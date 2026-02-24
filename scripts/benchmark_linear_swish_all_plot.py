import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cupy as cp

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neunet
from neunet.nn.experimental import CUDALinearSwish, CUDALinear, CUDASwish
from neunet.nn.layers import Linear
from neunet.nn.activations import Swish


@dataclass
class Variant:
    key: str
    label: str
    color: str
    linear: object
    activation: object | None = None

    def forward(self, x):
        if self.activation is None:
            return self.linear(x)
        return self.activation(self.linear(x))

    def parameters(self):
        params = []
        if hasattr(self.linear, "weight"):
            params.append(self.linear.weight)
        if hasattr(self.linear, "bias") and self.linear.bias is not None:
            params.append(self.linear.bias)
        return params

    def sync_weights(self, weight, bias):
        self.linear.weight.data = cp.copy(weight)
        if hasattr(self.linear, "bias") and self.linear.bias is not None:
            self.linear.bias.data = cp.copy(bias)


def make_variants(in_features, out_features, device, swish_beta):
    return [
        Variant(
            key="fused_save_true",
            label="CUDALinearSwish(save_preactivation=True)",
            color="#0ea5e9",
            linear=CUDALinearSwish(
                in_features,
                out_features,
                bias=True,
                swish_beta=swish_beta,
                save_preactivation=True,
                device=device,
            ),
        ),
        Variant(
            key="fused_save_false",
            label="CUDALinearSwish(save_preactivation=False)",
            color="#0284c7",
            linear=CUDALinearSwish(
                in_features,
                out_features,
                bias=True,
                swish_beta=swish_beta,
                save_preactivation=False,
                device=device,
            ),
        ),
        Variant(
            key="cutlass_plus_swish",
            label='CUDALinear(backend="cutlass") + CUDASwish',
            color="#16a34a",
            linear=CUDALinear(in_features, out_features, bias=True, device=device, backend="cutlass"),
            activation=CUDASwish(beta=swish_beta),
        ),
        Variant(
            key="cublaslt_plus_swish",
            label='CUDALinear(backend="cublaslt") + CUDASwish',
            color="#f59e0b",
            linear=CUDALinear(in_features, out_features, bias=True, device=device, backend="cublaslt"),
            activation=CUDASwish(beta=swish_beta),
        ),
        Variant(
            key="linear_plus_swish",
            label="Linear + Swish",
            color="#ef4444",
            linear=Linear(in_features, out_features, bias=True, device=device),
            activation=Swish(beta=swish_beta),
        ),
    ]


def reset_grads(params):
    for p in params:
        p.grad = None


def benchmark_forward(variant, x_data, device, warmup, iters):
    for _ in range(warmup):
        x = neunet.tensor(cp.copy(x_data), device=device, requires_grad=False)
        _ = variant.forward(x)
    cp.cuda.Device().synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    start.record()
    for _ in range(iters):
        x = neunet.tensor(cp.copy(x_data), device=device, requires_grad=False)
        _ = variant.forward(x)
    end.record()
    end.synchronize()

    return cp.cuda.get_elapsed_time(start, end) / iters


def benchmark_backward(variant, x_data, device, warmup, iters):
    x_tmp = neunet.tensor(cp.copy(x_data), device=device, requires_grad=True)
    out_tmp = variant.forward(x_tmp)
    grad_shape = out_tmp.shape
    del x_tmp, out_tmp

    params = variant.parameters()

    for _ in range(warmup):
        x = neunet.tensor(cp.copy(x_data), device=device, requires_grad=True)
        out = variant.forward(x)
        grad = cp.random.uniform(-1, 1, grad_shape).astype(cp.float32)
        out.backward(grad)
        reset_grads(params)
    cp.cuda.Device().synchronize()

    total_ms = 0.0
    for _ in range(iters):
        x = neunet.tensor(cp.copy(x_data), device=device, requires_grad=True)
        out = variant.forward(x)
        grad = cp.random.uniform(-1, 1, grad_shape).astype(cp.float32)

        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        out.backward(grad)
        end.record()
        end.synchronize()

        total_ms += cp.cuda.get_elapsed_time(start, end)
        reset_grads(params)

    return total_ms / iters


def run_benchmark(configs, device, swish_beta, warmup, iters):
    forward_results = {}
    backward_results = {}
    labels = [f"{b}x{i}x{o}" for b, i, o in configs]

    print("=" * 120)
    print("Benchmark: 5 variants of Linear+Swish")
    print("=" * 120)

    for config_idx, (batch_size, in_features, out_features) in enumerate(configs):
        print(f"\nConfig {config_idx + 1}/{len(configs)}: {batch_size} x {in_features} x {out_features}")

        variants = make_variants(in_features, out_features, device, swish_beta)

        # Keep all variants on identical weights for fair comparison.
        base_weight = cp.copy(variants[-1].linear.weight.data)
        base_bias = cp.copy(variants[-1].linear.bias.data)
        for v in variants:
            v.sync_weights(base_weight, base_bias)

        x_data = cp.random.uniform(-1, 1, (batch_size, in_features)).astype(cp.float32)

        for v in variants:
            fw_ms = benchmark_forward(v, x_data, device, warmup=warmup, iters=iters)
            bw_ms = benchmark_backward(v, x_data, device, warmup=warmup, iters=iters)

            forward_results.setdefault(v.key, []).append(fw_ms)
            backward_results.setdefault(v.key, []).append(bw_ms)
            print(f"  {v.label:<52} FWD: {fw_ms:>8.4f} ms | BWD: {bw_ms:>8.4f} ms")

    return labels, forward_results, backward_results, make_variants(configs[0][1], configs[0][2], device, swish_beta)


def print_summary_table(configs, variants, forward_results, backward_results):
    header = f"{'Config (B x In x Out)':<22} {'Variant':<52} {'Forward (ms)':>13} {'Backward (ms)':>14}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    labels = [f"{b} x {i} x {o}" for b, i, o in configs]
    for idx, cfg in enumerate(labels):
        for v in variants:
            fw = forward_results[v.key][idx]
            bw = backward_results[v.key][idx]
            print(f"{cfg:<22} {v.label:<52} {fw:>13.4f} {bw:>14.4f}")


def plot_results(labels, variants, forward_results, backward_results, output_dir, show):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is not installed. Install with: pip install matplotlib") from exc

    # plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "axes.facecolor": "#f8fafc",
            "figure.facecolor": "#ffffff",
            "axes.edgecolor": "#cbd5e1",
            "grid.color": "#d9e2ec",
            "grid.linestyle": "--",
            "grid.alpha": 0.5,
        }
    )

    x = list(range(len(labels)))
    baseline_key = "linear_plus_swish"

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), sharex=True)

    for v in variants:
        fw = forward_results[v.key]
        bw = backward_results[v.key]
        fw_speedup = [forward_results[baseline_key][i] / fw[i] for i in range(len(labels))]
        bw_speedup = [backward_results[baseline_key][i] / bw[i] for i in range(len(labels))]

        axes[0, 0].plot(x, fw, marker="o", linewidth=2.0, color=v.color, label=v.label)
        axes[0, 1].plot(x, bw, marker="o", linewidth=2.0, color=v.color, label=v.label)
        axes[1, 0].plot(x, fw_speedup, marker="o", linewidth=2.0, color=v.color, label=v.label)
        axes[1, 1].plot(x, bw_speedup, marker="o", linewidth=2.0, color=v.color, label=v.label)

    axes[0, 0].set_title("Forward Latency")
    axes[0, 1].set_title("Backward Latency")
    axes[1, 0].set_title("Forward Speedup vs Linear + Swish")
    axes[1, 1].set_title("Backward Speedup vs Linear + Swish")

    axes[0, 0].set_ylabel("Latency (ms, log scale)")
    axes[0, 1].set_ylabel("Latency (ms, log scale)")
    axes[1, 0].set_ylabel("Speedup (x)")
    axes[1, 1].set_ylabel("Speedup (x)")
    axes[1, 0].set_xlabel("Config (B x In x Out)")
    axes[1, 1].set_xlabel("Config (B x In x Out)")

    axes[0, 0].set_yscale("log")
    axes[0, 1].set_yscale("log")
    axes[1, 0].axhline(1.0, linestyle=":", linewidth=1.2, color="#6b7280")
    axes[1, 1].axhline(1.0, linestyle=":", linewidth=1.2, color="#6b7280")

    for ax in axes.flatten():
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.minorticks_on()
        ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.65)
        ax.grid(True, which="minor", linestyle=":", linewidth=0.55, alpha=0.45)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle("Linear+Swish Variants: Forward/Backward Performance Comparison")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "linear_swish_variants_comparison.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")

    print(f"\nSaved plot: {out_path.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Benchmark 5 Linear+Swish variants and plot results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--swish-beta", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/plots"))
    parser.add_argument("--show", action="store_true", help="Show matplotlib window")
    args = parser.parse_args()

    cp.random.seed(42)

    configs = [
        (32, 256, 512),
        (64, 256, 512),
        (128, 256, 512),
        (256, 512, 1024),
        (512, 512, 1024),
        (1024, 512, 1024),
        (1024, 1024, 2048),
        (2048, 1024, 2048),
        (4096, 1024, 4096),
    ]

    labels, forward_results, backward_results, variants = run_benchmark(
        configs=configs,
        device=args.device,
        swish_beta=args.swish_beta,
        warmup=args.warmup,
        iters=args.iters,
    )

    print_summary_table(configs, variants, forward_results, backward_results)
    plot_results(labels, variants, forward_results, backward_results, args.output_dir, args.show)


if __name__ == "__main__":
    main()
