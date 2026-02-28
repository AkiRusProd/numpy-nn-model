import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cupy as cp

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import neunet
from neunet.nn.experimental import (
    CUDALinear,
    CUDALinearSwish,
    CUDAFusedSwishAndMul,
    CUDASwish,
)
from neunet.nn.layers import Linear
from neunet.nn.activations import Swish


@dataclass
class SwiGLUVariant:
    key: str
    label: str
    color: str

    def forward(self, x):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def sync_from_reference(self, ref_gate_w, ref_gate_b, ref_up_w, ref_up_b):
        raise NotImplementedError


class SwiGLUFromLinearSwish(SwiGLUVariant):
    def __init__(self, in_features, hidden_size, device, swish_beta, save_preactivation):
        key_suffix = "save_true" if save_preactivation else "save_false"
        label_suffix = "save_preactivation=True" if save_preactivation else "save_preactivation=False"
        color = "#0ea5e9" if save_preactivation else "#0284c7"
        super().__init__(
            key=f"linear_swish_plus_linear_{key_suffix}",
            label=f"CUDALinearSwish({label_suffix}) + CUDALinear + mul",
            color=color,
        )
        self.gate_linear_swish = CUDALinearSwish(
            in_features,
            hidden_size,
            bias=True,
            swish_beta=swish_beta,
            save_preactivation=save_preactivation,
            device=device,
        )
        self.up_linear = CUDALinear(
            in_features, hidden_size, bias=True, device=device, backend="cutlass"
        )

    def forward(self, x):
        return self.gate_linear_swish(x) * self.up_linear(x)

    def parameters(self):
        params = [
            self.gate_linear_swish.weight,
            self.up_linear.weight,
        ]
        if self.gate_linear_swish.bias is not None:
            params.append(self.gate_linear_swish.bias)
        if self.up_linear.bias is not None:
            params.append(self.up_linear.bias)
        return params

    def sync_from_reference(self, ref_gate_w, ref_gate_b, ref_up_w, ref_up_b):
        self.gate_linear_swish.weight.data = cp.copy(ref_gate_w)
        self.up_linear.weight.data = cp.copy(ref_up_w)
        if self.gate_linear_swish.bias is not None:
            self.gate_linear_swish.bias.data = cp.copy(ref_gate_b)
        if self.up_linear.bias is not None:
            self.up_linear.bias.data = cp.copy(ref_up_b)


class SwiGLUFromFusedSwishMul(SwiGLUVariant):
    def __init__(self, in_features, hidden_size, device, swish_beta):
        super().__init__(
            key="linear_2h_plus_fused_swish_mul",
            label="CUDALinear(2H) + CUDAFusedSwishAndMul",
            color="#16a34a",
        )
        self.linear_2h = CUDALinear(
            in_features, hidden_size * 2, bias=True, device=device, backend="cutlass"
        )
        self.fused_swish_mul = CUDAFusedSwishAndMul(beta=swish_beta)

    def forward(self, x):
        return self.fused_swish_mul(self.linear_2h(x))

    def parameters(self):
        params = [self.linear_2h.weight]
        if self.linear_2h.bias is not None:
            params.append(self.linear_2h.bias)
        return params

    def sync_from_reference(self, ref_gate_w, ref_gate_b, ref_up_w, ref_up_b):
        self.linear_2h.weight.data = cp.concatenate(
            [cp.copy(ref_gate_w), cp.copy(ref_up_w)], axis=0
        )
        if self.linear_2h.bias is not None:
            self.linear_2h.bias.data = cp.concatenate(
                [cp.copy(ref_gate_b), cp.copy(ref_up_b)], axis=1
            )


class SwiGLUFromLinearAndSwish(SwiGLUVariant):
    def __init__(self, in_features, hidden_size, device, swish_beta):
        super().__init__(
            key="linear_plus_swish_baseline",
            label="Linear + Swish + Linear + mul (baseline)",
            color="#ef4444",
        )
        self.gate_linear = Linear(in_features, hidden_size, bias=True, device=device)
        self.up_linear = Linear(in_features, hidden_size, bias=True, device=device)
        self.swish = Swish(beta=swish_beta)

    def forward(self, x):
        return self.swish(self.gate_linear(x)) * self.up_linear(x)

    def parameters(self):
        params = [self.gate_linear.weight, self.up_linear.weight]
        if self.gate_linear.bias is not None:
            params.append(self.gate_linear.bias)
        if self.up_linear.bias is not None:
            params.append(self.up_linear.bias)
        return params

    def sync_from_reference(self, ref_gate_w, ref_gate_b, ref_up_w, ref_up_b):
        self.gate_linear.weight.data = cp.copy(ref_gate_w)
        self.up_linear.weight.data = cp.copy(ref_up_w)
        if self.gate_linear.bias is not None:
            self.gate_linear.bias.data = cp.copy(ref_gate_b)
        if self.up_linear.bias is not None:
            self.up_linear.bias.data = cp.copy(ref_up_b)


class SwiGLUFromCUDALinearAndCUDASwish(SwiGLUVariant):
    def __init__(self, in_features, hidden_size, device, swish_beta):
        super().__init__(
            key="cudalinear_plus_cudaswish",
            label="CUDALinear + CUDASwish + CUDALinear + mul",
            color="#f59e0b",
        )
        self.gate_linear = CUDALinear(
            in_features, hidden_size, bias=True, device=device, backend="cutlass"
        )
        self.up_linear = CUDALinear(
            in_features, hidden_size, bias=True, device=device, backend="cutlass"
        )
        self.swish = CUDASwish(beta=swish_beta)

    def forward(self, x):
        return self.swish(self.gate_linear(x)) * self.up_linear(x)

    def parameters(self):
        params = [self.gate_linear.weight, self.up_linear.weight]
        if self.gate_linear.bias is not None:
            params.append(self.gate_linear.bias)
        if self.up_linear.bias is not None:
            params.append(self.up_linear.bias)
        return params

    def sync_from_reference(self, ref_gate_w, ref_gate_b, ref_up_w, ref_up_b):
        self.gate_linear.weight.data = cp.copy(ref_gate_w)
        self.up_linear.weight.data = cp.copy(ref_up_w)
        if self.gate_linear.bias is not None:
            self.gate_linear.bias.data = cp.copy(ref_gate_b)
        if self.up_linear.bias is not None:
            self.up_linear.bias.data = cp.copy(ref_up_b)


def make_variants(in_features, hidden_size, device, swish_beta):
    return [
        SwiGLUFromLinearAndSwish(in_features, hidden_size, device, swish_beta),
        SwiGLUFromCUDALinearAndCUDASwish(in_features, hidden_size, device, swish_beta),
        SwiGLUFromLinearSwish(
            in_features,
            hidden_size,
            device,
            swish_beta,
            save_preactivation=True,
        ),
        SwiGLUFromLinearSwish(
            in_features,
            hidden_size,
            device,
            swish_beta,
            save_preactivation=False,
        ),
        SwiGLUFromFusedSwishMul(in_features, hidden_size, device, swish_beta),
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
    labels = [f"{b}x{h}x{s}" for b, h, s in configs]

    print("=" * 120)
    print("Benchmark: SwiGLU variants")
    print("=" * 120)

    for config_idx, (batch_size, hidden_dim, seq_len) in enumerate(configs):
        print(
            f"\nConfig {config_idx + 1}/{len(configs)}: "
            f"{batch_size} x {hidden_dim} x {seq_len}"
        )

        variants = make_variants(hidden_dim, hidden_dim, device, swish_beta)

        stdv = 1.0 / (hidden_dim ** 0.5)
        ref_gate_w = cp.random.uniform(-stdv, stdv, (hidden_dim, hidden_dim)).astype(
            cp.float32
        )
        ref_up_w = cp.random.uniform(-stdv, stdv, (hidden_dim, hidden_dim)).astype(
            cp.float32
        )
        ref_gate_b = cp.random.uniform(-stdv, stdv, (1, hidden_dim)).astype(cp.float32)
        ref_up_b = cp.random.uniform(-stdv, stdv, (1, hidden_dim)).astype(cp.float32)

        for v in variants:
            v.sync_from_reference(ref_gate_w, ref_gate_b, ref_up_w, ref_up_b)

        x_data = cp.random.uniform(-1, 1, (batch_size, seq_len, hidden_dim)).astype(
            cp.float32
        )

        for v in variants:
            fw_ms = benchmark_forward(v, x_data, device, warmup=warmup, iters=iters)
            bw_ms = benchmark_backward(v, x_data, device, warmup=warmup, iters=iters)
            forward_results.setdefault(v.key, []).append(fw_ms)
            backward_results.setdefault(v.key, []).append(bw_ms)
            print(f"  {v.label:<50} FWD: {fw_ms:>8.4f} ms | BWD: {bw_ms:>8.4f} ms")

    return (
        labels,
        forward_results,
        backward_results,
        make_variants(configs[0][1], configs[0][1], device, swish_beta),
    )


def print_summary_table(configs, variants, forward_results, backward_results):
    header = (
        f"{'Config (B x H x S)':<22} {'Variant':<50} "
        f"{'Forward (ms)':>13} {'Backward (ms)':>14}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    labels = [f"{b} x {h} x {s}" for b, h, s in configs]
    for idx, cfg in enumerate(labels):
        for v in variants:
            fw = forward_results[v.key][idx]
            bw = backward_results[v.key][idx]
            print(f"{cfg:<22} {v.label:<50} {fw:>13.4f} {bw:>14.4f}")


def plot_results(labels, variants, forward_results, backward_results, output_dir, show):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is not installed. Install with: pip install matplotlib"
        ) from exc

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
    baseline_key = "linear_plus_swish_baseline"

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)

    for v in variants:
        fw = forward_results[v.key]
        bw = backward_results[v.key]
        fw_speedup = [forward_results[baseline_key][i] / fw[i] for i in range(len(labels))]
        bw_speedup = [backward_results[baseline_key][i] / bw[i] for i in range(len(labels))]

        axes[0, 0].plot(x, fw, marker="o", linewidth=2.0, color=v.color, label=v.label)
        axes[0, 1].plot(x, bw, marker="o", linewidth=2.0, color=v.color, label=v.label)
        axes[1, 0].plot(
            x, fw_speedup, marker="o", linewidth=2.0, color=v.color, label=v.label
        )
        axes[1, 1].plot(
            x, bw_speedup, marker="o", linewidth=2.0, color=v.color, label=v.label
        )

    axes[0, 0].set_title("Forward Latency")
    axes[0, 1].set_title("Backward Latency")
    axes[1, 0].set_title("Forward Speedup vs Linear+Swish baseline")
    axes[1, 1].set_title("Backward Speedup vs Linear+Swish baseline")

    axes[0, 0].set_ylabel("Latency (ms, log scale)")
    axes[0, 1].set_ylabel("Latency (ms, log scale)")
    axes[1, 0].set_ylabel("Speedup (x)")
    axes[1, 1].set_ylabel("Speedup (x)")
    axes[1, 0].set_xlabel("Config (B x H x S)")
    axes[1, 1].set_xlabel("Config (B x H x S)")

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
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.suptitle("SwiGLU Variants: Forward/Backward Performance Comparison")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "swiglu_two_variants_comparison.png"
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"\nSaved plot: {out_path.resolve()}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark two SwiGLU variants and plot results"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--swish-beta", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path("scripts/plots"))
    parser.add_argument("--show", action="store_true", help="Show matplotlib window")
    args = parser.parse_args()

    cp.random.seed(42)

    configs = [
        # (batch_size, hidden_dim, seq_len)
        (1, 256, 256),
        (1, 256, 512),
        (1, 512, 512),
        (1, 512, 1024),
        (1, 1024, 1024),
        (1, 1024, 2048),
        (1, 2048, 2048),
        (1, 2048, 4096),
        (1, 4096, 4096),
        (1, 4096, 8192),
        # (1, 8192, 8192),
        # (1, 8192, 16384),
        # (1, 16384, 32768),
    ]

    labels, forward_results, backward_results, variants = run_benchmark(
        configs=configs,
        device=args.device,
        swish_beta=args.swish_beta,
        warmup=args.warmup,
        iters=args.iters,
    )

    print_summary_table(configs, variants, forward_results, backward_results)
    plot_results(
        labels,
        variants,
        forward_results,
        backward_results,
        args.output_dir,
        args.show,
    )


if __name__ == "__main__":
    main()
