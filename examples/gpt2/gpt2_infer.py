"""
Minimal GPT-2 (small) inference using neunet + Hugging Face weights.

Usage:
  python examples/gpt2/gpt2_infer.py --prompt "Hello" --max-new-tokens 50 --device cpu
"""

from __future__ import annotations

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"          # if MKL is used anywhere
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"   # macOS Accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import neunet
import neunet.nn as nn
from neunet import Tensor


REPO_ID_DEFAULT = "openai-community/gpt2"


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def _load_state_dict(model_path: Path) -> dict[str, Any]:
    if model_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "safetensors is required to load .safetensors weights. "
                "Install it with: pip install safetensors"
            ) from exc
        state = load_file(str(model_path))
        return _unwrap_state_dict(state)

    state = torch.load(str(model_path), map_location="cpu")
    return _unwrap_state_dict(state)


def _unwrap_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    # Some checkpoints wrap the actual state dict.
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    elif "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    # Strip optional "module." prefix
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    return state


def _get_key(state: dict[str, Any], key: str) -> Any:
    if key in state:
        return state[key]

    prefixes = ["", "transformer.", "gpt2.", "model.", "module."]
    for pref in prefixes:
        candidate = f"{pref}{key}" if pref else key
        if candidate in state:
            return state[candidate]

    for pref in ["transformer.", "gpt2.", "model.", "module."]:
        if key.startswith(pref):
            candidate = key[len(pref):]
            if candidate in state:
                return state[candidate]

    sample = list(state.keys())[:20]
    raise KeyError(f"Key '{key}' not found in state dict. Sample keys: {sample}")


def download_gpt2_files(repo_id: str, cache_dir: Path) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    filenames = [
        "config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
    ]

    paths: dict[str, Path] = {}
    for name in filenames:
        path = hf_hub_download(repo_id=repo_id, filename=name, cache_dir=str(cache_dir))
        paths[name] = Path(path)

    # Prefer pytorch_model.bin, fallback to safetensors
    try:
        weights = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", cache_dir=str(cache_dir))
        paths["weights"] = Path(weights)
    except Exception:
        weights = hf_hub_download(repo_id=repo_id, filename="model.safetensors", cache_dir=str(cache_dir))
        paths["weights"] = Path(weights)

    return paths


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, attn_pdrop: float, resid_pdrop: float):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.scale = 1.0 / np.sqrt(self.head_dim)

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

    def _split_heads(self, x: Tensor) -> Tensor:
        # (B, T, C) -> (B, nh, T, hs)
        b, t, c = x.shape
        x = x.reshape(b, t, self.n_head, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x: Tensor) -> Tensor:
        # (B, nh, T, hs) -> (B, T, C)
        b, nh, t, hs = x.shape
        return x.transpose(0, 2, 1, 3).reshape(b, t, nh * hs)

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape

        qkv = self.c_attn(x)
        q = qkv[:, :, : self.n_head * self.head_dim]
        k = qkv[:, :, self.n_head * self.head_dim : 2 * self.n_head * self.head_dim]
        v = qkv[:, :, 2 * self.n_head * self.head_dim :]

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        att = neunet.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale

        # Causal mask
        mask = np.tril(np.ones((t, t), dtype=np.int32))
        mask = mask[None, None, :, :]
        mask = neunet.tensor(mask, device=x.device, dtype=neunet.int32)
        att = neunet.where(mask == 0, -1e9, att)

        att = nn.Softmax(axis=-1)(att)
        att = self.attn_drop(att)

        y = neunet.matmul(att, v)
        y = self._merge_heads(y)
        y = self.c_proj(y)
        y = self.resid_drop(y)
        return y


class MLP(nn.Module):
    def __init__(self, n_embd: int, resid_pdrop: float):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=True)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(resid_pdrop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.drop(x)
        return x


class GPT2Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, ln_eps: float, attn_pdrop: float, resid_pdrop: float):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, eps=ln_eps)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.ln_2 = nn.LayerNorm(n_embd, eps=ln_eps)
        self.mlp = MLP(n_embd, resid_pdrop)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        n_embd = cfg["n_embd"]
        n_head = cfg["n_head"]
        n_layer = cfg["n_layer"]
        vocab_size = cfg["vocab_size"]
        n_positions = cfg.get("n_positions", cfg.get("n_ctx", 1024))
        ln_eps = cfg.get("layer_norm_epsilon", 1e-5)
        attn_pdrop = cfg.get("attn_pdrop", 0.0)
        resid_pdrop = cfg.get("resid_pdrop", 0.0)
        embd_pdrop = cfg.get("embd_pdrop", 0.0)

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_positions, n_embd)
        self.drop = nn.Dropout(embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(n_embd, n_head, ln_eps, attn_pdrop, resid_pdrop) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd, eps=ln_eps)

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.wte.weight
        self.to("cpu")

    def forward(self, idx: np.ndarray | Tensor) -> Tensor:
        if not isinstance(idx, Tensor):
            idx = neunet.tensor(idx, dtype=neunet.int32, device=self.device)

        b, t = idx.shape
        pos = np.arange(t, dtype=np.int32)[None, :]
        pos = neunet.tensor(pos, dtype=neunet.int32, device=self.device)

        x = self.wte(idx) + self.wpe(pos)
        x = self.drop(x)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


def load_gpt2_weights(model: GPT2, state: dict[str, Any]) -> None:
    # Embeddings
    model.wte.weight.data = _to_numpy(_get_key(state, "transformer.wte.weight"))
    model.wpe.weight.data = _to_numpy(_get_key(state, "transformer.wpe.weight"))

    # Blocks
    n_layer = len(model.h)
    for i in range(n_layer):
        prefix = f"transformer.h.{i}."
        block = model.h[i]

        block.ln_1.weight.data = _to_numpy(_get_key(state, prefix + "ln_1.weight"))
        block.ln_1.bias.data = _to_numpy(_get_key(state, prefix + "ln_1.bias"))

        block.attn.c_attn.weight.data = _to_numpy(_get_key(state, prefix + "attn.c_attn.weight")).T
        block.attn.c_attn.bias.data = _to_numpy(_get_key(state, prefix + "attn.c_attn.bias")).reshape(1, -1)

        block.attn.c_proj.weight.data = _to_numpy(_get_key(state, prefix + "attn.c_proj.weight")).T
        block.attn.c_proj.bias.data = _to_numpy(_get_key(state, prefix + "attn.c_proj.bias")).reshape(1, -1)

        block.ln_2.weight.data = _to_numpy(_get_key(state, prefix + "ln_2.weight"))
        block.ln_2.bias.data = _to_numpy(_get_key(state, prefix + "ln_2.bias"))

        block.mlp.c_fc.weight.data = _to_numpy(_get_key(state, prefix + "mlp.c_fc.weight")).T
        block.mlp.c_fc.bias.data = _to_numpy(_get_key(state, prefix + "mlp.c_fc.bias")).reshape(1, -1)

        block.mlp.c_proj.weight.data = _to_numpy(_get_key(state, prefix + "mlp.c_proj.weight")).T
        block.mlp.c_proj.bias.data = _to_numpy(_get_key(state, prefix + "mlp.c_proj.bias")).reshape(1, -1)

    # Final LN
    model.ln_f.weight.data = _to_numpy(_get_key(state, "transformer.ln_f.weight"))
    model.ln_f.bias.data = _to_numpy(_get_key(state, "transformer.ln_f.bias"))

    model.lm_head.weight = neunet.clone(model.wte.weight)


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


@dataclass
class GenerationResult:
    text: str
    generated_tokens: int
    elapsed_sec: float


@dataclass
class GenerationRequest:
    prompt: str
    max_new_tokens: int
    temperature: float
    top_k: int


class NeunetGPT2Runner:
    def __init__(self, model: GPT2, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, req: GenerationRequest) -> GenerationResult:
        encoded = self.tokenizer.encode(req.prompt)
        input_ids = encoded.ids

        start_time = time.perf_counter()
        for _ in range(req.max_new_tokens):
            idx = np.array([input_ids], dtype=np.int32)
            logits = self.model(idx)

            last = logits.data[0, -1]
            if hasattr(last, "get"):
                last = last.get()

            if req.temperature != 1.0:
                last = last / max(req.temperature, 1e-6)

            if req.top_k > 0:
                top_k = min(req.top_k, last.shape[0])
                inds = np.argpartition(last, -top_k)[-top_k:]
                probs = softmax_np(last[inds])
                next_id = int(np.random.choice(inds, p=probs))
            else:
                probs = softmax_np(last)
                next_id = int(np.argmax(probs))

            input_ids.append(next_id)
        elapsed = time.perf_counter() - start_time

        return GenerationResult(
            text=self.tokenizer.decode(input_ids),
            generated_tokens=req.max_new_tokens,
            elapsed_sec=elapsed,
        )


class TransformersGPT2Runner:
    def __init__(self, config_path: Path, tokenizer_path: Path, weights_path: Path, device: str, seed: int | None):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "transformers is required for comparison. Install it with: pip install transformers"
            ) from exc

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available")
            hf_device = "cuda"
        else:
            hf_device = "cpu"

        snapshot_dir = str(weights_path.parent)
        use_safetensors = weights_path.suffix == ".safetensors"

        self.device = hf_device
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            snapshot_dir,
            local_files_only=True,
            use_safetensors=use_safetensors,
        )
        self.model.to(hf_device)
        self.model.eval()

    def generate(self, req: GenerationRequest) -> GenerationResult:
        inputs = self.tokenizer(req.prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        input_len = input_ids.shape[1]

        do_sample = req.temperature != 1.0 or req.top_k > 0

        start_time = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=req.max_new_tokens,
                do_sample=do_sample,
                temperature=req.temperature,
                top_k=req.top_k if req.top_k > 0 else 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - start_time

        generated_tokens = int(output_ids.shape[1] - input_len)
        text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return GenerationResult(text=text, generated_tokens=generated_tokens, elapsed_sec=elapsed)


def _print_result(result: GenerationResult) -> None:
    print(result.text)
    if result.elapsed_sec > 0:
        print(f"tokens_per_sec: {result.generated_tokens / result.elapsed_sec:.2f}")
    else:
        print("tokens_per_sec: inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--backend", type=str, default="neunet", choices=["neunet", "transformers"])
    parser.add_argument("--repo-id", type=str, default=REPO_ID_DEFAULT)
    parser.add_argument("--cache-dir", type=str, default="saved models/gpt2_hf")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    cache_dir = Path(args.cache_dir)
    paths = download_gpt2_files(args.repo_id, cache_dir)

    with paths["config.json"].open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    tokenizer = Tokenizer.from_file(str(paths["tokenizer.json"]))

    state = _load_state_dict(paths["weights"])

    req = GenerationRequest(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    if args.backend == "neunet":
        model = GPT2(cfg)
        load_gpt2_weights(model, state)
        model.eval()

        if args.device == "cuda":
            model = model.to("cuda")

        result = NeunetGPT2Runner(model=model, tokenizer=tokenizer).generate(req)
    else:
        result = TransformersGPT2Runner(
            config_path=paths["config.json"],
            tokenizer_path=paths["tokenizer.json"],
            weights_path=paths["weights"],
            device=args.device,
            seed=args.seed,
        ).generate(req)

    _print_result(result)


if __name__ == "__main__":
    main()
