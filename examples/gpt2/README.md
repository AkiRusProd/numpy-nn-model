# GPT-2 Inference

## Run

```bash
python examples/gpt2/gpt2_infer.py --backend neunet --prompt "Hello" --max-new-tokens 50 --device cpu
```

```bash
python examples/gpt2/gpt2_infer.py --backend transformers --prompt "Hello" --max-new-tokens 50 --device cpu
```

### Options

- `--backend` (`neunet` or `transformers`, default: `neunet`)
- `--temperature` (default: 1.0)
- `--top-k` (default: 40)
- `--device` (`cpu` or `cuda`)
- `--seed` (default: not set)
- `--repo-id` (default: `openai-community/gpt2`)
- `--cache-dir` (default: `saved models/gpt2_hf`)

## Notes

- The script prints generated text and `tokens_per_sec`.
