"""Phase 0 checkpoint: load the configured model and run one generation.

    python scripts/smoke_load.py --config tiny
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs import get_config, parse_overrides  # noqa: E402
from utils import load_model_and_tokenizer, set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tiny")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    args = ap.parse_args()

    cfg = get_config(args.config, **parse_overrides(args.set))
    set_seed(cfg.seed)
    print(f"[smoke] loading {cfg.model_name} ...")
    model, tokenizer, device = load_model_and_tokenizer(cfg, for_training=False)
    print(f"[smoke] loaded on {device}, dtype={cfg.dtype}")

    messages = [{"role": "user", "content": "In one sentence, what is 17 + 25?"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens, do_sample=False)
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)
    print("[smoke] completion:\n" + text.strip())
    print("[smoke] OK")


if __name__ == "__main__":
    main()
