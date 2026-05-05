from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cpo.llm.generation import GenerationRuntime, complete_text


def _resolve_checkpoint_entry(entry_path: str) -> Path:
    path = Path(entry_path)
    if not path.exists():
        raise SystemExit(f"Model entry path not found: {path}")

    if path.is_dir() and path.name == "final_checkpoint":
        return path

    direct = path / "final_checkpoint"
    if direct.exists() and direct.is_dir():
        return direct

    candidates = sorted(path.rglob("final_checkpoint"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return path
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage4 text inference runner (local model)")
    parser.add_argument("--input", required=True, help="Path to input JSON with {text, meta}")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--model", default="outputs/checkpoints/gsm8k/stage3_cpo_20260405_183114")
    parser.add_argument("--device", default="")
    parser.add_argument("--do-sample", type=int, choices=[0, 1], default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    text = str(payload.get("text", "")).strip()
    if not text:
        raise SystemExit("Input JSON missing 'text'")

    model_path = _resolve_checkpoint_entry(args.model).as_posix()

    runtime_kwargs = {
        "model_name": model_path,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "provider": "local",
        "allow_fallback": False,
    }
    if args.device.strip():
        runtime_kwargs["device"] = args.device.strip()
    runtime = GenerationRuntime(**runtime_kwargs)

    result = complete_text(
        prompt=text,
        runtime=runtime,
        max_new_tokens=args.max_new_tokens,
        do_sample=bool(args.do_sample),
    )

    output_payload = {
        "input": text,
        "meta": payload.get("meta", {}),
        "model": model_path,
        "result": result,
    }

    (output_dir / "result.json").write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
