from __future__ import annotations

from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def download_gsm8k() -> None:
	target = RAW_DIR / "gsm8k"
	print("=" * 60)
	print("Downloading GSM8K...")
	ds = load_dataset("gsm8k", "main")
	ds.save_to_disk(str(target))
	print(f"Saved: {target}")


def download_strategyqa() -> None:
	target = RAW_DIR / "strategyqa"
	print("=" * 60)
	print("Downloading StrategyQA...")
	ds = load_dataset("metaeval/strategy-qa")
	ds.save_to_disk(str(target))
	print(f"Saved: {target}")


def main() -> None:
	RAW_DIR.mkdir(parents=True, exist_ok=True)
	download_gsm8k()
	download_strategyqa()
	print("=" * 60)
	print("Done.")


if __name__ == "__main__":
	main()