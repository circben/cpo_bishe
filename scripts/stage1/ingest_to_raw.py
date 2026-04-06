from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def _copy_tree(src: Path, dst: Path, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source path not found: {src}")
    if dst.exists() and overwrite:
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=not overwrite)


def _copy_file(src: Path, dst: Path, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local datasets into data/raw")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing raw data")
    args = parser.parse_args()

    gsm8k_src = DATA_DIR / "gsm8k"
    strategyqa_src = DATA_DIR / "strategyqa"

    gsm8k_dst = DATA_DIR / "raw" / "gsm8k"
    strategyqa_dst = DATA_DIR / "raw" / "strategyqa"

    _copy_tree(gsm8k_src, gsm8k_dst, overwrite=args.overwrite)
    _copy_tree(strategyqa_src, strategyqa_dst, overwrite=args.overwrite)

    print("Ingestion complete.")
    print(f"GSM8K raw path: {gsm8k_dst}")
    print(f"StrategyQA raw path: {strategyqa_dst}")


if __name__ == "__main__":
    main()
