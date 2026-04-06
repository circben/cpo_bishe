from __future__ import annotations

from cpo.eval.inference import run_cot_inference


def main() -> None:
    sample = "Is the sky blue during a clear day?"
    out = run_cot_inference(sample)
    print(out)


if __name__ == "__main__":
    main()
