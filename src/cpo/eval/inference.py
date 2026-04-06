from __future__ import annotations

import time


def run_cot_inference(question: str) -> dict:
    start = time.perf_counter()
    answer = f"Let's think step by step. Final answer: {question[:16]}..."
    latency = time.perf_counter() - start
    return {"prediction": answer, "latency": latency}
