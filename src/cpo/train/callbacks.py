from __future__ import annotations


class EarlyStopper:
    def __init__(self, patience: int = 3) -> None:
        self.patience = patience
        self.best = None
        self.bad_steps = 0

    def update(self, metric: float) -> bool:
        if self.best is None or metric > self.best:
            self.best = metric
            self.bad_steps = 0
            return False
        self.bad_steps += 1
        return self.bad_steps >= self.patience
