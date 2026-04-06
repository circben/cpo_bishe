from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(func: Callable[[T], R], items: Iterable[T], max_workers: int = 8) -> list[R]:
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(func, items))
