from __future__ import annotations

from pathlib import Path

from cpo.common.io_utils import write_json
from cpo.tot.tree import ToTTree


def save_tree_json(tree: ToTTree, path: str | Path) -> None:
    write_json(path, tree.to_dict())
