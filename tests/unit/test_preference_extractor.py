from cpo.preference.extractor import extract_preference_pairs
from cpo.tot.bfs_search import run_bfs_tot
from cpo.tot.success_path import backtrack_path


def test_extract_pairs_returns_list() -> None:
    tree = run_bfs_tot("test", task="gsm8k", max_depth=1, width=2, beam=2)
    best = max(tree.nodes.values(), key=lambda n: n.score).node_id
    path = backtrack_path(tree, best)
    pairs = extract_preference_pairs(tree, path, "test")
    assert isinstance(pairs, list)
