from cpo.tot.bfs_search import run_bfs_tot


def test_bfs_builds_tree() -> None:
    tree = run_bfs_tot("2+2=?", task="gsm8k", max_depth=2, width=2, beam=2)
    assert "root" in tree.nodes
    assert len(tree.nodes) > 1
