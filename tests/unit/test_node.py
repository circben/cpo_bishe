from cpo.tot.node import ToTNode


def test_node_defaults() -> None:
    n = ToTNode(node_id="n1", parent_id="root", depth=1, state="s", thought="t")
    assert n.score == 0.0
    assert n.children == []
