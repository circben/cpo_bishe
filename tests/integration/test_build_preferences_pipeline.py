from cpo.preference.formatter import to_dpo_format


def test_dpo_formatter() -> None:
    rows = [{"input": "q", "prefix": "p", "chosen": "a", "rejected": "b"}]
    out = to_dpo_format(rows)
    assert out[0]["prompt"].startswith("Question:")
