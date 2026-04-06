from cpo.eval.metrics_gsm8k import exact_match


def test_exact_match() -> None:
    assert exact_match("42", "42") == 1.0
    assert exact_match("41", "42") == 0.0
