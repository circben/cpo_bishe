from cpo.train.dpo_trainer import DPOConfig, train_dpo_stub


def test_train_stub() -> None:
    result = train_dpo_stub(DPOConfig(), train_size=10)
    assert result["status"] == "ok"
