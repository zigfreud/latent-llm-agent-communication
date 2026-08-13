from types import SimpleNamespace

from src.pipelines import infer


def test_load_source_forwards_explicit_pytorch_bin_selection(monkeypatch):
    revision = "source-revision"
    tokenizer = SimpleNamespace(
        pad_token=None,
        eos_token="<eos>",
        init_kwargs={"_commit_hash": revision},
    )
    model = SimpleNamespace(
        config=SimpleNamespace(_commit_hash=revision),
        eval=lambda: None,
    )
    captured = {}

    monkeypatch.setattr(
        infer.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer,
    )

    def fake_model_loader(*args, **kwargs):
        captured.update(kwargs)
        return model

    monkeypatch.setattr(
        infer.AutoModelForCausalLM,
        "from_pretrained",
        fake_model_loader,
    )

    loaded_model, loaded_tokenizer = infer.load_source(
        "source/model",
        "cpu",
        revision=revision,
        use_safetensors=False,
    )

    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
    assert tokenizer.pad_token == tokenizer.eos_token
    assert captured["revision"] == revision
    assert captured["use_safetensors"] is False
