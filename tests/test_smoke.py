from metabolic_plm.data.ingest import DEFAULT_MODELS


def test_default_models_populated() -> None:
    assert len(DEFAULT_MODELS) >= 5
    assert "iML1515" in DEFAULT_MODELS
