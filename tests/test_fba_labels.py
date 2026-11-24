"""Offline unit tests for COBRApy FBA essentiality labeling using a tiny BiGG JSON fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

from metabolic_plm.data.fba_labels import build_dataset, extract_gene_labels

FIXTURE_MODEL_PATH = Path(__file__).parent / "fixtures" / "toy_bigg_model.json"


def test_extract_gene_labels_on_real_cobra_fixture() -> None:
    """Call the real `extract_gene_labels` on a 5-gene COBRA JSON model without network calls."""
    df = extract_gene_labels(FIXTURE_MODEL_PATH, essential_threshold=0.05)

    assert not df.empty
    assert len(df) == 5
    assert set(df.columns) == {
        "model_id",
        "gene_id",
        "uniprot_id",
        "ncbigene_id",
        "wt_growth",
        "ko_growth",
        "growth_ratio",
        "is_essential",
        "primary_subsystem",
    }

    by_gene = df.set_index("gene_id")

    # Wild-type optimal growth is 10.0 (9.8 via PATH_MAJOR + 0.2 via PATH_MINOR)
    assert by_gene.loc["g_essential", "wt_growth"] == pytest.approx(10.0, rel=1e-4)

    # 1. Deleting g_essential blocks GLC_TRANS completely -> ko_growth = 0.0 -> essential (1)
    assert by_gene.loc["g_essential", "ko_growth"] == pytest.approx(0.0, abs=1e-6)
    assert by_gene.loc["g_essential", "growth_ratio"] == pytest.approx(0.0, abs=1e-6)
    assert int(by_gene.loc["g_essential", "is_essential"]) == 1
    assert by_gene.loc["g_essential", "uniprot_id"] == "P00001"
    assert by_gene.loc["g_essential", "primary_subsystem"] == "Transport"

    # 2. Deleting g_subthreshold leaves only PATH_MINOR (0.2 / 10.0 = 0.02 < 0.05) -> essential (1)
    assert by_gene.loc["g_subthreshold", "ko_growth"] == pytest.approx(0.2, rel=1e-4)
    assert by_gene.loc["g_subthreshold", "growth_ratio"] == pytest.approx(0.02, rel=1e-4)
    assert int(by_gene.loc["g_subthreshold", "is_essential"]) == 1

    # 3. Deleting g_minor leaves PATH_MAJOR (9.8 / 10.0 = 0.98 >= 0.05) -> non-essential (0)
    assert by_gene.loc["g_minor", "ko_growth"] == pytest.approx(9.8, rel=1e-4)
    assert by_gene.loc["g_minor", "growth_ratio"] == pytest.approx(0.98, rel=1e-4)
    assert int(by_gene.loc["g_minor", "is_essential"]) == 0

    # 4. Redundant isozymes (g_iso1 or g_iso2) have growth_ratio = 1.0 -> non-essential (0)
    assert by_gene.loc["g_iso1", "growth_ratio"] == pytest.approx(1.0, rel=1e-4)
    assert int(by_gene.loc["g_iso1", "is_essential"]) == 0
    assert by_gene.loc["g_iso2", "growth_ratio"] == pytest.approx(1.0, rel=1e-4)
    assert int(by_gene.loc["g_iso2", "is_essential"]) == 0


def test_build_dataset_offline_with_mocked_uniprot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `build_dataset` writes Parquet and FASTA from local fixture without UniProt HTTP."""
    import shutil

    raw_dir = tmp_path / "raw_bigg"
    raw_dir.mkdir(parents=True)
    shutil.copy(FIXTURE_MODEL_PATH, raw_dir / "iTOY4.json")

    fake_sequences = {
        "P00001": "MKKVLVLGASGGIGQPLSLLLKNSPLVSELSLY",
        "P00002": "MNTQIIIFALVILLGLFFGIVIGRLPFKPVRKW",
        "P00003": "MSHQHTSQVWHRGWACAQPEGENWYACMPNFGT",
        "P00004": "MTDTRRTRRGRRDRRNRRDRRERRQRRDRRHRR",
        "P00005": "MFCCCPPPCCCPPPCCCPPPCCCPPPCCCPPPC",
    }

    def _mock_fetch_uniprot_sequences(
        ids: list[str],
        from_db: str = "UniProtKB_AC-ID",
        batch_size: int = 500,
    ) -> dict[str, str]:
        return {uid: fake_sequences[uid] for uid in ids if uid in fake_sequences}

    monkeypatch.setattr(
        "metabolic_plm.data.fba_labels.fetch_uniprot_sequences",
        _mock_fetch_uniprot_sequences,
    )

    processed_dir = tmp_path / "processed"
    parquet_path = build_dataset(raw_dir=raw_dir, processed_dir=processed_dir)
    fasta_path = processed_dir / "sequences.fasta"

    assert parquet_path.exists()
    assert fasta_path.exists()

    import pandas as pd

    out_df = pd.read_parquet(parquet_path)
    assert len(out_df) == 5
    assert out_df["is_essential"].tolist() == [1, 1, 0, 0, 0]
    assert ">iTOY4|g_essential\nMKKVLVLGASGGIGQPLSLLLKNSPLVSELSLY" in fasta_path.read_text(
        encoding="utf-8"
    )
