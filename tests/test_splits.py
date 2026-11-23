"""Tests verifying ZERO sequence leakage across train/val/test splits at 30% identity."""

from __future__ import annotations

import pandas as pd
import pytest

from metabolic_plm.data.split import (
    SplitConfig,
    compute_pairwise_identity,
    create_homology_aware_splits,
    verify_zero_sequence_leakage,
)


def _build_synthetic_homolog_dataset() -> pd.DataFrame:
    """Build a synthetic protein dataset containing distinct homology families (>30% identity)."""
    # Family 1: E. coli / S. enterica / K. pneumoniae orthologous enzyme cluster (~85-95% identity)
    fam1_base = "MKKVLVLGASGGIGQPLSLLLKNSPLVSELSLYDIAHTPGVAADLSHIETKAAVKGYLGPEQLPDCLKGCDVVVIPAGVPR"
    fam1_orth1 = "MKKVLVLGASGGIGQPLSLLLKNSPLVSELSLYDIAHTPGVAADLSHIETKAAVKGYVGPEQLPDCLKGCDVVVIPAGVPR"
    fam1_orth2 = "MKKILVLGASGGIGQPLSLLLKNSPLVSELSLYDIAHTPGVAADLSHIETRAAVKGYVGPEQLPDCLKGCDVVVIPAGVPR"

    # Family 2: B. subtilis / S. aureus Gram-positive transporter family (~88% identity)
    fam2_base = "MNTQIIIFALVILLGLFFGIVIGRLPFKPVRKWLQDHFNPEDPEKVRQRAIDLMKKHGIQPEEFEKLLAEGHYDPN"
    fam2_orth1 = "MNTQIIIFALVILLGLFFGIVIGRLPFKPVRKWLQDHFNPEDPEKVRQRAIDLMKKHGIEPEEFEKLLAEGHYDPN"
    fam2_orth2 = "MNTQIIVFALVILLGLFFGIVIGRLPFKPVRKWLQDHFNPEDPEKVRQRAVDLMKKHGIEPEEFEKLLAEGHYDPN"

    # Family 3: P. putida aromatic dioxygenase family (~90% identity)
    fam3_base = "MSHQHTSQVWHRGWACAQPEGENWYACMPNFGTCYCRWPPQEYQTWCFYWHNCKMPWHWCYHCHQWHQWWY"
    fam3_orth1 = "MSHQHTSQVWHRGWACAQPEGENWYACMPNFGTCYCRWPPQEYQTWCFYWHNCKMPWHWCYHCHQWHQWWY"

    # Family 4: M. tuberculosis mycolic acid synthase family (~92% identity)
    fam4_base = "MTDTRRTRRGRRDRRNRRDRRERRQRRDRRHRRDRRTRRGRRDRRNRRDRRERRQRRDRRHRRDRRTRRGRR"
    fam4_orth1 = "MTDTRRTRRGRRDRRNRRDRRERRQRRDRRHRRDRRTRRGRRDRRNRRDRRERRQRRDRRHRRDRRTRRGRK"

    # Family 5: H. pylori urease accessory protein family (~90% identity)
    fam5_base = "MFCCCPPPCCCPPPCCCPPPCCCPPPCCCPPPCCCPPPCCCPPPCCCPPPCCCPPPCCCPPPCCCPPPCCC"
    fam5_orth1 = "MFCCCPPPCCCPPPCCCPPPCCCPPPCCCPPPCCCPPPCCCPPPCCCPPPCCCPPPCCCPPPCCCPPPCCM"

    rows = [
        ("iML1515__b3236", fam1_base, 1),
        ("STM_v1_0__STM3359", fam1_orth1, 1),
        ("iYL1228__KPN_03646", fam1_orth2, 1),
        ("iYO844__BSU00010", fam2_base, 0),
        ("iSB619__SA0001", fam2_orth1, 0),
        ("iCN900__CD0001", fam2_orth2, 0),
        ("iJN1463__PP_0001", fam3_base, 1),
        ("iJN1463__PP_0002", fam3_orth1, 1),
        ("iNJ661__Rv0001", fam4_base, 1),
        ("iEK1008__Rv0002", fam4_orth1, 1),
        ("iIT341__HP0072", fam5_base, 0),
        ("iIT341__HP0073", fam5_orth1, 0),
    ]
    return pd.DataFrame(rows, columns=["protein_uid", "sequence", "is_essential"])


def test_homologs_never_cross_train_val_test_splits() -> None:
    """Verify that orthologous sequences (>=30% identity) never cross splits."""
    df = _build_synthetic_homolog_dataset()
    cfg = SplitConfig(
        min_seq_id=0.30,
        coverage=0.80,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        seed=42,
        mmseqs_binary="nonexistent_mmseqs_for_deterministic_unit_test",
    )

    split_df = create_homology_aware_splits(df, config=cfg)

    assert set(split_df["split"].unique()) == {"train", "val", "test"}
    assert split_df["cluster_id"].isna().sum() == 0

    # Verify zero cluster overlap and zero pairwise sequence identity >= 30% across splits
    verify_zero_sequence_leakage(split_df, max_allowed_identity=0.30)

    # Explicitly assert Family 1 orthologs (E. coli, Salmonella, Klebsiella) share one split
    fam1_splits = set(
        split_df.loc[
            split_df["protein_uid"].isin(
                ["iML1515__b3236", "STM_v1_0__STM3359", "iYL1228__KPN_03646"]
            ),
            "split",
        ]
    )
    assert len(fam1_splits) == 1


def test_leakage_detector_catches_injected_cross_split_homolog() -> None:
    """Verify that `verify_zero_sequence_leakage` raises AssertionError if a homolog leaks."""
    df = _build_synthetic_homolog_dataset()
    cfg = SplitConfig(
        min_seq_id=0.30,
        mmseqs_binary="nonexistent_mmseqs_for_deterministic_unit_test",
    )
    split_df = create_homology_aware_splits(df, config=cfg)

    # Force a homologous sequence into a different split & fake cluster_id to test pairwise check
    corrupted = split_df.copy()
    ecoli_idx = corrupted.index[corrupted["protein_uid"] == "iML1515__b3236"][0]
    salm_idx = corrupted.index[corrupted["protein_uid"] == "STM_v1_0__STM3359"][0]

    corrupted.loc[ecoli_idx, "split"] = "train"
    corrupted.loc[ecoli_idx, "cluster_id"] = "forced_cluster_train"
    corrupted.loc[salm_idx, "split"] = "test"
    corrupted.loc[salm_idx, "cluster_id"] = "forced_cluster_test"

    assert (
        compute_pairwise_identity(
            str(corrupted.loc[ecoli_idx, "sequence"]),
            str(corrupted.loc[salm_idx, "sequence"]),
        )
        >= 0.30
    )

    with pytest.raises(AssertionError, match="Homology leakage"):
        verify_zero_sequence_leakage(corrupted, max_allowed_identity=0.30)
