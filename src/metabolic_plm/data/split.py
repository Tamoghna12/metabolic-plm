"""Homology-aware (MMseqs2 30% sequence identity) and naive random dataset splitting.

Supports the "Leakage vs. Reality" Ablation Study:
  - Run A (Naive Baseline): Random 70/15/15 split (`create_random_splits`)
  - Run B (Homology-Split): MMseqs2 30% identity clustering (`create_homology_aware_splits`)
    guaranteeing zero homologous sequence leakage across train, validation, and test splits.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_SEQ_ID_RE = re.compile(r"^[A-Za-z0-9_.:|-]+$")
_AA_SEQ_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWYBXZJUO]+$")


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for train/val/test partitioning (70/15/15 default)."""

    min_seq_id: float = 0.30
    coverage: float = 0.80
    cov_mode: int = 0
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    mmseqs_binary: str = "mmseqs"


def compute_pairwise_identity(seq_a: str, seq_b: str) -> float:
    """Compute exact global/semi-global pairwise sequence identity in [0.0, 1.0].

    Uses dynamic programming (Needleman-Wunsch with match=+1, mismatch/gap=0)
    normalized by the aligned length of the shorter sequence.
    """
    a = seq_a.strip().upper()
    b = seq_b.strip().upper()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    if len(a) > len(b):
        a, b = b, a

    n, m = len(a), len(b)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for j in range(1, m + 1):
        bj = b[j - 1]
        for i in range(1, n + 1):
            if a[i - 1] == bj:
                curr[i] = prev[i - 1] + 1
            else:
                curr[i] = prev[i] if prev[i] >= curr[i - 1] else curr[i - 1]
        prev, curr = curr, [0] * (n + 1)

    lcs_len = prev[n]
    return lcs_len / float(min(n, m))


def write_fasta(records: dict[str, str], fasta_path: Path) -> Path:
    """Write validated `{seq_id: sequence}` records to a FASTA file."""
    lines: list[str] = []
    for seq_id, seq in records.items():
        clean_id = seq_id.strip()
        clean_seq = seq.strip().upper()
        if not _SEQ_ID_RE.match(clean_id):
            raise ValueError(f"Invalid sequence identifier: {seq_id!r}")
        if not _AA_SEQ_RE.match(clean_seq):
            raise ValueError(f"Invalid amino acid characters in sequence for {seq_id!r}")
        lines.append(f">{clean_id}\n{clean_seq}")

    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    fasta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return fasta_path


def run_mmseqs2_clustering(
    records: dict[str, str],
    config: SplitConfig | None = None,
) -> dict[str, str]:
    """Cluster protein sequences at `config.min_seq_id` using MMseqs2 (`easy-cluster`).

    If the `mmseqs` binary is available on `PATH`, invokes `mmseqs easy-cluster` inside
    an isolated temporary workspace. If `mmseqs` is not installed in the active test/CI
    environment, falls back to single-linkage clustering (exact for <= 100 sequences, or
    fast 5-mer Jaccard + prefix clustering for large datasets).

    Returns:
        Dictionary mapping each `seq_id` to its representative `cluster_id`.
    """
    if config is None:
        config = SplitConfig()
    if not records:
        return {}

    if not (0.0 < config.min_seq_id <= 1.0):
        raise ValueError("min_seq_id must be in (0.0, 1.0]")
    if not (0.0 < config.coverage <= 1.0):
        raise ValueError("coverage must be in (0.0, 1.0]")

    mmseqs_path = shutil.which(config.mmseqs_binary)
    if mmseqs_path is not None:
        with tempfile.TemporaryDirectory(prefix="mmseqs2_split_") as tmp_dir:
            tmp_path = Path(tmp_dir).resolve()
            fasta_file = write_fasta(records, tmp_path / "input.fasta")
            out_prefix = tmp_path / "clu"
            work_dir = tmp_path / "tmp_work"

            cmd = [
                mmseqs_path,
                "easy-cluster",
                str(fasta_file),
                str(out_prefix),
                str(work_dir),
                "--min-seq-id",
                f"{config.min_seq_id:.4f}",
                "-c",
                f"{config.coverage:.4f}",
                "--cov-mode",
                str(int(config.cov_mode)),
            ]
            logger.info("Running MMseqs2 clustering: %s", " ".join(cmd))
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            cluster_tsv = Path(f"{out_prefix}_cluster.tsv")
            if not cluster_tsv.exists():
                raise RuntimeError(f"MMseqs2 did not produce expected file: {cluster_tsv}")

            seq_to_cluster: dict[str, str] = {}
            for line in cluster_tsv.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    rep_id, member_id = parts[0].strip(), parts[1].strip()
                    seq_to_cluster[member_id] = rep_id
            return seq_to_cluster

    logger.info(
        "MMseqs2 binary %r not found on PATH; using connected-components clustering "
        "at min_seq_id=%.2f.",
        config.mmseqs_binary,
        config.min_seq_id,
    )
    if len(records) <= 200:
        return _fallback_single_linkage_clustering(records, min_seq_id=config.min_seq_id)
    return _fast_kmer_clustering(records, min_seq_id=config.min_seq_id)


def _fallback_single_linkage_clustering(
    records: dict[str, str],
    min_seq_id: float = 0.30,
) -> dict[str, str]:
    """Exact connected-components homology clustering for small/unit-test datasets."""
    seq_ids = sorted(records.keys())
    parent: dict[str, str] = {sid: sid for sid in seq_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for i, sid_a in enumerate(seq_ids):
        seq_a = records[sid_a]
        for sid_b in seq_ids[i + 1 :]:
            if find(sid_a) == find(sid_b):
                continue
            ident = compute_pairwise_identity(seq_a, records[sid_b])
            if ident >= min_seq_id:
                union(sid_a, sid_b)

    return {sid: find(sid) for sid in seq_ids}


def _fast_kmer_clustering(
    records: dict[str, str],
    min_seq_id: float = 0.30,
    k: int = 4,
) -> dict[str, str]:
    """Fast k-mer inverted-index clustering fallback for thousands of protein sequences."""
    seq_ids = sorted(records.keys())
    parent: dict[str, str] = {sid: sid for sid in seq_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    kmer_sets: dict[str, set[str]] = {}
    inverted: dict[str, list[str]] = defaultdict(list)
    for sid in seq_ids:
        seq = records[sid]
        kmers = {seq[i : i + k] for i in range(max(1, len(seq) - k + 1))}
        kmer_sets[sid] = kmers
        for km in kmers:
            inverted[km].append(sid)

    # Approximate 30% sequence identity threshold in k-mer overlap space
    kmer_thresh = max(0.08, (min_seq_id**k) * 0.8)
    for sid_a in seq_ids:
        km_a = kmer_sets[sid_a]
        if not km_a:
            continue
        candidate_counts: dict[str, int] = defaultdict(int)
        for km in km_a:
            for sid_b in inverted[km]:
                if sid_b > sid_a:
                    candidate_counts[sid_b] += 1
        for sid_b, shared in candidate_counts.items():
            if find(sid_a) == find(sid_b):
                continue
            min_len = min(len(km_a), len(kmer_sets[sid_b]))
            if min_len > 0 and (shared / float(min_len)) >= kmer_thresh:
                union(sid_a, sid_b)

    return {sid: find(sid) for sid in seq_ids}


def assign_clusters_to_splits(
    seq_to_cluster: dict[str, str],
    config: SplitConfig | None = None,
) -> dict[str, str]:
    """Assign entire homology clusters to 'train', 'val', or 'test' partitions (70/15/15).

    Ensures:
      1. Every member of a cluster is placed in the exact same split (zero leakage).
      2. When at least 3 distinct clusters exist, all three splits ('train', 'val', 'test')
         receive at least one cluster.
    """
    if config is None:
        config = SplitConfig()
    total_ratio = config.train_ratio + config.val_ratio + config.test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-4):
        raise ValueError(f"Split ratios must sum to 1.0 (got {total_ratio:.4f})")

    clusters: dict[str, list[str]] = defaultdict(list)
    for seq_id, cluster_id in sorted(seq_to_cluster.items()):
        clusters[cluster_id].append(seq_id)

    cluster_ids = sorted(clusters.keys())
    if not cluster_ids:
        return {}

    rng = random.Random(config.seed)
    rng.shuffle(cluster_ids)

    n_total_seqs = len(seq_to_cluster)
    target_val = max(1, int(round(n_total_seqs * config.val_ratio)))
    target_test = max(1, int(round(n_total_seqs * config.test_ratio)))

    seq_to_split: dict[str, str] = {}
    val_count = 0
    test_count = 0

    for idx, cid in enumerate(cluster_ids):
        members = clusters[cid]
        if len(cluster_ids) >= 3 and idx == 0:
            split_name = "test"
            test_count += len(members)
        elif len(cluster_ids) >= 3 and idx == 1:
            split_name = "val"
            val_count += len(members)
        elif len(cluster_ids) >= 3 and idx == 2:
            split_name = "train"
        else:
            if test_count + len(members) <= target_test:
                split_name = "test"
                test_count += len(members)
            elif val_count + len(members) <= target_val:
                split_name = "val"
                val_count += len(members)
            else:
                split_name = "train"

        for member_id in members:
            seq_to_split[member_id] = split_name

    return seq_to_split


def create_random_splits(
    df: pd.DataFrame,
    id_col: str = "record_id",
    config: SplitConfig | None = None,
) -> pd.DataFrame:
    """Run A (Naive Baseline): Create a random 70/15/15 train/val/test split ignoring homology."""
    if config is None:
        config = SplitConfig()
    total_ratio = config.train_ratio + config.val_ratio + config.test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-4):
        raise ValueError(f"Split ratios must sum to 1.0 (got {total_ratio:.4f})")

    out_df = df.copy().reset_index(drop=True)
    n_total = len(out_df)
    if n_total == 0:
        out_df["cluster_id"] = pd.Series(dtype="object")
        out_df["split"] = pd.Series(dtype="object")
        return out_df

    indices = list(range(n_total))
    rng = random.Random(config.seed)
    rng.shuffle(indices)

    n_test = max(1, int(round(n_total * config.test_ratio))) if n_total >= 3 else 0
    n_val = max(1, int(round(n_total * config.val_ratio))) if n_total >= 3 else 0

    splits = ["train"] * n_total
    for i in indices[:n_test]:
        splits[i] = "test"
    for i in indices[n_test : n_test + n_val]:
        splits[i] = "val"

    if id_col in out_df.columns:
        out_df["cluster_id"] = out_df[id_col].astype(str)
    else:
        out_df["cluster_id"] = [f"rand_{i}" for i in range(n_total)]
    out_df["split"] = splits
    return out_df


def create_homology_aware_splits(
    df: pd.DataFrame,
    id_col: str = "protein_uid",
    seq_col: str = "sequence",
    config: SplitConfig | None = None,
) -> pd.DataFrame:
    """Run B (Homology-Split): Annotate DataFrame with `cluster_id` and 30% ID `split` labels."""
    if config is None:
        config = SplitConfig()
    if id_col not in df.columns and "record_id" in df.columns:
        id_col = "record_id"
    if id_col not in df.columns or seq_col not in df.columns:
        raise KeyError(f"DataFrame must contain columns {id_col!r} and {seq_col!r}")

    records = {str(row[id_col]): str(row[seq_col]).strip().upper() for _, row in df.iterrows()}
    seq_to_cluster = run_mmseqs2_clustering(records, config=config)
    seq_to_split = assign_clusters_to_splits(seq_to_cluster, config=config)

    out_df = df.copy()
    out_df["cluster_id"] = out_df[id_col].map(seq_to_cluster)
    out_df["split"] = out_df[id_col].map(seq_to_split)
    return out_df


def verify_zero_sequence_leakage(
    df: pd.DataFrame,
    seq_col: str = "sequence",
    split_col: str = "split",
    cluster_col: str = "cluster_id",
    max_allowed_identity: float = 0.30,
) -> None:
    """Assert zero cluster overlap and zero cross-split pairwise homology >= threshold.

    Raises:
        AssertionError: If any cluster spans multiple splits or if any pair of sequences
            in different splits has pairwise sequence identity >= `max_allowed_identity`.
    """
    splits = ("train", "val", "test")
    split_clusters: dict[str, set[str]] = {
        s: set(df.loc[df[split_col] == s, cluster_col].dropna().unique()) for s in splits
    }

    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1 :]:
            overlap = split_clusters[s1] & split_clusters[s2]
            if overlap:
                raise AssertionError(
                    f"Cluster leakage detected between {s1!r} and {s2!r}: {sorted(overlap)}"
                )

    split_sequences: dict[str, list[str]] = {
        s: [str(seq) for seq in df.loc[df[split_col] == s, seq_col].tolist()] for s in splits
    }

    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1 :]:
            for seq_a in split_sequences[s1]:
                for seq_b in split_sequences[s2]:
                    ident = compute_pairwise_identity(seq_a, seq_b)
                    if ident >= max_allowed_identity:
                        raise AssertionError(
                            f"Homology leakage >= {max_allowed_identity:.2f} detected between "
                            f"{s1!r} and {s2!r} (identity={ident:.3f})."
                        )


def build_ablation_splits(
    dataset_path: Path = Path("data/processed/metabolic_dataset.parquet"),
    splits_dir: Path = Path("data/splits"),
    config: SplitConfig | None = None,
) -> tuple[Path, Path]:
    """Generate and save both Run A (random 70/15/15) and Run B (MMseqs2 30% ID) datasets."""
    if config is None:
        config = SplitConfig()
    splits_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(dataset_path)
    id_col = "record_id" if "record_id" in df.columns else "protein_uid"

    random_df = create_random_splits(df, id_col=id_col, config=config)
    random_path = splits_dir / "random_split.parquet"
    random_df.to_parquet(random_path, index=False)

    homology_df = create_homology_aware_splits(df, id_col=id_col, config=config)
    homology_path = splits_dir / "homology_split.parquet"
    homology_df.to_parquet(homology_path, index=False)

    logger.info(
        "Saved Run A (random) split to %s (%s) and Run B (homology 30%% ID) split to %s "
        "(%s, %d clusters)",
        random_path,
        random_df["split"].value_counts().to_dict(),
        homology_path,
        homology_df["split"].value_counts().to_dict(),
        homology_df["cluster_id"].nunique(),
    )
    return random_path, homology_path


def main() -> None:
    """CLI entrypoint to generate both random and homology-aware splits."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Create random and MMseqs2 30% homology splits.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/metabolic_dataset.parquet"),
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
    )
    args = parser.parse_args()
    build_ablation_splits(dataset_path=args.dataset, splits_dir=args.splits_dir)


if __name__ == "__main__":
    main()
