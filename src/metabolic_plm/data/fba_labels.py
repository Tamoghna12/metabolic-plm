"""Generate COBRApy FBA essentiality labels and map BiGG genes to UniProt sequences."""

from __future__ import annotations

import logging
import time
from io import StringIO
from pathlib import Path

import cobra
import pandas as pd
import requests
from cobra.flux_analysis import single_gene_deletion

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("cobra.core.model").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

UNIPROT_API = "https://rest.uniprot.org/idmapping"


def extract_gene_labels(model_path: Path, essential_threshold: float = 0.05) -> pd.DataFrame:
    """Run single-gene deletion FBA on a BiGG JSON model and extract annotations."""
    model = cobra.io.load_json_model(str(model_path))
    model.solver.configuration.tolerances.integrality = 1e-7
    wt_solution = model.optimize()
    wt_growth = float(wt_solution.objective_value)

    if wt_growth <= 1e-6:
        logger.warning("Model %s has near-zero WT growth (%.6f); skipping.", model.id, wt_growth)
        return pd.DataFrame()

    logger.info("Running single-gene deletions for %s (WT growth: %.4f)...", model.id, wt_growth)
    deletions = single_gene_deletion(model)

    # Map frozenset({gene_id}) -> (growth, status)
    del_map: dict[str, tuple[float, str]] = {}
    for row in deletions.itertuples():
        gene_ids = list(row.ids)
        if len(gene_ids) == 1:
            growth = 0.0 if pd.isna(row.growth) else float(row.growth)
            del_map[gene_ids[0]] = (growth, str(row.status))

    records: list[dict[str, object]] = []
    for gene in model.genes:
        if gene.id not in del_map:
            continue
        ko_growth, status = del_map[gene.id]
        growth_ratio = ko_growth / wt_growth
        is_essential = int(status != "optimal" or growth_ratio < essential_threshold)

        # Extract UniProt or NCBI Gene cross-references from BiGG annotations
        uniprot_ids = gene.annotation.get("uniprot", [])
        ncbigene_ids = gene.annotation.get("ncbigene", [])
        uniprot_id = (
            uniprot_ids[0]
            if isinstance(uniprot_ids, list) and uniprot_ids
            else (uniprot_ids if isinstance(uniprot_ids, str) else None)
        )
        ncbigene_id = (
            ncbigene_ids[0]
            if isinstance(ncbigene_ids, list) and ncbigene_ids
            else (ncbigene_ids if isinstance(ncbigene_ids, str) else None)
        )

        # Capture primary metabolic subsystem for optional multi-task learning
        subsystems = sorted({rxn.subsystem for rxn in gene.reactions if rxn.subsystem})
        records.append(
            {
                "model_id": model.id,
                "gene_id": gene.id,
                "uniprot_id": uniprot_id,
                "ncbigene_id": ncbigene_id,
                "wt_growth": wt_growth,
                "ko_growth": ko_growth,
                "growth_ratio": growth_ratio,
                "is_essential": is_essential,
                "primary_subsystem": subsystems[0] if subsystems else "Unassigned",
            }
        )

    return pd.DataFrame(records)


def fetch_uniprot_sequences(
    ids: list[str],
    from_db: str = "UniProtKB_AC-ID",
    batch_size: int = 500,
) -> dict[str, str]:
    """Batch-map database IDs to amino acid sequences via UniProt REST API."""
    unique_ids = sorted({str(x) for x in ids if pd.notna(x) and str(x).strip()})
    seq_map: dict[str, str] = {}

    for start in range(0, len(unique_ids), batch_size):
        batch = unique_ids[start : start + batch_size]
        logger.info("Submitting UniProt batch %d–%d (%s)...", start, start + len(batch), from_db)

        resp = requests.post(
            f"{UNIPROT_API}/run",
            data={"from": from_db, "to": "UniProtKB", "ids": ",".join(batch)},
            timeout=60,
        )
        resp.raise_for_status()
        job_id = resp.json()["jobId"]

        # Poll until job finishes
        while True:
            status_resp = requests.get(f"{UNIPROT_API}/status/{job_id}", timeout=60)
            status_resp.raise_for_status()
            payload = status_resp.json()
            if "results" in payload or "failedIds" in payload:
                break
            if payload.get("jobStatus") == "RUNNING":
                time.sleep(2)
            else:
                break

        # Stream results as TSV containing input ID, accession, and sequence
        stream_url = (
            f"{UNIPROT_API}/uniprotkb/results/stream/{job_id}?format=tsv&fields=accession,sequence"
        )
        tsv_resp = requests.get(stream_url, timeout=120)
        tsv_resp.raise_for_status()

        if tsv_resp.text.strip():
            df_res = pd.read_csv(StringIO(tsv_resp.text), sep="\t")
            if "From" in df_res.columns and "Sequence" in df_res.columns:
                for row in df_res.itertuples():
                    src_id = str(row.From)
                    if src_id not in seq_map and isinstance(row.Sequence, str):
                        seq_map[src_id] = row.Sequence

    return seq_map


def build_dataset(
    raw_dir: Path = Path("data/raw/bigg"),
    processed_dir: Path = Path("data/processed"),
) -> Path:
    """Run FBA across all downloaded BiGG models, fetch sequences, and save artifacts."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    model_paths = sorted(raw_dir.glob("*.json"))
    if not model_paths:
        raise FileNotFoundError(f"No BiGG JSON models found in {raw_dir}. Run ingest.py first.")

    dfs = [extract_gene_labels(p) for p in model_paths]
    df = pd.concat([d for d in dfs if not d.empty], ignore_index=True)

    # 1. Fetch via UniProt IDs where available
    uniprot_seqs = fetch_uniprot_sequences(
        df["uniprot_id"].dropna().tolist(), from_db="UniProtKB_AC-ID"
    )
    df["sequence"] = df["uniprot_id"].map(uniprot_seqs)

    # 2. Fallback to NCBI Gene IDs for unmapped rows
    missing_mask = df["sequence"].isna() & df["ncbigene_id"].notna()
    if missing_mask.any():
        ncbi_seqs = fetch_uniprot_sequences(
            df.loc[missing_mask, "ncbigene_id"].tolist(), from_db="GeneID"
        )
        df.loc[missing_mask, "sequence"] = df.loc[missing_mask, "ncbigene_id"].map(ncbi_seqs)

    total_genes = len(df)
    df = df.dropna(subset=["sequence"]).copy()
    df["seq_len"] = df["sequence"].str.len()
    df["record_id"] = df["model_id"] + "|" + df["gene_id"]

    # Save Parquet + FASTA (for Day 4 MMseqs2 clustering)
    parquet_path = processed_dir / "metabolic_dataset.parquet"
    fasta_path = processed_dir / "sequences.fasta"
    df.to_parquet(parquet_path, index=False)

    with fasta_path.open("w", encoding="utf-8") as fh:
        for row in df.itertuples():
            fh.write(f">{row.record_id}\n{row.sequence}\n")

    essential_pct = 100.0 * df["is_essential"].mean()
    logger.info(
        "Saved %d/%d mapped genes (%.1f%% essential, median len %d AA) to %s",
        len(df),
        total_genes,
        essential_pct,
        int(df["seq_len"].median()),
        parquet_path,
    )
    return parquet_path


if __name__ == "__main__":
    build_dataset()
