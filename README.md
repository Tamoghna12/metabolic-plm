# metabolic-plm

Production-grade, containerized Python package that fine-tunes **ESM-2 (650M)** (`facebook/esm2_t33_650M_UR50D`) with **PEFT/LoRA** and **bfloat16 (`bf16`)** mixed precision on a **12GB NVIDIA RTX 4070** to predict microbial metabolic phenotypes (single-gene deletion essentiality validated via **COBRApy Flux Balance Analysis**), using strict **30% sequence identity** homology-aware train/validation/test splits.

---

## 1. The "Leakage vs. Reality" Ablation Study (Empirical Results)

Standard random splits on multi-species microbial datasets leak orthologous genes (e.g., *E. coli* `iML1515` vs. *S. enterica* `STM_v1_0` vs. *P. putida* `iJN1463`) across train and test sets, allowing protein language models to memorize sequence homology rather than learning metabolic essentiality.

We quantified this exact generalization gap across **3,416 FBA-labeled protein sequences** (`565` essential [16.54%], `2,851` non-essential [83.46%], `pos_weight=5.06`, `2,392` train / `512` val / `512` test) over 5 epochs on a single **NVIDIA GeForce RTX 4070 (12GB)**:

| Experiment | Split Strategy | Distinct Clusters | Best Val AUPRC | Test AUPRC | Test AUROC | Test Accuracy | What It Proves |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Run A (Naive Baseline)** | Random 70/15/15 split (`--strategy random`) | 3,416 (Unclustered) | **0.7212** | **0.6629** | **0.8960** | **84.96%** | Memorizing near-identical orthologs across related bacterial species yields an artificially inflated **0.6629 Test AUPRC** and **0.8960 AUROC**. |
| **Run B (Homology-Split)** | 30% sequence identity split (`--strategy homology`) | 1,336 homology families | **0.5277** | **0.4130** | **0.7539** | **81.84%** | True out-of-distribution generalization to unseen protein families (<30% identity) drops Test AUPRC by **-24.99 percentage points (-37.7% relative)** and Test AUROC by **-14.21 points**, exposing the true difficulty of zero-shot metabolic phenotype prediction. |

### Epoch-by-Epoch Generalization Dynamics

| Epoch | Run A (Random) Train Loss | Run A Val AUPRC | Run A Val AUROC | Run B (30% Homology) Train Loss | Run B Val AUPRC | Run B Val AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1.0552 | 0.5408 | 0.8359 | 1.0823 | 0.4993 | 0.8464 |
| **2** | 0.7855 | 0.6400 | 0.8746 | 0.7659 | **0.5277** | **0.8586** |
| **3** | 0.5416 | 0.6690 | 0.8950 | 0.4602 | 0.4482 | 0.8302 |
| **4** | 0.3609 | 0.7109 | 0.9104 | 0.2517 | 0.4191 | 0.8090 |
| **5** | 0.2740 | **0.7212** | **0.9135** | 0.1723 | 0.4079 | 0.7980 |

> **Key Takeaway**: Under the naive random split (Run A), validation AUPRC climbs monotonically through Epoch 5 (`0.5408 -> 0.7212`) as the model memorizes homologous sequence families shared between train and validation. Under the strict 30% homology split (Run B), out-of-distribution generalization peaks at **Epoch 2 (`0.5277` Val AUPRC, `0.8586` Val AUROC)** before overfitting to training-family sequence motifs in Epochs 3–5.

---

## 2. Engineering & Hardware Scaling Benchmarks (NVIDIA RTX 4070 12GB)

Fine-tuning a 650M-parameter Transformer in full FP32 requires >24GB VRAM. By combining `bfloat16` mixed precision, activation gradient checkpointing, rank-8 LoRA adapters on attention projections (`query`, `key`, `value`), and per-batch dynamic sequence padding (`max_length=1022`), the entire training loop runs in **under 4.0 GB of VRAM** on a consumer 12GB RTX 4070:

| Metric | Run A (`random`) | Run B (`homology`) | Configuration Details |
| :--- | :---: | :---: | :--- |
| **Backbone Checkpoint** | `esm2_t33_650M_UR50D` | `esm2_t33_650M_UR50D` | 33 layers, 1,280 hidden dim |
| **Total Parameters** | 654,709,142 | 654,709,142 | Frozen base backbone in `bf16` |
| **Trainable Parameters** | **3,668,481 (0.56%)** | **3,668,481 (0.56%)** | LoRA ($r=8, \alpha=16$, `q/k/v`) + MLP classifier head |
| **Peak Allocated VRAM** | **3.96 GB** / 12.0 GB | **3.95 GB** / 12.0 GB | Leaves >8 GB VRAM headroom for larger batch sizes |
| **Mean Throughput (Samples/s)** | **10.34 samples/sec** | **10.32 samples/sec** | Micro-batch = 4, Grad Accum = 8 (Effective Batch = 32) |
| **Mean Throughput (Tokens/s)** | **3,985.9 tokens/sec** | **3,990.5 tokens/sec** | Dynamic batch padding (`dynamic_pad_collate_fn`) |
| **Mean Epoch Wall Time** | **231.4 sec (~3.85 min)** | **231.9 sec (~3.86 min)** | ~19.3 minutes per 5-epoch ablation run |

---

## 3. Repository Structure

```text
metabolic-plm/
├── .github/
│   └── workflows/
│       └── ci.yml                          # GitHub Actions CI (ruff check/format + pytest)
├── configs/
│   └── train_esm2_lora.yaml                # Hydra config (ESM-2 650M, LoRA r=8, bf16, RTX 4070)
├── data/                                   # Git-ignored datasets & splits
│   ├── raw/
│   │   └── bigg/                           # 8 curated BiGG bacterial GEMs (JSON)
│   ├── processed/
│   │   ├── metabolic_dataset.parquet       # 3,416 FBA-labeled proteins mapped to UniProt/NCBI
│   │   └── sequences.fasta                 # FASTA sequences for MMseqs2 clustering
│   └── splits/
│       ├── random_split.parquet            # Run A: Naive 70/15/15 random split
│       └── homology_split.parquet          # Run B: 30% sequence identity homology split
├── docker/
│   └── Dockerfile                          # Reproducible CUDA 12.4 + PyTorch + COBRApy + MMseqs2 image
├── outputs/                                # Git-ignored training & ablation telemetry
│   ├── random_esm2_lora/
│   │   └── ablation_summary.json           # Run A metrics & RTX 4070 hardware telemetry
│   └── homology_esm2_lora/
│       └── ablation_summary.json           # Run B metrics & RTX 4070 hardware telemetry
├── src/
│   └── metabolic_plm/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── ingest.py                   # Downloads curated BiGG bacterial GEMs
│       │   ├── fba_labels.py               # Single-gene deletion FBA + UniProt sequence mapping
│       │   └── split.py                    # Random & MMseqs2 30% identity clustering splits
│       ├── models/
│       │   ├── __init__.py
│       │   └── esm_lora.py                 # ESM-2 (650M) + PEFT/LoRA + masked mean pooling head
│       └── train.py                        # bf16 training loop, dynamic padding, W&B & GPU telemetry
├── tests/
│   ├── test_smoke.py                       # Curated BiGG model list test
│   ├── test_fba_labels.py                  # FBA essentiality threshold logic test
│   ├── test_splits.py                      # Verifies ZERO sequence leakage across splits
│   └── test_model_forward.py               # Verifies tensor shapes, dynamic padding & gradient flow
├── wandb/                                  # Git-ignored Weights & Biases run logs
├── .gitignore
├── pyproject.toml                          # Package dependencies, ruff & pytest configuration
├── README.md                               # Architecture, ablation results & reproducibility guide
└── uv.lock                                 # Deterministic uv dependency lockfile
```

---

## 4. Scoped Dataset (8 Curated Bacterial GEMs from BiGG)

Instead of downloading all 7,302 AGORA2 models in Week 1, `metabolic_plm.data.ingest` pulls 8 high-quality bacterial Genome-Scale Metabolic Models (GEMs) from **BiGG Models**:
- `iML1515` & `iAF1260` (*Escherichia coli* str. K-12 substr. MG1655)
- `iYO844` (*Bacillus subtilis* subsp. *subtilis* str. 168)
- `iJN1463` (*Pseudomonas putida* KT2440)
- `STM_v1_0` (*Salmonella enterica* serovar Typhimurium str. LT2)
- `iSB619` (*Staphylococcus aureus* subsp. *aureus* N315)
- `iNJ661` (*Mycobacterium tuberculosis* H37Rv)
- `iIT341` (*Helicobacter pylori* 26695)

`metabolic_plm.data.fba_labels` executes single-gene deletion Flux Balance Analysis (`cobra.flux_analysis.single_gene_deletion`) with GLPK `integrality = 1e-7`, labels genes as essential (`is_essential = 1`) when knockout biomass flux drops below 5% of wild-type (`growth_ratio < 0.05` or non-optimal solver status), and maps BiGG gene annotations to amino acid sequences via the UniProt ID Mapping REST API (`UniProtKB_AC-ID` with `GeneID` fallback).

---

## 5. End-to-End Reproducibility Guide

```bash
# 1. Create virtual environment & install dependencies via uv
uv venv
uv pip install -e ".[dev]"

# 2. Run Ruff linter, format check, and Pytest suite
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest

# 3. Download curated BiGG JSON models into data/raw/bigg/
uv run python -m metabolic_plm.data.ingest

# 4. Run COBRApy single-gene deletion FBA & fetch UniProt sequences
uv run python -m metabolic_plm.data.fba_labels

# 5. Build Run A (random 70/15/15) & Run B (30% homology 70/15/15) splits
uv run python -m metabolic_plm.data.split

# 6. Execute Run A (Naive Baseline) & Run B (Homology-Split) on RTX 4070
uv run python -m metabolic_plm.train --strategy random
uv run python -m metabolic_plm.train --strategy homology
```
