# metabolic-plm

Production-grade, containerized Python package that fine-tunes **ESM-2 (650M)** (`facebook/esm2_t33_650M_UR50D`) with **PEFT/LoRA** and **bfloat16 (`bf16`)** mixed precision on a **12GB NVIDIA RTX 4070** to predict microbial metabolic phenotypes (single-gene deletion essentiality and metabolic subsystem participation validated via **COBRApy Flux Balance Analysis**), using strict **MMseqs2 30% sequence identity** homology-aware train/validation/test splits.

---

## 1. Repository Architecture

```text
metabolic-plm/
├── .github/
│   └── workflows/
│       └── ci.yml              # Runs ruff linting & pytest on every push
├── configs/
│   └── train_esm2_lora.yaml    # Hydra config (LR, LoRA rank, batch size, paths)
├── data/                       # Git-ignored; raw & processed datasets
├── docker/
│   └── Dockerfile              # Reproducible CUDA + PyTorch + COBRApy + MMseqs2 image
├── src/
│   └── metabolic_plm/
│       ├── __init__.py
│       ├── data/
│       │   ├── ingest.py       # Pulls BiGG/AGORA2 models & maps gene IDs to sequences
│       │   ├── fba_labels.py   # Runs COBRApy gene knockout / flux essentiality
│       │   └── split.py        # Wraps MMseqs2 clustering (30% seq identity splits)
│       ├── models/
│       │   └── esm_lora.py     # ESM-2 + PEFT/LoRA + classification head
│       └── train.py            # Main training loop (bf16, grad checkpointing, W&B)
├── tests/
│   ├── test_smoke.py           # Verifies curated BiGG model list configuration
│   ├── test_splits.py          # Verifies ZERO sequence leakage between train/val/test
│   └── test_model_forward.py   # Verifies tensor shapes & gradient flow on dummy batch
├── pyproject.toml              # Dependencies, ruff config, pytest config
└── README.md                   # Architecture, benchmarks, and reproducibility guide
```

---

## 2. Scoped Dataset Choice (BiGG Curated Bacterial GEMs)

Instead of downloading all 7,302 AGORA2 models in Week 1, `metabolic_plm.data.ingest` pulls a curated set of well-annotated bacterial Genome-Scale Metabolic Models (GEMs) from **BiGG Models**:
- `iML1515` & `iAF1260` (*Escherichia coli* str. K-12 substr. MG1655)
- `iYO844` (*Bacillus subtilis* subsp. *subtilis* str. 168)
- `iJN1463` (*Pseudomonas putida* KT2440)
- `STM_v1_0` (*Salmonella enterica* serovar Typhimurium str. LT2)
- `iSB619` (*Staphylococcus aureus* subsp. *aureus* N315)
- `iNJ661` (*Mycobacterium tuberculosis* H37Rv)
- `iIT341` (*Helicobacter pylori* 26695)

These models provide clean Gene-Protein-Reaction (GPR) boolean rules and standardized UniProt/NCBI cross-references, enabling single-gene knockout Flux Balance Analysis (`cobra.flux_analysis.single_gene_deletion`) across 15,000–20,000 labeled protein sequences in minutes.

---

## 3. Quickstart & Reproducibility

### Local Setup with `uv`

```bash
# 1. Create virtual environment & install package in editable mode
uv venv
uv pip install -e ".[dev]"

# 2. Run Ruff linter and Pytest suite
uv run ruff check src tests
uv run pytest

# 3. Download curated BiGG JSON models into data/raw/bigg/
uv run python -m metabolic_plm.data.ingest
```

### Hardware Optimization (12GB NVIDIA RTX 4070)

Configured in `configs/train_esm2_lora.yaml`:
- **Base Model**: `facebook/esm2_t33_650M_UR50D` (33 layers, 1280 hidden dim)
- **Precision**: `bf16` mixed precision (`torch.amp.autocast`)
- **Memory Saving**: Activation/gradient checkpointing enabled (`gradient_checkpointing: true`)
- **LoRA Adapters**: Rank $r = 16$, $\alpha = 32$, dropout $0.05$ on attention modules (`query`, `key`, `value`, `dense`)
- **Effective Batch Size**: 32 (`per_device_train_batch_size: 4`, `gradient_accumulation_steps: 8`)
