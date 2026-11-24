"""Main training & ablation loop for ESM-2 (650M) + LoRA (bf16) on 12GB RTX 4070.

Implements the "Leakage vs. Reality" Ablation Study:
  - Run A (Naive Baseline): Random 70/15/15 split (`split.strategy=random`)
  - Run B (Homology-Split): MMseqs2 30% identity split (`split.strategy=homology`)

Key Engineering & Optimization Specifications:
  - Backbone: `facebook/esm2_t33_650M_UR50D`
  - Precision & Memory: `bf16` mixed precision + `gradient_checkpointing=True`
  - LoRA: `r=8`, `lora_alpha=16`, `target_modules=["query", "key", "value"]`, `lora_dropout=0.1`
  - Batching: `per_device_batch_size=4`, `grad_accumulation_steps=8` (effective batch size = 32),
    `max_length=1022` residues (+ `<cls>` and `<eos>`) with dynamic padding per batch
  - Optimization: `AdamW`, `lr=3e-4`, cosine schedule with 10% warmup, `pos_weight=5.06`
  - Systems Telemetry Logged Every Epoch (and to W&B):
      * Peak VRAM (GB): `torch.cuda.max_memory_allocated() / 1e9`
      * Throughput: `samples_per_sec` & `tokens_per_sec` (non-padding tokens / epoch wall time)
      * Classification Metrics: `loss`, `auprc` (Average Precision), `auroc` (ROC-AUC), `accuracy`
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from metabolic_plm.data.split import (
    SplitConfig,
    create_homology_aware_splits,
    create_random_splits,
)
from metabolic_plm.models.esm_lora import (
    ClassifierOutput,
    EsmLoraConfig,
    EsmLoraForMetabolicPhenotype,
)

logger = logging.getLogger(__name__)

# Standard ESM-2 amino acid vocabulary indices (cls=0, pad=1, eos=2, unk=3)
ESM2_AA_TO_ID: dict[str, int] = {
    "<cls>": 0,
    "<pad>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "L": 4,
    "A": 5,
    "G": 6,
    "V": 7,
    "S": 8,
    "E": 9,
    "R": 10,
    "T": 11,
    "I": 12,
    "D": 13,
    "P": 14,
    "K": 15,
    "Q": 16,
    "N": 17,
    "F": 18,
    "Y": 19,
    "M": 20,
    "H": 21,
    "W": 22,
    "C": 23,
    "X": 24,
    "B": 25,
    "U": 26,
    "Z": 27,
    "O": 28,
}


@dataclass(frozen=True)
class TrainerConfig:
    """Hyperparameters for bf16 LoRA fine-tuning on a 12GB RTX 4070."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    num_train_epochs: int = 5
    warmup_ratio: float = 0.10
    pos_weight: float = 5.06
    max_length: int = 1022
    per_device_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    use_bf16: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class EpochTelemetry:
    """Training and hardware systems metrics captured at the end of an epoch."""

    train_loss: float
    peak_vram_gb: float
    epoch_time_sec: float
    samples_per_sec: float
    tokens_per_sec: float
    total_samples: int
    total_tokens: int


class ProteinSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Tokenizes protein sequences up to `max_length` residues (default 1022 + <cls>/<eos>)."""

    def __init__(
        self,
        sequences: list[str],
        labels: list[int],
        max_length: int = 1022,
        dynamic_padding: bool = True,
    ) -> None:
        if len(sequences) != len(labels):
            raise ValueError("sequences and labels must have identical length")
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        self.dynamic_padding = dynamic_padding

    def __len__(self) -> int:
        return len(self.sequences)

    @staticmethod
    def tokenize_unpadded(seq: str, max_residues: int = 1022) -> list[int]:
        """Convert an amino acid string to unpadded ESM-2 token IDs `[<cls>, ..., <eos>]`."""
        clean = seq.strip().upper()[:max_residues]
        token_ids = [ESM2_AA_TO_ID["<cls>"]]
        token_ids.extend(ESM2_AA_TO_ID.get(aa, ESM2_AA_TO_ID["<unk>"]) for aa in clean)
        token_ids.append(ESM2_AA_TO_ID["<eos>"])
        return token_ids

    @staticmethod
    def encode_sequence(seq: str, max_length: int = 1022) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an amino acid string with fixed padding up to `max_length + 2` tokens."""
        token_ids = ProteinSequenceDataset.tokenize_unpadded(seq, max_residues=max_length)
        target_len = max_length + 2
        valid_len = len(token_ids)
        pad_len = max(0, target_len - valid_len)
        padded_ids = token_ids + [ESM2_AA_TO_ID["<pad>"]] * pad_len
        attention_mask = [1] * valid_len + [0] * pad_len

        return (
            torch.tensor(padded_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.dynamic_padding:
            token_ids = self.tokenize_unpadded(self.sequences[index], max_residues=self.max_length)
            return {
                "input_ids": torch.tensor(token_ids, dtype=torch.long),
                "attention_mask": torch.ones(len(token_ids), dtype=torch.long),
                "labels": torch.tensor(self.labels[index], dtype=torch.long),
            }

        input_ids, attention_mask = self.encode_sequence(
            self.sequences[index],
            max_length=self.max_length,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def dynamic_pad_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Dynamically pad variable-length protein token sequences to the longest sequence in batch."""
    max_batch_len = max(int(item["input_ids"].size(0)) for item in batch)
    pad_id = ESM2_AA_TO_ID["<pad>"]

    batch_size = len(batch)
    padded_ids = torch.full((batch_size, max_batch_len), fill_value=pad_id, dtype=torch.long)
    padded_mask = torch.zeros((batch_size, max_batch_len), dtype=torch.long)
    labels = torch.empty(batch_size, dtype=torch.long)

    for idx, item in enumerate(batch):
        seq_len = int(item["input_ids"].size(0))
        padded_ids[idx, :seq_len] = item["input_ids"]
        padded_mask[idx, :seq_len] = item["attention_mask"]
        labels[idx] = item["labels"]

    return {
        "input_ids": padded_ids,
        "attention_mask": padded_mask,
        "labels": labels,
    }


def build_optimizer(
    model: EsmLoraForMetabolicPhenotype,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
) -> AdamW:
    """Create AdamW optimizer over trainable LoRA and classification head parameters only."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("Model has zero trainable parameters.")
    return AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)


def build_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """Create a cosine learning rate schedule with linear warmup (default 10% warmup)."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch_with_telemetry(
    model: EsmLoraForMetabolicPhenotype,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    cfg: TrainerConfig,
    scheduler: LambdaLR | None = None,
) -> EpochTelemetry:
    """Execute one training epoch with bf16 mixed precision, grad accumulation, and telemetry."""
    model.train()
    device = torch.device(cfg.device)
    model.to(device)

    device_type = "cuda" if device.type == "cuda" else "cpu"
    use_amp = cfg.use_bf16 and (device_type == "cpu" or torch.cuda.is_bf16_supported())

    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    num_batches = 0
    total_samples = 0
    total_tokens = 0

    for step, batch in enumerate(dataloader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        total_samples += int(labels.size(0))
        total_tokens += int(attention_mask.sum().item())

        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
            out: ClassifierOutput = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            if out.loss is None:
                raise RuntimeError("Model forward pass did not return a loss tensor.")
            scaled_loss = out.loss / float(cfg.gradient_accumulation_steps)

        scaled_loss.backward()
        total_loss += float(out.loss.detach().cpu().item())
        num_batches += 1

        if step % cfg.gradient_accumulation_steps == 0 or step == len(dataloader):
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=cfg.max_grad_norm,
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        peak_vram_gb = float(torch.cuda.max_memory_allocated(device)) / 1e9
    else:
        peak_vram_gb = 0.0

    elapsed = max(1e-9, time.perf_counter() - start_time)
    return EpochTelemetry(
        train_loss=total_loss / float(max(1, num_batches)),
        peak_vram_gb=peak_vram_gb,
        epoch_time_sec=elapsed,
        samples_per_sec=float(total_samples) / elapsed,
        tokens_per_sec=float(total_tokens) / elapsed,
        total_samples=total_samples,
        total_tokens=total_tokens,
    )


def train_one_epoch(
    model: EsmLoraForMetabolicPhenotype,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    cfg: TrainerConfig,
    scheduler: LambdaLR | None = None,
) -> float:
    """Execute one training epoch and return mean training loss (backwards-compatible helper)."""
    telemetry = train_one_epoch_with_telemetry(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        cfg=cfg,
        scheduler=scheduler,
    )
    return telemetry.train_loss


@torch.no_grad()
def evaluate_model(
    model: EsmLoraForMetabolicPhenotype,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    cfg: TrainerConfig,
) -> dict[str, float]:
    """Evaluate loss, accuracy, AUPRC (Average Precision), and AUROC on a DataLoader."""
    model.eval()
    device = torch.device(cfg.device)
    model.to(device)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    use_amp = cfg.use_bf16 and (device_type == "cpu" or torch.cuda.is_bf16_supported())

    total_loss = 0.0
    correct = 0
    total = 0
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=use_amp):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            if out.loss is not None:
                total_loss += float(out.loss.detach().cpu().item()) * labels.size(0)

        if model.config.num_labels == 1:
            probs = torch.sigmoid(out.logits.squeeze(-1).float())
            preds = (probs >= 0.5).long()
        else:
            probs = torch.softmax(out.logits.float(), dim=-1)[:, 1]
            preds = torch.argmax(out.logits, dim=-1)

        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))
        all_probs.append(probs.detach().cpu().numpy())
        all_targets.append(labels.detach().cpu().numpy())

    denom = float(max(1, total))
    y_prob = np.concatenate(all_probs) if all_probs else np.array([])
    y_true = np.concatenate(all_targets) if all_targets else np.array([])

    auprc = 0.0
    auroc = 0.5
    if len(y_true) > 0 and len(np.unique(y_true)) > 1:
        auprc = float(average_precision_score(y_true, y_prob))
        auroc = float(roc_auc_score(y_true, y_prob))

    return {
        "eval_loss": total_loss / denom,
        "eval_accuracy": correct / denom,
        "eval_auprc": auprc,
        "eval_auroc": auroc,
    }


def run_ablation_experiment(
    cfg_path: Path = Path("configs/train_esm2_lora.yaml"),
    strategy_override: str | None = None,
    epochs_override: int | None = None,
    model_name_override: str | None = None,
    wandb_mode_override: str | None = None,
) -> dict[str, Any]:
    """Run an end-to-end tracked experiment (Run A: random or Run B: homology) with W&B."""
    from omegaconf import OmegaConf

    raw_cfg: Any = OmegaConf.load(cfg_path)
    seed = int(raw_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    strategy = strategy_override or str(raw_cfg.split.get("strategy", "homology"))
    if strategy not in {"random", "homology"}:
        raise ValueError(f"Unsupported split.strategy: {strategy!r}. Use 'random' or 'homology'.")

    dataset_path = Path(str(raw_cfg.paths.processed_dataset))
    splits_dir = Path(str(raw_cfg.paths.splits_dir))
    splits_dir.mkdir(parents=True, exist_ok=True)
    split_file = splits_dir / f"{strategy}_split.parquet"

    split_cfg = SplitConfig(
        min_seq_id=float(raw_cfg.split.min_seq_id),
        coverage=float(raw_cfg.split.coverage),
        cov_mode=int(raw_cfg.split.cov_mode),
        train_ratio=float(raw_cfg.split.train_ratio),
        val_ratio=float(raw_cfg.split.val_ratio),
        test_ratio=float(raw_cfg.split.test_ratio),
        seed=seed,
        mmseqs_binary=str(raw_cfg.split.mmseqs_binary),
    )

    if split_file.exists():
        logger.info("Loading cached %s split from %s", strategy, split_file)
        df = pd.read_parquet(split_file)
    else:
        base_df = pd.read_parquet(dataset_path)
        id_col = "record_id" if "record_id" in base_df.columns else "protein_uid"
        if strategy == "random":
            df = create_random_splits(base_df, id_col=id_col, config=split_cfg)
        else:
            df = create_homology_aware_splits(base_df, id_col=id_col, config=split_cfg)
        df.to_parquet(split_file, index=False)
        logger.info("Saved %s split to %s", strategy, split_file)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    max_length = int(raw_cfg.model.get("max_length", 1022))
    train_ds = ProteinSequenceDataset(
        train_df["sequence"].tolist(),
        train_df["is_essential"].astype(int).tolist(),
        max_length=max_length,
        dynamic_padding=True,
    )
    val_ds = ProteinSequenceDataset(
        val_df["sequence"].tolist(),
        val_df["is_essential"].astype(int).tolist(),
        max_length=max_length,
        dynamic_padding=True,
    )
    test_ds = ProteinSequenceDataset(
        test_df["sequence"].tolist(),
        test_df["is_essential"].astype(int).tolist(),
        max_length=max_length,
        dynamic_padding=True,
    )

    batch_size = int(
        raw_cfg.training.get(
            "per_device_batch_size",
            raw_cfg.training.get("per_device_train_batch_size", 4),
        )
    )
    eval_batch_size = int(raw_cfg.training.get("per_device_eval_batch_size", 8))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dynamic_pad_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=dynamic_pad_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=dynamic_pad_collate_fn,
    )

    pos_weight = float(raw_cfg.training.get("pos_weight", 5.06))
    pretrained_name = model_name_override or str(raw_cfg.model.pretrained_model_name)
    model_cfg = EsmLoraConfig(
        pretrained_model_name=pretrained_name,
        num_labels=int(raw_cfg.model.get("num_labels", 1)),
        lora_r=int(raw_cfg.model.lora.r),
        lora_alpha=int(raw_cfg.model.lora.lora_alpha),
        lora_dropout=float(raw_cfg.model.lora.lora_dropout),
        target_modules=tuple(raw_cfg.model.lora.target_modules),
        classifier_dropout=float(raw_cfg.model.classifier_dropout),
        pos_weight=pos_weight,
        gradient_checkpointing=bool(raw_cfg.model.gradient_checkpointing),
    )

    model = EsmLoraForMetabolicPhenotype(model_cfg)
    trainable_params, total_params = model.count_parameters()
    logger.info(
        "[%s] Initialized %s | Trainable params: %s / %s (%.2f%%)",
        strategy.upper(),
        pretrained_name,
        f"{trainable_params:,}",
        f"{total_params:,}",
        100.0 * trainable_params / max(1, total_params),
    )

    epochs = epochs_override or int(
        raw_cfg.training.get("epochs", raw_cfg.training.get("num_train_epochs", 5))
    )
    grad_accum = int(
        raw_cfg.training.get(
            "grad_accumulation_steps",
            raw_cfg.training.get("gradient_accumulation_steps", 8),
        )
    )
    lr = float(raw_cfg.training.get("lr", raw_cfg.training.get("learning_rate", 3e-4)))
    warmup_ratio = float(raw_cfg.training.get("warmup_ratio", 0.10))

    trainer_cfg = TrainerConfig(
        learning_rate=lr,
        weight_decay=float(raw_cfg.training.get("weight_decay", 0.01)),
        gradient_accumulation_steps=grad_accum,
        max_grad_norm=float(raw_cfg.training.get("max_grad_norm", 1.0)),
        num_train_epochs=epochs,
        warmup_ratio=warmup_ratio,
        pos_weight=pos_weight,
        max_length=max_length,
        per_device_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        use_bf16=str(raw_cfg.training.get("precision", "bf16")) == "bf16",
    )

    optimizer = build_optimizer(
        model,
        learning_rate=trainer_cfg.learning_rate,
        weight_decay=trainer_cfg.weight_decay,
    )
    steps_per_epoch = math.ceil(len(train_loader) / float(grad_accum))
    total_opt_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = int(round(total_opt_steps * warmup_ratio))
    scheduler = build_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps,
    )

    wandb_enabled = bool(raw_cfg.wandb.get("enabled", False))
    wandb_mode = wandb_mode_override or str(raw_cfg.wandb.get("mode", "online"))
    wandb_run = None
    if wandb_enabled and wandb_mode != "disabled":
        import wandb

        run_name = f"{strategy}_split_esm2_lora_r{model_cfg.lora_r}"
        wandb_run = wandb.init(
            project=str(raw_cfg.wandb.get("project", "metabolic-plm")),
            name=run_name,
            mode=wandb_mode,
            config={
                "split_strategy": strategy,
                "model": asdict(model_cfg),
                "trainer": asdict(trainer_cfg),
                "trainable_params": trainable_params,
                "total_params": total_params,
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size": len(test_df),
                "num_clusters": int(df["cluster_id"].nunique()),
            },
        )

    history: list[dict[str, float]] = []
    best_val_auprc = -1.0
    out_dir = Path(f"outputs/{strategy}_esm2_lora")
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        telemetry = train_one_epoch_with_telemetry(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            cfg=trainer_cfg,
            scheduler=scheduler,
        )
        val_metrics = evaluate_model(model, val_loader, trainer_cfg)

        epoch_record = {
            "epoch": float(epoch),
            "train_loss": telemetry.train_loss,
            "val_loss": val_metrics["eval_loss"],
            "val_auprc": val_metrics["eval_auprc"],
            "val_auroc": val_metrics["eval_auroc"],
            "val_accuracy": val_metrics["eval_accuracy"],
            "peak_vram_gb": telemetry.peak_vram_gb,
            "samples_per_sec": telemetry.samples_per_sec,
            "tokens_per_sec": telemetry.tokens_per_sec,
            "epoch_time_sec": telemetry.epoch_time_sec,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)

        logger.info(
            "[%s | Epoch %d/%d] train_loss=%.4f | val_loss=%.4f | val_AUPRC=%.4f | val_AUROC=%.4f "
            "| Peak VRAM=%.2f GB | Throughput=%.1f samples/s (%.0f tok/s)",
            strategy.upper(),
            epoch,
            epochs,
            telemetry.train_loss,
            val_metrics["eval_loss"],
            val_metrics["eval_auprc"],
            val_metrics["eval_auroc"],
            telemetry.peak_vram_gb,
            telemetry.samples_per_sec,
            telemetry.tokens_per_sec,
        )

        if wandb_run is not None:
            wandb_run.log(epoch_record, step=epoch)

        if val_metrics["eval_auprc"] > best_val_auprc:
            best_val_auprc = val_metrics["eval_auprc"]

    test_metrics = evaluate_model(model, test_loader, trainer_cfg)
    summary = {
        "strategy": strategy,
        "pretrained_model_name": pretrained_name,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "num_clusters": int(df["cluster_id"].nunique()),
        "best_val_auprc": best_val_auprc,
        "test_loss": test_metrics["eval_loss"],
        "test_auprc": test_metrics["eval_auprc"],
        "test_auroc": test_metrics["eval_auroc"],
        "test_accuracy": test_metrics["eval_accuracy"],
        "peak_vram_gb": max((h["peak_vram_gb"] for h in history), default=0.0),
        "mean_samples_per_sec": float(np.mean([h["samples_per_sec"] for h in history])),
        "mean_tokens_per_sec": float(np.mean([h["tokens_per_sec"] for h in history])),
        "history": history,
    }

    logger.info(
        "[%s TEST RESULTS] test_AUPRC=%.4f | test_AUROC=%.4f | test_acc=%.4f | Peak VRAM=%.2f GB",
        strategy.upper(),
        summary["test_auprc"],
        summary["test_auroc"],
        summary["test_accuracy"],
        summary["peak_vram_gb"],
    )

    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "test_loss": summary["test_loss"],
                "test_auprc": summary["test_auprc"],
                "test_auroc": summary["test_auroc"],
                "test_accuracy": summary["test_accuracy"],
                "peak_vram_gb": summary["peak_vram_gb"],
                "mean_samples_per_sec": summary["mean_samples_per_sec"],
                "mean_tokens_per_sec": summary["mean_tokens_per_sec"],
            }
        )
        wandb_run.finish()

    summary_path = out_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    """CLI entrypoint for running Run A (random) or Run B (homology) ESM-2 LoRA training."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(
        description="Train ESM-2 LoRA metabolic essentiality classifier (Run A vs Run B)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_esm2_lora.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--strategy",
        choices=["random", "homology"],
        default=None,
        help="Override split strategy: 'random' (Run A) or 'homology' (Run B).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override Hugging Face pretrained model ID (e.g., facebook/esm2_t33_650M_UR50D).",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default=None,
        help="Override Weights & Biases mode.",
    )
    args = parser.parse_args()

    run_ablation_experiment(
        cfg_path=args.config,
        strategy_override=args.strategy,
        epochs_override=args.epochs,
        model_name_override=args.model_name,
        wandb_mode_override=args.wandb_mode,
    )


if __name__ == "__main__":
    main()
