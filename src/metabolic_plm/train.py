"""Main training loop for ESM-2 (650M) + LoRA with bf16 mixed precision on 12GB RTX 4070.

Features:
  - Hydra configuration support (`configs/train_esm2_lora.yaml`)
  - Automatic `bfloat16` autocast (`torch.amp.autocast`) on Ampere/Ada GPUs (RTX 4070)
  - Gradient accumulation & gradient clipping for stable effective batch size = 32
  - Optional Weights & Biases (`wandb`) experiment tracking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

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
    """Hyperparameters for bf16 LoRA fine-tuning on a 12GB GPU."""

    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    num_train_epochs: int = 10
    use_bf16: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ProteinSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Tokenizes protein sequences into fixed-length padded ESM-2 token tensors."""

    def __init__(
        self,
        sequences: list[str],
        labels: list[int],
        max_length: int = 1024,
    ) -> None:
        if len(sequences) != len(labels):
            raise ValueError("sequences and labels must have identical length")
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sequences)

    @staticmethod
    def encode_sequence(seq: str, max_length: int = 1024) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an amino acid string using the ESM-2 token mapping with `<cls>` and `<eos>`."""
        clean = seq.strip().upper()[: max_length - 2]
        token_ids = [ESM2_AA_TO_ID["<cls>"]]
        token_ids.extend(ESM2_AA_TO_ID.get(aa, ESM2_AA_TO_ID["<unk>"]) for aa in clean)
        token_ids.append(ESM2_AA_TO_ID["<eos>"])

        valid_len = len(token_ids)
        pad_len = max(0, max_length - valid_len)
        padded_ids = token_ids + [ESM2_AA_TO_ID["<pad>"]] * pad_len
        attention_mask = [1] * valid_len + [0] * pad_len

        return (
            torch.tensor(padded_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids, attention_mask = self.encode_sequence(
            self.sequences[index],
            max_length=self.max_length,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


def train_one_epoch(
    model: EsmLoraForMetabolicPhenotype,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    cfg: TrainerConfig,
) -> float:
    """Execute one training epoch with bf16 mixed precision and gradient accumulation."""
    model.train()
    device = torch.device(cfg.device)
    model.to(device)

    device_type = "cuda" if device.type == "cuda" else "cpu"
    use_amp = cfg.use_bf16 and (device_type == "cpu" or torch.cuda.is_bf16_supported())

    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    num_batches = 0

    for step, batch in enumerate(dataloader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

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
            optimizer.zero_grad(set_to_none=True)

    return total_loss / float(max(1, num_batches))


@torch.no_grad()
def evaluate_model(
    model: EsmLoraForMetabolicPhenotype,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    cfg: TrainerConfig,
) -> dict[str, float]:
    """Evaluate classification loss and accuracy on a validation or test DataLoader."""
    model.eval()
    device = torch.device(cfg.device)
    model.to(device)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    use_amp = cfg.use_bf16 and (device_type == "cpu" or torch.cuda.is_bf16_supported())

    total_loss = 0.0
    correct = 0
    total = 0

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

        preds = torch.argmax(out.logits, dim=-1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))

    denom = float(max(1, total))
    return {
        "eval_loss": total_loss / denom,
        "eval_accuracy": correct / denom,
    }


def build_optimizer(
    model: EsmLoraForMetabolicPhenotype,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
) -> AdamW:
    """Create AdamW optimizer over trainable LoRA and classification head parameters only."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("Model has zero trainable parameters.")
    return AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)


def main(cfg_path: Path = Path("configs/train_esm2_lora.yaml")) -> None:
    """Entrypoint for Hydra-configured ESM-2 LoRA training."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    from omegaconf import OmegaConf

    raw_cfg: Any = OmegaConf.load(cfg_path)
    model_cfg = EsmLoraConfig(
        pretrained_model_name=str(raw_cfg.model.pretrained_model_name),
        num_labels=int(raw_cfg.model.num_labels),
        lora_r=int(raw_cfg.model.lora.r),
        lora_alpha=int(raw_cfg.model.lora.lora_alpha),
        lora_dropout=float(raw_cfg.model.lora.lora_dropout),
        target_modules=tuple(raw_cfg.model.lora.target_modules),
        classifier_dropout=float(raw_cfg.model.classifier_dropout),
        gradient_checkpointing=bool(raw_cfg.model.gradient_checkpointing),
    )
    logger.info("Loaded EsmLoraConfig: %s", model_cfg)


if __name__ == "__main__":
    main()
