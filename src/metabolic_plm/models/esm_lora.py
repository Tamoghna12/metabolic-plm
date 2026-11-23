"""ESM-2 (650M) with PEFT/LoRA adapters and masked-mean-pooled classification head.

Optimized for 12GB VRAM (NVIDIA RTX 4070) using:
  - `bfloat16` mixed precision
  - Activation / gradient checkpointing on encoder layers
  - Low-Rank Adaptation (LoRA) targeting self-attention projections (`query`, `key`, `value`)
  - Masked mean pooling over non-padding residue embeddings before classification
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EsmLoraConfig:
    """Configuration for ESM-2 + LoRA metabolic phenotype classifier."""

    pretrained_model_name: str = "facebook/esm2_t33_650M_UR50D"
    num_labels: int = 2
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = field(
        default_factory=lambda: ("query", "key", "value", "dense")
    )
    classifier_dropout: float = 0.10
    gradient_checkpointing: bool = True
    use_pretrained: bool = True
    # Lightweight synthetic backbone settings when use_pretrained=False (unit tests / CI)
    vocab_size: int = 33
    hidden_size: int = 1280
    num_hidden_layers: int = 2
    num_attention_heads: int = 8


@dataclass
class ClassifierOutput:
    """Standardized forward output containing logits, optional loss, and pooled embeddings."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None
    pooled_embeddings: torch.Tensor | None = None


class LoRALinear(nn.Module):
    """Parameter-efficient Low-Rank Adaptation (LoRA) wrapper around a frozen `nn.Linear`.

    Implements W0*x + (alpha / r) * B * A * x where W0 is frozen and only low-rank A and B
    receive gradients.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank r must be positive, got {r}")

        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = float(lora_alpha) / float(r)
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        self.lora_A = nn.Parameter(torch.empty((r, in_features), device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r), device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1)) @ self.lora_B.transpose(
            0, 1
        )
        return base_out + self.scaling * lora_out


class _SyntheticEsmAttention(nn.Module):
    """Self-attention module with standard ESM-2 names (`query`, `key`, `value`, `dense`)."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.scale = float(hidden_size) ** -0.5

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        mask = attention_mask.unsqueeze(1).to(dtype=scores.dtype)
        scores = scores.masked_fill(mask == 0, -1e4)
        probs = torch.softmax(scores, dim=-1)
        context = torch.matmul(probs, v)
        return self.dense(context)


class _SyntheticEsmBackbone(nn.Module):
    """Lightweight offline ESM-2-compatible encoder for fast unit and CI testing."""

    def __init__(self, cfg: EsmLoraConfig) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=1)
        self.layers = nn.ModuleList(
            [_SyntheticEsmAttention(cfg.hidden_size) for _ in range(cfg.num_hidden_layers)]
        )
        self.layer_norm = nn.LayerNorm(cfg.hidden_size)
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> Any:
        if attention_mask is None:
            attention_mask = (input_ids != 1).long()
        hidden = self.embeddings(input_ids)
        for layer in self.layers:
            hidden = hidden + layer(hidden, attention_mask)
        hidden = self.layer_norm(hidden)

        class _BackboneOut:
            def __init__(self, last_hidden_state: torch.Tensor) -> None:
                self.last_hidden_state = last_hidden_state

        return _BackboneOut(hidden)


def inject_lora_adapters(
    backbone: nn.Module,
    target_modules: tuple[str, ...],
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> nn.Module:
    """Freeze all backbone weights and wrap matching `nn.Linear` layers with LoRA adapters.

    Uses Hugging Face `peft.get_peft_model` when the backbone is a `PreTrainedModel`
    and `peft` is installed; otherwise injects `LoRALinear` directly on matching modules.
    """
    for param in backbone.parameters():
        param.requires_grad = False

    try:
        from peft import LoraConfig, get_peft_model
        from transformers import PreTrainedModel

        if isinstance(backbone, PreTrainedModel):
            peft_cfg = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=list(target_modules),
            )
            return get_peft_model(backbone, peft_cfg)
    except ImportError:
        pass

    target_set = set(target_modules)
    for parent_name, parent_module in list(backbone.named_modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if child_name in target_set and isinstance(child_module, nn.Linear):
                setattr(
                    parent_module,
                    child_name,
                    LoRALinear(
                        child_module,
                        r=r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                    ),
                )
                logger.debug("Injected LoRALinear into %s.%s", parent_name, child_name)

    return backbone


class EsmLoraForMetabolicPhenotype(nn.Module):
    """ESM-2 encoder with LoRA adapters and a masked-mean-pooled classification head."""

    def __init__(self, config: EsmLoraConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = EsmLoraConfig()
        self.config = config

        if config.use_pretrained:
            from transformers import AutoConfig, AutoModel

            hf_cfg = AutoConfig.from_pretrained(config.pretrained_model_name)
            hidden_size = int(hf_cfg.hidden_size)
            backbone = AutoModel.from_pretrained(config.pretrained_model_name, config=hf_cfg)
        else:
            hidden_size = config.hidden_size
            backbone = _SyntheticEsmBackbone(config)

        if config.gradient_checkpointing and hasattr(backbone, "gradient_checkpointing_enable"):
            backbone.gradient_checkpointing_enable()

        self.encoder = inject_lora_adapters(
            backbone=backbone,
            target_modules=config.target_modules,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=config.classifier_dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(p=config.classifier_dropout),
            nn.Linear(hidden_size, config.num_labels),
        )

    @staticmethod
    def masked_mean_pool(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean residue embedding across non-padded tokens `(B, L, D) -> (B, D)`."""
        mask_expanded = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        summed = torch.sum(hidden_states * mask_expanded, dim=1)
        counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return summed / counts

    def count_parameters(self) -> tuple[int, int]:
        """Return `(trainable_params, total_params)` to verify parameter efficiency."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> ClassifierOutput:
        """Run encoder + masked mean pooling + classification head."""
        if attention_mask is None:
            attention_mask = (input_ids != 1).long()

        encoder_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = encoder_out.last_hidden_state
        pooled = self.masked_mean_pool(last_hidden, attention_mask)
        logits = self.classifier(pooled)

        loss: torch.Tensor | None = None
        if labels is not None:
            if self.config.num_labels == 1:
                loss_fn = nn.BCEWithLogitsLoss()
                loss = loss_fn(logits.squeeze(-1), labels.float())
            else:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels.long())

        return ClassifierOutput(
            logits=logits,
            loss=loss,
            pooled_embeddings=pooled,
        )
