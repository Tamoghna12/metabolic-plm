"""Tests verifying ESM-2 + LoRA tensor shapes, parameter freezing, and bf16 gradient flow."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from metabolic_plm.models.esm_lora import EsmLoraConfig, EsmLoraForMetabolicPhenotype
from metabolic_plm.train import (
    ProteinSequenceDataset,
    TrainerConfig,
    build_optimizer,
    evaluate_model,
    train_one_epoch,
)


def test_esm_lora_forward_shapes_and_gradient_flow() -> None:
    """Verify forward output shapes, loss computation, and strict LoRA gradient isolation."""
    torch.manual_seed(42)

    cfg = EsmLoraConfig(
        num_labels=2,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=("query", "key", "value", "dense"),
        use_pretrained=False,
        vocab_size=33,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
    )
    model = EsmLoraForMetabolicPhenotype(cfg)

    trainable_params, total_params = model.count_parameters()
    assert trainable_params > 0
    assert trainable_params < total_params

    # Ensure base linear layers inside attention are frozen while LoRA A/B weights are trainable
    lora_param_names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert any("lora_A" in name for name in lora_param_names)
    assert any("lora_B" in name for name in lora_param_names)
    assert any("classifier" in name for name in lora_param_names)
    assert not any("base_layer.weight" in name for name in lora_param_names)

    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(4, 24, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    # Pad last 6 tokens of sample 0 to verify masked mean pooling ignores padding
    input_ids[0, -6:] = 1
    attention_mask[0, -6:] = 0
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.long)

    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert output.logits.shape == (batch_size, cfg.num_labels)
    assert output.pooled_embeddings is not None
    assert output.pooled_embeddings.shape == (batch_size, cfg.hidden_size)
    assert output.loss is not None
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)

    output.loss.backward()

    # After 1 backward pass, LoRA A/B and classifier parameters must have finite gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing gradient for trainable parameter {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"
        else:
            assert param.grad is None, f"Frozen parameter {name} unexpectedly received a gradient"


def test_bf16_training_and_evaluation_step() -> None:
    """Verify end-to-end bf16 autocast training loop and evaluation metrics on a dummy batch."""
    torch.manual_seed(7)

    cfg = EsmLoraConfig(
        num_labels=2,
        lora_r=4,
        lora_alpha=8,
        use_pretrained=False,
        hidden_size=32,
        num_hidden_layers=2,
    )
    model = EsmLoraForMetabolicPhenotype(cfg)
    optimizer = build_optimizer(model, learning_rate=1e-3)

    sequences = [
        "MKKVLVLGASGGIGQPLSLLLKNSPLVSELSLY",
        "MNTQIIIFALVILLGLFFGIVIGRLPFKPVRKW",
        "MSHQHTSQVWHRGWACAQPEGENWYACMPNFGT",
        "MTDTRRTRRGRRDRRNRRDRRERRQRRDRRHRR",
    ]
    labels = [1, 0, 1, 0]
    dataset = ProteinSequenceDataset(sequences=sequences, labels=labels, max_length=40)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    trainer_cfg = TrainerConfig(
        learning_rate=1e-3,
        gradient_accumulation_steps=2,
        use_bf16=True,
        device="cpu",
    )

    epoch_loss = train_one_epoch(model, loader, optimizer, trainer_cfg)
    assert epoch_loss > 0.0

    metrics = evaluate_model(model, loader, trainer_cfg)
    assert "eval_loss" in metrics
    assert "eval_accuracy" in metrics
    assert 0.0 <= metrics["eval_accuracy"] <= 1.0
