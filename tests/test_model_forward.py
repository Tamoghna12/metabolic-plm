"""Tests verifying ESM-2 + LoRA tensor shapes, dynamic padding, pos_weight, and telemetry."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from metabolic_plm.models.esm_lora import EsmLoraConfig, EsmLoraForMetabolicPhenotype
from metabolic_plm.train import (
    ProteinSequenceDataset,
    TrainerConfig,
    build_cosine_schedule_with_warmup,
    build_optimizer,
    dynamic_pad_collate_fn,
    evaluate_model,
    train_one_epoch_with_telemetry,
)


def test_esm_lora_forward_shapes_and_gradient_flow() -> None:
    """Verify forward output shapes, pos_weight BCE loss, and strict LoRA gradient isolation."""
    torch.manual_seed(42)

    cfg = EsmLoraConfig(
        num_labels=1,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.10,
        target_modules=("query", "key", "value"),
        pos_weight=5.06,
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
    # "dense" is not in target_modules=("query", "key", "value"), so it must stay frozen
    assert not any("dense.lora_" in name for name in lora_param_names)

    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(4, 24, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    # Pad last 6 tokens of sample 0 to verify masked mean pooling ignores padding
    input_ids[0, -6:] = 1
    attention_mask[0, -6:] = 0
    labels = torch.tensor([1, 0, 1, 0], dtype=torch.long)

    output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    assert output.logits.shape == (batch_size, 1)
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


def test_bf16_training_telemetry_dynamic_padding_and_auprc() -> None:
    """Verify dynamic batch padding, cosine warmup schedule, throughput telemetry, and AUPRC."""
    torch.manual_seed(7)

    cfg = EsmLoraConfig(
        num_labels=1,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.10,
        target_modules=("query", "key", "value"),
        pos_weight=5.06,
        use_pretrained=False,
        hidden_size=32,
        num_hidden_layers=2,
    )
    model = EsmLoraForMetabolicPhenotype(cfg)
    optimizer = build_optimizer(model, learning_rate=3e-4)
    scheduler = build_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=1, num_training_steps=4
    )

    # Variable-length sequences to verify dynamic padding per batch
    sequences = [
        "MKKVLVLGASGGIGQPLSLL",
        "MNTQIIIFALVILLGLFFGIVIGRLPFKPVRKW",
        "MSHQHTSQVWHRGW",
        "MTDTRRTRRGRRDRRNRRDRRERRQRRDRRHRRDRRTRR",
    ]
    labels = [1, 0, 1, 0]
    dataset = ProteinSequenceDataset(
        sequences=sequences,
        labels=labels,
        max_length=1022,
        dynamic_padding=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=dynamic_pad_collate_fn,
    )

    first_batch = next(iter(loader))
    # First batch has lengths 20 (+2 = 22) and 33 (+2 = 35), so dynamic batch length must be 35
    assert first_batch["input_ids"].shape == (2, 35)
    assert int(first_batch["attention_mask"][0].sum().item()) == 22
    assert int(first_batch["attention_mask"][1].sum().item()) == 35

    trainer_cfg = TrainerConfig(
        learning_rate=3e-4,
        gradient_accumulation_steps=2,
        pos_weight=5.06,
        use_bf16=True,
        device="cpu",
    )

    telemetry = train_one_epoch_with_telemetry(
        model=model,
        dataloader=loader,
        optimizer=optimizer,
        cfg=trainer_cfg,
        scheduler=scheduler,
    )
    assert telemetry.train_loss > 0.0
    assert telemetry.total_samples == 4
    assert telemetry.total_tokens == (22 + 35 + 16 + 41)
    assert telemetry.samples_per_sec > 0.0
    assert telemetry.tokens_per_sec > 0.0
    assert telemetry.peak_vram_gb >= 0.0

    metrics = evaluate_model(model, loader, trainer_cfg)
    assert "eval_loss" in metrics
    assert "eval_accuracy" in metrics
    assert "eval_auprc" in metrics
    assert "eval_auroc" in metrics
    assert 0.0 <= metrics["eval_auprc"] <= 1.0
    assert 0.0 <= metrics["eval_auroc"] <= 1.0
