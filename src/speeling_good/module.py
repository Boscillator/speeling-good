"""Lightning training module for spelling-error token classification."""

from typing import Any

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    MulticlassAccuracy,
)
from transformers import get_linear_schedule_with_warmup
from transformers.modeling_outputs import TokenClassifierOutput

from speeling_good.models import SpeelingGoodDescriminator


class SpeelingGood(L.LightningModule):
    """Fine-tune ModernBERT; B/I mark errors and O marks clean tokens.

    Checkpoints include the backbone configuration so loading a Lightning
    checkpoint does not download the pretrained weights again. For inference,
    call the module with tokenizer tensors; the result has BIO ``logits``.
    """

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        learning_rate: float = 3e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        gradient_checkpointing: bool = False,
        backbone_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if learning_rate <= 0 or weight_decay < 0 or not 0 <= warmup_ratio <= 1:
            raise ValueError(
                "Require learning_rate > 0, weight_decay >= 0, warmup_ratio in [0, 1]"
            )
        self.discriminator = SpeelingGoodDescriminator(
            model_name=model_name, backbone_config=backbone_config
        )
        if gradient_checkpointing:
            self.discriminator.backbone.gradient_checkpointing_enable()
        self.save_hyperparameters(ignore=["backbone_config"])
        self.hparams["backbone_config"] = self.discriminator.backbone.config.to_dict()
        self.validation_metrics = MetricCollection(
            {
                "bio_accuracy": MulticlassAccuracy(num_classes=3, average="micro"),
                "error_precision": BinaryPrecision(),
                "error_recall": BinaryRecall(),
                "error_f1": BinaryF1Score(),
            },
            prefix="val/",
        )

    def forward(self, **inputs: torch.Tensor) -> TokenClassifierOutput:
        return self.discriminator(**inputs)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self(**batch).loss
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=int((batch["labels"] != -100).sum()),
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        output = self(**batch)
        mask = batch["labels"] != -100
        labels = batch["labels"][mask]
        predictions = output.logits.argmax(dim=-1)[mask]
        self.log(
            "val/loss",
            output.loss,
            on_epoch=True,
            batch_size=labels.numel(),
            sync_dist=True,
        )
        self.validation_metrics["bio_accuracy"].update(predictions, labels)
        for name in ("error_precision", "error_recall", "error_f1"):
            self.validation_metrics[name].update(predictions != 0, labels != 0)
        self.log_dict(self.validation_metrics, on_epoch=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        decay, no_decay = [], []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                (
                    no_decay if parameter.ndim < 2 or name.endswith("bias") else decay
                ).append(parameter)
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.hparams["weight_decay"]},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.hparams["learning_rate"],
        )
        steps = int(self.trainer.estimated_stepping_batches)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(steps * self.hparams["warmup_ratio"]),
            num_training_steps=steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
