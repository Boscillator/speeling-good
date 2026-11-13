"""Lightning training module for spelling-error token classification."""

from typing import Any

import lightning as L
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryF1Score,
    BinaryPrecision,
    BinaryPrecisionRecallCurve,
    BinaryRecall,
    MulticlassAccuracy,
)
from transformers import get_linear_schedule_with_warmup
from transformers.modeling_outputs import TokenClassifierOutput

from speeling_good.evaluation import (
    TABLE_COLUMNS,
    SpanDetectionMetrics,
    bio_spans,
    evaluated_word_count,
    passage_evaluation,
)
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
                "misspelling_token_precision": BinaryPrecision(),
                "misspelling_token_recall": BinaryRecall(),
                "misspelling_token_f1": BinaryF1Score(),
            },
            prefix="val/",
        )

        self.training_metrics = MetricCollection(
            {
                "misspelling_token_precision": BinaryPrecision(),
                "misspelling_token_recall": BinaryRecall(),
            },
            prefix="train/",
        )
        self.span_metrics = SpanDetectionMetrics()
        self.token_pr_curve = BinaryPrecisionRecallCurve(thresholds=1001)
        self.evaluation_rows: list[tuple[int, list[Any]]] = []

    def forward(self, **inputs: torch.Tensor) -> TokenClassifierOutput:
        return self.discriminator(**inputs)

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        output = self(**batch)
        loss = output.loss
        mask = batch["labels"] != -100
        self.training_metrics.update(
            output.logits.detach().argmax(dim=-1)[mask] != 0, batch["labels"][mask] != 0
        )
        self.log_dict(self.training_metrics, on_step=False, on_epoch=True)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=int((batch["labels"] != -100).sum()),
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        output = self(
            **{key: value for key, value in batch.items() if key != "metadata"}
        )
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
        for name in (
            "misspelling_token_precision",
            "misspelling_token_recall",
            "misspelling_token_f1",
        ):
            self.validation_metrics[name].update(predictions != 0, labels != 0)
        self.log_dict(self.validation_metrics, on_epoch=True)
        probabilities = output.logits.detach().float().softmax(dim=-1)
        self.token_pr_curve.update(
            probabilities[..., 1:].sum(dim=-1)[mask], (labels != 0).long()
        )
        batch_predictions = output.logits.detach().argmax(dim=-1).cpu().tolist()
        for prediction, target, metadata in zip(
            batch_predictions,
            batch["labels"].cpu().tolist(),
            batch["metadata"],
            strict=True,
        ):
            prediction = [
                value if label != -100 else -100
                for value, label in zip(prediction, target, strict=True)
            ]
            predicted, reference = bio_spans(prediction), bio_spans(target)
            self.span_metrics.update(
                predicted,
                reference,
                evaluated_word_count(
                    metadata["text"], metadata["offset_mapping"], target
                ),
            )
            if reference and len(self.evaluation_rows) < 100:
                self.evaluation_rows.append(
                    (
                        metadata["passage_index"],
                        passage_evaluation(
                            metadata["text"],
                            metadata["offset_mapping"],
                            target,
                            predicted,
                            reference,
                        ),
                    )
                )

    def on_validation_epoch_start(self) -> None:
        self.span_metrics.reset()
        self.token_pr_curve.reset()
        self.evaluation_rows.clear()

    def on_validation_epoch_end(self) -> None:
        # All ranks participate in metric synchronization before rank-zero logging.
        metrics = self.span_metrics.compute()
        self.log_dict(
            {f"val/{name}": value for name, value in metrics.items()}, on_epoch=True
        )
        precision, recall, thresholds = self.token_pr_curve.compute()
        if self.trainer.sanity_checking:
            return
        wandb_loggers = [
            run_logger
            for run_logger in self.loggers
            if isinstance(run_logger, WandbLogger)
        ]
        if not wandb_loggers:
            return
        precision, recall = precision.nan_to_num(0), recall.nan_to_num(0)
        rows = self.evaluation_rows
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered: list[Any] = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, rows)
            rows = [row for rank_rows in gathered for row in rank_rows]
        if not self.trainer.is_global_zero:
            return
        # Deduplicate passages repeated by distributed validation samplers.
        selected = sorted(dict(rows).items())[:100]
        curve = wandb.Table(
            columns=["recall", "precision", "threshold"],
            data=[
                [float(r), float(p), float(t)]
                for p, r, t in zip(
                    precision[:-1].cpu(),
                    recall[:-1].cpu(),
                    thresholds.cpu(),
                    strict=True,
                )
            ]
            + [[float(recall[-1]), float(precision[-1]), None]],
        )
        for run_logger in wandb_loggers:
            run_logger.experiment.log(
                {
                    "val/misspelling_token_pr_curve": wandb.plot.line(
                        curve,
                        x="recall",
                        y="precision",
                        title="Misspelling token precision–recall",
                    ),
                    "val/passage_evaluation": wandb.EvalTable(
                        columns=TABLE_COLUMNS,
                        data=[row for _, row in selected],
                        input_columns=["passage", "evaluated_end"],
                        output_columns=[
                            "predicted_spans",
                            "reference_spans",
                            "marked_passage",
                        ],
                        score_columns=[
                            "true_positives",
                            "false_positives",
                            "false_negatives",
                        ],
                    ),
                },
                # Lightning supplies the step when it commits validation scalars.
                commit=False,
            )

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
