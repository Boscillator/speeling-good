"""Train a BIO spelling discriminator on prepared part-*.jsonl shards."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from speeling_good.dataloaders import create_dataloaders
from speeling_good.module import SpeelingGood

PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str = "answerdotai/ModernBERT-base"
    learning_rate: Annotated[float, Field(gt=0)] = 3e-5
    weight_decay: Annotated[float, Field(ge=0)] = 0.01
    warmup_ratio: Annotated[float, Field(ge=0, le=1)] = 0.1
    gradient_checkpointing: bool = False
    clean_passage_weight: Annotated[float, Field(ge=0, le=1)] = 1.0


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    directory: Path = Path("data/fineweb")
    validation_fraction: Annotated[float, Field(gt=0, lt=1)] = 0.2
    batch_size: PositiveInt = 2
    max_length: PositiveInt = 128
    num_workers: Annotated[int, Field(strict=True, ge=0)] = 0


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    accelerator: str = "auto"
    devices: PositiveInt = 1
    precision: str = "32-true"
    max_epochs: PositiveInt = 1
    max_steps: Annotated[int, Field(strict=True, ge=-1)] = 2
    limit_val_batches: PositiveInt | Annotated[float, Field(gt=0, le=1)] = 2
    val_check_interval: PositiveInt | Annotated[float, Field(gt=0, le=1)] = 2
    accumulate_grad_batches: PositiveInt = 1
    gradient_clip_val: Annotated[float, Field(ge=0)] = 1.0
    log_every_n_steps: PositiveInt = 1
    num_sanity_val_steps: Annotated[int, Field(strict=True, ge=0)] = 1
    fast_dev_run: bool = False


class EarlyStoppingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    patience: Annotated[int, Field(strict=True, ge=0)] = 3
    min_delta: Annotated[float, Field(ge=0)] = 0.0


class WandbConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project: Annotated[str, Field(min_length=1)] = "speeling-good"
    entity: Annotated[str, Field(min_length=1)] | None = None
    name: Annotated[str, Field(min_length=1)] | None = None


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = 42
    output: Path = Path("runs/discriminator")
    resume_from: Path | None = None
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)


@contextmanager
def tracking_run(
    config: TrainConfig, run_dir: Path, run_id: str
) -> Generator[WandbLogger | Literal[False]]:
    """Own the W&B lifecycle, including errors outside Trainer.fit."""
    if config.trainer.fast_dev_run:
        yield False
        return
    wandb_logger = WandbLogger(
        **config.wandb.model_dump(exclude_none=True),
        save_dir=str(run_dir),
        id=run_id,
        resume="never",
        log_model=False,
    )
    experiment = wandb_logger.experiment
    try:
        wandb_logger.log_hyperparams(config.model_dump(mode="json"))
        logger.info("W&B run: {}", experiment.url)
        yield wandb_logger
    except BaseException:
        experiment.finish(exit_code=1)
        raise
    else:
        experiment.finish(exit_code=0)


def train(config: TrainConfig) -> L.Trainer:
    torch.set_float32_matmul_precision("high")
    L.seed_everything(config.seed, workers=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, use_fast=True)
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError("Expected a pretrained tokenizer")
    train_loader, validation_loader = create_dataloaders(
        tokenizer=tokenizer,
        seed=config.seed,
        **config.data.model_dump(),
    )
    # Resuming training starts a new experiment with its own local artifacts.
    run_id = uuid4().hex
    run_dir = config.output / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        OmegaConf.to_yaml(OmegaConf.create(config.model_dump(mode="json")))
    )
    tokenizer.save_pretrained(run_dir / "tokenizer")
    checkpoint = ModelCheckpoint(
        dirpath=run_dir / "checkpoints",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="epoch={epoch:02d}-step={step}",
        auto_insert_metric_name=False,
    )
    if config.resume_from:
        model = SpeelingGood.load_from_checkpoint(
            config.resume_from, map_location="cpu"
        )
    else:
        model = SpeelingGood(**config.model.model_dump())
    callbacks: list[L.Callback] = [checkpoint]
    if config.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=config.early_stopping.patience,
                min_delta=config.early_stopping.min_delta,
                check_on_train_epoch_end=False,
            )
        )
    if not config.trainer.fast_dev_run:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    with tracking_run(config, run_dir, run_id) as run_logger:
        trainer = L.Trainer(
            **config.trainer.model_dump(),
            default_root_dir=str(run_dir),
            logger=run_logger,
            callbacks=callbacks,
        )
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=validation_loader,
            ckpt_path=config.resume_from,
        )
    if trainer.should_stop and config.early_stopping.enabled:
        logger.info(
            "Early stopping triggered at epoch {}, step {}",
            trainer.current_epoch,
            trainer.global_step,
        )
    logger.info(
        "Run artifacts: {}; best checkpoint: {}",
        run_dir,
        checkpoint.best_model_path or "none saved",
    )
    return trainer


@hydra.main(
    version_base="1.3",
    config_path=str(Path(__file__).resolve().parents[3] / "config"),
    config_name="train_smoke",
)
def main(config: DictConfig) -> None:
    train(TrainConfig.model_validate(OmegaConf.to_container(config, resolve=True)))


if __name__ == "__main__":
    main()
