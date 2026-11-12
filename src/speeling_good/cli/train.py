"""Train a BIO spelling discriminator on prepared part-*.jsonl shards."""

from itertools import islice
from pathlib import Path
from typing import Annotated, cast

import hydra
import lightning as L
import torch
from datasets import Dataset, load_dataset
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerBase,
)

from speeling_good.dataloaders import prepare_dataset
from speeling_good.module import SpeelingGood

PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_name: str = "answerdotai/ModernBERT-base"
    learning_rate: Annotated[float, Field(gt=0)] = 3e-5
    weight_decay: Annotated[float, Field(ge=0)] = 0.01
    warmup_ratio: Annotated[float, Field(ge=0, le=1)] = 0.1
    gradient_checkpointing: bool = False


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    directory: Path = Path("data/fineweb")
    take: PositiveInt | None = 64
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


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = 42
    output: Path = Path("runs/discriminator")
    resume_from: Path | None = None
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)


def load_rows(config: DataConfig) -> Dataset:
    """Bound smoke-run input before reading/tokenizing the whole dataset."""
    shards = sorted(config.directory.glob("part-*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No part-*.jsonl shards found in {config.directory}")
    files = [str(path) for path in shards]
    if config.take is None:
        rows = load_dataset("json", data_files=files, split="train")
    else:
        stream = load_dataset("json", data_files=files, split="train", streaming=True)
        rows = Dataset.from_list(list(islice(stream, config.take)))
    rows = rows.filter(lambda row: row["annotations"] is not None)
    # Repeated passages must not leak between training and validation.
    seen: set[str] = set()
    indices = []
    for index, text in enumerate(rows["text"]):
        if text.strip() and text not in seen:
            seen.add(text)
            indices.append(index)
    rows = rows.select(indices)
    if len(rows) < 2:
        raise ValueError("Need at least two distinct passages with usable annotations")
    return rows


def train(config: TrainConfig) -> L.Trainer:
    torch.set_float32_matmul_precision("high")
    L.seed_everything(config.seed, workers=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name, use_fast=True)
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        raise TypeError("Expected a pretrained tokenizer")
    rows = load_rows(config.data)
    split = rows.train_test_split(
        test_size=config.data.validation_fraction, seed=config.seed
    )
    loaders = {}
    sizes = {}
    for name in ("train", "test"):
        dataset = prepare_dataset(
            split[name], tokenizer, max_length=config.data.max_length
        )
        # Exclude passages whose tokenization contains no supervised tokens.
        dataset = dataset.filter(
            lambda row: any(label != -100 for label in row["labels"])
        )
        if not len(dataset):
            raise ValueError(f"No supervised passages in {name} split")
        sizes[name] = len(dataset)
        loaders[name] = DataLoader(
            cast(TorchDataset, dataset),
            batch_size=config.data.batch_size,
            shuffle=name == "train",
            num_workers=config.data.num_workers,
            generator=torch.Generator().manual_seed(config.seed),
            collate_fn=DataCollatorForTokenClassification(tokenizer),
        )
    logger.info(
        "Training on {} passages; validating on {}",
        sizes["train"],
        sizes["test"],
    )
    # Each invocation gets a separate version directory, including resumed runs.
    csv_logger = CSVLogger(str(config.output), name="")
    _ = csv_logger.experiment  # Initialize before writing artifacts into log_dir.
    run_dir = Path(csv_logger.log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
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
    trainer = L.Trainer(
        **config.trainer.model_dump(),
        default_root_dir=str(run_dir),
        logger=csv_logger,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="step")],
    )
    trainer.fit(
        model,
        train_dataloaders=loaders["train"],
        val_dataloaders=loaders["test"],
        ckpt_path=config.resume_from,
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
