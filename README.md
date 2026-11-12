# speeling-good

A minimal Python and PyTorch project with CUDA support.


## Setup
```sh
direnv allow
```

## Data preprocessing

Set `MODEL_ENDPOINT` and `MODEL_TOKEN`, then run with the Hydra configuration
loaded automatically from [config/prepare_fineweb.yaml](config/prepare_fineweb.yaml).
Override settings with `key=value`. Reusing an output directory resumes after its
completed shards; `take` remains the total row limit across runs:

```sh
uv run prepare-fineweb output=data/fineweb-experiment annotation.temperature=0.5
```

Preview the resolved configuration without running annotation:

```sh
uv run prepare-fineweb --cfg job --resolve
```


## Training

Train the ModernBERT-base BIO spelling discriminator on prepared `part-*.jsonl`
shards. The default [smoke configuration](config/train_smoke.yaml) reads up to
64 passages from `data/fineweb`, uses batches of two with a maximum sequence
length of 128 tokens, and runs two optimizer steps plus validation. The first
run downloads the pretrained model and tokenizer. Lightning automatically selects
an available accelerator, falling back to CPU.

```sh
uv run train-discriminator
```

Override settings with Hydra's `key=value` syntax:

```sh
uv run train-discriminator data.directory=data/fineweb-experiment model.learning_rate=2e-5
```

Preview the resolved configuration without loading data or model weights:

```sh
uv run train-discriminator --cfg job --resolve
```

For a single training and validation batch, with logging and checkpointing disabled:

```sh
uv run train-discriminator trainer.fast_dev_run=true
```

Each run writes to `runs/discriminator/version_*/`: `config.yaml`, the tokenizer,
CSV metrics, and best/last checkpoints. Metrics include training and validation
loss, BIO token accuracy, and spelling-error token precision, recall, and F1.
The best checkpoint minimizes validation loss.

Resume with the same model and data settings and a higher step limit:

```sh
uv run train-discriminator resume_from=runs/discriminator/version_0/checkpoints/last.ckpt trainer.max_steps=4
```

Resume restores model, optimizer, and scheduler state, but does not restore the
exact mid-epoch shuffle position. Resumed runs write to a new version directory.

When ready for a longer run, remove the passage and step limits, and validate
on the entire held-out split once per epoch:

```sh
uv run train-discriminator data.take=null trainer.max_steps=-1 trainer.max_epochs=3 \
  trainer.val_check_interval=1.0 trainer.limit_val_batches=1.0
```

## Development
```sh
pytest
ruff check
ruff format
```
