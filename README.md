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

## Training

Use [config/train.yaml](config/train.yaml) for the first full run: three epochs
on `data/fineweb`, with full validation after each epoch. Requires a GPU with
bfloat16 support.

```sh
uv run wandb login
uv run train-discriminator --config-name train
```

## Development
```sh
pytest
ruff check
ruff format
```
