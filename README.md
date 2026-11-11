# speeling-good

A minimal Python and PyTorch project with CUDA support.


## Setup
```sh
direnv allow
```

## Data preprocessing

Set `MODEL_ENDPOINT` and `MODEL_TOKEN`, then run with the Hydra configuration
loaded automatically from [config/prepare_fineweb.yaml](config/prepare_fineweb.yaml).
Override settings with `key=value`; choose a fresh output directory for each run:

```sh
uv run prepare-fineweb output=data/fineweb-experiment annotation.temperature=0.5
```

Preview the resolved configuration without running annotation:

```sh
uv run prepare-fineweb --cfg job --resolve
```


## Development
```sh
pytest
ruff check
ruff format
```
