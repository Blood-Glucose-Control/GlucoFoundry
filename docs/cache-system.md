# Centralized Cache System

GlucoFoundry caches each dataset's raw and processed data under a consistent
layout and resolves raw data from the dataset's configured source on first use.
Fetching is **automatic for HuggingFace and local sources**; the bundled study
datasets require a one-time **manual download** — the cache manager raises a
clear error telling you where to put the files.

## Cache structure

```
cache/
└── data/
    └── <dataset_name>/
        ├── raw/          # original downloaded or copied files
        └── processed/    # cleaned, standardized, cached outputs
```

## How raw data is resolved

`CacheManager.ensure_raw_data(dataset_name, dataset_config)` checks the cache and,
if the raw files are missing, fetches them based on the dataset's `source`
(declared in `src/data/dataset_configs.py`):

| Source | Behavior |
|---|---|
| **HuggingFace** | Auto-downloaded via the `datasets` library using the config's `hf_dataset_id`. |
| **Local** | Copied from the configured local `cache_path`. |
| **Study datasets** (for example `aleppo_2017`, `lynch_2022`, `brown_2019`, `metabonet`, `cgmacros`) | Require a **manual download**: the manager raises a `RuntimeError` with instructions. See each dataset's `README.md` under `src/data/diabetes_datasets/<name>/`. |

## Usage

Data access normally goes through the dataset loader, which uses the cache
manager internally:

```python
from src.data.diabetes_datasets.data_loader import get_loader

loader = get_loader(data_source_name="gluroo", use_cached=True)
data = loader.load_data()
```

### Cache management

```python
from src.data.cache_manager import get_cache_manager

cache = get_cache_manager()

# Path to a dataset's cache directory
path = cache.get_dataset_cache_path("gluroo")

# Clear one dataset's cache, or all of it
cache.clear_cache("gluroo")
cache.clear_cache()
```

## Adding a dataset

1. Register the dataset in `src/data/dataset_configs.py` with its `source`
   (`DatasetSourceType`), `required_files`, and — depending on the source — a
   `hf_dataset_id` (HuggingFace) or `cache_path` (local).
2. Add the loader under `src/data/diabetes_datasets/<name>/`.

See [Data Preparation](user-guide/datasets/data-preparation.md) for the full
loader and standardization contract.

## Troubleshooting

- **Manual-download error**: the dataset's `source` requires you to place its
  `required_files` under `cache/data/<dataset>/raw/` yourself; follow the path in
  the raised error and the dataset's `README.md`.
- **Corrupted cache**: clear it with `cache.clear_cache("<dataset>")` (or
  `cache.clear_cache()` for everything), then re-run; check disk space and
  directory permissions.
