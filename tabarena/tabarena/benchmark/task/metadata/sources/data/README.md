# Committed reference task-metadata CSVs

This directory holds the **git-committed reference metadata** for every task
metadata source, one CSV per source named `<name>_tasks_metadata.csv`:

| File | Source | `name` |
| --- | --- | --- |
| `tabarena-v0.1_tasks_metadata.csv` | `TabArenaV0pt1TaskMetadataSource` | `tabarena-v0.1` |
| `BeyondArena_tasks_metadata.csv` | `DataFoundryTaskMetadataSource` (official collection) | `BeyondArena` |
| `<collection>_tasks_metadata.csv` | `DataFoundryTaskMetadataSource` (any collection) | `<collection.name>` |

Each CSV is one row per `(task, split)` with every `TabArenaTaskMetadata` column
(the `TabArenaTaskMetadata.to_dataframe()` format), plus `data_foundry_uri` for
Data Foundry collections. The files carry **no dataset contents**, only metadata,
so they are small and safe to commit. They let users filter tasks (by name, dtype,
size, problem type) **before downloading any data**, and let the sources skip
rebuilding / re-deriving the metadata at runtime.

Each CSV also carries **warehouse-level metadata** (`task_type`, `domain`,
`dataset_year`, `source`, `num_text_cols`, `num_high_cardinality_cats`,
`num_cols_after_preprocessing`, `missing_value_fraction`) so the table is
self-contained for filtering / plotting — no separate `warehouse_metadata.csv`
merge is needed. Dataset-derived stats (text / cardinality / cols-after-preprocessing
/ missing fraction) are computed at creation when the dataset is available
(Data Foundry collections); for v0.1 only the tabular fields (`task_type`, `domain`,
`dataset_year`, `source`) are populated and the dataset-derived stats stay empty.

`task_id_str` is stored in its portable form (no machine-specific cache path); it
resolves against the local cache at load time.

If a CSV is missing, the corresponding source regenerates the metadata on the fly
(see below) — committing the CSV is purely an optimization (and, for Data Foundry,
enables filter-before-download).

## Regenerating

The committed CSVs are produced by maintainer scripts under `scripts/`:

### TabArena v0.1

Pure local transform of the curated v0.1 metadata (no downloads):

```bash
python scripts/generate_tabarena_v0pt1_metadata.py
```

Equivalently, in code:

```python
from tabarena.benchmark.task.metadata.sources.tabarena_v0pt1 import (
    generate_tabarena_v0_1_reference_metadata,
)

generate_tabarena_v0_1_reference_metadata()  # writes tabarena-v0.1_tasks_metadata.csv
```

### Data Foundry collections (e.g. BeyondArena)

Downloads + converts the whole collection (large, one-off):

```bash
python scripts/generate_beyond_arena_metadata.py
```

For an arbitrary collection:

```python
from tabarena.benchmark.task.data_foundry import (
    generate_reference_metadata,
    reference_metadata_package_path,
)

generate_reference_metadata(
    collection=my_collection,
    out_path=reference_metadata_package_path(my_collection.name),
)
```

Then commit the resulting CSV.
