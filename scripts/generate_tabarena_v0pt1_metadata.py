"""Regenerate the committed TabArena v0.1 reference-metadata CSV (maintainer tool).

Rebuilds the per-task x split metadata from the curated v0.1 metadata and writes it
to the package-data location read by ``TabArenaV0pt1TaskMetadataSource``. Run this
when the curated v0.1 metadata changes, then commit the resulting CSV.

    python scripts/generate_tabarena_v0pt1_metadata.py
"""

from __future__ import annotations

from tabarena.benchmark.task.metadata.sources.tabarena_v0pt1 import (
    generate_tabarena_v0_1_reference_metadata,
)

if __name__ == "__main__":
    out_path = generate_tabarena_v0_1_reference_metadata()
    print(f"Wrote reference metadata to {out_path}. Commit it to skip the on-the-fly rebuild.")
