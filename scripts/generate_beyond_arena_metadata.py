"""Regenerate the committed BeyondArena reference-metadata CSV (maintainer tool).

Run this once when the ``BeyondArena`` collection contents change. It downloads
and converts every container (large, one-off), then writes a portable CSV to the
package-data location. Commit the resulting file so users can filter datasets
before downloading anything.

    python scripts/generate_beyond_arena_metadata.py

Requires the optional ``data-foundry`` dependency (``tabarena[data-foundry]``).
"""

from __future__ import annotations

from tabarena.benchmark.task.data_foundry import (
    generate_reference_metadata,
    get_beyond_arena_collection,
    reference_metadata_package_path,
)

if __name__ == "__main__":
    collection = get_beyond_arena_collection()
    out_path = reference_metadata_package_path(collection.name)
    generate_reference_metadata(collection=collection, out_path=out_path)
    print(f"Wrote reference metadata to {out_path}. Commit it to ship the fast filter path.")
