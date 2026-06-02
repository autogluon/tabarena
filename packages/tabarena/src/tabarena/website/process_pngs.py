"""Helper script to go from PNG to PNG ZIP Files we can use in HTML on the LB."""

from __future__ import annotations

import zipfile
from pathlib import Path
from multiprocessing import Pool, cpu_count


def process_png(png_path_str: str) -> None:
    png_path = Path(png_path_str).resolve()
    zip_path = png_path.with_suffix(".png.zip")

    print(f"Converting {png_path}...")

    # Zip all PNGs into one archive
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(png_path, arcname=png_path.name)

    # Clean up PNGs
    png_path.unlink(missing_ok=True)


def process_png_bulk(path: Path) -> None:
    png_paths = [str(p) for p in path.rglob("*.png")]

    if not png_paths:
        print("No PNGs found.")
        return

    # Use one process per CPU, but not more than number of PDFs
    n_procs = min(cpu_count(), len(png_paths))
    print(f"Found {len(png_paths)} PNGs. Using {n_procs} processes.")

    with Pool(processes=n_procs) as pool:
        # imap_unordered gives you streaming results + simple progress printing
        for _ in pool.imap_unordered(process_png, png_paths):
            pass
