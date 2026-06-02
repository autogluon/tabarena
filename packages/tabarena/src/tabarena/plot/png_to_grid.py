from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from PIL import Image


def make_png_grid(
    image_paths: Sequence[str | Path],
    output_path: str | Path,
    n_rows: int,
    n_cols: int,
    padding: int = 10,
    bg_color: tuple[int, int, int, int] = (255, 255, 255, 255),
    resize_mode: str = "fit",
    output_size: tuple[int, int] | None = None,
    max_output_size: tuple[int, int] | None = None,
    scale: float | None = None,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> None:
    """
    Combine PNG images into an n_rows x n_cols composite PNG.

    Parameters
    ----------
    image_paths
        List of input PNG file paths.
    output_path
        Path to save the composite PNG.
    n_rows
        Number of rows in the output grid.
    n_cols
        Number of columns in the output grid.
    padding
        Pixels between images and around the border.
    bg_color
        RGBA background color.
    resize_mode
        How to size images within each cell:
        - "fit": preserve aspect ratio, fit inside cell
        - "stretch": resize exactly to cell size
    output_size
        If provided, resize the final composite to exactly (width, height).
        This may distort the final aspect ratio.
    max_output_size
        If provided, resize the final composite to fit within (max_width, max_height)
        while preserving aspect ratio.
    scale
        If provided, uniformly scale the final composite by this factor.
        For example, 0.5 halves the size and 2.0 doubles it.
    resample
        PIL resampling filter used for resizing.

    Notes
    -----
    - If fewer images than grid cells are provided, remaining cells are left blank.
    - If more images than grid cells are provided, raises ValueError.
    - At most one of output_size, max_output_size, or scale may be specified.
    """
    image_paths = [Path(p) for p in image_paths]
    output_path = Path(output_path)

    if len(image_paths) > n_rows * n_cols:
        raise ValueError(
            f"Too many images ({len(image_paths)}) for grid {n_rows}x{n_cols} "
            f"({n_rows * n_cols} cells)."
        )

    if not image_paths:
        raise ValueError("image_paths must contain at least one image.")

    resize_args_provided = sum(
        x is not None for x in (output_size, max_output_size, scale)
    )
    if resize_args_provided > 1:
        raise ValueError(
            "Specify at most one of output_size, max_output_size, or scale."
        )

    if scale is not None and scale <= 0:
        raise ValueError(f"scale must be > 0, but got {scale}.")

    images = [Image.open(p).convert("RGBA") for p in image_paths]

    try:
        # Use the largest input width/height as cell size
        cell_w = max(img.width for img in images)
        cell_h = max(img.height for img in images)

        canvas_w = n_cols * cell_w + (n_cols + 1) * padding
        canvas_h = n_rows * cell_h + (n_rows + 1) * padding
        canvas = Image.new("RGBA", (canvas_w, canvas_h), bg_color)

        for idx, img in enumerate(images):
            row = idx // n_cols
            col = idx % n_cols

            x0 = padding + col * (cell_w + padding)
            y0 = padding + row * (cell_h + padding)

            if resize_mode == "stretch":
                placed = img.resize((cell_w, cell_h), resample)
                x = x0
                y = y0
            elif resize_mode == "fit":
                placed = img.copy()
                placed.thumbnail((cell_w, cell_h), resample)
                x = x0 + (cell_w - placed.width) // 2
                y = y0 + (cell_h - placed.height) // 2
            else:
                raise ValueError("resize_mode must be 'fit' or 'stretch'")

            canvas.alpha_composite(placed, (x, y))

        # Resize final composite if requested
        if output_size is not None:
            out_w, out_h = output_size
            if out_w <= 0 or out_h <= 0:
                raise ValueError(
                    f"output_size dimensions must be > 0, but got {output_size}."
                )
            canvas = canvas.resize((out_w, out_h), resample)

        elif max_output_size is not None:
            max_w, max_h = max_output_size
            if max_w <= 0 or max_h <= 0:
                raise ValueError(
                    f"max_output_size dimensions must be > 0, but got {max_output_size}."
                )

            scale_factor = min(max_w / canvas.width, max_h / canvas.height)
            # Avoid resizing if already within bounds
            if scale_factor < 1:
                new_w = max(1, round(canvas.width * scale_factor))
                new_h = max(1, round(canvas.height * scale_factor))
                canvas = canvas.resize((new_w, new_h), resample)

        elif scale is not None:
            new_w = max(1, round(canvas.width * scale))
            new_h = max(1, round(canvas.height * scale))
            canvas = canvas.resize((new_w, new_h), resample)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, format="PNG")

    finally:
        for img in images:
            img.close()



def make_png_grid_old(
    image_paths: Sequence[str | Path],
    output_path: str | Path,
    n_rows: int,
    n_cols: int,
    padding: int = 10,
    bg_color: tuple[int, int, int, int] = (255, 255, 255, 255),
    resize_mode: str = "fit",
) -> None:
    """
    Combine PNG images into an n_rows x n_cols composite PNG.

    Parameters
    ----------
    image_paths
        List of input PNG file paths.
    output_path
        Path to save the composite PNG.
    n_rows
        Number of rows in the output grid.
    n_cols
        Number of columns in the output grid.
    padding
        Pixels between images and around the border.
    bg_color
        RGBA background color.
    resize_mode
        How to size images within each cell:
        - "fit": preserve aspect ratio, fit inside cell
        - "stretch": resize exactly to cell size

    Notes
    -----
    - If fewer images than grid cells are provided, remaining cells are left blank.
    - If more images than grid cells are provided, raises ValueError.
    """
    image_paths = [Path(p) for p in image_paths]
    output_path = Path(output_path)

    if len(image_paths) > n_rows * n_cols:
        raise ValueError(
            f"Too many images ({len(image_paths)}) for grid {n_rows}x{n_cols} "
            f"({n_rows * n_cols} cells)."
        )

    if not image_paths:
        raise ValueError("image_paths must contain at least one image.")

    images = [Image.open(p).convert("RGBA") for p in image_paths]

    # Use the largest input width/height as cell size
    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)

    canvas_w = n_cols * cell_w + (n_cols + 1) * padding
    canvas_h = n_rows * cell_h + (n_rows + 1) * padding
    canvas = Image.new("RGBA", (canvas_w, canvas_h), bg_color)

    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols

        x0 = padding + col * (cell_w + padding)
        y0 = padding + row * (cell_h + padding)

        if resize_mode == "stretch":
            placed = img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            x = x0
            y = y0
        elif resize_mode == "fit":
            placed = img.copy()
            placed.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
            x = x0 + (cell_w - placed.width) // 2
            y = y0 + (cell_h - placed.height) // 2
        else:
            raise ValueError("resize_mode must be 'fit' or 'stretch'")

        canvas.alpha_composite(placed, (x, y))

    canvas.save(output_path, format="PNG")
