from __future__ import annotations

from pathlib import Path


def figure_path(prefix: str | None = None, suffix: str | None = None, mkdir: bool = True) -> Path:
    fig_save_path_dir = Path()
    if prefix:
        fig_save_path_dir = fig_save_path_dir / prefix
    fig_save_path_dir = fig_save_path_dir / "figures"
    if suffix:
        fig_save_path_dir = fig_save_path_dir / suffix
    if mkdir:
        fig_save_path_dir.mkdir(parents=True, exist_ok=True)
    return fig_save_path_dir
