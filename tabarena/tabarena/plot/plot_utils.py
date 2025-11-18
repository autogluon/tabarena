from __future__ import annotations

from pathlib import Path

import pandas as pd


def figure_path(path: str | Path, prefix: str = None, suffix: str = None):
    fig_save_path_dir = Path(path)
    if prefix:
        fig_save_path_dir = fig_save_path_dir / prefix
    fig_save_path_dir = fig_save_path_dir / "figures"
    if suffix:
        fig_save_path_dir = fig_save_path_dir / suffix
    fig_save_path_dir.mkdir(parents=True, exist_ok=True)
    return fig_save_path_dir


def table_path(path: str | Path, prefix: str = None, suffix: str = None):
    table_save_path_dir = Path(path)
    if prefix:
        table_save_path_dir = table_save_path_dir / prefix
    table_save_path_dir = table_save_path_dir / "tables"
    if suffix:
        table_save_path_dir = table_save_path_dir / suffix
    table_save_path_dir.mkdir(parents=True, exist_ok=True)
    return table_save_path_dir


def save_latex_table(df: pd.DataFrame, title: str, show_table: bool = False, latex_kwargs: dict | None = None, n_digits = None, save_prefix: str = None):
    if n_digits:
        for col in df.columns:
            if (not df[col].dtype == "object") and (not df[col].dtype == "int64"):
                n_digit = n_digits.get(col, 2)
                df[col] = df[col].astype("object")
                df.loc[:, col] = df.loc[:, col].apply(lambda s: f'{s:.{n_digit}f}')

    if latex_kwargs is None:
        latex_kwargs = dict()
    s = df.to_latex(**latex_kwargs)
    latex_folder = table_path(path=save_prefix)
    latex_file = latex_folder / f"{title}.tex"
    print(f"Writing latex result in {latex_file}")
    with open(latex_file, "w") as f:
        f.write(s)
    if show_table:
        print(s)
