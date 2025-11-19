from __future__ import annotations

import shutil

from tabarena.nips2025_utils.tabarena_context import TabArenaContext


if __name__ == "__main__":
    elo_bootstrap_rounds = 200  # 1 for toy, 200 for official
    save_path = "output_website_artifacts"  # folder to save all figures and tables
    use_latex: bool = False  # Set to True if you have the appropriate latex packages installed for nicer figure style

    include_unverified = True

    tabarena_context = TabArenaContext(include_unverified=include_unverified)

    tabarena_context.evaluate_all(
        save_path=save_path,
        elo_bootstrap_rounds=elo_bootstrap_rounds,
        use_latex=use_latex,
    )

    zip_results = True
    upload_to_s3 = False
    if zip_results:
        file_prefix = f"tabarena_website_artifacts"
        file_name = f"{file_prefix}.zip"
        shutil.make_archive(file_prefix, "zip", root_dir=save_path)
        if upload_to_s3:
            from autogluon.common.utils.s3_utils import upload_file
            upload_file(file_name=file_name, bucket="tabarena", prefix=save_path)
