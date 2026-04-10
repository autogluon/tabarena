from __future__ import annotations

from pathlib import Path


def pre_generate_text_cache(task_id_str: str, *, ignore_cache: bool = False) -> Path:
    """Generate the cache as it would be generated on-the-fly during preprocessing,
    and save it to a parquet file for later loading.
    """
    from tabarena.benchmark.preprocessing.model_agnostic_default_preprocessing import TabArenaModelAgnosticPreprocessing
    from tabarena.benchmark.preprocessing.text_feature_generators import SemanticTextFeatureGenerator
    from tabarena.benchmark.task.openml import OpenMLTaskWrapper
    from tabarena.benchmark.task.user_task import UserTask

    task_id_or_object = UserTask.from_task_id_str(task_id_str)
    task = OpenMLTaskWrapper(
        task=task_id_or_object.load_local_openml_task(),
    )
    cache_path = SemanticTextFeatureGenerator.get_text_cache_dir(task_id_str=str(task.task_id))
    if (not ignore_cache) and cache_path.exists():
        print(f"Cache already exists for {task_id_str} at {cache_path}, skipping generation.")
        return cache_path

    print(f"Loaded {task_id_str}, with {len(task.X)} rows and {len(task.X.columns)} columns.")
    preprocessing = TabArenaModelAgnosticPreprocessing(
        enable_sematic_text_features=True,
        enable_raw_text_features=False,
        enable_text_special_features=False,
        enable_statistical_text_features=False,
        enable_text_ngram_features=False,
        enable_datetime_features=False,
        verbosity=4,
    )
    preprocessing.fit_transform(X=task.X)

    cache_path = SemanticTextFeatureGenerator.get_text_cache_dir(task_id_str=str(task.task_id))
    SemanticTextFeatureGenerator.save_embedding_cache(
        cache=SemanticTextFeatureGenerator._embedding_look_up, path=cache_path
    )
    SemanticTextFeatureGenerator._embedding_look_up.clear()
    print(f"Cache generated and saved to: {cache_path}")
    return cache_path


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(levelname)s: %(message)s")

    # TODO: add support for setting the OpenML cache dir here as well.
    parser = argparse.ArgumentParser()
    # Require tasks settings
    parser.add_argument(
        "--task_id_str",
        type=str,
        required=True,
        help="User Task ID for a dataset with text.",
    )
    args = parser.parse_args()

    pre_generate_text_cache(args.task_id_str)
