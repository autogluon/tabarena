from __future__ import annotations

from autogluon.common.loaders import load_json


def load_configs(config_files: list[str]) -> dict:
    if config_files is None:
        config_files = []
    configs = {}
    for c in config_files:
        configs.update(load_json.load(path=c))
    return configs
