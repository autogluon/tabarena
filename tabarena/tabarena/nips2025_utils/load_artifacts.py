from __future__ import annotations

import io
from pathlib import Path
import pickle

from tabarena.benchmark.result import AGBagResult, BaselineResult
from tabarena.utils.parallel_for import parallel_for
from tabarena.utils.pickle_utils import read_pickle_bytes


# TODO: This is a hack to ensure old result artifacts still load properly after renaming tabrepo to tabarena.
# TODO: We should ensure all artifacts are saved as a dictionary so this doesn't need to be here.
class _RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("tabrepo"):
            module = module.replace("tabrepo", "tabarena", 1)
        return super().find_class(module, name)


def _rename_load(file_obj):
    try:
        return _RenameUnpickler(file_obj).load()
    except EOFError as e:
        raise e from Exception(
            f"Failed to load artifact {file_obj}. This may be due to a corrupted file or"
            " an incompatible format. Please ensure the file is a valid artifact and"
            " try again."
        )


def load_and_align(path, convert_to_holdout: bool = False) -> BaselineResult:
    # Transparently handles both raw and gzip-compressed ``.pkl`` artifacts.
    data: dict | BaselineResult = _rename_load(io.BytesIO(read_pickle_bytes(path)))

    data_aligned = BaselineResult.from_dict(data)
    if convert_to_holdout:
        return result_to_holdout(result=data_aligned)
    return data_aligned


def load_all_artifacts(
    file_paths: list[str | Path],
    engine: str = "sequential",
    convert_to_holdout: bool = False,
    progress_bar: bool = True,
) -> list[BaselineResult]:
    file_paths_lst = []
    for file_path in file_paths:
        file_paths_lst.append(
            {
                "path": str(file_path),
                "convert_to_holdout": convert_to_holdout,
            }
        )

    results_lst: list[BaselineResult] = parallel_for(
        f=load_and_align,
        inputs=file_paths_lst,
        engine=engine,
        progress_bar=progress_bar,
        desc=f"Loading raw artifacts"
    )
    return results_lst


def result_to_holdout(result: BaselineResult) -> BaselineResult:
    assert isinstance(result, AGBagResult)
    result_holdout = result.bag_artifacts(as_baseline=False)
    if len(result_holdout) > 0:
        assert len(result_holdout) == 1
        result_holdout = result_holdout[0]
    else:
        result_holdout = None
    return result_holdout


def results_to_holdout(result_lst: list[BaselineResult]) -> list[BaselineResult]:
    return [result_to_holdout(result) for result in result_lst]
