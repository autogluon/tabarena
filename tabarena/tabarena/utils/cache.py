from __future__ import annotations

import os
from typing import Union

import pandas as pd
import pickle
import sys
import boto3
from pathlib import Path
from contextlib import contextmanager
from time import perf_counter
from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

from autogluon.common.loaders import load_pkl
from autogluon.common.savers import save_pkl
from autogluon.common.utils import s3_utils

from tabarena.utils import catchtime

default_cache_path = Path("~/cache-zeroshot/").expanduser()

T = TypeVar("T")


# TODO: Cache `fun` details to verify cache equivalence
#  Store kwargs, don't use lambda
# TODO: Cache aux info such as date
# TODO: Enable "only_cache", to enforce that no code execution occurs, and make `fun` optional
# TODO: Docstring
class AbstractCacheFunction(Generic[T]):
    _save_after_run = True
    _load_after_cache = True

    def __init__(self, include_self_in_call: bool = False):
        self.include_self_in_call = include_self_in_call

    def cache(
        self,
        fun: Callable[..., T] | None,
        *,
        fun_kwargs: dict | None = None,
        ignore_cache: bool = False,
    ) -> T:
        exists = self.exists
        cache_file = self.cache_file
        if exists:
            if ignore_cache:
                run = True
            else:
                run = False
        else:
            run = True

        if run:
            msg = f'Generating cache (exists={exists}, ignore_cache={ignore_cache}, cache_file="{cache_file}")'
        else:
            msg = f'Loading cache (exists={exists}, ignore_cache={ignore_cache}, cache_file="{cache_file}")'
        print(msg)

        if not run:
            return self.load_cache()
        with catchtime("Evaluate function"):
            data = self._run(fun=fun, fun_kwargs=fun_kwargs)
            if self._save_after_run:
                self.save_cache(data=data)
        if self._save_after_run and self._load_after_cache:
            return self.load_cache()
        else:
            return data

    def _run(self, fun: Callable[..., T], fun_kwargs: dict | None = None) -> T:
        assert fun is not None, f"`fun` must not be None if a saving a new cache is required!"
        if fun_kwargs is None:
            fun_kwargs = {}
        assert isinstance(fun_kwargs, dict)
        if self.include_self_in_call:
            return fun(cacher=self, **fun_kwargs)
        else:
            return fun(**fun_kwargs)

    @property
    def exists(self) -> bool:
        return Path(self.cache_file).exists()

    @property
    def cache_file(self) -> str | None:
        raise NotImplementedError

    def save_cache(self, data: T) -> None:
        raise NotImplementedError

    def load_cache(self) -> T:
        raise NotImplementedError


# TODO: Avoid storing results as pickle for safety
class CacheFunctionDummy(AbstractCacheFunction[object]):
    """
    No caching
    """
    _save_after_run = False
    _load_after_cache = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def exists(self) -> bool:
        return False

    @property
    def cache_file(self):
        return None


# TODO: Avoid storing results as pickle for safety
class CacheFunctionPickle(AbstractCacheFunction[object]):
    """
    Note: This is unsafe. Know that loading pickle files can execute arbitrary code.
    Only use this for trusted cache files, and consider switching to a safe cache format as soon as possible.

    Most generic cache function that saves and loads pickle files.
    Should work with almost any function.
    """
    _load_after_cache = False  # Loading a pickle is unnecessary when the contents are already in memory

    def __init__(self, cache_name: str, cache_path: Path | str | None = None, include_self_in_call: bool = False):
        super().__init__(include_self_in_call=include_self_in_call)
        self.cache_name = cache_name
        if cache_path is None:
            # TODO: Remove default_cache_path?
            cache_path = default_cache_path
        cache_path = str(cache_path)
        if cache_path.endswith(os.path.sep):
            raise ValueError(f"cache_path must not end with a directory separator! (cache_path='{cache_path}'")
        self.cache_path = cache_path
        self.is_s3 = self.cache_path.startswith("s3://")

    @property
    def cache_file(self) -> str:
        if self.is_s3:
            return f"{self.cache_path}/{self.cache_name}.pkl"
        return str(Path(self.cache_path) / (self.cache_name + ".pkl"))

    @property
    def exists(self) -> bool:
        if self.is_s3:
            try:
                s3 = boto3.client('s3')
                bucket, key = s3_utils.s3_path_to_bucket_prefix(self.cache_file)
                s3.head_object(Bucket=bucket, Key=key)
                return True
            except s3.exceptions.ClientError as e:
                if e.response['Error']['Code'] == '404':
                    return False
                else:
                    raise
        else:
            return Path(self.cache_file).exists()

    def save_cache(self, data: object) -> None:
        if self.is_s3:
            s3 = boto3.client('s3')
            bucket, key = s3_utils.s3_path_to_bucket_prefix(self.cache_file)
            cache = pickle.dumps(data)
            print(f'Writing cache with size {round(sys.getsizeof(cache) / 1e6, 3)} MB to {self.cache_file}')
            s3.put_object(Bucket=bucket, Key=key, Body=cache)
        else:
            cache_file = self.cache_file
            Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                cache = pickle.dumps(data)
                print(f'Writing cache with size {round(sys.getsizeof(cache) / 1e6, 3)} MB')
                f.write(cache)

    def load_cache(self) -> object:
        with open(self.cache_file, "rb") as f:
            return pickle.loads(f.read())


# TODO: Delete and use CacheFunctionPickle
def cache_function(
        fun: Callable[[], object],
        cache_name: str,
        ignore_cache: bool = False,
        cache_path: Optional[Path | str] = None
):
    f"""
    :param fun: a function whose result obtained `fun()` will be cached, the output of the function must be serializable.
    :param cache_name: the cache of the function result is written into `{cache_path}/{cache_name}.pkl`
    :param ignore_cache: whether to recompute even if the cache is present
    :param cache_path: folder where to write cache files, default to ~/cache-zeroshot/
    :return: result of fun()
    """
    if cache_path is None:
        cache_path = default_cache_path
    cache_file = Path(cache_path) / (cache_name + ".pkl")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists() and not ignore_cache:
        print(f"Loading cache {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.loads(f.read())
    else:
        print(f"Cache {cache_file} not found or ignore_cache set to True, regenerating the file")
        with catchtime("Evaluate function."):
            res = fun()
            with open(cache_file, "wb") as f:
                cache = pickle.dumps(res)
                print(f'Writing cache with size {round(sys.getsizeof(cache) / 1e6, 3)} MB')
                f.write(cache)
            return res


class CacheFunctionDF(AbstractCacheFunction[pd.DataFrame]):
    _load_after_cache = True  # Loading from cache is necessary because .to_csv loses information from the original DataFrame

    def __init__(self, cache_name: str, cache_path: Path | str, include_self_in_call: bool = False):
        super().__init__(include_self_in_call=include_self_in_call)
        self.cache_name = cache_name
        self.cache_path = cache_path

    @property
    def cache_file(self) -> str:
        return str(Path(self.cache_path) / (self.cache_name + ".csv"))

    def load_cache(self) -> pd.DataFrame:
        return pd.read_csv(self.cache_file)

    def save_cache(self, data: pd.DataFrame):
        assert isinstance(data, pd.DataFrame)
        cache_file = self.cache_file
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(cache_file, index=False)


@contextmanager
def catchtime(name: str, logger=None) -> float:
    start = perf_counter()
    print_fun = print if logger is None else logger.info
    try:
        print_fun(f"start: {name}")
        yield lambda: perf_counter() - start
    finally:
        print_fun(f"Time for {name}: {perf_counter() - start:.4f} secs")


# TODO: Delete and use CacheFunctionDF?
@dataclass
class Experiment:
    expname: str  # name of the parent experiment used to store the file
    name: str  # name of the specific experiment, e.g. "localsearch"
    run_fun: Callable[..., list]  # function to execute to obtain results
    kwargs: dict = None

    def data(self, ignore_cache: bool = False):
        kwargs = self.kwargs
        if kwargs is None:
           kwargs = {}
        cacher = CacheFunctionDF(cache_name=self.name, cache_path=self.expname)
        return cacher.cache(
            lambda: pd.DataFrame(self.run_fun(**kwargs)),
            ignore_cache=ignore_cache,
        )


# TODO: Delete and use CacheFunctionPickle?
@dataclass
class SimulationExperiment(Experiment):
    def data(self, ignore_cache: bool = False) -> object:
        kwargs = self.kwargs
        if kwargs is None:
           kwargs = {}
        cacher = CacheFunctionPickle(cache_name=self.name, cache_path=self.expname)
        return cacher.cache(
            fun=lambda: self.run_fun(**kwargs),
            ignore_cache=ignore_cache,
        )


# TODO: Delete and use CacheFunctionDummy?
@dataclass
class DummyExperiment(Experiment):
    """
    Dummy Experiment class that doesn't perform caching and simply runs the run_fun and returns the result.
    """

    def data(self, ignore_cache: bool = False):
        return self.run_fun()


class SaveLoadMixin:
    """
    Mixin class to add generic pickle save/load methods.
    """
    def save(self, path: Union[str, Path]):
        path = str(path)
        assert path.endswith('.pkl')
        save_pkl.save(path=path, object=self)

    @classmethod
    def load(cls, path: Union[str, Path]):
        path = str(path)
        assert path.endswith('.pkl')
        obj = load_pkl.load(path=path)
        assert isinstance(obj, cls)
        return obj


if __name__ == '__main__':
    def f():
        import time
        time.sleep(0.5)
        return [1, 2, 3]

    res = cache_function(f, "f", ignore_cache=True)
    assert res == [1, 2, 3]

    res = cache_function(f, "f", ignore_cache=False)
    assert res == [1, 2, 3]
