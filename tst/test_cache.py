from __future__ import annotations

import pandas as pd

from tabarena.utils.cache import CacheFunctionDF, CacheFunctionDummy, CacheFunctionPickle


def test_cache_pickle(tmp_path):
    def f():
        return [1, 2, 3]

    for ignore_cache in [True, False]:
        cacher = CacheFunctionPickle(cache_name="f", cache_path=str(tmp_path))
        data = cacher.cache(fun=f, ignore_cache=ignore_cache)
        assert cacher.exists
        assert data == [1, 2, 3]


def test_cache_dataframe(tmp_path):
    def f():
        return pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    for ignore_cache in [True, False]:
        cacher = CacheFunctionDF(cache_name="f", cache_path=str(tmp_path))
        data = cacher.cache(fun=f, ignore_cache=ignore_cache)
        assert cacher.exists
        pd.testing.assert_frame_equal(data, pd.DataFrame({"a": [1, 2], "b": [3, 4]}))


def test_cache_dummy():
    def f():
        return [1, 2, 3]

    for ignore_cache in [True, False]:
        cacher = CacheFunctionDummy()
        assert not cacher.exists
        data = cacher.cache(fun=f, ignore_cache=ignore_cache)
        assert not cacher.exists
        assert data == [1, 2, 3]
