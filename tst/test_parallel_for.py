from __future__ import annotations

from tabarena.utils.parallel_for import parallel_for


def f(x, v1, v2):
    return x + v1 + v2


def test_parallel_for_list_arguments():
    for engine in [
        "sequential",
        "joblib",
        # TODO fix me in CI
        # "ray"
    ]:
        res = parallel_for(
            f,
            [[1], [2], [3]],
            context=dict(v1=2, v2=3),
            engine=engine,
        )
        print(res)
        assert res == [6, 7, 8]


def test_parallel_for_dict_arguments():
    for engine in [
        "sequential",
        "joblib",
        # TODO fix me in CI
        # "ray"
    ]:
        res = parallel_for(
            f,
            [{"x": 1}, {"x": 2}, {"x": 3}],
            context=dict(v1=2, v2=3),
            engine=engine,
        )
        print(res)
        assert res == [6, 7, 8]
