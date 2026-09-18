from __future__ import annotations

import pytest

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


def g(x, payload):
    import os

    return os.getpid(), id(payload), x + payload[0]


def test_parallel_for_ray_context_deserialized_once_per_worker():
    """With ray, a worker resolves the shared context once and reuses it across its tasks."""
    ray = pytest.importorskip("ray")
    try:
        ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)
    except Exception as e:
        pytest.skip(f"ray could not start: {e}")
    try:
        n = 40
        res = parallel_for(g, [{"x": i} for i in range(n)], context=dict(payload=[100]), engine="ray")
    finally:
        ray.shutdown()
    assert [r[2] for r in res] == [i + 100 for i in range(n)]
    ids_per_worker: dict[int, set[int]] = {}
    for pid, payload_id, _ in res:
        ids_per_worker.setdefault(pid, set()).add(payload_id)
    assert len(ids_per_worker) <= 2
    # every task on a given worker saw the same deserialized context object
    assert all(len(ids) == 1 for ids in ids_per_worker.values()), ids_per_worker


class _RecordsShipping:
    """Pickles to whether it was pickled inside `shipping_context`."""

    def __init__(self):
        self.shipped = None

    def __getstate__(self):
        from tabarena.utils.shipping import is_shipping

        return {"shipped": is_shipping()}


def test_parallel_for_ray_put_marks_shipping():
    """The context handed to ray is pickled inside `shipping_context`; a plain pickle is not."""
    import pickle

    ray = pytest.importorskip("ray")
    try:
        ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)
    except Exception as e:
        pytest.skip(f"ray could not start: {e}")
    try:
        assert pickle.loads(pickle.dumps(_RecordsShipping())).shipped is False
        res = parallel_for(
            lambda x, probe: probe.shipped, [{"x": 0}], context=dict(probe=_RecordsShipping()), engine="ray"
        )
        assert res == [True]
    finally:
        ray.shutdown()
