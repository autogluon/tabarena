from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from tqdm import tqdm

from tabarena.utils.shipping import shipping_context

if TYPE_CHECKING:
    from collections.abc import Callable

A = TypeVar("A")
B = TypeVar("B")


def parallel_for(
    f: Callable[[object], B],
    inputs: list[list | dict],
    context: dict | None = None,
    engine: str = "ray",
    progress_bar: bool = True,
    desc: str | None = None,
) -> list[B]:
    """Evaluates an embarrasingly parallel for-loop.
    :param f: the function to be evaluated, the function is evaluated on `f(x, **context)` for all `x` in `inputs`
    if inputs are a list and on the union of x and context keyword arguments else
    :param inputs: list of inputs to be evaluated
    :param context: additional constant arguments to be passed to `f`. When using ray, the context is put in the local
     object store once and each worker process deserializes it once (not once per input, see
     `_resolve_ray_context`), so `f` must treat it as read-only: with ray, inputs that run on the same worker see the
     same context objects. When using joblib, the context is serialized for each input.
    :param engine: can be ["sequential", "ray", "joblib"]
    :return: a list where the function is evaluated on all inputs together with the context, i.e.
    `[f(x, **context) for x in inputs]`.
    """
    assert engine in ["sequential", "ray", "joblib"]
    if context is None:
        context = {}
    if engine == "sequential":
        return [
            f(**x, **context) if isinstance(x, dict) else f(*x, **context)
            for x in tqdm(inputs, desc=desc, disable=not progress_bar, mininterval=1)
        ]
    if engine == "joblib":
        from joblib import Parallel, delayed

        return Parallel(n_jobs=-1, verbose=50)(
            delayed(f)(**x, **context) if isinstance(x, dict) else delayed(f)(*x, **context) for x in inputs
        )
    if engine == "ray":
        import ray

        if not ray.is_initialized():
            ray.init()

        @ray.remote
        def remote_f(x, context_ref):
            # `context_ref` arrives inside a list, so ray hands over the reference itself
            # instead of resolving it: resolution goes through the per-worker cache.
            context = _resolve_ray_context(context_ref[0])
            return f(**x, **context) if isinstance(x, dict) else f(*x, **context)

        with shipping_context():
            remote_context = ray.put(context)
        remote_results = [remote_f.remote(x, [remote_context]) for x in inputs]
        return [ray.get(res) for res in tqdm(remote_results, desc=desc, disable=not progress_bar, mininterval=1)]
    return None


# Per worker process: the object-store reference (hex id) of the most recently used context and
# its deserialized value. Ray deserializes a task argument anew for every task invocation, so
# without this a context such as an EvaluationRepositoryCollection (hundreds of MB to GBs of
# pandas frames) is unpickled once per task instead of once per worker; with thousands of tasks
# that dominated the wall time. Keyed by reference, so a new `ray.put` (a fresh `parallel_for`
# call, or a mutated context) can never hit a stale entry. Bounded to one entry so a worker holds
# at most one context at a time.
_RAY_CONTEXT_CACHE: dict[str, object] = {}


def _resolve_ray_context(context_ref) -> dict:
    import ray

    key = context_ref.hex()
    context = _RAY_CONTEXT_CACHE.get(key)
    if context is None:
        _RAY_CONTEXT_CACHE.clear()
        context = ray.get(context_ref)
        _RAY_CONTEXT_CACHE[key] = context
    return context
