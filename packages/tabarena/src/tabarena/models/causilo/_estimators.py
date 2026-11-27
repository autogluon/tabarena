"""Developer fix: the fits of Causilo-1.5 share one network per process through the weights registry.

causilo 1.5.1 offers what AutoGluon's ``SharedWeights`` asks of a library: a loader with plain inputs,
``causilo.load_pretrained_model(task, model_version, device)``, and fitted state that pickles without the
library's ``__getstate__``. A declaration works::

    shared_weights = SharedWeights(loader="causilo:load_pretrained_model", key=("task", "model_version"))

but its weightless pickle still carries most of the network. Every fitted execution runner keeps a
``ModelRunner(model)``, which holds the network's child modules (``feature_embedding``, ``row``, ``columns``,
``prediction``), and AutoGluon detaches only the module the loader returned. The children are pickled with
their weights, about 1.2 GB in ``model_tensors.pt`` for every saved fit (the refit model is saved inside
the timed fit), and come back as a second copy of the network on load.

Until AutoGluon also detaches the submodules of a shared network (https://github.com/autogluon/autogluon/pull/6100),
the wrapper keeps causilo's own pickle, which leaves the network out, and shares only the load. :func:`attach_shared_network` takes the network from the
registry (:mod:`tabarena.models._weights`), loaded through the public loader, and gives the estimator,
before ``fit``, the engine its ``fit`` would build, holding that network. ``fit`` reuses an engine whose
task, checkpoint, device and memory budgets match the estimator's parameters; if a later causilo changed
those conditions, ``fit`` would build its own engine and load its own network, slower but with the same
predictions. A pickled fit reloads its own network through causilo's ``import_estimator`` when the
benchmark loads it, outside the timed fit.

Remove this module and declare the ``SharedWeights`` above on ``Causilo15Model`` once TabArena's AutoGluon
floor includes that fix. Written against causilo 1.5.1.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: The causilo release whose engine this module builds; any other release fits without sharing.
WRITTEN_AGAINST = "1.5.1"
#: The library call whose result the registry entry holds, as the key's variant.
LOADER = "causilo:load_pretrained_model"


def attach_shared_network(estimator: Any, *, task: str) -> dict[str, Any] | None:
    """Hand ``estimator`` an engine that holds the registry's network for its checkpoint and device.

    Returns ``{"key": WeightsKey, "present_before_fit": bool}`` for the model's ``get_info``, or ``None``
    when the estimator is left to load its own network: another causilo release, a local checkpoint, a
    device list or memory budgets (the engine places the network differently then).
    """
    import causilo
    from causilo.estimators import device_key
    from causilo.model_selection import RUNTIME15_FORMAT, runtime_for, select_checkpoint

    from tabarena.models import _weights

    if causilo.__version__ != WRITTEN_AGAINST:
        logger.warning(
            "Causilo-1.5 shares its network only with causilo %s, found %s; every fit loads its own.",
            WRITTEN_AGAINST,
            causilo.__version__,
        )
        return None
    if (
        estimator.checkpoint_path is not None
        or isinstance(estimator.device, (list, tuple))
        or estimator.device_memory_budget_bytes
    ):
        return None
    source = select_checkpoint(estimator.model_version, task)
    if source.architecture != RUNTIME15_FORMAT:
        return None
    engine = runtime_for(source.architecture).make_engine(task, device_key(estimator.device), source, memory_budgets={})
    inputs = {"task": task, "repository": source.repository, "revision": source.revision, "folder": source.folder}
    key = _weights.make_key("causilo", json.dumps(inputs, sort_keys=True), LOADER, engine.device)

    def load():
        # Places the network on the device and prepares the bf16 output projections, as the engine does.
        return causilo.load_pretrained_model(task, model_version=estimator.model_version, device=str(engine.device))

    present_before_fit = _weights.contains(key)
    network = _weights.get_or_load(key, load, stage="fit", source={"loader": LOADER})
    engine._model_replicas[engine.device] = network
    engine.model = network
    estimator._engine = engine
    estimator._checkpoint_request = (estimator.model_version, None)
    return {"key": key, "present_before_fit": present_before_fit}


def shared_network_info(record: dict[str, Any]) -> dict[str, Any]:
    """``info["shared_weights"]`` in the shape AutoGluon reports for a declaring wrapper."""
    from tabarena.models import _weights

    key = record["key"]
    return {
        **key.to_dict(),
        "inputs": json.loads(key.checkpoint),
        "checkpoint": None,
        "loaded_by": _weights.loaded_by(key),
        "present_before_fit": record["present_before_fit"],
    }
