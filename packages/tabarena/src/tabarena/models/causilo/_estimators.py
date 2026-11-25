"""Developer fix: the fits of Causilo-1.5 share one network per process through the weights registry.

causilo 1.5.0 builds the 1.5 network inside ``Engine.initialize_model`` with
``causilo.v1_5.checkpoints.load_model(task, source)``, and AutoGluon's ``SharedWeights`` cannot take that
call over. The registry records a function loader's inputs to rebuild the network in another process, and
``source`` is a ``ModelSource`` dataclass it can only record as its ``repr``. The weightless pickle that
sharing writes (the estimator's attributes, without causilo's own ``export_estimator``) also fails on the
``threading.Lock`` that every fitted execution runner holds.

causilo's own pickle already leaves the network out, so the wrapper keeps it and shares only the load.
:func:`attach_shared_network` takes the placed network from the registry (:mod:`tabarena.models._weights`)
and gives the estimator, before ``fit``, the engine its ``fit`` would build, holding that network. ``fit``
reuses an engine whose task, checkpoint, device and memory budgets match the estimator's parameters; if a
later causilo changed those conditions, ``fit`` would build its own engine and load its own network,
slower but with the same predictions. A pickled fit reloads its own network through causilo's
``import_estimator`` when the benchmark loads it, outside the timed fit.

Upstream ask (nums-ai/causilo): a loader whose inputs are plain values, the device among them, and fitted
state that pickles without the library's ``__getstate__`` (the runners' lock), as causilo 1.0.2's
``load_pretrained_model(task)`` offered for 1.0. ``Causilo15Model`` then declares ``shared_weights`` like
``CausiloModel`` and this module goes. Written against causilo 1.5.0.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: The causilo release whose engine this module builds; any other release fits without sharing.
WRITTEN_AGAINST = "1.5.0"
#: The library call whose result the registry entry holds, as the key's variant.
LOADER = "causilo.v1_5.checkpoints:load_model"


def attach_shared_network(estimator: Any, *, task: str) -> dict[str, Any] | None:
    """Hand ``estimator`` an engine that holds the registry's network for its checkpoint and device.

    Returns ``{"key": WeightsKey, "present_before_fit": bool}`` for the model's ``get_info``, or ``None``
    when the estimator is left to load its own network: another causilo release, a local checkpoint, a
    device list or memory budgets (the engine places the network differently then).
    """
    import causilo
    from causilo.estimators import device_key
    from causilo.model_selection import RUNTIME15_FORMAT, runtime_for, select_checkpoint
    from causilo.v1_5 import checkpoints

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
        network, _ = checkpoints.load_model(task, source)
        # As ``initialize_model`` does: move it to the device and prepare the bf16 output projections.
        return engine._place_model(network, engine.device)

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
