"""The inner validation protocol of a benchmark run, as one object.

A :class:`ValidationProtocol` states how a bagged model is validated inside one outer split: the
bagging fold and repeat counts, an optional tiny-data regime with its own counts, whether the inner
splits follow the task's group / time structure, and whether AutoGluon may shrink the fold count to
the minority-class count. It is the only place these numbers live:

* an arena context declares its official protocol (:data:`TABARENA_V0PT1_VALIDATION_PROTOCOL` for
  ``TabArenaContext``, :data:`BEYONDARENA_VALIDATION_PROTOCOL` for ``BeyondArenaContext``) and stamps it onto
  the experiments it runs,
  asserting that every bagged experiment agrees with it unless the context was built with
  ``official_validation_protocol=False``;
* a bundle, a ``ModelJob`` or a single experiment may carry an explicit protocol (the opt-out at group or config
  level);
* the AutoGluon wrappers resolve the counts from the stamped protocol right before ``TabularPredictor.fit`` and
  record what they resolved and what was fitted in every result (``results["validation_protocol"]``).

Which flavour of experiment is official is a property of the context (``OFFICIAL_VALIDATION_FLAVOURS``): bagged
models and systems are official; outer, holdout and full-predictor fits run outside the protocol and are recorded
as such, without needing any flag.

Identity is the field-derived :meth:`ValidationProtocol.key` (``"8x1"``, ``"8x1+tiny5x5<=500+task-specific+adapt-classes"``),
never the ``name`` label, so a relabelled custom protocol can never read as official. The three label fields
(``name``, ``arena``, ``enforced``) are excluded from equality: ``enforced`` is ``True`` when an enforcing context
stamped the protocol, ``False`` when a context in opt-out mode did (or the flavour is non-official by construction),
and ``None`` when no context was involved.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, replace
from typing import Any, Literal

ValidationFlavour = Literal["bagged", "holdout", "outer", "system", "predictor", "bag-child-holdout"]
"""How an experiment validates: ``bagged`` (k-fold bag, the protocol applies in full), ``holdout`` (one
task-aware train/validation split sized like one bagging fold), ``outer`` (no inner validation),
``system`` (the system carves its own validation), ``predictor`` (a full ``TabularPredictor`` whose presets
decide), ``bag-child-holdout`` (a holdout result derived from one child of a bagged result)."""

DEFAULT_SEED_BLOCK_SIZE = 8
"""Seed block per config for ``fold-config-wise`` seeding when no protocol is known at generation time
(the protocol arrives later from the arena context). Equals the TabArena protocol's 8 children, which is
what every official run used."""

_STRUCTURE_FIELDS = ("group_on", "time_on", "stratify_on", "group_labels")


class ValidationProtocolError(ValueError):
    """An experiment, bundle or task collection deviates from the arena's enforced validation protocol."""


@dataclass(frozen=True)
class ValidationProtocol:
    """How a bagged model is validated inside one outer split.

    Attributes:
        num_bag_folds: Bagging folds (at least 2).
        num_bag_sets: Bagging repeats (at least 1).
        tiny_num_bag_folds: Folds of the tiny-data regime, or ``None`` when the protocol has no such regime.
        tiny_num_bag_sets: Repeats of the tiny-data regime, or ``None``.
        tiny_max_group_instances: The regime applies at or below this many training (group) instances; the
            three ``tiny_*`` fields are set together or not at all.
        task_specific_validation: Build the inner splits from the task's group / time structure (and a
            task-aware holdout split for holdout fits) instead of plain stratified folds.
        adapt_num_folds_to_n_classes: Let AutoGluon shrink the fold count to the minority-class count
            (``TabularPredictor.fit(adapt_num_bag_folds_to_n_classes=True)``).
        name: Display label, e.g. ``"TabArena-v0.1"``. Excluded from equality.
        arena: ``benchmark_name`` of the context that stamped this protocol; ``None`` when set by hand.
            Excluded from equality.
        enforced: Whether the stamping context asserted this protocol (``True``), stamped it in opt-out mode or
            on a non-official flavour (``False``), or no context was involved (``None``). Excluded from equality.
    """

    num_bag_folds: int = 8
    num_bag_sets: int = 1
    tiny_num_bag_folds: int | None = None
    tiny_num_bag_sets: int | None = None
    tiny_max_group_instances: int | None = None
    task_specific_validation: bool = False
    adapt_num_folds_to_n_classes: bool = False
    name: str | None = field(default=None, compare=False)
    arena: str | None = field(default=None, compare=False)
    enforced: bool | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.num_bag_folds, int) or isinstance(self.num_bag_folds, bool) or self.num_bag_folds < 2:
            raise ValueError(f"num_bag_folds must be an int >= 2, got {self.num_bag_folds!r}")
        if not isinstance(self.num_bag_sets, int) or isinstance(self.num_bag_sets, bool) or self.num_bag_sets < 1:
            raise ValueError(f"num_bag_sets must be an int >= 1, got {self.num_bag_sets!r}")
        tiny = (self.tiny_num_bag_folds, self.tiny_num_bag_sets, self.tiny_max_group_instances)
        if any(v is None for v in tiny) and any(v is not None for v in tiny):
            raise ValueError(
                "tiny_num_bag_folds, tiny_num_bag_sets and tiny_max_group_instances must be set together "
                f"(or all be None), got {tiny}"
            )
        if self.tiny_num_bag_folds is not None:
            if self.tiny_num_bag_folds < 2:
                raise ValueError(f"tiny_num_bag_folds must be >= 2, got {self.tiny_num_bag_folds!r}")
            if self.tiny_num_bag_sets < 1:
                raise ValueError(f"tiny_num_bag_sets must be >= 1, got {self.tiny_num_bag_sets!r}")
            if self.tiny_max_group_instances < 0:
                raise ValueError(f"tiny_max_group_instances must be >= 0, got {self.tiny_max_group_instances!r}")

    # --- Resolution ------------------------------------------------------------------------------------
    @property
    def has_tiny_regime(self) -> bool:
        """Whether a tiny-data regime is declared."""
        return self.tiny_num_bag_folds is not None

    def regime(self, num_group_instances: int | None) -> Literal["default", "tiny"]:
        """The regime for a training split with ``num_group_instances`` (group) instances.

        ``None`` means the size is unknown or must not decide, which is the default regime.
        """
        if (
            self.has_tiny_regime
            and num_group_instances is not None
            and num_group_instances <= self.tiny_max_group_instances
        ):
            return "tiny"
        return "default"

    def resolve_num_splits(self, num_group_instances: int | None) -> tuple[int, int]:
        """The ``(num_bag_folds, num_bag_sets)`` to fit for a training split of the given size."""
        if self.regime(num_group_instances) == "tiny":
            return self.tiny_num_bag_folds, self.tiny_num_bag_sets
        return self.num_bag_folds, self.num_bag_sets

    # --- Identity and display -----------------------------------------------------------------------------
    def key(self) -> str:
        """Compact identity built from the compared fields only, e.g. ``"8x1"`` or
        ``"8x1+tiny5x5<=500+task-specific+adapt-classes"``. Always starts with ``<folds>x<sets>``.
        """
        parts = [f"{self.num_bag_folds}x{self.num_bag_sets}"]
        if self.has_tiny_regime:
            parts.append(f"tiny{self.tiny_num_bag_folds}x{self.tiny_num_bag_sets}<={self.tiny_max_group_instances}")
        if self.task_specific_validation:
            parts.append("task-specific")
        if self.adapt_num_folds_to_n_classes:
            parts.append("adapt-classes")
        return "+".join(parts)

    def describe(self) -> str:
        """Human-readable summary for messages, e.g. ``"TabArena-v0.1 (8 folds x 1 set)"``."""
        details = [f"{self.num_bag_folds} folds x {self.num_bag_sets} {'set' if self.num_bag_sets == 1 else 'sets'}"]
        if self.has_tiny_regime:
            details.append(
                f"{self.tiny_num_bag_folds} x {self.tiny_num_bag_sets} at or below "
                f"{self.tiny_max_group_instances} training group instances"
            )
        if self.task_specific_validation:
            details.append("task-specific inner splits")
        if self.adapt_num_folds_to_n_classes:
            details.append("class-adaptive folds")
        return f"{self.name or 'custom'} ({'; '.join(details)})"

    @property
    def is_official(self) -> bool:
        """Whether the compared fields equal one of the registered official protocols."""
        return any(self == official for official in OFFICIAL_VALIDATION_PROTOCOLS.values())

    # --- (De)serialization ----------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Plain dict of every field (labels included); JSON and YAML safe."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationProtocol:
        """Rebuild from :meth:`to_dict` output; unknown keys are ignored so newer records still load."""
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    @classmethod
    def from_config(cls, config: ValidationProtocol | dict[str, Any] | None) -> ValidationProtocol | None:
        """Normalize a constructor input: an instance is returned as is, a dict is parsed, ``None`` stays ``None``."""
        if config is None or isinstance(config, ValidationProtocol):
            return config
        if isinstance(config, dict):
            return cls.from_dict(config)
        raise TypeError(
            f"validation_protocol must be a ValidationProtocol, a dict or None, got {type(config).__name__}"
        )

    def with_origin(self, *, arena: str | None, enforced: bool | None) -> ValidationProtocol:
        """Copy with the ``arena`` / ``enforced`` labels set (the compared fields are untouched)."""
        return replace(self, arena=arena, enforced=enforced)

    @classmethod
    def custom(cls, num_bag_folds: int, num_bag_sets: int = 1, **fields: Any) -> ValidationProtocol:
        """A user protocol with the given counts, named after them unless ``name`` is passed."""
        fields.setdefault("name", f"custom-{num_bag_folds}x{num_bag_sets}")
        return cls(num_bag_folds=num_bag_folds, num_bag_sets=num_bag_sets, **fields)


TABARENA_V0PT1_VALIDATION_PROTOCOL = ValidationProtocol(name="TabArena-v0.1")
"""TabArena-v0.1: 8 folds x 1 set on every task, plain stratified inner splits, no class adaptation."""

BEYONDARENA_VALIDATION_PROTOCOL = ValidationProtocol(
    name="BeyondArena",
    tiny_num_bag_folds=5,
    tiny_num_bag_sets=5,
    tiny_max_group_instances=500,
    task_specific_validation=True,
    adapt_num_folds_to_n_classes=True,
)
"""BeyondArena: 8 folds x 1 set, 5 x 5 at or below 500 training group instances, group / time-aware inner
splits (with their data-dependent clamps) and class-adaptive folds."""

OFFICIAL_VALIDATION_PROTOCOLS: dict[str, ValidationProtocol] = {
    p.name: p for p in (TABARENA_V0PT1_VALIDATION_PROTOCOL, BEYONDARENA_VALIDATION_PROTOCOL)
}
"""The registered official protocols by name."""


@dataclass(frozen=True)
class ValidationExpectation:
    """What an arena context expects of the experiments of a run, shipped to compute nodes in the ``JobBatch``.

    Attributes:
        protocol: The context's protocol (``None`` for a context without one).
        enforced: Whether the context asserts it on bagged experiments.
        arena: The context's ``benchmark_name``.
        official_flavours: Experiment flavours the arena treats as official.
    """

    protocol: ValidationProtocol | None
    enforced: bool
    arena: str
    official_flavours: tuple[str, ...] = ("bagged", "system")

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-safe dict."""
        return {
            "protocol": None if self.protocol is None else self.protocol.to_dict(),
            "enforced": self.enforced,
            "arena": self.arena,
            "official_flavours": list(self.official_flavours),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationExpectation:
        """Rebuild from :meth:`to_dict` output (a record without ``official_flavours`` uses the default)."""
        protocol = data.get("protocol")
        return cls(
            protocol=None if protocol is None else ValidationProtocol.from_dict(protocol),
            enforced=bool(data["enforced"]),
            arena=data["arena"],
            official_flavours=tuple(data.get("official_flavours") or ("bagged", "system")),
        )


@dataclass(frozen=True)
class ValidationResolution:
    """What an AutoGluon wrapper resolved from the protocol for one fit, before ``TabularPredictor.fit``.

    Attributes:
        num_group_instances: Training (group) instances the regime decision saw; ``None`` when the size did
            not decide.
        regime: ``"default"``, ``"tiny"``, or ``"explicit"`` for counts given directly to a full predictor.
        num_bag_folds_nominal: Folds the protocol asked for (before the data-dependent clamps).
        num_bag_sets_nominal: Repeats the protocol asked for.
        num_bag_folds_resolved: Folds handed to AutoGluon (after the clamps); ``None`` for a holdout fit.
        num_bag_sets_resolved: Repeats handed to AutoGluon; ``None`` for a holdout fit.
        clamps: Names of the data-dependent reductions that applied.
        custom_splits: Whether explicit group / time-aware folds were passed.
        num_custom_splits: How many custom splits were passed.
        task_specific_holdout: Whether a task-aware holdout split replaced AutoGluon's default one.
        holdout_rows: Rows of that holdout split.
        structure: The effective split structure (``group_on``, ``time_on``, ``stratify_on``, ``group_labels``).
    """

    num_group_instances: int | None = None
    regime: Literal["default", "tiny", "explicit"] | None = None
    num_bag_folds_nominal: int | None = None
    num_bag_sets_nominal: int | None = None
    num_bag_folds_resolved: int | None = None
    num_bag_sets_resolved: int | None = None
    clamps: tuple[str, ...] = ()
    custom_splits: bool = False
    num_custom_splits: int | None = None
    task_specific_holdout: bool = False
    holdout_rows: int | None = None
    structure: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        """Plain dict for the result record (``clamps`` as a list)."""
        record = dataclasses.asdict(self)
        record["clamps"] = list(self.clamps)
        return record


def structure_of(validation_metadata: Any) -> dict[str, Any]:
    """The split-structure fields of a ``ValidationMetadata`` as a plain dict (``group_labels`` as its value)."""
    out = {}
    for name in _STRUCTURE_FIELDS:
        value = getattr(validation_metadata, name, None)
        out[name] = getattr(value, "value", value)
    return out


def validation_protocol_key(record: dict[str, Any] | None) -> str | None:
    """The identity string of a result's ``validation_protocol`` record.

    ``None`` without a record (results that predate it); ``"system"`` for systems; the protocol key for bagged
    fits (``"custom"`` when a bagged record carries no protocol); ``"<flavour>:<key>"`` or the flavour word for
    every other flavour, so a holdout, outer or predictor fit can never read as an official bagged protocol.
    """
    if not isinstance(record, dict) or not record:
        return None
    flavour = record.get("flavour")
    protocol = record.get("protocol")
    key = ValidationProtocol.from_dict(protocol).key() if isinstance(protocol, dict) else None
    if flavour == "system":
        return "system"
    if flavour == "bagged":
        return key if key is not None else "custom"
    if flavour is None:
        return "custom" if key is None else f"custom:{key}"
    return f"{flavour}:{key}" if key is not None else str(flavour)


def experiment_violations(
    experiment: Any,
    *,
    protocol: ValidationProtocol,
    arena: str,
    bundle_hint: str | None = None,
) -> list[str]:
    """Why a *bagged* experiment does not comply with an enforced arena protocol (empty when it does).

    Only bagged experiments (``VALIDATION_FLAVOUR == "bagged"``) can violate a protocol: their stamp must equal
    the arena's protocol (a missing stamp is fine, the context fills it) and they must not override the task's
    split structure through ``method_kwargs["validation_metadata"]``. Every other flavour returns no violation.
    Objects without a ``VALIDATION_FLAVOUR`` attribute (test stand-ins) are ignored.
    """
    if getattr(experiment, "VALIDATION_FLAVOUR", None) != "bagged":
        return []
    name = getattr(experiment, "name", repr(experiment))
    violations: list[str] = []
    stamp = getattr(experiment, "validation_protocol", None)
    if stamp is not None and stamp != protocol:
        hint = f" ({arena} pairs with {bundle_hint})" if bundle_hint else ""
        violations.append(
            f"{name!r} carries validation protocol {stamp.describe()} [{stamp.key()}] but {arena} requires "
            f"{protocol.describe()} [{protocol.key()}]{hint}"
        )
    method_kwargs = getattr(experiment, "method_kwargs", None) or {}
    if isinstance(method_kwargs, dict) and method_kwargs.get("validation_metadata") is not None:
        violations.append(
            f"{name!r} overrides the task's split structure via method_kwargs['validation_metadata']; the task "
            f"collection is the only source of split structure under an enforced protocol"
        )
    return violations


def format_violations(violations: list[str], *, arena: str) -> str:
    """One error message for a list of violations, ending with how to fix them."""
    lines = "\n  - ".join(violations)
    return (
        f"[{arena} official validation protocol] {len(violations)} experiment(s) deviate:\n  - {lines}\n"
        "Fix: build the experiments with the bundle paired to this context and without an explicit "
        "validation_protocol, or pass official_validation_protocol=False to the context to run them under a custom "
        "protocol (recorded and marked as such)."
    )


def check_experiments(experiments: list[Any], *, expectation: ValidationExpectation) -> None:
    """Assert every experiment complies with an enforced :class:`ValidationExpectation` (the worker-side re-check).

    A no-op when the expectation is not enforced or has no protocol. Raises :class:`ValidationProtocolError`
    listing every offender.
    """
    if not expectation.enforced or expectation.protocol is None:
        return
    violations: list[str] = []
    for experiment in experiments:
        violations += experiment_violations(experiment, protocol=expectation.protocol, arena=expectation.arena)
    if violations:
        raise ValidationProtocolError(format_violations(violations, arena=expectation.arena))
