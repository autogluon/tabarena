from __future__ import annotations

from autogluon.common.space import Categorical

from tabarena.models.fintfm_linprobe.model import FinTFMLinearProbeModel
from tabarena.utils.config_utils import ConfigGenerator

#: PROPOSED search space for "FinTFM with a trained linear readout", not a tuned one. No
#: sampled config has been run on TabArena. The keys are the hyperparameters ``model.py`` reads,
#: and the first value of each is the measured entry's value (fintfm ``docs/results/FINDINGS.md``
#: §232, §235, §237), so the default config is the entry and a test pins that:
#:
#: - ``probe_binary``, ``probe_multiclass``, ``probe_regression``: the readout options dict.
#:   ``scale`` standardises the readout's inputs, ``cv`` chooses the logistic regression's ``C``
#:   by 3-fold log loss over 10^-3 to 10^2, ``raw`` appends the row's own squashed features.
#:   The candidates are the entry and its ablations (without ``raw``; plain). The regression
#:   readout is a ridge and reads only ``raw``. ``head`` is left out: it is refused while
#:   ``probe_crossfit_folds`` is 2 or more, which is the default.
#: - ``partition_binary``, ``partition_multiclass``: the tree partition's arguments. ``{}`` is
#:   one bag at the class defaults (the binary entry), ``{"n_bags": 8}`` the multiclass entry.
#:   The other bag counts are candidates that have not been measured on TabArena. Regression is
#:   not sampled: ``partition_regression`` stays at the entry's 4 bags.
#:
#: Left out on purpose: ``no_rep`` (a control, not a candidate), the partition depth and leaf
#: floor (only depth 2 and 0.05 were measured), ``max_context`` (4,000 changed nothing on
#: multiclass, §224). A sampled dict replaces the entry's whole dict for its problem type. The
#: readout adds a fit of up to 20,000 representation rows per leaf, so a tuned run is GPU work,
#: as for FinTFM.
#:
#: The default config is ``manual_configs=[{}]``: it carries no override, so it is whatever the
#: wrapper's own defaults are, which must equal the measured entry.
search_space = {
    "probe_binary": Categorical(
        {"scale": True, "cv": True, "raw": True},
        {"scale": True, "cv": True},
        {},
    ),
    "probe_multiclass": Categorical(
        {"scale": True, "cv": True, "raw": True},
        {"scale": True, "cv": True},
        {},
    ),
    "probe_regression": Categorical({"raw": True}, {}),
    "partition_binary": Categorical({}, {"n_bags": 4}),
    "partition_multiclass": Categorical({"n_bags": 8}, {"n_bags": 4}, {}),
}

#: PROPOSAL, not part of ``search_space``: ``probe_crossfit_folds`` (``model.py``) is 5 by default,
#: cross-fitting the readout's training rows against a context that leaves each out. The measured
#: 932 Elo entry was produced with it off (``0``). Sampling it only makes sense after the default
#: has been re-measured, to ask whether 2, 3, 5 or 10 folds matter; until then it is a setting to
#: fix, not to tune. ``0`` also makes the ``head`` probe option available again.
proposed_search_space_additions = {
    "probe_crossfit_folds": Categorical(5, 0),
}

gen_fintfm_linprobe = ConfigGenerator(
    model_cls=FinTFMLinearProbeModel,
    search_space=search_space,
    manual_configs=[{}],
)
