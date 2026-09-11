"""The frozen Mitra-v2 deployment recipe: constants and torch-free helpers.

Everything the wrapper needs to know about *how* Mitra-v2 is deployed lives here, so that the
numbers are in one place and the pure logic can be unit-tested on CPU. The values are the
defaults of the ``mitra-finetune`` package (``huggingface.co/autogluon/mitra-finetune``,
v0.2.0, commit ``8ffe799``) and Section 3.1 of the Mitra-v2 technical report
(arXiv:2609.04540). They are frozen: the report selected them once and confirmed them
prospectively, so they are constants of the method rather than tunables.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# --- Checkpoints ------------------------------------------------------------------------------
#: Hugging Face repos of the second-generation checkpoints. The revisions are pinned so the
#: weights a fit uses never silently change when a repo's default branch moves (the weights were
#: uploaded on 2026-09-03; later commits only touch the model card). Bump deliberately.
HF_CLASSIFIER_REPO = "autogluon/mitra-classifier-2"
HF_CLASSIFIER_REVISION = "edada0d20759c58ada8c8605c25f22f6e98ea5f0"
HF_REGRESSOR_REPO = "autogluon/mitra-regressor-2"
HF_REGRESSOR_REVISION = "246c7d310fff04296a2388df5f448e3ffe809283"
#: The files of a ``Tab2D.save_pretrained`` checkpoint directory.
CHECKPOINT_FILES = ("config.json", "model.safetensors")

# --- Fine-tuning schedule -----------------------------------------------------------------------
FINE_TUNE_STEPS = 50
LEARNING_RATE = 1e-5
#: Binary tasks whose whole outer training table fits the fine-tuning support cap use a gentler
#: learning rate; multiclass and large tables keep the recipe rate.
SMALL_BINARY_LEARNING_RATE = 3e-6
SMALL_BINARY_MAX_ROWS = 16_384
WARMUP_STEPS = 10
WEIGHT_DECAY = 0.3
#: Wall-clock budget of the fine-tuning loop of one bag child under the TabArena one-hour
#: protocol (seconds). The 50-step loop stops early when it is exceeded.
FINE_TUNE_BUDGET_S = 250.0
#: Query rows per validation pass of the fine-tuning loop. Stock AutoGluon validates in chunks of
#: ``max_samples_query`` (1,024) rows and redraws the in-context support for every chunk, so the
#: validation pass that follows every step costs as much as several steps on large tables. One
#: wide chunk is a single forward pass; under out-of-memory it is halved back to the stock chunk.
#: Query rows attend only to the support, so the chunking does not change what is predicted.
FINETUNE_EVAL_QUERY_CHUNK = 16_384
#: Attention kernel of the fine-tuning loop. ``"sdpa"`` runs the loop's attention through PyTorch's
#: fused ``scaled_dot_product_attention`` (the path stock AutoGluon takes when ``flash-attn`` is
#: not installed); ``"stock"`` keeps the construction-time choice, ``flash_attn_varlen_func``
#: whenever the package imports. Both compute exact attention and agree to bf16 rounding. Per
#: fine-tuning step ``sdpa`` is as fast as flash-attn 2 on an H100 and 1.3 to 1.7 times faster on
#: an RTX PRO 6000 Blackwell. Prediction always keeps the stock choice: at prediction shapes
#: (up to 16,384 support and 16,384 query rows) the flash path is faster and needs far less memory.
FINETUNE_ATTENTION_BACKEND = "sdpa"

# --- In-context support -------------------------------------------------------------------------
#: Rows the fine-tuning loop may draw as in-context support (stock AutoGluon: 8,192).
FINETUNE_SUPPORT_CAP_CLASSIFICATION = 16_384
FINETUNE_SUPPORT_CAP_REGRESSION = 20_480
#: Rows a fitted child conditions on at prediction time. Binary is kept lower on purpose: on the
#: largest binary tables a wider context slowed the bag enough to be truncated at the time limit.
PREDICT_SUPPORT_CAP_BINARY = 16_384
PREDICT_SUPPORT_CAP_MULTICLASS = 32_768
PREDICT_SUPPORT_CAP_REGRESSION = 32_768
#: Under CUDA out-of-memory the prediction cap is halved down to this floor (the stock cap).
PREDICT_SUPPORT_FLOOR = 8_192
#: Query rows per prediction chunk when the whole support fits in context (stock: 1,024), and the
#: floor the out-of-memory ratchet halves it down to before it touches the support cap.
PREDICT_QUERY_CHUNK = 16_384
PREDICT_QUERY_CHUNK_FLOOR = 256

# --- Wide tables --------------------------------------------------------------------------------
#: Tables wider than this are reduced to the budget before fine-tuning (train-only).
MAX_FEATURES_BUDGET = 256
#: Classification reduces only predominantly continuous tables (fraction of columns with more
#: than two distinct values); wide binary or categorical tables keep every column.
FEATURE_SELECTION_MIN_CONTINUOUS_FRACTION = 0.5

_CLASSIFICATION_PROBLEM_TYPES = ("binary", "multiclass")
_BAGGED_CHILD_NAME = re.compile(r"S\d+F\d+$")


def is_classification(problem_type: str) -> bool:
    """Whether an AutoGluon ``problem_type`` is a classification task."""
    return problem_type in _CLASSIFICATION_PROBLEM_TYPES


def finetune_learning_rate(problem_type: str, n_outer_train_rows: int) -> float:
    """The fine-tuning learning rate for a task.

    ``n_outer_train_rows`` is the size of the whole training table the child is bagged from (its
    fit fold plus its held-out fold), which is what the recipe conditions on.
    """
    if problem_type == "binary" and n_outer_train_rows <= SMALL_BINARY_MAX_ROWS:
        return SMALL_BINARY_LEARNING_RATE
    return LEARNING_RATE


def default_finetune_support_cap(problem_type: str) -> int:
    """Rows the fine-tuning loop may draw as in-context support."""
    if is_classification(problem_type):
        return FINETUNE_SUPPORT_CAP_CLASSIFICATION
    return FINETUNE_SUPPORT_CAP_REGRESSION


def default_predict_support_cap(problem_type: str) -> int:
    """Rows a fitted child conditions on at prediction time."""
    if problem_type == "binary":
        return PREDICT_SUPPORT_CAP_BINARY
    if problem_type == "multiclass":
        return PREDICT_SUPPORT_CAP_MULTICLASS
    return PREDICT_SUPPORT_CAP_REGRESSION


def is_bagged_child_name(model_name: str) -> bool:
    """Whether an AutoGluon model name is that of a bagged-ensemble child.

    AutoGluon names the children of a bag ``<parent name>S<repeat>F<fold>`` (see
    ``BaggedEnsembleModel._fit_folds``). The wrapper uses this to tell a child's held-out fold,
    which the recipe adds to the prediction-time support, from an external validation set,
    which it does not.
    """
    return _BAGGED_CHILD_NAME.search(model_name) is not None


def balanced_binary_support_indices(
    y_support: np.ndarray,
    size: int,
    rng: np.random.RandomState,
) -> np.ndarray | None:
    """Class-balanced subsample of a binary support set, or ``None`` when it does not apply.

    Fills a per-class quota, smallest class first, releasing any unmet quota to the other class
    and topping up a shortfall at random. Returns ``None`` when no subsample is needed
    (``size`` covers every row) or the labels are not binary, in which case the caller draws
    uniformly at random as stock AutoGluon does.
    """
    y = np.asarray(y_support).reshape(-1)
    n = y.shape[0]
    if size >= n:
        return None
    classes, counts = np.unique(y, return_counts=True)
    if classes.shape[0] != 2:
        return None
    budget = int(size)
    parts = []
    for i, class_index in enumerate(np.argsort(counts)):
        quota = budget // (classes.shape[0] - i)
        take = int(min(int(counts[class_index]), quota))
        budget -= take
        pool = np.flatnonzero(y == classes[class_index])
        parts.append(pool if take >= pool.shape[0] else rng.choice(pool, size=take, replace=False))
    chosen = np.concatenate(parts)
    if chosen.shape[0] < size:
        rest = np.setdiff1d(np.arange(n), chosen)
        chosen = np.concatenate([chosen, rng.choice(rest, size=size - chosen.shape[0], replace=False)])
    return chosen


def mean_decode_bins(logits: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Decode bin logits to the softmax-weighted mean of the bin centers.

    Stock AutoGluon decodes Mitra's cross-entropy regression head with the most likely bin;
    the recipe uses the mean of the predicted distribution, which the report found to lower RMSE
    materially. Non-finite logits are clamped first so one bad row cannot poison the softmax.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(logits).all():
        logits = np.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
    edges = np.asarray(bin_edges, dtype=np.float64)
    centers = edges[:-1] + (edges[1] - edges[0]) / 2
    logits = logits - logits.max(axis=-1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=-1, keepdims=True)
    return probs @ centers


@dataclass(frozen=True)
class WideTableReducer:
    """Train-only column reduction for tables wider than the feature budget.

    Classification keeps the ``budget`` columns with the largest ANOVA F-statistic, and only on
    predominantly continuous tables (the dtype gate); regression projects onto ``budget``
    principal components of the mean-imputed, standardized table. Both are fit on the training
    rows alone and applied unchanged to validation and test rows. ``fit`` returns ``None`` when
    the table needs no reduction, so callers keep the frame as is.
    """

    kind: str
    """``"select"`` (column indices) or ``"project"`` (a fitted sklearn pipeline)."""
    columns: np.ndarray | None = None
    projector: object | None = None

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        *,
        problem_type: str,
        budget: int = MAX_FEATURES_BUDGET,
    ) -> WideTableReducer | None:
        """Fit the reduction on the training table, or return ``None`` when it does not apply."""
        X = np.asarray(X, dtype=np.float64)
        n_rows, n_features = X.shape
        if budget <= 0 or n_features <= budget:
            return None
        from sklearn.impute import SimpleImputer

        X_imputed = SimpleImputer(strategy="mean", keep_empty_features=True).fit_transform(X)
        if is_classification(problem_type):
            n_continuous = sum(np.unique(X_imputed[:, j]).shape[0] > 2 for j in range(n_features))
            if n_continuous < n_features * FEATURE_SELECTION_MIN_CONTINUOUS_FRACTION:
                return None
            from sklearn.feature_selection import f_classif

            with np.errstate(divide="ignore", invalid="ignore"):
                scores, _ = f_classif(X_imputed, np.asarray(y))
            scores = np.where(np.isfinite(scores), scores, -np.inf)
            # Rank by score without re-sorting the survivors, as the reference pipeline does.
            return cls(kind="select", columns=np.argsort(scores)[::-1][:budget])

        from sklearn.decomposition import PCA
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        n_components = int(min(budget, n_rows, n_features))
        projector = make_pipeline(
            SimpleImputer(strategy="mean", keep_empty_features=True),
            StandardScaler(),
            PCA(n_components=n_components, random_state=0),
        )
        projector.fit(X)
        return cls(kind="project", projector=projector)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted reduction to a table with the training columns."""
        X = np.asarray(X, dtype=np.float64)
        if self.kind == "select":
            return X[:, self.columns]
        return self.projector.transform(X)

    @property
    def n_features_out(self) -> int:
        """Number of columns the reduced table has."""
        if self.kind == "select":
            return int(self.columns.shape[0])
        return int(self.projector[-1].n_components_)


@dataclass(frozen=True)
class RecipeSettings:
    """The per-fit recipe values handed to the estimator by the AutoGluon wrapper."""

    weight_decay: float = WEIGHT_DECAY
    finetune_support_cap: int = FINETUNE_SUPPORT_CAP_CLASSIFICATION
    predict_support_cap: int = PREDICT_SUPPORT_CAP_MULTICLASS
    predict_support_floor: int = PREDICT_SUPPORT_FLOOR
    predict_query_chunk: int = PREDICT_QUERY_CHUNK
    predict_query_chunk_floor: int = PREDICT_QUERY_CHUNK_FLOOR
    balanced_binary_support: bool = True
    fine_tune_budget: float | None = FINE_TUNE_BUDGET_S
    heldout_in_support: bool = True
    """Whether the validation rows given to ``fit`` join the prediction-time support once fitting
    and validation are over (the recipe's heldout-in-support rule for bag children)."""
    finetune_eval_query_chunk: int = FINETUNE_EVAL_QUERY_CHUNK
    """Query rows per validation pass during fine-tuning (:data:`FINETUNE_EVAL_QUERY_CHUNK`)."""
    finetune_memory_preflight: bool = True
    """Whether one throw-away forward and backward pass at the fine-tuning context size runs before
    the first validation pass, so a context that does not fit the GPU fails in seconds rather than
    after a full validation pass at that size."""
    finetune_attention_backend: str = FINETUNE_ATTENTION_BACKEND
    """``"sdpa"`` or ``"stock"`` (:data:`FINETUNE_ATTENTION_BACKEND`)."""
    n_bins: int | None = None
    """Width of the regression head (bins); ``None`` for classification."""

    @classmethod
    def for_problem_type(
        cls,
        problem_type: str,
        *,
        finetune_support_cap: int | None = None,
        predict_support_cap: int | None = None,
        **overrides,
    ) -> RecipeSettings:
        """The recipe for a task: the two support caps default to their task-dependent values.

        ``None`` for either cap selects the recipe value; every other keyword is stored as given
        (``fine_tune_budget=None`` is a real setting: no fixed budget, the fold's time limit).
        """
        if finetune_support_cap is None:
            finetune_support_cap = default_finetune_support_cap(problem_type)
        if predict_support_cap is None:
            predict_support_cap = default_predict_support_cap(problem_type)
        return cls(finetune_support_cap=finetune_support_cap, predict_support_cap=predict_support_cap, **overrides)
