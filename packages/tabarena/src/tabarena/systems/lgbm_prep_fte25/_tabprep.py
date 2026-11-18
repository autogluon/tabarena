"""TabPrep feature engineering for the LGBM-Prep bag child, on AutoGluon's public generators.

Three parts, each a port of the code the reference runs used:

* :class:`TabPrepFeatureGenerator` -- the scikit-learn wrapper of TabPrep (``atschalz/tabprep`` at commit
  ``fee7f189``): group-by statistics, random-subset feature compression, arithmetic combinations, categorical
  interactions and out-of-fold target encoding, composed from ``autogluon.features.generators``.
* :class:`FastGroupByFeatureGenerator` -- the same group statistics and percentile ranks as AutoGluon's
  ``GroupByFeatureGenerator`` with the group lookups vectorised (bit-identical output).
* :class:`SortedOOFTargetEncodingFeatureGenerator` and :class:`StringKeyRSFCGenerator` -- AutoGluon's out-of-fold
  target encoder and random-subset compression with the category order of the reference runs (sorted category
  values; string hash keys for the subsets). The public generators number categories in order of appearance,
  which changes floating-point summation order; on some tasks that makes a near-constant feature an exact
  duplicate of another and it is dropped. Pinning the order keeps the features those runs built.
* :func:`select_columns` -- the 2,000-feature cap: columns ranked by absolute Spearman correlation with the
  target on the training rows only, ties broken by column position, original order kept.

:func:`tabprep_frames` runs all three on one bag child: fit on the child's training rows, transform its
validation rows, cap. Rows that are scored are never used to fit anything here.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd
from autogluon.features.generators import (
    ArithmeticFeatureGenerator,
    BulkFeatureGenerator,
    CategoricalInteractionFeatureGenerator,
    GroupByFeatureGenerator,
    OOFTargetEncodingFeatureGenerator,
    RandomSubsetFeatureCompressionGenerator,
)
from autogluon.features.generators.abstract import AbstractFeatureGenerator
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from threadpoolctl import threadpool_limits

FEATURE_CAP = 2000
"""At most this many columns enter LightGBM after TabPrep."""


# ---------------------------------------------------------------------------------------------------------------------
# Group-by statistics with vectorised lookups
# ---------------------------------------------------------------------------------------------------------------------


def segmented_percentile_rank(codes: np.ndarray, values: np.ndarray, distributions: np.ndarray) -> np.ndarray:
    """Right-sided ``searchsorted`` of each row in its group's fitted distribution, divided by its size."""
    result = np.full(len(values), 0.5, dtype=float)
    sizes = np.fromiter((0 if x is None else len(x) for x in distributions), dtype=np.int64, count=len(distributions))
    if not sizes.sum():
        return result
    starts = np.r_[0, np.cumsum(sizes)]
    rows = np.flatnonzero((codes >= 0) & ~np.isnan(values))
    rows = rows[sizes[codes[rows]] > 0]
    if not len(rows):
        return result
    groups = np.unique(codes[rows])
    if len(groups) < 64:
        # A handful of long distributions is faster with one searchsorted per group.
        for group in groups:
            selected = rows[codes[rows] == group]
            arr = distributions[group]
            result[selected] = np.searchsorted(arr, values[selected], side="right") / len(arr)
        return result
    flat = np.concatenate([x for x in distributions if x is not None and len(x)])
    group = codes[rows]
    low = starts[group].copy()
    high = starts[group + 1].copy()
    query = values[rows]
    # Binary search over many independent sorted segments at once.
    active = low < high
    while active.any():
        positions = np.flatnonzero(active)
        mid = (low[positions] + high[positions]) // 2
        right = flat[mid] <= query[positions]
        low[positions[right]] = mid[right] + 1
        high[positions[~right]] = mid[~right]
        active = low < high
    result[rows] = (low - starts[group]) / sizes[group]
    return result


class FastGroupByFeatureGenerator(GroupByFeatureGenerator):
    """``GroupByFeatureGenerator`` with the transform's group lookups vectorised; same output."""

    def _transform(self, X):
        X = self._to_dataframe(X)

        if len(getattr(self, "pairs_", [])) == 0:
            empty = pd.DataFrame(index=X.index)
            return empty if self.return_dataframe else empty.values

        _group_aggs, rowwise_aggs = self._split_aggs()
        drop_basic = self._drop_basic()
        rel_enabled = self._relative_enabled()

        n = len(X)
        m = len(self.output_columns_)
        # One generated column at a time: a Fortran-ordered matrix keeps each column contiguous.
        out = np.empty((n, m), dtype=float, order="F")
        col_i = 0

        used_nums = {num for _, num in self.pairs_}
        num_cache = {num: X[num].astype(float).to_numpy(copy=False) for num in used_nums}

        codes_cache: dict = {}
        missing_cache: dict = {}
        safe_cache: dict = {}

        for cat, num in self.pairs_:
            x = num_cache[num]
            idx = self.group_index_.get((cat, num), None)
            vals_dict = self.group_values_.get((cat, num), None)
            mapped = {}

            # Category codes once per categorical column, from the first index met for it.
            if cat not in codes_cache:
                codes = np.full(n, -1, dtype=int) if idx is None else idx.get_indexer(X[cat].to_numpy())
                codes_cache[cat] = codes
                missing = codes == -1
                missing_cache[cat] = missing
                safe_codes = codes.copy()
                safe_codes[missing] = 0
                safe_cache[cat] = safe_codes

            codes = codes_cache[cat]
            missing = missing_cache[cat]
            safe_codes = safe_cache[cat]

            # Group aggregations. An empty group table has no value to take for unseen categories.
            if idx is not None and len(idx) > 0 and vals_dict is not None and len(vals_dict) > 0:
                for agg, vals in vals_dict.items():
                    col = vals.take(safe_codes).astype(float, copy=False)
                    if missing.any():
                        col = col.copy()
                        col[missing] = np.nan
                    mapped[agg] = col
                    if not drop_basic:
                        out[:, col_i] = (
                            col if self.fill_value is np.nan else np.where(np.isnan(col), self.fill_value, col)  # noqa: PLW0177 (AutoGluon's identity test)
                        )
                        col_i += 1
            else:
                group_stats = self.group_stats_.get((cat, num), None)
                if group_stats is not None and len(group_stats.columns) > 0:
                    cat_series = X[cat]
                    for agg in group_stats.columns:
                        col = cat_series.map(group_stats[agg]).astype(float).to_numpy(copy=False)
                        mapped[agg] = col
                        if not drop_basic:
                            out[:, col_i] = (
                                col if self.fill_value is np.nan else np.where(np.isnan(col), self.fill_value, col)  # noqa: PLW0177 (AutoGluon's identity test)
                            )
                            col_i += 1

            # Percentile rank within the group, always emitted when requested.
            if "pct_rank" in rowwise_aggs:
                pr = np.full(n, 0.5, dtype=float)
                valid = (codes != -1) & (~np.isnan(x))
                if valid.any():
                    keys = self.pct_rank_keys_.get((cat, num), None)
                    dists = self.pct_rank_vals_.get((cat, num), None)
                    if idx is not None and keys is not None and dists is not None and len(keys) > 0:
                        dist_by_code = self.pct_rank_by_code_.get((cat, num), None)
                        if dist_by_code is None:
                            dist_by_code = np.empty(len(idx), dtype=object)
                            dist_by_code[:] = None
                            pos = idx.get_indexer(keys)
                            ok = pos != -1
                            dist_by_code[pos[ok]] = dists[ok]
                            self.pct_rank_by_code_[(cat, num)] = dist_by_code
                        pr = segmented_percentile_rank(codes, x, dist_by_code)
                out[:, col_i] = pr
                col_i += 1

            # Differences and ratios to the group statistics.
            if rel_enabled:
                missing_aggs = set(self.relative_to_aggs) - set(mapped)
                if missing_aggs:
                    raise ValueError(
                        f"Requested relative_to_aggs {missing_aggs} absent from group aggregations {list(mapped)} "
                        f"for pair ({cat}, {num}); include them in `aggregations=`."
                    )
                for agg in self.relative_to_aggs:
                    ref = mapped[agg]
                    if "diff" in self.relative_ops:
                        out[:, col_i] = x - ref
                        col_i += 1
                    if "ratio" in self.relative_ops:
                        out[:, col_i] = x / (ref + self.eps)
                        col_i += 1

        if col_i != m:
            raise RuntimeError(f"Wrote {col_i} columns but expected {m}: output columns and writing order differ.")

        result = pd.DataFrame(out, columns=self.output_columns_, index=X.index)
        return result if self.return_dataframe else result.values


# ---------------------------------------------------------------------------------------------------------------------
# Out-of-fold target encoding with the reference runs' category order
# ---------------------------------------------------------------------------------------------------------------------


def _sorted_categories(X: pd.DataFrame) -> pd.DataFrame:
    """Object and category columns as categoricals whose categories are their sorted values.

    The encoder numbers categories by these codes, so this reproduces ``pd.factorize(values, sort=True)`` on
    the object values, which is what the reference runs did.
    """
    columns = X.select_dtypes(include=["object", "category"]).columns
    if len(columns) == 0:
        return X
    X = X.copy()
    for column in columns:
        X[column] = pd.Categorical(X[column].astype("object"))
    return X


class SortedOOFTargetEncodingFeatureGenerator(OOFTargetEncodingFeatureGenerator):
    """AutoGluon's out-of-fold target encoder with categories numbered in sorted order."""

    def _fit_transform(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        return super()._fit_transform(_sorted_categories(X), y, **kwargs)

    def _transform(self, X, **kwargs):
        return super()._transform(_sorted_categories(X), **kwargs)


class StringKeyRSFCGenerator(RandomSubsetFeatureCompressionGenerator):
    """Random-subset feature compression keyed by the string row hash of each subset, as the reference runs.

    The keys are the same row hashes as AutoGluon's; as strings with sorted categories they give the
    encoder the category order those runs had (the public generator orders the integer hashes by first
    appearance).
    """

    def _make_keys(self, X: pd.DataFrame, subsets) -> pd.DataFrame:
        if not subsets:
            return pd.DataFrame(index=X.index)
        keys = {
            f"{self._KEY_PREFIX}{i}": pd.Categorical(self._make_key(X[list(subset)]))
            for i, subset in enumerate(subsets)
        }
        return pd.DataFrame(keys, index=X.index)


# ---------------------------------------------------------------------------------------------------------------------
# The TabPrep wrapper (atschalz/tabprep @ fee7f189)
# ---------------------------------------------------------------------------------------------------------------------

_GENERATORS = {
    cls.__name__: cls
    for cls in (
        ArithmeticFeatureGenerator,
        BulkFeatureGenerator,
        CategoricalInteractionFeatureGenerator,
        GroupByFeatureGenerator,
        OOFTargetEncodingFeatureGenerator,
        RandomSubsetFeatureCompressionGenerator,
    )
}


class TabPrepFeatureGenerator(BaseEstimator, TransformerMixin):
    """The five TabPrep generators in one leakage-safe transformer (fit on training rows only)."""

    def __init__(
        self,
        *,
        target_type: str = "binary",
        random_state: int | None = 0,
        use_groupby: bool = True,
        groupby_kwargs: dict[str, Any] | None = None,
        use_rsfc: bool = True,
        rsfc_kwargs: dict[str, Any] | None = None,
        use_arithmetic: bool = True,
        arithmetic_kwargs: dict[str, Any] | None = None,
        use_cat_interact: bool = True,
        cat_interact_kwargs: dict[str, Any] | None = None,
        use_oofte: bool = True,
        oofte_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.target_type = target_type
        self.random_state = random_state
        self.use_groupby = use_groupby
        self.groupby_kwargs = dict(groupby_kwargs or {})
        self.use_rsfc = use_rsfc
        self.rsfc_kwargs = dict(rsfc_kwargs or {})
        self.use_arithmetic = use_arithmetic
        self.arithmetic_kwargs = dict(arithmetic_kwargs or {})
        self.use_cat_interact = use_cat_interact
        self.cat_interact_kwargs = dict(cat_interact_kwargs or {})
        self.use_oofte = use_oofte
        self.oofte_kwargs = dict(oofte_kwargs or {})
        self.preprocessor_: AbstractFeatureGenerator | None = None
        self.is_fitted_ = False

    def _config(self) -> dict[str, Any]:
        stage: list[Any] = []
        passthrough_types = None
        if self.use_groupby:
            stage.append(["GroupByFeatureGenerator", self.groupby_kwargs])
        if self.use_rsfc:
            stage.append(["RandomSubsetFeatureCompressionGenerator", self.rsfc_kwargs])
        if self.use_arithmetic:
            kwargs = dict(self.arithmetic_kwargs)
            kwargs.setdefault("passthrough", True)
            stage.append(["ArithmeticFeatureGenerator", kwargs])
        categorical: list[Any] = []
        if self.use_cat_interact:
            kwargs = dict(self.cat_interact_kwargs)
            kwargs.setdefault("passthrough", True)
            categorical.append(["CategoricalInteractionFeatureGenerator", kwargs])
        if self.use_oofte:
            categorical.append(["OOFTargetEncodingFeatureGenerator", self.oofte_kwargs])
            passthrough_types = {"invalid_raw_types": ["category", "object"]}
        if categorical:
            stage.append(categorical)
        result: dict[str, Any] = {"ag.prep_params": [stage] if stage else []}
        if passthrough_types is not None:
            result["ag.prep_params.passthrough_types"] = passthrough_types
        return result

    def _init_generator(
        self, generator: type[AbstractFeatureGenerator] | str, kwargs: dict | None
    ) -> AbstractFeatureGenerator:
        if generator == "GroupByFeatureGenerator":
            generator = FastGroupByFeatureGenerator
        elif generator == "RandomSubsetFeatureCompressionGenerator":
            generator = StringKeyRSFCGenerator
        elif generator == "OOFTargetEncodingFeatureGenerator":
            generator = SortedOOFTargetEncodingFeatureGenerator
        elif isinstance(generator, str):
            try:
                generator = _GENERATORS[generator]
            except KeyError as error:
                raise ValueError(f"Unknown TabPrep generator: {generator}") from error
        params = dict(kwargs or {})
        params.setdefault("verbosity", 0)
        params.setdefault("target_type", self.target_type)
        if "random_state" not in params and self.random_state is not None:
            params["random_state"] = self.random_state
        return generator(**params)

    def _recursive_init(self, value):
        if isinstance(value, list):
            if len(value) == 2 and isinstance(value[0], str | AbstractFeatureGenerator):
                return self._init_generator(value[0], value[1])
            return [self._recursive_init(item) for item in value]
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError(f"Invalid TabPrep generator tuple: {value!r}")
            return self._init_generator(value[0], value[1])
        raise ValueError(f"Invalid TabPrep generator specification: {value!r}")

    def get_preprocessor(self) -> AbstractFeatureGenerator | None:
        config = self._config()
        prep_params = config["ag.prep_params"]
        if not prep_params:
            return None
        preprocessors = self._recursive_init(prep_params)
        if len(preprocessors) == 1 and isinstance(preprocessors[0], AbstractFeatureGenerator):
            return preprocessors[0]
        return BulkFeatureGenerator(
            generators=preprocessors,
            remove_unused_features="false_recursive",
            post_drop_duplicates=True,
            passthrough=True,
            passthrough_types=config.get("ag.prep_params.passthrough_types"),
            verbosity=0,
        )

    def _reorder(self, frame: pd.DataFrame) -> pd.DataFrame:
        numeric = [x for x in self.numeric_features_in_ if x in frame.columns]
        remaining = [x for x in frame.columns if x not in numeric]
        return frame.loc[:, [*numeric, *remaining]]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, **fit_params):
        del fit_params
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a DataFrame, got {type(X)}")
        if y is None:
            raise AssertionError("y is required for TabPrep")
        frame = X.copy()
        target = pd.Series(y).copy()
        self.feature_names_in_ = list(frame.columns)
        self.n_features_in_ = frame.shape[1]
        self.numeric_features_in_ = list(frame.select_dtypes(include="number").columns)
        self.preprocessor_ = self.get_preprocessor()
        output = frame.copy() if self.preprocessor_ is None else self.preprocessor_.fit_transform(frame, target)
        self.is_fitted_ = True
        return self._reorder(output)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.fit_transform(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise AttributeError("TabPrepFeatureGenerator is not fitted")
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a DataFrame, got {type(X)}")
        output = X.copy() if self.preprocessor_ is None else self.preprocessor_.transform(X)
        return self._reorder(output)


# ---------------------------------------------------------------------------------------------------------------------
# The feature cap
# ---------------------------------------------------------------------------------------------------------------------


class SpearmanRanker:
    """Absolute Spearman correlation with a fixed target, with SciPy's complete / pairwise-missing semantics."""

    def __init__(self, target):
        self.target = np.asarray(target)
        if self.target.ndim != 1:
            raise ValueError("The training target must be one-dimensional")
        self.target_finite = np.isfinite(self.target)
        self.complete_target = bool(self.target_finite.all())
        self.target_has_nan = bool(np.isnan(self.target).any())
        self.target_ranks = stats.rankdata(self.target) if self.complete_target else None
        self.constant_target = len(self.target) < 2 or bool(np.all(self.target == self.target[0]))
        self._masked_ranks: OrderedDict = OrderedDict()
        self._cache_bytes = 0
        self._cache_limit = 64 * 1024**2
        self._cache_lock = Lock()

    def _ranks_for_mask(self, mask):
        key = np.packbits(mask).tobytes()
        with self._cache_lock:
            ranks = self._masked_ranks.get(key)
            if ranks is not None:
                self._masked_ranks.move_to_end(key)
                return ranks
        ranks = stats.rankdata(self.target[mask])
        size = ranks.nbytes + len(key)
        with self._cache_lock:
            if key not in self._masked_ranks and size <= self._cache_limit:
                while self._masked_ranks and self._cache_bytes + size > self._cache_limit:
                    old_key, old_ranks = self._masked_ranks.popitem(last=False)
                    self._cache_bytes -= len(old_key) + old_ranks.nbytes
                self._masked_ranks[key] = ranks
                self._cache_bytes += size
        return ranks

    def strength(self, numeric) -> float:
        numeric = np.asarray(numeric)
        if numeric.ndim != 1 or numeric.shape != self.target.shape:
            raise ValueError("Feature and target must have equal one-dimensional shapes")
        if len(numeric) < 2 or self.constant_target or np.all(numeric == numeric[0]):
            return -math.inf
        if self.complete_target and np.isfinite(numeric).all():
            # scipy.spearmanr's rankdata / corrcoef, without re-ranking the fixed target.
            corr = np.corrcoef(stats.rankdata(numeric), self.target_ranks)[0, 1]
        elif self.target_has_nan or np.isnan(numeric).any():
            # scipy.spearmanr(nan_policy='omit') with a NaN: drop rows with a nonfinite member, rank both columns,
            # masked corrcoef -- reproduced without the repeated masked-array construction.
            mask = np.isfinite(numeric) & self.target_finite
            if np.count_nonzero(mask) < 2:
                return -math.inf
            ranks = np.column_stack([stats.rankdata(numeric[mask]), self._ranks_for_mask(mask)])
            corr = np.ma.corrcoef(ranks, rowvar=False).data[0, 1]
        else:
            # Infinities without NaNs follow SciPy's ordinary ranking path.
            corr = stats.spearmanr(numeric, self.target, nan_policy="omit").statistic
        return -math.inf if not np.isfinite(corr) else abs(float(corr))


def select_columns(frame: pd.DataFrame, target, *, cap: int = FEATURE_CAP, num_cpus: int = 2) -> list[str]:
    """The ``cap`` columns most correlated with ``target`` (ties to the earlier column), in their original order."""
    columns = list(frame.columns)
    if len(columns) <= cap:
        return columns
    ranker = SpearmanRanker(target)

    def score(item):
        index, column = item
        values = frame[column]
        if isinstance(values.dtype, pd.CategoricalDtype):
            numeric = values.cat.codes.to_numpy(dtype=np.float64)
            numeric[numeric < 0] = np.nan
        else:
            numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
        return ranker.strength(numeric), -index, column

    workers = max(1, int(num_cpus))
    # The ranking workers are the parallelism; keep BLAS single-threaded inside them.
    with threadpool_limits(limits=1, user_api="blas"):
        if workers == 1:
            correlations = list(map(score, enumerate(columns)))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                correlations = list(executor.map(score, enumerate(columns)))
    correlations.sort(reverse=True)
    keep = {column for _, _, column in correlations[:cap]}
    return [column for column in columns if column in keep]


def _take(frame: pd.DataFrame, selected: list[str]) -> pd.DataFrame:
    """Apply the cap; a 0-row frame may lack generated columns (some generators emit none for it)."""
    if len(frame) == 0 and not set(selected) <= set(frame.columns):
        return frame.reindex(columns=selected)
    return frame.loc[:, selected]


def tabprep_frames(
    frames: dict[str, pd.DataFrame], y_train, problem_type: str, tabprep: dict, *, num_cpus: int = 2
) -> tuple[dict[str, pd.DataFrame], dict, TabPrepFeatureGenerator, list[str]]:
    """Fit TabPrep on ``frames['train']`` / ``y_train`` and transform every frame; apply the feature cap.

    ``tabprep`` holds the config's generator budgets and seeds. Returns the transformed frames, a report,
    the fitted transformer and the kept columns.
    """
    prep = TabPrepFeatureGenerator(
        target_type={"binclass": "binary"}.get(problem_type, problem_type),
        random_state=42,
        groupby_kwargs={"max_features": int(tabprep["groupby_max_feats"])},
        rsfc_kwargs={"n_subsets": int(tabprep["rsfc_n_subsets"]), "random_state": int(tabprep["rsfc_generation_seed"])},
        arithmetic_kwargs={
            "max_new_feats": int(tabprep["oafe_max_feats"]),
            "random_state": int(tabprep["oafe_generation_seed"]),
        },
        cat_interact_kwargs={
            "max_new_feats": int(tabprep["cat_combine_max_feats"]),
            "random_state": int(tabprep["cat_combine_generation_seed"]),
        },
        oofte_kwargs={"random_state": int(tabprep["oofte_generation_seed"])},
    )
    started = time.perf_counter()
    output = {"train": prep.fit_transform(frames["train"], pd.Series(y_train))}
    timings = {"fit_transform_seconds": time.perf_counter() - started}
    for part, frame in frames.items():
        if part != "train":
            started = time.perf_counter()
            output[part] = prep.transform(frame)
            timings[part + "_transform_seconds"] = time.perf_counter() - started
    before_cap = output["train"].shape[1]
    started = time.perf_counter()
    selected = select_columns(output["train"], y_train, cap=FEATURE_CAP, num_cpus=num_cpus)
    timings["feature_cap_seconds"] = time.perf_counter() - started
    if before_cap > FEATURE_CAP:
        output = {part: _take(frame, selected) for part, frame in output.items()}
    prep.timings_ = timings
    report = {
        "features_before_tabprep": frames["train"].shape[1],
        "features_after_tabprep_before_cap": before_cap,
        "features_after_cap": len(selected),
        "feature_cap_applied": before_cap > FEATURE_CAP,
    }
    return output, report, prep, selected
