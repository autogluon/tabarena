from __future__ import annotations

from collections import defaultdict
from typing import List
import logging
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EloHelper:
    def __init__(
        self,
        method_col: str = "method",
        task_col: str = "task",
        error_col: str = "metric_error",
        split_col: str | None = None,
    ):
        self.method_col = method_col
        self.task_col = task_col
        self.error_col = error_col
        self.split_col = split_col

        self.method_1 = f"{self.method_col}_1"
        self.method_2 = f"{self.method_col}_2"

    def compute_mle_elo(
        self,
        battles: pd.DataFrame,
        SCALE: int | float = 400,
        BASE: int = 10,
        INIT_RATING: int | float = 1000,
        calibration_framework: str = None,
        calibration_elo: float = None,
        use_pair_aggregation: bool = True,
        force_iterative_elo: bool = False,
        max_iter: int = 1000,
    ) -> pd.Series:
        """
        MLE ELO with per-task equal weighting via sample_weight.

        Adapted from ChatBot Arena: https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH#scrollTo=4_x-vXL4yxvC
        """
        models = pd.concat([battles[self.method_1], battles[self.method_2]]).unique()
        models = pd.Series(np.arange(len(models)), index=models)
        df_original = battles

        if use_pair_aggregation and not force_iterative_elo:
            X, Y, sample_weight, model_to_idx = self._aggregate_battles_for_mle(battles, BASE=BASE)
            if len(Y) == 0 or (np.unique(Y).size < 2):
                # Degenerate draw (e.g., total dominance in a bootstrap): fall back.
                logger.warning(
                    "compute_mle_elo: only one class present after aggregation; falling back to iterative ELO.")
                elo_scores = self.compute_iterative_elo_scores(
                    df_original, INIT_RATING=INIT_RATING, SCALE=SCALE, models=models
                )
                SeriesOut = pd.Series(elo_scores, index=models.index)
            else:
                lr = LogisticRegression(fit_intercept=False, max_iter=max_iter, C=1e6)
                lr.fit(X, Y, sample_weight=sample_weight)
                coef = lr.coef_[0]
                # map coef -> ELO
                elo_vec = SCALE * coef + INIT_RATING
                # place into full model order
                out = np.ones(len(models)) * INIT_RATING
                out[model_to_idx.values] = elo_vec[model_to_idx.values]
                SeriesOut = pd.Series(out, index=models.index)
        else:
            # duplicate battles
            battles = pd.concat([battles, battles], ignore_index=True)
            p = len(models.index)
            n = battles.shape[0]

            X = np.zeros([n, p])
            X[np.arange(n), models[battles[self.method_1]]] = +math.log(BASE)
            X[np.arange(n), models[battles[self.method_2]]] = -math.log(BASE)

            Y = np.zeros(n)
            Y[battles["winner"] == "1"] = 1.0

            # tie handling: make half wins for each side
            tie_idx = battles["winner"] == "tie"
            tie_idx[len(tie_idx) // 2:] = False
            Y[tie_idx] = 1.0

            # Build sample weights so each task contributes total weight 1 across its splits
            if "weight" in battles.columns:
                sample_weight = battles["weight"].to_numpy(dtype=float).copy()
            else:
                sample_weight = np.ones(n, dtype=float)

            if len(np.unique(Y)) == 1 or force_iterative_elo:
                logger.warning(
                    "compute_mle_elo fell back to iterative ELO (dominance in labels or forced)."
                )
                elo_scores = self.compute_iterative_elo_scores(
                    df_original,
                    INIT_RATING=INIT_RATING,
                    SCALE=SCALE,
                    models=models,
                )
            else:
                lr = LogisticRegression(fit_intercept=False, max_iter=max_iter, C=1e6)
                lr.fit(X, Y, sample_weight=sample_weight)
                elo_scores = SCALE * lr.coef_[0] + INIT_RATING

            SeriesOut = pd.Series(elo_scores, index=models.index)

        if calibration_framework is not None:
            if calibration_elo is None:
                calibration_elo = INIT_RATING
            SeriesOut += (calibration_elo - SeriesOut[calibration_framework])

        return SeriesOut.sort_values(ascending=False)

    def compute_iterative_elo_scores(
        self,
        battles: pd.DataFrame,
        INIT_RATING: int = 1000,
        SCALE: int = 400,
        K_factor: float = 1,
        models: pd.Series = None,
        *,
        epochs: int = 10,
        shuffle: bool = True,
        seed: int | np.random.Generator | None = 0,
    ):
        """
        Iterative ELO with optional deterministic shuffling and multi-epoch passes.

        Parameters
        ----------
        battles : pd.DataFrame
            Battles dataframe (from convert_results_to_battles). If present,
            uses df['weight'] to scale each update so tasks contribute equally.
        INIT_RATING : int
            Initial rating for all models.
        SCALE : int
            ELO scale (e.g., 400).
        K_factor : float
            Step size per update (will be multiplied by per-row 'weight' if present).
        models : pd.Series
            Mapping: index=model name, value=integer id. Built if None.
        epochs : int, default=10
            Number of full passes over the (possibly shuffled) battles.
        shuffle : bool, default=True
            Shuffle battle order each epoch.
        seed : int | np.random.Generator | None, default=0
            Controls the shuffle order. If an int or Generator is provided and
            shuffle=True, the traversal order is deterministic across runs.

        Returns
        -------
        np.ndarray
            ELO scores aligned to `models.index` order.
        """
        if models is None:
            model_names = pd.concat([battles[self.method_1], battles[self.method_2]]).unique()
            models = pd.Series(np.arange(len(model_names)), index=model_names)

        # Build numeric arrays for fast iteration
        # Map methods -> indices
        m1_idx = battles[self.method_1].map(models).to_numpy()
        m2_idx = battles[self.method_2].map(models).to_numpy()

        # Winner encoding: 1 -> m1 wins, 2 -> m2 wins, tie -> 0
        winner_raw = battles["winner"].to_numpy()
        # encode: 1, 2, tie
        w_code = np.zeros(len(winner_raw), dtype=np.int8)
        w_code[winner_raw == "1"] = 1
        w_code[winner_raw == "2"] = 2
        # weight per row (defaults to 1)
        if "weight" in battles.columns:
            w_row = battles["weight"].to_numpy(dtype=float, copy=False)
        else:
            w_row = np.ones(len(battles), dtype=float)

        def expected_result(r1, r2):
            return 1.0 / (1.0 + 10.0 ** ((r2 - r1) / SCALE))

        elo_scores = np.full(len(models.index), float(INIT_RATING))

        # RNG (only used if shuffle=True)
        if isinstance(seed, np.random.Generator):
            rng = seed
        else:
            rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        n = len(battles)
        base_order = np.arange(n)

        for _ in range(max(1, int(epochs))):
            if shuffle:
                order = rng.permutation(n)
            else:
                order = base_order

            for i in order:
                a = m1_idx[i]
                b = m2_idx[i]
                w = float(w_row[i])
                outcome = w_code[i]

                ra = elo_scores[a]
                rb = elo_scores[b]
                ea = expected_result(ra, rb)
                eb = 1.0 - ea  # symmetry

                if outcome == 0:  # tie
                    elo_scores[a] += (K_factor * w) * (0.5 - ea)
                    elo_scores[b] += (K_factor * w) * (0.5 - eb)
                elif outcome == 1:  # m1 wins
                    elo_scores[a] += (K_factor * w) * (1.0 - ea)
                    elo_scores[b] += (K_factor * w) * (0.0 - eb)
                else:  # outcome == 2, m2 wins
                    elo_scores[b] += (K_factor * w) * (1.0 - eb)
                    elo_scores[a] += (K_factor * w) * (0.0 - ea)

        return elo_scores

    def get_bootstrap_result(
        self,
        battles: pd.DataFrame,
        func_compute_elo,  # kept for API symmetry when not using compression
        rng=None,
        num_round: int | None = None,
        func_kwargs=None,
        *,
        BASE: int = 10,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        show_process: bool = True,
    ):
        """
        Task-level bootstrap with pair-compressed MLE.
        """
        if func_kwargs is None:
            func_kwargs = {}

        # RNG
        if isinstance(rng, np.random.Generator):
            _rng = rng
        elif rng is None:
            _rng = np.random.default_rng()
        else:
            _rng = np.random.default_rng(int(rng))

        # Non-bootstrap or legacy path
        if num_round is None:
            res = func_compute_elo(battles, **func_kwargs)
            df = pd.DataFrame([res])
            return df[df.median().sort_values(ascending=False).index]

        # One-time precompute
        model_to_idx, pairs, X2, Y2, SU, SV, task_ids = self._precompute_pair_agg_for_bootstrap(
            battles=battles, BASE=BASE
        )
        n_tasks = SU.shape[0]
        if SU.shape[1] == 0:
            raise ValueError("No (method_1, method_2) pairs present; cannot bootstrap.")

        rows = []
        for _ in tqdm(range(num_round), desc="bootstrap", disable=not show_process):
            # Sample task indices with replacement; convert to multiplicity vector
            sampled = _rng.choice(np.arange(n_tasks), size=n_tasks, replace=True)
            counts = np.bincount(sampled, minlength=n_tasks).astype(float)

            # Delegate per-draw work to the helper
            ser = self._bootstrap_draw_compressed(
                counts=counts,
                task_ids=task_ids,
                SU=SU,
                SV=SV,
                X2=X2,
                Y2=Y2,
                model_to_idx=model_to_idx,
                battles=battles,
                solver=solver,
                max_iter=max_iter,
                **func_kwargs,
            )
            rows.append(ser)

        df = pd.DataFrame(rows)
        return df[df.median().sort_values(ascending=False).index]

    def calc_battle_outcome(self, error_1: float, error_2: float) -> str:
        if error_1 < error_2:
            winner = "1"
        elif error_1 > error_2:
            winner = "2"
        else:
            winner = "tie"
        return winner

    def convert_results_to_battles(
        self,
        results_df: pd.DataFrame,
        frameworks: List[str] = None,
        datasets: List[str] = None,
    ) -> pd.DataFrame:
        # Keep only needed columns (+ split if available)
        cols = [self.method_col, self.task_col, self.error_col]
        if self.split_col is not None:
            cols.append(self.split_col)
        results_df = results_df[cols].copy()

        if datasets is not None:
            results_df = results_df[results_df[self.task_col].isin(datasets)]
        if frameworks is not None:
            results_df = results_df[results_df[self.method_col].isin(frameworks)]

        # Determine how to pair: by (task, split) if split_col is given; else by task only
        if self.split_col is not None:
            on_cols = [self.task_col, self.split_col]
        else:
            on_cols = [self.task_col]

        results_df_after_dedupe = results_df.drop_duplicates(subset=[self.method_col, *on_cols])
        if len(results_df_after_dedupe) != len(results_df):
            raise AssertionError(f"Found {len(results_df) - len(results_df_after_dedupe)} duplicate rows!")

        # Pair each method with every other method on the same task (and split if provided)
        results_pairs_df = pd.merge(
            results_df,
            results_df,
            on=on_cols,
            suffixes=('_1', '_2'),
            how='inner',
        )

        # Keep only pairs with different methods
        mask_diff_method = results_pairs_df[self.method_1] != results_pairs_df[self.method_2]
        results_pairs_df = results_pairs_df.loc[mask_diff_method].copy()

        # Winner of the pair
        results_pairs_df["winner"] = [
            self.calc_battle_outcome(e1, e2)
            for e1, e2 in zip(results_pairs_df[f"{self.error_col}_1"], results_pairs_df[f"{self.error_col}_2"])
        ]

        # Avoid counting each battle twice (dedupe A vs B with B vs A) by method pair (orderless) within the same on_cols
        # Build a canonical key for the unordered pair
        pair_key = list(zip(
            np.minimum(results_pairs_df[self.method_1], results_pairs_df[self.method_2]),
            np.maximum(results_pairs_df[self.method_1], results_pairs_df[self.method_2]),
        ))

        # Create a boolean mask that keeps the first occurrence of each (task[, split], unordered pair)
        # Easiest way: use a DataFrame of keys and drop_duplicates, then reindex
        key_df = pd.DataFrame({**{c: results_pairs_df[c].values for c in on_cols},
                               "__a": [k[0] for k in pair_key],
                               "__b": [k[1] for k in pair_key]})
        keep_idx = key_df.drop_duplicates(subset=on_cols + ["__a", "__b"]).index
        results_pairs_df = results_pairs_df.iloc[keep_idx].copy()

        # Compute per-task weights so each task contributes total weight 1.
        # If split_col exists: n_splits = nunique splits per task, else n_splits := 1
        if self.split_col is not None:
            splits_per_task = results_df.groupby(self.task_col)[self.split_col].nunique()
        else:
            splits_per_task = results_df.groupby(self.task_col).size()
            splits_per_task[:] = 1  # treat as 1 split per task

        # Map weight = 1 / n_splits(task)
        results_pairs_df["weight"] = 1.0 / results_pairs_df[self.task_col].map(splits_per_task).astype(float)

        # Final output columns (keep weight; keep split if present for debugging/analysis)
        out_cols = [self.method_1, self.method_2, "winner", self.task_col, "weight"]
        if self.split_col is not None and self.split_col in results_pairs_df.columns:
            out_cols.append(self.split_col)
        return results_pairs_df[out_cols]

    def compute_elo_ratings(
        self,
        battles: pd.DataFrame,
        seed: int = 0,
        calibration_framework=None,
        calibration_elo=None,
        INIT_RATING: float = 1000,
        BOOTSTRAP_ROUNDS: int = 100,
        SCALE: int = 400,
        show_process: bool = True,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed=seed)
        bootstrap_elo_lu = self.get_bootstrap_result(
            battles=battles,
            func_compute_elo=self.compute_mle_elo,
            num_round=BOOTSTRAP_ROUNDS,
            rng=rng,
            func_kwargs={
                "INIT_RATING": INIT_RATING,
                "SCALE": SCALE,
                "calibration_framework": calibration_framework,
                "calibration_elo": calibration_elo,
            },
            show_process=show_process,
        )
        return bootstrap_elo_lu

    def compute_elo_rating_dataset_contributon(
        self,
        results_ranked_fillna_df: pd.DataFrame,
        seed: int = 0,
        calibration_framework=None,
        calibration_elo=None,
        INIT_RATING: float = 1000,
        SCALE: int = 400,
    ) -> pd.DataFrame:
        datasets = list(results_ranked_fillna_df[self.task_col].unique())
        battles = self.convert_results_to_battles(results_ranked_fillna_df, datasets=datasets)

        rng = np.random.default_rng(seed=seed)
        bootstrap_elo_lu = self.get_bootstrap_result(
            battles=battles,
            func_compute_elo=self.compute_mle_elo,
            num_round=None,
            rng=rng,
            func_kwargs={
                "INIT_RATING": INIT_RATING,
                "SCALE": SCALE,
                "calibration_framework": calibration_framework,
                "calibration_elo": calibration_elo,
            }
        )

        bars = pd.DataFrame(dict(
            rating=bootstrap_elo_lu.quantile(.5),
        )).sort_values("rating", ascending=False)

        elo_impact_by_dataset_list = []
        for dataset_to_skip in datasets:
            battles_w_dataset_removed = battles[battles[self.task_col] != dataset_to_skip]
            bootstrap_elo_lu_w_dataset_removed = self.get_bootstrap_result(
                battles=battles_w_dataset_removed,
                func_compute_elo=self.compute_mle_elo,
                num_round=None,
                rng=rng,
                func_kwargs={
                    "INIT_RATING": INIT_RATING,
                    "SCALE": SCALE,
                    "calibration_framework": calibration_framework,
                    "calibration_elo": calibration_elo,
                }
            )
            bars_by_dataset = pd.DataFrame(dict(
                rating=bootstrap_elo_lu_w_dataset_removed.quantile(.5),
            ))

            delta = bars["rating"] - bars_by_dataset["rating"]
            delta.name = dataset_to_skip
            elo_impact_by_dataset_list.append(delta)
        elo_impact_by_dataset = pd.concat(elo_impact_by_dataset_list, axis=1)
        return elo_impact_by_dataset

    def get_rank_confidence(self, df: pd.DataFrame):
        df = df.copy()
        df = df.sort_values(by=["Arena Elo"], ascending=False)

        elo_ratings = df["Arena Elo"].to_list()
        uppers = df["upper"].to_list()
        lowers = df["lower"].to_list()

        ranks = []

        cur_rank = 0
        prev_lower = None
        num_models = len(elo_ratings)
        for i in range(num_models):
            cur_elo = elo_ratings[i]
            cur_upper = uppers[i]
            cur_lower = lowers[i]
            if prev_lower is None or cur_upper < prev_lower:
                cur_rank = i + 1
                prev_lower = cur_lower
            ranks.append(cur_rank)

        df["Rank"] = ranks

        return df

    def predict_win_rate(self, elo_ratings: dict, SCALE=400, BASE=10):
        names = sorted(list(elo_ratings.keys()))
        wins = defaultdict(lambda: defaultdict(lambda: 0))
        for a in names:
            for b in names:
                ea = 1 / (1 + BASE ** ((elo_ratings[b] - elo_ratings[a]) / SCALE))
                wins[a][b] = ea
                wins[b][a] = 1 - ea

        data = {
            a: [wins[a][b] if a != b else np.NAN for b in names]
            for a in names
        }

        df = pd.DataFrame(data, index=names)
        df.index.name = "model_a"
        df.columns.name = "model_b"
        return df.T

    def compute_winrate_from_battles(self, battles) -> pd.Series:
        framework_battle_counts = defaultdict(int)
        framework_win_counts = defaultdict(float)
        for f, win_val in [(self.method_1, "1"), (self.method_2, "2")]:
            counts = battles[f].value_counts().to_dict()
            win_counts = battles[[f, "winner"]].value_counts().reset_index()
            win_counts["count"] = win_counts["count"].astype(float)
            win_counts.loc[win_counts["winner"] == "tie", "count"] *= 0.5
            win_counts = win_counts.loc[win_counts["winner"].isin([win_val, "tie"]), :]
            win_counts = win_counts.drop(columns=["winner"]).groupby(f)["count"].sum().to_dict()
            for framework in counts:
                framework_battle_counts[framework] += counts[framework]
            for framework in win_counts:
                framework_win_counts[framework] += win_counts[framework]
        framework_battle_counts = dict(framework_battle_counts)

        results_wins = pd.Series(framework_win_counts, name="wins")
        results_battles = pd.Series(framework_battle_counts, name="battles")
        results_winrate = results_wins / results_battles
        results_winrate.name = "winrate"
        results_winrate.index.name = self.method_col
        return results_winrate

    def _fix_missing(self, df, missing_A, missing_B):
        df = df.copy()
        for b in missing_B:
            if b not in df:
                df[b] = 0
        df = df.T
        for a in missing_A:
            if a not in df:
                df[a] = 0
        df = df.T
        return df

    def compute_pairwise_win_fraction(self, battles, max_num_models=30) -> pd.DataFrame:
        unique_A = list(battles[self.method_1].unique())
        unique_B = list(battles[self.method_2].unique())
        missing_A = [b for b in unique_B if b not in unique_A]
        missing_B = [a for a in unique_A if a not in unique_B]
        unique_all = unique_A + missing_A
        # Times each model wins as Model A
        a_win_ptbl = pd.pivot_table(
            battles[battles['winner'] == "1"],
            index=self.method_1, columns=self.method_2, aggfunc="size", fill_value=0)

        # Table counting times each model wins as Model B
        b_win_ptbl = pd.pivot_table(
            battles[battles['winner'] == "2"],
            index=self.method_1, columns=self.method_2, aggfunc="size", fill_value=0)

        # Table counting times each model wins as Model B
        tie_ptbl = pd.pivot_table(
            battles[battles['winner'] == "tie"],
            index=self.method_1, columns=self.method_2, aggfunc="size", fill_value=0)
        tie_ptbl *= 0.5

        # Table counting number of A-B pairs
        num_battles_ptbl = pd.pivot_table(battles,
            index=self.method_1, columns=self.method_2, aggfunc="size", fill_value=0)

        missing_A = unique_all
        missing_B = unique_all
        a_win_ptbl = self._fix_missing(df=a_win_ptbl, missing_A=missing_A, missing_B=missing_B)
        b_win_ptbl = self._fix_missing(df=b_win_ptbl, missing_A=missing_A, missing_B=missing_B)
        tie_missing_A = [a for a in unique_all if a not in tie_ptbl.index]
        tie_missing_B = [b for b in unique_all if b not in tie_ptbl.columns]
        tie_ptbl = self._fix_missing(df=tie_ptbl, missing_A=tie_missing_A, missing_B=tie_missing_B)
        num_battles_ptbl = self._fix_missing(df=num_battles_ptbl, missing_A=missing_A, missing_B=missing_B)

        # Computing the proportion of wins for each model as A and as B
        # against all other models
        row_beats_col_freq = (
            (a_win_ptbl + b_win_ptbl.T + tie_ptbl + tie_ptbl.T) /
            (num_battles_ptbl + num_battles_ptbl.T)
        )

        # Arrange ordering according to proprition of wins
        prop_wins = row_beats_col_freq.mean(axis=1).sort_values(ascending=False)
        prop_wins = prop_wins[:max_num_models]
        model_names = list(prop_wins.keys())
        row_beats_col = row_beats_col_freq.loc[model_names, model_names]
        return row_beats_col

    def _aggregate_battles_for_mle(self, battles: pd.DataFrame, BASE: int = 10):
        """
        Compress battles to two rows per unordered (A,B) pair for exact MLE equivalence.

        For each unordered pair (A,B):
          - Build features X for "A vs B" once: +log(BASE) at A, -log(BASE) at B.
          - Create two rows with the *same X*:
              y=1, weight = w_Awins + 0.5 * w_ties
              y=0, weight = w_Bwins + 0.5 * w_ties
        """
        # model index mapping
        all_models = pd.concat([battles[self.method_1], battles[self.method_2]]).unique()
        model_to_idx = pd.Series(np.arange(len(all_models)), index=all_models)

        a = battles[self.method_1].to_numpy()
        b = battles[self.method_2].to_numpy()
        w = battles["weight"].to_numpy(dtype=float) if "weight" in battles.columns else np.ones(len(battles), float)
        win = battles["winner"].to_numpy()  # '1','2','tie'

        # canonical unordered keys u<=v
        u = np.where(a < b, a, b)
        v = np.where(a < b, b, a)

        # accumulate weighted wins/ties in canonical coordinates
        wins_u = {}  # A-side under (u,v)
        wins_v = {}
        for ui, vi, ai, bi, wi, yi in zip(u, v, a, b, w, win):
            key = (ui, vi)
            su = wins_u.get(key, 0.0)
            sv = wins_v.get(key, 0.0)
            if yi == '1':  # method_1 (ai) beat method_2 (bi)
                if ui == ai:
                    su += wi
                else:
                    sv += wi
            elif yi == '2':  # method_2 (bi) beat method_1 (ai)
                if ui == bi:
                    su += wi
                else:
                    sv += wi
            else:  # tie: half win each side
                su += 0.5 * wi
                sv += 0.5 * wi
            wins_u[key] = su
            wins_v[key] = sv

        # build compressed design: one X per pair; two rows with y in {0,1}
        pairs = list(wins_u.keys())
        p = len(all_models)
        X_rows, y_rows, sw_rows = [], [], []
        logB = math.log(BASE)

        for (ui, vi) in pairs:
            su = wins_u[(ui, vi)]
            sv = wins_v[(ui, vi)]
            if su == 0.0 and sv == 0.0:
                continue  # no information

            x = np.zeros(p, dtype=float)
            x[model_to_idx[ui]] = +logB
            x[model_to_idx[vi]] = -logB

            # y=1 row with A-side effective wins
            if su > 0.0:
                X_rows.append(x)
                y_rows.append(1.0)
                sw_rows.append(su)
            # y=0 row with B-side effective wins
            if sv > 0.0:
                X_rows.append(x)
                y_rows.append(0.0)
                sw_rows.append(sv)

        X = np.vstack(X_rows) if X_rows else np.zeros((0, p))
        y = np.asarray(y_rows, dtype=float)
        sample_weight = np.asarray(sw_rows, dtype=float)
        return X, y, sample_weight, model_to_idx

    def get_arena_leaderboard(self, bootstrap_elo_lu: pd.DataFrame, results_df: pd.DataFrame):
        bars = pd.DataFrame(dict(
            lower=bootstrap_elo_lu.quantile(.025),
            rating=bootstrap_elo_lu.quantile(.5),
            upper=bootstrap_elo_lu.quantile(.975))).reset_index(names="model").sort_values("rating", ascending=False)
        bars['error_y'] = bars['upper'] - bars["rating"]
        bars['error_y_minus'] = bars['rating'] - bars["lower"]
        bars['rating_rounded'] = np.round(bars['rating'], 2)
        battles = self.convert_results_to_battles(results_df=results_df)
        from collections import defaultdict
        framework_battle_counts = defaultdict(int)
        framework_win_counts = defaultdict(float)
        for f, win_val in [(self.method_1, "1"), (self.method_2, "2")]:
            counts = battles[f].value_counts().to_dict()
            win_counts = battles[[f, "winner"]].value_counts().reset_index()
            win_counts["count"] = win_counts["count"].astype(float)
            win_counts.loc[win_counts["winner"] == "tie", "count"] *= 0.5
            win_counts = win_counts.loc[win_counts["winner"].isin([win_val, "tie"]), :]
            win_counts = win_counts.drop(columns=["winner"]).groupby(f)["count"].sum().to_dict()
            for framework in counts:
                framework_battle_counts[framework] += counts[framework]
            for framework in win_counts:
                framework_win_counts[framework] += win_counts[framework]
        framework_battle_counts = dict(framework_battle_counts)

        def _get_95_ci(upper, lower):
            return f"+{upper:.0f}/-{lower:.0f}"

        leaderboard = bars.copy()
        leaderboard["95% CI"] = [_get_95_ci(upper, lower) for upper, lower in zip(leaderboard["error_y"], leaderboard["error_y_minus"])]
        leaderboard["Arena Elo"] = np.round(leaderboard['rating'], 0).astype(int)
        leaderboard["Battles"] = leaderboard["model"].map(framework_battle_counts)
        leaderboard["Wins"] = np.round(leaderboard["model"].map(framework_win_counts), decimals=0).astype(int)
        leaderboard["Winrate"] = np.round(leaderboard["Wins"] / leaderboard["Battles"], decimals=2)
        leaderboard["Rank (Simple)"] = leaderboard["Arena Elo"].rank(method="min", ascending=False).astype(int)
        leaderboard["Model"] = leaderboard["model"]
        leaderboard = self.get_rank_confidence(df=leaderboard)

        results_mean_agg = results_df[[self.method_col, "rank", "bestdiff", "loss_rescaled"]].groupby("framework").mean()
        leaderboard["mean_rank"] = leaderboard["model"].map(results_mean_agg["rank"])
        leaderboard["mean_bestdiff"] = leaderboard["model"].map(results_mean_agg["bestdiff"])
        leaderboard["mean_loss_rescaled"] = leaderboard["model"].map(results_mean_agg["loss_rescaled"])

        leaderboard["Rank Avg"] = np.round(leaderboard["mean_rank"], decimals=1)
        leaderboard["Champ Delta %"] = np.round(leaderboard["mean_bestdiff"] * 100, decimals=1)
        # leaderboard["Champ Delta % 2"] = np.round((1/(1 - leaderboard["mean_bestdiff"]) - 1) * 100, decimals=1)
        leaderboard["Rescaled Acc"] = np.round(1 - leaderboard["mean_loss_rescaled"], decimals=2)
        leaderboard["Elo"] = leaderboard["Arena Elo"]

        leaderboard_print = leaderboard[[
            "Rank",
            "Model",
            "Elo",
            "95% CI",
            # "Battles",
            # "Wins",
            "Winrate",
            "Rescaled Acc",
            # "Rank Avg",
            "Champ Delta %",
            # "Champ Delta % 2",
        ]]

        return leaderboard, leaderboard_print

    def _precompute_pair_agg_for_bootstrap(
        self,
        battles: pd.DataFrame,
        BASE: int = 10,
    ):
        """
        Precompute sufficient statistics for MLE ELO under task-cluster bootstrap.

        Returns
        -------
        model_to_idx : pd.Series
            Index mapping for models -> column index in design matrix.
        pairs : List[Tuple[str,str]]
            Canonical unordered pairs (u<=v), in a stable order.
        X2 : np.ndarray, shape (2 * n_pairs, n_models)
            Fixed design matrix. For each pair i:
              - Row 2*i   : features for (u beats v)  -> y=1
              - Row 2*i+1 : same features            -> y=0 (v beats u row)
        Y2 : np.ndarray, shape (2 * n_pairs,)
            Fixed targets (1,0,1,0,...) aligned with X2 rows.
        SU : np.ndarray, shape (n_tasks, n_pairs)
            Per-task effective wins for the 'u' side of each pair (wins + 0.5*ties).
        SV : np.ndarray, shape (n_tasks, n_pairs)
            Per-task effective wins for the 'v' side of each pair (wins + 0.5*ties).
        task_ids : np.ndarray, shape (n_tasks,)
            Array of task identifiers aligned with SU/SV rows.
        """
        # map models -> columns
        all_models = pd.concat([battles[self.method_1], battles[self.method_2]]).unique()
        model_to_idx = pd.Series(np.arange(len(all_models)), index=all_models)

        # pull arrays
        a = battles[self.method_1].to_numpy()
        b = battles[self.method_2].to_numpy()
        win = battles["winner"].to_numpy()  # '1','2','tie'
        task = battles[self.task_col].to_numpy()
        w = battles["weight"].to_numpy(dtype=float) if "weight" in battles.columns else np.ones(len(battles), float)

        # canonicalize unordered pair keys and compute (u,v), also keep which side was m1/m2
        u = np.where(a < b, a, b)
        v = np.where(a < b, b, a)

        # enumerate tasks
        uniq_tasks, task_inverse = np.unique(task, return_inverse=True)
        n_tasks = uniq_tasks.shape[0]

        # index unordered pairs
        pair_keys = np.core.defchararray.add(u.astype(str), "||")
        pair_keys = np.core.defchararray.add(pair_keys, v.astype(str))
        uniq_pairs, pair_inverse = np.unique(pair_keys, return_inverse=True)
        # recover the actual (u,v) strings in the uniq_pairs order
        # (fast: take the first occurrence indices)
        first_idx = np.zeros_like(uniq_pairs, dtype=int)
        seen = {}
        for i, k in enumerate(pair_keys):
            if k not in seen:
                seen[k] = i
        for j, k in enumerate(uniq_pairs):
            first_idx[j] = seen[k]
        pairs = list(zip(u[first_idx], v[first_idx]))
        n_pairs = len(pairs)

        # accumulate per-task, per-pair effective wins for u and v sides
        SU = np.zeros((n_tasks, n_pairs), dtype=float)
        SV = np.zeros((n_tasks, n_pairs), dtype=float)

        for ti, pi, ai, bi, ui, vi, yi, wi in zip(task_inverse, pair_inverse, a, b, u, v, win, w):
            if yi == '1':  # method_1 (a) beat method_2 (b)
                if ui == ai:
                    SU[ti, pi] += wi
                else:
                    SV[ti, pi] += wi
            elif yi == '2':
                if ui == bi:
                    SU[ti, pi] += wi
                else:
                    SV[ti, pi] += wi
            else:  # tie
                SU[ti, pi] += 0.5 * wi
                SV[ti, pi] += 0.5 * wi

        # Build fixed design matrix X2 and Y2: two rows per pair with identical features
        p = len(all_models)
        X2 = np.zeros((2 * n_pairs, p), dtype=float)
        Y2 = np.empty(2 * n_pairs, dtype=float)
        logB = math.log(BASE)
        for i, (ui, vi) in enumerate(pairs):
            ui_idx = model_to_idx[ui]
            vi_idx = model_to_idx[vi]
            # row for y=1 (u beats v)
            X2[2 * i, ui_idx] = +logB
            X2[2 * i, vi_idx] = -logB
            Y2[2 * i] = 1.0
            # row for y=0 (v beats u) has identical X
            X2[2 * i + 1, ui_idx] = +logB
            X2[2 * i + 1, vi_idx] = -logB
            Y2[2 * i + 1] = 0.0

        return model_to_idx, pairs, X2, Y2, SU, SV, uniq_tasks

    def _bootstrap_draw_compressed(
        self,
        *,
        counts: np.ndarray,             # shape (n_tasks,)
        task_ids: np.ndarray,           # shape (n_tasks,), aligns with counts
        SU: np.ndarray,                 # shape (n_tasks, n_pairs)
        SV: np.ndarray,                 # shape (n_tasks, n_pairs)
        X2: np.ndarray,                 # shape (2*n_pairs, n_models)
        Y2: np.ndarray,                 # shape (2*n_pairs,)
        model_to_idx: pd.Series,        # index = model names, values = col indices
        battles: pd.DataFrame,          # original battles (used only for fallback)
        SCALE: int,
        INIT_RATING: float,
        solver: str,
        max_iter: int,
        calibration_framework: str | None = None,
        calibration_elo: float | None = None,
    ) -> pd.Series:
        """
        One bootstrap draw using the pair-compressed MLE path.
        Returns a pd.Series of ELO scores indexed by model names.
        """
        # Totals per unordered pair from task multiplicities
        SU_tot = counts @ SU    # (n_pairs,)
        SV_tot = counts @ SV    # (n_pairs,)

        # If one class is missing, fall back to iterative path for this draw
        has_pos = np.any(SU_tot > 0.0)
        has_neg = np.any(SV_tot > 0.0)
        if not (has_pos and has_neg):
            # Build multiplicity map for tasks in this draw
            mult_map = dict(zip(task_ids, counts))
            battles_boot = battles.copy()
            if "weight" not in battles_boot.columns:
                battles_boot["weight"] = 1.0
            battles_boot["weight"] = (
                battles_boot["weight"].to_numpy(dtype=float)
                * battles_boot[self.task_col].map(mult_map).fillna(0.0).to_numpy(dtype=float)
            )
            elo_scores = self.compute_iterative_elo_scores(
                battles_boot,
                INIT_RATING=INIT_RATING,
                SCALE=SCALE,
                models=pd.Series(np.arange(len(model_to_idx)), index=model_to_idx.index),
            )
            ser = pd.Series(elo_scores, index=model_to_idx.index)
        else:
            # Interleaved sample weights: [SU_tot[0], SV_tot[0], SU_tot[1], SV_tot[1], ...]
            n_pairs = SU_tot.size
            sw = np.empty(2 * n_pairs, dtype=float)
            sw[0::2] = SU_tot
            sw[1::2] = SV_tot

            # Optionally drop zero-weight pairs to shrink the fit
            active_pairs = (SU_tot + SV_tot) > 0.0
            if not np.all(active_pairs):
                idx_pairs = np.flatnonzero(active_pairs)
                row_idx = np.empty(2 * idx_pairs.size, dtype=int)
                row_idx[0::2] = 2 * idx_pairs
                row_idx[1::2] = 2 * idx_pairs + 1
                X_fit, Y_fit, sw_fit = X2[row_idx], Y2[row_idx], sw[row_idx]
            else:
                X_fit, Y_fit, sw_fit = X2, Y2, sw

            # Fit LR on the fixed design with per-draw weights
            lr = LogisticRegression(fit_intercept=False, C=1e6, solver=solver, max_iter=max_iter)
            lr.fit(X_fit, Y_fit, sample_weight=sw_fit)

            # Map coefficients -> ELO
            coef = lr.coef_[0]                  # aligned with model_to_idx order
            elo_vec = SCALE * coef + INIT_RATING
            out = np.ones(len(model_to_idx), dtype=float) * INIT_RATING
            out[:] = elo_vec
            ser = pd.Series(out, index=model_to_idx.index)

        # ---- Per-draw calibration (apply to the Series) ----
        if calibration_framework is not None:
            target = INIT_RATING if calibration_elo is None else float(calibration_elo)
            if calibration_framework in ser.index:
                ser = ser + (target - ser[calibration_framework])
            else:
                # Optional: warn once; skip calibration if anchor missing in this draw
                logger.warning(
                    "Calibration framework '%s' not present in bootstrap draw; skipping calibration for this draw.",
                    calibration_framework,
                )

        return ser
