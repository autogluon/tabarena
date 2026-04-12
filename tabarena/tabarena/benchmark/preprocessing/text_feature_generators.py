from __future__ import annotations

import re
import unicodedata
import warnings
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
from autogluon.common.features.types import (
    S_IMAGE_BYTEARRAY,
    S_IMAGE_PATH,
    S_TEXT,
    S_TEXT_EMBEDDING,
    S_TEXT_EMBEDDING_DR,
    S_TEXT_SPECIAL,
)
from autogluon.features import AbstractFeatureGenerator
from sklearn.decomposition import PCA
from tqdm import tqdm

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


# Non-printable ASCII control characters excluding whitespace (tab \x09, LF \x0a, CR \x0d).
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0e-\x1f\x7f]")


def sanitize_text(text_data: pd.Series, fillna_str: str = "Missing Data") -> pd.Series:
    """Normalize text by applying unicode normalization, removing control characters,
    stripping whitespace, replacing multiple spaces with a single space, and converting
    to lowercase.
    """
    return (
        text_data.fillna(fillna_str)
        .astype(str)
        # Unicode & canonical form normalization
        .map(lambda x: _CONTROL_CHAR_RE.sub("", unicodedata.normalize("NFKC", x)))
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )


class TabArenaDefaultTextEncoder:
    @staticmethod
    def get_default_encoder():
        """Get the default sentence transformer model for encoding text features."""
        import torch
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            "Qwen/Qwen3-Embedding-8B",
            truncate_dim=32,  # minimal MRL dimension for Qwen3-Embedding
            model_kwargs={"dtype": torch.float16, "attn_implementation": "sdpa"},
            processor_kwargs={"padding_side": "left"},
        )

    # TODO: could optimize this much more + ideally compute on-the-fly
    # Length-bucket thresholds (chars) and batch sizes, ordered from longest to shortest.
    # Texts longer than the threshold get the corresponding batch size.
    _LENGTH_BUCKETS: ClassVar[list[tuple[int, int]]] = [
        (20_000, 8),
        (15_000, 16),
        (5_000, 32),
        (500, 64),
        (0, 128),
    ]

    @staticmethod
    def encode_texts(*, texts: list[str], encoder_model: SentenceTransformer) -> np.ndarray:
        """Encode texts with adaptive batch sizes based on text length.

        Texts are sorted by character length and split into buckets.
        Longer texts use smaller batch sizes to limit peak memory, while
        shorter texts use larger batch sizes for throughput.
        """
        # guess-timate + overhead for characters per token
        max_chars = int(encoder_model.max_seq_length * 3)
        long_texts = [t for t in texts if len(t) > max_chars]
        if long_texts:
            warnings.warn(
                f"{len(long_texts)} of {len(texts)} text value(s) exceed "
                f"~{max_chars:,} characters and may be truncated by the model's "
                f"{encoder_model.max_seq_length:,}-token context window. "
                f"Longest text: {max(len(t) for t in long_texts):,} characters.",
                stacklevel=2,
            )

        # Sort by character length (good proxy for token length, avoids tokenization overhead).
        sorted_indices = sorted(range(len(texts)), key=lambda k: len(texts[k]))
        sorted_texts = [texts[k] for k in sorted_indices]

        print(f"Encoding {len(texts)} unique text values...")
        print(f"\tShortest text: {len(sorted_texts[0])} chars, longest text: {len(sorted_texts[-1])} chars.")
        print(f"\tAverage text length: {sum(len(t) for t in sorted_texts) / len(sorted_texts):.1f} chars.")

        # Split sorted texts into length buckets and encode each with its own batch size.
        buckets = TabArenaDefaultTextEncoder._LENGTH_BUCKETS
        all_embs_parts: list[np.ndarray] = []
        start = len(sorted_texts)  # walk backwards (longest first)
        for char_threshold, batch_size in buckets:
            # Find the first text that is shorter than the threshold.
            end = start
            start = end
            while start > 0 and len(sorted_texts[start - 1]) >= char_threshold:
                start -= 1
            bucket_texts = sorted_texts[start:end]
            if not bucket_texts:
                continue
            print(f"\tBucket >={char_threshold} chars: {len(bucket_texts)} texts, batch_size={batch_size}")
            embs = encoder_model.encode(
                bucket_texts,
                prompt_name="query",
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=True,
            )
            all_embs_parts.append(embs)

        # Reverse parts so they follow the original sorted order (shortest first).
        all_embs_parts.reverse()
        all_embs_sorted = np.concatenate(all_embs_parts, axis=0)

        # Unsort back to original ordering.
        return all_embs_sorted[np.argsort(sorted_indices)]

    @staticmethod
    def get_text_to_encode(*, X: pd.DataFrame, seen_texts: set[str]) -> list[str]:
        """Collect unique (column, value) pairs from *X* that are not present in *embedding_look_up*."""
        unseen_keys = []

        # Pass 1: discover unseen (col, value)
        for col in tqdm(X.columns, desc="Collecting text to encode..."):
            s = sanitize_text(X[col])

            for val in s.unique():
                if val not in seen_texts:
                    unseen_keys.append(val)

        return unseen_keys

    @staticmethod
    def get_cache_data_for_dataset(*, X: pd.DataFrame, seen_texts: set[str]) -> dict:
        """Get the cache data for the given dataset, which is a dict mapping (col, value) pairs to their embeddings."""
        text_to_encode = TabArenaDefaultTextEncoder.get_text_to_encode(
            X=X,
            seen_texts=seen_texts,
        )
        if not text_to_encode:
            return {}

        new_embeddings = TabArenaDefaultTextEncoder.encode_texts(
            texts=text_to_encode,
            encoder_model=TabArenaDefaultTextEncoder.get_default_encoder(),
        )
        return dict(zip(text_to_encode, new_embeddings))


class SemanticTextFeatureGenerator(AbstractFeatureGenerator):
    """Create semantic text embeddings using a pre-trained sentencetransformer model.

    Uses ``Qwen/Qwen3-Embedding-0.6B`` with Matryoshka Representation Learning
    (MRL) to produce compact ``_MRL_DIM``-dimensional embeddings per text value.
    """

    _embedding_look_up: dict[str, np.ndarray] = {}
    """Class-level cache for the embeddings of unique text values, shared across all instances within a process."""
    _expected_columns: list[str]
    """Expected columns during transform, set during fit."""
    _feature_names: list[str]
    """Stable feature names for the generated embedding features."""
    only_load_from_cache: bool = False
    """Whether to only load embeddings from cache and crash for on-the-fly encoding of unseen text values."""

    def _fit_transform(self, X: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
        """See parameters of the parent class AbstractFeatureGenerator for more details
        on the parameters.
        """
        if self.only_load_from_cache:
            self._encoder_model = None
        else:
            self._encoder_model = TabArenaDefaultTextEncoder.get_default_encoder()
        X_out = self._transform(X, is_train=True)
        return X_out, {S_TEXT_EMBEDDING: list(X_out.columns)}

    def _transform(self, X: pd.DataFrame, *, is_train: bool = False) -> pd.DataFrame:
        """See parameters of the parent class AbstractFeatureGenerator for more details
        on the parameters.
        """
        n_rows = len(X)
        n_cols = len(X.columns)

        # Empty guard
        if n_rows == 0 or n_cols == 0:
            raise ValueError("Input DataFrame is empty!")

        # Encode text
        unseen_text = TabArenaDefaultTextEncoder.get_text_to_encode(X=X, seen_texts=set(self._embedding_look_up.keys()))
        if unseen_text:

            if self.only_load_from_cache:
                raise ValueError(
                    "Cache miss for text values during transform with only_load_from_cache=True. "
                    f"Unseen text values: {unseen_text[:10]} (showing up to 10)."
                )

            embeddings = TabArenaDefaultTextEncoder.encode_texts(
                texts=list(unseen_text),
                encoder_model=self._encoder_model,
            )
            self._embedding_look_up.update(zip(unseen_text, embeddings))

        # Infer embedding dimension
        emb_dim = len(next(iter(self._embedding_look_up.values())))
        # --- Stable feature names: source column is the prefix ---
        if is_train:
            self._feature_names = [f"{col}.semantic_embedding_{i}" for col in X.columns for i in range(emb_dim)]
            self._expected_columns = list(X.columns)
        elif list(X.columns) != self._expected_columns:
            raise ValueError(
                "Column mismatch between training and transform.\n"
                f"Expected: {self._expected_columns}\n"
                f"Got: {list(X.columns)}"
            )

        # Pass 2: build matrix (optimized for repeated values)
        # Preallocate
        semantic_embedding = np.empty((n_rows, n_cols * emb_dim), dtype=np.float32)
        for j, col in tqdm(
            enumerate(X.columns),
            desc="Building semantic embedding matrix...",
            total=n_cols,
        ):
            s = sanitize_text(X[col])
            arr = s.to_numpy()

            uniques, inverse = np.unique(arr, return_inverse=True)
            unique_embs = np.vstack([self._embedding_look_up[val] for val in uniques])
            col_matrix = unique_embs[inverse]

            start = j * emb_dim
            end = start + emb_dim
            semantic_embedding[:, start:end] = col_matrix

        return pd.DataFrame(
            semantic_embedding,
            columns=self._feature_names,
            index=X.index,
        )

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        """Define the original feature's dtypes that this feature generator works on.

        See autogluon.features.FeatureMetadata.get_features for all options how to filter input data.
        See autogluon.features.types for all available raw and special types.
        """
        return {
            "required_special_types": [S_TEXT],
            "invalid_special_types": [S_IMAGE_PATH, S_IMAGE_BYTEARRAY],
        }

    def _more_tags(self):
        return {"feature_interactions": True}

    @staticmethod
    def save_embedding_cache(cache: dict[str, np.ndarray], path: str | Path) -> None:
        keys = list(cache.keys())
        embs = np.vstack(list(cache.values()))
        df = pd.DataFrame(embs, index=pd.Index(keys, name="text"))
        df.to_parquet(path)

    @staticmethod
    def load_embedding_cache(path: str | Path) -> dict[str, np.ndarray]:
        df = pd.read_parquet(path)
        return dict(zip(df.index, df.to_numpy()))

    @staticmethod
    def get_text_cache_dir(task_id_str: str) -> Path:
        import openml

        base_path = (openml.config._root_cache_directory / "tabarena_text_cache").expanduser().resolve() / "text_cache"
        Path(base_path).mkdir(parents=True, exist_ok=True)

        return base_path / f"{task_id_str}_cache.parquet"

class StatisticalTextFeatureGenerator(AbstractFeatureGenerator):
    """Generate a statistical embedding of text features using skrub.

    Uses a ``TableVectorizer`` backed by ``StringEncoder`` to produce dense numeric
    embeddings for each text column.  Output columns are renamed from the
    ``{col}_{i}`` format produced by ``TableVectorizer`` to ``{col}.{i}`` so that the
    source column can be recovered by splitting on the first ``"."``, consistent with
    the naming convention used by ``TextSpecialFeatureGenerator`` and
    ``SemanticTextFeatureGenerator``.
    """

    MAX_N_OUTPUT_FEATURES = 32

    def _fit_transform(self, X: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
        from skrub import StringEncoder, TableVectorizer

        self._vectorizer = TableVectorizer(
            cardinality_threshold=0,
            high_cardinality=StringEncoder(
                n_components=self.MAX_N_OUTPUT_FEATURES,
                random_state=0,
            ),
            numeric="drop",
            datetime="drop",
            n_jobs=-1,
        )

        X_out = self._transform(X, is_train=True)
        type_family_groups_special = {S_TEXT_EMBEDDING: list(X_out.columns)}
        return X_out, type_family_groups_special

    def _transform(self, X: pd.DataFrame, *, is_train: bool = False) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            X[col] = sanitize_text(X[col], fillna_str="NA")
        if is_train:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    module="skrub._string_encoder",
                )
                X = self._vectorizer.fit_transform(X)
            # TableVectorizer produces "{col}_{i}"; remap to "{col}.{i}" so that
            # the source column prefix is separated by "." (the project convention).
            self._col_rename_map_: dict[str, str] = {c: re.sub(r"_(\d+)$", r".\1", c) for c in X.columns}
        else:
            X = self._vectorizer.transform(X)

        return X.rename(columns=self._col_rename_map_)

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return {
            "required_special_types": [S_TEXT],
            "invalid_special_types": [S_IMAGE_PATH, S_IMAGE_BYTEARRAY],
        }


class TextEmbeddingDimensionalityReductionFeatureGenerator(AbstractFeatureGenerator):
    """Model-specific preprocessing to reduce the dimensionality of text embeddings.

    Features are grouped by their source column. A separate PCA is fitted for each group
    (or sub-group when ``max_features_per_group`` is finite).

    Pipeline:
        - Validate and remember training feature names.
        - Standard-scale with vectorized NumPy/Pandas ops.
        - Group columns by source column prefix.
        - Optionally split large groups into sub-batches of ``max_features_per_group``.
        - Run PCA on each (sub-)batch, with up to 30 components.
        - Keep only the components needed to explain 99% cumulative variance.

    Parameters
    ----------
    max_features_per_group:
        Maximum number of input features per PCA batch.  When ``float("inf")``
        (default) all features sharing the same source-column prefix are reduced
        with a single PCA.  When finite, each group is further split into
        sub-batches of at most this many features.
    """

    _MAX_COMPONENTS_PER_BATCH = 30
    _EXPLAINED_VARIANCE_THRESHOLD = 0.99

    def __init__(self, max_features_per_group: int | float = float("inf"), **kwargs):
        super().__init__(**kwargs)
        self.max_features_per_group = max_features_per_group

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return {
            "valid_special_types": [S_TEXT_EMBEDDING, S_TEXT_SPECIAL],
            "required_at_least_one_special": True,
        }

    def _more_tags(self):
        return {"feature_interactions": True}

    @staticmethod
    def get_infer_features_in_args_to_drop() -> dict:
        return {"invalid_special_types": [S_TEXT_EMBEDDING, S_TEXT_SPECIAL]}

    @staticmethod
    def _parse_source_column(feature_name: str) -> str:
        """Return the source column name encoded in *feature_name*.

        The project-wide convention is ``{source_col}.{rest}`` — a single ``"."``
        separates the original column name from the feature-specific suffix.
        If ``"."`` is absent the entire feature name is treated as the source column.

        This covers all text-feature naming schemes:

        * ``TextSpecialFeatureGenerator``: ``{col}.char_count``, ``{col}.word_count``
        * ``SemanticTextFeatureGenerator``: ``{col}.semantic_embedding_{i}``
        * ``StatisticalTextFeatureGenerator``: ``{col}.{i}``
        * ``TextEmbeddingDimensionalityReductionFeatureGenerator`` output: ``{col}.dr{b}_{i}``
        """
        return feature_name.split(".", 1)[0]

    def _make_batch_plan(self, feature_names: list[str]) -> list[tuple[str, int, list[str]]]:
        """Build a PCA batch plan grouped by source column.

        Parameters
        ----------
        feature_names:
            Ordered list of input feature names.

        Returns:
        -------
        list of ``(source_col, sub_batch_idx, feature_list)`` tuples, one entry
        per PCA that will be fitted.
        """
        # Group features by source column, preserving encounter order.
        groups: dict[str, list[str]] = defaultdict(list)
        for feat in feature_names:
            src = self._parse_source_column(feat)
            groups[src].append(feat)

        plan: list[tuple[str, int, list[str]]] = []
        max_n = self.max_features_per_group

        for src_col, feats in groups.items():
            if max_n == float("inf") or len(feats) <= max_n:
                plan.append((src_col, 0, feats))
            else:
                max_n_int = int(max_n)
                sub_batches = [feats[i : i + max_n_int] for i in range(0, len(feats), max_n_int)]
                for sub_idx, sub_feats in enumerate(sub_batches):
                    plan.append((src_col, sub_idx, sub_feats))

        return plan

    def _fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **kwargs,
    ) -> tuple[pd.DataFrame, dict]:
        # Persist original fit-time schema.
        self.feature_names_in_ = list(X.columns)
        self.expected_features_ = list(X.columns)

        X_out = self._fit_preprocess_and_transform(X=X, y=y)
        self.feature_names_out_ = list(X_out.columns)

        type_family_groups_special = {S_TEXT_EMBEDDING_DR: self.feature_names_out_}
        return X_out, type_family_groups_special

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if list(X.columns) != self.expected_features_:
            raise ValueError(
                "Column mismatch between training and transform.\n"
                f"Expected: {self.expected_features_}\n"
                f"Got: {list(X.columns)}"
            )

        X_out = self._transform_inference(X)
        missing_output = [c for c in self.feature_names_out_ if c not in X_out.columns]
        if missing_output:
            raise ValueError(f"Transformed output is missing expected columns: {missing_output[:10]}")
        return X_out[self.feature_names_out_]

    def _fit_preprocess_and_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        X = X.copy()

        self.pre_pca_feature_names_ = list(X)

        # Standard-scale with vectorized ops.
        X, means, stds = self._standard_scale_fit(X)
        self._scale_means_ = means
        self._scale_stds_ = stds

        # Build batch plan based on source column grouping.
        batch_plan = self._make_batch_plan(list(X.columns))

        # Fit PCA per batch.
        self._batch_pcas_: list[PCA] = []
        self._batch_input_columns_: list[list[str]] = []
        self._batch_output_columns_: list[list[str]] = []

        transformed_batches: list[pd.DataFrame] = []

        for src_col, sub_batch_idx, batch_cols in tqdm(batch_plan, desc="Fitting PCA batches..."):
            X_batch = X[batch_cols]

            n_samples, n_features = X_batch.shape
            n_components = min(self._MAX_COMPONENTS_PER_BATCH, n_samples, n_features)

            if n_components <= 0:
                continue

            pca = PCA(
                n_components=n_components,
                copy=False,
                random_state=0,
            )

            X_pca = pca.fit_transform(X_batch.to_numpy(copy=False))

            # Keep only components needed for 99% explained variance.
            keep_count = self._num_components_for_variance(
                pca.explained_variance_ratio_,
                threshold=self._EXPLAINED_VARIANCE_THRESHOLD,
            )
            X_pca = X_pca[:, :keep_count]

            output_cols = [f"{src_col}.dr{sub_batch_idx}_{i}" for i in range(keep_count)]

            X_pca_df = pd.DataFrame(X_pca, index=X.index, columns=output_cols)

            self._batch_pcas_.append(pca)
            self._batch_input_columns_.append(batch_cols)
            self._batch_output_columns_.append(output_cols)
            transformed_batches.append(X_pca_df)

        if not transformed_batches:
            raise ValueError("No PCA features were generated.")

        X = pd.concat(transformed_batches, axis=1)
        self._log(
            20,
            f"Total PCA features generated: {X.shape[1]} from {len(self.pre_pca_feature_names_)} original features.",
        )
        return X

    def _transform_inference(self, X: pd.DataFrame) -> pd.DataFrame:
        # Match fit-time raw schema/order exactly.
        X = X[self.expected_features_]
        # Apply fit-time pre-PCA column selection.
        X = X[self.pre_pca_feature_names_]
        # Apply fit-time scaling.
        X_scaled = self._standard_scale_transform(
            X,
            means=self._scale_means_,
            stds=self._scale_stds_,
        )

        transformed_batches: list[pd.DataFrame] = []
        for pca, batch_cols, output_cols in zip(
            self._batch_pcas_,
            self._batch_input_columns_,
            self._batch_output_columns_,
        ):
            X_batch = X_scaled[batch_cols]
            X_pca = pca.transform(X_batch.to_numpy(copy=False))
            X_pca = X_pca[:, : len(output_cols)]

            X_pca_df = pd.DataFrame(X_pca, index=X.index, columns=output_cols)
            transformed_batches.append(X_pca_df)

        if not transformed_batches:
            raise ValueError("No PCA features were produced during transform.")

        return pd.concat(transformed_batches, axis=1)

    @staticmethod
    def _standard_scale_fit(
        X: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        # Vectorized mean/std; avoid sklearn scaler.
        means = X.mean(axis=0)
        stds = X.std(axis=0, ddof=0)

        # Guard against numerical zeros.
        stds = stds.mask(stds == 0.0, 1.0)

        X_scaled = (X - means) / stds
        return X_scaled, means, stds

    @staticmethod
    def _standard_scale_transform(
        X: pd.DataFrame,
        means: pd.Series,
        stds: pd.Series,
    ) -> pd.DataFrame:
        return (X - means) / stds

    @staticmethod
    def _encode_target_for_correlation(y: pd.Series) -> np.ndarray:
        if pd.api.types.is_numeric_dtype(y):
            y_num = pd.to_numeric(y, errors="coerce").to_numpy(dtype=np.float64, copy=False)
        else:
            # Deterministic encoding for non-numeric labels.
            y_num = pd.Series(pd.factorize(y)[0], index=y.index).to_numpy(dtype=np.float64, copy=False)

        if np.isnan(y_num).any():
            # Fill NaNs with mean to keep correlation computation vectorized/stable.
            mean_val = np.nanmean(y_num)
            if np.isnan(mean_val):
                mean_val = 0.0
            y_num = np.where(np.isnan(y_num), mean_val, y_num)

        return y_num

    def _sort_columns_by_target_correlation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> list[str]:
        """Sort columns by absolute Pearson correlation to the target.

        Assumes X is already standardized. Uses vectorized NumPy ops.
        """
        y_num = self._encode_target_for_correlation(y)

        y_centered = y_num - y_num.mean()
        y_std = y_centered.std(ddof=0)
        if y_std == 0.0:
            # Degenerate target: preserve existing order.
            return list(X.columns)

        y_scaled = y_centered / y_std

        x_mat = X.to_numpy(dtype=np.float64, copy=False)
        # Since X is standardized already, corr ~= mean(x * y_scaled).
        corr = np.abs(np.mean(x_mat * y_scaled[:, None], axis=0))

        order = np.argsort(-corr, kind="stable")
        return X.columns[order].tolist()

    @staticmethod
    def _num_components_for_variance(
        explained_variance_ratio: np.ndarray,
        threshold: float,
    ) -> int:
        cumulative = np.cumsum(explained_variance_ratio)
        keep_count = int(np.searchsorted(cumulative, threshold, side="left") + 1)
        return max(1, min(keep_count, len(explained_variance_ratio)))
