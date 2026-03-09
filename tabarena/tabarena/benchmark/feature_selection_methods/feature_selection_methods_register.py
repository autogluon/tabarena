from __future__ import annotations

from copy import deepcopy


def get_default_encoding_pipeline():
    """Return the default pipeline for encoding non-numerical or categorical data.

    The default pipeline handles:
        - Text Features
        - Date Time Features

    Text features are used to generate semantic and statistical embeddings.
    """
    from autogluon.features.generators import (
        AsTypeFeatureGenerator,
        FillNaFeatureGenerator,
    )
    from autogluon.features.generators.selection import FeatureSelectionGenerator
    return {
        # Use default pre-generators but disabled conversion of bool features.
        "pre_generators": [
            AsTypeFeatureGenerator(convert_bool=False),
            FillNaFeatureGenerator(),
        ],
        "pre_enforce_types": False,
        "post_generators": [FeatureSelectionGenerator()]
    }


def get_fs_pipeline(fs_method):
    """Return the pipeline for encoding non-numerical or categorical data and doing feature selection.

        The default pipeline handles:
            - Text Features
            - Date Time Features

        Text features are used to generate semantic and statistical embeddings.
        """
    from autogluon.features.generators import (
        AsTypeFeatureGenerator,
        FillNaFeatureGenerator,
    )
    from autogluon.features.generators.selection import FeatureSelectionGenerator
    return {
        # Use default pre-generators but disabled conversion of bool features.
        "pre_generators": [
            AsTypeFeatureGenerator(convert_bool=False),
            FillNaFeatureGenerator(),
        ],
        "pre_enforce_types": False,
        "post_generators": [FeatureSelectionGenerator(fs_method)]
    }


def get_dimensionality_reduction_pipeline():
    from autogluon.features.generators.identity import IdentityFeatureGenerator
    return [
        # Passthrough for all non-text-embedding features
        IdentityFeatureGenerator(
            infer_features_in_args={
                "valid_raw_types": None,
            }
        ),
    ]


def no_feature_selection(experiment):
    default_pipeline = get_default_encoding_pipeline()
    new_experiment = deepcopy(experiment)
    new_experiment.method_kwargs["fit_kwargs"][
        "_feature_generator_kwargs"
    ] = default_pipeline
    return new_experiment


def get_lsflip_feature_selection(experiment):
    pipeline = get_fs_pipeline("LS_Flip")
    new_experiment = deepcopy(experiment)
    new_experiment.method_kwargs["fit_kwargs"][
        "_feature_generator_kwargs"
    ] = pipeline
    return new_experiment


def get_lsflipswap_feature_selection(experiment):
    pipeline = get_fs_pipeline("LS_FlipSwap")
    new_experiment = deepcopy(experiment)
    new_experiment.method_kwargs["fit_kwargs"][
        "_feature_generator_kwargs"
    ] = pipeline
    return new_experiment


def get_boruta_feature_selection(experiment):
    pipeline = get_fs_pipeline("Boruta")
    new_experiment = deepcopy(experiment)
    new_experiment.method_kwargs["fit_kwargs"][
        "_feature_generator_kwargs"
    ] = pipeline
    return new_experiment


DEFAULT_PIPELINE_WITHOUT_FEATURE_SELECTION = "NoFS"
LSFLIP_FEATURE_SELECTION = "LS_Flip"
LSFLIPSWAP_FEATURE_SELECTION = "LS_FlipSwap"
BORUTA_FEATURE_SELECTION = "Boruta"


PREPROCESSING_METHODS = {
    # DEFAULT_PIPELINE_WITHOUT_FEATURE_SELECTION: default_feature_selection,
    LSFLIP_FEATURE_SELECTION: get_lsflip_feature_selection,
    LSFLIPSWAP_FEATURE_SELECTION: get_lsflipswap_feature_selection,
    BORUTA_FEATURE_SELECTION: get_boruta_feature_selection
}
