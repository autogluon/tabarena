

from __future__ import annotations

from tabarena.benchmark.models.prep_ag.prep_lgb.prep_lgb_model import PrepLGBModel
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

from tabarena.benchmark.experiment import YamlExperimentSerializer
from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator

import numpy as np

def generate_configs_lightgbm(num_random_configs=200) -> list:
    search_space = ConfigurationSpace(
        space=[
            Float("learning_rate", (5e-3, 1e-1), log=True),
            Float("feature_fraction", (0.4, 1.0)),
            Float("bagging_fraction", (0.7, 1.0)),
            Categorical("bagging_freq", [1]),
            Integer("num_leaves", (2, 200), log=True),
            Integer("min_data_in_leaf", (1, 64), log=True),
            Categorical("extra_trees", [False, True]),
            # categorical hyperparameters
            Integer("min_data_per_group", (2, 100), log=True),
            Float("cat_l2", (5e-3, 2), log=True),
            Float("cat_smooth", (1e-3, 100), log=True),
            Integer("max_cat_to_onehot", (8, 100), log=True),
            # these seem to help a little bit but can also make things slower
            Float("lambda_l1", (1e-4, 1.0)),
            Float("lambda_l2", (1e-4, 2.0)),
            # could search max_bin but this is expensive
        ],
        seed=1234,
    )

    prep_search_space = ConfigurationSpace(
        space=[
            # Preprocessing hyperparameters
            Categorical("use_arithmetic_preprocessor", [True, False]),
            Categorical("use_cat_fe", [True, False]),
            Categorical("use_rstafc", [True, False]),
            Categorical("use_groupby", [True, False]), 
            Categorical("use_select_spearman", [True]), # Might rather tune no. of features, i.e. in {1000, 1500, 2000}
        ],
        seed=123,
    )       

    configs = search_space.sample_configuration(num_random_configs)
    prep_configs = prep_search_space.sample_configuration(num_random_configs)
    
    if num_random_configs == 1:
        configs = [configs]
        prep_configs = [prep_configs]
    configs = [dict(config) for config in configs]
    prep_configs = [dict(config) for config in prep_configs]

    for i in range(len(prep_configs)):
        if 'ag.prep_params' not in configs[i]:
            configs[i]['ag.prep_params'] = []
        pipeline = []
        prep_params_passthrough_types = None
        use_arithmetic_preprocessor = prep_configs[i].pop('use_arithmetic_preprocessor', False)
        use_cat_fe = prep_configs[i].pop('use_cat_fe', False)
        use_tafc = prep_configs[i].pop('use_tafc', False)
        use_rstafc = prep_configs[i].pop('use_rstafc', False)
        use_neighbor_interactions = prep_configs[i].pop('use_neighbor_interactions', False)
        use_neighbor_structure = prep_configs[i].pop('use_neighbor_structure', False)
        use_groupby = prep_configs[i].pop('use_groupby', False)
        use_linear_feature = prep_configs[i].pop('use_linear_feature', False)
        use_select_spearman = prep_configs[i].pop('use_select_spearman', False)
        
        if use_groupby:
            pipeline.append(['GroupByFeatureGenerator', {}])
            
        if use_tafc:
            pipeline.append(['TargetAwareFeatureCompressionFeatureGenerator', {}])

        if use_rstafc:
            pipeline.append(['RandomSubsetTAFC', {}])

        if use_neighbor_interactions:
            pipeline.append(['NeighborInteractionFeatureGenerator', {}])
        
        if use_neighbor_structure:
            pipeline.append(['NeighborStructureFeatureGenerator', {}])

        if use_linear_feature:
            pipeline.append(['LinearFeatureGenerator', {}])
        
        if use_arithmetic_preprocessor:
            _generator_params = {}
            pipeline.append(['ArithmeticFeatureGenerator', _generator_params])

        cat_pipeline = [['OOFTargetEncodingFeatureGenerator', {}]]
        prep_params_passthrough_types = {"invalid_raw_types": ["category", "object"]}
        if use_cat_fe:
            cat_pipeline.append([
                ['CategoricalInteractionFeatureGenerator', {"passthrough": True}],
            ])
            cat_pipeline.reverse()

        if use_select_spearman:
            configs[i]['ag.prep_params'].append(pipeline)
            configs[i]['ag.prep_params'].append([
                ['SpearmanFeatureSelector', {'max_features': 2000}],
            ])
        else:
            configs[i]['ag.prep_params'].extend(pipeline)


        if prep_params_passthrough_types:
            configs[i]['ag.prep_params.passthrough_types'] = prep_params_passthrough_types


    return [convert_numpy_dtypes(config) for config in configs]

gen_lightgbm = CustomAGConfigGenerator(
    model_cls=PrepLGBModel,
    search_space_func=generate_configs_lightgbm,
    manual_configs=[
        {
        'ag.prep_params': [
            [
                ['GroupByFeatureGenerator', {}],
                ['RandomSubsetTAFC', {}],
                ['ArithmeticFeatureGenerator', {}],
                [
                    ['CategoricalInteractionFeatureGenerator', {"passthrough": True}],
                    ['OOFTargetEncodingFeatureGenerator', {}],
                ]],            
        ['SpearmanFeatureSelector', {'max_features': 2000}]            
        ],
        'ag.prep_params.passthrough_types': {"invalid_raw_types": ["category", "object"]}, # We never keep categorical features
        'ag.use_residuals': False,
        'ag.residual_type': 'oof',
        'ag.max_dataset_size_for_residuals': 1000,
        'ag.residual_init_kwargs': {},
        },
    ],
)

if __name__ == "__main__":
    experiments = gen_lightgbm.generate_all_bag_experiments(num_random_configs=200)
    YamlExperimentSerializer.to_yaml(
        experiments=experiments, path="configs_lightgbm_alt.yaml"
    )
