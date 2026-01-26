

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
            Categorical("use_arithmetic_preprocessor", [True]),
            Categorical("use_cat_fe", [True]),
            Categorical("use_rstafc", [True]),
            Categorical("use_groupby", [True]), 
            Categorical("use_select_spearman", [True]), # Might rather tune no. of features, i.e. in {1000, 1500, 2000}

            Categorical("arithmetic_max_feats", [2000, 1000]),
            Categorical("arithmetic_random_state", [42,84,168,336,672]),

            Categorical("cat_fe_max_feats", [100, 500]),
            Categorical("cat_fe_random_state", [42,84,168,336,672]),

            Categorical("rstafc_n_subsets", [50,100, 1]),
            Categorical("rstafc_random_state", [42,84,168,336,672]),

            Categorical("oofte_random_state", [42,84,168,336,672]),

            Categorical("groupby_max_feats", [500, 100, 1000]), 

            Categorical("spearman_max_feats", [2000]),

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
        cur_config = prep_configs[i]
        cur_config = convert_numpy_dtypes(cur_config)

        if 'ag.prep_params' not in configs[i]:
            configs[i]['ag.prep_params'] = []
        pipeline = []
        prep_params_passthrough_types = None
        use_arithmetic_preprocessor = cur_config.pop('use_arithmetic_preprocessor', False)
        use_cat_fe = cur_config.pop('use_cat_fe', False)
        use_tafc = cur_config.pop('use_tafc', False)
        use_rstafc = cur_config.pop('use_rstafc', False)
        use_neighbor_interactions = cur_config.pop('use_neighbor_interactions', False)
        use_neighbor_structure = cur_config.pop('use_neighbor_structure', False)
        use_groupby = cur_config.pop('use_groupby', False)
        use_linear_feature = cur_config.pop('use_linear_feature', False)
        use_select_spearman = cur_config.pop('use_select_spearman', False)
        arithmetic_max_feats = cur_config.pop('arithmetic_max_feats', 2000)
        arithmetic_random_state = cur_config.pop('arithmetic_max_feats_random_state', 42)
        cat_fe_max_feats = cur_config.pop('cat_fe_max_feats', 100)
        cat_fe_random_state = cur_config.pop('cat_fe_random_state', 42)
        rstafc_n_subsets = cur_config.pop('rstafc_n_subsets', 50)
        rstafc_random_state = cur_config.pop('rstafc_random_state', 42)
        oofte_random_state = cur_config.pop('oofte_random_state', 42)
        groupby_max_feats = cur_config.pop('groupby_max_feats', 500)
        spearman_max_feats = cur_config.pop('spearman_max_feats', 2000)
        

        if use_groupby:
            pipeline.append(['GroupByFeatureGenerator', {"max_features": groupby_max_feats}])
            
        if use_tafc:
            pipeline.append(['TargetAwareFeatureCompressionFeatureGenerator', {"random_state": oofte_random_state}])

        if use_rstafc:
            pipeline.append(['RandomSubsetTAFC', {"n_subsets": rstafc_n_subsets, "random_state": rstafc_random_state}])

        if use_neighbor_interactions:
            pipeline.append(['NeighborInteractionFeatureGenerator', {}])
        
        if use_neighbor_structure:
            pipeline.append(['NeighborStructureFeatureGenerator', {}])

        if use_linear_feature:
            pipeline.append(['LinearFeatureGenerator', {}])
        
        if use_arithmetic_preprocessor:
            _generator_params = {"max_new_feats": arithmetic_max_feats, "random_state": arithmetic_random_state}
            pipeline.append(['ArithmeticFeatureGenerator', _generator_params])

        cat_pipeline = [['OOFTargetEncodingFeatureGenerator', {}]]
        prep_params_passthrough_types = {"invalid_raw_types": ["category", "object"]}
        if use_cat_fe:
            cat_pipeline.append([
                ['CategoricalInteractionFeatureGenerator', {"passthrough": True, "max_new_feats": cat_fe_max_feats, "random_state": cat_fe_random_state}],
            ])
            cat_pipeline.reverse()#
        pipeline.append(cat_pipeline)

        if use_select_spearman:
            configs[i]['ag.prep_params'].append(pipeline)
            configs[i]['ag.prep_params'].append([
                ['SpearmanFeatureSelector', {'max_features': spearman_max_feats}],
            ])
        else:
            configs[i]['ag.prep_params'].append(pipeline)


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
