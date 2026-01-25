from __future__ import annotations

import numpy as np

from tabarena.benchmark.experiment import YamlExperimentSerializer
from tabarena.benchmark.models.prep_ag.prep_realmlp.prep_realmlp_model import PrepRealMLPModel
from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator, generate_bag_experiments
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

prep_manual_configs = [
    {
        "use_arithmetic_preprocessor": True,
        "use_cat_fe": True,
        "use_groupby": True,
        "use_rstafc": True,
        "use_select_spearman": True,
    }]

def generate_single_config_realmlp(rng):
    # common search space
    params = {
        "n_hidden_layers": rng.integers(2, 4, endpoint=True),
        "hidden_sizes": "rectangular",
        "hidden_width": rng.choice([256, 384, 512]),
        "p_drop": rng.uniform(0.0, 0.5),
        "act": "mish",
        "plr_sigma": np.exp(rng.uniform(np.log(1e-2), np.log(50))),
        "sq_mom": 1.0 - np.exp(rng.uniform(np.log(5e-3), np.log(5e-2))),
        "plr_lr_factor": np.exp(rng.uniform(np.log(5e-2), np.log(3e-1))),
        "scale_lr_factor": np.exp(rng.uniform(np.log(2.0), np.log(10.0))),
        "first_layer_lr_factor": np.exp(rng.uniform(np.log(0.3), np.log(1.5))),
        "ls_eps_sched": "coslog4",
        "ls_eps": np.exp(rng.uniform(np.log(5e-3), np.log(1e-1))),
        "p_drop_sched": "flat_cos",
        "lr": np.exp(rng.uniform(np.log(2e-2), np.log(3e-1))),
        "wd": np.exp(rng.uniform(np.log(1e-3), np.log(5e-2))),
        "use_ls": rng.choice(
            [False, True]  # changed "auto" to False, to have it equal for all metrics
        ),  # use label smoothing (will be ignored for regression)
        "max_one_hot_cat_size": np.floor(np.exp(rng.uniform(np.log(4.0), np.log(33.0)))).item(),
        "embedding_size": rng.choice([4, 8, 16]),
        "n_ens": 8,  # 16 might still be feasible
        "ens_av_before_softmax": False,
    }

    if rng.uniform(0.0, 1.0) > 0.5:
        # large configs
        params["plr_hidden_1"] = rng.choice([8, 16, 32, 64])
        params["plr_hidden_2"] = rng.choice([8, 16, 32, 64])
        params["n_epochs"] = rng.choice([256, 512])
        params["use_early_stopping"] = True
    else:
        # default values, used here to always set the same set of parameters
        params["plr_hidden_1"] = 16
        params["plr_hidden_2"] = 4
        params["n_epochs"] = 256
        params["use_early_stopping"] = False

    return convert_numpy_dtypes(params)


def generate_configs_realmlp(num_random_configs=200, seed=1234):
    # note: this doesn't set val_metric_name, which should be set outside
    rng = np.random.default_rng(seed)
    configs = [generate_single_config_realmlp(rng) for _ in range(num_random_configs)]


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
        arithmetic_max_feats = prep_configs[i].pop('arithmetic_max_feats', 2000)
        arithmetic_random_state = prep_configs[i].pop('arithmetic_max_feats_random_state', 42)
        cat_fe_max_feats = prep_configs[i].pop('cat_fe_max_feats', 100)
        cat_fe_random_state = prep_configs[i].pop('cat_fe_random_state', 42)
        rstafc_n_subsets = prep_configs[i].pop('rstafc_n_subsets', 50)
        rstafc_random_state = prep_configs[i].pop('rstafc_random_state', 42)
        oofte_random_state = prep_configs[i].pop('oofte_random_state', 42)
        groupby_max_feats = prep_configs[i].pop('groupby_max_feats', 500)
        spearman_max_feats = prep_configs[i].pop('spearman_max_feats', 2000)
        

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
            configs[i]['ag.prep_params'].extend(pipeline)


        if prep_params_passthrough_types:
            configs[i]['ag.prep_params.passthrough_types'] = prep_params_passthrough_types


    return configs

gen_realmlp = CustomAGConfigGenerator(
    model_cls=PrepRealMLPModel,
    search_space_func=generate_configs_realmlp,
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
    configs_yaml = []
    config_defaults = [{}]
    configs = generate_configs_realmlp(100, seed=1234)

    experiments_realmlp_streamlined = gen_realmlp.generate_all_bag_experiments(100)

    experiments_default = generate_bag_experiments(
        model_cls=PrepRealMLPModel,
        configs=config_defaults,
        time_limit=3600,
        name_id_prefix="c",
    )
    experiments_random = generate_bag_experiments(
        model_cls=PrepRealMLPModel, configs=configs, time_limit=3600
    )
    experiments = experiments_default + experiments_random
    YamlExperimentSerializer.to_yaml(
        experiments=experiments, path="configs_realmlp.yaml"
    )
