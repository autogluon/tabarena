from __future__ import annotations

import numpy as np
# from autogluon.common.space import Categorical
from tabarena.benchmark.models.prep_ag.prep_tabm.prep_tabm_model import PrepTabMModel
from tabarena.models.utils import convert_numpy_dtypes
from ...utils.config_utils import PrepConfigGenerator
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer
from tabarena.benchmark.experiment import YamlExperimentSerializer
from tabarena.utils.config_utils import CustomAGConfigGenerator

name = "prep_TabM"

prep_manual_configs = [
    {
        "use_arithmetic_preprocessor": True,
        "use_cat_fe": True,
        "use_groupby": True,
        "use_rstafc": True,
        "use_select_spearman": True,
    }]

# prep_search_space = {
#         # Preprocessing hyperparameters
#         "use_arithmetic_preprocessor": Categorical(True),
#         "use_cat_fe": Categorical(True),
#         "use_rstafc": Categorical(True),
#         "use_groupby": Categorical(True), 
#         "use_select_spearman": Categorical(True), # Might rather tune no. of features, i.e. in {1000, 1500, 2000}

#         "arithmetic_max_feats": Categorical(2000, 1000),
#         "arithmetic_random_state": Categorical(42,84,168,336,672),

#         "cat_fe_max_feats": Categorical(100, 500),
#         "cat_fe_random_state": Categorical(42,84,168,336,672),

#         "rstafc_n_subsets": Categorical(50,100, 1),
#         "rstafc_random_state": Categorical(42,84,168,336,672),

#         "oofte_random_state": Categorical(42,84,168,336,672),

#         "groupby_max_feats": Categorical(500, 100, 1000), 

#         "spearman_max_feats": Categorical(2000),
# }       

def generate_single_config_tabm(rng):
    # taken from https://github.com/yandex-research/tabm/blob/main/exp/tabm-mini-piecewiselinear/adult/0-tuning.toml
    # discussed with the authors
    params = {
        "batch_size": "auto",
        "patience": 16,
        "amp": False,  # only for GPU, maybe we should change it to True?
        "arch_type": "tabm-mini",
        "tabm_k": 32,
        "gradient_clipping_norm": 1.0,
        # this makes it probably slower with numerical embeddings, and also more RAM intensive
        # according to the paper it's not very important but should be a bit better (?)
        "share_training_batches": False,
        "lr": np.exp(rng.uniform(np.log(1e-4), np.log(3e-3))),
        "weight_decay": rng.choice(
            [0.0, np.exp(rng.uniform(np.log(1e-4), np.log(1e-1)))]
        ),
        # removed n_blocks=1 according to Yury Gurishniy's advice
        "n_blocks": rng.choice([2, 3, 4, 5]),
        # increased lower limit from 64 to 128 according to Yury Gorishniy's advice
        "d_block": rng.choice([i for i in range(128, 1024 + 1) if i % 16 == 0]),
        "dropout": rng.choice([0.0, rng.uniform(0.0, 0.5)]),
        # numerical embeddings
        "num_emb_type": "pwl",
        "d_embedding": rng.choice([i for i in range(8, 32 + 1) if i % 4 == 0]),
        "num_emb_n_bins": rng.integers(2, 128, endpoint=True),
        # could reduce eval_batch_size in case of OOM
    }

    return convert_numpy_dtypes(params)


def generate_configs_tabm(num_random_configs=200, seed=1234):
    # note: this doesn't set val_metric_name, which should be set outside
    rng = np.random.default_rng(seed)
    configs = [generate_single_config_tabm(rng) for _ in range(num_random_configs)]


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

gen_tabm = CustomAGConfigGenerator(
    model_cls=PrepTabMModel,
    search_space_func=generate_configs_tabm,
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
    experiments = gen_tabm.generate_all_bag_experiments(num_random_configs=200)
    YamlExperimentSerializer.to_yaml(
        experiments=experiments, path="configs_prep_tabm_alt.yaml"
    )
