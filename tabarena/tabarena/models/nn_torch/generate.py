from autogluon.common.space import Real, Int, Categorical
from autogluon.tabular.models import TabularNeuralNetTorchModel

from ...utils.config_utils import ConfigGenerator


name = 'NeuralNetTorch'
manual_configs = [
    {},
    {'use_batchnorm': True},
    {'num_layers': 2},
    {'use_batchnorm': True, 'num_layers': 2},
]
search_space = {
    'learning_rate': Real(1e-4, 3e-2, default=3e-4, log=True),
    'weight_decay': Real(1e-12, 0.1, default=1e-6, log=True),
    'dropout_prob': Real(0.0, 0.4, default=0.1),
    # 'embedding_size_factor': Categorical(1.0, 0.5, 1.5, 0.7, 0.6, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4),
    # 'proc.embed_min_categories': Categorical(4, 3, 10, 100, 1000),
    # 'proc.impute_strategy': Categorical('median', 'mean', 'most_frequent'),
    # 'proc.max_category_levels': Categorical(100, 10, 20, 200, 300, 400, 500, 1000, 10000),
    # 'proc.skew_threshold': Categorical(0.99, 0.2, 0.3, 0.5, 0.8, 0.9, 0.999, 1.0, 10.0, 100.0),
    'use_batchnorm': Categorical(False, True),
    'num_layers': Int(1, 5, default=2),
    'hidden_size': Int(8, 256, default=128),
    'activation': Categorical('relu', 'elu'),
}

gen_nn_torch = ConfigGenerator(model_cls=TabularNeuralNetTorchModel, manual_configs=[{}], search_space=search_space)


def generate_configs_nn_torch(num_random_configs=200):
    config_generator = ConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
