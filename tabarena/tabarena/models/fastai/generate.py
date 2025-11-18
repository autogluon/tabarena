from autogluon.common.space import Real, Int, Categorical
from autogluon.tabular.models import NNFastAiTabularModel

from ...utils.config_utils import ConfigGenerator


name = 'NeuralNetFastAI'
manual_configs = [
    {},
]
search_space = {
    # See docs: https://docs.fast.ai/tabular.learner.html
    'layers': Categorical([200], [400], [200, 100], [400, 200], [800, 400], [200, 100, 50], [400, 200, 100]),  # FIXME: None crashes when saved to json, turns to null
    'emb_drop': Real(0.0, 0.7, default=0.1),
    'ps': Real(0.0, 0.7, default=0.1),
    'bs': Categorical(256, 128, 512, 1024, 2048),
    'lr': Real(5e-4, 1e-1, default=1e-2, log=True),
    'epochs': Int(lower=20, upper=50, default=30),
}

gen_fastai = ConfigGenerator(model_cls=NNFastAiTabularModel, manual_configs=manual_configs, search_space=search_space)


def generate_configs_fastai(num_random_configs=200):
    config_generator = ConfigGenerator(name=name, manual_configs=manual_configs, search_space=search_space)
    return config_generator.generate_all_configs(num_random_configs=num_random_configs)
