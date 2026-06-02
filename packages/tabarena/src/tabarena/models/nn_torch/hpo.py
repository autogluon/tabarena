from __future__ import annotations

from autogluon.common.space import Categorical, Int, Real
from autogluon.tabular.models import TabularNeuralNetTorchModel

from tabarena.utils.config_utils import ConfigGenerator

search_space = {
    "learning_rate": Real(1e-4, 3e-2, default=3e-4, log=True),
    "weight_decay": Real(1e-12, 0.1, default=1e-6, log=True),
    "dropout_prob": Real(0.0, 0.4, default=0.1),
    "use_batchnorm": Categorical(False, True),
    "num_layers": Int(1, 5, default=2),
    "hidden_size": Int(8, 256, default=128),
    "activation": Categorical("relu", "elu"),
}


gen_nn_torch = ConfigGenerator(
    model_cls=TabularNeuralNetTorchModel,
    manual_configs=[{}],
    search_space=search_space,
)
