from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.lattice.hpo import gen_lattice, gen_lattice_tiny
from tabarena.models.lattice.model import LatticeModel, LatticeTinyModel

_PIP_EXTRA = (
    "lattice[fast] @ git+https://github.com/hachure-ai/Lattice.git@e106f9fba5994f85aca9593ae82af561054ab77a",
    "torch>=2.7.1",  # AutoGluon's tensor serialization needs torch.utils.serialization.
)


def _model_info(name: str, model_cls: type[LatticeModel], generator) -> ModelInfo:
    descriptor = ModelDescriptor(
        display_name=name,
        compute="gpu",
        is_bag=False,
        reference_url="https://github.com/hachure-ai/Lattice",
        organization="Nanjing University",
        license="CC-BY-NC-4.0 (weights)",
        commercial_use=False,
        date_introduced="2026-11",
    )
    metadata = descriptor.method_metadata(
        method=name,
        ag_key=model_cls.ag_key,
        config_default=f"{name}_c1_default_BAG_L1",
        can_hpo=False,
        verified=False,
        validation_protocol="8x1",
    )
    return ModelInfo(
        model_cls=model_cls,
        search_space=generator,
        method_metadata=metadata,
        pip_extra=_PIP_EXTRA,
        prefetch_weights=model_cls.prefetch_weights,
    )


lattice_info = _model_info("Lattice", LatticeModel, gen_lattice)
lattice_tiny_info = _model_info("Lattice-tiny", LatticeTinyModel, gen_lattice_tiny)
