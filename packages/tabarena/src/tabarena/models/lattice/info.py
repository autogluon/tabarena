from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata, ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.lattice.hpo import gen_lattice, gen_lattice_tiny
from tabarena.models.lattice.model import LatticeModel, LatticeTinyModel

_PIP_EXTRA = (
    "lattice @ git+https://github.com/hachure-ai/Lattice.git@4ab2558dba4001cb69dfd552ace14358a8af1b7d",
    "torch==2.14.1",
    "triton==3.8.0; sys_platform == 'linux'",
)


def _descriptor(name: str) -> ModelDescriptor:
    return ModelDescriptor(
        display_name=name,
        compute="gpu",
        is_bag=False,
        reference_url="https://github.com/hachure-ai/Lattice",
        organization="Nanjing University",
        license="CC-BY-NC-4.0 (weights)",
        commercial_use=False,
        date_introduced="2026-11",
    )


lattice_descriptor = _descriptor("Lattice")
lattice_tiny_descriptor = _descriptor("Lattice-tiny")

# Run lattice_19112026 (BENCHMARK_LOG.md): both sizes on all 51 datasets, every split, on a SkyPilot pool of
# RTX PRO 6000 with the CUDA 12.8 build of torch 2.7.1 and triton 3.3.1 (the stack of Lattice's Triton fast path).
_tabarena_run_kwargs = dict(
    can_hpo=False,
    date="2026-11-19",
    verified=True,
    validation_protocol="8x1",
    suite="tabarena-2026-11-19",
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)


def _method_metadata(descriptor: ModelDescriptor, model_cls: type[LatticeModel]) -> MethodMetadata:
    name = descriptor.display_name
    return descriptor.method_metadata(
        method=name, ag_key=model_cls.ag_key, config_default=f"{name}_c1_default_BAG_L1", **_tabarena_run_kwargs
    )


lattice_method_metadata = _method_metadata(lattice_descriptor, LatticeModel)
lattice_tiny_method_metadata = _method_metadata(lattice_tiny_descriptor, LatticeTinyModel)


def _model_info(model_cls: type[LatticeModel], generator, metadata: MethodMetadata) -> ModelInfo:
    return ModelInfo(
        model_cls=model_cls,
        search_space=generator,
        method_metadata=metadata,
        pip_extra=_PIP_EXTRA,
        prefetch_weights=model_cls.prefetch_weights,
    )


lattice_info = _model_info(LatticeModel, gen_lattice, lattice_method_metadata)
lattice_tiny_info = _model_info(LatticeTinyModel, gen_lattice_tiny, lattice_tiny_method_metadata)
