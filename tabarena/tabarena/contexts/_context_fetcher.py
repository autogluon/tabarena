from typing import Dict, List

from .context import BenchmarkContext
from .subcontext import BenchmarkSubcontext
from .context_2023_11_14 import (
    D244_F3_C1530,
    D244_F3_C1530_200,
    D244_F3_C1530_175,
    D244_F3_C1530_100,
    D244_F3_C1530_30,
    D244_F3_C1530_10,
    D244_F3_C1530_3,
)

__all__ = [
    'list_contexts',
    'list_subcontexts',
    'get_context',
    'get_subcontext',
]


_context_map: Dict[str, BenchmarkContext] = dict()
_subcontext_map: Dict[str, BenchmarkSubcontext] = dict()


def _add_subcontext(subcontext: BenchmarkSubcontext):
    if subcontext.name in _subcontext_map:
        raise AssertionError(f'ERROR: Multiple subcontexts have the same name: {subcontext.name}')
    _subcontext_map[subcontext.name] = subcontext


for c in [
    D244_F3_C1530,
    D244_F3_C1530_200,
    D244_F3_C1530_175,
    D244_F3_C1530_100,
    D244_F3_C1530_30,
    D244_F3_C1530_10,
    D244_F3_C1530_3,
]:
    if c.name in _context_map:
        raise AssertionError(f'ERROR: Multiple contexts have the same name: {c.name}')
    _context_map[c.name] = c


for c_name, c in _context_map.items():
    subcontext = BenchmarkSubcontext(parent=c)
    _add_subcontext(subcontext=subcontext)


def list_contexts() -> List[str]:
    return sorted(list(_context_map.keys()))


def list_subcontexts() -> List[str]:
    return sorted(list(_subcontext_map.keys()))


def get_context(name: str) -> BenchmarkContext:
    if name not in _context_map:
        available_names = list_contexts()
        raise ValueError(f'Could not find context with name="{name}". '
                         f'Valid names: {available_names}')
    return _context_map[name]


def get_subcontext(name: str) -> BenchmarkSubcontext:
    if name not in _subcontext_map:
        available_names = list_subcontexts()
        raise ValueError(f'Could not find context with name="{name}". '
                         f'Valid names: {available_names}')
    return _subcontext_map[name]
