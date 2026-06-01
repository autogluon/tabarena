"""Evaluation contexts: pluggable benchmark definitions (methods + task metadata + subsets).

A *context* bundles everything an evaluation needs to know about a benchmark family — its method
collection, its task metadata, and its subset predicates.
:class:`~tabarena.nips2025_utils.tabarena_context.TabArenaContext` is the base/TabArena-v0.1
context; :class:`~tabarena.evaluation.context.beyond_arena.BeyondArenaContext` is the data-foundry
(BeyondArena) variant. New benchmark families add a context here.

Imported lazily by the runners, so this subpackage is not pulled in at ``import tabarena.evaluation``.
"""

from __future__ import annotations

from tabarena.evaluation.context.beyond_arena import BeyondArenaContext

__all__ = ["BeyondArenaContext"]
