from __future__ import annotations

import copy
from threading import RLock

import numpy as np
import pandas as pd
import torch
from sdm import TableTensor, Task
from sdm.models import KumoTabular
from torch import nn
from torch.nn import functional as F

from .weights import checkpoint_path


class CellRidgeNetwork(nn.Module):
    """Kumo Small with four row blocks and a support-fitted affine ridge readout."""

    def __init__(self, task: str):
        super().__init__()
        self.capture_lock = RLock()
        self.task = task
        self.outer = KumoTabular(task=task, size="small", pretrained=False, device="cpu")
        backbone = self.outer.models[Task(task)]
        self.blocks = nn.ModuleList([copy.deepcopy(backbone.row_embedding.row_blocks[-1]) for _ in range(4)])
        self.norm = copy.deepcopy(backbone.row_embedding.norm)

    @torch.inference_mode()
    def embed(self, xs, ys, xq, cat, classes):
        """Capture final row-cell inputs and final ICL tokens from the same native forward."""
        backbone = self.outer.models[Task(self.task)]
        device = next(backbone.parameters()).device
        columns = [f"x{i}" for i in range(xs.shape[1])]
        stypes = {name: "categorical" if cat[i] else "numerical" for i, name in enumerate(columns)}
        frames = [pd.DataFrame(matrix, columns=columns) for matrix in (xs, xq)]
        for j in np.flatnonzero(cat):
            domain = np.unique(xs[np.isfinite(xs[:, j]), j])
            for frame in frames:
                frame[columns[j]] = pd.Categorical(frame[columns[j]], categories=domain)
        xx, qq = [
            TableTensor.from_pandas(frame, stypes=stypes).to(device=device, dtype=torch.float32) for frame in frames
        ]
        target = pd.Categorical(ys, categories=np.arange(classes)) if classes else ys
        yy = TableTensor.from_pandas(
            pd.DataFrame({"y": target}), stypes={"y": "categorical" if classes else "numerical"}
        )
        captured = {}

        def row_hook(_module, _args, kwargs):
            if "cells" in captured:
                raise ValueError("CellRidge requires a single native estimator pass")
            captured["cells"] = kwargs["key_value"][..., 4:, :].clone()

        def support_hook(_module, _args, kwargs):
            captured["kv"] = kwargs["key_value"].clone()

        def query_hook(_module, _args, result):
            captured["query"] = result.clone()

        with self.capture_lock:
            handles = [
                backbone.row_embedding.row_blocks[-1].register_forward_pre_hook(row_hook, with_kwargs=True),
                backbone.icl_block.layers[-1].register_forward_pre_hook(support_hook, with_kwargs=True),
                backbone.icl_block.norm.register_forward_hook(query_hook),
            ]
            try:
                self.outer(
                    x_context=xx,
                    y_context=yy.to(device=device, dtype=torch.float32),
                    x_query=qq,
                    num_estimators=1,
                    estimator_batch_size=1,
                    generator=torch.Generator(device=device).manual_seed(104),
                )
            finally:
                for handle in handles:
                    handle.remove()
            kv = captured.pop("kv")
            support = backbone.icl_block.norm(backbone.icl_block.layers[-1](query=kv.clone(), key_value=kv))
        rows = len(xs) + len(xq)
        units = torch.cat((support, captured["query"]), dim=-2).reshape(rows, 4, 128)
        cells = captured["cells"].reshape(rows, -1, 128)
        # Each tail block acts on the cells of one row.
        chunks = []
        for rows_i in range(0, rows, 64):
            sl = slice(rows_i, rows_i + 64)
            features = torch.cat((units[sl], cells[sl]), dim=1)
            for block in self.blocks:
                features = block(query=features, key_value=features)
            chunks.append(self.norm(features[:, 4:]).flatten(1))
        return torch.cat(chunks)

    @torch.inference_mode()
    def scores(self, xs, ys, xq, cat, classes):
        features = self.embed(xs, ys, xq, cat, classes)
        support, query = features.split((len(xs), len(xq)))
        labels = torch.as_tensor(ys, device=support.device)
        targets = F.one_hot(labels.long(), classes).float() if classes else labels.float()[:, None]
        feature_mean = support.mean(0)
        target_mean = targets.mean(0)
        a = support - feature_mean
        q = query - feature_mean
        y = targets - target_mean
        alpha = 0.01 * (a.square().sum() / a.shape[1]).clamp_min(1e-6)
        if a.shape[1] <= len(a):
            eye = torch.eye(a.shape[1], device=a.device, dtype=a.dtype)
            aug = torch.cat((a, alpha.sqrt() * eye))
            ortho, r = torch.linalg.qr(aug, mode="reduced")
            rhs = torch.cat((y, y.new_zeros(a.shape[1], y.shape[1])))
            pred = q @ torch.linalg.solve_triangular(r, ortho.T @ rhs, upper=True) + target_mean
        else:
            eye = torch.eye(len(a), device=a.device, dtype=a.dtype)
            _, r = torch.linalg.qr(torch.cat((a.T, alpha.sqrt() * eye)), mode="reduced")
            dual = torch.linalg.solve_triangular(r, torch.linalg.solve_triangular(r.T, y, upper=False), upper=True)
            pred = (q @ a.T) @ dual + target_mean
        if not torch.isfinite(pred).all():
            raise ValueError("Nonfinite ridge predictions")
        return pred


def load_network(task: str, device: str):
    from safetensors.torch import load_file

    state = load_file(str(checkpoint_path(task)))
    with torch.random.fork_rng(devices=[]):
        m = CellRidgeNetwork(task)
    backbone = m.outer.models[Task(task)]
    backbone.load_state_dict(
        {k.removeprefix("native."): v for k, v in state.items() if k.startswith("native.")}, strict=True
    )
    m.blocks.load_state_dict(
        {k.removeprefix("tail.blocks."): v for k, v in state.items() if k.startswith("tail.blocks.")}, strict=True
    )
    m.norm.load_state_dict(
        {k.removeprefix("tail.norm."): v for k, v in state.items() if k.startswith("tail.norm.")}, strict=True
    )
    return m.eval().requires_grad_(False).to(device)
