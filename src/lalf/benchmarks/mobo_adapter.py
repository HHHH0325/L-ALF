"""Fit only queried labels and call the retained MOBO selection kernels."""

from __future__ import annotations

import numpy as np

from .core import COLS


def compute_targets(history):
    y = history[COLS].to_numpy(float)
    return np.column_stack(
        [
            np.log1p(np.maximum(y[:, 2], 0)),
            np.log1p(np.maximum(y[:, 3], 0)),
            y[:, 1] - np.nextafter(0.30, -np.inf),
            1.3 - y[:, 0],
            y[:, 0] - 3.2,
        ]
    )


def select_mobo(method, lib, history, remaining, seed, batch_size=20, maxiter=20, device="cpu"):
    import torch
    from botorch.acquisition.multi_objective.parego import qLogNParEGO
    from botorch.optim import optimize_acqf_discrete
    from botorch.sampling.normal import SobolQMCNormalSampler
    from . import mobo

    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if selected_device == "auto":
        selected_device = "cpu"
    setup = dict(device=torch.device(selected_device), dtype=torch.float64)
    x = torch.as_tensor(lib.x, **setup)
    train_x = x[history.id.to_numpy(int)]
    train_y = torch.as_tensor(compute_targets(history), **setup)
    # Isolate seeds from the caller's torch random state.
    devices = [x.device.index or 0] if x.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        model = mobo.build_model(train_x, train_y)
        mobo.fit_model(model, maxiter)
        model.eval()
        ids, meta, _ = mobo.prescreen_candidates(model, x, list(remaining), 500, 0.5, 1024)
        if method == "qpots":
            selected, _, _ = mobo.select_batch_qpots(
                model,
                train_x,
                x,
                ids,
                meta,
                batch_size,
                posterior_draws=8,
                qpots_candidate_multiplier=4,
            )
        elif method == "qlognparego":
            remaining_ids = list(ids)
            selected = []
            pending = None
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([8]), seed=seed)
            for _ in range(batch_size):
                acq = qLogNParEGO(
                    model=model,
                    X_baseline=train_x,
                    scalarization_weights=mobo.sample_scalarization_weights(**setup),
                    sampler=sampler,
                    objective=mobo.select_objective(),
                    constraints=mobo.get_constraints(),
                    X_pending=pending,
                    prune_baseline=True,
                    cache_root=False,
                )
                choices = x[remaining_ids]
                point, _ = optimize_acqf_discrete(
                    acq,
                    q=1,
                    choices=choices,
                    max_batch_size=64,
                    unique=True,
                )
                pos = mobo.locate_choice_index(choices, point)
                selected.append(remaining_ids.pop(pos))
                pending = x[selected]
        else:
            raise ValueError(f"Unknown MOBO method: {method}")
    return selected
