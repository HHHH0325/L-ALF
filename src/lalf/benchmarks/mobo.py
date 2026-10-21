"""Historical finite-library MOBO selection kernels, isolated from research I/O."""

from typing import Callable
import gc
import math
import numpy as np
import torch
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize as OutcomeStandardize
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

OUTPUT_NOISES = (1e-4, 1e-3, 1e-4, 1e-4, 1e-4)
SCIPY_FIT_DISABLED = False


def build_model(train_x: torch.Tensor, train_y: torch.Tensor) -> ModelListGP:
    models = []
    for output_idx in range(train_y.shape[-1]):
        target = train_y[:, output_idx : output_idx + 1]
        noise_level = OUTPUT_NOISES[output_idx] if output_idx < len(OUTPUT_NOISES) else 1e-4
        target_var = torch.full_like(target, noise_level)
        covar_module = ScaleKernel(MaternKernel(nu=2.5))
        models.append(
            SingleTaskGP(
                train_X=train_x,
                train_Y=target,
                train_Yvar=target_var,
                covar_module=covar_module,
                outcome_transform=OutcomeStandardize(m=1),
            )
        )
    model = ModelListGP(*models)
    return model.to(device=train_x.device, dtype=train_x.dtype)


def fit_model(model: ModelListGP, maxiter: int) -> None:
    global SCIPY_FIT_DISABLED
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    if not SCIPY_FIT_DISABLED:
        try:
            fit_gpytorch_mll(
                mll,
                optimizer_kwargs={"options": {"maxiter": maxiter}},
            )
            return
        except Exception as exc:
            SCIPY_FIT_DISABLED = True
            print(f"[Fit] scipy optimizer failed; switching this run to torch optimizer: {exc}")

    fit_gpytorch_mll_torch(
        mll,
        step_limit=max(200, maxiter * 5),
    )


def get_constraints() -> list[Callable[[torch.Tensor], torch.Tensor]]:
    return [
        lambda samples: samples[..., 2],
        lambda samples: samples[..., 3],
        lambda samples: samples[..., 4],
    ]


def select_objective() -> IdentityMCMultiOutputObjective:
    return IdentityMCMultiOutputObjective(outcomes=[0, 1])


def sample_scalarization_weights(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    concentration = torch.ones(2, device=device, dtype=dtype)
    return torch.distributions.Dirichlet(concentration).sample()


def locate_choice_index(choices: torch.Tensor, chosen_x: torch.Tensor) -> int:
    chosen_flat = chosen_x.reshape(1, -1)
    exact = torch.nonzero((choices == chosen_flat).all(dim=1), as_tuple=False)
    if len(exact):
        return int(exact[0].item())
    close = torch.nonzero(
        torch.isclose(choices, chosen_flat, atol=1e-8, rtol=1e-6).all(dim=1), as_tuple=False
    )
    if len(close):
        return int(close[0].item())
    raise RuntimeError("Failed to map optimized candidate back to the discrete choice set.")


def posterior_summary(model: ModelListGP, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        posterior = model.posterior(x)
        mean = posterior.mean.detach().cpu().numpy().reshape(-1)
        std = posterior.variance.clamp_min(0).sqrt().detach().cpu().numpy().reshape(-1)
    return mean.astype(np.float64), std.astype(np.float64)


def standard_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def prescreen_candidates(
    model: ModelListGP,
    pool_x: torch.Tensor,
    available_indices: list[int],
    topk: int,
    beta: float,
    chunk_size: int,
) -> tuple[list[int], dict[int, dict], dict[str, float]]:
    model.eval()
    meta_rows: list[dict] = []

    with torch.no_grad():
        for start in range(0, len(available_indices), chunk_size):
            idx_chunk = available_indices[start : start + chunk_size]
            x_chunk = pool_x[idx_chunk]
            posterior = model.posterior(x_chunk)
            mean = posterior.mean
            std = posterior.variance.clamp_min(1e-8).sqrt()

            feasible_prob = torch.ones(len(idx_chunk), device=mean.device, dtype=mean.dtype)
            for out_idx in (2, 3, 4):
                z = (-mean[:, out_idx]) / std[:, out_idx].clamp_min(1e-6)
                feasible_prob = feasible_prob * standard_normal_cdf(z)

            ucb_logf = (mean[:, 0] + beta * std[:, 0]).clamp_min(0.0)
            ucb_logratio = (mean[:, 1] + beta * std[:, 1]).clamp_min(0.0)
            prescore = feasible_prob * (ucb_logf + ucb_logratio)

            rows = (
                torch.stack([prescore, feasible_prob, ucb_logf, ucb_logratio], dim=1).cpu().numpy()
            )
            for idx_val, row in zip(idx_chunk, rows):
                meta_rows.append(
                    {
                        "global_idx": int(idx_val),
                        "pre_score": float(row[0]),
                        "feasible_prob": float(row[1]),
                        "ucb_logf": float(row[2]),
                        "ucb_logratio": float(row[3]),
                    }
                )

    meta_rows = [row for row in meta_rows if np.isfinite(row["pre_score"])]
    meta_rows.sort(key=lambda row: row["pre_score"], reverse=True)
    top_rows = meta_rows[: min(topk, len(meta_rows))]

    ordered_indices = [row["global_idx"] for row in top_rows]
    meta = {}
    for rank, row in enumerate(top_rows, start=1):
        meta[row["global_idx"]] = {
            "prescreen_rank": rank,
            "pre_score": row["pre_score"],
            "feasible_prob": row["feasible_prob"],
            "ucb_logf": row["ucb_logf"],
            "ucb_logratio": row["ucb_logratio"],
        }

    summary = {
        "pool_size": float(len(available_indices)),
        "prescreen_size": float(len(top_rows)),
        "best_pre_score": float(top_rows[0]["pre_score"]) if top_rows else float("nan"),
        "worst_pre_score": float(top_rows[-1]["pre_score"]) if top_rows else float("nan"),
        "mean_feasible_prob_topk": float(np.mean([row["feasible_prob"] for row in top_rows]))
        if top_rows
        else float("nan"),
    }
    return ordered_indices, meta, summary


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sample_qpots_union(
    model,
    candidate_x: torch.Tensor,
    ordered_candidate_indices: list[int],
    candidate_meta: dict[int, dict],
    posterior_draws: int,
    pool_limit: int,
) -> tuple[list[int], dict[int, dict], dict[str, float]]:
    candidate_stats: dict[int, dict] = {}
    feasible_counts: list[int] = []
    pareto_sizes: list[int] = []

    with torch.no_grad():
        posterior = model.posterior(candidate_x)
        samples = posterior.sample(sample_shape=torch.Size([posterior_draws]))

    for draw_idx in range(posterior_draws):
        draw = samples[draw_idx]
        feasible_mask = (draw[:, 2] <= 0) & (draw[:, 3] <= 0) & (draw[:, 4] <= 0)
        feasible_local = torch.nonzero(feasible_mask, as_tuple=False).reshape(-1)
        feasible_counts.append(int(feasible_local.numel()))
        if feasible_local.numel() == 0:
            pareto_sizes.append(0)
            continue

        feasible_obj = draw[feasible_local, :2]
        pareto_mask = is_non_dominated(feasible_obj, maximize=True)
        pareto_local = feasible_local[pareto_mask]
        pareto_sizes.append(int(pareto_local.numel()))

        for local_idx in pareto_local.tolist():
            global_idx = int(ordered_candidate_indices[local_idx])
            sample_obj_sum = float(draw[local_idx, 0].item() + draw[local_idx, 1].item())
            sample_obj1 = float(draw[local_idx, 0].item())
            sample_obj2 = float(draw[local_idx, 1].item())
            stat = candidate_stats.setdefault(
                global_idx,
                {
                    "pareto_draw_count": 0,
                    "best_draw_obj_sum": float("-inf"),
                    "best_draw_obj1": float("-inf"),
                    "best_draw_obj2": float("-inf"),
                },
            )
            stat["pareto_draw_count"] += 1
            stat["best_draw_obj_sum"] = max(stat["best_draw_obj_sum"], sample_obj_sum)
            stat["best_draw_obj1"] = max(stat["best_draw_obj1"], sample_obj1)
            stat["best_draw_obj2"] = max(stat["best_draw_obj2"], sample_obj2)

    del posterior, samples
    clear_cuda()

    ordered_union = sorted(
        candidate_stats.keys(),
        key=lambda idx: (
            -int(candidate_stats[idx]["pareto_draw_count"]),
            -float(candidate_stats[idx]["best_draw_obj_sum"]),
            -float(candidate_meta.get(idx, {}).get("pre_score", float("-inf"))),
            int(candidate_meta.get(idx, {}).get("prescreen_rank", 10**9)),
        ),
    )
    limited_union = ordered_union[: min(len(ordered_union), pool_limit)]

    summary = {
        "sampled_union_size": float(len(ordered_union)),
        "candidate_pool_size": float(len(limited_union)),
        "mean_feasible_count": float(np.mean(feasible_counts)) if feasible_counts else float("nan"),
        "mean_pareto_size": float(np.mean(pareto_sizes)) if pareto_sizes else float("nan"),
        "max_pareto_draw_count": float(
            max((stats["pareto_draw_count"] for stats in candidate_stats.values()), default=0)
        ),
    }
    return limited_union, candidate_stats, summary


def _maximin_pick(
    candidate_ids: list[int],
    reference_x: torch.Tensor,
    pool_x: torch.Tensor,
    candidate_meta: dict[int, dict],
    candidate_stats: dict[int, dict],
) -> tuple[int, float]:
    candidate_x = pool_x[candidate_ids]
    distances = torch.cdist(candidate_x, reference_x)
    min_dist = distances.min(dim=1).values

    best_pos = max(
        range(len(candidate_ids)),
        key=lambda pos: (
            float(min_dist[pos].item()),
            int(candidate_stats.get(candidate_ids[pos], {}).get("pareto_draw_count", 0)),
            float(candidate_meta.get(candidate_ids[pos], {}).get("pre_score", float("-inf"))),
            -int(candidate_meta.get(candidate_ids[pos], {}).get("prescreen_rank", 10**9)),
        ),
    )
    return int(candidate_ids[best_pos]), float(min_dist[best_pos].item())


def select_batch_qpots(
    model,
    train_x: torch.Tensor,
    pool_x: torch.Tensor,
    ordered_candidate_indices: list[int],
    candidate_meta: dict[int, dict],
    batch_size: int,
    posterior_draws: int,
    qpots_candidate_multiplier: int,
) -> tuple[list[int], list[dict], dict[str, float]]:
    candidate_x = pool_x[ordered_candidate_indices]
    candidate_pool_limit = max(batch_size * qpots_candidate_multiplier, batch_size)
    union_ids, candidate_stats, qpots_summary = _sample_qpots_union(
        model=model,
        candidate_x=candidate_x,
        ordered_candidate_indices=ordered_candidate_indices,
        candidate_meta=candidate_meta,
        posterior_draws=posterior_draws,
        pool_limit=candidate_pool_limit,
    )

    selected_indices: list[int] = []
    trace_rows: list[dict] = []
    selected_set: set[int] = set()
    fill_count = 0

    for step in range(batch_size):
        available_union = [idx for idx in union_ids if idx not in selected_set]
        reference_ids = selected_indices
        reference_x = (
            train_x if not reference_ids else torch.cat([train_x, pool_x[reference_ids]], dim=0)
        )

        if available_union:
            chosen_idx, distance_score = _maximin_pick(
                candidate_ids=available_union,
                reference_x=reference_x,
                pool_x=pool_x,
                candidate_meta=candidate_meta,
                candidate_stats=candidate_stats,
            )
            selection_reason = "qpots"
        else:
            fill_candidates = [idx for idx in ordered_candidate_indices if idx not in selected_set]
            if not fill_candidates:
                break
            chosen_idx, distance_score = _maximin_pick(
                candidate_ids=fill_candidates,
                reference_x=reference_x,
                pool_x=pool_x,
                candidate_meta=candidate_meta,
                candidate_stats=candidate_stats,
            )
            selection_reason = "qpots_fill_prescreen"
            fill_count += 1

        chosen_x = pool_x[chosen_idx : chosen_idx + 1]
        pred_mean, pred_std = posterior_summary(model, chosen_x)
        meta = candidate_meta.get(chosen_idx, {})
        stats = candidate_stats.get(
            chosen_idx,
            {
                "pareto_draw_count": 0,
                "best_draw_obj_sum": np.nan,
                "best_draw_obj1": np.nan,
                "best_draw_obj2": np.nan,
            },
        )
        trace_rows.append(
            {
                "candidate_rank": step + 1,
                "selection_reason": selection_reason,
                "distance_score": distance_score,
                "sampled_union_size": int(qpots_summary["sampled_union_size"]),
                "candidate_pool_size": int(qpots_summary["candidate_pool_size"]),
                "pareto_draw_count": int(stats["pareto_draw_count"]),
                "best_draw_obj_sum": float(stats["best_draw_obj_sum"]),
                "best_draw_obj1": float(stats["best_draw_obj1"]),
                "best_draw_obj2": float(stats["best_draw_obj2"]),
                "prescreen_rank": int(meta.get("prescreen_rank", -1)),
                "pre_score": float(meta.get("pre_score", np.nan)),
                "feasible_prob": float(meta.get("feasible_prob", np.nan)),
                "ucb_logf": float(meta.get("ucb_logf", np.nan)),
                "ucb_logratio": float(meta.get("ucb_logratio", np.nan)),
                "pred_log1p_f_mean": pred_mean[0],
                "pred_log1p_ratio_mean": pred_mean[1],
                "pred_c_est_mean": pred_mean[2],
                "pred_c_s1_low_mean": pred_mean[3],
                "pred_c_s1_high_mean": pred_mean[4],
                "pred_log1p_f_std": pred_std[0],
                "pred_log1p_ratio_std": pred_std[1],
                "pred_c_est_std": pred_std[2],
                "pred_c_s1_low_std": pred_std[3],
                "pred_c_s1_high_std": pred_std[4],
            }
        )
        selected_indices.append(chosen_idx)
        selected_set.add(chosen_idx)

    qpots_summary["fill_count"] = float(fill_count)
    return selected_indices, trace_rows, qpots_summary
