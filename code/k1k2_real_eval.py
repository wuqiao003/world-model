#!/usr/bin/env python3
"""
K1 + K2 on real action-conditioned world models (LeWM PushT-FR3 family).

Model zoo = 3 released checkpoints x 5 action pipelines (identity + 4 controlled
corruptions). Each (ckpt, pipeline) pair is one "system"; corruptions give the
spread needed for rank correlation between offline metrics and planning utility.

Offline metric families
  fidelity      : fact_err_h1 / fact_err_H / pred_id_ratio   (latent analogue of pixel metrics)
  sensitivity   : self_cf_gap                                (frozen context, flip current action)
  validity      : pv_nn        nearest-neighbour counterfactual oracle on real data
  composite     : caf = gap/pv, caf_gated, score_iv

Downstream planning utility (no simulator required)
  gt_rank_pct   : percentile rank of the expert action sequence under model cost
  cost_cv       : discriminativity of the cost surface
  mppi_cos      : cos(first MPPI action, expert action)  <- primary utility
  mppi_ratio    : |MPPI action| / |expert action|
  cl_progress   : closed-loop receding-horizon progress toward the goal, stepped
                  by a nearest-neighbour transition oracle built from recorded
                  end-effector proprioception (a state space no model touches)

Outputs one JSON with per-system metrics plus Spearman(metric, utility) with a
bootstrap CI over anchors.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.adapters.lewm_stub import BUNDLES, LeWMAdapter, find_bundle  # noqa: E402
from eval.metrics import save_json, spearman  # noqa: E402

EPS = 1e-8


# ----------------------------------------------------------------------------
# action pipelines: (name, fn) operating on normalized actions (..., BLOCK, 2)
# ----------------------------------------------------------------------------
def pipe_identity(a):
    return a


def pipe_zero(a):
    """Action-blind: always feed the dataset-mean action (0 in scaler space)."""
    return torch.zeros_like(a)


def pipe_neg(a):
    """Sign-flipped: full sensitivity, wrong semantics."""
    return -a


def pipe_swap(a):
    """dx <-> dy: preserves magnitude statistics, breaks direction."""
    return a.flip(-1)


def pipe_atten(a):
    return 0.25 * a


def make_rot(deg: float):
    """Rotate (dx, dy) by a fixed angle: graded semantic corruption, magnitude kept."""
    th = np.deg2rad(deg)
    c, s = float(np.cos(th)), float(np.sin(th))

    def f(a):
        x, y = a[..., 0], a[..., 1]
        return torch.stack([c * x - s * y, s * x + c * y], dim=-1)

    return f


def make_scale(k: float):
    def f(a):
        return k * a

    return f


def make_noise(sigma: float):
    """Additive noise in scaler space: graded loss of action information."""

    def f(a):
        return a + sigma * torch.randn_like(a)

    return f


PIPELINES = {
    "id": pipe_identity,
    "zero": pipe_zero,
    "neg": pipe_neg,
    "swap": pipe_swap,
    "atten": pipe_atten,
}
# Graded corruption family: needed for statistical power. 15 systems (3 ckpts x
# 5 pipelines) leaves the paired-bootstrap CI crossing zero; the graded family
# raises the zoo to ~50 systems without any new modelling assumption.
for _d in (15, 30, 45, 60, 90, 120, 150):
    PIPELINES[f"rot{_d}"] = make_rot(_d)
for _k in (0.1, 0.5, 0.75, 1.5, 2.0):
    PIPELINES[f"scale{_k}"] = make_scale(_k)
for _s in (0.25, 0.5, 1.0, 2.0):
    PIPELINES[f"noise{_s}"] = make_noise(_s)

ALL_PIPELINES = ",".join(PIPELINES)

# Model-side variants. The action-only zoo is confounded: every action
# corruption also degrades one-step fidelity, because the true next state
# depends on the true action. Perturbing predictor weights degrades dynamics
# quality *without* touching the action pipeline, so fidelity and
# controllability can move independently.
WEIGHT_NOISE_SIGMAS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
VARIANT_PIPELINES = "id,zero,neg"  # pipelines run for each perturbed variant


@torch.no_grad()
def perturb_predictor(adapter: LeWMAdapter, sigma: float, generator: torch.Generator) -> None:
    """In-place N(0, sigma*std(W)) noise on predictor weights."""
    for p in adapter.model.predictor.parameters():
        if p.numel() < 2:
            continue
        s = float(p.detach().float().std())
        if s == 0.0:
            continue
        noise = torch.randn(p.shape, generator=generator, device=p.device, dtype=p.dtype)
        p.add_(sigma * s * noise)


# ----------------------------------------------------------------------------
def _load_one(path: Path):
    import h5py

    with h5py.File(path, "r") as f:
        if "pixels_jpeg" in f:
            import io

            from PIL import Image

            enc = f["pixels_jpeg"][:]
            pixels = np.stack(
                [np.asarray(Image.open(io.BytesIO(e.tobytes()))) for e in enc]
            ).astype(np.uint8)
        else:
            pixels = f["pixels"][:]                  # (M,224,224,3) uint8
        order = str(f.attrs.get("channel_order", "BGR"))
        actions = f["actions"][:][..., :2]           # (A,H,BLOCK,2) raw meters
        chain = f["chain"][:]                        # (A,H+1) idx into pixels
        block = int(f.attrs["block"])
        horizon = int(f.attrs["horizon"])
    if order.upper() == "RGB":
        pixels = np.ascontiguousarray(pixels[..., ::-1])
    return pixels, actions.astype(np.float32), chain, block, horizon


def _shard_paths(path: Path):
    spec = str(path)
    if any(ch in spec for ch in "*?"):
        files = sorted(Path().glob(spec))
        if not files:
            raise FileNotFoundError(f"no shards match {spec}")
        return files
    return [Path(spec)]


def load_proprio(path: Path):
    """
    Per-frame end-effector proprioception (M,7), or None if the subset lacks it.

    Kept separate from load_subset so the closed-loop utility can use a state
    space that no model under test participates in.
    """
    import h5py

    out = []
    for f in _shard_paths(path):
        with h5py.File(f, "r") as h:
            if "proprio" not in h:
                return None
            out.append(h["proprio"][:].astype(np.float32))
    return np.concatenate(out, 0)


def load_subset(path: Path):
    """
    Load one subset file or a glob of shards, concatenating frames and anchors.

    Frames are returned in the channel order the LeWM encoders were trained on:
    the upstream dataset stores OpenCV BGR and the released inference code feeds
    it unconverted, so BGR is the in-distribution order. Feeding RGB collapses
    the encoder (pred/id degrades by ~500x) -- see code/diag_lewm.py.
    """
    files = _shard_paths(path)
    px_all, act_all, chain_all = [], [], []
    offset, block, horizon = 0, None, None
    for f in files:
        px, act, ch, blk, hor = _load_one(f)
        if block is not None and (blk, hor) != (block, horizon):
            raise ValueError(f"shard {f} has block/horizon {(blk, hor)} != {(block, horizon)}")
        block, horizon = blk, hor
        px_all.append(px)
        act_all.append(act)
        chain_all.append(ch + offset)
        offset += len(px)
    if len(files) > 1:
        print(f"loaded {len(files)} shards: {offset} frames", flush=True)
    return (
        np.concatenate(px_all, 0),
        np.concatenate(act_all, 0),
        np.concatenate(chain_all, 0),
        block,
        horizon,
    )


@torch.no_grad()
def encode_all(adapter: LeWMAdapter, pixels: np.ndarray, batch: int = 64) -> torch.Tensor:
    outs = []
    for i in range(0, len(pixels), batch):
        chunk = torch.from_numpy(pixels[i : i + batch]).unsqueeze(1)  # (b,1,H,W,3)
        px = adapter.preprocess_rgb(chunk)
        outs.append(adapter.encode_pixels(px)[:, 0])
    return torch.cat(outs, 0)


@torch.no_grad()
def rollout(adapter: LeWMAdapter, z0: torch.Tensor, acts_norm: torch.Tensor) -> torch.Tensor:
    """z0 (B,D) -> predicted latents (B,H,D) for normalized acts (B,H,BLOCK,2)."""
    return adapter.predict(z0.unsqueeze(1), acts_norm, normalized=True)


@torch.no_grad()
def nn_counterfactual_pairs(z0: torch.Tensor, a1: torch.Tensor, sep_quantile: float = 0.6):
    """
    Build a counterfactual oracle from real data alone.

    For anchor i pick j minimising the latent distance ||z0_i - z0_j|| subject to
    the first-plan-step actions being far apart (above `sep_quantile` of the
    pairwise action-distance distribution, so the threshold is scale-free). Then
    (a_j applied at z0_i) has approximate real outcome z1_j, and the state
    mismatch ||z0_i - z0_j||^2 is an explicit noise floor for that oracle.
    """
    n = z0.size(0)
    d_state = torch.cdist(z0, z0).pow(2) / z0.size(-1)       # mean-squared latent gap
    af = a1.reshape(n, -1)
    d_act = torch.cdist(af, af)
    off = ~torch.eye(n, dtype=torch.bool, device=z0.device)
    thr = torch.quantile(d_act[off], sep_quantile)
    mask = (d_act < thr) | (~off)
    d_state = d_state.masked_fill(mask, float("inf"))
    j = d_state.argmin(dim=1)
    gap = d_state.gather(1, j.unsqueeze(1)).squeeze(1)
    return j, torch.isfinite(gap), gap


@torch.no_grad()
def mppi_first_action(
    adapter: LeWMAdapter,
    z0: torch.Tensor,
    z_goal: torch.Tensor,
    pipeline,
    horizon: int,
    block: int,
    k: int = 128,
    iters: int = 10,
    beta: float = 3.0,
    var: float = 1.0,
    return_block: bool = False,
) -> torch.Tensor:
    """
    Vanilla MPPI in normalized action space.

    Returns the mean first-tick action (2,), or the whole first plan-step
    (block, 2) when `return_block` -- the closed-loop utility needs the net
    displacement over the block, not just its first tick.
    """
    dev = z0.device
    mean = torch.zeros(horizon, block, 2, device=dev)
    std = torch.full_like(mean, var)
    z0k = z0.unsqueeze(0).expand(k, -1)
    for _ in range(iters):
        cand = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
            k, horizon, block, 2, device=dev
        )
        pred = rollout(adapter, z0k, pipeline(cand))
        cost = (pred[:, -1] - z_goal.unsqueeze(0)).pow(2).mean(-1)
        w = F.softmax(-beta * cost, dim=0)
        mean = (w.view(-1, 1, 1, 1) * cand).sum(0)
    return mean[0] if return_block else mean[0, 0]


def rank_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman between two 1-D tensors; nan when either side is constant."""
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = ra.norm() * rb.norm()
    return float(ra.dot(rb) / den) if float(den) > 0 else float("nan")


class ProprioNNSim:
    """
    Transition oracle built from the dataset, in a space no model touches.

    The two utility axes we have so far (`mppi_cos`, `gt_rank_pct`) both score a
    model against a cost surface, so a reviewer can object that they measure
    agreement rather than control. A closed loop needs a simulator, and we have
    no PushT-FR3 simulator -- but the dataset itself supplies transitions.

    State is end-effector proprioception, which is recorded by the robot and is
    independent of every model under test, so the simulator cannot favour the
    model whose latent space it was built from. `step` transfers the *observed
    displacement* of the nearest matching transition rather than teleporting to
    that transition's successor, which keeps the rollout anchored at the queried
    state and only borrows local dynamics.
    """

    def __init__(self, proprio, chain, actions_raw, w_act: float = 1.0):
        # PushT-FR3 is planar pushing: ee_z and the orientation quaternion are
        # near-constant, so the planar end-effector position is the state.
        self.p_frame = proprio[:, :2]
        self.P = self.p_frame[chain[:, 0]]
        self.A = actions_raw[:, 0].sum(1)             # net raw dxy of plan-step 0
        self.D = self.p_frame[chain[:, 1]] - self.P   # observed displacement
        self.s_p = float(self.P.std(0).mean()) + EPS
        self.s_a = float(self.A.std(0).mean()) + EPS
        self.w_act = w_act

    def step(self, p, a):
        d = (
            (self.P - p).norm(dim=-1) / self.s_p
            + self.w_act * (self.A - a).norm(dim=-1) / self.s_a
        )
        k = int(d.argmin())
        return p + self.D[k], float(d[k])

    def step_batch(self, P, A):
        """`step` over a batch of (state, action) pairs: P,A are (B,2)."""
        d = (
            torch.cdist(P, self.P) / self.s_p
            + self.w_act * torch.cdist(A, self.A) / self.s_a
        )
        k = d.argmin(dim=1)
        return P + self.D[k], d.gather(1, k[:, None]).squeeze(1)

    def rollout_cost(self, p0, plans, p_goal):
        """
        True terminal cost of each candidate plan, from one start state.

        plans is (K, horizon, 2) of net raw displacements per plan step. Returns
        (K,) squared distance from the simulated terminal state to the goal.
        """
        P = p0.unsqueeze(0).expand(plans.size(0), -1).clone()
        for h in range(plans.size(1)):
            P, _ = self.step_batch(P, plans[:, h])
        return (P - p_goal.unsqueeze(0)).pow(2).mean(-1)


@torch.no_grad()
def eval_system(
    adapter: LeWMAdapter,
    pipe_name: str,
    Z: torch.Tensor,
    chain: np.ndarray,
    acts_norm: torch.Tensor,
    horizon: int,
    block: int,
    n_plan: int,
    mppi_k: int,
    mppi_iters: int,
    sim: Optional["ProprioNNSim"] = None,
    n_closed: int = 24,
) -> Dict[str, object]:
    pipeline = PIPELINES[pipe_name]
    dev = Z.device
    ch = torch.from_numpy(chain).to(dev)
    z0 = Z[ch[:, 0]]                                     # (A,D)
    z_tgt = Z[ch[:, 1:]]                                 # (A,H,D)
    A = z0.size(0)

    # ---- fidelity -----------------------------------------------------------
    pred = rollout(adapter, z0, pipeline(acts_norm))     # (A,H,D)
    per_anchor_fact1 = (pred[:, 0] - z_tgt[:, 0]).pow(2).mean(-1)
    per_anchor_factH = (pred[:, -1] - z_tgt[:, -1]).pow(2).mean(-1)
    id_err1 = (z0 - z_tgt[:, 0]).pow(2).mean(-1)
    id_errH = (z0 - z_tgt[:, -1]).pow(2).mean(-1)

    # ---- sensitivity: frozen context, flip current plan-step only ----------
    acts_cf = acts_norm.clone()
    acts_cf[:, 0] = -acts_cf[:, 0]
    pred_cf = rollout(adapter, z0, pipeline(acts_cf))
    per_anchor_gap = (pred[:, 0] - pred_cf[:, 0]).pow(2).mean(-1)

    # ---- validity: nearest-neighbour counterfactual oracle -----------------
    j, valid, state_gap = nn_counterfactual_pairs(z0, acts_norm[:, 0])
    acts_swapped = acts_norm.clone()
    acts_swapped[:, 0] = acts_norm[j, 0]
    pred_nn = rollout(adapter, z0, pipeline(acts_swapped))
    oracle_nn = Z[ch[j, 1]]
    per_anchor_pv = (pred_nn[:, 0] - oracle_nn).pow(2).mean(-1)

    # Magnitude alone under-penalises a sign-flipped model: its counterfactual
    # MSE grows only mildly while its direction is maximally wrong. Alignment of
    # the raw displacement is also diluted, because most of the displacement is
    # action-independent drift shared by every action.
    cf_align = F.cosine_similarity(pred_nn[:, 0] - z0, oracle_nn - z0, dim=-1)
    fact_align = F.cosine_similarity(pred[:, 0] - z0, z_tgt[:, 0] - z0, dim=-1)

    # Counterfactual action-contrast alignment: differencing both the prediction
    # and the oracle over the *same* pair of actions cancels the common drift and
    # leaves only the action-attributable component. A sign-flipped model then
    # scores strongly negative instead of mildly positive.
    d_pred = pred_nn[:, 0] - pred[:, 0]              # switching a_i -> a_j
    d_oracle = oracle_nn - z_tgt[:, 0]               # real outcome of that switch
    caca = F.cosine_similarity(d_pred, d_oracle, dim=-1)
    degenerate = d_pred.norm(dim=-1) < 1e-6 * (z0.norm(dim=-1) + EPS)
    caca = caca.masked_fill(degenerate, 0.0)         # action-blind -> no credit

    # ---- downstream planning ----------------------------------------------
    sel = torch.linspace(0, A - 1, min(n_plan, A)).long().to(dev)
    ranks, ranks_id, cvs, coss, ratios, pccs = [], [], [], [], [], []
    for i in sel.tolist():
        zi, zg = z0[i], z_tgt[i, -1]
        cand = torch.randn(mppi_k, horizon, block, 2, device=dev)
        expert = acts_norm[i].unsqueeze(0)
        allc = torch.cat([cand, expert], 0)
        p = rollout(adapter, zi.unsqueeze(0).expand(allc.size(0), -1), pipeline(allc))
        cost = (p[:, -1] - zg.unsqueeze(0)).pow(2).mean(-1)
        c_rand, c_exp = cost[:-1], cost[-1]
        # tie-corrected rank: a degenerate (action-blind) cost surface must score
        # 0.5, not 0.0, or action-blind systems look perfect.
        tol = 1e-6 * (c_rand.abs().mean() + EPS)
        better = (c_rand < c_exp - tol).float().mean()
        tied = ((c_rand - c_exp).abs() <= tol).float().mean()
        ranks.append(float(better + 0.5 * tied))
        cvs.append(float(c_rand.std() / (c_rand.mean() + EPS)))

        # Same question, in-distribution distractors: other anchors' *real*
        # action sequences instead of Gaussian noise. Random plans are off the
        # action manifold, so a model can rank the expert first simply by
        # blowing up on inputs it never saw -- that is novelty rejection, not
        # planning. The gap between the two ranks separates the two abilities.
        other = torch.randint(0, A, (mppi_k,), device=dev)
        other = torch.where(other == i, (other + 1) % A, other)
        alld = torch.cat([acts_norm[other], expert], 0)
        pd = rollout(adapter, zi.unsqueeze(0).expand(alld.size(0), -1), pipeline(alld))
        cd = (pd[:, -1] - zg.unsqueeze(0)).pow(2).mean(-1)
        d_rand, d_exp = cd[:-1], cd[-1]
        dtol = 1e-6 * (d_rand.abs().mean() + EPS)
        ranks_id.append(float((d_rand < d_exp - dtol).float().mean()
                              + 0.5 * ((d_rand - d_exp).abs() <= dtol).float().mean()))

        # Plan-cost correlation: does the model order *these* candidate plans the
        # way the world does? Unlike recovering a first action from a terminal
        # goal, this target is identified -- each plan has one true cost -- and it
        # yields mppi_k paired observations per anchor instead of one vector, so
        # it is the axis that survives the discriminability check in
        # code/utility_oracle.py. The model rollout is already paid for above.
        if sim is not None:
            raw = (cand * adapter.a_std + adapter.a_mean).sum(2)   # (K,H,2) metres
            true_cost = sim.rollout_cost(sim.p_frame[ch[i, 0]], raw,
                                         sim.p_frame[ch[i, -1]])
            pccs.append(rank_corr(c_rand, true_cost))

        a_mppi = mppi_first_action(
            adapter, zi, zg, pipeline, horizon, block, k=mppi_k, iters=mppi_iters
        )
        a_true = acts_norm[i, 0, 0]
        coss.append(float(F.cosine_similarity(a_mppi, a_true, dim=0)))
        ratios.append(float(a_mppi.norm() / (a_true.norm() + EPS)))

    # ---- closed-loop receding horizon (third utility axis) -----------------
    progs = []
    if sim is not None and n_closed > 0:
        cl_sel = sel[:: max(1, len(sel) // max(1, n_closed))][:n_closed]
        for i in cl_sel.tolist():
            p = sim.p_frame[ch[i, 0]].clone()
            p_goal = sim.p_frame[ch[i, -1]]
            d0 = float((p - p_goal).norm()) + EPS
            z, zg = z0[i], z_tgt[i, -1]
            for _ in range(horizon):
                blk = mppi_first_action(
                    adapter, z, zg, pipeline, horizon, block,
                    k=mppi_k, iters=max(2, mppi_iters // 2), return_block=True,
                )
                a_net = (blk * adapter.a_std + adapter.a_mean).sum(0)  # raw metres
                p, _ = sim.step(p, a_net)
                # No state estimator: the model must also carry its own belief
                # forward, so its drift is part of the closed-loop cost.
                z = rollout(adapter, z.unsqueeze(0), pipeline(blk.unsqueeze(0).unsqueeze(0)))[0, 0]
            progs.append(1.0 - float((p - p_goal).norm()) / d0)

    # Latent scale differs by an order of magnitude across checkpoints, so every
    # latent error is also reported relative to a scale-matched baseline:
    #   fidelity   -> the no-op (identity) predictor
    #   validity   -> the state-matching noise floor of the NN oracle
    id1 = float(id_err1.mean())
    nvalid = int(valid.sum())
    pv_raw = float(per_anchor_pv[valid].mean()) if nvalid else float("nan")
    sgap = float(state_gap[valid].mean()) if nvalid else float("nan")
    m = {
        "latent_rms": float(Z.pow(2).mean().sqrt()),
        "id_err_h1": id1,
        "fact_err_h1": float(per_anchor_fact1.mean()),
        "fact_err_hH": float(per_anchor_factH.mean()),
        "pred_id_ratio_h1": float(per_anchor_fact1.mean() / (id1 + EPS)),
        "pred_id_ratio_hH": float(per_anchor_factH.mean() / (id_errH.mean() + EPS)),
        "self_cf_gap": float(per_anchor_gap.mean()),
        "gap_ratio": float(per_anchor_gap.mean() / (id1 + EPS)),
        "pv_nn": pv_raw,
        "nn_state_gap": sgap,
        "pv_ratio": float(pv_raw / (sgap + EPS)) if nvalid else float("nan"),
        "cf_align": float(cf_align[valid].mean()) if nvalid else float("nan"),
        "caca": float(caca[valid].mean()) if nvalid else float("nan"),
        "fact_align": float(fact_align.mean()),
        "n_valid_nn": nvalid,
        "gt_rank_pct": float(np.mean(ranks)),
        "gt_rank_id_pct": float(np.mean(ranks_id)),
        # >0 means the expert plan only wins against off-manifold distractors
        "ood_credit": float(np.mean(ranks_id) - np.mean(ranks)),
        "cost_cv": float(np.mean(cvs)),
        "mppi_cos": float(np.mean(coss)),
        "mppi_ratio": float(np.mean(ratios)),
        "plan_cost_corr": float(np.nanmean(pccs)) if pccs else float("nan"),
        "n_pcc": len(pccs),
        "cl_progress": float(np.mean(progs)) if progs else float("nan"),
        "n_closed_loop": len(progs),
    }
    # scale-free CAF: relative sensitivity over relative counterfactual error
    m["caf"] = m["gap_ratio"] / (m["pv_ratio"] + EPS)
    m["caf_unnorm"] = m["self_cf_gap"] / (m["pv_nn"] + EPS)
    return {
        "metrics": m,
        "per_anchor": {
            "fact1": per_anchor_fact1.cpu().numpy(),
            "id1": id_err1.cpu().numpy(),
            "gap": per_anchor_gap.cpu().numpy(),
            "pv": per_anchor_pv.cpu().numpy(),
            "state_gap": state_gap.cpu().numpy(),
            "caca": caca.cpu().numpy(),
            "cf_align": cf_align.cpu().numpy(),
            "fact_align": fact_align.cpu().numpy(),
            "valid": valid.cpu().numpy(),
        },
        "per_plan": {
            "cos": np.array(coss), "rank": np.array(ranks),
            "rank_id": np.array(ranks_id), "cv": np.array(cvs),
            **({"pcc": np.array(pccs)} if pccs else {}),
        },
        "per_closed": {"progress": np.array(progs)},
    }


def add_composites(systems: Dict[str, Dict], tau: float = 3.0, beta: float = 0.3):
    """
    Composite scores over the zoo.

    `caf_soft` is the intended headline: bounded action sensitivity divided by
    counterfactual error, with no on-policy fidelity term. The fidelity-gated
    (`caf_gated`) and fidelity-weighted (`score_iv`) variants are kept as an
    ablation, because the released checkpoints dissociate one-step fidelity from
    planning utility (v4b is worst on pred/id yet best on published cost-surface
    CV), so a fidelity gate can demote the best planner.
    """
    best_ratio = min(s["metrics"]["pred_id_ratio_h1"] for s in systems.values())
    for s in systems.values():
        m = s["metrics"]
        sens = float(np.tanh(m["gap_ratio"]))
        m["caf_soft"] = float(sens / (1.0 + m["pv_ratio"]))
        # headline: sensitivity x counterfactual action-contrast correctness
        m["caf_caca"] = float(sens * 0.5 * (1.0 + m["caca"]))
        m["caf_caca_mag"] = float(m["caf_caca"] / (1.0 + m["pv_ratio"]))
        m["caf_align"] = float(sens * 0.5 * (1.0 + m["cf_align"]))
        m["caf_align_mag"] = float(m["caf_align"] / (1.0 + m["pv_ratio"]))
        gate = m["pred_id_ratio_h1"] <= best_ratio * tau
        m["caf_gated"] = m["caf"] if gate else 0.0
        m["score_iv"] = float(m["caf_soft"] * np.exp(-beta * m["pred_id_ratio_h1"]))
        m["dual"] = (-m["pv_ratio"]) if m["gap_ratio"] > 1e-3 else -1e3
        m["neg_fact_h1"] = -m["fact_err_h1"]
        m["neg_fact_hH"] = -m["fact_err_hH"]
        m["neg_pred_id_ratio_h1"] = -m["pred_id_ratio_h1"]
        m["neg_pred_id_ratio_hH"] = -m["pred_id_ratio_hH"]
        m["neg_pv"] = -m["pv_ratio"]
        m["neg_gt_rank"] = -m["gt_rank_pct"]
        m["neg_gt_rank_id"] = -m["gt_rank_id_pct"]


# Ours (controllability-aware) vs fidelity baselines.
OURS_KEYS = ["caf_caca", "caf_caca_mag", "caf_align", "caf_align_mag", "caf_soft", "caf", "dual"]
ABLATION_KEYS = ["caf_gated", "score_iv", "gap_ratio", "neg_pv", "cf_align", "caca"]
# fidelity family, including a *directional* fidelity baseline so the comparison
# is not just magnitude-vs-direction
FIDELITY_KEYS = [
    "neg_fact_h1", "neg_fact_hH",
    "neg_pred_id_ratio_h1", "neg_pred_id_ratio_hH",
    "fact_align",
]
METRIC_KEYS = OURS_KEYS + ABLATION_KEYS + ["cost_cv"] + FIDELITY_KEYS


def correlations(systems: Dict[str, Dict], utility_key: str) -> Dict[str, float]:
    names = list(systems)
    util = [systems[n]["metrics"][utility_key] for n in names]
    return {
        k: spearman([systems[n]["metrics"][k] for n in names], util) for k in METRIC_KEYS
    }


def bootstrap_corr(
    systems: Dict[str, Dict], utility_key: str, n_boot: int = 400, seed: int = 0
) -> Dict[str, Dict[str, float]]:
    """Resample planning anchors to get CIs on the metric-vs-utility correlation."""
    rng = np.random.default_rng(seed)
    names = list(systems)
    key = {"mppi_cos": "cos", "neg_gt_rank": "rank",
           "neg_gt_rank_id": "rank_id", "plan_cost_corr": "pcc"}.get(utility_key)
    if key is None:
        return {}
    n = len(systems[names[0]]["per_plan"][key])
    acc: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        util = []
        for nm in names:
            v = np.nanmean(systems[nm]["per_plan"][key][idx])
            util.append(-v if utility_key.startswith("neg_") else v)
        for k in METRIC_KEYS:
            acc[k].append(spearman([systems[nm]["metrics"][k] for nm in names], util))
    return {
        k: {
            "mean": float(np.nanmean(v)),
            "lo": float(np.nanpercentile(v, 2.5)),
            "hi": float(np.nanpercentile(v, 97.5)),
        }
        for k, v in acc.items()
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pusht_subset.h5")
    ap.add_argument("--out", default="runs/k1k2_real/result.json")
    ap.add_argument("--dump-units", action="store_true",
                    help="also write <out>.units.npz with the per-anchor, "
                         "per-plan and per-closed-loop values behind each mean")
    ap.add_argument("--models", default="lewm_v2,lewm_v3,lewm_v4b")
    ap.add_argument("--pipelines", default="id,zero,neg,swap,atten",
                    help=f"comma list, or 'all' for the graded family ({len(PIPELINES)} entries)")
    ap.add_argument("--n-plan", type=int, default=64)
    ap.add_argument("--mppi-k", type=int, default=128)
    ap.add_argument("--mppi-iters", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--strata", type=int, default=1,
        help="split anchors into N equal groups by |net action| and evaluate each "
             "separately; low-|a| groups have a low action-attributable fraction of "
             "the one-step signal, which is where fidelity metrics should go blind",
    )
    ap.add_argument(
        "--weight-noise", action="store_true",
        help="also evaluate predictor-weight-perturbed variants (model-side "
             "corruptions, so fidelity and controllability degrade independently)",
    )
    ap.add_argument(
        "--n-closed", type=int, default=0,
        help="closed-loop receding-horizon anchors per system (0 = skip). Each "
             "costs `horizon` extra MPPI solves, so keep it well below --n-plan",
    )
    args = ap.parse_args()
    if args.pipelines.strip() == "all":
        args.pipelines = ALL_PIPELINES

    t0 = time.time()
    torch.manual_seed(args.seed)
    pixels, actions_raw, chain, block, horizon = load_subset(Path(args.data))
    print(f"frames={len(pixels)} anchors={len(chain)} block={block} H={horizon}", flush=True)

    # The simulator now serves two axes, so it is built whenever proprioception
    # exists: plan_cost_corr needs it for every planning anchor, while n_closed
    # only controls the far more expensive closed-loop rollout.
    sim = None
    proprio = load_proprio(Path(args.data))
    if proprio is None:
        print("no proprio in subset -> plan-cost and closed-loop axes disabled",
              flush=True)
    else:
        sim = ProprioNNSim(
            torch.from_numpy(proprio).to(args.device),
            torch.from_numpy(chain).to(args.device),
            torch.from_numpy(actions_raw).to(args.device),
        )
        print(
            f"proprio sim: {len(sim.P)} transitions, "
            f"mean |displacement|={float(sim.D.norm(dim=-1).mean()):.4f} m",
            flush=True,
        )

    # Stratify anchors by net first-plan-step action magnitude. Low-|a| anchors
    # carry little action-attributable signal, which is the regime where a
    # fidelity metric should stop tracking planning utility.
    net_a = np.linalg.norm(actions_raw[:, 0].sum(axis=-2), axis=-1)
    order = np.argsort(net_a)
    groups = np.array_split(order, max(1, args.strata))
    strata = {
        (f"q{i}" if args.strata > 1 else "all"): {
            "idx": np.sort(g),
            "net_action_mean": float(net_a[g].mean()),
        }
        for i, g in enumerate(groups)
    }
    for sk, sv in strata.items():
        print(f"stratum {sk}: n={len(sv['idx'])} mean|a|={sv['net_action_mean']:.4f} m", flush=True)

    per_stratum: Dict[str, Dict[str, Dict]] = {sk: {} for sk in strata}
    published = {}
    sigmas = WEIGHT_NOISE_SIGMAS if args.weight_noise else ()
    for mkey in args.models.split(","):
        d, ckpt = find_bundle(mkey)
        print(f"[{mkey}] {ckpt}", flush=True)
        adapter = LeWMAdapter(ckpt_path=ckpt, device=args.device, a_block=block, name=mkey)
        pred_state = {
            k: v.detach().clone() for k, v in adapter.model.predictor.state_dict().items()
        }
        Z = encode_all(adapter, pixels)
        acts_norm = adapter.normalize_actions(torch.from_numpy(actions_raw))
        print(f"[{mkey}] latents {tuple(Z.shape)}  {time.time()-t0:.0f}s", flush=True)
        published[mkey] = {"ckpt": str(ckpt), "latent_dim": int(Z.size(-1))}

        # (variant tag, sigma, pipelines to run)
        variants = [("", 0.0, args.pipelines.split(","))]
        variants += [
            (f"/w{s}", s, VARIANT_PIPELINES.split(",")) for s in sigmas
        ]

        for vtag, sigma, pipes in variants:
            adapter.model.predictor.load_state_dict(pred_state)
            if sigma > 0:
                gen = torch.Generator(device=adapter.device)
                gen.manual_seed(hash((mkey, sigma)) % (2 ** 31) + args.seed)
                perturb_predictor(adapter, sigma, gen)
            for sk, sv in strata.items():
                idx = sv["idx"]
                n_plan = max(8, int(round(args.n_plan * len(idx) / len(chain))))
                for pkey in pipes:
                    name = f"{mkey}{vtag}:{pkey}"
                    n_cl = max(0, int(round(args.n_closed * len(idx) / len(chain))))
                    r = eval_system(
                        adapter, pkey, Z, chain[idx], acts_norm[idx], horizon, block,
                        n_plan, args.mppi_k, args.mppi_iters,
                        sim=sim, n_closed=n_cl,
                    )
                    per_stratum[sk][name] = r
                    m = r["metrics"]
                    print(
                        f"  [{sk}] {name:26s} pid={m['pred_id_ratio_h1']:.3f} "
                        f"gapR={m['gap_ratio']:.3f} pvR={m['pv_ratio']:.3f} "
                        f"caca={m['caca']:+.3f} fAl={m['fact_align']:+.3f} "
                        f"| cos={m['mppi_cos']:+.3f} rank={m['gt_rank_pct']:.3f}"
                        f"/{m['gt_rank_id_pct']:.3f}"
                        + (f" pcc={m['plan_cost_corr']:+.3f}" if m["n_pcc"] else "")
                        + (f" clp={m['cl_progress']:+.3f}" if m["n_closed_loop"] else ""),
                        flush=True,
                    )
        del adapter, Z, pred_state
        torch.cuda.empty_cache()

    # Top-level summary fields describe this one stratum only; with >1 strata the
    # per-stratum block in `strata` is the result, not these fields.
    summary_stratum = "all" if "all" in per_stratum else list(per_stratum)[-1]
    systems = per_stratum[summary_stratum]

    def best(corr, keys):
        vals = [corr[k] for k in keys if corr[k] == corr[k]]
        return max(vals) if vals else float("nan")

    stratum_report = {}
    for sk, sysd in per_stratum.items():
        add_composites(sysd)
        c_cos = correlations(sysd, "mppi_cos")
        c_rank = correlations(sysd, "neg_gt_rank")
        stratum_report[sk] = {
            "n_anchors": int(len(strata[sk]["idx"])),
            "net_action_mean": strata[sk]["net_action_mean"],
            "systems": {k: v["metrics"] for k, v in sysd.items()},
            "spearman_vs_mppi_cos": c_cos,
            "spearman_vs_neg_gt_rank": c_rank,
            "best_ours_vs_cos": best(c_cos, OURS_KEYS),
            "best_fidelity_vs_cos": best(c_cos, FIDELITY_KEYS),
            "best_ours_vs_rank": best(c_rank, OURS_KEYS),
            "best_fidelity_vs_rank": best(c_rank, FIDELITY_KEYS),
        }
        c_rank_id = correlations(sysd, "neg_gt_rank_id")
        stratum_report[sk]["spearman_vs_neg_gt_rank_id"] = c_rank_id
        stratum_report[sk]["best_ours_vs_rank_id"] = best(c_rank_id, OURS_KEYS)
        stratum_report[sk]["best_fidelity_vs_rank_id"] = best(c_rank_id, FIDELITY_KEYS)
        if any(v["metrics"]["n_pcc"] for v in sysd.values()):
            c_pcc = correlations(sysd, "plan_cost_corr")
            stratum_report[sk]["spearman_vs_plan_cost_corr"] = c_pcc
            stratum_report[sk]["best_ours_vs_pcc"] = best(c_pcc, OURS_KEYS)
            stratum_report[sk]["best_fidelity_vs_pcc"] = best(c_pcc, FIDELITY_KEYS)
        if any(v["metrics"]["n_closed_loop"] for v in sysd.values()):
            c_cl = correlations(sysd, "cl_progress")
            stratum_report[sk]["spearman_vs_cl_progress"] = c_cl
            stratum_report[sk]["best_ours_vs_cl"] = best(c_cl, OURS_KEYS)
            stratum_report[sk]["best_fidelity_vs_cl"] = best(c_cl, FIDELITY_KEYS)
        sr = stratum_report[sk]
        print(
            f"[{sk}] ours_vs_cos={sr['best_ours_vs_cos']:+.3f} "
            f"fid_vs_cos={sr['best_fidelity_vs_cos']:+.3f} "
            f"| ours_vs_rank={sr['best_ours_vs_rank']:+.3f} "
            f"fid_vs_rank={sr['best_fidelity_vs_rank']:+.3f}"
            + (
                f" | ours_vs_cl={sr['best_ours_vs_cl']:+.3f} "
                f"fid_vs_cl={sr['best_fidelity_vs_cl']:+.3f}"
                if "best_ours_vs_cl" in sr else ""
            ),
            flush=True,
        )

    corr_cos = correlations(systems, "mppi_cos")
    corr_rank = correlations(systems, "neg_gt_rank")
    boot_cos = bootstrap_corr(systems, "mppi_cos")
    ours = best(corr_cos, OURS_KEYS)
    fid = best(corr_cos, FIDELITY_KEYS)
    # Harness acceptance check: the model card reports pred/id ~0.5 at one
    # plan-step. Ratios in the tens mean the preprocessing convention is wrong.
    sanity = {
        mkey: {
            "pred_id_ratio_h1": systems[f"{mkey}:id"]["metrics"]["pred_id_ratio_h1"],
            "plausible": systems[f"{mkey}:id"]["metrics"]["pred_id_ratio_h1"] < 20.0,
        }
        for mkey in args.models.split(",")
        if f"{mkey}:id" in systems
    }

    out = {
        "seconds": time.time() - t0,
        "seed": args.seed,
        "data": args.data,
        "n_frames": int(len(pixels)),
        "n_anchors": int(len(chain)),
        "n_plan": args.n_plan,
        "harness_sanity": sanity,
        "published_reference": {
            "lewm_v2": {"cost_cv_h5": 0.180, "gt_rank_pct": 0.366, "pred_id_h1": 0.465},
            "lewm_v3": {"cost_cv_h5": 0.132, "gt_rank_pct": 0.33},
            "lewm_v4b": {"cost_cv_h5": 0.286, "gt_rank_pct": 0.44},
        },
        "models": published,
        "strata": stratum_report,
        "systems": {k: v["metrics"] for k, v in systems.items()},
        "spearman_vs_mppi_cos": corr_cos,
        "spearman_vs_neg_gt_rank": corr_rank,
        "bootstrap_vs_mppi_cos": boot_cos,
        "best_ours_vs_cos": ours,
        "best_fidelity_vs_cos": fid,
        "best_ours_vs_rank": best(corr_rank, OURS_KEYS),
        "best_fidelity_vs_rank": best(corr_rank, FIDELITY_KEYS),
        "summary_stratum": summary_stratum,
        "pass_k1k2": (
            bool(ours > fid + 0.1 and ours > 0.4) if summary_stratum == "all" else None
        ),
        "n_systems": len(systems),
        "note": (
            "Real LeWM PushT-FR3 checkpoints x action-corruption zoo "
            "(+ optional predictor-weight perturbations). "
            "pv_nn = nearest-neighbour counterfactual oracle from real data. "
            f"Top-level spearman/pass fields cover stratum '{summary_stratum}' only."
        ),
    }
    save_json(out, Path(args.out))

    # Companion archive of the per-unit values behind every mean above. Without
    # it a metric's reliability can only be probed across whole evaluation runs,
    # which cannot separate anchor-sampling noise from a real stratum effect and
    # says nothing at all about the deterministic metrics.
    if args.dump_units:
        units = {}
        for sk, sysd in per_stratum.items():
            for name, r in sysd.items():
                for grp in ("per_anchor", "per_plan", "per_closed"):
                    for field, arr in r[grp].items():
                        units[f"{sk}|{name}|{field}"] = np.asarray(arr)
        p = Path(args.out).with_suffix(".units.npz")
        np.savez_compressed(p, **units)
        print(f"WROTE {p} ({len(units)} arrays, {p.stat().st_size/1e6:.1f} MB)")

    print(json.dumps(
        {
            "spearman_vs_mppi_cos": corr_cos,
            "best_ours_vs_cos": ours,
            "best_fidelity_vs_cos": fid,
            "pass_k1k2": out["pass_k1k2"],
        },
        indent=2,
    ))
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
