"""Pluggable eval harness for main-track CAF / Mem–Ctrl experiments."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import json
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class Batch:
    """Minimal batch for action-conditioned world models."""
    obs: torch.Tensor          # (B, T, ...) observation / latent / image
    actions: torch.Tensor      # (B, T-1, A) or (B, T, A)
    next_obs: Optional[torch.Tensor] = None  # teacher targets if available
    meta: Optional[Dict[str, Any]] = None


@dataclass
class CAFResult:
    fact_err: float
    self_cf_gap: float
    pred_validity: float
    caf: float
    extra: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class WorldModelAdapter(ABC):
    """Implement for each real / toy WM."""

    name: str = "base"

    @abstractmethod
    def predict(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict next obs/latents given context obs and action sequence."""

    def predict_cf(self, obs: torch.Tensor, actions: torch.Tensor, actions_cf: torch.Tensor) -> torch.Tensor:
        return self.predict(obs, actions_cf)


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.mse_loss(a, b).item())


def compute_caf(
    adapter: WorldModelAdapter,
    batch: Batch,
    actions_cf: torch.Tensor,
    oracle_next: Optional[torch.Tensor] = None,
    fact_gate_max: Optional[float] = None,
    beta: float = 40.0,
) -> CAFResult:
    """
    CAF = self_cf_gap / (pred_validity + eps).

    Main-track lesson (K2v2): raw CAF promotes OOD gap inflation (weak listeners).
    Prefer reporting also:
      - caf_gated: CAF if fact_err <= fact_gate_max else 0
      - score_iv: tanh(gap)/(1+pv)*exp(-beta*fact_err)

    If oracle_next is None, pred_validity uses batch.next_obs (weak stand-in).
    """
    pred_f = adapter.predict(batch.obs, batch.actions)
    pred_cf = adapter.predict_cf(batch.obs, batch.actions, actions_cf)

    target = batch.next_obs
    fact_err = mse(pred_f, target) if target is not None else float("nan")
    self_gap = mse(pred_f, pred_cf)

    weak = False
    if oracle_next is not None:
        pv = mse(pred_cf, oracle_next)
    elif target is not None:
        pv = mse(pred_cf, target)
        weak = True
    else:
        pv = float("nan")
        weak = True

    caf = self_gap / (pv + 1e-8) if pv == pv else float("nan")
    extra: Dict[str, float] = {}
    if weak:
        extra["pv_is_weak_standin"] = 1.0
    if fact_err == fact_err and pv == pv:
        import math
        extra["score_iv"] = math.tanh(self_gap) / (1.0 + pv) * math.exp(-beta * fact_err)
        if fact_gate_max is not None:
            extra["caf_gated"] = caf if fact_err <= fact_gate_max else 0.0
    return CAFResult(fact_err=fact_err, self_cf_gap=self_gap, pred_validity=pv, caf=caf, extra=extra or None)


def spearman(x, y) -> float:
    """Rank correlation without scipy."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for rank_i, i in enumerate(order):
            r[i] = float(rank_i)
        return r
    rx, ry = rank(x), rank(y)
    n = len(x)
    if n < 2:
        return float("nan")
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = sum((a - mx) ** 2 for a in rx) ** 0.5
    deny = sum((b - my) ** 2 for b in ry) ** 0.5
    if denx < 1e-12 or deny < 1e-12:
        return float("nan")
    return num / (denx * deny)


def predictive_validity_table(
    model_scores: Dict[str, Dict[str, float]],
    downstream: Dict[str, float],
    metric_keys=("caf", "fact_err", "self_cf_gap"),
) -> Dict[str, float]:
    """
    model_scores[model_name][metric] = value
    downstream[model_name] = utility (higher better)
    Returns Spearman(metric, downstream); for error-like metrics use negated.
    """
    names = [n for n in model_scores if n in downstream]
    out = {}
    for k in metric_keys:
        xs = [model_scores[n][k] for n in names]
        ys = [downstream[n] for n in names]
        # higher CAF better; higher fact_err worse
        if k in ("fact_err", "pred_validity", "fvd", "mse"):
            xs = [-v for v in xs]
        out[f"spearman_{k}"] = spearman(xs, ys)
    return out


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
