"""
LeWM adapter — latent CAF on YuhaiW/lewm-pusht-fr3-v2.

predict(): one plan-step latent transition given RGB context + normalized actions.
Uses the same encode / action_encoder / predict path as PushtLewmInference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Tuple  # noqa: F401

import numpy as np
import torch
import torch.nn.functional as F

from eval.metrics import WorldModelAdapter, Batch, compute_caf, save_json

CKPT_DIR = Path("/mnt/group/jxdong/wm_exp/ckpts/lewm_pusht")
LOCAL_CKPT = Path("ckpts/lewm_pusht")

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Released bundles sharing the LeWM JEPA architecture.
BUNDLES = {
    "lewm_v2": ("ckpts/lewm_pusht", "lewm_pusht_fr3_v2.ckpt"),
    "lewm_v3": ("ckpts/lewm_v3", "lewm_pusht_fr3_v3.ckpt"),
    "lewm_v4b": ("ckpts/lewm_v4b", "lewm_pusht_lewm_fr3_v4b.ckpt"),
}


def resolve_ckpt_dir(ckpt_dir: Optional[Path] = None) -> Path:
    if ckpt_dir is not None:
        return Path(ckpt_dir)
    if (CKPT_DIR / "lewm_pusht_fr3_v2.ckpt").exists():
        return CKPT_DIR
    if (LOCAL_CKPT / "lewm_pusht_fr3_v2.ckpt").exists():
        return LOCAL_CKPT
    raise FileNotFoundError(f"No LeWM ckpt under {CKPT_DIR} or {LOCAL_CKPT}")


def find_bundle(key: str, roots=(Path("/mnt/group/jxdong/wm_exp"), Path("."))) -> Tuple[Path, Path]:
    """
    Return (dir, ckpt_path) for a released bundle or a model we trained.

    Unknown keys are looked up as ckpts/trained/<key>/<key>.ckpt, the layout
    code/train_wm.py writes.
    """
    if key in BUNDLES:
        rel_dir, ckpt_name = BUNDLES[key]
    else:
        rel_dir, ckpt_name = f"ckpts/trained/{key}", f"{key}.ckpt"
    for root in roots:
        d = root / rel_dir
        if (d / ckpt_name).exists():
            return d, d / ckpt_name
    raise FileNotFoundError(f"bundle {key}: {ckpt_name} not found under {roots}")


class LeWMAdapter(WorldModelAdapter):
    """Action-conditioned latent world model (JEPA)."""

    name = "lewm_pusht"

    def __init__(
        self,
        ckpt_dir: Optional[Path] = None,
        device: str = "cuda",
        history_size: int = 3,
        a_block: int = 5,
        ckpt_path: Optional[Path] = None,
        name: Optional[str] = None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.history_size = history_size
        self.a_block = a_block
        self.meta: dict = {}

        if ckpt_path is not None:
            ckpt = Path(ckpt_path)
            self.ckpt_dir = ckpt.parent
        else:
            self.ckpt_dir = resolve_ckpt_dir(ckpt_dir)
            found = sorted(self.ckpt_dir.glob("*.ckpt"))
            if not found:
                raise FileNotFoundError(f"no *.ckpt in {self.ckpt_dir}")
            ckpt = found[0]
        if name:
            self.name = name

        # Models we trained record their own context length; the released
        # bundles do not, and all of them use 3.
        meta_path = self.ckpt_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
            self.history_size = int(self.meta.get("history_size", history_size))
            self.a_block = int(self.meta.get("a_block", a_block))

        # Make jepa/module importable for unpickle
        sys.path.insert(0, str(self.ckpt_dir))
        import jepa  # noqa: F401
        import module  # noqa: F401

        self.model = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._patch_vit_config(self.model)
        self._patch_vit_modules(self.model)

        scaler = self.ckpt_dir / "action_scaler.json"
        if scaler.exists():
            with open(scaler) as f:
                sc = json.load(f)
            mean, std = sc["mean"], sc["std"]
        else:
            # v4b ships the scaler inside the prior head checkpoint
            mean, std = self._scaler_from_prior(self.ckpt_dir)
        self.a_mean = torch.tensor(mean, dtype=torch.float32, device=self.device)
        self.a_std = torch.tensor(std, dtype=torch.float32, device=self.device).clamp_min(1e-9)
        self.action_dim = len(mean)

        self._mean = IMAGENET_MEAN.to(self.device)
        self._std = IMAGENET_STD.to(self.device)

    @staticmethod
    def _scaler_from_prior(ckpt_dir: Path):
        """Recover StandardScaler stats from a bundled prior-head checkpoint."""
        cands = sorted(ckpt_dir.glob("prior_head*.pt"))
        if not cands:
            raise FileNotFoundError(f"no action_scaler.json nor prior_head*.pt in {ckpt_dir}")
        sys.path.insert(0, str(ckpt_dir))
        blob = torch.load(cands[0], map_location="cpu", weights_only=False)

        def dig(obj, depth=0):
            if depth > 4:
                return None
            if isinstance(obj, dict):
                for k in ("action_scaler", "scaler", "act_scaler"):
                    if k in obj:
                        s = obj[k]
                        if isinstance(s, dict) and "mean" in s and "std" in s:
                            return list(np.asarray(s["mean"]).ravel()), list(np.asarray(s["std"]).ravel())
                if "mean" in obj and "std" in obj:
                    return list(np.asarray(obj["mean"]).ravel()), list(np.asarray(obj["std"]).ravel())
                for v in obj.values():
                    r = dig(v, depth + 1)
                    if r:
                        return r
            return None

        got = dig(blob)
        if got is None:
            raise KeyError(f"could not find scaler stats in {cands[0]}")
        return got

    @staticmethod
    def _patch_vit_config(model) -> None:
        """Pickled ViT configs miss attrs expected by current transformers."""
        enc = getattr(model, "encoder", None)
        cfg = getattr(enc, "config", None) if enc is not None else None
        if cfg is None:
            return
        try:
            from transformers import ViTConfig
            fresh = ViTConfig()
            for k, v in fresh.to_dict().items():
                if not hasattr(cfg, k):
                    try:
                        setattr(cfg, k, v)
                    except Exception:
                        pass
        except Exception:
            pass
        for k, v in {
            "output_attentions": False,
            "output_hidden_states": False,
            "use_return_dict": True,
            "return_dict": True,
            "torchscript": False,
            "tie_word_embeddings": True,
            "is_encoder_decoder": False,
        }.items():
            if not hasattr(cfg, k):
                try:
                    setattr(cfg, k, v)
                except Exception:
                    pass

    @staticmethod
    def _patch_vit_modules(model) -> None:
        """Bridge pickled ViT modules ↔ transformers 4.41 attention API."""
        for m in model.modules():
            name = m.__class__.__name__
            if name == "ViTSelfAttention" and not hasattr(m, "dropout"):
                # older checkpoints: dropout lived elsewhere / was absent
                m.dropout = torch.nn.Identity()
            if name == "ViTAttention" and not hasattr(m, "attention"):
                # unlikely; leave for forward diagnostics
                pass
            if name in ("ViTSelfAttention", "ViTAttention", "ViTLayer") and not hasattr(m, "pruned_heads"):
                try:
                    m.pruned_heads = set()
                except Exception:
                    pass

    def preprocess_rgb(self, frames: torch.Tensor) -> torch.Tensor:
        """(B,T,H,W,3) uint8 or float[0,1] → ImageNet-normalized (B,T,3,224,224)."""
        if frames.dtype == torch.uint8:
            x = frames.float() / 255.0
        else:
            x = frames.float()
            if x.max() > 1.5:
                x = x / 255.0
        if x.shape[-1] == 3:
            x = x.permute(0, 1, 4, 2, 3)  # B,T,C,H,W
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w).to(self.device)
        if (h, w) != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self._mean) / self._std
        return x.view(b, t, c, 224, 224)

    def encode_pixels(self, pixels_btchw: torch.Tensor) -> torch.Tensor:
        info = {"pixels": pixels_btchw.to(self.device)}
        return self.model.encode(info)["emb"]  # (B,T,D)

    def normalize_actions(self, actions_raw: torch.Tensor) -> torch.Tensor:
        """raw meters → normalized. actions (B,T,2) or (B,T,A_block,2)."""
        return (actions_raw.to(self.device) - self.a_mean) / self.a_std

    def flatten_plan_step(self, actions_norm: torch.Tensor) -> torch.Tensor:
        """
        Accept:
          (B, T, 2) — repeat across A_block
          (B, T, A_block*2) — already flat
          (B, T, A_block, 2)
        Return (B, T, A_block*2) for action_encoder.
        """
        if actions_norm.ndim == 4:
            b, t, ab, d = actions_norm.shape
            return actions_norm.reshape(b, t, ab * d)
        if actions_norm.ndim == 3 and actions_norm.size(-1) == self.action_dim:
            # repeat single Δxy across frameskip block
            rep = actions_norm.unsqueeze(2).expand(-1, -1, self.a_block, -1)
            b, t, ab, d = rep.shape
            return rep.reshape(b, t, ab * d)
        return actions_norm

    @torch.no_grad()
    def predict(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        normalized: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        obs: (B,T,D) latents OR (B,T,H,W,3)/(B,T,3,H,W) pixels
        actions: plan-step actions (see flatten_plan_step)
        normalized: True if actions are already in StandardScaler space.
                    None auto-detects from magnitude (raw deltas are ~mm).
        Returns predicted next latents (B, T_act, D) aligned with action length.
        """
        if obs.ndim >= 4:
            pixels = self.preprocess_rgb(obs)
            emb = self.encode_pixels(pixels)
        else:
            emb = obs.to(self.device)

        acts = actions.to(self.device)
        if normalized is None:
            normalized = not (
                acts.size(-1) == self.action_dim and acts.abs().mean() < 0.05
            )
        if not normalized and acts.size(-1) == self.action_dim:
            acts = self.normalize_actions(acts)
        acts_flat = self.flatten_plan_step(acts)
        t_act = acts_flat.size(1)

        # Build context: use last history_size frames of emb, pad if needed
        preds = []
        hist = emb[:, : max(1, emb.size(1))]
        for t in range(t_act):
            ctx_emb = hist[:, -self.history_size :]
            if ctx_emb.size(1) < self.history_size:
                pad = hist[:, :1].expand(-1, self.history_size - ctx_emb.size(1), -1)
                ctx_emb = torch.cat([pad, ctx_emb], dim=1)
            act_t = acts_flat[:, t : t + 1]
            act_emb = self.model.action_encoder(act_t)
            ctx_act = torch.zeros(
                hist.size(0), self.history_size, act_emb.size(-1), device=self.device
            )
            ctx_act[:, -1:] = act_emb
            pred = self.model.predict(ctx_emb, ctx_act)
            next_e = pred[:, -1:]
            preds.append(next_e)
            hist = torch.cat([hist, next_e], dim=1)
        return torch.cat(preds, dim=1)

    def predict_cf(self, obs, actions, actions_cf, normalized: Optional[bool] = None):
        return self.predict(obs, actions_cf, normalized=normalized)


def status_report(out_path: str = "runs/lewm_status/result.json"):
    info = {"dirs": {}}
    for d in [CKPT_DIR, LOCAL_CKPT]:
        files = [p.name for p in d.glob("*")] if d.exists() else []
        info["dirs"][str(d)] = {"exists": d.exists(), "files": files}
    try:
        resolve_ckpt_dir()
        info["ready_for_adapter"] = True
        info["next"] = "Run code/k1_lewm_latent_caf.py"
    except Exception as e:
        info["ready_for_adapter"] = False
        info["error"] = str(e)
    save_json(info, Path(out_path))
    print(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    status_report()
