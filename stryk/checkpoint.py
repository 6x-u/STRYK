from __future__ import annotations

import datetime
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from .config import model_layout
from .version import __version__

# .pt payloads carry everything needed to resume bit-for-bit-ish :)
# actor + critic + optimizer + scheduler + norm stats + rng + obs/act versions


def _rng_states() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _sha12(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def _states(policy) -> dict:
    if hasattr(policy, "state_dict"):
        return {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in policy.state_dict().items()}
    return dict(policy)


def save_checkpoint(pt_path: str | Path, policy, optimizer, step: int, cfg, extra: dict | None = None,
                    scheduler=None, norm_stats: dict | None = None, weights_only: bool = False) -> Path:
    p = Path(pt_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "actor_state": _states(policy),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "norm_stats": norm_stats or {},
        "rng": _rng_states() if not weights_only else {},
        "step": int(step),
        "version": __version__,
        "obs_version": int(getattr(cfg, "obs_version", 2)),
        "act_version": int(getattr(cfg, "act_version", 2)),
        "config": cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg),
        "extra": extra or {},
    }
    torch.save(payload, str(p))
    meta = {
        "file": p.name,
        "step": int(step),
        "version": __version__,
        "mode": getattr(cfg, "mode", None),
        "stadium": getattr(cfg, "stadium", None),
        "stage": getattr(cfg, "stage", None),
        "seed": getattr(cfg, "seed", None),
        "obs_version": payload["obs_version"],
        "act_version": payload["act_version"],
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if extra:
        for k in ("league", "curriculum", "benchmark", "metrics"):
            if k in extra:
                meta[k] = extra[k]
    meta["sha256_12"] = _sha12(p)
    p.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return p


def load_checkpoint(pt_path: str | Path, policy=None, optimizer=None, device: str = "cpu",
                    strict: bool = True, scheduler=None) -> dict:
    p = Path(pt_path)
    payload = torch.load(str(p), map_location=device, weights_only=False)
    if policy is not None and "actor_state" in payload:
        policy.load_state_dict(payload["actor_state"], strict=strict)
    if optimizer is not None and payload.get("optimizer_state"):
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state"):
        scheduler.load_state_dict(payload["scheduler_state"])
    if payload.get("rng"):
        try:
            torch.set_rng_state(payload["rng"]["torch"])
            np.random.set_state(payload["rng"]["numpy"])
            random.setstate(payload["rng"]["python"])
        except Exception:
            pass
    return payload


# ---- triple-save: latest (train state) / best (eval winner) / final (approved) ----

def _model_name(cfg) -> str:
    name = str(getattr(cfg, "model_name", "STRYK") or "STRYK").strip().lower().replace(" ", "_")
    return name or "STRYK"


def save_latest(policy, optimizer, step, cfg, extra=None, scheduler=None) -> Path:
    name = _model_name(cfg)
    lay = model_layout(getattr(cfg, "models_root", "Models"))
    lay["checkpoints"].mkdir(parents=True, exist_ok=True)
    return save_checkpoint(lay["checkpoints"] / f"{name}_latest.pt",
                           policy, optimizer, step, cfg, extra, scheduler=scheduler)


def save_best(policy, optimizer, step, cfg, score: float, extra=None, scheduler=None) -> Path:
    name = _model_name(cfg)
    lay = model_layout(getattr(cfg, "models_root", "Models"))
    lay["checkpoints"].mkdir(parents=True, exist_ok=True)
    extra = dict(extra or {})
    extra["benchmark"] = {"score": float(score)}
    return save_checkpoint(lay["checkpoints"] / f"{name}_best.pt",
                           policy, optimizer, step, cfg, extra, scheduler=scheduler)


def promote_to_final(best_path: str | Path, cfg, benchmark: dict | None = None) -> Path:
    name = _model_name(cfg)
    lay = model_layout(getattr(cfg, "models_root", "Models"))
    lay["best"].mkdir(parents=True, exist_ok=True)
    src = Path(best_path)
    dst = lay["best"] / f"{name}_final.pt"
    # copy bytes so final survives later overwrites of best :)
    dst.write_bytes(src.read_bytes())
    meta = json.loads(src.with_suffix(".meta.json").read_text(encoding="utf-8")) if src.with_suffix(".meta.json").exists() else {}
    meta["promoted"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if benchmark:
        meta["benchmark"] = benchmark
    dst.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dst


def export_final_onnx(policy, cfg, device: str = "cpu") -> dict:
    name = _model_name(cfg)
    lay = model_layout(getattr(cfg, "models_root", "Models"))
    lay["exports"].mkdir(parents=True, exist_ok=True)
    lay["metadata"].mkdir(parents=True, exist_ok=True)
    onnx_p = lay["exports"] / f"{name}_final.onnx"
    # dynamic-obs wrapper: entities+mask+global -> cont mean + disc argmax baked as dummy path
    from .obs_dynamic import MAX_ENTITIES, ENT_DIM, GLOBAL_DIM
    policy = policy.to(device)
    policy.eval()

    class DynWrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, ent, mask, glob):
            mu, logits, _ = self.m(ent, mask, glob)
            return mu, logits

    w = DynWrap(policy)
    dummy_e = torch.zeros(1, MAX_ENTITIES, ENT_DIM)
    dummy_m = torch.zeros(1, MAX_ENTITIES)
    dummy_g = torch.zeros(1, GLOBAL_DIM)
    torch.onnx.export(w, (dummy_e, dummy_m, dummy_g), str(onnx_p),
                      input_names=["entities", "mask", "global"],
                      output_names=["cont_mean", "disc_logits"],
                      dynamic_axes={"entities": {0: "N"}, "mask": {0: "N"}, "global": {0: "N"}},
                      opset_version=17, dynamo=False)
    manifest = {
        "model": f"{name}_final.onnx",
        "model_name": name,
        "version": __version__,
        "obs_version": int(getattr(cfg, "obs_version", 2)),
        "act_version": int(getattr(cfg, "act_version", 2)),
        "mode": getattr(cfg, "mode", None),
        "stadium": getattr(cfg, "stadium", None),
        "exported": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sha256_12": _sha12(onnx_p),
        "inputs": ["entities", "mask", "global"],
        "outputs": ["cont_mean", "disc_logits"],
    }
    (lay["exports"] / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (lay["metadata"] / "model_info.json").write_text(json.dumps(
        {**manifest, "kind": "final", "mappo": True, "hidden": getattr(cfg, "hidden", 128)}, indent=2), encoding="utf-8")
    return {"onnx": str(onnx_p), "manifest": str(lay["exports"] / "manifest.json"),
            "info": str(lay["metadata"] / "model_info.json")}


def export_onnx(policy, obs_dim: int, out_path: str | Path, device: str = "cpu", opset: int = 17) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    policy = policy.to(device)
    policy.eval()
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32, device=device)

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            mu, _, _ = self.m(x)
            return mu

    w = Wrapper(policy)
    torch.onnx.export(w, dummy, str(p), input_names=["obs"], output_names=["act"],
                      dynamic_axes={"obs": {0: "N"}, "act": {0: "N"}},
                      opset_version=opset, dynamo=False)
    return p


def export_torchscript(policy, out_path: str | Path, device: str = "cpu") -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    policy = policy.to(device)
    policy.eval()
    scripted = torch.jit.script(policy)
    scripted.save(str(p))
    return p



