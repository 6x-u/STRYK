"""kaggle training entry — run this file on kaggle.com, nothing else to touch :)
upload the STRYK folder as a Dataset, open a GPU kernel, run training.py.
it auto-detects kaggle paths + gpu, then trains the merobot model and keeps
saving .pt updates so a dropped session never loses progress.
"""
# training.py lives at the STRYK root so kaggle kernels find it at /kaggle/working/training.py
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# dataset layout on kaggle: /kaggle/input/<dataset>/stryk, /kaggle/input/<dataset>/configs ...
for cand in (Path("/kaggle/input"), Path("/kaggle/working"), Path("/kaggle/input/stryk")):
    if cand.exists():
        for sub in cand.rglob("stryk"):
            if (sub / "sim.py").exists() and str(sub.parent) not in sys.path:
                sys.path.insert(0, str(sub.parent))
                break

import torch
from stryk.config import TrainCfg, load_config
from stryk.train_multi import train_multi


def find_config() -> Path | None:
    for c in (ROOT / "configs" / "pro11.json",
              ROOT / "configs" / "mappo_5v5.json",
              ROOT / "configs" / "club5.json",
              Path("/kaggle/input/stryk/configs/pro11.json")):
        if c.exists():
            return c
    return None


def main() -> None:
    # env overrides win — handy on kaggle notebooks :)
    hours = float(os.environ.get("TRAIN_HOURS", os.environ.get("KAGGLE_TIME", "8")))
    updates = int(os.environ.get("UPDATES", "0"))
    model = os.environ.get("MODEL_NAME", "merobot")
    gpu = torch.cuda.is_available()
    print(f"[training] cuda={gpu} torch={torch.__version__} hours={hours} model={model}", flush=True)
    if gpu:
        print(f"[training] device={torch.cuda.get_device_name(0)}", flush=True)
    cfg_path = find_config()
    if cfg_path:
        cfg = load_config(cfg_path)
    else:
        cfg = TrainCfg(mode="11v11", stadium="etihad_fc27")
    cfg.mode = "11v11"  # this entry trains the full team, not 5v5 :)
    cfg.model_name = model
    cfg.use_mappo = True
    cfg.train_hours = hours
    cfg.total_updates = updates if updates > 0 else int(cfg.total_updates)
    # /kaggle/input is read-only, so checks go to /kaggle/working on kaggle :)
    if Path("/kaggle/working").exists():
        out_root = Path("/kaggle/working")
    else:
        out_root = ROOT
    cfg.models_root = str(out_root / "Models")
    cfg.telemetry_csv = str(out_root / "Models" / "Metadata" / "telemetry.csv")
    # speed up on gpu without touching physics — same 240Hz/60Hz FC27 sim, just bigger batches :)
    if gpu:
        if cfg.rollout_steps < 8192:
            cfg.rollout_steps = 8192
        if cfg.minibatch < 2048:
            cfg.minibatch = 2048
        if cfg.hidden < 256:
            cfg.hidden = 256
    else:
        cfg.rollout_steps = min(int(cfg.rollout_steps), 2048)
        cfg.hidden = min(int(cfg.hidden), 128)
    # resume nearest merobot_latest if present so restarts do not start dumb again
    if Path("/kaggle/working").exists():
        resume = Path("/kaggle/working") / "Models" / "Checkpoints" / f"{model}_latest.pt"
    else:
        resume = ROOT / "Models" / "Checkpoints" / f"{model}_latest.pt"
    # kaggle kernels may unpack the dataset read-only — checkpoints snapshot next to the script
    train_multi(cfg, resume=str(resume) if resume.exists() else None)
    print("[training] done — download the Models/ folder", flush=True)


if __name__ == "__main__":
    main()
