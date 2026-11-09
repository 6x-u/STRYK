# STRYK kaggle_run.py
# Kaggle entry — tiny wrapper so kernels do not care where the code lives :)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from stryk.config import TrainCfg
from stryk.train_multi import train_multi


def main() -> None:
    gpu = torch.cuda.is_available()
    print(f"[kaggle] cuda={gpu} torch={torch.__version__} ", end="")
    if gpu:
        print(f"device={torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("cpu fallback — long runs will be slow", flush=True)
    # tune the budget to the kernel session you chose; 8h of GPU beats 8h of laptop :)
    cfg = TrainCfg(
        mode="5v5",
        stadium="etihad_fc27",
        seed=2026,
        rollout_steps=8192 if gpu else 2048,
        minibatch=2048 if gpu else 512,
        hidden=256 if gpu else 128,
        total_updates=400,
        save_every=20,
        eval_every=20,
        stage=0,
        curriculum=True,
        selfplay=True,
        use_mappo=True,
        train_hours=8.0,
        models_root="Models",
        telemetry_csv="Models/Metadata/telemetry.csv",
    )
    train_multi(cfg)
    print("[kaggle] done — download the Models/ folder", flush=True)


if __name__ == "__main__":
    main()
