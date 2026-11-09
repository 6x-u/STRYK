from __future__ import annotations

import time

import numpy as np
import torch

from .multi_env import MultiEnv
from .policy_multi import MultiAgentPolicy
from .checkpoint import load_checkpoint
from .metrics import EpisodeStats
from .obs_dynamic import build_entities
from .policy_multi import action_mask
from .act import zero_action, decode


def _run_one(env: MultiEnv, policy: MultiAgentPolicy, device: str, deterministic: bool = True,
             max_steps: int | None = None) -> dict:
    obs = env.reset()
    total = 0.0
    done = False
    steps = 0
    while not done:
        if max_steps is not None and steps >= max_steps:
            break
        steps += 1
        with torch.no_grad():
            ent = torch.as_tensor(obs["entities"], dtype=torch.float32, device=device)
            msk = torch.as_tensor(obs["mask"], dtype=torch.float32, device=device)
            glo = torch.as_tensor(obs["global"], dtype=torch.float32, device=device)
            ams = torch.as_tensor(obs["action_mask"], dtype=torch.float32, device=device)
            cont, disc, _, _, _ = policy.sample(ent, msk, glo, ams, deterministic=deterministic)
        obs, rew, done, info = env.step(cont.cpu().numpy(), disc.cpu().numpy())
        total += float(np.sum(rew))
    return {"reward": total, "score": info["score"],
            "tracker": env.tracker.summary(), "world": env.world}


def benchmark_checkpoint(checkpoint: str, modes=("1v1", "5v5"), stadiums=("wembley_fc27", "etihad_fc27"),
                         episodes_per: int = 2, seeds=(2026, 7, 99), device: str = "cpu",
                         opponent_styles=("balanced", "press", "counter"), fast_sec: float = 0.0) -> dict:
    # unseen styles on purpose â€” if it only wins vs its own mirror it is just memorising lol
    import torch as _t
    _pay = _t.load(checkpoint, map_location=device, weights_only=False)
    _hid = int(_pay.get("config", {}).get("hidden", 256))
    policy = MultiAgentPolicy(hidden=_hid).to(device)
    load_checkpoint(checkpoint, policy, None, device=device)
    policy.eval()
    stats = EpisodeStats()
    per_mode: dict[str, dict] = {}
    wins = draws = losses = 0
    t0 = time.time()
    eps = 0
    for mode in modes:
        mg, mw = 0, 0
        for stad in stadiums:
            for s in seeds:
                env = MultiEnv(mode, stad, int(s), style=str(opponent_styles[eps % len(opponent_styles)]))
                max_steps = int(fast_sec * 60) if fast_sec > 0 else None
                out = _run_one(env, policy, device, max_steps=max_steps)
                if out["score"][0] > out["score"][1]:
                    wins += 1
                    mw += 1
                elif out["score"][0] == out["score"][1]:
                    draws += 1
                else:
                    losses += 1
                # episode ledger for the fancy numbers
                stats.goals_for += int(out["score"][0])
                stats.goals_against += int(out["score"][1])
                tr = out["tracker"]
                stats.passes_tried += int(tr.get("passes", 0))
                stats.passes_done += int(tr.get("passes", 0) * tr.get("pass_completion", 0.0))
                stats.shots += int(tr.get("shots", 0))
                stats.on_target += int(tr.get("shots", 0) * tr.get("shot_accuracy", 0.0))
                stats.poss_home += float(tr.get("possession_pct", 0.0))
                stats.turnovers += int(tr.get("turnovers", 0))
                stats.def_wins += int(tr.get("def_wins", 0))
                stats.episodes += 1
                mg += 1
                eps += 1
        per_mode[mode] = {"episodes": mg, "wins": mw}
    total_eps = max(wins + draws + losses, 1)
    snap = stats.summary()
    snap.update({
        "win_rate": wins / total_eps,
        "draw_rate": draws / total_eps,
        "loss_rate": losses / total_eps,
        "wins": wins, "draws": draws, "losses": losses,
        "modes": per_mode,
        "wall_sec": time.time() - t0,
        "checkpoint": checkpoint,
    })
    return snap


def bench_sim(mode: str, stadium: str, seconds: int = 2, seed: int = 2026) -> dict:
    env = MultiEnv(mode, stadium, seed)
    env.reset()
    steps = int(seconds * 60)
    t0 = time.time()
    for _ in range(steps):
        env.step(np.zeros((len(env.home_ids()), 10), dtype=np.float32),
                 np.zeros(len(env.home_ids()), dtype=np.int64))
    real = time.time() - t0
    return {"mode": mode, "stadium": stadium, "sim_sec": float(steps) / 60.0,
            "real_sec": real, "speedup": (float(steps) / 60.0) / max(real, 1e-6)}

