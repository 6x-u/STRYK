from __future__ import annotations

import csv
import time

import numpy as np
import torch

from .config import TrainCfg
from .curriculum import Curriculum
from .env import StrykEnv
from .policy import ActorCritic, np_sample
from .buffer import RolloutBuffer
from .ppo import ppo_update
from .checkpoint import save_checkpoint, load_checkpoint
from .selfplay import League
from .version import __version__


def _device(cfg: TrainCfg) -> str:
    return cfg.device if torch.cuda.is_available() else "cpu"


def train(cfg: TrainCfg, resume: str | None = None) -> dict:
    dev = _device(cfg)
    print(f"[stryk] train mode={cfg.mode} stadium={cfg.stadium} dev={dev} v={__version__}", flush=True)
    cur = Curriculum(start=cfg.stage, enabled=cfg.curriculum)
    league = League(seed=cfg.seed)
    env = StrykEnv(cfg.mode, cfg.stadium, cfg.seed, stage=cur.stage)
    policy = ActorCritic().to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
    start_step = 0
    if resume:
        payload = load_checkpoint(resume, policy, opt, device=dev)
        start_step = int(payload.get("step", 0))
        print(f"[stryk] resumed step {start_step} from {resume}", flush=True)
    buf = RolloutBuffer(env.observation_space_dim, env.action_space_dim, cfg.rollout_steps)
    obs, _ = env.reset()
    total_updates = int(cfg.total_updates)
    stats = {"updates": 0, "rewards": [], "goals": 0, "episodes": 0}
    t0 = time.time()
    for upd in range(start_step + 1, start_step + total_updates + 1):
        buf.reset()
        ep_rew, ep_len = 0.0, 0
        # collect one rollout
        for _ in range(cfg.rollout_steps):
            style, diff = league.sample()
            env.style, env.difficulty = style, diff
            env.stage = cur.stage
            act, logp, val, _ent = np_sample(policy, obs, dev)
            act = act.reshape(-1)
            obs2, rew, done, _info = env.step(act)
            buf.add(obs, act, float(logp[0]), float(rew), float(done), float(val[0]))
            ep_rew += float(rew)
            ep_len += 1
            obs = obs2
            if done:
                stats["rewards"].append(ep_rew)
                stats["episodes"] += 1
                goal = float(getattr(env.world, "score_home", 0) + getattr(env.world, "score_away", 0))
                stats["goals"] += int(goal > 0)
                # success = agent contributed to attack or survived window
                cur.record(ep_rew > 0.15)
                league.record(env.world.score_home > env.world.score_away)
                obs, _ = env.reset()
                ep_rew, ep_len = 0.0, 0
        with torch.no_grad():
            last_v = float(policy(torch.as_tensor(obs, dtype=torch.float32, device=dev))[2].cpu())
        buf.compute_gae(last_v, cfg.gamma, cfg.gae_lambda)
        diag = ppo_update(policy, opt, buf.flat(), cfg, upd)
        league.step()
        league.maybe_snapshot(policy, upd, cur.current()["difficulty"])
        stats["updates"] = upd
        if upd % 5 == 0 or upd == start_step + 1:
            print(f"[stryk] upd {upd}/{start_step+total_updates} "
                  f"stage={cur.stage}:{cur.current()['name']} "
                  f"pg={diag['policy_loss']:.4f} vl={diag['value_loss']:.4f} "
                  f"ent={diag['entropy']:.3f} clip={diag['clip_frac']:.3f} "
                  f"std={policy.mean_std():.3f}", flush=True)
        if upd % max(1, cfg.save_every) == 0:
            save_checkpoint(cfg.checkpoint, policy, opt, upd, cfg,
                            {"curriculum": cur.summary(), "league": league.summary(),
                             "episodes": stats["episodes"], "goals": stats["goals"]})
            _write_csv(cfg, upd, diag, cur, stats)
            print(f"[stryk] saved {cfg.checkpoint}", flush=True)
    save_checkpoint(cfg.checkpoint, policy, opt, stats["updates"], cfg,
                    {"curriculum": cur.summary(), "league": league.summary(),
                     "episodes": stats["episodes"], "goals": stats["goals"]})
    _write_csv(cfg, stats["updates"], {"policy_loss": 0, "value_loss": 0, "entropy": 0, "clip_frac": 0}, cur, stats)
    print(f"[stryk] done episodes={stats['episodes']} goals={stats['goals']} "
          f"elapsed={time.time()-t0:.1f}s ckpt={cfg.checkpoint}", flush=True)
    return {"checkpoint": cfg.checkpoint, "updates": stats["updates"],
            "episodes": stats["episodes"], "goals": stats["goals"]}


def _write_csv(cfg: TrainCfg, upd: int, diag: dict, cur: Curriculum, stats: dict):
    import os
    os.makedirs("checkpoints", exist_ok=True)
    fresh = not os.path.exists(cfg.telemetry_csv)
    with open(cfg.telemetry_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["update", "stage", "pg", "vl", "ent", "clip", "episodes", "goals"])
        w.writerow([upd, cur.stage, diag.get("policy_loss", 0), diag.get("value_loss", 0),
                    diag.get("entropy", 0), diag.get("clip_frac", 0),
                    stats.get("episodes", 0), stats.get("goals", 0)])
