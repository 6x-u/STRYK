from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import numpy as np
import torch

from .config import TrainCfg, model_layout
from .curriculum import Curriculum
from .multi_env import MultiEnv
from .policy_multi import MultiAgentPolicy, np_sample_multi
from .mappo import gae_per_agent, mappo_update
from .checkpoint import save_latest, save_best, load_checkpoint, promote_to_final, export_final_onnx
from .metrics import EpisodeStats
from .replay import ReplayStore
from .selfplay import League
from .benchmark import benchmark_checkpoint
from .version import __version__


def _pick_device(cfg: TrainCfg) -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_multi(cfg: TrainCfg, resume: str | None = None) -> dict:
    dev = _pick_device(cfg)
    # kaggle gpus are greedy — scale the rollout up when we actually have one :)
    rollout = int(cfg.rollout_steps)
    if dev == "cuda" and rollout < 4096:
        rollout = 4096
    print(f"[stryk] multi train mode={cfg.mode} stadium={cfg.stadium} dev={dev} v={__version__} rollout={rollout}", flush=True)
    cur = Curriculum(start=cfg.stage, enabled=cfg.curriculum)
    league = League(seed=cfg.seed)
    mode = cur.mode_for() if cfg.curriculum else cfg.mode
    env = MultiEnv(mode, cfg.stadium, cfg.seed, stage=cur.stage)
    policy = MultiAgentPolicy(hidden=int(cfg.hidden)).to(dev)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(cfg.total_updates), 1))
    start = 0
    if resume:
        payload = load_checkpoint(resume, policy, opt, device=dev, scheduler=sched)
        start = int(payload.get("step", 0))
        print(f"[stryk] resumed step {start} from {resume}", flush=True)
    N = len(env.home_ids())
    # rollout storage [T,N,...] — T*N feeds the minibatches
    T = max(64, rollout // max(N, 1))
    replay = ReplayStore(capacity=max(20000, rollout * 2))
    stats = EpisodeStats()
    best_score = -1e18
    t_start = time.time()
    budget = float(cfg.train_hours) * 3600.0 if float(cfg.train_hours) > 0 else 0.0
    obs = env.reset()
    for upd in range(start + 1, start + int(cfg.total_updates) + 1):
        if budget and (time.time() - t_start) > budget:
            print("[stryk] hour budget hit — wrapping up early", flush=True)
            break
        # sync env mode to curriculum stage (fine-tuning > from-scratch every phase)
        want_mode = cur.mode_for()
        if want_mode != env.mode:
            env = MultiEnv(want_mode, cfg.stadium, cfg.seed + upd)
            obs = env.reset()
        T0 = T
        ents = np.zeros((T0, N, obs["entities"].shape[1], obs["entities"].shape[2]), dtype=np.float32)
        msks = np.zeros((T0, N, obs["mask"].shape[1]), dtype=np.float32)
        glos = np.zeros((T0, N, obs["global"].shape[1]), dtype=np.float32)
        amss = np.zeros((T0, N, obs["action_mask"].shape[1]), dtype=np.float32)
        conts = np.zeros((T0, N, 10), dtype=np.float32)
        discs = np.zeros((T0, N), dtype=np.int64)
        logps = np.zeros((T0, N), dtype=np.float32)
        rews = np.zeros((T0, N), dtype=np.float32)
        dones = np.zeros((T0, N), dtype=np.float32)
        vals = np.zeros((T0, N), dtype=np.float32)
        ep_rew = 0.0
        for t in range(T0):
            style, diff = league.sample()
            env.style, env.difficulty = style, diff
            batch = {"entities": obs["entities"], "mask": obs["mask"],
                     "global": obs["global"], "action_mask": obs["action_mask"]}
            cont, disc, logp, val, _ = np_sample_multi(policy, batch, dev)
            obs2, rew, done, info = env.step(cont, disc)
            ents[t], msks[t], glos[t], amss[t] = obs["entities"], obs["mask"], obs["global"], obs["action_mask"]
            conts[t], discs[t], logps[t], rews[t], vals[t] = cont, disc, logp, rew, val
            dones[t] = float(done)
            replay.push({"entities": obs["entities"].reshape(-1), "actions": cont.reshape(-1),
                         "reward": float(np.sum(rew))})
            ep_rew += float(np.sum(rew))
            obs = obs2
            if done:
                stats.add_episode(env.world, env.tracker)
                cur.record(ep_rew > 0.2)
                league.record(env.world.score_home > env.world.score_away)
                obs = env.reset()
                ep_rew = 0.0
        with torch.no_grad():
            ent_t = torch.as_tensor(obs["entities"], dtype=torch.float32, device=dev)
            msk_t = torch.as_tensor(obs["mask"], dtype=torch.float32, device=dev)
            glo_t = torch.as_tensor(obs["global"], dtype=torch.float32, device=dev)
            _, _, last_v = policy(ent_t, msk_t, glo_t)
            last_val = float(last_v.mean().cpu())
        adv, ret = gae_per_agent(rews, dones, vals, last_val, cfg.gamma, cfg.gae_lambda)
        diag = mappo_update(policy, opt, {"entities": ents, "mask": msks, "global": glos,
                                          "action_mask": amss, "cont": conts, "disc": discs,
                                          "logp": logps}, adv, ret, cfg, upd)
        sched.step()
        league.step()
        league.maybe_snapshot(policy, upd, cur.current()["difficulty"])
        if upd % 5 == 0 or upd == start + 1:
            print(f"[stryk] upd {upd} stage={cur.stage}:{cur.current()['name']} mode={env.mode} "
                  f"pg={diag['policy_loss']:.4f} vl={diag['value_loss']:.4f} "
                  f"ent={diag['entropy']:.3f} clip={diag['clip_frac']:.3f} std={policy.mean_std():.3f}", flush=True)
        # triple save rhythm: latest every save_every, bench-gated best, final at the end
        if upd % max(1, int(cfg.save_every)) == 0:
            extra = {"curriculum": cur.summary(), "league": league.summary(), "metrics": stats.summary()}
            save_latest(policy, opt, upd, cfg, extra, scheduler=sched)
            print(f"[stryk] saved STRYK_latest.pt @ {upd}", flush=True)
            # pop-benchmark every eval_every — cheap gate for best
            if upd % max(1, int(cfg.eval_every)) == 0:
                lay = model_layout(cfg.models_root)
                latest = lay["checkpoints"] / f"{getattr(cfg, 'model_name', 'STRYK')}_latest.pt"
                bench = benchmark_checkpoint(str(latest), modes=(env.mode,), episodes_per=1, device=dev, fast_sec=10.0)
                score = float(bench["win_rate"] * 2.0 + bench.get("goal_diff", 0.0) + bench.get("pass_completion", 0.0))
                print(f"[stryk] bench score={score:.3f} wins={bench['wins']}/{bench['wins']+bench['draws']+bench['losses']}", flush=True)
                if score > best_score:
                    best_score = score
                    save_best(policy, opt, upd, cfg, score, extra, scheduler=sched)
                    print("[stryk] saved STRYK_best.pt", flush=True)
        _write_csv(cfg, upd, diag, cur, stats)
    # final promotion: best only goes final if it actually beat the bar :)
    extra = {"curriculum": cur.summary(), "league": league.summary(), "metrics": stats.summary()}
    save_latest(policy, opt, upd if "upd" in dir() else start, cfg, extra, scheduler=sched)
    lay = model_layout(cfg.models_root)
    best_p = lay["checkpoints"] / f"{getattr(cfg, 'model_name', 'STRYK')}_best.pt"
    if best_p.exists():
        bench = benchmark_checkpoint(str(best_p), modes=(env.mode,), episodes_per=1, device=dev, fast_sec=10.0)
        promote_to_final(best_p, cfg, bench)
        export_final_onnx(policy, cfg, device=dev)
        print("[stryk] exported STRYK_final.onnx + manifest", flush=True)
    print(f"[stryk] done episodes={stats.episodes} elapsed={time.time()-t_start:.1f}s", flush=True)
    return {"updates": int(upd) if "upd" in dir() else start, "episodes": stats.episodes,
            "best_score": best_score}


def _write_csv(cfg: TrainCfg, upd: int, diag: dict, cur: Curriculum, stats: EpisodeStats):
    lay = model_layout(cfg.models_root)
    lay["metadata"].mkdir(parents=True, exist_ok=True)
    p = Path(cfg.telemetry_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    fresh = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(["update", "stage", "mode", "pg", "vl", "ent", "clip", "episodes"])
        w.writerow([upd, cur.stage, cur.mode_for(), diag.get("policy_loss", 0), diag.get("value_loss", 0),
                    diag.get("entropy", 0), diag.get("clip_frac", 0), stats.episodes])
