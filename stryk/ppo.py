from __future__ import annotations

import numpy as np
import torch


def ppo_update(policy, optimizer, batch: dict, cfg, step: int) -> dict:
    dev = next(policy.parameters()).device
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=dev)
    act = torch.as_tensor(batch["act"], dtype=torch.float32, device=dev)
    old_lp = torch.as_tensor(batch["logp"], dtype=torch.float32, device=dev)
    ret = torch.as_tensor(batch["ret"], dtype=torch.float32, device=dev)
    adv = torch.as_tensor(batch["adv"], dtype=torch.float32, device=dev)
    n = obs.shape[0]
    mb = max(1, int(cfg.minibatch))
    order = np.random.permutation(n)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
    rounds = 0
    # LR anneal over total updates
    if cfg.total_updates > 0:
        frac = 1.0 - (step / float(max(cfg.total_updates, 1)))
        for g in optimizer.param_groups:
            g["lr"] = float(cfg.lr) * max(frac, 0.1)
    for ep in range(int(cfg.update_epochs)):
        for s in range(0, n, mb):
            idx = order[s:s + mb]
            ob, ac = obs[idx], act[idx]
            logp, val, ent = policy.evaluate(ob, ac)
            ratio = (logp - old_lp[idx]).exp()
            clipped = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip)
            pg = torch.min(ratio * adv[idx], clipped * adv[idx])
            pg_loss = -pg.mean()
            v_loss = torch.nn.functional.smooth_l1_loss(val, ret[idx])
            loss = pg_loss + cfg.value_coef * v_loss - cfg.entropy_coef * ent.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad)
            optimizer.step()
            rounds += 1
    # aggregate diagnostics on full batch
    with torch.no_grad():
        logp, val, ent = policy.evaluate(obs, act)
        ratio = (logp - old_lp).exp()
        clipped = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip)
        pg = torch.min(ratio * adv, clipped * adv)
        stats["policy_loss"] = float((-pg.mean()).cpu())
        stats["value_loss"] = float(torch.nn.functional.smooth_l1_loss(val, ret).cpu())
        stats["entropy"] = float(ent.mean().cpu())
        stats["clip_frac"] = float(((ratio < 1 - cfg.clip) | (ratio > 1 + cfg.clip)).float().mean().cpu())
    return stats
