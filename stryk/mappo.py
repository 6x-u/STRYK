from __future__ import annotations

import numpy as np
import torch


def gae_per_agent(rewards: np.ndarray, dones: np.ndarray, values: np.ndarray,
                  last_val: float, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    # rewards/dones/values: [T, N]; last_val: scalar bootstrap for terminal step
    T, N = rewards.shape
    adv = np.zeros_like(rewards, dtype=np.float32)
    ret = np.zeros_like(rewards, dtype=np.float32)
    last = np.full((N,), float(last_val), dtype=np.float32)
    for t in reversed(range(T)):
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * last * nonterm - values[t]
        adv[t] = delta + gamma * lam * nonterm * adv[t + 1] if t + 1 < T else delta
        ret[t] = adv[t] + values[t]
        last = values[t]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv.astype(np.float32), ret.astype(np.float32)


def mappo_update(policy, optimizer, batch: dict, adv: np.ndarray, ret: np.ndarray, cfg, step: int) -> dict:
    # batch arrays: [T,N,...] — flatten T*N for minibatches :)
    ent = torch.as_tensor(batch["entities"].reshape(-1, batch["entities"].shape[-2], batch["entities"].shape[-1]), dtype=torch.float32)
    msk = torch.as_tensor(batch["mask"].reshape(-1, batch["mask"].shape[-1]), dtype=torch.float32)
    glo = torch.as_tensor(batch["global"].reshape(-1, batch["global"].shape[-1]), dtype=torch.float32)
    ams = torch.as_tensor(batch["action_mask"].reshape(-1, batch["action_mask"].shape[-1]), dtype=torch.float32)
    cont = torch.as_tensor(batch["cont"].reshape(-1, batch["cont"].shape[-1]), dtype=torch.float32)
    disc = torch.as_tensor(batch["disc"].reshape(-1), dtype=torch.long)
    old_lp = torch.as_tensor(batch["logp"].reshape(-1), dtype=torch.float32)
    adv_t = torch.as_tensor(adv.reshape(-1), dtype=torch.float32)
    ret_t = torch.as_tensor(ret.reshape(-1), dtype=torch.float32)
    n = ent.shape[0]
    mb = max(1, int(cfg.minibatch))
    if cfg.total_updates > 0:
        frac = 1.0 - (step / float(max(cfg.total_updates, 1)))
        for g in optimizer.param_groups:
            g["lr"] = float(cfg.lr) * max(frac, 0.1)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
    for _ in range(int(cfg.update_epochs)):
        idx_all = np.random.permutation(n)
        for s in range(0, n, mb):
            idx = idx_all[s:s + mb]
            logp, val, ent_c = policy.evaluate(ent[idx], msk[idx], glo[idx], ams[idx], cont[idx], disc[idx])
            ratio = (logp - old_lp[idx]).exp()
            clipped = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip)
            pg = torch.min(ratio * adv_t[idx], clipped * adv_t[idx])
            loss = -pg.mean() + cfg.value_coef * torch.nn.functional.smooth_l1_loss(val, ret_t[idx]) - cfg.entropy_coef * ent_c.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad)
            optimizer.step()
    with torch.no_grad():
        logp, val, ent_c = policy.evaluate(ent, msk, glo, ams, cont, disc)
        ratio = (logp - old_lp).exp()
        clipped = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip)
        pg = torch.min(ratio * adv_t, clipped * adv_t)
        stats["policy_loss"] = float((-pg.mean()).cpu())
        stats["value_loss"] = float(torch.nn.functional.smooth_l1_loss(val, ret_t).cpu())
        stats["entropy"] = float(ent_c.mean().cpu())
        stats["clip_frac"] = float(((ratio < 1 - cfg.clip) | (ratio > 1 + cfg.clip)).float().mean().cpu())
    return stats
