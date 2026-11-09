from __future__ import annotations

import argparse
import json

from .config import TrainCfg, load_config
from .version import version_string


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stryk", description=version_string())
    sub = ap.add_subparsers(dest="cmd", required=False)
    tr = sub.add_parser("train")
    tr.add_argument("--config", default=None)
    tr.add_argument("--mode", default="5v5")
    tr.add_argument("--stadium", default="wembley_fc27")
    tr.add_argument("--seed", type=int, default=2026)
    tr.add_argument("--updates", type=int, default=200)
    tr.add_argument("--rollout", type=int, default=2048)
    tr.add_argument("--resume", default=None)
    tr.add_argument("--checkpoint", default="checkpoints/last.pt")
    tr.add_argument("--stage", type=int, default=0)
    tm = sub.add_parser("train-multi")
    tm.add_argument("--config", default=None)
    tm.add_argument("--mode", default="5v5")
    tm.add_argument("--stadium", default="wembley_fc27")
    tm.add_argument("--seed", type=int, default=2026)
    tm.add_argument("--updates", type=int, default=200)
    tm.add_argument("--rollout", type=int, default=2048)
    tm.add_argument("--resume", default=None)
    tm.add_argument("--hours", type=float, default=0.0)
    tm.add_argument("--models-root", default="Models")
    tm.add_argument("--stage", type=int, default=0)
    ev = sub.add_parser("eval")
    ev.add_argument("--checkpoint", required=True)
    ev.add_argument("--mode", default="5v5")
    ev.add_argument("--stadium", default="wembley_fc27")
    ev.add_argument("--episodes", type=int, default=5)
    bm = sub.add_parser("benchmark")
    bm.add_argument("--checkpoint", required=True)
    bm.add_argument("--modes", default="1v1,5v5")
    bm.add_argument("--episodes", type=int, default=1)
    bm.add_argument("--fast-sec", type=float, default=0.0)
    be = sub.add_parser("bench")
    be.add_argument("--mode", default="5v5")
    be.add_argument("--stadium", default="wembley_fc27")
    be.add_argument("--seconds", type=int, default=3)
    ex = sub.add_parser("export")
    ex.add_argument("--checkpoint", required=True)
    ex.add_argument("--stem", default="checkpoints/policy")
    ef = sub.add_parser("export-final")
    ef.add_argument("--checkpoint", required=True)
    ef.add_argument("--models-root", default="Models")
    args = ap.parse_args(argv)
    if args.cmd is None:
        ap.print_help()
        return 0
    if args.cmd == "train":
        cfg = load_config(args.config) if args.config else TrainCfg(
            mode=args.mode, stadium=args.stadium, seed=args.seed,
            total_updates=args.updates, rollout_steps=args.rollout,
            checkpoint=args.checkpoint, stage=args.stage)
        if args.checkpoint and not args.config:
            cfg.checkpoint = args.checkpoint
        from .train import train
        train(cfg, resume=args.resume)
    elif args.cmd == "train-multi":
        cfg = load_config(args.config) if args.config else TrainCfg(
            mode=args.mode, stadium=args.stadium, seed=args.seed,
            total_updates=args.updates, rollout_steps=args.rollout,
            stage=args.stage, use_mappo=True, train_hours=float(args.hours),
            models_root=args.models_root)
        if not args.config:
            cfg.train_hours = float(args.hours)
            cfg.models_root = args.models_root
        from .train_multi import train_multi
        train_multi(cfg, resume=args.resume)
    elif args.cmd == "eval":
        from .eval import evaluate
        print(json.dumps(evaluate(args.checkpoint, args.mode, args.stadium, args.episodes), indent=2))
    elif args.cmd == "benchmark":
        from .benchmark import benchmark_checkpoint
        modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())
        print(json.dumps(benchmark_checkpoint(args.checkpoint, modes=modes, episodes_per=args.episodes, fast_sec=float(args.fast_sec)), indent=2))
    elif args.cmd == "bench":
        from .eval import bench
        print(json.dumps(bench(args.mode, args.stadium, args.seconds), indent=2))
    elif args.cmd == "export":
        from .policy import ActorCritic
        from .checkpoint import load_checkpoint, export_both
        from .obs import OBS_DIM
        p = ActorCritic()
        load_checkpoint(args.checkpoint, p, None)
        print(export_both(p, OBS_DIM, args.stem))
    elif args.cmd == "export-final":
        from .policy_multi import MultiAgentPolicy
        from .checkpoint import load_checkpoint, export_final_onnx
        import torch as _t
        _pay = _t.load(args.checkpoint, map_location="cpu", weights_only=False)
        _hid = int(_pay.get("config", {}).get("hidden", 256))
        p = MultiAgentPolicy(hidden=_hid)
        load_checkpoint(args.checkpoint, p, None)
        print(export_final_onnx(p, TrainCfg(models_root=args.models_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
