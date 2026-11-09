from __future__ import annotations

from .stadiums import register_stadium as _reg_stadium
from .opponents import register_opponent_style as _reg_style

_REWARD_HOOKS: list = []


def register_stadium(fn):
    spec = fn()
    return _reg_stadium(spec)


def register_opponent_style(name: str, fn):
    return _reg_style(name, fn)


def register_reward(fn):
    _REWARD_HOOKS.append(fn)
    return fn


def reward_hooks() -> list:
    return list(_REWARD_HOOKS)


def hello() -> str:
    return "STRYK SDK ok"
