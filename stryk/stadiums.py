from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StadiumSpec:
    name: str
    display: str
    length: float = 105.0
    width: float = 68.0
    turf_grip: float = 0.95       # 1 dry .. 0.4 soaked
    restitution: float = 0.72
    wind_amp: float = 1.0
    bowl_swirl: float = 1.0
    wet_prob: float = 0.0
    drag_mult: float = 1.0
    ball_mass: float = 0.43


STADIUMS: dict[str, StadiumSpec] = {
    "bernabeu_fc27": StadiumSpec(
        "bernabeu_fc27", "Santiago Bernabeu FC27",
        turf_grip=0.99, restitution=0.70, wind_amp=0.6, bowl_swirl=0.6, drag_mult=1.0),
    "campnou_fc27": StadiumSpec(
        "campnou_fc27", "Spotify Camp Nou FC27",
        turf_grip=0.96, restitution=0.72, wind_amp=0.9, bowl_swirl=0.9),
    "etihad_fc27": StadiumSpec(
        "etihad_fc27", "Etihad FC27",
        turf_grip=0.95, restitution=0.73, wind_amp=1.2, bowl_swirl=1.1, wet_prob=0.25),
    "wembley_fc27": StadiumSpec(
        "wembley_fc27", "Wembley FC27 Arch",
        turf_grip=0.93, restitution=0.72, wind_amp=1.4, bowl_swirl=1.3, wet_prob=0.15),
    "signal_fc27": StadiumSpec(
        "signal_fc27", "Signal Iduna FC27 Wall",
        turf_grip=0.92, restitution=0.74, wind_amp=1.0, bowl_swirl=0.8),
    "allianz_fc27": StadiumSpec(
        "allianz_fc27", "Allianz Arena FC27",
        turf_grip=0.97, restitution=0.71, wind_amp=0.7, bowl_swirl=0.7),
    "sansiro_fc27": StadiumSpec(
        "sansiro_fc27", "San Siro FC27",
        turf_grip=0.94, restitution=0.72, wind_amp=1.1, bowl_swirl=1.0, wet_prob=0.2),
    "cruyff_fc27": StadiumSpec(
        "cruyff_fc27", "Johan Cruyff FC27",
        turf_grip=0.95, restitution=0.72, wind_amp=0.8, bowl_swirl=0.8),
    "meadow_fc27": StadiumSpec(
        "meadow_fc27", "Meadow Park FC27",
        turf_grip=0.90, restitution=0.75, wind_amp=1.6, bowl_swirl=1.5, wet_prob=0.35),
    "dome_fc27": StadiumSpec(
        "dome_fc27", "FC27 Dome 2100",
        turf_grip=1.00, restitution=0.69, wind_amp=0.0, bowl_swirl=0.0, drag_mult=0.98),
    "desert_fc27": StadiumSpec(
        "desert_fc27", "Desert Bowl FC27",
        turf_grip=0.88, restitution=0.68, wind_amp=1.8, bowl_swirl=1.6, drag_mult=1.06),
    "alpine_fc27": StadiumSpec(
        "alpine_fc27", "Alpine FC27 Altitude",
        turf_grip=0.96, restitution=0.72, wind_amp=1.3, bowl_swirl=1.2, drag_mult=0.95),
}


# SDK registry: user grounds can plug in without touching core.
_CUSTOM: dict[str, StadiumSpec] = {}


def register_stadium(spec: StadiumSpec) -> StadiumSpec:
    _CUSTOM[spec.name] = spec
    STADIUMS[spec.name] = spec
    return spec


def all_stadiums() -> dict[str, StadiumSpec]:
    return dict(STADIUMS)


def get_stadium(name: str) -> StadiumSpec:
    try:
        return STADIUMS[name]
    except KeyError as e:
        raise KeyError(f"unknown stadium {name!r}, known: {sorted(STADIUMS)}") from e
