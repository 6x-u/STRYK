from __future__ import annotations

from pathlib import Path


def save_topdown(world, out_path: str | Path, w: int = 800, h: int = 600):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pix = bytearray(w * h * 3)
    sx, sy = w / 112.0, h / 84.0
    ox, oy = w / 2.0, h / 2.0

    def fill(cx, cy, r, rgb):
        x0 = max(0, int(ox + cx * sx - r))
        x1 = min(w - 1, int(ox + cx * sx + r))
        y0 = max(0, int(oy + cy * sy - r))
        y1 = min(h - 1, int(oy + cy * sy + r))
        for y in range(y0, y1 + 1):
            base = (y * w + x0) * 3
            for x in range(x0, x1 + 1):
                base = (y * w + x) * 3
                pix[base] = rgb[0]
                pix[base + 1] = rgb[1]
                pix[base + 2] = rgb[2]

    # grass
    for y in range(h):
        base = y * w * 3
        for x in range(w):
            pix[base + x * 3] = 28
            pix[base + x * 3 + 1] = 96
            pix[base + x * 3 + 2] = 28
    for pl in world.players:
        c = (64, 120, 255) if pl.side == 0 else (255, 64, 64)
        fill(float(pl.pos[0]), float(pl.pos[1]), 6, c)
    fill(float(world.ball.pos[0]), float(world.ball.pos[1]), 5, (255, 255, 255))
    header = f"P6\n{w} {h}\n255\n".encode()
    p.write_bytes(header + bytes(pix))
    return p
