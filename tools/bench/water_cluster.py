# SPDX-License-Identifier: GPL-3.0-or-later
"""Generate a roughly spherical water cluster of N molecules on a jittered lattice."""
import math, sys, random

n_mol = int(sys.argv[1])
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260826
rng = random.Random(seed)

# Rigid TIP-like geometry (Angstrom), PM3 water is close enough for a benchmark.
OH, ANG = 0.957, math.radians(104.5)
base = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (OH, 0.0, 0.0)),
    ("H", (OH * math.cos(ANG), OH * math.sin(ANG), 0.0)),
]

spacing = 3.1  # roughly the O-O distance in liquid water
side = math.ceil(n_mol ** (1 / 3))
sites = [
    (i * spacing, j * spacing, k * spacing)
    for i in range(side)
    for j in range(side)
    for k in range(side)
]
centre = tuple(spacing * (side - 1) / 2 for _ in range(3))
sites.sort(key=lambda s: sum((a - b) ** 2 for a, b in zip(s, centre)))
sites = sites[:n_mol]

def random_rotation():
    # Uniform random rotation matrix from a random quaternion.
    u1, u2, u3 = rng.random(), rng.random(), rng.random()
    q = (
        math.sqrt(1 - u1) * math.sin(2 * math.pi * u2),
        math.sqrt(1 - u1) * math.cos(2 * math.pi * u2),
        math.sqrt(u1) * math.sin(2 * math.pi * u3),
        math.sqrt(u1) * math.cos(2 * math.pi * u3),
    )
    x, y, z, w = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]

lines = []
for sx, sy, sz in sites:
    r = random_rotation()
    jitter = [rng.uniform(-0.15, 0.15) for _ in range(3)]
    for sym, (px, py, pz) in base:
        x = r[0][0] * px + r[0][1] * py + r[0][2] * pz + sx + jitter[0]
        y = r[1][0] * px + r[1][1] * py + r[1][2] * pz + sy + jitter[1]
        z = r[2][0] * px + r[2][1] * py + r[2][2] * pz + sz + jitter[2]
        lines.append(f"{sym} {x:.6f} {y:.6f} {z:.6f}")

print(len(lines))
print(f"water cluster, {n_mol} molecules, seed {seed}")
print("\n".join(lines))