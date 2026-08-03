# SPDX-License-Identifier: GPL-3.0-or-later
"""Comprehensive element-pair (diatomic) sweep: pm3-rs vs MOPAC.

PM3/PM7 apply per-element-pair empirical core-core scalings (alpb/xfac) plus the
two-center integrals and resonance. A bug in extracting any one pair parameter, or
in the two-center d/sp integral assembly for a particular element combination, shows
up as a diatomic heat-of-formation mismatch. This sweeps every element pair (or just
the parametrized ones), builds a neutral diatomic at ~covalent-bond distance, runs
both codes at the SAME charge/multiplicity, and flags any disagreement.

Usage:
    python pair_sweep.py [params|all] [--tol 0.05] [--workers 8] [--limit N]

  params (default): only the pairs that carry an explicit alpb/xfac (the empirical
                    scalings) — the direct extraction test.
  all:              every element pair among the parametrized elements.

Both codes use the same multiplicity (parity of the valence-electron count), so a
mismatch is a real discrepancy regardless of which spin state is the true ground
state. Non-convergence / collapse is reported separately from clean mismatches.
"""
import argparse
import csv
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import run_mopac  # noqa: E402

CLI = None
for cand in (
    os.path.join(ROOT, "target", "release", "pm3_rs_cli.exe"),
    os.path.join(ROOT, "target", "debug", "pm3_rs_cli.exe"),
):
    if os.path.exists(cand):
        CLI = cand
        break

# Single-bond covalent radii (Å), Cordero 2008 / MOPAC radii_C, Z=1..102.
# Zeros (undefined) fall back to 1.45 Å for bond-length estimation only.
COVR = {
    1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
    10: 0.58, 11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02,
    18: 1.06, 19: 2.03, 20: 1.76, 21: 1.70, 22: 1.60, 23: 1.53, 24: 1.39, 25: 1.50,
    26: 1.42, 27: 1.38, 28: 1.24, 29: 1.32, 30: 1.22, 31: 1.22, 32: 1.20, 33: 1.19,
    34: 1.20, 35: 1.20, 36: 1.16, 37: 2.20, 38: 1.95, 39: 1.90, 40: 1.75, 41: 1.64,
    42: 1.54, 43: 1.47, 44: 1.46, 45: 1.42, 46: 1.39, 47: 1.45, 48: 1.44, 49: 1.42,
    50: 1.39, 51: 1.39, 52: 1.38, 53: 1.39, 54: 1.40, 55: 2.44, 56: 2.15, 57: 2.07,
    72: 1.75, 73: 1.70, 74: 1.62, 75: 1.51, 76: 1.44, 77: 1.41, 78: 1.36, 79: 1.36,
    80: 1.32, 81: 1.45, 82: 1.46, 83: 1.48, 90: 2.06, 92: 1.96,
}
# Lanthanide sparkles (58..71) — large ionic radii.
for z in range(58, 72):
    COVR.setdefault(z, 1.85)


def read_rows(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


def load():
    el = read_rows(os.path.join(ROOT, "src", "data", "pm3_parameters.csv"))
    tore, sym, main = {}, {}, {}
    edata = read_rows(os.path.join(ROOT, "src", "data", "element_data.csv"))
    for r in edata:
        z = int(r["z"])
        tore[z] = float(r["tore"])
        sym[z] = r["sym"]
        main[z] = r["main_group"] == "1"
    elements = sorted(int(r["z"]) for r in el)
    pairs = read_rows(os.path.join(ROOT, "src", "data", "pm3_pair_parameters.csv"))
    param_pairs = {(int(r["zi"]), int(r["zj"])) for r in pairs}
    return elements, tore, sym, main, param_pairs


def one_pair(zi, zj, tore, sym, tmpdir):
    zlo, zhi = min(zi, zj), max(zi, zj)
    r = 1.05 * (COVR.get(zi, 1.45) + COVR.get(zj, 1.45))
    r = max(r, 1.0)
    nelec = tore.get(zi, 0) + tore.get(zj, 0)
    mult = 1 if int(round(nelec)) % 2 == 0 else 2
    name = f"{sym[zi]}{sym[zj]}_{zi}_{zj}"
    xyz = os.path.join(tmpdir, name + ".xyz")
    with open(xyz, "w") as fh:
        fh.write(f"2\n{name}\n{sym[zi]} 0 0 0\n{sym[zj]} 0 0 {r:.5f}\n")
    # MOPAC
    try:
        mo = run_mopac.run(xyz, "PM3", 0, mult, "1scf")
        mo_hof = mo["heat_of_formation_kcal"]
    except Exception as e:
        return (name, zi, zj, mult, None, None, None, f"MOPAC_ERR:{str(e)[:40]}")
    if mo_hof is None:
        return (name, zi, zj, mult, None, None, None, "MOPAC_NO_HOF")
    mo_q0 = None
    if isinstance(mo.get("charges"), list) and mo["charges"]:
        mo_q0 = mo["charges"][0]
    # pm3-rs energy
    try:
        out = subprocess.run(
            [CLI, "energy", xyz, "--charge", "0", "--multiplicity", str(mult)],
            capture_output=True, text=True, timeout=90,
        )
        rs_hof = None
        for line in out.stdout.splitlines():
            if "Heat of formation" in line:
                rs_hof = float(line.split(":")[1].split("kcal")[0])
    except Exception as e:
        return (name, zi, zj, mult, None, mo_hof, None, f"RS_EXC:{str(e)[:40]}")
    if rs_hof is None:
        return (name, zi, zj, mult, None, mo_hof, None, "RS_FAIL")
    # Charge discriminator (heteronuclear): same charge → same state.
    dq = None
    if mo_q0 is not None and zi != zj:
        try:
            cq = subprocess.run(
                [CLI, "charges", xyz, "--charge", "0", "--multiplicity", str(mult)],
                capture_output=True, text=True, timeout=90,
            )
            rs_q0 = None
            for line in cq.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 3 and parts[0] == "1":
                    rs_q0 = float(parts[-1])
                    break
            if rs_q0 is not None:
                dq = abs(rs_q0 - mo_q0)
        except Exception:
            dq = None
    return (name, zi, zj, mult, rs_hof, mo_hof, dq, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="params", choices=["params", "all"])
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    elements, tore, sym, main, param_pairs = load()
    if args.mode == "params":
        pairs = sorted(param_pairs)
    else:
        pairs = [(a, b) for i, a in enumerate(elements) for b in elements[i:]]
    if args.limit:
        pairs = pairs[: args.limit]

    tmpdir = os.path.join(HERE, "_pairsweep")
    os.makedirs(tmpdir, exist_ok=True)
    print(f"Sweeping {len(pairs)} pairs ({args.mode}) with {args.workers} workers, tol={args.tol} kcal")

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one_pair, a, b, tore, sym, tmpdir): (a, b) for a, b in pairs}
        done = 0
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 100 == 0:
                print(f"  ... {done}/{len(pairs)}")

    real_bug, state_diff, homo_mism, rs_fail, mo_fail = [], [], [], [], []
    for name, zi, zj, mult, rs, mo, dq, err in results:
        if err:
            if err.startswith("MOPAC"):
                mo_fail.append((name, err))
            else:
                rs_fail.append((name, mult, mo, err))
            continue
        d = rs - mo
        if abs(d) <= args.tol:
            continue
        row = (abs(d), name, zi, zj, mult, rs, mo, d, dq)
        if zi == zj:
            homo_mism.append(row)  # symmetric → charge can't discriminate
        elif dq is not None and dq < 0.02:
            real_bug.append(row)  # SAME charges but different energy → real bug
        else:
            state_diff.append(row)  # different charges → SCF landed on different state

    for lst in (real_bug, state_diff, homo_mism):
        lst.sort(reverse=True)

    print(f"\n### REAL BUGS: same charges (Δq<0.02) but |ΔE|>{args.tol} : {len(real_bug)} ###")
    for ad, name, zi, zj, mult, rs, mo, d, dq in real_bug:
        print(f"  {name:14s} m={mult} pm3-rs={rs:12.4f} MOPAC={mo:12.4f} d={d:+10.4f} dq={dq:.4f}")

    print(f"\n=== homonuclear mismatches (state-sensitive) : {len(homo_mism)} ===")
    for ad, name, zi, zj, mult, rs, mo, d, dq in homo_mism[:40]:
        print(f"  {name:14s} m={mult} pm3-rs={rs:12.4f} MOPAC={mo:12.4f} d={d:+10.4f}")

    print(f"\n=== different-state (Δq>0.02, convergence) : {len(state_diff)} ===")
    for ad, name, zi, zj, mult, rs, mo, d, dq in state_diff[:40]:
        print(f"  {name:14s} m={mult} pm3-rs={rs:12.4f} MOPAC={mo:12.4f} d={d:+10.4f} dq={dq}")

    print(f"\n=== pm3-rs FAILED (MOPAC ok) : {len(rs_fail)} ===")
    for name, mult, mo, err in rs_fail[:40]:
        print(f"  {name:14s} m={mult} MOPAC={mo} [{err}]")

    print(f"\n=== MOPAC issues (skipped) : {len(mo_fail)} ===")
    for name, err in mo_fail[:20]:
        print(f"  {name:14s} [{err}]")

    nmis = len(real_bug) + len(state_diff) + len(homo_mism)
    ok = len(results) - nmis - len(rs_fail) - len(mo_fail)
    print(f"\nSUMMARY: {ok} match | {len(real_bug)} REAL-BUG | {len(homo_mism)} homo | "
          f"{len(state_diff)} state-diff | {len(rs_fail)} rs-fail | {len(mo_fail)} mopac-skip | of {len(results)}")


if __name__ == "__main__":
    main()
