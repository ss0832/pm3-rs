# SPDX-License-Identifier: GPL-3.0-or-later
"""Independently verify every PM3 pairwise alpb/xfac against the MOPAC source.

This is a static, SCF-free check of the empirical element-pair core-core scalings:
parse `alpb(i,j)`/`xfac(i,j)` directly from parameters_for_PM3_C.F90 and diff against
the extracted src/data/pm3_pair_parameters.csv. Any extraction error (wrong value,
missing pair, extra pair, swapped index) is reported. No quantum calculation involved.
"""
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "third_party", "mopac-source", "src", "models", "parameters_for_PM3_C.F90")
CSV = os.path.join(ROOT, "src", "data", "pm3_pair_parameters.csv")

PAT = re.compile(r"\b(alpb|xfac)\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*([-\d.]+)[dDeE]?([-+]?\d*)")


def parse_source():
    alpb, xfac = {}, {}
    with open(SRC, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = PAT.search(line)
            if not m:
                continue
            kind, i, j, mant, exp = m.groups()
            val = float(mant + ("e" + exp if exp else ""))
            key = (int(i), int(j))
            (alpb if kind == "alpb" else xfac)[key] = val
    return alpb, xfac


def parse_csv():
    alpb, xfac = {}, {}
    with open(CSV, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(l for l in fh if not l.startswith("#")):
            key = (int(r["zi"]), int(r["zj"]))
            alpb[key] = float(r["alpb"])
            xfac[key] = float(r["xfac"])
    return alpb, xfac


def norm(key):
    return (max(key), min(key))


def main():
    src_a, src_x = parse_source()
    csv_a, csv_x = parse_csv()
    print(f"source: {len(src_a)} alpb, {len(src_x)} xfac ; csv: {len(csv_a)} alpb, {len(csv_x)} xfac")

    # Normalize both to (max,min) keys for comparison.
    def normmap(d):
        out = {}
        for k, v in d.items():
            out[norm(k)] = v
        return out

    sa, sx = normmap(src_a), normmap(src_x)
    ca, cx = normmap(csv_a), normmap(csv_x)

    problems = 0
    all_keys = sorted(set(sa) | set(ca) | set(sx) | set(cx))
    for k in all_keys:
        for name, s, c in (("alpb", sa, ca), ("xfac", sx, cx)):
            sv, cv = s.get(k), c.get(k)
            if sv is None and cv is None:
                continue
            if sv is None:
                print(f"  {k} {name}: in CSV ({cv}) but NOT in source")
                problems += 1
            elif cv is None:
                print(f"  {k} {name}: in source ({sv}) but MISSING from CSV")
                problems += 1
            elif abs(sv - cv) > 1e-6:
                print(f"  {k} {name}: source={sv} csv={cv} d={cv - sv:+.6f}")
                problems += 1

    print(f"\n{'OK — all pair parameters match the source.' if problems == 0 else f'{problems} PROBLEM(S) FOUND.'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
