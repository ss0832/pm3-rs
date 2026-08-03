# SPDX-License-Identifier: GPL-3.0-or-later
"""Extract Grimme D3 reference data from MOPAC v23.2.5 dftd3 sources into CSVs.

Usage: python extract_d3_data.py <copyc6.F90> <dftd3_bits.F90>

Emits under src/data/:
  d3_c6_reference.csv  — pars tuples (iat, jat, c6, cn_a, cn_b)  [Grimme C6 refs]
  d3_radii.csv         — per-element r2r4 and rcov

PROVENANCE: openmopac/mopac v23.2.5, src/corrections/{copyc6,dftd3_bits}.F90
(Apache-2.0; the D3 model and its reference data are Grimme et al., adapted by
MOPAC with permission). See THIRD_PARTY_NOTICES.md.
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "src", "data")

PROV = (
    "# PROVENANCE: Grimme D3 reference data via MOPAC v23.2.5 "
    "(github.com/openmopac/mopac tag v23.2.5, Apache-2.0), {src}.\n"
    "# The D3 model and reference data are S. Grimme et al., J. Chem. Phys. 132, "
    "154104 (2010); see THIRD_PARTY_NOTICES.md.\n"
)


def conv(t):
    return float(t.replace("D", "E").replace("d", "e"))


def decode_elem(iat):
    """Decode a DFTD3 `pars` element index to the real atomic number.

    DFTD3/MOPAC `copyc6` encode the reference-geometry index in the element
    number: the `limit` subroutine subtracts 100 from `iat` once per extra
    reference (iadr), so `iat = Z + 100*(iadr-1)`. The real element is therefore
    `((iat-1) mod 100) + 1` (max_elem = 94, so no collision), and the stored
    `cn_a`/`cn_b` are that reference's coordination numbers.
    """
    return ((iat - 1) % 100) + 1


def extract_c6(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    # data pars( a: b) / iat, jat, c6, cna, cnb /   (iat/jat encode iadr/jadr)
    rows = []
    for m in re.finditer(r"data\s+pars\(\s*\d+\s*:\s*\d+\s*\)\s*/([^/]+)/", text, re.I):
        toks = [t.strip() for t in m.group(1).split(",")]
        if len(toks) != 5:
            raise ValueError(f"bad pars tuple: {m.group(1)!r}")
        iat = decode_elem(int(float(toks[0])))
        jat = decode_elem(int(float(toks[1])))
        c6, cna, cnb = conv(toks[2]), conv(toks[3]), conv(toks[4])
        rows.append((iat, jat, c6, cna, cnb))
    return rows


def extract_list(text, name, n):
    """Extract a Fortran list array `name = (/ ... /)` or data-initialized."""
    # Try `name = (/ v1, v2, ... /)` possibly across continuation lines.
    m = re.search(rf"{name}\s*=\s*\(/(.*?)/\)", text, re.I | re.S)
    if not m:
        # Try data statement
        m = re.search(rf"data\s+{name}\s*/(.*?)/", text, re.I | re.S)
    if not m:
        return None
    body = m.group(1).replace("&", " ").replace("\n", " ")
    vals = []
    for tok in body.split(","):
        tok = tok.strip()
        if not tok:
            continue
        # handle N*x repeats
        if "*" in tok:
            rep, v = tok.split("*")
            vals.extend([conv(v)] * int(rep.strip()))
        else:
            vals.append(conv(tok))
    return vals[:n] if n else vals


def main():
    copyc6, bits = sys.argv[1], sys.argv[2]
    os.makedirs(DATA, exist_ok=True)

    rows = extract_c6(copyc6)
    with open(os.path.join(DATA, "d3_c6_reference.csv"), "w", newline="\n") as fh:
        fh.write(PROV.format(src="src/corrections/copyc6.F90"))
        fh.write("iat,jat,c6,cn_a,cn_b\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]!r},{r[3]!r},{r[4]!r}\n")
    print(f"c6 reference: {len(rows)} tuples")

    # r0ab packed lower triangle from setr0ab (dftd3_bits.F90).
    bits_text = open(bits, encoding="utf-8", errors="replace").read()
    r0 = []
    for m in re.finditer(r"r0ab\(\s*\d+\s*:\s*\d+\s*\)\s*=\s*\(/(.*?)/\)", bits_text, re.S):
        for tok in m.group(1).replace("&", " ").split(","):
            tok = tok.strip()
            if tok:
                r0.append(conv(tok))
    if r0:
        with open(os.path.join(DATA, "d3_r0ab.csv"), "w", newline="\n") as fh:
            fh.write(PROV.format(src="src/corrections/dftd3_bits.F90 (setr0ab)"))
            fh.write("# packed lower triangle r0ab(k), k=1..4465 (Angstrom); "
                     "index into 94x94 symmetric via k=i(i-1)/2+j, i>=j\n")
            fh.write("value\n")
            for v in r0:
                fh.write(f"{v!r}\n")
        print(f"r0ab: {len(r0)} packed values")

    # r2r4/rcov live in the dftd3.F90 driver (arg 3).
    text = open(sys.argv[3], encoding="utf-8", errors="replace").read()
    r2r4 = extract_list(text, "r2r4", 94)
    rcov = extract_list(text, "rcov", 94)
    with open(os.path.join(DATA, "d3_radii.csv"), "w", newline="\n") as fh:
        fh.write(PROV.format(src="src/corrections/dftd3_bits.F90"))
        fh.write("z,r2r4,rcov\n")
        n = max(len(r2r4 or []), len(rcov or []))
        for i in range(n):
            z = i + 1
            a = r2r4[i] if r2r4 and i < len(r2r4) else 0.0
            b = rcov[i] if rcov and i < len(rcov) else 0.0
            fh.write(f"{z},{a!r},{b!r}\n")
    print(f"r2r4: {len(r2r4 or [])}, rcov: {len(rcov or [])}")


if __name__ == "__main__":
    main()
