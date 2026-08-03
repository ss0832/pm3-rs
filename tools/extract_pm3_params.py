#!/usr/bin/env python3
"""Extract PM3 parameters from the MOPAC v23.2.5 Fortran sources."""

from __future__ import annotations

import csv
import os
import re
import sys
from pathlib import Path

REPO_URL = "https://github.com/openmopac/mopac"
TAG = "v23.2.5"
LICENSE = "Apache-2.0"
SCRIPT = "tools/extract_pm3_params.py"
NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[DdEe][-+]?\d+)?"

SYMBOLS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Mi", "XX", "+3",
    "-3", "Cb", "++", "+", "--", "-", "Tv",
]

PM3_NAMES = {
    "usspm3", "upppm3", "zspm3", "zppm3", "betasp", "betapp",
    "betadp", "alppm3", "gsspm3", "gsppm3", "gpppm3", "gp2pm3",
    "hsppm3", "polvolpm3", "zsnpm3", "zpnpm3", "zdnpm3",
    "f0sdpm3", "g2sdpm3", "guesp1", "guesp2", "guesp3",
}
SPARKLE_NAMES = {
    "gsspm3sp", "alppm3sp", "guespm3sp1", "guespm3sp2",
    "guespm3sp3",
}


def normalized(value: str) -> str:
    value = value.strip()
    mantissa, *exponent = re.split(r"[DdEe]", value)
    return mantissa if not exponent or int(exponent[0]) == 0 else f"{mantissa}e{int(exponent[0])}"


def parse_data(path: Path, allowed: set[str]) -> dict[tuple[str, int, int | None], str]:
    pattern = re.compile(
        rf"^\s*data\s+([A-Za-z_]\w*)\s*\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\)\s*/\s*({NUM})\s*/",
        re.IGNORECASE,
    )
    values: dict[tuple[str, int, int | None], str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = pattern.match(line)
        if not match:
            continue
        name = match.group(1).lower()
        if name not in allowed:
            continue
        key = (name, int(match.group(2)), int(match.group(3)) if match.group(3) else None)
        if key in values:
            raise RuntimeError(f"{path}:{number}: duplicate {key}")
        values[key] = normalized(match.group(4))
    return values


def parse_pairs(path: Path) -> dict[tuple[int, int], tuple[str, str]]:
    pattern = re.compile(
        rf"^\s*(alpb|xfac)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*({NUM})",
        re.IGNORECASE,
    )
    partial: dict[tuple[int, int], dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            key = (int(match.group(2)), int(match.group(3)))
            partial.setdefault(key, {})[match.group(1).lower()] = normalized(match.group(4))
    result = {}
    for key, pair in partial.items():
        if set(pair) != {"alpb", "xfac"}:
            raise RuntimeError(f"incomplete alpb/xfac pair {key}")
        result[key] = (pair["alpb"], pair["xfac"])
    return result


def provenance(*sources: str) -> list[str]:
    return [
        "# PROVENANCE: extracted from MOPAC (Molecular Orbital PACkage)",
        f"# source repo: {REPO_URL}  tag: {TAG}",
        *(f"# source file: {source}" for source in sources),
        f"# license: {LICENSE}",
        f"# extraction script: {SCRIPT}",
        "# note: values are published scientific constants; see THIRD_PARTY_NOTICES.md",
    ]


def write_csv(path: Path, sources: tuple[str, ...], header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        for line in provenance(*sources):
            stream.write(line + "\n")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: extract_pm3_params.py <mopac-source-root>")
    root = Path(sys.argv[1]).resolve()
    project = Path(__file__).resolve().parents[1]
    data_dir = project / "src" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    pm3_path = root / "src" / "models" / "parameters_for_PM3_C.F90"
    sparkle_path = root / "src" / "models" / "parameters_for_PM3_Sparkles_C.F90"
    pair_path = root / "src" / "models" / "alpb_and_xfac_pm3.F90"
    common_path = root / "src" / "models" / "parameters_C.F90"
    for path in (pm3_path, sparkle_path, pair_path, common_path):
        if not path.is_file():
            raise SystemExit(f"missing MOPAC source file: {path}")

    pm3 = parse_data(pm3_path, PM3_NAMES)
    sparkles = parse_data(sparkle_path, SPARKLE_NAMES)
    pairs = parse_pairs(pair_path)
    eheat_sparkles = parse_data(common_path, {"eheat_sparkles"})

    def get(table: dict[tuple[str, int, int | None], str], name: str, z: int, n: int | None = None) -> str:
        return table.get((name, z, n), "0")

    if get(pm3, "usspm3", 1) != "-13.0733210" or get(pm3, "guesp1", 1, 1) != "1.1287500":
        raise RuntimeError("PM3 hydrogen anchor check failed")
    if pairs.get((11, 1)) != ("1.800472", "3.171946"):
        raise RuntimeError("PM3 Na-H pair anchor check failed")

    columns = [
        "z", "sym", "uss", "upp", "udd", "zs", "zp", "zd", "betas",
        "betap", "betad", "gss", "gsp", "gpp", "gp2", "hsp", "zsn",
        "zpn", "zdn", "f0sd", "g2sd", "alp", "poc", "g1_k", "g1_l",
        "g1_m", "g2_k", "g2_l", "g2_m", "g3_k", "g3_l", "g3_m",
        "g4_k", "g4_l", "g4_m",
    ]
    scalar_mapping = [
        "usspm3", "upppm3", None, "zspm3", "zppm3", None, "betasp",
        "betapp", "betadp", "gsspm3", "gsppm3", "gpppm3", "gp2pm3",
        "hsppm3", "zsnpm3", "zpnpm3", "zdnpm3", "f0sdpm3", "g2sdpm3",
        "alppm3", None,
    ]
    elements = sorted({z for _, z, _ in pm3})
    rows = []
    for z in elements:
        row: list[object] = [z, SYMBOLS[z - 1]]
        row.extend(get(pm3, name, z) if name else "0" for name in scalar_mapping)
        for gaussian in range(1, 5):
            row.extend(get(pm3, name, z, gaussian) for name in ("guesp1", "guesp2", "guesp3"))
        rows.append(row)
    write_csv(data_dir / "pm3_parameters.csv", ("src/models/parameters_for_PM3_C.F90",), columns, rows)

    pair_rows = [[zi, zj, alpb, xfac] for (zi, zj), (alpb, xfac) in sorted(pairs.items())]
    write_csv(data_dir / "pm3_pair_parameters.csv", ("src/models/alpb_and_xfac_pm3.F90",), ["zi", "zj", "alpb", "xfac"], pair_rows)

    write_csv(data_dir / "pm3_global.csv", ("src/models/parameters_for_PM3_C.F90",), ["index", "value"], [[i, 0] for i in range(1, 61)])

    sparkle_elements = sorted({z for _, z, _ in sparkles})
    sparkle_rows = []
    for z in sparkle_elements:
        sparkle_rows.append([
            z, SYMBOLS[z - 1], get(sparkles, "gsspm3sp", z),
            get(sparkles, "alppm3sp", z), get(eheat_sparkles, "eheat_sparkles", z),
            get(sparkles, "guespm3sp1", z, 1), get(sparkles, "guespm3sp2", z, 1),
            get(sparkles, "guespm3sp3", z, 1), get(sparkles, "guespm3sp1", z, 2),
            get(sparkles, "guespm3sp2", z, 2), get(sparkles, "guespm3sp3", z, 2),
        ])
    write_csv(
        data_dir / "pm3_sparkles.csv",
        ("src/models/parameters_for_PM3_Sparkles_C.F90", "src/models/parameters_C.F90"),
        ["z", "sym", "gss", "alp", "eheat_kcal", "g1_k", "g1_l", "g1_m", "g2_k", "g2_l", "g2_m"],
        sparkle_rows,
    )

    report = [
        "# PM3 parameter extraction report", "", f"MOPAC {TAG}, {LICENSE}.", "",
        f"- PM3 elements including special point atoms: {len(elements)}",
        f"- Pair-specific alpha/xfac entries: {len(pairs)}",
        f"- Lanthanide(III) Sparkles: {len(sparkle_elements)}",
        "- PM3 has no d-orbital or PM6 v_par parameters; corresponding schema fields are zero.",
        "- Supported MOPAC special model atoms in the PM3 table are Cb (102), + (104), and - (106).", "",
    ]
    (project / "tools" / "extract_report.md").write_text("\n".join(report), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
