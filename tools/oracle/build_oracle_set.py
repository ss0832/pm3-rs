"""Build the hundred-molecule MOPAC oracle that `tests/mopac_oracle.rs` reads.

Every number in `tests/data/mopac_oracle.tsv` comes out of MOPAC v23.2.5 and nothing
in this file computes chemistry itself. The procedure per molecule is two MOPAC runs:

1. **Optimize** from a rough starting geometry. The fixture is then a PM3 stationary
   point rather than a guess, which matters for the frequency and gradient checks the
   Rust side layers on top, and removes "the geometry was silly" as an explanation for
   any disagreement.
2. **1SCF at those optimized coordinates.** This is the row that gets written: heat of
   formation, dipole, ionization potential and the net atomic charges, all at a
   geometry recorded to the same nine decimals MOPAC reported.

Starting geometries come from ASE's G2/G3 collection where it has the molecule, and
from the small builders below for everything G2 has never heard of --- the heavy
main-group hydrides and halides, the organometallics, the ions.

`--check` additionally runs pm3-rs on each fixture and prints the deviation, which is
how the tolerance in the Rust test was chosen. It is a report, not a gate: the TSV is
MOPAC's answer whether or not pm3-rs agrees with it.

    python build_oracle_set.py                # regenerate the TSV
    python build_oracle_set.py --check        # regenerate, then compare against pm3-rs
    python build_oracle_set.py --check --only water,benzene

**Every run passes `NOMM`, and that is not optional.** MOPAC applies `MMOK` by default: a
molecular-mechanics correction to the O=C-N-H torsion of an amide, added to the heat of
formation after the SCF. It is not part of the PM3 Hamiltonian --- the total energy, the
density, and every orbital energy are identical with and without it --- so a MOPAC default
run is not a PM3 oracle for any molecule containing an amide linkage. Acetamide is the one
in this set, where the correction is 0.655 kcal/mol; that number was mistaken for a pm3-rs
defect until `NOMM` was tried.
"""

import argparse
import math
import os
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_mopac  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "tests", "data", "mopac_oracle.tsv")

# --------------------------------------------------------------------------------------
# Geometry builders, for the molecules ASE's G2 collection does not carry.
#
# None of these need to be accurate. They are starting points for MOPAC's optimizer, and
# the fixture that gets written is where the optimizer lands, not where it began.
# --------------------------------------------------------------------------------------

DEG = math.pi / 180.0


def diatomic(a, b, r):
    return [(a, 0.0, 0.0, 0.0), (b, 0.0, 0.0, r)]


def bent(a, x, r, angle):
    """AX2, C2v."""
    half = 0.5 * angle * DEG
    return [
        (a, 0.0, 0.0, 0.0),
        (x, r * math.sin(half), 0.0, r * math.cos(half)),
        (x, -r * math.sin(half), 0.0, r * math.cos(half)),
    ]


def linear_ax2(a, x, r):
    return [(a, 0.0, 0.0, 0.0), (x, 0.0, 0.0, r), (x, 0.0, 0.0, -r)]


def pyramidal(a, x, r, angle):
    """AX3, C3v. `angle` is the X-A-X angle."""
    # Solve for the polar angle theta from the apex that reproduces the X-A-X angle.
    cos_xax = math.cos(angle * DEG)
    # For three equivalent bonds at polar angle t: cos(XAX) = cos^2 t + sin^2 t * cos(120)
    #                                             = cos^2 t - 0.5 sin^2 t
    # => cos^2 t = (2 cos(XAX) + 1) / 3
    cos_t = math.sqrt(max(0.0, (2.0 * cos_xax + 1.0) / 3.0))
    sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
    out = [(a, 0.0, 0.0, 0.0)]
    for k in range(3):
        phi = k * 120.0 * DEG
        out.append((x, r * sin_t * math.cos(phi), r * sin_t * math.sin(phi), r * cos_t))
    return out


def planar3(a, x, r):
    """AX3, D3h."""
    out = [(a, 0.0, 0.0, 0.0)]
    for k in range(3):
        phi = k * 120.0 * DEG
        out.append((x, r * math.cos(phi), r * math.sin(phi), 0.0))
    return out


def tetrahedral(a, x, r):
    d = r / math.sqrt(3.0)
    return [
        (a, 0.0, 0.0, 0.0),
        (x, d, d, d),
        (x, -d, -d, d),
        (x, -d, d, -d),
        (x, d, -d, -d),
    ]


def octahedral(a, x, r):
    out = [(a, 0.0, 0.0, 0.0)]
    for axis in range(3):
        for sign in (1.0, -1.0):
            pos = [0.0, 0.0, 0.0]
            pos[axis] = sign * r
            out.append((x, *pos))
    return out


def trigonal_bipyramidal(a, x, r_eq, r_ax):
    out = [(a, 0.0, 0.0, 0.0)]
    for k in range(3):
        phi = k * 120.0 * DEG
        out.append((x, r_eq * math.cos(phi), r_eq * math.sin(phi), 0.0))
    out += [(x, 0.0, 0.0, r_ax), (x, 0.0, 0.0, -r_ax)]
    return out


def seesaw(a, x, r_eq, r_ax):
    """AX4 with a lone pair, C2v --- SF4's shape."""
    out = [(a, 0.0, 0.0, 0.0)]
    for sign in (1.0, -1.0):
        out.append((x, sign * r_eq * math.sin(52.0 * DEG), 0.0, r_eq * math.cos(52.0 * DEG)))
    for sign in (1.0, -1.0):
        out.append((x, 0.0, sign * r_ax, -0.25))
    return out


def dimethyl_metal(m, r_mc, r_ch=1.09):
    """M(CH3)2, linear C-M-C with staggered methyls --- the real organometallics."""
    out = [(m, 0.0, 0.0, 0.0)]
    for sign, twist in ((1.0, 0.0), (-1.0, 60.0)):
        cz = sign * r_mc
        out.append(("C", 0.0, 0.0, cz))
        # Methyl hydrogens on a cone opening away from the metal.
        theta = 110.0 * DEG
        for k in range(3):
            phi = (twist + k * 120.0) * DEG
            out.append(
                (
                    "H",
                    r_ch * math.sin(theta) * math.cos(phi),
                    r_ch * math.sin(theta) * math.sin(phi),
                    cz + sign * r_ch * abs(math.cos(theta)) * -1.0,
                )
            )
    return out


def dimer(a, r):
    """A homonuclear pair at a fixed separation.

    For the noble gases this is the whole molecule set PM3 admits: none of them binds, so
    there is nothing to optimize, and a fixed separation is still a complete test of the
    element's one-center terms, its two-center integrals and its core repulsion.
    """
    return [(a, 0.0, 0.0, 0.0), (a, 0.0, 0.0, r)]


def sparkle_trifluoride(ln, r=2.10):
    """LnF3, trigonal planar --- the geometry MOPAC's Sparkle validation uses.

    A Sparkle carries a core charge and no valence orbitals, so it contributes only through
    the core-core repulsion and the field it puts on the fluorines.
    """
    y = 0.8660254038 * r
    return [
        (ln, 0.0, 0.0, 0.0),
        ("F", r, 0.0, 0.0),
        ("F", -0.5 * r, y, 0.0),
        ("F", -0.5 * r, -y, 0.0),
    ]


def water_with_point_charge(sign, distance=3.0):
    """Water beside one of MOPAC's `+`/`-` special atoms: a bare unit charge, no orbitals."""
    return [
        ("O", 0.0, 0.0, 0.0),
        ("H", 0.9584, 0.0, 0.0),
        ("H", -0.24, 0.9278, 0.0),
        (sign, 0.0, 0.0, distance),
    ]


def capped_methyl():
    """Methane with one hydrogen replaced by `Cb`, MOPAC's capped-bond atom."""
    d = 0.6276
    return [
        ("C", 0.0, 0.0, 0.0),
        ("Cb", d, d, d),
        ("H", -d, -d, d),
        ("H", -d, d, -d),
        ("H", d, -d, -d),
    ]


def oxo_halide(a, o, x, r_o, r_x):
    """AOX3, C3v --- POCl3's shape: one short double bond up, three halides down."""
    out = [(a, 0.0, 0.0, 0.0), (o, 0.0, 0.0, r_o)]
    theta = 108.0 * DEG
    for k in range(3):
        phi = k * 120.0 * DEG
        out.append(
            (x, r_x * math.sin(theta) * math.cos(phi), r_x * math.sin(theta) * math.sin(phi), r_x * math.cos(theta))
        )
    return out


def methyl_halide_like(a, x, r_ax, y, r_ay, angle):
    """AX3 pyramid plus one axial substituent --- for CH3-Br style cases built by hand."""
    out = pyramidal(a, x, r_ax, angle)
    out.append((y, 0.0, 0.0, -r_ay))
    return out


# --------------------------------------------------------------------------------------
# The molecule set.
#
# `(key, g2_name_or_None, charge, multiplicity, builder_or_None)`. G2 supplies the light
# organics and second-row inorganics; everything below period 3 is built here, because
# the heavy main group is exactly where a semiempirical code's element coverage is worth
# testing and exactly where G2 stops.
# --------------------------------------------------------------------------------------

G2 = [
    # (key, g2 name, charge, multiplicity)
    ("hydrogen", "H2", 0, 1),
    ("nitrogen", "N2", 0, 1),
    ("fluorine", "F2", 0, 1),
    ("chlorine", "Cl2", 0, 1),
    ("oxygen_triplet", "O2", 0, 3),
    ("carbon_monoxide", "CO", 0, 1),
    ("carbon_dioxide", "CO2", 0, 1),
    ("carbon_disulfide", "CS2", 0, 1),
    ("carbonyl_sulfide", "OCS", 0, 1),
    ("hydrogen_fluoride", "HF", 0, 1),
    ("hydrogen_chloride", "HCl", 0, 1),
    ("water", "H2O", 0, 1),
    ("hydrogen_peroxide", "H2O2", 0, 1),
    ("hydrogen_sulfide", "SH2", 0, 1),
    ("sulfur_dioxide", "SO2", 0, 1),
    ("ammonia", "NH3", 0, 1),
    ("hydrazine", "N2H4", 0, 1),
    ("nitrous_oxide", "N2O", 0, 1),
    ("nitrogen_trifluoride", "NF3", 0, 1),
    ("ozone", "O3", 0, 1),
    ("hypochlorous_acid", "HOCl", 0, 1),
    ("chlorine_monofluoride", "ClF", 0, 1),
    ("nitrosyl_chloride", "ClNO", 0, 1),
    ("phosphine", "PH3", 0, 1),
    ("phosphorus_trifluoride", "PF3", 0, 1),
    ("silane", "SiH4", 0, 1),
    ("disilane", "Si2H6", 0, 1),
    ("silicon_tetrafluoride", "SiF4", 0, 1),
    ("silicon_tetrachloride", "SiCl4", 0, 1),
    ("methylsilane", "CH3SiH3", 0, 1),
    ("aluminium_trichloride", "AlCl3", 0, 1),
    ("aluminium_trifluoride", "AlF3", 0, 1),
    ("boron_trifluoride", "BF3", 0, 1),
    ("boron_trichloride", "BCl3", 0, 1),
    ("lithium_hydride", "LiH", 0, 1),
    ("lithium_fluoride", "LiF", 0, 1),
    ("sodium_chloride", "NaCl", 0, 1),
    ("methane", "CH4", 0, 1),
    ("ethane", "C2H6", 0, 1),
    ("ethene", "C2H4", 0, 1),
    ("ethyne", "C2H2", 0, 1),
    ("propane", "C3H8", 0, 1),
    ("isobutane", "isobutane", 0, 1),
    ("butane", "trans-butane", 0, 1),
    ("isobutene", "isobutene", 0, 1),
    ("butadiene", "butadiene", 0, 1),
    ("butyne", "2-butyne", 0, 1),
    ("allene", "C3H4_D2d", 0, 1),
    ("cyclopropane", "C3H6_D3h", 0, 1),
    ("cyclobutane", "cyclobutane", 0, 1),
    ("benzene", "C6H6", 0, 1),
    ("pyridine", "C5H5N", 0, 1),
    ("pyrrole", "C4H4NH", 0, 1),
    ("furan", "C4H4O", 0, 1),
    ("thiophene", "C4H4S", 0, 1),
    ("methanol", "CH3OH", 0, 1),
    ("ethanol", "CH3CH2OH", 0, 1),
    ("dimethyl_ether", "CH3OCH3", 0, 1),
    ("methyl_ethyl_ether", "CH3CH2OCH3", 0, 1),
    ("oxirane", "CH2OCH2", 0, 1),
    ("formaldehyde", "H2CO", 0, 1),
    ("acetaldehyde", "CH3CHO", 0, 1),
    ("acetone", "CH3COCH3", 0, 1),
    ("ketene", "H2CCO", 0, 1),
    ("glyoxal", "OCHCHO", 0, 1),
    ("formic_acid", "HCOOH", 0, 1),
    ("acetic_acid", "CH3COOH", 0, 1),
    ("methyl_formate", "HCOOCH3", 0, 1),
    ("acetyl_chloride", "CH3COCl", 0, 1),
    ("acetamide", "CH3CONH2", 0, 1),
    ("methylamine", "H3CNH2", 0, 1),
    ("ethylamine", "CH3CH2NH2", 0, 1),
    ("dimethylamine", "C2H6NH", 0, 1),
    ("trimethylamine", "C3H9N", 0, 1),
    ("aziridine", "CH2NHCH2", 0, 1),
    ("hydrogen_cyanide", "HCN", 0, 1),
    ("acetonitrile", "CH3CN", 0, 1),
    ("cyanogen", "NCCN", 0, 1),
    ("acrylonitrile", "H2CCHCN", 0, 1),
    ("nitromethane", "CH3NO2", 0, 1),
    ("methyl_nitrite", "CH3ONO", 0, 1),
    ("methanethiol", "CH3SH", 0, 1),
    ("dimethyl_sulfide", "CH3SCH3", 0, 1),
    ("ethanethiol", "CH3CH2SH", 0, 1),
    ("thiirane", "CH2SCH2", 0, 1),
    ("dimethyl_sulfoxide", "C2H6SO", 0, 1),
    ("chloromethane", "CH3Cl", 0, 1),
    ("dichloromethane", "H2CCl2", 0, 1),
    ("chloroform", "HCCl3", 0, 1),
    ("carbon_tetrachloride", "CCl4", 0, 1),
    ("fluoromethane_difluoro", "H2CF2", 0, 1),
    ("fluoroform", "HCF3", 0, 1),
    ("carbon_tetrafluoride", "CF4", 0, 1),
    ("tetrafluoroethene", "C2F4", 0, 1),
    ("tetrachloroethene", "C2Cl4", 0, 1),
    ("vinyl_chloride", "H2CCHCl", 0, 1),
    ("chloroethane", "CH3CH2Cl", 0, 1),
    ("carbonyl_fluoride", "COF2", 0, 1),
    # Open shell. PM3's UHF path is a separate code path from RHF and deserves its own
    # oracles rather than being represented by one methyl radical.
    ("methyl_radical", "CH3", 0, 2),
    ("amino_radical", "NH2", 0, 2),
    ("hydroxyl_radical", "OH", 0, 2),
    ("formyl_radical", "HCO", 0, 2),
    ("nitric_oxide", "NO", 0, 2),
    ("nitrogen_dioxide", "NO2", 0, 2),
    ("cyano_radical", "CN", 0, 2),
    ("thiyl_radical", "SH", 0, 2),
    ("phosphino_radical", "PH2", 0, 2),
    ("silyl_radical", "SiH3", 0, 2),
    ("ethyl_radical", "C2H5", 0, 2),
    ("vinyl_radical", "C2H3", 0, 2),
    ("methoxy_radical", "CH3O", 0, 2),
    ("chlorine_monoxide", "ClO", 0, 2),
    ("methylene_triplet", "CH2_s3B1d", 0, 3),
]

BUILT = [
    # --- heavy main group: the elements G2 has never heard of ---
    ("germane", 0, 1, lambda: tetrahedral("Ge", "H", 1.53)),
    ("germanium_tetrachloride", 0, 1, lambda: tetrahedral("Ge", "Cl", 2.11)),
    ("arsine", 0, 1, lambda: pyramidal("As", "H", 1.51, 92.0)),
    ("arsenic_trichloride", 0, 1, lambda: pyramidal("As", "Cl", 2.17, 98.0)),
    ("hydrogen_selenide", 0, 1, lambda: bent("Se", "H", 1.46, 91.0)),
    ("selenium_dioxide", 0, 1, lambda: bent("Se", "O", 1.61, 114.0)),
    ("bromine", 0, 1, lambda: diatomic("Br", "Br", 2.28)),
    ("hydrogen_bromide", 0, 1, lambda: diatomic("Br", "H", 1.41)),
    ("bromomethane", 0, 1, lambda: methyl_halide_like("C", "H", 1.09, "Br", 1.93, 111.0)),
    ("iodine", 0, 1, lambda: diatomic("I", "I", 2.67)),
    ("hydrogen_iodide", 0, 1, lambda: diatomic("I", "H", 1.61)),
    ("iodomethane", 0, 1, lambda: methyl_halide_like("C", "H", 1.09, "I", 2.13, 111.0)),
    ("stannane", 0, 1, lambda: tetrahedral("Sn", "H", 1.70)),
    ("tin_tetrachloride", 0, 1, lambda: tetrahedral("Sn", "Cl", 2.28)),
    ("stibine", 0, 1, lambda: pyramidal("Sb", "H", 1.70, 92.0)),
    ("antimony_trichloride", 0, 1, lambda: pyramidal("Sb", "Cl", 2.33, 97.0)),
    ("hydrogen_telluride", 0, 1, lambda: bent("Te", "H", 1.66, 90.0)),
    ("bismuthine", 0, 1, lambda: pyramidal("Bi", "H", 1.78, 92.0)),
    ("bismuth_trichloride", 0, 1, lambda: pyramidal("Bi", "Cl", 2.48, 97.0)),
    ("lead_tetrachloride", 0, 1, lambda: tetrahedral("Pb", "Cl", 2.43)),
    ("gallium_trichloride", 0, 1, lambda: planar3("Ga", "Cl", 2.11)),
    ("indium_trichloride", 0, 1, lambda: planar3("In", "Cl", 2.29)),
    ("thallium_chloride", 0, 1, lambda: diatomic("Tl", "Cl", 2.48)),
    # --- the three transition metals PM3 actually has ---
    ("zinc_dichloride", 0, 1, lambda: linear_ax2("Zn", "Cl", 2.07)),
    ("cadmium_dichloride", 0, 1, lambda: linear_ax2("Cd", "Cl", 2.24)),
    ("mercury_dichloride", 0, 1, lambda: linear_ax2("Hg", "Cl", 2.25)),
    ("dimethylzinc", 0, 1, lambda: dimethyl_metal("Zn", 1.93)),
    ("dimethylcadmium", 0, 1, lambda: dimethyl_metal("Cd", 2.11)),
    ("dimethylmercury", 0, 1, lambda: dimethyl_metal("Hg", 2.09)),
    # --- hypervalent sulfur and phosphorus, where the d-orbital path runs ---
    ("sulfur_hexafluoride", 0, 1, lambda: octahedral("S", "F", 1.56)),
    ("sulfur_tetrafluoride", 0, 1, lambda: seesaw("S", "F", 1.55, 1.65)),
    ("phosphorus_trichloride", 0, 1, lambda: pyramidal("P", "Cl", 2.04, 100.0)),
    ("phosphorus_pentachloride", 0, 1, lambda: trigonal_bipyramidal("P", "Cl", 2.02, 2.12)),
    ("phosphoryl_chloride", 0, 1, lambda: oxo_halide("P", "O", "Cl", 1.45, 1.99)),
    # --- s-block salts and hydrides ---
    ("beryllium_dichloride", 0, 1, lambda: linear_ax2("Be", "Cl", 1.75)),
    ("magnesium_dichloride", 0, 1, lambda: linear_ax2("Mg", "Cl", 2.18)),
    ("sodium_fluoride", 0, 1, lambda: diatomic("Na", "F", 1.93)),
    ("potassium_chloride", 0, 1, lambda: diatomic("K", "Cl", 2.67)),
    ("potassium_fluoride", 0, 1, lambda: diatomic("K", "F", 2.17)),
    ("calcium_dichloride", 0, 1, lambda: linear_ax2("Ca", "Cl", 2.48)),
    # --- ions: the charged SCF path, closed and open shell ---
    ("ammonium", 1, 1, lambda: tetrahedral("N", "H", 1.03)),
    ("hydronium", 1, 1, lambda: pyramidal("O", "H", 0.98, 113.0)),
    ("hydroxide", -1, 1, lambda: diatomic("O", "H", 0.96)),
    ("cyanide", -1, 1, lambda: diatomic("C", "N", 1.18)),
    ("methyl_cation", 1, 1, lambda: planar3("C", "H", 1.09)),
    ("methyl_anion", -1, 1, lambda: pyramidal("C", "H", 1.10, 108.0)),
    ("water_cation", 1, 2, lambda: bent("O", "H", 0.99, 110.0)),
    ("nitrite", -1, 1, lambda: bent("N", "O", 1.25, 115.0)),
    # --- the heavy s-block, which G2 stops well short of ---
    ("rubidium_chloride", 0, 1, lambda: diatomic("Rb", "Cl", 2.79)),
    ("strontium_dichloride", 0, 1, lambda: bent("Sr", "Cl", 2.63, 140.0)),
    ("caesium_chloride", 0, 1, lambda: diatomic("Cs", "Cl", 2.91)),
    ("barium_dichloride", 0, 1, lambda: bent("Ba", "Cl", 2.77, 140.0)),
    # Francium is deliberately absent. Its row in `src/data/pm3_parameters.csv` is all
    # zeros, and MOPAC answers `DATA ARE NOT AVAILABLE FOR ELEMENT NO. 87 ... CALCULATION
    # STOPPED` --- there is no PM3 parameterization to compare against. pm3-rs refuses it
    # too ("missing PM3 parameter block for Z=87"), which is the agreement that matters.
]

# Species whose geometry is **not** optimized, with the reason. Everything above is relaxed
# by MOPAC before the reference single point; these are not, because there is nothing to
# relax to --- a noble-gas pair has no bound minimum in PM3 and would fly apart, and a
# Sparkle or point charge has no valence orbitals and so no chemistry to minimize. Their
# rows are still full oracles: the reference is MOPAC's 1SCF at the stated geometry, which
# is exactly what pm3-rs is asked to reproduce.
FIXED = [
    # --- noble gases, at a fixed separation ---
    ("helium_dimer", 0, 1, lambda: dimer("He", 3.0)),
    ("neon_dimer", 0, 1, lambda: dimer("Ne", 3.0)),
    ("argon_dimer", 0, 1, lambda: dimer("Ar", 3.0)),
    ("krypton_dimer", 0, 1, lambda: dimer("Kr", 3.5)),
    ("xenon_dimer", 0, 1, lambda: dimer("Xe", 4.0)),
    ("helium_hydride_cation", 1, 1, lambda: diatomic("He", "H", 0.9)),
    # --- MOPAC's three special atoms ---
    ("capped_methyl", 0, 1, capped_methyl),
    ("water_with_positive_sparkle", 1, 1, lambda: water_with_point_charge("+")),
    ("water_with_negative_sparkle", -1, 1, lambda: water_with_point_charge("-")),
]
# --- the fifteen La-Lu Sparkles, as LnF3 ---
FIXED += [
    (f"{ln.lower()}_trifluoride", 0, 1, (lambda l=ln: sparkle_trifluoride(l)))
    for ln in "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split()
]


def g2_atoms(name):
    from ase.collections import g2 as collection

    atoms = collection[name]
    return [
        (a.symbol, float(a.position[0]), float(a.position[1]), float(a.position[2]))
        for a in atoms
    ]


def cases():
    """Every molecule, as `(key, charge, multiplicity, builder, relax)`."""
    out = []
    for key, name, charge, mult in G2:
        out.append((key, charge, mult, (lambda n=name: g2_atoms(n)), True))
    for key, charge, mult, builder in BUILT:
        out.append((key, charge, mult, builder, True))
    for key, charge, mult, builder in FIXED:
        out.append((key, charge, mult, builder, False))
    return out


def write_xyz(path, key, atoms):
    with open(path, "w") as fh:
        fh.write(f"{len(atoms)}\n{key}\n")
        for sym, x, y, z in atoms:
            fh.write(f"{sym} {x:.9f} {y:.9f} {z:.9f}\n")


def formula(atoms):
    counts = {}
    for sym, *_ in atoms:
        counts[sym] = counts.get(sym, 0) + 1
    # Hill order, which is what a chemist reading the TSV expects.
    keys = sorted(counts)
    ordered = []
    for lead in ("C", "H"):
        if lead in counts:
            ordered.append(lead)
            keys.remove(lead)
    if "C" not in counts:
        ordered = []
        keys = sorted(counts)
    ordered += keys
    return "".join(f"{s}{counts[s]}" if counts[s] > 1 else s for s in ordered)


def geometry_field(symbols, coords):
    parts = []
    for i, sym in enumerate(symbols):
        x, y, z = coords[3 * i : 3 * i + 3]
        parts.append(f"{sym} {x:.9f} {y:.9f} {z:.9f}")
    return ";".join(parts)


def run_one(key, charge, mult, atoms, workroot, relax=True):
    """Optimize (unless `relax` is false), then take a single point. Returns a TSV row."""
    tmp = os.path.join(workroot, key)
    os.makedirs(tmp, exist_ok=True)
    start = os.path.join(tmp, key + ".xyz")
    write_xyz(start, key, atoms)
    symbols = [sym for sym, *_ in atoms]

    if relax:
        # NOMM on both runs: see the module docstring. Without it the optimizer minimizes a
        # different surface for an amide and the reported heat of formation is not PM3's.
        opt = run_mopac.run(
            start, charge=charge, mult=mult, mode="optimize", workdir=tmp,
            extra_keywords=("NOMM",),
        )
        aux = run_mopac.parse_aux(os.path.join(tmp, key + ".aux"))
        coords = aux.get("ATOM_X_OPT")
        if not isinstance(coords, list) or len(coords) != 3 * len(atoms):
            raise RuntimeError(f"{key}: MOPAC did not return optimized coordinates")
        if opt["heat_of_formation_kcal"] is None:
            raise RuntimeError(f"{key}: optimization produced no heat of formation")
        relaxed = [(symbols[i], *coords[3 * i : 3 * i + 3]) for i in range(len(atoms))]
    else:
        coords = [c for _, *xyz in atoms for c in xyz]
        relaxed = list(atoms)

    final = os.path.join(tmp, key + "_opt.xyz")
    write_xyz(final, key, relaxed)
    single = run_mopac.run(
        final, charge=charge, mult=mult, mode="1scf", workdir=tmp, extra_keywords=("NOMM",)
    )
    if single["heat_of_formation_kcal"] is None:
        raise RuntimeError(f"{key}: single point produced no heat of formation")

    charges = single["charges"] or []
    return {
        "name": key,
        "formula": formula(atoms),
        "charge": charge,
        "multiplicity": mult,
        "heat_of_formation_kcal": f"{single['heat_of_formation_kcal']:.9f}",
        "dipole_debye": f"{single['dipole_magnitude_debye']:.6f}"
        if single["dipole_magnitude_debye"] is not None
        else "",
        "ionization_potential_ev": f"{single['ionization_potential_ev']:.6f}"
        if single["ionization_potential_ev"] is not None
        else "",
        "charges_e": ";".join(f"{q:.6f}" for q in charges),
        "relaxed": "yes" if relax else "no",
        "geometry_angstrom": geometry_field(symbols, coords),
    }


COLUMNS = [
    "name",
    "formula",
    "charge",
    "multiplicity",
    "heat_of_formation_kcal",
    "dipole_debye",
    "ionization_potential_ev",
    "charges_e",
    "relaxed",
    "geometry_angstrom",
]

HEADER = """\
# MOPAC PM3 reference values for pm3-rs.
#
# Generated by tools/oracle/build_oracle_set.py against MOPAC v23.2.5 (PM3 PRECISE NOMM,
# AUX(PRECISION=9)). NOMM turns off MOPAC's default MMOK amide correction, which is a
# molecular-mechanics term added after the SCF and not part of PM3; without it the row for
# any amide would not be a PM3 reference. Each geometry is a PM3 stationary point: MOPAC
# optimized it from a
# rough guess, and every number on the row is a 1SCF run at the coordinates in the last
# column. Nothing here was computed by pm3-rs --- it is the oracle, not a snapshot of
# our own output --- and `tests/mopac_oracle.rs` is what reads it.
#
# Columns are tab separated. `charges_e` is one Mulliken-style net atomic charge per
# atom and `geometry_angstrom` one "Sym x y z" per atom, both semicolon separated and
# in input order. Heats of formation are kcal/mol, dipoles debye, potentials eV.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="also compare against pm3-rs")
    ap.add_argument("--only", default="", help="comma-separated subset of names")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    wanted = {n.strip() for n in args.only.split(",") if n.strip()}
    selected = [c for c in cases() if not wanted or c[0] in wanted]
    print(f"{len(selected)} molecules")

    rows, failures = [], []
    workroot = tempfile.mkdtemp(prefix="pm3rs_oracle_set_")
    for index, (key, charge, mult, builder, relax) in enumerate(selected, 1):
        try:
            atoms = builder()
            row = run_one(key, charge, mult, atoms, workroot, relax)
            rows.append(row)
            print(f"  [{index:3d}/{len(selected)}] {key:<28} {row['formula']:<10} "
                  f"{float(row['heat_of_formation_kcal']):12.4f} kcal/mol")
        except Exception as exc:  # noqa: BLE001 - a failure here is a report, not a crash
            failures.append((key, str(exc).splitlines()[0]))
            print(f"  [{index:3d}/{len(selected)}] {key:<28} FAILED: {exc}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="\n", encoding="utf-8") as fh:
        fh.write(HEADER)
        fh.write("\t".join(COLUMNS) + "\n")
        for row in rows:
            fh.write("\t".join(str(row[c]) for c in COLUMNS) + "\n")
    print(f"\nwrote {len(rows)} rows to {args.out}")
    if failures:
        print(f"{len(failures)} failed:")
        for key, why in failures:
            print(f"  {key}: {why}")

    if args.check:
        compare(rows)


def compare(rows):
    """Report pm3-rs against the oracle. A report, not a gate --- the TSV stands either way."""
    from ase.data import atomic_numbers
    from pm3_rs import native

    print("\npm3-rs against the oracle:")
    worst = []
    for row in rows:
        atoms = [p.split() for p in row["geometry_angstrom"].split(";")]
        numbers = [atomic_numbers[a[0]] for a in atoms]
        positions = [[float(a[1]), float(a[2]), float(a[3])] for a in atoms]
        try:
            result = native.single_point(
                numbers,
                positions,
                charge=int(row["charge"]),
                multiplicity=int(row["multiplicity"]),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {row['name']:<28} pm3-rs FAILED: {str(exc).splitlines()[0]}")
            worst.append((float("inf"), row["name"]))
            continue
        delta = abs(result["heat_of_formation_kcal"] - float(row["heat_of_formation_kcal"]))
        worst.append((delta, row["name"]))
    worst.sort(reverse=True)
    for delta, name in worst[:15]:
        print(f"  {name:<28} {delta:.3e} kcal/mol")
    finite = [d for d, _ in worst if math.isfinite(d)]
    if finite:
        print(f"\n  max {max(finite):.3e}, median {sorted(finite)[len(finite)//2]:.3e} kcal/mol")


if __name__ == "__main__":
    main()
