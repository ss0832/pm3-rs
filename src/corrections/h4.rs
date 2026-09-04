// SPDX-License-Identifier: GPL-3.0-or-later

//! H4 hydrogen-bond correction (the "H4" of PM3-D3H4) and the accompanying
//! H-H repulsion. The functional form follows the public H4 implementation;
//! all coefficients in this module are the PM3-specific D3H4 parameter set.
//!
//! For each O/N···H–O/N contact the correction is
//! `E = e_para · e_radial(R_DA) · e_angular(∠D–H–A) · e_bond_switch
//!      · e_scale_water · e_scale_NR4+ · e_scale_COO⁻`, summed in kcal/mol.
//! The donor D is the O/N nearer the bridging hydrogen; a triple is counted
//! only when that donor–H bond is covalent (< 1.15 Å), matching MOPAC's
//! `all_h_bonds`/`connected` selection. The water- and charged-group scaling
//! factors use the distance-based covalent-valence measure
//! [`cvalence_contribution`] (MOPAC `cvalence_contribution`, covalent radii
//! from `radii_C`). The separate H–H repulsion [`hh_rep_energy`]
//! (MOPAC `energy_corr_hh_rep`/`poly`) is added for the D3H4 family.
//! PDB-residue-name-only HIP/GUA overrides from the Cuby interface are not part
//! of this coordinate-only library; the standard continuous group corrections
//! remain active for every input format.
//!
//! PROVENANCE: J. Řezáč, P. Hobza, *J. Chem. Theory Comput.* **8**, 141 (2012);
//! via MOPAC (Apache-2.0). See THIRD_PARTY_NOTICES.md.

use super::dist_g;
use crate::constants::{BOHR_TO_ANGSTROM, KCAL_TO_EV};
use crate::dual::Scalar;
use crate::system::Molecule;
use std::f64::consts::PI;

/// Clamp a cosine to `[-1, 1]` (roundoff guard) while preserving derivatives inside the range.
#[inline]
fn clamp_unit<S: Scalar>(c: S) -> S {
    if c.val() >= 1.0 {
        S::cst(1.0)
    } else if c.val() <= -1.0 {
        S::cst(-1.0)
    } else {
        c
    }
}

/// `(1 − |target − cv|)` clamped at 0 — the charged-group valence membership factor.
#[inline]
fn f_dev<S: Scalar>(cv: S, target: f64) -> S {
    let dev = (cv - target).abs();
    if dev.val() < 1.0 {
        S::cst(1.0) - dev
    } else {
        S::cst(0.0)
    }
}

const PARA_OH_O: f64 = 2.71;
const PARA_OH_N: f64 = 4.37;
const PARA_NH_O: f64 = 2.29;
const PARA_NH_N: f64 = 3.86;

// PM3 charged-/water-group multipliers from the D3H4 reference implementation.
const MULTIPLIER_WH_O: f64 = 0.91;
const MULTIPLIER_NH4: f64 = 2.539_063_393_162_7;
const MULTIPLIER_COO: f64 = 0.886_102_644_137_03;

/// Donor–hydrogen covalent-bond cutoff (Å) used to select H-bond triples
/// (MOPAC `all_h_bonds`, `RAH = 1.15` for D3H4/D3H4X).
const RAH: f64 = 1.15;
/// Donor–acceptor cutoff (Å); `e_radial` has its minimum at 5.5 Å.
const CUTOFF_DA: f64 = 5.5;

/// Covalent radii (Å), elements 1..=118 — MOPAC `radii_C::covalent_radii`
/// (H_bonds4.F90:18). Used only by [`cvalence_contribution`].
const COVALENT_RADII: [f64; 118] = [
    0.37, 0.32, 1.34, 0.90, 0.82, 0.77, 0.75, 0.73, 0.71, 0.69, 1.54, 1.30, 1.18, 1.11, 1.06, 1.02,
    0.99, 0.97, 1.96, 1.74, 1.44, 1.36, 1.25, 1.27, 1.39, 1.25, 1.26, 1.21, 1.38, 1.31, 1.26, 1.22,
    1.19, 1.16, 1.14, 1.10, 2.11, 1.92, 1.62, 1.48, 1.37, 1.45, 1.56, 1.26, 1.35, 1.31, 1.53, 1.48,
    1.44, 1.41, 1.38, 1.35, 1.33, 1.30, 2.25, 1.98, 1.69, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.60, 1.50, 1.38, 1.46, 1.59, 1.28, 1.37, 1.28, 1.44, 1.49, 0.0, 0.0, 1.46,
    0.0, 0.0, 1.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];

#[inline]
fn covalent_radius(z: u8) -> f64 {
    COVALENT_RADII
        .get((z as usize).wrapping_sub(1))
        .copied()
        .unwrap_or(0.0)
}

/// Radial term (7th-order polynomial in the donor–acceptor distance, Å), generic.
fn e_radial_g<S: Scalar>(rda: S) -> S {
    rda.powi(7) * -0.003_034_074_074_073_135
        + rda.powi(6) * 0.073_576_296_296_270_92
        + rda.powi(5) * -0.700_871_111_110_828
        + rda.powi(4) * 3.253_096_296_294_617_5
        + rda.powi(3) * -7.206_874_074_068_388
        + rda.powi(2) * 5.317_546_666_655_722
        + rda * 3.407_360_000_011_028
        - 4.685_120_000_004_504
}

/// Angular term for the D–H–A angle `theta` (radians). `angle = π − theta`. Generic.
fn e_angular_g<S: Scalar>(angle: S) -> S {
    let a = angle * (1.0 / (PI / 2.0));
    let x = a.powi(7) * -20.0 + a.powi(6) * 70.0 + a.powi(5) * -84.0 + a.powi(4) * 35.0;
    -(x * x) + 1.0
}

/// Coefficient for a donor/acceptor element pair (O=8, N=7); 0 otherwise.
fn e_para(donor_z: u8, acceptor_z: u8) -> f64 {
    match (donor_z, acceptor_z) {
        (8, 8) => PARA_OH_O,
        (8, 7) => PARA_OH_N,
        (7, 8) => PARA_NH_O,
        (7, 7) => PARA_NH_N,
        _ => 0.0,
    }
}

/// Distance (Å) between two atoms, generic.
#[inline]
fn dist_ang_g<S: Scalar>(pos: &[[S; 3]], a: usize, b: usize) -> S {
    dist_g(&pos[a], &pos[b]) * BOHR_TO_ANGSTROM
}

/// Covalent-valence contribution of the `a–b` contact (MOPAC `cvalence_contribution`): 1 inside
/// the covalent-radii sum `r0`, smoothly switching to 0 at `r1 = 1.6·r0` via the C² 7th-order
/// switch. Generic over the scalar.
fn cvalence_contribution_g<S: Scalar>(numbers: &[u8], pos: &[[S; 3]], a: usize, b: usize) -> S {
    let ri = covalent_radius(numbers[a]);
    let rj = covalent_radius(numbers[b]);
    let r0 = ri + rj;
    let r1 = r0 * 1.6;
    let r = dist_ang_g(pos, a, b);
    let rv = r.val();
    if rv == 0.0 || rv >= r1 {
        S::cst(0.0)
    } else if rv <= r0 {
        S::cst(1.0)
    } else {
        let x = (r - r0) / (r1 - r0);
        -(x.powi(7) * -20.0 + x.powi(6) * 70.0 + x.powi(5) * -84.0 + x.powi(4) * 35.0) + 1.0
    }
}

/// Water scaling `e_scale_w` for an O(donor)···O(acceptor) contact (MOPAC H_bonds4.F90:184).
fn e_scale_water_g<S: Scalar>(numbers: &[u8], pos: &[[S; 3]], donor: usize, acceptor: usize) -> S {
    if numbers[donor] != 8 || numbers[acceptor] != 8 {
        return S::cst(1.0);
    }
    let n = numbers.len();
    let mut hydrogens = S::cst(0.0);
    let mut others = S::cst(0.0);
    for k in 0..n {
        let c = cvalence_contribution_g(numbers, pos, donor, k);
        if numbers[k] == 1 {
            hydrogens = hydrogens + c;
        } else {
            others = others + c;
        }
    }
    if hydrogens.val() < 1.0 {
        return S::cst(1.0);
    }
    let slope = MULTIPLIER_WH_O - 1.0;
    let vv = hydrogens.val();
    let fv = if vv > 1.0 && vv <= 2.0 {
        hydrogens - 1.0
    } else if vv > 2.0 && vv < 3.0 {
        -hydrogens + 3.0
    } else {
        S::cst(0.0)
    };
    let fv2 = if (1.0 - others.val()) > 0.0 {
        -others + 1.0
    } else {
        S::cst(0.0)
    };
    fv * fv2 * slope + 1.0
}

/// NR4⁺ donor scaling `e_scale_chd` (MOPAC H_bonds4.F90:220).
fn e_scale_charged_donor_g<S: Scalar>(numbers: &[u8], pos: &[[S; 3]], donor: usize) -> S {
    if numbers[donor] != 7 {
        return S::cst(1.0);
    }
    let mut v = S::cst(0.0);
    for k in 0..numbers.len() {
        v = v + cvalence_contribution_g(numbers, pos, donor, k);
    }
    let vv = if v.val() > 3.0 { v - 3.0 } else { S::cst(0.0) };
    vv * (MULTIPLIER_NH4 - 1.0) + 1.0
}

/// Carboxylate COO⁻ acceptor scaling `e_scale_cha` (MOPAC H_bonds4.F90:236).
fn e_scale_charged_acceptor_g<S: Scalar>(numbers: &[u8], pos: &[[S; 3]], acceptor: usize) -> S {
    if numbers[acceptor] != 8 {
        return S::cst(1.0);
    }
    let n = numbers.len();
    let o1 = acceptor;
    // Closest bonded carbon to O1, plus O1's total covalent valence.
    let mut cdist = f64::INFINITY;
    let mut cc: Option<usize> = None;
    let mut cv_o1 = S::cst(0.0);
    for k in 0..n {
        let v = cvalence_contribution_g(numbers, pos, o1, k);
        cv_o1 = cv_o1 + v;
        let d = dist_ang_g(pos, o1, k).val();
        if v.val() > 0.0 && numbers[k] == 6 && d < cdist {
            cdist = d;
            cc = Some(k);
        }
    }
    let cc = match cc {
        Some(c) => c,
        None => return S::cst(1.0),
    };
    // Second oxygen bonded to that carbon (≠ O1), plus the carbon's valence.
    let mut odist = f64::INFINITY;
    let mut o2: Option<usize> = None;
    let mut cv_cc = S::cst(0.0);
    for k in 0..n {
        let v = cvalence_contribution_g(numbers, pos, cc, k);
        cv_cc = cv_cc + v;
        let d = dist_ang_g(pos, cc, k).val();
        if v.val() > 0.0 && k != o1 && numbers[k] == 8 && d < odist {
            odist = d;
            o2 = Some(k);
        }
    }
    let o2 = match o2 {
        Some(o) => o,
        None => return S::cst(1.0),
    };
    let mut cv_o2 = S::cst(0.0);
    for k in 0..n {
        cv_o2 = cv_o2 + cvalence_contribution_g(numbers, pos, o2, k);
    }
    let f_o1 = f_dev(cv_o1, 1.0);
    let f_o2 = f_dev(cv_o2, 1.0);
    let f_cc = f_dev(cv_cc, 3.0);
    f_o1 * f_o2 * f_cc * (MULTIPLIER_COO - 1.0) + 1.0
}

/// H4 hydrogen-bond correction energy (eV), f64 entry point.
pub fn h4_energy(mol: &Molecule) -> f64 {
    let (numbers, pos) = super::geometry_f64(mol);
    h4_energy_g::<f64>(&numbers, &pos)
}

/// H4 hydrogen-bond correction energy (eV), generic over the scalar.
pub fn h4_energy_g<S: Scalar>(numbers: &[u8], pos: &[[S; 3]]) -> S {
    h4_energy_cluster_g(numbers, pos, numbers.len())
}

/// H4 over an image-expanded cluster, generic over the scalar.
///
/// `n_cell` is how many leading entries belong to the reference cell. Each hydrogen-bond triple
/// has exactly one hydrogen, so assigning the triple to the cell that hydrogen sits in counts
/// every crystal-distinct triple exactly once — no fractional weights, and no ambiguity about
/// which cell owns a bond that straddles a boundary. Donor and acceptor may be anywhere in the
/// cluster, which is what lets hydrogen bonds cross the cell edge; in ice or a molecular crystal
/// most of them do.
///
/// With `n_cell = numbers.len()` this is exactly the molecular sum.
pub fn h4_energy_cluster_g<S: Scalar>(numbers: &[u8], pos: &[[S; 3]], n_cell: usize) -> S {
    let n = numbers.len();
    let is_don_acc = |z: u8| z == 7 || z == 8;
    let a = BOHR_TO_ANGSTROM;
    let mut sum_kcal = S::cst(0.0);
    for i in 0..n {
        if !is_don_acc(numbers[i]) {
            continue;
        }
        for j in (i + 1)..n {
            if !is_don_acc(numbers[j]) {
                continue;
            }
            let rda = dist_g(&pos[j], &pos[i]) * a;
            if rda.val() >= CUTOFF_DA {
                continue; // donor–acceptor beyond the correction's range
            }
            for h in 0..n_cell {
                if numbers[h] != 1 {
                    continue;
                }
                let vih = [
                    pos[i][0] - pos[h][0],
                    pos[i][1] - pos[h][1],
                    pos[i][2] - pos[h][2],
                ];
                let vjh = [
                    pos[j][0] - pos[h][0],
                    pos[j][1] - pos[h][1],
                    pos[j][2] - pos[h][2],
                ];
                let rih = (vih[0] * vih[0] + vih[1] * vih[1] + vih[2] * vih[2]).sqrt();
                let rjh = (vjh[0] * vjh[0] + vjh[1] * vjh[1] + vjh[2] * vjh[2]).sqrt();
                // D–H–A angle at H, then the H4 `angle = π − ∠(D-H-A)`.
                let cos_dha = (vih[0] * vjh[0] + vih[1] * vjh[1] + vih[2] * vjh[2]) / (rih * rjh);
                let dha = clamp_unit(cos_dha).acos();
                let angle = -dha + PI;
                if angle.val() >= PI / 2.0 {
                    continue; // ∠D-H-A ≤ 90°: not a hydrogen bond
                }
                // Donor is the O/N nearer the bridging hydrogen; the donor–H bond must be
                // covalent (< 1.15 Å) for the triple to count.
                let (donor, acceptor, rdh, rah) = if rih.val() < rjh.val() {
                    (i, j, rih * a, rjh * a)
                } else {
                    (j, i, rjh * a, rih * a)
                };
                if rdh.val() >= RAH {
                    continue;
                }
                let ep = e_para(numbers[donor], numbers[acceptor]);
                if ep == 0.0 {
                    continue;
                }
                // Bond switching (long D–H, e.g. a proton shared with the acceptor).
                let e_bond_switch = if rdh.val() > 1.15 {
                    let rdhs = rdh - 1.15;
                    let ravgs = rdh * 0.5 + rah * 0.5 - 1.15;
                    let x = rdhs / ravgs;
                    -(x.powi(7) * -20.0 + x.powi(6) * 70.0 + x.powi(5) * -84.0 + x.powi(4) * 35.0)
                        + 1.0
                } else {
                    S::cst(1.0)
                };
                let e_sw = e_scale_water_g(numbers, pos, donor, acceptor);
                let e_chd = e_scale_charged_donor_g(numbers, pos, donor);
                let e_cha = e_scale_charged_acceptor_g(numbers, pos, acceptor);
                sum_kcal = sum_kcal
                    + e_radial_g(rda)
                        * e_angular_g(angle)
                        * e_bond_switch
                        * e_sw
                        * e_chd
                        * e_cha
                        * ep;
            }
        }
    }
    sum_kcal * KCAL_TO_EV
}

/// H–H repulsion `poly(r)` (kcal/mol) for two hydrogens `r` Å apart (MOPAC `poly`), generic.
fn poly_hh_g<S: Scalar>(r: S) -> S {
    const K: f64 = 1.55;
    const EXPONENT: f64 = 6.86;
    const R0: f64 = 2.23;
    (S::cst(1.0) - (((r / R0 - 1.0) * -EXPONENT).exp() + 1.0).recip()) * K
}

/// H–H repulsion energy (eV) summed over all hydrogen pairs (MOPAC `energy_corr_hh_rep`),
/// f64 entry point. Added for the PM3-D3H4 family.
pub fn hh_rep_energy(mol: &Molecule) -> f64 {
    let (numbers, pos) = super::geometry_f64(mol);
    hh_rep_energy_g::<f64>(&numbers, &pos)
}

/// Generic (over the scalar) H–H repulsion energy (eV).
pub fn hh_rep_energy_g<S: Scalar>(numbers: &[u8], pos: &[[S; 3]]) -> S {
    hh_rep_energy_cluster_g(numbers, pos, numbers.len())
}

/// H–H repulsion over an image-expanded cluster, generic over the scalar. See
/// [`h4_energy_cluster_g`] for what `n_cell` means; here the tuple is a pair, so the sum runs
/// over cell atoms against the whole cluster with weight ½.
pub fn hh_rep_energy_cluster_g<S: Scalar>(numbers: &[u8], pos: &[[S; 3]], n_cell: usize) -> S {
    let n = numbers.len();
    let a = BOHR_TO_ANGSTROM;
    let mut sum_kcal = S::cst(0.0);
    for i in 0..n_cell {
        if numbers[i] != 1 {
            continue;
        }
        for j in 0..n {
            if j == i || numbers[j] != 1 {
                continue;
            }
            let r = dist_g(&pos[i], &pos[j]) * a;
            sum_kcal = sum_kcal + poly_hh_g(r);
        }
    }
    // The ½ pairs with the ordered loop: `½ Σ_i Σ_{j≠i} = Σ_{i<j}` for the molecular case.
    sum_kcal * (0.5 * KCAL_TO_EV)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_hbond_no_correction() {
        let mol = Molecule::from_xyz_str("2\nn2\nN 0 0 0\nN 0 0 1.1\n", 0.0).unwrap();
        // Two N with no bridging H → 0.
        assert_eq!(h4_energy(&mol), 0.0);
    }

    #[test]
    fn water_dimer_is_attractive() {
        // Linear-ish water dimer O–H···O.
        let mol = Molecule::from_xyz_str(
            "6\nwater dimer\nO 0.0 0.0 0.0\nH 0.0 0.0 0.96\nH 0.93 0.0 -0.24\nO 0.0 0.0 2.90\nH 0.0 0.76 3.15\nH 0.0 -0.76 3.15\n",
            0.0,
        )
        .unwrap();
        let e = h4_energy(&mol);
        assert!(e < 0.0, "water dimer H4 energy {e} eV should be attractive");
    }

    #[test]
    fn hh_repulsion_is_positive() {
        // Two hydrogens 1.2 Å apart repel.
        let mol = Molecule::from_xyz_str("2\nh2\nH 0 0 0\nH 0 0 1.2\n", 0.0).unwrap();
        assert!(hh_rep_energy(&mol) > 0.0);
    }
}
