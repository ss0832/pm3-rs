// SPDX-License-Identifier: GPL-3.0-or-later
//! MOPAC's `MMOK` molecular-mechanics correction to the amide bond.
//!
//! PM3 (like AM1 and MNDO) puts the barrier to rotation about a peptide bond far too low,
//! because an NDDO valence Hamiltonian has no term for the amide's partial double-bond
//! character. MOPAC's answer is not to change the Hamiltonian but to add a classical
//! torsion term to the heat of formation afterwards, switched on by the `MMOK` keyword —
//! which MOPAC applies **by default**, and `NOMM` turns off.
//!
//! ## The form, and where it comes from
//!
//! The correction is, per hydrogen on an amide nitrogen,
//!
//! ```text
//! E = K · sin²(φ),    φ = the O=C–N–H torsion,    K = 7.1853 kcal/mol
//! ```
//!
//! summed over every such hydrogen. It vanishes for a planar amide and is largest at 90°,
//! so what it does is *raise the twisted geometry* — that is, restore the rotation barrier
//! without touching the planar minimum.
//!
//! Both the form and the constant were measured against MOPAC v23.2.5 rather than copied:
//! `MMOK` and `NOMM` differ only in this term, so scanning the torsion of formamide and
//! subtracting gives it directly. The fit is exact to the digits MOPAC prints — at 45° the
//! measured correction is 3.592650 kcal/mol for one twisted hydrogen against a predicted
//! `7.1853 × sin²(45°) = 3.592650` — and on acetamide's own oracle geometry, where the two
//! torsions are 27.730° and 160.879°, the prediction of 2.326650 kcal/mol meets MOPAC's
//! 2.3266497. `tests/mopac_oracle.rs::the_oracle_is_plain_pm3_not_mopacs_default` pins the
//! acetamide half against MOPAC's own default-keyword number.
//!
//! ## Why it is off by default here
//!
//! It is not part of PM3. A heat of formation with it in is not comparable to a published
//! PM3 number, and `pm3-rs` reports the Hamiltonian's own answer unless asked otherwise.
//! Turn it on with [`crate::scf::Pm3Options::mmok`] when reproducing a MOPAC default run.
//!
//! The term is written over [`Scalar`], so the same code supplies its gradient and Hessian
//! contributions through the crate's forward-mode dual numbers — an optimization with
//! `mmok` on minimizes the surface it actually reports.

use crate::dual::Scalar;

/// Torsion force constant, kcal/mol per unit `sin²φ`. Measured against MOPAC v23.2.5.
pub const AMIDE_K_KCAL: f64 = 7.185_300;

/// `AMIDE_K_KCAL` in eV, which is what the rest of the crate works in.
const AMIDE_K_EV: f64 = AMIDE_K_KCAL / crate::constants::EV_TO_KCAL;

// Bond-perception cutoffs, **in Bohr** because that is what the correction module's
// coordinates are in. Written as Angstrom and converted, because the chemistry is what the
// numbers mean: generous enough to survive a distorted geometry, tight enough that a
// carbonyl is not confused with an ether (C=O 1.22 vs C-O 1.43) and an amide N-H is not
// confused with a hydrogen bond.
const C_O_DOUBLE_MAX: f64 = 1.35 / crate::constants::BOHR_TO_ANGSTROM;
const C_N_MAX: f64 = 1.65 / crate::constants::BOHR_TO_ANGSTROM;
const N_H_MAX: f64 = 1.30 / crate::constants::BOHR_TO_ANGSTROM;

const H: u8 = 1;
const C: u8 = 6;
const N: u8 = 7;
const O: u8 = 8;

fn distance2(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).powi(2)).sum()
}

/// `sin²` of the `p0–p1–p2–p3` torsion, without trigonometry.
///
/// With `b1 = p1−p0`, `b2 = p2−p1`, `b3 = p3−p2` and `n1 = b1×b2`, `n2 = b2×b3`, the
/// torsion satisfies `sin φ = ((n1×n2)·b̂2)/(|n1||n2|)`, so
/// `sin²φ = ((n1×n2)·b2)² / (|n1|²|n2|²|b2|²)` — all polynomial, which keeps the dual
/// numbers exact and avoids the `acos` branch cut at a planar amide, exactly where these
/// molecules sit.
fn sin2_torsion<S: Scalar>(p0: &[S; 3], p1: &[S; 3], p2: &[S; 3], p3: &[S; 3]) -> S {
    let sub = |a: &[S; 3], b: &[S; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [S; 3], b: [S; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: [S; 3], b: [S; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

    let b1 = sub(p1, p0);
    let b2 = sub(p2, p1);
    let b3 = sub(p3, p2);
    let n1 = cross(b1, b2);
    let n2 = cross(b2, b3);
    let numerator = dot(cross(n1, n2), b2).powi(2);
    let denominator = dot(n1, n1) * dot(n2, n2) * dot(b2, b2);
    // Collinear atoms leave the torsion undefined; the term is zero there rather than NaN.
    if denominator.val() <= 1.0e-30 {
        return S::cst(0.0);
    }
    numerator / denominator
}

/// Every `(O, C, N, H)` quadruple that MOPAC would treat as an amide linkage.
///
/// The rule is the chemical one: a nitrogen carrying at least one hydrogen, bonded to a
/// carbon that carries a short (double) bond to oxygen. Each N–H on such a nitrogen
/// contributes its own term, which is what the measurement says MOPAC does — moving one
/// hydrogen of formamide's `NH2` out of plane and leaving the other gives exactly half the
/// correction of moving both.
pub fn amide_torsions(numbers: &[u8], pos: &[[f64; 3]]) -> Vec<(usize, usize, usize, usize)> {
    let bonded = |i: usize, j: usize, max: f64| distance2(&pos[i], &pos[j]) < max * max;
    let mut out = Vec::new();
    for (n_i, &z_n) in numbers.iter().enumerate() {
        if z_n != N {
            continue;
        }
        for (c_i, &z_c) in numbers.iter().enumerate() {
            if z_c != C || !bonded(n_i, c_i, C_N_MAX) {
                continue;
            }
            // The carbonyl oxygen. If a carbon carries two of them (a carbamate, say) each
            // defines a torsion, and MOPAC counts each — so this does not break early.
            for (o_i, &z_o) in numbers.iter().enumerate() {
                if z_o != O || !bonded(c_i, o_i, C_O_DOUBLE_MAX) {
                    continue;
                }
                for (h_i, &z_h) in numbers.iter().enumerate() {
                    if z_h == H && bonded(n_i, h_i, N_H_MAX) {
                        out.push((o_i, c_i, n_i, h_i));
                    }
                }
            }
        }
    }
    out
}

/// The correction in eV, generic over the scalar so gradients come from the same code.
///
/// Topology is decided on the *values* of the coordinates, not the dual parts: which atoms
/// form an amide is a discrete fact about the structure, and letting it depend on the
/// derivative direction would make the energy discontinuous at every cutoff.
pub fn amide_correction_g<S: Scalar>(numbers: &[u8], pos: &[[S; 3]]) -> S {
    let values: Vec<[f64; 3]> = pos
        .iter()
        .map(|p| [p[0].val(), p[1].val(), p[2].val()])
        .collect();
    let mut e = S::cst(0.0);
    for (o_i, c_i, n_i, h_i) in amide_torsions(numbers, &values) {
        e = e + sin2_torsion(&pos[o_i], &pos[c_i], &pos[n_i], &pos[h_i]) * AMIDE_K_EV;
    }
    e
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A planar amide is the minimum of the term, and a perpendicular one its maximum.
    #[test]
    fn the_correction_vanishes_for_a_planar_amide_and_peaks_at_ninety_degrees() {
        // O=C-N-H laid out in one plane, then the hydrogen lifted out of it.
        let numbers = [O, C, N, H];
        let planar = [
            [0.0, 1.22, 0.0],
            [0.0, 0.0, 0.0],
            [1.36, 0.0, 0.0],
            [1.86, 0.87, 0.0],
        ];
        let flat = amide_correction_g(&numbers, &planar);
        assert!(
            flat.abs() < 1e-12,
            "planar amide should cost nothing: {flat}"
        );

        let perpendicular = [
            [0.0, 1.22, 0.0],
            [0.0, 0.0, 0.0],
            [1.36, 0.0, 0.0],
            [1.86, 0.0, 0.87],
        ];
        let twisted = amide_correction_g(&numbers, &perpendicular);
        let expected = AMIDE_K_KCAL / crate::constants::EV_TO_KCAL;
        assert!(
            (twisted - expected).abs() < 1e-9,
            "perpendicular amide should cost the full constant: {twisted} vs {expected}"
        );
    }

    /// An amine is not an amide, and neither is an ester: the term must not fire on either.
    #[test]
    fn only_an_amide_nitrogen_counts() {
        // Methylamine-like N-H with no carbonyl anywhere.
        let amine = [N, H, C];
        let amine_pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.47, 0.0]];
        assert_eq!(amide_torsions(&amine, &amine_pos).len(), 0);

        // A carbonyl with no N-H: the nitrogen is there but carries no hydrogen.
        let ester = [O, C, N, C];
        let ester_pos = [
            [0.0, 1.22, 0.0],
            [0.0, 0.0, 0.0],
            [1.36, 0.0, 0.0],
            [1.86, 1.4, 0.0],
        ];
        assert_eq!(amide_torsions(&ester, &ester_pos).len(), 0);
    }

    /// Both hydrogens of an `NH2` amide contribute, independently.
    #[test]
    fn each_amide_hydrogen_contributes_its_own_term() {
        let numbers = [O, C, N, H, H];
        let pos = [
            [0.0, 1.22, 0.0],
            [0.0, 0.0, 0.0],
            [1.36, 0.0, 0.0],
            [1.86, 0.0, 0.87],
            [1.86, 0.0, -0.87],
        ];
        assert_eq!(amide_torsions(&numbers, &pos).len(), 2);
        let e = amide_correction_g(&numbers, &pos);
        let one = AMIDE_K_KCAL / crate::constants::EV_TO_KCAL;
        assert!(
            (e - 2.0 * one).abs() < 1e-9,
            "expected two full terms, got {e}"
        );
    }
}
