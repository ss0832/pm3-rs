// SPDX-License-Identifier: GPL-3.0-or-later

//! Halogen-bond correction (the "X" of PM3-D3H4X), using the published
//! D3H4X correction form and MOPAC `corrections/disp_DnX.F90` parameters.
//!
//! `E_X = Σ_{X, A} a_X[X,A]·exp(b_X[X,A]·R_XA)` over halogen atoms
//! `X ∈ {Cl, Br, I}` and acceptor atoms `A ∈ {N, O, S}` (Å, kcal/mol).
//!
//! PROVENANCE: J. Řezáč, P. Hobza, *Chem. Phys. Lett.* **506**, 286 (2011);
//! parameters via MOPAC (Apache-2.0). See THIRD_PARTY_NOTICES.md.

use super::dist_g;
use crate::constants::{BOHR_TO_ANGSTROM, KCAL_TO_EV};
use crate::dual::Scalar;
use crate::system::Molecule;

/// Halogen index (Cl=0, Br=1, I=2) or `None`.
fn halogen_index(z: u8) -> Option<usize> {
    match z {
        17 => Some(0),
        35 => Some(1),
        53 => Some(2),
        _ => None,
    }
}

/// Acceptor index (N=0, O=1, S=2) or `None`.
fn acceptor_index(z: u8) -> Option<usize> {
    match z {
        7 => Some(0),
        8 => Some(1),
        16 => Some(2),
        _ => None,
    }
}

// a_X[halogen][acceptor] (kcal/mol) and b_X[halogen][acceptor] (Å⁻¹),
// disp_DnX.F90 first parameter set. S entries present only for I.
const A_X: [[f64; 3]; 3] = [
    [1.0489e12, 4.6783e8, 0.0],     // Cl - N, Cl - O, Cl - S
    [1.0226e5, 9.6021e3, 0.0],      // Br - N, Br - O, Br - S
    [1.2751e12, 6.0912e5, 1.051e6], // I - N, I - O, I - S
];
const B_X: [[f64; 3]; 3] = [
    [-9.946, -6.867, 0.0],
    [-3.236, -2.900, 0.0],
    [-9.534, -4.154, -3.82],
];

/// Halogen-bond correction energy (eV), f64 entry point.
pub fn hx_energy(mol: &Molecule) -> f64 {
    let (numbers, pos) = super::geometry_f64(mol);
    hx_energy_g::<f64>(&numbers, &pos)
}

/// Halogen-bond correction energy (eV), generic over the scalar.
pub fn hx_energy_g<S: Scalar>(numbers: &[u8], pos: &[[S; 3]]) -> S {
    hx_energy_cluster_g(numbers, pos, numbers.len())
}

/// Halogen-bond correction over an image-expanded cluster, generic over the scalar.
///
/// `n_cell` is how many leading entries belong to the reference cell. The term is *directed* —
/// halogen `i` donating to acceptor `j` is not the same as the reverse, and the parameter table
/// `A_X`/`B_X` is asymmetric — so each ordered pair has exactly one halogen, and assigning the
/// pair to the cell that halogen sits in counts every crystal-distinct pair once with no
/// fractional weight. With `n_cell = numbers.len()` this is the molecular sum unchanged.
pub fn hx_energy_cluster_g<S: Scalar>(numbers: &[u8], pos: &[[S; 3]], n_cell: usize) -> S {
    let n = numbers.len();
    let mut sum_kcal = S::cst(0.0);
    for i in 0..n_cell {
        for j in 0..n {
            if i == j {
                continue;
            }
            let (Some(x), Some(a)) = (halogen_index(numbers[i]), acceptor_index(numbers[j])) else {
                continue;
            };
            let a_x = A_X[x][a];
            if a_x == 0.0 {
                continue;
            }
            let rab = dist_g(&pos[j], &pos[i]) * BOHR_TO_ANGSTROM;
            if rab.val() > 8.0 {
                continue;
            }
            sum_kcal = sum_kcal + (rab * B_X[x][a]).exp() * a_x;
        }
    }
    sum_kcal * KCAL_TO_EV
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_halogen_no_correction() {
        let mol = Molecule::from_xyz_str("2\nh2\nH 0 0 0\nH 0 0 0.74\n", 0.0).unwrap();
        assert_eq!(hx_energy(&mol), 0.0);
    }

    #[test]
    fn halogen_acceptor_pair_is_small_and_finite() {
        // Cl···N at ~3 Å: a small short-range repulsive correction that
        // reshapes PM3's halogen-bond potential (a_X·exp(b_X·R), a_X > 0).
        let mol = Molecule::from_xyz_str("2\ncln\nCl 0 0 0\nN 0 0 3.0\n", 0.0).unwrap();
        let e = hx_energy(&mol);
        assert!(
            e.is_finite() && e.abs() < 0.5 && e > 0.0,
            "Cl-N X energy {e} eV"
        );
        // The correction decays with distance.
        let far = Molecule::from_xyz_str("2\ncln\nCl 0 0 0\nN 0 0 5.0\n", 0.0).unwrap();
        assert!(hx_energy(&far) < e, "X correction must decay with distance");
    }
}
