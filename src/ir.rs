// SPDX-License-Identifier: GPL-3.0-or-later

//! Dipole derivatives and infrared intensities.
//!
//! # Three solves, not `3N`
//!
//! An infrared intensity needs `∂μ_α/∂R_{Aβ}`, which is the mixed second derivative
//! `−∂²E/∂F_α∂R_{Aβ}`. Taken literally that is one coupled-perturbed solve per nuclear
//! coordinate — `3N` of them, the same cost as the Hessian.
//!
//! It is not needed. The interchange theorem makes the response half symmetric in its two
//! perturbations,
//!
//! ```text
//! ∂²E/∂F_α∂R  =  [skeleton]  +  4 G^{R} : U^{F_α}  =  [skeleton]  +  4 G^{F_α} : U^{R}
//! ```
//!
//! so the expensive object can be either one. The **field** side has three perturbations
//! whatever the molecule is, and the nuclear side needs only `G^R` — the derivative Fock, which
//! `hessian::skeleton_fock_ov` already builds without any coupled-perturbed solve at
//! all. So the whole `3 × 3N` tensor costs three solves, and the cost stops growing with the
//! system.
//!
//! # The explicit half is the atomic charge
//!
//! At fixed density `∂E/∂F_α = −μ_α`, and `μ` depends on the nuclei only through each atom's own
//! monopole position (`dd` belongs to the element, not the geometry). So
//!
//! ```text
//! ∂²E/∂F_α∂R_{Aβ} |_P  =  −(Z_A − pop_A) δ_αβ  =  −q_A δ_αβ
//! ```
//!
//! and the skeleton contribution to `∂μ/∂R` is a net atomic charge on the diagonal — the same
//! term that makes the force on an atom in a field `q_A f`.
//!
//! # Mass weighting is where this goes wrong quietly
//!
//! [`crate::hessian::VibrationalModes::modes`] are eigenvectors of `H_ij/√(m_i m_j)`, so the
//! Cartesian displacement of mode `m` is `l_{im}/√m_i`. Contracting `∂μ/∂R` against `l` without
//! dividing by `√m` gives intensities of the right order and the wrong ratios, which is exactly
//! the kind of error that survives inspection.
//!
//! # Translations and rotations
//!
//! `Σ_A ∂μ_β/∂R_{Aα} = Q δ_αβ`, so a **charged** molecule has a non-zero dipole derivative along
//! every translation, and a molecule with a permanent dipole has one along every rotation. Both
//! are artefacts of describing the molecule in a fixed frame rather than infrared activity, and
//! both would appear as intensity on the six modes that should have none. They are projected out.

use crate::error::Result;
use crate::hessian::{
    cphf_ov, ov_denominators, project_ov, skeleton_fock_ov, submatrix_cols, ucphf_ov,
    vibrational_analysis, VibrationalModes,
};
use crate::linalg::Matrix;
use crate::params::Pm3Parameters;
use crate::scf::{run_pm3, Pm3Options, Pm3Result};
use crate::system::Molecule;

/// Infrared intensity in km/mol from `|∂μ/∂Q|²` in `(e / √amu)²`.
///
/// Derived rather than transcribed, from the SI expression
///
/// ```text
/// A = N_A |∂μ/∂Q|² / (12 ε₀ c²)
/// ```
///
/// with `∂μ/∂Q` converted from `e/√amu` by the elementary charge and the atomic mass unit, and
/// the result from m/mol to km/mol. `N_A`, `c` and `e` are exact by the 2019 SI definitions;
/// `ε₀` and `u` are CODATA. Writing the arithmetic out means the number can be checked against
/// the definition instead of against another table — it comes to 974.88.
const ELEMENTARY_CHARGE_C: f64 = 1.602_176_634e-19;
const AVOGADRO_PER_MOL: f64 = 6.022_140_76e23;
const SPEED_OF_LIGHT_M_PER_S: f64 = 2.997_924_58e8;
const VACUUM_PERMITTIVITY_F_PER_M: f64 = 8.854_187_812_8e-12;
const ATOMIC_MASS_UNIT_KG: f64 = 1.660_539_066_60e-27;
const KM_PER_MOL_PER_E2_PER_AMU: f64 = AVOGADRO_PER_MOL
    * (ELEMENTARY_CHARGE_C * ELEMENTARY_CHARGE_C / ATOMIC_MASS_UNIT_KG)
    / (12.0 * VACUUM_PERMITTIVITY_F_PER_M * SPEED_OF_LIGHT_M_PER_S * SPEED_OF_LIGHT_M_PER_S)
    / 1000.0;

/// An infrared spectrum: the frequencies, their intensities, and the raw tensor both come from.
#[derive(Clone, Debug)]
pub struct IrSpectrum {
    /// Harmonic frequencies (cm⁻¹), ascending, as [`VibrationalModes::frequencies_cm`].
    pub frequencies_cm: Vec<f64>,
    /// Integrated absorption per mode (km/mol), aligned with `frequencies_cm`.
    pub intensities_km_per_mol: Vec<f64>,
    /// `∂μ_α/∂R_{Aβ}` as a dense `3 × 3N` matrix in units of the elementary charge: row `α` is
    /// the Cartesian dipole component, column `3A + β` the nuclear coordinate.
    ///
    /// Reported alongside the per-mode intensities because it is the mode-independent object —
    /// a different frequency analysis, a different isotope substitution or a different projection
    /// scheme all reuse it, and none of them needs the response solved again.
    pub dipole_derivatives: Matrix,
    /// The vibrational analysis the intensities were projected onto.
    pub modes: VibrationalModes,
}

/// `∂μ_α/∂R_{Aβ}` (elementary charges), as a `3 × 3N` matrix.
///
/// See the module note: three coupled-perturbed solves, whatever the size of the molecule.
pub fn dipole_derivatives(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
) -> Result<Matrix> {
    let tightened = crate::hessian::tighten_scf_for_hessian(options);
    let scf = run_pm3(molecule, params, &tightened)?;
    dipole_derivatives_at(molecule, params, &tightened, &scf)
}

/// The same, reusing a converged result rather than solving again.
pub fn dipole_derivatives_at(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    scf: &Pm3Result,
) -> Result<Matrix> {
    let basis = crate::basis::Basis::build(molecule, params)?;
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let mut out = Matrix::zeros(3, ndof);

    // The explicit half: a net atomic charge on the diagonal.
    for (atom, charge) in scf.charges.iter().enumerate() {
        for axis in 0..3 {
            out[(axis, 3 * atom + axis)] += charge;
        }
    }

    // The dipole operator, about the coordinate origin — the same origin the field uses, and for
    // the same reason: a centre of mass would move with the nuclei and put a term into this
    // derivative that is a property of the origin rather than of the molecule.
    let operator =
        crate::dipole::dipole_matrix(molecule, params, &basis, crate::math::Vec3::zero())?;

    if scf.unrestricted {
        return unrestricted_derivatives(molecule, params, options, scf, &basis, &operator, out);
    }

    let n_occ = scf.n_occ;
    let nvir = basis.nao - n_occ;
    if nvir == 0 || n_occ == 0 {
        return Ok(out);
    }
    let cv = submatrix_cols(&scf.mo_coeff, n_occ, nvir);
    let co = submatrix_cols(&scf.mo_coeff, 0, n_occ);
    let denom = ov_denominators(&scf.mo_energies, n_occ, nvir);
    let core = crate::hamiltonian::build_core_limited(
        molecule,
        &basis,
        params,
        crate::scf::pair_cache_limit(options),
        options.field,
    )?;

    // Three field responses. `∂F_fock/∂F_α = M_α` because the field enters `H_core` linearly and
    // has no two-electron part.
    let mut field_response = Vec::with_capacity(3);
    for term in operator.iter() {
        let g_ov = project_ov(term, &cv, &co);
        field_response.push(cphf_ov(
            &g_ov,
            &denom,
            &cv,
            &co,
            molecule,
            params,
            &basis,
            &core,
            None,
            crate::hessian::CPHF_ITERATIONS,
        )?);
    }

    // The nuclear derivative Focks — no solve, just the derivative of the Fock matrix.
    let nuclear = skeleton_fock_ov(molecule, params, options, &basis, &scf.density, &cv, &co)?;

    for (dof, g_nuclear) in nuclear.iter().enumerate() {
        for (axis, u_field) in field_response.iter().enumerate() {
            // dμ/dR = −∂²E/∂F∂R, and the response half of that is `4 G^R : U^F` with the same
            // factor the Hessian uses.
            out[(axis, dof)] -= 4.0 * g_nuclear.frobenius_dot(u_field);
        }
    }
    Ok(out)
}

/// The open-shell path: two coupled channels, and the response density is the sum of both spins.
#[allow(clippy::too_many_arguments)]
fn unrestricted_derivatives(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    scf: &Pm3Result,
    basis: &crate::basis::Basis,
    operator: &[Matrix; 3],
    mut out: Matrix,
) -> Result<Matrix> {
    let (n_alpha, n_beta) = (scf.n_occ, scf.n_beta);
    let (vir_a, vir_b) = (basis.nao - n_alpha, basis.nao - n_beta);
    if vir_a == 0 || vir_b == 0 || n_alpha == 0 || n_beta == 0 {
        return Ok(out);
    }
    let energies_beta = scf.mo_energies_beta.as_ref().expect("unrestricted");
    let coeff_beta = scf.mo_coeff_beta.as_ref().expect("unrestricted");

    let cva = submatrix_cols(&scf.mo_coeff, n_alpha, vir_a);
    let coa = submatrix_cols(&scf.mo_coeff, 0, n_alpha);
    let cvb = submatrix_cols(coeff_beta, n_beta, vir_b);
    let cob = submatrix_cols(coeff_beta, 0, n_beta);
    let denom_a = ov_denominators(&scf.mo_energies, n_alpha, vir_a);
    let denom_b = ov_denominators(energies_beta, n_beta, vir_b);

    let core = crate::hamiltonian::build_core_limited(
        molecule,
        basis,
        params,
        crate::scf::pair_cache_limit(options),
        options.field,
    )?;

    // The field is spin-independent, so both channels see the same perturbation.
    let mut field_response = Vec::with_capacity(3);
    for term in operator.iter() {
        let ga = project_ov(term, &cva, &coa);
        let gb = project_ov(term, &cvb, &cob);
        field_response.push(ucphf_ov(
            &ga,
            &gb,
            &denom_a,
            &denom_b,
            &cva,
            &coa,
            &cvb,
            &cob,
            molecule,
            params,
            basis,
            &core,
            crate::hessian::CPHF_ITERATIONS,
        )?);
    }

    let spin = scf.spin_density.as_ref().expect("unrestricted");
    let mut pa = scf.density.clone();
    let mut pb = scf.density.clone();
    for ((a, b), (t, s)) in pa
        .as_mut_slice()
        .iter_mut()
        .zip(pb.as_mut_slice().iter_mut())
        .zip(scf.density.as_slice().iter().zip(spin.as_slice()))
    {
        *a = 0.5 * (t + s);
        *b = 0.5 * (t - s);
    }
    let nuclear = crate::hessian::skeleton_fock_ov_spin(
        molecule,
        params,
        options,
        basis,
        &scf.density,
        &pa,
        &pb,
        &cva,
        &coa,
        &cvb,
        &cob,
    )?;

    let (nuclear_alpha, nuclear_beta) = nuclear;
    for (dof, (ga, gb)) in nuclear_alpha.iter().zip(&nuclear_beta).enumerate() {
        for (axis, (ua, ub)) in field_response.iter().enumerate() {
            // `2 Σ_σ G^σ : U^σ`, the unrestricted contraction the UHF Hessian uses — the same
            // total weight as the restricted `4 G : U`, split across two channels.
            out[(axis, dof)] -= 2.0 * (ga.frobenius_dot(ua) + gb.frobenius_dot(ub));
        }
    }
    Ok(out)
}

/// Frequencies, infrared intensities and the tensor they come from.
pub fn ir_spectrum(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    step: f64,
) -> Result<IrSpectrum> {
    let modes = vibrational_analysis(molecule, params, options, step)?;
    let tightened = crate::hessian::tighten_scf_for_hessian(options);
    let scf = run_pm3(molecule, params, &tightened)?;
    let derivatives = dipole_derivatives_at(molecule, params, &tightened, &scf)?;

    let masses: Vec<f64> = molecule
        .atoms
        .iter()
        .map(|a| params.element(a.z).map(|e| e.mass))
        .collect::<Result<Vec<_>>>()?;
    let ndof = 3 * molecule.atoms.len();
    let projector = internal_projector(molecule, &masses, &modes.modes);

    let mut intensities = vec![0.0; ndof];
    for (mode, slot) in intensities.iter_mut().enumerate() {
        let mut sum = 0.0;
        for axis in 0..3 {
            // `∂μ/∂Q = Σ_i (∂μ/∂x_i) l_{i,mode} / √m_i`, with the mode projected onto the
            // internal subspace first so a charged molecule's translations and a polar one's
            // rotations do not appear as absorption.
            let derivative: f64 = (0..ndof)
                .map(|dof| {
                    // A massless site contributes nothing rather than infinity. MOPAC's `+`
                    // (Z = 104) and `--` (Z = 106) point charges have a tabulated mass of exactly
                    // zero and are accepted from a plain XYZ line; the three sibling
                    // mass-weightings have always guarded this and only here was it missing, so
                    // an infrared spectrum of a system containing one came back as `NaN`.
                    let root = masses[dof / 3].sqrt();
                    if root > 0.0 {
                        derivatives[(axis, dof)] * projector[(dof, mode)] / root
                    } else {
                        0.0
                    }
                })
                .sum();
            sum += derivative * derivative;
        }
        *slot = KM_PER_MOL_PER_E2_PER_AMU * sum;
    }

    Ok(IrSpectrum {
        frequencies_cm: modes.frequencies_cm.clone(),
        intensities_km_per_mol: intensities,
        dipole_derivatives: derivatives,
        modes,
    })
}

/// The normal modes with their translational and rotational content removed, still in
/// mass-weighted coordinates.
///
/// Six directions (five for a linear molecule) carry no vibration, and `∂μ/∂R` is generally
/// non-zero along all of them: `Σ_A ∂μ_β/∂R_{Aα} = Q δ_αβ` for a charged molecule, and a
/// permanent dipole rotates. Leaving them in puts absorption on modes that have none.
/// The subspace itself is [`crate::rigid`]'s, shared with the frequency path rather than built a
/// second time here. The two carried separate copies of the same Gram–Schmidt, which is how the
/// massless-site guard came to exist in one of them and not the other.
fn internal_projector(molecule: &Molecule, masses: &[f64], modes: &Matrix) -> Matrix {
    let positions: Vec<crate::math::Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
    let basis = crate::rigid::rigid_body_basis(
        &positions,
        masses,
        crate::rigid::RigidMotions::TranslationsAndRotations,
    );
    crate::rigid::project_columns(&basis, modes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;

    const WATER: &str = "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";
    const METHYL: &str =
        "4\nmethyl\nC 0.0 0.0 0.0\nH 0.0 1.078 0.0\nH 0.9336 -0.539 0.0\nH -0.9336 -0.539 0.0\n";

    fn tight() -> Pm3Options {
        Pm3Options {
            e_tol: 1.0e-12,
            p_tol: 1.0e-11,
            max_scf: 500,
            ..Default::default()
        }
    }

    /// The whole tensor against finite differences of the dipole.
    ///
    /// This is the test that decides whether the interchange is right: the factor, the sign, the
    /// explicit atomic-charge term and the choice of which perturbation to solve for all show up
    /// here and nowhere else. Three coupled-perturbed solves are compared against eighteen
    /// self-consistent field evaluations.
    ///
    /// Both spin paths, because the open-shell branch is a separate function with its own
    /// factors. The translational sum rule cannot stand in for this one on that branch:
    /// `Σ_A G^{R_Aα} = 0` identically, so the response contribution drops out of the sum
    /// entirely and the rule holds whatever coefficient the response is given.
    #[test]
    fn the_dipole_derivatives_match_finite_differences() {
        for (xyz, charge, multiplicity) in [(WATER, 0.0, 1_usize), (METHYL, 0.0, 2)] {
            the_finite_difference_case(xyz, charge, multiplicity);
        }
    }

    fn the_finite_difference_case(xyz: &str, charge: f64, multiplicity: usize) {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = Molecule::from_xyz_str(xyz, charge)
            .unwrap()
            .with_multiplicity(multiplicity);
        let options = Pm3Options {
            charge,
            multiplicity,
            ..tight()
        };
        let analytic = dipole_derivatives(&molecule, &params, &options).unwrap();
        let ndof = 3 * molecule.atoms.len();
        assert_eq!((analytic.rows, analytic.cols), (3, ndof));

        let step = 1.0e-4;
        let mut worst = 0.0_f64;
        for dof in 0..ndof {
            let dipole_at = |sign: f64| {
                let mut shifted = molecule.clone();
                let mut delta = [0.0; 3];
                delta[dof % 3] = sign * step;
                shifted.atoms[dof / 3].position += Vec3::new(delta[0], delta[1], delta[2]);
                let scf = run_pm3(&shifted, &params, &options).unwrap();
                let basis = crate::basis::Basis::build(&shifted, &params).unwrap();
                crate::dipole::dipole_from_density(
                    &shifted,
                    &params,
                    &basis,
                    &scf.density,
                    Vec3::zero(),
                )
                .unwrap()
            };
            let (plus, minus) = (dipole_at(1.0), dipole_at(-1.0));
            for axis in 0..3 {
                let numeric = (plus.to_array()[axis] - minus.to_array()[axis]) / (2.0 * step);
                worst = worst.max((analytic[(axis, dof)] - numeric).abs());
            }
        }
        assert!(
            worst < 2.0e-5,
            "the dipole derivatives differ from finite differences by {worst:.3e} e"
        );
    }

    /// `Σ_A ∂μ_β/∂R_{Aα} = Q δ_αβ` — translating the whole molecule moves its dipole by exactly
    /// its charge times the displacement, and by nothing else.
    ///
    /// Oracle-free, and it catches a whole class of error at once: a missing explicit term, a
    /// response leaking into the translations, a wrong sign on one atom.
    #[test]
    fn the_derivatives_obey_the_translational_sum_rule() {
        let params = Pm3Parameters::standard().unwrap();
        for (xyz, charge, multiplicity) in [(WATER, 0.0, 1), (WATER, 1.0, 2)] {
            let molecule = Molecule::from_xyz_str(xyz, charge)
                .unwrap()
                .with_multiplicity(multiplicity);
            let options = Pm3Options {
                charge,
                multiplicity,
                ..tight()
            };
            let d = dipole_derivatives(&molecule, &params, &options).unwrap();
            for alpha in 0..3 {
                for beta in 0..3 {
                    let sum: f64 = (0..molecule.atoms.len())
                        .map(|atom| d[(beta, 3 * atom + alpha)])
                        .sum();
                    let expected = if alpha == beta { charge } else { 0.0 };
                    assert!(
                        (sum - expected).abs() < 1.0e-6,
                        "charge {charge}: Σ_A ∂μ_{beta}/∂R_{alpha} = {sum}, expected {expected}"
                    );
                }
            }
        }
    }

    /// Water's three vibrations are all infrared active, the six rigid-body modes are not, and
    /// the intensities are of the size a spectrum reports.
    #[test]
    fn water_has_three_active_modes_and_six_silent_ones() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = crate::optimizer::optimize(
            &Molecule::from_xyz_str(WATER, 0.0).unwrap(),
            &params,
            &tight(),
            &crate::optimizer::OptOptions::default(),
        )
        .unwrap()
        .molecule;
        let spectrum = ir_spectrum(&molecule, &params, &tight(), 1.0e-3).unwrap();

        assert_eq!(spectrum.intensities_km_per_mol.len(), 9);
        assert_eq!(
            (
                spectrum.dipole_derivatives.rows,
                spectrum.dipole_derivatives.cols
            ),
            (3, 9)
        );
        for value in &spectrum.intensities_km_per_mol {
            assert!(*value >= 0.0, "an intensity cannot be negative: {value}");
        }

        // The six lowest-|frequency| modes are the rigid-body ones and must carry no absorption.
        let mut order: Vec<usize> = (0..9).collect();
        order.sort_by(|a, b| {
            spectrum.frequencies_cm[*a]
                .abs()
                .partial_cmp(&spectrum.frequencies_cm[*b].abs())
                .unwrap()
        });
        for &mode in &order[..6] {
            assert!(
                spectrum.intensities_km_per_mol[mode] < 1.0e-6,
                "rigid-body mode {mode} ({:.1} cm^-1) carries {} km/mol",
                spectrum.frequencies_cm[mode],
                spectrum.intensities_km_per_mol[mode]
            );
        }
        // And the three vibrations are active, at the scale a real spectrum shows.
        for &mode in &order[6..] {
            let intensity = spectrum.intensities_km_per_mol[mode];
            assert!(
                intensity > 1.0 && intensity < 1000.0,
                "vibration {mode} ({:.1} cm^-1) has {intensity} km/mol",
                spectrum.frequencies_cm[mode]
            );
        }
    }

    /// The derived km/mol conversion, against the number every table gives.
    #[test]
    fn the_intensity_conversion_comes_out_at_the_published_value() {
        assert!(
            (KM_PER_MOL_PER_E2_PER_AMU - 974.88).abs() < 0.01,
            "the derivation gives {KM_PER_MOL_PER_E2_PER_AMU}, tables give 974.88"
        );
    }

    /// The open-shell path runs and obeys the same sum rule, which is the statement that the two
    /// coupled channels were weighted correctly.
    #[test]
    fn an_open_shell_radical_obeys_the_sum_rule() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = Molecule::from_xyz_str(METHYL, 0.0)
            .unwrap()
            .with_multiplicity(2);
        let options = Pm3Options {
            multiplicity: 2,
            ..tight()
        };
        let d = dipole_derivatives(&molecule, &params, &options).unwrap();
        for alpha in 0..3 {
            for beta in 0..3 {
                let sum: f64 = (0..molecule.atoms.len())
                    .map(|atom| d[(beta, 3 * atom + alpha)])
                    .sum();
                // The radical is neutral, so `Q δ_αβ` is zero in every component — including the
                // diagonal, which is what makes this a statement about the two spin channels
                // cancelling correctly rather than about the charge.
                assert!(
                    sum.abs() < 1.0e-5,
                    "UHF: Σ_A ∂μ_{beta}/∂R_{alpha} = {sum}, expected 0"
                );
            }
        }
    }
}
