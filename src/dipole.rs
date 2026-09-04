// SPDX-License-Identifier: GPL-3.0-or-later

//! The dipole operator in the AO basis, and the two things that read it.
//!
//! # One operator, two uses
//!
//! A molecule's dipole moment and its coupling to a uniform electric field are the same object
//! seen from two sides: `μ = −∂E/∂F`, so whatever matrix gives the dipole as an expectation value
//! is the matrix the field multiplies in the Hamiltonian. Writing them separately is how the two
//! drift apart, and the drift is invisible — each looks right on its own and only their
//! *relationship* is wrong.
//!
//! # What survives zero differential overlap
//!
//! NDDO treats the AO basis as orthonormal and neglects differential overlap between atoms, so
//! `⟨μ_A | r | ν_B⟩` vanishes for `A ≠ B`. Only on-atom matrix elements survive, and within an
//! atom the PM3 multipole model leaves exactly two:
//!
//! | element | value | what it is |
//! |---|---|---|
//! | `M_α[μ_A, μ_A]` | `(R_A − O)_α`, every AO on `A` | the monopole: charge sits at the nucleus |
//! | `M_α[s_A, p_{α,A}]` | `dd_A` | the `s`–`p` hybridization dipole |
//!
//! so that
//!
//! ```text
//! μ_α = Σ_A Z_A^core (R_A − O)_α  −  Tr[P M_α]
//! ```
//!
//! Expanding the trace gives `Σ_A q_A (R_A − O)_α − 2 dd_A P[s, p_α]`, which is MOPAC's
//! `dipole.F90` term for term — the factor two being the `μν` and `νμ` halves of the trace.
//!
//! # The origin is part of the definition, and the two uses need different ones
//!
//! `Σ q_i r_i` moves with the origin whenever the system carries a net charge. MOPAC references
//! the **dipole** to the centre of mass, with its massless point atoms `+`/`−` moving the charge
//! distribution without moving the origin, and [`centre_of_mass`] reproduces that.
//!
//! The **field** must not use it. The centre of mass depends on the geometry, so
//! `∂O/∂R_A = m_A/M ≠ 0`, and an origin that moves with the nuclei injects a term into every
//! force and every force constant that has nothing to do with the physics. The field is
//! referenced to the coordinate origin, fixed, which is also what MOPAC's `FIELD=` does. For a
//! charged system that makes the energy depend on where the molecule sits — correctly so: a net
//! charge in a uniform field really does have a position-dependent energy.
//!
//! # `d` elements
//!
//! The `p`–`d` hybridization dipole (`ddp[5]`) is **not** included, because MOPAC's dipole does
//! not include it either and matching MOPAC is the requirement. No PM3 element has a `d` shell,
//! so nothing in the shipped parameterization reaches this.

use crate::basis::Basis;
use crate::error::Result;
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::params::Pm3Parameters;
use crate::system::Molecule;

/// MOPAC's dipole origin: the centre of mass, with massless point atoms contributing charge but
/// not mass.
///
/// A system built only from massless point atoms has no centre of mass; the coordinate origin
/// stands in rather than dividing by zero.
pub fn centre_of_mass(molecule: &Molecule, params: &Pm3Parameters) -> Result<Vec3> {
    let mut total = 0.0;
    let mut moment = Vec3::zero();
    for atom in &molecule.atoms {
        let mass = params.element(atom.z)?.mass;
        total += mass;
        moment += atom.position * mass;
    }
    Ok(if total > 0.0 {
        moment / total
    } else {
        Vec3::zero()
    })
}

/// The three Cartesian components of the dipole operator, in the AO basis (atomic units).
///
/// See the module note for the structure and for how to choose `origin`.
pub fn dipole_matrix(
    molecule: &Molecule,
    params: &Pm3Parameters,
    basis: &Basis,
    origin: Vec3,
) -> Result<[Matrix; 3]> {
    let n = basis.nao;
    let mut out = [
        Matrix::zeros(n, n),
        Matrix::zeros(n, n),
        Matrix::zeros(n, n),
    ];
    // `PM3_BORN_NO_DD` drops the intra-atomic `s`–`p` moment, leaving the atom-centred position
    // operator the Berry phase uses. A diagnostic, not a model: it exists so the two formalisms
    // can be asked whether they differ by this term and by nothing else, rather than argued about.
    // Read once rather than per atom per axis — `env::var` allocates and takes a lock.
    let with_dd = std::env::var("PM3_BORN_NO_DD").is_err();

    for (index, atom) in molecule.atoms.iter().enumerate() {
        let element = params.element(atom.z)?;
        let offset = basis.atom_offset[index];
        let norb = basis.atom_norb[index];
        let displacement = (atom.position - origin).to_array();

        for (axis, matrix) in out.iter_mut().enumerate() {
            // The monopole: every orbital on this atom sees the nucleus's own position.
            for mu in 0..norb {
                matrix[(offset + mu, offset + mu)] = displacement[axis];
            }
            // The s–p hybridization dipole, symmetric. `p_x` is AO 1, `p_y` 2, `p_z` 3, so the
            // component along `axis` is AO `axis + 1`.
            if element.has_p() && with_dd {
                let p = offset + axis + 1;
                matrix[(offset, p)] = element.dd;
                matrix[(p, offset)] = element.dd;
            }
        }
    }
    Ok(out)
}

/// `μ` in atomic units (e·Bohr) from a converged total density.
///
/// `Σ_A Z_A^core (R_A − O) − Tr[P M]`, which is the module note's expression evaluated.
pub fn dipole_from_density(
    molecule: &Molecule,
    params: &Pm3Parameters,
    basis: &Basis,
    density: &Matrix,
    origin: Vec3,
) -> Result<Vec3> {
    let operator = dipole_matrix(molecule, params, basis, origin)?;
    let mut nuclear = Vec3::zero();
    for atom in &molecule.atoms {
        nuclear += (atom.position - origin) * params.element(atom.z)?.core_charge;
    }
    let electronic = Vec3::new(
        density.frobenius_dot(&operator[0]),
        density.frobenius_dot(&operator[1]),
        density.frobenius_dot(&operator[2]),
    );
    Ok(nuclear - electronic)
}

/// The one-electron matrix a uniform field `f` adds to `H_core` (eV), and the nuclear term that
/// goes alongside `core_ev`.
///
/// `E_field = −μ·f`, and with `μ = Σ_A Z_A R_A − Tr[P M]` that splits into a density-independent
/// piece `−Σ_A Z_A R_A·f` and a one-electron operator `+M·f` whose expectation the SCF picks up
/// through `H_core`. Because the self-consistent energy counts `H_core` exactly once, adding the
/// operator there is the whole electronic coupling — there is no two-electron field term.
///
/// `f` is in eV per Bohr per elementary charge, so the product with a length in Bohr is directly
/// an energy in eV.
pub fn field_terms(
    molecule: &Molecule,
    params: &Pm3Parameters,
    basis: &Basis,
    field: Vec3,
) -> Result<(Matrix, f64)> {
    // The coordinate origin, deliberately: see the module note.
    let operator = dipole_matrix(molecule, params, basis, Vec3::zero())?;
    let components = field.to_array();
    let mut matrix = Matrix::zeros(basis.nao, basis.nao);
    for (axis, term) in operator.iter().enumerate() {
        let weight = components[axis];
        if weight == 0.0 {
            continue;
        }
        for (slot, value) in matrix.as_mut_slice().iter_mut().zip(term.as_slice()) {
            *slot += weight * value;
        }
    }

    let mut nuclear = 0.0;
    for atom in &molecule.atoms {
        nuclear -= params.element(atom.z)?.core_charge * atom.position.dot(field);
    }
    Ok((matrix, nuclear))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbc::multipole::AtomSites;

    fn water() -> Molecule {
        Molecule::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
            0.0,
        )
        .unwrap()
    }

    /// The operator and the multipole site model must agree, and the agreement is the statement
    /// that a **uniform** field couples to the monopole and the dipole and to nothing else.
    ///
    /// [`AtomSites`] realizes the whole charge distribution — monopole, `s`–`p` dipole *and* the
    /// quadrupole configurations. Put it in a linear potential `−f·r` and every quadrupole
    /// configuration contributes zero, because its charges sum to zero and its first moment
    /// vanishes. That cancellation is what makes the two-element operator above complete rather
    /// than truncated, and it is checked here rather than asserted.
    #[test]
    fn a_uniform_field_sees_only_the_monopole_and_the_dipole() {
        let molecule = water();
        let params = Pm3Parameters::standard().unwrap();
        let basis = Basis::build(&molecule, &params).unwrap();
        let field = Vec3::new(0.031, -0.017, 0.023);
        let (matrix, _) = field_terms(&molecule, &params, &basis, field).unwrap();

        for (index, atom) in molecule.atoms.iter().enumerate() {
            let element = params.element(atom.z).unwrap();
            let sites = AtomSites::build(element).unwrap();
            if sites.n_orb == 0 {
                continue;
            }
            // The potential of the uniform field at each of the atom's charge sites.
            let potential: Vec<f64> = sites
                .offsets
                .iter()
                .map(|offset| -(atom.position + *offset).dot(field))
                .collect();
            let offset = basis.atom_offset[index];
            for mu in 0..sites.n_orb {
                for nu in 0..=mu {
                    let from_sites = sites.fock_contribution(mu, nu, &potential);
                    let from_operator = matrix[(offset + mu, offset + nu)];
                    assert!(
                        (from_sites - from_operator).abs() < 1.0e-12,
                        "atom {index} element ({mu},{nu}): sites give {from_sites}, \
                         the operator gives {from_operator}"
                    );
                }
            }
        }
    }

    /// `μ = −∂E/∂F`, which is the definition of the dipole and therefore the sharpest statement
    /// that the field and the reported dipole are the same object.
    ///
    /// It needs no oracle: a wrong sign, a factor of two, a missing nuclear term or a field that
    /// reached `H_core` but not the energy all break it. What it cannot see is an error shared
    /// by both sides, which is why the operator is also checked against the multipole sites and
    /// against MOPAC.
    ///
    /// Neutral only. `−∂E/∂F` is the dipole about the **field's** origin (the coordinate origin);
    /// the reported one is about the centre of mass. For a neutral system `Σ q_i = 0` makes the
    /// two the same, and for a charged one they differ by `Q·R_com` — see the next test.
    #[test]
    fn the_dipole_is_minus_the_energy_derivative_in_the_field() {
        let molecule = water();
        let params = Pm3Parameters::standard().unwrap();
        let mut options = crate::scf::Pm3Options {
            e_tol: 1.0e-12,
            p_tol: 1.0e-11,
            ..Default::default()
        };
        let reference = crate::scf::run_pm3(&molecule, &params, &options).unwrap();

        let step = 1.0e-4;
        for axis in 0..3 {
            let mut energy_at = |sign: f64| {
                let mut field = [0.0; 3];
                field[axis] = sign * step;
                options.field = Some(Vec3::new(field[0], field[1], field[2]));
                crate::scf::run_pm3(&molecule, &params, &options)
                    .unwrap()
                    .total_ev
            };
            let numeric = -(energy_at(1.0) - energy_at(-1.0)) / (2.0 * step);
            let analytic =
                reference.dipole_debye.to_array()[axis] / crate::constants::AU_DIPOLE_TO_DEBYE;
            assert!(
                (numeric - analytic).abs() < 1.0e-7,
                "axis {axis}: −dE/dF gives {numeric} e·Bohr, the reported dipole {analytic}"
            );
        }
    }

    /// The same identity for an **ion**, where the two origins genuinely disagree.
    ///
    /// `μ_0 = μ_com + Q R_com`. Getting this wrong is invisible on every neutral molecule, and
    /// it is the reason the field is referenced to the coordinate origin rather than borrowing
    /// the dipole's centre of mass — a geometry-dependent origin would put `∂O/∂R_A = m_A/M`
    /// into every force.
    #[test]
    fn the_charged_identity_holds_once_the_origins_are_reconciled() {
        let mut molecule = water();
        molecule.charge = 1.0;
        // Off the origin on purpose: at the origin the correction would vanish and prove nothing.
        for atom in &mut molecule.atoms {
            atom.position += Vec3::new(2.5, -1.5, 3.0);
        }
        let params = Pm3Parameters::standard().unwrap();
        let mut options = crate::scf::Pm3Options {
            charge: 1.0,
            multiplicity: 2,
            e_tol: 1.0e-12,
            p_tol: 1.0e-11,
            ..Default::default()
        };
        let reference = crate::scf::run_pm3(&molecule, &params, &options).unwrap();
        let com = centre_of_mass(&molecule, &params).unwrap();
        let charge: f64 = options.charge;

        let step = 1.0e-4;
        for axis in 0..3 {
            let mut energy_at = |sign: f64| {
                let mut field = [0.0; 3];
                field[axis] = sign * step;
                options.field = Some(Vec3::new(field[0], field[1], field[2]));
                crate::scf::run_pm3(&molecule, &params, &options)
                    .unwrap()
                    .total_ev
            };
            let numeric = -(energy_at(1.0) - energy_at(-1.0)) / (2.0 * step);
            let about_com =
                reference.dipole_debye.to_array()[axis] / crate::constants::AU_DIPOLE_TO_DEBYE;
            let about_origin = about_com + charge * com.to_array()[axis];
            assert!(
                (numeric - about_origin).abs() < 1.0e-6,
                "axis {axis}: −dE/dF gives {numeric}, μ_com + Q·R_com gives {about_origin}"
            );
            // And the correction is not a rounding-scale afterthought.
            assert!(
                (about_origin - about_com).abs() > 0.5,
                "axis {axis}: the origin shift is too small to be testing anything"
            );
        }
    }

    /// The shared operator has to reproduce the dipole the SCF reports, which is the number
    /// pinned against MOPAC for all sixty oracle cases. Anything else is a regression dressed up
    /// as a refactor.
    #[test]
    fn the_operator_reproduces_the_reported_dipole() {
        let molecule = water();
        let params = Pm3Parameters::standard().unwrap();
        let options = crate::scf::Pm3Options::default();
        let result = crate::scf::run_pm3(&molecule, &params, &options).unwrap();
        let basis = Basis::build(&molecule, &params).unwrap();
        let origin = centre_of_mass(&molecule, &params).unwrap();
        let mu = dipole_from_density(&molecule, &params, &basis, &result.density, origin).unwrap();
        let debye = mu * crate::constants::AU_DIPOLE_TO_DEBYE;
        for (got, want) in debye.to_array().iter().zip(result.dipole_debye.to_array()) {
            assert!(
                (got - want).abs() < 1.0e-12,
                "the shared operator gives {debye:?} where the SCF reports {:?}",
                result.dipole_debye
            );
        }
    }
}
