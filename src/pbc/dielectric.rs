// SPDX-License-Identifier: GPL-3.0-or-later

//! Electronic polarizability and the dielectric tensor it gives.
//!
//! ```text
//! α_αβ = ∂(Ω P_α) / ∂F_β          ε∞ = 1 + 4π α / Ω
//! ```
//!
//! Both come from the same coupled-perturbed solve the phonons use, driven by the dipole
//! operator instead of by a nuclear displacement — see `pbc::dfpt::field_response` for
//! why that is legitimate under periodic boundary conditions when a *finite* field is not.
//!
//! # What `ε∞` is and is not
//!
//! It is the **electronic** (clamped-ion, high-frequency) dielectric constant: the response of
//! the electrons at fixed nuclei. It is not the static dielectric constant, which adds the ionic
//! contribution `(4π/Ω) Σ_m Z*·Z*/ω_m²` over the infrared-active modes and is a different
//! calculation. And PM3 was parameterized against molecular heats of formation, geometries,
//! dipoles and ionization potentials — not against solid-state dielectric response — so the
//! number this returns is the correct `ε∞` *of this model*, which is a different claim from
//! agreement with a measurement.
//!
//! # Why `ε∞` is three-dimensional and `α` is not
//!
//! `ε∞ = 1 + 4πα/Ω` needs `Ω` to be a volume. A slab has an area and a chain a length, so the
//! conversion has no meaning there and is refused rather than performed against a number that
//! happens to be finite because a vacuum padding was chosen. The polarizability itself is
//! well defined in every dimensionality and is returned by [`polarizability`] unconverted.
//!
//! The macroscopic longitudinal response at finite `q` behaves differently in each dimensionality
//! — see `dielectric_function` — and only in 3D is it the constant that `ε∞` names.

// A note on `clippy::needless_range_loop`, allowed below.
//
// The loop variables here are Cartesian directions (`alpha`, `beta`, `axis`), atom indices, or
// the rows and columns of a matrix being eliminated. The index *is* the meaning: `for alpha in
// 0..3` says which direction, where `for (alpha, row) in out.iter_mut().enumerate()` says it
// less clearly and no more safely. Several of these loops also index two different tensors by
// the same direction, which no single iterator expresses.
#![allow(clippy::needless_range_loop)]

use crate::constants::HARTREE_TO_EV;
use crate::error::{Pm3Error, Result};
use crate::params::Pm3Parameters;
use crate::pbc::gamma::PeriodicOptions;
use crate::scf::Pm3Options;
use crate::system::Molecule;

/// The polarizability and, in 3D, the dielectric tensor built from it.
#[derive(Clone, Copy, Debug)]
pub struct DielectricTensors {
    /// `α_αβ` in Bohr³ — the cell's electronic polarizability.
    pub polarizability: [[f64; 3]; 3],
    /// `ε∞ = 1 + 4πα/Ω`, dimensionless. Only meaningful for a fully periodic cell.
    pub epsilon: [[f64; 3]; 3],
}

/// `α_αβ = ∂(Ω P_α)/∂F_β` in Bohr³, in every dimensionality.
///
/// The conversion to a dielectric constant is left to the caller, because it is the step that
/// needs a volume — see the module note.
pub fn polarizability(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
) -> Result<[[f64; 3]; 3]> {
    Ok(polarizability_and_dielectric(molecule, params, options, periodic)?.polarizability)
}

/// The electronic dielectric tensor `ε∞ = 1 + 4πα/Ω`, for a fully periodic cell.
///
/// A chain or a slab is refused: `Ω` would be a length or an area, and dividing by the vacuum
/// padding of a supercell would make the answer a statement about the padding. Use
/// [`polarizability`] there, which returns the same `α` and leaves the conversion alone.
pub fn dielectric_tensor(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
) -> Result<DielectricTensors> {
    let cell = molecule
        .cell
        .ok_or_else(|| Pm3Error::InvalidInput("a dielectric tensor needs a cell".to_string()))?;
    if cell.n_periodic() != 3 {
        return Err(Pm3Error::InvalidInput(
            "the electronic dielectric tensor is three-dimensional: ε∞ = 1 + 4πα/Ω needs Ω to \
             be a volume, and a chain or a slab has only a length or an area. Use \
             `pbc::dielectric::polarizability`, which returns the same α and leaves the \
             conversion — a claim about where the material stops — to you."
                .to_string(),
        ));
    }
    polarizability_and_dielectric(molecule, params, options, periodic)
}

fn polarizability_and_dielectric(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
) -> Result<DielectricTensors> {
    let cell = molecule.cell.ok_or_else(|| {
        Pm3Error::InvalidInput("a polarizability needs a periodic cell".to_string())
    })?;
    let response = crate::pbc::dfpt::field_response(molecule, params, options, periodic)?;
    let basis = &response.basis;

    // The same diagnostic switch `pbc::born` carries, and for the same reason: it drops the
    // intra-atomic `s`–`p` moment, which is exactly what the Berry-phase route omits by placing
    // every orbital at its atom's centre. `tests/pbc_finite_field.rs` uses it to establish that
    // the two routes differ by that term and by nothing else. Read once, not three loops deep.
    let with_dd = std::env::var("PM3_BORN_NO_DD").is_err();

    let mut alpha = [[0.0_f64; 3]; 3];
    for beta in 0..3 {
        let delta = &response.delta[beta];
        for alpha_index in 0..3 {
            let mut total = 0.0;
            for (b, atom) in molecule.atoms.iter().enumerate() {
                let offset = basis.atom_offset[b];
                let norb = basis.atom_norb[b];
                if norb == 0 {
                    continue;
                }
                // Charge transfer: `∂Q_b = −∂p_b`, the same sign as the Born charges.
                let population: f64 = (0..norb)
                    .map(|mu| delta[(offset + mu, offset + mu)].re)
                    .sum();
                total += -population * atom.position.to_array()[alpha_index];

                // The on-site `sp` hybridization moment, carried as `−Tr[P M]`.
                let element = params.element(atom.z)?;
                if element.has_p() && with_dd {
                    let p = offset + alpha_index + 1;
                    total += -2.0 * element.dd * delta[(offset, p)].re;
                }
            }
            alpha[alpha_index][beta] = total;
        }
    }

    // Hartree per eV, and it is load-bearing.
    //
    // The response was solved in this crate's internal units, where energies are eV and lengths
    // Bohr, so `∂P/∂F` comes out per (eV/Bohr) of field. A polarizability in Bohr³ is per
    // (Hartree/Bohr). Omitting this leaves `α` — and therefore `ε∞ − 1` — smaller by 27.21 while
    // remaining symmetric, positive-definite and origin-independent, so every structural check
    // still passes. The check that catches it is the one comparing a molecule in a large box
    // against its own finite-field polarizability, which is why that test exists.
    for row in &mut alpha {
        for value in row.iter_mut() {
            *value *= HARTREE_TO_EV;
        }
    }

    let volume = cell.measure();
    let mut epsilon = [[0.0_f64; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let identity = if a == b { 1.0 } else { 0.0 };
            epsilon[a][b] = identity + 4.0 * std::f64::consts::PI * alpha[a][b] / volume;
        }
    }
    Ok(DielectricTensors {
        polarizability: alpha,
        epsilon,
    })
}

/// The **static** dielectric tensor, `ε₀ = ε∞ + ionic`.
///
/// ```text
/// ε₀_αβ = ε∞_αβ + (4π/Ω) Σ_m  Z̄_{m,α} Z̄_{m,β} / ω_m²,
///     Z̄_{m,α} = Σ_{a,γ} Z*_{a,αγ} e_{m,aγ} / √M_a
/// ```
///
/// The electrons alone give `ε∞`; letting the nuclei move as well adds one term per infrared-
/// active mode, weighted by how strongly that mode carries a dipole (`Z̄`, its **mode effective
/// charge**) and by how soft it is (`1/ω²`). A soft polar mode dominates, which is why
/// ferroelectrics have enormous static constants and why this quantity is far more sensitive to
/// the model than `ε∞` is.
///
/// # What it requires, and what it refuses
///
/// Three-dimensional cells only, for the same reason [`dielectric_tensor`] is. Modes at or below
/// [`SOFT_MODE_FLOOR`] in `ω²` are **skipped, and counted**, and the count comes back in
/// [`StaticDielectric::skipped_modes`] rather than being swallowed.
///
/// The floor is not zero, and it cannot be. The sum weights each mode by `1/ω²`, so a mode that
/// lands a hair *above* zero contributes an enormous term rather than a negligible one — the
/// three acoustic modes come out at `±0.01 cm⁻¹` here, and whether each is `+0.005` or `−0.007`
/// is arithmetic noise. Testing `ω² ≤ 0` would therefore let an acoustic mode through whenever it
/// landed on the positive side and blow the tensor up. The floor is what makes the acoustic
/// branch reliably excluded instead of excluded by luck.
///
/// Three skipped modes is the acoustic branch. **A fourth is a real imaginary mode**, and the
/// first thing to check is not the geometry but the Γ-point margin: `examples/soft_mode_floor.rs`
/// finds one at `−1.14e-1` for water in a box a hair *under* 14 Bohr and none at all a hair over,
/// because [`crate::pbc::gamma::DEFAULT_SHORT_RANGE_CUTOFF`] is exactly 14 Bohr and below it the
/// SCF converges cleanly to a well-defined wrong answer — 35 eV away from the one just across the
/// boundary. A response built on that ground state inherits the error with no other symptom. Check
/// the margin, then relax the structure.
///
/// # What it cost to add
///
/// Nothing new. The ingredients are the Born charges and the Γ-point dynamical matrix, both of
/// which this crate already had, so this is a contraction of existing quantities rather than
/// another response to solve. Semiempirical codes commonly stop at `ε∞` for want of the first of
/// those, not for want of this formula.
/// **Deprecated and no longer used.** Kept so a caller that read it still compiles.
///
/// This was the `ω²` below which a mode was left out of the ionic sum, in the units the
/// mass-weighted Γ matrix is diagonalized in, `eV/(Å²·amu)`. It was placed in a measured gap
/// rather than guessed — `examples/soft_mode_floor.rs` shows the acoustic branch landing between
/// `1e-17` and `5e-15` across five cell widths, and the softest genuine mode at `+2.5e-2` — and
/// that gap is thirteen orders of magnitude wide, so the classification was never close to the
/// line.
///
/// It is still gone, because a measured gap is a property of the systems it was measured on. The
/// acoustic branch is now **projected out** of the mass-weighted matrix before diagonalization
/// (see [`crate::rigid`]), which puts it at exactly zero, so the sum tests `ω² ≤ 0` and no
/// magnitude enters. A soft mode in a cell near a phase transition is then kept because it is a
/// real mode, not dropped because it was smaller than a constant chosen elsewhere.
#[deprecated(
    since = "0.2.4",
    note = "the acoustic branch is projected out, so the ionic sum needs no floor; \
            StaticDielectric::skipped_modes now counts exact zeros and imaginary modes"
)]
pub const SOFT_MODE_FLOOR: f64 = 1.0e-6;

pub fn static_dielectric_tensor(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
) -> Result<StaticDielectric> {
    let cell = molecule.cell.ok_or_else(|| {
        Pm3Error::InvalidInput("a static dielectric tensor needs a cell".to_string())
    })?;
    if cell.n_periodic() != 3 {
        return Err(Pm3Error::InvalidInput(
            "the static dielectric tensor is three-dimensional: both halves of it, ε∞ and the \
             ionic term, carry a `4π/Ω` that needs Ω to be a volume."
                .to_string(),
        ));
    }
    let electronic = dielectric_tensor(molecule, params, options, periodic)?;
    let born = crate::pbc::born::born_charges(molecule, params, options, periodic)?;

    // The Γ-point dynamical matrix, mass weighted, and its eigenvectors. The frequencies come
    // from the same decomposition so the modes and the `1/ω²` cannot come from different matrices.
    let dynamical =
        crate::pbc::dfpt::dynamical_matrix(molecule, params, options, periodic, [0.0; 3])?;
    let ndof = dynamical.matrix.rows;
    let masses = &dynamical.masses;
    let per_angstrom_squared =
        crate::constants::ANGSTROM_TO_BOHR * crate::constants::ANGSTROM_TO_BOHR;
    let mut weighted = crate::cmatrix::CMatrix::zeros(ndof, ndof);
    for i in 0..ndof {
        for j in 0..ndof {
            let scale = (masses[i / 3] * masses[j / 3]).sqrt();
            if scale > 0.0 {
                weighted[(i, j)] = dynamical.matrix[(i, j)] * (per_angstrom_squared / scale);
            }
        }
    }
    // Project the acoustic branch out before diagonalizing, rather than recognising it afterwards
    // by being small. The three translations are exact null vectors of `D(0)`, so removing them
    // leaves the sum below with nothing to classify: what used to be "`ω²` below `1e-6`, which is
    // where a measured gap put the line" is now "`ω² ≤ 0`", and the only modes that meets are the
    // three that were projected (exactly zero) and any genuinely imaginary one.
    let positions: Vec<crate::math::Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
    let acoustic = crate::rigid::rigid_body_basis(
        &positions,
        masses,
        crate::rigid::RigidMotions::TranslationsOnly,
    );
    crate::rigid::project_out_hermitian(&acoustic, &mut weighted);
    let (mut eigenvalues, vectors) = crate::cmatrix::hermitian_eigen(&weighted)?;
    for index in crate::rigid::rigid_mode_indices_complex(&acoustic, &vectors) {
        eigenvalues[index] = 0.0;
    }

    // `4π/Ω · e²Å²/eV` to dimensionless. `e²/Hartree = Bohr`, so the chain is
    // `e²Å²/eV → (eV per Hartree) → e²Å²/Hartree = Bohr·Å² → Bohr³` via `Å² = a0² Bohr²`.
    let conversion = HARTREE_TO_EV * per_angstrom_squared;
    let prefactor = 4.0 * std::f64::consts::PI / cell.measure() * conversion;

    let mut ionic = [[0.0_f64; 3]; 3];
    let mut skipped_modes = 0usize;
    for mode in 0..ndof {
        let omega2 = eigenvalues[mode];
        // No floor. The acoustic branch is exactly zero because it was projected out above, and
        // what is left below zero is a genuinely imaginary mode, which the sum cannot use either.
        if omega2 <= 0.0 {
            skipped_modes += 1;
            continue;
        }
        // The mode effective charge `Z̄_{m,α}`. The eigenvector is complex in general; at Γ with a
        // real symmetric matrix it is real up to a global phase, and only its modulus enters a
        // product of two of them, so the real part is taken after fixing that phase by using the
        // same component for both factors.
        let mut bar = [0.0_f64; 3];
        for alpha in 0..3 {
            let mut total = 0.0;
            for a in 0..ndof / 3 {
                let root_mass = masses[a].sqrt();
                if root_mass <= 0.0 {
                    continue;
                }
                for gamma in 0..3 {
                    total += born[a][alpha][gamma] * vectors[(3 * a + gamma, mode)].re / root_mass;
                }
            }
            bar[alpha] = total;
        }
        for alpha in 0..3 {
            for beta in 0..3 {
                ionic[alpha][beta] += prefactor * bar[alpha] * bar[beta] / omega2;
            }
        }
    }

    let mut epsilon = electronic.epsilon;
    for alpha in 0..3 {
        for beta in 0..3 {
            epsilon[alpha][beta] += ionic[alpha][beta];
        }
    }
    Ok(StaticDielectric {
        epsilon,
        electronic: electronic.epsilon,
        ionic,
        skipped_modes,
    })
}

/// The static dielectric tensor and the two halves it is made of.
#[derive(Clone, Copy, Debug)]
pub struct StaticDielectric {
    /// `ε₀ = ε∞ + ionic`, dimensionless.
    pub epsilon: [[f64; 3]; 3],
    /// The electronic (clamped-ion) half — the same `ε∞` [`dielectric_tensor`] returns.
    pub electronic: [[f64; 3]; 3],
    /// The lattice half, `(4π/Ω) Σ_m Z̄ Z̄ / ω_m²`. Positive semi-definite for a stable structure.
    pub ionic: [[f64; 3]; 3],
    /// How many modes were left out because `ω² ≤ 0`.
    ///
    /// Three of them are the acoustic branch and are expected. **More than three means the
    /// geometry is not a minimum**, and the ionic term is then missing whatever those modes would
    /// have contributed — which for a soft mode is most of it. Reported rather than hidden,
    /// because an unrelaxed structure gives a number that looks like an answer.
    pub skipped_modes: usize,
}

/// How much the polarizability moves when the cell origin does — the size of the approximation.
///
/// The position operator this is built on is not a well-defined periodic operator, and the
/// argument that the *response* is nevertheless well defined (an origin shift adds a constant to
/// the diagonal, which the occupied–virtual projection kills) is an argument. This measures it
/// instead: it recomputes with every atom displaced by `offset` and returns the largest change in
/// any component of `α`. A number near machine precision is what says the argument holds for this
/// system; a large one says it does not.
pub fn dielectric_origin_sensitivity(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
    offset: crate::math::Vec3,
) -> Result<f64> {
    let here = polarizability(molecule, params, options, periodic)?;
    let mut shifted = molecule.clone();
    for atom in &mut shifted.atoms {
        atom.position += offset;
    }
    let there = polarizability(&shifted, params, options, periodic)?;
    let mut worst = 0.0_f64;
    for a in 0..3 {
        for b in 0..3 {
            worst = worst.max((here[a][b] - there[a][b]).abs());
        }
    }
    Ok(worst)
}
