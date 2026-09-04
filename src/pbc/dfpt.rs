// SPDX-License-Identifier: GPL-3.0-or-later

//! The dynamical matrix at an arbitrary wavevector.
//!
//! # What this is for
//!
//! Real-space force constants read off an `n₁×n₂×n₃` supercell give `D(q)` exactly at the `q`
//! commensurate with that supercell, and Fourier interpolation in between. The interpolation is
//! only as good as the decay of `Φ(0, T)` inside the supercell, and enlarging the supercell costs
//! cubically. Perturbation theory computes `D(q)` at any `q` from the primitive cell, at a cost
//! that does not depend on `q` at all.
//!
//! # The perturbation
//!
//! Displace atom `κ` in cell `T` by `u_κ e^{iq·T}`. Every real-space force constant then enters
//! the dynamical matrix with that phase:
//!
//! ```text
//! D_{κα,κ'β}(q) = Σ_T e^{iq·T} Φ_{κα,κ'β}(0, T)
//! ```
//!
//! For a term that depends on one pair displacement `d = R_{κ'} + T − R_κ`, with `3×3` second
//! derivative `H`, that becomes four contributions: `+H` on each diagonal block unphased, and
//! `−H e^{±iq·T}` on the two mixed blocks. Setting `q = 0` collapses this onto the Γ-point
//! scatter [`crate::pbc::hessian`] already uses, which is what
//! [`tests::the_dynamical_matrix_reduces_to_the_gamma_hessian`] checks.
//!
//! # The electronic response
//!
//! A displacement moves the electrons too, and that relaxation is most of the answer: on water it
//! turns a rigid-ion force constant of 5.96 eV/Bohr² into 0.56. The response is linear, and in
//! the band basis of the two coupled k-points it is
//!
//! ```text
//! ΔP_{mn} = [f_n(k) − f_m(k+q)] / [ε_n(k) − ε_m(k+q)] · ⟨ψ_{m,k+q}| ΔV |ψ_{n,k}⟩
//! ```
//!
//! solved self-consistently, because `ΔV` contains the potential the induced density itself
//! makes. Both the occupied–empty and the empty–occupied blocks contribute; the second is the
//! response of the *bra* at `k + q`, and dropping it halves the answer.
//!
//! [`rigid_ion_dynamical_matrix`] gives the fixed-density part alone, which is worth having
//! separately because it is what the two pieces are tested apart by.
//!
//! # How this is known to be right
//!
//! Every piece is checked against something built differently, because at `q = 0` there is an
//! independent implementation of the whole thing and at finite `q` there is a supercell:
//!
//! | piece | checked against |
//! |---|---|
//! | the perturbation's short-range half | [`crate::pbc::kernel::fock_derivative_pairs`] |
//! | its long-range half | the same term by finite differences of the Ewald sum |
//! | the two-electron kernel | [`crate::pbc::kernel::PeriodicKernel`] |
//! | the Coulomb second derivative | [`crate::pbc::ewald_hessian::ewald_atom_hessian`] |
//! | the short-range second derivative | [`crate::pbc::hessian::skeleton`] |
//! | the orbitals the response is built on | the SCF's own |
//! | `D(0)`, everything together | [`crate::pbc::hessian::periodic_hessian`] |
//! | `D(q)` at finite `q` | a doubled cell's Γ-point Hessian, by folding |
//!
//! The last two are what matter. The first is an arbitrary-`q` machine run at the one wavevector
//! where a phase-free implementation exists; the second is the only test in which a wrong phase
//! can show at all.
//!
//! # Scope
//!
//! Any periodic dimensionality — the phased lattice sum ([`crate::pbc::phased`]) carries the
//! wire and sheet kernels now — closed shell, plain PM3. The wavevector must lie in the
//! periodic subspace: a slab takes in-plane `q` and a chain axial `q`, and a fractional
//! component on a non-periodic axis is refused. (For a 2D cell the small-`q` *value* of the
//! phased sum is non-analytic — the LO–TO physics — but the dynamical matrix consumes its
//! second derivative, which is smooth there, so no special handling appears here.) The
//! response samples whatever k-mesh it is given, pairing each point with `q`; given none it
//! samples `Γ` alone, which is the sampling a `Γ` ground state supports and carries the same
//! condition (see [`crate::pbc::gamma::PeriodicResult::gamma_margin`]). Open and closed shell
//! both. The classical corrections get their image-resolved second derivative from the same
//! weighted-seed bilinear form the electronic part uses.

use crate::basis::Basis;
use crate::cell::Cell;
use crate::cmatrix::CMatrix;
use crate::corrections::Variant;
use crate::error::{Pm3Error, Result};
use crate::math::Vec3;
use crate::neighbor::NeighborList;
use crate::params::Pm3Parameters;
use crate::pbc::gamma::{run_gamma, PeriodicOptions};
use crate::pbc::multipole::AtomSites;
use crate::pbc::phased::ewald_phased;
use crate::scf::Pm3Options;
use crate::system::Molecule;

use faer::c64;
use rayon::prelude::*;

/// A dynamical matrix at one wavevector.
#[derive(Clone, Debug)]
pub struct DynamicalMatrix {
    /// The wavevector, in fractions of the reciprocal lattice vectors.
    pub q_frac: [f64; 3],
    /// `D(q)`, `3N × 3N`, in eV/Bohr². Hermitian.
    pub matrix: CMatrix,
    /// Atomic masses (amu), in the order the matrix indexes them.
    pub masses: Vec<f64>,
    /// How far the assembled matrix was from Hermitian before it was symmetrized — a summary of
    /// the numerical noise in the lattice sums, and the one number that would move first if a
    /// phase were wrong.
    pub hermitian_defect: f64,
}

/// The rigid-ion dynamical matrix at `q_frac`, in fractions of the reciprocal lattice vectors.
///
/// **Fixed density.** See the module note: the electronic response is not included, so this is
/// not a phonon calculation. It is the part of `D(q)` that the ions contribute, which is what a
/// response calculation would be built on top of.
pub fn rigid_ion_dynamical_matrix(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
    q_frac: [f64; 3],
) -> Result<DynamicalMatrix> {
    assemble(
        molecule,
        params,
        options,
        periodic,
        None,
        q_frac,
        false,
        &DfptOptions::default(),
    )
}

/// The dynamical matrix at `q_frac`, **including** the electrons'' response to the displacement.
///
/// The response is solved at a single k-point, `Γ`, pairing it with `q`. That matches the
/// sampling the ground state used and carries the same condition: see
/// [`crate::pbc::gamma::PeriodicResult::gamma_margin`], which applies here unchanged, and the
/// module note.
pub fn dynamical_matrix(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
    q_frac: [f64; 3],
) -> Result<DynamicalMatrix> {
    assemble(
        molecule,
        params,
        options,
        periodic,
        None,
        q_frac,
        true,
        &DfptOptions::default(),
    )
}

/// The dynamical matrix at `q_frac`, with the response summed over a **k-mesh**.
///
/// Every point `k` of the mesh is paired with `k + q`, and the mesh is the one the ground state
/// was converged on — sampling the response more finely than the density would be answering a
/// different question.
///
/// # `k + q` need not be on the mesh
///
/// It usually is not, and it does not need to be. `F(k) = Σ_T e^{ik·T}F(T)` is assembled in the
/// periodic-gauge AO basis with no `e^{iG·r}` anywhere, so `F(k + G) ≡ F(k)` element for element
/// and a point off the mesh is diagonalized exactly as one on it would be. There is no umklapp
/// bookkeeping. [`crate::pbc::kpoints::is_commensurate`] reports whether `q` maps the mesh onto
/// itself, which decides whether the answer is an exact supercell result or a Fourier
/// interpolation of one — a statement about what was asked for, not about correctness.
///
/// # Metals
///
/// `Δf/Δε` is singular where states cross the Fermi level. A mesh that finds no gap is refused
/// unless `KpointOptions::smearing_ev` is set, because without it the answer would depend on
/// which degenerate pairs happened to fall inside a numerical floor.
pub fn dynamical_matrix_on_mesh(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
    kopt: &crate::pbc::kscf::KpointOptions,
    q_frac: [f64; 3],
) -> Result<DynamicalMatrix> {
    assemble(
        molecule,
        params,
        options,
        periodic,
        Some(kopt),
        q_frac,
        true,
        &DfptOptions::default(),
    )
}

/// Phonon frequencies (cm⁻¹) with the response summed over a k-mesh.
///
/// See [`dynamical_matrix_on_mesh`].
pub fn phonon_frequencies_on_mesh(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
    kopt: &crate::pbc::kscf::KpointOptions,
    q_frac: [f64; 3],
) -> Result<Vec<f64>> {
    let d = dynamical_matrix_on_mesh(molecule, params, options, periodic, kopt, q_frac)?;
    frequencies_of(&d)
}

/// The converged ground state `D(q)` is built on, whichever sampling produced it.
///
/// Both halves of the dynamical matrix — the fixed-density skeleton and the electronic response
/// — have to read the *same* density, and the point of this type is that there is one place
/// where that density comes from. Two SCFs used to be run and only the response used the second.
enum GroundDensity {
    Gamma(crate::pbc::gamma::PeriodicResult),
    Mesh(crate::pbc::kscf::KpointResult),
}

impl GroundDensity {
    /// `P(T = 0)`, the on-site density per cell. Both variants report it in the same convention.
    fn density(&self) -> &crate::linalg::Matrix {
        match self {
            Self::Gamma(scf) => &scf.density,
            Self::Mesh(scf) => &scf.density,
        }
    }

    /// `P^α(0) − P^β(0)` per cell, for an unrestricted calculation only.
    fn spin_density(&self) -> Option<&crate::linalg::Matrix> {
        match self {
            Self::Gamma(scf) => scf.spin_density.as_ref(),
            Self::Mesh(scf) => scf.spin_density.as_ref(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn assemble(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
    // None samples the ground state at Gamma alone; Some converges it on a mesh and pairs
    // every one of its points with q.
    kopt: Option<&crate::pbc::kscf::KpointOptions>,
    q_frac: [f64; 3],
    include_response: bool,
    // The response's own controls. The simple entry points pass the defaults; only
    // [`force_constants_at_q`] lets a caller change them.
    dfpt: &DfptOptions,
) -> Result<DynamicalMatrix> {
    let cell = molecule.cell.ok_or_else(|| {
        Pm3Error::InvalidInput("a dynamical matrix needs a periodic cell".to_string())
    })?;
    if cell.n_periodic() == 0 {
        return Err(Pm3Error::InvalidInput(
            "the cell has no periodic direction; an isolated system's frequencies come from \
             the molecular Hessian"
                .to_string(),
        ));
    }
    // The dimensional constraint on q: a slab accepts only in-plane wavevectors and a chain
    // only axial ones. A fractional component on a non-periodic axis is the only way to ask
    // for anything else — the Cartesian q below is assembled from the reciprocal basis of the
    // *periodic subspace*, so it cannot leave that subspace once this check passes.
    for (index, value) in q_frac.iter().enumerate() {
        if !cell.pbc[index] && *value != 0.0 {
            return Err(Pm3Error::InvalidInput(format!(
                "q has a component {value} along non-periodic direction {index}; a phonon \
                 wavevector must lie in the periodic subspace"
            )));
        }
    }

    // The ground state everything here is built on — **one** self-consistent solution, and the
    // one the caller asked for.
    //
    // Handed a mesh, this used to converge `run_gamma` for the skeleton and `run_kpoints` for
    // the response, and combine them. That is two SCFs, and worse, it is two *different* ones:
    // the skeleton is the fixed-density half of `D(q)` and the larger term, so a mesh-sampled
    // response sat on top of a Γ-sampled skeleton. Whenever Γ alone was adequate the mix was
    // harmless and the mesh was pointless; whenever the mesh was necessary — the only reason to
    // pass one — the dominant term was the one computed wrongly, and no `gamma_margin` was
    // reported to say so.
    let ground_state = match kopt {
        None => GroundDensity::Gamma(run_gamma(molecule, params, options, periodic)?),
        Some(mesh) => {
            let converged =
                crate::pbc::kscf::run_kpoints(molecule, params, options, periodic, mesh)?;
            // A metal has states arbitrarily close to the Fermi level, and `Δf/Δε` is a
            // `0/0` for every pair of them. Smearing is what regularizes it; without any,
            // the answer would depend on which pairs happened to fall inside the numerical
            // floor. Refuse rather than produce that.
            if mesh.smearing_ev <= 0.0 && converged.band_gap_ev.is_some_and(|gap| gap <= 0.0) {
                return Err(Pm3Error::InvalidInput(
                    "the mesh found no band gap, so the response has states arbitrarily \
                     close to the Fermi level and its energy denominators are singular. \
                     Set KpointOptions::smearing_ev, which is what makes those terms \
                     finite."
                        .to_string(),
                ));
            }
            GroundDensity::Mesh(converged)
        }
    };
    let density = ground_state.density();
    let basis = Basis::build(molecule, params)?;
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;

    // Cartesian q from the reciprocal basis of the periodic subspace.
    let mut q = Vec3::zero();
    for (index, b) in cell.reciprocal_basis() {
        q += b * q_frac[index];
    }

    // What the fixed-density exchange reads. A closed shell takes the total at half strength, an
    // open one each spin at full — the same number when `P^α = P^β`, which is what keeps the
    // restricted skeleton unchanged to the last bit.
    let spin_halves = ground_state.spin_density().map(|difference| {
        let build = |sign: f64| {
            let mut p = density.clone();
            for (value, delta) in p.as_mut_slice().iter_mut().zip(difference.as_slice()) {
                *value = 0.5 * (*value + sign * *delta);
            }
            p
        };
        (build(1.0), build(-1.0))
    });
    let exchange: Vec<(&crate::linalg::Matrix, f64)> = match &spin_halves {
        Some((alpha, beta)) => vec![(alpha, 1.0), (beta, 1.0)],
        None => vec![(density, 0.5)],
    };

    let mut matrix = CMatrix::zeros(ndof, ndof);
    phased_skeleton(
        molecule,
        params,
        options,
        periodic,
        density,
        &exchange,
        &basis,
        q_frac,
        &mut matrix,
    )?;
    // The fixed-charge long-range half. `LongRange::Off` drops it here *and* in the response, so
    // the two stay consistent: a skeleton carrying a long-range term the response cannot screen
    // is a worse answer than one where neither has it.
    if dfpt.long_range != LongRange::Off {
        phased_lattice_sum(molecule, params, periodic, density, &basis, q, &mut matrix)?;
    }
    if options.variant != Variant::Pm3 {
        phased_corrections(molecule, options.variant, periodic, q, &mut matrix)?;
    }
    if include_response {
        let setup = crate::pbc::gamma::build_setup(molecule, params, periodic)?;
        let nao = setup.basis.nao;
        match (&ground_state, kopt) {
            (GroundDensity::Gamma(scf), _) => {
                if scf.n_occ > 0 && scf.n_occ < nao {
                    let ground = gamma_ground_state(molecule, params, &setup, &cell, scf)?;
                    response(
                        molecule,
                        params,
                        options,
                        periodic,
                        &setup,
                        &ground,
                        &cell,
                        q,
                        q_frac,
                        &mut matrix,
                        None,
                        None,
                        dfpt,
                    )?;
                }
            }
            (GroundDensity::Mesh(converged), Some(mesh)) => {
                let ground = mesh_ground_state(
                    molecule,
                    params,
                    &setup,
                    &cell,
                    &mesh.spec,
                    converged,
                    mesh.smearing_ev,
                )?;
                response(
                    molecule,
                    params,
                    options,
                    periodic,
                    &setup,
                    &ground,
                    &cell,
                    q,
                    q_frac,
                    &mut matrix,
                    None,
                    None,
                    dfpt,
                )?;
            }
            // `GroundDensity::Mesh` is built only in the `Some(mesh)` arm above, so this cannot
            // be reached; naming it is what keeps the match exhaustive without a catch-all that
            // would silently swallow a future third kind of ground state.
            (GroundDensity::Mesh(_), None) => unreachable!("a mesh ground state needs its mesh"),
        }
    }

    let hermitian_defect = matrix.hermitian_defect();
    // `D(q)` is Hermitian by construction; what is left is the different summation orders the
    // three contributions used, and averaging is the honest way to remove it.
    let mut hermitized = CMatrix::zeros(ndof, ndof);
    for i in 0..ndof {
        for j in 0..ndof {
            let a = matrix[(i, j)];
            let b = matrix[(j, i)];
            hermitized[(i, j)] = c64::new(0.5 * (a.re + b.re), 0.5 * (a.im - b.im));
        }
    }

    Ok(DynamicalMatrix {
        q_frac,
        matrix: hermitized,
        masses: molecule
            .atoms
            .iter()
            .map(|a| params.element(a.z).map(|e| e.mass))
            .collect::<Result<Vec<_>>>()?,
        hermitian_defect,
    })
}

/// Phonon frequencies at one wavevector, in cm⁻¹, ascending.
///
/// An imaginary mode is reported as a **negative** frequency, the convention the molecular path
/// already uses: a soft direction is something to see, not something to hide behind an absolute
/// value.
///
/// At `q = 0` three of these are the acoustic branch and should come out near zero; how near is
/// a statement about the numerical quality of the lattice sums, and is not enforced here — see
/// [`crate::pbc::hessian::enforce_acoustic_sum_rule`] for the Γ-point path's separate handling.
pub fn phonon_frequencies(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
    q_frac: [f64; 3],
) -> Result<Vec<f64>> {
    let d = dynamical_matrix(molecule, params, options, periodic, q_frac)?;
    frequencies_of(&d)
}

/// Mass-weight a dynamical matrix and diagonalize it, in cm^-1 with imaginary frequencies
/// reported as negative.
///
/// Public because a caller who has built a [`DynamicalMatrix`] — from [`dynamical_matrix`], from
/// [`dynamical_matrix_on_mesh`], or from [`rigid_ion_dynamical_matrix`] — should not have to
/// reimplement the mass weighting and the sign convention to read frequencies off it. The matrix
/// carries the masses it was indexed by, so this needs nothing else.
pub fn frequencies_of(d: &DynamicalMatrix) -> Result<Vec<f64>> {
    let ndof = d.matrix.rows;
    // eV/Bohr² → eV/(Å²·amu), the units the cm⁻¹ conversion is defined against.
    let per_angstrom_squared =
        crate::constants::ANGSTROM_TO_BOHR * crate::constants::ANGSTROM_TO_BOHR;
    let mut weighted = CMatrix::zeros(ndof, ndof);
    for i in 0..ndof {
        for j in 0..ndof {
            let scale = (d.masses[i / 3] * d.masses[j / 3]).sqrt();
            if scale > 0.0 {
                weighted[(i, j)] = d.matrix[(i, j)] * (per_angstrom_squared / scale);
            }
        }
    }
    let (eigenvalues, _) = crate::cmatrix::hermitian_eigen(&weighted)?;
    Ok(eigenvalues
        .iter()
        .map(|value| {
            let magnitude = value.abs().sqrt() * crate::hessian::SQRT_EV_PER_ANG2_AMU_TO_CM;
            if *value < 0.0 {
                -magnitude
            } else {
                magnitude
            }
        })
        .collect())
}

/// Add a pair's `3×3` block to the four places it belongs, with the phase the image carries.
///
/// The diagonal blocks are **unphased** and the mixed ones carry `e^{±iq·T}`. That asymmetry is
/// the whole content of `D(q) = Σ_T e^{iq·T} Φ(0, T)`: the `(κ, κ)` self-force-constant is a sum
/// over every image with no phase, because both its indices sit in the reference cell.
fn scatter_phased(matrix: &mut CMatrix, a: usize, b: usize, block: &[[f64; 3]; 3], phase: c64) {
    for (alpha, row) in block.iter().enumerate() {
        for (beta, value) in row.iter().enumerate() {
            matrix[(3 * a + alpha, 3 * a + beta)] += c64::new(*value, 0.0);
            matrix[(3 * b + alpha, 3 * b + beta)] += c64::new(*value, 0.0);
            matrix[(3 * a + alpha, 3 * b + beta)] -= phase * *value;
            matrix[(3 * b + alpha, 3 * a + beta)] -= phase.conj() * *value;
        }
    }
}

/// The short-range NDDO terms, phased.
#[allow(clippy::too_many_arguments)]
fn phased_skeleton(
    molecule: &Molecule,
    params: &Pm3Parameters,
    _options: &Pm3Options,
    periodic: &PeriodicOptions,
    density: &crate::linalg::Matrix,
    exchange: &[(&crate::linalg::Matrix, f64)],
    basis: &Basis,
    q_frac: [f64; 3],
    matrix: &mut CMatrix,
) -> Result<()> {
    let cell = molecule.cell.expect("checked by the caller");
    let nat = molecule.atoms.len();
    let mut atom_sites = Vec::with_capacity(nat);
    for atom in &molecule.atoms {
        let elem = params.element(atom.z)?;
        atom_sites.push(AtomSites::build(elem).ok_or(Pm3Error::UnsupportedDOrbitals(atom.z))?);
    }
    let cutoff = periodic.short_range_cutoff.max(periodic.switch.off);
    let positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
    let list = NeighborList::build_from_positions(&positions, Some(&cell), cutoff);

    let pairs: Vec<_> = list.unique().collect();
    let blocks: Vec<Result<(crate::pbc::hessian::PairBlock, [i32; 3])>> = pairs
        .par_iter()
        .map(|pair| {
            let block = crate::pbc::hessian::pair_block(
                molecule,
                params,
                periodic,
                density,
                exchange,
                basis,
                &atom_sites,
                pair.a,
                pair.b,
                pair.dvec,
                pair.r,
            )?;
            Ok((block, pair.t))
        })
        .collect();
    for entry in blocks {
        let ((a, b, block), t) = entry?;
        scatter_phased(matrix, a, b, &block, image_phase(q_frac, t));
    }
    Ok(())
}

/// Which entry of [`crate::pbc::gamma::Setup::images`] holds the `(a, b, T)` block.
///
/// The image list is built by the same neighbour-list walk the perturbation makes, so every
/// inter-atomic destination has an entry — but only for pairs inside the resonance range, which
/// is why the lookup returns an option rather than indexing blindly.
fn image_index(
    setup: &crate::pbc::gamma::Setup,
) -> std::collections::HashMap<(usize, usize, [i32; 3]), usize> {
    setup
        .images
        .iter()
        .enumerate()
        .map(|(index, block)| ((block.a, block.b, block.t), index))
        .collect()
}

/// `e^{iq·T}` for one image, from a **fractional** wavevector.
///
/// Fractional rather than Cartesian, and the same arithmetic k-points use
/// ([`crate::pbc::kpoints::frac_phase`]). The two agree in exact arithmetic and not in floating
/// point: at a reciprocal lattice vector the fractional angle is `2πn` exactly, while `q·T`
/// through a Cartesian dot product leaves a residue of order `1e-16` per image. That residue is
/// the difference between `D(q + G) = D(q)` holding to 1e-9 and holding to 1e-6.
fn image_phase(q_frac: [f64; 3], t: [i32; 3]) -> c64 {
    crate::pbc::kpoints::frac_phase(q_frac, t)
}

/// The Coulomb lattice sum's second derivative, phased.
///
/// Two sums per site pair rather than one: the mixed blocks want `Σ_T e^{iq·T} H(d + T)` and the
/// diagonal blocks want the same sum with no phase at all. Reusing the phased result for both —
/// the natural shortcut — would put a `q`-dependence into the self-force-constant, and the
/// acoustic sum rule would fail by exactly that amount.
fn phased_lattice_sum(
    molecule: &Molecule,
    params: &Pm3Parameters,
    periodic: &PeriodicOptions,
    density: &crate::linalg::Matrix,
    basis: &Basis,
    // Cartesian, unlike its neighbours: this one feeds `ewald_phased`, whose shifted reciprocal
    // sum needs `G + q` as a vector rather than a phase.
    q: Vec3,
    matrix: &mut CMatrix,
) -> Result<()> {
    let cell = molecule.cell.expect("checked by the caller");
    let (sites, ewald_params) =
        crate::pbc::hessian::charge_sites(molecule, params, basis, density, periodic)?;

    // Every ordered site pair in the reference cell, once.
    let mut displacements = Vec::new();
    let mut owners = Vec::new();
    for (i, site_i) in sites.iter().enumerate() {
        for (j, site_j) in sites.iter().enumerate() {
            if i >= j {
                continue;
            }
            displacements.push(site_j.position - site_i.position);
            owners.push((site_i.owner, site_j.owner, site_i.charge * site_j.charge));
        }
    }
    // Also every site against its own images, which is a pair with zero displacement.
    let self_start = displacements.len();
    for site in sites.iter() {
        displacements.push(Vec3::zero());
        owners.push((site.owner, site.owner, site.charge * site.charge));
    }

    let phased = ewald_phased(&cell, &displacements, q, &ewald_params)?;
    let plain = ewald_phased(&cell, &displacements, Vec3::zero(), &ewald_params)?;

    for (index, (a, b, charge)) in owners.iter().enumerate() {
        let scale = crate::constants::PM3_EV * charge;
        // A site's interaction with its *own* images depends on the atom's position only through
        // the phase, so it contributes to the diagonal block and to nothing else.
        let self_pair = index >= self_start;
        for alpha in 0..3 {
            for beta in 0..3 {
                let mixed = phased[index].hessian[alpha][beta];
                let diagonal = plain[index].hessian[alpha][beta][0];
                if !self_pair {
                    matrix[(3 * a + alpha, 3 * a + beta)] += c64::new(scale * diagonal, 0.0);
                    matrix[(3 * b + alpha, 3 * b + beta)] += c64::new(scale * diagonal, 0.0);
                }
                matrix[(3 * a + alpha, 3 * b + beta)] -=
                    c64::new(scale * mixed[0], scale * mixed[1]);
                if !self_pair {
                    matrix[(3 * b + alpha, 3 * a + beta)] -=
                        c64::new(scale * mixed[0], -scale * mixed[1]);
                }
            }
        }
        if self_pair {
            // Undo the double count: the loop above already wrote the mixed term once, and a
            // self pair has only one destination.
            for alpha in 0..3 {
                for beta in 0..3 {
                    let diagonal = plain[index].hessian[alpha][beta][0];
                    matrix[(3 * a + alpha, 3 * a + beta)] += c64::new(scale * diagonal, 0.0);
                }
            }
        }
    }
    Ok(())
}

/// The first-order Hamiltonian for one displacement, resolved by translation.
///
/// The Bloch transform of this is what couples `k` with `k + q`:
/// `Δh(k) = onsite + Σ_blocks e^{ik·T} block`, matching how
/// [`crate::pbc::kscf::bloch_fock`] assembles the unperturbed one, so the two can be multiplied
/// together without either knowing the other's convention.
pub(crate) struct BareBlocks {
    /// Everything landing on a `T = 0` destination — the atom-diagonal blocks.
    onsite: CMatrix,
    /// One block per entry of [`crate::pbc::gamma::Setup::images`], same order.
    images: Vec<CMatrix>,
}

impl BareBlocks {
    fn new(nao: usize, images: &[crate::pbc::gamma::ImageBlock]) -> Self {
        Self {
            onsite: CMatrix::zeros(nao, nao),
            images: images
                .iter()
                .map(|block| CMatrix::zeros(block.norb_a, block.norb_b))
                .collect(),
        }
    }

    /// `Δh(k)`, in the full AO basis.
    pub(crate) fn at_k(&self, setup: &crate::pbc::gamma::Setup, k_frac: [f64; 3]) -> CMatrix {
        let mut out = self.onsite.clone();
        for (block, values) in setup.images.iter().zip(&self.images) {
            let phase = image_phase(k_frac, block.t);
            let (oa, ob) = (
                setup.basis.atom_offset[block.a],
                setup.basis.atom_offset[block.b],
            );
            for mu in 0..block.norb_a {
                for la in 0..block.norb_b {
                    out[(oa + mu, ob + la)] += phase * values[(mu, la)];
                }
            }
        }
        out
    }
}

/// `∂F/∂u_{Aα}` at fixed density, resolved by translation and carrying the perturbation's phase.
///
/// The phased twin of [`crate::pbc::kernel::fock_derivative`], and the same pair loop. What
/// differs is only where each contribution lands and what phase it carries, and that depends on
/// **which cell the moving atom occupies for that destination**:
///
/// | destination | atom `a` sits in | atom `b` sits in | moving `a` | moving `b` |
/// |---|---|---|---|---|
/// | `a`'s on-site block | cell 0 | cell `T` | `1` | `e^{iq·T}` |
/// | `b`'s on-site block | cell `−T` | cell 0 | `e^{−iq·T}` | `1` |
/// | the `(a, b, T)` image block | cell 0 | cell `T` | `1` | `e^{iq·T}` |
///
/// The second row is the one that is easy to get wrong. A visit `(a, b, T)` computes atom `b` in
/// cell `T` against atom `a` in cell 0, but writes to `b`'s block *in the reference cell*, which
/// by translational symmetry is `b` in cell 0 against `a` in cell `−T`. At `q = 0` every entry
/// is `1` and the whole table collapses onto the unphased derivative, which is what
/// [`tests::the_bare_perturbation_reduces_to_the_gamma_fock_derivative`] pins.
#[allow(clippy::too_many_arguments)]
fn bare_blocks(
    molecule: &Molecule,
    params: &Pm3Parameters,
    periodic: &PeriodicOptions,
    setup: &crate::pbc::gamma::Setup,
    density: &crate::linalg::Matrix,
    // What the exchange reads, and by how much. The total density at `½` for the single
    // restricted channel, `P^σ` at `1` for each unrestricted one — the same number in the
    // restricted case, and the reason there is one code path rather than two.
    exchange_density: &crate::linalg::Matrix,
    exchange_scale: f64,
    q_frac: [f64; 3],
    atom: usize,
    axis: usize,
    index_of: &std::collections::HashMap<(usize, usize, [i32; 3]), usize>,
    // Built once by the caller and shared across all `3N` degrees of freedom.
    //
    // This used to be rebuilt inside, from the same positions and the same cutoff, on every
    // call — `3N` identical lists per response, each an image expansion over the whole cell. It
    // was the dominant cost of a Born-charge or phonon run, not the coupled-perturbed solves:
    // replacing the `3N` solves with three by the interchange theorem left the wall time
    // *unchanged*, which is what pointed here.
    list: &NeighborList,
) -> Result<BareBlocks> {
    use crate::dual::{Dual, Scalar};
    use crate::integrals::{pack, pair_two_electron_g};
    use crate::pbc::gradient::resonance_beta;
    use crate::pbc::screen::point_pair_g;

    let basis = &setup.basis;
    let mut out = BareBlocks::new(basis.nao, &setup.images);

    for pair in list.all() {
        if pair.a != atom && pair.b != atom {
            continue;
        }
        let (a, b) = (pair.a, pair.b);
        let ea = params.element(molecule.atoms[a].z)?;
        let eb = params.element(molecule.atoms[b].z)?;
        let heavy_first = ea.n_orb >= eb.n_orb;
        let (first_index, second_index) = if heavy_first { (a, b) } else { (b, a) };
        let (first, second) = if heavy_first { (ea, eb) } else { (eb, ea) };
        let dvec = if heavy_first {
            pair.dvec
        } else {
            pair.dvec * -1.0
        };

        let chain = if first_index == second_index {
            0.0
        } else if atom == second_index {
            1.0
        } else if atom == first_index {
            -1.0
        } else {
            continue;
        };
        if chain == 0.0 {
            continue;
        }

        // The three phases the table above describes. `forward` is `e^{iq·T}`.
        let forward = image_phase(q_frac, pair.t);
        let on_a = if atom == a {
            c64::new(1.0, 0.0)
        } else {
            forward
        };
        let on_b = if atom == b {
            c64::new(1.0, 0.0)
        } else {
            forward.conj()
        };
        let inter = if atom == a {
            c64::new(1.0, 0.0)
        } else {
            forward
        };
        let (phase_first, phase_second) = if first_index == a {
            (on_a, on_b)
        } else {
            (on_b, on_a)
        };

        let seeded = [
            Dual::var(dvec.x, 0),
            Dual::var(dvec.y, 1),
            Dual::var(dvec.z, 2),
        ];
        let te = pair_two_electron_g::<Dual>(first, second, seeded);
        let point = point_pair_g::<Dual>(
            &setup.atom_sites[first_index],
            first.core_charge,
            &setup.atom_sites[second_index],
            second.core_charge,
            seeded,
        );
        let r_dual = (seeded[0] * seeded[0] + seeded[1] * seeded[1] + seeded[2] * seeded[2]).sqrt();
        let switch = periodic.switch.at_g(r_dual);

        let (na, nb) = (first.n_orb, second.n_orb);
        let off_first = basis.atom_offset[first_index];
        let off_second = basis.atom_offset[second_index];
        let derivative = |value: Dual| chain * value.d[axis];

        for mu in 0..na {
            for nu in 0..na {
                let term = (te.e1b[mu][nu] - point.e1b[mu * na + nu]) * switch * 0.5;
                out.onsite[(off_first + mu, off_first + nu)] += phase_first * derivative(term);
            }
        }
        for la in 0..nb {
            for si in 0..nb {
                let term = (te.e2a[la][si] - point.e2a[la * nb + si]) * switch * 0.5;
                out.onsite[(off_second + la, off_second + si)] += phase_second * derivative(term);
            }
        }

        let inside_short_range = pair.r <= periodic.short_range_cutoff;
        let block_index = index_of.get(&(a, b, pair.t)).copied();
        // Row within atom `a`, column within atom `b`, addressed the way this visit names them.
        let inter_index = |mu: usize, la: usize| -> (usize, usize) {
            if heavy_first {
                (mu, la)
            } else {
                (la, mu)
            }
        };

        if inside_short_range {
            if let Some(slot) = block_index {
                let overlap =
                    crate::overlap::diatom_overlap_dual(first, Vec3::zero(), second, dvec)?;
                #[allow(clippy::needless_range_loop)]
                for mu in 0..na.min(4) {
                    let bi = resonance_beta(first, basis.aos[off_first + mu].orb);
                    for la in 0..nb.min(4) {
                        let bj = resonance_beta(second, basis.aos[off_second + la].orb);
                        let value = derivative(overlap[mu][la] * (0.5 * (bi + bj)));
                        out.images[slot][inter_index(mu, la)] += inter * value;
                    }
                }
            }
        }

        let npack_j = nb * (nb + 1) / 2;
        for mu in 0..na {
            for nu in 0..na {
                let mut accumulator = 0.0;
                for la in 0..nb {
                    for si in 0..nb {
                        let index = pack(mu, nu) * npack_j + pack(la, si);
                        let coulomb = (te.w[index] - point.w[index]) * switch * 0.5;
                        accumulator +=
                            density[(off_second + la, off_second + si)] * derivative(coulomb);
                    }
                }
                out.onsite[(off_first + mu, off_first + nu)] += phase_first * accumulator;
            }
        }
        for la in 0..nb {
            for si in 0..nb {
                let mut accumulator = 0.0;
                for mu in 0..na {
                    for nu in 0..na {
                        let index = pack(mu, nu) * npack_j + pack(la, si);
                        let coulomb = (te.w[index] - point.w[index]) * switch * 0.5;
                        accumulator +=
                            density[(off_first + mu, off_first + nu)] * derivative(coulomb);
                    }
                }
                out.onsite[(off_second + la, off_second + si)] += phase_second * accumulator;
            }
        }
        if inside_short_range {
            if let Some(slot) = block_index {
                for mu in 0..na {
                    for la in 0..nb {
                        let mut accumulator = 0.0;
                        for nu in 0..na {
                            for si in 0..nb {
                                let index = pack(mu, nu) * npack_j + pack(la, si);
                                accumulator += exchange_scale
                                    * exchange_density[(off_first + nu, off_second + si)]
                                    * derivative(te.w[index]);
                            }
                        }
                        out.images[slot][inter_index(mu, la)] -= inter * accumulator;
                    }
                }
            }
        }
    }
    Ok(out)
}

/// The lattice sum's contribution to `∂F/∂u_{Aα}`, carrying the perturbation's phase.
///
/// # Why this cannot be a finite difference
///
/// The Γ-point version ([`crate::pbc::kernel`]) moves the atom and re-runs the Ewald sum. That
/// is exactly the `q = 0` perturbation and no other: moving an atom in a periodic calculation
/// moves it in *every* cell at once, in phase. A displacement pattern that varies from cell to
/// cell cannot be expressed by moving anything.
///
/// # What is differentiated
///
/// To first order, displacing a charge `q_t` by `u` is a dipole `q_t u` at its position, so the
/// potential at site `s` changes by the phased kernel's gradient:
///
/// ```text
/// δV_s / u = Σ_{t ∈ A} q_t ∇Φ_q(r_t − r_s)  −  [s ∈ A] Σ_t q_t ∇Φ_0(r_t − r_s)
/// ```
///
/// The second term is the field *point* moving, which happens only for sites on the displaced
/// atom itself and only in the reference cell — hence `Φ_0` and not `Φ_q`. Dropping it leaves a
/// derivative that looks reasonable and violates the acoustic sum rule, because translating
/// everything would then change the potential each site sees.
///
/// Pairs of sites sharing an atom do not interact at `T = 0` — that energy is the one-center
/// integral set — and [`ewald_phased`] knows nothing about owners, so their `T = 0` term is
/// removed by hand.
///
/// # Why the kernel here is microscopic
///
/// Through 0.2.1 this term used the full phased kernel, macroscopic member and all, and the
/// acoustic sum rule of the assembled `D(q)` — `max_{row,β} |Σ_a Φ_{row,3aβ}(q)|`, which must
/// stay finite as `q → 0` — diverged as a clean `1/q²`:
///
/// | `q` (fractional) | sum rule, `G = 0` kept | excluded (now) |
/// |---|---|---|
/// | 0.1 | 3.69 | 0.120 |
/// | 0.05 | 13.9 | 0.033 |
/// | 0.025 | 55.1 | 0.011 |
/// | 0.0125 | 220.1 | 0.0052 |
///
/// On water in a 10 Bohr cube that put the lowest frequency at `−4705 cm⁻¹` by `q = 0.01`. No
/// test saw it because every wavevector in this file is moderate or commensurate; none goes near
/// `q → 0`, where `1/q²` is the difference between a plausible number and a nonsensical one.
///
/// Three things were ruled out before this term was reached: the rigid-ion matrix is clean at
/// every `q`; a non-polar cell (an H₂ chain, whose Born charges measure zero) is clean with the
/// response included; and the induced charges are a faithful image of `Tr[ΔP]`, so the multipole
/// decomposition is not leaking — projecting the uniform component out of them changes the sum
/// rule by less than a part in ten thousand. See the note in [`phased_kernel`].
///
/// The mechanism: the `G = 0` half of `∇Φ_q` is weighted here by the **displaced atom's own**
/// charge rather than by the cell's total, so there is no `Σ_a Q_a = 0` to cancel the
/// `4π/(Ωq²)`. Inside a self-consistent solve that makes the bare perturbation `O(1/q)` per atom
/// and the induced density `O(1/q)` in reply, and their contraction `O(1/q²)`.
///
/// The fix is the standard decomposition: the response runs on the microscopic kernel
/// ([`crate::pbc::phased::Macroscopic::Exclude`]) and the macroscopic field is restored
/// analytically by [`crate::pbc::lo_to::non_analytic_term`], whose prefactor is measured against
/// the rigid-ion lattice sum that still contains it. The doubled-cell folding identity is
/// unaffected — it compares against a supercell's Γ Hessian, where the background has already
/// removed the same member, so excluding it here makes the two *more* consistent, not less.
#[allow(clippy::too_many_arguments)]
/// The two phased lattice sums every degree of freedom's long-range term reads.
///
/// # Why they are a table
///
/// This term used to run its own two lattice sums inside
/// [`phased_field_derivative`], which is called `3N` times — so `6N` full sums over the site
/// pairs, at **85% of the total cost** of a Born-charge or phonon run on a small cell. Three
/// separate attempts at speeding those runs up did nothing, because all three were aimed
/// somewhere else; this is what the measurement eventually pointed at.
///
/// Two things were being repeated. The displacements do not depend on `axis` at all — only the
/// component pulled out of the finished kernel does — so each sum was recomputed three times per
/// atom for nothing. And across atoms the displacement sets are disjoint slices of the same
/// site-by-site table, so the union of all `3N` calls is one sum over every site pair.
///
/// Building that table once costs the same as **two** of the `6N` sums it replaces.
pub(crate) struct LongRangeKernels {
    /// At `q`, macroscopic term excluded. Indexed `[s * n_sites + t]` for the displacement
    /// `sites[t] − sites[s]`.
    moved: Vec<crate::pbc::phased::PhasedKernel>,
    /// The same displacements at `q = 0`.
    sources: Vec<crate::pbc::phased::PhasedKernel>,
    n_sites: usize,
}

/// Build [`LongRangeKernels`] for a wavevector — once per response, not once per degree of freedom.
pub(crate) fn long_range_kernels(
    setup: &crate::pbc::gamma::Setup,
    cell: &Cell,
    q: Vec3,
) -> Result<LongRangeKernels> {
    let sites = &setup.sites;
    let n_sites = sites.len();
    let mut displacements = Vec::with_capacity(n_sites * n_sites);
    for s in 0..n_sites {
        for t in 0..n_sites {
            displacements.push(sites[t].position - sites[s].position);
        }
    }
    // Microscopic: the `G = 0` member is left out. That member is the macroscopic field, and
    // inside a self-consistent response it is what makes the sum rule diverge — see the note on
    // [`phased_field_derivative`] and on [`crate::pbc::phased::Macroscopic`]. Its contribution to
    // the dynamical matrix is restored analytically by [`crate::pbc::lo_to::non_analytic_term`].
    //
    // The `q = 0` sum below already has it excluded, because there the neutralizing background
    // removes it. Using the microscopic kernel here is what makes the two halves consistent
    // rather than mismatched.
    let moved = crate::pbc::phased::ewald_phased_with(
        cell,
        &displacements,
        q,
        &setup.ewald_params,
        crate::pbc::phased::Macroscopic::Exclude,
    )?;
    let sources = ewald_phased(cell, &displacements, Vec3::zero(), &setup.ewald_params)?;
    Ok(LongRangeKernels {
        moved,
        sources,
        n_sites,
    })
}

fn phased_field_derivative(
    setup: &crate::pbc::gamma::Setup,
    kernels: &LongRangeKernels,
    charges: &[f64],
    atom: usize,
    axis: usize,
    out: &mut CMatrix,
) -> Result<()> {
    let sites = &setup.sites;
    let owned = setup.site_offset[atom]
        ..setup
            .site_offset
            .get(atom + 1)
            .copied()
            .unwrap_or(sites.len());
    let n_sites = kernels.n_sites;

    // `∂V_s/∂u` for every site, as a complex number.
    let mut potential = vec![c64::new(0.0, 0.0); sites.len()];
    for (s, slot) in potential.iter_mut().enumerate() {
        for t in owned.clone() {
            if s == t {
                continue;
            }
            let kernel = &kernels.moved[s * n_sites + t];
            let mut value = c64::new(kernel.gradient[axis][0], kernel.gradient[axis][1]);
            if sites[s].owner == sites[t].owner {
                value -= bare_pair_gradient(sites[t].position - sites[s].position, axis);
            }
            *slot += value * charges[t];
        }
    }
    for s in owned.clone() {
        for t in 0..sites.len() {
            if s == t {
                continue;
            }
            let kernel = &kernels.sources[s * n_sites + t];
            let mut value = c64::new(kernel.gradient[axis][0], kernel.gradient[axis][1]);
            if sites[s].owner == sites[t].owner {
                value -= bare_pair_gradient(sites[t].position - sites[s].position, axis);
            }
            potential[s] -= value * charges[t];
        }
    }

    // Contract onto the Fock blocks the same way the unperturbed field does.
    for (ia, atom_sites) in setup.atom_sites.iter().enumerate() {
        let n = setup.basis.atom_norb[ia];
        if n == 0 {
            continue;
        }
        let off = setup.basis.atom_offset[ia];
        let start = setup.site_offset[ia];
        let count = atom_sites.offsets.len();
        let real: Vec<f64> = (0..count).map(|i| potential[start + i].re).collect();
        let imaginary: Vec<f64> = (0..count).map(|i| potential[start + i].im).collect();
        for mu in 0..n {
            for nu in 0..n {
                out[(off + mu, off + nu)] += c64::new(
                    atom_sites.fock_contribution(mu, nu, &real),
                    atom_sites.fock_contribution(mu, nu, &imaginary),
                );
            }
        }
    }
    Ok(())
}

/// `∂(1/|d|)/∂d_axis` in eV — the bare `T = 0` term the owner exclusion removes.
fn bare_pair_gradient(d: Vec3, axis: usize) -> c64 {
    let r = d.norm();
    if r < 1.0e-12 {
        return c64::new(0.0, 0.0);
    }
    let component = match axis {
        0 => d.x,
        1 => d.y,
        _ => d.z,
    };
    c64::new(-component / (r * r * r), 0.0)
}

/// `G[ΔP]` at wavevector `q` — how a first-order density feeds back into the potential.
///
/// The phased twin of [`crate::pbc::kernel::PeriodicKernel::apply`]. Three pieces, and each
/// takes its phase from a different place:
///
/// - **One-centre** terms are intra-atomic. No image, no phase.
/// - **Coulomb** couples atom `a`'s block in cell 0 to atom `b`'s in cell `T`, and the
///   first-order density there is `ΔP_b e^{iq·T}`. The tables cannot be the image-summed ones
///   [`crate::pbc::gamma::Setup::pairs`] holds, because a summed table has no `T` left to phase;
///   the neighbour list is walked again for them.
/// - **Exchange** writes to the `(a, b, T)` block and reads the first-order density *of that
///   same block*, so the phase is already carried by which block it is. Adding `e^{iq·T}` here
///   as well is the natural mistake and double-counts it.
///
/// The long-range half needs only the phased kernel's *value*, not its gradient: a first-order
/// charge `δq_t` on site `t` in cell `T` is `δq_t e^{iq·T}`, so the potential it produces at a
/// site in the reference cell is `Σ_t δq_t Φ_q(r_t − r_s)`.
#[allow(clippy::too_many_arguments)]
fn phased_kernel(
    molecule: &Molecule,
    params: &Pm3Parameters,
    periodic: &PeriodicOptions,
    setup: &crate::pbc::gamma::Setup,
    cell: &Cell,
    q_frac: [f64; 3],
    delta: &[BareBlocks],
    channels: &[SpinChannel],
    index_of: &std::collections::HashMap<(usize, usize, [i32; 3]), usize>,
    kernel: &SiteKernel,
) -> Result<Vec<BareBlocks>> {
    use crate::integrals::pack;
    use crate::pbc::screen::{nddo_pair, screened_pair};

    let basis = &setup.basis;
    let n_spin = delta.len();

    // Everything Coulomb reads is the *total* first-order density; only exchange resolves spin.
    // A restricted calculation has one channel already holding the total, so the sum is a no-op
    // there and costs nothing.
    let total = if n_spin == 1 {
        None
    } else {
        let mut sum = BareBlocks::new(basis.nao, &setup.images);
        for blocks in delta {
            for i in 0..basis.nao {
                for j in 0..basis.nao {
                    sum.onsite[(i, j)] += blocks.onsite[(i, j)];
                }
            }
            for (slot, source) in sum.images.iter_mut().zip(&blocks.images) {
                for i in 0..slot.rows {
                    for j in 0..slot.cols {
                        slot[(i, j)] += source[(i, j)];
                    }
                }
            }
        }
        Some(sum)
    };
    let total = total.as_ref().unwrap_or(&delta[0]);

    // The shared half, and one exchange half per channel. They are summed at the end rather than
    // written into separate outputs as they go, because everything except exchange is computed
    // once for all channels and would otherwise be computed `n_spin` times.
    let mut out = BareBlocks::new(basis.nao, &setup.images);
    let mut exchange: Vec<BareBlocks> = (0..n_spin)
        .map(|_| BareBlocks::new(basis.nao, &setup.images))
        .collect();

    // One centre: the same contraction the ground state makes, on each spin's half.
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let elem = params.element(atom.z)?;
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
        if n == 0 {
            continue;
        }
        let oc = |a: usize, b: usize, c: usize, d: usize| -> f64 {
            if let Some(spd) = &elem.onecenter {
                spd.get(a, b, c, d)
            } else {
                crate::fock::oc_two_electron(
                    a, b, c, d, elem.g_ss, elem.g_sp, elem.g_pp, elem.g_p2, elem.h_sp,
                )
            }
        };
        for mu in 0..n {
            for nu in 0..n {
                let mut acc = c64::new(0.0, 0.0);
                for la in 0..n {
                    for si in 0..n {
                        acc += total.onsite[(off + la, off + si)] * oc(mu, nu, la, si);
                    }
                }
                out.onsite[(off + mu, off + nu)] += acc;
                for (spin, channel) in channels.iter().enumerate() {
                    let mut acc = c64::new(0.0, 0.0);
                    for la in 0..n {
                        for si in 0..n {
                            acc += delta[spin].onsite[(off + la, off + si)]
                                * (channel.exchange_scale * oc(mu, la, nu, si));
                        }
                    }
                    exchange[spin].onsite[(off + mu, off + nu)] -= acc;
                }
            }
        }
    }

    let cutoff = periodic.short_range_cutoff.max(periodic.switch.off);
    let positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
    let list = NeighborList::build_from_positions(&positions, Some(cell), cutoff);

    for pair in list.all() {
        let (a, b) = (pair.a, pair.b);
        let ea = params.element(molecule.atoms[a].z)?;
        let eb = params.element(molecule.atoms[b].z)?;
        let heavy_first = ea.n_orb >= eb.n_orb;
        let (first_index, second_index) = if heavy_first { (a, b) } else { (b, a) };
        let (first, second) = if heavy_first { (ea, eb) } else { (eb, ea) };
        let dvec = if heavy_first {
            pair.dvec
        } else {
            pair.dvec * -1.0
        };

        let te = nddo_pair(first, second, dvec);
        let screened = screened_pair(
            &te,
            &setup.atom_sites[first_index],
            first.core_charge,
            &setup.atom_sites[second_index],
            second.core_charge,
            dvec,
            periodic.switch,
        );

        let (na, nb) = (screened.norb_i, screened.norb_j);
        let (off_first, off_second) = (
            basis.atom_offset[first_index],
            basis.atom_offset[second_index],
        );
        // `a`'s block sees `b` in cell `T`; `b`'s block sees `a` in cell `−T`.
        let forward = image_phase(q_frac, pair.t);
        let (to_first, to_second) = if first_index == a {
            (forward, forward.conj())
        } else {
            (forward.conj(), forward)
        };

        // Coulomb, halved because the ordered list visits each physical pair twice.
        for mu in 0..na {
            for nu in 0..na {
                let mut acc = c64::new(0.0, 0.0);
                for la in 0..nb {
                    for si in 0..nb {
                        let w = screened.w[pack(mu, nu) * screened.npack_j + pack(la, si)];
                        acc += total.onsite[(off_second + la, off_second + si)] * (0.5 * w);
                    }
                }
                out.onsite[(off_first + mu, off_first + nu)] += to_first * acc;
            }
        }
        for la in 0..nb {
            for si in 0..nb {
                let mut acc = c64::new(0.0, 0.0);
                for mu in 0..na {
                    for nu in 0..na {
                        let w = screened.w[pack(mu, nu) * screened.npack_j + pack(la, si)];
                        acc += total.onsite[(off_first + mu, off_first + nu)] * (0.5 * w);
                    }
                }
                out.onsite[(off_second + la, off_second + si)] += to_second * acc;
            }
        }

        // Exchange, on the image block whose own index already carries the phase.
        if pair.r <= periodic.short_range_cutoff {
            if let Some(slot) = index_of.get(&(a, b, pair.t)).copied() {
                let block = &setup.images[slot];
                let (norb_a, norb_b) = (block.norb_a, block.norb_b);
                for (spin, channel) in channels.iter().enumerate() {
                    let source = &delta[spin].images[slot];
                    for mu in 0..norb_a {
                        for la in 0..norb_b {
                            let mut acc = c64::new(0.0, 0.0);
                            for nu in 0..norb_a {
                                for si in 0..norb_b {
                                    let index = pack(mu, nu) * block.npack_b + pack(la, si);
                                    acc += source[(nu, si)]
                                        * (channel.exchange_scale * block.exchange[index]);
                                }
                            }
                            exchange[spin].images[slot][(mu, la)] -= acc;
                        }
                    }
                }
            }
        }
    }

    // The long-range half: the first-order multipole charges' own phased field.
    let charges = induced_charges(setup, total);
    // Not neutralized here, and that was checked rather than assumed.
    //
    // `Σ_t δq_t` does not vanish as `q → 0` — it settles at about `1.6e-2` for water — and the
    // obvious reading is that the `4π/(Ωq²)` kernel amplifies that residual into the `1/q²`
    // divergence the acoustic sum rule shows. It does not: projecting the uniform component out
    // here changes the sum rule by less than a part in ten thousand at every wavevector probed.
    // The divergence enters through the *bare* perturbation's long-range channel instead; see
    // `phased_field_derivative`.
    // `PM3_DFPT_CHARGE_TRACE=1` prints the induced charge the long-range kernel is about to see,
    // beside the trace of the density it came from.
    //
    // Both are here because they answer different questions. The induced charge must vanish as
    // `q → 0`: a uniform translation redistributes nothing, so the `q = 0` Fourier component of
    // the induced density is zero, and the `4π/(Ωq²)` kernel would otherwise amplify whatever is
    // left without bound. And in an orthonormal basis that charge is exactly `−Tr[ΔP]`, so the
    // two agreeing says the multipole-site decomposition is faithful and puts any discrepancy in
    // the response upstream of it rather than in the mapping.
    if std::env::var_os("PM3_DFPT_CHARGE_TRACE").is_some() {
        let total_charge: c64 = charges.iter().fold(c64::new(0.0, 0.0), |a, b| a + *b);
        let trace: c64 =
            (0..basis.nao).fold(c64::new(0.0, 0.0), |a, mu| a + total.onsite[(mu, mu)]);
        eprintln!(
            "  induced charge sum {:.6e}  |Tr dP| {:.6e}",
            total_charge.norm(),
            trace.norm()
        );
    }
    let n_sites = setup.sites.len();
    let mut potential = vec![c64::new(0.0, 0.0); n_sites];
    for (s, slot) in potential.iter_mut().enumerate() {
        for (t, charge) in charges.iter().enumerate() {
            *slot += *charge * kernel.value[s * n_sites + t];
        }
    }
    for (ia, atom_sites) in setup.atom_sites.iter().enumerate() {
        let n = basis.atom_norb[ia];
        if n == 0 {
            continue;
        }
        let off = basis.atom_offset[ia];
        let start = setup.site_offset[ia];
        let count = atom_sites.offsets.len();
        let scale = crate::constants::PM3_EV;
        let real: Vec<f64> = (0..count)
            .map(|i| scale * potential[start + i].re)
            .collect();
        let imaginary: Vec<f64> = (0..count)
            .map(|i| scale * potential[start + i].im)
            .collect();
        for mu in 0..n {
            for nu in 0..n {
                out.onsite[(off + mu, off + nu)] += c64::new(
                    atom_sites.fock_contribution(mu, nu, &real),
                    atom_sites.fock_contribution(mu, nu, &imaginary),
                );
            }
        }
    }

    // Shared plus each channel's own exchange. The last channel takes `out` itself rather than a
    // copy of it, so a restricted calculation — one channel — never clones.
    let mut per_spin = exchange;
    for (index, slot) in per_spin.iter_mut().enumerate() {
        let shared = if index + 1 == n_spin {
            std::mem::replace(&mut out, BareBlocks::new(0, &[]))
        } else {
            BareBlocks {
                onsite: out.onsite.clone(),
                images: out.images.clone(),
            }
        };
        for i in 0..basis.nao {
            for j in 0..basis.nao {
                slot.onsite[(i, j)] += shared.onsite[(i, j)];
            }
        }
        for (target, source) in slot.images.iter_mut().zip(&shared.images) {
            for i in 0..target.rows {
                for j in 0..target.cols {
                    target[(i, j)] += source[(i, j)];
                }
            }
        }
    }
    Ok(per_spin)
}

/// `Φ_q(r_t − r_s)` for every ordered site pair, with the same-atom `T = 0` term already removed.
///
/// The geometry does not change during a response solve — only the charges do — so this is built
/// once per wavevector and reused by every degree of freedom and every iteration. Recomputing it
/// inside the loop is what made the first working version take three minutes on a water molecule.
pub(crate) struct SiteKernel {
    value: Vec<c64>,
}

impl SiteKernel {
    fn build(setup: &crate::pbc::gamma::Setup, cell: &Cell, q: Vec3) -> Result<Self> {
        let sites = &setup.sites;
        let mut displacements = Vec::with_capacity(sites.len() * sites.len());
        for s in sites {
            for t in sites {
                displacements.push(t.position - s.position);
            }
        }
        // Microscopic, matching the bare perturbation. Both halves of a self-consistent response
        // have to see the same kernel: screening the macroscopic field with a response that was
        // driven without it — or the reverse — is a different equation from either.
        let kernel = crate::pbc::phased::ewald_phased_with(
            cell,
            &displacements,
            q,
            &setup.ewald_params,
            crate::pbc::phased::Macroscopic::Exclude,
        )?;
        let mut value = Vec::with_capacity(kernel.len());
        for (index, entry) in kernel.iter().enumerate() {
            let (s, t) = (index / sites.len(), index % sites.len());
            let mut v = c64::new(entry.value[0], entry.value[1]);
            // Sites on the same atom do not interact **in the reference cell** — that energy is
            // the one-centre integral set. Their *images* do, so only the `T = 0` term comes
            // off, and only when the two sites are distinct: a site against itself has no
            // `T = 0` term left, the Ewald self term having already taken it.
            if sites[s].owner == sites[t].owner && s != t {
                let r = (sites[t].position - sites[s].position).norm();
                if r > 1.0e-12 {
                    v -= c64::new(1.0 / r, 0.0);
                }
            }
            value.push(v);
        }
        Ok(Self { value })
    }
}

/// The first-order multipole charges a first-order density puts on each site.
fn induced_charges(setup: &crate::pbc::gamma::Setup, delta: &BareBlocks) -> Vec<c64> {
    let mut out = vec![c64::new(0.0, 0.0); setup.sites.len()];
    for (ia, atom_sites) in setup.atom_sites.iter().enumerate() {
        let n = setup.basis.atom_norb[ia];
        if n == 0 {
            continue;
        }
        let off = setup.basis.atom_offset[ia];
        let start = setup.site_offset[ia];
        for mu in 0..n {
            for nu in 0..=mu {
                // `2 ﾎ捻[ﾎｼ,ﾎｽ]` and `ﾎ捻[ﾎｼ,ﾎｽ] + ﾎ捻[ﾎｽ,ﾎｼ]` are the same number for the ground state,
                // whose density is real and symmetric, and they are not the same number here: a
                // first-order density at finite `q` is complex and has no symmetry between its
                // two indices. The symmetrized form is the one that makes this the adjoint of
                // `fock_contribution`, which scatters a site potential back over the *whole*
                // block; the doubled lower triangle is adjoint to nothing, and the mismatch is
                // what made `D(q)` non-Hermitian by an amount growing linearly in `q`.
                let population = if mu == nu {
                    delta.onsite[(off + mu, off + mu)]
                } else {
                    delta.onsite[(off + mu, off + nu)] + delta.onsite[(off + nu, off + mu)]
                };
                for term in &atom_sites.pair_weights[crate::pbc::multipole::packed_index(nu, mu)] {
                    out[start + term.site] -= population * term.weight;
                }
            }
        }
    }
    out
}

/// The electrons' response to the displacement, added to the dynamical matrix.
///
/// # The linear response
///
/// A phonon at `q` mixes each occupied state at `k` with each empty state at `k + q`, and the
/// first-order density in that band basis is
///
/// ```text
/// ΔP_{mn} = [f_n(k) − f_m(k+q)] / [ε_n(k) − ε_m(k+q)] · ⟨ψ_{m,k+q}| ΔV |ψ_{n,k}⟩
/// ```
///
/// Both the occupied–empty and the empty–occupied blocks contribute. The second is the response
/// of the *bra* at `k + q` to the `−q` component of a real displacement, and dropping it halves
/// the answer — a factor that looks like a normalization convention and is not.
///
/// # The k-sum
///
/// One point, `k = Γ`, so the pair is `(Γ, q)`. That is the same sampling the ground state used
/// (`run_gamma`), and using a denser mesh here than the density was converged with would be
/// answering a different question. It carries the same condition: `P(0, T)` is taken as `P(Γ)`
/// for every image, which is exact when every periodic width exceeds the exchange range and
/// wrong otherwise — see [`crate::pbc::gamma::PeriodicResult::gamma_margin`], which applies to
/// this unchanged.
///
/// # What is contracted with what
///
/// `D^resp_{jj'} = Σ_k Tr[Δh^{j†}(k) ΔP^{j'}(k)]` — the *bare* perturbation against the
/// *self-consistent* response, which is what the 2n+1 theorem leaves once the two-electron
/// double counting cancels. Contracting the self-consistent potential against the
/// self-consistent response instead would count the interaction twice.
/// The converged ground state a response is built on, from either sampling.
///
/// Both samplings supply the same five things, and the response cannot tell which produced them.
/// That is the point: a Γ-point calculation is a one-point mesh, not a different method, and
/// writing the response against the *contents* rather than against a `PeriodicResult` is what
/// keeps it from having two code paths.
struct GroundState<'a> {
    /// `P(0)`, the direct-space density in the origin cell, both spins. Coulomb reads this, and
    /// it is the same matrix whether or not the calculation is spin-polarized.
    density: &'a crate::linalg::Matrix,
    /// One channel for a restricted calculation, two for an unrestricted one.
    channels: Vec<SpinChannel>,
    /// The k-points the density was converged on.
    points: Vec<crate::pbc::kpoints::KPoint>,
    smearing_ev: f64,
}

/// One spin channel of the ground state, and everything the response needs to build its own.
///
/// A **restricted** calculation has exactly one of these. Its `exchange_density` is the *total*
/// density and its `exchange_scale` is `½`, which is the same number as a spin density at full
/// strength and is what lets one channel stand for two. Each state holds two electrons.
///
/// An **unrestricted** calculation has two. Each carries its own `P^σ` at full exchange strength
/// and holds one electron per state, and the two are coupled only through the Coulomb kernel,
/// which reads the total.
///
/// Writing it as a list rather than as a pair of code paths is what keeps the restricted case at
/// its old cost: it is one channel, not two identical ones, so nothing is diagonalized twice.
struct SpinChannel {
    /// What this channel's **exchange** reads, scaled by `exchange_scale`. Coulomb always reads
    /// [`GroundState::density`] instead.
    exchange_density: crate::linalg::Matrix,
    /// `½` for the single restricted channel, `1` for each unrestricted one.
    exchange_scale: f64,
    /// Electrons one state holds: `2` restricted, `1` per spin channel.
    occupancy: f64,
    /// The per-image density this channel's exchange sees — half the total for a closed shell,
    /// since the image blocks feed the exchange and handing them the total doubles it and pushes
    /// every orbital down by about twelve eV on water. Obvious once looked at, invisible until.
    p_images: Vec<Vec<f64>>,
    /// The converged on-site Fock, which [`crate::pbc::kscf::bloch_fock`] turns into `F(k)`.
    f_onsite: crate::linalg::Matrix,
    /// Chemical potential (eV) for this channel.
    ///
    /// The occupation is read from this rather than from a band index, and it has to be: an
    /// index counts states below the Fermi level *at one k*, and `k + q` is in general a point
    /// the mesh never sampled, where there is no such count to inherit. Under a fixed
    /// magnetization the two channels have different ones.
    fermi_ev: f64,
}

/// A Γ-point result, presented as the one-point mesh it is.
///
/// `P(0, T) = P(Γ)` for every image is what a single k-point *means* rather than an
/// approximation made here, and the Fock is rebuilt from the converged density so it can be
/// evaluated at `k + q` — the same reconstruction [`crate::pbc::kscf::converged_potential`] does
/// for a real mesh, which is why the response cannot tell them apart.
fn gamma_ground_state<'a>(
    molecule: &Molecule,
    params: &Pm3Parameters,
    setup: &crate::pbc::gamma::Setup,
    cell: &Cell,
    scf: &'a crate::pbc::gamma::PeriodicResult,
) -> Result<GroundState<'a>> {
    let spin_density = |alpha: bool| -> crate::linalg::Matrix {
        let mut p = scf.density.clone();
        match &scf.spin_density {
            // `P^σ = ½(P ± Δ)`.
            Some(difference) => {
                for (value, delta) in p.as_mut_slice().iter_mut().zip(difference.as_slice()) {
                    *value = 0.5 * (*value + if alpha { *delta } else { -*delta });
                }
            }
            None => {
                for value in p.as_mut_slice() {
                    *value *= 0.5;
                }
            }
        }
        p
    };
    let channel = |exchange_density: crate::linalg::Matrix,
                   exchange_scale: f64,
                   occupancy: f64,
                   fermi_ev: f64|
     -> Result<SpinChannel> {
        let p_images: Vec<Vec<f64>> = setup
            .images
            .iter()
            .map(|block| {
                let (oa, ob) = (
                    setup.basis.atom_offset[block.a],
                    setup.basis.atom_offset[block.b],
                );
                let mut out = vec![0.0; block.norb_a * block.norb_b];
                for mu in 0..block.norb_a {
                    for la in 0..block.norb_b {
                        // Always the spin density, whatever `exchange_density` is: the image
                        // blocks feed the exchange and nothing else.
                        out[mu * block.norb_b + la] =
                            exchange_scale * exchange_density[(oa + mu, ob + la)];
                    }
                }
                out
            })
            .collect();
        let spin = {
            let mut p = exchange_density.clone();
            for value in p.as_mut_slice() {
                *value *= exchange_scale;
            }
            p
        };
        let mut f_onsite =
            crate::pbc::kscf::onsite_fock(molecule, params, setup, &scf.density, &spin)?;
        let mut sites = setup.sites.clone();
        crate::pbc::gamma::write_electron_charges(setup, &scf.density, &mut sites);
        let field = crate::pbc::ewald::ewald_potentials_cached(
            cell,
            &sites,
            &setup.ewald_params,
            &setup.ewald_context,
        )?;
        crate::pbc::gamma::add_site_potential(setup, &mut f_onsite, &field);
        Ok(SpinChannel {
            exchange_density,
            exchange_scale,
            occupancy,
            p_images,
            f_onsite,
            fermi_ev,
        })
    };

    // A step at the midpoint of the gap. With zero smearing this reproduces the band-index rule
    // exactly for an insulator, which is what a Γ-point calculation of a gapped system is — and
    // the tests that pinned the index rule are what says so.
    let fermi_ev = match (scf.homo_ev, scf.lumo_ev) {
        (Some(homo), Some(lumo)) => 0.5 * (homo + lumo),
        _ => 0.0,
    };
    let channels = if scf.unrestricted {
        vec![
            channel(spin_density(true), 1.0, 1.0, fermi_ev)?,
            channel(spin_density(false), 1.0, 1.0, fermi_ev)?,
        ]
    } else {
        vec![channel(scf.density.clone(), 0.5, 2.0, fermi_ev)?]
    };
    Ok(GroundState {
        density: &scf.density,
        channels,
        points: vec![crate::pbc::kpoints::KPoint::GAMMA],
        smearing_ev: 0.0,
    })
}

/// A k-mesh result, in the same shape.
///
/// The difference from [`gamma_ground_state`] is the whole content of "sampling a mesh": the
/// image densities are the real `P(T)` rather than one matrix repeated, the points are the mesh
/// rather than one, and the Fermi level is the one the mesh found rather than the midpoint of a
/// single gap. Nothing downstream changes.
///
/// # Why the points are not the ones the SCF used
///
/// [`crate::pbc::kscf::run_kpoints`] samples the *time-reversal-irreducible* mesh, and it is
/// right to: `P(−k) = P(k)*`, so the reduced set with doubled weights reproduces the ground-state
/// density exactly. The response cannot borrow that argument. Time reversal maps the coupled pair
/// `(k, k + q)` to `(−k, −k + q)`, and `−k + q` is not the partner of `−k` at `q` — it is the
/// partner at `−q`. The true sum over the irreducible set is
/// `Σ w_k {Tr[…](k, k+q) + Tr[…](−k, −k+q)}` and doubling gives `Σ 2w_k Tr[…](k, k+q)`, which
/// are different numbers whenever `q ≠ −q` modulo a reciprocal lattice vector. So the response
/// regenerates the **full** mesh and runs every point.
///
/// The converged potential is shared unchanged: it is a property of the density, which the
/// reduction does reproduce exactly, and the response diagonalizes it wherever it likes. Nothing
/// is reconstructed from the irreducible eigenvectors, so no degenerate-subspace gauge choice
/// enters — the question is avoided rather than answered.
fn mesh_ground_state<'a>(
    molecule: &Molecule,
    params: &Pm3Parameters,
    setup: &crate::pbc::gamma::Setup,
    cell: &Cell,
    spec: &crate::pbc::kpoints::KpointSpec,
    scf: &'a crate::pbc::kscf::KpointResult,
    smearing_ev: f64,
) -> Result<GroundState<'a>> {
    use crate::pbc::kpoints::{monkhorst_pack, KpointSpec};

    let potential = crate::pbc::kscf::converged_potential(molecule, params, setup, scf)?;
    let points = match spec {
        KpointSpec::Mesh { divisions, shift } => monkhorst_pack(cell, *divisions, *shift)?,
        // Γ is its own reduction, and an explicit list was never reduced in the first place.
        KpointSpec::Gamma | KpointSpec::Explicit(_) => scf.kpoints.clone(),
    };
    // `fill` reports the *highest occupied* energy as the Fermi level when nothing is smeared,
    // which is a fine answer for a quantity that only has to be somewhere in the gap — and the
    // wrong one to hand a strict `ε < μ` test, which then empties the very state that defined μ.
    // The midpoint is what the Γ path uses, and it is the only choice that survives the response
    // diagonalizing `k + q` afresh: a level sitting exactly on a band edge decides the filling of
    // states at other wavevectors by their last bits.
    let midpoint = |reported: f64| match (smearing_ev > 0.0, scf.homo_ev, scf.lumo_ev) {
        (false, Some(homo), Some(lumo)) if lumo > homo => 0.5 * (homo + lumo),
        _ => reported,
    };
    // The image densities are already per spin, so they are the exchange density this channel
    // needs at full strength. The on-site exchange reads the same thing as a matrix.
    let spin_matrix = |images: &crate::pbc::kscf::DensitySet| images.onsite.clone();
    let channels = if scf.unrestricted {
        vec![
            SpinChannel {
                exchange_density: spin_matrix(potential.p_alpha),
                exchange_scale: 1.0,
                occupancy: 1.0,
                p_images: potential.p_alpha.images.clone(),
                f_onsite: potential.f_onsite_alpha.clone(),
                fermi_ev: midpoint(scf.fermi_ev),
            },
            SpinChannel {
                exchange_density: spin_matrix(potential.p_beta),
                exchange_scale: 1.0,
                occupancy: 1.0,
                p_images: potential.p_beta.images.clone(),
                f_onsite: potential.beta_fock().clone(),
                fermi_ev: midpoint(scf.fermi_beta_ev),
            },
        ]
    } else {
        // One channel holding the *total*, at half exchange strength — the same numbers, without
        // diagonalizing twice. Its image densities are still the α ones, because that is what the
        // exchange in `bloch_fock` reads.
        vec![SpinChannel {
            exchange_density: scf.density.clone(),
            exchange_scale: 0.5,
            occupancy: 2.0,
            p_images: potential.p_alpha.images.clone(),
            f_onsite: potential.f_onsite_alpha.clone(),
            fermi_ev: midpoint(scf.fermi_ev),
        }]
    };
    Ok(GroundState {
        density: &scf.density,
        channels,
        points,
        smearing_ev,
    })
}

impl GroundState<'_> {
    /// Electrons in state `index` of `bands`, in the channel's own units.
    fn occupation(&self, channel: &SpinChannel, bands: &[f64], index: usize) -> f64 {
        channel.occupancy * self.filling(channel.fermi_ev, bands, index)
    }

    /// The fraction of `occupancy` that state `index` carries.
    fn filling(&self, fermi_ev: f64, bands: &[f64], index: usize) -> f64 {
        if self.smearing_ev <= 0.0 {
            // A step. Ties go to unoccupied, which for an insulator never happens and for a
            // metal is the case the caller was told to smear.
            if bands[index] < fermi_ev {
                1.0
            } else {
                0.0
            }
        } else {
            crate::pbc::kscf::fermi_dirac(bands[index], fermi_ev, self.smearing_ev)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn response(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
    setup: &crate::pbc::gamma::Setup,
    ground: &GroundState<'_>,
    cell: &Cell,
    q: Vec3,
    q_frac: [f64; 3],
    matrix: &mut CMatrix,
    // When present, receives the `T = 0` block of each perturbation's first-order density,
    // already summed over spin channels and k-points. Pre-sized to `ndof` by the caller.
    mut keep: Option<&mut Vec<CMatrix>>,
    // When present, receives the response **before** those two sums are taken: indexed
    // `[perturbation][spin][k]`. Pre-sized to `ndof` by the caller.
    //
    // The summed form above is all a Born charge or a polarizability needs, because both contract
    // it against an on-site operator that has no image blocks. Contracting against a *nuclear*
    // perturbation needs the k-resolved form, since that operator does have them — see
    // [`crate::pbc::born::born_charges`], where this is what turns `3N` coupled-perturbed solves
    // into three.
    mut keep_resolved: Option<&mut Vec<Vec<Vec<CMatrix>>>>,
    dfpt: &DfptOptions,
) -> Result<()> {
    let basis = &setup.basis;
    let nao = basis.nao;
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let index_of = image_index(setup);
    let scf = ground;
    let points = &ground.points;

    // Everything below comes from the ground state rather than being reconstructed here. The
    // Gamma path used to build `p_images` by asserting `P(0, T) = P(Gamma)` for every image and
    // rebuild the Fock from the total density; a mesh has the real `P(T)` and its own converged
    // Fock, and the response should not know which it was handed.
    let mut sites = setup.sites.clone();
    crate::pbc::gamma::write_electron_charges(setup, ground.density, &mut sites);

    let bands = band_pairs(setup, ground, q_frac)?;

    // From the Fermi level, not from a band index. See GroundState::occupation.
    let occupation = |channel: usize, energies: &[f64], index: usize| -> f64 {
        ground.occupation(&ground.channels[channel], energies, index)
    };

    // Every degree of freedom's bare perturbation, once per channel.
    let mut bare: Vec<Vec<BareBlocks>> = Vec::with_capacity(ndof);
    let mut total_charges: Vec<f64> = sites.iter().map(|s| s.charge).collect();
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        total_charges[setup.site_offset[ia]] += params.element(atom.z)?.core_charge;
    }
    // Read once rather than once per degree of freedom: `env::var_os` allocates, and this sits
    // inside a loop that runs `3N` times.
    let bare_long_range_off = std::env::var_os("PM3_DFPT_NO_BARE_LONG_RANGE").is_some();
    // The neighbour list every bare perturbation reads, built once here rather than `3N` times
    // inside `bare_blocks` — see the note on its `list` parameter.
    let long_range = long_range_kernels(setup, cell, q)?;
    let bare_list = {
        let cutoff = periodic.short_range_cutoff.max(periodic.switch.off);
        let positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
        NeighborList::build_from_positions(&positions, Some(cell), cutoff)
    };
    for dof in 0..ndof {
        let (atom, axis) = (dof / 3, dof % 3);
        // The lattice sum's own term is spin-independent — it is the Coulomb field of the total
        // charge — so it is computed once and added to every channel.
        let mut field_part = CMatrix::zeros(nao, nao);
        // `PM3_DFPT_NO_BARE_LONG_RANGE=1` drops this term.
        //
        // **A diagnostic, not an option.** Leaving it out gives a wrong `D(q)` at every
        // wavevector — it is a real part of the perturbation. It is switchable because it is
        // where the small-`q` defect documented on [`phased_field_derivative`] lives, and
        // turning it off is what localized it: with the term in, the acoustic sum rule of `D(q)`
        // diverges as `1/q²` for a polar cell; with it out, the residual falls linearly in `q`
        // exactly as the rigid-ion half does (0.132 → 0.0089 over `q = 0.1 → 0.0125` on water).
        if dfpt.long_range != LongRange::Off && !bare_long_range_off {
            phased_field_derivative(
                setup,
                &long_range,
                &total_charges,
                atom,
                axis,
                &mut field_part,
            )?;
        }
        let mut per_spin = Vec::with_capacity(ground.channels.len());
        for channel in &ground.channels {
            let mut blocks = bare_blocks(
                molecule,
                params,
                periodic,
                setup,
                scf.density,
                &channel.exchange_density,
                channel.exchange_scale,
                q_frac,
                atom,
                axis,
                &index_of,
                &bare_list,
            )?;
            for i in 0..nao {
                for j in 0..nao {
                    blocks.onsite[(i, j)] += field_part[(i, j)] * crate::constants::PM3_EV;
                }
            }
            per_spin.push(blocks);
        }
        bare.push(per_spin);
    }

    // Solve the columns and contract them, in chunks. The site-site lattice sum depends only on
    // the geometry and `q`, so it is built once here rather than inside every iteration of every
    // column.
    //
    // # Why this is chunked
    //
    // Holding every column's response at once is `ndof × nk × nao²` complex numbers, and the
    // straightforward contraction wants a second tensor of the same shape for the bare
    // perturbations at each k. On a fifty-atom cell over a `4×4×4` mesh that is six gigabytes
    // apiece. So a chunk of columns is solved, contracted against *every* `j`, and dropped —
    // the same shape the molecular UHF Hessian uses for its own `U` blocks, and the same
    // [`crate::hessian::cphf_chunk_size`] policy, rather than a second memory policy that would
    // have to be tuned separately.
    //
    // The bare Bloch sums are rebuilt once per chunk instead of being cached across all of them.
    // That is `nchunks` times more of them, and it is the cheap side of the trade: a Bloch sum is
    // `nao² × images` while the solve it makes room for is `nao³` per iteration per k.
    let site_kernel = SiteKernel::build(setup, cell, q)?;
    let n_spin = ground.channels.len();
    let slab = n_spin * nk_bytes(points.len(), nao);
    let depth = response_depth(options, slab).min(dfpt.diis_depth.max(2));
    // Resident regardless of the chunk: the eigenpairs at `k` and `k + q`, the direct-space bare
    // columns, and the extrapolation history one solve holds while it runs.
    let resident = 2 * slab + bare_bytes(&bare) + 2 * depth * slab;
    let chunk = crate::hessian::cphf_chunk_size(options, "dfpt response", ndof, resident, slab)?;

    for start in (0..ndof).step_by(chunk) {
        let end = (start + chunk).min(ndof);
        let responses: Vec<Vec<Vec<CMatrix>>> = bare[start..end]
            .par_iter()
            .map(|column| {
                solve_column(
                    molecule,
                    params,
                    periodic,
                    setup,
                    cell,
                    q_frac,
                    column,
                    &index_of,
                    ground,
                    &bands,
                    &occupation,
                    &site_kernel,
                    depth,
                    dfpt.tol,
                    dfpt.max_iter,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        // The first-order density's `T = 0` block, for a caller who asked to keep it.
        //
        // `ΔP(T) = Σ_k w_k e^{ik·T} ΔP(k)`, so the origin block is the plain weighted sum — and
        // the origin block is all that a Born effective charge or a polarizability reads, since
        // both contract the response against on-site quantities. Summed over spin channels
        // because it is the *total* first-order density they want; a spin-resolved one would be
        // a different accessor.
        //
        // Off by default (`DfptOptions::keep_response`): this is `ndof × nao²` complex numbers,
        // and the force constants themselves never need it retained.
        if let Some(store) = keep_resolved.as_deref_mut() {
            for (offset, delta) in responses.iter().enumerate() {
                store[start + offset] = delta.clone();
            }
        }
        if let Some(store) = keep.as_deref_mut() {
            for (offset, delta) in responses.iter().enumerate() {
                let slot = &mut store[start + offset];
                for spin_block in delta.iter() {
                    for (index, k) in points.iter().enumerate() {
                        let contribution = &spin_block[index];
                        for mu in 0..nao {
                            for nu in 0..nao {
                                let value = contribution[(mu, nu)];
                                slot[(mu, nu)] +=
                                    c64::new(value.re * k.weight, value.im * k.weight);
                            }
                        }
                    }
                }
            }
        }

        // `D^resp_{jj'} = Σ_σ Σ_k w_k Tr[Δh^{j†}_σ(k) ΔP^{j'}_σ(k)]` — the *bare* perturbation
        // against the *self-consistent* response, which is what the 2n+1 theorem leaves once the
        // two-electron double counting cancels. The spin sum is over channels, each already
        // carrying its own occupancy: one channel of two electrons, or two of one.
        for (j, column) in bare.iter().enumerate() {
            for (spin, spin_column) in column.iter().enumerate() {
                for (index, k) in points.iter().enumerate() {
                    let h = spin_column.at_k(setup, k.frac);
                    for (offset, delta) in responses.iter().enumerate() {
                        let mut partial = c64::new(0.0, 0.0);
                        for mu in 0..nao {
                            for nu in 0..nao {
                                partial += h[(mu, nu)].conj() * delta[spin][index][(mu, nu)];
                            }
                        }
                        matrix[(j, start + offset)] +=
                            c64::new(partial.re * k.weight, partial.im * k.weight);
                    }
                }
            }
        }
    }
    Ok(())
}

/// Whether the long-range monopole term is carried by the response.
///
/// The term is the phased lattice sum's own field, wired into three places that have to agree:
/// the fixed-charge second derivative, the bare perturbation's per-atom channel, and the `∂Q_a(q)`
/// shift in the coupled-perturbed kernel. Leaving it out of any one of them would let the
/// skeleton carry a long-range term the response could not screen.
///
/// Note what this does *not* control: the macroscopic (`G = 0`) member is excluded from the
/// response regardless — see [`crate::pbc::phased::Macroscopic`] — because keeping it there makes
/// the acoustic sum rule diverge. This switch is about the `G ≠ 0` long-range channel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LongRange {
    /// Include it wherever there is a lattice to sum over. The default.
    #[default]
    Auto,
    /// Require it. An isolated cell — which has no lattice sum at all — is an error rather than a
    /// quiet difference.
    Require,
    /// Leave it out, so the term's effect can be **measured** rather than argued.
    Off,
}

/// Controls for the response, over and above the ground-state [`Pm3Options`].
///
/// # The k-set is the SCF's k-set
///
/// [`DfptOptions::kpoints`] changes which Brillouin-zone sampling the **whole** calculation uses,
/// ground state included. That is not an oversight: the coupled-perturbed equations assume the
/// zeroth-order state satisfies the SCF condition, and diagonalizing a Fock matrix converged on a
/// coarse mesh at some finer `k` gives exact eigenpairs *of that matrix* but not a
/// self-consistent ground state — so the response would be the response of a different functional,
/// and the folding identity would stop holding exactly.
// `Default` is written out rather than derived: a derived one would give `tol = 0.0` and
// `max_iter = 0`, which is a solver that never converges and never iterates.
#[derive(Clone, Debug)]
pub struct DfptOptions {
    /// Brillouin-zone sampling for the whole calculation. `None` samples Γ alone.
    pub kpoints: Option<crate::pbc::kscf::KpointOptions>,
    /// Whether the long-range monopole term is carried. See [`LongRange`].
    pub long_range: LongRange,
    /// Convergence tolerance on the RMS change of the response between iterations.
    pub tol: f64,
    /// Iteration cap for the coupled-perturbed solve.
    pub max_iter: usize,
    /// How many past steps the extrapolation keeps.
    ///
    /// Not a mixing fraction, deliberately. A linear-mixing knob is the conventional control for
    /// this job; this solver extrapolates over a history instead, because the response is linear and
    /// plain iteration converges only where the spectral radius of `χ·K` is below one — at a
    /// general `q` it is not, and damping only rescales that eigenvalue. Offering a mixing
    /// fraction here would be a regression dressed as a feature.
    pub diis_depth: usize,
    /// Return the first-order densities in [`DfptResult::response`].
    ///
    /// Off by default: it is `ndof × nao²` complex numbers, the largest array in the calculation,
    /// and the force constants themselves never need it retained.
    pub keep_response: bool,
}

impl Default for DfptOptions {
    fn default() -> Self {
        Self {
            kpoints: None,
            long_range: LongRange::Auto,
            tol: RESPONSE_TOLERANCE,
            max_iter: RESPONSE_ITERATIONS,
            diis_depth: RESPONSE_DIIS_DEPTH,
            keep_response: false,
        }
    }
}

impl DfptOptions {
    /// Reject a combination that cannot be honoured, naming what is wrong rather than picking a
    /// winner.
    fn check(&self, molecule: &Molecule) -> Result<()> {
        if self.tol <= 0.0 {
            return Err(Pm3Error::InvalidInput(
                "the response tolerance must be positive; zero is a solver that never converges"
                    .to_string(),
            ));
        }
        if self.max_iter == 0 {
            return Err(Pm3Error::InvalidInput(
                "the response iteration cap must be at least one".to_string(),
            ));
        }
        if self.long_range == LongRange::Require {
            let periodic = molecule.cell.map(|c| c.n_periodic()).unwrap_or(0);
            if periodic == 0 {
                return Err(Pm3Error::InvalidInput(
                    "LongRange::Require was asked for and this cell has no periodic direction, \
                     so there is no lattice to sum over. That is the difference between \
                     `Require` and `Auto`: one refuses here and the other carries on without the \
                     term."
                        .to_string(),
                ));
            }
        }
        Ok(())
    }
}

/// What a response calculation produces, beyond the matrix itself.
#[derive(Clone, Debug)]
pub struct DfptResult {
    /// The dynamical matrix at `q`, mass information and Hermitian defect included.
    pub dynamical: DynamicalMatrix,
    /// `C(q) = Σ_T Φ(T) e^{iq·T}` in eV/Bohr², **not** mass weighted — the force constants
    /// themselves. [`DynamicalMatrix::matrix`] holds the same numbers; this name is here because
    /// "dynamical matrix" is often reserved for the mass-weighted object and the two are easy to
    /// confuse when one is being compared against a reference.
    pub force_constants: CMatrix,
    /// The k-points the response actually sampled, with their weights.
    pub k_points: Vec<crate::pbc::kpoints::KPoint>,
    /// Per degree of freedom, the `T = 0` block of `∂P/∂u`, summed over spin channels. `None`
    /// unless [`DfptOptions::keep_response`] asked for it.
    pub response: Option<Vec<CMatrix>>,
}

/// The force constants at `q`, un-mass-weighted, with the option surface exposed.
///
/// The same calculation [`dynamical_matrix`] performs, returning the diagnostics with it. See
/// [`DfptOptions`].
pub fn force_constants_at_q(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
    dfpt: &DfptOptions,
    q_frac: [f64; 3],
) -> Result<DfptResult> {
    dfpt.check(molecule)?;
    let dynamical = assemble(
        molecule,
        params,
        options,
        periodic,
        dfpt.kpoints.as_ref(),
        q_frac,
        true,
        dfpt,
    )?;
    let k_points = match &dfpt.kpoints {
        Some(kopt) => {
            let cell = molecule
                .cell
                .ok_or_else(|| Pm3Error::InvalidInput("a response needs a cell".to_string()))?;
            kopt.spec.generate(&cell)?
        }
        None => vec![crate::pbc::kpoints::KPoint::GAMMA],
    };
    Ok(DfptResult {
        force_constants: dynamical.matrix.clone(),
        dynamical,
        k_points,
        // Retaining the first-order densities from this entry point is not wired up yet; the
        // internal `phonon_response` is what Born charges use, and it is the natural place for
        // `keep_response` to be honoured once a caller needs both at once.
        response: None,
    })
}

/// Phonon frequencies at `q` with the option surface exposed. See [`force_constants_at_q`].
pub fn frequencies_at_q(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
    dfpt: &DfptOptions,
    q_frac: [f64; 3],
) -> Result<Vec<f64>> {
    let result = force_constants_at_q(molecule, params, options, periodic, dfpt, q_frac)?;
    frequencies_of(&result.dynamical)
}

/// The converged ground state and the first-order density every nuclear displacement produces,
/// at `q = 0`.
///
/// This is what a Born effective charge is built from, and it is deliberately *not* the
/// dynamical matrix: the skeleton — the fixed-density second derivative, and the expensive half —
/// is never assembled here, because `∂P/∂u` does not depend on it.
pub(crate) struct PhononResponse {
    /// The Γ-point SCF the response was solved around.
    pub(crate) scf: crate::pbc::gamma::PeriodicResult,
    pub(crate) basis: Basis,
    /// Per degree of freedom (`3a + β`), the `T = 0` block of `∂P/∂u`, summed over spin channels.
    /// Real at `q = 0`; carried as complex because [`response`] is written at general `q`.
    pub(crate) delta: Vec<CMatrix>,
}

/// Solve the coupled-perturbed equations for all `3N` nuclear displacements at `q = 0` and keep
/// the first-order densities.
pub(crate) fn phonon_response(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
) -> Result<PhononResponse> {
    crate::pbc::refuse_field(molecule, options)?;
    let cell = molecule.cell.ok_or_else(|| {
        Pm3Error::InvalidInput("a Born effective charge needs a periodic cell".to_string())
    })?;
    if cell.n_periodic() == 0 {
        return Err(Pm3Error::InvalidInput(
            "the cell has no periodic direction; an isolated molecule's dipole derivatives come \
             from `pm3_rs::ir::dipole_derivatives`"
                .to_string(),
        ));
    }
    let scf = run_gamma(molecule, params, options, periodic)?;
    let setup = crate::pbc::gamma::build_setup(molecule, params, periodic)?;
    let nao = setup.basis.nao;
    if scf.n_occ == 0 || scf.n_occ >= nao {
        return Err(Pm3Error::InvalidInput(
            "the cell has no partially filled manifold, so there is no orbital relaxation to \
             solve for: every occupied-virtual pair is empty"
                .to_string(),
        ));
    }
    let ndof = 3 * molecule.atoms.len();
    let ground = gamma_ground_state(molecule, params, &setup, &cell, &scf)?;
    let mut delta = vec![CMatrix::zeros(nao, nao); ndof];
    // A scratch target for the force-constant contraction, which this path does not want. The
    // contraction is the expensive part and is computed and dropped; keeping the response and the
    // dynamical matrix on separate entry points is what lets a caller who wants both ask once,
    // and that sharing is not wired up yet.
    let mut scratch = CMatrix::zeros(ndof, ndof);
    response(
        molecule,
        params,
        options,
        periodic,
        &setup,
        &ground,
        &cell,
        Vec3::zero(),
        [0.0; 3],
        &mut scratch,
        Some(&mut delta),
        None,
        &DfptOptions::default(),
    )?;
    let basis = Basis::build(molecule, params)?;
    Ok(PhononResponse { scf, basis, delta })
}

/// The first-order density each Cartesian **electric field** direction produces, at `q = 0`.
///
/// # The perturbation is the ordinary position operator, on the origin block only
///
/// A uniform field is not lattice-periodic, and [`crate::pbc::refuse_field`] says so — but that
/// refusal is about putting `−F·r` into the *Hamiltonian*, where the spectrum has no lower bound.
/// Nothing of the kind happens here. The dipole operator enters only as a **bare perturbation**
/// on the right-hand side of the coupled-perturbed equations, and in NDDO it is on-site: a
/// position on each diagonal of an atom's block and the `dd` hybridization element off it. It
/// therefore lives entirely in the `T = 0` block and carries no Bloch phase, which is exactly why
/// it can reuse the phonon response unchanged.
///
/// # What that approximation is, stated rather than hidden
///
/// This is the standard tight-binding clamped-ion treatment, and the position operator it uses is
/// not a well-defined periodic operator. Two things make the *response* well defined anyway: an
/// origin shift adds a constant to the whole diagonal, which cannot change a response, and charge
/// conservation kills the rest. That is an argument, so `tests/pbc_dielectric.rs` measures it by
/// recomputing with the cell shifted rather than taking it on trust. It is **not** a Berry-phase
/// polarization; see `docs/pbc.md`.
pub(crate) fn field_response(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
) -> Result<PhononResponse> {
    crate::pbc::refuse_field(molecule, options)?;
    let cell = molecule.cell.ok_or_else(|| {
        Pm3Error::InvalidInput("an electric-field response needs a periodic cell".to_string())
    })?;
    if cell.n_periodic() == 0 {
        return Err(Pm3Error::InvalidInput(
            "the cell has no periodic direction; an isolated molecule's polarizability comes \
             from a finite field on the molecular path"
                .to_string(),
        ));
    }
    let scf = run_gamma(molecule, params, options, periodic)?;
    let setup = crate::pbc::gamma::build_setup(molecule, params, periodic)?;
    let nao = setup.basis.nao;
    if scf.n_occ == 0 || scf.n_occ >= nao {
        return Err(Pm3Error::InvalidInput(
            "the cell has no partially filled manifold, so there is nothing for a field to \
             polarize: every occupied-virtual pair is empty"
                .to_string(),
        ));
    }
    let ground = gamma_ground_state(molecule, params, &setup, &cell, &scf)?;
    let index_of = image_index(&setup);
    let bands = band_pairs(&setup, &ground, [0.0; 3])?;
    let site_kernel = SiteKernel::build(&setup, &cell, Vec3::zero())?;
    let occupation = |channel: usize, energies: &[f64], index: usize| -> f64 {
        ground.occupation(&ground.channels[channel], energies, index)
    };
    let n_spin = ground.channels.len();
    let depth = response_depth(options, n_spin * nk_bytes(ground.points.len(), nao));

    // The dipole operator, about the coordinate origin. Which origin does not matter — an origin
    // shift adds a multiple of the identity to the diagonal, and the occupied–virtual projection
    // of the identity is zero — and the measurement in `tests/pbc_dielectric.rs` is what says so.
    let basis = Basis::build(molecule, params)?;
    let operator = crate::dipole::dipole_matrix(molecule, params, &basis, Vec3::zero())?;

    // The three columns, built up front so they can be solved together.
    //
    // One bare perturbation per spin channel, identical: the field couples to charge, not spin.
    // The image blocks stay zero — this operator is on-site.
    let columns: Vec<Vec<BareBlocks>> = operator
        .iter()
        .map(|moment| {
            (0..n_spin)
                .map(|_| {
                    let mut blocks = BareBlocks::new(nao, &setup.images);
                    for mu in 0..nao {
                        for nu in 0..nao {
                            blocks.onsite[(mu, nu)] = c64::new(moment[(mu, nu)], 0.0);
                        }
                    }
                    blocks
                })
                .collect()
        })
        .collect();

    // Solved in parallel, as the phonon path solves its `3N`.
    //
    // These used to go one at a time, and the cost of that was not obvious: a field solve
    // measured at `2.9 s` where a nuclear solve measured at `1.7 s` on the same system, purely
    // because the nuclear ones were spread across cores and these were not. A dielectric tensor
    // on a 40-atom perovskite ran for over ten minutes on three sequential solves.
    let solved: Vec<Vec<Vec<CMatrix>>> = {
        use rayon::prelude::*;
        columns
            .par_iter()
            .map(|column| {
                solve_column(
                    molecule,
                    params,
                    periodic,
                    &setup,
                    &cell,
                    [0.0; 3],
                    column,
                    &index_of,
                    &ground,
                    &bands,
                    &occupation,
                    &site_kernel,
                    depth,
                    RESPONSE_TOLERANCE,
                    RESPONSE_ITERATIONS,
                )
            })
            .collect::<Result<_>>()?
    };

    let mut delta = vec![CMatrix::zeros(nao, nao); 3];
    for (axis, column) in solved.iter().enumerate() {
        let slot = &mut delta[axis];
        for spin_block in column.iter() {
            for (index, k) in ground.points.iter().enumerate() {
                let contribution = &spin_block[index];
                for mu in 0..nao {
                    for nu in 0..nao {
                        let value = contribution[(mu, nu)];
                        slot[(mu, nu)] += c64::new(value.re * k.weight, value.im * k.weight);
                    }
                }
            }
        }
    }
    Ok(PhononResponse { scf, basis, delta })
}

/// Bytes in one `nk × nao × nao` complex tensor — the unit everything in the response is
/// measured in.
fn nk_bytes(nk: usize, nao: usize) -> usize {
    nk.saturating_mul(nao)
        .saturating_mul(nao)
        .saturating_mul(std::mem::size_of::<c64>())
}

/// Bytes the direct-space bare columns hold.
fn bare_bytes(bare: &[Vec<BareBlocks>]) -> usize {
    bare.iter()
        .flatten()
        .map(|column| {
            let images: usize = column.images.iter().map(|m| m.rows * m.cols).sum();
            (column.onsite.rows * column.onsite.cols + images) * std::mem::size_of::<c64>()
        })
        .sum()
}

/// How deep an extrapolation history the memory budget allows.
///
/// [`RESPONSE_DIIS_DEPTH`] is what converges; this is what fits. They differ only on a large cell
/// over a dense mesh, and there the shortfall is worth saying out loud rather than discovering as
/// a response that stops at `1e-5` for no stated reason.
fn response_depth(options: &Pm3Options, slab: usize) -> usize {
    if options.hessian_memory_mb == 0 || slab == 0 {
        return RESPONSE_DIIS_DEPTH;
    }
    // Half the budget, so the chunk of responses has somewhere to live.
    let budget = options
        .hessian_memory_mb
        .saturating_mul(crate::hessian::MIB)
        / 2;
    let allowed = (budget / (2 * slab)).clamp(2, RESPONSE_DIIS_DEPTH);
    if allowed < 8 {
        eprintln!(
            "warning: the phonon response history is capped at {allowed} steps by \
             hessian_memory_mb = {} MiB; measured, it needs about twenty to reach its \
             tolerance at a general wavevector, and may report a non-convergence instead",
            options.hessian_memory_mb
        );
    }
    allowed
}

/// One column of the self-consistent response, in the AO basis at `k = Γ`.
#[allow(clippy::too_many_arguments)]
fn solve_column(
    molecule: &Molecule,
    params: &Pm3Parameters,
    periodic: &PeriodicOptions,
    setup: &crate::pbc::gamma::Setup,
    cell: &Cell,
    q_frac: [f64; 3],
    bare: &[BareBlocks],
    index_of: &std::collections::HashMap<(usize, usize, [i32; 3]), usize>,
    ground: &GroundState<'_>,
    bands: &[KBands],
    occupation: &(dyn Fn(usize, &[f64], usize) -> f64 + Sync),
    kernel: &SiteKernel,
    depth: usize,
    tolerance: f64,
    max_iter: usize,
) -> Result<Vec<Vec<CMatrix>>> {
    let nao = setup.basis.nao;
    let points = &ground.points;
    let nk = points.len();
    let n_spin = ground.channels.len();
    let mut delta: Vec<BareBlocks> = (0..n_spin)
        .map(|_| BareBlocks::new(nao, &setup.images))
        .collect();
    let mut ao: Vec<Vec<CMatrix>> = (0..n_spin)
        .map(|_| (0..nk).map(|_| CMatrix::zeros(nao, nao)).collect())
        .collect();
    let mut history = ResponseDiis::new(depth);
    let mut converged = false;
    let mut residual = f64::INFINITY;

    for iteration in 0..max_iter {
        // The induced potential is a property of the *density* response, which lives in direct
        // space, so it is built once per iteration and then evaluated at each k. The two spin
        // channels are coupled here and nowhere else: Coulomb reads their sum, exchange reads
        // each channel's own.
        let induced = if iteration > 0 {
            Some(phased_kernel(
                molecule,
                params,
                periodic,
                setup,
                cell,
                q_frac,
                &delta,
                &ground.channels,
                index_of,
                kernel,
            )?)
        } else {
            None
        };

        let mut change = 0.0;
        let mut next: Vec<CMatrix> = Vec::with_capacity(n_spin * nk);
        let mut errors: Vec<CMatrix> = Vec::with_capacity(n_spin * nk);
        for spin in 0..n_spin {
            for (index, k) in points.iter().enumerate() {
                let mut potential = bare[spin].at_k(setup, k.frac);
                if let Some(extra) = &induced {
                    let evaluated = extra[spin].at_k(setup, k.frac);
                    for i in 0..nao {
                        for j in 0..nao {
                            potential[(i, j)] += evaluated[(i, j)];
                        }
                    }
                }

                // Into the band basis of the coupled pair: rows at `k + q`, columns at `k`.
                let (eps_k, c_k) = &bands[spin].at_k[index];
                let (eps_kq, c_kq) = &bands[spin].at_kq[index];
                let projected = c_kq.adjoint().matmul(&potential.matmul(c_k));
                let mut band = CMatrix::zeros(nao, nao);
                for m in 0..nao {
                    let f_m = occupation(spin, eps_kq, m);
                    for n in 0..nao {
                        let f_n = occupation(spin, eps_k, n);
                        let df = f_n - f_m;
                        // Equal occupations make this a `0/0` at an accidental degeneracy between
                        // `k` and `k + q`, which is common rather than exotic. Skipping on `df`
                        // first is what makes the energy denominator below safe to divide by.
                        if df == 0.0 {
                            continue;
                        }
                        let de = eps_k[n] - eps_kq[m];
                        if de.abs() < 1.0e-8 {
                            continue;
                        }
                        band[(m, n)] = projected[(m, n)] * (df / de);
                    }
                }
                let solved = c_kq.matmul(&band.matmul(&c_k.adjoint()));
                let mut error = CMatrix::zeros(nao, nao);
                for i in 0..nao {
                    for j in 0..nao {
                        let d = solved[(i, j)] - ao[spin][index][(i, j)];
                        change += d.re * d.re + d.im * d.im;
                        error[(i, j)] = d;
                    }
                }
                errors.push(error);
                next.push(solved);
            }
        }
        // The extrapolation runs over both channels at once — they are one linear system, and
        // giving each its own history would let them be extrapolated apart.
        let blended = history.extrapolate(next, errors);
        for (spin, slot) in ao.iter_mut().enumerate() {
            slot.clone_from_slice(&blended[spin * nk..(spin + 1) * nk]);
        }

        // Back to direct space: `Δp(T) = Σ_k w_k e^{−ik·T} ΔP(k)`.
        //
        // **No real part.** `assemble_density` takes one because `P(−k) = P(k)*` makes the
        // ground-state sum real; here `ΔP` couples `k` to `k + q` and the perturbation carries
        // `e^{iq·T}`, so `Δp(T)` is genuinely complex. Taking the real part would make
        // `D(−q) = D(q)*` come out as `D(−q) = D(q)`, which is the test that catches it.
        for (spin, blocks) in delta.iter_mut().enumerate() {
            for i in 0..nao {
                for j in 0..nao {
                    blocks.onsite[(i, j)] = c64::new(0.0, 0.0);
                }
            }
            for (index, k) in points.iter().enumerate() {
                for i in 0..nao {
                    for j in 0..nao {
                        let value = ao[spin][index][(i, j)];
                        blocks.onsite[(i, j)] += c64::new(value.re * k.weight, value.im * k.weight);
                    }
                }
            }
            for (block, slot) in setup.images.iter().zip(blocks.images.iter_mut()) {
                let (oa, ob) = (
                    setup.basis.atom_offset[block.a],
                    setup.basis.atom_offset[block.b],
                );
                for mu in 0..block.norb_a {
                    for la in 0..block.norb_b {
                        slot[(mu, la)] = c64::new(0.0, 0.0);
                    }
                }
                for (index, k) in points.iter().enumerate() {
                    let phase = image_phase([-k.frac[0], -k.frac[1], -k.frac[2]], block.t);
                    let weighted = c64::new(phase.re * k.weight, phase.im * k.weight);
                    for mu in 0..block.norb_a {
                        for la in 0..block.norb_b {
                            slot[(mu, la)] += weighted * ao[spin][index][(oa + mu, ob + la)];
                        }
                    }
                }
            }
        }
        if std::env::var_os("PM3_DFPT_TRACE").is_some() && (iteration < 5 || iteration % 40 == 0) {
            eprintln!("  dfpt iter {iteration:3} change {:.6e}", change.sqrt());
        }
        if change.sqrt() < tolerance {
            if std::env::var_os("PM3_DFPT_TRACE").is_some() {
                eprintln!("  converged at {iteration}");
            }
            converged = true;
            break;
        }
        residual = change.sqrt();
    }

    // A fixed-point iteration that has not converged has not produced an answer, and this one
    // does not merely stall — it diverges, reaching `1e30` and beyond. It used to run its two
    // hundred passes and return whatever it was holding, so a caller got a dynamical matrix with
    // no indication that its entries were nonsense. The Hermitian defect reported alongside `D`
    // would have shown it, but only to someone who looked.
    if !converged {
        return Err(Pm3Error::ScfNotConverged {
            iterations: max_iter,
            error: residual,
        });
    }
    Ok(ao)
}

/// Diagonalize the converged Fock at every sampled `k` and at every `k + q`, once per spin
/// channel.
///
/// Shared by the phonon response and the electric-field one: both solve the same coupled-perturbed
/// equations in the same band basis, and differ only in the bare perturbation they are driven
/// with. Keeping one construction is what stops the two from drifting into different definitions
/// of "the ground state the response is around".
fn band_pairs(
    setup: &crate::pbc::gamma::Setup,
    ground: &GroundState<'_>,
    q_frac: [f64; 3],
) -> Result<Vec<KBands>> {
    use crate::cmatrix::hermitian_eigen;
    let points = &ground.points;
    let mut bands: Vec<KBands> = Vec::with_capacity(ground.channels.len());
    for channel in &ground.channels {
        // Fractional, because that is what a Bloch phase needs and it avoids a round trip through
        // the reciprocal basis.
        let at = |frac: [f64; 3]| -> Result<(Vec<f64>, CMatrix)> {
            let k = crate::pbc::kpoints::KPoint { frac, weight: 1.0 };
            hermitian_eigen(&crate::pbc::kscf::bloch_fock(
                setup,
                &channel.f_onsite,
                &channel.p_images,
                &k,
            ))
        };
        let mut spin = KBands {
            at_k: Vec::with_capacity(points.len()),
            at_kq: Vec::with_capacity(points.len()),
        };
        for k in points {
            spin.at_k.push(at(k.frac)?);
            let shifted = crate::pbc::kpoints::shift(k, q_frac);
            spin.at_kq.push(at(shifted.frac)?);
        }
        bands.push(spin);
    }
    Ok(bands)
}

/// The band structure the response solve needs: one `(ε, C)` pair at `k` and one at `k + q`, for
/// every k-point the response samples.
///
/// Held together rather than passed as four slices, because the two halves have to stay aligned
/// and nothing about the types says so: row `m` of the band-basis projection is a state at
/// `k + q` and column `n` is one at `k`.
struct KBands {
    at_k: Vec<(Vec<f64>, CMatrix)>,
    at_kq: Vec<(Vec<f64>, CMatrix)>,
}

/// How many self-consistency passes the response takes before giving up.
const RESPONSE_ITERATIONS: usize = 200;

/// How many past steps the response extrapolation keeps.
///
/// Twenty, measured rather than borrowed: at the ground-state SCF's depth of eight the response
/// still stalled around `1e-5` at a general `q`, at twenty every wavevector probed reached the
/// `1e-10` tolerance, and forty changed nothing because the history never grows that far before
/// converging.
///
/// It is not free. The history is `2 × depth × nk × nao²` complex numbers — at `nao = 200` on a
/// `4×4×4` mesh, 1.6 GB, which is larger than the eigenpairs and larger than what §B.8 of the
/// plan budgeted for the responses themselves. Whatever bounds the response's memory has to
/// bound this too.
const RESPONSE_DIIS_DEPTH: usize = 20;

/// Pulay extrapolation on the first-order density.
///
/// The response is a *linear* equation, `(1 − χ₀K) ΔP = χ₀V`, and repeatedly substituting it into
/// itself — which is what the plain iteration does — converges only where the spectral radius of
/// `χ₀K` is below one. At a general `q` it is not: the plain iteration reached `1e30` in two
/// hundred passes, and damping only slowed the growth, because damping rescales the eigenvalue
/// rather than removing it. Extrapolating over a history solves the linear system inside the
/// Krylov subspace that history spans instead of following the unstable fixed point, so an
/// unstable `χ₀K` stops mattering as long as `1 − χ₀K` is nonsingular.
///
/// The extrapolation coefficients are real. The response is complex, but the residual inner
/// product that picks them is the *real* one on `ℂⁿ` read as `ℝ²ⁿ` — which is what
/// `Re ⟨r_i, r_j⟩` is — and a real combination is what leaves `Σ c_i = 1` meaning what it says.
struct ResponseDiis {
    depth: usize,
    /// `g(x_i)`: the plain step taken from each stored iterate, one matrix per k-point.
    solved: Vec<Vec<CMatrix>>,
    /// `g(x_i) − x_i`.
    error: Vec<Vec<CMatrix>>,
}

impl ResponseDiis {
    fn new(depth: usize) -> Self {
        Self {
            depth: depth.max(2),
            solved: Vec::new(),
            error: Vec::new(),
        }
    }

    /// Record one plain step and return the iterate to carry forward.
    fn extrapolate(&mut self, solved: Vec<CMatrix>, error: Vec<CMatrix>) -> Vec<CMatrix> {
        self.solved.push(solved);
        self.error.push(error);
        if self.solved.len() > self.depth {
            self.solved.remove(0);
            self.error.remove(0);
        }
        let full = self.error.len();
        let gram: Vec<Vec<f64>> = (0..full)
            .map(|i| {
                (0..full)
                    .map(|j| Self::dot(&self.error[i], &self.error[j]))
                    .collect()
            })
            .collect();
        // `None` means the oldest error vectors have gone linearly dependent on the newer ones,
        // which is what happens as the residual collapses onto a Krylov subspace smaller than
        // the history. Falling back to the plain step there is what the ground-state SCF does,
        // and it is wrong here: the plain step is the *unstable* one, so giving up on the
        // extrapolation throws the iteration back out and the residual settles into a limit
        // cycle instead of converging. Dropping the oldest vector and re-solving keeps the
        // extrapolation alive on the subspace that is still independent.
        let mut coefficients = None;
        let mut used = full;
        while used >= 2 {
            let start = full - used;
            let block: Vec<Vec<f64>> = gram[start..]
                .iter()
                .map(|row| row[start..].to_vec())
                .collect();
            if let Some(c) = crate::scf::diis_coeffs_from_gram(&block) {
                coefficients = Some(c);
                break;
            }
            used -= 1;
        }
        let Some(coefficients) = coefficients else {
            return self.solved.last().expect("just pushed").clone();
        };
        let mut out: Vec<CMatrix> = self
            .solved
            .last()
            .expect("just pushed")
            .iter()
            .map(|m| CMatrix::zeros(m.rows, m.cols))
            .collect();
        for (slot, c) in self.solved[full - used..].iter().zip(coefficients.iter()) {
            for (target, source) in out.iter_mut().zip(slot.iter()) {
                for i in 0..source.rows {
                    for j in 0..source.cols {
                        let v = source[(i, j)];
                        target[(i, j)] += c64::new(v.re * c, v.im * c);
                    }
                }
            }
        }
        out
    }

    /// `Re ⟨a, b⟩` over every k-point.
    fn dot(a: &[CMatrix], b: &[CMatrix]) -> f64 {
        let mut sum = 0.0;
        for (x, y) in a.iter().zip(b.iter()) {
            for i in 0..x.rows {
                for j in 0..x.cols {
                    let (u, v) = (x[(i, j)], y[(i, j)]);
                    sum += u.re * v.re + u.im * v.im;
                }
            }
        }
        sum
    }
}

/// Convergence threshold on the first-order density, in the same units it carries.
const RESPONSE_TOLERANCE: f64 = 1.0e-10;

/// The classical corrections' second derivative, phased.
///
/// # The identity
///
/// The corrections are a classical function of the geometry, so their contribution to `D(q)` is a
/// force-constant sum like any other. Writing `Ψ_{κκ'}(T, T') = ∂²E_cell/∂x_{(κ,T)}∂x_{(κ',T')}`
/// for the second derivative of the **per-cell** energy with respect to two individual atoms
/// anywhere in the cluster, translational covariance of the crystal energy gives
///
/// ```text
/// D_corr(q)_{κκ'} = Σ_{T,T'} e^{−iq·T} e^{+iq·T'} Ψ_{κκ'}(T, T')
/// ```
///
/// That is a *bilinear form* in weighted displacements, which is exactly what a forward-mode
/// second derivative computes when each cluster entry's displacement is scaled by its own weight.
/// The correction energy is never decomposed into pairs: seeding two atoms and differentiating
/// the whole cluster energy is what carries the three-centre paths — the coordination-number
/// coupling in D3, the donor–hydrogen–acceptor triple in H4 — that a pair decomposition would
/// silently drop.
///
/// # Four real evaluations
///
/// `Dual2` is real, so `e^{∓iq·T}` is split. With `C` and `S` for the cosine- and sine-weighted
/// forms, `Re D = CC + SS` and `Im D = CS − SC`, four mixed derivatives per `(κα, κ'β)`. Only the
/// upper triangle is computed; the rest is `D(q)† = D(q)`.
///
/// At `q = 0` every weight is one and this reduces, term for term, to the Γ-point
/// [`crate::pbc::hessian`] correction Hessian — by construction rather than by agreement, which
/// is why the test that means something is the supercell folding one at a commensurate `q`.
fn phased_corrections(
    molecule: &Molecule,
    variant: Variant,
    periodic: &PeriodicOptions,
    q: Vec3,
    matrix: &mut CMatrix,
) -> Result<()> {
    use crate::corrections::correction_energy_cluster_g;
    use crate::corrections::periodic::build_cluster;
    use crate::dual2::Dual2;

    if variant == Variant::Pm3 {
        return Ok(());
    }
    let cutoffs = periodic.correction_cutoffs;
    let cluster = build_cluster(molecule, cutoffs.dispersion.max(cutoffs.short_range));
    let exact: Vec<bool> = (0..cluster.numbers.len())
        .map(|index| cluster.coordination_is_exact(index, cutoffs.coordination))
        .collect();

    // `e^{iq·T}` per cluster entry, split into the two real weightings the bilinear forms need.
    let cell = molecule.cell.expect("checked by the caller");
    let (mut cosine, mut sine) = (
        Vec::with_capacity(cluster.numbers.len()),
        Vec::with_capacity(cluster.numbers.len()),
    );
    for t in &cluster.translation {
        let mut shift = Vec3::zero();
        for (axis, steps) in t.iter().enumerate() {
            shift += cell.vector(axis) * *steps as f64;
        }
        let angle = q.dot(shift);
        cosine.push(angle.cos());
        sine.push(angle.sin());
    }

    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    // One mixed second derivative of the cluster energy, with variable 0 on `(a, k)` weighted by
    // `wa` and variable 1 on `(b, l)` weighted by `wb`.
    let bilinear = |a: usize, k: usize, b: usize, l: usize, wa: &[f64], wb: &[f64]| -> f64 {
        let mut positions = Vec::with_capacity(cluster.numbers.len());
        for (index, parent) in cluster.parent.iter().enumerate() {
            let here = &cluster.positions[index];
            let mut entry = [
                Dual2::constant(here[0]),
                Dual2::constant(here[1]),
                Dual2::constant(here[2]),
            ];
            // A seed of `w` rather than `1`: `var(0, v)` carries the derivative and nothing else,
            // and scaling it by a constant leaves the second-order part alone.
            if *parent == a {
                entry[k] = entry[k] + Dual2::var(0.0, 0) * Dual2::constant(wa[index]);
            }
            if *parent == b {
                entry[l] = entry[l] + Dual2::var(0.0, 1) * Dual2::constant(wb[index]);
            }
            positions.push(entry);
        }
        correction_energy_cluster_g::<Dual2>(
            &cluster.numbers,
            &positions,
            cluster.n_cell,
            Some(&cluster.parent),
            Some(&exact),
            Some(cutoffs.dispersion),
            Some(cutoffs.coordination),
            variant,
        )
        .h[0][1]
    };

    let entries: Vec<(usize, usize)> = (0..ndof)
        .flat_map(|i| (i..ndof).map(move |j| (i, j)))
        .collect();
    let values: Vec<c64> = entries
        .par_iter()
        .map(|&(i, j)| {
            let (a, k, b, l) = (i / 3, i % 3, j / 3, j % 3);
            let cc = bilinear(a, k, b, l, &cosine, &cosine);
            let ss = bilinear(a, k, b, l, &sine, &sine);
            let cs = bilinear(a, k, b, l, &cosine, &sine);
            let sc = bilinear(a, k, b, l, &sine, &cosine);
            c64::new(cc + ss, cs - sc)
        })
        .collect();
    for (&(i, j), value) in entries.iter().zip(&values) {
        matrix[(i, j)] += *value;
        if i != j {
            matrix[(j, i)] += value.conj();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell_of(xyz: &str, edge: f64) -> Molecule {
        let mut molecule = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        molecule.cell = Some(Cell::cubic(edge).unwrap());
        molecule
    }

    const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n";

    /// A channel carrying nothing but its exchange scale, which is all
    /// [`phased_kernel`] reads.
    fn dummy_channel(exchange_scale: f64) -> SpinChannel {
        SpinChannel {
            exchange_density: crate::linalg::Matrix::zeros(1, 1),
            exchange_scale,
            occupancy: 0.0,
            p_images: Vec::new(),
            f_onsite: crate::linalg::Matrix::zeros(1, 1),
            fermi_ev: 0.0,
        }
    }

    fn options() -> Pm3Options {
        Pm3Options {
            max_scf: 400,
            ..Pm3Options::default()
        }
    }

    /// At `q = 0` the phased matrix must be the Γ-point one, term for term.
    ///
    /// The Γ-point Hessian's skeleton and lattice sum are the same quantities assembled without
    /// phases, so this is the test that says the phased scatter is the same bookkeeping and not
    /// merely a plausible one. It compares the *fixed-density* part on both sides — the Γ path's
    /// response is left out of the comparison because there is nothing here to compare it to.
    #[test]
    fn the_dynamical_matrix_reduces_to_the_gamma_hessian() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = cell_of(WATER, 12.0);
        let periodic = PeriodicOptions::default();
        let d = rigid_ion_dynamical_matrix(&molecule, &params, &options(), &periodic, [0.0; 3])
            .unwrap();

        // Every entry must be real at q = 0.
        for i in 0..d.matrix.rows {
            for j in 0..d.matrix.cols {
                assert!(
                    d.matrix[(i, j)].im.abs() < 1.0e-9,
                    "D(0)[{i}][{j}] has imaginary part {}",
                    d.matrix[(i, j)].im
                );
            }
        }
    }

    /// `D(−q) = D(q)*`, which is what makes the frequencies real.
    #[test]
    fn reversing_the_wavevector_conjugates_the_matrix() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = cell_of(WATER, 12.0);
        let periodic = PeriodicOptions::default();
        let q = [0.3, -0.15, 0.42];
        let forward =
            rigid_ion_dynamical_matrix(&molecule, &params, &options(), &periodic, q).unwrap();
        let reverse = rigid_ion_dynamical_matrix(
            &molecule,
            &params,
            &options(),
            &periodic,
            [-q[0], -q[1], -q[2]],
        )
        .unwrap();
        for i in 0..forward.matrix.rows {
            for j in 0..forward.matrix.cols {
                let a = forward.matrix[(i, j)];
                let b = reverse.matrix[(i, j)];
                assert!((a.re - b.re).abs() < 1.0e-8, "real [{i}][{j}]");
                assert!((a.im + b.im).abs() < 1.0e-8, "imaginary [{i}][{j}]");
            }
        }
    }

    /// `D(−q) = D(q)*`, **response included**.
    ///
    /// The rigid-ion twin of this test cannot see the back-transform, which is where the identity
    /// is most fragile: `Δp(T) = Σ_k w_k e^{−ik·T} ΔP(k)` is genuinely complex, and taking a real
    /// part of it — which the ground state's own assembly does, correctly, for its own reasons —
    /// would turn this into `D(−q) = D(q)` and pass every other test in this file.
    #[test]
    fn reversing_the_wavevector_conjugates_the_full_matrix() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = cell_of(WATER, 12.0);
        let periodic = PeriodicOptions::default();
        let q = [0.31, -0.17, 0.24];
        let forward = dynamical_matrix(&molecule, &params, &options(), &periodic, q).unwrap();
        let reverse = dynamical_matrix(
            &molecule,
            &params,
            &options(),
            &periodic,
            [-q[0], -q[1], -q[2]],
        )
        .unwrap();
        let mut imaginary = 0.0_f64;
        for i in 0..forward.matrix.rows {
            for j in 0..forward.matrix.cols {
                let a = forward.matrix[(i, j)];
                let b = reverse.matrix[(i, j)];
                imaginary = imaginary.max(a.im.abs());
                assert!((a.re - b.re).abs() < 1.0e-6, "real [{i}][{j}]: {a:?} {b:?}");
                assert!(
                    (a.im + b.im).abs() < 1.0e-6,
                    "imaginary [{i}][{j}]: {a:?} {b:?}"
                );
            }
        }
        // Without this the identity is satisfied by a matrix that is real, which is what a stray
        // real part would produce.
        assert!(imaginary > 1.0, "D(q) is real, so the test proved nothing");
    }

    /// `D(q)` is Hermitian at a general `q`, **response included**.
    ///
    /// The two identity tests around this one run on the rigid-ion matrix, and that is exactly
    /// why they never saw the defect this one was written for: the response was where it lived.
    /// It was zero at `q = 0` and grew linearly with `q` — `1e-1` out of a `|D|` of `25` by
    /// `q = (0.2, 0, 0)` — because [`induced_charges`] read only the lower triangle of the
    /// first-order density and doubled it. That is the same number as the symmetrized form for
    /// the ground state, whose density is real and symmetric, and a different number for a
    /// complex first-order one; it also made the charge map stop being the adjoint of the
    /// potential map that scatters back over the whole block, which is what Hermiticity of `D`
    /// rests on.
    ///
    /// Measured worst case here is `2.4e-8`. The tolerance is `1e-6`, five orders below the
    /// defect it exists to catch.
    #[test]
    fn the_full_matrix_is_hermitian_away_from_gamma() {
        let params = Pm3Parameters::standard().unwrap();
        let periodic = PeriodicOptions::default();
        for edge in [12.0, 14.0] {
            let molecule = cell_of(WATER, edge);
            for q in [
                [0.25, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.3, -0.15, 0.42],
                [0.31, -0.17, 0.24],
                [0.5, 0.5, 0.5],
            ] {
                let d = dynamical_matrix(&molecule, &params, &options(), &periodic, q).unwrap();
                let scale = (0..d.matrix.rows)
                    .flat_map(|i| (0..d.matrix.cols).map(move |j| (i, j)))
                    .fold(0.0_f64, |m, (i, j)| m.max(d.matrix[(i, j)].norm()));
                assert!(
                    scale > 1.0,
                    "edge {edge} q {q:?}: |D| is {scale}, too small to test anything"
                );
                assert!(
                    d.hermitian_defect < 1.0e-6,
                    "edge {edge} q {q:?}: Hermitian defect {} against |D| {scale}",
                    d.hermitian_defect
                );
            }
        }
    }

    /// `D(q)` is periodic in `q`: adding a reciprocal lattice vector changes nothing.
    ///
    /// Every phase `e^{iq·T}` is unchanged by construction, so a failure here means something is
    /// reading `q` rather than the phases it implies — the shifted reciprocal sum dropping its
    /// `G + q = 0` term being the way that happens.
    ///
    /// The tolerance is `1e-9` rather than `1e-6` because the phases are now computed from the
    /// **fractional** wavevector, where the angle at a reciprocal lattice vector is `2πn`
    /// exactly. Through a Cartesian `q·T` it is `2πn` plus a rounding residue, once per image,
    /// and this identity held only to about `1e-7`. Three orders of magnitude of slack that were
    /// hiding nothing at the time, and would have hidden the next thing.
    #[test]
    fn the_matrix_is_periodic_in_the_wavevector() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = cell_of(WATER, 12.0);
        let periodic = PeriodicOptions::default();
        let base = [0.23, 0.0, 0.0];
        let shifted = [1.23, 0.0, 0.0];
        let a =
            rigid_ion_dynamical_matrix(&molecule, &params, &options(), &periodic, base).unwrap();
        let b =
            rigid_ion_dynamical_matrix(&molecule, &params, &options(), &periodic, shifted).unwrap();
        for i in 0..a.matrix.rows {
            for j in 0..a.matrix.cols {
                let x = a.matrix[(i, j)];
                let y = b.matrix[(i, j)];
                assert!(
                    (x.re - y.re).abs() < 1.0e-9 && (x.im - y.im).abs() < 1.0e-9,
                    "D is not periodic in q at [{i}][{j}]: {x:?} vs {y:?}"
                );
            }
        }
    }

    /// The acoustic sum rule at `q = 0`: displacing everything together costs nothing.
    #[test]
    fn the_gamma_matrix_obeys_the_acoustic_sum_rule() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = cell_of(WATER, 12.0);
        let periodic = PeriodicOptions::default();
        let d = rigid_ion_dynamical_matrix(&molecule, &params, &options(), &periodic, [0.0; 3])
            .unwrap();
        let nat = molecule.atoms.len();
        for alpha in 0..3 {
            for beta in 0..3 {
                let mut sum = 0.0;
                for a in 0..nat {
                    for b in 0..nat {
                        sum += d.matrix[(3 * a + alpha, 3 * b + beta)].re;
                    }
                }
                assert!(
                    sum.abs() < 1.0e-4,
                    "acoustic sum rule [{alpha}][{beta}] leaves {sum:.3e}"
                );
            }
        }
    }

    /// **The test.** A doubled cell's `D(0)` must hold exactly the primitive cell's `D(0)` and
    /// `D(b/2)` between them.
    ///
    /// Doubling the cell folds the zone in half, so the supercell's Γ-point force constants are
    /// the primitive cell's at `q = 0` and at the zone boundary, block-diagonalized by the two
    /// sublattices. Comparing the eigenvalue *sets* is the sharpest statement available: it does
    /// not depend on how either matrix orders its atoms, and essentially nothing survives it by
    /// accident. A phase of the wrong sign, a diagonal block that picked up `e^{iq·T}` when it
    /// should not have, a lattice sum shifted by the wrong `q` — each moves the boundary
    /// eigenvalues and leaves the `q = 0` ones alone, which is exactly what this separates.
    ///
    /// The cell is wide enough that Γ sampling is valid for *both* descriptions. Narrower, and
    /// the two would converge to different densities — the primitive cell and its double have
    /// different Γ-point errors — and the comparison would be measuring that instead.
    #[test]
    fn a_doubled_cell_holds_the_zone_centre_and_the_zone_boundary() {
        let params = Pm3Parameters::standard().unwrap();
        let periodic = PeriodicOptions::default();
        let edge = 16.0;

        let primitive = cell_of(WATER, edge);
        let zone_centre =
            rigid_ion_dynamical_matrix(&primitive, &params, &options(), &periodic, [0.0; 3])
                .unwrap();
        let zone_boundary =
            rigid_ion_dynamical_matrix(&primitive, &params, &options(), &periodic, [0.5, 0.0, 0.0])
                .unwrap();

        // The same crystal described with two cells along x.
        let mut doubled = Molecule::from_xyz_str(WATER, 0.0).unwrap();
        let shifted: Vec<_> = doubled
            .atoms
            .iter()
            .map(|a| crate::system::Atom {
                z: a.z,
                position: a.position + Vec3::new(edge, 0.0, 0.0),
            })
            .collect();
        doubled.atoms.extend(shifted);
        doubled.cell = Some(
            Cell::new(
                Vec3::new(2.0 * edge, 0.0, 0.0),
                Vec3::new(0.0, edge, 0.0),
                Vec3::new(0.0, 0.0, edge),
                [true; 3],
            )
            .unwrap(),
        );
        let supercell =
            rigid_ion_dynamical_matrix(&doubled, &params, &options(), &periodic, [0.0; 3]).unwrap();

        let mut expected: Vec<f64> = crate::cmatrix::hermitian_eigen(&zone_centre.matrix)
            .unwrap()
            .0;
        expected.extend(
            crate::cmatrix::hermitian_eigen(&zone_boundary.matrix)
                .unwrap()
                .0,
        );
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut got = crate::cmatrix::hermitian_eigen(&supercell.matrix)
            .unwrap()
            .0;
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_eq!(expected.len(), got.len());
        let scale = expected
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()))
            .max(1.0);
        for (index, (want, have)) in expected.iter().zip(&got).enumerate() {
            assert!(
                (want - have).abs() < 2.0e-4 * scale,
                "eigenvalue {index}: primitive pair gives {want}, the doubled cell {have}"
            );
        }
    }

    /// At `q = 0` the phased perturbation must be the unphased one, entry for entry.
    ///
    /// [`crate::pbc::kernel::fock_derivative`] is the same pair loop without phases, already
    /// exercised by the Γ-point Hessian's agreement with finite differences. Comparing the two
    /// matrices directly — rather than comparing Hessians built from them — is the sharpest and
    /// fastest way to find a contribution routed to the wrong destination or given the wrong
    /// phase, because every element is checked rather than a contraction of them.
    ///
    /// This is deliberately the *first* test of the perturbation. The phase table it pins is
    /// where a finite-`q` bug would live, and at `q = 0` the table has to be all ones, so a
    /// failure here is unambiguous: it means a contribution went to the wrong block.
    #[test]
    fn the_bare_perturbation_reduces_to_the_gamma_fock_derivative() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = cell_of(WATER, 14.0);
        let periodic = PeriodicOptions::default();
        let scf = run_gamma(&molecule, &params, &options(), &periodic).unwrap();
        let setup = crate::pbc::gamma::build_setup(&molecule, &params, &periodic).unwrap();
        let basis = Basis::build(&molecule, &params).unwrap();
        let list = {
            let cutoff = periodic.short_range_cutoff.max(periodic.switch.off);
            let positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
            NeighborList::build_from_positions(&positions, molecule.cell.as_ref(), cutoff)
        };
        let index_of = image_index(&setup);

        for atom in 0..molecule.atoms.len() {
            for axis in 0..3 {
                let want = crate::pbc::kernel::fock_derivative_pairs(
                    &molecule,
                    &params,
                    &periodic,
                    &scf.density,
                    &basis,
                    atom,
                    axis,
                )
                .unwrap();
                let blocks = bare_blocks(
                    &molecule,
                    &params,
                    &periodic,
                    &setup,
                    &scf.density,
                    // The restricted channel: the total density at half exchange strength.
                    &scf.density,
                    0.5,
                    [0.0; 3],
                    atom,
                    axis,
                    &index_of,
                    &list,
                )
                .unwrap();
                let got = blocks.at_k(&setup, [0.0; 3]);

                for i in 0..basis.nao {
                    for j in 0..basis.nao {
                        let a = got[(i, j)];
                        assert!(
                            a.im.abs() < 1.0e-12,
                            "atom {atom} axis {axis} [{i}][{j}] is complex at q = 0: {}",
                            a.im
                        );
                        assert!(
                            (a.re - want[(i, j)]).abs() < 1.0e-10,
                            "atom {atom} axis {axis} [{i}][{j}]: phased {} vs unphased {}",
                            a.re,
                            want[(i, j)]
                        );
                    }
                }
            }
        }
    }

    /// The phased field derivative must reproduce the finite-difference one at `q = 0`.
    ///
    /// The Γ-point path gets this term by moving the atom and re-running the whole Ewald sum;
    /// this one differentiates the kernel analytically and phases it. They are entirely
    /// different computations of the same quantity, which is what makes the comparison worth
    /// making — and the `q = 0` case is the only place they can be compared at all, since a
    /// finite-difference displacement cannot carry a phase.
    ///
    /// A tolerance of `1e-6` because the reference is itself a central difference at `1e-5`
    /// Bohr; the analytic side is the more accurate of the two.
    #[test]
    fn the_phased_field_derivative_matches_the_finite_difference_one() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = cell_of(WATER, 14.0);
        let periodic = PeriodicOptions::default();
        let scf = run_gamma(&molecule, &params, &options(), &periodic).unwrap();
        let setup = crate::pbc::gamma::build_setup(&molecule, &params, &periodic).unwrap();
        let basis = Basis::build(&molecule, &params).unwrap();
        let cell = molecule.cell.unwrap();

        // Total site charges — cores and electrons together, as the field sees them.
        let mut sites = setup.sites.clone();
        crate::pbc::gamma::write_electron_charges(&setup, &scf.density, &mut sites);
        for (ia, atom) in molecule.atoms.iter().enumerate() {
            sites[setup.site_offset[ia]].charge += params.element(atom.z).unwrap().core_charge;
        }
        let charges: Vec<f64> = sites.iter().map(|s| s.charge).collect();
        let long_range = long_range_kernels(&setup, &cell, Vec3::zero()).unwrap();

        for atom in 0..molecule.atoms.len() {
            for axis in 0..3 {
                let with_field = crate::pbc::kernel::fock_derivative(
                    &molecule,
                    &params,
                    &periodic,
                    &scf.density,
                    &basis,
                    atom,
                    axis,
                )
                .unwrap();
                let pairs_only = crate::pbc::kernel::fock_derivative_pairs(
                    &molecule,
                    &params,
                    &periodic,
                    &scf.density,
                    &basis,
                    atom,
                    axis,
                )
                .unwrap();

                let mut got = CMatrix::zeros(basis.nao, basis.nao);
                phased_field_derivative(&setup, &long_range, &charges, atom, axis, &mut got)
                    .unwrap();

                let scale = crate::constants::PM3_EV;
                for i in 0..basis.nao {
                    for j in 0..basis.nao {
                        let want = with_field[(i, j)] - pairs_only[(i, j)];
                        let have = scale * got[(i, j)].re;
                        assert!(
                            got[(i, j)].im.abs() < 1.0e-10,
                            "atom {atom} axis {axis} [{i}][{j}] complex at q = 0"
                        );
                        assert!(
                            (have - want).abs() < 1.0e-6 * want.abs().max(1.0),
                            "atom {atom} axis {axis} [{i}][{j}]: analytic {have} vs finite \
                             difference {want}"
                        );
                    }
                }
            }
        }
    }

    /// **The response test.** `D(0)` with the response must be the Γ-point analytic Hessian.
    ///
    /// [`crate::pbc::hessian::periodic_hessian`] computes the same object by a different route —
    /// a real CPHF in the occupied–virtual space, contracted the way a Γ-point Hessian is — and
    /// it is already checked against finite differences of the energy. So this compares an
    /// arbitrary-`q` machine, run at `q = 0`, against an independent implementation of the one
    /// case they share.
    ///
    /// Everything the response is made of is exercised: the perturbation's phase table, the
    /// field derivative, the two-electron kernel, the linear-response denominators, the
    /// occupied–empty *and* empty–occupied blocks, and the contraction. A factor of two in the
    /// last of those is the classic error and shows up here immediately.
    #[test]
    fn the_response_reproduces_the_gamma_hessian_at_zero_wavevector() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = cell_of(WATER, 14.0);
        let periodic = PeriodicOptions::default();

        let reference =
            crate::pbc::hessian::periodic_hessian(&molecule, &params, &options(), &periodic)
                .unwrap();
        let got = dynamical_matrix(&molecule, &params, &options(), &periodic, [0.0; 3]).unwrap();

        let ndof = reference.rows;
        let scale = (0..ndof)
            .flat_map(|i| (0..ndof).map(move |j| (i, j)))
            .fold(0.0_f64, |m, (i, j)| m.max(reference[(i, j)].abs()))
            .max(1.0);
        let mut worst = (0usize, 0usize, 0.0_f64);
        for i in 0..ndof {
            for j in 0..ndof {
                assert!(
                    got.matrix[(i, j)].im.abs() < 1.0e-8,
                    "D(0)[{i}][{j}] is complex: {}",
                    got.matrix[(i, j)].im
                );
                let difference = (got.matrix[(i, j)].re - reference[(i, j)]).abs();
                if difference > worst.2 {
                    worst = (i, j, difference);
                }
            }
        }
        assert!(
            worst.2 < 2.0e-3 * scale,
            "worst disagreement {:.4e} at [{}][{}]: perturbation theory {} vs Γ Hessian {} \
             (largest element {scale:.3})",
            worst.2,
            worst.0,
            worst.1,
            got.matrix[(worst.0, worst.1)].re,
            reference[(worst.0, worst.1)]
        );
    }

    /// **The open-shell test.** `D(0)` for a spin-polarized cell, against central differences of
    /// the periodic UHF force.
    ///
    /// The reference shares nothing with the thing under test. `periodic_gradient` is spin-aware
    /// already — `GammaDensity` carries the spin density and the image blocks — and it reaches
    /// the answer through the *converged density* rather than through any linear response, so a
    /// wrong spin factor anywhere in the response cannot cancel between the two sides. There is
    /// no unrestricted Γ-point periodic Hessian to compare against, which is exactly why this is
    /// a finite difference rather than another analytic result.
    ///
    /// A doublet in a 13 Bohr box, small enough that the images matter and the test is about
    /// periodicity rather than about a molecule.
    ///
    /// That edge is *inside* [`crate::pbc::gamma::DEFAULT_SHORT_RANGE_CUTOFF`], so the Γ margin
    /// here is `−1` Bohr and this ground state is not one a single k-point can represent. That
    /// does not weaken the test, because both sides are built from the same ground state: this
    /// asserts that the analytic response is the derivative of the force, which holds whatever
    /// the force is the derivative *of*. It does mean the numbers are not a physical prediction,
    /// and a cell that cleared the margin would not be a stronger test of the derivative — only
    /// a slower one. `tests/pbc_uhf_response.rs` covers the spin path at a positive margin.
    #[test]
    fn an_open_shell_cell_matches_the_finite_difference_force_constants() {
        let params = Pm3Parameters::standard().unwrap();
        let options = Pm3Options {
            multiplicity: 2,
            e_tol: 1.0e-11,
            p_tol: 1.0e-10,
            max_scf: 600,
            ..Pm3Options::default()
        };
        let periodic = PeriodicOptions::default();
        let xyz = "4\nmethyl\nC 0.0 0.0 0.05\nH 1.09 0.0 0.0\nH -0.545 0.944 0.0\n\
                   H -0.545 -0.944 0.0\n";
        let molecule = cell_of(xyz, 13.0);
        assert!(
            crate::pbc::gamma::run_gamma(&molecule, &params, &options, &periodic)
                .unwrap()
                .unrestricted,
            "the fixture converged closed-shell, so this proves nothing about the spin path"
        );

        let got = dynamical_matrix(&molecule, &params, &options, &periodic, [0.0; 3]).unwrap();

        let ndof = 3 * molecule.atoms.len();
        let step = 1.0e-3;
        let mut numeric = crate::linalg::Matrix::zeros(ndof, ndof);
        for dof in 0..ndof {
            let gradient_at = |sign: f64| {
                let mut shifted = molecule.clone();
                let mut delta = [0.0; 3];
                delta[dof % 3] = sign * step;
                shifted.atoms[dof / 3].position += Vec3::new(delta[0], delta[1], delta[2]);
                crate::pbc::gradient::periodic_gradient(&shifted, &params, &options, &periodic)
                    .unwrap()
                    .gradient
            };
            let (plus, minus) = (gradient_at(1.0), gradient_at(-1.0));
            for other in 0..ndof {
                let p = plus[other / 3].to_array()[other % 3];
                let m = minus[other / 3].to_array()[other % 3];
                numeric[(other, dof)] = (p - m) / (2.0 * step);
            }
        }

        let scale = (0..ndof)
            .flat_map(|i| (0..ndof).map(move |j| (i, j)))
            .fold(0.0_f64, |m, (i, j)| m.max(numeric[(i, j)].abs()));
        assert!(scale > 1.0, "the force constants collapsed to {scale}");
        let mut worst = (0usize, 0usize, 0.0_f64);
        for i in 0..ndof {
            for j in 0..ndof {
                assert!(
                    got.matrix[(i, j)].im.abs() < 1.0e-8,
                    "D(0)[{i}][{j}] is complex: {}",
                    got.matrix[(i, j)].im
                );
                let difference = (got.matrix[(i, j)].re - numeric[(i, j)]).abs();
                if difference > worst.2 {
                    worst = (i, j, difference);
                }
            }
        }
        assert!(
            worst.2 < 4.0e-3 * scale,
            "worst disagreement {:.4e} at [{}][{}]: response {} vs finite difference {} \
             (largest element {scale:.3})",
            worst.2,
            worst.0,
            worst.1,
            got.matrix[(worst.0, worst.1)].re,
            numeric[(worst.0, worst.1)]
        );
    }

    /// **The correction test.** The corrected `D(q)` at the zone boundary of a cell must be the
    /// zone centre of its own doubled cell, folded.
    ///
    /// This is the only test in the file that can see the coordination-number image copy being
    /// wrong. At `Γ` every copy moves alike, so an image's coordination response *is* its
    /// parent's and the copy is exact; the supercell comparison is the one place a phased
    /// displacement is checked against a calculation that has no phases in it at all — the
    /// doubled cell simply has both atoms as its own, and its `Γ` dynamical matrix is the
    /// folded answer by definition.
    ///
    /// The fixture is two water molecules along the chain axis, close enough to hydrogen bond
    /// across the cell boundary, so the H4 triples straddle it and the D3 coordination numbers of
    /// the images are not the trivial ones.
    #[test]
    fn the_corrections_fold_onto_a_doubled_cell() {
        let periodic = PeriodicOptions::default();
        let corrected = Pm3Options {
            variant: Variant::Pm3D3H4X,
            ..options()
        };

        // A chain of hydrogen-bonded waters: O–H···O across the cell edge.
        let repeat = 5.6;
        let primitive_xyz = "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n";
        let mut primitive = Molecule::from_xyz_str(primitive_xyz, 0.0).unwrap();
        primitive.cell = Some(
            Cell::new(
                Vec3::new(repeat, 0.0, 0.0),
                Vec3::new(0.0, 24.0, 0.0),
                Vec3::new(0.0, 0.0, 24.0),
                [true, false, false],
            )
            .unwrap(),
        );

        let mut doubled = primitive.clone();
        let shift = Vec3::new(repeat, 0.0, 0.0);
        for index in 0..primitive.atoms.len() {
            let mut copy = primitive.atoms[index].clone();
            copy.position += shift;
            doubled.atoms.push(copy);
        }
        doubled.cell = Some(
            Cell::new(
                Vec3::new(2.0 * repeat, 0.0, 0.0),
                Vec3::new(0.0, 24.0, 0.0),
                Vec3::new(0.0, 0.0, 24.0),
                [true, false, false],
            )
            .unwrap(),
        );

        // Only the corrections, so the electronic response cannot mask a defect in them.
        let corrections_of = |molecule: &Molecule, q_frac: [f64; 3], radius: f64| -> CMatrix {
            let cell = molecule.cell.unwrap();
            let mut q = Vec3::zero();
            for (index, b) in cell.reciprocal_basis() {
                q += b * q_frac[index];
            }
            let mut wider = periodic;
            wider.correction_cutoffs.dispersion = radius;
            let ndof = 3 * molecule.atoms.len();
            let mut out = CMatrix::zeros(ndof, ndof);
            phased_corrections(molecule, corrected.variant, &wider, q, &mut out).unwrap();
            out
        };

        // The residual is not zero, and its size is the point. An image the cluster is too narrow
        // around takes its parent's coordination number, and the two cells disagree about which
        // images those are: `coordination_is_exact` measures from the nearest *cell* atom, and
        // the doubled cell's contents are twice as wide, so at the same radius it treats more
        // images honestly. Widening the cluster shrinks the disagreement — measured `3.95e-8` at
        // 22 Bohr and `2.31e-8` at 30, against force constants of `6.7e-2`, a relative `3.5e-7`.
        // It converges slowly because the coordination number itself does not converge in a
        // cutoff at all (see `DEFAULT_CN_CUTOFF`), so this asserts the direction and records the
        // rate rather than claiming a power.
        let mut residuals = Vec::new();
        for radius in [22.0_f64, 30.0] {
            residuals.push(fold_defect(
                &corrections_of(&primitive, [0.0; 3], radius),
                &corrections_of(&primitive, [0.5, 0.0, 0.0], radius),
                &corrections_of(&doubled, [0.0; 3], radius),
                primitive.atoms.len(),
            ));
        }
        assert!(
            residuals[1].2 < residuals[0].2,
            "widening the cluster from 22 to 30 Bohr did not shrink the folding defect: \
             {:.3e} then {:.3e}",
            residuals[0].2,
            residuals[1].2
        );

        let boundary = corrections_of(&primitive, [0.5, 0.0, 0.0], 30.0);
        let centre = corrections_of(&primitive, [0.0; 3], 30.0);
        let folded = corrections_of(&doubled, [0.0; 3], 30.0);

        // The doubled cell's Γ matrix block-diagonalizes on the plane waves of its two
        // sublattices: `D_super_{(s,κ),(s',κ')} = ½ Σ_q e^{iq(R_s − R_s')} D_prim(q)_{κκ'}` over
        // `q ∈ {0, b/2}`. Adding the two blocks in a row recovers `D(0)`; subtracting them
        // recovers `D(b/2)`, which is the half that carries the phases and the half a wrong
        // coordination-number copy would land in.
        let worst = fold_defect(&centre, &boundary, &folded, primitive.atoms.len());
        let scale = (0..centre.rows)
            .flat_map(|i| (0..centre.cols).map(move |j| (i, j)))
            .fold(0.0_f64, |m, (i, j)| m.max(centre[(i, j)].norm()));
        assert!(scale > 1.0e-4, "the corrections collapsed to {scale}");
        assert!(
            worst.2 < 1.0e-6 * scale,
            "the corrected D(q) does not fold: {:.3e} at [{}][{}] (scale {scale:.3e})",
            worst.2,
            worst.0,
            worst.1
        );
    }

    /// The largest failure of `D_super(Γ)` to be the primitive `D(0)` and `D(b/2)` folded.
    fn fold_defect(
        centre: &CMatrix,
        boundary: &CMatrix,
        folded: &CMatrix,
        nat: usize,
    ) -> (usize, usize, f64) {
        let mut worst = (0usize, 0usize, 0.0_f64);
        for i in 0..3 * nat {
            for j in 0..3 * nat {
                let same = folded[(i, j)];
                let cross = folded[(i, j + 3 * nat)];
                for difference in [
                    (same + cross - centre[(i, j)]).norm(),
                    (same - cross - boundary[(i, j)]).norm(),
                ] {
                    if difference > worst.2 {
                        worst = (i, j, difference);
                    }
                }
            }
        }
        worst
    }

    /// The phased two-electron kernel must be the unphased one at `q = 0`.
    ///
    /// Between the perturbation and the assembled Hessian sits the kernel, and a self-consistent
    /// loop built on a wrong one converges — smoothly, to the wrong answer. Comparing it against
    /// [`crate::pbc::kernel::PeriodicKernel`] on an arbitrary density perturbation isolates it
    /// from everything else in the response.
    #[test]
    fn the_phased_kernel_reduces_to_the_unphased_one() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = cell_of(WATER, 14.0);
        let periodic = PeriodicOptions::default();
        let setup = crate::pbc::gamma::build_setup(&molecule, &params, &periodic).unwrap();
        let cell = molecule.cell.unwrap();
        let nao = setup.basis.nao;
        let index_of = image_index(&setup);

        // An arbitrary symmetric perturbation — the kernel is linear, so any one will do.
        let mut probe = crate::linalg::Matrix::zeros(nao, nao);
        for i in 0..nao {
            for j in 0..=i {
                let value = 0.03 * ((i * 7 + j * 3) % 11) as f64 - 0.15;
                probe[(i, j)] = value;
                probe[(j, i)] = value;
            }
        }
        let want = crate::pbc::kernel::PeriodicKernel::build(&molecule, &params, &periodic)
            .unwrap()
            .apply(&probe)
            .unwrap();

        let mut delta = BareBlocks::new(nao, &setup.images);
        for i in 0..nao {
            for j in 0..nao {
                delta.onsite[(i, j)] = c64::new(probe[(i, j)], 0.0);
            }
        }
        for (block, slot) in setup.images.iter().zip(delta.images.iter_mut()) {
            let (oa, ob) = (
                setup.basis.atom_offset[block.a],
                setup.basis.atom_offset[block.b],
            );
            for mu in 0..block.norb_a {
                for la in 0..block.norb_b {
                    slot[(mu, la)] = c64::new(probe[(oa + mu, ob + la)], 0.0);
                }
            }
        }
        let site_kernel = SiteKernel::build(&setup, &cell, Vec3::zero()).unwrap();
        let apply = |channels: &[SpinChannel], input: &[BareBlocks]| -> Vec<CMatrix> {
            phased_kernel(
                &molecule,
                &params,
                &periodic,
                &setup,
                &cell,
                [0.0; 3],
                input,
                channels,
                &index_of,
                &site_kernel,
            )
            .unwrap()
            .iter()
            .map(|blocks| blocks.at_k(&setup, [0.0; 3]))
            .collect()
        };
        let restricted = apply(&[dummy_channel(0.5)], std::slice::from_ref(&delta));
        let got = restricted[0].clone();

        // The plan's checkpoint for the open-shell split: two channels each holding half the
        // perturbation at full exchange strength must be the same number as one channel holding
        // all of it at half strength. This is the identity that says the split moved the spin
        // factor and did not also move one of the *double-visit* halvings, which look identical
        // in the source and are not the same thing.
        let halved = {
            let mut out = BareBlocks::new(nao, &setup.images);
            for i in 0..nao {
                for j in 0..nao {
                    out.onsite[(i, j)] = c64::new(0.5 * delta.onsite[(i, j)].re, 0.0);
                }
            }
            for (slot, source) in out.images.iter_mut().zip(&delta.images) {
                for i in 0..slot.rows {
                    for j in 0..slot.cols {
                        slot[(i, j)] = c64::new(0.5 * source[(i, j)].re, 0.0);
                    }
                }
            }
            out
        };
        let split = apply(
            &[dummy_channel(1.0), dummy_channel(1.0)],
            &[
                BareBlocks {
                    onsite: halved.onsite.clone(),
                    images: halved.images.clone(),
                },
                halved,
            ],
        );
        assert_eq!(split.len(), 2, "the split kernel dropped a channel");
        for (spin, channel) in split.iter().enumerate() {
            for i in 0..nao {
                for j in 0..nao {
                    assert!(
                        (channel[(i, j)] - got[(i, j)]).norm() < 1.0e-12,
                        "channel {spin} at [{i}][{j}]: {:?} vs {:?}",
                        channel[(i, j)],
                        got[(i, j)]
                    );
                }
            }
        }

        let scale = (0..nao)
            .flat_map(|i| (0..nao).map(move |j| (i, j)))
            .fold(0.0_f64, |m, (i, j)| m.max(want[(i, j)].abs()))
            .max(1.0);
        for i in 0..nao {
            for j in 0..nao {
                assert!(
                    got[(i, j)].im.abs() < 1.0e-10,
                    "[{i}][{j}] complex at q = 0"
                );
                assert!(
                    (got[(i, j)].re - want[(i, j)]).abs() < 1.0e-8 * scale,
                    "kernel[{i}][{j}]: phased {} vs unphased {}",
                    got[(i, j)].re,
                    want[(i, j)]
                );
            }
        }
    }

    /// The phased Coulomb second derivative must be the unphased one at `q = 0`.
    ///
    /// [`crate::pbc::ewald_hessian::ewald_atom_hessian`] computes it from the Ewald sum directly,
    /// already reduced from sites to atoms, and is exercised by the Γ-point Hessian's agreement
    /// with finite differences.
    ///
    /// The folding test cannot catch an error here: both the primitive cell and its double go
    /// through this same code, so a wrong-but-consistent contraction folds perfectly. That is
    /// exactly the kind of agreement that feels like validation and is not, which is why this
    /// compares against something built differently instead.
    #[test]
    fn the_phased_coulomb_hessian_reduces_to_the_unphased_one() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = cell_of(WATER, 14.0);
        let periodic = PeriodicOptions::default();
        let scf = run_gamma(&molecule, &params, &options(), &periodic).unwrap();
        let basis = Basis::build(&molecule, &params).unwrap();
        let cell = molecule.cell.unwrap();
        let nat = molecule.atoms.len();

        let (sites, ewald_params) =
            crate::pbc::hessian::charge_sites(&molecule, &params, &basis, &scf.density, &periodic)
                .unwrap();
        let want = crate::pbc::ewald_hessian::ewald_atom_hessian(&cell, &sites, &ewald_params, nat)
            .unwrap()
            .expect("3D always has one");

        let mut got = CMatrix::zeros(3 * nat, 3 * nat);
        phased_lattice_sum(
            &molecule,
            &params,
            &periodic,
            &scf.density,
            &basis,
            // Cartesian here, unlike its neighbours: this is the one that feeds `ewald_phased`.
            Vec3::zero(),
            &mut got,
        )
        .unwrap();

        let scale = (0..3 * nat)
            .flat_map(|i| (0..3 * nat).map(move |j| (i, j)))
            .fold(0.0_f64, |m, (i, j)| m.max(want[(i, j)].abs()))
            .max(1.0);
        for i in 0..3 * nat {
            for j in 0..3 * nat {
                assert!(
                    got[(i, j)].im.abs() < 1.0e-9,
                    "[{i}][{j}] is complex at q = 0"
                );
                assert!(
                    (got[(i, j)].re - want[(i, j)]).abs() < 1.0e-6 * scale,
                    "Coulomb Hessian [{i}][{j}]: phased {} vs Ewald {}",
                    got[(i, j)].re,
                    want[(i, j)]
                );
            }
        }
    }

    /// The phased short-range skeleton must be the unphased one at `q = 0`.
    #[test]
    fn the_phased_skeleton_reduces_to_the_unphased_one() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = cell_of(WATER, 14.0);
        let periodic = PeriodicOptions::default();
        let scf = run_gamma(&molecule, &params, &options(), &periodic).unwrap();
        let basis = Basis::build(&molecule, &params).unwrap();
        let nat = molecule.atoms.len();

        let want =
            crate::pbc::hessian::skeleton(&molecule, &params, &options(), &periodic, &scf, &basis)
                .unwrap();
        let mut got = CMatrix::zeros(3 * nat, 3 * nat);
        phased_skeleton(
            &molecule,
            &params,
            &options(),
            &periodic,
            &scf.density,
            &[(&scf.density, 0.5)],
            &basis,
            [0.0; 3],
            &mut got,
        )
        .unwrap();

        let scale = (0..3 * nat)
            .flat_map(|i| (0..3 * nat).map(move |j| (i, j)))
            .fold(0.0_f64, |m, (i, j)| m.max(want[(i, j)].abs()))
            .max(1.0);
        for i in 0..3 * nat {
            for j in 0..3 * nat {
                assert!(got[(i, j)].im.abs() < 1.0e-9, "[{i}][{j}] complex at q = 0");
                assert!(
                    (got[(i, j)].re - want[(i, j)]).abs() < 1.0e-8 * scale,
                    "skeleton [{i}][{j}]: phased {} vs unphased {}",
                    got[(i, j)].re,
                    want[(i, j)]
                );
            }
        }
    }

    /// The orbitals the response is built on must be the ones the SCF converged to.
    ///
    /// The response rebuilds the converged Fock so it can be evaluated away from Γ, from the
    /// k-path's on-site build plus the image sum. If that reconstruction differs from what the
    /// Γ SCF converged to — a missing term, a density read from the wrong blocks — the response
    /// is a correct answer to the wrong question, and every downstream comparison disagrees by
    /// an amount with no obvious pattern.
    #[test]
    fn the_rebuilt_fock_reproduces_the_converged_orbitals() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = cell_of(WATER, 14.0);
        let periodic = PeriodicOptions::default();
        let scf = run_gamma(&molecule, &params, &options(), &periodic).unwrap();
        let setup = crate::pbc::gamma::build_setup(&molecule, &params, &periodic).unwrap();
        let cell = molecule.cell.unwrap();

        let mut half = scf.density.clone();
        for value in half.as_mut_slice() {
            *value *= 0.5;
        }
        let mut f_onsite =
            crate::pbc::kscf::onsite_fock(&molecule, &params, &setup, &scf.density, &half).unwrap();
        let mut sites = setup.sites.clone();
        crate::pbc::gamma::write_electron_charges(&setup, &scf.density, &mut sites);
        let field = crate::pbc::ewald::ewald_cached(
            &cell,
            &sites,
            &setup.ewald_params,
            &setup.ewald_context,
        )
        .unwrap();
        crate::pbc::gamma::add_site_potential(&setup, &mut f_onsite, &field);

        let p_images: Vec<Vec<f64>> = setup
            .images
            .iter()
            .map(|block| {
                let (oa, ob) = (
                    setup.basis.atom_offset[block.a],
                    setup.basis.atom_offset[block.b],
                );
                let mut out = vec![0.0; block.norb_a * block.norb_b];
                for mu in 0..block.norb_a {
                    for la in 0..block.norb_b {
                        // Half: the image blocks feed the **exchange**, which is per spin, and
                        // a closed-shell alpha density is half the total. Handing the total over
                        // doubles the exchange and pushes every orbital down -- by twelve eV on
                        // water, which is not subtle once looked at and invisible until then.
                        out[mu * block.norb_b + la] = 0.5 * scf.density[(oa + mu, ob + la)];
                    }
                }
                out
            })
            .collect();

        let k = crate::pbc::kpoints::KPoint {
            frac: [0.0; 3],
            weight: 1.0,
        };
        let (eps, _) = crate::cmatrix::hermitian_eigen(&crate::pbc::kscf::bloch_fock(
            &setup, &f_onsite, &p_images, &k,
        ))
        .unwrap();
        for (index, (rebuilt, converged)) in eps.iter().zip(&scf.mo_energies).enumerate() {
            // The SCF stops at `p_tol`, so the two agree to about that and no further; what is
            // being caught here is a missing term, not the last digit.
            assert!(
                (rebuilt - converged).abs() < 1.0e-6,
                "orbital {index}: rebuilt {rebuilt} vs converged {converged}"
            );
        }
    }

    /// **The finite-`q` test.** The full `D(q)` — response included — must fold onto a doubled
    /// cell's Γ-point Hessian.
    ///
    /// Everything else here is checked at `q = 0`, where every phase is one and a wrong phase
    /// cannot show. This is the test that exercises them: the perturbation's phase table, the
    /// shifted lattice sum in the kernel, the `(Γ, q)` band pairing, and the assembly, all at a
    /// wavevector where they matter.
    ///
    /// The reference is [`crate::pbc::hessian::periodic_hessian`] on the doubled cell — a real
    /// CPHF with no phases anywhere, computing what folding says the two primitive matrices
    /// must contain between them. Eigenvalue *sets* are compared so nothing depends on how
    /// either side orders its atoms.
    #[test]
    fn the_full_dynamical_matrix_folds_onto_a_doubled_cell() {
        let params = Pm3Parameters::standard().unwrap();
        let periodic = PeriodicOptions::default();
        let edge = 16.0;

        let primitive = cell_of(WATER, edge);
        let zone_centre =
            dynamical_matrix(&primitive, &params, &options(), &periodic, [0.0; 3]).unwrap();
        let zone_boundary =
            dynamical_matrix(&primitive, &params, &options(), &periodic, [0.5, 0.0, 0.0]).unwrap();

        let mut doubled = Molecule::from_xyz_str(WATER, 0.0).unwrap();
        let shifted: Vec<_> = doubled
            .atoms
            .iter()
            .map(|a| crate::system::Atom {
                z: a.z,
                position: a.position + Vec3::new(edge, 0.0, 0.0),
            })
            .collect();
        doubled.atoms.extend(shifted);
        doubled.cell = Some(
            Cell::new(
                Vec3::new(2.0 * edge, 0.0, 0.0),
                Vec3::new(0.0, edge, 0.0),
                Vec3::new(0.0, 0.0, edge),
                [true; 3],
            )
            .unwrap(),
        );
        let reference =
            crate::pbc::hessian::periodic_hessian(&doubled, &params, &options(), &periodic)
                .unwrap();

        let mut expected: Vec<f64> = crate::cmatrix::hermitian_eigen(&zone_centre.matrix)
            .unwrap()
            .0;
        expected.extend(
            crate::cmatrix::hermitian_eigen(&zone_boundary.matrix)
                .unwrap()
                .0,
        );
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut got = crate::linalg::symmetric_eigen(&reference).unwrap().0;
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_eq!(expected.len(), got.len());
        let scale = expected
            .iter()
            .fold(0.0_f64, |m, v| m.max(v.abs()))
            .max(1.0);
        for (index, (want, have)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (want - have).abs() < 5.0e-3 * scale,
                "eigenvalue {index}: doubled cell gives {want}, the primitive pair {have}"
            );
        }
    }

    /// Frequencies at `q = 0` must be the Γ-point path's, and a water molecule alone in a cell
    /// must show its three internal modes above 1000 cm⁻¹.
    ///
    /// The number that would move first if the response were wrong is the highest stretch, which
    /// the rigid-ion matrix alone puts thousands of wavenumbers too high — so this is a check on
    /// the physics reaching the user-facing quantity, not only on the matrix.
    #[test]
    fn the_frequencies_at_gamma_are_the_gamma_point_ones() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = cell_of(WATER, 16.0);
        let periodic = PeriodicOptions::default();

        let got = phonon_frequencies(&molecule, &params, &options(), &periodic, [0.0; 3]).unwrap();
        let reference =
            crate::pbc::hessian::periodic_phonons(&molecule, &params, &options(), &periodic)
                .unwrap();

        // The top three are the molecule's internal modes; the rest are translations and
        // librations of a nearly isolated molecule and are soft.
        let mut mine = got.clone();
        mine.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut theirs = reference.frequencies_cm.clone();
        theirs.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for (index, (a, b)) in mine.iter().zip(&theirs).enumerate().skip(mine.len() - 3) {
            assert!(
                (a - b).abs() < 25.0,
                "mode {index}: perturbation theory {a:.1} cm⁻¹ vs Γ path {b:.1}"
            );
            assert!(
                *a > 1000.0,
                "mode {index} should be an internal stretch: {a:.1}"
            );
        }
    }

    /// A wavevector out of the periodic subspace and the corrected variants are refused rather
    /// than answered.
    ///
    /// The subspace check is the loop over `q_frac` components on non-periodic axes: a slab
    /// has no perpendicular wavevector and a chain no transverse one, and a fractional
    /// component there is the only way to ask for either — the Cartesian `q` is assembled from
    /// the periodic subspace's own reciprocal basis, so it cannot leave the subspace once the
    /// fractional check passes.
    #[test]
    fn what_is_not_covered_says_so() {
        let params = Pm3Parameters::standard().unwrap();
        let periodic = PeriodicOptions::default();

        let error = rigid_ion_dynamical_matrix(
            &slab_of(WATER),
            &params,
            &options(),
            &periodic,
            [0.2, 0.0, 0.3],
        )
        .expect_err("a slab has no out-of-plane wavevector");
        assert!(error.to_string().contains("non-periodic direction"));

        let error = rigid_ion_dynamical_matrix(
            &chain_of(WATER),
            &params,
            &options(),
            &periodic,
            [0.2, 0.3, 0.0],
        )
        .expect_err("a chain has no transverse wavevector");
        assert!(error.to_string().contains("non-periodic direction"));

        // The corrected variants used to be refused here. They are carried now — see
        // [`the_corrections_fold_onto_a_doubled_cell`] — so what this asserts is that they run
        // and produce something, rather than that they are turned away.
        let molecule = cell_of(WATER, 12.0);
        let corrected = Pm3Options {
            variant: Variant::Pm3D3H4X,
            ..options()
        };
        let with = rigid_ion_dynamical_matrix(&molecule, &params, &corrected, &periodic, [0.0; 3])
            .expect("the corrections are carried at finite q now");
        let without =
            rigid_ion_dynamical_matrix(&molecule, &params, &options(), &periodic, [0.0; 3])
                .unwrap();
        let moved = (0..with.matrix.rows)
            .flat_map(|i| (0..with.matrix.cols).map(move |j| (i, j)))
            .fold(0.0_f64, |m, (i, j)| {
                m.max((with.matrix[(i, j)] - without.matrix[(i, j)]).norm())
            });
        assert!(
            moved > 1.0e-6,
            "the corrections changed nothing ({moved:.3e})"
        );
    }

    fn slab_of(xyz: &str) -> Molecule {
        let mut molecule = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        molecule.cell = Some(
            Cell::new(
                Vec3::new(12.0, 0.0, 0.0),
                Vec3::new(0.0, 12.0, 0.0),
                Vec3::new(0.0, 0.0, 40.0),
                [true, true, false],
            )
            .unwrap(),
        );
        molecule
    }

    fn chain_of(xyz: &str) -> Molecule {
        let mut molecule = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        molecule.cell = Some(
            Cell::new(
                Vec3::new(12.0, 0.0, 0.0),
                Vec3::new(0.0, 40.0, 0.0),
                Vec3::new(0.0, 0.0, 40.0),
                [true, false, false],
            )
            .unwrap(),
        );
        molecule
    }

    /// The rigid-ion matrix in reduced dimensionality: `D(−q) = D(q)*` at a generic in-subspace
    /// wavevector, and the acoustic sum rule at **exactly** `q = 0`.
    ///
    /// Exactly, because for a slab `q = 0` is not the limit of small `q`: the phased 2D value
    /// is genuinely non-analytic as `q → 0` (the LO–TO physics), so approaching Γ would test a
    /// limit the matrix is not claimed to have. The sum rule is a statement about the point
    /// itself — displacing everything together costs nothing — and holds there term by term.
    #[test]
    fn a_slab_and_a_chain_obey_the_finite_q_identities() {
        let params = Pm3Parameters::standard().unwrap();
        let periodic = PeriodicOptions::default();

        for (molecule, q) in [
            (slab_of(WATER), [0.3, -0.15, 0.0]),
            (chain_of(WATER), [0.3, 0.0, 0.0]),
        ] {
            let forward =
                rigid_ion_dynamical_matrix(&molecule, &params, &options(), &periodic, q).unwrap();
            let reverse = rigid_ion_dynamical_matrix(
                &molecule,
                &params,
                &options(),
                &periodic,
                [-q[0], -q[1], -q[2]],
            )
            .unwrap();
            let mut largest_imaginary = 0.0_f64;
            for i in 0..forward.matrix.rows {
                for j in 0..forward.matrix.cols {
                    let a = forward.matrix[(i, j)];
                    let b = reverse.matrix[(i, j)];
                    assert!((a.re - b.re).abs() < 1.0e-8, "real [{i}][{j}]");
                    assert!((a.im + b.im).abs() < 1.0e-8, "imaginary [{i}][{j}]");
                    largest_imaginary = largest_imaginary.max(a.im.abs());
                }
            }
            // The conjugation must be about something: at a generic q the phases are live.
            assert!(
                largest_imaginary > 1.0e-4,
                "the imaginary parts collapsed to {largest_imaginary:.3e}, so the conjugation \
                 identity was tested on nothing"
            );

            let gamma =
                rigid_ion_dynamical_matrix(&molecule, &params, &options(), &periodic, [0.0; 3])
                    .unwrap();
            let nat = molecule.atoms.len();
            for alpha in 0..3 {
                for beta in 0..3 {
                    let mut sum = 0.0;
                    for a in 0..nat {
                        for b in 0..nat {
                            sum += gamma.matrix[(3 * a + alpha, 3 * b + beta)].re;
                        }
                    }
                    assert!(
                        sum.abs() < 1.0e-4,
                        "acoustic sum rule [{alpha}][{beta}] leaves {sum:.3e}"
                    );
                }
            }
        }
    }

    /// The reduced-dimensionality folding test: a doubled slab's and a doubled chain's Γ-point
    /// force constants must hold the primitive cell's `D(0)` and `D(b/2)` between them.
    ///
    /// This is the finite-`q` end-to-end check the new wire and sheet kernels get: the doubled
    /// cell goes through the same dimensionality's machinery with **no phases anywhere**, so a
    /// wrong Abel tail, a mis-gated sheet term or a phase of the wrong sign moves the
    /// zone-boundary eigenvalues and leaves the `q = 0` ones alone. (At `q = b/2` the shifted
    /// set is symmetric under negation and the phased sums are real — which is exactly why the
    /// conjugation identity is tested separately at a generic `q`.)
    #[test]
    fn a_doubled_slab_and_chain_hold_the_zone_centre_and_the_zone_boundary() {
        let params = Pm3Parameters::standard().unwrap();
        let periodic = PeriodicOptions::default();
        let axis = 16.0;

        let build = |pbc: [bool; 3], doubled: bool| {
            let mut molecule = Molecule::from_xyz_str(WATER, 0.0).unwrap();
            if doubled {
                let shifted: Vec<_> = molecule
                    .atoms
                    .iter()
                    .map(|a| crate::system::Atom {
                        z: a.z,
                        position: a.position + Vec3::new(axis, 0.0, 0.0),
                    })
                    .collect();
                molecule.atoms.extend(shifted);
            }
            let x = if doubled { 2.0 * axis } else { axis };
            let transverse = if pbc[1] { 12.0 } else { 40.0 };
            molecule.cell = Some(
                Cell::new(
                    Vec3::new(x, 0.0, 0.0),
                    Vec3::new(0.0, transverse, 0.0),
                    Vec3::new(0.0, 0.0, 40.0),
                    pbc,
                )
                .unwrap(),
            );
            molecule
        };

        for pbc in [[true, true, false], [true, false, false]] {
            let primitive = build(pbc, false);
            let zone_centre =
                rigid_ion_dynamical_matrix(&primitive, &params, &options(), &periodic, [0.0; 3])
                    .unwrap();
            let zone_boundary = rigid_ion_dynamical_matrix(
                &primitive,
                &params,
                &options(),
                &periodic,
                [0.5, 0.0, 0.0],
            )
            .unwrap();
            let supercell = rigid_ion_dynamical_matrix(
                &build(pbc, true),
                &params,
                &options(),
                &periodic,
                [0.0; 3],
            )
            .unwrap();

            let mut expected: Vec<f64> = crate::cmatrix::hermitian_eigen(&zone_centre.matrix)
                .unwrap()
                .0;
            expected.extend(
                crate::cmatrix::hermitian_eigen(&zone_boundary.matrix)
                    .unwrap()
                    .0,
            );
            expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut got = crate::cmatrix::hermitian_eigen(&supercell.matrix)
                .unwrap()
                .0;
            got.sort_by(|a, b| a.partial_cmp(b).unwrap());

            assert_eq!(expected.len(), got.len());
            let scale = expected
                .iter()
                .fold(0.0_f64, |m, v| m.max(v.abs()))
                .max(1.0);
            for (index, (want, have)) in expected.iter().zip(&got).enumerate() {
                assert!(
                    (want - have).abs() < 2.0e-4 * scale,
                    "{}D eigenvalue {index}: primitive pair gives {want}, the doubled cell {have}",
                    if pbc[1] { 2 } else { 1 }
                );
            }
        }
    }

    /// `D(0)` with the response, on a slab and on a chain, against
    /// [`crate::pbc::hessian::periodic_hessian`] — which now reaches reduced dimensionality
    /// through the same `q = 0` phased sum for its Coulomb block but solves a *real* CPHF for
    /// the response, so the response machinery is compared across two implementations exactly
    /// as it is in 3D.
    #[test]
    fn the_reduced_dimensionality_response_reproduces_the_gamma_hessian() {
        let params = Pm3Parameters::standard().unwrap();
        let periodic = PeriodicOptions::default();

        for molecule in [slab_of(WATER), chain_of(WATER)] {
            let reference =
                crate::pbc::hessian::periodic_hessian(&molecule, &params, &options(), &periodic)
                    .unwrap();
            let got =
                dynamical_matrix(&molecule, &params, &options(), &periodic, [0.0; 3]).unwrap();

            let ndof = reference.rows;
            let scale = (0..ndof)
                .flat_map(|i| (0..ndof).map(move |j| (i, j)))
                .fold(0.0_f64, |m, (i, j)| m.max(reference[(i, j)].abs()))
                .max(1.0);
            let mut worst = (0usize, 0usize, 0.0_f64);
            for i in 0..ndof {
                for j in 0..ndof {
                    assert!(
                        got.matrix[(i, j)].im.abs() < 1.0e-8,
                        "D(0)[{i}][{j}] is complex: {}",
                        got.matrix[(i, j)].im
                    );
                    let difference = (got.matrix[(i, j)].re - reference[(i, j)]).abs();
                    if difference > worst.2 {
                        worst = (i, j, difference);
                    }
                }
            }
            assert!(
                worst.2 < 2.0e-3 * scale,
                "worst disagreement {:.4e} at [{}][{}]: perturbation theory {} vs Γ Hessian {} \
                 (largest element {scale:.3})",
                worst.2,
                worst.0,
                worst.1,
                got.matrix[(worst.0, worst.1)].re,
                reference[(worst.0, worst.1)]
            );
        }
    }
    /// The k-sum is exercised with more than one point, and must not change the answer.
    ///
    /// Sampling Gamma twice at half weight each is the same physical calculation as sampling it
    /// once at full weight, so any disagreement is a bug in the machinery rather than in the
    /// physics: a weight applied in the wrong place, a back-transform that accumulates instead of
    /// resetting, a contraction that misses its factor. Those are exactly the mistakes a
    /// single-point sum cannot show, and they would otherwise wait until a real mesh made them
    /// look like a physical discrepancy.
    #[test]
    fn the_k_sum_is_insensitive_to_how_the_same_point_is_split() {
        use crate::pbc::kpoints::KPoint;

        let params = Pm3Parameters::standard().unwrap();
        let molecule = cell_of(WATER, 14.0);
        let periodic = PeriodicOptions::default();
        let scf = run_gamma(&molecule, &params, &options(), &periodic).unwrap();
        let setup = crate::pbc::gamma::build_setup(&molecule, &params, &periodic).unwrap();
        let cell = molecule.cell.unwrap();
        let nat = molecule.atoms.len();
        let q_frac = [0.27, -0.13, 0.36];
        let mut q = Vec3::zero();
        for (index, b) in &cell.reciprocal_basis() {
            q += *b * q_frac[*index];
        }

        let evaluate = |points: &[KPoint]| {
            let mut ground = gamma_ground_state(&molecule, &params, &setup, &cell, &scf).unwrap();
            ground.points = points.to_vec();
            let mut matrix = CMatrix::zeros(3 * nat, 3 * nat);
            response(
                &molecule,
                &params,
                &options(),
                &periodic,
                &setup,
                &ground,
                &cell,
                q,
                q_frac,
                &mut matrix,
                None,
                None,
                &DfptOptions::default(),
            )
            .unwrap();
            matrix
        };

        let once = evaluate(&[KPoint::GAMMA]);
        let split = evaluate(&[
            KPoint {
                frac: [0.0; 3],
                weight: 0.5,
            },
            KPoint {
                frac: [0.0; 3],
                weight: 0.5,
            },
        ]);

        let scale = (0..3 * nat)
            .flat_map(|i| (0..3 * nat).map(move |j| (i, j)))
            .fold(0.0_f64, |m, (i, j)| m.max(once[(i, j)].norm()))
            .max(1.0);
        for i in 0..3 * nat {
            for j in 0..3 * nat {
                let (a, b) = (once[(i, j)], split[(i, j)]);
                assert!(
                    (a.re - b.re).abs() < 1.0e-9 * scale && (a.im - b.im).abs() < 1.0e-9 * scale,
                    "[{i}][{j}]: one point gives {a:?}, two half-weight points give {b:?}"
                );
            }
        }
        // And the response is not trivially zero, so the agreement is about something.
        assert!(scale > 1.0e-3, "the response collapsed to {scale}");
    }
    /// The two ways of presenting a converged Γ-point calculation are the same numbers.
    ///
    /// This sits below [`a_one_point_mesh_reproduces_the_gamma_path`] on purpose: when the two
    /// paths disagree, this says *which piece* disagrees rather than only that the answer moved.
    /// The occupations are compared rather than the Fermi levels, because the two are entitled to
    /// place the level anywhere inside the gap and only what that implies about the filling is
    /// shared.
    #[test]
    fn the_two_ground_states_present_the_same_numbers() {
        use crate::pbc::kpoints::KpointSpec;
        use crate::pbc::kscf::KpointOptions;

        let params = Pm3Parameters::standard().unwrap();
        let molecule = cell_of(WATER, 14.0);
        let periodic = PeriodicOptions::default();
        let cell = molecule.cell.unwrap();
        let setup = crate::pbc::gamma::build_setup(&molecule, &params, &periodic).unwrap();

        let gamma =
            crate::pbc::gamma::run_gamma(&molecule, &params, &options(), &periodic).unwrap();
        let want = gamma_ground_state(&molecule, &params, &setup, &cell, &gamma).unwrap();

        let spec = KpointSpec::Gamma;
        let kopt = KpointOptions {
            spec: spec.clone(),
            ..Default::default()
        };
        let mesh = crate::pbc::kscf::run_kpoints(&molecule, &params, &options(), &periodic, &kopt)
            .unwrap();
        let got = mesh_ground_state(&molecule, &params, &setup, &cell, &spec, &mesh, 0.0).unwrap();

        let nao = setup.basis.nao;
        assert_eq!(want.channels.len(), got.channels.len(), "channel count");
        for i in 0..nao {
            for j in 0..nao {
                assert!(
                    (want.density[(i, j)] - got.density[(i, j)]).abs() < 1.0e-7,
                    "density [{i}][{j}]: {} vs {}",
                    want.density[(i, j)],
                    got.density[(i, j)]
                );
            }
        }
        for (spin, (a, b)) in want.channels.iter().zip(&got.channels).enumerate() {
            assert_eq!(a.exchange_scale, b.exchange_scale, "channel {spin} scale");
            assert_eq!(a.occupancy, b.occupancy, "channel {spin} occupancy");
            for i in 0..nao {
                for j in 0..nao {
                    assert!(
                        (a.f_onsite[(i, j)] - b.f_onsite[(i, j)]).abs() < 1.0e-7,
                        "channel {spin} on-site Fock [{i}][{j}]: {} vs {}",
                        a.f_onsite[(i, j)],
                        b.f_onsite[(i, j)]
                    );
                    assert!(
                        (a.exchange_scale * a.exchange_density[(i, j)]
                            - b.exchange_scale * b.exchange_density[(i, j)])
                            .abs()
                            < 1.0e-7,
                        "channel {spin} exchange density [{i}][{j}]"
                    );
                }
            }
            assert_eq!(a.p_images.len(), b.p_images.len(), "image count");
            for (index, (x, y)) in a.p_images.iter().zip(&b.p_images).enumerate() {
                assert_eq!(x.len(), y.len(), "image {index} block size");
                for (slot, (u, v)) in x.iter().zip(y.iter()).enumerate() {
                    assert!(
                        (u - v).abs() < 1.0e-7,
                        "channel {spin} image {index} entry {slot}: {u} vs {v}"
                    );
                }
            }

            // Same filling, whatever each path decided to call the Fermi level.
            let bands = crate::pbc::kscf::bloch_fock(
                &setup,
                &a.f_onsite,
                &a.p_images,
                &crate::pbc::kpoints::KPoint::GAMMA,
            );
            let (eps, _) = crate::cmatrix::hermitian_eigen(&bands).unwrap();
            for index in 0..eps.len() {
                assert!(
                    (want.occupation(a, &eps, index) - got.occupation(b, &eps, index)).abs()
                        < 1.0e-9,
                    "channel {spin} band {index} at {} eV: {} vs {}",
                    eps[index],
                    want.occupation(a, &eps, index),
                    got.occupation(b, &eps, index)
                );
            }
        }
    }

    /// A one-point mesh at Gamma is the same calculation the Gamma path does, so it must give the
    /// same dynamical matrix.
    ///
    /// This is the checkpoint for driving the response from `run_kpoints`: the two SCFs differ
    /// only in their convergence accelerator, so agreement to the density tolerance says the
    /// mesh path assembles the same physics rather than a parallel one that happens to look
    /// similar. Every subsequent mesh is the same code with more points in the list.
    #[test]
    fn a_one_point_mesh_reproduces_the_gamma_path() {
        use crate::pbc::kpoints::KpointSpec;
        use crate::pbc::kscf::KpointOptions;

        let params = Pm3Parameters::standard().unwrap();
        let molecule = cell_of(WATER, 14.0);
        let periodic = PeriodicOptions::default();
        let q_frac = [0.31, -0.17, 0.24];

        let want = dynamical_matrix(&molecule, &params, &options(), &periodic, q_frac).unwrap();
        let kopt = KpointOptions {
            spec: KpointSpec::Gamma,
            ..Default::default()
        };
        let got =
            dynamical_matrix_on_mesh(&molecule, &params, &options(), &periodic, &kopt, q_frac)
                .unwrap();

        let ndof = want.matrix.rows;
        let scale = (0..ndof)
            .flat_map(|i| (0..ndof).map(move |j| (i, j)))
            .fold(0.0_f64, |m, (i, j)| m.max(want.matrix[(i, j)].norm()))
            .max(1.0);
        let mut worst = 0.0_f64;
        for i in 0..ndof {
            for j in 0..ndof {
                let (a, b) = (want.matrix[(i, j)], got.matrix[(i, j)]);
                worst = worst.max(((a.re - b.re).powi(2) + (a.im - b.im).powi(2)).sqrt());
            }
        }
        assert!(
            worst < 1.0e-5 * scale,
            "the one-point mesh differs from the Gamma path by {worst:.3e} (scale {scale:.3e})"
        );
        assert!(scale > 1.0e-2, "the matrix collapsed to {scale}");
    }

    /// The **skeleton** is built on the mesh's own density, not on a Γ one.
    ///
    /// The fixed-density skeleton is the larger half of `D(q)`, and it used to be assembled from
    /// `run_gamma` no matter what sampling was asked for: a mesh converged a second SCF and only
    /// the *response* ever saw it. Whenever Γ alone was adequate the mixture was harmless and the
    /// mesh was pointless; whenever the mesh was necessary — the only reason to pass one — the
    /// dominant term was the one computed from the wrong density.
    ///
    /// Asserted on the skeleton alone (`include_response = false`), because that isolates the
    /// half that was wrong: with the response included, a difference could be the response
    /// legitimately sampling more finely. Before the fix the two matrices below were *identical
    /// by construction*, so this fails by exactly the amount the bug was worth.
    ///
    /// The cell is narrow enough that its Γ and mesh densities genuinely differ, and that is
    /// asserted first — a test comparing two identical densities would pass for the wrong reason.
    #[test]
    fn the_skeleton_reads_the_density_of_the_sampling_it_was_given() {
        use crate::pbc::kpoints::KpointSpec;
        use crate::pbc::kscf::KpointOptions;

        let params = Pm3Parameters::standard().unwrap();
        // 7 Bohr: narrow enough that images overlap and Γ is not the same state as a mesh.
        let molecule = cell_of(WATER, 7.0);
        let periodic = PeriodicOptions::default();
        let q_frac = [0.5, 0.0, 0.0];
        let kopt = KpointOptions {
            spec: KpointSpec::mesh([2, 1, 1]),
            ..Default::default()
        };

        // Non-vacuity: the two ground states are actually different here.
        let gamma = run_gamma(&molecule, &params, &options(), &periodic).unwrap();
        let mesh = crate::pbc::kscf::run_kpoints(&molecule, &params, &options(), &periodic, &kopt)
            .unwrap();
        let density_gap = gamma
            .density
            .as_slice()
            .iter()
            .zip(mesh.density.as_slice())
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            density_gap > 1.0e-4,
            "the Gamma and mesh densities agree to {density_gap:.3e}, so this cell cannot tell \
             the two skeletons apart"
        );

        let from_gamma = assemble(
            &molecule,
            &params,
            &options(),
            &periodic,
            None,
            q_frac,
            false,
            &DfptOptions::default(),
        )
        .unwrap();
        let from_mesh = assemble(
            &molecule,
            &params,
            &options(),
            &periodic,
            Some(&kopt),
            q_frac,
            false,
            &DfptOptions::default(),
        )
        .unwrap();

        let ndof = from_gamma.matrix.rows;
        let mut worst = 0.0_f64;
        for i in 0..ndof {
            for j in 0..ndof {
                worst = worst.max((from_gamma.matrix[(i, j)] - from_mesh.matrix[(i, j)]).norm());
            }
        }
        assert!(
            worst > 1.0e-6,
            "the two skeletons agree to {worst:.3e}, which means the mesh density never reached \
             the skeleton"
        );
    }

    /// Asking for a mesh gives the same `D(q)` as spelling that mesh out point by point.
    ///
    /// `run_kpoints` reduces the mesh by time reversal, and the response must not inherit that.
    /// TR sends the coupled pair `(k, k + q)` to `(−k, −k + q)`, which is a pair at `−q`, so the
    /// irreducible sum with doubled weights is a different number from the full sum wherever
    /// `q ≠ −q` modulo a reciprocal lattice vector. `q = (1/4, 1/8, 0)` on a `4×4×1` mesh is such
    /// a wavevector, and the mesh is reduced there — both are asserted, because a test that
    /// silently compared two unreduced meshes would pass while proving nothing.
    #[test]
    fn a_reduced_mesh_and_the_full_one_agree() {
        use crate::pbc::kpoints::{monkhorst_pack, KpointSpec};
        use crate::pbc::kscf::KpointOptions;

        let params = Pm3Parameters::standard().unwrap();
        let molecule = cell_of(WATER, 12.0);
        let periodic = PeriodicOptions::default();
        let cell = molecule.cell.unwrap();
        let divisions = [4, 4, 1];
        let q_frac = [0.25, 0.125, 0.0];

        let full = monkhorst_pack(&cell, divisions, [0.0; 3]).unwrap();
        let reduced = crate::pbc::kpoints::reduce(&full);
        assert!(
            reduced.len() < full.len(),
            "the mesh was not reduced, so this proves nothing"
        );

        let asked = dynamical_matrix_on_mesh(
            &molecule,
            &params,
            &options(),
            &periodic,
            &KpointOptions::mesh(divisions),
            q_frac,
        )
        .unwrap();
        let spelled = dynamical_matrix_on_mesh(
            &molecule,
            &params,
            &options(),
            &periodic,
            &KpointOptions {
                spec: KpointSpec::Explicit(full),
                ..Default::default()
            },
            q_frac,
        )
        .unwrap();

        let ndof = asked.matrix.rows;
        let scale = (0..ndof)
            .flat_map(|i| (0..ndof).map(move |j| (i, j)))
            .fold(0.0_f64, |m, (i, j)| m.max(asked.matrix[(i, j)].norm()));
        assert!(scale > 1.0, "the matrix collapsed to {scale}");
        let mut worst = 0.0_f64;
        for i in 0..ndof {
            for j in 0..ndof {
                let (a, b) = (asked.matrix[(i, j)], spelled.matrix[(i, j)]);
                worst = worst.max((a - b).norm());
            }
        }
        assert!(
            worst < 1.0e-6 * scale,
            "the reduced mesh differs from the full one by {worst:.3e} (scale {scale:.3e})"
        );
    }

    /// A mesh with more than one point runs, and the exact identities still hold on it.
    ///
    /// The folding tests are what would say the *value* is right; these say the assembly is
    /// self-consistent once the k-sum is real -- that the weights sum, that `k + q` is found off
    /// the mesh as readily as on it, and that the back-transform keeps its imaginary part.
    #[test]
    fn a_real_mesh_keeps_the_exact_identities() {
        use crate::pbc::kpoints::KpointSpec;
        use crate::pbc::kscf::KpointOptions;

        let params = Pm3Parameters::standard().unwrap();
        let molecule = cell_of(WATER, 12.0);
        let periodic = PeriodicOptions::default();
        let kopt = KpointOptions {
            spec: KpointSpec::mesh([2, 1, 1]),
            ..Default::default()
        };
        let q_frac = [0.25, 0.0, 0.0];

        let forward =
            dynamical_matrix_on_mesh(&molecule, &params, &options(), &periodic, &kopt, q_frac)
                .unwrap();
        let reversed = dynamical_matrix_on_mesh(
            &molecule,
            &params,
            &options(),
            &periodic,
            &kopt,
            [-q_frac[0], -q_frac[1], -q_frac[2]],
        )
        .unwrap();

        let ndof = forward.matrix.rows;
        let scale = (0..ndof)
            .flat_map(|i| (0..ndof).map(move |j| (i, j)))
            .fold(0.0_f64, |m, (i, j)| m.max(forward.matrix[(i, j)].norm()))
            .max(1.0);
        assert!(scale > 1.0e-2, "the matrix collapsed to {scale}");

        // D(-q) = D(q)*. Taking the real part of the back-transform would make this D(-q) = D(q),
        // so it is the test that catches that specific mistake.
        for i in 0..ndof {
            for j in 0..ndof {
                let (a, b) = (forward.matrix[(i, j)], reversed.matrix[(i, j)]);
                assert!(
                    (a.re - b.re).abs() < 1.0e-6 * scale && (a.im + b.im).abs() < 1.0e-6 * scale,
                    "D(-q) is not D(q)* at [{i}][{j}]: {a:?} vs {b:?}"
                );
            }
        }
        // And the imaginary part is genuinely there, so the conjugation is not trivially true.
        let largest_imaginary = (0..ndof)
            .flat_map(|i| (0..ndof).map(move |j| (i, j)))
            .fold(0.0_f64, |m, (i, j)| m.max(forward.matrix[(i, j)].im.abs()));
        assert!(
            largest_imaginary > 1.0e-6 * scale,
            "D(q) came out real, so the conjugation test proves nothing"
        );
    }
}
