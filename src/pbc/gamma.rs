// SPDX-License-Identifier: GPL-3.0-or-later

//! Γ-point periodic SCF.
//!
//! # Why the Γ point needs so little new machinery
//!
//! At `k = 0` the Bloch orbital is just an atomic orbital summed over every cell, so the Fock
//! element between two of them is the ordinary Fock element summed over every lattice
//! translation:
//!
//! ```text
//! F_μν(Γ) = Σ_T F_μν(0, T)
//! ```
//!
//! The image index never touches the AO index — both orbitals still belong to atoms of the
//! reference cell — so once the two-center tables are summed over `T`, the molecular Fock
//! builder in [`crate::fock`] applies verbatim. Almost all of this module is about *building*
//! those summed tables, not about a new SCF.
//!
//! # The counting rule, stated once
//!
//! The neighbour list visits every ordered pair-with-image, so a given physical interaction is
//! seen twice: once as `(a, b, T)` and once as `(b, a, −T)`. Each visit writes its term into
//! both destination blocks. Halving every such contribution is therefore exactly right, and
//! uniformly so — including for `a = b`, where the two visits are `±T` and the two writes land
//! in the same block. Quantities that are summed once per *interaction* rather than accumulated
//! into blocks — the core–core energy — use the unique half instead.
//!
//! # An atom interacts with its own images
//!
//! `a = b, T ≠ 0` pairs are real two-electron terms between orbitals that all sit on one atom.
//! They belong with the one-center block, not in the two-center pair loop, which would scatter
//! them into that block twice. They are kept in [`Setup::self_images`] and applied alongside the
//! ordinary one-center integrals.
//!
//! # Coulomb is split; exchange is not
//!
//! Both halves of the NDDO two-electron term are built from the same integrals `(μν|λσ)`, but
//! they cannot be treated the same way under a lattice sum. Subtracting the point-charge model
//! from the integral table removes it from the Coulomb term *and* from the exchange term, while
//! the Ewald sum only puts the Coulomb part back — leaving the exchange short by exactly the
//! point-charge contribution. For H₂ that showed up as a 9.73 eV error, precisely half the
//! `19.46 eV` point-charge pair energy.
//!
//! So the two are separated at the source:
//!
//! * **Coulomb** uses the screened correction `f·(W_NDDO − W_point)` summed over images, with
//!   the lattice sum supplying the rest.
//! * **Exchange** uses the **full** `W_NDDO` inside a cutoff and gets no lattice sum at all.
//!   Nothing is missing: exchange is weighted by the inter-atomic density-matrix element
//!   `P_AB`, which decays exponentially, so the long-range exchange it would contribute is zero
//!   for exactly the reason the Coulomb one is not.
//!
//! # When Γ alone is enough
//!
//! One k-point cannot resolve the density matrix by image. Bloch theory gives
//!
//! ```text
//! P(0, T) = Σ_k w_k e^{−ik·T} P(k)
//! ```
//!
//! and with `k = Γ` the only term available, `P(0, T) = P(Γ)` for **every** `T`. That is the
//! defining approximation of Γ-only sampling, and it is what limits this module.
//!
//! It costs nothing in the Coulomb terms, which depend on the on-site density and are lattice
//! summed properly. It is fatal in the exchange, whose true weight `P(0, T)` decays exponentially
//! with `|T|` while the substitute `P(Γ)` does not decay at all. For a cubic lattice of water at
//! 7.4 Å the true weight between an atom and its own image is around `1e-6`; `P(Γ)` supplies the
//! on-site population instead, and the result is **38 eV** of spurious binding per cell. Widen the
//! cell so that no image sits inside the exchange cutoff and the error disappears completely: a
//! `2×2×2` supercell agrees with the single cell to `5.7e-11 eV` at an 18 Bohr edge and to
//! `1.4e-4 eV` at 16, against `3.9e1 eV` at 14.
//!
//! So the condition is sharp and purely geometric — **every periodic width must exceed
//! [`PeriodicOptions::short_range_cutoff`]**, which is what makes `P(Γ)` attributable to a single
//! image — and [`PeriodicResult::gamma_margin`] reports it. Nothing about the SCF detects a
//! violation: it converges cleanly to a well-defined wrong answer.
//!
//! This is not a defect peculiar to NDDO. It is the ordinary reason a Γ-only calculation needs a
//! supercell, and a crystal whose bonding runs through its images — graphene, a metal, a chain —
//! genuinely requires a k-mesh rather than a bigger cell.
//!
//! # Where the electrostatics comes from
//!
//! The long-range Coulomb is not in these tables at all: [`crate::pbc::screen`] has subtracted
//! the point-charge model from them and [`crate::pbc::ewald`] sums that model over the lattice.
//! The Ewald half reaches the SCF as a potential at each multipole site, and *which* part of the
//! energy expression it joins depends on what produced it:
//!
//! * the potential from the **cores** is linear in the density, so it belongs in `H_core` and is
//!   computed once per geometry;
//! * the potential from the **electrons** is quadratic, so it belongs in the Fock only.
//!
//! Putting the electron term in both — the obvious mistake, since `F = H + G` makes it look like
//! it should appear twice — double-counts the electron–core energy while leaving the Fock matrix
//! itself correct, so the SCF still converges and only the energy is wrong. Writing the energy as
//! `E = ½ Σ P (H + F)` with `H` strictly the one-electron part is what keeps that honest.

use crate::basis::Basis;
use crate::constants::{EV_TO_KCAL, PM3_EV};
use crate::corrections::periodic::{periodic_correction_energy, CorrectionCutoffs};
use crate::error::{Pm3Error, Result};
use crate::integrals::pack;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::neighbor::NeighborList;
use crate::params::{Pm3Element, Pm3Parameters};
use crate::pbc::ewald::{ChargeSite, EwaldOutput, EwaldParams};
use crate::pbc::multipole::AtomSites;
use crate::pbc::screen::{nddo_pair, point_pair, screened_core_core, SwitchRange};
use crate::scf::{commutator, diis_coeffs, Pm3Options, Reference};
use crate::system::Molecule;

/// Knobs specific to the periodic path; the electronic-structure ones stay in [`Pm3Options`].
#[derive(Clone, Copy, Debug)]
pub struct PeriodicOptions {
    /// Where the Klopman–Ohno correction is handed over to the lattice sum.
    pub switch: SwitchRange,
    /// Ewald splitting and cutoffs. `None` derives them from the cell, which is what keeps the
    /// reciprocal sum affordable as the cell grows — see [`EwaldParams::for_cell`].
    pub ewald: Option<EwaldParams>,
    /// Cutoff (Bohr) for the resonance `β·S` and the exchange, both of which follow the overlap
    /// and so decay exponentially.
    pub short_range_cutoff: f64,
    /// Cluster radii for the classical D3/H4/X lattice sums.
    pub correction_cutoffs: CorrectionCutoffs,
}

/// Default resonance/exchange cutoff (Bohr). Valence Slater overlaps are below `1e-12` well
/// inside this for every PM3 element.
pub const DEFAULT_SHORT_RANGE_CUTOFF: f64 = 14.0;

impl Default for PeriodicOptions {
    fn default() -> Self {
        Self {
            switch: SwitchRange::default(),
            ewald: None,
            short_range_cutoff: DEFAULT_SHORT_RANGE_CUTOFF,
            correction_cutoffs: CorrectionCutoffs::default(),
        }
    }
}

/// Result of a Γ-point periodic calculation. All energies are **per unit cell**.
#[derive(Clone, Debug)]
pub struct PeriodicResult {
    pub density: Matrix,
    /// `P^α − P^β` per cell, for an unrestricted calculation only.
    pub spin_density: Option<Matrix>,
    /// True when the unrestricted path was used.
    pub unrestricted: bool,
    /// α-spin orbital energies. For a restricted calculation these are *the* orbital energies.
    pub mo_energies: Vec<f64>,
    /// β-spin orbital energies, for an unrestricted calculation. `None` when restricted, where
    /// they would be a copy of the α ones.
    ///
    /// The molecular [`crate::scf::Pm3Result`] has carried these since 0.2.1; this did not, and
    /// the β channel was discarded the moment it was diagonalized — which is what made the
    /// reported frontier α-only.
    pub mo_energies_beta: Option<Vec<f64>>,
    pub mo_coeff: Matrix,
    pub n_occ: usize,
    /// Electronic energy per cell (eV), `½ Σ P (H + F)`.
    pub electronic_ev: f64,
    /// Core–core energy per cell (eV): the short-range remainder plus the cores' Ewald sum.
    pub core_ev: f64,
    /// Classical D3/H4/X correction energy per cell (eV), lattice-summed.
    pub correction_ev: f64,
    /// The full Ewald contribution (cores and electrons), reported separately because it is the
    /// piece carrying a boundary convention.
    pub ewald_ev: f64,
    pub total_ev: f64,
    pub heat_of_formation_kcal: f64,
    pub charges: Vec<f64>,
    pub homo_ev: Option<f64>,
    pub lumo_ev: Option<f64>,
    pub iterations: usize,
    pub converged: bool,
    /// Narrowest periodic width minus the resonance/exchange cutoff, in Bohr.
    ///
    /// **Positive means the Γ point alone is enough; negative means it is not** — see the module
    /// note "When Γ alone is enough". This is a property of the cell and the cutoff, not of the
    /// convergence: nothing in the SCF reacts to a violation, so this is the one number that says
    /// whether a Γ-only result can be trusted without a k-mesh or a larger supercell.
    pub gamma_margin: f64,
}

/// The image-summed two-electron tables for one atom pair (or one atom against its own images).
///
/// Two tables rather than one, because Coulomb and exchange are summed differently — see the
/// module note.
pub(crate) struct PairTables {
    pub a: usize,
    pub b: usize,
    /// `Σ_T f·(W_NDDO − W_point)`: the Coulomb correction, with the lattice sum supplying the
    /// point-charge part it leaves out.
    pub coulomb: Vec<f64>,
    /// `Σ_T W_NDDO` inside the exchange cutoff: the full integral, because exchange has no
    /// lattice-sum counterpart and needs none.
    pub exchange: Vec<f64>,
    pub norb_i: usize,
    pub norb_j: usize,
    pub npack_j: usize,
}

/// One `(a, b, T)` block of the two terms that connect *different* cells.
///
/// Everything else in the Fock matrix — the one-center integrals, the electron–core attraction,
/// the Coulomb term, the Ewald potential — writes into an on-site block and so belongs to `T = 0`
/// whatever image produced it. Only the resonance and the exchange put a value at `(0, T)`, and
/// only those two therefore have to be kept image by image for
/// `F(k) = Σ_T e^{ik·T} F(T)`.
///
/// Both tables are stored in `a`-major orientation, `a` being the atom in the reference cell.
pub(crate) struct ImageBlock {
    pub a: usize,
    pub b: usize,
    pub t: [i32; 3],
    /// `½(β_μ + β_ν) S_μν`, `norb_a × norb_b` row-major.
    pub resonance: Vec<f64>,
    /// The full NDDO two-electron table, packed.
    pub exchange: Vec<f64>,
    pub norb_a: usize,
    pub norb_b: usize,
    pub npack_b: usize,
}

/// Everything derived from the geometry alone, built once per calculation.
pub(crate) struct Setup {
    pub basis: Basis,
    pub atom_sites: Vec<AtomSites>,
    /// Flattened site list; charges are rewritten each iteration.
    pub sites: Vec<ChargeSite>,
    /// First site index of each atom.
    pub site_offset: Vec<usize>,
    /// The cell these tables were built for. Carried so every consumer uses the same one — a
    /// lattice sum run against a different cell than the tables assume is silent nonsense.
    pub cell: crate::cell::Cell,
    /// Γ-point one-electron Hamiltonian: atomic `U`, the screened electron–core attraction, the
    /// image-summed resonance `Σ_T β·S`, and the potential the cores exert through the lattice
    /// sum.
    pub h_core: Matrix,
    /// The same thing with the resonance left out — that is, everything that lands on a `T = 0`
    /// block. The k-point path needs this, because it takes the resonance from [`Setup::images`]
    /// with a Bloch phase instead of summed.
    pub h_onsite: Matrix,
    /// Resonance and exchange resolved by image; see [`ImageBlock`].
    pub images: Vec<ImageBlock>,
    /// Two-center tables for `a < b`.
    pub pairs: Vec<PairTables>,
    /// Tables for each atom against its own images, structurally one-center.
    pub self_images: Vec<Option<PairTables>>,
    /// Core–core energy per cell (eV).
    pub core_ev: f64,
    /// The cores' Ewald energy alone.
    pub core_ewald_ev: f64,
    /// Ewald parameters actually used, resolved from the cell when the caller left them open.
    pub ewald_params: EwaldParams,
    /// The geometry half of the lattice sum, built once and reused by every SCF iteration.
    /// Only the charges change between them; see [`crate::pbc::ewald::EwaldContext`].
    pub ewald_context: crate::pbc::ewald::EwaldContext,
    /// See [`PeriodicResult::gamma_margin`].
    pub gamma_margin: f64,
}

/// How many electrons the cell holds and how they are split by spin.
pub(crate) struct Occupancy {
    /// Electrons per cell. For a charged cell this is `Σ Z_val − Q`.
    pub n_elec: f64,
    pub n_alpha: usize,
    pub n_beta: usize,
    pub unrestricted: bool,
}

/// Resolve charge, multiplicity, and reference into an electron count.
///
/// `Molecule::charge` means the **net charge per unit cell** for a periodic system, which is why
/// this is shared between the Γ-point and k-point paths rather than duplicated: getting the two
/// to disagree about what a charged cell contains would be very hard to notice.
pub(crate) fn occupancy(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
) -> Result<Occupancy> {
    let charge = if options.charge != 0.0 {
        options.charge
    } else {
        molecule.charge
    };
    let multiplicity = options.multiplicity.max(molecule.multiplicity).max(1);

    let mut n_elec = 0.0;
    for atom in &molecule.atoms {
        n_elec += params.element(atom.z)?.core_charge;
    }
    n_elec -= charge;
    let n_elec_int = n_elec.round() as i64;
    if (n_elec - n_elec_int as f64).abs() > 1.0e-6 || n_elec_int < 0 {
        return Err(Pm3Error::InvalidInput(format!(
            "invalid electron count per cell: {n_elec}"
        )));
    }
    let n_unpaired = (multiplicity - 1) as i64;
    if (n_elec_int - n_unpaired) < 0 || (n_elec_int - n_unpaired) % 2 != 0 {
        return Err(Pm3Error::InvalidInput(format!(
            "electron count {n_elec_int} per cell is incompatible with multiplicity {multiplicity}"
        )));
    }
    let n_alpha = ((n_elec_int + n_unpaired) / 2) as usize;
    let n_beta = ((n_elec_int - n_unpaired) / 2) as usize;
    let unrestricted = match options.reference {
        Reference::Auto => n_alpha != n_beta,
        Reference::Uhf => true,
        Reference::Rhf => {
            if n_alpha != n_beta {
                return Err(Pm3Error::InvalidInput(format!(
                    "RHF requested for an open-shell cell (n_alpha={n_alpha} != n_beta={n_beta})"
                )));
            }
            false
        }
    };
    Ok(Occupancy {
        n_elec: n_elec_int as f64,
        n_alpha,
        n_beta,
        unrestricted,
    })
}

/// Run a Γ-point periodic PM3 calculation.
pub fn run_gamma(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
) -> Result<PeriodicResult> {
    crate::pbc::refuse_field(molecule, options)?;
    // A cell with no periodic direction is not an error: it is the isolated case, and running it
    // here rather than through [`crate::scf::run_pm3`] is what gives a large molecule the same
    // linear-scaling near field a crystal gets. See [`crate::cell::Cell::isolated`] and the
    // module note "What the split costs an isolated system".
    molecule.cell.ok_or_else(|| {
        Pm3Error::InvalidInput("a periodic calculation needs a cell on the molecule".to_string())
    })?;

    let Occupancy {
        n_alpha,
        n_beta,
        unrestricted,
        ..
    } = occupancy(molecule, params, options)?;
    let setup = build_setup(molecule, params, periodic)?;
    if n_alpha > setup.basis.nao {
        return Err(Pm3Error::InvalidInput(format!(
            "{n_alpha} occupied orbitals do not fit in {} basis functions",
            setup.basis.nao
        )));
    }

    let state = scf_loop(
        molecule,
        params,
        options,
        &setup,
        n_alpha,
        n_beta,
        unrestricted,
    )?;

    // Classical corrections are post-SCF and lattice-summed once per geometry; they never enter
    // the Fock matrix, so they are added here and nowhere else. Keeping the single addition site
    // is what stops a k-point loop from ever multiplying them by the mesh size.
    let correction_ev =
        periodic_correction_energy(molecule, options.variant, &periodic.correction_cutoffs);
    let total_ev = state.electronic_ev + setup.core_ev + correction_ev;
    let mut e_isol_sum = 0.0;
    let mut eheat_sum = 0.0;
    for atom in &molecule.atoms {
        let e = params.element(atom.z)?;
        e_isol_sum += e.e_isol;
        eheat_sum += e.eheat_ev;
    }

    let mut charges = vec![0.0; molecule.atoms.len()];
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let off = setup.basis.atom_offset[ia];
        let n = setup.basis.atom_norb[ia];
        let population: f64 = (0..n).map(|mu| state.density[(off + mu, off + mu)]).sum();
        charges[ia] = params.element(atom.z)?.core_charge - population;
    }

    let nao = setup.basis.nao;
    // Both spin channels. An open-shell cell has `n_α > n_β`, so its β LUMO sits below its α one
    // and the α-only frontier reported a gap that is not the gap. `pbc::kscf::band_edges` scans
    // both; this is the Γ-point version of the same rule.
    let frontier = |energies: &[f64], occupied: usize| -> (Option<f64>, Option<f64>) {
        (
            (occupied >= 1).then(|| energies[occupied - 1]),
            (occupied < nao).then(|| energies[occupied]),
        )
    };
    let (homo_alpha, lumo_alpha) = frontier(&state.mo_energies, n_alpha);
    let (homo_beta, lumo_beta) = match &state.mo_energies_beta {
        Some(beta) => frontier(beta, n_beta),
        None => (None, None),
    };
    let homo_ev = match (homo_alpha, homo_beta) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (only, None) | (None, only) => only,
    };
    let lumo_ev = match (lumo_alpha, lumo_beta) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (only, None) | (None, only) => only,
    };
    Ok(PeriodicResult {
        density: state.density,
        spin_density: state.spin_density,
        unrestricted,
        mo_energies: state.mo_energies.clone(),
        mo_coeff: state.mo_coeff,
        n_occ: n_alpha,
        electronic_ev: state.electronic_ev,
        core_ev: setup.core_ev,
        correction_ev,
        ewald_ev: setup.core_ewald_ev + state.electron_ewald_ev,
        total_ev,
        heat_of_formation_kcal: (total_ev - e_isol_sum + eheat_sum) * EV_TO_KCAL,
        charges,
        homo_ev,
        lumo_ev,
        mo_energies_beta: state.mo_energies_beta,
        iterations: state.iterations,
        converged: state.converged,
        gamma_margin: setup.gamma_margin,
    })
}

struct ScfState {
    density: Matrix,
    spin_density: Option<Matrix>,
    mo_energies: Vec<f64>,
    /// β-spin orbital energies, for an unrestricted calculation. Kept rather than discarded: the
    /// frontier of an open shell is not an α-only question, and the β channel used to be dropped
    /// on the floor the moment it was diagonalized.
    mo_energies_beta: Option<Vec<f64>>,
    mo_coeff: Matrix,
    electronic_ev: f64,
    electron_ewald_ev: f64,
    iterations: usize,
    converged: bool,
}

/// Damped SCF over the Γ-point Fock matrix, restricted or unrestricted.
///
/// The two references share every line here. An unrestricted step diagonalizes two Fock matrices
/// instead of one and occupies each with weight 1; a restricted step is the special case
/// `P^α = P^β = ½P`, which is why the same [`build_periodic_fock`] serves both — it already takes
/// the total density for Coulomb and a spin density for exchange.
///
/// The long-range field is built from the **total** density in both cases: the lattice sum sees
/// charge, not spin.
///
/// Convergence is damping for the first few cycles and then plain Pulay CDIIS. The periodic Fock
/// carries an extra density-dependent Ewald term on top of the molecular one, but `[F, P]` still
/// vanishes exactly at the fixed point, so the ordinary extrapolation applies to it unchanged.
/// Simpler than [`crate::scf`]'s A-DIIS/CDIIS hybrid, whose incremental history bookkeeping is
/// tangled with the molecular Fock builder — but worth having: damping alone needed roughly 38
/// cycles on the systems here, and every cycle costs a full lattice sum.
fn scf_loop(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    setup: &Setup,
    n_alpha: usize,
    n_beta: usize,
    unrestricted: bool,
) -> Result<ScfState> {
    let nao = setup.basis.nao;
    let total = initial_density(molecule, params, &setup.basis)?;
    let n_electrons = (n_alpha + n_beta) as f64;
    let scale = |fraction: f64| -> Matrix {
        let mut out = total.clone();
        for value in out.as_mut_slice() {
            *value *= fraction;
        }
        out
    };
    // Two different guesses, for the reason [`crate::scf`] gives at its own UHF entry point.
    //
    // A restricted run starts from the superposition of atomic densities, which is close to the
    // converged density and keeps the frozen RHF results reproducible.
    //
    // An unrestricted run cannot: scaling one SAD density by `n_α/n_elec` and `n_β/n_elec` gives
    // two densities that differ only in magnitude, so α and β see the same potential shape and
    // the iteration starts inside the spin-symmetric subspace. For the methyl radical that
    // subspace contains a *stationary point* 4.8 eV above the UHF minimum — plain damping needs
    // roughly 25 wasted cycles to be repelled from it, and CDIIS, which solves `[F,P] = 0`
    // without regard to whether the solution is a minimum, converges neatly onto it instead. So
    // occupy core-Hamiltonian orbitals to the two aufbau counts and let the unequal occupations
    // break the symmetry from the first cycle.
    let (mut p_alpha, mut p_beta) = if unrestricted {
        let (_, guess) = symmetric_eigen(&setup.h_core)?;
        (
            guess.leading_columns_gram(n_alpha, 1.0),
            guess.leading_columns_gram(n_beta, 1.0),
        )
    } else {
        (
            scale(if n_electrons > 0.0 {
                n_alpha as f64 / n_electrons
            } else {
                0.5
            }),
            scale(if n_electrons > 0.0 {
                n_beta as f64 / n_electrons
            } else {
                0.5
            }),
        )
    };

    let mut last_energy = f64::INFINITY;
    // The most recent RMS density change, kept so a failure can report how far off it was.
    let mut last_change = f64::INFINITY;
    let mut converged = false;
    let mut iterations = 0;
    let mut mo_energies = vec![0.0; nao];
    let mut mo_energies_beta: Option<Vec<f64>> = None;
    let mut mo_coeff = Matrix::zeros(nao, nao);
    let mut history: Vec<DiisSlot> = Vec::new();
    // Enough damping to hold the density steady while the Ewald potential is still moving.
    let damping = if options.damping > 0.0 {
        options.damping.clamp(0.0, 0.95)
    } else {
        0.3
    };

    // A restricted run uses the molecular path's accelerator: A-DIIS far from the fixed point,
    // Pulay CDIIS once the commutator is small.
    //
    // This is not a refinement. Plain CDIIS interpolates the Fock matrix with unconstrained
    // weights and has no notion of the energy going down, so far from convergence it can move
    // to a worse iterate; damping is the only thing holding it, and how much damping is enough
    // grows with the system. Measured on a chain of identical waters, CDIIS with the standard
    // damping converged at 56 molecules, failed at 58, converged again at 66, and failed at
    // every size beyond — an erratic pattern that says the iteration is being held rather than
    // driven. A-DIIS constrains its weights to the simplex and minimizes an energy surrogate,
    // which is what makes the large end converge at all.
    //
    // An unrestricted run keeps plain CDIIS, exactly as the molecular unrestricted path does:
    // the A-DIIS surrogate is built from `⟨D, F⟩` and there is one such product per spin, so
    // feeding it a single channel would optimize the wrong functional.
    let accelerator = if !unrestricted
        && options.use_diis
        && crate::scf::diis_depth_fits(nao, 3, options.scf_memory_mb)
    {
        Some(crate::scf::AccelHistory::new(
            crate::scf::diis_depth(nao, 3, options.scf_memory_mb),
            true,
        ))
    } else {
        None
    };
    let mut accelerator = accelerator;

    for iteration in 1..=options.max_scf {
        iterations = iteration;
        let p_total = add(&p_alpha, &p_beta);

        // The electrons' own long-range field, from the total density.
        let electrons = electron_field(setup, &p_total)?;

        let mut fock_alpha = build_periodic_fock(molecule, params, setup, &p_total, &p_alpha)?;
        add_site_potential(setup, &mut fock_alpha, &electrons);
        let mut fock_beta = if unrestricted {
            let mut beta = build_periodic_fock(molecule, params, setup, &p_total, &p_beta)?;
            add_site_potential(setup, &mut beta, &electrons);
            beta
        } else {
            fock_alpha.clone()
        };

        // The energy the convergence test watches, evaluated *here* — at the current density,
        // with the Fock matrix that density produces, before any extrapolation touches it.
        //
        // This is the only pairing that is the actual PM3 energy of an actual state. Taking it
        // after extrapolation pairs a combination of history Fock matrices with a density that
        // belongs to none of them, and the result stops moving once the extrapolation weights
        // settle — which is not the same thing as the SCF having converged. On long chains that
        // let the iteration stop early enough to leave tens of millielectronvolts on the table,
        // varying with something as inert as where the molecule sat in space.
        let electronic = 0.5
            * (p_total.frobenius_dot(&setup.h_core)
                + p_alpha.frobenius_dot(&fock_alpha)
                + p_beta.frobenius_dot(&fock_beta));

        // The `[F, P]` commutator, the same error the molecular path uses.
        let error = spin_commutator(&fock_alpha, &p_alpha, &fock_beta, &p_beta, unrestricted);
        let error_norm = error.frobenius_dot(&error).sqrt();

        if let Some(accelerator) = accelerator.as_mut() {
            let norm_squared = accelerator.push(fock_alpha.clone(), error, Some(p_alpha.clone()));
            let extrapolated = if norm_squared.sqrt() > options.adiis_switch {
                accelerator.adiis()
            } else {
                accelerator.cdiis()
            };
            // A refused solve (a singular Gram, or a history of one) leaves the plain Fock.
            fock_alpha =
                extrapolated.unwrap_or_else(|| accelerator.focks[accelerator.len() - 1].clone());
            fock_beta = fock_alpha.clone();
        } else {
            history.push(DiisSlot {
                fock_alpha: fock_alpha.clone(),
                fock_beta: unrestricted.then(|| fock_beta.clone()),
                error,
            });
            if history.len() > DIIS_DEPTH {
                history.remove(0);
            }
            if let Some((first, coefficients)) = extrapolation_weights(&history) {
                let kept = &history[first..];
                fock_alpha = extrapolate(&coefficients, kept.iter().map(|s| &s.fock_alpha));
                fock_beta = if unrestricted {
                    extrapolate(
                        &coefficients,
                        kept.iter().map(|s| {
                            s.fock_beta
                                .as_ref()
                                .expect("unrestricted slots always carry a beta Fock")
                        }),
                    )
                } else {
                    fock_alpha.clone()
                };
            }
        }

        let (energies_alpha, coefficients_alpha) = symmetric_eigen(&fock_alpha)?;
        let new_alpha = coefficients_alpha.leading_columns_gram(n_alpha, 1.0);

        let (new_beta, energies_beta) = if unrestricted {
            let (energies_beta, coefficients_beta) = symmetric_eigen(&fock_beta)?;
            let new_beta = coefficients_beta.leading_columns_gram(n_beta, 1.0);
            (new_beta, energies_beta)
        } else {
            (new_alpha.clone(), energies_alpha.clone())
        };

        let new_total = add(&new_alpha, &new_beta);
        let change = new_total.rms_difference(&p_total);

        // Damping carries the first few cycles, where the Ewald potential is still moving and the
        // extrapolation has nothing to extrapolate from. Once the commutator error is small,
        // CDIIS is doing the work and damping would only slow it down — but the handover has to
        // wait for a genuinely small error: dropping damping while the density is still far out
        // lets an open-shell run slide into the spin-symmetric solution instead of the UHF one.
        let effective_damping = if accelerator.is_some() || error_norm < DAMPING_HANDOVER {
            0.0
        } else {
            damping
        };
        damp_into(&mut p_alpha, &new_alpha, effective_damping);
        damp_into(&mut p_beta, &new_beta, effective_damping);
        mo_energies = energies_alpha;
        mo_coeff = coefficients_alpha;
        mo_energies_beta = unrestricted.then_some(energies_beta);

        if (electronic - last_energy).abs() < options.e_tol && change < options.p_tol {
            converged = true;
            break;
        }
        last_energy = electronic;
        last_change = change;
    }

    if !converged {
        // The density change, not a hardcoded `NaN`.
        //
        // This used to report `f64::NAN` unconditionally, so every non-convergence here came
        // back as "error=NaN" whether the run had blown up or merely stopped a decade short. A
        // caller cannot tell those apart from that message, and they call for opposite responses.
        //
        // The Γ margin goes in the message too, because when it is negative it is usually the
        // cause and never appears in what the user was looking at. `P(Γ)` stands in for `P(0, T)`
        // at every image, and in a cell narrower than the exchange range that substitution is
        // being asked to hold for images that genuinely overlap — a diamond or a perovskite in
        // its conventional cell, where the answer is not "damp harder" but "use a k-mesh".
        let margin = setup.gamma_margin;
        if margin <= 0.0 {
            return Err(Pm3Error::InvalidInput(format!(
                "the Gamma-point SCF did not converge after {iterations} iterations (density \
                 change {last_change:.3e} against a tolerance of {:.1e}), and the Gamma margin is \
                 {margin:+.2} Bohr. A margin at or below zero means one k-point cannot represent \
                 this cell: `P(Gamma)` is standing in for the density at images that overlap the \
                 exchange range. More iterations or damping will not fix that -- use a k-mesh \
                 (`--kpts` / `KpointOptions`), or a supercell wide enough to make the margin \
                 positive.",
                options.p_tol
            )));
        }
        return Err(Pm3Error::ScfNotConverged {
            iterations,
            error: last_change,
        });
    }

    // Evaluate the energy at the density that is actually returned, with the Fock matrix that
    // density produces — not with the last one the loop happened to hold.
    //
    // Inside the loop, `fock_alpha` is the *extrapolated* Fock (a combination of history
    // entries, which is the whole point of DIIS) and the density paired with it is the freshly
    // diagonalized `new_alpha`, while what leaves this function is the damped and extrapolated
    // `p_alpha`. Neither mismatch is visible in the convergence test, which only asks that the
    // energy stop moving: a stable set of extrapolation weights makes a *wrong* energy stop
    // moving just as convincingly as a right one.
    //
    // The error is normally far below the tolerance, which is why it went unnoticed. It is not
    // always: on a chain of 92 water molecules the reported energy was 1.03 eV above the
    // molecular path's while the two converged densities agreed to 6e-5 electrons per atom —
    // an energy discrepancy with no corresponding difference in the state. The molecular path
    // (`crate::scf::rhf_loop`) and the divide-and-conquer path have always rebuilt here; this
    // one had not.
    let density = add(&p_alpha, &p_beta);
    let electrons = electron_field(setup, &density)?;

    let electron_ewald_ev = electrons.energy_ev;
    let mut fock_alpha = build_periodic_fock(molecule, params, setup, &density, &p_alpha)?;
    add_site_potential(setup, &mut fock_alpha, &electrons);
    let fock_beta = if unrestricted {
        let mut beta = build_periodic_fock(molecule, params, setup, &density, &p_beta)?;
        add_site_potential(setup, &mut beta, &electrons);
        beta
    } else {
        fock_alpha.clone()
    };
    let electronic_ev = 0.5
        * (density.frobenius_dot(&setup.h_core)
            + p_alpha.frobenius_dot(&fock_alpha)
            + p_beta.frobenius_dot(&fock_beta));

    let spin_density = unrestricted.then(|| subtract(&p_alpha, &p_beta));
    Ok(ScfState {
        density,
        spin_density,
        mo_energies,
        mo_energies_beta,
        mo_coeff,
        electronic_ev,
        electron_ewald_ev,
        iterations,
        converged,
    })
}

/// How many Fock/error pairs the periodic CDIIS keeps.
const DIIS_DEPTH: usize = 8;

/// Commutator norm below which damping is switched off and CDIIS runs unassisted.
const DAMPING_HANDOVER: f64 = 1.0e-2;

/// Largest `Σ|c_i|` accepted from the CDIIS solve.
///
/// The extrapolation is an interpolation only in spirit — the coefficients are unconstrained in
/// sign, and once the error vectors go nearly linearly dependent the solve answers with large
/// cancelling weights that amplify whatever noise is left. Refusing those and retrying on a
/// shorter history is what keeps the tail of the SCF monotone.
const MAX_DIIS_WEIGHT: f64 = 20.0;

/// CDIIS weights for the longest suffix of the history that gives a well-behaved solve.
///
/// Returns the index the suffix starts at along with its coefficients.
fn extrapolation_weights(history: &[DiisSlot]) -> Option<(usize, Vec<f64>)> {
    for first in 0..history.len().saturating_sub(1) {
        let errors: Vec<Matrix> = history[first..]
            .iter()
            .map(|slot| slot.error.clone())
            .collect();
        let Some(coefficients) = diis_coeffs(&errors) else {
            continue;
        };
        // `diis_coeffs` returns the Lagrange multiplier as a trailing entry; the weights are
        // the leading `errors.len()`.
        let weight: f64 = coefficients
            .iter()
            .take(errors.len())
            .map(|c| c.abs())
            .sum();
        if weight.is_finite() && weight <= MAX_DIIS_WEIGHT {
            return Some((first, coefficients));
        }
    }
    None
}

/// One CDIIS history entry. `fock_beta` is `None` for a restricted run, where it would only ever
/// be a copy of the alpha matrix.
struct DiisSlot {
    fock_alpha: Matrix,
    fock_beta: Option<Matrix>,
    error: Matrix,
}

/// The CDIIS error for one SCF cycle.
///
/// Restricted runs use `[F, P^α]` directly. Unrestricted runs stack the two spin commutators into
/// one `2·nao × nao` matrix so that [`diis_coeffs`]'s Frobenius products come out as the sum over
/// both channels — the standard UHF metric — with no change to the solver.
fn spin_commutator(
    fock_alpha: &Matrix,
    p_alpha: &Matrix,
    fock_beta: &Matrix,
    p_beta: &Matrix,
    unrestricted: bool,
) -> Matrix {
    let alpha = commutator(fock_alpha, p_alpha);
    if !unrestricted {
        return alpha;
    }
    let beta = commutator(fock_beta, p_beta);
    let mut stacked = Matrix::zeros(2 * alpha.rows, alpha.cols);
    let split = alpha.as_slice().len();
    stacked.as_mut_slice()[..split].copy_from_slice(alpha.as_slice());
    stacked.as_mut_slice()[split..].copy_from_slice(beta.as_slice());
    stacked
}

/// `Σ_i c_i F_i` over the history.
fn extrapolate<'a>(coefficients: &[f64], focks: impl Iterator<Item = &'a Matrix>) -> Matrix {
    let mut out: Option<Matrix> = None;
    for (c, fock) in coefficients.iter().zip(focks) {
        let accumulator = out.get_or_insert_with(|| Matrix::zeros(fock.rows, fock.cols));
        for (slot, value) in accumulator
            .as_mut_slice()
            .iter_mut()
            .zip(fock.as_slice().iter())
        {
            *slot += c * value;
        }
    }
    out.expect("diis_coeffs only returns coefficients for a non-empty history")
}

fn add(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = a.clone();
    for (slot, value) in out.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *slot += value;
    }
    out
}

fn subtract(a: &Matrix, b: &Matrix) -> Matrix {
    let mut out = a.clone();
    for (slot, value) in out.as_mut_slice().iter_mut().zip(b.as_slice()) {
        *slot -= value;
    }
    out
}

fn damp_into(current: &mut Matrix, fresh: &Matrix, damping: f64) {
    for (slot, value) in current.as_mut_slice().iter_mut().zip(fresh.as_slice()) {
        *slot = damping * *slot + (1.0 - damping) * value;
    }
}

/// Superposition of neutral atomic densities — the guess the molecular path uses.
pub fn initial_density(
    molecule: &Molecule,
    params: &Pm3Parameters,
    basis: &Basis,
) -> Result<Matrix> {
    let mut p = Matrix::zeros(basis.nao, basis.nao);
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let elem = params.element(atom.z)?;
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
        if n == 0 {
            continue;
        }
        p[(off, off)] = elem.occ_s;
        if n >= 4 {
            let per_p = elem.occ_p / 3.0;
            for k in 1..4 {
                p[(off + k, off + k)] = per_p;
            }
        }
    }
    Ok(p)
}

/// Electronic multipole charges for the current density; core monopoles stay at zero so the
/// resulting field is the electrons' alone.
pub(crate) fn write_electron_charges(setup: &Setup, density: &Matrix, sites: &mut [ChargeSite]) {
    for (ia, atom_sites) in setup.atom_sites.iter().enumerate() {
        let n = setup.basis.atom_norb[ia];
        if n == 0 {
            continue;
        }
        let off = setup.basis.atom_offset[ia];
        let mut block = vec![0.0; n * n];
        for mu in 0..n {
            for nu in 0..n {
                block[mu * n + nu] = density[(off + mu, off + nu)];
            }
        }
        let charges = atom_sites.charges(&block, 0.0);
        for (index, charge) in charges.iter().enumerate() {
            sites[setup.site_offset[ia] + index].charge = *charge;
        }
    }
}

/// The electrons' own long-range field at a given density.
///
/// Four call sites used to spell this out — write the electronic multipole charges onto a copy
/// of the site list, then run the lattice sum — and a fifth would have been a fifth chance to
/// pass the wrong density or the wrong cell.
pub(crate) fn electron_field(setup: &Setup, density: &Matrix) -> Result<EwaldOutput> {
    let mut sites = setup.sites.clone();
    write_electron_charges(setup, density, &mut sites);
    crate::pbc::ewald::ewald_potentials_cached(
        &setup.cell,
        &sites,
        &setup.ewald_params,
        &setup.ewald_context,
    )
}

/// Add a set of site potentials to the diagonal atom blocks of a matrix.
pub(crate) fn add_site_potential(setup: &Setup, target: &mut Matrix, field: &EwaldOutput) {
    for (ia, atom_sites) in setup.atom_sites.iter().enumerate() {
        let n = setup.basis.atom_norb[ia];
        if n == 0 {
            continue;
        }
        let off = setup.basis.atom_offset[ia];
        let start = setup.site_offset[ia];
        let potential = &field.site_potential_ev[start..start + atom_sites.offsets.len()];
        for mu in 0..n {
            for nu in 0..n {
                target[(off + mu, off + nu)] += atom_sites.fock_contribution(mu, nu, potential);
            }
        }
    }
}

/// `H_core` plus every density-dependent term except the electrons' own lattice field.
///
/// Written here rather than reusing [`crate::fock::build_fock_spin`] because Coulomb and
/// exchange need different integral tables — that routine takes one table and uses it for both,
/// which is exactly the coupling this module has to break.
pub(crate) fn build_periodic_fock(
    molecule: &Molecule,
    params: &Pm3Parameters,
    setup: &Setup,
    p_tot: &Matrix,
    p_spin: &Matrix,
) -> Result<Matrix> {
    let basis = &setup.basis;
    let mut fock = setup.h_core.clone();

    add_one_center(molecule, params, basis, p_tot, p_spin, &mut fock)?;

    // An atom against its own images: structurally one-center, so the same pattern with the
    // image-summed tables standing in for the `Gss`/`Gsp`/… integrals.
    for (ia, table) in setup.self_images.iter().enumerate() {
        let Some(tables) = table else { continue };
        let n = basis.atom_norb[ia];
        let off = basis.atom_offset[ia];
        let npack = tables.npack_j;
        for mu in 0..n {
            for nu in 0..n {
                let mut acc = 0.0;
                for la in 0..n {
                    for si in 0..n {
                        acc += p_tot[(off + la, off + si)]
                            * tables.coulomb[pack(mu, nu) * npack + pack(la, si)];
                        acc -= p_spin[(off + la, off + si)]
                            * tables.exchange[pack(mu, la) * npack + pack(nu, si)];
                    }
                }
                fock[(off + mu, off + nu)] += acc;
            }
        }
    }

    // Two-center blocks. Coulomb from the screened table, exchange from the full one.
    for tables in &setup.pairs {
        let (oa, ob) = (basis.atom_offset[tables.a], basis.atom_offset[tables.b]);
        let (na, nb) = (tables.norb_i, tables.norb_j);
        let npack_j = tables.npack_j;
        for mu in 0..na {
            for nu in 0..na {
                let mut acc = 0.0;
                for la in 0..nb {
                    for si in 0..nb {
                        acc += p_tot[(ob + la, ob + si)]
                            * tables.coulomb[pack(mu, nu) * npack_j + pack(la, si)];
                    }
                }
                fock[(oa + mu, oa + nu)] += acc;
            }
        }
        for la in 0..nb {
            for si in 0..nb {
                let mut acc = 0.0;
                for mu in 0..na {
                    for nu in 0..na {
                        acc += p_tot[(oa + mu, oa + nu)]
                            * tables.coulomb[pack(mu, nu) * npack_j + pack(la, si)];
                    }
                }
                fock[(ob + la, ob + si)] += acc;
            }
        }
        for mu in 0..na {
            for la in 0..nb {
                let mut acc = 0.0;
                for nu in 0..na {
                    for si in 0..nb {
                        acc += p_spin[(oa + nu, ob + si)]
                            * tables.exchange[pack(mu, nu) * npack_j + pack(la, si)];
                    }
                }
                let value = fock[(oa + mu, ob + la)] - acc;
                fock[(oa + mu, ob + la)] = value;
                fock[(ob + la, oa + mu)] = value;
            }
        }
    }
    Ok(fock)
}

/// [`write_electron_charges`] from a sparse density.
///
/// Only the atom-diagonal blocks are read — a multipole charge is a property of one atom — and
/// a subsystem holding an atom holds its whole block, so every one of them is inside the
/// pattern and the sparse read is exact.
pub(crate) fn write_electron_charges_sparse(
    setup: &Setup,
    pattern: &crate::dc::pattern::DensityPattern,
    density: &crate::dc::pattern::SparseMatrix,
    sites: &mut [ChargeSite],
) {
    for (ia, atom_sites) in setup.atom_sites.iter().enumerate() {
        let n = setup.basis.atom_norb[ia];
        if n == 0 {
            continue;
        }
        let off = setup.basis.atom_offset[ia];
        let mut block = vec![0.0; n * n];
        for mu in 0..n {
            for nu in 0..n {
                block[mu * n + nu] = density.get(pattern, off + mu, off + nu);
            }
        }
        let charges = atom_sites.charges(&block, 0.0);
        for (index, charge) in charges.iter().enumerate() {
            sites[setup.site_offset[ia] + index].charge = *charge;
        }
    }
}

/// [`add_site_potential`] onto a sparsity pattern. Atom-diagonal blocks only, as above.
pub(crate) fn add_site_potential_sparse(
    setup: &Setup,
    pattern: &crate::dc::pattern::DensityPattern,
    target: &mut crate::dc::pattern::SparseMatrix,
    field: &EwaldOutput,
) {
    for (ia, atom_sites) in setup.atom_sites.iter().enumerate() {
        let n = setup.basis.atom_norb[ia];
        if n == 0 {
            continue;
        }
        let off = setup.basis.atom_offset[ia];
        let start = setup.site_offset[ia];
        let potential = &field.site_potential_ev[start..start + atom_sites.offsets.len()];
        for mu in 0..n {
            for nu in 0..n {
                target.add(
                    pattern,
                    off + mu,
                    off + nu,
                    atom_sites.fock_contribution(mu, nu, potential),
                );
            }
        }
    }
}

/// [`build_periodic_fock`] onto a sparsity pattern, for divide-and-conquer.
///
/// The same arithmetic with a different destination, and the same argument for why it changes no
/// number: this Fock matrix is read only where the subsystem gather looks, which is inside the
/// pattern by construction, and the density it reads is structurally zero outside it. See
/// [`crate::dc::pattern`].
///
/// Every position touched here — the atom-diagonal blocks, the self-image blocks, and the
/// `(a, b)` blocks of [`Setup::pairs`] — is one the dense form touches too. What is dropped is
/// the pairs no subsystem shares, whose entries the solve never consults.
#[allow(clippy::too_many_arguments)] // the dense twin's arguments, plus the pattern and its core
pub(crate) fn build_periodic_fock_sparse(
    molecule: &Molecule,
    params: &Pm3Parameters,
    setup: &Setup,
    h_core: &crate::dc::pattern::SparseMatrix,
    pattern: &crate::dc::pattern::DensityPattern,
    p_tot: &crate::dc::pattern::SparseMatrix,
    p_spin: &crate::dc::pattern::SparseMatrix,
    out: &mut crate::dc::pattern::SparseMatrix,
) -> Result<()> {
    let basis = &setup.basis;
    out.values_mut().copy_from_slice(h_core.values());

    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let elem = params.element(atom.z)?;
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
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
                let mut acc = 0.0;
                for la in 0..n {
                    for si in 0..n {
                        acc += p_tot.get(pattern, off + la, off + si) * oc(mu, nu, la, si);
                        acc -= p_spin.get(pattern, off + la, off + si) * oc(mu, la, nu, si);
                    }
                }
                out.add(pattern, off + mu, off + nu, acc);
            }
        }
    }

    for (ia, table) in setup.self_images.iter().enumerate() {
        let Some(tables) = table else { continue };
        let n = basis.atom_norb[ia];
        let off = basis.atom_offset[ia];
        let npack = tables.npack_j;
        for mu in 0..n {
            for nu in 0..n {
                let mut acc = 0.0;
                for la in 0..n {
                    for si in 0..n {
                        acc += p_tot.get(pattern, off + la, off + si)
                            * tables.coulomb[pack(mu, nu) * npack + pack(la, si)];
                        acc -= p_spin.get(pattern, off + la, off + si)
                            * tables.exchange[pack(mu, la) * npack + pack(nu, si)];
                    }
                }
                out.add(pattern, off + mu, off + nu, acc);
            }
        }
    }

    for tables in &setup.pairs {
        let (oa, ob) = (basis.atom_offset[tables.a], basis.atom_offset[tables.b]);
        let (na, nb) = (tables.norb_i, tables.norb_j);
        let npack_j = tables.npack_j;
        for mu in 0..na {
            for nu in 0..na {
                let mut acc = 0.0;
                for la in 0..nb {
                    for si in 0..nb {
                        acc += p_tot.get(pattern, ob + la, ob + si)
                            * tables.coulomb[pack(mu, nu) * npack_j + pack(la, si)];
                    }
                }
                out.add(pattern, oa + mu, oa + nu, acc);
            }
        }
        for la in 0..nb {
            for si in 0..nb {
                let mut acc = 0.0;
                for mu in 0..na {
                    for nu in 0..na {
                        acc += p_tot.get(pattern, oa + mu, oa + nu)
                            * tables.coulomb[pack(mu, nu) * npack_j + pack(la, si)];
                    }
                }
                out.add(pattern, ob + la, ob + si, acc);
            }
        }
        // The dense form mirrors by reading one entry to set the other; here both are added,
        // which is the same number because everything written before this point is symmetric.
        for mu in 0..na {
            for la in 0..nb {
                let mut acc = 0.0;
                for nu in 0..na {
                    for si in 0..nb {
                        acc += p_spin.get(pattern, oa + nu, ob + si)
                            * tables.exchange[pack(mu, nu) * npack_j + pack(la, si)];
                    }
                }
                out.add(pattern, oa + mu, ob + la, -acc);
                out.add(pattern, ob + la, oa + mu, -acc);
            }
        }
    }
    Ok(())
}

/// The one-center two-electron block: intra-atomic, so no image ever enters it and the k-point
/// path uses it unchanged.
pub(crate) fn add_one_center(
    molecule: &Molecule,
    params: &Pm3Parameters,
    basis: &Basis,
    p_tot: &Matrix,
    p_spin: &Matrix,
    fock: &mut Matrix,
) -> Result<()> {
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let elem = params.element(atom.z)?;
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
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
                let mut acc = 0.0;
                for la in 0..n {
                    for si in 0..n {
                        acc += p_tot[(off + la, off + si)] * oc(mu, nu, la, si);
                        acc -= p_spin[(off + la, off + si)] * oc(mu, la, nu, si);
                    }
                }
                fock[(off + mu, off + nu)] += acc;
            }
        }
    }
    Ok(())
}

/// Build everything that depends only on the geometry.
pub(crate) fn build_setup(
    molecule: &Molecule,
    params: &Pm3Parameters,
    periodic: &PeriodicOptions,
) -> Result<Setup> {
    let cell = molecule.cell.expect("checked by the caller");
    let basis = Basis::build(molecule, params)?;
    let nat = molecule.atoms.len();

    // Every element realizes: a `d` shell through MOPAC's own multipole table, a Sparkle or a
    // point atom through the one site it has — its nucleus. See [`AtomSites::build`].
    let mut atom_sites = Vec::with_capacity(nat);
    for atom in &molecule.atoms {
        let elem = params.element(atom.z)?;
        atom_sites.push(AtomSites::build(elem).ok_or(Pm3Error::UnsupportedDOrbitals(atom.z))?);
    }

    let mut sites = Vec::new();
    let mut site_offset = Vec::with_capacity(nat);
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        site_offset.push(sites.len());
        for offset in &atom_sites[ia].offsets {
            sites.push(ChargeSite {
                position: atom.position + *offset,
                charge: 0.0,
                owner: ia,
            });
        }
    }
    let mut core_sites = sites.clone();
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        core_sites[site_offset[ia]].charge = params.element(atom.z)?.core_charge;
    }

    // The electrostatics is split into a core half and an electron half so that one can enter
    // `H_core` — linear in the density — and the other the Fock matrix, which is quadratic. Each
    // half carries the full nuclear charge with opposite sign, so **each is a charged lattice sum
    // on its own**, even for a perfectly neutral cell.
    //
    // In 3D and 2D that is harmless: a neutralizing background makes each half finite and the two
    // backgrounds cancel in the total. In 1D there is none, and a charged chain's potential grows
    // logarithmically with transverse distance.
    //
    // What saves it is that the divergence cancels *exactly* between the three pieces, provided
    // they share one truncation. Writing `H_N = Σ_{n≤N} 1/n` for the divergent lattice sum and `Q`
    // for the nuclear charge, the monopole part of each piece is
    //
    //     ½Σ_cc → +Q²H_N/L,    Σ_ce → −2Q²H_N/L,    ½Σ_ee → +Q²H_N/L
    //
    // which sums to zero for any `N`. So the chain is summed with a common image count and the
    // answer is finite and `N`-independent even though no single term is.
    //
    // Each half now also carries its own neutralizing line charge (see
    // [`crate::pbc::ewald::direct_1d`]), which makes each finite on its own. That changes nothing
    // for a neutral cell — the line terms cancel across the three pieces exactly as the
    // divergences did — and it is what lets a genuinely charged cell work at all, where the
    // halves no longer balance and the leftover *is* the physical background term.
    let ewald_params = periodic
        .ewald
        .unwrap_or_else(|| EwaldParams::for_cell(&cell, crate::pbc::ewald::DEFAULT_ACCURACY));

    let ewald_context = crate::pbc::ewald::EwaldContext::build(&cell, &sites, &ewald_params);
    let core_field = crate::pbc::ewald::ewald_potentials_cached(
        &cell,
        &core_sites,
        &ewald_params,
        &ewald_context,
    )?;

    let cutoff = periodic.short_range_cutoff.max(periodic.switch.off);
    let positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
    let list = NeighborList::build_from_positions(&positions, Some(&cell), cutoff);

    let mut h_core = Matrix::zeros(basis.nao, basis.nao);
    for (mu, ao) in basis.aos.iter().enumerate() {
        let elem = params.element(ao.z)?;
        h_core[(mu, mu)] = match ao.orb {
            0 => elem.u_ss,
            1..=3 => elem.u_pp,
            _ => elem.u_dd,
        };
    }

    // The image-summed pair tables, keyed by the pair rather than indexed by `a·nat + b`.
    //
    // Two `nat × nat` arrays of `Option<Vec<f64>>` is 48 bytes per *possible* pair — 432 MB at
    // three thousand atoms and 4.8 GB at ten thousand — to hold the `O(N)` pairs the neighbour
    // list above actually produced. That is a quadratic term sitting on the screened
    // divide-and-conquer path, which is the one this crate calls linear scaling.
    //
    // Insertion order is the neighbour list's, and the drain below sorts, so nothing here
    // depends on hash iteration order: a Fourier sum or an accumulation whose order varies
    // between runs is a real hazard in this kind of code and is avoided by construction rather
    // than by hoping.
    let mut summed: std::collections::HashMap<(usize, usize), PairSums> =
        std::collections::HashMap::new();
    let mut images: Vec<ImageBlock> = Vec::new();
    let mut core_ev = 0.0;

    for pair in list.all() {
        let (a, b) = (pair.a, pair.b);
        let ea = params.element(molecule.atoms[a].z)?;
        let eb = params.element(molecule.atoms[b].z)?;
        // The integral kernels branch on heavy/light and have no light/heavy case, so tables are
        // always built with the heavier atom first and re-oriented afterwards.
        let heavy_first = ea.n_orb >= eb.n_orb;
        let (first_index, second_index) = if heavy_first { (a, b) } else { (b, a) };
        let (first, second) = if heavy_first { (ea, eb) } else { (eb, ea) };
        let dvec = if heavy_first {
            pair.dvec
        } else {
            pair.dvec * -1.0
        };

        let te = nddo_pair(first, second, dvec);
        let point = point_pair(
            &atom_sites[first_index],
            first.core_charge,
            &atom_sites[second_index],
            second.core_charge,
            dvec,
        );
        let f = periodic.switch.at(pair.r);

        // Every physical term is visited twice by the ordered neighbour list and written into
        // both destination blocks, so each contribution is halved. See the module note.
        let (na, nb) = (first.n_orb, second.n_orb);
        let off_first = basis.atom_offset[first_index];
        let off_second = basis.atom_offset[second_index];
        for mu in 0..na {
            for nu in 0..na {
                h_core[(off_first + mu, off_first + nu)] +=
                    0.5 * f * (te.e1b[mu][nu] - point.e1b[mu * na + nu]);
            }
        }
        for la in 0..nb {
            for si in 0..nb {
                h_core[(off_second + la, off_second + si)] +=
                    0.5 * f * (te.e2a[la][si] - point.e2a[la * nb + si]);
            }
        }

        // Resonance and exchange are the two terms that land on an *image* block rather than on
        // an on-site one, so they are collected per `(a, b, T)`. The Γ path folds them down
        // immediately; the k-point path needs them resolved, because `F(k) = Σ_T e^{ik·T} F(T)`.
        //
        // No halving here: unlike the on-site terms, each `(a, b, T)` block is visited exactly
        // once by the neighbour list, and the visit `(b, a, −T)` fills the transposed block.
        //
        // Both are cut off at the same radius, beyond which the exchange contribution is zero
        // however large the integral is, because the inter-atomic density matrix multiplying it
        // has decayed. That reasoning is sound for the partners it was written for and breaks
        // down once the cell is narrower than the cutoff, since `P(Γ)` is then attributable to
        // more than one image — see the module note "When Γ alone is enough" and
        // [`PeriodicResult::gamma_margin`], which reports the condition.
        if pair.r <= periodic.short_range_cutoff {
            let s_block = pair_overlap(first, second, dvec)?;
            let mut resonance = vec![0.0; na * nb];
            for mu in 0..na {
                let bi = resonance_beta(first, basis.aos[off_first + mu].orb);
                for la in 0..nb {
                    let bj = resonance_beta(second, basis.aos[off_second + la].orb);
                    resonance[mu * nb + la] = 0.5 * (bi + bj) * s_block[mu][la];
                }
            }
            let (norb_a, norb_b) = (basis.atom_norb[a], basis.atom_norb[b]);
            let exchange = oriented(&te.w, None, 1.0, te.npack_i, te.npack_j, heavy_first);
            // The Γ path wants these summed over images; it contracts the sum against `P(Γ)`,
            // which is the one density it has.
            accumulate(
                &mut summed.entry((a, b)).or_default().exchange,
                exchange.clone(),
            );
            images.push(ImageBlock {
                a,
                b,
                t: pair.t,
                resonance: orient_rectangular(&resonance, na, nb, heavy_first),
                exchange,
                norb_a,
                norb_b,
                npack_b: norb_b * (norb_b + 1) / 2,
            });
        }

        // Coulomb: the screened correction, over the whole switch range.
        accumulate(
            &mut summed.entry((a, b)).or_default().coulomb,
            oriented(
                &te.w,
                Some(&point.w),
                f,
                te.npack_i,
                te.npack_j,
                heavy_first,
            ),
        );
        // The core–core energy is a sum over interactions, not an accumulation into blocks, so
        // it takes the unique half rather than a factor of one half.
        if pair.is_unique_representative() {
            core_ev += periodic_core_core(
                params,
                ea,
                eb,
                molecule.atoms[a].z,
                molecule.atoms[b].z,
                pair.r,
                periodic.switch,
            );
        }
    }

    // Everything accumulated so far lands on a `T = 0` block, so this is the on-site Hamiltonian
    // the k-point path wants; only the resonance, folded in next, belongs to an image.
    let mut h_onsite = h_core.clone();

    // Fold the image-resolved resonance into the Γ-point one-electron matrix, which is exactly
    // `Σ_T H(T)`. Each visit's block goes to its own destination and is not symmetrized: the
    // transposed block is filled by the `(b, a, −T)` visit, which the neighbour list also makes.
    for block in &images {
        let (oa, ob) = (basis.atom_offset[block.a], basis.atom_offset[block.b]);
        for mu in 0..block.norb_a {
            for la in 0..block.norb_b {
                h_core[(oa + mu, ob + la)] += block.resonance[mu * block.norb_b + la];
            }
        }
    }

    core_ev += core_field.energy_ev;
    add_field_to_blocks(&basis, &atom_sites, &site_offset, &mut h_core, &core_field);
    add_field_to_blocks(
        &basis,
        &atom_sites,
        &site_offset,
        &mut h_onsite,
        &core_field,
    );

    let mut pairs = Vec::new();
    let mut self_images: Vec<Option<PairTables>> = (0..nat).map(|_| None).collect();
    // Sorted, which reproduces the `for a { for b }` order the dense arrays were drained in
    // exactly — so this change is a memory one and not a numerical one. Iterating the map
    // directly would make the accumulation order depend on the hash seed.
    let mut keys: Vec<(usize, usize)> = summed.keys().copied().collect();
    keys.sort_unstable();
    for key in keys {
        let (a, b) = key;
        let sums = summed.remove(&key).expect("the key came from the map");
        let (coulomb, exchange) = (sums.coulomb, sums.exchange);
        if coulomb.is_none() && exchange.is_none() {
            continue;
        }
        if a > b {
            // The transpose of a pair already stored.
            continue;
        }
        let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);
        let (npack_i, npack_j) = (na * (na + 1) / 2, nb * (nb + 1) / 2);
        let tables = PairTables {
            a,
            b,
            coulomb: coulomb.unwrap_or_else(|| vec![0.0; npack_i * npack_j]),
            exchange: exchange.unwrap_or_else(|| vec![0.0; npack_i * npack_j]),
            norb_i: na,
            norb_j: nb,
            npack_j,
        };
        if a == b {
            self_images[a] = Some(tables);
        } else {
            pairs.push(tables);
        }
    }

    Ok(Setup {
        cell,
        basis,
        atom_sites,
        sites,
        site_offset,
        h_core,
        h_onsite,
        images,
        pairs,
        self_images,
        core_ev,
        core_ewald_ev: core_field.energy_ev,
        ewald_params,
        ewald_context,
        // The narrowest periodic width, against the range over which `P(Γ)` has to stand in for a
        // single image's `P(0, T)`.
        gamma_margin: cell
            .periodic_widths()
            .into_iter()
            .fold(f64::INFINITY, |narrowest, (_, width)| narrowest.min(width))
            - periodic.short_range_cutoff,
    })
}

/// Add `table` into `slot`, creating it if this is the first image of that pair.
/// One pair's image-summed Coulomb and exchange tables, held together so the map has one entry
/// per pair rather than two.
#[derive(Default)]
struct PairSums {
    coulomb: Option<Vec<f64>>,
    exchange: Option<Vec<f64>>,
}

fn accumulate(slot: &mut Option<Vec<f64>>, table: Vec<f64>) {
    match slot {
        Some(existing) => {
            for (target, value) in existing.iter_mut().zip(&table) {
                *target += value;
            }
        }
        empty => *empty = Some(table),
    }
}

/// `f · (w − point)` (or `f · w` when `point` is `None`), in `a`-major orientation.
fn oriented(
    w: &[f64],
    point: Option<&[f64]>,
    f: f64,
    npack_i: usize,
    npack_j: usize,
    heavy_first: bool,
) -> Vec<f64> {
    let mut out = vec![0.0; npack_i * npack_j];
    for p in 0..npack_i {
        for q in 0..npack_j {
            let index = p * npack_j + q;
            let subtract = point.map_or(0.0, |values| values[index]);
            out[index] = f * (w[index] - subtract);
        }
    }
    if heavy_first {
        return out;
    }
    let mut transposed = vec![0.0; npack_i * npack_j];
    for p in 0..npack_i {
        for q in 0..npack_j {
            transposed[q * npack_i + p] = out[p * npack_j + q];
        }
    }
    transposed
}

/// Re-orient a rectangular `rows × cols` block from heavy-major to `a`-major.
fn orient_rectangular(block: &[f64], rows: usize, cols: usize, heavy_first: bool) -> Vec<f64> {
    if heavy_first {
        return block.to_vec();
    }
    let mut transposed = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            transposed[j * rows + i] = block[i * cols + j];
        }
    }
    transposed
}

fn resonance_beta(elem: &Pm3Element, orb: u8) -> f64 {
    match orb {
        0 => elem.beta_s,
        1..=3 => elem.beta_p,
        _ => elem.beta_d,
    }
}

fn pair_overlap(ea: &Pm3Element, eb: &Pm3Element, dvec: Vec3) -> Result<[[f64; 9]; 9]> {
    let s = crate::overlap::diatom_overlap(ea, Vec3::zero(), eb, dvec)?;
    let mut out = [[0.0; 9]; 9];
    for (i, row) in s.iter().enumerate() {
        out[i][..4].copy_from_slice(row);
    }
    Ok(out)
}

/// PM3's core–core energy for one image pair with the point-charge monopole removed.
///
/// The Klopman–Ohno monopole is the only part the lattice sum owns; the exponential and Gaussian
/// terms decay far too fast to need it and are summed as they stand.
fn periodic_core_core(
    params: &Pm3Parameters,
    ei: &Pm3Element,
    ej: &Pm3Element,
    zi: u8,
    zj: u8,
    r: f64,
    switch: SwitchRange,
) -> f64 {
    let full = crate::repulsion::pair_core_energy_scalar::<f64>(params, ei, ej, zi, zj, r);
    let rho = ei.po[9] + ej.po[9];
    let bare = PM3_EV / (r * r + rho * rho).sqrt() * ei.core_charge * ej.core_charge;
    full - bare + screened_core_core(ei, ej, r, switch)
}

fn add_field_to_blocks(
    basis: &Basis,
    atom_sites: &[AtomSites],
    site_offset: &[usize],
    target: &mut Matrix,
    field: &EwaldOutput,
) {
    for (ia, sites) in atom_sites.iter().enumerate() {
        let n = basis.atom_norb[ia];
        if n == 0 {
            continue;
        }
        let off = basis.atom_offset[ia];
        let start = site_offset[ia];
        let potential = &field.site_potential_ev[start..start + sites.offsets.len()];
        for mu in 0..n {
            for nu in 0..n {
                target[(off + mu, off + nu)] += sites.fock_contribution(mu, nu, potential);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Cell;
    use crate::corrections::Variant;
    use crate::scf::{run_pm3, Reference};

    const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n";
    const H2: &str = "2\nh2\nH 0 0 0\nH 0.74 0 0\n";
    const METHANE: &str = "5\nmethane\nC 0.0 0.0 0.0\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\nH -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276\n";

    fn periodic(xyz: &str, edge: f64) -> Result<PeriodicResult> {
        let params = Pm3Parameters::standard().unwrap();
        let mut molecule = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        molecule.cell = Some(Cell::cubic(edge).unwrap());
        run_gamma(
            &molecule,
            &params,
            &Pm3Options::default(),
            &PeriodicOptions::default(),
        )
    }

    /// `n×n×n` copies of `base`'s contents in a cell `n` times as wide, describing exactly the
    /// same infinite crystal.
    fn supercell(base: &Molecule, edge: f64, n: usize) -> Molecule {
        let mut atoms = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let shift = Vec3::new(edge * i as f64, edge * j as f64, edge * k as f64);
                    atoms.extend(base.atoms.iter().map(|atom| {
                        let mut copy = atom.clone();
                        copy.position += shift;
                        copy
                    }));
                }
            }
        }
        let mut out = base.clone();
        out.atoms = atoms;
        out.cell = Some(Cell::cubic(edge * n as f64).unwrap());
        out
    }

    fn molecular(xyz: &str) -> crate::scf::Pm3Result {
        let params = Pm3Parameters::standard().unwrap();
        run_pm3(
            &Molecule::from_xyz_str(xyz, 0.0).unwrap(),
            &params,
            &Pm3Options::default(),
        )
        .unwrap()
    }

    /// Folding: one cell of edge `L` and a `2×2×2` supercell of edge `2L` holding eight copies
    /// describe the *same infinite crystal*, so the supercell energy must be exactly eight times
    /// the single-cell one.
    ///
    /// This is the strongest statement available about the lattice sums, and it is strictly
    /// stronger than the large-cell molecular limit: it is sensitive to every counting
    /// convention — the ordered-visit halving, the unique half in the core–core term, the
    /// self-image block, the owner exclusion in the Ewald sum — none of which the isolated limit
    /// probes, because there the image terms it would miscount are all zero anyway.
    ///
    /// It holds only where the Γ point is sufficient, which is exactly the condition
    /// [`PeriodicResult::gamma_margin`] reports; the companion test below pins down what happens
    /// when it is not.
    #[test]
    fn a_supercell_reproduces_the_single_cell_energy_eight_times_over() {
        let params = Pm3Parameters::standard().unwrap();
        for edge in [18.0_f64, 22.0] {
            let mut one = Molecule::from_xyz_str(WATER, 0.0).unwrap();
            one.cell = Some(Cell::cubic(edge).unwrap());
            let eight = supercell(&one, edge, 2);
            let options = Pm3Options::default();
            let periodic = PeriodicOptions::default();
            let single = run_gamma(&one, &params, &options, &periodic).unwrap();
            let folded = run_gamma(&eight, &params, &options, &periodic).unwrap();
            assert!(
                single.gamma_margin > 0.0,
                "edge {edge} is inside the Γ-point validity condition"
            );
            let difference = (single.total_ev - folded.total_ev / 8.0).abs();
            assert!(
                difference < 1.0e-8,
                "edge {edge}: {} vs {} per cell ({difference:.3e} eV)",
                single.total_ev,
                folded.total_ev / 8.0
            );
        }
    }

    /// The other side of that condition, stated so it cannot regress silently.
    ///
    /// Squeezing the cell until an image enters the exchange cutoff makes the Γ-point energy
    /// wrong by tens of eV — not because a sum is miscounted, but because one k-point substitutes
    /// `P(Γ)` for a `P(0, T)` that has decayed to nothing. The SCF converges perfectly well onto
    /// it, so nothing but the geometry reveals the problem. See the module note.
    #[test]
    fn an_image_inside_the_exchange_cutoff_breaks_the_gamma_point() {
        let params = Pm3Parameters::standard().unwrap();
        let mut one = Molecule::from_xyz_str(WATER, 0.0).unwrap();
        one.cell = Some(Cell::cubic(14.0).unwrap());
        let eight = supercell(&one, 14.0, 2);
        let options = Pm3Options::default();
        let periodic = PeriodicOptions::default();
        let single = run_gamma(&one, &params, &options, &periodic).unwrap();
        let folded = run_gamma(&eight, &params, &options, &periodic).unwrap();

        assert!(
            single.gamma_margin <= 0.0,
            "a 14 Bohr cell is exactly at the 14 Bohr exchange cutoff, not beyond it"
        );
        assert!(single.converged, "the wrong answer converges cleanly");
        assert!(
            (single.total_ev - folded.total_ev / 8.0).abs() > 1.0,
            "the Γ-point sampling error is supposed to be large here"
        );
        // The supercell holds the same eight molecules 14 Bohr apart, and its own images are 28
        // Bohr out — so it satisfies the condition and lands near the isolated-molecule value.
        // That the *contents* are 14 Bohr apart is beside the point: the criterion is the cell
        // width, because that is what decides whether `P(Γ)` belongs to one image or several.
        assert!(folded.gamma_margin > 0.0);
    }

    /// A non-polar molecule alone in a cell has essentially no interaction with its own images,
    /// so the periodic energy must equal the molecular one — at *every* cell size, not just
    /// asymptotically. This is the sharpest end-to-end check available: it exercises the whole
    /// split at once, and only an exact cancellation between the screened short-range tables,
    /// the Ewald sum, the core field and the core–core remainder can produce it.
    #[test]
    fn a_nonpolar_molecule_in_a_cell_reproduces_the_molecular_energy() {
        let reference = molecular(H2);
        for edge in [30.0, 60.0, 120.0] {
            let result = periodic(H2, edge).unwrap();
            assert!(result.converged);
            let difference = (result.total_ev - reference.total_ev).abs();
            assert!(
                difference < 1.0e-7,
                "H2 in a {edge} Bohr cell: {} vs molecular {} ({difference:.3e} eV)",
                result.total_ev,
                reference.total_ev
            );
            for (periodic_charge, molecular_charge) in result.charges.iter().zip(&reference.charges)
            {
                assert!((periodic_charge - molecular_charge).abs() < 1.0e-8);
            }
        }
    }

    /// Zero periodic directions is an isolated system, and this path must then reproduce the
    /// molecular one.
    ///
    /// Not asymptotically and not to a fudge factor: with no lattice there are no images, the
    /// long-range half is [`crate::pbc::ewald::direct_0d`]'s exact pair sum, and the only
    /// difference left from [`crate::scf::run_pm3`] is the documented switch — the
    /// Klopman–Ohno correction handed back to the point-charge limit beyond
    /// [`crate::pbc::screen::DEFAULT_SWITCH_OFF`]. Pushing the switch past the whole molecule
    /// removes even that, and then the two must agree to round-off. Both halves are checked,
    /// because that is what separates "the split is exactly reversible" from "the truncation
    /// happens to be small here".
    #[test]
    fn an_isolated_cell_reproduces_the_molecular_energy() {
        for xyz in [H2, WATER, METHANE] {
            let reference = molecular(xyz);
            let mut molecule = Molecule::from_xyz_str(xyz, 0.0).unwrap();
            molecule.cell = Some(Cell::isolated());
            let params = Pm3Parameters::standard().unwrap();

            // With every truncation removed the two are the same calculation.
            let untruncated = PeriodicOptions {
                switch: crate::pbc::screen::SwitchRange {
                    on: 400.0,
                    off: 410.0,
                },
                short_range_cutoff: 400.0,
                ..PeriodicOptions::default()
            };
            let exact =
                run_gamma(&molecule, &params, &Pm3Options::default(), &untruncated).unwrap();
            let difference = (exact.total_ev - reference.total_ev).abs();
            assert!(
                difference < 1.0e-8,
                "untruncated isolated {} vs molecular {} ({difference:.3e} eV)",
                exact.total_ev,
                reference.total_ev
            );
            assert!(
                exact.gamma_margin.is_infinite(),
                "an isolated system has no image to be too close to"
            );

            // And with the defaults, the switch is the only thing between them.
            let switched = run_gamma(
                &molecule,
                &params,
                &Pm3Options::default(),
                &PeriodicOptions::default(),
            )
            .unwrap();
            let switch_error = (switched.total_ev - reference.total_ev).abs();
            assert!(
                switch_error < 1.0e-3,
                "the switch cost {switch_error:.3e} eV, which is too much to call a tail"
            );
        }
    }

    /// Sparkles and point atoms are ordinary members of the lattice sum.
    ///
    /// They have no orbitals, so they have no electronic multipoles — their realization is the
    /// one site every atom has, the nucleus with its core charge. Nothing else about them is
    /// special, and this checks that by demanding the isolated case reproduce the molecular
    /// path exactly rather than approximately.
    ///
    /// The periodic half then does double duty. Each of these systems carries a net charge, so
    /// the cell is charged and the energy is shifted by the neutralizing background. That shift
    /// must scale as `q²`: the `Gd` Sparkle is trivalent, so its offset has to be nine times the
    /// singly-charged one in the same cell. A background that entered the energy without
    /// entering the Fock potential, or one normalized by the wrong power, would not do that.
    #[test]
    fn sparkles_and_point_atoms_are_ordinary_lattice_sites() {
        let params = Pm3Parameters::standard().unwrap();
        let options = Pm3Options::default();
        let mut offsets = Vec::new();

        for (label, extra, charge) in [
            ("point charge +", "+ 4.0 0.0 0.0", 1.0),
            ("point charge -", "- 4.0 0.0 0.0", -1.0),
            ("Gd sparkle", "Gd 4.0 0.0 0.0", 3.0),
        ] {
            let xyz = format!("4\n{label}\nO 0 0 0\nH 0.9584 0 0\nH -0.24 0.9278 0\n{extra}\n");
            let molecule = Molecule::from_xyz_str(&xyz, charge).unwrap();
            let reference = crate::scf::run_pm3(&molecule, &params, &options).unwrap();

            let mut isolated = molecule.clone();
            isolated.cell = Some(Cell::isolated());
            let alone = run_gamma(&isolated, &params, &options, &PeriodicOptions::default())
                .unwrap_or_else(|e| panic!("{label} isolated: {e}"));
            let difference = (alone.total_ev - reference.total_ev).abs();
            assert!(
                difference < 1.0e-9,
                "{label}: isolated {} vs molecular {} ({difference:.3e} eV)",
                alone.total_ev,
                reference.total_ev
            );

            let mut periodic = molecule.clone();
            periodic.cell = Some(Cell::cubic(40.0).unwrap());
            let in_cell = run_gamma(&periodic, &params, &options, &PeriodicOptions::default())
                .unwrap_or_else(|e| panic!("{label} in a cell: {e}"));
            offsets.push((charge, in_cell.total_ev - reference.total_ev));
        }

        // Same cell, so the background shift is q² times a common constant.
        let per_unit_charge: Vec<f64> = offsets
            .iter()
            .map(|(charge, offset)| offset / (charge * charge))
            .collect();
        let first = per_unit_charge[0];
        for (value, (charge, offset)) in per_unit_charge.iter().zip(&offsets) {
            assert!(
                (value - first).abs() < 0.02 * first.abs(),
                "the q={charge} cell is offset by {offset:.4} eV, which is {value:.4} per q² \
                 against {first:.4} — the background is not scaling as the square of the charge"
            );
        }
        assert!(
            first < -0.5,
            "a charged cell should be pulled down by its background, not {first:.4} eV"
        );
    }

    /// A **polar** molecule does interact with its images, through the dipole–dipole term, so the
    /// approach to the molecular limit is not exact — it is `1/L³`. Asserting the exponent rather
    /// than a tolerance is what makes this a physics test instead of a fudge: a leftover monopole
    /// error would fall off as `1/L`, and a residual charge-dipole one as `1/L²`.
    #[test]
    fn a_polar_molecule_approaches_the_molecular_limit_as_one_over_l_cubed() {
        let reference = molecular(WATER);
        let mut differences = Vec::new();
        for edge in [30.0, 60.0, 120.0] {
            let result = periodic(WATER, edge).unwrap();
            assert!(result.converged);
            differences.push((result.total_ev - reference.total_ev).abs());
        }
        for window in differences.windows(2) {
            let ratio = window[0] / window[1];
            assert!(
                (ratio.log2() - 3.0).abs() < 0.2,
                "the image interaction falls off as 1/L^{:.2}, not 1/L³ (differences {:?})",
                ratio.log2(),
                differences
            );
        }
        // And it really is small by the largest cell tested.
        assert!(
            *differences.last().unwrap() < 1.0e-4,
            "residual {:.3e} eV at 120 Bohr",
            differences.last().unwrap()
        );
    }

    /// The same for a molecule with `p` orbitals on the heavy atom and a tetrahedral shape, which
    /// exercises the dipole and quadrupole multipole sites that H₂ never touches.
    #[test]
    fn methane_reproduces_the_molecular_energy() {
        let reference = molecular(METHANE);
        let result = periodic(METHANE, 40.0).unwrap();
        assert!(result.converged);
        let difference = (result.total_ev - reference.total_ev).abs();
        assert!(
            difference < 1.0e-3,
            "methane in a 40 Bohr cell: {} vs molecular {} ({difference:.3e} eV)",
            result.total_ev,
            reference.total_ev
        );
    }

    /// Translating the contents of a cell cannot change anything.
    #[test]
    fn the_energy_does_not_depend_on_where_the_molecule_sits() {
        let params = Pm3Parameters::standard().unwrap();
        let mut base = Molecule::from_xyz_str(WATER, 0.0).unwrap();
        base.cell = Some(Cell::cubic(40.0).unwrap());
        let at_origin = run_gamma(
            &base,
            &params,
            &Pm3Options::default(),
            &PeriodicOptions::default(),
        )
        .unwrap();

        let mut shifted = base.clone();
        for atom in &mut shifted.atoms {
            atom.position += Vec3::new(7.3, -4.1, 11.9);
        }
        let moved = run_gamma(
            &shifted,
            &params,
            &Pm3Options::default(),
            &PeriodicOptions::default(),
        )
        .unwrap();
        assert!(
            (at_origin.total_ev - moved.total_ev).abs() < 1.0e-9,
            "a rigid translation changed the energy by {:.3e} eV",
            at_origin.total_ev - moved.total_ev
        );
    }

    /// The Ewald parameters have to be derived from the cell, not fixed. With a fixed splitting
    /// the reciprocal-vector count grows with the cell volume — 4.5e7 vectors for a 240 Bohr box
    /// at the default 12 Bohr cutoff — and the calculation stops finishing.
    #[test]
    fn ewald_parameters_track_the_cell_size() {
        let small = EwaldParams::for_cell(&Cell::cubic(20.0).unwrap(), 1.0e-12);
        let large = EwaldParams::for_cell(&Cell::cubic(240.0).unwrap(), 1.0e-12);
        assert!(
            large.alpha < 0.2 * small.alpha,
            "alpha did not shrink with the cell ({} vs {})",
            large.alpha,
            small.alpha
        );
        // gmax falls in step with the reciprocal lattice spacing, so the vector count stays put.
        let count = |edge: f64, params: &EwaldParams| {
            let spacing = std::f64::consts::TAU / edge;
            (params.gmax / spacing).ceil()
        };
        let small_count = count(20.0, &small);
        let large_count = count(240.0, &large);
        assert!(
            large_count < 3.0 * small_count,
            "reciprocal vector count blew up with the cell ({large_count} vs {small_count})"
        );
    }

    /// A closed-shell system must give the same energy whether the reference is restricted or
    /// forced unrestricted: `P^α = P^β` is a fixed point of the unrestricted equations, so the
    /// two paths have to agree exactly rather than approximately. Anything else means the α and
    /// β Fock matrices are not being built from the same expression.
    #[test]
    fn forcing_uhf_on_a_closed_shell_cell_changes_nothing() {
        let params = Pm3Parameters::standard().unwrap();
        let mut molecule = Molecule::from_xyz_str(WATER, 0.0).unwrap();
        molecule.cell = Some(Cell::cubic(40.0).unwrap());
        let restricted = run_gamma(
            &molecule,
            &params,
            &Pm3Options::default(),
            &PeriodicOptions::default(),
        )
        .unwrap();
        let forced = run_gamma(
            &molecule,
            &params,
            &Pm3Options {
                reference: Reference::Uhf,
                ..Pm3Options::default()
            },
            &PeriodicOptions::default(),
        )
        .unwrap();
        assert!(forced.unrestricted);
        assert!(!restricted.unrestricted);
        assert!(
            (forced.total_ev - restricted.total_ev).abs() < 1.0e-8,
            "UHF {} vs RHF {} on a closed shell",
            forced.total_ev,
            restricted.total_ev
        );
        // And the spin density vanishes, as it must for a spin-symmetric solution.
        let spin = forced.spin_density.expect("UHF reports a spin density");
        let largest = spin.as_slice().iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(largest < 1.0e-7, "spurious spin polarization {largest:.3e}");
    }

    /// An open-shell cell must reproduce the molecular UHF result in the large-cell limit, the
    /// same way the closed-shell one does.
    #[test]
    fn an_open_shell_cell_reproduces_the_molecular_uhf_energy() {
        const METHYL: &str =
            "4\nmethyl radical\nC 0.0 0.0 0.0\nH 0.0 1.078 0.0\nH 0.9336 -0.539 0.0\nH -0.9336 -0.539 0.0\n";
        let params = Pm3Parameters::standard().unwrap();
        let options = Pm3Options {
            multiplicity: 2,
            ..Pm3Options::default()
        };
        let reference = run_pm3(
            &Molecule::from_xyz_str(METHYL, 0.0)
                .unwrap()
                .with_multiplicity(2),
            &params,
            &options,
        )
        .unwrap();
        assert!(reference.unrestricted);

        let mut molecule = Molecule::from_xyz_str(METHYL, 0.0)
            .unwrap()
            .with_multiplicity(2);
        molecule.cell = Some(Cell::cubic(40.0).unwrap());
        let result = run_gamma(&molecule, &params, &options, &PeriodicOptions::default()).unwrap();
        assert!(result.unrestricted);
        assert!(result.converged);
        let difference = (result.total_ev - reference.total_ev).abs();
        assert!(
            difference < 5.0e-3,
            "periodic UHF {} vs molecular UHF {} ({difference:.3e} eV)",
            result.total_ev,
            reference.total_ev
        );
        // The unpaired electron has to show up as a real spin density.
        let spin = result.spin_density.expect("UHF reports a spin density");
        let trace: f64 = (0..spin.rows).map(|i| spin[(i, i)]).sum();
        assert!(
            (trace - 1.0).abs() < 1.0e-6,
            "the spin density integrates to {trace}, not one unpaired electron"
        );
    }
    /// A corrected variant must behave exactly like the plain one in the large-cell limit, and
    /// its correction has to be a real number rather than a silently dropped term.
    #[test]
    fn corrected_variants_reproduce_the_molecular_energy() {
        let params = Pm3Parameters::standard().unwrap();
        for variant in [Variant::Pm3D3, Variant::Pm3D3H4, Variant::Pm3D3H4X] {
            let options = Pm3Options {
                variant,
                ..Pm3Options::default()
            };
            let reference = run_pm3(
                &Molecule::from_xyz_str(WATER, 0.0).unwrap(),
                &params,
                &options,
            )
            .unwrap();
            let mut molecule = Molecule::from_xyz_str(WATER, 0.0).unwrap();
            molecule.cell = Some(Cell::cubic(60.0).unwrap());
            let result =
                run_gamma(&molecule, &params, &options, &PeriodicOptions::default()).unwrap();
            // Compare what the correction *adds*, in both paths, rather than the totals: water
            // is polar, so its total still carries the 1/L³ dipole image term (1.27e-4 eV at
            // 60 Bohr) that has nothing to do with the corrections. Differencing against the
            // plain variant at the same cell removes it and leaves the correction alone.
            let plain_options = Pm3Options::default();
            let plain_molecular = run_pm3(
                &Molecule::from_xyz_str(WATER, 0.0).unwrap(),
                &params,
                &plain_options,
            )
            .unwrap();
            let plain_periodic = run_gamma(
                &molecule,
                &params,
                &plain_options,
                &PeriodicOptions::default(),
            )
            .unwrap();
            let added_periodic = result.total_ev - plain_periodic.total_ev;
            let added_molecular = reference.total_ev - plain_molecular.total_ev;
            let difference = (added_periodic - added_molecular).abs();
            assert!(
                difference < 1.0e-8,
                "{variant:?}: the correction adds {added_periodic} periodically but \
                 {added_molecular} molecularly ({difference:.3e} eV)"
            );
            assert!(
                result.correction_ev.abs() > 1.0e-6,
                "{variant:?}: the correction is {} — it is not reaching the total",
                result.correction_ev
            );
        }
    }

    /// The correction is post-SCF and lattice-summed once, so it must be exactly the standalone
    /// per-cell value — not scaled by anything the SCF does.
    #[test]
    fn the_reported_correction_is_the_per_cell_lattice_sum() {
        use crate::corrections::periodic::periodic_correction_energy;
        let params = Pm3Parameters::standard().unwrap();
        let options = Pm3Options {
            variant: Variant::Pm3D3H4,
            ..Pm3Options::default()
        };
        let mut molecule = Molecule::from_xyz_str(WATER, 0.0).unwrap();
        molecule.cell = Some(Cell::cubic(14.0).unwrap());
        let periodic = PeriodicOptions::default();
        let result = run_gamma(&molecule, &params, &options, &periodic).unwrap();
        let standalone =
            periodic_correction_energy(&molecule, Variant::Pm3D3H4, &periodic.correction_cutoffs);
        assert!(
            (result.correction_ev - standalone).abs() < 1.0e-12,
            "{} vs {standalone}",
            result.correction_ev
        );
        // The identity above would also hold if the lattice sum silently returned the isolated
        // molecule's value, so check the premise separately: at a cell edge where images really
        // do interact, the per-cell correction must differ from the isolated one.
        let isolated = crate::corrections::correction_energy(
            &Molecule::from_xyz_str(WATER, 0.0).unwrap(),
            Variant::Pm3D3H4,
        );
        let mut tight = Molecule::from_xyz_str(WATER, 0.0).unwrap();
        tight.cell = Some(Cell::cubic(6.0).unwrap());
        let with_images =
            periodic_correction_energy(&tight, Variant::Pm3D3H4, &periodic.correction_cutoffs);
        assert!(
            (with_images - isolated).abs() > 1.0e-3,
            "the lattice sum returns the isolated value ({with_images} vs {isolated})"
        );
    }
    /// A non-periodic structure must be refused rather than silently treated as a molecule.
    #[test]
    fn a_missing_cell_is_reported() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = Molecule::from_xyz_str(WATER, 0.0).unwrap();
        let error = run_gamma(
            &molecule,
            &params,
            &Pm3Options::default(),
            &PeriodicOptions::default(),
        )
        .expect_err("a molecule with no cell is not a periodic system");
        assert!(error.to_string().contains("cell"));
    }
}
