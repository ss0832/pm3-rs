// SPDX-License-Identifier: GPL-3.0-or-later

//! NDDO SCF driver — restricted (RHF) closed shell and unrestricted (UHF) open shell.
//!
//! Because NDDO assumes an orthonormal AO basis, the working equations are the plain
//! eigenproblem `F C = C ε` (no `S`); overlap enters only the resonance term of `H_core`.
//! The initial density is a **superposition of atomic densities** ([`sad_density`]) — the
//! exact free-atom density in a minimal valence basis, far better than the bare-core guess —
//! and charge convergence is accelerated with the A-DIIS→CDIIS hybrid on the `[F,P]` commutator.

use crate::basis::Basis;
use crate::constants::{AU_DIPOLE_TO_DEBYE, EV_TO_KCAL};
use crate::error::{Pm3Error, Result};
use crate::fock::{build_fock, build_fock_spin};
use crate::hamiltonian::{build_core_limited, CoreHamiltonian};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::params::Pm3Parameters;
use crate::repulsion::core_core_energy;
use crate::system::Molecule;

/// Choice of SCF reference (restricted vs unrestricted), independent of the
/// spin multiplicity. `Auto` picks RHF for a closed shell and UHF for an open
/// shell; `Rhf`/`Uhf` force the reference even for a singlet (a UHF singlet can
/// break spin symmetry, e.g. stretched bonds or diradicals).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Reference {
    #[default]
    Auto,
    Rhf,
    Uhf,
}

/// SCF charge-convergence accelerator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScfAccelerator {
    /// No extrapolation (plain iteration).
    None,
    /// Pulay CDIIS on the `[F,P]` commutator throughout.
    Cdiis,
    /// **A-DIIS** (Hu & Yang, *J. Chem. Phys.* **132**, 054109 (2010)) while far from
    /// convergence, switching to CDIIS once the commutator error drops below a threshold —
    /// the robust hybrid recommended for hard cases (radicals, small gaps, poor guesses).
    AdiisCdiis,
}

#[derive(Clone, Debug)]
pub struct Pm3Options {
    pub charge: f64,
    pub multiplicity: usize,
    pub max_scf: usize,
    pub e_tol: f64,
    pub p_tol: f64,
    /// Legacy flag: `false` forces [`ScfAccelerator::None`] regardless of `accelerator`.
    pub use_diis: bool,
    pub accelerator: ScfAccelerator,
    /// Commutator-error norm below which the ADIIS→CDIIS hybrid switches to CDIIS.
    pub adiis_switch: f64,
    /// Restricted/unrestricted reference. `Auto` (default) uses RHF for closed
    /// shells and UHF for open shells; `Uhf` forces the unrestricted path even
    /// for a singlet.
    pub reference: Reference,
    /// PM3 method variant: which post-SCF corrections (D3/H4/X) to add.
    pub variant: crate::corrections::Variant,
    /// Level shift (eV) applied to the virtual space during diagonalization
    /// (Saunders–Hillier). It changes the SCF path/basin without changing the
    /// converged fixed point, preventing the variational collapse to
    /// unphysical charge-transfer states seen for open-shell transition metals.
    pub level_shift_ev: f64,
    /// Density damping factor `λ ∈ [0,1)`: the density fed to the next Fock is
    /// `(1−λ)·P_new + λ·P_old`. Slows early-iteration charge sloshing, which —
    /// together with the level shift — keeps the SCF in the physical basin.
    pub damping: f64,
    /// Transition-metal d-block penalty (eV) added to the metal d-orbital Fock
    /// diagonal at iteration 0 and linearly annealed to zero over
    /// `d_penalty_iters`. This raises the metal d orbitals early so electrons
    /// stay on the ligands (the physical charge distribution), then relaxes to
    /// the true Fock — steering open-shell transition-metal SCF away from the
    /// unphysical metal-anion collapse that a plain iteration / level shift
    /// cannot escape (the collapse fills the *low* metal-d orbitals, which a
    /// virtual-space shift leaves untouched).
    pub d_penalty_start: f64,
    /// Number of iterations over which `d_penalty_start` anneals to zero.
    pub d_penalty_iters: usize,
    /// Optional CPHF/Hessian interaction cutoff radius `r_off` (Bohr). When set,
    /// the coupled-perturbed response damps atom-pair contributions smoothly to
    /// zero between `r_off − 2` and `r_off` (a C² quintic switch), so distant
    /// pairs — whose response is negligible — are skipped. `None` (the default)
    /// computes the exact response and is **bit-identical** to the uncut result.
    pub hessian_cutoff: Option<f64>,
    /// Soft memory budget (MiB) for Hessian workspaces. It bounds concurrent
    /// finite-difference jobs and CPHF response batches, and rejects an analytic
    /// Hessian before allocation if its resident occupied-virtual blocks alone
    /// exceed the limit. Set to `0` to disable the guard.
    pub hessian_memory_mb: usize,
    /// Soft memory budget (MiB) for the resident `O(N²)` two-electron pair
    /// cache built by [`crate::hamiltonian::build_core_limited`]. Exceeding it
    /// raises [`Pm3Error::ResourceLimit`] before the allocation is attempted,
    /// instead of letting the allocator abort the process. `0` disables the
    /// check. Overridden per-process by `PM3_MAX_PAIR_CACHE_MB`.
    pub integral_memory_mb: usize,
    /// Soft memory budget (MiB) for the SCF convergence-accelerator history.
    /// Each retained iterate costs `O(nao²)`; the effective DIIS depth is
    /// reduced (never below 2) so the history fits. `0` disables the cap and
    /// always keeps the full depth. This is the difference between a
    /// 4800-basis-function job needing 4.4 GiB of history and 0.5 GiB.
    pub scf_memory_mb: usize,
    /// Uniform external electric field (eV per Bohr per elementary charge), or `None` for no
    /// field. **Molecular only** — a uniform field is not compatible with periodic boundary
    /// conditions, and the periodic entry points refuse it rather than ignoring it.
    ///
    /// Referenced to the **coordinate origin**, not to the centre of mass: see
    /// [`crate::dipole`]. For a system with a net charge that makes the energy depend on where
    /// the molecule sits, which is physics rather than a convention — and it means such a system
    /// has no energy minimum in a field, since the field can always pull it further.
    pub field: Option<crate::math::Vec3>,
}

impl Default for Pm3Options {
    fn default() -> Self {
        Self {
            charge: 0.0,
            multiplicity: 1,
            max_scf: 200,
            e_tol: 1.0e-8,
            p_tol: 1.0e-7,
            use_diis: true,
            accelerator: ScfAccelerator::AdiisCdiis,
            adiis_switch: 0.1,
            reference: Reference::Auto,
            variant: crate::corrections::Variant::Pm3,
            level_shift_ev: 0.0,
            damping: 0.0,
            d_penalty_start: 0.0,
            d_penalty_iters: 0,
            hessian_cutoff: None,
            hessian_memory_mb: 1024,
            integral_memory_mb: 0,
            scf_memory_mb: 512,
            field: None,
        }
    }
}

/// Resolve the two-electron pair-cache ceiling: the explicit option when set,
/// otherwise the process default (`PM3_MAX_PAIR_CACHE_MB`, else 4 GiB).
pub(crate) fn pair_cache_limit(options: &Pm3Options) -> usize {
    if options.integral_memory_mb > 0 {
        options.integral_memory_mb
    } else {
        crate::hamiltonian::default_pair_cache_limit_mb()
    }
}

/// Deepest convergence-accelerator history the SCF will keep.
const MAX_DIIS_DEPTH: usize = 8;

/// Effective DIIS depth under the configured memory budget.
///
/// A depth-`k` history holds `k` Fock matrices, `k` error matrices and (for
/// A-DIIS) `k` densities — `3 k nao²` doubles. At `nao = 4800` the full depth
/// is 4.4 GiB, which is an out-of-memory abort on most machines; trimming the
/// depth costs a few extra iterations instead. Never drops below 2, the
/// minimum for any extrapolation.
pub(crate) fn diis_depth(nao: usize, matrices_per_slot: usize, budget_mb: usize) -> usize {
    if budget_mb == 0 {
        return MAX_DIIS_DEPTH;
    }
    let per_slot = slot_bytes(nao, matrices_per_slot);
    if per_slot == 0 {
        return MAX_DIIS_DEPTH;
    }
    let budget = budget_mb.saturating_mul(1024 * 1024);
    (budget / per_slot).clamp(2, MAX_DIIS_DEPTH)
}

fn slot_bytes(nao: usize, matrices_per_slot: usize) -> usize {
    matrices_per_slot
        .saturating_mul(nao)
        .saturating_mul(nao)
        .saturating_mul(std::mem::size_of::<f64>())
}

/// Whether the minimum useful history (depth 2) fits the budget.
pub(crate) fn diis_depth_fits(nao: usize, matrices_per_slot: usize, budget_mb: usize) -> bool {
    budget_mb == 0 || 2 * slot_bytes(nao, matrices_per_slot) <= budget_mb * 1024 * 1024
}

/// Apply a Saunders–Hillier level shift to a Fock matrix before diagonalization:
/// `F' = F + shift·(I − P)`, raising the virtual space by ~`shift`. `P` is the
/// (idempotent, orthonormal-basis) density projector for the spin channel
/// (occupation 1 per orbital); for RHF pass `½P_tot`.
fn level_shift(f: &Matrix, p_proj: &Matrix, shift: f64) -> Matrix {
    if shift == 0.0 {
        return f.clone();
    }
    let mut out = f.clone();
    let n = f.rows;
    for i in 0..n {
        for j in 0..n {
            let ident = if i == j { 1.0 } else { 0.0 };
            out[(i, j)] += shift * (ident - p_proj[(i, j)]);
        }
    }
    out
}

#[derive(Clone, Debug)]
pub struct Pm3Result {
    pub density: Matrix,
    /// Spin density `P_α − P_β` (open-shell UHF only; `None` for RHF).
    pub spin_density: Option<Matrix>,
    pub mo_energies: Vec<f64>,
    pub mo_coeff: Matrix,
    pub n_occ: usize,
    /// The β orbital energies and coefficients, and the β occupation count.
    ///
    /// `None` for a restricted run — not "not computed", but "there is one set of orbitals and
    /// [`Pm3Result::mo_coeff`] is it, doubly occupied up to [`Pm3Result::n_occ`]". An
    /// unrestricted run has two genuinely different sets, and both are needed to write a
    /// wavefunction out or to take an open-shell orbital response.
    pub mo_energies_beta: Option<Vec<f64>>,
    pub mo_coeff_beta: Option<Matrix>,
    /// Occupied β count. Equal to `n_occ` for a restricted run.
    pub n_beta: usize,
    pub electronic_ev: f64,
    pub core_ev: f64,
    pub total_ev: f64,
    pub heat_of_formation_kcal: f64,
    pub charges: Vec<f64>,
    pub dipole_debye: Vec3,
    pub dipole_magnitude: f64,
    pub homo_ev: Option<f64>,
    pub lumo_ev: Option<f64>,
    pub iterations: usize,
    /// Always `true`, and kept because a reader should not have to know that to trust the
    /// result: a run that does **not** converge returns
    /// [`crate::error::Pm3Error::ScfNotConverged`] rather than a `Pm3Result` with this set to
    /// `false`. There is no path that produces an unconverged result here. `iterations` beside
    /// it is the number that actually varies.
    pub converged: bool,
    /// True when the UHF (open-shell) path was used.
    pub unrestricted: bool,
}

pub struct Pm3Calculator {
    pub params: Pm3Parameters,
    pub options: Pm3Options,
}

impl Pm3Calculator {
    pub fn new(params: Pm3Parameters) -> Self {
        Self {
            params,
            options: Pm3Options::default(),
        }
    }
    pub fn with_options(params: Pm3Parameters, options: Pm3Options) -> Self {
        Self { params, options }
    }
    pub fn calculate(&self, molecule: &Molecule) -> Result<Pm3Result> {
        run_pm3(molecule, &self.params, &self.options)
    }
}

struct ScfState {
    density: Matrix,
    spin_density: Option<Matrix>,
    mo_energies: Vec<f64>,
    mo_coeff: Matrix,
    n_occ: usize,
    /// The β set, when the two spins have different orbitals. `None` for a restricted run,
    /// where there is one set and `mo_coeff` is it — not "not computed".
    mo_energies_beta: Option<Vec<f64>>,
    mo_coeff_beta: Option<Matrix>,
    n_beta: usize,
    electronic_ev: f64,
    converged: bool,
    /// The RMS density change the loop stopped at, so a failure can say how far off it was
    /// rather than reporting a placeholder.
    last_change: f64,
    iterations: usize,
    unrestricted: bool,
}

/// Enable faer's multi-threaded (rayon) parallelism once per process. faer 0.24
/// defaults to sequential, leaving the O(N³) matmul/eigendecomposition in the
/// SCF single-threaded; on a many-core machine this is the single biggest
/// large-system speedup.
fn enable_faer_parallelism() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        faer::set_global_parallelism(faer::Par::rayon(0));
    });
}

pub fn run_pm3(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
) -> Result<Pm3Result> {
    enable_faer_parallelism();
    let charge =
        reconcile_scalar_metadata("charge", molecule.charge, options.charge, 0.0, 1.0e-12)?;
    let multiplicity = reconcile_usize_metadata(
        "multiplicity",
        molecule.multiplicity,
        options.multiplicity,
        1,
    )?;
    if multiplicity < 1 {
        return Err(Pm3Error::InvalidInput(
            "multiplicity must be >= 1".to_string(),
        ));
    }
    let timing = std::env::var("PM3_TIMING").is_ok();
    let t_all = std::time::Instant::now();
    let basis = Basis::build(molecule, params)?;
    let core = build_core_limited(
        molecule,
        &basis,
        params,
        pair_cache_limit(options),
        options.field,
    )?;
    if timing {
        eprintln!(
            "[timing] basis+core (integrals): {:.3}s  (nao={})",
            t_all.elapsed().as_secs_f64(),
            basis.nao
        );
    }

    let mut n_elec = 0.0;
    for atom in &molecule.atoms {
        n_elec += params.element(atom.z)?.core_charge;
    }
    n_elec -= charge;
    let n_elec_int = n_elec.round() as i64;
    if (n_elec - n_elec_int as f64).abs() > 1.0e-6 || n_elec_int < 0 {
        return Err(Pm3Error::InvalidInput(format!(
            "invalid electron count {n_elec}"
        )));
    }
    let n_unpaired = (multiplicity - 1) as i64;
    if (n_elec_int - n_unpaired) < 0 || (n_elec_int - n_unpaired) % 2 != 0 {
        return Err(Pm3Error::InvalidInput(format!(
            "electron count {n_elec_int} is incompatible with multiplicity {} (need same parity)",
            multiplicity
        )));
    }
    let n_alpha = ((n_elec_int + n_unpaired) / 2) as usize;
    let n_beta = ((n_elec_int - n_unpaired) / 2) as usize;

    // Restricted vs unrestricted: honour an explicit request, else pick by shell.
    let use_uhf = match options.reference {
        Reference::Auto => n_alpha != n_beta,
        Reference::Uhf => true,
        Reference::Rhf => {
            if n_alpha != n_beta {
                return Err(Pm3Error::InvalidInput(format!(
                    "RHF requested for an open-shell system (n_alpha={n_alpha} != n_beta={n_beta}); use UHF or Auto"
                )));
            }
            false
        }
    };
    // Open-shell transition-metal systems can collapse to an unphysical
    // charge-transfer minimum without a level shift. When the caller has not
    // set one, default to a moderate shift for the UHF path; it guides the SCF
    // to the aufbau state and does not change the converged energy of a
    // well-behaved case (Saunders–Hillier).
    let has_open_d = use_uhf
        && molecule
            .atoms
            .iter()
            .any(|a| params.element(a.z).map(|e| e.has_d()).unwrap_or(false));
    let run_scf = |shift: f64, damping: f64, d_pen: f64, d_pen_iters: usize| -> Result<ScfState> {
        let mut opts = options.clone();
        opts.level_shift_ev = shift;
        opts.damping = damping;
        opts.d_penalty_start = d_pen;
        opts.d_penalty_iters = d_pen_iters;
        if use_uhf {
            uhf_loop(molecule, &basis, params, &core, n_alpha, n_beta, &opts)
        } else {
            rhf_loop(molecule, &basis, params, &core, n_alpha, &opts)
        }
    };
    // Open-shell transition-metal SCF collapses to an unphysical metal-anion
    // fixed point from any guess (a genuine lower/adjacent SCF solution; the
    // Hamiltonian itself is bit-exact vs MOPAC). A metal-d Fock penalty, held
    // high early and annealed to zero, keeps electrons on the ligands and
    // tracks the physical basin down to the true (penalty-free) fixed point.
    // Level shift + damping alone cannot escape it (the collapse fills the low
    // metal-d orbitals, which a virtual-space shift leaves untouched).
    let state = if options.level_shift_ev != 0.0 || options.damping != 0.0 {
        run_scf(options.level_shift_ev, options.damping, 0.0, 0)?
    } else if has_open_d {
        // Anneal a 60 eV metal-d penalty over 60 iterations, with mild damping.
        // Recovers the physical basin for several open-shell TM halides; for the
        // remainder it is at worst neutral (the penalty vanishes before
        // convergence, so any converged result is a true fixed point).
        run_scf(0.0, 0.3, 60.0, 60)?
    } else {
        run_scf(0.0, 0.0, 0.0, 0)?
    };
    if timing {
        eprintln!(
            "[timing] SCF ({} iters): {:.3}s",
            state.iterations,
            t_all.elapsed().as_secs_f64()
        );
    }

    // The external field's density-independent half, `−Σ_A Z_A R_A·f`. Its electronic partner
    // `+Tr[P M·f]` is already in `H_core`, and the self-consistent energy counts `H_core` once,
    // so between them the whole of `E_field = −μ·f` is accounted for exactly once.
    let field_nuclear_ev = match options.field {
        Some(f) => crate::dipole::field_terms(molecule, params, &basis, f)?.1,
        None => 0.0,
    };
    let core_ev = core_core_energy(molecule, params)? + field_nuclear_ev;
    let capped_bond_ev = crate::hamiltonian::capped_bond_energy_correction(
        molecule,
        &basis,
        &state.density,
        &core.h_core,
    );
    let electronic_ev = state.electronic_ev + capped_bond_ev;
    // Post-SCF classical corrections (D3/H4/X) per the method variant.
    let correction_ev = crate::corrections::correction_energy(molecule, options.variant);
    let total_ev = electronic_ev + core_ev + correction_ev;

    let mut e_isol_sum = 0.0;
    let mut eheat_sum = 0.0;
    for atom in &molecule.atoms {
        let e = params.element(atom.z)?;
        e_isol_sum += e.e_isol;
        eheat_sum += e.eheat_ev;
    }
    let heat_of_formation_kcal = (total_ev - e_isol_sum + eheat_sum) * EV_TO_KCAL;

    if !state.converged {
        return Err(Pm3Error::ScfNotConverged {
            iterations: state.iterations,
            error: state.last_change,
        });
    }

    // Mulliken net charges from the total density.
    let mut charges = vec![0.0; molecule.atoms.len()];
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
        let mut pop = 0.0;
        for mu in 0..n {
            pop += state.density[(off + mu, off + mu)];
        }
        charges[ia] = params.element(atom.z)?.core_charge - pop;
    }

    // Dipole: the point-charge term plus the s–p hybrid polarization, both in e·Bohr, from the
    // operator in [`crate::dipole`]. It is the same matrix an external field multiplies, and
    // that is the reason it lives there rather than here: `μ = −∂E/∂F` ties the two together,
    // and two separate expressions of one object drift apart silently, each looking right on its
    // own while only their relationship is wrong.
    //
    // The origin is the centre of mass, which is MOPAC's `dipole.F90` convention and matters
    // only for a charged system — `Σ q_i = 0` makes the shift cancel for a neutral one.
    let dipole_origin = crate::dipole::centre_of_mass(molecule, params)?;
    let dip = crate::dipole::dipole_from_density(
        molecule,
        params,
        &basis,
        &state.density,
        dipole_origin,
    )?;
    let dipole_debye = dip * AU_DIPOLE_TO_DEBYE;
    let dipole_magnitude = dipole_debye.norm();

    let nao = basis.nao;
    // Both spin channels, not just α.
    //
    // `mo_energies` is the α spectrum and `n_occ` the α count, so reading the frontier off them
    // alone is right for a closed shell and wrong for an open one: a radical has `n_α > n_β`, so
    // the β LUMO sits below the α LUMO essentially always, and the two HOMOs are separated by
    // whatever the exchange splitting is rather than being equal. The reported gap was the
    // α–α one, which is not the gap.
    //
    // `pbc::kscf::band_edges` has scanned both channels all along; this is the molecular version
    // of the same rule.
    let frontier = |energies: &[f64], occupied: usize| -> (Option<f64>, Option<f64>) {
        (
            (occupied >= 1).then(|| energies[occupied - 1]),
            (occupied < nao).then(|| energies[occupied]),
        )
    };
    let (homo_alpha, lumo_alpha) = frontier(&state.mo_energies, state.n_occ);
    let (homo_beta, lumo_beta) = match &state.mo_energies_beta {
        Some(beta) => frontier(beta, state.n_beta),
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

    Ok(Pm3Result {
        density: state.density,
        spin_density: state.spin_density,
        mo_energies: state.mo_energies,
        mo_coeff: state.mo_coeff,
        n_occ: state.n_occ,
        mo_energies_beta: state.mo_energies_beta,
        mo_coeff_beta: state.mo_coeff_beta,
        n_beta: state.n_beta,
        electronic_ev,
        core_ev,
        total_ev,
        heat_of_formation_kcal,
        charges,
        dipole_debye,
        dipole_magnitude,
        homo_ev,
        lumo_ev,
        iterations: state.iterations,
        converged: state.converged,
        unrestricted: state.unrestricted,
    })
}

/// **Superposition of Atomic Densities (SAD)** initial guess: a block-diagonal density built
/// from each atom's spherically-averaged neutral valence configuration `s^{min(2,Zv)} p^{…}`.
/// In a minimal valence NDDO basis a free atom's density *is* diagonal (spherical symmetry), so
/// this superposition is the exact isolated-atom density and a far better SCF start than the
/// bare core (zero-density) guess — fewer iterations, more robust for larger systems.
fn sad_density(molecule: &Molecule, basis: &Basis, params: &Pm3Parameters) -> Result<Matrix> {
    let nao = basis.nao;
    let mut p = Matrix::zeros(nao, nao);
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let elem = params.element(atom.z)?;
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
        if n == 0 {
            continue;
        }
        // MOPAC initial shell occupancies (ios/iop/iod), spherically averaged.
        p[(off, off)] = elem.occ_s;
        if n >= 4 {
            let per_p = elem.occ_p / 3.0;
            for k in 1..4 {
                p[(off + k, off + k)] = per_p;
            }
        }
        if n >= 9 {
            let per_d = elem.occ_d / 5.0;
            for k in 4..9 {
                p[(off + k, off + k)] = per_d;
            }
        }
    }
    Ok(p)
}

/// Build a density `P = w Σ_{k<n_occ} c_k c_kᵀ` from MO coefficients (`w` = 2 for RHF, 1 for UHF).
fn density_from_coeff(c: &Matrix, n_occ: usize, weight: f64) -> Matrix {
    let nao = c.rows;
    if nao == 0 {
        return Matrix::zeros(0, 0);
    }
    if n_occ == 0 {
        return Matrix::zeros(nao, nao);
    }
    // P = weight · C_occ C_occᵀ, directly into the row-major result.
    c.leading_columns_gram(n_occ, weight)
}

fn reconcile_scalar_metadata(
    name: &str,
    molecule_value: f64,
    option_value: f64,
    default: f64,
    tolerance: f64,
) -> Result<f64> {
    let molecule_set = (molecule_value - default).abs() > tolerance;
    let option_set = (option_value - default).abs() > tolerance;
    if molecule_set && option_set && (molecule_value - option_value).abs() > tolerance {
        return Err(Pm3Error::InvalidInput(format!(
            "conflicting {name}: molecule={molecule_value}, options={option_value}"
        )));
    }
    Ok(if option_set {
        option_value
    } else {
        molecule_value
    })
}

fn reconcile_usize_metadata(
    name: &str,
    molecule_value: usize,
    option_value: usize,
    default: usize,
) -> Result<usize> {
    let molecule_set = molecule_value != default;
    let option_set = option_value != default;
    if molecule_set && option_set && molecule_value != option_value {
        return Err(Pm3Error::InvalidInput(format!(
            "conflicting {name}: molecule={molecule_value}, options={option_value}"
        )));
    }
    Ok(if option_set {
        option_value
    } else {
        molecule_value
    })
}

/// DIIS error `[F,P] = FP − PF`.
///
/// `F` and `P` are both symmetric, so `PF = (FP)ᵀ` and the second `O(nao³)`
/// matrix product is redundant: form `FP` once and antisymmetrize it in place.
pub(crate) fn commutator(f: &Matrix, p: &Matrix) -> Matrix {
    debug_assert_eq!(f.rows, p.rows);
    let mut e = f.matmul(p);
    let n = e.rows;
    let data = e.as_mut_slice();
    for i in 0..n {
        data[i * n + i] = 0.0;
        for j in 0..i {
            let upper = data[j * n + i];
            let lower = data[i * n + j];
            data[i * n + j] = lower - upper;
            data[j * n + i] = upper - lower;
        }
    }
    e
}

/// Rolling history for the SCF convergence accelerators.
///
/// The Gram matrices the extrapolators need — `b[i][j] = ⟨E_i,E_j⟩` for CDIIS
/// and `m[i][j] = ⟨D_i,F_j⟩` for A-DIIS — are maintained **incrementally**:
/// only the new row and column are evaluated each iteration (`2k−1` dot
/// products instead of `2k²`), and the A-DIIS difference matrices `D_i − D_n`
/// and `F_j − F_n` are never materialized. The previous formulation allocated
/// `2k` full `nao × nao` temporaries per iteration — 416 MiB of allocator churn
/// per iteration at `nao = 1800`, which dominated the SCF wall time.
pub(crate) struct AccelHistory {
    depth: usize,
    keep_densities: bool,
    pub(crate) focks: Vec<Matrix>,
    errors: Vec<Matrix>,
    densities: Vec<Matrix>,
    /// `⟨E_i, E_j⟩`, same order as `errors`.
    b: Vec<Vec<f64>>,
    /// `⟨D_i, F_j⟩`, same order as `densities`/`focks`.
    m: Vec<Vec<f64>>,
}

impl AccelHistory {
    pub(crate) fn new(depth: usize, keep_densities: bool) -> Self {
        Self {
            depth,
            keep_densities,
            focks: Vec::with_capacity(depth),
            errors: Vec::with_capacity(depth),
            densities: Vec::with_capacity(depth),
            b: Vec::with_capacity(depth),
            m: Vec::with_capacity(depth),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.focks.len()
    }

    /// Append one iterate, updating the Gram matrices with only the new
    /// row/column, then drop the oldest entry if the depth is exceeded.
    /// `density` is required exactly when the history keeps densities (A-DIIS).
    /// Returns `‖E_new‖²`, which the Gram update computes anyway.
    pub(crate) fn push(&mut self, fock: Matrix, error: Matrix, density: Option<Matrix>) -> f64 {
        let mut b_row: Vec<f64> = self
            .errors
            .iter()
            .map(|e| e.frobenius_dot(&error))
            .collect();
        let error_norm_sq = error.frobenius_dot(&error);
        b_row.push(error_norm_sq);
        for (i, row) in self.b.iter_mut().enumerate() {
            row.push(b_row[i]);
        }
        self.b.push(b_row);

        if let Some(density) = density.filter(|_| self.keep_densities) {
            // New row `⟨D_new, F_j⟩` and new column `⟨D_i, F_new⟩`.
            let mut m_row: Vec<f64> = self
                .focks
                .iter()
                .map(|f| density.frobenius_dot(f))
                .collect();
            m_row.push(density.frobenius_dot(&fock));
            for (row, d) in self.m.iter_mut().zip(&self.densities) {
                row.push(d.frobenius_dot(&fock));
            }
            self.m.push(m_row);
            self.densities.push(density);
        }
        self.focks.push(fock);
        self.errors.push(error);

        while self.focks.len() > self.depth {
            self.focks.remove(0);
            self.errors.remove(0);
            self.b.remove(0);
            for row in &mut self.b {
                row.remove(0);
            }
            if self.keep_densities {
                self.densities.remove(0);
                self.m.remove(0);
                for row in &mut self.m {
                    row.remove(0);
                }
            }
        }
        error_norm_sq
    }

    /// Pulay CDIIS: extrapolated Fock from the stored `⟨E_i,E_j⟩` Gram matrix.
    pub(crate) fn cdiis(&self) -> Option<Matrix> {
        let coeffs = diis_coeffs_from_gram(&self.b)?;
        Some(combine(&self.focks, &coeffs))
    }

    /// A-DIIS (Hu & Yang 2010) from the stored `⟨D_i,F_j⟩` Gram matrix.
    pub(crate) fn adiis(&self) -> Option<Matrix> {
        let n = self.len();
        if n < 2 || !self.keep_densities {
            return None;
        }
        let last = n - 1;
        let mnn = self.m[last][last];
        // d_i = ⟨D_i − D_n, F_n⟩ and s_ij = ⟨D_i − D_n, F_j − F_n⟩, expanded
        // from the stored inner products (no difference matrices formed).
        let d: Vec<f64> = (0..n).map(|i| self.m[i][last] - mnn).collect();
        let s: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| self.m[i][j] - self.m[i][last] - self.m[last][j] + mnn)
                    .collect()
            })
            .collect();
        let c = solve_adiis_simplex(&d, &s);
        Some(combine(&self.focks, &c))
    }
}

fn rms_diff(a: &Matrix, b: &Matrix) -> f64 {
    a.rms_difference(b)
}

fn rhf_loop(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm3Parameters,
    core: &CoreHamiltonian,
    n_occ: usize,
    options: &Pm3Options,
) -> Result<ScfState> {
    let nao = basis.nao;
    let mut density = sad_density(molecule, basis, params)?; // SAD initial guess
    let mut e_old = 0.0;
    let mut mo_energies = vec![0.0; nao];
    let mut mo_coeff = Matrix::zeros(nao, nao);
    let mut converged = false;
    // The RMS density change the loop last saw, so a failure reports how far off it was.
    let mut last_change = f64::INFINITY;
    let mut iterations = 0;
    // MOPAC's PM3 He parameterization has formal p AOs but an s-only core
    // attraction.  On charged He systems DIIS can converge to a different
    // charge-transfer fixed point; the unaccelerated Roothaan path follows the
    // MOPAC basin reproducibly.
    let contains_helium = molecule.atoms.iter().any(|atom| atom.z == 2);
    let accel = if options.use_diis && !contains_helium {
        options.accelerator
    } else {
        ScfAccelerator::None
    };
    let timing = std::env::var("PM3_TIMING").is_ok();
    let (mut t_fock, mut t_eigen, mut t_accel, mut t_density) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    // A-DIIS needs a density history on top of the Fock/error pair; plain CDIIS
    // does not. When the budget cannot even hold a depth-2 A-DIIS history
    // (`3 · nao²` doubles per slot), degrade to CDIIS rather than blowing the
    // budget: it is the weaker accelerator but reaches the same fixed point.
    let mut keep_densities = accel == ScfAccelerator::AdiisCdiis;
    if keep_densities && !diis_depth_fits(nao, 3, options.scf_memory_mb) {
        keep_densities = false;
    }
    let accel = if accel == ScfAccelerator::AdiisCdiis && !keep_densities {
        ScfAccelerator::Cdiis
    } else {
        accel
    };
    let mut history = AccelHistory::new(
        diis_depth(
            nao,
            if keep_densities { 3 } else { 2 },
            options.scf_memory_mb,
        ),
        keep_densities,
    );

    for iter in 0..options.max_scf {
        iterations = iter + 1;
        let t0 = std::time::Instant::now();
        let f = build_fock(molecule, basis, params, core, &density)?;
        if timing {
            t_fock += t0.elapsed().as_secs_f64();
        }
        let e_elec = 0.5 * (density.frobenius_dot(&core.h_core) + density.frobenius_dot(&f));

        let t_a = std::time::Instant::now();
        let f_use = match accel {
            ScfAccelerator::None => f,
            _ => {
                let err = commutator(&f, &density);
                // History (Fock, commutator, density) for CDIIS / A-DIIS; the
                // Gram update returns ‖[F,P]‖² so it is not evaluated twice.
                let err_norm = history
                    .push(f, err, keep_densities.then(|| density.clone()))
                    .sqrt();
                let extrapolated =
                    if accel == ScfAccelerator::AdiisCdiis && err_norm > options.adiis_switch {
                        history.adiis()
                    } else {
                        history.cdiis()
                    };
                // No usable extrapolation (a singular Gram matrix, or a
                // single-entry history): fall back to the newest plain Fock.
                extrapolated.unwrap_or_else(|| history.focks[history.len() - 1].clone())
            }
        };
        let t1 = std::time::Instant::now();
        if timing {
            t_accel += t_a.elapsed().as_secs_f64();
        }
        let (eps, c) = symmetric_eigen(&f_use)?;
        if timing {
            t_eigen += t1.elapsed().as_secs_f64();
        }
        let t2 = std::time::Instant::now();
        let p_new = density_from_coeff(&c, n_occ, 2.0);
        let dp = rms_diff(&p_new, &density);
        last_change = dp;
        if timing {
            t_density += t2.elapsed().as_secs_f64();
        }
        let de = (e_elec - e_old).abs();

        mo_energies = eps;
        mo_coeff = c;
        density = p_new;
        e_old = e_elec;
        // The commutator above belongs to the *pre-extrapolation* density.  In
        // a DIIS step it can be small even when the extrapolated Fock produces
        // a materially different density, which falsely accepted an excited
        // UHF fixed point.  Density self-consistency is the authoritative
        // condition here; the next iteration evaluates its commutator.
        if iter > 0 && de < options.e_tol && dp < options.p_tol {
            converged = true;
            break;
        }
    }
    if timing {
        eprintln!(
            "[timing]   RHF fock: {t_fock:.3}s  accel: {t_accel:.3}s  eigen: {t_eigen:.3}s  density: {t_density:.3}s  ({iterations} iters, diis depth {})",
            history.depth
        );
    }
    let f_final = build_fock(molecule, basis, params, core, &density)?;
    let electronic_ev =
        0.5 * (density.frobenius_dot(&core.h_core) + density.frobenius_dot(&f_final));

    Ok(ScfState {
        last_change,
        density,
        spin_density: None,
        mo_energies,
        mo_coeff,
        n_occ,
        mo_energies_beta: None,
        mo_coeff_beta: None,
        n_beta: n_occ,
        electronic_ev,
        converged,
        iterations,
        unrestricted: false,
    })
}

#[allow(clippy::too_many_arguments)]
fn uhf_loop(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm3Parameters,
    core: &CoreHamiltonian,
    n_alpha: usize,
    n_beta: usize,
    options: &Pm3Options,
) -> Result<ScfState> {
    let nao = basis.nao;
    // Transition-metal d AOs to penalize during annealing (open-shell d collapse fix).
    let d_pen_aos: Vec<usize> = if options.d_penalty_start > 0.0 {
        basis
            .aos
            .iter()
            .enumerate()
            .filter(|(_, ao)| {
                ao.orb >= 4 && params.element(ao.z).map(|e| !e.main_group).unwrap_or(false)
            })
            .map(|(i, _)| i)
            .collect()
    } else {
        Vec::new()
    };
    // SAD guess split by spin population; the different α/β aufbau counts break spin symmetry.
    // Use a common core-Hamiltonian orbital guess. Unequal occupations break
    // spin symmetry without fractionally removing charge from every atom, as
    // a globally scaled SAD guess would do for radicals.
    let (_, guess_coefficients) = symmetric_eigen(&core.h_core)?;
    let mut pa = density_from_coeff(&guess_coefficients, n_alpha, 1.0);
    let mut pb = density_from_coeff(&guess_coefficients, n_beta, 1.0);
    let mut e_old = 0.0;
    let mut eps_a = vec![0.0; nao];
    let mut c_a = Matrix::zeros(nao, nao);
    // Kept rather than discarded: an unrestricted calculation has two orbital sets, and the β
    // one is what a Molden export, an open-shell response and the UHF Hessian all need. It used
    // to be recomputed by re-diagonalizing both spin Focks after the fact.
    let mut eps_b = vec![0.0; nao];
    let mut c_b = Matrix::zeros(nao, nao);
    let mut converged = false;
    // The RMS density change the loop last saw, so a failure reports how far off it was.
    let mut last_change = f64::INFINITY;
    let mut iterations = 0;

    let mut hist_fa: Vec<Matrix> = Vec::new();
    let mut hist_fb: Vec<Matrix> = Vec::new();
    let mut hist_err: Vec<Matrix> = Vec::new();
    // One slot holds F_α, F_β and the stacked 2·nao × nao error matrix.
    let max_diis = diis_depth(nao, 4, options.scf_memory_mb);
    // The RHF A-DIIS implementation extrapolates one density/Fock pair.  It
    // cannot be reused for UHF without a spin-resolved energy functional.
    // Plain UHF iterations are robust from the SAD spin guess; only an
    // explicitly requested CDIIS mode uses the paired-Fock extrapolator.
    let use_cdiis = options.use_diis && options.accelerator == ScfAccelerator::Cdiis;

    for iter in 0..options.max_scf {
        iterations = iter + 1;
        let mut p_tot = pa.clone();
        for (t, b) in p_tot.as_mut_slice().iter_mut().zip(pb.as_slice()) {
            *t += *b;
        }
        let fa = build_fock_spin(molecule, basis, params, core, &p_tot, &pa)?;
        let fb = build_fock_spin(molecule, basis, params, core, &p_tot, &pb)?;

        let e_elec = 0.5
            * (p_tot.frobenius_dot(&core.h_core) + pa.frobenius_dot(&fa) + pb.frobenius_dot(&fb));

        // Combined DIIS error = [F_a,P_a] ⊕ [F_b,P_b].
        let ea = commutator(&fa, &pa);
        let eb = commutator(&fb, &pb);
        let mut err = Matrix::zeros(2 * nao, nao);
        for i in 0..nao {
            for j in 0..nao {
                err[(i, j)] = ea[(i, j)];
                err[(nao + i, j)] = eb[(i, j)];
            }
        }
        let (fa_use, fb_use) = if use_cdiis {
            hist_fa.push(fa.clone());
            hist_fb.push(fb.clone());
            hist_err.push(err);
            if hist_fa.len() > max_diis {
                hist_fa.remove(0);
                hist_fb.remove(0);
                hist_err.remove(0);
            }
            match diis_coeffs(&hist_err) {
                Some(coeffs) => (combine(&hist_fa, &coeffs), combine(&hist_fb, &coeffs)),
                None => (fa, fb),
            }
        } else {
            (fa, fb)
        };

        // Level shift (path only; the converged fixed point is unchanged).
        let shift = options.level_shift_ev;
        let mut fa_diag = level_shift(&fa_use, &pa, shift);
        let mut fb_diag = level_shift(&fb_use, &pb, shift);
        // Transition-metal d-block penalty, linearly annealed to zero (path
        // only; vanishes before convergence so the fixed point is the true one).
        if !d_pen_aos.is_empty() {
            let frac = 1.0 - (iter as f64) / (options.d_penalty_iters.max(1) as f64);
            let pen = options.d_penalty_start * frac.max(0.0);
            if pen > 0.0 {
                for &k in &d_pen_aos {
                    fa_diag[(k, k)] += pen;
                    fb_diag[(k, k)] += pen;
                }
            }
        }
        let (ea_eps, ca) = symmetric_eigen(&fa_diag)?;
        let (eb_eps, cb) = symmetric_eigen(&fb_diag)?;
        let mut pa_new = density_from_coeff(&ca, n_alpha, 1.0);
        let mut pb_new = density_from_coeff(&cb, n_beta, 1.0);

        let dp = rms_diff(&pa_new, &pa) + rms_diff(&pb_new, &pb);

        last_change = dp;
        // Density damping (path only): blend with the previous density.
        let lambda = options.damping;
        if lambda > 0.0 {
            for (n, o) in pa_new.as_mut_slice().iter_mut().zip(pa.as_slice()) {
                *n = (1.0 - lambda) * *n + lambda * *o;
            }
            for (n, o) in pb_new.as_mut_slice().iter_mut().zip(pb.as_slice()) {
                *n = (1.0 - lambda) * *n + lambda * *o;
            }
        }
        let de = (e_elec - e_old).abs();
        eps_a = ea_eps;
        c_a = ca;
        eps_b = eb_eps;
        c_b = cb;
        pa = pa_new;
        pb = pb_new;
        e_old = e_elec;
        // See the RHF loop: `err_norm` is evaluated before DIIS extrapolation,
        // so it must not by itself accept the newly generated density. Never
        // accept convergence while the d-block penalty is still active.
        let penalty_active = !d_pen_aos.is_empty() && iter < options.d_penalty_iters;
        if iter > 0 && !penalty_active && de < options.e_tol && dp < options.p_tol {
            converged = true;
            break;
        }
    }

    let mut density = pa.clone();
    for (t, b) in density.as_mut_slice().iter_mut().zip(pb.as_slice()) {
        *t += *b;
    }
    let mut spin = pa.clone();
    for (s, b) in spin.as_mut_slice().iter_mut().zip(pb.as_slice()) {
        *s -= *b;
    }
    // Final energy.
    let fa = build_fock_spin(molecule, basis, params, core, &density, &pa)?;
    let fb = build_fock_spin(molecule, basis, params, core, &density, &pb)?;
    let electronic_ev =
        0.5 * (density.frobenius_dot(&core.h_core) + pa.frobenius_dot(&fa) + pb.frobenius_dot(&fb));

    Ok(ScfState {
        last_change,
        density,
        spin_density: Some(spin),
        mo_energies: eps_a,
        mo_coeff: c_a,
        n_occ: n_alpha,
        mo_energies_beta: Some(eps_b),
        mo_coeff_beta: Some(c_b),
        n_beta,
        electronic_ev,
        converged,
        iterations,
        unrestricted: true,
    })
}

/// Linear combination `Σ c_i F_i` of the history, chunked over rayon: the
/// history is up to 8 matrices of `nao²` doubles, so this is a pure
/// memory-bandwidth pass that is worth splitting for large bases.
fn combine(fs: &[Matrix], coeffs: &[f64]) -> Matrix {
    use rayon::prelude::*;
    let (r, c) = (fs[0].rows, fs[0].cols);
    let mut out = Matrix::zeros(r, c);
    let n = r * c;
    if n < (1 << 18) {
        for (i, f) in fs.iter().enumerate() {
            let ci = coeffs[i];
            for (o, v) in out.as_mut_slice().iter_mut().zip(f.as_slice()) {
                *o += ci * v;
            }
        }
        return out;
    }
    const CHUNK: usize = 1 << 16;
    out.as_mut_slice()
        .par_chunks_mut(CHUNK)
        .enumerate()
        .for_each(|(k, dst)| {
            let lo = k * CHUNK;
            for (i, f) in fs.iter().enumerate() {
                let ci = coeffs[i];
                let src = &f.as_slice()[lo..lo + dst.len()];
                for (o, v) in dst.iter_mut().zip(src) {
                    *o += ci * v;
                }
            }
        });
    out
}

/// Solve the Pulay DIIS coefficient system from a stack of error matrices.
/// Convenience wrapper that forms the `⟨E_i,E_j⟩` Gram matrix from scratch;
/// the RHF path maintains it incrementally instead ([`AccelHistory`]).
pub(crate) fn diis_coeffs(es: &[Matrix]) -> Option<Vec<f64>> {
    let gram: Vec<Vec<f64>> = es
        .iter()
        .map(|ei| es.iter().map(|ej| ei.frobenius_dot(ej)).collect())
        .collect();
    diis_coeffs_from_gram(&gram)
}

/// Solve the Pulay DIIS coefficient system from a precomputed `⟨E_i,E_j⟩` Gram matrix.
pub(crate) fn diis_coeffs_from_gram(gram: &[Vec<f64>]) -> Option<Vec<f64>> {
    let n = gram.len();
    if n < 2 {
        return None;
    }
    // Normalize the Gram block before solving. The coefficients are invariant under
    // `B → B/s` (only the Lagrange multiplier scales), but the pivot test in
    // [`solve_bordered_small`] is an *absolute* threshold, and `⟨E_i,E_j⟩ ~ ‖E‖²` shrinks
    // quadratically as the SCF converges. Without this, a run whose error reaches ~1e-5
    // hands the solver a matrix whose entries are all ~1e-10 and gets either a spurious
    // `None` or wildly amplified coefficients — which shows up as DIIS converging nicely
    // and then walking back out into a limit cycle.
    let scale = gram
        .iter()
        .flat_map(|row| row.iter())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let dim = n + 1;
    let mut b = Matrix::zeros(dim, dim);
    for i in 0..n {
        for j in 0..n {
            b[(i, j)] = gram[i][j] / scale;
        }
        b[(i, n)] = -1.0;
        b[(n, i)] = -1.0;
    }
    let mut rhs = vec![0.0; dim];
    rhs[n] = -1.0;
    // The DIIS matrix (a small bordered saddle-point system) becomes singular near
    // convergence when the error vectors turn linearly dependent. A pivot-guarded
    // Gaussian elimination returns `None` there so the caller falls back to the plain
    // Fock — faer's LU instead returns a degenerate solution that derails DIIS. The heavy
    // O(n^3) eigendecomposition still uses faer; only this tiny solve is bespoke.
    solve_bordered_small(&b, &rhs)
}

/// Gaussian elimination with partial pivoting for the small DIIS system; returns `None`
/// if the matrix is (near-)singular.
fn solve_bordered_small(a: &Matrix, b: &[f64]) -> Option<Vec<f64>> {
    let n = a.rows;
    let mut m = a.clone();
    let mut rhs = b.to_vec();
    for col in 0..n {
        let mut pivot = col;
        let mut best = m[(col, col)].abs();
        for row in (col + 1)..n {
            let v = m[(row, col)].abs();
            if v > best {
                best = v;
                pivot = row;
            }
        }
        if best < 1.0e-12 {
            return None;
        }
        if pivot != col {
            for j in 0..n {
                let t = m[(col, j)];
                m[(col, j)] = m[(pivot, j)];
                m[(pivot, j)] = t;
            }
            rhs.swap(col, pivot);
        }
        for row in (col + 1)..n {
            let factor = m[(row, col)] / m[(col, col)];
            if factor == 0.0 {
                continue;
            }
            for j in col..n {
                let v = m[(col, j)];
                m[(row, j)] -= factor * v;
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    let mut x = vec![0.0; n];
    for col in (0..n).rev() {
        let mut sum = rhs[col];
        for j in (col + 1)..n {
            sum -= m[(col, j)] * x[j];
        }
        x[col] = sum / m[(col, col)];
    }
    Some(x)
}

/// Projected-gradient minimization of the A-DIIS quadratic
/// `f(c) = 2 Σ c_i ⟨D_i−D_n|F_n⟩ + Σ c_i c_j ⟨D_i−D_n|F_j−F_n⟩`
/// (Hu & Yang 2010) on the probability simplex `{c ≥ 0, Σc = 1}`. The
/// nonnegative weights prevent the runaway extrapolation plain DIIS can produce
/// far from convergence.
fn solve_adiis_simplex(d: &[f64], s: &[Vec<f64>]) -> Vec<f64> {
    let n = d.len();
    // Lipschitz estimate for the step size from (S + Sᵀ).
    let mut l: f64 = 1.0e-12;
    for (i, s_i) in s.iter().enumerate().take(n) {
        let mut row = 0.0;
        for (j, &s_ij) in s_i.iter().enumerate().take(n) {
            row += (s_ij + s[j][i]).abs();
        }
        l = l.max(row);
    }
    let lr = 1.0 / l;
    // Start from the latest point (all weight on the newest Fock/density).
    let mut c = vec![0.0; n];
    c[n - 1] = 1.0;
    for _ in 0..400 {
        // grad_k = 2 d_k + Σ_j (s_kj + s_jk) c_j
        let mut g = vec![0.0; n];
        for k in 0..n {
            let mut acc = 2.0 * d[k];
            for j in 0..n {
                acc += (s[k][j] + s[j][k]) * c[j];
            }
            g[k] = acc;
        }
        let trial: Vec<f64> = (0..n).map(|i| c[i] - lr * g[i]).collect();
        let proj = simplex_project(&trial);
        let mut delta = 0.0;
        for i in 0..n {
            delta += (proj[i] - c[i]).abs();
        }
        c = proj;
        if delta < 1.0e-12 {
            break;
        }
    }
    c
}

/// Euclidean projection of `v` onto the probability simplex `{c ≥ 0, Σc = 1}`.
fn simplex_project(v: &[f64]) -> Vec<f64> {
    let mut u = v.to_vec();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut css = 0.0;
    let mut rho = 0;
    let mut theta = 0.0;
    for (j, &uj) in u.iter().enumerate() {
        css += uj;
        let t = (css - 1.0) / (j as f64 + 1.0);
        if uj - t > 0.0 {
            rho = j + 1;
            theta = t;
        }
    }
    let _ = rho;
    v.iter().map(|&vi| (vi - theta).max(0.0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(xyz: &str, charge: f64) -> Pm3Result {
        let mol = Molecule::from_xyz_str(xyz, charge).unwrap();
        let params = Pm3Parameters::standard().unwrap();
        run_pm3(&mol, &params, &Pm3Options::default()).unwrap()
    }

    /// The reported frontier is the frontier of **both** spin channels.
    ///
    /// `mo_energies` is the α spectrum and `n_occ` the α count, so reading HOMO and LUMO off
    /// them alone gives the α–α gap, which for an open shell is not the gap: a radical has
    /// `n_α > n_β`, so its β LUMO lies below its α LUMO. The number was surfaced all the way to
    /// `pm3_rs.single_point()["lumo_ev"]`.
    #[test]
    fn the_open_shell_frontier_reads_both_spins() {
        const METHYL: &str = "4\nmethyl\nC 0.0 0.0 0.0\nH 1.0787 0.0 0.0\n\
                              H -0.5393 0.9341 0.0\nH -0.5393 -0.9341 0.0\n";
        let result = run_mult(METHYL, 0.0, 2);
        assert!(result.unrestricted, "a doublet takes the unrestricted path");

        let beta = result
            .mo_energies_beta
            .as_ref()
            .expect("an unrestricted result carries the beta spectrum");
        let alpha_lumo = result.mo_energies[result.n_occ];
        let beta_lumo = beta[result.n_beta];

        // Non-vacuity: the two channels really do disagree here, so the test can tell the
        // both-spin rule from the alpha-only one.
        assert!(
            (alpha_lumo - beta_lumo).abs() > 1.0e-3,
            "the two LUMOs agree to {:.3e}; this system cannot distinguish the two rules",
            (alpha_lumo - beta_lumo).abs()
        );
        assert!(
            beta_lumo < alpha_lumo,
            "a radical's singly occupied level leaves the beta LUMO below the alpha one"
        );

        let lumo = result.lumo_ev.expect("a frontier exists");
        assert!(
            (lumo - beta_lumo).abs() < 1.0e-12,
            "the reported LUMO is {lumo}, the alpha-only answer; the real one is {beta_lumo}"
        );
        let homo = result.homo_ev.expect("a frontier exists");
        let expected_homo = result.mo_energies[result.n_occ - 1].max(beta[result.n_beta - 1]);
        assert!((homo - expected_homo).abs() < 1.0e-12);
        assert!(homo <= lumo, "the gap came out negative: {homo} to {lumo}");
    }

    fn run_mult(xyz: &str, charge: f64, mult: usize) -> Pm3Result {
        let mol = Molecule::from_xyz_str(xyz, charge).unwrap();
        let params = Pm3Parameters::standard().unwrap();
        let opts = Pm3Options {
            charge,
            multiplicity: mult,
            ..Pm3Options::default()
        };
        run_pm3(&mol, &params, &opts).unwrap()
    }

    #[test]
    fn diis_depth_respects_the_memory_budget() {
        // Unlimited budget keeps the full depth.
        assert_eq!(diis_depth(1800, 3, 0), MAX_DIIS_DEPTH);
        // 512 MiB / (3 · 1800² · 8 B = 74.2 MiB) = 6 slots.
        assert_eq!(diis_depth(1800, 3, 512), 6);
        // A huge basis is floored at 2 rather than 0.
        assert_eq!(diis_depth(8000, 3, 512), 2);
        // ...and at that size a depth-2 A-DIIS history does not fit, so the
        // caller degrades to CDIIS instead of blowing the budget.
        assert!(!diis_depth_fits(8000, 3, 512));
        assert!(diis_depth_fits(1800, 3, 512));
        // A small system is unaffected.
        assert_eq!(diis_depth(60, 3, 512), MAX_DIIS_DEPTH);
    }

    #[test]
    fn scf_memory_budget_does_not_change_the_converged_energy() {
        // Trimming the accelerator history is a path change only: the SCF fixed
        // point (and hence every reported quantity) must be identical.
        let xyz =
            "4\nformaldehyde\nC 0.0 0.0 0.0\nO 0.0 0.0 1.21\nH 0.94 0.0 -0.54\nH -0.94 0.0 -0.54\n";
        let molecule = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Pm3Parameters::standard().unwrap();
        let full = run_pm3(&molecule, &params, &Pm3Options::default()).unwrap();
        // 1 MiB forces the minimum depth and the A-DIIS → CDIIS degradation.
        let tight = run_pm3(
            &molecule,
            &params,
            &Pm3Options {
                scf_memory_mb: 1,
                ..Pm3Options::default()
            },
        )
        .unwrap();
        assert!(tight.converged);
        assert!(
            (full.total_ev - tight.total_ev).abs() < 1.0e-8,
            "full={} tight={}",
            full.total_ev,
            tight.total_ev
        );
    }

    #[test]
    fn pair_cache_budget_is_reported_not_aborted() {
        // 180 atoms → 16 110 pairs → ~3 MiB of pair cache.
        let mut xyz = String::from("180\nwater lattice\n");
        for i in 0..60 {
            let (x, y, z) = (
                3.1 * (i % 5) as f64,
                3.1 * ((i / 5) % 4) as f64,
                3.1 * (i / 20) as f64,
            );
            xyz.push_str(&format!(
                "O {x} {y} {z}\nH {} {y} {z}\nH {} {} {z}\n",
                x + 0.96,
                x - 0.24,
                y + 0.93
            ));
        }
        let molecule = Molecule::from_xyz_str(&xyz, 0.0).unwrap();
        let params = Pm3Parameters::standard().unwrap();
        let basis = Basis::build(&molecule, &params).unwrap();
        let required = crate::hamiltonian::pair_cache_bytes(&basis);
        assert!(required > 2 * 1024 * 1024, "cache estimate {required} B");

        // A budget below the requirement must fail cleanly, before allocating.
        let err = match crate::hamiltonian::build_core_limited(&molecule, &basis, &params, 1, None)
        {
            Err(err) => err,
            Ok(_) => panic!("a 1 MiB budget must reject a 3 MiB pair cache"),
        };
        let message = err.to_string();
        assert!(
            matches!(err, Pm3Error::ResourceLimit { .. }) && message.contains("pair cache"),
            "unexpected error: {message}"
        );
        // Zero disables the check; a large budget passes.
        assert!(
            crate::hamiltonian::build_core_limited(&molecule, &basis, &params, 0, None).is_ok()
        );
        assert!(
            crate::hamiltonian::build_core_limited(&molecule, &basis, &params, 4096, None).is_ok()
        );
    }

    #[test]
    fn molecule_charge_and_multiplicity_are_honored() {
        let xyz = "2\nh2+\nH 0.0 0.0 0.0\nH 0.74 0.0 0.0\n";
        let params = Pm3Parameters::standard().unwrap();
        let molecule_metadata = Molecule::from_xyz_str(xyz, 1.0)
            .unwrap()
            .with_multiplicity(2);
        let from_molecule = run_pm3(&molecule_metadata, &params, &Pm3Options::default()).unwrap();

        let neutral_metadata = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let explicit = run_pm3(
            &neutral_metadata,
            &params,
            &Pm3Options {
                charge: 1.0,
                multiplicity: 2,
                ..Pm3Options::default()
            },
        )
        .unwrap();
        assert!(from_molecule.unrestricted);
        assert!((from_molecule.total_ev - explicit.total_ev).abs() < 1.0e-12);
        assert!((from_molecule.charges.iter().sum::<f64>() - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn conflicting_molecule_and_option_metadata_is_rejected() {
        let xyz = "2\nh2\nH 0.0 0.0 0.0\nH 0.74 0.0 0.0\n";
        let molecule = Molecule::from_xyz_str(xyz, 1.0).unwrap();
        let params = Pm3Parameters::standard().unwrap();
        let err = run_pm3(
            &molecule,
            &params,
            &Pm3Options {
                charge: -1.0,
                multiplicity: 1,
                ..Pm3Options::default()
            },
        )
        .unwrap_err();
        assert!(err.to_string().contains("conflicting charge"));
    }

    #[test]
    fn water_heat_of_formation() {
        let xyz =
            "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.9584 0.0000 0.0000\nH -0.2400 0.9278 0.0000\n";
        let r = run(xyz, 0.0);
        eprintln!(
            "H2O: dHf={:.3} kcal/mol  elec={:.4} eV core={:.4} eV  dipole={:.3} D  charges={:?}  iters={}",
            r.heat_of_formation_kcal, r.electronic_ev, r.core_ev, r.dipole_magnitude, r.charges, r.iterations
        );
        assert!(r.converged);
        // MOPAC v23.2.5, PM3 1SCF, same geometry: -53.2301514568744 kcal/mol
        // and 1.762063533710513 D (see tools/oracle/PM3_VALIDATION.md).
        assert!((r.heat_of_formation_kcal - (-53.2301514568744)).abs() < 0.005);
        assert!((r.dipole_magnitude - 1.762063533710513).abs() < 0.002);
        let qsum: f64 = r.charges.iter().sum();
        assert!(qsum.abs() < 1e-6);
        assert!(r.charges[0] < 0.0 && r.charges[1] > 0.0);
    }

    #[test]
    fn water_no_diis_debug() {
        let xyz =
            "3\nwater\nO 0.0000 0.0000 0.0000\nH 0.9584 0.0000 0.0000\nH -0.2400 0.9278 0.0000\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Pm3Parameters::standard().unwrap();
        let opts = Pm3Options {
            use_diis: false,
            max_scf: 500,
            ..Pm3Options::default()
        };
        let r = run_pm3(&mol, &params, &opts).unwrap();
        eprintln!(
            "H2O(no-diis): dHf={:.3} conv={} iters={}",
            r.heat_of_formation_kcal, r.converged, r.iterations
        );
    }

    #[test]
    fn methane_heat_of_formation() {
        let xyz = "5\nmethane\nC 0.0000 0.0000 0.0000\nH 0.6276 0.6276 0.6276\nH -0.6276 -0.6276 0.6276\nH -0.6276 0.6276 -0.6276\nH 0.6276 -0.6276 -0.6276\n";
        let r = run(xyz, 0.0);
        eprintln!(
            "CH4: dHf={:.3} kcal/mol dipole={:.3} D iters={}",
            r.heat_of_formation_kcal, r.dipole_magnitude, r.iterations
        );
        assert!(r.converged);
    }

    #[test]
    fn accelerators_agree_on_energy() {
        // A-DIIS→CDIIS, plain CDIIS, and no acceleration must reach the same converged
        // energy (same SCF fixed point); the hybrid should not need more iterations.
        let xyz =
            "4\nformaldehyde\nC 0.0 0.0 0.0\nO 0.0 0.0 1.21\nH 0.94 0.0 -0.54\nH -0.94 0.0 -0.54\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Pm3Parameters::standard().unwrap();
        let run = |acc: ScfAccelerator| {
            let opts = Pm3Options {
                accelerator: acc,
                ..Pm3Options::default()
            };
            run_pm3(&mol, &params, &opts).unwrap()
        };
        let hybrid = run(ScfAccelerator::AdiisCdiis);
        let cdiis = run(ScfAccelerator::Cdiis);
        let none = run(ScfAccelerator::None);
        eprintln!(
            "iters: hybrid={} cdiis={} none={}  E(hybrid)={:.6}",
            hybrid.iterations, cdiis.iterations, none.iterations, hybrid.total_ev
        );
        assert!((hybrid.total_ev - cdiis.total_ev).abs() < 1e-6);
        assert!((hybrid.total_ev - none.total_ev).abs() < 1e-6);
        // Both accelerated paths should be far faster than plain iteration.
        assert!(hybrid.iterations < none.iterations);
    }

    #[test]
    fn methyl_radical_uhf() {
        // Planar CH3 radical (doublet): UHF must converge, be open-shell, and carry net spin.
        let xyz = "4\nmethyl\nC 0.0 0.0 0.0\nH 1.079 0.0 0.0\nH -0.5395 0.9344 0.0\nH -0.5395 -0.9344 0.0\n";
        let r = run_mult(xyz, 0.0, 2);
        eprintln!(
            "CH3.: dHf={:.3} kcal/mol unrestricted={} iters={}",
            r.heat_of_formation_kcal, r.unrestricted, r.iterations
        );
        assert!(r.converged);
        assert!(r.unrestricted);
        // Total spin population (∫ P_α − P_β) should be ≈ 1 unpaired electron.
        let spin = r.spin_density.as_ref().unwrap();
        let n_spin: f64 = (0..spin.rows).map(|i| spin[(i, i)]).sum();
        assert!((n_spin - 1.0).abs() < 1e-6, "net spin {n_spin}");
        // Frozen regression value at this geometry (r(C-H) = 1.079 Å). The
        // MOPAC v23.2.5 UHF/DOUBLET oracle comparison for CH3 uses the
        // r = 1.078 Å geometry and lives in `tests/molecules.rs`
        // (28.0055283157567 kcal/mol); see tools/oracle/PM3_VALIDATION.md.
        assert!((r.heat_of_formation_kcal - 28.0222138930039).abs() < 0.005);
    }
}
