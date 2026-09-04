// SPDX-License-Identifier: GPL-3.0-or-later

//! The divide-and-conquer SCF, molecular and Γ-point periodic.
//!
//! Both share everything except how the Fock matrix is built, which is the only place the
//! boundary conditions appear at all: a subsystem does not know or care whether the potential it
//! sits in came from a lattice sum. That is why the periodic path here is a few lines rather than
//! a second implementation — the hard part of periodicity lives in [`crate::pbc`], and by the time
//! the Fock matrix exists it has already been dealt with.

use rayon::prelude::*;

use crate::basis::Basis;
use crate::dc::partition::{partition, weight, DcOptions, Partition};
use crate::dc::pattern::{DensityPattern, SparseMatrix};
use crate::densitydiis::DensityDiis;
use crate::error::{Pm3Error, Result};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::params::Pm3Parameters;
use crate::pbc::gamma::PeriodicOptions;
use crate::scf::{Pm3Options, Reference};
use crate::system::Molecule;

/// A molecular divide-and-conquer result. Mirrors [`crate::scf::Pm3Result`] where the two overlap.
#[derive(Clone, Debug)]
pub struct DcResult {
    pub density: Matrix,
    pub spin_density: Option<Matrix>,
    pub unrestricted: bool,
    pub electronic_ev: f64,
    pub core_ev: f64,
    pub total_ev: f64,
    pub heat_of_formation_kcal: f64,
    pub charges: Vec<f64>,
    /// The global chemical potential the subsystems were filled to (eV).
    pub fermi_ev: f64,
    /// β-spin Fermi level; equal to [`DcResult::fermi_ev`] unless the magnetization is fixed.
    pub fermi_beta_ev: f64,
    pub iterations: usize,
    pub converged: bool,
    /// How many orbital pairs the partitioning dropped, and the largest subsystem it produced.
    /// Together these say what was traded for what.
    pub dropped_pairs: usize,
    pub largest_subsystem: usize,
    pub n_subsystems: usize,
}

/// A Γ-point periodic divide-and-conquer result.
#[derive(Clone, Debug)]
pub struct DcPeriodicResult {
    pub density: Matrix,
    pub spin_density: Option<Matrix>,
    pub unrestricted: bool,
    pub electronic_ev: f64,
    pub core_ev: f64,
    pub correction_ev: f64,
    pub ewald_ev: f64,
    pub total_ev: f64,
    pub heat_of_formation_kcal: f64,
    pub charges: Vec<f64>,
    pub fermi_ev: f64,
    pub fermi_beta_ev: f64,
    pub iterations: usize,
    pub converged: bool,
    pub dropped_pairs: usize,
    pub largest_subsystem: usize,
    pub n_subsystems: usize,
    /// See [`crate::pbc::gamma::PeriodicResult::gamma_margin`] — the condition applies here too,
    /// and divide-and-conquer does nothing to relax it.
    pub gamma_margin: f64,
}

/// Run a molecular divide-and-conquer SCF.
pub fn run_dc(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    dc: &DcOptions,
) -> Result<DcResult> {
    if dc.long_range_cutoff.is_some() {
        return run_dc_screened(molecule, params, options, dc);
    }
    let basis = Basis::build(molecule, params)?;
    let core = crate::hamiltonian::build_core_limited(
        molecule,
        &basis,
        params,
        crate::scf::pair_cache_limit(options),
        options.field,
    )?;
    let counts = electron_counts(molecule, params, options)?;
    let partitioning = partition(molecule, &basis, dc)?;

    // The core Hamiltonian is read onto the pattern once and the dense copy released. Nothing
    // downstream needs it: the Fock build works from the sparse one, the reported energy is a
    // trace against a density that is zero outside the pattern, and the capped-bond correction
    // is a term-by-term product with that same density — so every entry outside the pattern is
    // multiplied by a structural zero wherever it appears.
    let pattern = DensityPattern::from_partition(&partitioning, basis.nao);
    let h_core = pattern.read_dense(&core.h_core);
    let core = crate::hamiltonian::CoreHamiltonian {
        h_core: Matrix::zeros(0, 0),
        pairs: core.pairs,
    };
    let build =
        |p_total: &SparseMatrix, p_spin: &SparseMatrix, out: &mut SparseMatrix| -> Result<()> {
            crate::fock::build_fock_spin_sparse(
                molecule, &basis, params, &core, &h_core, &pattern, p_total, p_spin, out,
            )
        };
    let atomic = crate::pbc::gamma::initial_density(molecule, params, &basis)?;
    let state = loop_scf(&basis, &partitioning, options, dc, &counts, &atomic, build)?;

    let electronic = 0.5
        * (pattern.dot_sparse(&state.density, &h_core)
            + pattern.dot_sparse(&state.alpha, &state.fock_alpha)
            + pattern.dot_sparse(&state.beta, &state.fock_beta))
        + crate::hamiltonian::capped_bond_energy_correction_sparse(
            molecule,
            &basis,
            &pattern,
            &state.density,
            &h_core,
        );
    // The external field's density-independent half, exactly as `run_pm3` accounts for it: the
    // electronic partner `+Tr[P M·f]` is already inside `h_core` above, and dropping the nuclear
    // half here would leave a divide-and-conquer energy in a field wrong by `−Σ_A Z_A R_A·f`
    // while every convergence diagnostic still looked healthy.
    let field_nuclear_ev = match options.field {
        Some(f) => crate::dipole::field_terms(molecule, params, &basis, f)?.1,
        None => 0.0,
    };
    let core_ev = crate::repulsion::core_core_energy(molecule, params)?
        + field_nuclear_ev
        + crate::corrections::correction_energy(molecule, options.variant);
    let total_ev = electronic + core_ev;
    let (e_isol, eheat) = isolated_sums(molecule, params)?;

    Ok(DcResult {
        charges: mulliken(molecule, params, &basis, &pattern, &state.density)?,
        density: pattern.to_dense(&state.density),
        spin_density: counts
            .unrestricted
            .then(|| subtract(&pattern, &state.alpha, &state.beta)),
        unrestricted: counts.unrestricted,
        electronic_ev: electronic,
        core_ev,
        total_ev,
        heat_of_formation_kcal: (total_ev - e_isol + eheat) * crate::constants::EV_TO_KCAL,
        fermi_ev: state.fermi_alpha,
        fermi_beta_ev: state.fermi_beta,
        iterations: state.iterations,
        converged: state.converged,
        dropped_pairs: partitioning.dropped_pairs(),
        largest_subsystem: partitioning.largest_subsystem(),
        n_subsystems: partitioning.subsystems.len(),
    })
}

/// Run a Γ-point periodic divide-and-conquer SCF.
pub fn run_dc_gamma(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
    dc: &DcOptions,
) -> Result<DcPeriodicResult> {
    crate::pbc::refuse_field(molecule, options)?;
    let cell = molecule.cell.ok_or_else(|| {
        Pm3Error::InvalidInput("a periodic calculation needs a cell on the molecule".to_string())
    })?;
    let setup = crate::pbc::gamma::build_setup(molecule, params, periodic)?;
    let basis = Basis::build(molecule, params)?;
    let counts = electron_counts(molecule, params, options)?;
    let partitioning = partition(molecule, &basis, dc)?;

    // The lattice sum's contribution to the Fock matrix depends on the density, so it is rebuilt
    // each iteration exactly as the full periodic SCF does. Divide-and-conquer changes how the
    // density is obtained from the Fock matrix, and nothing about how the Fock matrix is obtained
    // from the density.
    // The core Hamiltonian goes onto the pattern and the dense copy is released, as in the
    // molecular path. Nothing here reads it outside the pattern: the Fock build works from the
    // sparse one and the reported energy is a trace against a density that is structurally zero
    // there.
    let pattern = DensityPattern::from_partition(&partitioning, basis.nao);
    let h_core = pattern.read_dense(&setup.h_core);
    let mut setup = setup;
    setup.h_core = Matrix::zeros(0, 0);
    let setup = setup;
    let build =
        |p_total: &SparseMatrix, p_spin: &SparseMatrix, out: &mut SparseMatrix| -> Result<()> {
            let mut sites = setup.sites.clone();
            crate::pbc::gamma::write_electron_charges_sparse(&setup, &pattern, p_total, &mut sites);
            let electrons = crate::pbc::ewald::ewald_potentials_cached(
                &cell,
                &sites,
                &setup.ewald_params,
                &setup.ewald_context,
            )?;
            crate::pbc::gamma::build_periodic_fock_sparse(
                molecule, params, &setup, &h_core, &pattern, p_total, p_spin, out,
            )?;
            crate::pbc::gamma::add_site_potential_sparse(&setup, &pattern, out, &electrons);
            Ok(())
        };
    let atomic = crate::pbc::gamma::initial_density(molecule, params, &basis)?;
    let state = loop_scf(&basis, &partitioning, options, dc, &counts, &atomic, build)?;

    // One more lattice sum at the converged density, for the reported Ewald energy.
    let electron_ewald_ev = {
        let mut sites = setup.sites.clone();
        crate::pbc::gamma::write_electron_charges_sparse(
            &setup,
            &pattern,
            &state.density,
            &mut sites,
        );
        crate::pbc::ewald::ewald_potentials_cached(
            &cell,
            &sites,
            &setup.ewald_params,
            &setup.ewald_context,
        )?
        .energy_ev
    };

    let electronic = 0.5
        * (pattern.dot_sparse(&state.density, &h_core)
            + pattern.dot_sparse(&state.alpha, &state.fock_alpha)
            + pattern.dot_sparse(&state.beta, &state.fock_beta));
    let correction_ev = crate::corrections::periodic::periodic_correction_energy(
        molecule,
        options.variant,
        &periodic.correction_cutoffs,
    );
    let total_ev = electronic + setup.core_ev + correction_ev;
    let (e_isol, eheat) = isolated_sums(molecule, params)?;

    Ok(DcPeriodicResult {
        charges: mulliken(molecule, params, &basis, &pattern, &state.density)?,
        density: pattern.to_dense(&state.density),
        spin_density: counts
            .unrestricted
            .then(|| subtract(&pattern, &state.alpha, &state.beta)),
        unrestricted: counts.unrestricted,
        electronic_ev: electronic,
        core_ev: setup.core_ev,
        correction_ev,
        ewald_ev: setup.core_ewald_ev + electron_ewald_ev,
        total_ev,
        heat_of_formation_kcal: (total_ev - e_isol + eheat) * crate::constants::EV_TO_KCAL,
        fermi_ev: state.fermi_alpha,
        fermi_beta_ev: state.fermi_beta,
        iterations: state.iterations,
        converged: state.converged,
        dropped_pairs: partitioning.dropped_pairs(),
        largest_subsystem: partitioning.largest_subsystem(),
        n_subsystems: partitioning.subsystems.len(),
        gamma_margin: setup.gamma_margin,
    })
}

/// Molecular divide-and-conquer with the long-range Coulomb handed to a point-charge model.
///
/// An isolated system is the zero-dimensional member of the periodic family, so this is
/// [`run_dc_gamma`] on a cell with no periodic direction — no images, no reciprocal space, and
/// a lattice sum that is a plain pair sum. What it buys is the crystal's near field: a
/// neighbour-list pair table that grows linearly with the system instead of the dense
/// `O(N²)` cache, which is what dominates a large run.
///
/// See [`DcOptions::long_range_cutoff`] for the accuracy this trades away and how much of it.
fn run_dc_screened(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    dc: &DcOptions,
) -> Result<DcResult> {
    let cutoff = dc
        .long_range_cutoff
        .expect("only called when the cutoff is set");
    if molecule.atoms.iter().any(|atom| atom.z == 102) {
        return Err(Pm3Error::InvalidInput(
            "the capped-bond (Cb) energy correction is not part of the screened long-range \
             path; leave DcOptions::long_range_cutoff unset for a molecule containing Cb"
                .to_string(),
        ));
    }
    // The periodic machinery this delegates to does not read `Pm3Options::field`, so a field
    // handed here would be dropped rather than applied — and the unscreened path beside it does
    // apply one, which makes the silence worse than the refusal.
    if options.field.is_some() {
        return Err(Pm3Error::InvalidInput(
            "a uniform electric field is not carried by the screened long-range path, which \
             runs on the periodic machinery; leave DcOptions::long_range_cutoff unset to use \
             the field"
                .to_string(),
        ));
    }

    let mut isolated = molecule.clone();
    if isolated.cell.is_none() {
        isolated.cell = Some(crate::cell::Cell::isolated());
    }
    // The switch is faded over the last few Bohr before the cutoff, as everywhere else: a hard
    // cut would put a step in the energy and therefore a delta function in the force.
    //
    // The resonance and exchange range moves with it. Those are separate physics — they follow
    // the overlap and so decay exponentially, where the Coulomb correction decays as a power —
    // and their own default is shorter. Tying them anyway is what makes this a *single knob
    // that converges*: widening it must recover the dense answer, and it cannot if a second,
    // fixed radius is still cutting the exchange. It only ever widens, never narrows.
    let periodic = PeriodicOptions {
        switch: crate::pbc::screen::SwitchRange {
            on: (cutoff - crate::pbc::screen::DEFAULT_SWITCH_WIDTH).max(1.0),
            off: cutoff,
        },
        short_range_cutoff: cutoff.max(crate::pbc::gamma::DEFAULT_SHORT_RANGE_CUTOFF),
        ..PeriodicOptions::default()
    };
    let result = run_dc_gamma(&isolated, params, options, &periodic, dc)?;

    Ok(DcResult {
        density: result.density,
        spin_density: result.spin_density,
        unrestricted: result.unrestricted,
        electronic_ev: result.electronic_ev,
        // The molecular path's `core_ev` is the core–core repulsion plus the classical
        // corrections; the periodic one reports them separately.
        core_ev: result.core_ev + result.correction_ev,
        total_ev: result.total_ev,
        heat_of_formation_kcal: result.heat_of_formation_kcal,
        charges: result.charges,
        fermi_ev: result.fermi_ev,
        fermi_beta_ev: result.fermi_beta_ev,
        iterations: result.iterations,
        converged: result.converged,
        dropped_pairs: result.dropped_pairs,
        largest_subsystem: result.largest_subsystem,
        n_subsystems: result.n_subsystems,
    })
}

/// Electron bookkeeping, shared with the full paths' conventions.
struct Counts {
    n_alpha: f64,
    n_beta: f64,
    unrestricted: bool,
}

fn electron_counts(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
) -> Result<Counts> {
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
    let n_unpaired = (multiplicity - 1) as f64;
    if n_elec < n_unpaired || n_elec < 0.0 {
        return Err(Pm3Error::InvalidInput(format!(
            "electron count {n_elec} is incompatible with multiplicity {multiplicity}"
        )));
    }
    let n_alpha = 0.5 * (n_elec + n_unpaired);
    let n_beta = 0.5 * (n_elec - n_unpaired);
    let unrestricted = match options.reference {
        Reference::Auto => n_alpha != n_beta,
        Reference::Uhf => true,
        Reference::Rhf => {
            if n_alpha != n_beta {
                return Err(Pm3Error::InvalidInput(
                    "RHF requested for an open-shell system".to_string(),
                ));
            }
            false
        }
    };
    Ok(Counts {
        n_alpha,
        n_beta,
        unrestricted,
    })
}

struct DcState {
    density: SparseMatrix,
    alpha: SparseMatrix,
    beta: SparseMatrix,
    fock_alpha: SparseMatrix,
    fock_beta: SparseMatrix,
    fermi_alpha: f64,
    fermi_beta: f64,
    iterations: usize,
    converged: bool,
}

/// The SCF loop, with the Fock build supplied by the caller.
fn loop_scf<F>(
    basis: &Basis,
    partitioning: &Partition,
    options: &Pm3Options,
    dc: &DcOptions,
    counts: &Counts,
    atomic: &Matrix,
    build_fock: F,
) -> Result<DcState>
where
    F: Fn(&SparseMatrix, &SparseMatrix, &mut SparseMatrix) -> Result<()>,
{
    let nao = basis.nao;
    let total_electrons = counts.n_alpha + counts.n_beta;
    let share = |electrons: f64| {
        if total_electrons > 0.0 {
            electrons / total_electrons
        } else {
            0.5
        }
    };
    // DIIS sees only the entries the partitioning can populate. Everything else is
    // structurally zero in both the current and the fresh density, so it contributes nothing
    // to the extrapolation and nothing to the residual — it only costs memory traffic, and at
    // a thousand atoms it costs most of the iteration. See [`DensityPattern`].
    let pattern = DensityPattern::from_partition(partitioning, nao);
    // The starting guess arrives dense and is read onto the pattern once. Nothing after this
    // point holds an `nao × nao` array until the results are handed back.
    let mut alpha = pattern.read_dense(&scaled(atomic, share(counts.n_alpha)));
    let mut beta = pattern.read_dense(&scaled(atomic, share(counts.n_beta)));

    let damping = if options.damping > 0.0 {
        options.damping.clamp(0.0, 0.95)
    } else {
        0.3
    };
    let mut last_energy = f64::INFINITY;
    let mut converged = false;
    let mut iterations = 0;
    let mut fock_alpha = pattern.zeros();
    let mut fock_beta = pattern.zeros();
    let mut fermi_alpha = 0.0;
    let mut fermi_beta = 0.0;

    let mut history = DensityDiis::new(2 * pattern.nnz(), options.scf_memory_mb);
    let mut current = vec![0.0; 2 * pattern.nnz()];
    let mut fresh = vec![0.0; 2 * pattern.nnz()];

    // Every matrix the loop touches is allocated once here, and on the pattern rather than
    // dense. Each holds `nnz` doubles instead of `nao²`: at 960 atoms that is a couple of
    // megabytes against a hundred and twenty-eight, and the six of them together were most of a
    // gigabyte of zeros. Nothing here reads outside the pattern — the workspaces only ever go
    // through it, and the subsystem solve looks exactly where some subsystem holds both
    // orbitals, which is what the pattern *is* — so the numbers are unchanged.
    let mut p_total = pattern.zeros();
    let mut new_total = pattern.zeros();
    let mut fresh_alpha = pattern.zeros();
    // The beta workspace is only ever written by an unrestricted run; a restricted one reads the
    // alpha matrices instead, so it does not allocate a second copy of them.
    let mut fresh_beta = pattern.zeros();

    let profile = std::env::var("PM3_DC_PROFILE").is_ok();
    let (mut t_fock, mut t_solve, mut t_rest) = (0.0, 0.0, 0.0);
    for iteration in 1..=options.max_scf {
        iterations = iteration;
        let mut clock = std::time::Instant::now();
        pattern.add_sparse(&alpha, &beta, &mut p_total);
        build_fock(&p_total, &alpha, &mut fock_alpha)?;
        // Restricted: the two spins see the same Fock and reach the same density, so the β
        // channel reads the α matrices rather than owning copies of them. Cloning cost a full
        // `nao × nao` write twice per iteration — sixty megabytes at a thousand atoms, moved so
        // that a matrix already in hand could have a second name. The converged Fock is rebuilt
        // below the loop, which is where `fock_beta` gets its real value.
        if counts.unrestricted {
            build_fock(&p_total, &beta, &mut fock_beta)?;
        }

        t_fock += clock.elapsed().as_secs_f64();
        clock = std::time::Instant::now();
        let mu_alpha = solve_channel(
            &fock_alpha,
            partitioning,
            counts.n_alpha,
            dc,
            &pattern,
            &mut fresh_alpha,
        )?;
        let mu_beta = if counts.unrestricted {
            solve_channel(
                &fock_beta,
                partitioning,
                counts.n_beta,
                dc,
                &pattern,
                &mut fresh_beta,
            )?
        } else {
            mu_alpha
        };
        fermi_alpha = mu_alpha;
        fermi_beta = mu_beta;

        t_solve += clock.elapsed().as_secs_f64();
        clock = std::time::Instant::now();
        let (beta_fock, beta_fresh) = if counts.unrestricted {
            (&fock_beta, &fresh_beta)
        } else {
            (&fock_alpha, &fresh_alpha)
        };
        pattern.add_sparse(&fresh_alpha, beta_fresh, &mut new_total);
        let electronic = 0.5
            * (pattern.dot_sparse(&new_total, &fock_alpha)
                + pattern.dot_sparse(&fresh_alpha, &fock_alpha)
                + pattern.dot_sparse(beta_fresh, beta_fock));
        let change = pattern.rms_difference_sparse(&new_total, &p_total);

        pattern.damp_sparse(&mut alpha, &fresh_alpha, damping);
        let blended_beta = beta_fresh.clone();
        pattern.damp_sparse(&mut beta, &blended_beta, damping);

        // Pulay DIIS on the density, as in the k-point path and for the same reason: the iterated
        // quantity here is a density, not a Fock matrix. Damping alone does not reliably converge
        // a truncated partitioning — the local diagonalizations shift occupations between
        // subsystems as the chemical potential moves, and the fixed point is stiff.
        pattern.gather_pair_sparse(&alpha, &beta, &mut current);
        pattern.gather_pair_sparse(&fresh_alpha, &blended_beta, &mut fresh);
        let residual: Vec<f64> = fresh.iter().zip(&current).map(|(a, b)| a - b).collect();
        history.push(current.clone(), residual);
        if let Some(blended) = history.extrapolate() {
            pattern.scatter_pair_sparse(&blended, &mut alpha, &mut beta);
        }

        t_rest += clock.elapsed().as_secs_f64();
        if (electronic - last_energy).abs() < options.e_tol && change < options.p_tol {
            converged = true;
            break;
        }
        last_energy = electronic;
    }
    if profile {
        eprintln!(
            "  [dc nao={nao} its={iterations}]  fock {t_fock:.3}s  solve {t_solve:.3}s  \
             density {t_rest:.3}s"
        );
    }

    if !converged {
        return Err(Pm3Error::ScfNotConverged {
            iterations,
            error: f64::NAN,
        });
    }

    // Rebuild the Fock matrices at the converged density so the reported energy is evaluated
    // there rather than one iteration behind.
    let mut sparse_density = pattern.zeros();
    pattern.add_sparse(&alpha, &beta, &mut sparse_density);
    build_fock(&sparse_density, &alpha, &mut fock_alpha)?;
    if counts.unrestricted {
        build_fock(&sparse_density, &beta, &mut fock_beta)?;
    } else {
        fock_beta = fock_alpha.clone();
    }

    // Handed back on the pattern. The callers densify exactly what their public result holds and
    // nothing else — the energies contract sparsely, the charges read a diagonal, and only the
    // density a caller asked for becomes an `nao × nao` array.
    Ok(DcState {
        density: sparse_density,
        alpha,
        beta,
        fock_alpha,
        fock_beta,
        fermi_alpha,
        fermi_beta,
        iterations,
        converged,
    })
}

/// One spin channel: diagonalize every subsystem, then bisect the chemical potential until the
/// assembled density holds `target` electrons.
///
/// The assembled density is written into `density`, which the caller owns across iterations —
/// see the allocation note in [`loop_scf`]. It is cleared through the pattern rather than
/// wholesale, so both the clearing and the writing cost `O(N)`.
fn solve_channel(
    fock: &SparseMatrix,
    partitioning: &Partition,
    target: f64,
    dc: &DcOptions,
    pattern: &DensityPattern,
    density: &mut SparseMatrix,
) -> Result<f64> {
    // Each subsystem is diagonalized once; only the occupation depends on `μ`, so the bisection
    // reuses the eigenpairs rather than re-solving. This is what keeps the bisection free.
    //
    // The subsystems are independent by construction, which is the other half of what
    // divide-and-conquer buys: not just a smaller problem, but many of them. Each solve is small
    // enough that faer's own parallelism would be pure overhead, so the parallelism goes here,
    // across subsystems, rather than inside any one of them.
    let solved: Vec<(Vec<f64>, Matrix)> = partitioning
        .subsystems
        .par_iter()
        .map(|subsystem| {
            let n = subsystem.n_orbitals();
            let mut local = Matrix::zeros(n, n);
            for (i, &mu) in subsystem.orbitals.iter().enumerate() {
                for (j, &nu) in subsystem.orbitals.iter().enumerate() {
                    // Every orbital pair here is inside this subsystem, so it is inside the
                    // pattern; the sparse read is exact rather than truncating.
                    local[(i, j)] = fock.get(pattern, mu, nu);
                }
            }
            symmetric_eigen(&local)
        })
        .collect::<Result<Vec<_>>>()?;

    let electrons_at = |mu: f64| -> f64 {
        let mut total = 0.0;
        for (subsystem, (energies, vectors)) in partitioning.subsystems.iter().zip(&solved) {
            let n = subsystem.n_orbitals();
            for (index, energy) in energies.iter().enumerate() {
                let f = fermi_dirac(*energy, mu, dc.smearing_ev);
                if f <= 0.0 {
                    continue;
                }
                // Only the diagonal is needed for the electron count, and only where the weight
                // is one — a `½`-weighted diagonal element cannot occur, since a diagonal element
                // has both indices on the same orbital.
                for i in 0..n {
                    if subsystem.in_core[i] {
                        let c = vectors[(i, index)];
                        total += f * c * c;
                    }
                }
            }
        }
        total
    };

    let (mut low, mut high) = bracket(&solved, dc, target, &electrons_at);
    for _ in 0..200 {
        let mid = 0.5 * (low + high);
        if electrons_at(mid) > target {
            high = mid;
        } else {
            low = mid;
        }
        if (high - low).abs() < dc.electron_tol {
            break;
        }
    }
    let mu = 0.5 * (low + high);

    density.fill(0.0);
    for (subsystem, (energies, vectors)) in partitioning.subsystems.iter().zip(&solved) {
        let n = subsystem.n_orbitals();
        for (index, energy) in energies.iter().enumerate() {
            let f = fermi_dirac(*energy, mu, dc.smearing_ev);
            if f <= 1.0e-14 {
                continue;
            }
            for i in 0..n {
                let ci = vectors[(i, index)];
                if ci == 0.0 {
                    continue;
                }
                for j in 0..n {
                    let w = weight(subsystem.in_core[i], subsystem.in_core[j]);
                    if w == 0.0 {
                        continue;
                    }
                    density.add(
                        pattern,
                        subsystem.orbitals[i],
                        subsystem.orbitals[j],
                        w * f * ci * vectors[(j, index)],
                    );
                }
            }
        }
    }
    Ok(mu)
}

/// A bracket for the chemical potential that is guaranteed to contain the answer.
fn bracket<F: Fn(f64) -> f64>(
    solved: &[(Vec<f64>, Matrix)],
    dc: &DcOptions,
    target: f64,
    electrons_at: &F,
) -> (f64, f64) {
    let lowest = solved
        .iter()
        .flat_map(|(energies, _)| energies.iter())
        .fold(f64::INFINITY, |a, b| a.min(*b));
    let highest = solved
        .iter()
        .flat_map(|(energies, _)| energies.iter())
        .fold(f64::NEG_INFINITY, |a, b| a.max(*b));
    let pad = 40.0 * dc.smearing_ev.max(1.0);
    let mut low = lowest - pad;
    let mut high = highest + pad;
    // Widen if the bracket somehow fails to straddle the target; cheap insurance against an
    // unusual spectrum.
    let mut widening = 0;
    while electrons_at(low) > target && widening < 40 {
        low -= pad;
        widening += 1;
    }
    widening = 0;
    while electrons_at(high) < target && widening < 40 {
        high += pad;
        widening += 1;
    }
    (low, high)
}

#[inline]
fn fermi_dirac(energy: f64, mu: f64, smearing: f64) -> f64 {
    if smearing <= 0.0 {
        return if energy <= mu { 1.0 } else { 0.0 };
    }
    let x = (energy - mu) / smearing;
    if x > 40.0 {
        0.0
    } else if x < -40.0 {
        1.0
    } else {
        1.0 / (1.0 + x.exp())
    }
}

/// One spin channel's share of the superposition of neutral atomic densities.
fn scaled(atomic: &Matrix, fraction: f64) -> Matrix {
    let mut out = atomic.clone();
    for value in out.as_mut_slice() {
        *value *= fraction;
    }
    out
}

fn mulliken(
    molecule: &Molecule,
    params: &Pm3Parameters,
    basis: &Basis,
    pattern: &DensityPattern,
    density: &SparseMatrix,
) -> Result<Vec<f64>> {
    let mut charges = vec![0.0; molecule.atoms.len()];
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
        // The diagonal of an atom's own block, which every subsystem holding that atom holds.
        let population: f64 = (0..n)
            .map(|mu| density.get(pattern, off + mu, off + mu))
            .sum();
        charges[ia] = params.element(atom.z)?.core_charge - population;
    }
    Ok(charges)
}

fn isolated_sums(molecule: &Molecule, params: &Pm3Parameters) -> Result<(f64, f64)> {
    let mut e_isol = 0.0;
    let mut eheat = 0.0;
    for atom in &molecule.atoms {
        let e = params.element(atom.z)?;
        e_isol += e.e_isol;
        eheat += e.eheat_ev;
    }
    Ok((e_isol, eheat))
}

/// `α − β` on the pattern, handed back dense because that is what the public result holds.
fn subtract(pattern: &DensityPattern, a: &SparseMatrix, b: &SparseMatrix) -> Matrix {
    let mut out = a.clone();
    for (slot, value) in out.values_mut().iter_mut().zip(b.values()) {
        *slot -= value;
    }
    pattern.to_dense(&out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Cell;
    use crate::corrections::Variant;

    /// A chain of `n` water molecules, far enough apart to be nearly independent.
    fn water_chain(n: usize) -> Molecule {
        let mut lines = format!("{}\nchain\n", 3 * n);
        for i in 0..n {
            let x = 3.2 * i as f64;
            lines.push_str(&format!("O {:.4} 0.0 0.0\n", x));
            lines.push_str(&format!("H {:.4} 0.0 0.0\n", x + 0.9584));
            lines.push_str(&format!("H {:.4} 0.9278 0.0\n", x - 0.24));
        }
        Molecule::from_xyz_str(&lines, 0.0).unwrap()
    }

    fn options() -> Pm3Options {
        Pm3Options {
            max_scf: 400,
            ..Pm3Options::default()
        }
    }

    /// The linear-scaling path is the same calculation with the long range handed to a
    /// point-charge model, so pushing the handover past the whole molecule must give the dense
    /// path back exactly.
    ///
    /// That is the check that separates "the split is reversible" from "the truncation happens
    /// to be small here", and it is the one that would catch a near-field table built for the
    /// wrong pairs, a double-counted point term, or a switch applied in the wrong direction.
    #[test]
    fn the_linear_path_reproduces_the_dense_one_when_nothing_is_truncated() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = water_chain(4);
        let dense = DcOptions {
            core_radius: 4.0,
            buffer_radius: 9.0,
            ..DcOptions::default()
        };
        let untruncated = DcOptions {
            long_range_cutoff: Some(400.0),
            ..dense
        };

        // Converged well past the default: the two build their Fock matrices with different
        // code, so at the default tolerance they stop at different points *within* it and the
        // residual would be the stopping rule rather than the physics.
        let tight = Pm3Options {
            e_tol: 1.0e-12,
            p_tol: 1.0e-11,
            ..options()
        };
        let a = run_dc(&molecule, &params, &tight, &dense).unwrap();
        let b = run_dc(&molecule, &params, &tight, &untruncated).unwrap();
        let difference = (a.total_ev - b.total_ev).abs();
        assert!(
            difference < 1.0e-7,
            "untruncated linear {} vs dense {} ({difference:.3e} eV)",
            b.total_ev,
            a.total_ev
        );
        assert_eq!(a.n_subsystems, b.n_subsystems, "same partitioning");
        for (x, y) in a.charges.iter().zip(&b.charges) {
            assert!((x - y).abs() < 1.0e-8, "charge {x} vs {y}");
        }
    }

    /// And with the default handover, what it costs is the switch — an error that stays put as
    /// the system grows rather than accumulating with it.
    ///
    /// An extensive error would be fatal: it would mean the linear path is only usable on the
    /// small systems that do not need it. Asserting the *per atom* figure at two sizes is what
    /// distinguishes the two cases; a single tolerance could not.
    ///
    /// Both sizes have to be long compared with the handover radius. A chain shorter than that
    /// has almost no pairs beyond it and so almost no switch error, and comparing such a chain
    /// with a long one measures how many pairs crossed the cutoff, not whether the error
    /// accumulates. At 3.2 A spacing, 12 waters span 73 Bohr and 36 span 220, against a 22 Bohr
    /// handover.
    #[test]
    fn the_switch_costs_the_linear_path_a_fixed_amount_per_atom() {
        let params = Pm3Parameters::standard().unwrap();
        let dense = DcOptions {
            core_radius: 4.0,
            buffer_radius: 9.0,
            ..DcOptions::default()
        };
        let linear = DcOptions {
            long_range_cutoff: Some(22.0),
            ..dense
        };

        let per_atom = |n: usize| {
            let molecule = water_chain(n);
            let a = run_dc(&molecule, &params, &options(), &dense).unwrap();
            let b = run_dc(&molecule, &params, &options(), &linear).unwrap();
            (a.total_ev - b.total_ev).abs() / molecule.atoms.len() as f64
        };

        let small = per_atom(12);
        let large = per_atom(36);
        assert!(
            large < 1.0e-4,
            "the switch cost {:.1} ueV/atom, which is too much to call a tail",
            1.0e6 * large
        );
        assert!(
            large < 2.0 * small.max(1.0e-9),
            "tripling the chain took the per-atom error from {:.1} to {:.1} ueV, \
             so it is accumulating with system size rather than saturating",
            1.0e6 * small,
            1.0e6 * large
        );
    }

    /// **The divide-and-conquer test.** Widening the buffer must converge the energy onto the full
    /// diagonalization.
    ///
    /// Nothing else establishes that the method is right rather than merely self-consistent: the
    /// partitioning, the Yang–Lee weights, the global chemical potential and the reassembly are
    /// all exercised at once, against a reference that shares none of them.
    #[test]
    fn widening_the_buffer_converges_onto_the_full_diagonalization() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = water_chain(6);
        let reference = crate::scf::run_pm3(&molecule, &params, &options())
            .unwrap()
            .total_ev;

        let errors: Vec<f64> = [3.0_f64, 6.0, 10.0, 16.0]
            .iter()
            .map(|buffer| {
                let dc = DcOptions {
                    core_radius: 3.0,
                    buffer_radius: *buffer,
                    ..DcOptions::default()
                };
                let result = run_dc(&molecule, &params, &options(), &dc).unwrap();
                (result.total_ev - reference).abs()
            })
            .collect();

        // Convergence is asserted end to end rather than step by step. The error is *not*
        // guaranteed monotone in the buffer radius: widening it changes which atoms fall in which
        // subsystem, and a term that happened to cancel at one radius need not cancel at the next.
        // What must hold is that the trend is downward and that the widest buffer essentially
        // reproduces the reference.
        assert!(
            errors[3] < errors[0],
            "widening the buffer did not help: {errors:?}"
        );
        assert!(
            errors[3] < 5.0e-4,
            "the widest buffer should essentially reproduce the reference, not miss by {:.3e}",
            errors[3]
        );
    }

    /// A buffer that reaches the whole system makes every subsystem the whole system, so the
    /// result has to be the full diagonalization exactly — not approximately.
    ///
    /// This separates the *partitioning* from the *truncation*: if this fails, the reassembly or
    /// the chemical potential is wrong, not the buffer radius.
    #[test]
    fn a_reaching_buffer_reproduces_the_full_result_exactly() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = water_chain(3);
        let reference = crate::scf::run_pm3(&molecule, &params, &options()).unwrap();
        let dc = DcOptions {
            core_radius: 2.0,
            buffer_radius: 500.0,
            // With one subsystem covering everything, a step occupation is exact and avoids the
            // broadening's own (small) effect on the energy.
            smearing_ev: 1.0e-4,
            ..DcOptions::default()
        };
        let result = run_dc(&molecule, &params, &options(), &dc).unwrap();
        assert_eq!(result.dropped_pairs, 0);
        let difference = (result.total_ev - reference.total_ev).abs();
        assert!(
            difference < 1.0e-6,
            "full-coverage DC {} vs full diagonalization {} ({difference:.3e} eV)",
            result.total_ev,
            reference.total_ev
        );
        for (a, b) in result.charges.iter().zip(&reference.charges) {
            assert!((a - b).abs() < 1.0e-5, "charge {a} vs {b}");
        }
    }

    /// The electron count is the constraint the chemical potential enforces, so it must come out
    /// exactly — whatever the partitioning, and whether or not the system is charged.
    #[test]
    fn the_electron_count_is_exact() {
        let params = Pm3Parameters::standard().unwrap();
        for charge in [0.0, 1.0, -1.0] {
            let mut molecule = water_chain(4);
            molecule.charge = charge;
            if charge != 0.0 {
                molecule.multiplicity = 2;
            }
            let dc = DcOptions {
                core_radius: 3.0,
                buffer_radius: 8.0,
                ..DcOptions::default()
            };
            let result = run_dc(&molecule, &params, &options(), &dc).unwrap();
            let trace: f64 = (0..result.density.rows)
                .map(|i| result.density[(i, i)])
                .sum();
            let expected = 32.0 - charge;
            assert!(
                (trace - expected).abs() < 1.0e-6,
                "charge {charge}: {trace} electrons, expected {expected}"
            );
            let total: f64 = result.charges.iter().sum();
            assert!(
                (total - charge).abs() < 1.0e-6,
                "charge {charge}: Mulliken charges sum to {total}"
            );
        }
    }

    /// How the system is cut up must not change the answer beyond the truncation. Two different
    /// core radii with a generous buffer have to agree.
    #[test]
    fn the_result_does_not_depend_on_where_the_cuts_fall() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = water_chain(5);
        let energy_at = |core_radius: f64| {
            run_dc(
                &molecule,
                &params,
                &options(),
                &DcOptions {
                    core_radius,
                    buffer_radius: 14.0,
                    ..DcOptions::default()
                },
            )
            .unwrap()
            .total_ev
        };
        let coarse = energy_at(6.0);
        let fine = energy_at(2.5);
        assert!(
            (coarse - fine).abs() < 1.0e-4,
            "partitioning changed the answer: {coarse} vs {fine}"
        );
    }

    /// An open-shell system goes through the unrestricted path and polarizes.
    #[test]
    fn an_open_shell_system_converges_with_the_right_moment() {
        const METHYL: &str =
            "4\nmethyl\nC 0.0 0.0 0.0\nH 0.0 1.078 0.0\nH 0.9336 -0.539 0.0\nH -0.9336 -0.539 0.0\n";
        let params = Pm3Parameters::standard().unwrap();
        let mut molecule = Molecule::from_xyz_str(METHYL, 0.0).unwrap();
        molecule.multiplicity = 2;
        let dc = DcOptions {
            core_radius: 4.0,
            buffer_radius: 500.0,
            smearing_ev: 1.0e-4,
            ..DcOptions::default()
        };
        let result = run_dc(&molecule, &params, &options(), &dc).unwrap();
        assert!(result.unrestricted);
        let spin = result.spin_density.expect("UHF reports a spin density");
        let moment: f64 = (0..spin.rows).map(|i| spin[(i, i)]).sum();
        assert!(
            (moment - 1.0).abs() < 1.0e-5,
            "the unpaired electron should give a moment of 1, not {moment}"
        );
    }

    /// The corrections are classical and global — they do not depend on the partitioning at all.
    #[test]
    fn the_corrections_are_independent_of_the_partitioning() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = water_chain(4);
        let options = Pm3Options {
            variant: Variant::Pm3D3H4X,
            ..options()
        };
        let correction_at = |core_radius: f64| {
            let result = run_dc(
                &molecule,
                &params,
                &options,
                &DcOptions {
                    core_radius,
                    buffer_radius: 10.0,
                    ..DcOptions::default()
                },
            )
            .unwrap();
            result.core_ev - crate::repulsion::core_core_energy(&molecule, &params).unwrap()
        };
        let a = correction_at(3.0);
        let b = correction_at(7.0);
        assert!(a.abs() > 1.0e-6, "the test needs a live correction");
        assert!(
            (a - b).abs() < 1.0e-12,
            "the correction moved with the partitioning: {a} vs {b}"
        );
    }

    /// The periodic path, against the full periodic diagonalization.
    #[test]
    fn the_periodic_path_converges_onto_the_full_periodic_result() {
        let params = Pm3Parameters::standard().unwrap();
        let mut molecule = water_chain(4);
        molecule.cell = Some(Cell::cubic(30.0).unwrap());
        let periodic = PeriodicOptions::default();
        let reference =
            crate::pbc::gamma::run_gamma(&molecule, &params, &options(), &periodic).unwrap();

        let dc = DcOptions {
            core_radius: 3.0,
            buffer_radius: 500.0,
            smearing_ev: 1.0e-4,
            ..DcOptions::default()
        };
        let result = run_dc_gamma(&molecule, &params, &options(), &periodic, &dc).unwrap();
        assert_eq!(result.dropped_pairs, 0);
        let difference = (result.total_ev - reference.total_ev).abs();
        assert!(
            difference < 1.0e-6,
            "periodic DC {} vs full periodic {} ({difference:.3e} eV)",
            result.total_ev,
            reference.total_ev
        );
        // And the condition that governs the Γ point is reported unchanged: DC does not relax it.
        assert!(result.gamma_margin > 0.0);
    }

    /// A truncated periodic run still converges with the buffer.
    #[test]
    fn the_periodic_path_converges_with_the_buffer() {
        let params = Pm3Parameters::standard().unwrap();
        let mut molecule = water_chain(5);
        molecule.cell = Some(Cell::cubic(34.0).unwrap());
        let periodic = PeriodicOptions::default();
        let reference =
            crate::pbc::gamma::run_gamma(&molecule, &params, &options(), &periodic).unwrap();
        let errors: Vec<f64> = [4.0_f64, 8.0, 14.0]
            .iter()
            .map(|buffer| {
                let dc = DcOptions {
                    core_radius: 3.0,
                    buffer_radius: *buffer,
                    ..DcOptions::default()
                };
                let result = run_dc_gamma(&molecule, &params, &options(), &periodic, &dc).unwrap();
                (result.total_ev - reference.total_ev).abs()
            })
            .collect();
        assert!(
            errors[2] < errors[0],
            "widening the buffer did not help: {errors:?}"
        );
        assert!(
            errors[2] < 1.0e-3,
            "the widest buffer still misses by {:.3e}",
            errors[2]
        );
    }
}
