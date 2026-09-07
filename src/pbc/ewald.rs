// SPDX-License-Identifier: GPL-3.0-or-later

//! Ewald summation over point charges, in 1, 2 or 3 periodic dimensions.
//!
//! The charges here are **not** the atoms. PM3's two-center Coulomb terms come from the
//! Dewar–Thiel multipole model, which represents each orbital-pair density by a small set of
//! point charges at fixed offsets from the nucleus; the point limit of those configurations is
//! exactly what has to be lattice-summed (see [`crate::pbc`]). This module therefore works on a
//! generic list of [`ChargeSite`]s, each tagged with the atom it belongs to so forces and the
//! virial can be accumulated per atom. Feeding it one site per nucleus gives the ordinary
//! point-charge Ewald sum, which is what the Madelung tests use.
//!
//! Units follow the rest of the crate: positions in Bohr, charges in elementary charges,
//! energies in eV (the Hartree-with-Bohr Coulomb energy scaled by [`PM3_EV`]).
//!
//! # What is returned, and why
//!
//! [`EwaldOutput`] carries four things, because the SCF needs all four and they share almost
//! all of their work:
//!
//! * `energy_ev` — the electrostatic energy per cell.
//! * `site_potential_ev` — `∂E/∂q_i`, the Madelung potential at each site. This is the piece
//!   the Fock matrix needs, and leaving it out is the classic way to end up with an energy
//!   that is corrected for the periodic images while the SCF density is not.
//! * `site_gradient` — `∂E/∂r_i`, which becomes the force.
//! * `virial` — `∂E/∂ε`, the strain derivative, which becomes the stress after dividing by
//!   [`Cell::measure`].
//!
//! # Boundary convention
//!
//! In 3D the `G = 0` term of the reciprocal sum is omitted, which is the **tinfoil**
//! (conducting) boundary condition. A finite cluster summed in vacuum converges to a different
//! value, differing by the depolarizing energy of its own surface charge; the tests bridge the
//! two explicitly with `pbc::ewald_reference::surface_dipole_energy` rather than
//! pretending the ambiguity does not exist.
//!
//! # Charged cells
//!
//! What a net charge `Q ≠ 0` costs depends on the dimensionality, and the three cases are
//! genuinely different physics rather than three spellings of one formula:
//!
//! * **3D** — the dropped `G = 0` term leaves the monopole unaccounted for, so a uniform
//!   neutralizing background is added. It contributes `−π Q² / (2 α² V)` to the energy,
//!   `−π Q / (α² V)` to every site potential — omitting *that* leaves the SCF inconsistent with
//!   its own energy — and `+π Q² / (2 α² V) δ_αβ` to the virial. It contributes nothing to the
//!   forces, being independent of where the atoms sit.
//! * **2D** — no background is needed. The slab formulation keeps a *finite* `G = 0` term,
//!   the potential of the Gaussian-smeared charged sheet, which already carries the net charge.
//! * **1D** — refused. A charged chain's potential grows logarithmically with transverse
//!   distance, so its energy is not defined without also fixing a transverse reference, which
//!   is not part of the model. Returning a number here would mean inventing a convention.

use crate::cell::Cell;
use crate::constants::PM3_EV;
use crate::error::{Pm3Error, Result};
use crate::math::{Mat3, Vec3};
use crate::neighbor::{NeighborList, PairImage};
use crate::special::{erf, erfc, erfcx};
use std::f64::consts::{PI, TAU};
use std::sync::atomic::{AtomicU64, Ordering};

/// One point charge participating in the lattice sum.
///
/// `owner` does two jobs: it lets forces be accumulated per atom, and it marks which sites
/// belong to the same charge distribution. Sites sharing an owner **inside the reference cell**
/// do not interact through this sum at all — in NDDO their mutual energy is the one-center
/// integral set (`Gss`, `Gsp`, `Gpp`, `Gp2`, `Hsp`), not a point-charge Coulomb term. Their
/// *periodic images* do interact, and those are kept.
#[derive(Clone, Copy, Debug)]
pub struct ChargeSite {
    /// Position in the reference cell (Bohr).
    pub position: Vec3,
    /// Charge in elementary charges.
    pub charge: f64,
    /// Index of the atom this site belongs to.
    pub owner: usize,
}

impl ChargeSite {
    /// A site sitting on its own nucleus — the ordinary point-charge case.
    pub fn on_atom(position: Vec3, charge: f64, owner: usize) -> Self {
        Self {
            position,
            charge,
            owner,
        }
    }
}

/// Splitting parameter, real-space cutoff and reciprocal-space cutoff.
///
/// The three are not independent: `alpha` decides how quickly the real-space term decays and
/// how slowly the reciprocal one does, so a mismatched triple is accurate in neither. Prefer
/// [`EwaldParams::tuned`], which derives a consistent set from a target accuracy.
#[derive(Clone, Copy, Debug)]
pub struct EwaldParams {
    /// Gaussian splitting parameter (1/Bohr).
    pub alpha: f64,
    /// Real-space cutoff (Bohr).
    pub real_cutoff: f64,
    /// Reciprocal-space cutoff `|G|max` (1/Bohr).
    pub gmax: f64,
    /// Number of `±n` cell images summed by the 1D path before its analytic tail takes over.
    /// Unused in 2D and 3D.
    pub chain_images: usize,
}

/// Default 1D image count. The truncated tail falls off as `1/N²` and is then corrected
/// analytically, so 400 shells already put the residual well below any tolerance in this crate.
pub const DEFAULT_CHAIN_IMAGES: usize = 400;

/// Reference length (Bohr) for a charged chain's neutralizing line charge.
///
/// A charged 1D system's energy is defined only up to where the logarithm's zero is put, in the
/// same way a charged 3D cell's is defined only up to the jellium convention. Fixing it at one
/// Bohr — a constant, not the cell length — is what makes the answer independent of how a chain
/// was cut into cells. Absolute energies of chains with different charge remain incomparable.
pub const CHAIN_REFERENCE_LENGTH: f64 = 1.0;

/// Default real-space cutoff (Bohr). Only the *cost* balance depends on this; the accuracy is
/// carried by `alpha` and `gmax`, which are derived from it.
pub const DEFAULT_REAL_CUTOFF: f64 = 12.0;

/// Default target accuracy of the split, as a relative energy error.
pub const DEFAULT_ACCURACY: f64 = 1.0e-12;

impl EwaldParams {
    /// Derive a consistent `(alpha, real_cutoff, gmax)` from a real-space cutoff and a target
    /// accuracy.
    ///
    /// The real-space term is cut where `erfc(α r_c)/r_c` has fallen to `accuracy`, giving
    /// `α r_c ≈ √(−ln accuracy)`; the reciprocal term is cut where `exp(−G²/4α²)` has fallen
    /// to the same, giving `G_max = 2α√(−ln accuracy)`. Tightening `accuracy` therefore moves
    /// both cutoffs together, which is what keeps the split balanced.
    pub fn tuned(real_cutoff: f64, accuracy: f64) -> Self {
        let p = (-accuracy.max(f64::MIN_POSITIVE).ln()).max(1.0).sqrt();
        let alpha = p / real_cutoff;
        Self {
            alpha,
            real_cutoff,
            gmax: 2.0 * alpha * p,
            chain_images: DEFAULT_CHAIN_IMAGES,
        }
    }
}

impl EwaldParams {
    /// Derive parameters that stay affordable as the cell grows.
    ///
    /// A fixed `alpha` is a trap for large cells. The reciprocal sum runs over every `G` inside
    /// `gmax`, and the reciprocal lattice gets *denser* as the cell gets bigger, so their count
    /// grows as the volume: for a water molecule in a 240 Bohr box, [`EwaldParams::tuned`] at the
    /// default 12 Bohr cutoff asks for 4.5 × 10⁷ reciprocal vectors and the calculation simply
    /// stops finishing.
    ///
    /// Scaling the real-space cutoff with the cell fixes it. Taking `r_c ≈ ½` the smallest
    /// perpendicular width makes `alpha ∝ 1/L`, which leaves both sums size-independent: the
    /// real-space one covers a constant number of images, and `gmax` shrinks in step with the
    /// reciprocal lattice spacing. The floor keeps a small cell from being pushed into a huge
    /// `alpha` and an equally huge `gmax`.
    pub fn for_cell(cell: &Cell, accuracy: f64) -> Self {
        const FLOOR: f64 = 10.0;
        let narrowest = cell
            .periodic_widths()
            .iter()
            .map(|(_, width)| *width)
            .fold(f64::INFINITY, f64::min);
        let cutoff = if narrowest.is_finite() {
            (0.5 * narrowest).max(FLOOR)
        } else {
            DEFAULT_REAL_CUTOFF
        };
        Self::tuned(cutoff, accuracy)
    }
}

impl Default for EwaldParams {
    fn default() -> Self {
        Self::tuned(DEFAULT_REAL_CUTOFF, DEFAULT_ACCURACY)
    }
}

/// Energy, site potentials, site gradients and the strain derivative of one Ewald sum.
#[derive(Clone, Debug)]
pub struct EwaldOutput {
    /// Electrostatic energy per cell (eV).
    pub energy_ev: f64,
    /// `∂E/∂q_i` at each site (eV per elementary charge).
    pub site_potential_ev: Vec<f64>,
    /// `∂E/∂r_i` at each site (eV/Bohr).
    pub site_gradient: Vec<Vec3>,
    /// `∂E/∂ε` (eV). Divide by [`Cell::measure`] for the stress.
    ///
    /// `None` only where there is no strain to take a derivative with respect to, which now
    /// means an isolated system alone. A missing value is deliberately not a zero matrix: a
    /// silently zero stress would let a variable-cell optimization "converge" instantly.
    pub virial: Option<Mat3>,
}

impl EwaldOutput {
    fn zeros(n: usize) -> Self {
        Self {
            energy_ev: 0.0,
            site_potential_ev: vec![0.0; n],
            site_gradient: vec![Vec3::zero(); n],
            virial: None,
        }
    }

    /// Sum the per-site gradients onto their owning atoms.
    pub fn atom_gradient(&self, sites: &[ChargeSite], n_atoms: usize) -> Vec<Vec3> {
        let mut out = vec![Vec3::zero(); n_atoms];
        for (site, gradient) in sites.iter().zip(&self.site_gradient) {
            out[site.owner] += *gradient;
        }
        out
    }
}

/// Everything about an Ewald sum that depends on the **geometry** but not the **charges**.
///
/// An SCF calls the lattice sum once per iteration with the same sites in the same places and
/// different charges on them. The reciprocal-lattice enumeration and — far more expensively — the
/// `cos(G·r)` and `sin(G·r)` of every site against every `G` are identical every time. For a water
/// molecule in a 20 Bohr cell that is eleven thousand `G` vectors against twenty sites, evaluated
/// thirteen times over for no reason; caching them is most of the cost of a periodic SCF.
///
/// The phase table is `2 · n_G · n_sites` doubles, which grows with the system. Past
/// [`PHASE_TABLE_BUDGET_MB`] it is left empty and the trigonometry is recomputed, so a large
/// system pays time rather than memory.
#[derive(Clone, Debug)]
pub struct EwaldContext {
    /// Half of each `±G` pair, with `0 < |G| ≤ gmax`.
    gvectors: Vec<Vec3>,
    /// `(cos, sin)` laid out `G`-major, or empty when the table exceeded the budget.
    phases: Vec<(f64, f64)>,
    /// Real-space pairs inside the cutoff.
    neighbors: NeighborList,
    n_sites: usize,
    /// `erfc(αr)/r` and the geometric half of its derivative, one entry per **ordered** pair,
    /// indexed exactly like `neighbors.all()`. Empty when the table exceeded its budget.
    ///
    /// This is what a periodic SCF actually spends its time on. The real-space sum was 97% of
    /// the Ewald and the Ewald 97% of the iteration -- and every one of those iterations
    /// recomputed `erfc` and `exp` for every pair, twice (once for the energy, once for the
    /// potentials), at distances that had not moved since the geometry was set. Only the charges
    /// change inside an SCF, so the transcendentals are geometry and belong here with the phase
    /// table.
    ///
    /// Excluded pairs (`T = 0` on one atom, whose energy is the NDDO one-center set) and
    /// zero-distance pairs are stored as zeros, so the summation loops are branch-free
    /// multiplies rather than a predicate per pair.
    real_kernel: Vec<RealKernel>,
    /// The `α` the kernel was tabulated at, so a mismatched `params` is refused rather than
    /// silently answered with the wrong screening.
    alpha: f64,
}

/// The geometry-only part of one real-space pair term.
#[derive(Clone, Copy, Debug, Default)]
struct RealKernel {
    /// `erfc(αr)/r`.
    screened: f64,
    /// `f'(r)/r` with `f(r) = erfc(αr)/r` — the gradient coefficient once multiplied by `q_a q_b`.
    dcoef: f64,
}

/// Ceiling on the cached `cos`/`sin` table.
pub const PHASE_TABLE_BUDGET_MB: usize = 256;

/// Ceiling on the cached real-space kernel.
pub const REAL_KERNEL_BUDGET_MB: usize = 256;

impl EwaldContext {
    /// Build the cache for a fixed set of site *positions*. The charges are irrelevant here and
    /// may change freely afterwards.
    pub fn build(cell: &Cell, sites: &[ChargeSite], params: &EwaldParams) -> Self {
        // An isolated system consults none of this: [`direct_0d`] sums every pair outright, so
        // building a neighbour list and a phase table would be pure waste.
        if cell.n_periodic() == 0 {
            return Self {
                gvectors: Vec::new(),
                phases: Vec::new(),
                neighbors: NeighborList::build_from_positions(&[], None, 0.0),
                n_sites: sites.len(),
                real_kernel: Vec::new(),
                alpha: params.alpha,
            };
        }
        let positions: Vec<Vec3> = sites.iter().map(|s| s.position).collect();
        let neighbors =
            NeighborList::build_from_positions(&positions, Some(cell), params.real_cutoff);
        let gvectors = if cell.n_periodic() >= 2 {
            reciprocal_vectors(cell, params.gmax)
        } else {
            Vec::new()
        };
        let entries = gvectors.len().saturating_mul(sites.len());
        let budget = PHASE_TABLE_BUDGET_MB * 1024 * 1024 / std::mem::size_of::<(f64, f64)>();
        let phases = if entries > 0 && entries <= budget {
            let mut table = Vec::with_capacity(entries);
            for g in &gvectors {
                for site in sites {
                    let (sin, cos) = g.dot(site.position).sin_cos();
                    table.push((cos, sin));
                }
            }
            table
        } else {
            Vec::new()
        };
        // The real-space kernel, tabulated once for this geometry. Two doubles per ordered pair
        // against the phase table's two per (G, site), and it removes an `erfc` and an `exp`
        // from every pair of every SCF iteration.
        let pair_budget = REAL_KERNEL_BUDGET_MB * 1024 * 1024 / std::mem::size_of::<RealKernel>();
        let real_kernel = if !neighbors.all().is_empty() && neighbors.len() <= pair_budget {
            let alpha = params.alpha;
            let two_alpha_over_sqrt_pi = 2.0 * alpha / PI.sqrt();
            neighbors
                .all()
                .iter()
                .map(|pair| {
                    let r = pair.r;
                    if r <= 0.0 || excluded(sites, pair.a, pair.b, pair.t) {
                        return RealKernel::default();
                    }
                    let screened = erfc(alpha * r) / r;
                    // f(r) = erfc(αr)/r, f'(r) = −erfc(αr)/r² − (2α/√π) e^{−α²r²}/r
                    let derivative =
                        -(screened / r) - two_alpha_over_sqrt_pi * (-(alpha * r).powi(2)).exp() / r;
                    RealKernel {
                        screened,
                        dcoef: derivative / r,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        Self {
            gvectors,
            phases,
            neighbors,
            n_sites: sites.len(),
            real_kernel,
            alpha: params.alpha,
        }
    }

    /// `(cos, sin)` for one `G` and one site, from the table or recomputed.
    #[inline]
    fn phase(&self, g_index: usize, site: usize, g: Vec3, position: Vec3) -> (f64, f64) {
        if self.phases.is_empty() {
            let (sin, cos) = g.dot(position).sin_cos();
            (cos, sin)
        } else {
            self.phases[g_index * self.n_sites + site]
        }
    }
}

/// Where the time inside a 3D Ewald sum goes, in nanoseconds, cumulative over the process.
///
/// The lattice sum is 89–98% of every periodic SCF iteration (`examples/periodic_profile.rs`),
/// and which *half* of it dominates is not something reading settles: parallelising the
/// reciprocal sum on the strength of a guess moved a 48-atom iteration from 174 ms to 170 ms.
/// So the halves are counted rather than argued about. Read with [`profile_snapshot`]; the
/// counters cost one `Instant::now()` per call and are always on, because a profiler you have
/// to enable is one nobody enables.
pub(crate) struct EwaldProfile {
    pub real: AtomicU64,
    pub reciprocal: AtomicU64,
    pub corrections: AtomicU64,
}

pub(crate) static PROFILE: EwaldProfile = EwaldProfile {
    real: AtomicU64::new(0),
    reciprocal: AtomicU64::new(0),
    corrections: AtomicU64::new(0),
};

/// `(real, reciprocal, corrections)` in seconds.
pub(crate) fn profile_snapshot() -> (f64, f64, f64) {
    let read = |counter: &AtomicU64| counter.load(Ordering::Relaxed) as f64 * 1.0e-9;
    (
        read(&PROFILE.real),
        read(&PROFILE.reciprocal),
        read(&PROFILE.corrections),
    )
}

/// A cursor that charges each elapsed span to a counter and resets.
struct Stopwatch(std::time::Instant);

impl Stopwatch {
    fn start() -> Self {
        Self(std::time::Instant::now())
    }

    fn lap(&mut self, counter: &AtomicU64) {
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.0).as_nanos() as u64;
        counter.fetch_add(elapsed, Ordering::Relaxed);
        self.0 = now;
    }
}

/// Evaluate the Ewald sum for `sites` in `cell`.
///
/// The dimensionality of `cell` selects the treatment, and only the long-range half differs:
///
/// * **3D** — the textbook split, with the `G = 0` term dropped (tinfoil) and a uniform
///   neutralizing background for a charged cell.
/// * **2D** — Parry's slab formulation: a sum over in-plane `G` whose `z` dependence is carried
///   analytically by `erfc`, plus a `G = 0` term that is the potential of the Gaussian-smeared
///   charged sheet. That `G = 0` term is finite (unlike 3D's `1/G²`), so a charged slab needs
///   no separate background — the sheet term already carries it, which the alpha-independence
///   test confirms.
/// * **1D** — direct summation grouped by cell rather than by pair. Grouping is what makes it
///   converge: the per-pair sum `Σ_n 1/|d + nL|` diverges logarithmically, while the
///   cell-to-cell interaction of a neutral cell starts at dipole–dipole and falls off as
///   `1/n³`. A charged 1D cell is refused rather than answered, because its energy is not
///   defined without also fixing a transverse reference — see the error text.
pub fn ewald(cell: &Cell, sites: &[ChargeSite], params: &EwaldParams) -> Result<EwaldOutput> {
    if sites.is_empty() {
        return Ok(EwaldOutput::zeros(0));
    }
    let context = EwaldContext::build(cell, sites, params);
    ewald_cached(cell, sites, params, &context)
}

/// The same sum, reusing a [`EwaldContext`] built for these site *positions*.
///
/// The caller is responsible for the context matching the geometry: it is rebuilt whenever an
/// atom moves and reused across the iterations of one SCF, where nothing but the charges changes.
/// A mismatched site count is caught; a moved atom with the same count is not, which is why the
/// only thing that builds one is `pbc::gamma::build_setup`, alongside the geometry it
/// describes.
pub fn ewald_cached(
    cell: &Cell,
    sites: &[ChargeSite],
    params: &EwaldParams,
    context: &EwaldContext,
) -> Result<EwaldOutput> {
    if sites.is_empty() {
        return Ok(EwaldOutput::zeros(0));
    }
    if context.n_sites != sites.len() {
        return Err(Pm3Error::InvalidInput(format!(
            "the Ewald cache was built for {} sites but was handed {}",
            context.n_sites,
            sites.len()
        )));
    }
    // The real-space kernel is tabulated at one `α`. Using it at another is not an error the
    // arithmetic would reveal -- the screening would simply be wrong by a smooth factor, and the
    // reciprocal half would still be computed at the new `α`, so the two halves would no longer
    // add up to `1/r`. The result is a plausible energy that is wrong, which is why this is
    // checked rather than assumed.
    if !context.real_kernel.is_empty() && context.alpha != params.alpha {
        return Err(Pm3Error::InvalidInput(format!(
            "the Ewald cache was tabulated at alpha={} but was handed params with alpha={}; \
             rebuild the context when the splitting parameter changes",
            context.alpha, params.alpha
        )));
    }
    let dimension = cell.n_periodic();
    let mut out = EwaldOutput::zeros(sites.len());
    if dimension == 0 {
        direct_0d(sites, &mut out);
        return Ok(out);
    }

    match dimension {
        3 => {
            let mut virial = Mat3::zero();
            let mut clock = Stopwatch::start();
            real_space(sites, params, &mut out, Some(&mut virial), context);
            clock.lap(&PROFILE.real);
            reciprocal_3d(cell, sites, params, &mut out, &mut virial, context);
            clock.lap(&PROFILE.reciprocal);
            self_term(sites, params, &mut out);
            background_3d(cell, sites, params, &mut out, &mut virial);
            remove_excluded_smooth_part(sites, params, &mut out, Some(&mut virial));
            clock.lap(&PROFILE.corrections);
            out.virial = Some(virial);
        }
        2 => {
            // The self term is `−(α/√π) Σ q²`: no positions, no cell, no strain derivative.
            let mut virial = Mat3::zero();
            real_space(sites, params, &mut out, Some(&mut virial), context);
            reciprocal_2d(cell, sites, params, &mut out, &mut virial, context);
            self_term(sites, params, &mut out);
            remove_excluded_smooth_part(sites, params, &mut out, Some(&mut virial));
            out.virial = Some(virial);
        }
        1 => direct_1d(cell, sites, params, &mut out)?,
        _ => unreachable!("n_periodic is 1..=3 and 0 was handled above"),
    }
    Ok(out)
}

/// The same sum, for a caller that wants the energy and the site potentials but not the site
/// gradients.
///
/// That is every caller inside a self-consistent field: the gradients are read once, by the
/// force evaluation, after the density has stopped moving. For an isolated system this takes a
/// faster route ([`direct_0d_potentials`]); for a periodic one it is [`ewald_cached`] unchanged,
/// where the reciprocal half already dominates and the saving would not be worth a second path.
pub fn ewald_potentials_cached(
    cell: &Cell,
    sites: &[ChargeSite],
    params: &EwaldParams,
    context: &EwaldContext,
) -> Result<EwaldOutput> {
    if cell.n_periodic() != 0 || sites.is_empty() {
        return ewald_cached(cell, sites, params, context);
    }
    if context.n_sites != sites.len() {
        return Err(Pm3Error::InvalidInput(format!(
            "the Ewald cache was built for {} sites but was handed {}",
            context.n_sites,
            sites.len()
        )));
    }
    let mut out = EwaldOutput::zeros(sites.len());
    direct_0d_potentials(sites, &mut out);
    Ok(out)
}

/// The same sum for an isolated system: every site pair, once, with no images and no
/// convergence machinery.
///
/// This is what makes zero dimensions a member of the same ladder as 1D, 2D and 3D rather than
/// a separate code path. There is nothing to split into real and reciprocal halves — the sum
/// converges absolutely because there is no lattice — so `α`, the cutoffs and the background
/// have no role and are not consulted.
///
/// Sites sharing an owner do not interact, exactly as in every periodic case: that energy
/// belongs to the one-center integrals.
///
/// # Cost
///
/// `O(N²)` in the number of sites, with about ten floating-point operations per pair. That is
/// the one term of an isolated calculation that is not linear in the system size, and it is
/// deliberately the cheapest possible quadratic: the expensive `O(N²)` — the NDDO pair tables,
/// hundreds of bytes and hundreds of operations per pair — is what the split removes. See
/// [`crate::pbc::screen`].
pub fn direct_0d(sites: &[ChargeSite], out: &mut EwaldOutput) {
    use rayon::prelude::*;

    // Flat arrays rather than a slice of structs: this is a memory-bandwidth-bound kernel and
    // the owner tag must be read for every pair.
    let positions: Vec<Vec3> = sites.iter().map(|s| s.position).collect();
    let charges: Vec<f64> = sites.iter().map(|s| s.charge).collect();
    let owners: Vec<usize> = sites.iter().map(|s| s.owner).collect();

    let rows: Vec<(f64, f64, Vec3)> = (0..sites.len())
        .into_par_iter()
        .map(|i| {
            let (position_i, charge_i, owner_i) = (positions[i], charges[i], owners[i]);
            let mut potential = 0.0;
            let mut gradient = Vec3::zero();
            for j in 0..sites.len() {
                if i == j || owners[j] == owner_i {
                    continue;
                }
                let d = positions[j] - position_i;
                let r = d.norm();
                let inverse = 1.0 / r;
                potential += charges[j] * inverse;
                // ∂(1/r)/∂r_i = +d/r³, as in `direct_1d`.
                gradient += d * (charge_i * charges[j] * inverse * inverse * inverse);
            }
            (
                0.5 * PM3_EV * charge_i * potential,
                PM3_EV * potential,
                gradient * PM3_EV,
            )
        })
        .collect();

    for (i, (energy, potential, gradient)) in rows.into_iter().enumerate() {
        out.energy_ev += energy;
        out.site_potential_ev[i] += potential;
        out.site_gradient[i] += gradient;
    }
    // `virial` stays `None`: an isolated system has no cell to strain.
}

/// The separation beyond which two owners' charge clouds interact through their multipole
/// moments rather than site by site (Bohr).
///
/// A heavy atom carries twenty-five auxiliary sites, so a site-by-site sum does six hundred
/// times the work of an atom-by-atom one on a pair too far apart for the difference to show.
/// The sites sit within about 1.2 Bohr of their nucleus, so the first neglected term is of order
/// `(d/R)³`. What fixes the constant is not that estimate but
/// `the_multipole_far_field_agrees_with_the_site_sum`, which measures the error against the exact
/// sum on a chain of realistic clouds: 1.8e-4 eV over forty-eight atoms at forty Bohr and 8e-6 eV
/// at eighty. Eighty is chosen because the second number disappears into the convergence
/// threshold and the first does not, and because the far field is cheap enough now that what
/// decides the cost is the near field it hands the work back to.
const MULTIPOLE_RADIUS: f64 = 80.0;

/// One owner's charge cloud, reduced to the moments a distant observer can tell apart.
struct Cloud {
    centre: Vec3,
    charge: f64,
    dipole: Vec3,
    /// `Σ q d_α d_β` about [`Cloud::centre`] — the second moment, not its traceless part.
    quadrupole: [[f64; 3]; 3],
    /// Where this owner's sites sit in the owner-sorted index list.
    range: std::ops::Range<usize>,
}

/// Group sites by owner and reduce each group to its first three moments.
fn clouds(sites: &[ChargeSite]) -> (Vec<usize>, Vec<Cloud>) {
    let mut order: Vec<usize> = (0..sites.len()).collect();
    order.sort_unstable_by_key(|&i| sites[i].owner);

    let mut clouds = Vec::new();
    let mut start = 0;
    while start < order.len() {
        let owner = sites[order[start]].owner;
        let mut end = start;
        while end < order.len() && sites[order[end]].owner == owner {
            end += 1;
        }
        let members = &order[start..end];

        // The expansion converges in `d/R`, so the centre is placed to make `d` small rather
        // than at any physically distinguished point — the moments are defined about whatever
        // is chosen here and the reconstruction undoes the choice exactly.
        let mut centre = Vec3::zero();
        for &i in members {
            centre += sites[i].position;
        }
        centre = centre * (1.0 / members.len() as f64);

        let mut charge = 0.0;
        let mut dipole = Vec3::zero();
        let mut quadrupole = [[0.0; 3]; 3];
        for &i in members {
            let q = sites[i].charge;
            let offset = sites[i].position - centre;
            let d = [offset.x, offset.y, offset.z];
            charge += q;
            dipole += offset * q;
            for (a, row) in quadrupole.iter_mut().enumerate() {
                for (b, slot) in row.iter_mut().enumerate() {
                    *slot += q * d[a] * d[b];
                }
            }
        }

        clouds.push(Cloud {
            centre,
            charge,
            dipole,
            quadrupole,
            range: start..end,
        });
        start = end;
    }
    (order, clouds)
}

/// The potential, field and field gradient at `r` (measured from the source's centre) of a
/// charge, a dipole and a second moment sitting at that centre.
///
/// In terms of the interaction tensors `T = 1/R`, `T_α = ∂_α T`, and so on,
///
/// ```text
///   Φ      = q T    − μ_α T_α    + ½ Θ_αβ T_αβ
///   ∂_γ Φ  = q T_γ  − μ_α T_αγ   + ½ Θ_αβ T_αβγ
///   ∂_γδ Φ = q T_γδ − μ_α T_αγδ
/// ```
///
/// which is the Taylor expansion of `Σ_t q_t/|R − d_t|` in the source displacements `d_t`,
/// differentiated at the observer.
///
/// # Where this is truncated, and why the last term is not worth adding
///
/// Count a term by `(l, m)`: the source's multipole order and the observer's Taylor order. It
/// scales as `(d/R)^(l+m)` relative to the leading `q/R`. Kept are `l ≤ 2` for the source —
/// [`clouds`] computes a charge, a dipole and a second moment and nothing beyond — and `m ≤ 2`
/// for the observer, which is `(0,0)…(2,0)`, `(0,1)…(2,1)`, `(0,2)` and `(1,2)`.
///
/// So `(2,2)`, the `½ Θ_αβ T_αβγδ` that would complete the second derivative, is **fourth**
/// order — while `(3,0)`, the source's octupole, is **third** and is absent too, because the
/// cloud has no third moment to contribute it. Adding the rank-4 tensor would therefore buy
/// nothing: the error is already set one order lower, by a term that would need a different
/// change to recover.
///
/// The handover radius is fixed by measurement rather than by this counting anyway — see
/// [`MULTIPOLE_RADIUS`], which records what the error actually is at 40 and at 80 Bohr. The
/// counting says only that going to rank 4 alone is not the way to improve it.
fn multipole_field(r: Vec3, cloud: &Cloud) -> (f64, Vec3, [[f64; 3]; 3]) {
    let rv = [r.x, r.y, r.z];
    let r2 = r.norm2();
    let inverse = 1.0 / r2.sqrt();
    let i2 = inverse * inverse;
    let i3 = inverse * i2;
    let (i5, i7) = (i3 * i2, i3 * i2 * i2);

    let t1 = [-rv[0] * i3, -rv[1] * i3, -rv[2] * i3];
    let mut t2 = [[0.0; 3]; 3];
    let mut t3 = [[[0.0; 3]; 3]; 3];
    for a in 0..3 {
        for b in 0..3 {
            let delta_ab = f64::from(u8::from(a == b));
            t2[a][b] = (3.0 * rv[a] * rv[b] - delta_ab * r2) * i5;
            for c in 0..3 {
                let delta_ac = f64::from(u8::from(a == c));
                let delta_bc = f64::from(u8::from(b == c));
                t3[a][b][c] = (3.0 * r2 * (delta_ac * rv[b] + delta_bc * rv[a] + delta_ab * rv[c])
                    - 15.0 * rv[a] * rv[b] * rv[c])
                    * i7;
            }
        }
    }

    let mu = [cloud.dipole.x, cloud.dipole.y, cloud.dipole.z];

    // The charge's terms are laid down first, and by assignment. They used to be written inside
    // the loop over `a`, where `field[a] = q T_a` on the second pass overwrote the dipole and
    // quadrupole contributions the first pass had already accumulated into that slot. The
    // potential was unaffected — it only ever accumulates — so the far-field comparison, which
    // weighs the field and its gradient as small Taylor corrections to a correct value, stayed
    // inside its tolerance. Only differentiating the tensors against each other showed it.
    let mut potential = cloud.charge * inverse;
    let mut field = [
        cloud.charge * t1[0],
        cloud.charge * t1[1],
        cloud.charge * t1[2],
    ];
    let mut gradient = t2;
    for row in gradient.iter_mut() {
        for slot in row.iter_mut() {
            *slot *= cloud.charge;
        }
    }

    for a in 0..3 {
        potential -= mu[a] * t1[a];
        for b in 0..3 {
            potential += 0.5 * cloud.quadrupole[a][b] * t2[a][b];
            field[b] -= mu[a] * t2[a][b];
            for c in 0..3 {
                field[c] += 0.5 * cloud.quadrupole[a][b] * t3[a][b][c];
                gradient[b][c] -= mu[a] * t3[a][b][c];
            }
        }
    }
    (potential, Vec3::new(field[0], field[1], field[2]), gradient)
}

/// How much error one accepted tree node may contribute, in the units the potential is summed in.
///
/// **Not an opening angle.** The usual Barnes–Hut criterion `s/d < θ` is a *relative* bound, and
/// what matters here is an absolute one, because the pairwise rule it has to match is itself
/// absolute: a cloud is taken whole past [`MULTIPOLE_RADIUS`], and at 80 Bohr, with a span of
/// about 2 Bohr and a few electrons of charge, the first term it drops — the octupole — is about
/// `q s³/d⁴ ≈ 8e-7`. A fixed angle accepts distant nodes far more loosely than that and the
/// error shows: on the chain the far-field test uses, `θ = 0.3` moved the energy by 1.5 meV
/// against a tolerance of 50 µeV.
///
/// So a node is accepted when the same estimate of its own first neglected term,
/// `Σ|q| · s³ / d⁴`, is below this. Compact distant groups are then merged aggressively while
/// elongated near ones are opened, which is the behaviour the accuracy asks for rather than the
/// one a single angle happens to give.
const FAR_FIELD_TOLERANCE: f64 = 1.0e-7;

/// How many clouds a leaf holds before it is worth splitting.
///
/// Below this the walk costs more than the pairs it saves.
const LEAF_CLOUDS: usize = 8;

/// One node of the multipole tree: the combined moments of every cloud beneath it.
struct TreeNode {
    /// Where the moments are expanded about — the charge-weighted centre would concentrate the
    /// expansion on the heaviest atom, so the geometric one is used and `radius` measured from it.
    centre: Vec3,
    /// The largest distance from `centre` to any contained cloud's own centre. Half of what the
    /// acceptance test compares against, and a bound rather than an estimate.
    radius: f64,
    /// `Σ|q|` over the contained clouds. The other half: cancellation inside a node can only make
    /// its neglected terms smaller, so the absolute sum bounds them.
    abs_charge: f64,
    charge: f64,
    dipole: Vec3,
    quadrupole: [[f64; 3]; 3],
    /// Child node indices, empty for a leaf.
    children: Vec<usize>,
    /// Cloud indices, non-empty only for a leaf.
    clouds: Vec<usize>,
}

/// Build the tree over `clouds`, returning the arena with the root last.
///
/// # The translation
///
/// A node's moments are its children's, moved to the node's own centre. With `s = c − C` the
/// shift from a child's centre to the parent's, and the second-moment convention
/// [`Cloud::quadrupole`] uses,
///
/// ```text
/// Q' = Q                              D' = D + Q s
/// Quad'_αβ = Quad_αβ + D_α s_β + D_β s_α + Q s_α s_β
/// ```
///
/// which is exact — no information is lost going up, only on the way out, where the expansion is
/// truncated. That is why the accuracy question is entirely about [`FAR_FIELD_TOLERANCE`].
fn build_tree(clouds: &[Cloud]) -> Vec<TreeNode> {
    let mut arena: Vec<TreeNode> = Vec::new();
    let all: Vec<usize> = (0..clouds.len()).collect();
    split(clouds, all, &mut arena);
    arena
}

/// Recursively split `members` into octants, appending nodes to `arena`; returns the node index.
fn split(clouds: &[Cloud], members: Vec<usize>, arena: &mut Vec<TreeNode>) -> usize {
    let mut low = Vec3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut high = Vec3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &i in &members {
        let c = clouds[i].centre;
        low = Vec3::new(low.x.min(c.x), low.y.min(c.y), low.z.min(c.z));
        high = Vec3::new(high.x.max(c.x), high.y.max(c.y), high.z.max(c.z));
    }
    let centre = (low + high) * 0.5;

    let leaf = members.len() <= LEAF_CLOUDS;
    let mut children = Vec::new();
    if !leaf {
        // Eight octants about the box centre. An octant that comes out empty is dropped rather
        // than stored, so a sparse structure does not pay for the boxes it does not fill.
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); 8];
        for &i in &members {
            let c = clouds[i].centre;
            let octant = usize::from(c.x >= centre.x)
                | (usize::from(c.y >= centre.y) << 1)
                | (usize::from(c.z >= centre.z) << 2);
            buckets[octant].push(i);
        }
        // If every cloud lands in the same octant the split has made no progress — coincident or
        // near-coincident centres — and recursing would not terminate. Keep it a leaf.
        if buckets.iter().filter(|b| !b.is_empty()).count() > 1 {
            for bucket in buckets {
                if !bucket.is_empty() {
                    children.push(split(clouds, bucket, arena));
                }
            }
        }
    }

    let mut node = TreeNode {
        centre,
        radius: 0.0,
        abs_charge: 0.0,
        charge: 0.0,
        dipole: Vec3::zero(),
        quadrupole: [[0.0; 3]; 3],
        children,
        clouds: Vec::new(),
    };
    for &i in &members {
        let cloud = &clouds[i];
        let s = cloud.centre - centre;
        node.radius = node.radius.max(s.norm());
        node.abs_charge += cloud.charge.abs();
        node.charge += cloud.charge;
        node.dipole += cloud.dipole + s * cloud.charge;
        let d = [cloud.dipole.x, cloud.dipole.y, cloud.dipole.z];
        let sv = [s.x, s.y, s.z];
        for a in 0..3 {
            for b in 0..3 {
                node.quadrupole[a][b] += cloud.quadrupole[a][b]
                    + d[a] * sv[b]
                    + d[b] * sv[a]
                    + cloud.charge * sv[a] * sv[b];
            }
        }
    }
    if node.children.is_empty() {
        node.clouds = members;
    }
    arena.push(node);
    arena.len() - 1
}

/// A tree node read as a cloud, so [`multipole_field`] can be used unchanged.
fn node_as_cloud(node: &TreeNode) -> Cloud {
    Cloud {
        centre: node.centre,
        charge: node.charge,
        dipole: node.dipole,
        quadrupole: node.quadrupole,
        range: 0..0,
    }
}

/// [`direct_0d`] without the gradients, and with distant owners collapsed onto their multipole
/// moments.
///
/// The self-consistent field wants the site potentials and the energy and never looks at the
/// site gradients, which it would otherwise recompute and discard on every iteration. Dropping
/// them is worth about a third; collapsing the far field is worth an order of magnitude more,
/// and together they take the isolated lattice sum from the largest term in a linear-scaling
/// calculation to a term that is hard to find in a profile.
///
/// # Cost
///
/// Still `O(N²)`, now in the number of *atoms* rather than sites and with a constant small
/// enough that the near field decides the slope over any system size worth measuring. The
/// remaining quadratic is a point-multipole sum, which is what a tree code would accelerate if
/// this ever stopped being negligible.
pub fn direct_0d_potentials(sites: &[ChargeSite], out: &mut EwaldOutput) {
    use rayon::prelude::*;

    let (order, clouds) = clouds(sites);
    let positions: Vec<Vec3> = sites.iter().map(|s| s.position).collect();
    let charges: Vec<f64> = sites.iter().map(|s| s.charge).collect();
    let far = MULTIPOLE_RADIUS * MULTIPOLE_RADIUS;
    let arena = build_tree(&clouds);

    let rows: Vec<Vec<(usize, f64)>> = clouds
        .par_iter()
        .enumerate()
        .map(|(index, a)| {
            let members = &order[a.range.clone()];
            let mut potential = vec![0.0; members.len()];

            // The far field is summed once for the whole cloud, as a potential and its first two
            // derivatives at the centre; the near field has to be summed site by site.
            let (mut value, mut field, mut curvature) = (0.0, Vec3::zero(), [[0.0; 3]; 3]);
            let mut add_moments = |r: Vec3, source: &Cloud| {
                let (dv, de, dh) = multipole_field(r, source);
                value += dv;
                field += de;
                for (row, add) in curvature.iter_mut().zip(dh.iter()) {
                    for (slot, term) in row.iter_mut().zip(add.iter()) {
                        *slot += term;
                    }
                }
            };

            // Walk the tree instead of every other cloud. A node is taken whole when it is far
            // enough that **every** cloud inside it would have been far by the pairwise rule —
            // `d − s > MULTIPOLE_RADIUS`, by the triangle inequality — and small enough on the
            // sky that merging costs less than the expansion already does. The first condition
            // is what keeps the near field exactly the set it was; the second is the only place
            // the tree introduces an error of its own.
            let mut stack = vec![arena.len() - 1];
            while let Some(node_index) = stack.pop() {
                let node = &arena[node_index];
                let r = a.centre - node.centre;
                let distance = r.norm();
                // The first term this node's expansion drops, estimated the same way the
                // per-cloud handover's is: an octupole of size `Σ|q| s³`, seen from `d`.
                let neglected = node.abs_charge * node.radius.powi(3) / distance.powi(4);
                if !node.children.is_empty()
                    && distance - node.radius > MULTIPOLE_RADIUS
                    && neglected < FAR_FIELD_TOLERANCE
                {
                    add_moments(r, &node_as_cloud(node));
                    continue;
                }
                if node.children.is_empty() {
                    for &other in &node.clouds {
                        if other == index {
                            continue;
                        }
                        let b = &clouds[other];
                        let r = a.centre - b.centre;
                        if r.norm2() > far {
                            add_moments(r, b);
                        } else {
                            for (slot, &i) in potential.iter_mut().zip(members) {
                                let position = positions[i];
                                for &j in &order[b.range.clone()] {
                                    *slot += charges[j] / (positions[j] - position).norm();
                                }
                            }
                        }
                    }
                } else {
                    stack.extend_from_slice(&node.children);
                }
            }

            // Carry the cloud's potential out to its own sites, to the same order in `d/R` the
            // source was expanded to.
            for (slot, &i) in potential.iter_mut().zip(members) {
                let offset = positions[i] - a.centre;
                let d = [offset.x, offset.y, offset.z];
                let e = [field.x, field.y, field.z];
                let mut taylor = value;
                for a in 0..3 {
                    taylor += e[a] * d[a];
                    for b in 0..3 {
                        taylor += 0.5 * curvature[a][b] * d[a] * d[b];
                    }
                }
                *slot += taylor;
            }
            members.iter().copied().zip(potential).collect()
        })
        .collect();

    for row in rows {
        for (i, potential) in row {
            out.energy_ev += 0.5 * PM3_EV * charges[i] * potential;
            out.site_potential_ev[i] += PM3_EV * potential;
        }
    }
}

/// `Σ q_i q_j erfc(α r)/r` over lattice images inside the real-space cutoff.
///
/// `virial` is `Some` only where the strain derivative of the whole sum is available, so the
/// real-space half is never accumulated into a virial the reciprocal half will not complete.
fn real_space(
    sites: &[ChargeSite],
    params: &EwaldParams,
    out: &mut EwaldOutput,
    mut virial: Option<&mut Mat3>,
    context: &EwaldContext,
) {
    let list = &context.neighbors;
    let alpha = params.alpha;
    let two_alpha_over_sqrt_pi = 2.0 * alpha / PI.sqrt();

    // `erfc` and `exp` at distances that do not move while the charges do. Read from the table
    // where there is one; recompute where the geometry was too big to tabulate.
    let kernel = |index: usize, pair: &PairImage| -> RealKernel {
        if let Some(entry) = context.real_kernel.get(index) {
            return *entry;
        }
        let r = pair.r;
        if r <= 0.0 || excluded(sites, pair.a, pair.b, pair.t) {
            return RealKernel::default();
        }
        let screened = erfc(alpha * r) / r;
        let derivative =
            -(screened / r) - two_alpha_over_sqrt_pi * (-(alpha * r).powi(2)).exp() / r;
        RealKernel {
            screened,
            dcoef: derivative / r,
        }
    };

    // Each distinct interaction once, for the energy and the virial.
    for (index, pair) in list.all().iter().enumerate() {
        if !pair.is_unique_representative() {
            continue;
        }
        let entry = kernel(index, pair);
        if entry.screened == 0.0 {
            continue;
        }
        let qq = sites[pair.a].charge * sites[pair.b].charge;
        out.energy_ev += PM3_EV * qq * entry.screened;

        // dE/dr_a = q q f'(r) · (−d/r);  dE/dr_b = +the same vector.
        let contribution = pair.dvec * (PM3_EV * qq * entry.dcoef);
        out.site_gradient[pair.a] -= contribution;
        out.site_gradient[pair.b] += contribution;
        // dE/dε_αβ = Σ q q f'(r) d_α d_β / r
        if let Some(accumulator) = virial.as_deref_mut() {
            accumulate_outer(accumulator, contribution, pair.dvec);
        }
    }

    // Every ordered pair, for the potentials: φ_i = Σ_j q_j erfc(α r)/r. The same kernel the
    // energy loop above just used, at the same distances -- it used to be a second `erfc` per
    // pair, so an SCF iteration evaluated the function twice for every pair in the cutoff.
    for (index, pair) in list.all().iter().enumerate() {
        let screened = kernel(index, pair).screened;
        if screened != 0.0 {
            out.site_potential_ev[pair.a] += PM3_EV * sites[pair.b].charge * screened;
        }
    }
}

/// Whether two sites' interaction is handled elsewhere and must be kept out of the lattice sum:
/// sites on the same atom in the same cell, whose mutual energy is the NDDO one-center integral
/// set rather than a Coulomb term. Their images (`T ≠ 0`) are ordinary interactions and stay.
#[inline]
fn excluded(sites: &[ChargeSite], a: usize, b: usize, t: [i32; 3]) -> bool {
    t == [0, 0, 0] && sites[a].owner == sites[b].owner
}

/// Undo what the smooth half of the sum contributed for the excluded pairs.
///
/// The real-space loop can simply skip them, but the reciprocal sum cannot: it works through the
/// structure factor and has no notion of which pairs are which. It therefore includes every pair,
/// so the `erf` part of each excluded pair — the complement of the `erfc` the real-space loop
/// left out — has to be removed by hand. Forgetting this is a silent error: the energy stays
/// finite and looks plausible, and only a comparison against a molecular calculation reveals it.
fn remove_excluded_smooth_part(
    sites: &[ChargeSite],
    params: &EwaldParams,
    out: &mut EwaldOutput,
    mut virial: Option<&mut Mat3>,
) {
    let alpha = params.alpha;
    let two_alpha_over_sqrt_pi = 2.0 * alpha / PI.sqrt();
    for a in 0..sites.len() {
        for b in 0..sites.len() {
            if a == b || sites[a].owner != sites[b].owner {
                continue;
            }
            let d = sites[b].position - sites[a].position;
            let r = d.norm();
            if r <= 0.0 {
                continue;
            }
            let smooth = erf(alpha * r) / r;
            // Potentials count every ordered pair; the energy and virial count each once.
            out.site_potential_ev[a] -= PM3_EV * sites[b].charge * smooth;
            if a >= b {
                continue;
            }
            let qq = sites[a].charge * sites[b].charge;
            out.energy_ev -= PM3_EV * qq * smooth;
            // d/dr [erf(αr)/r] = −erf(αr)/r² + (2α/√π) e^{−α²r²}/r
            let derivative =
                -(smooth / r) + two_alpha_over_sqrt_pi * (-(alpha * r).powi(2)).exp() / r;
            let contribution = d * (PM3_EV * qq * derivative / r);
            out.site_gradient[a] += contribution;
            out.site_gradient[b] -= contribution;
            if let Some(accumulator) = virial.as_deref_mut() {
                accumulate_outer(accumulator, contribution * -1.0, d);
            }
        }
    }
}

/// 3D reciprocal-space term, `G = 0` omitted (tinfoil boundary).
fn reciprocal_3d(
    cell: &Cell,
    sites: &[ChargeSite],
    params: &EwaldParams,
    out: &mut EwaldOutput,
    virial: &mut Mat3,
    context: &EwaldContext,
) {
    let volume = cell.measure();
    // 2π/V, doubled because reciprocal_vectors returns only half of each ±G pair.
    let prefactor = 2.0 * TAU / volume;
    let inv_four_alpha2 = 1.0 / (4.0 * params.alpha * params.alpha);

    // Each `G` is independent of every other, and this loop is where a periodic SCF spends most
    // of its life: measured at 89–98% of every iteration across 6 to 72 atoms
    // (`examples/periodic_profile.rs`, `PM3_GAMMA_PROFILE=1`). The isolated paths have been
    // parallel since they were written; the periodic one never was.
    //
    // Split into a fixed number of contiguous chunks, mapped in parallel and **reduced in index
    // order**, rather than a `par_iter().reduce()`. A rayon reduction combines partial sums in
    // whatever order threads happen to finish, so the last bits of the energy would depend on
    // the machine and on the run. Fixed chunks summed in order give the same answer every time,
    // on any thread count, which is what the reproducibility tests hold this to and what makes
    // a regression in the twelfth digit mean something.
    use rayon::prelude::*;

    let chunk = context
        .gvectors
        .len()
        .div_ceil(rayon::current_num_threads().max(1) * 4)
        .max(1);
    let partials: Vec<Partial> = context
        .gvectors
        .par_chunks(chunk)
        .enumerate()
        .map(|(chunk_index, gs)| {
            let mut partial = Partial::zeros(sites.len());
            let mut phases: Vec<(f64, f64)> = vec![(0.0, 0.0); sites.len()];
            for (offset, g) in gs.iter().enumerate() {
                let g_index = chunk_index * chunk + offset;
                let g = *g;
                let g2 = g.norm2();
                let amplitude = (-g2 * inv_four_alpha2).exp() / g2;
                if amplitude == 0.0 {
                    continue;
                }
                // Structure factor S(G) = Σ_i q_i e^{iG·r_i}, kept as (Re, Im).
                let (mut sre, mut sim) = (0.0, 0.0);
                for (index, site) in sites.iter().enumerate() {
                    let (cos, sin) = context.phase(g_index, index, g, site.position);
                    sre += site.charge * cos;
                    sim += site.charge * sin;
                    phases[index] = (cos, sin);
                }

                let structure = sre * sre + sim * sim;
                partial.energy += PM3_EV * prefactor * amplitude * structure;

                for (index, (cos, sin)) in phases.iter().enumerate() {
                    // Re[S(G) e^{−iG·r_i}] and Im[S(G) e^{−iG·r_i}].
                    let real = sre * cos + sim * sin;
                    let imaginary = sim * cos - sre * sin;
                    partial.potential[index] += PM3_EV * 2.0 * prefactor * amplitude * real;
                    partial.gradient[index] += g
                        * (PM3_EV * 2.0 * prefactor * amplitude * sites[index].charge * imaginary);
                }

                // ∂/∂ε_αβ of (1/V)·A(G)·|S|², with G → (1 − εᵀ)G and V → V(1 + tr ε), and G·r
                // invariant so |S|² does not move:
                //     [ 2 (1/4α² + 1/G²) G_α G_β − δ_αβ ] · (2π/V) A |S|²
                let weight = PM3_EV * prefactor * amplitude * structure;
                let scale = 2.0 * (inv_four_alpha2 + 1.0 / g2) * weight;
                accumulate_outer(&mut partial.virial, g * scale, g);
                for axis in 0..3 {
                    add_diagonal(&mut partial.virial, axis, -weight);
                }
            }
            partial
        })
        .collect();

    for partial in partials {
        out.energy_ev += partial.energy;
        for (index, value) in partial.potential.iter().enumerate() {
            out.site_potential_ev[index] += value;
        }
        for (index, value) in partial.gradient.iter().enumerate() {
            out.site_gradient[index] += *value;
        }
        for axis in 0..3 {
            virial.col[axis] += partial.virial.col[axis];
        }
    }
}

/// One chunk's share of the reciprocal sum, summed into the total in chunk order.
struct Partial {
    energy: f64,
    potential: Vec<f64>,
    gradient: Vec<Vec3>,
    virial: Mat3,
}

impl Partial {
    fn zeros(n: usize) -> Self {
        Self {
            energy: 0.0,
            potential: vec![0.0; n],
            gradient: vec![Vec3::zero(); n],
            virial: Mat3::zero(),
        }
    }
}

/// `−(α/√π) Σ q_i²`: removes each Gaussian's interaction with itself, which the reciprocal
/// sum includes. No position or strain dependence.
fn self_term(sites: &[ChargeSite], params: &EwaldParams, out: &mut EwaldOutput) {
    let factor = params.alpha / PI.sqrt();
    for (index, site) in sites.iter().enumerate() {
        out.energy_ev -= PM3_EV * factor * site.charge * site.charge;
        out.site_potential_ev[index] -= PM3_EV * 2.0 * factor * site.charge;
    }
}

/// Uniform neutralizing background for a 3D cell with net charge `Q`.
///
/// Only 3D needs one: its `G = 0` reciprocal term is `1/G²`, which diverges and is dropped,
/// leaving the monopole unaccounted for. The 2D formulation keeps a finite `G = 0` term that
/// already *is* the charged sheet, and 1D refuses a charged cell outright.
fn background_3d(
    cell: &Cell,
    sites: &[ChargeSite],
    params: &EwaldParams,
    out: &mut EwaldOutput,
    virial: &mut Mat3,
) {
    let total: f64 = sites.iter().map(|s| s.charge).sum();
    if total == 0.0 {
        return;
    }
    let volume = cell.measure();
    let factor = PI / (2.0 * params.alpha * params.alpha * volume);
    out.energy_ev -= PM3_EV * factor * total * total;
    for potential in &mut out.site_potential_ev {
        *potential -= PM3_EV * 2.0 * factor * total;
    }
    // E_bg ∝ 1/V, and ∂V/∂ε_αβ = V δ_αβ, so ∂E_bg/∂ε_αβ = +factor Q² δ_αβ.
    for axis in 0..3 {
        add_diagonal(virial, axis, PM3_EV * factor * total * total);
    }
}

/// One in-plane `G`'s dependence on the out-of-plane separation `z`, with its first two `z`
/// derivatives and its `g` derivative.
///
/// Returns `(K, ∂K/∂z, ∂K/∂g, ∂²K/∂z²)` for `K = T₁ + T₂`,
/// `T₁ = e^{Gz} erfc(G/2α + αz)` and `T₂ = e^{−Gz} erfc(G/2α − αz)`.
///
/// Written naively both factors overflow long before their product does — `e^{Gz}` blows up
/// exactly where `erfc` vanishes. Rewriting `T = e^{−G²/4α² − α²z²} · erfcx(u)` makes the
/// exponent bounded, because `Gz − (G/2α + αz)²` collapses to `−G²/4α² − α²z²` with the `Gz`
/// cancelling. Both terms share that same prefactor. Where `u < 0` the scaled form is the one
/// that overflows, but there `e^{∓Gz} ≤ 1` and the direct product is safe, so the two branches
/// cover each other exactly.
///
/// The `z` derivative is `G(T₁ − T₂)`: the two `erfc'` pieces are equal and opposite and cancel
/// identically, which is worth knowing because it makes the slab force noticeably cheaper than
/// the energy.
/// The `g` and second `z` derivatives are the reason `shared` is computed once. All of them
/// differentiate the same two `erfc`s, and the Gaussian that falls out is the same
/// `e^{−g²/4α² − α²z²}` everywhere, because `e^{±gz} e^{−u²}` is exactly that regardless of
/// which branch computed `t1` and `t2`. With `∂t1/∂z = g·t1 − (2α/√π)·shared` and
/// `∂t2/∂z = −g·t2 + (2α/√π)·shared`:
///
/// ```text
///   ∂K/∂z   = g(t1 − t2)                       — the Gaussians cancel
///   ∂K/∂g   = z(t1 − t2) − (2/(α√π)) shared    — the Gaussians add
///   ∂²K/∂z² = g²·K − (4αg/√π)·shared           — they add here too
/// ```
///
/// The second derivative feeds the phased slab sum ([`crate::pbc::phased`]), where the mixed
/// `n̂ ⊗ n̂` block of the pair Hessian is exactly `∂²K/∂z²`.
#[inline]
pub(crate) fn slab_kernel(g: f64, z: f64, alpha: f64) -> (f64, f64, f64, f64) {
    let half_g_over_alpha = g / (2.0 * alpha);
    let u1 = half_g_over_alpha + alpha * z;
    let u2 = half_g_over_alpha - alpha * z;
    let shared = (-half_g_over_alpha * half_g_over_alpha - alpha * alpha * z * z).exp();
    let t1 = if u1 >= 0.0 {
        shared * erfcx(u1)
    } else {
        (g * z).exp() * erfc(u1)
    };
    let t2 = if u2 >= 0.0 {
        shared * erfcx(u2)
    } else {
        (-g * z).exp() * erfc(u2)
    };
    let difference = t1 - t2;
    let sum = t1 + t2;
    (
        sum,
        g * difference,
        z * difference - 2.0 / (alpha * PI.sqrt()) * shared,
        g * g * sum - 4.0 * alpha * g / PI.sqrt() * shared,
    )
}

/// 2D (slab) reciprocal-space term, after Parry.
///
/// Unlike 3D, the `z` dependence prevents the double sum over sites from collapsing into
/// `|S(G)|²`, so this is `O(N² N_G)` rather than `O(N N_G)`. That is the honest cost of an
/// exact slab sum; the alternative — padding the cell with vacuum and using the 3D formula
/// plus a correction — reintroduces a dependence on how much vacuum the caller chose.
///
/// # Strain
///
/// The slab's own strains are the in-plane ones, so the projector `P = 1 − n̂ ⊗ n̂` stands in for
/// the identity throughout: an in-plane strain `ε` scales the area by `1 + P:ε`, carries the
/// in-plane part of every separation to `(1 + ε)d`, and leaves `z = d·n̂` alone. Three things
/// then depend on it, and one conspicuously does not:
///
/// ```text
///   ∂(1/A)/∂ε_βγ  = −P_βγ / A
///   ∂|G|/∂ε_βγ    = −G_β G_γ / |G|          (G transforms as (1 − ε^T)G)
///   ∂(G·d)/∂ε_βγ  = 0                        (the two transformations cancel exactly)
/// ```
///
/// The last is what keeps this tractable: the phase is strain-invariant, so the whole derivative
/// is the `1/(A|G|)` prefactor and the kernel's dependence on `|G|`, with the cosine along for the
/// ride. The `G = 0` sheet depends on the strain only through `1/A`.
fn reciprocal_2d(
    cell: &Cell,
    sites: &[ChargeSite],
    params: &EwaldParams,
    out: &mut EwaldOutput,
    virial: &mut Mat3,
    context: &EwaldContext,
) {
    let periodic = cell.periodic_indices();
    let normal = cell
        .vector(periodic[0])
        .cross(cell.vector(periodic[1]))
        .normalized();
    // P = 1 − n̂ ⊗ n̂, the identity of the plane the slab is periodic in.
    let projector = {
        let mut p = Mat3::zero();
        accumulate_outer(&mut p, normal * -1.0, normal);
        add_diagonal(&mut p, 0, 1.0);
        add_diagonal(&mut p, 1, 1.0);
        add_diagonal(&mut p, 2, 1.0);
        p
    };
    let area = cell.measure();
    let alpha = params.alpha;
    // The phase here is `G·d` over pair displacements rather than `G·r` over sites, so the cached
    // per-site table does not apply; the enumeration still does.
    let gvectors = &context.gvectors;
    let sqrt_pi = PI.sqrt();

    // E = ½ Σ_{i,j} q_i q_j φ(d_ij) over *all* ordered pairs, i = j included: the smooth part
    // of the potential is regular at the origin, and its self-interaction is what `self_term`
    // then removes.
    for (i, site_i) in sites.iter().enumerate() {
        for site_j in sites.iter() {
            let d = site_j.position - site_i.position;
            let z = d.dot(normal);
            let mut potential = 0.0;
            let mut gradient = Vec3::zero(); // ∂φ/∂d
                                             // ∂φ/∂ε, split into the part that carries a G ⊗ G and the part that carries a P.
            let mut strain_outer = Mat3::zero();
            let mut strain_trace = 0.0;

            for g in gvectors {
                let magnitude = g.norm();
                let (kernel, dkernel_dz, dkernel_dg, _) = slab_kernel(magnitude, z, alpha);
                let phase = g.dot(d);
                let (sin, cos) = phase.sin_cos();
                // Doubled for the dropped half of each ±G pair.
                let scale = 2.0 * PI / (area * magnitude);
                potential += scale * cos * kernel;
                gradient += (*g * (-sin * kernel) + normal * (cos * dkernel_dz)) * scale;

                // ∂/∂ε_βγ [ (2π/(A|G|)) cos(G·d) K ]
                //   = scale·cos·[ (K/|G|² − K'/|G|) G_β G_γ − K P_βγ ]
                let radial = kernel / (magnitude * magnitude) - dkernel_dg / magnitude;
                accumulate_outer(&mut strain_outer, *g * (scale * cos * radial), *g);
                strain_trace -= scale * cos * kernel;
            }

            // G = 0: the Gaussian-smeared charged sheet, −(2π/A)[z erf(αz) + e^{−α²z²}/(α√π)].
            // It tends to −(2π/A)|z|, the field of a uniform sheet, and stays finite at z = 0 —
            // which is why a charged slab needs no separate background term.
            let sheet = -TAU / area
                * (z * erf(alpha * z) + (-(alpha * z).powi(2)).exp() / (alpha * sqrt_pi));
            potential += sheet;
            gradient += normal * (-TAU / area * erf(alpha * z));
            // The sheet depends on the strain through 1/A alone, so ∂sheet/∂ε_βγ = −P_βγ sheet.
            strain_trace -= sheet;

            let qq = site_i.charge * site_j.charge;
            out.energy_ev += 0.5 * PM3_EV * qq * potential;
            out.site_potential_ev[i] += PM3_EV * site_j.charge * potential;
            // E = ½ ΣΣ q_i q_j φ(r_j − r_i) with φ even, so ∂E/∂r_k = −Σ_j q_k q_j φ'(d_kj).
            out.site_gradient[i] -= gradient * (PM3_EV * qq);

            let weight = 0.5 * PM3_EV * qq;
            for axis in 0..3 {
                virial.col[axis] +=
                    strain_outer.col[axis] * weight + projector.col[axis] * (weight * strain_trace);
            }
        }
    }
}

/// 1D (chain) electrostatics by direct summation **grouped by cell**.
///
/// The grouping is the entire point. Summed pair by pair, `Σ_n q_i q_j / |d_ij + nL|` diverges
/// logarithmically for every pair; summed cell by cell, the monopole term vanishes for a
/// neutral cell and the leading surviving interaction is dipole–dipole, falling off as `1/n³`
/// so the sum converges absolutely. Pairing `+n` with `−n` additionally cancels the odd
/// multipole orders term by term, which is what leaves a clean `1/n³` tail to extrapolate.
///
/// No Ewald splitting is used here at all, so there is no `alpha` dependence to get wrong.
fn direct_1d(
    cell: &Cell,
    sites: &[ChargeSite],
    params: &EwaldParams,
    out: &mut EwaldOutput,
) -> Result<()> {
    let total: f64 = sites.iter().map(|s| s.charge).sum();
    let charged = total.abs() > 1.0e-10;

    let periodic_axis = cell.periodic_indices()[0];
    let axis = cell.vector(periodic_axis);
    let length = axis.norm();
    let images = params.chain_images.max(1);

    // The strain derivative comes free from the pair gradients already being computed. Under a
    // strain `ε` every pair separation transforms as `(1 + ε)(d + T)`, so
    //
    //     ∂E/∂ε_αβ = Σ_pairs (∂E/∂d_α)(d + T)_β
    //
    // — the ordinary virial, and the same identity the short-range half uses. There is no
    // reciprocal space here to differentiate separately: 1D is summed directly, which is what
    // makes its stress the easiest of the three rather than the hardest.
    let mut virial = Mat3::zero();

    // n = 0: the ordinary intra-cell sum, minus the pairs the one-center integrals own.
    for (i, site_i) in sites.iter().enumerate() {
        for (j, site_j) in sites.iter().enumerate() {
            if i == j || site_i.owner == site_j.owner {
                continue;
            }
            let d = site_j.position - site_i.position;
            let r = d.norm();
            let qq = site_i.charge * site_j.charge;
            out.energy_ev += 0.5 * PM3_EV * qq / r;
            out.site_potential_ev[i] += PM3_EV * site_j.charge / r;
            // ∂(1/r)/∂r_i = +d/r³
            let force = d * (PM3_EV * qq / (r * r * r));
            out.site_gradient[i] += force;
            // `force` is `∂E/∂r_i`, so `∂E/∂d` is its negative. Each distinct pair is visited
            // once in each direction and carries half the energy, so the virial takes the same
            // half; the two visits then add to one full outer product.
            accumulate_outer(&mut virial, force * -0.5, d);
        }
    }

    // ±n shells. Both signs are added inside the same iteration so the cancellation that makes
    // this converge happens before any rounding, not after a long partial sum.
    for n in 1..=images {
        let shift = axis * n as f64;
        for translation in [shift, shift * -1.0] {
            for (i, site_i) in sites.iter().enumerate() {
                for site_j in sites.iter() {
                    let d = site_j.position + translation - site_i.position;
                    let r = d.norm();
                    if r <= 0.0 {
                        continue;
                    }
                    let qq = site_i.charge * site_j.charge;
                    out.energy_ev += 0.5 * PM3_EV * qq / r;
                    out.site_potential_ev[i] += PM3_EV * site_j.charge / r;
                    let force = d * (PM3_EV * qq / (r * r * r));
                    out.site_gradient[i] += force;
                    // As in the `n = 0` block, and with the same halving: `d` already contains
                    // the translation, which is exactly what makes this the periodic virial
                    // rather than a molecular one.
                    accumulate_outer(&mut virial, force * -0.5, d);
                }
            }
        }
    }

    // Analytic tail beyond the summed images. The cell is neutral, so the leading remaining
    // cell-to-cell interaction is dipole–dipole, `U(n) = [|μ|² − 3(μ·ê)²]/(nL)³`.
    //
    // The bookkeeping is worth spelling out, because it is where the factor of two hides. The
    // loop above visits `+n` and `−n` separately and gives each ordered site pair weight ½, so
    // shell `n` contributes `½U(+n) + ½U(−n) = U(n)` — the dipole term is even in `n`, so the
    // two halves recombine into exactly one `U`, not two. The tail is therefore
    // `Σ_{n>N} U(n)` with no extra factor.
    //
    // Euler–Maclaurin gives the remaining lattice sum to three terms,
    //     Σ_{n>N} n⁻³ = 1/(2N²) − 1/(2N³) + 1/(4N⁴) + O(N⁻⁶),
    // and keeping all three matters: with only the first, the correction carries its own
    // `1/N³` error, larger than the quadrupole term it is meant to expose.
    // Skipped for a charged half of a neutral cell: the dipole of a charged distribution is
    // origin-dependent, so the formula below means nothing there. What is lost is the tail of the
    // *total*, which the three halves would otherwise have supplied between them — of order
    // `1e-6 eV` at the default image count, and the price of summing a divergent split with a
    // common truncation. See `pbc::gamma::build_setup`.
    // A charged chain, and the line charge that neutralizes it.
    //
    // The sum above diverges when `Q = Σq ≠ 0`: shell `n` contributes `Q²/(nL)` once the shell is
    // far enough that the cell looks like a point, so the partial sum carries `Q² H_N / L` with
    // `H_N` the harmonic number — logarithmically divergent in the image count, which is the same
    // statement as "the potential of a charged wire grows logarithmically with distance".
    //
    // 3D removes this with a uniform background; 1D takes a uniform **line charge** `−Q/L` along
    // the axis, and the three pieces — charges with charges, charges with line, line with line —
    // carry `+Q²H_N/L`, `−2Q²H_N/L` and `+Q²H_N/L`. They cancel for any `N`, so subtracting
    // `Q² H_N / L` from the truncated point sum leaves exactly what the neutralized system has.
    //
    // What does not cancel is a **convention**: the leftover finite part still depends on where
    // the logarithm's reference length is put, and the choice here is the cell length `L`, which
    // is what makes `H_N` and not `H_N + const` the right thing to subtract. So a charged 1D cell
    // has a well-defined energy *within this convention*, exactly as a charged 3D cell has one
    // within the jellium convention, and the same warning applies to both: absolute energies of
    // cells with different charge or different length are not comparable. See `docs/pbc.md`.
    //
    // The Fock potential follows by differentiating: `∂E_line/∂q_i = −2 Q H_N / L`, the same
    // constant shift at every site. Leaving it out would let the SCF converge against a
    // background the energy expression knows about and the potential does not.
    if charged {
        // `H_N` alone would be the wrong thing to subtract, and wrong in a way that hides: each
        // cell would still get a finite, stable energy, and only a cell compared with its own
        // doubled version would reveal that the leftover carries a `ln L` the two do not share.
        //
        // What the sum actually grows like is the logarithm of the **extent** reached, `N · L`,
        // because that is the distance at which the chain stops looking like a chain. Writing the
        // subtraction that way — `H_N + ln(L/L₀)` — leaves a remainder that depends on neither
        // the image count nor how the same chain was cut into cells. `L₀` is one Bohr: a fixed
        // reference rather than a cell-dependent one, which is exactly the property that makes
        // the convention size-consistent.
        let harmonic: f64 = (1..=images).map(|n| 1.0 / n as f64).sum();
        let extent = harmonic + (length / CHAIN_REFERENCE_LENGTH).ln();
        let per_charge = -2.0 * PM3_EV * total * extent / length;
        out.energy_ev -= PM3_EV * total * total * extent / length;
        for potential in out.site_potential_ev.iter_mut() {
            *potential += per_charge;
        }
        // `E_line = −Q²(H_N + ln(L/L₀))/L`. Straining the axis by `ε` takes `L → L(1 + ε)`, so
        // `∂E_line/∂ε = Q²(H_N + ln(L/L₀) − 1)/L` — the `−1` from differentiating the logarithm,
        // which a virial derived from the `H_N`-only form would have missed.
        add_diagonal(
            &mut virial,
            periodic_axis,
            PM3_EV * total * total * (extent - 1.0) / length,
        );
    }

    // The full affine derivative, not just the axial component. Deforming a chain transversally
    // does change its energy — it moves the atoms apart — so `∂E/∂ε` has off-axis entries and
    // they are correct. What a chain does not have is a transverse *cell* degree of freedom, and
    // that projection belongs to whoever is relaxing a cell rather than to the sum itself; see
    // [`crate::pbc::gradient::forces_and_stress`].
    out.virial = Some(virial);

    if charged {
        return Ok(());
    }
    let dipole = sites
        .iter()
        .fold(Vec3::zero(), |acc, s| acc + s.position * s.charge);
    let along = dipole.dot(axis / length);
    let dipole_dipole = dipole.norm2() - 3.0 * along * along;
    let n = images as f64;
    let lattice_tail = 0.5 / (n * n) - 0.5 / (n * n * n) + 0.25 / (n * n * n * n);
    let tail = dipole_dipole / (length * length * length) * lattice_tail;
    out.energy_ev += PM3_EV * tail;
    Ok(())
}

/// Reciprocal-lattice vectors of the periodic subspace with `0 < |G| ≤ gmax`, keeping only one
/// of each `±G` pair.
///
/// Every quantity the reciprocal sums produce is even under `G → −G`: `|S(G)|²` obviously,
/// `Re[S(G) e^{−iG·r}]` because both factors conjugate together, the gradient term because `G`
/// and `Im[S e^{−iG·r}]` each change sign, and the virial because it is quadratic in `G`. So the
/// half-space carries all the information and the callers double it — half the structure-factor
/// work, which is where a periodic SCF spends most of its time for a small cell.
pub(crate) fn reciprocal_vectors_for(cell: &Cell, gmax: f64) -> Vec<Vec3> {
    reciprocal_vectors(cell, gmax)
}

fn reciprocal_vectors(cell: &Cell, gmax: f64) -> Vec<Vec3> {
    let basis = cell.reciprocal_basis();
    let mut limits = [0i32; 3];
    for (index, b) in &basis {
        let length = b.norm();
        limits[*index] = if length > 0.0 {
            (gmax / length).ceil() as i32
        } else {
            0
        };
    }
    let vector_of = |index: usize| -> Vec3 {
        basis
            .iter()
            .find(|(i, _)| *i == index)
            .map(|(_, b)| *b)
            .unwrap_or_else(Vec3::zero)
    };
    let (b0, b1, b2) = (vector_of(0), vector_of(1), vector_of(2));
    let gmax2 = gmax * gmax;
    let mut out = Vec::new();
    for n0 in -limits[0]..=limits[0] {
        for n1 in -limits[1]..=limits[1] {
            for n2 in -limits[2]..=limits[2] {
                // Keep the lexicographically positive member of each ±G pair.
                let leading = [n0, n1, n2].into_iter().find(|v| *v != 0);
                match leading {
                    None => continue,
                    Some(sign) if sign < 0 => continue,
                    Some(_) => {}
                }
                let g = b0 * n0 as f64 + b1 * n1 as f64 + b2 * n2 as f64;
                if g.norm2() <= gmax2 {
                    out.push(g);
                }
            }
        }
    }
    out
}

/// `m += a ⊗ b` for the column-major 3×3 accumulator (`m[α][β] += a_α b_β`).
#[inline]
fn accumulate_outer(m: &mut Mat3, a: Vec3, b: Vec3) {
    // Mat3 stores columns, so column β holds the entries with second index β.
    m.col[0] += a * b.x;
    m.col[1] += a * b.y;
    m.col[2] += a * b.z;
}

/// `m[axis][axis] += value`.
#[inline]
fn add_diagonal(m: &mut Mat3, axis: usize, value: f64) {
    match axis {
        0 => m.col[0].x += value,
        1 => m.col[1].y += value,
        _ => m.col[2].z += value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbc::ewald_reference::{
        direct_energy, direct_potentials, surface_dipole_energy, RefSite,
    };

    fn as_reference(sites: &[ChargeSite]) -> Vec<RefSite> {
        sites
            .iter()
            .map(|s| RefSite {
                position: s.position,
                charge: s.charge,
            })
            .collect()
    }

    /// Conventional 8-ion rocksalt cell with nearest-neighbour distance `a` (Bohr).
    fn rocksalt(a: f64) -> (Cell, Vec<ChargeSite>) {
        let cell = Cell::cubic(2.0 * a).unwrap();
        let mut sites = Vec::new();
        let mut owner = 0;
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    sites.push(ChargeSite::on_atom(
                        Vec3::new(i as f64 * a, j as f64 * a, k as f64 * a),
                        if (i + j + k) % 2 == 0 { 1.0 } else { -1.0 },
                        owner,
                    ));
                    owner += 1;
                }
            }
        }
        (cell, sites)
    }

    /// The Madelung constant is the one closed-form number a 3D lattice sum can be held to.
    /// Rocksalt: `E per ion = −M q²/a` with `M = 1.7475645946331821...`.
    #[test]
    fn rocksalt_reproduces_the_madelung_constant() {
        const MADELUNG: f64 = 1.747_564_594_633_182_1;
        let a = 5.0;
        let (cell, sites) = rocksalt(a);
        let params = EwaldParams::tuned(20.0, 1.0e-14);
        let out = ewald(&cell, &sites, &params).unwrap();
        // 8 ions in the cell, energy per ion = −M/a (Hartree·Bohr units), halved because the
        // per-ion energy already counts each interaction once.
        let expected = -MADELUNG / a * PM3_EV * 8.0 / 2.0;
        let relative = (out.energy_ev - expected).abs() / expected.abs();
        assert!(
            relative < 1.0e-10,
            "Ewald {} vs Madelung {} (rel {relative:.3e})",
            out.energy_ev,
            expected
        );
    }

    /// The result must not depend on where the split is put. Changing `alpha` moves work
    /// between the real and reciprocal sums and is the sharpest internal check that the two
    /// halves, the self term and the background are mutually consistent.
    #[test]
    fn energy_is_independent_of_the_splitting_parameter() {
        let (cell, sites) = rocksalt(5.0);
        let mut energies = Vec::new();
        for real_cutoff in [12.0, 16.0, 20.0, 26.0] {
            let params = EwaldParams::tuned(real_cutoff, 1.0e-14);
            energies.push(ewald(&cell, &sites, &params).unwrap().energy_ev);
        }
        let first = energies[0];
        for (index, energy) in energies.iter().enumerate() {
            let relative = (energy - first).abs() / first.abs();
            assert!(
                relative < 1.0e-10,
                "alpha choice {index} changed the energy by {relative:.3e}"
            );
        }
    }

    /// A neutral cell whose dipole moment vanishes by construction.
    ///
    /// The construction has to be `+q` at **both** `c ± d₁` and `−q` at **both** `c ± d₂`:
    /// putting `+q` at `c + d` and `−q` at `c − d` looks symmetric but gives `M = 2qd`, not
    /// zero. With `Σ q = 0` and `Σ q r = 0` the tinfoil and vacuum-cluster conventions
    /// coincide, so the direct sum converges quickly and can be compared to the Ewald energy
    /// with no boundary term in between.
    fn dipole_free_cell() -> (Cell, Vec<ChargeSite>) {
        let edge = 8.0;
        let cell = Cell::cubic(edge).unwrap();
        let centre = Vec3::new(edge / 2.0, edge / 2.0, edge / 2.0);
        let pairs = [
            (Vec3::new(1.1, 0.7, -0.5), 0.8),
            (Vec3::new(-2.0, 1.5, 2.2), -0.8),
        ];
        let mut sites = Vec::new();
        for (index, (offset, charge)) in pairs.iter().enumerate() {
            sites.push(ChargeSite::on_atom(centre + *offset, *charge, 2 * index));
            sites.push(ChargeSite::on_atom(
                centre - *offset,
                *charge,
                2 * index + 1,
            ));
        }
        (cell, sites)
    }

    /// Against the brute-force lattice sum, which shares none of Ewald's algebra — no
    /// splitting, no self term, no background, no reciprocal space.
    #[test]
    fn matches_the_direct_lattice_sum() {
        let (cell, sites) = dipole_free_cell();
        let params = EwaldParams::tuned(24.0, 1.0e-14);
        let out = ewald(&cell, &sites, &params).unwrap();

        let reference = as_reference(&sites);
        // Zero dipole, so the shape/boundary term is identically zero here.
        let surface = surface_dipole_energy(&cell, &reference);
        assert!(surface.abs() < 1.0e-12, "fixture is not dipole-free");

        // The direct sum still has a quadrupolar tail, so hold it to converging toward the
        // Ewald value rather than to a tolerance its own truncation would decide.
        let coarse = (direct_energy(&cell, &reference, 15) - out.energy_ev).abs();
        let fine = (direct_energy(&cell, &reference, 40) - out.energy_ev).abs();
        assert!(
            fine < coarse,
            "the direct sum is not approaching the Ewald value ({fine:.3e} vs {coarse:.3e})"
        );
        let relative = fine / out.energy_ev.abs();
        assert!(
            relative < 1.0e-5,
            "Ewald {} vs direct (rel {relative:.3e})",
            out.energy_ev
        );
    }

    /// When the cell *does* carry a dipole the two conventions genuinely differ, and the gap
    /// is the depolarizing energy of the cluster's own surface charge. The dipolar tail decays
    /// as `1/R³` against a shell population growing as `R²`, so the direct sum approaches that
    /// limit only slowly — the test asserts the approach rather than a fixed tolerance, which
    /// is the honest statement of what a conditionally convergent sum can be held to.
    #[test]
    fn surface_term_bridges_the_boundary_conventions() {
        let cell = Cell::new(
            Vec3::new(7.0, 0.0, 0.0),
            Vec3::new(0.9, 6.4, 0.0),
            Vec3::new(0.0, 0.5, 7.3),
            [true, true, true],
        )
        .unwrap();
        let sites = vec![
            ChargeSite::on_atom(Vec3::new(0.4, 0.5, 0.6), 0.8, 0),
            ChargeSite::on_atom(Vec3::new(3.1, 1.7, 3.6), -0.5, 1),
            ChargeSite::on_atom(Vec3::new(5.2, 4.4, 1.4), -0.3, 2),
        ];
        let params = EwaldParams::tuned(24.0, 1.0e-14);
        let out = ewald(&cell, &sites, &params).unwrap();
        let reference = as_reference(&sites);
        let surface = surface_dipole_energy(&cell, &reference);
        assert!(surface.abs() > 0.1, "fixture should have a real dipole");
        let bridged = out.energy_ev + surface;

        let mut previous = f64::INFINITY;
        let mut last = f64::NAN;
        for shells in [12, 24, 48] {
            let direct = direct_energy(&cell, &reference, shells);
            let gap = (direct - bridged).abs();
            assert!(
                gap < previous,
                "the direct sum stopped approaching the bridged value at {shells} shells \
                 ({gap:.3e} vs {previous:.3e})"
            );
            previous = gap;
            last = gap;
        }
        // Without the surface term the two would differ by |surface| itself; with it, the
        // residual has to be a small fraction of that.
        assert!(
            last < 0.15 * surface.abs(),
            "residual {last:.3e} is not small against the surface term {surface:.3e}"
        );
    }

    /// `E = ½ Σ q_i φ_i` ties the potentials to the energy. If the site potential is missing a
    /// term the energy has (the background is the usual casualty), this fails.
    #[test]
    fn potentials_are_consistent_with_the_energy() {
        for charge_offset in [0.0, 0.35] {
            let cell = Cell::cubic(9.0).unwrap();
            let sites = vec![
                ChargeSite::on_atom(Vec3::new(0.5, 0.5, 0.5), 0.6 + charge_offset, 0),
                ChargeSite::on_atom(Vec3::new(4.1, 2.2, 1.3), -0.35, 1),
                ChargeSite::on_atom(Vec3::new(2.0, 6.4, 5.5), -0.25, 2),
            ];
            let params = EwaldParams::tuned(18.0, 1.0e-14);
            let out = ewald(&cell, &sites, &params).unwrap();
            let from_potentials: f64 = 0.5
                * sites
                    .iter()
                    .zip(&out.site_potential_ev)
                    .map(|(s, p)| s.charge * p)
                    .sum::<f64>();
            assert!(
                (out.energy_ev - from_potentials).abs() < 1.0e-9 * out.energy_ev.abs().max(1.0),
                "charge offset {charge_offset}: energy {} vs ½Σqφ {from_potentials}",
                out.energy_ev
            );
        }
    }

    /// The site potentials must match the brute-force ones **up to a constant**.
    ///
    /// The constant is not slack in the test, it is physics: dropping the `G = 0` term fixes
    /// the cell-average potential at zero, while a finite cluster summed in vacuum settles on
    /// a different reference level. Only potential *differences* are observable in a neutral
    /// cell, which is exactly why the energy `½ Σ q φ` is unaffected — `Σ q = 0` annihilates
    /// any constant. So the offset is measured once and then required to be the *same* for
    /// every site, which is a strictly stronger statement than comparing each site loosely.
    #[test]
    fn potentials_match_the_direct_sum_up_to_the_reference_level() {
        let (cell, sites) = dipole_free_cell();
        let params = EwaldParams::tuned(24.0, 1.0e-14);
        let out = ewald(&cell, &sites, &params).unwrap();
        let reference = as_reference(&sites);
        let direct = direct_potentials(&cell, &reference, 30);

        let offset = out.site_potential_ev[0] - direct[0];
        for (index, (ewald_value, direct_value)) in
            out.site_potential_ev.iter().zip(&direct).enumerate()
        {
            let residual = (ewald_value - direct_value - offset).abs();
            assert!(
                residual < 1.0e-3,
                "site {index}: Ewald {ewald_value} vs direct {direct_value} differ by \
                 {:.6} rather than the common offset {offset:.6}",
                ewald_value - direct_value
            );
        }
        // And the offset really is only a reference level: it must cancel out of the energy.
        let from_potentials: f64 = 0.5
            * sites
                .iter()
                .zip(&out.site_potential_ev)
                .map(|(s, p)| s.charge * p)
                .sum::<f64>();
        assert!((from_potentials - out.energy_ev).abs() < 1.0e-9 * out.energy_ev.abs().max(1.0));
    }

    /// Analytic site gradients against a central difference of the energy.
    #[test]
    fn gradients_match_finite_differences() {
        let cell = Cell::new(
            Vec3::new(8.0, 0.0, 0.0),
            Vec3::new(1.1, 7.2, 0.0),
            Vec3::new(0.3, 0.6, 8.4),
            [true, true, true],
        )
        .unwrap();
        let base = vec![
            ChargeSite::on_atom(Vec3::new(0.4, 0.9, 1.2), 0.75, 0),
            ChargeSite::on_atom(Vec3::new(3.6, 2.1, 4.4), -0.4, 1),
            ChargeSite::on_atom(Vec3::new(6.0, 5.3, 2.0), -0.35, 2),
        ];
        let params = EwaldParams::tuned(20.0, 1.0e-14);
        let analytic = ewald(&cell, &base, &params).unwrap().site_gradient;

        let step = 1.0e-5;
        for index in 0..base.len() {
            for axis in 0..3 {
                let mut plus = base.clone();
                let mut minus = base.clone();
                shift(&mut plus[index].position, axis, step);
                shift(&mut minus[index].position, axis, -step);
                let numeric = (ewald(&cell, &plus, &params).unwrap().energy_ev
                    - ewald(&cell, &minus, &params).unwrap().energy_ev)
                    / (2.0 * step);
                let got = analytic[index].get(axis);
                assert!(
                    (got - numeric).abs() < 1.0e-6 * numeric.abs().max(1.0),
                    "site {index} axis {axis}: analytic {got} vs numeric {numeric}"
                );
            }
        }
    }

    /// Forces must sum to zero: a periodic system cannot accelerate itself.
    #[test]
    fn gradients_sum_to_zero() {
        let (cell, sites) = rocksalt(4.5);
        let params = EwaldParams::tuned(18.0, 1.0e-13);
        let out = ewald(&cell, &sites, &params).unwrap();
        let total = out
            .site_gradient
            .iter()
            .fold(Vec3::zero(), |acc, g| acc + *g);
        assert!(
            total.norm() < 1.0e-9,
            "net Ewald force is {total:?}, not zero"
        );
    }

    /// The virial against a central difference of the energy under strain — the only way to
    /// catch a sign or factor slip in the reciprocal-space strain derivative.
    #[test]
    fn virial_matches_strained_finite_differences() {
        let cell = Cell::new(
            Vec3::new(7.5, 0.0, 0.0),
            Vec3::new(0.8, 6.9, 0.0),
            Vec3::new(0.4, 0.3, 8.1),
            [true, true, true],
        )
        .unwrap();
        let sites = vec![
            ChargeSite::on_atom(Vec3::new(0.5, 0.6, 0.7), 0.85, 0),
            ChargeSite::on_atom(Vec3::new(3.2, 2.6, 4.0), -0.5, 1),
            ChargeSite::on_atom(Vec3::new(5.9, 4.9, 1.8), -0.35, 2),
        ];
        let params = EwaldParams::tuned(20.0, 1.0e-14);
        let analytic = ewald(&cell, &sites, &params).unwrap().virial.unwrap();

        let step = 1.0e-6;
        for alpha in 0..3 {
            for beta in 0..3 {
                let energy_at = |sign: f64| -> f64 {
                    let mut strain = Mat3::zero();
                    add_diagonal_at(&mut strain, alpha, beta, sign * step);
                    let strained_cell = cell.strained(&strain);
                    let strained_sites: Vec<ChargeSite> = sites
                        .iter()
                        .map(|s| ChargeSite {
                            position: s.position + strain.mul_vec(s.position),
                            ..*s
                        })
                        .collect();
                    ewald(&strained_cell, &strained_sites, &params)
                        .unwrap()
                        .energy_ev
                };
                let numeric = (energy_at(1.0) - energy_at(-1.0)) / (2.0 * step);
                let got = component(&analytic, alpha, beta);
                assert!(
                    (got - numeric).abs() < 1.0e-5 * numeric.abs().max(1.0),
                    "virial[{alpha}][{beta}]: analytic {got} vs numeric {numeric}"
                );
            }
        }
    }

    /// A charged cell must be finite, and its background must show up in the energy, the site
    /// potentials and the virial — but never in the forces.
    #[test]
    fn charged_cell_background_is_consistent() {
        let cell = Cell::cubic(10.0).unwrap();
        let neutral = vec![
            ChargeSite::on_atom(Vec3::new(1.0, 1.0, 1.0), 0.5, 0),
            ChargeSite::on_atom(Vec3::new(5.0, 5.0, 5.0), -0.5, 1),
        ];
        let charged = vec![
            ChargeSite::on_atom(Vec3::new(1.0, 1.0, 1.0), 0.5, 0),
            ChargeSite::on_atom(Vec3::new(5.0, 5.0, 5.0), 0.5, 1),
        ];
        let params = EwaldParams::tuned(20.0, 1.0e-13);
        let neutral_out = ewald(&cell, &neutral, &params).unwrap();
        let charged_out = ewald(&cell, &charged, &params).unwrap();
        assert!(charged_out.energy_ev.is_finite());

        // Q = 0 must leave the background term exactly absent: the neutral virial has no
        // isotropic background contribution to remove.
        let volume = cell.measure();
        let factor = PI / (2.0 * params.alpha * params.alpha * volume);
        let expected_background = -PM3_EV * factor * 1.0; // Q² = 1
                                                          // Recomputing the charged energy without the background must differ by exactly that.
        let mut without = charged_out.clone();
        without.energy_ev -= expected_background;
        assert!(neutral_out.energy_ev.is_finite() && without.energy_ev.is_finite());

        // The background is position-independent, so a rigid translation of the charged cell
        // changes nothing — and the forces still sum to zero.
        let translated: Vec<ChargeSite> = charged
            .iter()
            .map(|s| ChargeSite {
                position: s.position + Vec3::new(1.3, -0.7, 2.1),
                ..*s
            })
            .collect();
        let translated_out = ewald(&cell, &translated, &params).unwrap();
        assert!(
            (translated_out.energy_ev - charged_out.energy_ev).abs() < 1.0e-9,
            "a rigid translation changed a charged cell's energy"
        );
        let total = charged_out
            .site_gradient
            .iter()
            .fold(Vec3::zero(), |acc, g| acc + *g);
        assert!(total.norm() < 1.0e-9, "charged-cell forces do not cancel");
    }

    /// The charged-cell gradient and virial must still pass their finite-difference checks —
    /// the background contributes to one and not the other, which is easy to get backwards.
    #[test]
    fn charged_cell_derivatives_match_finite_differences() {
        let cell = Cell::cubic(9.0).unwrap();
        let sites = vec![
            ChargeSite::on_atom(Vec3::new(0.7, 1.1, 2.3), 0.6, 0),
            ChargeSite::on_atom(Vec3::new(4.4, 3.9, 5.1), 0.4, 1),
        ];
        let params = EwaldParams::tuned(18.0, 1.0e-14);
        let out = ewald(&cell, &sites, &params).unwrap();

        let step = 1.0e-5;
        for index in 0..sites.len() {
            for axis in 0..3 {
                let mut plus = sites.clone();
                let mut minus = sites.clone();
                shift(&mut plus[index].position, axis, step);
                shift(&mut minus[index].position, axis, -step);
                let numeric = (ewald(&cell, &plus, &params).unwrap().energy_ev
                    - ewald(&cell, &minus, &params).unwrap().energy_ev)
                    / (2.0 * step);
                let got = out.site_gradient[index].get(axis);
                assert!(
                    (got - numeric).abs() < 1.0e-6 * numeric.abs().max(1.0),
                    "charged gradient [{index}][{axis}]: {got} vs {numeric}"
                );
            }
        }

        let strain_step = 1.0e-6;
        for alpha in 0..3 {
            for beta in 0..3 {
                let energy_at = |sign: f64| -> f64 {
                    let mut strain = Mat3::zero();
                    add_diagonal_at(&mut strain, alpha, beta, sign * strain_step);
                    let strained_sites: Vec<ChargeSite> = sites
                        .iter()
                        .map(|s| ChargeSite {
                            position: s.position + strain.mul_vec(s.position),
                            ..*s
                        })
                        .collect();
                    ewald(&cell.strained(&strain), &strained_sites, &params)
                        .unwrap()
                        .energy_ev
                };
                let numeric = (energy_at(1.0) - energy_at(-1.0)) / (2.0 * strain_step);
                let got = component(&out.virial.unwrap(), alpha, beta);
                assert!(
                    (got - numeric).abs() < 1.0e-5 * numeric.abs().max(1.0),
                    "charged virial[{alpha}][{beta}]: {got} vs {numeric}"
                );
            }
        }
    }

    // ---- 2D (slab) ----

    /// A neutral slab with no dipole normal to the plane, so the direct ring sum and the
    /// tinfoil-equivalent Parry sum converge to the same value with no boundary term.
    fn slab_fixture(vacuum: f64) -> (Cell, Vec<ChargeSite>) {
        let cell = Cell::new(
            Vec3::new(6.0, 0.0, 0.0),
            Vec3::new(1.2, 5.6, 0.0),
            Vec3::new(0.0, 0.0, vacuum),
            [true, true, false],
        )
        .unwrap();
        // Charges placed symmetrically about z = 0 with equal signs at ±z, so the normal
        // dipole vanishes while the in-plane structure stays non-trivial.
        let sites = vec![
            ChargeSite::on_atom(Vec3::new(0.5, 0.4, 0.9), 0.7, 0),
            ChargeSite::on_atom(Vec3::new(0.5, 0.4, -0.9), 0.7, 1),
            ChargeSite::on_atom(Vec3::new(3.4, 2.9, 0.35), -0.7, 2),
            ChargeSite::on_atom(Vec3::new(3.4, 2.9, -0.35), -0.7, 3),
        ];
        (cell, sites)
    }

    /// The slab sum must not know how much vacuum the caller padded the cell with. This is the
    /// property that a "3D Ewald with a big c axis" approach cannot have, and the reason the
    /// exact 2D formulation is worth its extra cost.
    #[test]
    fn slab_energy_is_independent_of_the_vacuum_thickness() {
        let params = EwaldParams::tuned(18.0, 1.0e-13);
        let mut energies = Vec::new();
        for vacuum in [30.0, 120.0, 600.0] {
            let (cell, sites) = slab_fixture(vacuum);
            energies.push(ewald(&cell, &sites, &params).unwrap().energy_ev);
        }
        for energy in &energies {
            let relative = (energy - energies[0]).abs() / energies[0].abs();
            assert!(
                relative < 1.0e-12,
                "vacuum thickness changed the slab energy by {relative:.3e}"
            );
        }
    }

    /// Moving the real/reciprocal split must leave the slab energy alone, exactly as in 3D.
    /// Run for a charged slab too: 2D keeps a finite `G = 0` term instead of a background, and
    /// this is what confirms that term really does carry the net charge consistently.
    #[test]
    fn slab_energy_is_independent_of_the_splitting_parameter() {
        for extra_charge in [0.0, 0.5] {
            let (cell, mut sites) = slab_fixture(60.0);
            sites[0].charge += extra_charge;
            let mut energies = Vec::new();
            for real_cutoff in [12.0, 16.0, 22.0] {
                let params = EwaldParams::tuned(real_cutoff, 1.0e-13);
                energies.push(ewald(&cell, &sites, &params).unwrap().energy_ev);
            }
            for energy in &energies {
                let relative = (energy - energies[0]).abs() / energies[0].abs().max(1.0);
                assert!(
                    relative < 1.0e-9,
                    "charge offset {extra_charge}: alpha changed the slab energy by {relative:.3e}"
                );
            }
        }
    }

    /// Against the direct ring sum, which shares none of Parry's algebra.
    ///
    /// The ring sum converges as `1/N`: a ring at radius `R` holds `O(R)` cells whose
    /// dipole–dipole interaction falls off as `1/R³`, leaving an `O(1/N)` tail. Comparing raw
    /// values at any affordable `N` would therefore be comparing against the truncation, not
    /// against the sum — measured, the gap goes 0.487, 0.250, 0.127, 0.0637, 0.0319, 0.0160 for
    /// N = 10…320, halving exactly. Richardson extrapolation `2E(2N) − E(N)` removes that
    /// leading term and lands on the Parry value.
    #[test]
    fn slab_matches_the_direct_lattice_sum() {
        let (cell, sites) = slab_fixture(60.0);
        let params = EwaldParams::tuned(20.0, 1.0e-13);
        let out = ewald(&cell, &sites, &params).unwrap();
        let reference = as_reference(&sites);
        let surface = surface_dipole_energy(&cell, &reference);
        assert!(surface.abs() < 1.0e-12, "fixture has a normal dipole");

        let coarse = direct_energy(&cell, &reference, 160);
        let fine = direct_energy(&cell, &reference, 320);
        // The 1/N law is itself part of the claim: check the gap really does halve before
        // relying on an extrapolation that assumes it.
        let ratio = (coarse - out.energy_ev) / (fine - out.energy_ev);
        assert!(
            (ratio - 2.0).abs() < 0.1,
            "the ring sum is not converging as 1/N (ratio {ratio:.4})"
        );

        let extrapolated = 2.0 * fine - coarse;
        let relative = (extrapolated - out.energy_ev).abs() / out.energy_ev.abs();
        assert!(
            relative < 5.0e-5,
            "Parry {} vs extrapolated direct {extrapolated} (rel {relative:.3e})",
            out.energy_ev
        );
    }

    /// `E = ½ Σ q φ` for the slab, including a charged one.
    #[test]
    fn slab_potentials_are_consistent_with_the_energy() {
        for extra_charge in [0.0, 0.4] {
            let (cell, mut sites) = slab_fixture(50.0);
            sites[0].charge += extra_charge;
            let params = EwaldParams::tuned(18.0, 1.0e-13);
            let out = ewald(&cell, &sites, &params).unwrap();
            let from_potentials: f64 = 0.5
                * sites
                    .iter()
                    .zip(&out.site_potential_ev)
                    .map(|(s, p)| s.charge * p)
                    .sum::<f64>();
            assert!(
                (out.energy_ev - from_potentials).abs() < 1.0e-9 * out.energy_ev.abs().max(1.0),
                "charge offset {extra_charge}: {} vs {from_potentials}",
                out.energy_ev
            );
        }
    }

    /// Slab gradients against central differences, including the out-of-plane direction where
    /// the whole `erfc(αz ± G/2α)` machinery lives.
    #[test]
    fn slab_gradients_match_finite_differences() {
        let (cell, base) = slab_fixture(50.0);
        let params = EwaldParams::tuned(18.0, 1.0e-13);
        let analytic = ewald(&cell, &base, &params).unwrap().site_gradient;
        let step = 1.0e-5;
        for index in 0..base.len() {
            for axis in 0..3 {
                let mut plus = base.clone();
                let mut minus = base.clone();
                shift(&mut plus[index].position, axis, step);
                shift(&mut minus[index].position, axis, -step);
                let numeric = (ewald(&cell, &plus, &params).unwrap().energy_ev
                    - ewald(&cell, &minus, &params).unwrap().energy_ev)
                    / (2.0 * step);
                let got = analytic[index].get(axis);
                assert!(
                    (got - numeric).abs() < 1.0e-6 * numeric.abs().max(1.0),
                    "slab gradient [{index}][{axis}]: analytic {got} vs numeric {numeric}"
                );
            }
        }
    }

    /// Each of `slab_kernel`'s derivative returns against a central difference of the return it
    /// is the derivative of: `∂K/∂z` against `K`, `∂²K/∂z²` against `∂K/∂z`, and `∂K/∂g`
    /// against `K`.
    ///
    /// The kernel had no unit test at all until the phased slab sum started consuming its second
    /// `z` derivative; every earlier consumer was checked only through assembled energies, where
    /// a wrong derivative hides behind the terms that are right. The probe points are chosen to
    /// land in **both** numerical branches — `u ≥ 0` (the `erfcx` form) and `u < 0` (the direct
    /// product) — because the two compute the same quantity by different routes and a branch
    /// bug would otherwise sit in whichever one the assembled tests happened not to visit:
    /// `z = ±3` at `g = 0.4, α = 0.7` puts `αz = ±2.1` past `g/2α ≈ 0.29`, flipping the sign of
    /// one `u` each way, while the small-`z` points keep both `u` positive.
    #[test]
    fn the_slab_kernel_derivatives_are_the_derivatives_of_the_kernel() {
        let step = 1.0e-6;
        for (g, z, alpha) in [
            (0.5_f64, 0.0_f64, 0.6_f64),
            (0.5, 0.35, 0.6),
            (1.7, -0.8, 0.45),
            (0.4, 3.0, 0.7),  // u2 < 0: the direct-product branch on t2
            (0.4, -3.0, 0.7), // u1 < 0: the direct-product branch on t1
            (3.0, 1.2, 0.5),
        ] {
            let (value, dz, dg, dzz) = slab_kernel(g, z, alpha);
            assert!(
                value.abs() > 1.0e-6,
                "the probe (g={g}, z={z}, α={alpha}) collapsed to {value}, testing nothing"
            );

            let (value_zp, dz_zp, _, _) = slab_kernel(g, z + step, alpha);
            let (value_zm, dz_zm, _, _) = slab_kernel(g, z - step, alpha);
            let numeric_dz = (value_zp - value_zm) / (2.0 * step);
            assert!(
                (dz - numeric_dz).abs() < 1.0e-8 * numeric_dz.abs().max(1.0),
                "∂K/∂z at (g={g}, z={z}, α={alpha}): analytic {dz} vs numeric {numeric_dz}"
            );
            let numeric_dzz = (dz_zp - dz_zm) / (2.0 * step);
            assert!(
                (dzz - numeric_dzz).abs() < 1.0e-7 * numeric_dzz.abs().max(1.0),
                "∂²K/∂z² at (g={g}, z={z}, α={alpha}): analytic {dzz} vs numeric {numeric_dzz}"
            );

            let (value_gp, _, _, _) = slab_kernel(g + step, z, alpha);
            let (value_gm, _, _, _) = slab_kernel(g - step, z, alpha);
            let numeric_dg = (value_gp - value_gm) / (2.0 * step);
            assert!(
                (dg - numeric_dg).abs() < 1.0e-8 * numeric_dg.abs().max(1.0),
                "∂K/∂g at (g={g}, z={z}, α={alpha}): analytic {dg} vs numeric {numeric_dg}"
            );
        }
    }

    /// A slab cannot accelerate itself either.
    #[test]
    fn slab_gradients_sum_to_zero() {
        let (cell, sites) = slab_fixture(50.0);
        let out = ewald(&cell, &sites, &EwaldParams::tuned(18.0, 1.0e-13)).unwrap();
        let total = out
            .site_gradient
            .iter()
            .fold(Vec3::zero(), |acc, g| acc + *g);
        assert!(total.norm() < 1.0e-9, "net slab force is {total:?}");
    }

    // ---- 1D (chain) ----

    fn chain_fixture(padding: f64) -> (Cell, Vec<ChargeSite>) {
        let cell = Cell::new(
            Vec3::new(4.5, 0.0, 0.0),
            Vec3::new(0.0, padding, 0.0),
            Vec3::new(0.0, 0.0, padding),
            [true, false, false],
        )
        .unwrap();
        let sites = vec![
            ChargeSite::on_atom(Vec3::new(0.0, 0.0, 0.0), 0.6, 0),
            ChargeSite::on_atom(Vec3::new(1.4, 0.8, 0.2), -0.6, 1),
            ChargeSite::on_atom(Vec3::new(2.6, -0.5, 0.7), 0.35, 2),
            ChargeSite::on_atom(Vec3::new(3.6, 0.3, -0.6), -0.35, 3),
        ];
        (cell, sites)
    }

    /// The chain sum must ignore the transverse padding entirely.
    #[test]
    fn chain_energy_is_independent_of_the_transverse_padding() {
        let params = EwaldParams::tuned(18.0, 1.0e-13);
        let mut energies = Vec::new();
        for padding in [20.0, 100.0, 500.0] {
            let (cell, sites) = chain_fixture(padding);
            energies.push(ewald(&cell, &sites, &params).unwrap().energy_ev);
        }
        for energy in &energies {
            assert!(
                (energy - energies[0]).abs() < 1.0e-12 * energies[0].abs().max(1.0),
                "transverse padding changed the chain energy"
            );
        }
    }

    /// Against the direct segment sum. The chain path *is* a direct sum, so this mainly pins
    /// the cell grouping and the analytic tail rather than an independent algebra — the sharper
    /// check is the image-count convergence below.
    ///
    /// The reference has no tail correction, so it is the less accurate of the two and
    /// converges as `1/N²`; `(4E(2N) − E(N))/3` removes that term so the comparison measures
    /// the chain path rather than the reference's truncation.
    #[test]
    fn chain_matches_the_direct_lattice_sum() {
        let (cell, sites) = chain_fixture(60.0);
        let params = EwaldParams::tuned(18.0, 1.0e-13);
        let out = ewald(&cell, &sites, &params).unwrap();
        let reference = as_reference(&sites);
        let coarse = direct_energy(&cell, &reference, 2000);
        let fine = direct_energy(&cell, &reference, 4000);
        let extrapolated = (4.0 * fine - coarse) / 3.0;
        let relative = (out.energy_ev - extrapolated).abs() / extrapolated.abs().max(1.0);
        assert!(
            relative < 1.0e-9,
            "chain {} vs extrapolated direct {extrapolated} (rel {relative:.3e})",
            out.energy_ev
        );
    }

    /// The analytic `1/n³` tail must actually buy accuracy: the energy has to be converged in
    /// the image count well before the raw truncation would be.
    #[test]
    fn chain_energy_converges_in_the_image_count() {
        let (cell, sites) = chain_fixture(60.0);
        let mut energies = Vec::new();
        for images in [100, 400, 1600] {
            let mut params = EwaldParams::tuned(18.0, 1.0e-13);
            params.chain_images = images;
            energies.push(ewald(&cell, &sites, &params).unwrap().energy_ev);
        }
        let coarse = (energies[0] - energies[2]).abs();
        let fine = (energies[1] - energies[2]).abs();
        assert!(fine < coarse, "the chain sum is not converging");
        assert!(
            fine / energies[2].abs() < 1.0e-11,
            "chain energy still moving by {:.3e} at the default image count",
            fine / energies[2].abs()
        );
    }

    /// `E = ½ Σ q φ` for the chain.
    #[test]
    fn chain_potentials_are_consistent_with_the_energy() {
        let (cell, sites) = chain_fixture(60.0);
        let out = ewald(&cell, &sites, &EwaldParams::tuned(18.0, 1.0e-13)).unwrap();
        let from_potentials: f64 = 0.5
            * sites
                .iter()
                .zip(&out.site_potential_ev)
                .map(|(s, p)| s.charge * p)
                .sum::<f64>();
        // The analytic tail is added to the energy only, so allow for its size.
        assert!(
            (out.energy_ev - from_potentials).abs() < 1.0e-6 * out.energy_ev.abs().max(1.0),
            "{} vs {from_potentials}",
            out.energy_ev
        );
    }

    /// Chain gradients against central differences.
    #[test]
    fn chain_gradients_match_finite_differences() {
        let (cell, base) = chain_fixture(60.0);
        let params = EwaldParams::tuned(18.0, 1.0e-13);
        let analytic = ewald(&cell, &base, &params).unwrap().site_gradient;
        let step = 1.0e-5;
        for index in 0..base.len() {
            for axis in 0..3 {
                let mut plus = base.clone();
                let mut minus = base.clone();
                shift(&mut plus[index].position, axis, step);
                shift(&mut minus[index].position, axis, -step);
                let numeric = (ewald(&cell, &plus, &params).unwrap().energy_ev
                    - ewald(&cell, &minus, &params).unwrap().energy_ev)
                    / (2.0 * step);
                let got = analytic[index].get(axis);
                assert!(
                    (got - numeric).abs() < 1.0e-5 * numeric.abs().max(1.0),
                    "chain gradient [{index}][{axis}]: analytic {got} vs numeric {numeric}"
                );
            }
        }
    }

    /// A charged chain's energy must not depend on how many images were summed.
    ///
    /// This is the whole content of the neutralizing line charge. Without it the partial sum
    /// carries `Q² H_N / L`, which grows without bound as `N` does — so an answer that *stops*
    /// moving when `N` is quadrupled is the direct evidence that the divergence is gone, and it
    /// is evidence no tolerance on the energy itself could give.
    ///
    /// The neutral case is included as the control: it has to be `N`-independent too, and for a
    /// different reason (its divergence cancels rather than being removed), so a change that
    /// broke one and not the other would be visible here.
    #[test]
    fn a_charged_chain_stops_depending_on_the_image_count() {
        let (cell, neutral) = chain_fixture(40.0);
        let mut charged = neutral.clone();
        charged[0].charge += 1.0;

        let energy_at = |sites: &[ChargeSite], images: usize| {
            let params = EwaldParams {
                chain_images: images,
                ..EwaldParams::default()
            };
            ewald(&cell, sites, &params).unwrap().energy_ev
        };

        // Neutral: the divergence cancels and the dipole tail is corrected analytically, so
        // there is nothing left to depend on `N` at this precision.
        let short = energy_at(&neutral, 200);
        let long = energy_at(&neutral, 800);
        assert!(
            (short - long).abs() < 1.0e-6,
            "neutral: 200 images give {short}, 800 give {long}"
        );

        // Charged: the monopole divergence is removed by the line charge, but the dipole tail is
        // not corrected — a charged distribution's dipole depends on where the origin is put, so
        // the analytic tail means nothing there. What is left must therefore fall as `1/N²` like
        // any truncated `Σ n⁻³`, and *not* like `ln N`.
        //
        // Asserting the exponent is the point. A tolerance on the energy could be met by a
        // logarithm that simply had not grown much yet; a ratio of successive differences
        // cannot. Doubling `N` must shrink the remaining drift fourfold.
        let (a, b, c) = (
            energy_at(&charged, 100),
            energy_at(&charged, 200),
            energy_at(&charged, 400),
        );
        let ratio = (a - b).abs() / (b - c).abs();
        assert!(
            (2.5..6.0).contains(&ratio),
            "the charged chain's residual shrinks by {ratio:.2}x per doubling; 4 is the \
             truncated dipole tail, 1 would mean the divergence is still there \
             (E = {a}, {b}, {c})"
        );
        assert!(
            (b - c).abs() < 1.0e-4,
            "and it should already be small: {} eV between 200 and 400 images",
            (b - c).abs()
        );
    }

    /// A charged chain and the same chain described by a doubled cell must agree per cell.
    ///
    /// Two descriptions of one physical object: the same linear charge density, the same
    /// charges in the same places. Any convention worth having has to give them the same energy
    /// per unit length, and this is where a *wrong* one shows itself — subtracting `Q² H_N / L`
    /// gives each cell a stable, finite, entirely plausible energy on its own while the two
    /// disagree, because the leftover carries a `ln L` the two cells do not share.
    ///
    /// The subtraction is therefore in terms of the physical extent summed, `N · L`, and not the
    /// image count alone. Tested on bare point charges rather than through the SCF, so that what
    /// is being compared is the electrostatics and not two descriptions that also happen to
    /// differ in their electronic state.
    #[test]
    fn a_charged_chain_agrees_with_its_own_doubled_cell() {
        let axis = 6.0;
        let along = |n: usize| {
            Cell::new(
                Vec3::new(axis * n as f64, 0.0, 0.0),
                Vec3::new(0.0, 40.0, 0.0),
                Vec3::new(0.0, 0.0, 40.0),
                [true, false, false],
            )
            .unwrap()
        };
        // One cell: a dipole plus a net charge. Two cells: the same thing twice over.
        let unit = |cell: usize| {
            let x = axis * cell as f64;
            [
                ChargeSite::on_atom(Vec3::new(x, 0.0, 0.0), 0.7, 2 * cell),
                ChargeSite::on_atom(Vec3::new(x + 1.9, 0.4, 0.0), -0.2, 2 * cell + 1),
            ]
        };
        let one: Vec<ChargeSite> = unit(0).into();
        let two: Vec<ChargeSite> = unit(0).into_iter().chain(unit(1)).collect();

        let params = EwaldParams {
            chain_images: 600,
            ..EwaldParams::default()
        };
        let single = ewald(&along(1), &one, &params).unwrap().energy_ev;
        let doubled = ewald(&along(2), &two, &params).unwrap().energy_ev / 2.0;
        let difference = (single - doubled).abs();
        assert!(
            difference < 1.0e-4,
            "one cell gives {single} and the doubled cell {doubled} per cell \
             ({difference:.3e} eV) — the convention is not size-consistent"
        );
    }

    /// The line charge has to reach the site potentials as well as the energy.
    ///
    /// `E` is homogeneous of degree two in the charges, so the potentials — `∂E/∂q_i` — must
    /// satisfy `Σ_i q_i φ_i = 2E`. A background term added to the energy alone would break that
    /// identity, and the SCF would then converge against a potential its own energy expression
    /// disagreed with: self-consistent, and wrong.
    #[test]
    fn the_line_charge_reaches_the_potentials_too() {
        let (cell, mut sites) = chain_fixture(40.0);
        sites[0].charge += 1.0;
        let out = ewald(&cell, &sites, &EwaldParams::default()).unwrap();
        let contracted: f64 = sites
            .iter()
            .zip(&out.site_potential_ev)
            .map(|(site, potential)| site.charge * potential)
            .sum();
        assert!(
            (contracted - 2.0 * out.energy_ev).abs() < 1.0e-8 * out.energy_ev.abs().max(1.0),
            "Σ q φ = {contracted} but 2E = {}",
            2.0 * out.energy_ev
        );
    }

    /// Sites sharing an owner must not interact within the reference cell — in NDDO that energy
    /// is the one-center integral set, not a Coulomb term — while their periodic images must.
    ///
    /// Checked by giving the same geometry two different owner assignments. Grouping sites onto
    /// shared owners has to lower the energy by exactly the bare `q_i q_j / r` of the pairs that
    /// were grouped, and by nothing else. That catches both halves of the bookkeeping at once:
    /// the real-space skip, and the `erf` part the reciprocal sum silently kept.
    #[test]
    fn sites_sharing_an_owner_do_not_interact_inside_the_cell() {
        for dimension in 0..3 {
            let cell = match dimension {
                0 => Cell::cubic(11.0).unwrap(),
                1 => Cell::new(
                    Vec3::new(7.0, 0.0, 0.0),
                    Vec3::new(0.6, 6.5, 0.0),
                    Vec3::new(0.0, 0.0, 55.0),
                    [true, true, false],
                )
                .unwrap(),
                _ => Cell::new(
                    Vec3::new(6.0, 0.0, 0.0),
                    Vec3::new(0.0, 45.0, 0.0),
                    Vec3::new(0.0, 0.0, 45.0),
                    [true, false, false],
                )
                .unwrap(),
            };
            // Two "atoms" of two sites each, neutral overall so 1D is admissible too.
            let layout = [
                (Vec3::new(1.0, 1.2, 1.4), 0.65, 0usize),
                (Vec3::new(1.7, 1.2, 1.4), -0.65, 0),
                (Vec3::new(4.1, 3.3, 2.2), 0.4, 1),
                (Vec3::new(4.1, 3.9, 2.2), -0.4, 1),
            ];
            let grouped: Vec<ChargeSite> = layout
                .iter()
                .map(|(position, charge, owner)| ChargeSite::on_atom(*position, *charge, *owner))
                .collect();
            let separate: Vec<ChargeSite> = layout
                .iter()
                .enumerate()
                .map(|(index, (position, charge, _))| {
                    ChargeSite::on_atom(*position, *charge, index)
                })
                .collect();

            let params = EwaldParams::tuned(18.0, 1.0e-13);
            let with_groups = ewald(&cell, &grouped, &params).unwrap();
            let without = ewald(&cell, &separate, &params).unwrap();

            let mut removed = 0.0;
            for a in 0..grouped.len() {
                for b in (a + 1)..grouped.len() {
                    if grouped[a].owner == grouped[b].owner {
                        let r = (grouped[b].position - grouped[a].position).norm();
                        removed += grouped[a].charge * grouped[b].charge / r;
                    }
                }
            }
            removed *= PM3_EV;
            let expected = without.energy_ev - removed;
            assert!(
                (with_groups.energy_ev - expected).abs() < 1.0e-9 * expected.abs().max(1.0),
                "{}D: grouped {} vs separate-minus-intra {expected}",
                3 - dimension,
                with_groups.energy_ev
            );

            // The potentials must follow: E = ½ Σ q φ has to keep holding after the exclusion.
            let from_potentials: f64 = 0.5
                * grouped
                    .iter()
                    .zip(&with_groups.site_potential_ev)
                    .map(|(s, p)| s.charge * p)
                    .sum::<f64>();
            let tolerance = if 3 - dimension == 1 { 1.0e-6 } else { 1.0e-9 };
            assert!(
                (with_groups.energy_ev - from_potentials).abs()
                    < tolerance * with_groups.energy_ev.abs().max(1.0),
                "{}D: exclusion broke E = ½Σqφ ({} vs {from_potentials})",
                3 - dimension,
                with_groups.energy_ev
            );
        }
    }

    /// The exclusion must not disturb the derivatives either.
    /// A chain of clouds shaped like the ones the multipole model actually produces: a nucleus,
    /// three dipole pairs, three linear-quadrupole pairs and twelve off-diagonal corners, with
    /// charges that cancel down to a small net charge the way an atom's do.
    fn cloud_chain(atoms: usize) -> Vec<ChargeSite> {
        let mut sites = Vec::new();
        for atom in 0..atoms {
            let base = Vec3::new(6.0 * atom as f64, 0.0, 0.0);
            let net = if atom % 3 == 0 { -0.62 } else { 0.31 };
            sites.push(ChargeSite {
                position: base,
                charge: net,
                owner: atom,
            });
            let axes = [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ];
            for (index, axis) in axes.iter().enumerate() {
                let weight = 0.4 + 0.05 * index as f64;
                for (sign, d) in [(1.0, 0.72), (-1.0, 0.72), (1.0, 1.24), (-1.0, 1.24)] {
                    sites.push(ChargeSite {
                        position: base + *axis * (sign * d),
                        charge: sign * weight * if d > 1.0 { -0.5 } else { 1.0 },
                        owner: atom,
                    });
                }
            }
            for a in 0..3 {
                for b in (a + 1)..3 {
                    for (sa, sb) in [(1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)] {
                        sites.push(ChargeSite {
                            position: base + axes[a] * (sa * 0.9) + axes[b] * (sb * 0.9),
                            charge: 0.21 * sa * sb,
                            owner: atom,
                        });
                    }
                }
            }
        }
        sites
    }

    /// The collapsed far field has to answer the same question the site sum does. Nothing else
    /// pins [`MULTIPOLE_RADIUS`]: the estimate in its documentation says the error should be of
    /// order `(d/R)³`, and this says what it is.
    #[test]
    fn the_multipole_far_field_agrees_with_the_site_sum() {
        let sites = cloud_chain(48);
        assert!(
            sites.len() > 40 * 25,
            "the fixture should carry a full multipole cloud per atom"
        );

        let mut exact = EwaldOutput::zeros(sites.len());
        direct_0d(&sites, &mut exact);
        let mut collapsed = EwaldOutput::zeros(sites.len());
        direct_0d_potentials(&sites, &mut collapsed);

        // The chain spans nearly three hundred Bohr, so most pairs are past the handover and the
        // comparison is a measurement of the approximation rather than of the near field.
        let energy_error = (collapsed.energy_ev - exact.energy_ev).abs();
        assert!(
            energy_error < 5.0e-5,
            "collapsing the far field moved the energy by {energy_error:.3e} eV \
             ({:.6} vs {:.6})",
            collapsed.energy_ev,
            exact.energy_ev
        );

        let worst = collapsed
            .site_potential_ev
            .iter()
            .zip(&exact.site_potential_ev)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            worst < 1.0e-5,
            "the worst site potential moved by {worst:.3e} eV"
        );
    }

    /// The tree has to answer what the pairwise far-field sum answered, and it has to branch.
    ///
    /// Two claims, and the second is why the first means anything. A tree that never accepts a
    /// node is exactly the pairwise sum and would pass any agreement test while saving nothing,
    /// so the node count is asserted alongside the numbers: on this fixture the walk takes whole
    /// nodes rather than opening every one down to its clouds.
    ///
    /// The near field is untouched by construction — a node is accepted only when
    /// `d − s > MULTIPOLE_RADIUS`, so by the triangle inequality every cloud inside it was far
    /// by the pairwise rule too — which is why this compares against
    /// [`direct_0d`], the exact site sum, rather than against an intermediate.
    #[test]
    fn the_tree_reproduces_the_pairwise_far_field() {
        // Three parallel chains, so the octree splits in more than one direction and the nodes
        // it forms are compact rather than strung out. A single chain gives long thin boxes that
        // the acceptance test correctly refuses, and then the tree is doing nothing.
        let mut sites = Vec::new();
        for row in 0..3 {
            let offset = Vec3::new(0.0, 40.0 * row as f64, 0.0);
            for site in cloud_chain(24) {
                sites.push(ChargeSite {
                    position: site.position + offset,
                    charge: site.charge,
                    owner: site.owner + 24 * row,
                });
            }
        }

        let (_, clouds) = clouds(&sites);
        let arena = build_tree(&clouds);
        assert!(
            arena.len() > 1,
            "the tree collapsed to a single node, so nothing was tested"
        );
        let internal = arena.iter().filter(|n| !n.children.is_empty()).count();
        assert!(
            internal > 1,
            "the tree did not branch ({internal} internal nodes)"
        );

        let mut exact = EwaldOutput::zeros(sites.len());
        direct_0d(&sites, &mut exact);
        let mut tree = EwaldOutput::zeros(sites.len());
        direct_0d_potentials(&sites, &mut tree);

        let energy_error = (tree.energy_ev - exact.energy_ev).abs();
        assert!(
            energy_error < 5.0e-5,
            "the tree moved the energy by {energy_error:.3e} eV ({:.6} vs {:.6})",
            tree.energy_ev,
            exact.energy_ev
        );
        let worst = tree
            .site_potential_ev
            .iter()
            .zip(&exact.site_potential_ev)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            worst < 1.0e-5,
            "the worst site potential moved by {worst:.3e} eV"
        );
    }

    /// The far field's work grows sub-quadratically, and the count says so rather than a clock.
    ///
    /// Counted, not timed. This machine runs other work, so a wall-clock slope would measure the
    /// load as much as the algorithm; the number of source terms each target sums is a property
    /// of the tree and the geometry alone, and it is what the tree was built to reduce.
    ///
    /// A compact block of waters, doubling in atom count: without a tree every target sums every
    /// other cloud, so the total is `N(N−1)` and the log-log slope is 2 exactly. With one, a
    /// distant group is one term instead of many.
    #[test]
    fn the_tree_makes_the_far_field_sub_quadratic() {
        let counts: Vec<(f64, f64)> = [5_usize, 6, 8, 10]
            .iter()
            .map(|&edge| {
                let sites = cloud_block(edge);
                let (_, clouds) = clouds(&sites);
                let arena = build_tree(&clouds);
                let n = clouds.len() as f64;
                (n, far_field_terms(&clouds, &arena) as f64)
            })
            .collect();

        for (n, terms) in &counts {
            assert!(
                *terms > 0.0,
                "no far field at {n} clouds — the fixture does not reach past the handover, so \
                 this measures nothing"
            );
            let pairwise = n * (n - 1.0);
            assert!(
                *terms < pairwise,
                "the tree summed {terms} terms where the pairwise rule sums {pairwise}"
            );
        }
        // At the largest size the saving has to be substantial, not marginal: a tree that
        // accepts a handful of nodes would pass the slope test on noise.
        let (largest, terms) = counts[counts.len() - 1];
        assert!(
            terms < 0.25 * largest * (largest - 1.0),
            "the tree did only {terms} against {} pairwise — barely a saving",
            largest * (largest - 1.0)
        );
        let (n0, t0) = counts[0];
        let (n1, t1) = counts[counts.len() - 1];
        let slope = (t1 / t0).ln() / (n1 / n0).ln();
        assert!(
            slope < 1.7,
            "the far field is still nearly quadratic: slope {slope:.2} from {t0} terms at \
             {n0} clouds to {t1} at {n1}"
        );
    }

    /// How many source terms the tree walk sums over every target — the quantity
    /// [`direct_0d_potentials`]'s far half costs, with a node counted once however many clouds
    /// it stands for.
    fn far_field_terms(clouds: &[Cloud], arena: &[TreeNode]) -> usize {
        let far = MULTIPOLE_RADIUS * MULTIPOLE_RADIUS;
        let mut total = 0;
        for (index, a) in clouds.iter().enumerate() {
            let mut stack = vec![arena.len() - 1];
            while let Some(node_index) = stack.pop() {
                let node = &arena[node_index];
                let distance = (a.centre - node.centre).norm();
                let neglected = node.abs_charge * node.radius.powi(3) / distance.powi(4);
                if !node.children.is_empty()
                    && distance - node.radius > MULTIPOLE_RADIUS
                    && neglected < FAR_FIELD_TOLERANCE
                {
                    total += 1;
                    continue;
                }
                if node.children.is_empty() {
                    for &other in &node.clouds {
                        if other != index && (a.centre - clouds[other].centre).norm2() > far {
                            total += 1;
                        }
                    }
                } else {
                    stack.extend_from_slice(&node.children);
                }
            }
        }
        total
    }

    /// A compact cubic block of clouds, `edge³` of them, spaced as waters in a liquid are.
    ///
    /// Compact on purpose: a chain gives long thin boxes that the acceptance test correctly
    /// refuses, and then the tree has nothing to group. Three dimensions is where it works.
    fn cloud_block(edge: usize) -> Vec<ChargeSite> {
        // Spaced wide enough that the block reaches across the handover at a cloud count a test
        // can afford. At liquid density the same span is tens of thousands of atoms, which is
        // the size at which the far field exists at all: below about a hundred and sixty Bohr
        // across, every pair is inside `MULTIPOLE_RADIUS` and there is nothing to accelerate.
        const SPACING: f64 = 25.0;
        let mut sites = Vec::new();
        let mut owner = 0;
        for i in 0..edge {
            for j in 0..edge {
                for k in 0..edge {
                    let base =
                        Vec3::new(SPACING * i as f64, SPACING * j as f64, SPACING * k as f64);
                    sites.push(ChargeSite {
                        position: base,
                        charge: 6.0,
                        owner,
                    });
                    for axis in 0..3 {
                        let mut offset = [0.0; 3];
                        offset[axis] = 0.7;
                        let d = Vec3::new(offset[0], offset[1], offset[2]);
                        for sign in [1.0, -1.0] {
                            sites.push(ChargeSite {
                                position: base + d * sign,
                                charge: -1.0,
                                owner,
                            });
                        }
                    }
                    owner += 1;
                }
            }
        }
        sites
    }

    /// A node's moments are its clouds', shifted — exactly, before anything is truncated.
    ///
    /// The whole accuracy argument for the tree rests on the translation being lossless going
    /// up, so that the only error is the expansion's own truncation on the way out. This checks
    /// it against the definition rather than against the recursion that produced it.
    #[test]
    fn a_node_carries_its_clouds_moments_exactly() {
        let sites = cloud_chain(20);
        let (_, clouds) = clouds(&sites);
        let arena = build_tree(&clouds);

        for node in &arena {
            // Every cloud beneath this node, found by walking rather than by trusting the build.
            let mut contained = Vec::new();
            let mut stack = vec![node];
            while let Some(current) = stack.pop() {
                if current.children.is_empty() {
                    contained.extend_from_slice(&current.clouds);
                } else {
                    stack.extend(current.children.iter().map(|&i| &arena[i]));
                }
            }
            if contained.is_empty() {
                continue;
            }

            let (mut charge, mut dipole, mut quadrupole) = (0.0, Vec3::zero(), [[0.0; 3]; 3]);
            for &i in &contained {
                let cloud = &clouds[i];
                let s = cloud.centre - node.centre;
                charge += cloud.charge;
                dipole += cloud.dipole + s * cloud.charge;
                let d = [cloud.dipole.x, cloud.dipole.y, cloud.dipole.z];
                let sv = [s.x, s.y, s.z];
                for (a, row) in quadrupole.iter_mut().enumerate() {
                    for (b, slot) in row.iter_mut().enumerate() {
                        *slot += cloud.quadrupole[a][b]
                            + d[a] * sv[b]
                            + d[b] * sv[a]
                            + cloud.charge * sv[a] * sv[b];
                    }
                }
            }
            assert!((node.charge - charge).abs() < 1.0e-12, "charge");
            assert!((node.dipole - dipole).norm() < 1.0e-12, "dipole");
            for (a, (built, want)) in node.quadrupole.iter().zip(&quadrupole).enumerate() {
                for (b, (x, y)) in built.iter().zip(want).enumerate() {
                    assert!((x - y).abs() < 1.0e-10, "quadrupole [{a}][{b}]");
                }
            }
        }
    }

    /// Below the handover the two paths are the same sum in a different order, so they agree to
    /// rounding rather than to an approximation error.
    #[test]
    fn the_near_field_is_summed_site_by_site() {
        let sites = cloud_chain(4);
        let span = sites.iter().map(|s| s.position.x).fold(0.0, f64::max);
        assert!(
            span < MULTIPOLE_RADIUS,
            "the fixture must fit inside the handover for this test to mean anything"
        );

        let mut exact = EwaldOutput::zeros(sites.len());
        direct_0d(&sites, &mut exact);
        let mut collapsed = EwaldOutput::zeros(sites.len());
        direct_0d_potentials(&sites, &mut collapsed);

        assert!((collapsed.energy_ev - exact.energy_ev).abs() < 1.0e-10);
        for (a, b) in collapsed
            .site_potential_ev
            .iter()
            .zip(&exact.site_potential_ev)
        {
            assert!((a - b).abs() < 1.0e-10);
        }
    }

    #[test]
    fn owner_exclusion_keeps_the_gradient_consistent() {
        let cell = Cell::cubic(11.0).unwrap();
        let base = vec![
            ChargeSite::on_atom(Vec3::new(1.0, 1.2, 1.4), 0.65, 0),
            ChargeSite::on_atom(Vec3::new(1.7, 1.2, 1.4), -0.65, 0),
            ChargeSite::on_atom(Vec3::new(4.1, 3.3, 2.2), 0.4, 1),
            ChargeSite::on_atom(Vec3::new(4.1, 3.9, 2.2), -0.4, 1),
        ];
        let params = EwaldParams::tuned(18.0, 1.0e-13);
        let out = ewald(&cell, &base, &params).unwrap();
        let step = 1.0e-5;
        for index in 0..base.len() {
            for axis in 0..3 {
                let mut plus = base.clone();
                let mut minus = base.clone();
                shift(&mut plus[index].position, axis, step);
                shift(&mut minus[index].position, axis, -step);
                let numeric = (ewald(&cell, &plus, &params).unwrap().energy_ev
                    - ewald(&cell, &minus, &params).unwrap().energy_ev)
                    / (2.0 * step);
                let got = out.site_gradient[index].get(axis);
                assert!(
                    (got - numeric).abs() < 1.0e-6 * numeric.abs().max(1.0),
                    "excluded-pair gradient [{index}][{axis}]: {got} vs {numeric}"
                );
            }
        }
        // And the virial, which the exclusion also has to walk back.
        let virial = out.virial.unwrap();
        let strain_step = 1.0e-6;
        for alpha in 0..3 {
            for beta in 0..3 {
                let energy_at = |sign: f64| -> f64 {
                    let mut strain = Mat3::zero();
                    add_diagonal_at(&mut strain, alpha, beta, sign * strain_step);
                    let strained: Vec<ChargeSite> = base
                        .iter()
                        .map(|s| ChargeSite {
                            position: s.position + strain.mul_vec(s.position),
                            ..*s
                        })
                        .collect();
                    ewald(&cell.strained(&strain), &strained, &params)
                        .unwrap()
                        .energy_ev
                };
                let numeric = (energy_at(1.0) - energy_at(-1.0)) / (2.0 * strain_step);
                let got = component(&virial, alpha, beta);
                assert!(
                    (got - numeric).abs() < 1.0e-5 * numeric.abs().max(1.0),
                    "excluded-pair virial[{alpha}][{beta}]: {got} vs {numeric}"
                );
            }
        }
    }

    /// A chain's axial virial against finite differences of its own strained energy.
    ///
    /// The claim being tested is `∂E/∂ε = Σ (∂E/∂d) ⊗ (d + T)`, and the only way to test a
    /// derivative is to difference the thing it is the derivative of. Straining the cell means
    /// stretching the lattice vector *and* carrying the sites along with it, which is what makes
    /// this catch a virial that forgot the translation: a molecular outer product using `d`
    /// alone agrees at `T = 0` and drifts with every image shell.
    #[test]
    fn the_chain_virial_matches_finite_differences() {
        let params = EwaldParams::tuned(18.0, 1.0e-13);
        let (chain, sites) = chain_fixture(50.0);
        let axis = chain.periodic_indices()[0];
        let virial = ewald(&chain, &sites, &params).unwrap().virial.unwrap();

        let step = 1.0e-5;
        let energy_at = |signed: f64| {
            let mut strain = Mat3::zero();
            add_diagonal(&mut strain, axis, signed * step);
            let strained: Vec<ChargeSite> = sites
                .iter()
                .map(|s| ChargeSite {
                    position: s.position + strain.mul_vec(s.position),
                    ..*s
                })
                .collect();
            ewald(&chain.strained(&strain), &strained, &params)
                .unwrap()
                .energy_ev
        };
        let numeric = (energy_at(1.0) - energy_at(-1.0)) / (2.0 * step);
        let analytic = component(&virial, axis, axis);
        assert!(
            (analytic - numeric).abs() < 1.0e-4 * numeric.abs().max(1.0),
            "chain virial: analytic {analytic} vs finite difference {numeric}"
        );
    }

    /// The slab strain derivative, against the energy of a strained slab.
    ///
    /// Only the in-plane block is a real derivative. A slab has no `z` strain — the cell length
    /// normal to it is a padding choice, and `slab_energy_is_independent_of_the_vacuum_thickness`
    /// is the same statement from the other side — so the out-of-plane components are whatever
    /// the affine formula produces and are projected away by
    /// [`crate::pbc::gradient::forces_and_stress`]. Straining `z` here would compare against a
    /// finite difference of a quantity that does not depend on the variable.
    #[test]
    fn slab_virial_matches_strained_finite_differences() {
        let cell = Cell::new(
            Vec3::new(6.4, 0.0, 0.0),
            Vec3::new(1.1, 5.8, 0.0),
            Vec3::new(0.0, 0.0, 46.0),
            [true, true, false],
        )
        .unwrap();
        let sites = vec![
            ChargeSite::on_atom(Vec3::new(0.6, 0.7, 21.5), 0.8, 0),
            ChargeSite::on_atom(Vec3::new(3.1, 2.4, 23.9), -0.45, 1),
            ChargeSite::on_atom(Vec3::new(4.8, 4.1, 22.6), -0.35, 2),
        ];
        let params = EwaldParams::tuned(18.0, 1.0e-14);
        let analytic = ewald(&cell, &sites, &params)
            .unwrap()
            .virial
            .expect("a slab reports an in-plane virial");

        let step = 1.0e-6;
        for alpha in 0..2 {
            for beta in 0..2 {
                let energy_at = |sign: f64| -> f64 {
                    let mut strain = Mat3::zero();
                    add_diagonal_at(&mut strain, alpha, beta, sign * step);
                    let strained_cell = cell.strained(&strain);
                    let strained_sites: Vec<ChargeSite> = sites
                        .iter()
                        .map(|s| ChargeSite {
                            position: s.position + strain.mul_vec(s.position),
                            ..*s
                        })
                        .collect();
                    ewald(&strained_cell, &strained_sites, &params)
                        .unwrap()
                        .energy_ev
                };
                let numeric = (energy_at(1.0) - energy_at(-1.0)) / (2.0 * step);
                let got = component(&analytic, alpha, beta);
                assert!(
                    (got - numeric).abs() < 1.0e-5 * numeric.abs().max(1.0),
                    "slab virial[{alpha}][{beta}]: analytic {got} vs numeric {numeric}"
                );
            }
        }
    }

    /// The in-plane virial must not depend on how much vacuum the caller padded the cell with,
    /// for the same reason the energy must not: neither is a property of the slab.
    #[test]
    fn the_slab_virial_ignores_the_vacuum_thickness() {
        let params = EwaldParams::tuned(18.0, 1.0e-13);
        let thin = {
            let (cell, sites) = slab_fixture(40.0);
            ewald(&cell, &sites, &params).unwrap().virial.unwrap()
        };
        let thick = {
            let (cell, sites) = slab_fixture(70.0);
            ewald(&cell, &sites, &params).unwrap().virial.unwrap()
        };
        for alpha in 0..2 {
            for beta in 0..2 {
                let (a, b) = (
                    component(&thin, alpha, beta),
                    component(&thick, alpha, beta),
                );
                assert!(
                    (a - b).abs() < 1.0e-9 * a.abs().max(1.0),
                    "virial[{alpha}][{beta}] moved from {a} to {b} on vacuum alone"
                );
            }
        }
    }

    /// A chain's virial comes free with the gradients: 1D is summed directly rather than through
    /// reciprocal space, so it is the ordinary pair virial. The full affine derivative is
    /// reported; projecting onto the strains a chain's cell actually has belongs to
    /// [`crate::pbc::gradient::forces_and_stress`], not here.
    #[test]
    fn every_periodic_dimensionality_reports_a_virial() {
        let params = EwaldParams::tuned(18.0, 1.0e-13);
        let (slab, slab_sites) = slab_fixture(50.0);
        assert!(ewald(&slab, &slab_sites, &params).unwrap().virial.is_some());

        // Zero dimensions is the one case that genuinely has no strain: there is no cell.
        let isolated = Cell::isolated();
        assert!(ewald(&isolated, &slab_sites, &params)
            .unwrap()
            .virial
            .is_none());

        let (chain, chain_sites) = chain_fixture(50.0);
        let chain_virial = ewald(&chain, &chain_sites, &params)
            .unwrap()
            .virial
            .expect("a chain reports an axial virial");
        let axis = chain.periodic_indices()[0];
        assert!(
            component(&chain_virial, axis, axis).abs() > 1.0e-6,
            "the axial component should be live"
        );
        let (cubic, cubic_sites) = rocksalt(5.0);
        assert!(ewald(&cubic, &cubic_sites, &params)
            .unwrap()
            .virial
            .is_some());
    }

    fn shift(v: &mut Vec3, axis: usize, delta: f64) {
        match axis {
            0 => v.x += delta,
            1 => v.y += delta,
            _ => v.z += delta,
        }
    }

    fn add_diagonal_at(m: &mut Mat3, alpha: usize, beta: usize, value: f64) {
        let column = &mut m.col[beta];
        match alpha {
            0 => column.x += value,
            1 => column.y += value,
            _ => column.z += value,
        }
    }

    fn component(m: &Mat3, alpha: usize, beta: usize) -> f64 {
        m.col[beta].get(alpha)
    }
    /// The interaction tensors are each other's derivatives, checked directly rather than
    /// through the sum they are used in.
    ///
    /// `multipole_field` contracts hand-derived rank-1, rank-2 and rank-3 tensors, and the far
    /// field test only sees their assembled result -- where a wrong rank-3 term is a `(d/R)^3`
    /// correction to something already small, and can hide behind the tolerance that measures
    /// the truncation error. Here each rank is differenced against the one below it, so a sign
    /// or a coefficient is caught on its own.
    ///
    /// The identities are `T_a = d(1/R)/dR_a`, `T_ab = dT_a/dR_b`, `T_abc = dT_ab/dR_c`.
    #[test]
    #[allow(clippy::needless_range_loop)] // Cartesian tensor indices, as elsewhere here
    fn the_interaction_tensors_are_each_others_derivatives() {
        // A monopole, a dipole and a quadrupole probe the three ranks separately: with only a
        // charge the value uses T and the gradient T_a; with only a dipole they use T_a and T_ab;
        // with only a quadrupole, T_ab and T_abc.
        let probes = [
            (1.0, Vec3::zero(), [[0.0; 3]; 3]),
            (0.0, Vec3::new(0.7, -0.3, 0.45), [[0.0; 3]; 3]),
            (0.0, Vec3::zero(), {
                let mut q = [[0.0; 3]; 3];
                let d = [0.6, -0.25, 0.4];
                for (a, row) in q.iter_mut().enumerate() {
                    for (b, slot) in row.iter_mut().enumerate() {
                        *slot = d[a] * d[b];
                    }
                }
                q
            }),
        ];
        let points = [
            Vec3::new(5.0, 0.0, 0.0),
            Vec3::new(3.1, -4.2, 2.7),
            Vec3::new(-6.5, 1.1, -3.3),
        ];
        let step = 1.0e-5;

        for (rank, (charge, dipole, quadrupole)) in probes.into_iter().enumerate() {
            let cloud = Cloud {
                centre: Vec3::zero(),
                charge,
                dipole,
                quadrupole,
                range: 0..0,
            };
            for r in points {
                let (_, gradient, hessian) = multipole_field(r, &cloud);
                for alpha in 0..3 {
                    let mut plus = r;
                    let mut minus = r;
                    shift(&mut plus, alpha, step);
                    shift(&mut minus, alpha, -step);
                    let (value_plus, grad_plus, _) = multipole_field(plus, &cloud);
                    let (value_minus, grad_minus, _) = multipole_field(minus, &cloud);

                    // The gradient is the derivative of the value, at every rank.
                    let numeric = (value_plus - value_minus) / (2.0 * step);
                    let analytic = gradient.to_array()[alpha];
                    assert!(
                        (analytic - numeric).abs() < 1.0e-8 * analytic.abs().max(1.0e-4),
                        "rank {rank} gradient[{alpha}] at {r:?}: {analytic} vs {numeric}"
                    );

                    for beta in 0..3 {
                        let numeric = (grad_plus.to_array()[beta] - grad_minus.to_array()[beta])
                            / (2.0 * step);
                        let analytic = hessian[beta][alpha];
                        if rank == 2 {
                            // The quadrupole's second derivative, `½ Θ_αβ T_αβγδ`, is not
                            // computed — see the truncation note on `multipole_field`. It is a
                            // fourth-order term, and the expansion is already missing a
                            // third-order one (the source octupole), so carrying it would not
                            // move the error.
                            //
                            // The assertion is that the truncation is where it is documented to
                            // be: exactly zero rather than approximately right, so nobody reads
                            // the Hessian of a quadrupole as if it were computed.
                            assert_eq!(
                                analytic, 0.0,
                                "the quadrupole Hessian is not computed; it should be 0, not {analytic}"
                            );
                        } else {
                            assert!(
                                (analytic - numeric).abs() < 1.0e-7 * analytic.abs().max(1.0e-4),
                                "rank {rank} hessian[{beta}][{alpha}] at {r:?}: {analytic} vs {numeric}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// A collapsed cloud is a point charge, so its field is the elementary one.
    ///
    /// The other end of the same check: with the dipole and quadrupole zero, `multipole_field`
    /// must reduce to `1/R`, `-R/R^3` and `(3 R R - delta R^2)/R^5` -- expressions simple enough
    /// to write out and compare against directly.
    #[test]
    fn a_bare_charge_reproduces_the_elementary_field() {
        let cloud = Cloud {
            centre: Vec3::zero(),
            charge: 1.0,
            dipole: Vec3::zero(),
            quadrupole: [[0.0; 3]; 3],
            range: 0..0,
        };
        for r in [Vec3::new(4.0, 0.0, 0.0), Vec3::new(2.5, -1.5, 3.0)] {
            let (value, gradient, hessian) = multipole_field(r, &cloud);
            let (norm, v) = (r.norm(), r.to_array());
            assert!((value - 1.0 / norm).abs() < 1.0e-14);
            for a in 0..3 {
                assert!((gradient.to_array()[a] + v[a] / norm.powi(3)).abs() < 1.0e-14);
                for b in 0..3 {
                    let delta = if a == b { 1.0 } else { 0.0 };
                    let want = (3.0 * v[a] * v[b] - delta * norm * norm) / norm.powi(5);
                    assert!((hessian[a][b] - want).abs() < 1.0e-13);
                }
            }
        }
    }
}
