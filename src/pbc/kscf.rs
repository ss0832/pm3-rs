// SPDX-License-Identifier: GPL-3.0-or-later

//! k-point periodic SCF.
//!
//! # What changes and what does not
//!
//! Nothing about the *integrals* changes at finite `k`. The geometry setup — the screened
//! two-electron tables, the Ewald sum, the core–core energy, the classical corrections — is the
//! same `pbc::gamma::build_setup` the Γ path uses, and it is shared rather than
//! reimplemented.
//!
//! What changes is which density multiplies what. In direct space
//!
//! ```text
//! F_μν(T) = ⟨φ_μ(r) | F | φ_ν(r − T)⟩,     F(k) = Σ_T e^{ik·T} F(T)
//! P_μν(T) = Σ_k w_k e^{−ik·T} P(k),        P(k) = Σ_n f_nk c_n c_n†
//! ```
//!
//! and the whole point of sampling more than one `k` is that `P(T)` can then *decay with T* — the
//! thing the Γ point cannot represent, and the reason it needs a supercell (see the note in
//! [`crate::pbc::gamma`]).
//!
//! # Only two terms live on an image block
//!
//! Almost every term in the NDDO Fock matrix lands on an on-site block. The one-center integrals
//! are intra-atomic; the Coulomb term contracts atom `B`'s on-site density into atom `A`'s
//! on-site block, however far away the image of `B` is; the electron–core attraction and the
//! Ewald potential are the same. All of those need only `P(0) = Σ_k w_k P(k)` and are computed
//! exactly as at Γ.
//!
//! Only the **resonance** `β·S` and the **exchange** connect orbitals in different cells, and only
//! those two are therefore carried image by image, in `pbc::gamma::ImageBlock`. So the
//! Bloch sum is short: `F(k) = F_onsite + Σ_T e^{ik·T} [β·S(T) − K(T)]`.
//!
//! # No generalized eigenproblem
//!
//! ZDO makes `S(T) = δ_T0 δ_μν`, so `S(k) = I` at every `k` and each k-point is an ordinary
//! Hermitian eigenproblem. See [`crate::cmatrix`].
//!
//! # Occupations
//!
//! Band energies at different `k` interleave, so occupation is a global question: a single Fermi
//! level across the whole mesh, not an aufbau count per k-point. Two conventions are offered
//! because they answer different questions — see [`Magnetization`].

use faer::c64;

use crate::cmatrix::{hermitian_eigen, CMatrix};
use crate::constants::EV_TO_KCAL;
use crate::corrections::periodic::periodic_correction_energy;
use crate::densitydiis::DensityDiis;
use crate::error::{Pm3Error, Result};
use crate::integrals::pack;
use crate::linalg::Matrix;
use crate::params::Pm3Parameters;

use crate::pbc::gamma::{
    add_one_center, add_site_potential, build_setup, initial_density, occupancy,
    write_electron_charges, ImageBlock, Occupancy, PeriodicOptions, Setup,
};
use crate::pbc::kpoints::{KPoint, KpointSpec};
use crate::scf::Pm3Options;
use crate::system::Molecule;

/// How the two spin channels share (or do not share) a Fermi level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Magnetization {
    /// One Fermi level for both spins, so the cell's magnetization is whatever the electronic
    /// structure prefers. This is the right convention for a magnetic solid, where the moment is
    /// an output.
    #[default]
    Free,
    /// A separate Fermi level per spin, holding `n_α − n_β` at the value the multiplicity asks
    /// for. This is the periodic reading of a molecular multiplicity, and the only way to
    /// converge onto a chosen spin state rather than the lowest one.
    Fixed,
}

/// k-point specific knobs. The geometry ones stay in [`PeriodicOptions`].
#[derive(Clone, Debug)]
pub struct KpointOptions {
    pub spec: KpointSpec,
    /// Fermi–Dirac broadening in eV. Zero fills strictly by energy, which is exact for an
    /// insulator and can oscillate for a metal.
    pub smearing_ev: f64,
    pub magnetization: Magnetization,
}

impl Default for KpointOptions {
    fn default() -> Self {
        Self {
            spec: KpointSpec::Gamma,
            smearing_ev: 0.0,
            magnetization: Magnetization::Fixed,
        }
    }
}

impl KpointOptions {
    /// An unshifted Γ-centred `n₁×n₂×n₃` mesh with everything else at its default.
    pub fn mesh(divisions: [usize; 3]) -> Self {
        Self {
            spec: KpointSpec::mesh(divisions),
            ..Self::default()
        }
    }
}

/// Result of a k-point periodic calculation. All energies are **per unit cell**.
#[derive(Clone, Debug)]
pub struct KpointResult {
    /// The irreducible points actually sampled, with their weights.
    pub kpoints: Vec<KPoint>,
    /// `P(T = 0)`, the on-site density per cell.
    pub density: Matrix,
    /// `P^α(0) − P^β(0)` per cell, for an unrestricted calculation only.
    pub spin_density: Option<Matrix>,
    pub unrestricted: bool,
    /// Band energies per k-point (α spin), ascending within each point.
    pub bands: Vec<Vec<f64>>,
    /// β-spin bands, for an unrestricted calculation.
    pub bands_beta: Option<Vec<Vec<f64>>>,
    /// Occupation numbers matching [`KpointResult::bands`], in electrons per state.
    pub occupations: Vec<Vec<f64>>,
    /// α Fermi level (eV). Under [`Magnetization::Free`] this is *the* Fermi level.
    pub fermi_ev: f64,
    /// β Fermi level, which differs from the α one only under [`Magnetization::Fixed`].
    pub fermi_beta_ev: f64,
    pub electronic_ev: f64,
    pub core_ev: f64,
    pub correction_ev: f64,
    pub ewald_ev: f64,
    pub total_ev: f64,
    /// `T·S` (eV per cell) from the Fermi–Dirac smearing — the **electronic** entropy of
    /// fractional occupations, and nothing else. Zero when `smearing_ev` is zero.
    ///
    /// This is not a thermodynamic entropy of the nuclei: there is no vibrational, rotational or
    /// translational contribution anywhere in this crate. `smearing_ev` is `k_B T` for a
    /// fictitious electronic temperature chosen to make a metal's occupations converge, not for
    /// the temperature of an experiment.
    /// The largest swing any atom's electron population made **during** the iteration, in
    /// electrons.
    ///
    /// Zero for a well-behaved SCF, which walks downhill. A large value means the density passed
    /// through qualitatively different arrangements on its way — charge sloshing — and a
    /// converged result that got there by sloshing is one that may have settled on a spurious
    /// self-consistent branch rather than the intended one. Rocksalt NaCl on a `2×2×2` mesh
    /// swings **1.1 electrons** and converges to a state with `−4.94` electrons on the *sodium*,
    /// against `+0.17` on every odd mesh: converged, self-consistent, and chemical nonsense.
    ///
    /// Reported rather than judged, because the crate has no way to know what is physical for a
    /// given system, and a threshold here would be a chemistry opinion in a numerics library.
    /// What it can say is that the iteration did not go straight there.
    pub charge_swing: f64,
    /// Non-`None` if the SCF failed at the requested settings and a retry rescued it, naming what
    /// the retry changed.
    ///
    /// **Read this before trusting a result that carries it.** A convergence aid can land on a
    /// *different* self-consistent solution rather than on the one that was asked for, and the
    /// difference is not always small: rocksalt NaCl on a `2×2×2` mesh converges by itself to
    /// `−341.79 eV` and, with a 5 eV level shift, to `−379.22`. The retry uses a level shift
    /// precisely because it leaves the converged density alone where it works — diamond and
    /// silicon come out at the same energy that damping reaches — but "where it works" is a
    /// statement about the run, not a guarantee. Confirm a rescued answer against a different
    /// mesh before building on it.
    pub rescued_by: Option<String>,
    pub entropy_ts_ev: f64,
    /// The **Mermin electronic free energy** per cell (eV): `total_ev − entropy_ts_ev`.
    ///
    /// # Why this exists and is not the same as the energy
    ///
    /// With Fermi–Dirac occupations the variational functional is `A = E − TS`, and it is `A`
    /// whose nuclear derivative the Hellmann–Feynman force computes. Reporting `E` beside a
    /// force that is `−dA/dR` makes the two inconsistent by `∂(TS)/∂R` — invisible on an
    /// insulator, where `S = 0`, and present on exactly the metals smearing exists for. This is
    /// the quantity ASE means by `free_energy` and asks for as `force_consistent=True`.
    ///
    /// # What it is **not**
    ///
    /// It is **not** a Gibbs free energy. There is no zero-point energy, no vibrational
    /// partition function, no `pV` term and no nuclear entropy in it: `G = H − TS_total` is a
    /// different quantity requiring a normal-mode analysis this crate does not do. Nor is it
    /// [`KpointResult::heat_of_formation_kcal`], which is MOPAC's parameterized ΔH°f at 298 K —
    /// a fitted quantity, not a computed thermodynamic potential. Three different things with
    /// three different names, deliberately.
    pub free_energy_ev: f64,
    pub heat_of_formation_kcal: f64,
    pub charges: Vec<f64>,
    /// Highest occupied and lowest unoccupied band energies over the sampled mesh, and the gap
    /// between them. A negative or zero gap means the mesh found the system metallic.
    pub homo_ev: Option<f64>,
    pub lumo_ev: Option<f64>,
    pub band_gap_ev: Option<f64>,
    pub iterations: usize,
    pub converged: bool,
    /// The Γ-point validity margin for this cell — see
    /// [`crate::pbc::gamma::PeriodicResult::gamma_margin`].
    ///
    /// Reported here too, where it is a *diagnosis* rather than a warning: a negative margin says
    /// why the mesh was necessary. Graphene's is permanently negative at any cell size, because
    /// its bonding runs through its images.
    pub gamma_margin: f64,
    /// The converged direct-space density, kept so [`band_structure`] can diagonalize at points
    /// that were never part of the mesh. Not public: it is an internal representation, and the
    /// physically meaningful part of it is already exposed as [`KpointResult::density`].
    ///
    /// `pub(crate)` rather than private because the phonon skeleton needs `P(T)`, not just
    /// `P(0)`. It assembled meshed dynamical matrices out of `P(0)` alone for every image, which
    /// is the Γ identity `P(T) = P(0)` — true by construction at Γ and false on a mesh, and the
    /// cause of a meshed `D(0)` that broke cubic symmetry and missed a finite difference by 300%.
    pub(crate) converged_density: (DensitySet, DensitySet),
}

impl KpointResult {
    /// `P(T)` for one atom pair and lattice translation, `norb_a × norb_b` row-major.
    ///
    /// `None` when the pair is outside the image list the SCF was built with, which means the
    /// block is zero by construction rather than missing.
    ///
    /// The orientation is the one asked for: an image list holds each pair once, so a request for
    /// `(b, a, −T)` is served by transposing the stored `(a, b, T)`.
    pub(crate) fn density_image(
        &self,
        images: &[crate::pbc::gamma::ImageBlock],
        a: usize,
        b: usize,
        t: [i32; 3],
    ) -> Option<ImagePair<'_>> {
        let (alpha, beta) = &self.converged_density;
        let negated = [-t[0], -t[1], -t[2]];
        for (index, block) in images.iter().enumerate() {
            if block.a == a && block.b == b && block.t == t {
                return Some(ImagePair {
                    alpha: &alpha.images[index],
                    beta: &beta.images[index],
                    cols: block.norb_b,
                    transposed: false,
                });
            }
            if block.a == b && block.b == a && block.t == negated {
                return Some(ImagePair {
                    alpha: &alpha.images[index],
                    beta: &beta.images[index],
                    cols: block.norb_b,
                    transposed: true,
                });
            }
        }
        None
    }
}

/// One image block of the density, in the orientation the caller asked for.
pub(crate) struct ImagePair<'a> {
    alpha: &'a [f64],
    beta: &'a [f64],
    /// Row stride **as stored**, before any transposition.
    cols: usize,
    transposed: bool,
}

impl ImagePair<'_> {
    /// `P^α(T)[mu, nu] + P^β(T)[mu, nu]` in the requested orientation.
    pub(crate) fn total(&self, mu: usize, nu: usize) -> f64 {
        let index = if self.transposed {
            nu * self.cols + mu
        } else {
            mu * self.cols + nu
        };
        self.alpha[index] + self.beta[index]
    }

    /// The same element for one spin channel: `0` is α, anything else β.
    pub(crate) fn spin(&self, channel: usize, mu: usize, nu: usize) -> f64 {
        let index = if self.transposed {
            nu * self.cols + mu
        } else {
            mu * self.cols + nu
        };
        if channel == 0 {
            self.alpha[index]
        } else {
            self.beta[index]
        }
    }
}

/// Band energies along a path through the Brillouin zone.
///
/// Non-self-consistent by construction, and correctly so: the density comes from the converged
/// mesh, and the path is then diagonalized in that fixed potential. Sampling the path
/// self-consistently would be wrong — a path is not a quadrature, and its points carry no
/// meaningful integration weight.
#[derive(Clone, Debug)]
pub struct BandStructure {
    pub kpoints: Vec<KPoint>,
    /// Cumulative Cartesian path length (1/Bohr) at each point, which is the sensible horizontal
    /// axis for a plot: it makes segments of different reciprocal length look different.
    pub distances: Vec<f64>,
    pub bands: Vec<Vec<f64>>,
    pub bands_beta: Option<Vec<Vec<f64>>>,
    /// The Fermi level from the self-consistent mesh, for reference on the plot.
    pub fermi_ev: f64,
}

/// The converged self-consistent field, rebuilt so it can be evaluated at **any** wavevector.
///
/// The k-point SCF keeps its density in direct space, `P^σ(T)`, and throws away the per-`k`
/// objects each iteration. Anything that wants an orbital at a `k` the mesh never visited — a
/// band path, or the `k + q` half of a phonon response — has to put the potential back together
/// from that direct-space density. This is that step, factored out of [`band_structure`] because
/// [`crate::pbc::dfpt`] needs exactly the same thing.
///
/// Borrowed rather than owned: the densities are the ones inside `scf`, and copying an
/// `nao × nao` per spin to hand them over would be the largest allocation in a phonon
/// calculation that does not need to exist.
pub(crate) struct ConvergedPotential<'a> {
    pub(crate) f_onsite_alpha: Matrix,
    /// Only built when the calculation is unrestricted; [`ConvergedPotential::beta_fock`] is how
    /// to read it, and returns the α matrix when there is only one.
    f_onsite_beta: Option<Matrix>,
    pub(crate) p_alpha: &'a DensitySet,
    pub(crate) p_beta: &'a DensitySet,
    pub(crate) unrestricted: bool,
}

impl ConvergedPotential<'_> {
    /// The β on-site Fock — the α one when the two spins share it.
    pub(crate) fn beta_fock(&self) -> &Matrix {
        self.f_onsite_beta.as_ref().unwrap_or(&self.f_onsite_alpha)
    }
}

/// Rebuild the converged potential. See [`ConvergedPotential`].
pub(crate) fn converged_potential<'a>(
    molecule: &Molecule,
    params: &Pm3Parameters,
    setup: &Setup,
    scf: &'a KpointResult,
) -> Result<ConvergedPotential<'a>> {
    let cell = molecule.cell.ok_or_else(|| {
        Pm3Error::InvalidInput("a periodic calculation needs a cell on the molecule".to_string())
    })?;
    let (p_alpha, p_beta) = &scf.converged_density;
    let p_total = add(&p_alpha.onsite, &p_beta.onsite);

    let mut sites = setup.sites.clone();
    write_electron_charges(setup, &p_total, &mut sites);
    let electrons = crate::pbc::ewald::ewald_potentials_cached(
        &cell,
        &sites,
        &setup.ewald_params,
        &setup.ewald_context,
    )?;

    let mut f_onsite_alpha = onsite_fock(molecule, params, setup, &p_total, &p_alpha.onsite)?;
    add_site_potential(setup, &mut f_onsite_alpha, &electrons);
    let f_onsite_beta = if scf.unrestricted {
        let mut beta = onsite_fock(molecule, params, setup, &p_total, &p_beta.onsite)?;
        add_site_potential(setup, &mut beta, &electrons);
        Some(beta)
    } else {
        None
    };

    Ok(ConvergedPotential {
        f_onsite_alpha,
        f_onsite_beta,
        p_alpha,
        p_beta,
        unrestricted: scf.unrestricted,
    })
}

/// `(ε, C)` at one wavevector and one spin, in a converged potential.
///
/// Works at any `k`, on the mesh or off it. That is not a convenience: `F(k) = Σ_T e^{ik·T}F(T)`
/// is assembled in the periodic-gauge AO basis with no `e^{iG·r}` anywhere, so `F(k + G) ≡ F(k)`
/// element for element and there is no umklapp bookkeeping to get wrong. A `k + q` that falls
/// off the mesh is diagonalized here exactly as one on it would be.
// The beta argument has no caller until M6 opens the response to open shells; the alpha half
// is what dfpt::response builds inline, one bloch_fock away from this.
#[allow(dead_code)]
pub(crate) fn bands_at(
    setup: &Setup,
    potential: &ConvergedPotential<'_>,
    k: &KPoint,
    beta: bool,
) -> Result<(Vec<f64>, CMatrix)> {
    let (fock, density) = if beta && potential.unrestricted {
        (potential.beta_fock(), potential.p_beta)
    } else {
        (&potential.f_onsite_alpha, potential.p_alpha)
    };
    hermitian_eigen(&bloch_fock(setup, fock, &density.images, k))
}

/// Evaluate band energies along `path` in the potential of a converged calculation.
pub fn band_structure(
    molecule: &Molecule,
    params: &Pm3Parameters,
    periodic: &PeriodicOptions,
    scf: &KpointResult,
    path: &[KPoint],
) -> Result<BandStructure> {
    let cell = molecule.cell.ok_or_else(|| {
        Pm3Error::InvalidInput("a band structure needs a cell on the molecule".to_string())
    })?;
    if path.is_empty() {
        return Err(Pm3Error::InvalidInput(
            "a band structure needs at least one k-point".to_string(),
        ));
    }
    let setup = build_setup(molecule, params, periodic)?;
    let potential = converged_potential(molecule, params, &setup, scf)?;
    let (p_alpha, p_beta) = (potential.p_alpha, potential.p_beta);
    let (f_alpha, f_beta) = (&potential.f_onsite_alpha, potential.beta_fock());

    let mut bands = Vec::with_capacity(path.len());
    let mut bands_beta = Vec::with_capacity(path.len());
    let mut distances = Vec::with_capacity(path.len());
    let mut travelled = 0.0;
    let mut previous: Option<crate::math::Vec3> = None;
    for k in path {
        let position = k.cartesian(&cell);
        if let Some(last) = previous {
            travelled += (position - last).norm();
        }
        previous = Some(position);
        distances.push(travelled);

        let (energies, _) = hermitian_eigen(&bloch_fock(&setup, f_alpha, &p_alpha.images, k))?;
        bands.push(energies);
        if scf.unrestricted {
            let (energies_beta, _) =
                hermitian_eigen(&bloch_fock(&setup, f_beta, &p_beta.images, k))?;
            bands_beta.push(energies_beta);
        }
    }

    Ok(BandStructure {
        kpoints: path.to_vec(),
        distances,
        bands,
        bands_beta: scf.unrestricted.then_some(bands_beta),
        fermi_ev: scf.fermi_ev,
    })
}

/// Run a k-point periodic PM3 calculation.
/// Smearing for the retry, in eV. Only ever accepted when it turns out to have changed
/// nothing — see the note at the call site.
/// The mixing fraction the first rescue rung uses.
///
/// 0.9 rather than the 0.3–0.5 a textbook fallback recommends, because the failures this path
/// actually sees are not the gentle ones that fraction is for. Measured: silicon needs 0.9 and is
/// unrescued at 0.5; NaCl converges unaided; diamond is unrescued at any fraction because its
/// residual alternates rather than decays.
const RESCUE_DAMPING: f64 = 0.9;

const RESCUE_SMEARING_EV: f64 = 0.1;

/// Iteration budget for a retry.
///
/// A rescued iteration is slower as well as more stable, and the default 200 is the budget that
/// already failed. Diamond under a level shift is still coming down at iteration 2000.
const RESCUE_ITERATIONS: usize = 2000;

/// `T·S` below which the occupations are integral and smearing changed nothing.
///
/// Fermi–Dirac at 0.1 eV puts `f` within `1e-9` of 0 or 1 for any state more than 2 eV from the
/// chemical potential, so a gapped insulator lands far below this and a system with any state
/// near the Fermi level lands far above. There is no continuum in between at this smearing.
///
/// `pub(crate)` because the response uses the same certificate for a different decision: this
/// path asks whether a smearing rescue changed the answer, and `pbc::dfpt` asks whether the
/// occupations are integral enough for a response that is not a metallic DFPT to be valid. One
/// threshold for one question — "is any state fractionally filled" — rather than two that could
/// drift apart.
pub(crate) const INTEGRAL_OCCUPATION_ENTROPY_EV: f64 = 1.0e-6;

pub fn run_kpoints(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
    kopt: &KpointOptions,
) -> Result<KpointResult> {
    let first = run_kpoints_with_terms(molecule, params, options, periodic, kopt, None);
    let (iterations, error, diagnosis) = match first {
        Ok(result) => return Ok(result),
        Err(Pm3Error::ScfNotConverged {
            iterations,
            error,
            diagnosis,
        }) => (iterations, error, diagnosis),
        // Anything else is not a convergence failure and retrying it would only take longer to
        // report the same thing.
        Err(other) => return Err(other),
    };

    // One retry, with a level shift, and only if the caller had not already set one — otherwise
    // this would silently override a deliberate choice.
    //
    // A level shift and not smearing, though smearing is the faster rescue on the same ladder.
    // Smearing changes the state: silicon converges under `0.1 eV` to an energy `0.69 eV` away
    // from the one damping and the shift agree on, and NaCl under `0.5 eV` to one `37 eV` away.
    // A rescue that answers a different question is worse than a failure, because the failure is
    // visible.
    let asked_for_aid =
        options.level_shift_ev != 0.0 || kopt.smearing_ev != 0.0 || options.damping != 0.0;
    let give_up = || Pm3Error::ScfNotConverged {
        iterations,
        error,
        diagnosis: diagnosis.clone(),
    };
    if asked_for_aid {
        return Err(give_up());
    }

    let budget = options.max_scf.max(RESCUE_ITERATIONS);

    // **Rung one: damping.** Ordered first because its guarantee is the stronger of the two.
    //
    // Damping changes the *path* and not the equations: the density fed to the next Fock is
    // `(1−λ)P_new + λP_old`, and at a fixed point those coincide, so a converged damped run
    // satisfies exactly the equations that were asked for. Nothing has to be certified after the
    // fact the way smearing does.
    //
    // The one thing that could have made this dishonest is already ruled out by construction: the
    // convergence test above measures `fresh − p`, the **undamped** step, so damping cannot
    // flatter it. Had it measured the damped step, a `λ = 0.9` run would report a residual five
    // times smaller than the truth and "converged" would mean a fifth of what it says.
    //
    // Measured on `examples/scf_hardening.rs`: damping at 0.9 rescues silicon and does nothing
    // for diamond, whose residual sits in an alternating limit cycle on two density elements
    // rather than decaying — a symmetry-breaking oscillation that no mixing fraction reaches.
    let damped = Pm3Options {
        damping: RESCUE_DAMPING,
        max_scf: budget,
        ..options.clone()
    };
    if let Ok(mut rescued) = run_kpoints_with_terms(molecule, params, &damped, periodic, kopt, None)
    {
        rescued.rescued_by = Some(format!(
            "the SCF ran out at the requested settings (residual {error:.3e} after {iterations} \
             iterations) and converged with density damping of {RESCUE_DAMPING} over {budget} \
             iterations. Damping changes the path and not the fixed point, and the convergence \
             test measures the undamped step, so this solves the equations that were asked for. \
             It does not guarantee the same *basin*: a cell with more than one self-consistent \
             solution can be steered between them by the mixer, and NaCl on an even mesh has \
             three that differ by tens of eV. Compare against a different mesh before trusting a \
             rescued energy"
        ));
        return Ok(rescued);
    }

    // **Fermi smearing, accepted only when it provably changed nothing.**
    //
    // This is the one convergence aid with a certificate, and the certificate is why it is the
    // only one applied automatically. If the occupations come out integral then `T·S` is zero, no
    // state is fractionally filled, and the smeared fixed-point equations are *identical* to the
    // strict-filling ones -- smearing only changed how the chemical potential was found on the
    // way to them. So the answer is the one that was asked for, not a nearby one.
    //
    // The alternative aids were measured on the same three systems
    // (`examples/scf_hardening.rs`) and none of them can say that. A 5 eV level shift converges
    // diamond to `−248.64 eV` where smearing gives `−249.27`; diamond's gap is 15.9 eV, so
    // smearing cannot have moved an occupation and the shift is the one that found a different
    // solution. Damping to 0.9 rescues silicon and does nothing for diamond. An *unguarded*
    // smearing is worse still: it takes silicon 0.69 eV away from the right answer and NaCl 37 eV
    // away, because there the occupations really do go fractional.
    //
    // So: rung two, with a proof, and nothing after it. A level shift is *not* a third rung —
    // it converges diamond to a different solution, which is the one failure mode worse than not
    // converging. It stays in the diagnosis for the caller to choose deliberately.
    let smeared = KpointOptions {
        smearing_ev: RESCUE_SMEARING_EV,
        ..kopt.clone()
    };
    let longer = Pm3Options {
        max_scf: budget,
        ..options.clone()
    };
    if let Ok(mut rescued) =
        run_kpoints_with_terms(molecule, params, &longer, periodic, &smeared, None)
    {
        if rescued.entropy_ts_ev.abs() <= INTEGRAL_OCCUPATION_ENTROPY_EV {
            rescued.rescued_by = Some(format!(
                "the SCF ran out at the requested settings (residual {error:.3e} after \
                 {iterations} iterations) and converged with {RESCUE_SMEARING_EV} eV of Fermi \
                 smearing over {budget} iterations. The occupations came out integral -- the \
                 electronic entropy is {:.2e} eV -- so this is the same self-consistent state \
                 the request asked for, reached by a smoother path to the chemical potential",
                rescued.entropy_ts_ev
            ));
            return Ok(rescued);
        }
        // Fractional occupations: a real answer, but to a different question. Say so rather than
        // return it, and rather than say nothing.
        return Err(Pm3Error::ScfNotConverged {
            iterations,
            error,
            diagnosis: Some(format!(
                "{}. A retry with {RESCUE_SMEARING_EV} eV of smearing does converge, but to a \
                 state with fractional occupations (electronic entropy {:.3e} eV), which is a \
                 different calculation rather than the same one reached more carefully -- so it \
                 is reported rather than returned. Set KpointOptions::smearing_ev yourself if \
                 that is what you want, and compare two smearings before trusting it",
                diagnosis.unwrap_or_else(|| "the iteration ran out".to_string()),
                rescued.entropy_ts_ev
            )),
        });
    }

    Err(give_up())
}

/// The same, with an extra Hermitian term added to `H(k)` at each k-point.
///
/// One matrix per entry of the **generated** k-point list, in that order — so a caller supplying
/// them must pin the ordering with [`KpointSpec::Explicit`] rather than let a mesh be reduced
/// underneath them.
///
/// This exists for [`crate::pbc::finite_field`], where the field couples neighbouring k-points and
/// so cannot be written as a term in the on-site Fock matrix. It is held **fixed** across the SCF
/// and recomputed by an outer loop, which is what keeps the inner loop an ordinary SCF.
pub(crate) fn run_kpoints_with_terms(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
    kopt: &KpointOptions,
    k_terms: Option<&[CMatrix]>,
) -> Result<KpointResult> {
    crate::pbc::refuse_field(molecule, options)?;
    let cell = molecule.cell.ok_or_else(|| {
        Pm3Error::InvalidInput("a periodic calculation needs a cell on the molecule".to_string())
    })?;
    if cell.n_periodic() == 0 {
        return Err(Pm3Error::InvalidInput(
            "the cell has no periodic direction; use the molecular path".to_string(),
        ));
    }
    let kpoints = kopt.spec.generate(&cell)?;
    if let Some(terms) = k_terms {
        if terms.len() != kpoints.len() {
            return Err(Pm3Error::InvalidInput(format!(
                "{} k-point terms were supplied for {} generated k-points. They are matched by \
                 position, so a mesh that reduces would silently pair each term with the wrong \
                 point; supply them against `KpointSpec::Explicit`.",
                terms.len(),
                kpoints.len()
            )));
        }
    }
    let Occupancy {
        n_elec,
        n_alpha,
        n_beta,
        unrestricted,
    } = occupancy(molecule, params, options)?;
    let setup = build_setup(molecule, params, periodic)?;
    let nao = setup.basis.nao;
    if n_alpha > nao {
        return Err(Pm3Error::InvalidInput(format!(
            "{n_alpha} occupied orbitals do not fit in {nao} basis functions",
        )));
    }

    let state = scf_loop(
        molecule,
        params,
        options,
        &setup,
        &kpoints,
        kopt,
        Targets {
            total: n_elec,
            alpha: n_alpha as f64,
            beta: n_beta as f64,
        },
        unrestricted,
        k_terms,
    )?;

    // Classical corrections are post-SCF and lattice-summed once per geometry. They do not depend
    // on `k` at all, so they are added exactly here and nowhere else — the one place a k-point
    // loop could otherwise multiply them by the mesh size.
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

    let (homo_ev, lumo_ev) = band_edges(&state);
    // The smearing's entropy term. Zero without smearing, where the occupations are a step and
    // `f ln f + (1−f) ln(1−f)` vanishes at every state.
    let entropy_ts_ev = smearing_entropy_ts_ev(&state, &kpoints, kopt.smearing_ev);
    Ok(KpointResult {
        kpoints,
        density: state.density,
        spin_density: state.spin_density,
        unrestricted,
        bands: state.bands,
        bands_beta: state.bands_beta,
        occupations: state.occupations,
        fermi_ev: state.fermi_ev,
        fermi_beta_ev: state.fermi_beta_ev,
        electronic_ev: state.electronic_ev,
        core_ev: setup.core_ev,
        correction_ev,
        ewald_ev: setup.core_ewald_ev + state.electron_ewald_ev,
        total_ev,
        entropy_ts_ev,
        free_energy_ev: total_ev - entropy_ts_ev,
        heat_of_formation_kcal: (total_ev - e_isol_sum + eheat_sum) * EV_TO_KCAL,
        charges,
        homo_ev,
        lumo_ev,
        band_gap_ev: match (homo_ev, lumo_ev) {
            (Some(h), Some(l)) => Some(l - h),
            _ => None,
        },
        iterations: state.iterations,
        converged: state.converged,
        charge_swing: state.charge_swing,
        gamma_margin: setup.gamma_margin,
        converged_density: state.converged_density,
        // Set by `run_kpoints` if it had to retry; the inner routine never rescues itself.
        rescued_by: None,
    })
}

/// Forces and stress for a k-point calculation.
#[derive(Clone, Debug)]
pub struct KpointGradient {
    /// The converged SCF this was evaluated at.
    pub scf: KpointResult,
    /// Total energy per cell (eV).
    pub energy_ev: f64,
    /// `∂E/∂R` per atom (eV/Bohr).
    pub gradient: Vec<crate::math::Vec3>,
    /// `−∂E/∂R` per atom (eV/Bohr).
    pub forces: Vec<crate::math::Vec3>,
    /// `∂E/∂ε` (eV). `None` in 1D and 2D, as at Γ.
    pub virial: Option<crate::math::Mat3>,
    /// `σ = (1/V) ∂E/∂ε` (eV/Bohr³).
    pub stress: Option<crate::math::Mat3>,
    pub max_gradient: f64,
}

/// The density a k-point calculation converged to, resolved by image.
///
/// The lookup is by `(a, b, T)` because the gradient walks the neighbour list in a different
/// order from the Fock build — `unique()` rather than `all()` — so positional pairing between the
/// two would be wrong in a way that is easy to write and hard to see.
/// `P_ij(T)` and its spin difference, keyed by `(a, b, T)`.
type ImageBlocks =
    std::collections::HashMap<(usize, usize, [i32; 3]), (Vec<f64>, Option<Vec<f64>>)>;

struct ResolvedDensity<'a> {
    onsite: &'a Matrix,
    spin_onsite: Option<Matrix>,
    offsets: &'a [usize],
    blocks: ImageBlocks,
}

impl crate::pbc::gradient::PairDensity for ResolvedDensity<'_> {
    fn onsite(&self, atom: usize, mu: usize, nu: usize) -> f64 {
        let off = self.offsets[atom];
        self.onsite[(off + mu, off + nu)]
    }

    fn inter(
        &self,
        i: usize,
        j: usize,
        t: [i32; 3],
        norb_i: usize,
        norb_j: usize,
    ) -> crate::pbc::gradient::InterBlock {
        match self.blocks.get(&(i, j, t)) {
            Some((total, spin)) => crate::pbc::gradient::InterBlock {
                total: total.clone(),
                spin: spin.clone(),
                cols: norb_j,
            },
            // Outside the exchange cutoff no block was stored, and none is needed: the two terms
            // that read one are both gated on the same cutoff.
            None => crate::pbc::gradient::InterBlock {
                total: vec![0.0; norb_i * norb_j],
                spin: self
                    .spin_onsite
                    .as_ref()
                    .map(|_| vec![0.0; norb_i * norb_j]),
                cols: norb_j,
            },
        }
    }
}

/// Converge a k-point SCF and evaluate the analytic forces and stress at that density.
///
/// The only thing that differs from the Γ-point gradient is where the inter-atomic density comes
/// from: `P_ij(T)` for the image being differentiated, rather than one `P(Γ)` shared by all of
/// them. Everything else — the AD on the pair displacement, the Ewald derivatives, the virial, the
/// classical corrections — is the same code.
pub fn kpoint_gradient(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
    kopt: &KpointOptions,
) -> Result<KpointGradient> {
    let scf = run_kpoints(molecule, params, options, periodic, kopt)?;
    let setup = build_setup(molecule, params, periodic)?;
    let (p_alpha, p_beta) = &scf.converged_density;

    let mut blocks = std::collections::HashMap::new();
    for (index, block) in setup.images.iter().enumerate() {
        let total: Vec<f64> = p_alpha.images[index]
            .iter()
            .zip(&p_beta.images[index])
            .map(|(a, b)| a + b)
            .collect();
        let spin = scf.unrestricted.then(|| {
            p_alpha.images[index]
                .iter()
                .zip(&p_beta.images[index])
                .map(|(a, b)| a - b)
                .collect::<Vec<f64>>()
        });
        blocks.insert((block.a, block.b, block.t), (total, spin));
    }

    let density = ResolvedDensity {
        onsite: &scf.density,
        spin_onsite: scf.spin_density.clone(),
        offsets: &setup.basis.atom_offset,
        blocks,
    };
    let (gradient, virial, stress) = crate::pbc::gradient::forces_and_stress(
        molecule,
        params,
        options,
        periodic,
        &density,
        &scf.density,
    )?;
    let forces: Vec<crate::math::Vec3> = gradient.iter().map(|g| *g * -1.0).collect();
    let max_gradient = gradient
        .iter()
        .flat_map(|g| g.to_array())
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    Ok(KpointGradient {
        energy_ev: scf.total_ev,
        scf,
        gradient,
        forces,
        virial,
        stress,
        max_gradient,
    })
}

/// Electron counts the occupation search has to hit.
struct Targets {
    total: f64,
    alpha: f64,
    beta: f64,
}

struct KScfState {
    density: Matrix,
    spin_density: Option<Matrix>,
    bands: Vec<Vec<f64>>,
    bands_beta: Option<Vec<Vec<f64>>>,
    /// Electrons per state, both spins — the reported filling.
    occupations: Vec<Vec<f64>>,
    /// The per-spin tables the densities are built from.
    occupations_alpha: Vec<Vec<f64>>,
    occupations_beta: Vec<Vec<f64>>,
    fermi_ev: f64,
    fermi_beta_ev: f64,
    electronic_ev: f64,
    electron_ewald_ev: f64,
    iterations: usize,
    converged: bool,
    /// Largest per-atom population swing over the whole run; see KpointResult::charge_swing.
    charge_swing: f64,
    converged_density: (DensitySet, DensitySet),
}

/// The density matrix in the form the Fock build needs it: the on-site block plus one small
/// block per image.
#[derive(Clone, Debug)]
pub(crate) struct DensitySet {
    /// `P(0)` — everything that is not resonance or exchange contracts against this.
    pub(crate) onsite: Matrix,
    /// `P(T)` restricted to each [`ImageBlock`]'s atom pair, in the same `a`-major layout.
    pub(crate) images: Vec<Vec<f64>>,
}

impl DensitySet {
    fn zeros(nao: usize, blocks: &[ImageBlock]) -> Self {
        Self {
            onsite: Matrix::zeros(nao, nao),
            images: blocks
                .iter()
                .map(|b| vec![0.0; b.norb_a * b.norb_b])
                .collect(),
        }
    }
}

#[allow(clippy::too_many_arguments)] // `k_terms` travels with the k-point list it indexes
fn scf_loop(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    setup: &Setup,
    kpoints: &[KPoint],
    kopt: &KpointOptions,
    targets: Targets,
    unrestricted: bool,
    // One extra Hermitian matrix per k-point, held fixed across this loop. See
    // `run_kpoints_with_terms`.
    k_terms: Option<&[CMatrix]>,
) -> Result<KScfState> {
    let cell = molecule.cell.expect("checked by the caller");
    let nao = setup.basis.nao;
    let n_k = kpoints.len();

    // Start from the same superposition of atomic densities the Γ path uses, split by spin
    // population. Unlike the Γ path there is no saddle to avoid here: a k-mesh that resolves
    // `P(T)` also resolves the symmetry breaking, and the fixed-magnetization mode holds the
    // spin split open by construction.
    let atomic = initial_density(molecule, params, &setup.basis)?;
    let scale = |fraction: f64| -> Matrix {
        let mut out = atomic.clone();
        for value in out.as_mut_slice() {
            *value *= fraction;
        }
        out
    };
    let mut p_alpha = DensitySet::zeros(nao, &setup.images);
    let mut p_beta = DensitySet::zeros(nao, &setup.images);
    p_alpha.onsite = scale(if targets.total > 0.0 {
        targets.alpha / targets.total
    } else {
        0.5
    });
    p_beta.onsite = scale(if targets.total > 0.0 {
        targets.beta / targets.total
    } else {
        0.5
    });

    let damping = if options.damping > 0.0 {
        options.damping.clamp(0.0, 0.95)
    } else {
        0.3
    };

    let mut last_energy = f64::INFINITY;
    // The most recent RMS density change, so a failure reports how far off it was.
    let mut last_change = f64::INFINITY;
    let mut converged = false;
    let mut iterations = 0;
    let mut electron_ewald_ev = 0.0;
    let mut bands = vec![Vec::new(); n_k];
    let mut bands_beta = vec![Vec::new(); n_k];
    let mut occupations = vec![Vec::new(); n_k];
    let mut occupations_beta = vec![Vec::new(); n_k];
    // Last iteration's filling, so the trace can say whether states are changing sides.
    let mut previous_occupations: Vec<Vec<f64>> = vec![Vec::new(); n_k];
    // `(per-atom population, occupation flips, residual)` for the last few iterations, which is
    // what [`diagnose`] reads if this loop runs out.
    let mut recent: Vec<(Vec<f64>, usize, f64)> = Vec::with_capacity(DIAGNOSIS_WINDOW + 1);
    // `(min, max)` population per atom over the whole run, for `KpointResult::charge_swing`.
    let mut population_range: Vec<(f64, f64)> = Vec::new();
    let mut fermi_ev = 0.0;
    let mut fermi_beta_ev = 0.0;
    let mut history = DensityDiis::new(
        2 * nao * nao + 2 * total_image_elements(setup),
        options.scf_memory_mb,
    );

    // A level shift raises the empty states before the density is rebuilt, which slows the
    // charge sloshing that stops a hard cell from settling. Zero by default; the caller asks
    // for it the same way the molecular path does, through `Pm3Options::level_shift_ev`.
    //
    // The projector it needs is the *previous* iteration's occupied space, so the orbitals are
    // carried across iterations. Zero differential overlap is what makes this cheap: with
    // `S = I` the occupied projector is just `Σ_n f_n c_n c_n†`, no metric anywhere.
    let shift = options.level_shift_ev;
    let mut previous_alpha: Vec<CMatrix> = Vec::new();
    let mut previous_beta: Vec<CMatrix> = Vec::new();

    for iteration in 1..=options.max_scf {
        iterations = iteration;
        let p_total_onsite = add(&p_alpha.onsite, &p_beta.onsite);

        // The electrons' long-range field, from the total on-site density. The lattice sum sees
        // charge, not spin, and the multipole charges are built from `P(0)` alone.
        let mut sites = setup.sites.clone();
        write_electron_charges(setup, &p_total_onsite, &mut sites);
        let electrons = crate::pbc::ewald::ewald_potentials_cached(
            &cell,
            &sites,
            &setup.ewald_params,
            &setup.ewald_context,
        )?;
        electron_ewald_ev = electrons.energy_ev;

        // On-site Fock blocks: identical in form to the Γ path, because none of these terms
        // connects different cells.
        let mut f_onsite_alpha =
            onsite_fock(molecule, params, setup, &p_total_onsite, &p_alpha.onsite)?;
        add_site_potential(setup, &mut f_onsite_alpha, &electrons);
        let f_onsite_beta = if unrestricted {
            let mut beta = onsite_fock(molecule, params, setup, &p_total_onsite, &p_beta.onsite)?;
            add_site_potential(setup, &mut beta, &electrons);
            beta
        } else {
            f_onsite_alpha.clone()
        };

        // Bloch-transform and diagonalize, one k-point at a time.
        let mut coefficients_alpha = Vec::with_capacity(n_k);
        let mut coefficients_beta = Vec::with_capacity(n_k);
        let mut h_of_k = Vec::with_capacity(n_k);
        let mut f_alpha_of_k = Vec::with_capacity(n_k);
        let mut f_beta_of_k = Vec::with_capacity(n_k);
        for (index, k) in kpoints.iter().enumerate() {
            let mut fk = bloch_fock(setup, &f_onsite_alpha, &p_alpha.images, k);
            // The field term, if there is one. Added after the Bloch transform because it is not
            // a lattice sum: it couples this k-point to its neighbours on the string, and the
            // outer loop holds it fixed while this SCF converges around it.
            if let Some(terms) = k_terms {
                for row in 0..fk.rows {
                    for col in 0..fk.cols {
                        fk[(row, col)] += terms[index][(row, col)];
                    }
                }
            }
            let (energies, vectors) =
                hermitian_eigen(&level_shifted(&fk, previous_alpha.get(index), shift))?;
            bands[index] = energies;
            coefficients_alpha.push(vectors);
            h_of_k.push(bloch_one_electron(setup, k));
            f_alpha_of_k.push(fk);

            if unrestricted {
                let fk_beta = bloch_fock(setup, &f_onsite_beta, &p_beta.images, k);
                let (energies_beta, vectors_beta) =
                    hermitian_eigen(&level_shifted(&fk_beta, previous_beta.get(index), shift))?;
                bands_beta[index] = energies_beta;
                coefficients_beta.push(vectors_beta);
                f_beta_of_k.push(fk_beta);
            }
        }
        if !unrestricted {
            bands_beta.clone_from(&bands);
            coefficients_beta.clone_from(&coefficients_alpha);
            f_beta_of_k.clone_from(&f_alpha_of_k);
        }

        let weights: Vec<f64> = kpoints.iter().map(|k| k.weight).collect();
        let (new_occ_alpha, new_occ_beta, mu_alpha, mu_beta) =
            occupy(&bands, &bands_beta, &weights, &targets, kopt, unrestricted)?;
        occupations = new_occ_alpha;
        occupations_beta = new_occ_beta;
        fermi_ev = mu_alpha;
        fermi_beta_ev = mu_beta;

        // Remember this iteration's occupied space, so the next one can shift against it.
        if shift != 0.0 {
            previous_alpha = coefficients_alpha
                .iter()
                .zip(&occupations)
                .map(|(c, f)| c.occupied_density(f))
                .collect();
            previous_beta = if unrestricted {
                coefficients_beta
                    .iter()
                    .zip(&occupations_beta)
                    .map(|(c, f)| c.occupied_density(f))
                    .collect()
            } else {
                previous_alpha.clone()
            };
        }

        let fresh_alpha = assemble_density(setup, kpoints, &coefficients_alpha, &occupations, nao);
        let fresh_beta = if unrestricted {
            assemble_density(setup, kpoints, &coefficients_beta, &occupations_beta, nao)
        } else {
            fresh_alpha.clone()
        };

        // E = ½ Σ_k w_k [ ⟨P_tot(k), H(k)⟩ + Σ_σ ⟨P^σ(k), F^σ(k)⟩ ], the k-space form of the
        // direct-space `½ Σ_T P(T)·(H(T) + F(T))`. `real_dot` is `Re Σ a b*`, which is what the
        // transform of a real direct-space sum produces.
        let mut electronic = 0.0;
        for (index, k) in kpoints.iter().enumerate() {
            let pa = density_at_k(&coefficients_alpha[index], &occupations[index]);
            let pb = if unrestricted {
                density_at_k(&coefficients_beta[index], &occupations_beta[index])
            } else {
                pa.clone()
            };
            let mut ptot = pa.clone();
            for (slot, value) in ptot.as_mut_slice().iter_mut().zip(pb.as_slice()) {
                *slot += value;
            }
            electronic += 0.5
                * k.weight
                * (ptot.real_dot(&h_of_k[index])
                    + pa.real_dot(&f_alpha_of_k[index])
                    + pb.real_dot(&f_beta_of_k[index]));
        }

        let change = fresh_alpha
            .onsite
            .rms_difference(&p_alpha.onsite)
            .max(fresh_beta.onsite.rms_difference(&p_beta.onsite));

        // `PM3_KSCF_WHERE=1`: which element of the density is still moving, and how the on-site
        // block compares with the images. A residual that will not go below `1e-4` on a
        // wide-gap insulator is not what a Pulay iteration is supposed to do, and the first
        // question is which part of the density it lives in.
        if std::env::var_os("PM3_KSCF_WHERE").is_some() {
            let n = fresh_alpha.onsite.rows;
            let mut worst = (0.0_f64, 0usize, 0usize);
            for i in 0..n {
                for j in 0..n {
                    let d = (fresh_alpha.onsite[(i, j)] - p_alpha.onsite[(i, j)]).abs();
                    if d > worst.0 {
                        worst = (d, i, j);
                    }
                }
            }
            let image_worst = fresh_alpha
                .images
                .iter()
                .zip(&p_alpha.images)
                .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
                .fold(0.0_f64, f64::max);
            eprintln!(
                "  where {iterations:4}  onsite max |dP| {:.3e} at ({}, {})  images max |dP| {:.3e}",
                worst.0, worst.1, worst.2, image_worst
            );
        }

        // Constant damping, all the way to the end — deliberately, and it is worth recording why,
        // because the alternative looks obviously right and is not.
        //
        // The Γ path hands over: `gamma.rs` disables damping as soon as its accelerator engages.
        // This path does not, and that costs something measurable. Diamond on a `3×3×3` mesh
        // reaches a residual of `1.1e-4` in a dozen iterations and then crawls, losing only a
        // factor of three over the next 190 against a `1e-7` tolerance, with DIIS extrapolating
        // on every one of them; mixing at `0.3` forever shrinks each residual by the same factor
        // before DIIS sees it, so the history spans less and less of the error.
        //
        // Handing over here — on the residual alone, or on the residual *and* the accelerator
        // having worked — converges diamond and silicon and breaks four k-point identities that
        // do not otherwise fail: the supercell folding, the band path against the mesh, the
        // folded forces, and an open-shell mesh's magnetization. Those are the tests that say
        // this path computes what it claims to, so the trade is not available at that price.
        //
        // Three literature remedies were implemented and measured against this. None of them is
        // an improvement, and the measurements are worth more than another attempt:
        //
        // * **Deeper Pulay history** (Pulay, *Chem. Phys. Lett.* **73**, 393 (1980)). `MAX_DEPTH`
        //   8 → 24 moves diamond's residual from `3.72e-5` to `3.41e-5`. Depth is not the binding
        //   constraint, so the stall is not the history failing to span the slow modes.
        //
        // * **Kerker preconditioning** (Kerker, *Phys. Rev. B* **23**, 3082 (1981); the local-basis
        //   transcription is the atomic-charge channel, as in DFTB+). Mixing the monopole channel
        //   at a quarter strength is neutral on every odd mesh, as designed — diamond does not
        //   move an electron between atoms at any point — and on NaCl's `2×2×2` it makes the
        //   sloshing **worse**, 1.44 electrons of swing becoming 2.56.
        //
        // * **Handing damping over to the accelerator**, as the Γ path does — see above.
        //
        // The Kerker result is the informative one, because it says what NaCl's `2×2×2` failure
        // actually is. Three different mixers reach three different *converged* solutions on the
        // same cell and mesh — `−341.79`, `−366.34` and `−379.22 eV`, with `−2.10`, `−4.35` and
        // `−4.94` electrons of charge on the sodium — while every mesh from `3³` to `7³` agrees on
        // `−327.45` and `+0.17`. That is not an ill-conditioned iteration that a preconditioner
        // straightens out. The fixed-point map has several attractors at that sampling and the
        // mixer picks among them; the cure is not to use that mesh, and `charge_swing` is here so
        // that using it is visible.
        //
        // `PM3_KSCF_WHERE=1` says where the residual that will not go away actually lives, and on
        // diamond the answer is specific: elements `(2, 7)` and `(3, 6)` of the on-site block,
        // pinned at `9.3e-5` for the last hundred iterations and **alternating between the two**.
        // With four orbitals per atom those are atom 1's `p_y` against atom 2's `p_z` and its
        // mirror — two elements a symmetry operation exchanges. The iteration is not converging
        // slowly towards a fixed point; it is orbiting between two symmetry-equivalent bond-order
        // arrangements, and their average is not a fixed point either, which is why a Pulay
        // history of any depth extrapolates onto it and leaves again.
        //
        // That is a symmetry-breaking limit cycle, and mixing is the wrong tool for it. The
        // remedy in codes that have it is to symmetrize the density against the crystal's point
        // group each pass (VASP's `ISYM`, and the equivalent elsewhere), which needs symmetry
        // detection this crate does not have. It is worth knowing that the missing piece is
        // symmetry rather than a better mixer.
        //
        // The diagnosis below reports the stiff tail meanwhile, with the iteration count it would
        // need, so a caller can raise `max_scf` knowingly.
        damp_set(&mut p_alpha, &fresh_alpha, damping);
        damp_set(&mut p_beta, &fresh_beta, damping);

        // Pulay's original DIIS, on the density rather than on the Fock matrix. The Γ path
        // extrapolates `F` against the `[F, P]` commutator, which does not carry over cleanly:
        // here `F` is a different matrix at every k-point while the quantity actually iterated is
        // one real direct-space density. Extrapolating that against its own residual
        // `P_out − P_in` keeps the accelerator attached to the thing being iterated, and costs a
        // few flat vectors rather than a Fock matrix per k-point.
        let residual = difference(&fresh_alpha, &fresh_beta, &p_alpha, &p_beta);
        history.push(flatten(&p_alpha, &p_beta), residual);
        let extrapolated = match history.extrapolate() {
            Some(blended) => {
                unflatten(&blended, &mut p_alpha, &mut p_beta);
                true
            }
            // The solve refused every suffix. Plain damping then, which converges but slowly, and
            // the trace says so rather than leaving a 0.99-per-step tail unexplained.
            None => false,
        };

        // Two failure modes look identical in `dP` alone and want opposite remedies, so the trace
        // separates them:
        //
        // * **Charge sloshing** — a long-wavelength density mode the iteration keeps
        //   overshooting. Its signature is the per-atom electron population swinging while the
        //   *energy* barely moves, because moving charge between well-separated sites costs
        //   little. Damping and DIIS fight it; preconditioning is what cures it.
        // * **Band crossing** — states reordering across the Fermi level, so the occupation
        //   assignment flips between iterations. Its signature is `flips`: how many states
        //   changed occupation by more than a thousandth of an electron since the last pass. A
        //   non-zero steady value means the iteration is choosing a different filling each time
        //   and cannot converge by damping at all.
        //
        // Set `PM3_KSCF_TRACE=1`.
        if std::env::var_os("PM3_KSCF_TRACE").is_some() {
            let population: Vec<f64> = (0..setup.basis.atom_offset.len())
                .map(|atom| {
                    let off = setup.basis.atom_offset[atom];
                    (0..setup.basis.atom_norb[atom])
                        .map(|mu| fresh_alpha.onsite[(off + mu, off + mu)])
                        .sum::<f64>()
                })
                .collect();
            let flips = occupations
                .iter()
                .zip(previous_occupations.iter())
                .map(|(now, before)| {
                    now.iter()
                        .zip(before.iter())
                        .filter(|(a, b)| (*a - *b).abs() > 1.0e-3)
                        .count()
                })
                .sum::<usize>();
            let charges: Vec<String> = population.iter().map(|v| format!("{v:7.4}")).collect();
            eprintln!(
                "  k-scf {iterations:4}  E={electronic:16.6}  dE={:12.3e}  dP={change:10.3e}  \
                 mu={fermi_ev:9.4}  flips={flips:3}  diis={}{:<2}  n_e=[{}]",
                electronic - last_energy,
                if extrapolated { "y" } else { "N" },
                history.len(),
                charges.join(" ")
            );
        }
        // The history the failure diagnosis reads. A fixed, small window: what matters is what
        // the iteration was doing when it ran out, not what it did on its way there.
        {
            let population: Vec<f64> = (0..setup.basis.atom_offset.len())
                .map(|atom| {
                    let off = setup.basis.atom_offset[atom];
                    (0..setup.basis.atom_norb[atom])
                        .map(|mu| fresh_alpha.onsite[(off + mu, off + mu)])
                        .sum::<f64>()
                })
                .collect();
            let flips = occupations
                .iter()
                .zip(previous_occupations.iter())
                .map(|(now, before)| {
                    now.iter()
                        .zip(before.iter())
                        .filter(|(a, b)| (*a - *b).abs() > 1.0e-3)
                        .count()
                })
                .sum::<usize>();
            // The whole-run extremes, not just the window's: sloshing that happened early and
            // damped out is still what decided which branch the iteration ended on.
            if population_range.is_empty() {
                population_range = population.iter().map(|p| (*p, *p)).collect();
            } else {
                for (slot, value) in population_range.iter_mut().zip(&population) {
                    slot.0 = slot.0.min(*value);
                    slot.1 = slot.1.max(*value);
                }
            }
            recent.push((population, flips, change));
            if recent.len() > DIAGNOSIS_WINDOW {
                recent.remove(0);
            }
        }
        previous_occupations.clone_from(&occupations);
        last_change = change;
        if (electronic - last_energy).abs() < options.e_tol && change < options.p_tol {
            converged = true;
            last_energy = electronic;
            break;
        }
        last_energy = electronic;
    }

    if !converged {
        // The density change, not a hardcoded `NaN` — see the note on the Γ path in `gamma.rs`.
        return Err(Pm3Error::ScfNotConverged {
            iterations,
            error: last_change,
            diagnosis: Some(diagnose(&recent, options.p_tol, options.max_scf)),
        });
    }

    let density = add(&p_alpha.onsite, &p_beta.onsite);
    let spin_density = unrestricted.then(|| subtract(&p_alpha.onsite, &p_beta.onsite));
    Ok(KScfState {
        density,
        spin_density,
        bands,
        bands_beta: unrestricted.then_some(bands_beta),
        // Report the physical filling — electrons per state, both spins — rather than the
        // per-spin tables the density is built from.
        occupations: occupations
            .iter()
            .zip(&occupations_beta)
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| x + y).collect())
            .collect(),
        occupations_alpha: occupations,
        occupations_beta,
        fermi_ev,
        fermi_beta_ev,
        electronic_ev: last_energy,
        electron_ewald_ev,
        iterations,
        converged,
        charge_swing: population_range
            .iter()
            .map(|(lo, hi)| hi - lo)
            .fold(0.0_f64, f64::max),
        converged_density: (p_alpha, p_beta),
    })
}

/// The on-site part of the Fock matrix: everything that is not resonance or exchange.
///
/// Structurally identical to the Γ-point build with the resonance and the image exchange removed,
/// because those are the only two terms that do not land on a `T = 0` block.
pub(crate) fn onsite_fock(
    molecule: &Molecule,
    params: &Pm3Parameters,
    setup: &Setup,
    p_tot: &Matrix,
    p_spin: &Matrix,
) -> Result<Matrix> {
    let basis = &setup.basis;
    let mut fock = setup.h_onsite.clone();
    add_one_center(molecule, params, basis, p_tot, p_spin, &mut fock)?;

    // Coulomb from every other atom and from every image, contracted with on-site densities.
    for tables in setup.pairs.iter().chain(setup.self_images.iter().flatten()) {
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
        if tables.a == tables.b {
            // A self-image pair is one interaction, not two: its Coulomb term was already summed
            // over every image and belongs in the atom's own block exactly once.
            continue;
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
    }
    Ok(fock)
}

/// `H(k) = H_onsite + Σ_T e^{ik·T} β·S(T)`.
fn bloch_one_electron(setup: &Setup, k: &KPoint) -> CMatrix {
    let mut out = CMatrix::from_real(&setup.h_onsite);
    for block in &setup.images {
        let phase = k.phase(block.t);
        let (oa, ob) = (
            setup.basis.atom_offset[block.a],
            setup.basis.atom_offset[block.b],
        );
        for mu in 0..block.norb_a {
            for la in 0..block.norb_b {
                out[(oa + mu, ob + la)] +=
                    phase * c64::new(block.resonance[mu * block.norb_b + la], 0.0);
            }
        }
    }
    out
}

/// `F^σ(k) = F_onsite + Σ_T e^{ik·T} [ β·S(T) − K^σ(T) ]`.
pub(crate) fn bloch_fock(
    setup: &Setup,
    onsite: &Matrix,
    p_images: &[Vec<f64>],
    k: &KPoint,
) -> CMatrix {
    let mut out = CMatrix::from_real(onsite);
    for (block, p_block) in setup.images.iter().zip(p_images) {
        let phase = k.phase(block.t);
        let (oa, ob) = (
            setup.basis.atom_offset[block.a],
            setup.basis.atom_offset[block.b],
        );
        let (na, nb) = (block.norb_a, block.norb_b);
        for mu in 0..na {
            for la in 0..nb {
                // Exchange with the density at *this* image, which is exactly what the Γ point
                // cannot supply.
                let mut exchange = 0.0;
                for nu in 0..na {
                    for si in 0..nb {
                        exchange += p_block[nu * nb + si]
                            * block.exchange[pack(mu, nu) * block.npack_b + pack(la, si)];
                    }
                }
                let value = block.resonance[mu * nb + la] - exchange;
                out[(oa + mu, ob + la)] += phase * c64::new(value, 0.0);
            }
        }
    }
    out
}

/// `F(k) + shift · (I − Q)`, where `Q` projects onto the occupied space.
///
/// Raising the empty states without touching the occupied ones leaves the converged solution
/// exactly where it was — at self-consistency the shift acts only on states the density does not
/// occupy — while making each step take a smaller bite out of the occupied–empty gap. That is
/// what stops a cell whose bands nearly cross from cycling between two fillings.
///
/// `Q` is the previous iteration's occupied projector, so on the first pass there is none and
/// the Fock matrix is returned untouched.
fn level_shifted(fock: &CMatrix, occupied: Option<&CMatrix>, shift: f64) -> CMatrix {
    let Some(projector) = occupied.filter(|_| shift != 0.0) else {
        return fock.clone();
    };
    let mut out = fock.clone();
    for i in 0..out.rows {
        for j in 0..out.cols {
            let identity = if i == j { 1.0 } else { 0.0 };
            out[(i, j)] += (identity - projector[(i, j)]) * shift;
        }
    }
    out
}

/// `P(k) = Σ_n f_nk c_n c_n†`.
fn density_at_k(coefficients: &CMatrix, occupations: &[f64]) -> CMatrix {
    coefficients.occupied_density(occupations)
}

/// Back-transform the k-space densities into the direct-space blocks the Fock build needs.
///
/// `P(T) = Re[ Σ_k w_k e^{−ik·T} P(k) ]`. The real part is exact rather than a truncation: with a
/// real Hamiltonian `P(−k) = P(k)*`, so the full-zone sum is real, and taking `Re` after summing
/// the time-reversal-reduced set with doubled weights reproduces it exactly.
fn assemble_density(
    setup: &Setup,
    kpoints: &[KPoint],
    coefficients: &[CMatrix],
    occupations: &[Vec<f64>],
    nao: usize,
) -> DensitySet {
    let mut out = DensitySet::zeros(nao, &setup.images);
    for (index, k) in kpoints.iter().enumerate() {
        let pk = density_at_k(&coefficients[index], &occupations[index]);

        // T = 0: the phase is 1, so this is just the weighted sum of the real parts.
        for (slot, value) in out.onsite.as_mut_slice().iter_mut().zip(pk.as_slice()) {
            *slot += k.weight * value.re;
        }

        for (block, target) in setup.images.iter().zip(out.images.iter_mut()) {
            let phase = k.phase(block.t).conj();
            let (oa, ob) = (
                setup.basis.atom_offset[block.a],
                setup.basis.atom_offset[block.b],
            );
            for mu in 0..block.norb_a {
                for la in 0..block.norb_b {
                    let value = pk[(oa + mu, ob + la)] * phase;
                    target[mu * block.norb_b + la] += k.weight * value.re;
                }
            }
        }
    }
    out
}

/// One value per band per k-point: band energies, or occupation numbers.
type PerBand = Vec<Vec<f64>>;

/// Both spin channels' occupations and their Fermi levels.
type Filling = (PerBand, PerBand, f64, f64);

/// Fill the bands and report the Fermi level(s).
///
/// The two returned tables are **per spin**: at most one electron per state each, so a restricted
/// run gets the same table twice rather than a doubled one. Reporting a doubled occupation while
/// building the density from it is the natural mistake here, and it hides itself — every k-point
/// result stays self-consistent, so even a mesh-against-supercell comparison passes. Only the
/// cross-check against the Γ-point path catches it.
fn occupy(
    bands: &[Vec<f64>],
    bands_beta: &[Vec<f64>],
    weights: &[f64],
    targets: &Targets,
    kopt: &KpointOptions,
    unrestricted: bool,
) -> Result<Filling> {
    if !unrestricted {
        let (occ, mu) = fill(bands, weights, targets.alpha, kopt.smearing_ev, 1.0)?;
        return Ok((occ.clone(), occ, mu, mu));
    }
    match kopt.magnetization {
        Magnetization::Fixed => {
            let (alpha, mu_a) = fill(bands, weights, targets.alpha, kopt.smearing_ev, 1.0)?;
            let (beta, mu_b) = fill(bands_beta, weights, targets.beta, kopt.smearing_ev, 1.0)?;
            Ok((alpha, beta, mu_a, mu_b))
        }
        Magnetization::Free => {
            // One Fermi level over both channels: concatenate the spins into a single ladder,
            // fill it, then split the result back.
            let mut combined: Vec<Vec<f64>> = Vec::with_capacity(bands.len() * 2);
            combined.extend(bands.iter().cloned());
            combined.extend(bands_beta.iter().cloned());
            let mut doubled = weights.to_vec();
            doubled.extend_from_slice(weights);
            let (occ, mu) = fill(&combined, &doubled, targets.total, kopt.smearing_ev, 1.0)?;
            let (alpha, beta) = occ.split_at(bands.len());
            Ok((alpha.to_vec(), beta.to_vec(), mu, mu))
        }
    }
}

/// Occupy a ladder of weighted states to a target electron count.
///
/// `max_occ` is how many electrons one state holds — 2 for a restricted channel, 1 for a spin
/// channel. With `smearing = 0` the states are filled strictly in energy order and the Fermi
/// level is the highest occupied energy; otherwise a Fermi–Dirac distribution is bisected onto
/// the target.
fn fill(
    bands: &[Vec<f64>],
    weights: &[f64],
    target: f64,
    smearing: f64,
    max_occ: f64,
) -> Result<(Vec<Vec<f64>>, f64)> {
    let mut occupations: Vec<Vec<f64>> = bands.iter().map(|row| vec![0.0; row.len()]).collect();
    let capacity: f64 = bands
        .iter()
        .zip(weights)
        .map(|(row, w)| row.len() as f64 * w * max_occ)
        .sum();
    if target > capacity + 1.0e-9 {
        return Err(Pm3Error::InvalidInput(format!(
            "{target} electrons per cell do not fit in the sampled bands (capacity {capacity})"
        )));
    }
    if target <= 0.0 {
        return Ok((occupations, f64::NEG_INFINITY));
    }

    if smearing > 0.0 {
        let mut low =
            bands.iter().flatten().fold(f64::INFINITY, |a, b| a.min(*b)) - 20.0 * smearing;
        let mut high = bands
            .iter()
            .flatten()
            .fold(f64::NEG_INFINITY, |a, b| a.max(*b))
            + 20.0 * smearing;
        // The electron count is monotone in μ, so plain bisection converges; 200 halvings take
        // any physical bracket below machine precision.
        for _ in 0..200 {
            let mu = 0.5 * (low + high);
            let count = fermi_count(bands, weights, mu, smearing, max_occ);
            if count > target {
                high = mu;
            } else {
                low = mu;
            }
        }
        let mu = 0.5 * (low + high);
        for (row, (energies, weight)) in occupations.iter_mut().zip(bands.iter().zip(weights)) {
            let _ = weight;
            for (slot, energy) in row.iter_mut().zip(energies) {
                *slot = max_occ * fermi_dirac(*energy, mu, smearing);
            }
        }
        return Ok((occupations, mu));
    }

    // Zero smearing: fill strictly by energy. Sorting the whole ladder is what makes this a
    // *global* aufbau — bands at different k-points interleave, and filling each k-point to its
    // own count would be a different (and wrong) calculation.
    let mut ladder: Vec<(f64, usize, usize)> = Vec::new();
    for (index, row) in bands.iter().enumerate() {
        for (band, energy) in row.iter().enumerate() {
            ladder.push((*energy, index, band));
        }
    }
    ladder.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut remaining = target;
    let mut fermi = ladder.first().map_or(0.0, |s| s.0);
    for (energy, index, band) in ladder {
        if remaining <= 1.0e-12 {
            break;
        }
        let room = weights[index] * max_occ;
        let take = room.min(remaining);
        occupations[index][band] = max_occ * (take / room);
        remaining -= take;
        fermi = energy;
    }
    Ok((occupations, fermi))
}

pub(crate) fn fermi_dirac(energy: f64, mu: f64, smearing: f64) -> f64 {
    let x = (energy - mu) / smearing;
    if x > 40.0 {
        0.0
    } else if x < -40.0 {
        1.0
    } else {
        1.0 / (1.0 + x.exp())
    }
}

fn fermi_count(bands: &[Vec<f64>], weights: &[f64], mu: f64, smearing: f64, max_occ: f64) -> f64 {
    bands
        .iter()
        .zip(weights)
        .map(|(row, weight)| {
            weight
                * max_occ
                * row
                    .iter()
                    .map(|e| fermi_dirac(*e, mu, smearing))
                    .sum::<f64>()
        })
        .sum()
}

/// `T·S` in eV per cell, from the Fermi–Dirac occupations the SCF converged on.
///
/// ```text
/// S/k_B = −Σ_σ Σ_k w_k Σ_n [ f ln f + (1−f) ln(1−f) ],     T·S = smearing_ev · S/k_B
/// ```
///
/// `smearing_ev` **is** `k_B T` for the fictitious electronic temperature, so the conversion is a
/// multiplication and not a physical constant. The two spin tables each hold `f ∈ [0, 1]` and are
/// equal for a restricted run, which supplies the factor of two without a special case.
///
/// Both endpoints are removable singularities: `f ln f → 0` as `f → 0` and likewise at `f = 1`.
/// A strictly filled state is exactly one of those, which is why this is identically zero without
/// smearing rather than merely small.
fn smearing_entropy_ts_ev(state: &KScfState, kpoints: &[KPoint], smearing_ev: f64) -> f64 {
    if smearing_ev <= 0.0 {
        return 0.0;
    }
    let mut dimensionless = 0.0;
    for table in [&state.occupations_alpha, &state.occupations_beta] {
        for (row, k) in table.iter().zip(kpoints) {
            for occupancy in row {
                let f = occupancy.clamp(0.0, 1.0);
                let filled = if f > 0.0 { f * f.ln() } else { 0.0 };
                let empty = if f < 1.0 {
                    (1.0 - f) * (1.0 - f).ln()
                } else {
                    0.0
                };
                dimensionless -= k.weight * (filled + empty);
            }
        }
    }
    smearing_ev * dimensionless
}

/// Highest occupied and lowest unoccupied band energies over the whole mesh.
fn band_edges(state: &KScfState) -> (Option<f64>, Option<f64>) {
    let mut homo = f64::NEG_INFINITY;
    let mut lumo = f64::INFINITY;
    let mut channels: Vec<(&PerBand, &PerBand)> = vec![(&state.bands, &state.occupations_alpha)];
    if let Some(beta) = &state.bands_beta {
        channels.push((beta, &state.occupations_beta));
    }
    for (bands, occupations) in channels {
        for (row, occ) in bands.iter().zip(occupations) {
            for (energy, filling) in row.iter().zip(occ) {
                if *filling > 1.0e-6 {
                    homo = homo.max(*energy);
                }
                if *filling < 1.0e-6 {
                    lumo = lumo.min(*energy);
                }
            }
        }
    }
    (
        homo.is_finite().then_some(homo),
        lumo.is_finite().then_some(lumo),
    )
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

/// Both spins' densities as one flat vector: the on-site block followed by every image block.
/// How many doubles the image blocks add to a flattened density.
fn total_image_elements(setup: &Setup) -> usize {
    setup
        .images
        .iter()
        .map(|block| block.norb_a * block.norb_b)
        .sum()
}

fn flatten(alpha: &DensitySet, beta: &DensitySet) -> Vec<f64> {
    let mut out = Vec::new();
    for set in [alpha, beta] {
        out.extend_from_slice(set.onsite.as_slice());
        for block in &set.images {
            out.extend_from_slice(block);
        }
    }
    out
}

fn unflatten(values: &[f64], alpha: &mut DensitySet, beta: &mut DensitySet) {
    let mut cursor = 0;
    for set in [alpha, beta] {
        let n = set.onsite.as_slice().len();
        set.onsite
            .as_mut_slice()
            .copy_from_slice(&values[cursor..cursor + n]);
        cursor += n;
        for block in &mut set.images {
            let m = block.len();
            block.copy_from_slice(&values[cursor..cursor + m]);
            cursor += m;
        }
    }
    debug_assert_eq!(cursor, values.len());
}

/// `P_out − P_in`, the SCF residual, flattened the same way.
fn difference(
    fresh_alpha: &DensitySet,
    fresh_beta: &DensitySet,
    alpha: &DensitySet,
    beta: &DensitySet,
) -> Vec<f64> {
    let out = flatten(fresh_alpha, fresh_beta);
    let old = flatten(alpha, beta);
    out.iter().zip(&old).map(|(a, b)| a - b).collect()
}

/// How many iterations the failure diagnosis looks back over.
///
/// Long enough to see a limit cycle — the NaCl one has a period of a few steps — and short enough
/// that it describes where the iteration ended up rather than where it started.
const DIAGNOSIS_WINDOW: usize = 12;

/// Say what the iteration was doing when it ran out.
///
/// The three ways a periodic SCF fails look identical in the residual and want different
/// remedies, so reporting the residual alone tells a caller a calculation failed and nothing
/// about what to change. All three are distinguishable from what the loop already computes:
///
/// * **Charge sloshing** — the per-atom populations swing while nothing else settles. Measured on
///   rocksalt NaCl with a `2×2×2` mesh: a full **1.1 electrons** moving between Na and Cl every
///   iteration, with the chemical potential following it across twelve electronvolts. Damping and
///   DIIS fight this and do not win, because the mode is nearly free in energy.
/// * **Band crossing** — states change occupation between passes, so the iteration is solving a
///   different filling each time. Smearing is what makes that continuous.
/// * **A slow tail** — nothing oscillates and the residual falls, just not fast enough. Diamond
///   and silicon on a `3×3×3` mesh do this at about `0.993` per iteration. The remedy is
///   iterations, and the estimate below says how many.
///
/// The classification is measured per run, not assumed from the system: `docs/pbc.md` attributed
/// NaCl's failure to a Fermi level trapped in a degenerate manifold, and the flip count says the
/// occupations never change at all.
fn diagnose(recent: &[(Vec<f64>, usize, f64)], p_tol: f64, max_scf: usize) -> String {
    if recent.len() < 3 {
        return "too few iterations to say why".to_string();
    }
    let flips: usize = recent.iter().map(|(_, f, _)| *f).sum();
    let swing = {
        let atoms = recent[0].0.len();
        (0..atoms)
            .map(|atom| {
                let values: Vec<f64> = recent.iter().map(|(p, _, _)| p[atom]).collect();
                let hi = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let lo = values.iter().cloned().fold(f64::INFINITY, f64::min);
                hi - lo
            })
            .fold(0.0_f64, f64::max)
    };
    let residuals: Vec<f64> = recent.iter().map(|(_, _, r)| *r).collect();
    let first = residuals[0];
    let last = *residuals.last().unwrap();
    // Geometric mean of the per-step ratio: below 1 the residual is still coming down.
    let rate = if first > 0.0 && last > 0.0 {
        (last / first).powf(1.0 / (residuals.len() - 1) as f64)
    } else {
        1.0
    };

    // A tenth of an electron moving back and forth is not rounding; it is the mode. Below that,
    // whatever is left is not what is holding the calculation up.
    if swing > 0.1 {
        return format!(
            "charge is sloshing: an atom's population swung by {swing:.3} electrons over the last \
             {} iterations while the density never settled. That is a long-wavelength mode the \
             iteration keeps overshooting, and it is nearly free in energy, so damping and DIIS \
             do not catch it. An odd k-mesh usually avoids it where an even one does not (see \
             docs/pbc.md); failing that, a larger supercell at Γ",
            recent.len()
        );
    }
    if flips > 0 {
        return format!(
            "states are changing occupation: {flips} occupation changes over the last {} \
             iterations, so the filling itself is different each pass and no amount of damping \
             the density can settle it. Set KpointOptions::smearing_ev, which makes the \
             occupation a continuous function of the band energy — and compare the energy against \
             a smaller smearing, since a large one converges to a different state",
            recent.len()
        );
    }
    if rate < 1.0 && last > 0.0 {
        // How many more steps at this rate, if it holds.
        let needed = (p_tol / last).ln() / rate.ln();
        if needed.is_finite() && needed > 0.0 {
            return format!(
                "nothing is oscillating and the residual is still falling, at {rate:.4} per \
                 iteration — a stiff iteration rather than an unstable one. At that rate it needs \
                 roughly {} more to reach the tolerance; raise Pm3Options::max_scf above the \
                 current {max_scf}. {}",
                needed.ceil() as u64,
                UNVERIFIABLE_AIDS
            );
        }
    }
    format!(
        "the residual is flat at {last:.3e} with no charge oscillation and no occupation changes, \
         so more iterations at these settings will not help. {UNVERIFIABLE_AIDS}"
    )
}

/// The aids that work but cannot certify themselves, offered rather than applied.
///
/// [`run_kpoints`] retries only with smearing, and only keeps the result when the occupations
/// come out integral — a check that proves the answer is the one that was asked for. Damping and
/// a level shift have no equivalent certificate: on the measured ladder a 5 eV shift converges
/// diamond to an energy 0.63 eV from the one smearing reaches, on a crystal whose 15.9 eV gap
/// means smearing cannot have moved an occupation. Both are worth trying by hand, with the
/// comparison this sentence asks for.
const UNVERIFIABLE_AIDS: &str = "Pm3Options::damping near 0.9 and \
     Pm3Options::level_shift_ev near 5.0 each converge cases this does not, but neither can show \
     it found the same solution rather than another one -- so compare the energy against a \
     different k-mesh, or against the other aid, before relying on it";

fn damp_set(current: &mut DensitySet, fresh: &DensitySet, damping: f64) {
    for (slot, value) in current
        .onsite
        .as_mut_slice()
        .iter_mut()
        .zip(fresh.onsite.as_slice())
    {
        *slot = damping * *slot + (1.0 - damping) * value;
    }
    for (block, new_block) in current.images.iter_mut().zip(&fresh.images) {
        for (slot, value) in block.iter_mut().zip(new_block) {
            *slot = damping * *slot + (1.0 - damping) * value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Cell;
    use crate::math::Vec3;
    use crate::pbc::gamma::run_gamma;
    use crate::system::Atom;

    const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n";

    fn celled(xyz: &str, edge: f64) -> Molecule {
        let mut molecule = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        molecule.cell = Some(Cell::cubic(edge).unwrap());
        molecule
    }

    /// `n×n×n` copies of `base` in a cell `n` times as wide — the same infinite crystal.
    fn supercell(base: &Molecule, edge: f64, n: usize) -> Molecule {
        let mut atoms = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let shift = Vec3::new(edge * i as f64, edge * j as f64, edge * k as f64);
                    atoms.extend(base.atoms.iter().map(|atom| {
                        let mut copy: Atom = atom.clone();
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

    fn kpoint(molecule: &Molecule, kopt: &KpointOptions) -> KpointResult {
        let params = Pm3Parameters::standard().unwrap();
        run_kpoints(
            molecule,
            &params,
            &Pm3Options::default(),
            &PeriodicOptions::default(),
            kopt,
        )
        .unwrap()
    }

    /// The k-point path at `k = Γ` must reproduce the Γ-point path exactly.
    ///
    /// The two are independent implementations of the same physics — one sums the image tables
    /// and contracts them with `P(Γ)`, the other keeps them resolved and contracts each with its
    /// own `P(T)`, which at a single k-point are the same numbers. Agreement is therefore a real
    /// cross-check of the Bloch machinery, the density back-transform, and the occupation search,
    /// against code that is already validated against MOPAC in the isolated limit.
    #[test]
    fn the_gamma_point_agrees_with_the_dedicated_gamma_path() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = celled(WATER, 20.0);
        // Converged well past the default, because what is being compared is two implementations
        // of the same physics and not two stopping rules. The paths run different accelerators —
        // the Γ one extrapolates the Fock matrix, this one the density — so at the default
        // tolerance they halt at different points *within* it, and the charges, which are not
        // stationary at the fixed point, show that difference before anything else does.
        // Tightening removes the stopping rule from the comparison instead of widening the
        // assertion until it no longer notices.
        let options = Pm3Options {
            e_tol: 1.0e-11,
            p_tol: 1.0e-10,
            max_scf: 400,
            ..Pm3Options::default()
        };
        let reference =
            run_gamma(&molecule, &params, &options, &PeriodicOptions::default()).unwrap();
        let result = run_kpoints(
            &molecule,
            &params,
            &options,
            &PeriodicOptions::default(),
            &KpointOptions::default(),
        )
        .unwrap();

        let difference = (result.total_ev - reference.total_ev).abs();
        assert!(
            difference < 1.0e-8,
            "k-point Γ {} vs run_gamma {} ({difference:.3e} eV)",
            result.total_ev,
            reference.total_ev
        );
        for (a, b) in result.charges.iter().zip(&reference.charges) {
            assert!((a - b).abs() < 1.0e-8, "charge {a} vs {b}");
        }
        // And the band energies are the Γ-point orbital energies.
        for (a, b) in result.bands[0].iter().zip(&reference.mo_energies) {
            assert!((a - b).abs() < 1.0e-8, "orbital {a} vs {b}");
        }
    }

    /// **The k-point test.** An `n×n×n` Γ-centred mesh on one cell and the Γ point of an `n×n×n`
    /// supercell are two descriptions of the same crystal sampled identically, so the energy per
    /// cell must agree.
    ///
    /// This is what a k-point implementation is for, and essentially nothing survives it by
    /// accident: it pins the Bloch phase convention and its sign, the direction of the density
    /// back-transform, the weights, the time-reversal reduction, the global Fermi level across
    /// interleaved bands, and the rule that the classical corrections are added once per cell
    /// rather than once per k-point.
    ///
    /// The cell is deliberately narrow enough that the Γ point alone would be badly wrong — see
    /// the note in [`crate::pbc::gamma`] — so the mesh is doing real work here rather than
    /// reproducing an isolated molecule.
    #[test]
    fn a_mesh_reproduces_the_equivalent_supercell() {
        let edge = 12.0;
        let one = celled(WATER, edge);
        for n in [2usize, 3] {
            let mesh = kpoint(&one, &KpointOptions::mesh([n, n, n]));
            let folded = kpoint(&supercell(&one, edge, n), &KpointOptions::default());
            let per_cell = folded.total_ev / (n * n * n) as f64;
            let difference = (mesh.total_ev - per_cell).abs();
            assert!(
                difference < 1.0e-7,
                "{n}³ mesh {} vs supercell {per_cell} per cell ({difference:.3e} eV)",
                mesh.total_ev
            );
        }
    }

    /// The corrections are classical and post-SCF, so they cannot depend on the mesh. A path that
    /// added them inside the k loop would scale them by the number of points.
    #[test]
    fn the_correction_energy_does_not_depend_on_the_mesh() {
        let molecule = celled(WATER, 14.0);
        let options = Pm3Options {
            variant: crate::corrections::Variant::Pm3D3H4X,
            ..Pm3Options::default()
        };
        let params = Pm3Parameters::standard().unwrap();
        let of = |divisions: [usize; 3]| {
            run_kpoints(
                &molecule,
                &params,
                &options,
                &PeriodicOptions::default(),
                &KpointOptions::mesh(divisions),
            )
            .unwrap()
            .correction_ev
        };
        let reference = of([1, 1, 1]);
        assert!(reference.abs() > 1.0e-6, "the test needs a live correction");
        for divisions in [[2, 2, 2], [3, 1, 1], [2, 3, 1]] {
            assert!(
                (of(divisions) - reference).abs() < 1.0e-12,
                "{divisions:?} changed the correction"
            );
        }
    }

    /// Denser meshes must converge, and the energy must stop moving once `P(T)` is resolved.
    #[test]
    fn the_energy_converges_with_the_mesh() {
        let molecule = celled(WATER, 14.0);
        let energies: Vec<f64> = [1usize, 2, 3, 4]
            .iter()
            .map(|n| kpoint(&molecule, &KpointOptions::mesh([*n, *n, *n])).total_ev)
            .collect();
        // Γ is far off; the mesh converges rapidly after that.
        let gamma_error = (energies[0] - energies[3]).abs();
        let converged = (energies[2] - energies[3]).abs();
        assert!(
            gamma_error > 1.0,
            "the Γ point should be badly wrong here, not {gamma_error:.3e}"
        );
        assert!(
            converged < 1.0e-3,
            "3³ and 4³ should agree, not by {converged:.3e}"
        );
    }

    /// A slab may only be sampled in its own plane, and doing so must still work.
    #[test]
    fn a_slab_samples_only_its_plane() {
        let mut slab = Molecule::from_xyz_str(WATER, 0.0).unwrap();
        slab.cell = Some(
            Cell::new(
                Vec3::new(14.0, 0.0, 0.0),
                Vec3::new(0.0, 14.0, 0.0),
                Vec3::new(0.0, 0.0, 60.0),
                [true, true, false],
            )
            .unwrap(),
        );
        let result = kpoint(&slab, &KpointOptions::mesh([2, 2, 1]));
        assert!(result.converged);
        assert!(result.kpoints.iter().all(|k| k.frac[2] == 0.0));
    }

    /// Forcing UHF on a closed shell must return the RHF answer with no spin density — the same
    /// guarantee the Γ path gives, now with a real mesh.
    #[test]
    fn forcing_uhf_on_a_closed_shell_mesh_changes_nothing() {
        let molecule = celled(WATER, 14.0);
        let params = Pm3Parameters::standard().unwrap();
        let run = |reference| {
            run_kpoints(
                &molecule,
                &params,
                &Pm3Options {
                    reference,
                    ..Pm3Options::default()
                },
                &PeriodicOptions::default(),
                &KpointOptions::mesh([2, 2, 2]),
            )
            .unwrap()
        };
        let restricted = run(crate::scf::Reference::Rhf);
        let forced = run(crate::scf::Reference::Uhf);
        assert!(forced.unrestricted && !restricted.unrestricted);
        assert!(
            (forced.total_ev - restricted.total_ev).abs() < 1.0e-8,
            "UHF {} vs RHF {}",
            forced.total_ev,
            restricted.total_ev
        );
        let spin = forced.spin_density.expect("UHF reports a spin density");
        let largest = spin.as_slice().iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(largest < 1.0e-7, "spurious spin polarization {largest:.3e}");
    }

    /// Time-reversal reduction must be exact, not an approximation. Running the full mesh
    /// explicitly (which skips reduction) has to give the same energy as the reduced one.
    #[test]
    fn time_reversal_reduction_is_exact() {
        use crate::pbc::kpoints::monkhorst_pack;
        let molecule = celled(WATER, 14.0);
        let cell = molecule.cell.unwrap();
        for divisions in [[3, 3, 3], [2, 3, 1]] {
            let full = monkhorst_pack(&cell, divisions, [0.0; 3]).unwrap();
            let reduced = kpoint(&molecule, &KpointOptions::mesh(divisions));
            assert!(
                reduced.kpoints.len() < full.len(),
                "{divisions:?} was not reduced at all"
            );
            let explicit = kpoint(
                &molecule,
                &KpointOptions {
                    spec: KpointSpec::Explicit(full),
                    ..KpointOptions::default()
                },
            );
            let difference = (reduced.total_ev - explicit.total_ev).abs();
            assert!(
                difference < 1.0e-8,
                "{divisions:?}: reduced {} vs full {} ({difference:.3e} eV)",
                reduced.total_ev,
                explicit.total_ev
            );
        }
    }

    /// A charged cell has to fold onto its supercell just as a neutral one does.
    ///
    /// The neutralizing background is a `1/V` constant, so the supercell's is eight times smaller
    /// per cell while carrying eight times the charge — the two only agree if the background
    /// energy, its Fock potential, and the electron count all scale together.
    #[test]
    fn a_charged_cell_folds_onto_its_supercell() {
        let edge = 12.0;
        let params = Pm3Parameters::standard().unwrap();
        let mut one = celled(WATER, edge);
        one.charge = 1.0;
        one.multiplicity = 2;
        let mut eight = supercell(&one, edge, 2);
        eight.charge = 8.0;
        eight.multiplicity = 9;

        let run = |molecule: &Molecule, kopt: &KpointOptions| {
            run_kpoints(
                molecule,
                &params,
                &Pm3Options::default(),
                &PeriodicOptions::default(),
                kopt,
            )
            .unwrap()
        };
        let mesh = run(&one, &KpointOptions::mesh([2, 2, 2]));
        let folded = run(&eight, &KpointOptions::default());
        let per_cell = folded.total_ev / 8.0;
        let difference = (mesh.total_ev - per_cell).abs();
        assert!(
            difference < 1.0e-6,
            "charged 2³ mesh {} vs supercell {per_cell} per cell ({difference:.3e} eV)",
            mesh.total_ev
        );
        // And the cell really is charged: the Mulliken charges sum to the formal charge.
        let total: f64 = mesh.charges.iter().sum();
        assert!((total - 1.0).abs() < 1.0e-6, "charges sum to {total}");
    }

    /// An open-shell cell must actually polarize, and its magnetization must be the one the
    /// multiplicity asked for.
    #[test]
    fn an_open_shell_mesh_polarizes_by_the_requested_moment() {
        const METHYL: &str =
            "4\nmethyl\nC 0.0 0.0 0.0\nH 0.0 1.078 0.0\nH 0.9336 -0.539 0.0\nH -0.9336 -0.539 0.0\n";
        let mut molecule = Molecule::from_xyz_str(METHYL, 0.0).unwrap();
        molecule.multiplicity = 2;
        molecule.cell = Some(Cell::cubic(16.0).unwrap());
        let result = kpoint(&molecule, &KpointOptions::mesh([2, 2, 2]));
        assert!(result.unrestricted && result.converged);

        let spin = result.spin_density.expect("UHF reports a spin density");
        let moment: f64 = (0..spin.rows).map(|i| spin[(i, i)]).sum();
        assert!(
            (moment - 1.0).abs() < 1.0e-6,
            "the unpaired electron should give a moment of 1, not {moment}"
        );
        assert!(result.bands_beta.is_some());
    }

    /// A band path must run in the converged potential, agree with the mesh where they touch, and
    /// carry a plottable horizontal axis.
    #[test]
    fn a_band_path_reproduces_the_mesh_where_they_meet() {
        use crate::pbc::kpoints::band_path;
        let params = Pm3Parameters::standard().unwrap();
        let molecule = celled(WATER, 14.0);
        let scf = kpoint(&molecule, &KpointOptions::mesh([2, 2, 2]));
        let path = band_path(
            &molecule.cell.unwrap(),
            &[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]],
            3,
        )
        .unwrap();
        let bands =
            band_structure(&molecule, &params, &PeriodicOptions::default(), &scf, &path).unwrap();

        assert_eq!(bands.bands.len(), path.len());
        assert!(bands.distances[0] == 0.0);
        assert!(bands.distances.windows(2).all(|w| w[1] >= w[0]));

        // Γ is both the path's first point and a point of the mesh, so the two must agree there —
        // to within one iteration's worth of density change, since the SCF reports the bands of
        // the Fock matrix it last diagonalized while the path is built from the density that came
        // out of it.
        let mesh_gamma = scf
            .kpoints
            .iter()
            .position(|k| k.frac == [0.0, 0.0, 0.0])
            .expect("a Γ-centred mesh contains Γ");
        for (a, b) in bands.bands[0].iter().zip(&scf.bands[mesh_gamma]) {
            assert!((a - b).abs() < 1.0e-7, "band {a} vs {b}");
        }
    }

    /// k-point forces against finite differences of the k-point energy.
    ///
    /// This is what says the image-resolved density is being read for the right image: using
    /// `P(Γ)` — or the right blocks in the wrong order — gives forces that are wrong by a few
    /// percent and look perfectly reasonable until differenced.
    #[test]
    fn forces_match_finite_differences() {
        let params = Pm3Parameters::standard().unwrap();
        let options = Pm3Options::default();
        let periodic = PeriodicOptions::default();
        let kopt = KpointOptions::mesh([2, 2, 1]);
        let base = celled(WATER, 13.0);

        let analytic = kpoint_gradient(&base, &params, &options, &periodic, &kopt).unwrap();
        let energy_at = |molecule: &Molecule| {
            run_kpoints(molecule, &params, &options, &periodic, &kopt)
                .unwrap()
                .total_ev
        };
        let step = 2.0e-4;
        for atom in 0..base.atoms.len() {
            for axis in 0..3 {
                let mut plus = base.clone();
                let mut minus = base.clone();
                shift(&mut plus.atoms[atom].position, axis, step);
                shift(&mut minus.atoms[atom].position, axis, -step);
                let numerical = (energy_at(&plus) - energy_at(&minus)) / (2.0 * step);
                let exact = analytic.gradient[atom].to_array()[axis];
                assert!(
                    (numerical - exact).abs() < 2.0e-4,
                    "atom {atom} axis {axis}: analytic {exact} vs finite difference {numerical}"
                );
            }
        }
    }

    /// Forces must fold like energies: an `n×n×n` mesh and the equivalent supercell have to agree
    /// atom for atom.
    #[test]
    fn forces_fold_onto_the_supercell() {
        let edge = 12.0;
        let params = Pm3Parameters::standard().unwrap();
        let options = Pm3Options::default();
        let periodic = PeriodicOptions::default();
        let one = celled(WATER, edge);
        let two = supercell(&one, edge, 2);

        let mesh = kpoint_gradient(
            &one,
            &params,
            &options,
            &periodic,
            &KpointOptions::mesh([2, 2, 2]),
        )
        .unwrap();
        let folded = kpoint_gradient(
            &two,
            &params,
            &options,
            &periodic,
            &KpointOptions::default(),
        )
        .unwrap();
        for atom in 0..one.atoms.len() {
            for axis in 0..3 {
                let a = mesh.gradient[atom].to_array()[axis];
                // The supercell's first copy sits at the same place as the single cell's atoms.
                let b = folded.gradient[atom].to_array()[axis];
                assert!(
                    (a - b).abs() < 1.0e-6,
                    "atom {atom} axis {axis}: mesh {a} vs supercell {b}"
                );
            }
        }
        // And the stress, which is per cell and so needs no scaling either.
        let (sa, sb) = (mesh.stress.unwrap(), folded.stress.unwrap());
        for beta in 0..3 {
            for alpha in 0..3 {
                let (a, b) = (
                    sa.col[beta].to_array()[alpha],
                    sb.col[beta].to_array()[alpha],
                );
                assert!(
                    (a - b).abs() < 1.0e-8,
                    "stress ({alpha},{beta}): mesh {a} vs supercell {b}"
                );
            }
        }
    }

    /// Translational invariance: the forces on a cell in equilibrium with nothing external must
    /// sum to zero.
    #[test]
    fn forces_sum_to_zero() {
        let params = Pm3Parameters::standard().unwrap();
        let result = kpoint_gradient(
            &celled(WATER, 13.0),
            &params,
            &Pm3Options::default(),
            &PeriodicOptions::default(),
            &KpointOptions::mesh([2, 2, 2]),
        )
        .unwrap();
        let mut total = Vec3::zero();
        for force in &result.forces {
            total += *force;
        }
        let largest = total.to_array().iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(largest < 1.0e-8, "net force {largest:.3e}");
    }

    /// The stress against finite differences of the energy under strain.
    #[test]
    fn stress_matches_strained_finite_differences() {
        let params = Pm3Parameters::standard().unwrap();
        let options = Pm3Options::default();
        let periodic = PeriodicOptions::default();
        let kopt = KpointOptions::mesh([2, 2, 1]);
        let base = celled(WATER, 13.0);
        let analytic = kpoint_gradient(&base, &params, &options, &periodic, &kopt).unwrap();
        let virial = analytic.virial.expect("3D reports a virial");

        let step = 1.0e-4;
        for beta in 0..3 {
            for alpha in 0..3 {
                let strained = |sign: f64| {
                    let mut eps = crate::math::Mat3::zero();
                    set_component(&mut eps, alpha, beta, sign * step);
                    let mut molecule = base.clone();
                    molecule.cell = Some(base.cell.unwrap().strained(&eps));
                    for atom in &mut molecule.atoms {
                        atom.position += eps.mul_vec(atom.position);
                    }
                    run_kpoints(&molecule, &params, &options, &periodic, &kopt)
                        .unwrap()
                        .total_ev
                };
                let numerical = (strained(1.0) - strained(-1.0)) / (2.0 * step);
                let exact = virial.col[beta].to_array()[alpha];
                assert!(
                    (numerical - exact).abs() < 5.0e-4,
                    "({alpha},{beta}): analytic {exact} vs finite difference {numerical}"
                );
            }
        }
    }

    fn shift(v: &mut Vec3, axis: usize, delta: f64) {
        match axis {
            0 => v.x += delta,
            1 => v.y += delta,
            _ => v.z += delta,
        }
    }

    fn set_component(m: &mut crate::math::Mat3, alpha: usize, beta: usize, value: f64) {
        match alpha {
            0 => m.col[beta].x = value,
            1 => m.col[beta].y = value,
            _ => m.col[beta].z = value,
        }
    }

    /// The occupations must account for exactly the electrons in the cell, at every mesh size and
    /// with or without smearing.
    #[test]
    fn the_occupations_hold_the_electron_count() {
        let molecule = celled(WATER, 14.0);
        for smearing in [0.0, 0.2] {
            let result = kpoint(
                &molecule,
                &KpointOptions {
                    spec: KpointSpec::mesh([2, 2, 2]),
                    smearing_ev: smearing,
                    ..KpointOptions::default()
                },
            );
            let counted: f64 = result
                .kpoints
                .iter()
                .zip(&result.occupations)
                .map(|(k, row)| k.weight * row.iter().sum::<f64>())
                .sum();
            assert!(
                (counted - 8.0).abs() < 1.0e-6,
                "smearing {smearing}: {counted} electrons, expected 8"
            );
            // And the density's trace has to agree with it.
            let trace: f64 = (0..result.density.rows)
                .map(|i| result.density[(i, i)])
                .sum();
            assert!((trace - 8.0).abs() < 1.0e-6, "density trace {trace}");
        }
    }
}
