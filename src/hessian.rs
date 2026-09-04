// SPDX-License-Identifier: GPL-3.0-or-later

// Cartesian/MO tensor loops are kept index-explicit to preserve their direct
// correspondence with the CPHF equations and atom-block scatter operations.
#![allow(clippy::needless_range_loop, clippy::type_complexity)]

//! Nuclear Hessian and harmonic vibrational analysis.
//!
//! [`analytic_hessian`] is the primary path: a **fully analytic** RHF Hessian combining a
//! closed-form skeleton second derivative (second-order forward-AD, [`crate::dual2::Dual2`],
//! of the two-center integral kernels) with the CPHF orbital-relaxation response — no finite
//! differences. [`numerical_hessian`] (central differences of the analytic gradient, `3N`
//! columns in parallel on rayon) is retained as the independent validation reference and as
//! the fallback for open-shell / heavy-element systems. Mass-weighting and diagonalization
//! (faer) give harmonic frequencies; overall translations and rotations appear as the ~`6`
//! (5 for linear molecules) near-zero modes.

use crate::dual::Scalar;
use crate::error::{Pm3Error, Result};
use crate::gradient::closed_form_gradient;
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::params::Pm3Parameters;
use crate::scf::Pm3Options;
use crate::system::Molecule;
use std::sync::atomic::{AtomicU64, Ordering};

// CPHF profiling accumulators (µs), printed by analytic_hessian when PM3_TIMING is set.
static T_TRANSFORM_US: AtomicU64 = AtomicU64::new(0);
static T_FOCK_US: AtomicU64 = AtomicU64::new(0);
static T_PROJ_US: AtomicU64 = AtomicU64::new(0);
static N_CPHF_ITER: AtomicU64 = AtomicU64::new(0);
static N_LOC_AOS: AtomicU64 = AtomicU64::new(0);

pub(crate) const MIB: usize = 1024 * 1024;

fn checked_workspace_bytes(items: &[usize]) -> usize {
    items
        .iter()
        .copied()
        .try_fold(1usize, usize::checked_mul)
        .unwrap_or(usize::MAX)
}

fn hessian_worker_count(options: &Pm3Options, jobs: usize, bytes_per_job: usize) -> usize {
    let available = rayon::current_num_threads().min(jobs.max(1));
    if options.hessian_memory_mb == 0 || bytes_per_job == 0 {
        return available.max(1);
    }
    let budget = options.hessian_memory_mb.saturating_mul(MIB);
    available.min((budget / bytes_per_job).max(1)).max(1)
}

pub(crate) fn cphf_chunk_size(
    options: &Pm3Options,
    operation: &'static str,
    ndof: usize,
    resident_bytes: usize,
    bytes_per_response: usize,
) -> Result<usize> {
    if options.hessian_memory_mb == 0 {
        return Ok(96.min(ndof).max(1));
    }
    let budget = options.hessian_memory_mb.saturating_mul(MIB);
    if resident_bytes >= budget {
        return Err(crate::error::Pm3Error::ResourceLimit {
            operation,
            required_mb: resident_bytes.saturating_add(MIB - 1) / MIB,
            limit_mb: options.hessian_memory_mb,
        });
    }
    let remaining = budget - resident_bytes;
    if bytes_per_response > remaining {
        return Err(crate::error::Pm3Error::ResourceLimit {
            operation,
            required_mb: resident_bytes
                .saturating_add(bytes_per_response)
                .saturating_add(MIB - 1)
                / MIB,
            limit_mb: options.hessian_memory_mb,
        });
    }
    Ok((remaining / bytes_per_response.max(1)).clamp(1, 96.min(ndof).max(1)))
}

fn symmetrize_average_in_place(matrix: &mut Matrix) {
    debug_assert_eq!(matrix.rows, matrix.cols);
    for i in 0..matrix.rows {
        for j in 0..i {
            let value = 0.5 * (matrix[(i, j)] + matrix[(j, i)]);
            matrix[(i, j)] = value;
            matrix[(j, i)] = value;
        }
    }
}

fn add_transpose_in_place(matrix: &mut Matrix) {
    debug_assert_eq!(matrix.rows, matrix.cols);
    for i in 0..matrix.rows {
        matrix[(i, i)] *= 2.0;
        for j in 0..i {
            let value = matrix[(i, j)] + matrix[(j, i)];
            matrix[(i, j)] = value;
            matrix[(j, i)] = value;
        }
    }
}

/// `sqrt(eV / (Å²·amu))` → cm⁻¹ (standard vibrational conversion; 1 unit = 521.47 cm⁻¹).
pub const SQRT_EV_PER_ANG2_AMU_TO_CM: f64 = 521.470_9;

#[derive(Clone, Debug)]
pub struct VibrationalModes {
    /// Cartesian Hessian (eV/Bohr²), symmetric, size `3N × 3N`.
    pub hessian: Matrix,
    /// Harmonic frequencies (cm⁻¹), ascending; negative = imaginary (saddle/unconverged).
    pub frequencies_cm: Vec<f64>,
    /// Mass-weighted eigenvalues (eV/(Å²·amu)).
    pub eigenvalues: Vec<f64>,
    /// Normal modes as the **columns** of a `3N × 3N` matrix, in **mass-weighted** coordinates
    /// and ordered with `frequencies_cm`.
    ///
    /// Mass-weighted, not Cartesian: these are the eigenvectors of `H_ij/√(m_i m_j)`, so the
    /// Cartesian displacement of mode `m` is `l_{im} / √m_i`. Anything contracting a Cartesian
    /// quantity against a mode — an infrared intensity above all — has to divide by `√m` first,
    /// and forgetting to is the classic way to get intensities that look plausible and are
    /// systematically wrong.
    pub modes: Matrix,
}

/// Cartesian Hessian (eV/Bohr²) by central differences of the analytic gradient.
/// The analytic and numerical Hessians both assume a tightly-converged **stationary** density
/// (Brillouin: the occupied–virtual Fock block vanishes). The energy/gradient default tolerances
/// converge the energy well but leave the density — and hence `F_ov` — looser than a *second*
/// derivative needs, most visibly for slowly-converging near-degenerate systems (linear
/// molecules' π shells). Converging the SCF tighter here removes that error (ZnCl₂ 1.8e-5 →
/// 2.5e-7) while leaving the public energy/gradient tolerances untouched.
pub(crate) fn tighten_scf_for_hessian(options: &Pm3Options) -> Pm3Options {
    let mut o = options.clone();
    o.e_tol = o.e_tol.min(1.0e-12);
    o.p_tol = o.p_tol.min(1.0e-11);
    o.max_scf = o.max_scf.max(500);
    o
}

/// **Analytic** second derivatives of the classical D3/H4/X correction energy (eV/Bohr²,
/// `3N × 3N`), by forward-mode automatic differentiation ([`crate::dual2::Dual2`]) of the
/// scalar-generic correction energy — the D3 coordination-number coupling included, since the
/// coordination numbers are computed in the same generic arithmetic. No finite differences and no
/// SCF; added on top of the CPHF Hessian for the correction variants. Zero for plain PM3.
///
/// Each Dual2 tracks a 3-vector, so the diagonal 3×3 atom blocks come from one evaluation per
/// atom (seed that atom's `x,y,z`), and each off-diagonal element from an evaluation seeding one
/// coordinate of atom `a` and one of atom `b` — exact, `O(N²)` cheap classical evaluations.
fn correction_hessian(molecule: &Molecule, variant: crate::corrections::Variant) -> Matrix {
    use crate::dual2::Dual2;
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let mut h = Matrix::zeros(ndof, ndof);
    if variant == crate::corrections::Variant::Pm3 {
        return h;
    }
    let (numbers, p0) = crate::corrections::geometry_f64(molecule);
    let const_pos = || -> Vec<[Dual2; 3]> {
        p0.iter()
            .map(|p| {
                [
                    Dual2::constant(p[0]),
                    Dual2::constant(p[1]),
                    Dual2::constant(p[2]),
                ]
            })
            .collect()
    };
    // Diagonal 3×3 blocks: seed all three coordinates of atom `a` (variables 0,1,2).
    for a in 0..nat {
        let mut pos = const_pos();
        pos[a] = [
            Dual2::var(p0[a][0], 0),
            Dual2::var(p0[a][1], 1),
            Dual2::var(p0[a][2], 2),
        ];
        let e = crate::corrections::correction_energy_g::<Dual2>(&numbers, &pos, variant);
        for k in 0..3 {
            for l in 0..3 {
                h[(3 * a + k, 3 * a + l)] = e.h[k][l];
            }
        }
    }
    // Off-diagonal blocks (a < b): seed coordinate `k` of `a` (var 0) and `l` of `b` (var 1);
    // the mixed second derivative is `e.h[0][1]`. Each of these is an independent
    // O(N²) classical evaluation, so the O(N²) pair loop is the dominant cost of a
    // correction Hessian — run it on rayon, and build the constant-position vector
    // once per pair instead of once per the nine mixed derivatives.
    let pairs: Vec<(usize, usize)> = (0..nat)
        .flat_map(|a| ((a + 1)..nat).map(move |b| (a, b)))
        .collect();
    let blocks: Vec<[[f64; 3]; 3]> = {
        use rayon::prelude::*;
        pairs
            .par_iter()
            .map(|&(a, b)| {
                let base = const_pos();
                let mut block = [[0.0; 3]; 3];
                for k in 0..3 {
                    for l in 0..3 {
                        let mut pos = base.clone();
                        pos[a][k] = Dual2::var(p0[a][k], 0);
                        pos[b][l] = Dual2::var(p0[b][l], 1);
                        block[k][l] = crate::corrections::correction_energy_g::<Dual2>(
                            &numbers, &pos, variant,
                        )
                        .h[0][1];
                    }
                }
                block
            })
            .collect()
    };
    for (&(a, b), block) in pairs.iter().zip(&blocks) {
        for k in 0..3 {
            for l in 0..3 {
                h[(3 * a + k, 3 * b + l)] = block[k][l];
                h[(3 * b + l, 3 * a + k)] = block[k][l];
            }
        }
    }
    h
}

pub fn numerical_hessian(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    step: f64,
) -> Result<Matrix> {
    use rayon::prelude::*;
    let options = &tighten_scf_for_hessian(options);
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let nao = crate::basis::Basis::build(molecule, params)?.nao;
    // A gradient job holds an SCF DIIS history, Fock/density work matrices and
    // pair-integral storage. This conservative dense-matrix estimate limits the
    // number of simultaneous jobs; nested faer work shares the same local pool.
    let bytes_per_job = checked_workspace_bytes(&[48, nao, nao, std::mem::size_of::<f64>()]);
    let workers = hessian_worker_count(options, ndof, bytes_per_job);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|e| crate::error::Pm3Error::InvalidInput(format!("rayon pool: {e}")))?;

    // Column j = (grad(+step·e_j) − grad(−step·e_j)) / (2·step); columns are independent.
    let columns: Vec<Result<Vec<f64>>> = pool.install(|| {
        (0..ndof)
            .into_par_iter()
            .map(|j| {
                let (atom, k) = (j / 3, j % 3);
                let mut plus = molecule.clone();
                let mut minus = molecule.clone();
                displace(&mut plus.atoms[atom].position, k, step);
                displace(&mut minus.atoms[atom].position, k, -step);
                let gp = closed_form_gradient(&plus, params, options)?;
                let gm = closed_form_gradient(&minus, params, options)?;
                let mut col = vec![0.0; ndof];
                for a in 0..nat {
                    for c in 0..3 {
                        let idx = 3 * a + c;
                        col[idx] = (component(&gp.gradient[a], c) - component(&gm.gradient[a], c))
                            / (2.0 * step);
                    }
                }
                Ok(col)
            })
            .collect()
    });

    let mut h = Matrix::zeros(ndof, ndof);
    for (j, col) in columns.into_iter().enumerate() {
        let col = col?;
        for (i, &v) in col.iter().enumerate() {
            h[(i, j)] = v;
        }
    }
    // Symmetrize in place, avoiding a third `ndof × ndof` allocation.
    symmetrize_average_in_place(&mut h);
    Ok(h)
}

/// Harmonic vibrational analysis at the given geometry (should be a stationary point).
pub fn vibrational_analysis(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    step: f64,
) -> Result<VibrationalModes> {
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    // CPHF analytic Hessian (no SCF re-runs); UHF falls back to finite differences internally.
    let hessian = analytic_hessian(molecule, params, options, step)?;

    // Mass-weight: H'_ij = H_ij / sqrt(m_i m_j), converting eV/Bohr² → eV/Å².
    let a0_sq = crate::constants::ANGSTROM_TO_BOHR * crate::constants::ANGSTROM_TO_BOHR;
    let masses: Vec<f64> = molecule
        .atoms
        .iter()
        .map(|a| params.element(a.z).map(|e| e.mass))
        .collect::<Result<Vec<_>>>()?;
    let mass_of = |dof: usize| masses[dof / 3];
    let mut mw = Matrix::zeros(ndof, ndof);
    for i in 0..ndof {
        for j in 0..ndof {
            let mij = (mass_of(i) * mass_of(j)).sqrt();
            // A massless atom leaves the row and column at zero rather than at infinity.
            // MOPAC's special codes `+` (Z = 104) and `--` (Z = 106) are point charges with a
            // tabulated mass of exactly 0.000, `symbol_to_z` accepts both from a plain XYZ line,
            // and `tests/api_surface.rs` asserts they are supported — so this divided by zero on
            // an input the crate advertises. The two sibling mass-weightings
            // (`pbc::hessian::frequencies` and `pbc::dfpt::frequencies_of`) have always guarded
            // it this way; only this one did not.
            mw[(i, j)] = if mij > 0.0 {
                hessian[(i, j)] * a0_sq / mij // eV/(Å²·amu)
            } else {
                0.0
            };
        }
    }
    let (eigs, modes) = symmetric_eigen(&mw)?;
    let frequencies_cm: Vec<f64> = eigs
        .iter()
        .map(|&lam| {
            if lam >= 0.0 {
                SQRT_EV_PER_ANG2_AMU_TO_CM * lam.sqrt()
            } else {
                -SQRT_EV_PER_ANG2_AMU_TO_CM * (-lam).sqrt()
            }
        })
        .collect();

    Ok(VibrationalModes {
        hessian,
        frequencies_cm,
        eigenvalues: eigs,
        modes,
    })
}

/// **Analytic (CPHF) Cartesian Hessian** (eV/Bohr²), robust to axis-aligned d/sparkle geometries.
///
/// Thin wrapper over [`analytic_hessian_core`]. The d/sparkle two-center rotation
/// ([`crate::rotations::Rotation`]) parametrizes the diatomic frame by a polar/azimuthal angle
/// pair that is singular when a bond lies on the global **z-axis** (`sqb = √(x²+y²) → 0`, the
/// azimuth undefined): the analytic *second* derivatives of that one pair lose accuracy within
/// ~`1e-3°` of the axis, and the value-only degenerate branch zeroes them exactly on it. Since
/// the PM3 energy is rotationally invariant, we detect a near-z d/sparkle pair, rotate the whole
/// molecule into a generic frame (every such bond then well off the z-axis, where the analytic
/// path is correct to ~`1e-7`), evaluate the core Hessian there, and rotate it back **exactly**
/// via `H = Qᵀ H_rot Q`, `Q = blockdiag(R0)`. Inputs with no near-axis d/sparkle pair skip all of
/// this and are bit-identical to [`analytic_hessian_core`].
pub fn analytic_hessian(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    step: f64,
) -> Result<Matrix> {
    // A second derivative needs a tightly-converged stationary density (Brillouin F_ov ≈ 0);
    // the energy/gradient default tolerances are looser than that.
    let options = &tighten_scf_for_hessian(options);
    // MOPAC's capped bond is intentionally non-variational: its very large
    // resonance term shapes the SCF density but is removed from the reported
    // energy and pair gradient. Differentiate that gradient directly.
    if molecule.atoms.iter().any(|atom| atom.z == 102) {
        return numerical_hessian(molecule, params, options, step.max(1.0e-3));
    }
    // A d/sparkle bond on the global z-axis makes that pair's rotation singular; evaluate in a
    // generic frame and rotate the Hessian back (see [`crate::frame`]). No near-axis pair ⇒
    // bit-identical to the core routine.
    if !crate::frame::needs_generic_frame(molecule, params)? {
        return analytic_hessian_core(molecule, params, options, step);
    }
    match crate::frame::pick_generic_frame(molecule, params)? {
        Some(r0) => {
            let rotated = crate::frame::rotate_molecule(molecule, &r0);
            let h_rot = analytic_hessian_core(&rotated, params, options, step)?;
            Ok(crate::frame::back_rotate_hessian(&h_rot, &r0))
        }
        // No candidate frame cleared every d-pair off the z-axis (extremely unlikely):
        // fall back to the collinear-safe numerical Hessian.
        None => numerical_hessian(molecule, params, options, step.max(1.0e-3)),
    }
}

/// **Analytic (CPHF) Cartesian Hessian** (eV/Bohr²).
///
/// `H_ab = E^(2,skel)_ab + Σ_μν F^a_μν (∂P/∂R_b)_μν`, where:
///
/// * the **skeleton** (fixed-density) second derivative `E^(2,skel)` is computed in **closed
///   form** by second-order forward-mode automatic differentiation ([`crate::dual2::Dual2`])
///   of the two-center integral kernels — resonance `β·S`, electron–core attraction, the
///   Dewar–Sabelli–Klopman two-electron integrals, and the PM3 core–core repulsion — with **no
///   finite differences**; and
/// * the density response `∂P/∂R_b` solves the coupled-perturbed (CPHF) equations, whose kernel
///   is the **orbital Hessian** (the same object a second-order SOSCF would use). This is done
///   entirely in the compact MO occupied–virtual subspace (`H_relax[a][b] = 4 G^a·U^b`), so the
///   working set is `O(ndof · n_occ · n_vir)` — no dense `ndof × nao²` derivative-Fock or
///   response intermediates are ever materialized (memory-lean and rayon-parallel over DOFs).
///
/// Fully analytic for both closed-shell RHF and open-shell UHF (the latter via [`analytic_hessian_uhf`]
/// with coupled α/β CPHF), across all valence shells (`n ≤ 3` analytic kernel, `n ≥ 4` via AD
/// through the numerical overlap quadrature). No finite differences.
fn analytic_hessian_core(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    step: f64,
) -> Result<Matrix> {
    use crate::dual2::Dual2;

    let scf = crate::scf::run_pm3(molecule, params, options)?;
    if scf.unrestricted {
        // The open-shell (UHF) skeleton/response is still s/p-only; d-containing
        // open-shell systems use the numerical Hessian (central differences of the
        // analytic gradient). Closed-shell d systems take the analytic path below.
        if crate::gradient::has_d_atom(molecule, params)? {
            return numerical_hessian(molecule, params, options, step.max(1.0e-3));
        }
        let _ = step; // UHF path is fully analytic (no finite-difference step)
        return analytic_hessian_uhf(molecule, params, options, &scf);
    }

    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let basis = crate::basis::Basis::build(molecule, params)?;
    let core = crate::hamiltonian::build_core_limited(
        molecule,
        &basis,
        params,
        crate::scf::pair_cache_limit(options),
        options.field,
    )?;
    let p = scf.density.clone();
    let c = scf.mo_coeff.clone();
    let eps = scf.mo_energies.clone();
    let n_occ = scf.n_occ;

    // 1) Skeleton (fixed-density) second derivative — fully analytic via second-order AD
    //    (Dual2) of each two-center pair's energy contribution E_pair(R_ab). Since E_pair
    //    depends only on the displacement R_ab = R_b − R_a, its 3×3 Hessian block scatters as
    //    +H onto the (a,a) and (b,b) diagonal blocks and −H onto the (a,b)/(b,a) blocks.
    let mut hess = Matrix::zeros(ndof, ndof);
    let pairs: Vec<(usize, usize)> = (0..nat)
        .flat_map(|u| ((u + 1)..nat).map(move |v| (u, v)))
        .collect();
    let blocks: Vec<Result<(usize, usize, [[f64; 3]; 3])>> = {
        use crate::gradient::resonance_beta;
        use rayon::prelude::*;
        pairs
            .par_iter()
            .map(|&(u, v)| -> Result<(usize, usize, [[f64; 3]; 3])> {
                let eu = params.element(molecule.atoms[u].z)?;
                let ev = params.element(molecule.atoms[v].z)?;
                // d-containing / sparkle pairs use the spd Dual2 path (a,b)=(u,v);
                // pure s/p pairs keep the fast sp path with the p-orientation swap.
                let point_pair = eu.n_orb == 0 || ev.n_orb == 0;
                let d_pair = eu.has_d() || ev.has_d();
                let (a, b) = if point_pair {
                    if eu.n_orb > 0 {
                        (u, v)
                    } else {
                        (v, u)
                    }
                } else if d_pair || eu.has_p() || !ev.has_p() {
                    (u, v)
                } else {
                    (v, u)
                };
                let ea = params.element(molecule.atoms[a].z)?;
                let eb = params.element(molecule.atoms[b].z)?;
                let (pa, pb) = (molecule.atoms[a].position, molecule.atoms[b].position);
                let dvec = [
                    Dual2::var(pb.x - pa.x, 0),
                    Dual2::var(pb.y - pa.y, 1),
                    Dual2::var(pb.z - pa.z, 2),
                ];
                let (te, s): (crate::integrals::PairTwoElecG<Dual2>, [[Dual2; 9]; 9]) =
                    if point_pair {
                        (
                            crate::integrals::pair_with_point_core_g::<Dual2>(ea, eb, dvec),
                            [[Dual2::constant(0.0); 9]; 9],
                        )
                    } else if d_pair {
                        (
                            crate::integrals_d::pair_two_electron_spd::<Dual2>(ea, eb, dvec),
                            crate::overlap::diatom_overlap_spd::<Dual2>(ea, eb, dvec),
                        )
                    } else {
                        (
                            crate::integrals::pair_two_electron_g::<Dual2>(ea, eb, dvec),
                            crate::overlap::embed4_g(crate::overlap::diatom_overlap_dual2(
                                ea, pa, eb, pb,
                            )?),
                        )
                    };
                let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
                let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);

                let mut epair = Dual2::constant(0.0);
                // Resonance β·S energy (both μν and νμ orderings → factor (β_i+β_j)).
                for i in 0..na {
                    let bi = resonance_beta(ea, basis.aos[oa + i].orb);
                    for j in 0..nb {
                        let bj = resonance_beta(eb, basis.aos[ob + j].orb);
                        let coef = p[(oa + i, ob + j)] * (bi + bj);
                        epair = epair + s[i][j] * coef;
                    }
                }
                // Electron–core attraction.
                for i in 0..na {
                    for j in 0..na {
                        epair = epair + te.e1b[i][j] * p[(oa + i, oa + j)];
                    }
                }
                for k in 0..nb {
                    for l in 0..nb {
                        epair = epair + te.e2a[k][l] * p[(ob + k, ob + l)];
                    }
                }
                // Two-electron Coulomb (J) + exchange (K), fixed density.
                for mu in 0..na {
                    for nu in 0..na {
                        for la in 0..nb {
                            for si in 0..nb {
                                let coul = p[(oa + mu, oa + nu)] * p[(ob + la, ob + si)];
                                let exch = -0.5 * p[(oa + mu, ob + la)] * p[(oa + nu, ob + si)];
                                epair = epair + te.two_e(mu, nu, la, si) * (coul + exch);
                            }
                        }
                    }
                }
                // Core–core repulsion (function of |R_ab|).
                let r = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
                epair = epair
                    + crate::repulsion::pair_core_energy_scalar::<Dual2>(
                        params,
                        ea,
                        eb,
                        molecule.atoms[a].z,
                        molecule.atoms[b].z,
                        r,
                    );
                Ok((a, b, epair.h))
            })
            .collect()
    };
    for blk in blocks {
        let (a, b, hb) = blk?;
        for i in 0..3 {
            for j in 0..3 {
                let val = hb[i][j];
                hess[(3 * a + i, 3 * a + j)] += val;
                hess[(3 * b + i, 3 * b + j)] += val;
                hess[(3 * a + i, 3 * b + j)] -= val;
                hess[(3 * b + i, 3 * a + j)] -= val;
            }
        }
    }

    // 2+3) Orbital-relaxation (CPHF) term in the compact MO occupied–virtual subspace:
    //   H_relax[a][b] = 4 Σ_{ov} G^a_{ov} U^b_{ov},
    // where G^t is the skeleton derivative Fock projected to the occ–virt block (n_vir × n_occ)
    // and U^b solves the coupled-perturbed equations against the orbital Hessian. Keeping
    // everything in the n_vir × n_occ block (never ndof × nao²) makes this both fast and
    // memory-lean: the response density is formed by matrix products (O(nao²·n_occ)), not the
    // O(n_occ·n_vir·nao²) outer-product loop, and no 3N full Fock/response matrices are stored.
    let nvir = basis.nao - n_occ;
    if nvir > 0 && n_occ > 0 {
        let cv = submatrix_cols(&c, n_occ, nvir); // virtual MOs, nao × n_vir
        let co = submatrix_cols(&c, 0, n_occ); // occupied MOs, nao × n_occ
        let denom = ov_denominators(&eps, n_occ, nvir); // ε_i − ε_a, n_vir × n_occ

        // Skeleton derivative Fock ov-blocks, built one atom at a time (peak memory O(nao²)).
        let t_gov = std::time::Instant::now();
        let gov = skeleton_fock_ov(molecule, params, options, &basis, &p, &cv, &co)?;
        if std::env::var("PM3_TIMING").is_ok() {
            eprintln!(
                "[timing]   skeleton gov: {:.2}s",
                t_gov.elapsed().as_secs_f64()
            );
        }

        // Optional perturbation-local CPHF cutoff (smooth switch; None = exact/full).
        let cutoff = options.hessian_cutoff.or_else(|| {
            std::env::var("PM3_HESS_CUTOFF")
                .ok()
                .and_then(|v| v.parse().ok())
        });
        let atom_pairs = cutoff.map(|_| build_atom_pairs(&core, molecule.atoms.len()));

        // CPHF response + relaxation Hessian, streamed in perturbation chunks so
        // the O(N³) set of 3N response ov-blocks `U^b` is never all resident at
        // once (OOM guard for large systems): solve a chunk in parallel, contract
        // `H_relax[a][b] = 4 G^a : U^b` against all `G^a`, then drop the chunk.
        use rayon::prelude::*;
        let ov = nvir.saturating_mul(n_occ);
        let resident_bytes = checked_workspace_bytes(&[ndof, ov, std::mem::size_of::<f64>()]);
        // One active CPHF solve holds AO response/Fock work matrices plus its
        // compact DIIS history; include both in the concurrency estimate.
        let per_response = checked_workspace_bytes(&[
            8usize,
            6usize
                .saturating_mul(basis.nao.saturating_mul(basis.nao))
                .saturating_add(20usize.saturating_mul(ov)),
        ]);
        let chunk = cphf_chunk_size(
            options,
            "RHF analytic Hessian",
            ndof,
            resident_bytes,
            per_response,
        )?;
        let mut b0 = 0;
        while b0 < ndof {
            let b1 = (b0 + chunk).min(ndof);
            let uov_chunk: Vec<Matrix> = (b0..b1)
                .into_par_iter()
                .map(|b| match (cutoff, atom_pairs.as_ref()) {
                    (Some(r_off), Some(ap)) => cphf_ov_local(
                        b / 3,
                        &gov[b],
                        &denom,
                        &cv,
                        &co,
                        molecule,
                        params,
                        &basis,
                        &core,
                        ap,
                        r_off,
                    ),
                    _ => cphf_ov(
                        &gov[b],
                        &denom,
                        &cv,
                        &co,
                        molecule,
                        params,
                        &basis,
                        &core,
                        None,
                        CPHF_ITERATIONS,
                    ),
                })
                .collect::<Result<Vec<_>>>()?;
            let block: Vec<Vec<f64>> = (0..ndof)
                .into_par_iter()
                .map(|a| {
                    uov_chunk
                        .iter()
                        .map(|ub| 4.0 * gov[a].frobenius_dot(ub))
                        .collect()
                })
                .collect();
            for (a, row) in block.into_iter().enumerate() {
                for (j, v) in row.into_iter().enumerate() {
                    hess[(a, b0 + j)] += v;
                }
            }
            b0 = b1;
        }
        if std::env::var("PM3_TIMING").is_ok() {
            eprintln!(
                "[timing] CPHF (summed/threads): transform={:.1}s fock={:.1}s proj={:.1}s iters={} n_loc_max={}",
                T_TRANSFORM_US.swap(0, Ordering::Relaxed) as f64 / 1e6,
                T_FOCK_US.swap(0, Ordering::Relaxed) as f64 / 1e6,
                T_PROJ_US.swap(0, Ordering::Relaxed) as f64 / 1e6,
                N_CPHF_ITER.swap(0, Ordering::Relaxed),
                N_LOC_AOS.swap(0, Ordering::Relaxed),
            );
        }
    }

    // Symmetrize in place to keep peak Hessian storage bounded.
    symmetrize_average_in_place(&mut hess);
    // Classical D3/H4/X correction second derivatives (variant-dependent; zero for plain PM3).
    let ch = correction_hessian(molecule, options.variant);
    for i in 0..ndof {
        for j in 0..ndof {
            hess[(i, j)] += ch[(i, j)];
        }
    }
    Ok(hess)
}

/// Copy `count` columns of `c` starting at `start` into a fresh `nao × count` matrix.
pub(crate) fn submatrix_cols(c: &Matrix, start: usize, count: usize) -> Matrix {
    let nao = c.rows;
    let mut m = Matrix::zeros(nao, count);
    for mu in 0..nao {
        for k in 0..count {
            m[(mu, k)] = c[(mu, start + k)];
        }
    }
    m
}

/// Orbital-energy denominators `ε_i − ε_a` (occupied `i`, virtual `a`), as an `n_vir × n_occ`
/// matrix — the diagonal of the uncoupled orbital Hessian.
pub(crate) fn ov_denominators(eps: &[f64], n_occ: usize, nvir: usize) -> Matrix {
    let mut d = Matrix::zeros(nvir, n_occ);
    for a in 0..nvir {
        for i in 0..n_occ {
            d[(a, i)] = eps[i] - eps[n_occ + a];
        }
    }
    d
}

/// Project an AO-basis matrix `f` onto the MO occupied–virtual block `Cvᵀ F Co` (n_vir × n_occ).
pub(crate) fn project_ov(f: &Matrix, cv: &Matrix, co: &Matrix) -> Matrix {
    // Sequential: called per-perturbation inside the parallel CPHF loop.
    let m = f.matmul_seq(co); // nao × n_occ
    cv.transpose_matmul_seq(&m) // n_vir × n_occ
}

/// Skeleton derivative Fock, projected to the MO occ–virt block, one entry per Cartesian DOF.
///
/// Built **one atom at a time**: for atom `c` its three axis-derivative Fock matrices are
/// accumulated from the pairs `{c, x}` and immediately projected to the compact `n_vir × n_occ`
/// block, so peak memory is `O(nao²)` (a few transient matrices per thread) rather than
/// `O(ndof · nao²)`. Each pair's dual integrals are evaluated twice overall (once per endpoint),
/// a negligible cost next to the CPHF solve.
pub(crate) fn skeleton_fock_ov(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    basis: &crate::basis::Basis,
    p: &Matrix,
    cv: &Matrix,
    co: &Matrix,
) -> Result<Vec<Matrix>> {
    use crate::gradient::{pair_dual, resonance_beta};
    use rayon::prelude::*;
    let nat = molecule.atoms.len();
    let nao = basis.nao;

    // Per atom: the three projected ov-blocks (x, y, z). Limit simultaneous
    // dense AO work matrices according to the Hessian memory budget.
    let bytes_per_atom = checked_workspace_bytes(&[3, nao, nao, std::mem::size_of::<f64>()]);
    let workers = hessian_worker_count(options, nat, bytes_per_atom);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|e| crate::error::Pm3Error::InvalidInput(format!("rayon pool: {e}")))?;
    let per_atom: Vec<Result<[Matrix; 3]>> = pool.install(|| {
        (0..nat)
            .into_par_iter()
            .map(|c| -> Result<[Matrix; 3]> {
                let mut fmat = [
                    Matrix::zeros(nao, nao),
                    Matrix::zeros(nao, nao),
                    Matrix::zeros(nao, nao),
                ];

                // The external field's derivative. `H_core` carries `+M·f`, whose only geometry
                // is each atom's own monopole position, so `∂(M·f)[μ_c,μ_c]/∂R_{c,axis} = f_axis`
                // and every other entry is constant. Small, one-center, and easy to lose: the
                // loop below is over *pairs*, so a field term added anywhere else would never
                // reach the response, and the Hessian would quietly drop the cross term
                // `Tr[(∂P/∂R)(∂H'/∂R)]` while still looking like a converged analytic result.
                if let Some(field) = options.field {
                    let offset = basis.atom_offset[c];
                    let norb = basis.atom_norb[c];
                    for (axis, component) in field.to_array().iter().enumerate() {
                        for mu in 0..norb {
                            fmat[axis][(offset + mu, offset + mu)] += component;
                        }
                    }
                }

                for x in 0..nat {
                    if x == c {
                        continue;
                    }
                    let (u, v) = (c.min(x), c.max(x));
                    let (a, b, te, s) = pair_dual(molecule, params, u, v)?;
                    let ea = params.element(molecule.atoms[a].z)?;
                    let eb = params.element(molecule.atoms[b].z)?;
                    // E_pair depends on R_ab = R_b − R_a; ∂/∂R_c = +∂/∂R_ab if c==b, else −.
                    let sign = if c == b { 1.0 } else { -1.0 };
                    let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
                    let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);

                    for axis in 0..3 {
                        let fm = &mut fmat[axis];
                        // Resonance β·S.
                        for i in 0..na {
                            let bi = resonance_beta(ea, basis.aos[oa + i].orb);
                            for j in 0..nb {
                                let bj = resonance_beta(eb, basis.aos[ob + j].orb);
                                let val = sign * 0.5 * (bi + bj) * s[i][j].d[axis];
                                fm[(oa + i, ob + j)] += val;
                                fm[(ob + j, oa + i)] += val;
                            }
                        }
                        // Electron–core attraction.
                        for i in 0..na {
                            for j in 0..na {
                                fm[(oa + i, oa + j)] += sign * te.e1b[i][j].d[axis];
                            }
                        }
                        for k in 0..nb {
                            for l in 0..nb {
                                fm[(ob + k, ob + l)] += sign * te.e2a[k][l].d[axis];
                            }
                        }
                        // Two-electron Coulomb (J).
                        for mu in 0..na {
                            for nu in 0..na {
                                let mut acc = 0.0;
                                for la in 0..nb {
                                    for si in 0..nb {
                                        acc += p[(ob + la, ob + si)]
                                            * te.two_e(mu, nu, la, si).d[axis];
                                    }
                                }
                                fm[(oa + mu, oa + nu)] += sign * acc;
                            }
                        }
                        for la in 0..nb {
                            for si in 0..nb {
                                let mut acc = 0.0;
                                for mu in 0..na {
                                    for nu in 0..na {
                                        acc += p[(oa + mu, oa + nu)]
                                            * te.two_e(mu, nu, la, si).d[axis];
                                    }
                                }
                                fm[(ob + la, ob + si)] += sign * acc;
                            }
                        }
                        // Two-electron exchange (K).
                        for mu in 0..na {
                            for la in 0..nb {
                                let mut acc = 0.0;
                                for nu in 0..na {
                                    for si in 0..nb {
                                        acc += p[(oa + nu, ob + si)]
                                            * te.two_e(mu, nu, la, si).d[axis];
                                    }
                                }
                                let val = sign * (-0.5 * acc);
                                fm[(oa + mu, ob + la)] += val;
                                fm[(ob + la, oa + mu)] += val;
                            }
                        }
                    }
                }
                Ok([
                    project_ov(&fmat[0], cv, co),
                    project_ov(&fmat[1], cv, co),
                    project_ov(&fmat[2], cv, co),
                ])
            })
            .collect()
    });

    let mut gov: Vec<Matrix> = Vec::with_capacity(3 * nat);
    for res in per_atom {
        let [gx, gy, gz] = res?;
        gov.push(gx);
        gov.push(gy);
        gov.push(gz);
    }
    Ok(gov)
}

/// AO-basis first-order density response from the MO occ–virt response coefficients `u`
/// (n_vir × n_occ): `R = Cv (w·U) Coᵀ + Co (w·U)ᵀ Cvᵀ`, built by matrix products
/// (O(nao²·n_occ)). The occupation weight `w` is 2 for RHF (spin-summed) and 1 for a single UHF
/// spin channel.
fn ao_response_density_w(u: &Matrix, cv: &Matrix, co: &Matrix, weight: f64) -> Matrix {
    let mut uw = u.clone();
    for x in uw.as_mut_slice() {
        *x *= weight;
    }
    // Sequential: called per-perturbation inside the parallel CPHF loop.
    let a = cv.matmul_seq(&uw); // nao × n_occ
    let mut response = a.matmul_transpose_seq(co); // nao × nao
    add_transpose_in_place(&mut response);
    response
}

/// RHF response density (occupation weight 2).
fn ao_response_density(u: &Matrix, cv: &Matrix, co: &Matrix) -> Matrix {
    ao_response_density_w(u, cv, co, 2.0)
}

/// C² quintic switching function: `1` for `r ≤ r_on`, `0` for `r ≥ r_off`, smooth
/// (value + first + second derivative continuous) between — so the cutoff never
/// introduces a kink in the Hessian as atoms cross the boundary.
fn switch_fn(r: f64, r_on: f64, r_off: f64) -> f64 {
    if r <= r_on {
        1.0
    } else if r >= r_off {
        0.0
    } else {
        let x = (r - r_on) / (r_off - r_on);
        1.0 - (10.0 * x.powi(3) - 15.0 * x.powi(4) + 6.0 * x.powi(5))
    }
}

/// Solve the CPHF equations for one perturbation entirely in the MO occ–virt block: iterate
/// `U = (G_skel + [G(∂P(U))]_ov) / (ε_i − ε_a)` to self-consistency. `G(∂P) = F(∂P) − H_core`
/// is the two-electron response Fock (the orbital-Hessian coupling); the fixed point is the
/// coupled response. Returns the converged `U` (n_vir × n_occ), or
/// [`crate::error::Pm3Error::ScfNotConverged`] if it did not converge.
///
/// `max_iter` is a parameter rather than the constant it always is in production, so that the
/// failure path can be reached from a test. A cap that cannot be lowered is a branch that cannot
/// be exercised, and this one shipped returning a partially converged response for months.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cphf_ov(
    g_ov: &Matrix,
    denom: &Matrix,
    cv: &Matrix,
    co: &Matrix,
    molecule: &Molecule,
    params: &Pm3Parameters,
    basis: &crate::basis::Basis,
    core: &crate::hamiltonian::CoreHamiltonian,
    pair_sw: Option<&[f64]>,
    max_iter: usize,
) -> Result<Matrix> {
    // Uncoupled start: U0 = G / (ε_i − ε_a).
    let elem_div = |num: &Matrix| -> Matrix {
        let mut u = num.clone();
        for (uv, dv) in u.as_mut_slice().iter_mut().zip(denom.as_slice()) {
            *uv = if dv.abs() < 1.0e-10 { 0.0 } else { *uv / *dv };
        }
        u
    };
    let prof = std::env::var("PM3_TIMING").is_ok();
    let mut u = elem_div(g_ov);
    // DIIS (Pulay) acceleration of the fixed-point iteration — same converged
    // response, far fewer iterations (semiempirical CPHF is stiff otherwise).
    let mut hist_u: Vec<Matrix> = Vec::new();
    let mut hist_e: Vec<Matrix> = Vec::new();
    let max_diis = 8;
    let mut converged = false;
    let mut residual = f64::INFINITY;
    // `PM3_CPHF_TRACE=1` prints the residual per pass, the way `PM3_DFPT_TRACE` does for the
    // periodic response. Without it there was no way to tell a fixed point that is converging
    // slowly from one that has stalled, which is exactly the question the iteration cap turns on.
    let trace = std::env::var_os("PM3_CPHF_TRACE").is_some();
    let mut pass = 0usize;
    for _ in 0..max_iter {
        if prof {
            N_CPHF_ITER.fetch_add(1, Ordering::Relaxed);
        }
        let t0 = std::time::Instant::now();
        let r = ao_response_density(&u, cv, co);
        if prof {
            T_TRANSFORM_US.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        let t1 = std::time::Instant::now();
        // Response two-electron operator G(∂P) = J(∂P) − K(½∂P), no H_core, with
        // the optional per-pair distance switch. pair_sw = None → exact.
        let g_resp_full =
            crate::fock::response_fock_rhf(molecule, basis, params, core, &r, pair_sw)?;
        if prof {
            T_FOCK_US.fetch_add(t1.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        let g_resp = project_ov(&g_resp_full, cv, co);
        let mut rhs = g_ov.clone();
        for (rv, gv) in rhs.as_mut_slice().iter_mut().zip(g_resp.as_slice()) {
            *rv += *gv;
        }
        let u_new = elem_div(&rhs);
        // Residual of the fixed-point map.
        let mut err = u_new.clone();
        let mut diff = 0.0;
        for (ev, ov) in err.as_mut_slice().iter_mut().zip(u.as_slice()) {
            *ev -= *ov;
            diff += *ev * *ev;
        }
        residual = diff.sqrt();
        if trace {
            eprintln!("  cphf iter {pass:4} residual {residual:.6e}");
        }
        pass += 1;
        if residual < CPHF_TOLERANCE {
            u = u_new;
            converged = true;
            break;
        }
        // DIIS extrapolation: u ← Σ c_i u_new_i minimising Σ c_i e_i, Σ c_i = 1.
        hist_u.push(u_new.clone());
        hist_e.push(err);
        if hist_u.len() > max_diis {
            hist_u.remove(0);
            hist_e.remove(0);
        }
        match cphf_diis(&hist_u, &hist_e) {
            Some(mixed) => u = mixed,
            None => {
                if trace {
                    eprintln!(
                        "  cphf iter {pass:4} DIIS declined (history {})",
                        hist_u.len()
                    );
                }
                u = u_new;
            }
        }
    }
    // A fixed point that ran out of passes has not produced a response, and this used to return
    // whatever it was holding. The caller cannot tell: a partially converged `U` gives a Hessian
    // with plausible-looking frequencies, an infrared spectrum with plausible-looking
    // intensities, and no diagnostic anywhere. `pbc::dfpt`'s response already refuses in exactly
    // this situation; these three did not.
    if !converged {
        return Err(Pm3Error::ScfNotConverged {
            iterations: max_iter,
            error: residual,
        });
    }
    Ok(u)
}

/// Pulay DIIS extrapolation for the CPHF fixed point: solve `B c = [0,−1]`
/// (`B_ij = ⟨e_i,e_j⟩`, normalised + ridge-regularised so the near-linearly-dependent
/// residuals of a stiff fixed point stay solvable) and return `Σ c_i u_i`.
pub(crate) fn cphf_diis(hist_u: &[Matrix], hist_e: &[Matrix]) -> Option<Matrix> {
    // One channel, expressed through the many-channel form so there is a single implementation
    // of the B matrix, the ridge and the renormalisation to extrapolate against.
    let u: Vec<Vec<Matrix>> = hist_u.iter().map(|m| vec![m.clone()]).collect();
    let e: Vec<Vec<Matrix>> = hist_e.iter().map(|m| vec![m.clone()]).collect();
    cphf_diis_channels(&u, &e).map(|mut c| c.remove(0))
}

/// The same extrapolation over **several coupled channels at once**.
///
/// The α and β halves of an unrestricted response are one fixed point, not two: they couple
/// through the total response density in the Coulomb term. Extrapolating them separately would
/// build each channel's coefficients from a residual that ignores the other, which is a different
/// (and worse-conditioned) problem. So the inner product runs over the whole list and one set of
/// coefficients is applied to every channel — the same shape
/// [`crate::pbc::dfpt`]'s response extrapolation uses for its per-k blocks.
fn cphf_diis_channels(hist_u: &[Vec<Matrix>], hist_e: &[Vec<Matrix>]) -> Option<Vec<Matrix>> {
    let m = hist_u.len();
    if m < 2 {
        return None;
    }
    let dot = |a: &[Matrix], b: &[Matrix]| -> f64 {
        a.iter().zip(b).map(|(x, y)| x.frobenius_dot(y)).sum()
    };
    // The coefficients come from [`crate::scf::diis_coeffs_from_gram`] — the same solve the
    // ground-state SCF and the periodic response already use — rather than from a second
    // implementation here.
    //
    // The second implementation was wrong, and wrong in a way that only a trace would show.
    // Pulay's problem is "minimise `‖Σ cᵢ eᵢ‖²` subject to `Σ cᵢ = 1`", and it divided `B_ij`
    // by `sᵢ·sⱼ`, the two residuals' *own* magnitudes. That is not a conditioning transform:
    // it turns the constraint into `Σ dᵢ/sᵢ = 1` in the rescaled variables, and rescaling the
    // answer afterwards does not put it back. The shared version divides the whole matrix by
    // one scalar, which leaves the minimiser untouched — and says so in its own comment.
    //
    // What the bespoke copy cost: on water the residual fell for the eight passes it took to
    // fill the history and then settled onto a constant ratio of 0.892 per pass, which is
    // exactly the unaccelerated fixed point's spectral radius — a hundred and sixty passes to
    // cross eight decades, from a method whose entire purpose is not to do that. On the shared
    // solve the same response converges in **three**.
    let gram: Vec<Vec<f64>> = (0..m)
        .map(|i| (0..m).map(|j| dot(&hist_e[i], &hist_e[j])).collect())
        .collect();
    let c = crate::scf::diis_coeffs_from_gram(&gram)?;
    let mut out: Vec<Matrix> = hist_u[0]
        .iter()
        .map(|u| crate::linalg::Matrix::zeros(u.rows, u.cols))
        .collect();
    for (ci, entry) in c.iter().zip(hist_u) {
        for (slot, ui) in out.iter_mut().zip(entry) {
            for (ov, uv) in slot.as_mut_slice().iter_mut().zip(ui.as_slice()) {
                *ov += *ci * *uv;
            }
        }
    }
    Some(out)
}

/// How many passes a coupled-perturbed fixed point takes before it has failed.
///
/// Headroom, not a working limit. Water's response reaches `1e-16` in **three** passes now that
/// [`cphf_diis`] uses the shared coefficient solve; four hundred is there for a genuinely stiff
/// system, and a converged solve never approaches it.
///
/// It is worth recording what this looked like before, because the cap was the symptom and the
/// extrapolation was the disease. The limit was one hundred, and the loop `Ok`'d whatever
/// iterate it was holding when that ran out — so the declared tolerance of `1e-9` was never the
/// tolerance the solver met. Water stopped at `9.4e-7`. Every analytic Hessian, infrared
/// intensity and dipole derivative this crate produced was built on a response about three
/// digits looser than it claimed, and nothing said so. Raising the cap alone would have bought
/// correctness at 160 passes; fixing the extrapolation bought it at three.
pub(crate) const CPHF_ITERATIONS: usize = 400;

/// Convergence on the RMS change of the response between passes.
const CPHF_TOLERANCE: f64 = 1.0e-9;

/// For each atom, the `(other_atom, core.pairs index)` of every two-center pair it
/// belongs to — lets the local CPHF gather an atom's neighbour pairs in O(1).
fn build_atom_pairs(
    core: &crate::hamiltonian::CoreHamiltonian,
    nat: usize,
) -> Vec<Vec<(usize, usize)>> {
    let mut ap = vec![Vec::new(); nat];
    for (idx, p) in core.pairs.iter().enumerate() {
        ap[p.a].push((p.b, idx));
        ap[p.b].push((p.a, idx));
    }
    ap
}

/// **Perturbation-LOCAL** two-electron response operator `G(∂P)` restricted to the
/// active atoms `S_P` (those within the cutoff of the perturbed atom), in the packed
/// local AO basis (`n_loc × n_loc`). Contracts `r_loc` over the one-center blocks and
/// the two-center pairs whose *both* atoms are active, neglecting the (screened, hence
/// negligible) coupling to distant atoms. With every atom active this equals the full
/// `response_fock_rhf`.
#[allow(clippy::too_many_arguments)]
fn local_fock(
    atoms: &[usize],
    loc_off: &[usize],
    atom_local: &[usize],
    n_loc: usize,
    r_loc: &Matrix,
    in_sp: &[bool],
    atom_pairs: &[Vec<(usize, usize)>],
    molecule: &Molecule,
    params: &Pm3Parameters,
    basis: &crate::basis::Basis,
    core: &crate::hamiltonian::CoreHamiltonian,
) -> Result<Matrix> {
    use crate::fock::oc_two_electron;
    let mut g = Matrix::zeros(n_loc, n_loc);
    for (li, &a) in atoms.iter().enumerate() {
        let elem = params.element(molecule.atoms[a].z)?;
        let n = basis.atom_norb[a];
        let lo = loc_off[li];
        let (gss, gsp, gpp, gp2, hsp) = (elem.g_ss, elem.g_sp, elem.g_pp, elem.g_p2, elem.h_sp);
        let oc = |i: usize, j: usize, k: usize, l: usize| -> f64 {
            if let Some(spd) = &elem.onecenter {
                spd.get(i, j, k, l)
            } else {
                oc_two_electron(i, j, k, l, gss, gsp, gpp, gp2, hsp)
            }
        };
        for mu in 0..n {
            for nu in 0..n {
                let mut acc = 0.0;
                for la in 0..n {
                    for si in 0..n {
                        acc += r_loc[(lo + la, lo + si)] * oc(mu, nu, la, si);
                        acc -= 0.5 * r_loc[(lo + la, lo + si)] * oc(mu, la, nu, si);
                    }
                }
                g[(lo + mu, lo + nu)] += acc;
            }
        }
    }
    for (li, &a) in atoms.iter().enumerate() {
        for &(b, idx) in &atom_pairs[a] {
            if !in_sp[b] {
                continue;
            }
            let pair = &core.pairs[idx];
            if a != pair.a {
                continue; // process each pair exactly once (from its `a` side)
            }
            let te = &pair.te;
            let (na, nb) = (te.norb_i, te.norb_j);
            let oa = loc_off[li];
            let ob = loc_off[atom_local[b]];
            for mu in 0..na {
                for nu in 0..na {
                    let mut acc = 0.0;
                    for la in 0..nb {
                        for si in 0..nb {
                            acc += r_loc[(ob + la, ob + si)] * te.two_e(mu, nu, la, si);
                        }
                    }
                    g[(oa + mu, oa + nu)] += acc;
                }
            }
            for la in 0..nb {
                for si in 0..nb {
                    let mut acc = 0.0;
                    for mu in 0..na {
                        for nu in 0..na {
                            acc += r_loc[(oa + mu, oa + nu)] * te.two_e(mu, nu, la, si);
                        }
                    }
                    g[(ob + la, ob + si)] += acc;
                }
            }
            for mu in 0..na {
                for la in 0..nb {
                    let mut acc = 0.0;
                    for nu in 0..na {
                        for si in 0..nb {
                            acc += 0.5 * r_loc[(oa + nu, ob + si)] * te.two_e(mu, nu, la, si);
                        }
                    }
                    let v = -acc;
                    g[(oa + mu, ob + la)] += v;
                    g[(ob + la, oa + mu)] = g[(oa + mu, ob + la)];
                }
            }
        }
    }
    Ok(g)
}

/// **Perturbation-local CPHF solve** for one perturbation (perturbed atom `p_atom`).
/// The whole AO round-trip (response density → response fock → project) is confined to
/// the active atoms within `r_off` of `p_atom` (a C² switch damps the boundary), so
/// each perturbation costs `O(neighbours·nvir·nocc)` rather than `O(nao²·nocc)` —
/// turning the Hessian from O(N⁴) toward O(N³). The MO occ–virt amplitude `u` stays
/// full. With `r_off` larger than the molecule it is bit-identical to [`cphf_ov`].
#[allow(clippy::too_many_arguments)]
fn cphf_ov_local(
    p_atom: usize,
    g_ov: &Matrix,
    denom: &Matrix,
    cv: &Matrix,
    co: &Matrix,
    molecule: &Molecule,
    params: &Pm3Parameters,
    basis: &crate::basis::Basis,
    core: &crate::hamiltonian::CoreHamiltonian,
    atom_pairs: &[Vec<(usize, usize)>],
    r_off: f64,
) -> Result<Matrix> {
    let nat = molecule.atoms.len();
    let r_on = (r_off - 2.0).max(0.0);
    let pos_p = molecule.atoms[p_atom].position;
    let mut atoms = Vec::new();
    let mut weight = Vec::new();
    let mut in_sp = vec![false; nat];
    let mut atom_local = vec![usize::MAX; nat];
    for a in 0..nat {
        let d = (molecule.atoms[a].position - pos_p).norm();
        let w = switch_fn(d, r_on, r_off);
        if w > 0.0 {
            atom_local[a] = atoms.len();
            in_sp[a] = true;
            atoms.push(a);
            weight.push(w);
        }
    }
    let mut loc_off = Vec::with_capacity(atoms.len());
    let mut global_aos = Vec::new();
    for &a in &atoms {
        loc_off.push(global_aos.len());
        let off = basis.atom_offset[a];
        for k in 0..basis.atom_norb[a] {
            global_aos.push(off + k);
        }
    }
    let n_loc = global_aos.len();
    let (nvir, nocc) = (cv.cols, co.cols);
    let mut cv_loc = Matrix::zeros(n_loc, nvir);
    let mut co_loc = Matrix::zeros(n_loc, nocc);
    for (li, &gao) in global_aos.iter().enumerate() {
        for a in 0..nvir {
            cv_loc[(li, a)] = cv[(gao, a)];
        }
        for i in 0..nocc {
            co_loc[(li, i)] = co[(gao, i)];
        }
    }
    // Per-local-AO switch weight (of its atom).
    let mut aow = vec![0.0; n_loc];
    for (li, _) in atoms.iter().enumerate() {
        let hi = if li + 1 < loc_off.len() {
            loc_off[li + 1]
        } else {
            n_loc
        };
        for k in loc_off[li]..hi {
            aow[k] = weight[li];
        }
    }

    let elem_div = |num: &Matrix| -> Matrix {
        let mut u = num.clone();
        for (uv, dv) in u.as_mut_slice().iter_mut().zip(denom.as_slice()) {
            *uv = if dv.abs() < 1.0e-10 { 0.0 } else { *uv / *dv };
        }
        u
    };
    let mut u = elem_div(g_ov);
    let mut hist_u: Vec<Matrix> = Vec::new();
    let mut hist_e: Vec<Matrix> = Vec::new();
    let max_diis = 8;
    let co_loc_t = co_loc.transpose();
    let cv_loc_t = cv_loc.transpose();
    let prof = std::env::var("PM3_TIMING").is_ok();
    let mut converged = false;
    let mut residual = f64::INFINITY;
    for _ in 0..CPHF_ITERATIONS {
        if prof {
            N_CPHF_ITER.fetch_add(1, Ordering::Relaxed);
            N_LOC_AOS.fetch_max(n_loc as u64, Ordering::Relaxed);
        }
        let t0 = std::time::Instant::now();
        let mut uw = u.clone();
        for x in uw.as_mut_slice() {
            *x *= 2.0;
        }
        let a_loc = cv_loc.matmul_seq(&uw); // n_loc × nocc
        let mut r_loc = a_loc.matmul_seq(&co_loc_t); // n_loc × n_loc
        add_transpose_in_place(&mut r_loc);
        for i in 0..n_loc {
            let wi = aow[i];
            for j in 0..n_loc {
                r_loc[(i, j)] *= wi * aow[j];
            }
        }
        if prof {
            T_TRANSFORM_US.fetch_add(t0.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        let t1 = std::time::Instant::now();
        let g_loc = local_fock(
            &atoms,
            &loc_off,
            &atom_local,
            n_loc,
            &r_loc,
            &in_sp,
            atom_pairs,
            molecule,
            params,
            basis,
            core,
        )?;
        if prof {
            T_FOCK_US.fetch_add(t1.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        let t2 = std::time::Instant::now();
        let m = g_loc.matmul_seq(&co_loc); // n_loc × nocc
        let g_resp = cv_loc_t.matmul_seq(&m); // nvir × nocc
        if prof {
            T_PROJ_US.fetch_add(t2.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        let mut rhs = g_ov.clone();
        for (rv, gv) in rhs.as_mut_slice().iter_mut().zip(g_resp.as_slice()) {
            *rv += *gv;
        }
        let u_new = elem_div(&rhs);
        let mut err = u_new.clone();
        let mut diff = 0.0;
        for (ev, ov) in err.as_mut_slice().iter_mut().zip(u.as_slice()) {
            *ev -= *ov;
            diff += *ev * *ev;
        }
        residual = diff.sqrt();
        if residual < CPHF_TOLERANCE {
            u = u_new;
            converged = true;
            break;
        }
        hist_u.push(u_new.clone());
        hist_e.push(err);
        if hist_u.len() > max_diis {
            hist_u.remove(0);
            hist_e.remove(0);
        }
        u = cphf_diis(&hist_u, &hist_e).unwrap_or(u_new);
    }
    if !converged {
        return Err(Pm3Error::ScfNotConverged {
            iterations: CPHF_ITERATIONS,
            error: residual,
        });
    }
    Ok(u)
}

/// **Analytic open-shell (UHF) Cartesian Hessian** (eV/Bohr²). Same structure as the RHF path
/// but spin-resolved: the skeleton second derivative uses same-spin exchange
/// `−[Pα_μλ Pα_νσ + Pβ_μλ Pβ_νσ]`, and the response solves the **coupled** α/β CPHF equations
/// (the α and β responses are coupled through the total-density Coulomb term). Everything stays
/// in the per-spin MO occ–virt blocks — memory-lean, rayon-parallel. No finite differences.
fn analytic_hessian_uhf(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    scf: &crate::scf::Pm3Result,
) -> Result<Matrix> {
    use crate::dual2::Dual2;
    use crate::fock::build_fock_spin;
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let basis = crate::basis::Basis::build(molecule, params)?;
    let nao = basis.nao;
    let core = crate::hamiltonian::build_core_limited(
        molecule,
        &basis,
        params,
        crate::scf::pair_cache_limit(options),
        options.field,
    )?;

    // Spin densities Pα = (P_tot + S)/2, Pβ = (P_tot − S)/2.
    let pt = scf.density.clone();
    let spin = scf.spin_density.as_ref().ok_or_else(|| {
        crate::error::Pm3Error::InvalidInput("UHF Hessian requires a spin density".into())
    })?;
    let mut pa = pt.clone();
    let mut pb = pt.clone();
    {
        let (pas, pbs) = (pa.as_mut_slice(), pb.as_mut_slice());
        let (pts, ss) = (pt.as_slice(), spin.as_slice());
        for i in 0..pts.len() {
            pas[i] = 0.5 * (pts[i] + ss[i]);
            pbs[i] = 0.5 * (pts[i] - ss[i]);
        }
    }
    // Recover both spin orbital sets by diagonalizing the converged spin Fock matrices.
    //
    // `Pm3Result` now carries the β set as well, but these are not the same orbitals: the stored
    // ones come from the last iteration's Fock, these from the Fock at the *converged* density.
    // The two agree to `p_tol`, and a second derivative wants the converged pair — `tighten_scf`
    // exists for the same reason. So this stays, and the field it made redundant is the electron
    // count below.
    let fa = build_fock_spin(molecule, &basis, params, &core, &pt, &pa)?;
    let fb = build_fock_spin(molecule, &basis, params, &core, &pt, &pb)?;
    let (eps_a, ca) = symmetric_eigen(&fa)?;
    let (eps_b, cb) = symmetric_eigen(&fb)?;
    let n_alpha = scf.n_occ;
    // Read, not reconstructed. `n_occ − (multiplicity − 1)` re-derives what the SCF already
    // resolved, and disagrees with it whenever the caller's `multiplicity` is not the one the
    // SCF actually used.
    let n_beta = scf.n_beta;

    // 1) Skeleton (fixed-density) second derivative — spin-resolved exchange.
    let mut hess = Matrix::zeros(ndof, ndof);
    let beta = |elem: &crate::params::Pm3Element, orb: u8| {
        if orb == 0 {
            elem.beta_s
        } else {
            elem.beta_p
        }
    };
    let pairs: Vec<(usize, usize)> = (0..nat)
        .flat_map(|u| ((u + 1)..nat).map(move |v| (u, v)))
        .collect();
    let blocks: Vec<Result<(usize, usize, [[f64; 3]; 3])>> = {
        use rayon::prelude::*;
        pairs
            .par_iter()
            .map(|&(u, v)| -> Result<(usize, usize, [[f64; 3]; 3])> {
                let eu = params.element(molecule.atoms[u].z)?;
                let ev = params.element(molecule.atoms[v].z)?;
                let (a, b) = if eu.has_p() || !ev.has_p() {
                    (u, v)
                } else {
                    (v, u)
                };
                let ea = params.element(molecule.atoms[a].z)?;
                let eb = params.element(molecule.atoms[b].z)?;
                let (posa, posb) = (molecule.atoms[a].position, molecule.atoms[b].position);
                let dvec = [
                    Dual2::var(posb.x - posa.x, 0),
                    Dual2::var(posb.y - posa.y, 1),
                    Dual2::var(posb.z - posa.z, 2),
                ];
                let te = crate::integrals::pair_two_electron_g::<Dual2>(ea, eb, dvec);
                let s = crate::overlap::diatom_overlap_dual2(ea, posa, eb, posb)?;
                let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
                let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);

                let mut epair = Dual2::constant(0.0);
                for i in 0..na {
                    let bi = beta(ea, basis.aos[oa + i].orb);
                    for j in 0..nb {
                        let bj = beta(eb, basis.aos[ob + j].orb);
                        let coef = pt[(oa + i, ob + j)] * (bi + bj);
                        epair = epair + s[i][j] * coef;
                    }
                }
                for i in 0..na {
                    for j in 0..na {
                        epair = epair + te.e1b[i][j] * pt[(oa + i, oa + j)];
                    }
                }
                for k in 0..nb {
                    for l in 0..nb {
                        epair = epair + te.e2a[k][l] * pt[(ob + k, ob + l)];
                    }
                }
                for mu in 0..na {
                    for nu in 0..na {
                        for la in 0..nb {
                            for si in 0..nb {
                                let coul = pt[(oa + mu, oa + nu)] * pt[(ob + la, ob + si)];
                                // Same-spin exchange: −(Pα_μλ Pα_νσ + Pβ_μλ Pβ_νσ).
                                let exch = -(pa[(oa + mu, ob + la)] * pa[(oa + nu, ob + si)]
                                    + pb[(oa + mu, ob + la)] * pb[(oa + nu, ob + si)]);
                                epair = epair + te.two_e(mu, nu, la, si) * (coul + exch);
                            }
                        }
                    }
                }
                let r = (dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2]).sqrt();
                epair = epair
                    + crate::repulsion::pair_core_energy_scalar::<Dual2>(
                        params,
                        ea,
                        eb,
                        molecule.atoms[a].z,
                        molecule.atoms[b].z,
                        r,
                    );
                Ok((a, b, epair.h))
            })
            .collect()
    };
    for blk in blocks {
        let (a, b, hb) = blk?;
        for i in 0..3 {
            for j in 0..3 {
                let val = hb[i][j];
                hess[(3 * a + i, 3 * a + j)] += val;
                hess[(3 * b + i, 3 * b + j)] += val;
                hess[(3 * a + i, 3 * b + j)] -= val;
                hess[(3 * b + i, 3 * a + j)] -= val;
            }
        }
    }

    // 2+3) Coupled UCPHF relaxation term: H_relax[a][b] = 2 Σ_σ Gσ^a · Uσ^b.
    let nva = nao - n_alpha;
    let nvb = nao - n_beta;
    let have_a = nva > 0 && n_alpha > 0;
    let have_b = nvb > 0 && n_beta > 0;
    if have_a || have_b {
        let cva = submatrix_cols(&ca, n_alpha, nva);
        let coa = submatrix_cols(&ca, 0, n_alpha);
        let cvb = submatrix_cols(&cb, n_beta, nvb);
        let cob = submatrix_cols(&cb, 0, n_beta);
        let denom_a = ov_denominators(&eps_a, n_alpha, nva);
        let denom_b = ov_denominators(&eps_b, n_beta, nvb);

        let (gova, govb) = skeleton_fock_ov_spin(
            molecule, params, options, &basis, &pt, &pa, &pb, &cva, &coa, &cvb, &cob,
        )?;

        // Streamed in perturbation chunks (OOM guard): never hold all 3N coupled
        // α/β response ov-blocks at once.
        use rayon::prelude::*;
        let ov_total = nva
            .saturating_mul(n_alpha)
            .saturating_add(nvb.saturating_mul(n_beta));
        let resident_bytes = checked_workspace_bytes(&[ndof, ov_total, std::mem::size_of::<f64>()]);
        let per_response = checked_workspace_bytes(&[
            8usize,
            8usize
                .saturating_mul(nao.saturating_mul(nao))
                .saturating_add(20usize.saturating_mul(ov_total)),
        ]);
        let chunk = cphf_chunk_size(
            options,
            "UHF analytic Hessian",
            ndof,
            resident_bytes,
            per_response,
        )?;
        let mut b0 = 0;
        while b0 < ndof {
            let b1 = (b0 + chunk).min(ndof);
            let uov_chunk: Vec<(Matrix, Matrix)> = (b0..b1)
                .into_par_iter()
                .map(|t| {
                    ucphf_ov(
                        &gova[t],
                        &govb[t],
                        &denom_a,
                        &denom_b,
                        &cva,
                        &coa,
                        &cvb,
                        &cob,
                        molecule,
                        params,
                        &basis,
                        &core,
                        CPHF_ITERATIONS,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let block: Vec<Vec<f64>> = (0..ndof)
                .into_par_iter()
                .map(|a| {
                    uov_chunk
                        .iter()
                        .map(|ub| {
                            2.0 * (gova[a].frobenius_dot(&ub.0) + govb[a].frobenius_dot(&ub.1))
                        })
                        .collect()
                })
                .collect();
            for (a, row) in block.into_iter().enumerate() {
                for (j, v) in row.into_iter().enumerate() {
                    hess[(a, b0 + j)] += v;
                }
            }
            b0 = b1;
        }
    }

    // Symmetrize in place to avoid a duplicate Hessian allocation.
    symmetrize_average_in_place(&mut hess);
    // Classical D3/H4/X correction second derivatives (variant-dependent; zero for plain PM3).
    let ch = correction_hessian(molecule, options.variant);
    for i in 0..ndof {
        for j in 0..ndof {
            hess[(i, j)] += ch[(i, j)];
        }
    }
    Ok(hess)
}

/// Spin-resolved skeleton derivative Fock ov-blocks: returns `(Gα, Gβ)` per DOF. The resonance,
/// electron–core, and Coulomb `J(P_tot)` parts are spin-independent (shared); the exchange
/// differs — `Kα(Pα)` into the α Fock, `Kβ(Pβ)` into the β Fock. Built one atom at a time and
/// projected to each spin's occ–virt block (peak memory `O(nao²)`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn skeleton_fock_ov_spin(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    basis: &crate::basis::Basis,
    pt: &Matrix,
    pa: &Matrix,
    pb: &Matrix,
    cva: &Matrix,
    coa: &Matrix,
    cvb: &Matrix,
    cob: &Matrix,
) -> Result<(Vec<Matrix>, Vec<Matrix>)> {
    use crate::integrals::pair_two_electron_dual;
    use crate::overlap::diatom_overlap_dual;
    use rayon::prelude::*;
    let nat = molecule.atoms.len();
    let nao = basis.nao;
    let beta = |e: &crate::params::Pm3Element, orb: u8| if orb == 0 { e.beta_s } else { e.beta_p };

    let bytes_per_atom = checked_workspace_bytes(&[6, nao, nao, std::mem::size_of::<f64>()]);
    let workers = hessian_worker_count(options, nat, bytes_per_atom);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()
        .map_err(|e| crate::error::Pm3Error::InvalidInput(format!("rayon pool: {e}")))?;
    let per_atom: Vec<Result<[(Matrix, Matrix); 3]>> = pool.install(|| {
        (0..nat)
            .into_par_iter()
            .map(|c| -> Result<[(Matrix, Matrix); 3]> {
                let mut fa = [
                    Matrix::zeros(nao, nao),
                    Matrix::zeros(nao, nao),
                    Matrix::zeros(nao, nao),
                ];
                let mut fb = [
                    Matrix::zeros(nao, nao),
                    Matrix::zeros(nao, nao),
                    Matrix::zeros(nao, nao),
                ];
                // The external field's derivative, into **both** channels. The restricted twin
                // of this block carries the note about why it has to live in the skeleton rather
                // than in a one-centre loop of its own; the reasoning applies once per spin. The
                // field couples to charge and not to spin, so each channel gets the same term
                // rather than a halved one — halving it here is the shape the mistake would take.
                if let Some(field) = options.field {
                    let offset = basis.atom_offset[c];
                    let norb = basis.atom_norb[c];
                    for (axis, component) in field.to_array().iter().enumerate() {
                        for mu in 0..norb {
                            fa[axis][(offset + mu, offset + mu)] += component;
                            fb[axis][(offset + mu, offset + mu)] += component;
                        }
                    }
                }
                for x in 0..nat {
                    if x == c {
                        continue;
                    }
                    let (u, v) = (c.min(x), c.max(x));
                    let eu = params.element(molecule.atoms[u].z)?;
                    let ev = params.element(molecule.atoms[v].z)?;
                    let (a, b) = if eu.has_p() || !ev.has_p() {
                        (u, v)
                    } else {
                        (v, u)
                    };
                    let ea = params.element(molecule.atoms[a].z)?;
                    let eb = params.element(molecule.atoms[b].z)?;
                    let (posa, posb) = (molecule.atoms[a].position, molecule.atoms[b].position);
                    let sign = if c == b { 1.0 } else { -1.0 };
                    let te = pair_two_electron_dual(ea, eb, posb - posa);
                    let s = diatom_overlap_dual(ea, posa, eb, posb)?;
                    let (oa, ob) = (basis.atom_offset[a], basis.atom_offset[b]);
                    let (na, nb) = (basis.atom_norb[a], basis.atom_norb[b]);

                    for axis in 0..3 {
                        // Borrow both spin Fock matrices for this atom (distinct arrays fa/fb).
                        let fma = &mut fa[axis];
                        let fmb = &mut fb[axis];
                        // Shared (spin-independent): resonance, e-core, Coulomb J(P_tot) → into BOTH,
                        // per-pair (do NOT copy the running-accumulated matrix, which would
                        // re-add earlier neighbours' contributions).
                        for i in 0..na {
                            let bi = beta(ea, basis.aos[oa + i].orb);
                            for j in 0..nb {
                                let bj = beta(eb, basis.aos[ob + j].orb);
                                let val = sign * 0.5 * (bi + bj) * s[i][j].d[axis];
                                fma[(oa + i, ob + j)] += val;
                                fma[(ob + j, oa + i)] += val;
                                fmb[(oa + i, ob + j)] += val;
                                fmb[(ob + j, oa + i)] += val;
                            }
                        }
                        for i in 0..na {
                            for j in 0..na {
                                let val = sign * te.e1b[i][j].d[axis];
                                fma[(oa + i, oa + j)] += val;
                                fmb[(oa + i, oa + j)] += val;
                            }
                        }
                        for k in 0..nb {
                            for l in 0..nb {
                                let val = sign * te.e2a[k][l].d[axis];
                                fma[(ob + k, ob + l)] += val;
                                fmb[(ob + k, ob + l)] += val;
                            }
                        }
                        for mu in 0..na {
                            for nu in 0..na {
                                let mut acc = 0.0;
                                for la in 0..nb {
                                    for si in 0..nb {
                                        acc += pt[(ob + la, ob + si)]
                                            * te.two_e(mu, nu, la, si).d[axis];
                                    }
                                }
                                let val = sign * acc;
                                fma[(oa + mu, oa + nu)] += val;
                                fmb[(oa + mu, oa + nu)] += val;
                            }
                        }
                        for la in 0..nb {
                            for si in 0..nb {
                                let mut acc = 0.0;
                                for mu in 0..na {
                                    for nu in 0..na {
                                        acc += pt[(oa + mu, oa + nu)]
                                            * te.two_e(mu, nu, la, si).d[axis];
                                    }
                                }
                                let val = sign * acc;
                                fma[(ob + la, ob + si)] += val;
                                fmb[(ob + la, ob + si)] += val;
                            }
                        }
                        // Same-spin exchange Kσ (coefficient −1): Kα(Pα) → fa, Kβ(Pβ) → fb.
                        for mu in 0..na {
                            for la in 0..nb {
                                let mut acca = 0.0;
                                let mut accb = 0.0;
                                for nu in 0..na {
                                    for si in 0..nb {
                                        let dw = te.two_e(mu, nu, la, si).d[axis];
                                        acca += pa[(oa + nu, ob + si)] * dw;
                                        accb += pb[(oa + nu, ob + si)] * dw;
                                    }
                                }
                                let va = sign * (-acca);
                                let vb = sign * (-accb);
                                fma[(oa + mu, ob + la)] += va;
                                fma[(ob + la, oa + mu)] += va;
                                fmb[(oa + mu, ob + la)] += vb;
                                fmb[(ob + la, oa + mu)] += vb;
                            }
                        }
                    }
                }
                Ok([
                    (project_ov(&fa[0], cva, coa), project_ov(&fb[0], cvb, cob)),
                    (project_ov(&fa[1], cva, coa), project_ov(&fb[1], cvb, cob)),
                    (project_ov(&fa[2], cva, coa), project_ov(&fb[2], cvb, cob)),
                ])
            })
            .collect()
    });

    let mut gova: Vec<Matrix> = Vec::with_capacity(3 * nat);
    let mut govb: Vec<Matrix> = Vec::with_capacity(3 * nat);
    for res in per_atom {
        let arr = res?;
        for (ga, gb) in arr {
            gova.push(ga);
            govb.push(gb);
        }
    }
    Ok((gova, govb))
}

/// Coupled α/β CPHF solve for one perturbation (MO occ–virt blocks). Iterate
/// `Uσ = (Gσ_skel + [J(ΔP_tot) − Kσ(ΔPσ)]_ov) / (εσ_i − εσ_a)` to self-consistency; the α and β
/// channels couple through the total response density `ΔP_tot = ΔPα + ΔPβ` in the Coulomb term.
#[allow(clippy::too_many_arguments)]
pub(crate) fn ucphf_ov(
    ga: &Matrix,
    gb: &Matrix,
    denom_a: &Matrix,
    denom_b: &Matrix,
    cva: &Matrix,
    coa: &Matrix,
    cvb: &Matrix,
    cob: &Matrix,
    molecule: &Molecule,
    params: &Pm3Parameters,
    basis: &crate::basis::Basis,
    core: &crate::hamiltonian::CoreHamiltonian,
    max_iter: usize,
) -> Result<(Matrix, Matrix)> {
    let div = |num: &Matrix, denom: &Matrix| -> Matrix {
        let mut u = num.clone();
        for (uv, dv) in u.as_mut_slice().iter_mut().zip(denom.as_slice()) {
            *uv = if dv.abs() < 1.0e-10 { 0.0 } else { *uv / *dv };
        }
        u
    };
    let mut ua = div(ga, denom_a);
    let mut ub = div(gb, denom_b);
    // The same Pulay extrapolation the restricted solver has had all along. It was written there
    // because "semiempirical CPHF is stiff otherwise", and this is the *harder* problem — an
    // open shell with small α/β gaps and near-degenerate frontier orbitals — so the unaccelerated
    // iteration here was the one most likely to run out of passes, and (before the check below)
    // to do it silently.
    let mut hist_u: Vec<Vec<Matrix>> = Vec::new();
    let mut hist_e: Vec<Vec<Matrix>> = Vec::new();
    let max_diis = 8;
    let mut converged = false;
    let mut residual = f64::INFINITY;
    for _ in 0..max_iter {
        let dpa = ao_response_density_w(&ua, cva, coa, 1.0);
        let dpb = ao_response_density_w(&ub, cvb, cob, 1.0);
        let mut dpt = dpa.clone();
        for (t, x) in dpt.as_mut_slice().iter_mut().zip(dpb.as_slice()) {
            *t += *x;
        }
        // Response two-electron Focks Gσ(ΔP) = build_fock_spin(ΔP_tot, ΔPσ) − H_core.
        let mut fa_r = crate::fock::build_fock_spin(molecule, basis, params, core, &dpt, &dpa)?;
        let mut fb_r = crate::fock::build_fock_spin(molecule, basis, params, core, &dpt, &dpb)?;
        for (xv, hv) in fa_r.as_mut_slice().iter_mut().zip(core.h_core.as_slice()) {
            *xv -= *hv;
        }
        for (xv, hv) in fb_r.as_mut_slice().iter_mut().zip(core.h_core.as_slice()) {
            *xv -= *hv;
        }
        let ga_resp = project_ov(&fa_r, cva, coa);
        let gb_resp = project_ov(&fb_r, cvb, cob);
        let mut rhs_a = ga.clone();
        for (rv, gv) in rhs_a.as_mut_slice().iter_mut().zip(ga_resp.as_slice()) {
            *rv += *gv;
        }
        let mut rhs_b = gb.clone();
        for (rv, gv) in rhs_b.as_mut_slice().iter_mut().zip(gb_resp.as_slice()) {
            *rv += *gv;
        }
        let ua_new = div(&rhs_a, denom_a);
        let ub_new = div(&rhs_b, denom_b);
        let mut err_a = ua_new.clone();
        let mut err_b = ub_new.clone();
        let mut diff = 0.0;
        for (ev, ov) in err_a.as_mut_slice().iter_mut().zip(ua.as_slice()) {
            *ev -= *ov;
            diff += *ev * *ev;
        }
        for (ev, ov) in err_b.as_mut_slice().iter_mut().zip(ub.as_slice()) {
            *ev -= *ov;
            diff += *ev * *ev;
        }
        residual = diff.sqrt();
        if residual < CPHF_TOLERANCE {
            ua = ua_new;
            ub = ub_new;
            converged = true;
            break;
        }
        // Both channels extrapolate on one set of coefficients: they are one coupled fixed
        // point, not two independent ones.
        hist_u.push(vec![ua_new.clone(), ub_new.clone()]);
        hist_e.push(vec![err_a, err_b]);
        if hist_u.len() > max_diis {
            hist_u.remove(0);
            hist_e.remove(0);
        }
        match cphf_diis_channels(&hist_u, &hist_e) {
            Some(mut mixed) => {
                ub = mixed.remove(1);
                ua = mixed.remove(0);
            }
            None => {
                ua = ua_new;
                ub = ub_new;
            }
        }
    }
    if !converged {
        return Err(Pm3Error::ScfNotConverged {
            iterations: max_iter,
            error: residual,
        });
    }
    Ok((ua, ub))
}

#[inline]
fn displace(p: &mut Vec3, k: usize, d: f64) {
    match k {
        0 => p.x += d,
        1 => p.y += d,
        _ => p.z += d,
    }
}
#[inline]
fn component(v: &Vec3, k: usize) -> f64 {
    match k {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{optimize, OptOptions};

    /// A massless atom does not turn the frequencies into `NaN`.
    ///
    /// MOPAC's sparkle codes carry a tabulated mass of exactly zero, `symbol_to_z` accepts them
    /// from an ordinary XYZ line, and `tests/api_surface.rs` asserts they are supported — so
    /// `pm3-rs frequencies` on a geometry containing one divided by zero and returned `inf`
    /// wherever that atom appeared. Both sibling mass-weightings guarded it; this one did not.
    #[test]
    fn a_massless_atom_does_not_poison_the_frequencies() {
        let params = Pm3Parameters::standard().unwrap();
        // A water molecule with a `+` point charge sitting off to one side.
        // Charge `+1`, because the sparkle carries a `+1` core charge and no orbitals: at
        // neutrality the cell would hold nine electrons, which no singlet can.
        let molecule = Molecule::from_xyz_str(
            "4\nwater and a sparkle\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\n\
             H 0.0 -0.7572 -0.4692\n+ 0.0 0.0 4.0\n",
            1.0,
        )
        .unwrap();
        // Non-vacuity: the tabulated mass really is zero, which is what makes this a division
        // by zero rather than a merely small denominator.
        assert_eq!(params.element(104).unwrap().mass, 0.0);

        let options = Pm3Options {
            charge: 1.0,
            ..Pm3Options::default()
        };
        let vib = vibrational_analysis(&molecule, &params, &options, 1.0e-3)
            .expect("a sparkle is a supported atom");
        assert!(
            vib.frequencies_cm.iter().all(|f| f.is_finite()),
            "non-finite frequencies: {:?}",
            vib.frequencies_cm
        );
        assert!(vib.eigenvalues.iter().all(|e| e.is_finite()));
    }

    /// A coupled-perturbed solve that runs out of passes is an error, not an answer.
    ///
    /// Both solvers used to fall out of their loop and `Ok` whatever iterate they were holding.
    /// Nothing downstream could tell: a partially converged `U` gives a Hessian with plausible
    /// frequencies and an infrared spectrum with plausible intensities. It was not hypothetical
    /// — water's own response stops at `9.4e-7` after the hundred passes the cap used to allow,
    /// against a declared tolerance of `1e-9`, so *every* analytic Hessian this crate produced
    /// was built on a response three digits looser than it claimed.
    ///
    /// Driven at `max_iter = 1` because a cap that cannot be lowered is a branch that cannot be
    /// tested, which is how this survived.
    #[test]
    fn a_response_that_runs_out_of_passes_is_refused() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = Molecule::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
            0.0,
        )
        .unwrap();
        let options = Pm3Options::default();
        let scf = crate::run_pm3(&molecule, &params, &options).unwrap();
        let basis = crate::basis::Basis::build(&molecule, &params).unwrap();
        let core = crate::hamiltonian::build_core(&molecule, &basis, &params).unwrap();

        let n_occ = scf.n_occ;
        let nvir = basis.nao - n_occ;
        let cv = submatrix_cols(&scf.mo_coeff, n_occ, nvir);
        let co = submatrix_cols(&scf.mo_coeff, 0, n_occ);
        let denom = ov_denominators(&scf.mo_energies, n_occ, nvir);
        let gov =
            skeleton_fock_ov(&molecule, &params, &options, &basis, &scf.density, &cv, &co).unwrap();

        let one_pass = cphf_ov(
            &gov[0], &denom, &cv, &co, &molecule, &params, &basis, &core, None, 1,
        );
        match one_pass {
            Err(crate::error::Pm3Error::ScfNotConverged { iterations, error }) => {
                assert_eq!(iterations, 1);
                assert!(error > CPHF_TOLERANCE, "refused while inside tolerance");
            }
            Err(other) => panic!("wrong error: {other}"),
            Ok(_) => panic!("one pass cannot converge a stiff fixed point, and it said it did"),
        }

        // The production cap does converge it — which is the other half of the claim, and the
        // half that says 400 is enough rather than merely larger than 100.
        let converged = cphf_ov(
            &gov[0],
            &denom,
            &cv,
            &co,
            &molecule,
            &params,
            &basis,
            &core,
            None,
            CPHF_ITERATIONS,
        )
        .expect("the production cap converges water's response");
        assert_eq!(converged.rows, nvir);
        assert_eq!(converged.cols, n_occ);

        // And it converges *fast*, which is the part a correctness check alone would not catch.
        //
        // The extrapolation used to normalise each residual by its own magnitude, which solves a
        // different constrained problem than Pulay's. The answer was still the right fixed point
        // — every finite-difference test passed — so nothing failed; the iteration simply fell
        // back to the unaccelerated rate of 0.892 per pass and took 160 of them. Ten passes is
        // far above the three this needs and far below the 160 the broken extrapolation took, so
        // it pins the behaviour without pinning the exact count.
        let fast = cphf_ov(
            &gov[0], &denom, &cv, &co, &molecule, &params, &basis, &core, None, 10,
        );
        assert!(
            fast.is_ok(),
            "ten passes should be plenty; the extrapolation is not accelerating"
        );
        let quick = fast.unwrap();
        let drift = quick
            .as_slice()
            .iter()
            .zip(converged.as_slice())
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            drift < 1.0e-8,
            "the ten-pass and the full solve disagree by {drift:.3e}; extrapolation must change \
             the path to the fixed point, not the fixed point"
        );
    }

    /// The open-shell response is extrapolated too, and refuses the same way.
    ///
    /// `ucphf_ov` was a bare fixed-point iteration while its restricted twin had Pulay DIIS —
    /// written there because "semiempirical CPHF is stiff otherwise". So the acceleration was
    /// missing from the *harder* of the two problems.
    #[test]
    fn the_open_shell_response_is_accelerated_and_refuses_when_it_must() {
        const METHYL: &str = "4\nmethyl\nC 0.0 0.0 0.0\nH 1.0787 0.0 0.0\n\
                              H -0.5393 0.9341 0.0\nH -0.5393 -0.9341 0.0\n";
        let params = Pm3Parameters::standard().unwrap();
        let molecule = Molecule::from_xyz_str(METHYL, 0.0)
            .unwrap()
            .with_multiplicity(2);
        let options = Pm3Options {
            multiplicity: 2,
            reference: crate::Reference::Uhf,
            ..Pm3Options::default()
        };
        let scf = crate::run_pm3(&molecule, &params, &options).unwrap();
        let basis = crate::basis::Basis::build(&molecule, &params).unwrap();
        let core = crate::hamiltonian::build_core(&molecule, &basis, &params).unwrap();

        let (n_alpha, n_beta) = (scf.n_occ, scf.n_beta);
        let (nva, nvb) = (basis.nao - n_alpha, basis.nao - n_beta);
        let cva = submatrix_cols(&scf.mo_coeff, n_alpha, nva);
        let coa = submatrix_cols(&scf.mo_coeff, 0, n_alpha);
        let beta_coeff = scf
            .mo_coeff_beta
            .as_ref()
            .expect("an unrestricted result has beta MOs");
        let beta_energies = scf
            .mo_energies_beta
            .as_ref()
            .expect("an unrestricted result has beta energies");
        let cvb = submatrix_cols(beta_coeff, n_beta, nvb);
        let cob = submatrix_cols(beta_coeff, 0, n_beta);
        let denom_a = ov_denominators(&scf.mo_energies, n_alpha, nva);
        let denom_b = ov_denominators(beta_energies, n_beta, nvb);

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
        let (gova, govb) = skeleton_fock_ov_spin(
            &molecule,
            &params,
            &options,
            &basis,
            &scf.density,
            &pa,
            &pb,
            &cva,
            &coa,
            &cvb,
            &cob,
        )
        .unwrap();

        let one_pass = ucphf_ov(
            &gova[0], &govb[0], &denom_a, &denom_b, &cva, &coa, &cvb, &cob, &molecule, &params,
            &basis, &core, 1,
        );
        assert!(
            matches!(
                one_pass,
                Err(crate::error::Pm3Error::ScfNotConverged { .. })
            ),
            "one pass cannot converge the coupled alpha/beta fixed point"
        );

        let (ua, ub) = ucphf_ov(
            &gova[0],
            &govb[0],
            &denom_a,
            &denom_b,
            &cva,
            &coa,
            &cvb,
            &cob,
            &molecule,
            &params,
            &basis,
            &core,
            CPHF_ITERATIONS,
        )
        .expect("the production cap converges the methyl radical's response");
        assert_eq!((ua.rows, ua.cols), (nva, n_alpha));
        assert_eq!((ub.rows, ub.cols), (nvb, n_beta));
    }

    #[test]
    fn cphf_batch_respects_memory_budget() {
        let opts = Pm3Options {
            hessian_memory_mb: 1,
            ..Pm3Options::default()
        };
        let chunk = cphf_chunk_size(&opts, "test", 100, MIB / 2, MIB / 8).unwrap();
        assert_eq!(chunk, 4);
        let err = cphf_chunk_size(&opts, "test", 100, MIB, MIB / 8).unwrap_err();
        assert!(matches!(err, crate::error::Pm3Error::ResourceLimit { .. }));
    }

    #[test]
    fn analytic_hessian_matches_numerical() {
        // The CPHF analytic Hessian must match the finite-difference Hessian (FD of the
        // full-SCF gradient) — the independent ground truth.
        let mol = Molecule::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.97 0.02 0.0\nH -0.25 0.94 0.0\n",
            0.0,
        )
        .unwrap();
        let params = Pm3Parameters::standard().unwrap();
        let opts = Pm3Options::default();
        let ha = analytic_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
        let hn = numerical_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
        let ndof = ha.rows;
        let mut max_delta = 0.0_f64;
        for i in 0..ndof {
            for j in 0..ndof {
                max_delta = max_delta.max((ha[(i, j)] - hn[(i, j)]).abs());
            }
        }
        eprintln!("analytic-vs-numerical Hessian max delta = {max_delta:.2e} eV/Bohr^2");
        assert!(max_delta < 1.0e-3, "Hessian mismatch {max_delta:.3e}");
    }

    #[test]
    fn gradient_onaxis_dpair_is_correct() {
        // Regression: an on-axis d-pair in an ASYMMETRIC environment must NOT give a wrong
        // analytic gradient. Zn-Cl1 lies exactly on +z; Cl2 is off to the side (bent,
        // asymmetric density), so the transverse force on the on-axis pair is genuinely nonzero
        // — the singular-frame branch would drop it (≈0.2 eV/Bohr error) without the frame fix.
        let params = Pm3Parameters::standard().unwrap();
        let opts = Pm3Options::default();
        let mol = Molecule::from_xyz_str(
            "3\nZnCl2 bent, one bond on z\nZn 0 0 0\nCl 0 0 2.07\nCl 1.6 0.0 -0.9\n",
            0.0,
        )
        .unwrap();
        let ga = crate::gradient::closed_form_gradient(&mol, &params, &opts).unwrap();
        let gn = crate::gradient::numerical_gradient(&mol, &params, &opts, 5.0e-4).unwrap();
        let mut maxd = 0.0_f64;
        for a in 0..mol.atoms.len() {
            for k in 0..3 {
                maxd = maxd.max((ga.gradient[a].get(k) - gn.gradient[a].get(k)).abs());
            }
        }
        eprintln!("on-axis d-pair gradient: analytic vs FD-of-energy maxΔ = {maxd:.4e} eV/Bohr");
        assert!(
            maxd < 1e-5,
            "on-axis d-pair gradient wrong: {maxd:.3e} eV/Bohr"
        );
    }

    #[test]
    fn correction_variant_hessian_matches_numerical() {
        // The analytic Hessian must include the classical D3/H4/X second derivatives: for a
        // hydrogen-bonded water dimer under PM3-D3H4 it must still match the numerical Hessian
        // (which finite-differences the correction-inclusive gradient). A missing correction
        // Hessian would make the two disagree by the dispersion/H-bond curvature.
        let mol = Molecule::from_xyz_str(
            "6\nwater dimer\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n\
             O 0.0 0.0 2.9\nH 0.9 0.0 3.2\nH -0.24 0.0 2.0\n",
            0.0,
        )
        .unwrap();
        let params = Pm3Parameters::standard().unwrap();
        let opts = Pm3Options {
            variant: crate::corrections::Variant::Pm3D3H4,
            ..Pm3Options::default()
        };
        let ha = analytic_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
        let hn = numerical_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
        // Sanity: without the correction Hessian the two would differ; confirm the plain-PM3
        // analytic Hessian of the same geometry is *different* (i.e. corrections contribute).
        let opts_plain = Pm3Options::default();
        let ha_plain = analytic_hessian(&mol, &params, &opts_plain, 1.0e-3).unwrap();
        let mut maxd = 0.0_f64;
        let mut maxcorr = 0.0_f64;
        for i in 0..ha.rows {
            for j in 0..ha.rows {
                maxd = maxd.max((ha[(i, j)] - hn[(i, j)]).abs());
                maxcorr = maxcorr.max((ha[(i, j)] - ha_plain[(i, j)]).abs());
            }
        }
        eprintln!("D3H4 Hessian: analytic-vs-numerical maxΔ={maxd:.3e}  correction contribution maxΔ={maxcorr:.3e}");
        assert!(
            maxcorr > 1e-4,
            "correction Hessian is negligible — is it being added? {maxcorr:.2e}"
        );
        assert!(
            maxd < 5e-5,
            "D3H4 analytic vs numerical Hessian mismatch {maxd:.3e}"
        );
    }

    #[test]
    fn hessian_frame_rotation_is_exact() {
        // The frame-rotation back-transform must be exact: for a generic (non-degenerate)
        // d molecule, computing the Hessian in a rotated frame and rotating it back must
        // reproduce the directly-computed Hessian to round-off. Isolates the transform from
        // any electronic (degeneracy) effect.
        let params = Pm3Parameters::standard().unwrap();
        let opts = Pm3Options::default();
        let mol = Molecule::from_xyz_str(
            "3\nSCl2\nS 0.0 0.0 0.0\nCl 1.5 0.7 0.3\nCl -1.4 0.8 0.2\n",
            0.0,
        )
        .unwrap();
        let h_direct = analytic_hessian_core(&mol, &params, &opts, 1.0e-3).unwrap();
        let r0 = crate::frame::euler_rotation(0.6, 0.7, 0.9);
        let h_rot = analytic_hessian_core(
            &crate::frame::rotate_molecule(&mol, &r0),
            &params,
            &opts,
            1.0e-3,
        )
        .unwrap();
        let h_back = crate::frame::back_rotate_hessian(&h_rot, &r0);
        let mut maxd = 0.0_f64;
        for i in 0..h_direct.rows {
            for j in 0..h_direct.rows {
                maxd = maxd.max((h_direct[(i, j)] - h_back[(i, j)]).abs());
            }
        }
        eprintln!("frame-rotation round-trip maxΔ = {maxd:.3e}");
        assert!(
            maxd < 1e-8,
            "frame-rotation back-transform not exact: {maxd:.3e}"
        );
    }

    #[test]
    fn d_orbital_analytic_hessian_matches_numerical() {
        // Closed-shell d systems now take the analytic spd CPHF Hessian; it must
        // agree with the numerical (central-difference) Hessian. Test HBr (Br has a
        // PM3 valence d shell) and ZnCl2 (d on the metal and on Cl).
        let params = Pm3Parameters::standard().unwrap();
        let opts = Pm3Options::default();
        let mut fails: Vec<String> = Vec::new();
        for xyz in [
            "2\nHBr collinear\nH 0.0 0.0 0.0\nBr 0.0 0.0 1.48\n",
            "3\nZnCl2 collinear\nZn 0 0 0\nCl 0 0 2.07\nCl 0 0 -2.07\n",
            // Linear ZnCl2 along a generic axis (wrapper does NOT trigger) — isolates any
            // linear/triatomic issue from the z-axis singularity.
            "3\nZnCl2 genaxis\nZn 0 0 0\nCl 1.195 1.195 1.195\nCl -1.195 -1.195 -1.195\n",
            // Non-collinear (tetrahedral) — no bond lies on a coordinate axis.
            "3\nSCl2 generic\nS 0 0 0\nCl 1.5 0.7 0.3\nCl -1.4 0.8 0.2\n",
            // Bent HBr-like (sp-d): displace off the z axis.
            "2\nHBr bent\nH 0.3 0.2 0.0\nBr 0.0 0.0 1.48\n",
            // d-d diatomic (Cl2, closed shell), generic off-axis orientation.
            "2\nCl2 offaxis\nCl 0.0 0.0 0.0\nCl 1.3 0.9 0.7\n",
            // d-d diatomic (ZnS, closed shell), generic off-axis orientation.
            "2\nZnS offaxis\nZn 0.0 0.0 0.0\nS 1.4 1.0 0.8\n",
        ] {
            let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
            let ha = analytic_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
            let hn = numerical_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
            let n = ha.rows;
            let mut maxd = 0.0_f64;
            let mut maxasym = 0.0_f64;
            for i in 0..n {
                for j in 0..n {
                    assert!(ha[(i, j)].is_finite());
                    maxasym = maxasym.max((ha[(i, j)] - ha[(j, i)]).abs());
                    maxd = maxd.max((ha[(i, j)] - hn[(i, j)]).abs());
                }
            }
            let label = xyz.lines().nth(1).unwrap_or("");
            eprintln!("d-Hessian maxΔ = {maxd:.3e} (asym {maxasym:.1e}) for [{label}]");
            if maxasym >= 1e-6 || maxd >= 2e-5 {
                fails.push(format!("{label}: maxΔ={maxd:.3e} asym={maxasym:.1e}"));
            }
        }
        assert!(fails.is_empty(), "d-Hessian failures: {fails:?}");
    }

    #[test]
    fn d_hessian_fd_step_is_quadratic() {
        // The user's diagnostic, as a permanent check: a *correct* analytic Hessian differs from
        // the finite-difference Hessian only by FD truncation, which scales as h². We halve the
        // step twice and require the discrepancy to shrink by ≈4× each time (and be tiny). A
        // broken analytic Hessian would instead plateau at an h-independent error.
        let params = Pm3Parameters::standard().unwrap();
        let opts = Pm3Options::default();
        // Tetrahedral TiCl4: bonds on (1,1,1); all Cl-Cl bonds lie in coordinate planes — the
        // configuration that previously exposed the zero-valued-coefficient derivative drop.
        let mol = Molecule::from_xyz_str(
            "3\nSCl2 generic\nS 0 0 0\nCl 1.5 0.7 0.3\nCl -1.4 0.8 0.2\n",
            0.0,
        )
        .unwrap();
        let ha = analytic_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
        let dev = |h: f64| -> f64 {
            let hn = numerical_hessian(&mol, &params, &opts, h).unwrap();
            let mut m = 0.0_f64;
            for i in 0..ha.rows {
                for j in 0..ha.rows {
                    m = m.max((ha[(i, j)] - hn[(i, j)]).abs());
                }
            }
            m
        };
        let (d4, d2, d1) = (dev(4.0e-3), dev(2.0e-3), dev(1.0e-3));
        eprintln!("FD-step scaling: h=4e-3 {d4:.2e}  h=2e-3 {d2:.2e}  h=1e-3 {d1:.2e}");
        // Halving h must cut the discrepancy by ~4× (h²); allow slack for the FD noise floor.
        assert!(d1 < 5e-6, "analytic Hessian not converging to FD: {d1:.2e}");
        assert!(
            d4 / d2 > 2.5 && d2 / d1 > 2.5,
            "FD discrepancy not ~h² (analytic bug): {d4:.2e},{d2:.2e},{d1:.2e}"
        );
    }

    #[test]
    fn analytic_hessian_uhf_radical() {
        // Methyl radical (doublet, UHF): the coupled α/β CPHF (UCPHF) analytic Hessian must
        // match the finite-difference Hessian (FD of the analytic UHF gradient).
        let mol = Molecule::from_xyz_str(
            "4\nmethyl\nC 0.0 0.0 0.05\nH 1.09 0.0 0.0\nH -0.545 0.944 0.0\nH -0.545 -0.944 0.0\n",
            0.0,
        )
        .unwrap();
        let params = Pm3Parameters::standard().unwrap();
        let opts = Pm3Options {
            multiplicity: 2,
            ..Pm3Options::default()
        };
        let ha = analytic_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
        let hn = numerical_hessian(&mol, &params, &opts, 1.0e-3).unwrap();
        let ndof = ha.rows;
        let mut max_delta = 0.0_f64;
        for i in 0..ndof {
            for j in 0..ndof {
                max_delta = max_delta.max((ha[(i, j)] - hn[(i, j)]).abs());
            }
        }
        eprintln!("UHF analytic-vs-numerical Hessian max delta = {max_delta:.2e}");
        assert!(max_delta < 2.0e-3, "UHF Hessian mismatch {max_delta:.3e}");
    }

    #[test]
    fn water_vibrations() {
        // Optimize water, then compute PM3 harmonic frequencies. The oracle
        // values below are MOPAC v23.2.5 `PM3 FORCE LET` at the pm3-rs
        // optimized geometry: one bend and two stretches, plus six near-rigid
        // translations/rotations.
        let mol = Molecule::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n",
            0.0,
        )
        .unwrap();
        let params = Pm3Parameters::standard().unwrap();
        let opts = Pm3Options::default();
        let relaxed = optimize(&mol, &params, &opts, &OptOptions::default()).unwrap();
        let vib = vibrational_analysis(&relaxed.molecule, &params, &opts, 1.0e-3).unwrap();
        let freqs = &vib.frequencies_cm;
        eprintln!(
            "H2O frequencies (cm^-1): {:?}",
            freqs.iter().map(|f| f.round()).collect::<Vec<_>>()
        );
        // The three highest are the real vibrational modes.
        let n = freqs.len();
        let high = &freqs[n - 3..];
        let oracle = [1743.46, 3868.68, 3989.81];
        for (got, reference) in high.iter().zip(oracle) {
            assert!(
                (got - reference).abs() < 2.0,
                "PM3/MOPAC frequency mismatch: got {got}, expected {reference}"
            );
        }
        // The six lowest (trans/rot) should be small in magnitude.
        let six_low_max = freqs[..6].iter().map(|f| f.abs()).fold(0.0_f64, f64::max);
        assert!(
            six_low_max < 300.0,
            "trans/rot not near zero: {six_low_max}"
        );
    }
    /// The analytic Hessian in a field, against a finite difference of the analytic gradient.
    ///
    /// The field contributes nothing to the *skeleton*: it is linear in the nuclear coordinates,
    /// so its second derivative at fixed density vanishes identically. Everything it does to the
    /// Hessian comes through the response, and reaches it only because `skeleton_fock_ov` carries
    /// the field's own first derivative. Leave that out and the result is still symmetric, still
    /// has six near-zero modes, and is wrong by the cross term.
    ///
    /// Both spin paths are exercised. The open-shell case is not decoration: `skeleton_fock_ov`
    /// and `skeleton_fock_ov_spin` are separate functions, so wiring the field into one of them
    /// leaves the other silently short of exactly the cross term described above — and a
    /// closed-shell test cannot see it.
    #[test]
    fn the_hessian_in_a_field_matches_finite_differences() {
        for (xyz, multiplicity) in [
            (
                "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
                1_usize,
            ),
            (
                "4\nmethyl\nC 0.0 0.0 0.05\nH 1.09 0.0 0.0\nH -0.545 0.944 0.0\nH -0.545 -0.944 0.0\n",
                2,
            ),
        ] {
            the_field_hessian_case(xyz, multiplicity);
        }
    }

    fn the_field_hessian_case(xyz: &str, multiplicity: usize) {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let field = crate::math::Vec3::new(0.018, -0.011, 0.026);
        let options = Pm3Options {
            e_tol: 1.0e-12,
            p_tol: 1.0e-11,
            max_scf: 500,
            field: Some(field),
            multiplicity,
            ..Default::default()
        };

        let analytic = analytic_hessian(&molecule, &params, &options, 5.0e-4).unwrap();
        let ndof = 3 * molecule.atoms.len();
        let step = 1.0e-4;
        let mut numeric = Matrix::zeros(ndof, ndof);
        for dof in 0..ndof {
            let gradient_at = |sign: f64| {
                let mut shifted = molecule.clone();
                let mut delta = [0.0; 3];
                delta[dof % 3] = sign * step;
                shifted.atoms[dof / 3].position +=
                    crate::math::Vec3::new(delta[0], delta[1], delta[2]);
                crate::gradient::closed_form_gradient(&shifted, &params, &options)
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

        let mut worst = 0.0_f64;
        for i in 0..ndof {
            for j in 0..ndof {
                worst = worst.max((analytic[(i, j)] - numeric[(i, j)]).abs());
            }
        }
        assert!(
            worst < 2.0e-4,
            "the field Hessian differs from finite differences by {worst:.3e} eV/Bohr^2"
        );

        // And the field genuinely moved it, so the agreement is not agreement on the field-free
        // Hessian.
        let without = analytic_hessian(
            &molecule,
            &params,
            &Pm3Options {
                field: None,
                ..options.clone()
            },
            5.0e-4,
        )
        .unwrap();
        let mut moved = 0.0_f64;
        for i in 0..ndof {
            for j in 0..ndof {
                moved = moved.max((analytic[(i, j)] - without[(i, j)]).abs());
            }
        }
        assert!(
            moved > 1.0e-3,
            "the field barely moved the Hessian ({moved:.3e})"
        );
    }
}
