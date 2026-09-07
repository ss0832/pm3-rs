// SPDX-License-Identifier: GPL-3.0-or-later

//! Γ-point analytic Hessian and phonons for periodic systems.
//!
//! # What this is, in phonon language
//!
//! The matrix built here is the dynamical matrix at `q = 0`:
//!
//! ```text
//! D(0)_{κα,κ'β} = Σ_T Φ(0κα, Tκ'β)
//! ```
//!
//! — the second derivative of the energy per cell with respect to displacing *every copy* of atom
//! `κ` together. That is what a Γ-point calculation can produce and all it can produce: a
//! displacement pattern with a wavevector needs the density's response at that wavevector, which
//! one k-point cannot represent, for the same reason spelled out in [`crate::pbc::gamma`].
//!
//! # Two halves
//!
//! ```text
//! ∂²E/∂R∂R  =  skeleton (density held fixed)  +  response (the density moves)
//! ```
//!
//! The split is not a numerical convenience — it is what makes the two testable apart. The
//! skeleton is pure differentiation of the integrals and can be checked term by term; the response
//! is a linear solve whose right-hand side is the skeleton derivative of the Fock matrix.
//!
//! | contribution | how |
//! |---|---|
//! | two-center integrals, resonance, electron–core, core–core | second-order AD ([`Dual2`]) on the image displacement `d + T` |
//! | Ewald, at fixed charges | analytic, [`crate::pbc::ewald_hessian`] |
//! | D3/H4/X | second-order AD over the image cluster |
//! | density response | CPHF, with the periodic two-electron kernel |
//!
//! # Why a self-image contributes nothing
//!
//! A pair `(a, a, T)` has displacement `d = T·h`, which does not depend on where atom `a` is —
//! both endpoints move together. Its derivatives are therefore identically zero, and the scatter
//! below produces that automatically: `+H` on `(a,a)` twice and `−H` on `(a,a)` twice.
//!
//! # The acoustic sum rule
//!
//! Every term here is a function of interatomic displacements alone, so each one is individually
//! translation-invariant and the acoustic sum rule holds by construction rather than by
//! enforcement. [`enforce_acoustic_sum_rule`] exists to clean up rounding, not to rescue a missing
//! term — if it changes the result by more than roundoff, something is wrong.

use rayon::prelude::*;

use crate::basis::Basis;
use crate::corrections::periodic::{build_cluster, cluster_positions_g, CorrectionCutoffs};
use crate::corrections::{correction_energy_cluster_g, Variant};
use crate::dual::Scalar;
use crate::dual2::Dual2;
use crate::error::{Pm3Error, Result};
use crate::integrals::{pack, pair_two_electron_g};
use crate::linalg::{symmetric_eigen, Matrix};
use crate::math::Vec3;
use crate::neighbor::NeighborList;
use crate::params::Pm3Parameters;
use crate::pbc::ewald_hessian::ewald_atom_hessian;
use crate::pbc::gamma::{run_gamma, PeriodicOptions, PeriodicResult};
use crate::pbc::multipole::AtomSites;
use crate::pbc::screen::point_pair_g;
use crate::scf::Pm3Options;
use crate::system::Molecule;

/// A Γ-point phonon calculation.
#[derive(Clone, Debug)]
pub struct PeriodicPhonons {
    /// The converged SCF the Hessian was evaluated at.
    pub scf: PeriodicResult,
    /// `∂²E/∂R∂R` per cell (eV/Bohr²) — the dynamical matrix at `q = 0`, before mass weighting.
    ///
    /// The **raw** second derivative, symmetric, with nothing imposed on it. The acoustic
    /// projection that produces the frequencies below is applied to a mass-weighted copy and
    /// never to this.
    pub hessian: Matrix,
    /// Γ-point phonon frequencies (cm⁻¹), ascending. Negative values denote imaginary modes.
    ///
    /// The three acoustic modes are **exactly** `0.0`: the translations are projected out of the
    /// mass-weighted matrix before it is diagonalized, so those directions are empty rather than
    /// small. Nothing is selected by being below a threshold.
    pub frequencies_cm: Vec<f64>,
    /// Eigenvalues of the mass-weighted matrix, in the same order.
    pub eigenvalues: Vec<f64>,
    /// The **mass-weighted** eigenvectors, one mode per column, in the same order.
    ///
    /// Column `m` is the displacement pattern of mode `m` in mass-weighted coordinates, so the
    /// Cartesian displacement of atom `a` is its three rows divided by `sqrt(mass[a])` — the same
    /// convention as [`crate::VibrationalModes::modes`], and [`PeriodicPhonons::masses`] is what
    /// de-weights it.
    ///
    /// A frequency says how fast the crystal vibrates; this says how. Without it there is no way
    /// to tell an optical branch from an acoustic one, or to see which sublattice a soft mode
    /// moves — and it was being computed and discarded.
    pub modes: Matrix,
    /// Atomic masses (amu) in the molecule's atom order, for de-weighting [`Self::modes`].
    pub masses: Vec<f64>,
    /// The largest acoustic `|ω|` **before** the acoustic branch was projected out.
    ///
    /// Zero in exact arithmetic, so this measures how well the lattice sums cancelled — the same
    /// diagnostic `rigid_residual_cm` is for a molecule. Through 0.2.3 it was the third smallest
    /// `|ω|` of the *corrected* spectrum, which is the number the correction had just flattened.
    pub acoustic_residual_cm: f64,
}

/// The Γ-point Hessian: `∂²E/∂R_iα ∂R_jβ` per cell, in eV/Bohr².
pub fn periodic_hessian(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
) -> Result<Matrix> {
    let scf = run_gamma(molecule, params, options, periodic)?;
    hessian_at(molecule, params, options, periodic, &scf)
}

/// Γ-point phonon frequencies, from the analytic Hessian.
pub fn periodic_phonons(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
) -> Result<PeriodicPhonons> {
    let scf = run_gamma(molecule, params, options, periodic)?;
    // The Hessian is handed to `frequencies` by reference and comes back untouched: the acoustic
    // branch is projected out of the mass-weighted copy inside, so what is stored below is the
    // raw second derivative.
    //
    // It used to be `enforce_acoustic_sum_rule(&mut hessian)` here, which had two problems. The
    // row-wise correction is not symmetric -- it subtracts a per-row share from column entries
    // only -- so the stored `hessian` came back non-symmetric, and `symmetric_eigen` reads one
    // triangle and silently re-symmetrizes, which partly undid the sum rule that had just been
    // imposed. And the residual was then read off the *corrected* spectrum, so it reported the
    // number the correction had just flattened rather than the one worth seeing.
    let hessian = hessian_at(molecule, params, options, periodic, &scf)?;
    let solved = frequencies(molecule, params, &hessian)?;
    Ok(PeriodicPhonons {
        scf,
        hessian,
        frequencies_cm: solved.frequencies_cm,
        eigenvalues: solved.eigenvalues,
        modes: solved.modes,
        masses: solved.masses,
        acoustic_residual_cm: solved.acoustic_residual_cm,
    })
}

fn hessian_at(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
    scf: &PeriodicResult,
) -> Result<Matrix> {
    if scf.unrestricted {
        return Err(Pm3Error::InvalidInput(
            "the periodic analytic Hessian is restricted-only so far; an open-shell cell needs \
             the unrestricted CPHF response, which is not wired up yet"
                .to_string(),
        ));
    }
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let basis = Basis::build(molecule, params)?;

    // Where a Γ-point Hessian's time goes. Same discipline as `PM3_GAMMA_PROFILE`: the Ewald
    // finding in this release came from measuring rather than reading, after three plausible
    // guesses had each changed the wall clock by nothing. Set `PM3_HESSIAN_PROFILE=1`.
    let profile = std::env::var_os("PM3_HESSIAN_PROFILE").is_some();
    let clock = std::time::Instant::now();
    let mut mark = 0.0f64;
    let lap = |name: &str, clock: &std::time::Instant, mark: &mut f64| {
        if profile {
            let now = clock.elapsed().as_secs_f64();
            eprintln!("[hessian profile] {name:<14} {:>8.2} s", now - *mark);
            *mark = now;
        }
    };

    let mut hessian = skeleton(molecule, params, options, periodic, scf, &basis)?;
    lap("skeleton", &clock, &mut mark);

    // The lattice sum's own second derivative, at fixed charges.
    let cell = molecule.cell.expect("run_gamma requires a cell");
    let (sites, ewald_params) = charge_sites(molecule, params, &basis, &scf.density, periodic)?;
    if let Some(from_ewald) = ewald_atom_hessian(&cell, &sites, &ewald_params, nat)? {
        for i in 0..ndof {
            for j in 0..ndof {
                hessian[(i, j)] += from_ewald[(i, j)];
            }
        }
    } else {
        // 1D and 2D answer through the phased sum at q = 0 now; only an isolated cell reports
        // no Ewald Hessian, and it has a better home than this path.
        return Err(Pm3Error::InvalidInput(
            "the cell has no periodic direction; use the molecular Hessian for an isolated \
             system"
                .to_string(),
        ));
    }

    // The classical corrections, lattice summed.
    if options.variant != Variant::Pm3 {
        let from_corrections =
            correction_hessian(molecule, options.variant, &periodic.correction_cutoffs);
        for i in 0..ndof {
            for j in 0..ndof {
                hessian[(i, j)] += from_corrections[(i, j)];
            }
        }
    }

    // The density's response.
    lap("ewald + corr", &clock, &mut mark);
    response(
        molecule,
        params,
        options,
        periodic,
        scf,
        &basis,
        &mut hessian,
    )?;
    lap("response", &clock, &mut mark);

    // Average away the asymmetry that different summation orders leave behind.
    for i in 0..ndof {
        for j in (i + 1)..ndof {
            let mean = 0.5 * (hessian[(i, j)] + hessian[(j, i)]);
            hessian[(i, j)] = mean;
            hessian[(j, i)] = mean;
        }
    }
    Ok(hessian)
}

/// One pair's contribution: the two atom indices and the `3×3` block of its displacement Hessian.
pub(crate) type PairBlock = (usize, usize, [[f64; 3]; 3]);

/// Scatter a pair's `3×3` displacement Hessian onto the four atom blocks it belongs to.
///
/// For `E(d)` with `d = R_b − R_a`, `∂²E/∂R_a∂R_a = ∂²E/∂R_b∂R_b = +H` and the mixed blocks are
/// `−H`. A self-image, where `a == b`, cancels to zero — which is right, since its displacement
/// does not depend on the atom's position at all.
fn scatter_pair(hessian: &mut Matrix, a: usize, b: usize, block: &[[f64; 3]; 3]) {
    for (alpha, row) in block.iter().enumerate() {
        for (beta, value) in row.iter().enumerate() {
            hessian[(3 * a + alpha, 3 * a + beta)] += value;
            hessian[(3 * b + alpha, 3 * b + beta)] += value;
            hessian[(3 * a + alpha, 3 * b + beta)] -= value;
            hessian[(3 * b + alpha, 3 * a + beta)] -= value;
        }
    }
}

/// The fixed-density second derivative of everything that depends on an image displacement.
pub(crate) fn skeleton(
    molecule: &Molecule,
    params: &Pm3Parameters,
    _options: &Pm3Options,
    periodic: &PeriodicOptions,
    scf: &PeriodicResult,
    basis: &Basis,
) -> Result<Matrix> {
    let cell = molecule.cell.expect("run_gamma requires a cell");
    let nat = molecule.atoms.len();
    let mut hessian = Matrix::zeros(3 * nat, 3 * nat);

    let mut atom_sites = Vec::with_capacity(nat);
    for atom in &molecule.atoms {
        let elem = params.element(atom.z)?;
        atom_sites.push(AtomSites::build(elem).ok_or(Pm3Error::UnsupportedDOrbitals(atom.z))?);
    }
    let cutoff = periodic.short_range_cutoff.max(periodic.switch.off);
    let positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
    let list = NeighborList::build_from_positions(&positions, Some(&cell), cutoff);

    let pairs: Vec<_> = list.unique().collect();
    let blocks: Vec<Result<PairBlock>> = pairs
        .par_iter()
        .map(|pair| {
            pair_block(
                molecule,
                params,
                periodic,
                &scf.density,
                &[(&scf.density, 0.5)],
                basis,
                &atom_sites,
                pair.a,
                pair.b,
                pair.dvec,
                pair.r,
                // A Γ-point Hessian, where `P(T) = P(0)` is the sampling rather than an
                // approximation on top of it.
                None,
            )
        })
        .collect();
    for entry in blocks {
        let (a, b, block) = entry?;
        scatter_pair(&mut hessian, a, b, &block);
    }
    Ok(hessian)
}

/// One image pair's `3×3` second derivative with respect to its displacement.
///
/// The same terms the gradient differentiates once, differentiated twice. Keeping them in one
/// place is deliberate: the switch, the point-charge subtraction and the exchange cutoff all have
/// to agree between the energy, the gradient and the Hessian, and three copies of that agreement
/// is two too many to keep straight.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pair_block(
    molecule: &Molecule,
    params: &Pm3Parameters,
    periodic: &PeriodicOptions,
    // The density, not the whole result: this is a fixed-density second derivative, and taking
    // the matrix rather than the PeriodicResult is what lets the k-point path reuse it.
    density: &Matrix,
    // What **exchange** reads, and by how much: `[(P, ½)]` for a closed shell and
    // `[(P^α, 1), (P^β, 1)]` for an open one. The two agree exactly when `P^α = P^β = P/2`, which
    // is what keeps the restricted path bit-for-bit unchanged. Coulomb always reads `density`.
    exchange: &[(&Matrix, f64)],
    basis: &Basis,
    atom_sites: &[AtomSites],
    a: usize,
    b: usize,
    dvec_in: Vec3,
    r: f64,
    // `P(T)` for **this** image's `(a, b)` block, oriented as the `a` and `b` arguments are.
    //
    // `None` means "use `density`'s own `(a, b)` block", which asserts `P(T) = P(0)`. That is
    // exactly true at Γ — one k-point cannot distinguish images — and false on a k-mesh, where
    // `P(0)` is the Brillouin-zone average and `P(T)` decays with `T`. Passing `None` from a
    // meshed ground state is what made a meshed `D(0)` anisotropic on a cubic crystal and put it
    // 300% away from a finite difference; the on-site blocks below are `P(0)` either way and are
    // untouched by this.
    image: Option<&crate::pbc::kscf::ImagePair<'_>>,
) -> Result<PairBlock> {
    use crate::pbc::gradient::resonance_beta;

    let ea = params.element(molecule.atoms[a].z)?;
    let eb = params.element(molecule.atoms[b].z)?;
    let heavy_first = ea.n_orb >= eb.n_orb;
    let (first_index, second_index) = if heavy_first { (a, b) } else { (b, a) };
    let (first, second) = if heavy_first { (ea, eb) } else { (eb, ea) };
    let dvec = if heavy_first { dvec_in } else { dvec_in * -1.0 };

    let seeded = [
        Dual2::var(dvec.x, 0),
        Dual2::var(dvec.y, 1),
        Dual2::var(dvec.z, 2),
    ];
    let te = pair_two_electron_g::<Dual2>(first, second, seeded);
    let point = point_pair_g::<Dual2>(
        &atom_sites[first_index],
        first.core_charge,
        &atom_sites[second_index],
        second.core_charge,
        seeded,
    );
    let r_dual = (seeded[0] * seeded[0] + seeded[1] * seeded[1] + seeded[2] * seeded[2]).sqrt();
    let switch = periodic.switch.at_g(r_dual);

    let (na, nb) = (first.n_orb, second.n_orb);
    let off_first = basis.atom_offset[first_index];
    let off_second = basis.atom_offset[second_index];

    // Accumulate the whole pair energy as one Dual2, then read its Hessian off at the end. That
    // is cheaper than carrying a 3×3 accumulator through every term and, more to the point, it
    // cannot get the bookkeeping wrong.
    let mut energy = Dual2::constant(0.0);

    // The off-diagonal `(first, second)` density element, from `P(T)` where there is one.
    //
    // `pair_block` reorders its two atoms so the heavier basis comes first, and `image` is
    // oriented as the caller's `(a, b)` — so when the swap happened the indices swap with it.
    let off_diagonal_total = |mu: usize, la: usize| -> f64 {
        match image {
            None => density[(off_first + mu, off_second + la)],
            Some(p) if heavy_first => p.total(mu, la),
            Some(p) => p.total(la, mu),
        }
    };
    // The same element for one exchange channel. `exchange` is `[(P, ½)]` for a closed shell and
    // `[(P^α, 1), (P^β, 1)]` for an open one — the same two shapes `ImagePair` distinguishes, so
    // a single channel reads the total and two read the spins in order.
    let off_diagonal_exchange = |channel: usize, spin: &Matrix, mu: usize, la: usize| -> f64 {
        match image {
            None => spin[(off_first + mu, off_second + la)],
            Some(p) => {
                let (i, j) = if heavy_first { (mu, la) } else { (la, mu) };
                if exchange.len() == 1 {
                    p.total(i, j)
                } else {
                    p.spin(channel, i, j)
                }
            }
        }
    };

    for mu in 0..na {
        for nu in 0..na {
            let coefficient = density[(off_first + mu, off_first + nu)];
            energy = energy + (te.e1b[mu][nu] - point.e1b[mu * na + nu]) * switch * coefficient;
        }
    }
    for la in 0..nb {
        for si in 0..nb {
            let coefficient = density[(off_second + la, off_second + si)];
            energy = energy + (te.e2a[la][si] - point.e2a[la * nb + si]) * switch * coefficient;
        }
    }

    let inside_short_range = r <= periodic.short_range_cutoff;
    if inside_short_range {
        let overlap = crate::overlap::diatom_overlap_dual2(first, Vec3::zero(), second, dvec)?;
        #[allow(clippy::needless_range_loop)]
        for mu in 0..na.min(4) {
            let bi = resonance_beta(first, basis.aos[off_first + mu].orb);
            for la in 0..nb.min(4) {
                let bj = resonance_beta(second, basis.aos[off_second + la].orb);
                let coefficient = off_diagonal_total(mu, la) * (bi + bj);
                energy = energy + overlap[mu][la] * coefficient;
            }
        }
    }

    let npack_j = nb * (nb + 1) / 2;
    for mu in 0..na {
        for nu in 0..na {
            for la in 0..nb {
                for si in 0..nb {
                    let index = pack(mu, nu) * npack_j + pack(la, si);
                    let weight = density[(off_first + mu, off_first + nu)]
                        * density[(off_second + la, off_second + si)];
                    energy = energy + (te.w[index] - point.w[index]) * switch * weight;
                    if inside_short_range {
                        let mut weight = 0.0;
                        for (channel, (spin, scale)) in exchange.iter().enumerate() {
                            weight -= scale
                                * off_diagonal_exchange(channel, spin, mu, la)
                                * off_diagonal_exchange(channel, spin, nu, si);
                        }
                        energy = energy + te.w[index] * weight;
                    }
                }
            }
        }
    }

    let full = crate::repulsion::pair_core_energy_scalar::<Dual2>(
        params,
        first,
        second,
        molecule.atoms[first_index].z,
        molecule.atoms[second_index].z,
        r_dual,
    );
    let rho = first.po[9] + second.po[9];
    let charge_product = crate::constants::PM3_EV * first.core_charge * second.core_charge;
    let bare = (r_dual * r_dual + rho * rho).sqrt().recip() * charge_product;
    let point_monopole = r_dual.recip() * charge_product;
    energy = energy + full - bare + (bare - point_monopole) * switch;

    let mut block = [[0.0; 3]; 3];
    for (alpha, row) in block.iter_mut().enumerate() {
        for (beta, slot) in row.iter_mut().enumerate() {
            *slot = energy.h[alpha][beta];
        }
    }
    Ok((first_index, second_index, block))
}

/// The classical corrections' second derivative, lattice summed.
///
/// Structurally the molecular version with the cluster standing in for the molecule: seeding a
/// *parent* atom's coordinate propagates the seed to every image of it, which is exactly right,
/// because moving an atom moves all of its copies.
fn correction_hessian(
    molecule: &Molecule,
    variant: Variant,
    cutoffs: &CorrectionCutoffs,
) -> Matrix {
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let mut hessian = Matrix::zeros(ndof, ndof);
    if variant == Variant::Pm3 {
        return hessian;
    }
    let cluster = build_cluster(molecule, cutoffs.dispersion.max(cutoffs.short_range));
    let base: Vec<[Dual2; 3]> = cluster
        .positions
        .iter()
        .map(|p| {
            [
                Dual2::constant(p[0]),
                Dual2::constant(p[1]),
                Dual2::constant(p[2]),
            ]
        })
        .collect();
    let energy_of = |seeded: &[[Dual2; 3]]| -> Dual2 {
        let positions = cluster_positions_g::<Dual2>(&cluster, seeded);
        correction_energy_cluster_g::<Dual2>(
            &cluster.numbers,
            &positions,
            cluster.n_cell,
            Some(&cluster.parent),
            None,
            Some(cutoffs.dispersion),
            Some(cutoffs.coordination),
            variant,
        )
    };

    // Diagonal blocks: all three coordinates of one atom at once.
    let diagonal: Vec<[[f64; 3]; 3]> = (0..nat)
        .into_par_iter()
        .map(|a| {
            let mut seeded = base.clone();
            seeded[a] = [
                Dual2::var(cluster.positions[a][0], 0),
                Dual2::var(cluster.positions[a][1], 1),
                Dual2::var(cluster.positions[a][2], 2),
            ];
            energy_of(&seeded).h
        })
        .collect();
    for (a, block) in diagonal.iter().enumerate() {
        for (k, row) in block.iter().enumerate() {
            for (l, value) in row.iter().enumerate() {
                hessian[(3 * a + k, 3 * a + l)] = *value;
            }
        }
    }

    // Off-diagonal blocks: one coordinate of each atom, read off the mixed derivative.
    let pairs: Vec<(usize, usize)> = (0..nat)
        .flat_map(|a| ((a + 1)..nat).map(move |b| (a, b)))
        .collect();
    let blocks: Vec<[[f64; 3]; 3]> = pairs
        .par_iter()
        .map(|&(a, b)| {
            let mut block = [[0.0; 3]; 3];
            for (k, row) in block.iter_mut().enumerate() {
                for (l, slot) in row.iter_mut().enumerate() {
                    let mut seeded = base.clone();
                    seeded[a][k] = Dual2::var(cluster.positions[a][k], 0);
                    seeded[b][l] = Dual2::var(cluster.positions[b][l], 1);
                    *slot = energy_of(&seeded).h[0][1];
                }
            }
            block
        })
        .collect();
    for (&(a, b), block) in pairs.iter().zip(&blocks) {
        for (k, row) in block.iter().enumerate() {
            for (l, value) in row.iter().enumerate() {
                hessian[(3 * a + k, 3 * b + l)] = *value;
                hessian[(3 * b + l, 3 * a + k)] = *value;
            }
        }
    }
    hessian
}

/// The multipole charge sites at the converged density, cores included.
pub(crate) fn charge_sites(
    molecule: &Molecule,
    params: &Pm3Parameters,
    basis: &Basis,
    density: &Matrix,
    periodic: &PeriodicOptions,
) -> Result<(
    Vec<crate::pbc::ewald::ChargeSite>,
    crate::pbc::ewald::EwaldParams,
)> {
    let cell = molecule.cell.expect("periodic");
    let mut sites = Vec::new();
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let elem = params.element(atom.z)?;
        let n = basis.atom_norb[ia];
        let off = basis.atom_offset[ia];
        let mut block = vec![0.0; n * n];
        for mu in 0..n {
            for nu in 0..n {
                block[mu * n + nu] = density[(off + mu, off + nu)];
            }
        }
        let atom_sites = AtomSites::build(elem).ok_or(Pm3Error::UnsupportedDOrbitals(atom.z))?;
        let charges = atom_sites.charges(&block, elem.core_charge);
        for (offset, charge) in atom_sites.offsets.iter().zip(&charges) {
            sites.push(crate::pbc::ewald::ChargeSite {
                position: atom.position + *offset,
                charge: *charge,
                owner: ia,
            });
        }
    }
    let ewald_params = periodic.ewald.unwrap_or_else(|| {
        crate::pbc::ewald::EwaldParams::for_cell(&cell, crate::pbc::ewald::DEFAULT_ACCURACY)
    });
    Ok((sites, ewald_params))
}

/// Zero the residual net force on the whole cell, by projecting the three uniform translations
/// out of the Cartesian Hessian.
///
/// Every term in the Hessian is a function of interatomic displacements, so each row-block already
/// sums to zero up to rounding. This removes that rounding, which matters because the acoustic
/// frequencies come out as the square root of nearly-cancelling numbers.
///
/// [`periodic_phonons`] no longer calls this — it projects the mass-weighted copy instead and
/// leaves the Hessian it returns raw. This stays for a caller assembling force constants of its
/// own.
///
/// **Both sides, not one.** The row-wise version this replaces subtracted a per-row share from
/// column entries only, which left a symmetric matrix asymmetric by `c[j][β_i] − c[i][β_j]`. That
/// matters more than the size of the violation suggests: [`crate::linalg::symmetric_eigen`] reads
/// only the lower triangle, so the matrix actually diagonalized was `tril(H) + tril(H)ᵀ`, which
/// does not satisfy the sum rule that had just been imposed on it. The correction was undone, in
/// part, by the very step it existed to serve.
pub fn enforce_acoustic_sum_rule(hessian: &mut Matrix) {
    let nat = hessian.rows / 3;
    if nat == 0 {
        return;
    }
    // Unweighted translations: a Cartesian Hessian's null vectors are uniform displacements, so
    // every site carries the same amplitude here rather than `√m`.
    let positions = vec![crate::math::Vec3::new(0.0, 0.0, 0.0); nat];
    let unit = vec![1.0; nat];
    let basis = crate::rigid::rigid_body_basis(
        &positions,
        &unit,
        crate::rigid::RigidMotions::TranslationsOnly,
    );
    crate::rigid::project_out_symmetric(&basis, hessian);
}

/// Mass-weight and diagonalize, returning frequencies in cm⁻¹ (negative = imaginary), the
/// eigenvalues, and the largest acoustic wavenumber found **before** the acoustic branch was
/// projected out.
///
/// The three translations are removed by projection here, on the mass-weighted copy, exactly as
/// [`crate::hessian::vibrational_analysis`] does for a molecule. Only translations: a crystal is
/// not invariant under rotating its contents inside a fixed lattice, so the rotational generators
/// are not null vectors of this matrix and projecting them out would delete real restoring force.
///
/// The caller's Hessian is **not** modified. That is the point of doing it here rather than to
/// the matrix itself: `PeriodicPhonons::hessian` and `periodic_hessian` hand back the raw second
/// derivative, which is what anyone doing their own analysis needs.
/// What diagonalizing the mass-weighted Γ-point matrix produces.
///
/// `modes` are the **mass-weighted** eigenvectors, one per column, in the same order as the
/// frequencies. They used to be computed and dropped on the floor here, which left a Γ-point
/// phonon calculation able to say how fast the crystal vibrates and not how — and the "how" is
/// what tells an optical mode from an acoustic one, or says which sublattice a soft mode moves.
struct Diagonalized {
    frequencies_cm: Vec<f64>,
    eigenvalues: Vec<f64>,
    modes: Matrix,
    masses: Vec<f64>,
    acoustic_residual_cm: f64,
}

fn frequencies(
    molecule: &Molecule,
    params: &Pm3Parameters,
    hessian: &Matrix,
) -> Result<Diagonalized> {
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let mut masses = Vec::with_capacity(nat);
    for atom in &molecule.atoms {
        masses.push(params.element(atom.z)?.mass);
    }
    // eV/Bohr² → eV/(Å²·amu), the units the cm⁻¹ conversion is defined against.
    let per_angstrom_squared =
        crate::constants::ANGSTROM_TO_BOHR * crate::constants::ANGSTROM_TO_BOHR;
    let mut weighted = Matrix::zeros(ndof, ndof);
    for i in 0..ndof {
        for j in 0..ndof {
            let scale = (masses[i / 3] * masses[j / 3]).sqrt();
            weighted[(i, j)] = if scale > 0.0 {
                hessian[(i, j)] * per_angstrom_squared / scale
            } else {
                0.0
            };
        }
    }

    let positions: Vec<crate::math::Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
    let acoustic = crate::rigid::rigid_body_basis(
        &positions,
        &masses,
        crate::rigid::RigidMotions::TranslationsOnly,
    );
    // What the acoustic branch carried before it was removed. This is what
    // `acoustic_residual_cm` now reports: a measurement of the lattice sums' quality, taken
    // before the projection rather than after it. Reading it off the projected spectrum, as it
    // used to be, reported the number the projection had just set to zero.
    let residual_cm = crate::rigid::rayleigh_quotients(&acoustic, &weighted)
        .iter()
        .map(|&lam| crate::hessian::signed_wavenumber(lam).abs())
        .fold(0.0_f64, f64::max);
    crate::rigid::project_out_symmetric(&acoustic, &mut weighted);

    let (mut eigenvalues, modes) = symmetric_eigen(&weighted)?;
    for index in crate::rigid::rigid_mode_indices(&acoustic, &modes) {
        eigenvalues[index] = 0.0;
    }
    let frequencies_cm = eigenvalues
        .iter()
        .map(|&value| crate::hessian::signed_wavenumber(value))
        .collect();
    Ok(Diagonalized {
        frequencies_cm,
        eigenvalues,
        modes,
        masses,
        acoustic_residual_cm: residual_cm,
    })
}

/// The CPHF response: the part of the second derivative that comes from the density moving.
fn response(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
    scf: &PeriodicResult,
    basis: &Basis,
    hessian: &mut Matrix,
) -> Result<()> {
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let nao = basis.nao;
    let n_occ = scf.n_occ;
    let nvir = nao - n_occ;
    if n_occ == 0 || nvir == 0 {
        return Ok(());
    }

    let co = columns(&scf.mo_coeff, 0, n_occ);
    let cv = columns(&scf.mo_coeff, n_occ, nvir);
    let mut denominator = Matrix::zeros(nvir, n_occ);
    for a in 0..nvir {
        for i in 0..n_occ {
            denominator[(a, i)] = scf.mo_energies[i] - scf.mo_energies[n_occ + a];
        }
    }

    // The skeleton derivative of the Fock matrix, projected to the occupied–virtual block. This is
    // the right-hand side of the CPHF equations and also, contracted with the response, the
    // relaxation part of the Hessian.
    let kernel = super::kernel::PeriodicKernel::build(molecule, params, periodic)?;
    let g_ov: Vec<Matrix> = (0..ndof)
        .into_par_iter()
        .map(|dof| {
            let derivative =
                fock_derivative(molecule, params, periodic, scf, basis, dof / 3, dof % 3)?;
            Ok(project_ov(&derivative, &cv, &co))
        })
        .collect::<Result<Vec<_>>>()?;

    let responses: Vec<Matrix> = g_ov
        .par_iter()
        .map(|rhs| cphf(rhs, &denominator, &cv, &co, &kernel, options.cphf_max_iter))
        .collect::<Result<Vec<_>>>()?;

    for (b, u) in responses.iter().enumerate() {
        for (a, g) in g_ov.iter().enumerate() {
            hessian[(a, b)] += 4.0 * g.frobenius_dot(u);
        }
    }
    Ok(())
}

fn columns(c: &Matrix, start: usize, count: usize) -> Matrix {
    let mut out = Matrix::zeros(c.rows, count);
    for i in 0..c.rows {
        for j in 0..count {
            out[(i, j)] = c[(i, start + j)];
        }
    }
    out
}

fn project_ov(f: &Matrix, cv: &Matrix, co: &Matrix) -> Matrix {
    let m = f.matmul_seq(co);
    cv.transpose_matmul_seq(&m)
}

/// Solve the CPHF equations for one perturbation.
fn cphf(
    g_ov: &Matrix,
    denominator: &Matrix,
    cv: &Matrix,
    co: &Matrix,
    kernel: &super::kernel::PeriodicKernel,
    max_iter: usize,
) -> Result<Matrix> {
    let divide = |numerator: &Matrix| -> Matrix {
        let mut u = numerator.clone();
        for (value, d) in u.as_mut_slice().iter_mut().zip(denominator.as_slice()) {
            *value = if d.abs() < 1.0e-10 { 0.0 } else { *value / *d };
        }
        u
    };
    let mut u = divide(g_ov);
    // Pulay-extrapolated and *checked*, as the molecular solvers now are. This one ran its two
    // hundred passes and returned whatever it was holding, so a periodic Hessian could come back
    // built on a response that had not converged, with nothing in the result to say so.
    let mut hist_u: Vec<Matrix> = Vec::new();
    let mut hist_e: Vec<Matrix> = Vec::new();
    let max_diis = 8;
    let mut converged = false;
    let mut residual = f64::INFINITY;
    for _ in 0..max_iter {
        // The AO-basis response density for the current `U`, symmetrized and scaled the way the
        // closed-shell density is.
        let response_density = ao_response_density(&u, cv, co);
        let applied = kernel.apply(&response_density)?;
        let projected = project_ov(&applied, cv, co);
        let mut rhs = g_ov.clone();
        for (value, extra) in rhs.as_mut_slice().iter_mut().zip(projected.as_slice()) {
            *value += extra;
        }
        let next = divide(&rhs);
        let mut error = next.clone();
        let mut change = 0.0;
        for (value, old) in error.as_mut_slice().iter_mut().zip(u.as_slice()) {
            *value -= *old;
            change += *value * *value;
        }
        residual = change.sqrt();
        if residual < PERIODIC_CPHF_TOLERANCE {
            u = next;
            converged = true;
            break;
        }
        hist_u.push(next.clone());
        hist_e.push(error);
        if hist_u.len() > max_diis {
            hist_u.remove(0);
            hist_e.remove(0);
        }
        u = crate::hessian::cphf_diis(&hist_u, &hist_e).unwrap_or(next);
    }
    if !converged {
        return Err(Pm3Error::ScfNotConverged {
            iterations: max_iter,
            error: residual,
            diagnosis: None,
        });
    }
    Ok(u)
}

// How many passes the Γ-point periodic response takes before it has failed is
// `Pm3Options::cphf_max_iter`, whose default of 400 is the constant that used to live here.
// Headroom, for the same reason as `crate::hessian::CPHF_ITERATIONS`, and with the same history:
// at two hundred, and with no extrapolation at all, a water chain's response stopped at a
// residual of `1.2e-7` against a declared tolerance of `1e-10` and returned it as though it had
// converged. This loop now extrapolates with the shared Pulay solve and checks that it arrived.

/// Convergence on the RMS change of the response between passes.
const PERIODIC_CPHF_TOLERANCE: f64 = 1.0e-10;

/// `∂P = 4 Cv U Coᵀ`, symmetrized — the closed-shell response density in the AO basis.
fn ao_response_density(u: &Matrix, cv: &Matrix, co: &Matrix) -> Matrix {
    let temporary = cv.matmul_seq(u);
    let mut out = temporary.matmul_transpose_seq(co);
    let n = out.rows;
    for i in 0..n {
        for j in 0..i {
            // The *sum* of the two triangles, not their mean: the diagonal below takes ×4 and
            // an off-diagonal pair takes ×2 of the sum, which is the same ×4 spread over the two
            // entries. Naming it `mean` invited the wrong fix.
            let symmetrized = out[(i, j)] + out[(j, i)];
            out[(i, j)] = 2.0 * symmetrized;
            out[(j, i)] = 2.0 * symmetrized;
        }
        out[(i, i)] *= 4.0;
    }
    out
}

/// `∂F/∂R_{atom,axis}` at fixed density, in the AO basis.
fn fock_derivative(
    molecule: &Molecule,
    params: &Pm3Parameters,
    periodic: &PeriodicOptions,
    scf: &PeriodicResult,
    basis: &Basis,
    atom: usize,
    axis: usize,
) -> Result<Matrix> {
    super::kernel::fock_derivative(molecule, params, periodic, &scf.density, basis, atom, axis)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Cell;
    use crate::pbc::gradient::periodic_gradient;

    const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n";

    fn celled(xyz: &str, edge: f64) -> Molecule {
        let mut molecule = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        molecule.cell = Some(Cell::cubic(edge).unwrap());
        molecule
    }

    /// The analytic Hessian against central differences of the **analytic gradient**.
    ///
    /// This is the only test that matters for a Hessian. It exercises the skeleton second
    /// derivatives, the Ewald second derivative, the Fock derivative that forms the CPHF
    /// right-hand side, and the CPHF solve itself, against a reference that shares none of that
    /// machinery — the gradient is a first derivative and is independently validated.
    #[test]
    fn matches_finite_differences_of_the_analytic_gradient() {
        let params = Pm3Parameters::standard().unwrap();
        let options = Pm3Options::default();
        let periodic = PeriodicOptions::default();
        let base = celled(WATER, 18.0);
        let analytic = periodic_hessian(&base, &params, &options, &periodic).unwrap();

        let step = 2.0e-4;
        for atom in 0..base.atoms.len() {
            for axis in 0..3 {
                let gradient_at = |shift: f64| -> Vec<f64> {
                    let mut moved = base.clone();
                    match axis {
                        0 => moved.atoms[atom].position.x += shift,
                        1 => moved.atoms[atom].position.y += shift,
                        _ => moved.atoms[atom].position.z += shift,
                    }
                    periodic_gradient(&moved, &params, &options, &periodic)
                        .unwrap()
                        .gradient
                        .iter()
                        .flat_map(|g| g.to_array())
                        .collect()
                };
                let plus = gradient_at(step);
                let minus = gradient_at(-step);
                let row = 3 * atom + axis;
                for (column, (p, m)) in plus.iter().zip(&minus).enumerate() {
                    let numerical = (p - m) / (2.0 * step);
                    let exact = analytic[(row, column)];
                    assert!(
                        (numerical - exact).abs() < 3.0e-3,
                        "({row},{column}): analytic {exact} vs finite difference {numerical}"
                    );
                }
            }
        }
    }

    /// A second derivative of a scalar is symmetric.
    #[test]
    fn the_hessian_is_symmetric() {
        let params = Pm3Parameters::standard().unwrap();
        let hessian = periodic_hessian(
            &celled(WATER, 18.0),
            &params,
            &Pm3Options::default(),
            &PeriodicOptions::default(),
        )
        .unwrap();
        for i in 0..hessian.rows {
            for j in 0..hessian.rows {
                let difference = (hessian[(i, j)] - hessian[(j, i)]).abs();
                assert!(
                    difference < 1.0e-9,
                    "({i},{j}) asymmetric by {difference:.3e}"
                );
            }
        }
    }

    /// The acoustic sum rule, *before* it is enforced.
    ///
    /// Every term is a function of interatomic displacements, so the rows already sum to zero.
    /// Checking that here rather than trusting [`enforce_acoustic_sum_rule`] is the point: if a
    /// term were being scattered to the wrong atom, enforcement would hide it.
    #[test]
    fn the_acoustic_sum_rule_holds_before_enforcement() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = celled(WATER, 18.0);
        let hessian = periodic_hessian(
            &molecule,
            &params,
            &Pm3Options::default(),
            &PeriodicOptions::default(),
        )
        .unwrap();
        let nat = molecule.atoms.len();
        for row in 0..hessian.rows {
            for beta in 0..3 {
                let total: f64 = (0..nat).map(|atom| hessian[(row, 3 * atom + beta)]).sum();
                assert!(
                    total.abs() < 1.0e-6,
                    "row {row}, axis {beta}: sums to {total:.3e}"
                );
            }
        }
    }

    /// A molecule alone in a large cell must give the molecular Hessian.
    ///
    /// The bridge to the MOPAC-validated molecular path: whatever the periodic machinery does, in
    /// the isolated limit it has to reduce to something already checked against an external
    /// oracle.
    #[test]
    fn a_molecule_in_a_large_cell_reproduces_the_molecular_hessian() {
        let params = Pm3Parameters::standard().unwrap();
        let options = Pm3Options::default();
        let molecule = celled(WATER, 30.0);
        let periodic_result =
            periodic_hessian(&molecule, &params, &options, &PeriodicOptions::default()).unwrap();
        let molecular = crate::hessian::analytic_hessian(
            &Molecule::from_xyz_str(WATER, 0.0).unwrap(),
            &params,
            &options,
            1.0e-3,
        )
        .unwrap();
        for i in 0..molecular.rows {
            for j in 0..molecular.rows {
                let difference = (periodic_result[(i, j)] - molecular[(i, j)]).abs();
                assert!(
                    difference < 2.0e-3,
                    "({i},{j}): periodic {} vs molecular {} ({difference:.3e})",
                    periodic_result[(i, j)],
                    molecular[(i, j)]
                );
            }
        }
    }

    /// Γ-point phonons of a relaxed cell: three acoustic modes at zero and the rest real.
    #[test]
    fn phonons_have_three_acoustic_modes() {
        use crate::pbc::optimize::{relax, PeriodicOptOptions};
        let params = Pm3Parameters::standard().unwrap();
        let options = Pm3Options::default();
        let periodic = PeriodicOptions::default();
        let relaxed = relax(
            &celled(WATER, 18.0),
            &params,
            &options,
            &periodic,
            &PeriodicOptOptions {
                max_iter: 80,
                gtol: 1.0e-5,
                ..PeriodicOptOptions::default()
            },
        )
        .unwrap();
        let phonons = periodic_phonons(&relaxed.molecule, &params, &options, &periodic).unwrap();
        assert_eq!(phonons.frequencies_cm.len(), 9);
        let mut magnitudes: Vec<f64> = phonons.frequencies_cm.iter().map(|f| f.abs()).collect();
        magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Three acoustic branches at **exactly** zero. That is a statement about translational
        // invariance and holds in any crystal, so it is imposed by projecting the three
        // translations out of the mass-weighted matrix rather than checked to within 5 cm⁻¹.
        assert_eq!(
            phonons.frequencies_cm.iter().filter(|f| **f == 0.0).count(),
            3,
            "expected exactly three zeros, got {:?}",
            phonons.frequencies_cm
        );
        assert_eq!(magnitudes[2], 0.0);
        // `acoustic_residual_cm` is now the **pre**-projection number, so it is a measurement of
        // the lattice sums rather than a restatement of what the projection just did. Small, and
        // genuinely non-zero -- reading it off the projected spectrum, as 0.2.3 did, could only
        // ever return zero and so could never have failed.
        assert!(
            phonons.acoustic_residual_cm > 0.0 && phonons.acoustic_residual_cm < 5.0,
            "the pre-projection acoustic residual is {:.3} cm⁻¹",
            phonons.acoustic_residual_cm
        );
        // And the Hessian handed back is the raw one: symmetric, with nothing imposed on it.
        for i in 0..phonons.hessian.rows {
            for j in 0..i {
                assert_eq!(
                    phonons.hessian[(i, j)],
                    phonons.hessian[(j, i)],
                    "the returned Hessian is not symmetric at ({i}, {j})"
                );
            }
        }

        // The *librations* are also soft here, and deliberately not asserted against. An 18 Bohr
        // cell of water is essentially a gas of non-interacting molecules, so rotating one costs
        // almost nothing: three more near-zero modes, which is right rather than wrong. In a real
        // molecular crystal they would be the librational branches at a few hundred cm⁻¹.
        //
        // What must be there regardless are the molecule's three internal modes — a bend and two
        // stretches — which no amount of cell weirdness moves below 1000 cm⁻¹.
        assert!(
            magnitudes[6] > 1000.0,
            "the three internal modes should survive, lowest is {:.3} cm⁻¹",
            magnitudes[6]
        );
        // No *stiff* mode is imaginary. The soft ones are excluded on purpose: a nearly-free
        // libration has a true frequency of zero, and whether the square root of a number that
        // small comes out at `+3` or `−9 cm⁻¹` is rounding in the residual gradient, not a
        // saddle. Asserting a sign there would be asserting noise.
        for frequency in &phonons.frequencies_cm {
            if frequency.abs() > 50.0 {
                assert!(
                    *frequency > 0.0,
                    "imaginary stiff mode at a minimum: {frequency:.3} cm⁻¹"
                );
            }
        }
    }

    /// Corrections contribute, and they contribute the lattice-summed amount.
    #[test]
    fn the_corrected_variant_changes_the_hessian() {
        let params = Pm3Parameters::standard().unwrap();
        let periodic = PeriodicOptions::default();
        let molecule = celled(WATER, 18.0);
        let plain =
            periodic_hessian(&molecule, &params, &Pm3Options::default(), &periodic).unwrap();
        let corrected = periodic_hessian(
            &molecule,
            &params,
            &Pm3Options {
                variant: Variant::Pm3D3H4X,
                ..Pm3Options::default()
            },
            &periodic,
        )
        .unwrap();
        let largest = (0..plain.rows)
            .flat_map(|i| (0..plain.rows).map(move |j| (i, j)))
            .fold(0.0_f64, |m, (i, j)| {
                m.max((plain[(i, j)] - corrected[(i, j)]).abs())
            });
        assert!(largest > 1.0e-6, "the correction Hessian did nothing");
    }
}
