// SPDX-License-Identifier: GPL-3.0-or-later

//! Born effective charges: the dipole a cell acquires per unit displacement of one atom.
//!
//! ```text
//! Z*_{a,αβ} = ∂(Ω P_α) / ∂u_{a,β}
//! ```
//!
//! # What carries the long-range part of a lattice vibration
//!
//! Without them a polar crystal's longitudinal and transverse optical branches stay degenerate
//! as `q → 0`, which is wrong by an amount that is not small. They are the input the
//! non-analytic term of the dynamical matrix is built from — see [`crate::pbc::phonon`] — and
//! they are also the periodic analogue of the molecular atomic polar tensor that
//! [`crate::ir`] contracts for infrared intensities.
//!
//! # The expression, in this model
//!
//! The cell dipole is `Σ_b Q_b R_b + Σ_b μ_b`, with `Q_b` the net atomic charge and `μ_b` the
//! on-site `sp` hybridization moment that the same dipole operator [`crate::dipole`] uses.
//! Differentiating:
//!
//! ```text
//! Z*_{a,αβ} = Q_a δ_αβ  +  Σ_b R_{b,α} ∂Q_b/∂u_{a,β}  +  Σ_b ∂μ_{b,α}/∂u_{a,β}
//! ```
//!
//! The first term is the atom's own charge moving with it. The other two are the electrons
//! rearranging, and both come from the coupled-perturbed response
//! ([`crate::pbc::dfpt::phonon_response`]) — the same response the Γ-point force constants use.
//!
//! # Why this is well defined under periodic boundary conditions when the polarization is not
//!
//! `Σ_b R_b Q_b` depends on where the cell origin is put, and the polarization of a periodic
//! solid is famously defined only modulo a quantum. The **derivative** is not: charge is
//! conserved, so `Σ_b ∂Q_b/∂u_a = 0`, and the origin dependence cancels term by term. That is
//! what makes this computable here while an absolute polarization is not — and it is *measured*
//! rather than argued, by `tests/pbc_born_charges.rs`, which recomputes with the cell shifted.
//!
//! # The check that matters
//!
//! `Σ_a Z*_a = 0`, the acoustic sum rule for Born charges: translating the whole crystal
//! produces no dipole. It follows from charge conservation and nothing else, so a violation is a
//! bug in the response rather than a physical effect.
//!
//! Note what that check cannot see. An implementation returning only the `Q_a δ_αβ` term — the
//! rigid-ion charges, with no electronic response at all — satisfies it exactly, because the net
//! charges already sum to zero in a neutral cell. The test suite therefore also pins a
//! homonuclear chain, whose true `Z*` is near zero only *because* the response cancels the
//! point-charge term.

// A note on `clippy::needless_range_loop`, allowed below.
//
// The loop variables here are Cartesian directions (`alpha`, `beta`, `axis`), atom indices, or
// the rows and columns of a matrix being eliminated. The index *is* the meaning: `for alpha in
// 0..3` says which direction, where `for (alpha, row) in out.iter_mut().enumerate()` says it
// less clearly and no more safely. Several of these loops also index two different tensors by
// the same direction, which no single iterator expresses.
#![allow(clippy::needless_range_loop)]

use crate::error::Result;
use crate::params::Pm3Parameters;
use crate::pbc::gamma::PeriodicOptions;
use crate::scf::Pm3Options;
use crate::system::Molecule;

/// Born effective charges, one `3 × 3` tensor per atom, in units of the elementary charge.
///
/// Row index is the polarization direction `α`, column index the displacement direction `β`:
/// `tensor[α][β] = ∂(Ω P_α)/∂u_β`.
///
/// # The cost, and the route not taken
///
/// `3N` coupled-perturbed solves, one per degree of freedom, each contracted against the dipole
/// operator. `Z*` is a mixed second derivative, `∂²E/∂𝓔_α ∂u_{aβ}`, so the interchange theorem
/// says the **field** could carry the self-consistency instead — three solves whatever `N` is.
///
/// That was implemented and withdrawn. It gave the wrong tensor, and fixing it was not worth
/// pursuing at the time because the measurement that mattered pointed elsewhere: at the time it
/// was tried, **85% of a Born-charge run was one long-range lattice sum being recomputed `6N`
/// times**, and replacing `3N` solves with three changed the wall clock by nothing at all. With
/// that fixed (see [`crate::pbc::dfpt::LongRangeKernels`]) the solves are what remains, and the
/// interchange route is worth about another 3x — for whoever wants it, with the warning that its
/// contraction is not as obvious as the theorem makes it sound.
pub fn born_charges(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
) -> Result<Vec<[[f64; 3]; 3]>> {
    let response = crate::pbc::dfpt::phonon_response(molecule, params, options, periodic)?;
    let basis = &response.basis;
    let nat = molecule.atoms.len();

    // The ground state's net atomic charges, `Q_a = Z_a^core − Σ_{μ∈a} P_μμ`.
    let mut charges = Vec::with_capacity(nat);
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let offset = basis.atom_offset[ia];
        let norb = basis.atom_norb[ia];
        let population: f64 = (0..norb)
            .map(|mu| response.scf.density[(offset + mu, offset + mu)])
            .sum();
        charges.push(params.element(atom.z)?.core_charge - population);
    }

    // Diagnostic switch, read once rather than per orbital per degree of freedom: `env::var`
    // allocates and takes a lock, and this sits three loops deep.
    //
    // Dropping the `dd` term makes this route compute what the Berry phase computes, since the
    // phase places every orbital at its atom's centre and so carries no intra-atomic moment. That
    // is how `tests/pbc_berry.rs` established that the two formalisms differ by this term and by
    // nothing else: with it removed they agree to `1.9e-4 e`, and with it present they differ by
    // `0.063` along the bond and `0.147` across it. Kept because that is the only way to check the
    // claim again after a change, and because the alternative was to argue it.
    let with_dd = std::env::var("PM3_BORN_NO_DD").is_err();

    let mut out = vec![[[0.0_f64; 3]; 3]; nat];
    for a in 0..nat {
        for beta in 0..3 {
            let delta = &response.delta[3 * a + beta];
            for alpha in 0..3 {
                // 1) The atom's own charge, moving with it.
                let mut total = if alpha == beta { charges[a] } else { 0.0 };

                for (b, atom) in molecule.atoms.iter().enumerate() {
                    let offset = basis.atom_offset[b];
                    let norb = basis.atom_norb[b];
                    if norb == 0 {
                        continue;
                    }
                    // 2) Charge transfer. `∂Q_b = −∂p_b`: a population going up is a charge
                    // going down, and the minus sign is the whole content of the term.
                    let population: f64 = (0..norb)
                        .map(|mu| delta[(offset + mu, offset + mu)].re)
                        .sum();
                    total += -population * atom.position.to_array()[alpha];

                    // 3) The on-site `sp` hybridization moment moving. The dipole operator puts
                    // `dd` on both `(s, p_α)` and `(p_α, s)`, and the cell dipole carries it as
                    // `−Tr[P M]`, so the response contributes `−2·dd·ΔP_{s,p_α}`.
                    let element = params.element(atom.z)?;
                    if element.has_p() && with_dd {
                        let p = offset + alpha + 1;
                        total += -2.0 * element.dd * delta[(offset, p)].re;
                    }
                }
                out[a][alpha][beta] = total;
            }
        }
    }
    Ok(out)
}

/// The largest `|Σ_a Z*_{a,αβ}|` over the nine components — the acoustic sum rule's residual.
///
/// Reported rather than enforced. Translating the whole crystal produces no dipole, so this is
/// zero for an exact response and a measure of the response's own convergence otherwise. A
/// caller who wants the rule imposed can subtract `Σ_a Z*_a / N` from every tensor, but doing it
/// silently would hide exactly the number that says whether the response converged.
pub fn born_charge_sum_rule_residual(born: &[[[f64; 3]; 3]]) -> f64 {
    let mut worst = 0.0_f64;
    for alpha in 0..3 {
        for beta in 0..3 {
            let total: f64 = born.iter().map(|z| z[alpha][beta]).sum();
            worst = worst.max(total.abs());
        }
    }
    worst
}

/// Subtract the mean violation from every tensor, so that `Σ_a Z*_a = 0` exactly.
///
/// The counterpart of [`crate::pbc::hessian::enforce_acoustic_sum_rule`] for the charges, and
/// used for the same reason: the non-analytic term of a dynamical matrix is built from `Z*`, and
/// a residual there leaks into the acoustic branch at small `q` where the two nearly cancel.
/// Check [`born_charge_sum_rule_residual`] *before* calling this — if it is not already small,
/// something is wrong with the response and flattening it hides that.
pub fn enforce_born_sum_rule(born: &mut [[[f64; 3]; 3]]) {
    let nat = born.len();
    if nat == 0 {
        return;
    }
    for alpha in 0..3 {
        for beta in 0..3 {
            let share: f64 = born.iter().map(|z| z[alpha][beta]).sum::<f64>() / nat as f64;
            for z in born.iter_mut() {
                z[alpha][beta] -= share;
            }
        }
    }
}
