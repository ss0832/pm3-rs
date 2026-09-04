// SPDX-License-Identifier: GPL-3.0-or-later

//! The two pieces the periodic CPHF needs: the response operator `G[∂P]` and the skeleton
//! derivative of the Fock matrix.
//!
//! # The response operator comes free
//!
//! `F = H + G[P]`, and `G` is **linear** in the density — including its long-range half, because
//! the Ewald sum's site *potentials* are linear in the site charges even though its energy is
//! quadratic. So the operator the CPHF iteration needs is just
//!
//! ```text
//! G[∂P] = F(∂P) − H
//! ```
//!
//! with `F(·)` the ordinary periodic Fock build. Nothing new has to be derived, and — more
//! usefully — the CPHF kernel cannot drift out of agreement with the SCF's own Fock, because it
//! *is* the SCF's own Fock.
//!
//! # The Fock derivative cannot
//!
//! `∂F/∂R` at fixed density has no such shortcut: it is the derivative of the integrals
//! themselves. It is computed here by the same forward-mode AD the gradient uses, restricted to
//! the image pairs that touch the moving atom, and assembled with the same counting conventions
//! [`crate::pbc::gamma`] states once — the halving of the ordered-visit terms, the screened
//! Coulomb against the full exchange, the exchange cutoff.

use crate::basis::Basis;
use crate::dual::{Dual, Scalar};
use crate::error::{Pm3Error, Result};
use crate::integrals::{pack, pair_two_electron_g};
use crate::linalg::Matrix;
use crate::math::Vec3;
use crate::neighbor::NeighborList;
use crate::params::Pm3Parameters;
use crate::pbc::ewald::ewald;
use crate::pbc::gamma::{
    add_site_potential, build_periodic_fock, build_setup, write_electron_charges, PeriodicOptions,
    Setup,
};
use crate::pbc::multipole::AtomSites;
use crate::pbc::screen::point_pair_g;
use crate::system::Molecule;

/// `G[∂P]`, the periodic two-electron operator applied to a density perturbation.
pub struct PeriodicKernel<'a> {
    molecule: &'a Molecule,
    params: &'a Pm3Parameters,
    setup: Setup,
}

impl<'a> PeriodicKernel<'a> {
    pub fn build(
        molecule: &'a Molecule,
        params: &'a Pm3Parameters,
        periodic: &PeriodicOptions,
    ) -> Result<Self> {
        Ok(Self {
            molecule,
            params,
            setup: build_setup(molecule, params, periodic)?,
        })
    }

    /// Apply the operator to a **total** density perturbation, closed shell.
    pub fn apply(&self, delta_p: &Matrix) -> Result<Matrix> {
        let cell = self.molecule.cell.expect("periodic");
        // Closed shell: each spin carries half the perturbation, which is what makes the exchange
        // come out as `−½` of the Coulomb integral.
        let mut half = delta_p.clone();
        for value in half.as_mut_slice() {
            *value *= 0.5;
        }
        let mut out = build_periodic_fock(self.molecule, self.params, &self.setup, delta_p, &half)?;
        for (value, h) in out
            .as_mut_slice()
            .iter_mut()
            .zip(self.setup.h_core.as_slice())
        {
            *value -= h;
        }

        // The long-range half of the response: the perturbed multipole charges' own field.
        let mut sites = self.setup.sites.clone();
        write_electron_charges(&self.setup, delta_p, &mut sites);
        let field = crate::pbc::ewald::ewald_potentials_cached(
            &cell,
            &sites,
            &self.setup.ewald_params,
            &self.setup.ewald_context,
        )?;
        add_site_potential(&self.setup, &mut out, &field);
        Ok(out)
    }
}

/// The full periodic Fock matrix at a fixed density.
///
/// Exists so [`fock_derivative`] has something to be differenced against in a test. Building it
/// from the same pieces the SCF uses is the point — a reference assembled differently would only
/// test that two spellings of the same mistake agree.
pub fn fock_at(
    molecule: &Molecule,
    params: &Pm3Parameters,
    periodic: &PeriodicOptions,
    density: &Matrix,
) -> Result<Matrix> {
    let cell = molecule.cell.expect("periodic");
    let setup = build_setup(molecule, params, periodic)?;
    let mut half = density.clone();
    for value in half.as_mut_slice() {
        *value *= 0.5;
    }
    let mut out = build_periodic_fock(molecule, params, &setup, density, &half)?;
    let mut sites = setup.sites.clone();
    write_electron_charges(&setup, density, &mut sites);
    let field = crate::pbc::ewald::ewald_potentials_cached(
        &cell,
        &sites,
        &setup.ewald_params,
        &setup.ewald_context,
    )?;
    add_site_potential(&setup, &mut out, &field);
    Ok(out)
}

/// The short-range half of `∂F/∂R_{atom,axis}` at fixed density: every term that comes from a
/// neighbour-list pair, without the lattice sum's own contribution.
///
/// Only pairs touching `atom` contribute. A self-image pair contributes nothing at all: its
/// displacement is a lattice vector, which does not move when the atom does.
pub(crate) fn fock_derivative_pairs(
    molecule: &Molecule,
    params: &Pm3Parameters,
    periodic: &PeriodicOptions,
    density: &Matrix,
    basis: &Basis,
    atom: usize,
    axis: usize,
) -> Result<Matrix> {
    use crate::pbc::gradient::resonance_beta;

    let cell = molecule.cell.expect("periodic");
    let nat = molecule.atoms.len();
    let mut out = Matrix::zeros(basis.nao, basis.nao);

    let mut atom_sites = Vec::with_capacity(nat);
    for a in &molecule.atoms {
        let elem = params.element(a.z)?;
        atom_sites.push(AtomSites::build(elem).ok_or(Pm3Error::UnsupportedDOrbitals(a.z))?);
    }
    let cutoff = periodic.short_range_cutoff.max(periodic.switch.off);
    let positions: Vec<Vec3> = molecule.atoms.iter().map(|a| a.position).collect();
    let list = NeighborList::build_from_positions(&positions, Some(&cell), cutoff);

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

        // `d = R_second − R_first + T`, so moving `first` moves `d` backwards and moving `second`
        // moves it forwards. Both at once — a self-image — leaves it alone.
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

        let seeded = [
            Dual::var(dvec.x, 0),
            Dual::var(dvec.y, 1),
            Dual::var(dvec.z, 2),
        ];
        let te = pair_two_electron_g::<Dual>(first, second, seeded);
        let point = point_pair_g::<Dual>(
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
        let derivative = |value: Dual| chain * value.d[axis];

        // One-electron: the screened electron–core attraction, halved for the double visit.
        for mu in 0..na {
            for nu in 0..na {
                let term = (te.e1b[mu][nu] - point.e1b[mu * na + nu]) * switch * 0.5;
                out[(off_first + mu, off_first + nu)] += derivative(term);
            }
        }
        for la in 0..nb {
            for si in 0..nb {
                let term = (te.e2a[la][si] - point.e2a[la * nb + si]) * switch * 0.5;
                out[(off_second + la, off_second + si)] += derivative(term);
            }
        }

        // The inter-atomic terms go to the block the *visit* names, not the block the element
        // ordering names. The integrals are always evaluated heavy-atom-first, but the neighbour
        // list visits `(a, b, T)` and `(b, a, −T)` separately and each fills its own block, so the
        // destination has to follow the visit or one block gets everything and its transpose gets
        // nothing. [`crate::pbc::gamma::build_setup`] makes the same distinction.
        let (off_a, off_b) = (basis.atom_offset[a], basis.atom_offset[b]);
        // `(mu on first, la on second)` maps to `(row on a, column on b)` — the same pair of
        // orbitals, addressed the way this visit names them.
        let inter_index = |mu: usize, la: usize| -> (usize, usize) {
            if heavy_first {
                (off_a + mu, off_b + la)
            } else {
                (off_a + la, off_b + mu)
            }
        };

        let inside_short_range = pair.r <= periodic.short_range_cutoff;
        if inside_short_range {
            let overlap = crate::overlap::diatom_overlap_dual(first, Vec3::zero(), second, dvec)?;
            #[allow(clippy::needless_range_loop)]
            for mu in 0..na.min(4) {
                let bi = resonance_beta(first, basis.aos[off_first + mu].orb);
                for la in 0..nb.min(4) {
                    let bj = resonance_beta(second, basis.aos[off_second + la].orb);
                    // Not halved: each visit fills its own block once.
                    out[inter_index(mu, la)] += derivative(overlap[mu][la] * (0.5 * (bi + bj)));
                }
            }
        }

        // Two-electron. Coulomb writes into both atoms' on-site blocks from the other's density,
        // so — like the electron–core terms above — it is halved for the double visit. Exchange
        // writes into one inter-atomic block and is not.
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
                out[(off_first + mu, off_first + nu)] += accumulator;
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
                out[(off_second + la, off_second + si)] += accumulator;
            }
        }
        if inside_short_range {
            for mu in 0..na {
                for la in 0..nb {
                    let mut accumulator = 0.0;
                    for nu in 0..na {
                        for si in 0..nb {
                            let index = pack(mu, nu) * npack_j + pack(la, si);
                            accumulator += 0.5
                                * density[(off_first + nu, off_second + si)]
                                * derivative(te.w[index]);
                        }
                    }
                    out[inter_index(mu, la)] -= accumulator;
                }
            }
        }
    }

    Ok(out)
}

/// The same derivative with the lattice sum's own contribution added: the whole field moved,
/// cores and electrons together.
///
/// Split from the pair loop above so the two can be tested apart. The phased twin in
/// [`crate::pbc::dfpt`] reproduces the pair loop exactly at `q = 0`, and comparing against a
/// result that already carried the field term would have meant comparing two things that differ
/// by a term neither test could isolate.
pub fn fock_derivative(
    molecule: &Molecule,
    params: &Pm3Parameters,
    periodic: &PeriodicOptions,
    density: &Matrix,
    basis: &Basis,
    atom: usize,
    axis: usize,
) -> Result<Matrix> {
    let mut out = fock_derivative_pairs(molecule, params, periodic, density, basis, atom, axis)?;
    let nat = molecule.atoms.len();
    let mut atom_sites = Vec::with_capacity(nat);
    for a in &molecule.atoms {
        let elem = params.element(a.z)?;
        atom_sites.push(AtomSites::build(elem).ok_or(Pm3Error::UnsupportedDOrbitals(a.z))?);
    }
    add_field_derivative(
        molecule,
        params,
        periodic,
        basis,
        &atom_sites,
        density,
        atom,
        axis,
        &mut out,
    )?;
    Ok(out)
}

/// `∂/∂R` of the potential the lattice sum puts on every multipole site.
///
/// Both halves of the field belong here — the cores' (which reaches the Fock through `H_core`) and
/// the electrons' (which reaches it directly). Leaving the electron half out is the natural
/// mistake, because the *charges* are frozen at the converged density and it is easy to conclude
/// the term is therefore frozen too. It is not: the sites move with their atoms, and the field a
/// fixed charge distribution produces at a moving point still changes.
///
/// Done by central differences of the analytic Ewald potential, which is the one place in this
/// Hessian that is not closed-form. The potential is a smooth, cheap function of position with no
/// cancellation anywhere near it, so a central difference at `1e-5` Bohr is good to roughly
/// `1e-10` — orders below what the finite-difference test of the whole Hessian resolves. Doing it
/// analytically would mean differentiating the Ewald sum with respect to the *field point* rather
/// than the source: a second full derivation for a term this well behaved.
#[allow(clippy::too_many_arguments)]
fn add_field_derivative(
    molecule: &Molecule,
    params: &Pm3Parameters,
    periodic: &PeriodicOptions,
    basis: &Basis,
    atom_sites: &[AtomSites],
    density: &Matrix,
    atom: usize,
    axis: usize,
    out: &mut Matrix,
) -> Result<()> {
    const STEP: f64 = 1.0e-5;
    // The charges are those of the converged density and stay put; only the positions move.
    let mut charges: Vec<Vec<f64>> = Vec::with_capacity(molecule.atoms.len());
    for (ia, a) in molecule.atoms.iter().enumerate() {
        let elem = params.element(a.z)?;
        let n = basis.atom_norb[ia];
        let off = basis.atom_offset[ia];
        let mut block = vec![0.0; n * n];
        for mu in 0..n {
            for nu in 0..n {
                block[mu * n + nu] = density[(off + mu, off + nu)];
            }
        }
        charges.push(atom_sites[ia].charges(&block, elem.core_charge));
    }

    let field_at = |shift: f64| -> Result<Matrix> {
        let mut moved = molecule.clone();
        match axis {
            0 => moved.atoms[atom].position.x += shift,
            1 => moved.atoms[atom].position.y += shift,
            _ => moved.atoms[atom].position.z += shift,
        }
        let cell = moved.cell.expect("periodic");
        let mut sites = Vec::new();
        let mut site_offset = Vec::with_capacity(moved.atoms.len());
        for (ia, a) in moved.atoms.iter().enumerate() {
            site_offset.push(sites.len());
            for (index, offset) in atom_sites[ia].offsets.iter().enumerate() {
                sites.push(crate::pbc::ewald::ChargeSite {
                    position: a.position + *offset,
                    charge: charges[ia][index],
                    owner: ia,
                });
            }
        }
        let ewald_params = periodic.ewald.unwrap_or_else(|| {
            crate::pbc::ewald::EwaldParams::for_cell(&cell, crate::pbc::ewald::DEFAULT_ACCURACY)
        });
        let field = ewald(&cell, &sites, &ewald_params)?;
        let mut block = Matrix::zeros(basis.nao, basis.nao);
        for (ia, sites_of_atom) in atom_sites.iter().enumerate() {
            let n = basis.atom_norb[ia];
            let off = basis.atom_offset[ia];
            let potentials: Vec<f64> = (0..sites_of_atom.offsets.len())
                .map(|index| field.site_potential_ev[site_offset[ia] + index])
                .collect();
            for mu in 0..n {
                for nu in 0..n {
                    block[(off + mu, off + nu)] +=
                        sites_of_atom.fock_contribution(mu, nu, &potentials);
                }
            }
        }
        Ok(block)
    };

    let plus = field_at(STEP)?;
    let minus = field_at(-STEP)?;
    for (slot, (p, m)) in out
        .as_mut_slice()
        .iter_mut()
        .zip(plus.as_slice().iter().zip(minus.as_slice()))
    {
        *slot += (p - m) / (2.0 * STEP);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Cell;
    use crate::pbc::gamma::run_gamma;
    use crate::scf::Pm3Options;

    const WATER: &str = "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n";

    fn celled(edge: f64) -> Molecule {
        let mut molecule = Molecule::from_xyz_str(WATER, 0.0).unwrap();
        molecule.cell = Some(Cell::cubic(edge).unwrap());
        molecule
    }

    /// The Fock derivative against finite differences of the Fock matrix itself, at a frozen
    /// density. This is the CPHF right-hand side, so an error here scales the whole response.
    #[test]
    fn the_fock_derivative_matches_finite_differences() {
        let params = Pm3Parameters::standard().unwrap();
        let periodic = PeriodicOptions::default();
        let base = celled(18.0);
        let scf = run_gamma(&base, &params, &Pm3Options::default(), &periodic).unwrap();
        let basis = Basis::build(&base, &params).unwrap();
        let density = &scf.density;

        let step = 1.0e-5;
        for atom in 0..base.atoms.len() {
            for axis in 0..3 {
                let analytic =
                    fock_derivative(&base, &params, &periodic, density, &basis, atom, axis)
                        .unwrap();
                let moved = |shift: f64| {
                    let mut m = base.clone();
                    match axis {
                        0 => m.atoms[atom].position.x += shift,
                        1 => m.atoms[atom].position.y += shift,
                        _ => m.atoms[atom].position.z += shift,
                    }
                    fock_at(&m, &params, &periodic, density).unwrap()
                };
                let plus = moved(step);
                let minus = moved(-step);
                for i in 0..basis.nao {
                    for j in 0..basis.nao {
                        let numerical = (plus[(i, j)] - minus[(i, j)]) / (2.0 * step);
                        let exact = analytic[(i, j)];
                        assert!(
                            (numerical - exact).abs() < 2.0e-4,
                            "atom {atom} axis {axis} element ({i},{j}): \
                             analytic {exact} vs finite difference {numerical}"
                        );
                    }
                }
            }
        }
    }

    /// The response operator is linear in the perturbation, which is what lets the CPHF iteration
    /// treat it as a matrix rather than re-deriving it.
    #[test]
    fn the_response_operator_is_linear() {
        let params = Pm3Parameters::standard().unwrap();
        let periodic = PeriodicOptions::default();
        let molecule = celled(18.0);
        let kernel = PeriodicKernel::build(&molecule, &params, &periodic).unwrap();
        let scf = run_gamma(&molecule, &params, &Pm3Options::default(), &periodic).unwrap();

        let mut scaled = scf.density.clone();
        for value in scaled.as_mut_slice() {
            *value *= 0.25;
        }
        let single = kernel.apply(&scf.density).unwrap();
        let quarter = kernel.apply(&scaled).unwrap();
        for (a, b) in single.as_slice().iter().zip(quarter.as_slice()) {
            assert!(
                (a * 0.25 - b).abs() < 1.0e-10,
                "not linear: {} vs {}",
                a * 0.25,
                b
            );
        }
    }
}
