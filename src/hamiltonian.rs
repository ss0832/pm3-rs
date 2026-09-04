// SPDX-License-Identifier: GPL-3.0-or-later

//! Core (one-electron) Hamiltonian assembly.
//!
//! `H_core` holds the diagonal atomic energies `U_ss/U_pp/U_dd`, the
//! electron–core attraction to every other atom (from the NDDO integrals), and
//! the inter-atomic resonance `H_μν = ½(β_μ + β_ν) S_μν`. The per-pair
//! two-electron integrals are returned alongside for reuse in the Fock build.
//! Pairs where either atom carries d orbitals use the spd multipole integrals
//! ([`crate::integrals_d`]) and the spd overlap; all-sp pairs use the validated
//! s/p kernels ([`crate::integrals`], [`crate::overlap`]).

use crate::basis::Basis;
use crate::error::{Pm3Error, Result};
use crate::integrals::{pair_two_electron, PackedTwoElec, PairTwoElec};
use crate::integrals_d::pair_two_electron_spd;
use crate::linalg::Matrix;
use crate::overlap::{diatom_overlap, diatom_overlap_spd};
use crate::params::Pm3Parameters;
use crate::system::Molecule;

/// Rotated two-electron integrals for one atom pair, tagged with the ordered atom indices.
///
/// Only the packed two-electron table is retained ([`PackedTwoElec`]); the
/// electron–core attraction blocks are folded into `h_core` during
/// [`build_core`] and then dropped.
pub struct PairIntegral {
    pub a: usize,
    pub b: usize,
    pub te: PackedTwoElec,
}

pub struct CoreHamiltonian {
    pub h_core: Matrix,
    pub pairs: Vec<PairIntegral>,
}

const MIB: usize = 1024 * 1024;

/// Default soft ceiling (MiB) for the resident `O(N²)` two-electron pair cache,
/// overridable with the `PM3_MAX_PAIR_CACHE_MB` environment variable
/// (`0` disables the check). [`crate::scf::Pm3Options::integral_memory_mb`] is
/// the programmatic knob.
pub fn default_pair_cache_limit_mb() -> usize {
    std::env::var("PM3_MAX_PAIR_CACHE_MB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096)
}

/// Exact size of the pair cache [`build_core`] is about to allocate, in bytes.
///
/// The cache holds one packed table per atom pair, so its size is a closed-form
/// function of the per-atom packed dimensions `p_i = n_i(n_i+1)/2`:
/// `Σ_{i<j} p_i p_j = ((Σ p_i)² − Σ p_i²) / 2` elements. Computing it up front
/// turns "the allocator aborts the process" into an actionable error.
pub fn pair_cache_bytes(basis: &Basis) -> usize {
    let packed: Vec<usize> = basis
        .atom_norb
        .iter()
        .map(|&n| (n * (n + 1) / 2).max(1))
        .collect();
    let total: usize = packed.iter().sum();
    let squares: usize = packed.iter().map(|p| p * p).sum();
    let elements = (total.saturating_mul(total).saturating_sub(squares)) / 2;
    let atoms = basis.atom_norb.len();
    let n_pairs = atoms.saturating_mul(atoms.saturating_sub(1)) / 2;
    elements
        .saturating_mul(std::mem::size_of::<f64>())
        .saturating_add(n_pairs.saturating_mul(std::mem::size_of::<PairIntegral>()))
}

/// MOPAC capped-bond (`Cb`, atom code 102) energy correction.
///
/// The very large Cb resonance integral is retained in the SCF Fock matrix so
/// it terminates the dangling bond, but MOPAC `capcor` removes its contribution
/// from the reported electronic energy after the density has converged.
/// [`capped_bond_energy_correction`] from a sparse density and core.
///
/// Every term is a product `P_{μν} H_{μν}`, so wherever the divide-and-conquer density is
/// structurally zero the term is zero whatever the core holds there. That is what lets both
/// sides be read on the pattern and still give the dense answer exactly.
pub(crate) fn capped_bond_energy_correction_sparse(
    molecule: &Molecule,
    basis: &Basis,
    pattern: &crate::dc::pattern::DensityPattern,
    density: &crate::dc::pattern::SparseMatrix,
    h_core: &crate::dc::pattern::SparseMatrix,
) -> f64 {
    let mut sum = 0.0;
    for i in 0..molecule.atoms.len() {
        let il = basis.atom_offset[i];
        let iu = il + basis.atom_norb[i];
        if molecule.atoms[i].z == 102 {
            if iu == 0 {
                continue;
            }
            let last = iu - 1;
            for k in 0..last {
                sum += density.get(pattern, last, k) * h_core.get(pattern, last, k);
            }
        } else {
            for j in 0..=i {
                if molecule.atoms[j].z != 102 {
                    continue;
                }
                let capped_s = basis.atom_offset[j];
                for k in il..iu {
                    sum += density.get(pattern, k, capped_s) * h_core.get(pattern, k, capped_s);
                }
            }
        }
    }
    -2.0 * sum
}

pub(crate) fn capped_bond_energy_correction(
    molecule: &Molecule,
    basis: &Basis,
    density: &Matrix,
    h_core: &Matrix,
) -> f64 {
    // This intentionally follows MOPAC `capcor.F90` index-for-index.  Its
    // historical Cb convention selects the last AO row when Cb is the current
    // atom, but the first Cb AO column when Cb precedes the current atom.  The
    // resulting atom-order dependence is part of MOPAC's special-atom oracle.
    let mut sum = 0.0;
    for i in 0..molecule.atoms.len() {
        let il = basis.atom_offset[i];
        let iu = il + basis.atom_norb[i];
        if molecule.atoms[i].z == 102 {
            if iu == 0 {
                continue;
            }
            let last = iu - 1;
            for k in 0..last {
                sum += density[(last, k)] * h_core[(last, k)];
            }
        } else {
            for j in 0..=i {
                if molecule.atoms[j].z != 102 {
                    continue;
                }
                let capped_s = basis.atom_offset[j];
                for k in il..iu {
                    sum += density[(k, capped_s)] * h_core[(k, capped_s)];
                }
            }
        }
    }
    -2.0 * sum
}

/// Return the resonance β for orbital index `orb` (0 = s, 1..4 = p, 4..9 = d).
#[inline]
fn beta_of(elem: &crate::params::Pm3Element, orb: u8) -> f64 {
    match orb {
        0 => elem.beta_s,
        1..=3 => elem.beta_p,
        _ => elem.beta_d,
    }
}

/// Embed a 4×4 sp overlap into a 9×9 block.
fn embed4(s: [[f64; 4]; 4]) -> [[f64; 9]; 9] {
    let mut out = [[0.0; 9]; 9];
    for (i, row) in s.iter().enumerate() {
        out[i][..4].copy_from_slice(row);
    }
    out
}

/// [`build_core_limited`] with the process-wide default cache ceiling and **no external field**.
pub fn build_core(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm3Parameters,
) -> Result<CoreHamiltonian> {
    build_core_limited(molecule, basis, params, default_pair_cache_limit_mb(), None)
}

/// Assemble `H_core` and the resident two-electron pair cache, refusing to
/// start if the cache would exceed `limit_mb` MiB (`0` = unlimited).
///
/// `field` is a uniform external electric field (eV per Bohr per elementary charge), which
/// enters here rather than being added by the caller afterwards. That is deliberate: the field
/// is a one-electron operator, so it belongs in `H_core`, and every consumer — the SCF, the
/// fixed-density energy, the Hessian's derivative Fock — then sees it without having to
/// remember to. Making it an argument rather than a later mutation means the compiler asks each
/// call site the question instead of letting one of them silently answer "no field".
pub fn build_core_limited(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm3Parameters,
    limit_mb: usize,
    field: Option<crate::math::Vec3>,
) -> Result<CoreHamiltonian> {
    let nao = basis.nao;
    // Pre-flight guard: the pair cache is the single largest allocation of a
    // large PM3 run and grows as O(N²). Report the requirement instead of
    // letting the allocator abort the process mid-build.
    if limit_mb > 0 {
        let required = pair_cache_bytes(basis);
        if required > limit_mb.saturating_mul(MIB) {
            return Err(Pm3Error::ResourceLimit {
                operation: "PM3 two-electron pair cache",
                required_mb: required.div_ceil(MIB),
                limit_mb,
            });
        }
    }
    let mut h = Matrix::zeros(nao, nao);

    // Diagonal U_ss / U_pp / U_dd.
    for (mu, ao) in basis.aos.iter().enumerate() {
        let elem = params.element(ao.z)?;
        h[(mu, mu)] = match ao.orb {
            0 => elem.u_ss,
            1..=3 => elem.u_pp,
            _ => elem.u_dd,
        };
    }

    // The external field, as `+M·f` — the same dipole operator the reported dipole is an
    // expectation value of. It is added before the two-center terms because those use `=` on the
    // resonance block and would overwrite it; the one-center loop above uses `=` for the same
    // reason and this has to follow it.
    if let Some(f) = field {
        let (operator, _) = crate::dipole::field_terms(molecule, params, basis, f)?;
        for (slot, value) in h.as_mut_slice().iter_mut().zip(operator.as_slice()) {
            *slot += value;
        }
    }

    use rayon::prelude::*;

    let nat = molecule.atoms.len();
    let pair_indices: Vec<(usize, usize)> = (0..nat)
        .flat_map(|u| ((u + 1)..nat).map(move |v| (u, v)))
        .collect();

    // Per-pair two-electron integrals + overlap, computed in bounded parallel
    // batches. `PairTwoElec` remains cached for every pair because every SCF
    // iteration reuses it, but transient 9×9 overlap blocks no longer remain
    // resident for all O(N²) pairs at once.
    let batch_size = rayon::current_num_threads().saturating_mul(4).max(64);
    let mut pairs = Vec::with_capacity(pair_indices.len());
    for batch in pair_indices.chunks(batch_size) {
        let computed: Vec<(usize, usize, PairTwoElec, [[f64; 9]; 9])> = batch
            .par_iter()
            .map(
                |&(u, v)| -> Result<(usize, usize, PairTwoElec, [[f64; 9]; 9])> {
                    let eu = params.element(molecule.atoms[u].z)?;
                    let ev = params.element(molecule.atoms[v].z)?;
                    if eu.n_orb == 0 || ev.n_orb == 0 {
                        let (a, b) = if eu.n_orb > 0 { (u, v) } else { (v, u) };
                        let ea = params.element(molecule.atoms[a].z)?;
                        let eb = params.element(molecule.atoms[b].z)?;
                        let displacement = molecule.atoms[b].position - molecule.atoms[a].position;
                        let te = crate::integrals::pair_with_point_core_g::<f64>(
                            ea,
                            eb,
                            [displacement.x, displacement.y, displacement.z],
                        );
                        Ok((a, b, te, [[0.0; 9]; 9]))
                    } else if eu.has_d() || ev.has_d() {
                        // spd path (also handles zero-orbital sparkle partners); keep
                        // natural (u, v) ordering.
                        let pos_u = molecule.atoms[u].position;
                        let pos_v = molecule.atoms[v].position;
                        let d = pos_v - pos_u;
                        let te = pair_two_electron_spd::<f64>(eu, ev, [d.x, d.y, d.z]);
                        let s_block = diatom_overlap_spd::<f64>(eu, ev, [d.x, d.y, d.z]);
                        Ok((u, v, te, s_block))
                    } else {
                        // sp path: heavy atom first when the partner is H (validated convention).
                        let (a, b) = if eu.has_p() || !ev.has_p() {
                            (u, v)
                        } else {
                            (v, u)
                        };
                        let (ea, eb) = (
                            params.element(molecule.atoms[a].z)?,
                            params.element(molecule.atoms[b].z)?,
                        );
                        let pos_a = molecule.atoms[a].position;
                        let pos_b = molecule.atoms[b].position;
                        let d = pos_b - pos_a;
                        let r = d.norm();
                        let xij = d / r;
                        let te = pair_two_electron(ea, eb, xij, r);
                        let s_block = embed4(diatom_overlap(ea, pos_a, eb, pos_b)?);
                        Ok((a, b, te, s_block))
                    }
                },
            )
            .collect::<Result<Vec<_>>>()?;

        // Assemble each batch before computing the next one.
        for (a, b, te, s_block) in computed {
            let (ea, eb) = (
                params.element(molecule.atoms[a].z)?,
                params.element(molecule.atoms[b].z)?,
            );
            let off_a = basis.atom_offset[a];
            let off_b = basis.atom_offset[b];
            let na = basis.atom_norb[a];
            let nb = basis.atom_norb[b];

            // Electron–core attraction: e1b onto atom a's block, e2a onto atom b's block.
            for i in 0..na {
                for j in 0..na {
                    h[(off_a + i, off_a + j)] += te.e1b[i][j];
                }
            }
            for i in 0..nb {
                for j in 0..nb {
                    h[(off_b + i, off_b + j)] += te.e2a[i][j];
                }
            }

            // Resonance β·S (inter-atomic, symmetric).
            for i in 0..na {
                let bi = beta_of(ea, basis.aos[off_a + i].orb);
                for j in 0..nb {
                    let bj = beta_of(eb, basis.aos[off_b + j].orb);
                    let value = 0.5 * (bi + bj) * s_block[i][j];
                    h[(off_a + i, off_b + j)] = value;
                    h[(off_b + j, off_a + i)] = value;
                }
            }

            // `te.e1b`/`te.e2a` are fully consumed above; only the packed
            // two-electron table stays resident for the SCF/CPHF Fock builds.
            pairs.push(PairIntegral {
                a,
                b,
                te: te.into(),
            });
        }
    }

    Ok(CoreHamiltonian { h_core: h, pairs })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heh_plus_core_hamiltonian_matches_mopac() {
        let molecule = Molecule::from_xyz_str("2\nHeH+\nHe 0 0 0\nH 0.9 0 0\n", 1.0).unwrap();
        let params = Pm3Parameters::standard().unwrap();
        let basis = Basis::build(&molecule, &params).unwrap();
        let h = build_core(&molecule, &basis, &params).unwrap().h_core;
        let reference = [-44.232953, 9.9998433, 9.9998433, 9.9998433, -30.552992];
        for (i, expected) in reference.iter().enumerate() {
            assert!(
                (h[(i, i)] - expected).abs() < 2.0e-5,
                "i={i}: {}",
                h[(i, i)]
            );
        }
        assert!(
            (h[(4, 0)] - (-5.412820)).abs() < 2.0e-5,
            "s-s={}",
            h[(4, 0)]
        );
        assert!(
            (h[(4, 1)] - (-0.674201)).abs() < 2.0e-5,
            "s-px={}",
            h[(4, 1)]
        );

        let packed_density = [
            1.8084673365316,
            0.0052498481566,
            0.0000152399245,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.5885176334281,
            0.0017084235643,
            0.0,
            0.0,
            0.1915174235439,
        ];
        let mut density = Matrix::zeros(5, 5);
        let mut k = 0;
        for i in 0..5 {
            for j in 0..=i {
                density[(i, j)] = packed_density[k];
                density[(j, i)] = packed_density[k];
                k += 1;
            }
        }
        let core = build_core(&molecule, &basis, &params).unwrap();
        let f = crate::fock::build_fock(&molecule, &basis, &params, &core, &density).unwrap();
        let packed_fock = [
            -35.2717267527736,
            0.0678397814106,
            31.1231311689591,
            0.0,
            0.0,
            31.1051176606997,
            0.0,
            0.0,
            0.0,
            31.1051176606997,
            -7.9850063322581,
            -0.8239308128258,
            0.0,
            0.0,
            -13.3254020626222,
        ];
        k = 0;
        for i in 0..5 {
            for j in 0..=i {
                assert!(
                    (f[(i, j)] - packed_fock[k]).abs() < 2.0e-5,
                    "({i},{j}) rust={} mopac={}",
                    f[(i, j)],
                    packed_fock[k]
                );
                k += 1;
            }
        }
    }
}
