// SPDX-License-Identifier: GPL-3.0-or-later

//! NDDO Fock-matrix build, spin-resolved: `F^σ = H_core + J(P_tot) − K(P^σ)`.
//!
//! The Coulomb part `J` is built from the **total** density (both spins); the exchange
//! part `K` from the **same-spin** density. The RHF (closed-shell) Fock is the special case
//! `P^σ = ½ P_tot`, i.e. `F = H_core + J(P) − K(½P)`. The one-center block uses the exact
//! one-center two-electron integrals ([`oc_two_electron`]); the two-center block uses the
//! rotated integrals from [`crate::integrals`].

use crate::basis::Basis;
use crate::error::Result;
use crate::hamiltonian::{CoreHamiltonian, PairIntegral};
use crate::integrals::pack;
use crate::linalg::Matrix;
use crate::params::Pm3Parameters;
use crate::system::Molecule;
use rayon::prelude::*;

/// Pairs handled per parallel batch. The scratch buffer is `batch × stride`
/// doubles and is reused across batches, so peak scratch stays well under a
/// megabyte regardless of system size — the per-pair results are never all
/// resident at once, which is what keeps this `O(N²)` loop from becoming an
/// `O(N²)`-sized allocation.
const PAIR_BATCH: usize = 4096;

/// Layout of one pair's slot in the batch scratch buffer. The strides come from
/// the *actual* basis (`p_max = n_max(n_max+1)/2`), not a worst-case 9-AO bound:
/// for the s/p basis PM3 always uses, a slot is 36 doubles rather than 171, and
/// the per-iteration traffic through this buffer drops by the same factor.
#[derive(Clone, Copy)]
struct SlotLayout {
    p_max: usize,
    n_max: usize,
    stride: usize,
}

impl SlotLayout {
    fn for_basis(basis: &Basis) -> Self {
        let n_max = basis.atom_norb.iter().copied().max().unwrap_or(0).max(1);
        let p_max = n_max * (n_max + 1) / 2;
        Self {
            p_max,
            n_max,
            // [ J_a (p_max) | J_b (p_max) | K_ab (n_max²) | d_a (p_max) | d_b (p_max) ]
            // The packed densities live in the slot too, so the per-pair kernel
            // has no stack scratch to zero.
            stride: 4 * p_max + n_max * n_max,
        }
    }
}

/// How the two-electron kernels read a density.
///
/// A dense `Matrix` and a divide-and-conquer sparse one are both densities and differ only in
/// what they hold: outside its pattern the sparse one *is* zero, so a view that returns zero
/// there is exact rather than approximate. Reading through this is what lets the same kernel
/// serve the molecular SCF and the partitioned one without either knowing about the other.
pub(crate) trait DensityView {
    /// `P_{munu}`. A sparse view returns a structural zero outside its pattern, which for a
    /// divide-and-conquer density is the value and not an approximation of it.
    fn at(&self, mu: usize, nu: usize) -> f64;
}

impl DensityView for Matrix {
    #[inline]
    fn at(&self, mu: usize, nu: usize) -> f64 {
        self[(mu, nu)]
    }
}

/// A [`crate::dc::pattern::SparseMatrix`] with the pattern that gives it meaning.
pub(crate) struct PatternedDensity<'a> {
    pub pattern: &'a crate::dc::pattern::DensityPattern,
    pub values: &'a crate::dc::pattern::SparseMatrix,
}

impl DensityView for PatternedDensity<'_> {
    #[inline]
    fn at(&self, mu: usize, nu: usize) -> f64 {
        self.values.get(self.pattern, mu, nu)
    }
}

/// Pack atom `a`'s density block into the `(μν)` multipole vector the two-center
/// Coulomb kernel contracts against: off-diagonal orbital pairs are counted
/// twice, since `P` is symmetric and `(μν|λσ) = (νμ|λσ)`.
///
/// This is what turns the Coulomb term from the naive `n_a² · n_b²` four-index
/// loop into two packed mat-vecs of `p_a · p_b` (256 → 100 multiply-adds for an
/// sp/sp pair), with identical arithmetic.
#[inline]
fn packed_density<D: DensityView + ?Sized>(p: &D, off: usize, n: usize, out: &mut [f64]) {
    for mu in 0..n {
        out[pack(mu, mu)] = p.at(off + mu, off + mu);
        for nu in 0..mu {
            out[pack(mu, nu)] = 2.0 * p.at(off + mu, off + nu);
        }
    }
}

/// Two-center Coulomb + exchange for one pair, written into its scratch slot.
/// `exchange_scale` is 1 for a spin Fock (`K` from the same-spin density) and
/// ½ for the RHF response operator.
#[allow(clippy::too_many_arguments)] // per-pair kernel: buffers, densities, weights
fn pair_contribution<D: DensityView + ?Sized>(
    slot: &mut [f64],
    layout: SlotLayout,
    pair: &PairIntegral,
    basis: &Basis,
    p_coulomb: &D,
    p_exchange: &D,
    exchange_scale: f64,
    sw: f64,
) {
    slot.fill(0.0);
    if sw == 0.0 {
        return;
    }
    let te = &pair.te;
    let (oa, ob) = (basis.atom_offset[pair.a], basis.atom_offset[pair.b]);
    let (na, nb) = (te.norb_i, te.norb_j);
    let (pa, pb) = (na * (na + 1) / 2, nb * (nb + 1) / 2);
    let (j_a, rest) = slot.split_at_mut(layout.p_max);
    let (j_b, rest) = rest.split_at_mut(layout.p_max);
    let (k_ab, rest) = rest.split_at_mut(layout.n_max * layout.n_max);
    let (d_a, d_b) = rest.split_at_mut(layout.p_max);

    // --- Coulomb: J_a = W · d_b and J_b = Wᵀ · d_a on the packed tables.
    packed_density(p_coulomb, oa, na, d_a);
    packed_density(p_coulomb, ob, nb, d_b);
    for p in 0..pa {
        let row = &te.w_row(p)[..pb];
        let mut acc = 0.0;
        for (q, &w) in row.iter().enumerate() {
            acc += d_b[q] * w;
        }
        j_a[p] = sw * acc;
        let da = sw * d_a[p];
        for (q, &w) in row.iter().enumerate() {
            j_b[q] += da * w;
        }
    }

    // --- Exchange: K[μ][λ] = Σ_{ν,σ} P(ν_a, σ_b) (μν|λσ).
    let scale = sw * exchange_scale;
    for mu in 0..na {
        for nu in 0..na {
            let row = te.w_row(pack(mu, nu));
            for la in 0..nb {
                let mut acc = 0.0;
                for si in 0..nb {
                    acc += p_exchange.at(oa + nu, ob + si) * row[pack(la, si)];
                }
                k_ab[mu * layout.n_max + la] -= scale * acc;
            }
        }
    }
}

/// Scatter one pair's scratch slot into the shared Fock matrix.
#[inline]
fn scatter_pair(
    f: &mut Matrix,
    layout: SlotLayout,
    pair: &PairIntegral,
    basis: &Basis,
    slot: &[f64],
) {
    let te = &pair.te;
    let (oa, ob) = (basis.atom_offset[pair.a], basis.atom_offset[pair.b]);
    let (na, nb) = (te.norb_i, te.norb_j);
    let j_a = &slot[..layout.p_max];
    let j_b = &slot[layout.p_max..2 * layout.p_max];
    let k_ab = &slot[2 * layout.p_max..2 * layout.p_max + layout.n_max * layout.n_max];
    for mu in 0..na {
        for nu in 0..na {
            f[(oa + mu, oa + nu)] += j_a[pack(mu, nu)];
        }
    }
    for la in 0..nb {
        for si in 0..nb {
            f[(ob + la, ob + si)] += j_b[pack(la, si)];
        }
    }
    for mu in 0..na {
        for la in 0..nb {
            let value = f[(oa + mu, ob + la)] + k_ab[mu * layout.n_max + la];
            f[(oa + mu, ob + la)] = value;
            f[(ob + la, oa + mu)] = value;
        }
    }
}

/// Run the whole two-center pair loop: batched `par_chunks_mut` over pairs into
/// a reusable scratch buffer, then a serial scatter. `pair_sw` (the CPHF
/// distance cutoff) weights each pair; a zero weight skips it.
///
/// The two-center term is the `O(N²)` part of every SCF and CPHF iteration and
/// was previously a single serial loop; the pairs are independent, so only the
/// scatter (which does collide, on the shared diagonal blocks) stays serial.
fn accumulate_two_center(
    f: &mut Matrix,
    core: &CoreHamiltonian,
    basis: &Basis,
    p_coulomb: &Matrix,
    p_exchange: &Matrix,
    exchange_scale: f64,
    pair_sw: Option<&[f64]>,
) {
    let layout = SlotLayout::for_basis(basis);
    let mut scratch = vec![0.0f64; PAIR_BATCH.min(core.pairs.len().max(1)) * layout.stride];
    for (batch_index, batch) in core.pairs.chunks(PAIR_BATCH).enumerate() {
        let base = batch_index * PAIR_BATCH;
        let used = &mut scratch[..batch.len() * layout.stride];
        used.par_chunks_mut(layout.stride)
            .zip(batch.par_iter())
            .enumerate()
            .for_each(|(k, (slot, pair))| {
                let sw = pair_sw.map_or(1.0, |s| s[base + k]);
                pair_contribution(
                    slot,
                    layout,
                    pair,
                    basis,
                    p_coulomb,
                    p_exchange,
                    exchange_scale,
                    sw,
                );
            });
        for (pair, slot) in batch.iter().zip(used.chunks(layout.stride)) {
            scatter_pair(f, layout, pair, basis, slot);
        }
    }
}

/// One-center two-electron integral `(a b | c d)` (all orbitals on the same atom), from the
/// PM3 one-center parameters `Gss/Gsp/Gpp/Gp2/Hsp`. Orbital indices: 0 = s, 1..3 = p. Uses the
/// NDDO index symmetries `(ab|cd) = (ba|cd) = (ab|dc) = (cd|ab)`. (d elements use the
/// Slater–Condon table in [`crate::onecenter`] instead.)
#[inline]
#[allow(clippy::too_many_arguments)] // direct mathematical integral signature
pub fn oc_two_electron(
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    gss: f64,
    gsp: f64,
    gpp: f64,
    gp2: f64,
    hsp: f64,
) -> f64 {
    // Diagonal-pair cases: bra = (x,x), ket = (y,y).
    if a == b && c == d {
        return match (a == 0, c == 0) {
            (true, true) => gss,  // (ss|ss)
            (true, false) => gsp, // (ss|pp)
            (false, true) => gsp, // (pp|ss)
            (false, false) => {
                if a == c {
                    gpp // (pp|pp)
                } else {
                    gp2 // (pp|p'p')
                }
            }
        };
    }
    // Off-diagonal-pair cases: sort bra/ket index pairs.
    let (ba, bb) = (a.min(b), a.max(b));
    let (kc, kd) = (c.min(d), c.max(d));
    // (s p_i | s p_i) = H_sp
    if ba == 0 && bb != 0 && kc == 0 && kd != 0 && bb == kd {
        return hsp;
    }
    // (p_i p_j | p_i p_j) = ½(G_pp − G_p2),  i ≠ j
    if ba != 0 && bb != 0 && ba != bb && ba == kc && bb == kd {
        return 0.5 * (gpp - gp2);
    }
    0.0
}

/// Build the spin-σ Fock matrix `F = H_core + J(p_tot) − K(p_spin)`.
pub fn build_fock_spin(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm3Parameters,
    core: &CoreHamiltonian,
    p_tot: &Matrix,
    p_spin: &Matrix,
) -> Result<Matrix> {
    let mut f = core.h_core.clone();

    // One-center (intra-atomic) contributions.
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let elem = params.element(atom.z)?;
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
        let (gss, gsp, gpp, gp2, hsp) = (elem.g_ss, elem.g_sp, elem.g_pp, elem.g_p2, elem.h_sp);
        // For d elements the one-center (ab|cd) come from the spd Slater–Condon
        // table; for s/p elements the closed-form parameters suffice.
        let oc = |a: usize, b: usize, c: usize, d: usize| -> f64 {
            if let Some(spd) = &elem.onecenter {
                spd.get(a, b, c, d)
            } else {
                oc_two_electron(a, b, c, d, gss, gsp, gpp, gp2, hsp)
            }
        };
        for mu in 0..n {
            for nu in 0..n {
                let mut acc = 0.0;
                for la in 0..n {
                    for si in 0..n {
                        // Coulomb (μν|λσ) from total density.
                        acc += p_tot[(off + la, off + si)] * oc(mu, nu, la, si);
                        // Exchange (μλ|νσ) from same-spin density.
                        acc -= p_spin[(off + la, off + si)] * oc(mu, la, nu, si);
                    }
                }
                f[(off + mu, off + nu)] += acc;
            }
        }
    }

    // Two-center (inter-atomic) contributions: Coulomb J from the total density,
    // exchange K from the same-spin density. This is the O(N²) part of every SCF
    // iteration, so it runs batched-parallel over pairs.
    accumulate_two_center(&mut f, core, basis, p_tot, p_spin, 1.0, None);

    Ok(f)
}

/// CPHF **response** two-electron operator `G(∂P) = J(∂P) − K(½ ∂P)` (RHF), i.e.
/// the Fock built from the response density with **no** `H_core`. Optional
/// `pair_sw` gives a per-`core.pairs` switch factor in `[0,1]`; a pair with
/// factor 0 is skipped entirely (the CPHF/Hessian distance cutoff). `None`
/// applies every pair at full weight → bit-identical to `build_fock(∂P) − H_core`.
/// The one-center (intra-atomic) block is always exact and full (it is O(N), not
/// the bottleneck).
/// [`build_fock_spin`] onto a sparsity pattern, for divide-and-conquer.
///
/// The same arithmetic with a different destination. The partitioned solve reads the Fock matrix
/// only where some subsystem holds both orbitals — that is what a subsystem gather *is* — so
/// every other entry was computed, stored in an `nao × nao` array, and never looked at. At 960
/// atoms that array is 128 MB and the part of it that gets read is a couple of megabytes.
///
/// The density is read the same way and for the same reason: outside the pattern a
/// divide-and-conquer density is not small but structurally zero, so a sparse view returning
/// zero there is the value rather than an approximation of it. See the note on
/// [`crate::dc::pattern`].
///
/// The off-diagonal block is added twice rather than mirrored. The dense scatter writes
/// `f[(b, a)] = f[(a, b)] + k`, reading one entry to set the other; that is an addition in
/// disguise, because everything written before it — the core, the one-centre blocks, the
/// earlier pairs — is symmetric, so the two entries are equal when it runs. Adding to both is
/// the same number without the read, which on a sparse matrix costs a second search.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_fock_spin_sparse(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm3Parameters,
    core: &CoreHamiltonian,
    h_core: &crate::dc::pattern::SparseMatrix,
    pattern: &crate::dc::pattern::DensityPattern,
    p_tot: &crate::dc::pattern::SparseMatrix,
    p_spin: &crate::dc::pattern::SparseMatrix,
    out: &mut crate::dc::pattern::SparseMatrix,
) -> Result<()> {
    out.values_mut().copy_from_slice(h_core.values());

    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let elem = params.element(atom.z)?;
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
        let (gss, gsp, gpp, gp2, hsp) = (elem.g_ss, elem.g_sp, elem.g_pp, elem.g_p2, elem.h_sp);
        let oc = |a: usize, b: usize, c: usize, d: usize| -> f64 {
            if let Some(spd) = &elem.onecenter {
                spd.get(a, b, c, d)
            } else {
                oc_two_electron(a, b, c, d, gss, gsp, gpp, gp2, hsp)
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

    let coulomb = PatternedDensity {
        pattern,
        values: p_tot,
    };
    let exchange = PatternedDensity {
        pattern,
        values: p_spin,
    };
    let layout = SlotLayout::for_basis(basis);
    let mut scratch = vec![0.0f64; PAIR_BATCH.min(core.pairs.len().max(1)) * layout.stride];
    for batch in core.pairs.chunks(PAIR_BATCH) {
        let used = &mut scratch[..batch.len() * layout.stride];
        used.par_chunks_mut(layout.stride)
            .zip(batch.par_iter())
            .for_each(|(slot, pair)| {
                pair_contribution(slot, layout, pair, basis, &coulomb, &exchange, 1.0, 1.0);
            });
        for (pair, slot) in batch.iter().zip(used.chunks(layout.stride)) {
            let te = &pair.te;
            let (oa, ob) = (basis.atom_offset[pair.a], basis.atom_offset[pair.b]);
            let (na, nb) = (te.norb_i, te.norb_j);
            let j_a = &slot[..layout.p_max];
            let j_b = &slot[layout.p_max..2 * layout.p_max];
            let k_ab = &slot[2 * layout.p_max..2 * layout.p_max + layout.n_max * layout.n_max];
            for mu in 0..na {
                for nu in 0..na {
                    out.add(pattern, oa + mu, oa + nu, j_a[pack(mu, nu)]);
                }
            }
            for la in 0..nb {
                for si in 0..nb {
                    out.add(pattern, ob + la, ob + si, j_b[pack(la, si)]);
                }
            }
            for mu in 0..na {
                for la in 0..nb {
                    let value = k_ab[mu * layout.n_max + la];
                    out.add(pattern, oa + mu, ob + la, value);
                    out.add(pattern, ob + la, oa + mu, value);
                }
            }
        }
    }
    Ok(())
}

pub fn response_fock_rhf(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm3Parameters,
    core: &CoreHamiltonian,
    dp: &Matrix,
    pair_sw: Option<&[f64]>,
) -> Result<Matrix> {
    let nao = basis.nao;
    let mut f = Matrix::zeros(nao, nao);

    // One-center (intra-atomic) J − K, spin ½ (RHF).
    for (ia, atom) in molecule.atoms.iter().enumerate() {
        let elem = params.element(atom.z)?;
        let off = basis.atom_offset[ia];
        let n = basis.atom_norb[ia];
        let (gss, gsp, gpp, gp2, hsp) = (elem.g_ss, elem.g_sp, elem.g_pp, elem.g_p2, elem.h_sp);
        let oc = |a: usize, b: usize, c: usize, d: usize| -> f64 {
            if let Some(spd) = &elem.onecenter {
                spd.get(a, b, c, d)
            } else {
                oc_two_electron(a, b, c, d, gss, gsp, gpp, gp2, hsp)
            }
        };
        for mu in 0..n {
            for nu in 0..n {
                let mut acc = 0.0;
                for la in 0..n {
                    for si in 0..n {
                        acc += dp[(off + la, off + si)] * oc(mu, nu, la, si);
                        acc -= 0.5 * dp[(off + la, off + si)] * oc(mu, la, nu, si);
                    }
                }
                f[(off + mu, off + nu)] += acc;
            }
        }
    }

    // Two-center, with the per-pair distance switch.
    accumulate_two_center(&mut f, core, basis, dp, dp, 0.5, pair_sw);
    Ok(f)
}

/// RHF (closed-shell) Fock: `F = H_core + J(P) − K(½P)`.
pub fn build_fock(
    molecule: &Molecule,
    basis: &Basis,
    params: &Pm3Parameters,
    core: &CoreHamiltonian,
    density: &Matrix,
) -> Result<Matrix> {
    let mut half = density.clone();
    for v in half.as_mut_slice() {
        *v *= 0.5;
    }
    build_fock_spin(molecule, basis, params, core, density, &half)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hamiltonian::build_core;
    use crate::params::Pm3Parameters;

    /// Reference two-center accumulation: the literal four-index loops, kept as
    /// the correctness oracle for the packed/parallel kernel above.
    fn reference_two_center(
        f: &mut Matrix,
        core: &CoreHamiltonian,
        basis: &Basis,
        p_coulomb: &Matrix,
        p_exchange: &Matrix,
        exchange_scale: f64,
    ) {
        for pair in &core.pairs {
            let te = &pair.te;
            let (oa, ob) = (basis.atom_offset[pair.a], basis.atom_offset[pair.b]);
            let (na, nb) = (te.norb_i, te.norb_j);
            for mu in 0..na {
                for nu in 0..na {
                    let mut acc = 0.0;
                    for la in 0..nb {
                        for si in 0..nb {
                            acc += p_coulomb[(ob + la, ob + si)] * te.two_e(mu, nu, la, si);
                        }
                    }
                    f[(oa + mu, oa + nu)] += acc;
                }
            }
            for la in 0..nb {
                for si in 0..nb {
                    let mut acc = 0.0;
                    for mu in 0..na {
                        for nu in 0..na {
                            acc += p_coulomb[(oa + mu, oa + nu)] * te.two_e(mu, nu, la, si);
                        }
                    }
                    f[(ob + la, ob + si)] += acc;
                }
            }
            for mu in 0..na {
                for la in 0..nb {
                    let mut acc = 0.0;
                    for nu in 0..na {
                        for si in 0..nb {
                            acc += exchange_scale
                                * p_exchange[(oa + nu, ob + si)]
                                * te.two_e(mu, nu, la, si);
                        }
                    }
                    f[(oa + mu, ob + la)] -= acc;
                    f[(ob + la, oa + mu)] = f[(oa + mu, ob + la)];
                }
            }
        }
    }

    fn setup(xyz: &str) -> (Molecule, Pm3Parameters, Basis, CoreHamiltonian, Matrix) {
        let molecule = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Pm3Parameters::standard().unwrap();
        let basis = Basis::build(&molecule, &params).unwrap();
        let core = build_core(&molecule, &basis, &params).unwrap();
        // A dense, symmetric, non-trivial test density.
        let n = basis.nao;
        let mut p = Matrix::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let v = 0.37 * ((i * 7 + j * 3) % 11) as f64 - 1.1;
                p[(i, j)] = v;
                p[(j, i)] = v;
            }
        }
        (molecule, params, basis, core, p)
    }

    /// The packed Coulomb mat-vec and the batched-parallel pair loop must
    /// reproduce the literal four-index contraction element for element.
    #[test]
    fn packed_parallel_two_center_matches_four_index_loops() {
        // Mixed s-only / sp / halogen atoms so both packed dimensions appear.
        let xyz = "6\nmixed\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\nC 2.4 0.3 -0.7\nCl 3.9 -0.8 0.6\nH 2.2 1.3 -1.2\n";
        let (_m, _pa, basis, core, p) = setup(xyz);
        for &scale in &[1.0, 0.5] {
            let mut fast = Matrix::zeros(basis.nao, basis.nao);
            accumulate_two_center(&mut fast, &core, &basis, &p, &p, scale, None);
            let mut slow = Matrix::zeros(basis.nao, basis.nao);
            reference_two_center(&mut slow, &core, &basis, &p, &p, scale);
            let mut max_difference = 0.0_f64;
            for i in 0..basis.nao {
                for j in 0..basis.nao {
                    max_difference = max_difference.max((fast[(i, j)] - slow[(i, j)]).abs());
                }
            }
            assert!(
                max_difference < 1.0e-12,
                "exchange_scale={scale}: max difference {max_difference:.3e}"
            );
        }
    }

    /// The pair-cache size estimate that guards against OOM must match what is
    /// actually allocated, or the guard is meaningless.
    #[test]
    fn pair_cache_estimate_matches_allocation() {
        let xyz =
            "5\nmix\nS 0.0 0.0 0.0\nH 1.3 0.0 0.0\nH -0.3 1.3 0.0\nBr 2.9 0.4 0.5\nH 3.5 1.6 0.9\n";
        let (_m, _pa, basis, core, _p) = setup(xyz);
        let actual: usize = core
            .pairs
            .iter()
            .map(|p| {
                p.te.w.len() * std::mem::size_of::<f64>()
                    + std::mem::size_of::<crate::hamiltonian::PairIntegral>()
            })
            .sum();
        let estimated = crate::hamiltonian::pair_cache_bytes(&basis);
        assert!(
            estimated >= actual && estimated <= actual * 2,
            "estimate {estimated} vs actual {actual}"
        );
    }

    /// The two-center Fock builds must stay symmetric; the SCF commutator
    /// shortcut `PF = (FP)ᵀ` depends on it.
    #[test]
    fn built_fock_is_symmetric() {
        let xyz =
            "4\nformaldehyde\nC 0.0 0.0 0.0\nO 0.0 0.0 1.21\nH 0.94 0.0 -0.54\nH -0.94 0.0 -0.54\n";
        let (molecule, params, basis, core, p) = setup(xyz);
        let f = build_fock(&molecule, &basis, &params, &core, &p).unwrap();
        for i in 0..basis.nao {
            for j in 0..basis.nao {
                assert!(
                    (f[(i, j)] - f[(j, i)]).abs() < 1.0e-12,
                    "F[{i},{j}] asymmetric"
                );
            }
        }
    }
}
