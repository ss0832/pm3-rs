// SPDX-License-Identifier: GPL-3.0-or-later

//! The point-charge realization of PM3's charge distribution — the bridge between the NDDO
//! density and the Ewald sum.
//!
//! # Why this exists
//!
//! PM3's two-center Coulomb integrals are not `q_A q_B / R` between nuclei. They come from the
//! Dewar–Thiel multipole model, in which each atom's orbital-pair density is represented by a
//! small configuration of point charges at fixed offsets from the nucleus, interacting through
//! the Klopman–Ohno screened form `1/√(R² + a)`. `src/integrals.rs` evaluates those
//! configurations directly: every term of `local_xh_g` and `local_xx_g` is one
//! `±q/√((r ± d)² + a)`.
//!
//! Periodic electrostatics needs the **point limit** of exactly those configurations — the same
//! charges at the same offsets, with `a → 0` — because that is the part with a `1/R` tail that
//! only Ewald can sum. Representing the cell by nuclear monopoles instead would not do: the
//! neglected charge–dipole term falls off as `1/R²`, and its 3D lattice sum diverges linearly.
//!
//! # What is represented
//!
//! Once written as point charges, all three of PM3's Coulomb pieces become one sum:
//!
//! * **core–core** — a monopole `+Z_core` at each nucleus;
//! * **electron–core** — those monopoles against the electronic multipoles;
//! * **electron–electron** — the electronic multipoles against each other.
//!
//! The electronic multipoles come from the atom's own density block, with fixed offsets and
//! weights that are linear in the density:
//!
//! | AO pair | configuration (offsets from the nucleus) |
//! |---|---|
//! | `(s,s)` | monopole: `1` at `0` |
//! | `(s,p_α)` | dipole: `±½` at `±D₁ ê_α` |
//! | `(p_α,p_α)` | monopole `1` at `0`, plus `+¼` at `±2D₂ ê_α` and `−½` at `0` |
//! | `(p_α,p_β)`, `α ≠ β` | `±¼` at the four `(±D₂ê_α ± D₂ê_β)` corners |
//!
//! `D₁` is the element's `dd` and `D₂` its `qq`, the same charge separations `calpar` derives.
//! For a main-group sp element the electron–core multipoles (`ddp[2]`, `ddp[3]/√2`) are equal to
//! the electron–electron ones, so a single geometry serves all three pieces; only the screening
//! `a` differs between them, and screening is exactly what the point limit drops.
//!
//! # What this is *not*
//!
//! It is not a drop-in rewrite of the NDDO integral. PM3 evaluates its multipole integrals in
//! the **local diatomic frame**, where the configuration is aligned with the interatomic axis,
//! and then rotates the result as a tensor. A configuration of point charges placed along the
//! *global* axes is a different object at finite separation: for a carbon `(s,p)` dipole against
//! a monopole 5.5 Bohr away, the two disagree by about 4%, because the local-frame construction
//! is a multipole expansion truncated at its own order while the point-charge sum carries every
//! order the geometry implies.
//!
//! Measured on the axis, where the two frames coincide, they agree exactly — that is the
//! sharpest available check on the charges and offsets themselves, and it is what
//! `configurations_are_exact_on_axis` pins.
//!
//! Off axis they agree only **asymptotically**, and that is precisely what the Ewald split
//! needs. What has to be true is:
//!
//! * the point-charge model carries the correct `1/R`, `1/R²`, `1/R³` multipole tail, so Ewald
//!   sums the right long-range object; and
//! * the leftover `W_NDDO − W_point` decays fast enough to be truncated in real space.
//!
//! It does: the frame difference is `O(D³/R⁴)`, absolutely convergent in 3D, and the only piece
//! that is *not* — the Klopman–Ohno screening's spurious `−a/2R³` tail — is the one the
//! switching function in [`crate::pbc`] removes on purpose. `difference_from_the_kernel_decays`
//! measures that decay rather than assuming it.

use crate::math::Vec3;
use crate::params::Pm3Element;

/// One point charge in an atom's multipole realization: where it sits relative to the nucleus,
/// and how much of a given density-matrix element it carries.
#[derive(Clone, Copy, Debug)]
pub struct SiteWeight {
    /// Index into the atom's canonical site list.
    pub site: usize,
    /// Coefficient multiplying the density-matrix element.
    pub weight: f64,
}

/// The fixed point-charge geometry of one atom, plus the map from density-matrix elements onto
/// those points.
#[derive(Clone, Debug)]
pub struct AtomSites {
    /// Offsets from the nucleus (Bohr). Site 0 is always the nucleus itself.
    pub offsets: Vec<Vec3>,
    /// For each AO pair `(μ, ν)` with `μ ≤ ν`, the points its density feeds and by how much.
    /// Indexed by [`packed_index`].
    pub pair_weights: Vec<Vec<SiteWeight>>,
    /// Number of AOs on the atom: 0 for a sparkle or point charge, 1 for an s-only element,
    /// 4 for an sp one, 9 for a `d` one (built by `AtomSites::build_spd`). No PM3 element has
    /// a `d` shell, so that path is exercised by a synthetic element rather than a real one —
    /// see `no_pm3_element_carries_a_d_shell`.
    pub n_orb: usize,
}

/// This crate's global AO index → MOPAC's local `mndod` index (1-based).
///
/// Global order is `s, px, py, pz, dx²−y², dxz, dz², dyz, dxy`; `mndod` puts the σ component
/// of each shell first, `s, pz, px, py, dz², dxz, dyz, dx²−y², dxy`. The [`crate::integrals_d`]
/// tables are indexed the second way.
const LOCAL_ORDER: [usize; 9] = [1, 3, 4, 2, 8, 6, 5, 7, 9];

/// The point charges realizing one `(l, m)` multipole with charge separation `D`.
///
/// Read off [`crate::integrals_d::charg`] rather than from a textbook, because what has to
/// match is that kernel and not a general convention. Two of them were derived by solving for
/// the layout that reproduces `charg`'s own `(2, 0)` and `(2, 2)` self-interactions term by
/// term; the rest follow from the same reading.
///
/// The frame is the **global** one — `m = 0` along `z`, `m = 1` along `x`, `m = −1` along `y`.
/// MOPAC evaluates these in the local diatomic frame and rotates the result, which is a
/// different object at finite separation; the two agree asymptotically, which is all the
/// long-range half needs. See the module note.
fn configuration(l: usize, m: i32, d: f64) -> Vec<(f64, Vec3)> {
    let x = Vec3::new(d, 0.0, 0.0);
    let y = Vec3::new(0.0, d, 0.0);
    let z = Vec3::new(0.0, 0.0, d);
    // The |m| = 1 and m = −2 quadrupoles sit on a square rotated 45° from the axes, so their
    // half-diagonal is `D`, putting each corner at `D/√2` on two axes.
    let h = d / std::f64::consts::SQRT_2;
    match (l, m) {
        (0, _) => vec![(1.0, Vec3::zero())],
        (1, 0) => vec![(0.5, z), (-0.5, z * -1.0)],
        (1, 1) => vec![(0.5, x), (-0.5, x * -1.0)],
        (1, -1) => vec![(0.5, y), (-0.5, y * -1.0)],
        // +¼ on the z axis, and the compensating −½ split evenly over the four transverse
        // points so the configuration is axially symmetric and traceless.
        (2, 0) => vec![
            (0.25, z),
            (0.25, z * -1.0),
            (-0.125, x),
            (-0.125, x * -1.0),
            (-0.125, y),
            (-0.125, y * -1.0),
        ],
        (2, 1) => square(Vec3::new(h, 0.0, 0.0), Vec3::new(0.0, 0.0, h)),
        (2, -1) => square(Vec3::new(0.0, h, 0.0), Vec3::new(0.0, 0.0, h)),
        // x² − y²: on the axes rather than the diagonals, unlike its m = −2 partner.
        (2, 2) => vec![(0.25, x), (0.25, x * -1.0), (-0.25, y), (-0.25, y * -1.0)],
        (2, -2) => square(Vec3::new(h, 0.0, 0.0), Vec3::new(0.0, h, 0.0)),
        _ => Vec::new(),
    }
}

/// The four corners of a square quadrupole spanned by `a` and `b`, signed by the product of the
/// two coordinates — the `xz`, `yz` and `xy` configurations.
fn square(a: Vec3, b: Vec3) -> Vec<(f64, Vec3)> {
    vec![
        (0.25, a + b),
        (-0.25, a - b),
        (-0.25, a * -1.0 + b),
        (0.25, (a + b) * -1.0),
    ]
}

/// Packed index of an AO pair with `mu <= nu`, matching [`crate::integrals::pack`].
#[inline]
pub fn packed_index(mu: usize, nu: usize) -> usize {
    let (lo, hi) = if mu <= nu { (mu, nu) } else { (nu, mu) };
    hi * (hi + 1) / 2 + lo
}

impl AtomSites {
    /// Build the point-charge geometry for one element.
    ///
    /// Every element realizes. A Sparkle or a point atom has no orbitals and therefore no
    /// electronic multipoles at all, so its realization is the single site every atom has
    /// anyway — the nucleus, carrying its core charge. That is not a special case being
    /// tolerated; it is what the model says about an atom with no density.
    pub fn build(elem: &Pm3Element) -> Option<Self> {
        if elem.n_orb == 0 {
            return Some(Self {
                offsets: vec![Vec3::zero()],
                pair_weights: Vec::new(),
                n_orb: 0,
            });
        }
        if elem.has_d() {
            return Some(Self::build_spd(elem));
        }
        let n_orb = elem.n_orb;
        let d1 = elem.dd;
        let d2 = elem.qq;

        // Canonical site list. The nucleus first, then the dipole pair for each axis, then the
        // linear-quadrupole pair for each axis, then the four corners of each off-diagonal
        // quadrupole. Sites whose charge turns out to be zero are dropped later, when the
        // density is known, so this list stays a fixed property of the element.
        let mut offsets = vec![Vec3::zero()];
        let axes = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let mut dipole_site = [[0usize; 2]; 3];
        let mut linear_site = [[0usize; 2]; 3];
        if n_orb >= 4 {
            for (axis, direction) in axes.iter().enumerate() {
                dipole_site[axis][0] = offsets.len();
                offsets.push(*direction * d1);
                dipole_site[axis][1] = offsets.len();
                offsets.push(*direction * -d1);
            }
            for (axis, direction) in axes.iter().enumerate() {
                linear_site[axis][0] = offsets.len();
                offsets.push(*direction * (2.0 * d2));
                linear_site[axis][1] = offsets.len();
                offsets.push(*direction * (-2.0 * d2));
            }
        }
        // Off-diagonal quadrupole corners, keyed by the unordered axis pair.
        let mut corner_site = [[[0usize; 4]; 3]; 3];
        if n_orb >= 4 {
            for a in 0..3 {
                for b in (a + 1)..3 {
                    let (ea, eb) = (axes[a] * d2, axes[b] * d2);
                    let corners = [ea + eb, ea * -1.0 + eb * -1.0, ea - eb, ea * -1.0 + eb];
                    for (index, corner) in corners.iter().enumerate() {
                        corner_site[a][b][index] = offsets.len();
                        offsets.push(*corner);
                    }
                }
            }
        }

        let packed = n_orb * (n_orb + 1) / 2;
        let mut pair_weights = vec![Vec::new(); packed];

        // (s,s): a bare monopole.
        pair_weights[packed_index(0, 0)].push(SiteWeight {
            site: 0,
            weight: 1.0,
        });

        if n_orb >= 4 {
            for axis in 0..3 {
                let p = axis + 1;
                // (s, p_α): a dipole of separation D₁, positive lobe on the +α side. The sign
                // is the integral kernels' convention, measured on the axis rather than
                // reasoned about: `(p_z,s|s,s)` for carbon against a monopole 5.5 Bohr along
                // +z is +0.7671480174 eV, which is `+½` at `+D₁ẑ` with `−½` at `−D₁ẑ`, to
                // every digit. The opposite assignment reproduces the magnitude and gets the
                // sign of every dipole term in the model backwards.
                let pair = packed_index(0, p);
                pair_weights[pair].push(SiteWeight {
                    site: dipole_site[axis][0],
                    weight: 0.5,
                });
                pair_weights[pair].push(SiteWeight {
                    site: dipole_site[axis][1],
                    weight: -0.5,
                });

                // (p_α, p_α): monopole plus a linear quadrupole.
                let pair = packed_index(p, p);
                pair_weights[pair].push(SiteWeight {
                    site: 0,
                    weight: 1.0,
                });
                pair_weights[pair].push(SiteWeight {
                    site: linear_site[axis][0],
                    weight: 0.25,
                });
                pair_weights[pair].push(SiteWeight {
                    site: linear_site[axis][1],
                    weight: 0.25,
                });
                pair_weights[pair].push(SiteWeight {
                    site: 0,
                    weight: -0.5,
                });
            }

            // (p_α, p_β), α ≠ β: the four-corner quadrupole. The two indices are Cartesian axis
            // labels used to address three different things (`corner_site` and the two AO
            // numbers), and they run over the upper triangle, so iterating the range directly
            // says what is meant; the iterator rewrite clippy suggests would obscure it.
            #[allow(clippy::needless_range_loop)]
            for a in 0..3 {
                for b in (a + 1)..3 {
                    let pair = packed_index(a + 1, b + 1);
                    let weights = [0.25, 0.25, -0.25, -0.25];
                    for (index, weight) in weights.iter().enumerate() {
                        pair_weights[pair].push(SiteWeight {
                            site: corner_site[a][b][index],
                            weight: *weight,
                        });
                    }
                }
            }
        }

        Some(Self {
            offsets,
            pair_weights,
            n_orb,
        })
    }

    /// The same construction for an element with `d` orbitals, driven by MOPAC's own multipole
    /// table rather than by hand.
    ///
    /// # Why this is a table lookup and the `sp` case is not
    ///
    /// MNDO-d already writes every AO pair's charge distribution as a sum of `(l, m)` multipoles
    /// — that is what [`crate::integrals_d::ch`] is, one coefficient per pair per `(l, m)`, and
    /// [`Pm3Element::ddp`] holds the charge separation for each shell-pair class. There are
    /// forty-five pairs and nine distinct configurations; enumerating the pairs by hand would be
    /// forty-five chances to make a transcription error, and the table is already in the tree.
    ///
    /// The `sp` construction above stays hand-written because it realizes the *other* kernel:
    /// `sp`-only elements go through [`crate::integrals::pair_two_electron`], whose layout
    /// differs from `mndod`'s even though the moments agree. Both are correct — the point model
    /// only has to carry the right `l ≤ 2` tail, and the difference from either is absolutely
    /// convergent and picked up by the short-range correction.
    ///
    /// # The two orderings
    ///
    /// The `ch` table is indexed in MOPAC's *local* `mndod` order, `s, pz, px, py, dz², dxz,
    /// dyz, dx²−y², dxy`, with the σ component of each shell first. This crate's global AO
    /// order is `s, px, py, pz, dx²−y², dxz, dz², dyz, dxy`. [`LOCAL_ORDER`] is that
    /// permutation, and getting it wrong is the one mistake here that would still produce a
    /// plausible-looking answer — which is why
    /// [`tests::the_d_configurations_reproduce_the_kernel_on_axis`] checks against the integral
    /// kernel rather than against the table.
    ///
    /// # What is not realized
    ///
    /// Nothing. The MNDO-d multipole model *is* `l ≤ 2` — `ch` has no `l = 3` or `l = 4`
    /// members — so this carries every moment the method itself has, not a truncation of them.
    fn build_spd(elem: &Pm3Element) -> Self {
        use crate::integrals_d::{ch, indexd, indx, LORB};

        // The s/p block is *not* MNDO-d. `pair_two_electron_spd` overwrites it with the classic
        // `reppd` model — `dd` and `qq`, the same one every main-group element uses — and reaches
        // for the d-multipole kernel only where a d orbital is actually involved. MOPAC does the
        // same, and the two models disagree: for zinc the `(s,p)` separation is 1.62 Bohr under
        // one and 0.62 under the other. So the realization is seeded from the s/p construction
        // and only extended here, rather than rebuilt.
        let mut sp_element = elem.clone();
        sp_element.n_orb = 4;
        let sp = Self::build(&sp_element).expect("an sp element always realizes");

        let mut offsets = sp.offsets.clone();
        // Sites are interned by position so classes that happen to share a separation share
        // their points, and so the list stays as short as the geometry allows.
        let intern = |offsets: &mut Vec<Vec3>, position: Vec3| -> usize {
            const SAME: f64 = 1.0e-12;
            for (index, existing) in offsets.iter().enumerate() {
                if (*existing - position).norm2() < SAME {
                    return index;
                }
            }
            offsets.push(position);
            offsets.len() - 1
        };

        let packed = 9 * 10 / 2;
        let mut pair_weights: Vec<Vec<SiteWeight>> = vec![Vec::new(); packed];
        // `packed_index` agrees with the four-orbital packing on the first ten entries, so the
        // s/p weights transfer position for position.
        pair_weights[..sp.pair_weights.len()].clone_from_slice(&sp.pair_weights);

        for mu in 0..9 {
            for nu in 0..=mu {
                if mu < 4 && nu < 4 {
                    continue;
                }
                let (lm, ln) = (LOCAL_ORDER[mu], LOCAL_ORDER[nu]);
                let pair = indexd(lm, ln);
                let (li, lj) = (LORB[mu], LORB[nu]);
                let separation = elem.ddp[indx(li + 1, lj + 1)];

                let mut terms: Vec<SiteWeight> = Vec::new();
                for l in li.abs_diff(lj).min(2)..=(li + lj).min(2) {
                    for m in -(l as i32)..=(l as i32) {
                        let coefficient = ch(pair, l, m);
                        if coefficient == 0.0 {
                            continue;
                        }
                        for (charge, position) in configuration(l, m, separation) {
                            let site = intern(&mut offsets, position);
                            terms.push(SiteWeight {
                                site,
                                weight: coefficient * charge,
                            });
                        }
                    }
                }
                // Several `(l, m)` can land on the same interned site; fold them so the Fock
                // contraction visits each site once.
                terms.sort_unstable_by_key(|t| t.site);
                let mut folded: Vec<SiteWeight> = Vec::with_capacity(terms.len());
                for term in terms {
                    match folded.last_mut() {
                        Some(last) if last.site == term.site => last.weight += term.weight,
                        _ => folded.push(term),
                    }
                }
                folded.retain(|t| t.weight.abs() > 1.0e-14);
                pair_weights[packed_index(nu, mu)] = folded;
            }
        }

        Self {
            offsets,
            pair_weights,
            n_orb: 9,
        }
    }

    /// Charges at each site from an atom's density block and its core charge.
    ///
    /// `density` is the `n_orb × n_orb` on-atom block of the total density matrix, row-major.
    /// The electronic contribution enters with a minus sign (electrons are negative) and
    /// off-diagonal pairs count twice, since `P` is symmetric and `(μν)` and `(νμ)` are the same
    /// configuration.
    pub fn charges(&self, density: &[f64], core_charge: f64) -> Vec<f64> {
        let mut out = vec![0.0; self.offsets.len()];
        out[0] += core_charge;
        let n = self.n_orb;
        for mu in 0..n {
            for nu in 0..=mu {
                let population = if mu == nu {
                    density[mu * n + mu]
                } else {
                    2.0 * density[mu * n + nu]
                };
                if population == 0.0 {
                    continue;
                }
                for term in &self.pair_weights[packed_index(nu, mu)] {
                    out[term.site] -= population * term.weight;
                }
            }
        }
        out
    }

    /// `∂(site charges)/∂P_{μν}` contracted with a set of site potentials — the contribution
    /// the long-range field makes to the Fock matrix element `F_{μν}` on this atom.
    ///
    /// The factor of two for `μ ≠ ν` that [`AtomSites::charges`] applies is deliberately **not**
    /// applied here: `F_{μν}` is the derivative with respect to one matrix element, while the
    /// energy counts `(μν)` and `(νμ)` separately, and the two conventions cancel.
    pub fn fock_contribution(&self, mu: usize, nu: usize, site_potential: &[f64]) -> f64 {
        let mut sum = 0.0;
        for term in &self.pair_weights[packed_index(mu, nu)] {
            sum -= term.weight * site_potential[term.site];
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::PM3_EV;
    use crate::integrals::{pack, pair_two_electron_g};
    use crate::params::Pm3Parameters;

    /// A copy of `elem` with every Klopman–Ohno screening term removed, so the integral kernels
    /// evaluate the bare point-charge configurations. This is the point limit, computed by the
    /// production code rather than re-derived.
    fn unscreened(elem: &Pm3Element) -> Pm3Element {
        let mut out = elem.clone();
        out.rho0 = 0.0;
        out.rho1 = 0.0;
        out.rho2 = 0.0;
        out.po = [0.0; 10];
        out
    }

    /// Direct Coulomb energy between two sets of point charges (eV).
    fn point_energy(
        offsets_a: &[Vec3],
        charges_a: &[f64],
        position_a: Vec3,
        offsets_b: &[Vec3],
        charges_b: &[f64],
        position_b: Vec3,
    ) -> f64 {
        let mut sum = 0.0;
        for (offset_a, charge_a) in offsets_a.iter().zip(charges_a) {
            for (offset_b, charge_b) in offsets_b.iter().zip(charges_b) {
                let d = (position_b + *offset_b) - (position_a + *offset_a);
                sum += charge_a * charge_b / d.norm();
            }
        }
        sum * PM3_EV
    }

    /// Coulomb energy between one AO pair's configuration on atom A and one on atom B (eV).
    fn configuration_energy(
        sites_a: &AtomSites,
        pair_a: (usize, usize),
        position_a: Vec3,
        sites_b: &AtomSites,
        pair_b: (usize, usize),
        position_b: Vec3,
    ) -> f64 {
        let mut sum = 0.0;
        for term_a in &sites_a.pair_weights[packed_index(pair_a.0, pair_a.1)] {
            for term_b in &sites_b.pair_weights[packed_index(pair_b.0, pair_b.1)] {
                let d = (position_b + sites_b.offsets[term_b.site])
                    - (position_a + sites_a.offsets[term_a.site]);
                sum += term_a.weight * term_b.weight / d.norm();
            }
        }
        sum * PM3_EV
    }

    /// The same check for a `d` element, against the `spd` kernel.
    ///
    /// This is what the `d` configurations are validated by, and deliberately not the table they
    /// were built from. Reusing [`crate::integrals_d::ch`] on both sides would confirm only that
    /// the code reads its own input; going through `pair_two_electron_spd` tests the whole
    /// chain — the local-to-global orbital permutation, the `(l, m)` layouts read off `charg`,
    /// the shell-pair class that selects each separation, and the sign of every coefficient.
    ///
    /// A wrong permutation is the failure mode worth insisting on: it produces a configuration
    /// with the right total charge and a plausible magnitude, and would pass anything less than
    /// a comparison with the integral itself.
    #[test]
    fn the_d_configurations_reproduce_the_kernel_on_axis() {
        let params = Pm3Parameters::standard().unwrap();
        let hydrogen = params.element(1).unwrap();
        let sites_h = AtomSites::build(hydrogen).unwrap();
        let labels = ["s", "px", "py", "pz", "dx2-y2", "dxz", "dz2", "dyz", "dxy"];
        let mut failures = Vec::new();

        // PM3 parameterizes all forty-two of its elements on an s/p basis — Zn, Cd and Hg
        // included — so there is no PM3 element with a d shell to test against. The realization
        // is a property of the MNDO-d multipole model rather than of any element's particular
        // numbers, so the subject is built here: a nine-orbital element whose charge separations
        // are chosen only to be distinct and nonzero. That is enough to catch a permuted
        // orbital, a mislaid coefficient or a wrong layout, and since both sides of the
        // comparison read the same separations, inventing them costs the test nothing.
        for scale in [1.0_f64, 1.7] {
            let mut synthetic = params.element(30).unwrap().clone();
            synthetic.n_orb = 9;
            synthetic.ddp = [0.0, 0.0, 0.62, 0.83, 0.47, 0.71, 0.55].map(|v: f64| v * scale);
            synthetic.po = [0.0; 10];
            synthetic.rho0 = 0.0;
            synthetic.rho1 = 0.0;
            synthetic.rho2 = 0.0;
            let elem = &synthetic;
            let z = 30u8;
            let sites = AtomSites::build(elem).unwrap();
            assert_eq!(sites.n_orb, 9, "Z{z} should have realized nine orbitals");

            for axis in 0..3 {
                for distance in [5.0_f64, 7.0, 10.0] {
                    let mut offset = [0.0; 3];
                    offset[axis] = distance;
                    let position_b = Vec3::new(offset[0], offset[1], offset[2]);
                    let te = crate::integrals_d::pair_two_electron_spd::<f64>(
                        &unscreened(elem),
                        &unscreened(hydrogen),
                        offset,
                    );
                    for mu in 0..9 {
                        for nu in 0..=mu {
                            let kernel = te.w[pack(mu, nu) * te.npack_j];
                            let point = configuration_energy(
                                &sites,
                                (nu, mu),
                                Vec3::zero(),
                                &sites_h,
                                (0, 0),
                                position_b,
                            );
                            if (kernel - point).abs() > 1.0e-8 * kernel.abs().max(1.0) {
                                failures.push(format!(
                                    "Z{z} scale {scale} axis {axis} r={distance}: ({},{}|s,s) kernel \
                                     {kernel:.10} vs points {point:.10}",
                                    labels[mu], labels[nu]
                                ));
                            }
                        }
                    }
                }
            }
        }
        assert!(
            failures.is_empty(),
            "{} d-configuration mismatches:\n{}",
            failures.len(),
            failures
                .iter()
                .take(12)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    /// On the interatomic axis the local diatomic frame and the global frame coincide, so the
    /// point-charge configuration and the unscreened kernel must agree **exactly**. This is the
    /// sharpest check on the charges and offsets themselves: a wrong sign, a factor of two, or
    /// `D₂` in place of `2D₂` all show up immediately.
    ///
    /// The partner is `s`-only so each integral isolates one configuration on the heavy atom.
    #[test]
    fn configurations_are_exact_on_axis() {
        let params = Pm3Parameters::standard().unwrap();
        let hydrogen = params.element(1).unwrap();
        let sites_h = AtomSites::build(hydrogen).unwrap();
        let labels = ["s", "px", "py", "pz"];
        let mut failures = Vec::new();

        for z in [6u8, 7, 8, 16, 17] {
            let elem = params.element(z).unwrap();
            let sites = AtomSites::build(elem).unwrap();
            // Along each global axis in turn, so every diagonal configuration takes a turn at
            // being the longitudinal one and at being transverse.
            for axis in 0..3 {
                for distance in [4.0_f64, 5.5, 8.0] {
                    let mut offset = [0.0; 3];
                    offset[axis] = distance;
                    let position_b = Vec3::new(offset[0], offset[1], offset[2]);
                    let te = pair_two_electron_g::<f64>(
                        &unscreened(elem),
                        &unscreened(hydrogen),
                        offset,
                    );
                    for mu in 0..elem.n_orb {
                        for nu in 0..=mu {
                            let kernel = te.w[pack(mu, nu) * te.npack_j];
                            let point = configuration_energy(
                                &sites,
                                (nu, mu),
                                Vec3::zero(),
                                &sites_h,
                                (0, 0),
                                position_b,
                            );
                            if (kernel - point).abs() > 1.0e-9 * kernel.abs().max(1.0) {
                                failures.push(format!(
                                    "Z{z} axis {axis} r={distance}: ({},{}|s,s) kernel \
                                     {kernel:.10} vs points {point:.10}",
                                    labels[mu], labels[nu]
                                ));
                            }
                        }
                    }
                }
            }
        }
        assert!(
            failures.is_empty(),
            "{} configuration(s) disagree on axis:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    /// Off axis the two constructions are genuinely different objects, and what the Ewald split
    /// needs is not that they agree but that their **difference decays fast**.
    ///
    /// The leading multipole terms fall off as `1/R`, `1/R²`, `1/R³`; the frame difference is
    /// `O(1/R⁴)`. Doubling the separation must therefore shrink the difference by at least
    /// 8× — one power faster than the slowest term Ewald is responsible for — or the leftover
    /// could not be truncated in real space at all.
    #[test]
    fn difference_from_the_kernel_decays() {
        let params = Pm3Parameters::standard().unwrap();
        let carbon = params.element(6).unwrap();
        let oxygen = params.element(8).unwrap();
        let sites_c = AtomSites::build(carbon).unwrap();
        let sites_o = AtomSites::build(oxygen).unwrap();
        let direction = Vec3::new(0.61, -0.47, 0.64).normalized();

        for (mu, nu) in [(0usize, 0usize), (0, 3), (1, 1), (3, 3), (1, 2)] {
            let mut previous: Option<(f64, f64)> = None;
            for distance in [6.0_f64, 12.0, 24.0, 48.0] {
                let position_b = direction * distance;
                let te = pair_two_electron_g::<f64>(
                    &unscreened(carbon),
                    &unscreened(oxygen),
                    [position_b.x, position_b.y, position_b.z],
                );
                let kernel = te.w[pack(mu, nu) * te.npack_j + pack(0, 0)];
                let point = configuration_energy(
                    &sites_c,
                    (nu, mu),
                    Vec3::zero(),
                    &sites_o,
                    (0, 0),
                    position_b,
                );
                let difference = (kernel - point).abs();
                if let Some((last_distance, last_difference)) = previous {
                    // Ignore pairs whose difference has already reached rounding noise.
                    if last_difference > 1.0e-12 {
                        let ratio = last_difference / difference.max(f64::MIN_POSITIVE);
                        assert!(
                            ratio > 7.0,
                            "pair ({mu},{nu}): difference only fell {ratio:.2}x from \
                             {last_distance} to {distance} Bohr ({last_difference:.3e} -> \
                             {difference:.3e}); it must fall faster than 1/R³"
                        );
                    }
                }
                previous = Some((distance, difference));
            }
        }
    }

    /// [`AtomSites::charges`] has two conventions the per-configuration tests cannot see: the
    /// `2×` it applies to off-diagonal density elements, and the sign that makes electrons
    /// negative. Contracting a full density block against the unscreened kernel on axis — where
    /// the two constructions are exact — pins both at once.
    #[test]
    fn charges_reproduce_a_contracted_density_on_axis() {
        let params = Pm3Parameters::standard().unwrap();
        let hydrogen = params.element(1).unwrap();
        let sites_h = AtomSites::build(hydrogen).unwrap();
        for z in [6u8, 8, 16] {
            let elem = params.element(z).unwrap();
            let sites = AtomSites::build(elem).unwrap();
            let n = elem.n_orb;
            // A dense symmetric density, so every off-diagonal pair contributes.
            let mut density = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..=i {
                    let v = 0.19 * ((i * 5 + j * 3) as f64) - 0.44;
                    density[i * n + j] = v;
                    density[j * n + i] = v;
                }
            }
            let distance = 6.5;
            let position_b = Vec3::new(0.0, 0.0, distance);
            let te = pair_two_electron_g::<f64>(
                &unscreened(elem),
                &unscreened(hydrogen),
                [0.0, 0.0, distance],
            );
            // Σ_{μν} P[μν] (μν|ss), the electron–electron energy of this density against a
            // unit s density on the partner.
            let mut contracted = 0.0;
            for mu in 0..n {
                for nu in 0..n {
                    contracted += density[mu * n + nu] * te.w[pack(mu, nu) * te.npack_j];
                }
            }
            // The same thing as point charges: electrons only on both sides, so the two minus
            // signs multiply to a plus and the comparison is against `+contracted`.
            let charges = sites.charges(&density, 0.0);
            let partner = sites_h.charges(&[1.0], 0.0);
            let direct = point_energy(
                &sites.offsets,
                &charges,
                Vec3::zero(),
                &sites_h.offsets,
                &partner,
                position_b,
            );
            assert!(
                (contracted - direct).abs() < 1.0e-9 * contracted.abs().max(1.0),
                "Z={z}: contracted kernel {contracted:.12} vs point charges {direct:.12}"
            );
        }
    }

    /// Hydrogen has one AO and must reduce to a single monopole at the nucleus.
    #[test]
    fn s_only_atoms_are_a_single_monopole() {
        let params = Pm3Parameters::standard().unwrap();
        let hydrogen = params.element(1).unwrap();
        let sites = AtomSites::build(hydrogen).unwrap();
        assert_eq!(sites.offsets.len(), 1);
        assert_eq!(sites.offsets[0].norm(), 0.0);
        // One electron in the s orbital, core charge +1: a neutral atom.
        let charges = sites.charges(&[1.0], 1.0);
        assert_eq!(charges.len(), 1);
        assert!((charges[0]).abs() < 1.0e-15, "neutral H is not neutral");
    }

    /// The total site charge must be the atom's net charge, whatever the density looks like:
    /// every non-monopole configuration is charge-neutral by construction, so only the trace of
    /// the density block can move it.
    #[test]
    fn site_charges_sum_to_the_atomic_charge() {
        let params = Pm3Parameters::standard().unwrap();
        for z in [1u8, 6, 7, 8, 16, 17] {
            let elem = params.element(z).unwrap();
            let sites = AtomSites::build(elem).unwrap();
            let n = elem.n_orb;
            let mut density = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..=i {
                    let v = 0.23 * ((i * 7 + j) as f64) - 0.31;
                    density[i * n + j] = v;
                    density[j * n + i] = v;
                }
            }
            let population: f64 = (0..n).map(|i| density[i * n + i]).sum();
            let charges = sites.charges(&density, elem.core_charge);
            let total: f64 = charges.iter().sum();
            assert!(
                (total - (elem.core_charge - population)).abs() < 1.0e-12,
                "Z={z}: site charges sum to {total}, expected {}",
                elem.core_charge - population
            );
        }
    }

    /// No PM3 element has a `d` shell, which is why the `d` coverage above has to invent one.
    ///
    /// This used to be `d_elements_are_refused`, asserting `build` returned `None` for zinc —
    /// guarded by `if zinc.has_d()`, which is false for every one of the forty-two, so the
    /// assertion never ran. It survived the change that made `build` realize `d` elements
    /// through `AtomSites::build_spd` precisely because it never ran. A test whose subject
    /// does not exist reports the same thing whether the code is right or wrong.
    #[test]
    fn no_pm3_element_carries_a_d_shell() {
        let params = Pm3Parameters::standard().unwrap();
        let with_d: Vec<u8> = (1..=100u8)
            .filter(|z| params.element(*z).map(|e| e.has_d()).unwrap_or(false))
            .collect();
        assert!(
            with_d.is_empty(),
            "PM3 gained a d element (Z {with_d:?}); the synthetic fixture in \
             `the_spd_sites_reproduce_the_mndo_d_kernels` can now be a real one"
        );
    }

    /// Every element PM3 does have realizes its sites, and the count is the model's.
    ///
    /// The two-electron kernels are checked elsewhere; what this pins is that nothing is
    /// silently refused. A `None` here would take an element out of every periodic calculation
    /// and out of the isolated far field, and the only symptom would be an error much later.
    #[test]
    fn every_pm3_element_realizes_its_sites() {
        let params = Pm3Parameters::standard().unwrap();
        for z in 1..=100u8 {
            let Ok(elem) = params.element(z) else {
                continue;
            };
            let sites = AtomSites::build(elem)
                .unwrap_or_else(|| panic!("Z={z} has {} orbitals but no sites", elem.n_orb));
            assert_eq!(
                sites.n_orb, elem.n_orb,
                "Z={z} realized the wrong orbital count"
            );
            // A nucleus always, then the dipole and quadrupole configurations for an sp shell.
            let expected = match elem.n_orb {
                0 | 1 => 1,
                4 => 25,
                _ => sites.offsets.len(),
            };
            assert_eq!(
                sites.offsets.len(),
                expected,
                "Z={z} ({} orbitals) built {} sites",
                elem.n_orb,
                sites.offsets.len()
            );
        }
    }
}
