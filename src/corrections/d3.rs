// SPDX-License-Identifier: GPL-3.0-or-later

//! Grimme D3 dispersion correction with zero damping.
//!
//! `E_disp = −Σ_{A<B} [ s6·C6_AB·f6/R⁶ + s8·C8_AB·f8/R⁸ ]`, with the
//! coordination-number-dependent `C6_AB` interpolated from the Grimme reference
//! table and zero-damping `f_n = 1/(1 + 6·(rs_n·R0_AB/R)^{α_n})`,
//! `C8 = 3·C6·√(Q_A Q_B)`. Distances/energies in atomic units internally.
//!
//! PROVENANCE: S. Grimme, J. Antony, S. Ehrlich, H. Krieg, *J. Chem. Phys.*
//! **132**, 154104 (2010). The PM3-D3 parameter set is recorded in the public
//! MOPAC 5.022mn `anad3.f`; the PM3-D3H4 set follows the published D3H4
//! implementation. Reference tables/algorithm are from MOPAC (Apache-2.0).
//! See THIRD_PARTY_NOTICES.md.

use crate::constants::{BOHR_TO_ANGSTROM, HARTREE_TO_EV};
use crate::data_tables::{CsvTable, D3_C6_CSV, D3_R0AB_CSV, D3_RADII_CSV};
use crate::dual::Scalar;
use crate::system::Molecule;
use std::collections::HashMap;
use std::sync::OnceLock;

/// D3 zero-damping parameters for a PM3 variant.
#[derive(Clone, Copy, Debug)]
pub struct D3Params {
    pub s6: f64,
    pub s8: f64,
    pub rs6: f64,
    pub rs8: f64,
    pub alp6: f64,
    pub alp8: f64,
}

impl D3Params {
    /// PM3-D3H4 dispersion: s6=1, s8=0, rs6=0.90, alpha6=22.
    pub fn pm3_d3h4() -> Self {
        let alp = 22.0;
        Self {
            s6: 1.0,
            s8: 0.0,
            rs6: 0.90,
            rs8: 1.0e-10,
            alp6: alp,
            alp8: alp + 2.0,
        }
    }

    /// Plain PM3-D3 dispersion (Grimme's hard-wired PM3 set).
    pub fn pm3_d3() -> Self {
        let alp = 14.0;
        Self {
            s6: 1.0,
            s8: 0.612,
            rs6: 1.345,
            rs8: 1.0,
            alp6: alp,
            alp8: alp + 2.0,
        }
    }
}

/// Static D3 reference data (parsed once from the embedded CSVs).
type C6Reference = (f64, f64, f64);
type C6Table = HashMap<(u8, u8), Vec<C6Reference>>;

struct D3Data {
    /// `c6[(iat,jat)]` = reference `(c6, cn_a, cn_b)` tuples, keyed with `iat ≥ jat`.
    c6: C6Table,
    /// Cutoff radii `r0ab` (Bohr), 94×94 (0-indexed by Z-1).
    r0ab: Vec<Vec<f64>>,
    /// Per-element `r2r4` (transformed: `sqrt(0.5·r2r4·sqrt(Z))`) and scaled `rcov`.
    r2r4: Vec<f64>,
    rcov: Vec<f64>,
    max_elem: usize,
}

fn data() -> &'static D3Data {
    static D: OnceLock<D3Data> = OnceLock::new();
    D.get_or_init(load_data)
}

#[allow(clippy::needless_range_loop)] // symmetric packed-table expansion is clearer by index
fn load_data() -> D3Data {
    const MAX: usize = 94;
    // C6 reference tuples.
    let t = CsvTable::parse(D3_C6_CSV).expect("d3_c6_reference.csv");
    let (ci, cj, cc, ca, cb) = (
        t.col("iat").unwrap(),
        t.col("jat").unwrap(),
        t.col("c6").unwrap(),
        t.col("cn_a").unwrap(),
        t.col("cn_b").unwrap(),
    );
    let mut c6 = C6Table::new();
    for row in &t.rows {
        let iat = t.f64_at(row, ci) as u8;
        let jat = t.f64_at(row, cj) as u8;
        let key = (iat.max(jat), iat.min(jat));
        let (c6v, cna, cnb) = (t.f64_at(row, cc), t.f64_at(row, ca), t.f64_at(row, cb));
        // Store with cn ordered to match the (max,min) key orientation.
        let entry = if iat >= jat {
            (c6v, cna, cnb)
        } else {
            (c6v, cnb, cna)
        };
        c6.entry(key).or_default().push(entry);
    }

    // r0ab packed lower triangle → symmetric matrix (Å → Bohr).
    let tr = CsvTable::parse(D3_R0AB_CSV).expect("d3_r0ab.csv");
    let cval = tr.col("value").unwrap();
    let packed: Vec<f64> = tr.rows.iter().map(|r| tr.f64_at(r, cval)).collect();
    let mut r0ab = vec![vec![0.0; MAX]; MAX];
    let mut k = 0;
    for i in 0..MAX {
        for j in 0..=i {
            let v = packed[k] / BOHR_TO_ANGSTROM; // Å → Bohr
            r0ab[i][j] = v;
            r0ab[j][i] = v;
            k += 1;
        }
    }

    // r2r4 / rcov with the DFTD3 post-load transforms.
    let td = CsvTable::parse(D3_RADII_CSV).expect("d3_radii.csv");
    let (cz, c_r2r4, c_rcov) = (
        td.col("z").unwrap(),
        td.col("r2r4").unwrap(),
        td.col("rcov").unwrap(),
    );
    let mut r2r4 = vec![0.0; MAX + 1];
    let mut rcov = vec![0.0; MAX + 1];
    for row in &td.rows {
        let z = td.f64_at(row, cz) as usize;
        if z == 0 || z > MAX {
            continue;
        }
        // r2r4(i) = sqrt(0.5 * r2r4_raw * sqrt(Z)); rcov = 4/3 * rcov_raw / a0.
        r2r4[z] = (0.5 * td.f64_at(row, c_r2r4) * (z as f64).sqrt()).sqrt();
        rcov[z] = 4.0 / 3.0 * td.f64_at(row, c_rcov) / BOHR_TO_ANGSTROM;
    }

    D3Data {
        c6,
        r0ab,
        r2r4,
        rcov,
        max_elem: MAX,
    }
}

use super::{dist2_g, dist_g};

/// Coordination numbers (DFTD3 `ncoord`, k1 = 16), generic over the scalar so the analytic
/// Hessian differentiates through the CN coupling. `pos` is the Bohr geometry.
fn coordination_numbers_g<S: Scalar>(numbers: &[u8], pos: &[[S; 3]], d: &D3Data) -> Vec<S> {
    const K1: f64 = 16.0;
    let n = numbers.len();
    let mut cn = vec![S::cst(0.0); n];
    for i in 0..n {
        let zi = numbers[i] as usize;
        if zi == 0 || zi > d.max_elem {
            continue;
        }
        for j in 0..n {
            if i == j {
                continue;
            }
            let zj = numbers[j] as usize;
            if zj == 0 || zj > d.max_elem {
                continue;
            }
            let r = dist_g(&pos[j], &pos[i]);
            let rco = d.rcov[zi] + d.rcov[zj];
            // 1 / (1 + exp(-K1·(rco/r − 1)))
            cn[i] = cn[i] + (((r.recip() * rco - 1.0) * -K1).exp() + 1.0).recip();
        }
    }
    cn
}

/// Gaussian-weighted C6 interpolation (DFTD3 `getc6`, k3 = −4), generic over the scalar.
fn getc6_g<S: Scalar>(d: &D3Data, zi: u8, zj: u8, cni: S, cnj: S) -> S {
    const K3: f64 = -4.0;
    let key = (zi.max(zj), zi.min(zj));
    let refs = match d.c6.get(&key) {
        Some(r) => r,
        None => return S::cst(0.0),
    };
    // The stored tuples are oriented for (max,min); align the CN pair to match.
    let (cn_i, cn_j) = if zi >= zj { (cni, cnj) } else { (cnj, cni) };
    let mut num = S::cst(0.0);
    let mut den = S::cst(0.0);
    for &(c6, cna, cnb) in refs {
        let da = cn_i - cna;
        let db = cn_j - cnb;
        let t = ((da * da + db * db) * K3).exp();
        num = num + t * c6;
        den = den + t;
    }
    if den.val() > 0.0 {
        num / den
    } else {
        S::cst(0.0)
    }
}

/// Element → hbsimple type index (N=1, O=2, F=3, P=4, S=5, Cl=6), else 0.
fn hbpar(z: u8) -> usize {
    match z {
        7 => 1,
        8 => 2,
        9 => 3,
        15 => 4,
        16 => 5,
        17 => 6,
        _ => 0,
    }
}

/// hbsimple donor/acceptor cutoff radii `r0ab` (Bohr), 6×6 symmetric over
/// {N, O, F, P, S, Cl} (MOPAC `dftd3_bits.F90` hbsimple).
const HB_R0AB: [[f64; 6]; 6] = [
    [
        4.955_806_193_900_96,
        4.695_213_220_497_56,
        4.513_610_383_039_64,
        5.757_050_043_259_74,
        5.490_409_844_454_58,
        5.282_162_180_366_20,
    ],
    [
        4.695_213_220_497_56,
        4.689_732_781_681_66,
        4.442_934_618_839_33,
        5.428_615_689_131_10,
        5.443_355_744_729_65,
        5.188_620_316_956_41,
    ],
    [
        4.513_610_383_039_64,
        4.442_934_618_839_33,
        4.345_613_577_462_78,
        5.227_738_052_841_99,
        5.164_621_095_120_98,
        5.079_772_512_453_76,
    ],
    [
        5.757_050_043_259_74,
        5.428_615_689_131_10,
        5.227_738_052_841_99,
        6.617_253_213_921_16,
        6.270_110_847_547_18,
        6.031_249_499_089_98,
    ],
    [
        5.490_409_844_454_58,
        5.443_355_744_729_65,
        5.164_621_095_120_98,
        6.270_110_847_547_18,
        6.256_315_586_440_32,
        5.956_982_885_057_16,
    ],
    [
        5.282_162_180_366_20,
        5.188_620_316_956_41,
        5.079_772_512_453_76,
        6.031_249_499_089_98,
        5.956_982_885_057_16,
        5.866_843_092_799_86,
    ],
];

/// Per-hydrogen-bond energy `eabh` (Hartree), MOPAC `dftd3_bits.F90`, generic over the scalar.
fn eabh_g<S: Scalar>(pos: &[[S; 3]], a: usize, b: usize, h: usize, shortcut: f64, cab: f64) -> S {
    const LONGCUT: f64 = 7.50388; // 5 Å in Bohr
    const ALP: f64 = 20.0;
    let (pa, pb, ph) = (&pos[a], &pos[b], &pos[h]);
    let rab2 = dist2_g(pa, pb);
    let rab = rab2.sqrt();
    let d2ij = dist2_g(pa, ph);
    let d2jk = dist2_g(ph, pb);
    let d2ik = rab2;
    let xy = (d2ij * d2jk + 1.0e-14).sqrt();
    let cosabh = (d2ij + d2jk - d2ik) * 0.5 / xy;
    let aterm = (cosabh - 1.0).powi(4) * 0.125;
    let xm = (pa[0] + pb[0]) * 0.5;
    let ym = (pa[1] + pb[1]) * 0.5;
    let zm = (pa[2] + pb[2]) * 0.5;
    let rhm = ((ph[0] - xm).powi(2) + (ph[1] - ym).powi(2) + (ph[2] - zm).powi(2)).sqrt();
    // sigmoids: 1/(1+exp(-ALP·(x/y − 1))); dampm/dampl use 1 − sigmoid.
    let sig = |ratio: S| (((ratio - 1.0) * (-ALP)).exp() + 1.0).recip();
    let dampm = S::cst(1.0) - sig(rhm / rab);
    let damps = sig(rab / shortcut);
    let dampl = S::cst(1.0) - sig(rab / LONGCUT);
    -(dampl * damps * dampm * aterm * cab) / (rab2 * rab2)
}

/// Legacy MOPAC `hbsimple` helper (eV).
///
/// This is retained as a low-level utility for callers porting older workflows,
/// but it is not part of the PM3-D3, PM3-D3H4, or PM3-D3H4X variants.
pub fn hbsimple_energy(mol: &Molecule) -> f64 {
    let (numbers, pos) = super::geometry_f64(mol);
    hbsimple_energy_g::<f64>(&numbers, &pos)
}

/// Generic (over the scalar) hbsimple hydrogen-bond energy (eV).
pub fn hbsimple_energy_g<S: Scalar>(numbers: &[u8], pos: &[[S; 3]]) -> S {
    const HBSCALE: f64 = 1.301;
    const THR: f64 = 250.0; // Bohr²
    let n = numbers.len();
    let mut energy = S::cst(0.0);
    for i in 0..n {
        let ta = hbpar(numbers[i]);
        if ta == 0 {
            continue;
        }
        for j in (i + 1)..n {
            let tb = hbpar(numbers[j]);
            if tb == 0 {
                continue;
            }
            if dist2_g(&pos[i], &pos[j]).val() >= THR {
                continue;
            }
            let shortcut = HB_R0AB[ta - 1][tb - 1];
            let cab = HBSCALE; // 0.5*(scalehb+scalehb), all equal
            for (k, &z) in numbers.iter().enumerate() {
                if z != 1 {
                    continue;
                }
                energy = energy + eabh_g(pos, i, j, k, shortcut, cab);
            }
        }
    }
    energy * HARTREE_TO_EV
}

/// D3 dispersion energy (eV) for a molecule (f64 entry point).
pub fn d3_energy(mol: &Molecule, p: &D3Params) -> f64 {
    let (numbers, pos) = super::geometry_f64(mol);
    d3_energy_g::<f64>(&numbers, &pos, p)
}

/// D3 dispersion energy (eV), generic over the scalar. `pos` is the Bohr geometry; seeding it
/// with [`crate::dual2::Dual2`] gives the exact second derivatives (the CN coupling included,
/// since the coordination numbers are computed in the same generic arithmetic).
pub fn d3_energy_g<S: Scalar>(numbers: &[u8], pos: &[[S; 3]], p: &D3Params) -> S {
    let d = data();
    let cn = coordination_numbers_g(numbers, pos, d);
    let n = numbers.len();
    let mut e6 = S::cst(0.0);
    let mut e8 = S::cst(0.0);
    for i in 0..n.saturating_sub(1) {
        let zi = numbers[i];
        if zi == 0 || zi as usize > d.max_elem {
            continue;
        }
        for j in i + 1..n {
            let zj = numbers[j];
            if zj == 0 || zj as usize > d.max_elem {
                continue;
            }
            let r = dist_g(&pos[j], &pos[i]);
            let r2 = r * r;
            let r6 = r2 * r2 * r2;
            let r8 = r6 * r2;
            let c6 = getc6_g(d, zi, zj, cn[i], cn[j]);
            let c8 = c6 * (3.0 * d.r2r4[zi as usize] * d.r2r4[zj as usize]);
            let rr = r.recip() * d.r0ab[zi as usize - 1][zj as usize - 1];
            // 1 / (1 + 6·(rs·rr)^alp)
            let damp6 = ((rr * p.rs6).powf(p.alp6) * 6.0 + 1.0).recip();
            let damp8 = ((rr * p.rs8).powf(p.alp8) * 6.0 + 1.0).recip();
            e6 = e6 + c6 * damp6 / r6;
            e8 = e8 + c8 * damp8 / r8;
        }
    }
    // Dispersion is attractive; energy in Hartree → eV.
    -(e6 * p.s6 + e8 * p.s8) * HARTREE_TO_EV
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn d3_data_loads() {
        let d = data();
        assert!(!d.c6.is_empty());
        // C-C reference exists.
        assert!(d.c6.contains_key(&(6, 6)));
        assert!(d.r2r4[6] > 0.0 && d.rcov[6] > 0.0);
        assert!(d.r0ab[5][5] > 0.0);
    }

    #[test]
    fn pm3_parameter_sets_match_reference_implementations() {
        let d3 = D3Params::pm3_d3();
        assert_eq!((d3.s6, d3.s8, d3.rs6, d3.rs8), (1.0, 0.612, 1.345, 1.0));
        assert_eq!((d3.alp6, d3.alp8), (14.0, 16.0));

        let d3h4 = D3Params::pm3_d3h4();
        assert_eq!((d3h4.s6, d3h4.s8, d3h4.rs6), (1.0, 0.0, 0.90));
        assert_eq!((d3h4.alp6, d3h4.alp8), (22.0, 24.0));
    }

    #[test]
    fn c6_is_cn_interpolated() {
        // Each element carries several coordination-number reference points, so
        // getc6 must decrease as the atoms become more coordinated (less
        // polarizable). A single-reference table (the earlier extraction bug)
        // would return a constant, so these monotonicity checks guard it.
        let d = data();
        // Multiple references per pair are present (H at CN 0 and ~0.91; O 0..2).
        assert!(d.c6[&(1, 1)].len() >= 2, "H-H should have >1 CN reference");
        assert!(
            d.c6[&(8, 8)].len() >= 5,
            "O-O should have several CN references"
        );
        // Isolated (CN 0) is more polarizable → larger C6 than bonded.
        let h_iso = getc6_g::<f64>(d, 1, 1, 0.0, 0.0);
        let h_bond = getc6_g::<f64>(d, 1, 1, 1.0, 1.0);
        assert!(h_iso > 7.0 && h_iso < 7.6, "isolated H C6 {h_iso}");
        assert!(h_bond > 2.8 && h_bond < 3.3, "bonded H C6 {h_bond}");
        assert!(h_iso > h_bond);
        let o_iso = getc6_g::<f64>(d, 8, 8, 0.0, 0.0);
        let o_bond = getc6_g::<f64>(d, 8, 8, 2.0, 2.0);
        assert!(o_iso > 15.0 && o_iso < 15.6, "isolated O C6 {o_iso}");
        assert!(o_bond > 10.0 && o_bond < 11.0, "bonded O C6 {o_bond}");
        assert!(o_iso > o_bond);
    }

    #[test]
    fn water_dimer_d3_components_are_sane() {
        // Sanity: coordination numbers and C6 are physical for a water dimer.
        let d = data();
        let mol = Molecule::from_xyz_str(
            "6\nwd\nO -1.551007 -0.114520 0.0\nH -1.934259 0.762503 0.0\nH -0.599677 0.040712 0.0\nO 1.350625 0.111469 0.0\nH 1.680398 -0.373741 -0.758561\nH 1.680398 -0.373741 0.758561\n",
            0.0,
        )
        .unwrap();
        let (numbers, pos) = super::super::geometry_f64(&mol);
        let cn = coordination_numbers_g::<f64>(&numbers, &pos, d);
        assert!((cn[0] - 2.0).abs() < 0.1, "O coordination number {}", cn[0]);
        assert!((cn[1] - 1.0).abs() < 0.1, "H coordination number {}", cn[1]);
        assert!(
            getc6_g::<f64>(d, 8, 8, cn[0], cn[3]) > 10.0,
            "C6(O,O) too small"
        );
        assert!(d3_energy(&mol, &D3Params::pm3_d3()) < 0.0);
    }

    #[test]
    fn methane_dispersion_is_small_and_negative() {
        let mol = Molecule::from_xyz_str(
            "5\nch4\nC 0 0 0\nH 0.629 0.629 0.629\nH -0.629 -0.629 0.629\nH -0.629 0.629 -0.629\nH 0.629 -0.629 -0.629\n",
            0.0,
        )
        .unwrap();
        let e = d3_energy(&mol, &D3Params::pm3_d3h4());
        assert!(e < 0.0 && e > -1.0, "CH4 D3 energy {e} eV");
    }
}
