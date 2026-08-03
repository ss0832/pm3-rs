// SPDX-License-Identifier: GPL-3.0-or-later

//! PM3 element/pair parameters and the derived NDDO multipole quantities.
//!
//! Parses the embedded PM3 parameter tables (extracted from MOPAC v23.2.5,
//! see [`crate::data_tables`]) and, per element, precomputes the
//! dipole/quadrupole charge separations `dd`/`qq` and the Klopman–Ohno
//! additive terms `rho0/rho1/rho2` used by the two-center two-electron
//! integrals. The closed forms and the `rho1`/`rho2` secant solves follow
//! MOPAC `calpar.F90`; the d-shell multipole extension (MNDO-d `ddpo`/`poij`)
//! is added by the d-orbital milestone.

use crate::constants::{EV_TO_KCAL, PM3_EV};
use crate::data_tables::{self, CsvTable};
use crate::error::{Pm3Error, Result};
use std::collections::HashMap;

/// Per-element PM3 parameters plus derived NDDO quantities.
#[derive(Clone, Debug)]
pub struct Pm3Element {
    pub z: u8,
    /// Principal quantum number of the valence s shell (`npq_s`); the sp
    /// overlap fast path uses this as "the" quantum number, which is exact for
    /// every element with `npq_s == npq_p` (all sp-gate elements).
    pub n: u8,
    pub n_s: u8,
    pub n_p: u8,
    pub n_d: u8,
    pub u_ss: f64,
    pub u_pp: f64,
    pub u_dd: f64,
    pub zeta_s: f64,
    pub zeta_p: f64,
    pub zeta_d: f64,
    pub beta_s: f64,
    pub beta_p: f64,
    pub beta_d: f64,
    pub g_ss: f64,
    pub g_sp: f64,
    pub g_pp: f64,
    pub g_p2: f64,
    pub h_sp: f64,
    /// Internal one-center exponents for transition metals (`zsn/zpn/zdn`).
    pub zsn: f64,
    pub zpn: f64,
    pub zdn: f64,
    /// Explicit Slater–Condon overrides for d elements.
    pub f0sd: f64,
    pub g2sd: f64,
    /// Per-element core-core exponent `alp6` (only some elements; PM3 uses
    /// pairwise `alpb`/`xfac` for parametrized pairs).
    pub alpha: f64,
    /// Per-element core additive term `poc` (`poc_6`, MOPAC `pocord`): overrides
    /// the core monopole `po(9)` when nonzero. Only a few elements (e.g. Sc, Fe,
    /// Ni) define it; used in electron–core attraction and core–core repulsion.
    pub poc: f64,
    /// PM3 per-element core-core Gaussian corrections `(K, L, M)` from
    /// `gues61/62/63` (nonzero triples only; at most 2 for PM3).
    pub gauss: Vec<(f64, f64, f64)>,
    /// Number of valence AOs: 1 (s), 4 (s,p) or 9 (s,p,d).
    pub n_orb: usize,
    /// Core charge (valence-electron count, MOPAC `tore`).
    pub core_charge: f64,
    /// Initial shell occupancies (`ios`/`iop`/`iod`), used by the SAD guess
    /// and the isolated-atom energy.
    pub occ_s: f64,
    pub occ_p: f64,
    pub occ_d: f64,
    /// MOPAC `main_group` flag: one-center integrals from `Gss…Hsp` directly.
    pub main_group: bool,
    /// MOPAC `ndelec`: d electrons folded into the core in `Eisol` bookkeeping.
    pub ndelec: i32,
    /// Experimental atomic heat of formation (eV).
    pub eheat_ev: f64,
    /// Isolated-atom electronic energy (eV).
    pub e_isol: f64,
    /// Atomic mass (amu).
    pub mass: f64,
    // Derived NDDO multipole terms (Bohr).
    pub dd: f64,
    pub qq: f64,
    pub rho0: f64,
    pub rho1: f64,
    pub rho2: f64,
    /// MNDO-d charge separations `ddp(1..=6)` (index 0 unused; 2=sp dipole,
    /// 3=pp quadrupole, 4=sd quadrupole, 5=pd dipole, 6=dd quadrupole), Bohr.
    pub ddp: [f64; 7],
    /// MNDO-d additive Klopman terms `po(1..=9)` (1=ss, 2=sp, 3=pp, 4=sd, 5=pd,
    /// 6=dd, 7=pp-monopole, 8=dd-monopole, 9=core), Bohr.
    pub po: [f64; 10],
    /// One-center spd two-electron integrals (d elements only).
    pub onecenter: Option<crate::onecenter::OneCenterSpd>,
}

impl Pm3Element {
    pub fn has_p(&self) -> bool {
        self.n_orb >= 4
    }
    pub fn has_d(&self) -> bool {
        self.n_orb >= 9
    }
}

/// Pairwise PM3 core-core parameters (`alpb` in Å⁻¹-like units, unitless `xfac`).
#[derive(Clone, Copy, Debug)]
pub struct PairParams {
    pub alpha: f64,
    pub x: f64,
}

/// Lanthanide sparkle: a parametrized point core with no orbitals.
#[derive(Clone, Debug)]
pub struct SparkleElement {
    pub z: u8,
    pub core_charge: f64,
    pub g_ss: f64,
    pub alpha: f64,
    pub gauss: Vec<(f64, f64, f64)>,
    pub eheat_ev: f64,
    pub mass: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Pm3Parameters {
    pub elements: HashMap<u8, Pm3Element>,
    pub sparkles: HashMap<u8, SparkleElement>,
    /// Symmetric key `(max(zi,zj), min(zi,zj))`.
    pub pair: HashMap<(u8, u8), PairParams>,
    /// MOPAC `v_par6(1..60)` global parameters (index 0 unused).
    pub global: Vec<f64>,
}

impl Pm3Parameters {
    /// Load the standard embedded PM3 parameter set.
    pub fn standard() -> Result<Self> {
        Self::from_embedded()
    }

    pub fn element(&self, z: u8) -> Result<&Pm3Element> {
        self.elements.get(&z).ok_or(Pm3Error::MissingElement(z))
    }

    /// Pairwise `alpb`/`xfac`; error if MOPAC has no parameters for the pair.
    pub fn pair(&self, zi: u8, zj: u8) -> Result<&PairParams> {
        let key = (zi.max(zj), zi.min(zj));
        self.pair.get(&key).ok_or_else(|| {
            Pm3Error::MissingParameter(format!("PM3 diatomic parameters for Z={zi},Z={zj}"))
        })
    }

    pub fn v_par(&self, index: usize) -> f64 {
        self.global.get(index).copied().unwrap_or(0.0)
    }

    fn from_embedded() -> Result<Self> {
        let edata = data_tables::element_data();

        // --- per-element parameters ---
        let t = CsvTable::parse(data_tables::PM3_PARAM_CSV)
            .ok_or_else(|| Pm3Error::InvalidInput("empty pm3_parameters.csv".to_string()))?;
        let col = |n: &str| -> Result<usize> {
            t.col(n)
                .ok_or_else(|| Pm3Error::MissingParameter(format!("pm3_parameters.csv column {n}")))
        };
        let c_z = col("z")?;
        let names = [
            "uss", "upp", "udd", "zs", "zp", "zd", "betas", "betap", "betad", "gss", "gsp", "gpp",
            "gp2", "hsp", "zsn", "zpn", "zdn", "f0sd", "g2sd", "alp", "poc",
        ];
        let mut ci = HashMap::new();
        for n in names {
            ci.insert(n, col(n)?);
        }
        let gcols = [
            (col("g1_k")?, col("g1_l")?, col("g1_m")?),
            (col("g2_k")?, col("g2_l")?, col("g2_m")?),
            (col("g3_k")?, col("g3_l")?, col("g3_m")?),
            (col("g4_k")?, col("g4_l")?, col("g4_m")?),
        ];

        let mut elements = HashMap::new();
        for row in &t.rows {
            let z = t.f64_at(row, c_z) as u8;
            let get = |n: &str| t.f64_at(row, ci[n]);
            let (u_ss, beta_s) = (get("uss"), get("betas"));
            let point_charge = matches!(z, 104 | 106);
            if u_ss == 0.0 && beta_s == 0.0 && !point_charge {
                continue;
            }
            if z as usize >= edata.len() || z == 0 {
                continue;
            }
            let ed = edata[z as usize];
            let (zeta_p, zeta_d) = (get("zp"), get("zd"));
            let n_orb = if point_charge {
                0
            } else if zeta_d > 0.0 {
                9
            } else if zeta_p > 0.0 {
                4
            } else {
                1
            };
            let mut gauss = Vec::new();
            for (kk, ll, mm) in gcols {
                let (k, l, m) = (t.f64_at(row, kk), t.f64_at(row, ll), t.f64_at(row, mm));
                if k != 0.0 || l != 0.0 {
                    gauss.push((k, l, m));
                }
            }

            let n = ed.npq_s;
            let zeta_s = get("zs");
            let (mut g_ss, mut g_sp, mut g_pp, mut g_p2) =
                (get("gss"), get("gsp"), get("gpp"), get("gp2"));
            // MOPAC calpar floors: zeta_p >= 0.3, hsp >= 1e-7, hpp >= 0.1.
            let mut h_sp = get("hsp").max(1.0e-7);
            // Transition metals (main_group = false): the one-center
            // Coulomb/exchange integrals are NOT the fitted values but are
            // recomputed from the internal exponents `zsn`/`zpn` via the
            // Slater–Condon radial integrals (MOPAC `sp_two_electron`). This is
            // essential — using the fitted values makes the d shell far too
            // deep and the SCF collapses to an unphysical charge-transfer state.
            let (zsn, zpn) = (get("zsn"), get("zpn"));
            if !ed.main_group && zsn > 1.0e-4 && zpn > 1.0e-4 {
                use crate::onecenter::slater_rsc;
                let ns = ed.npq_s as i32;
                let np = ed.npq_p as i32;
                g_ss = slater_rsc(0, ns, zsn, ns, zsn, ns, zsn, ns, zsn);
                g_sp = slater_rsc(0, ns, zsn, ns, zsn, np, zpn, np, zpn);
                h_sp = (slater_rsc(1, ns, zsn, np, zpn, ns, zsn, np, zpn) / 3.0).max(1.0e-7);
                let r033 = slater_rsc(0, np, zpn, np, zpn, np, zpn, np, zpn);
                let r233 = slater_rsc(2, np, zpn, np, zpn, np, zpn, np, zpn);
                g_pp = r033 + 0.16 * r233;
                g_p2 = r033 - 0.08 * r233;
            }

            let (dd, qq, rho1, rho2) = if z == 102 {
                // `calpar` derives dipole/quadrupole terms only through Z=97.
                // Cb then receives am=1e-10, while ad/aq/dd/qq stay zero.  The
                // resulting electron-electron multipole interactions vanish;
                // the distinct pre-override `po` values used for electron-core
                // attraction are installed by `derive_multipoles` below.
                (0.0, 0.0, 0.5e10, 0.5e10)
            } else if n_orb >= 4 && zeta_p > 0.0 {
                let zp = zeta_p.max(0.3);
                // MOPAC uses the periodic-table row PQN (`nspqn`) for the multipole charge
                // separations, not the per-element Slater `n` (see `multipole_pqn`).
                let qn = multipole_pqn(z);
                let dd = dd_charge_sep(qn, zeta_s, zp);
                let qq = qq_charge_sep(qn, zp);
                let hpp = (0.5 * (g_pp - g_p2)).max(0.1);
                let rho1 = additive_rho1(h_sp, dd);
                let rho2 = additive_rho2(hpp, qq);
                (dd, qq, rho1, rho2)
            } else {
                (0.0, 0.0, 0.0, 0.0)
            };
            let rho0 = match z {
                // Point charges with Gss=0 receive the generic am=1 fallback.
                // Cb's electron-electron `am` is the post-`inid` safeguard.
                102 => 0.5e10,
                104 | 106 => 0.5,
                _ if g_ss > 0.0 => 0.5 * PM3_EV / g_ss,
                _ => 0.0,
            };

            // Isolated-atom electronic energy: average-of-configuration
            // coefficients in the MOPAC shell occupancies (`calpar.F90:75-113`).
            // Exact for main-group elements; the d-block adds `eiscor` terms in
            // the d milestone. Hydrogen is special-cased (eisol = uss).
            let (ios, iop) = (ed.occ_s, ed.occ_p);
            let e_isol = if point_charge || z == 102 {
                // `calpar` forms isolated-atom reference energies only for
                // Z=2..97 (plus the explicit H case). Cb therefore retains
                // its initialized zero reference energy in MOPAC.
                0.0
            } else if z == 1 {
                u_ss
            } else {
                u_ss * ios
                    + get("upp") * iop
                    + get("udd") * ed.occ_d
                    + g_ss * gssc(ios)
                    + g_pp * gppc(iop)
                    + g_sp * gspc(ios, iop)
                    + g_p2 * gp2c(iop)
                    + h_sp * hspc(ios, iop)
            };

            let mut elem = Pm3Element {
                z,
                n,
                n_s: ed.npq_s,
                n_p: ed.npq_p,
                n_d: ed.npq_d,
                u_ss,
                u_pp: get("upp"),
                u_dd: get("udd"),
                zeta_s,
                zeta_p,
                zeta_d,
                beta_s,
                beta_p: get("betap"),
                beta_d: get("betad"),
                g_ss,
                g_sp,
                g_pp,
                g_p2,
                h_sp,
                zsn: get("zsn"),
                zpn: get("zpn"),
                zdn: get("zdn"),
                f0sd: get("f0sd"),
                g2sd: get("g2sd"),
                alpha: get("alp"),
                poc: get("poc"),
                gauss,
                n_orb,
                core_charge: ed.tore,
                occ_s: if point_charge { 0.0 } else { ed.occ_s },
                occ_p: if point_charge { 0.0 } else { ed.occ_p },
                occ_d: if point_charge { 0.0 } else { ed.occ_d },
                main_group: ed.main_group,
                ndelec: ed.ndelec,
                eheat_ev: ed.eheat_kcal / EV_TO_KCAL,
                e_isol,
                mass: ed.mass,
                dd,
                qq,
                rho0,
                rho1,
                rho2,
                ddp: [0.0; 7],
                po: [0.0; 10],
                onecenter: None,
            };
            // Derived MNDO-d multipole terms. For a d element the one-center
            // spd integrals are built first (they feed the d-multipole solver
            // and add the d-shell isolated-atom energy); the sp additive terms
            // then follow MOPAC `inid`'s main-group overwrite.
            derive_multipoles(&mut elem);
            elements.insert(z, elem);
        }
        if elements.is_empty() {
            return Err(Pm3Error::InvalidInput(
                "no PM3 elements parsed from parameter table".to_string(),
            ));
        }

        // --- pairwise alpb/xfac ---
        let tp = CsvTable::parse(data_tables::PM3_PAIR_CSV)
            .ok_or_else(|| Pm3Error::InvalidInput("empty pm3_pair_parameters.csv".to_string()))?;
        let (c_zi, c_zj, c_a, c_x) = (
            tp.col("zi")
                .ok_or_else(|| Pm3Error::MissingParameter("zi".into()))?,
            tp.col("zj")
                .ok_or_else(|| Pm3Error::MissingParameter("zj".into()))?,
            tp.col("alpb")
                .ok_or_else(|| Pm3Error::MissingParameter("alpb".into()))?,
            tp.col("xfac")
                .ok_or_else(|| Pm3Error::MissingParameter("xfac".into()))?,
        );
        let mut pair = HashMap::new();
        for row in &tp.rows {
            let zi = tp.f64_at(row, c_zi) as u8;
            let zj = tp.f64_at(row, c_zj) as u8;
            let key = (zi.max(zj), zi.min(zj));
            pair.insert(
                key,
                PairParams {
                    alpha: tp.f64_at(row, c_a),
                    x: tp.f64_at(row, c_x),
                },
            );
        }

        // --- global v_par vector ---
        let tg = CsvTable::parse(data_tables::PM3_GLOBAL_CSV)
            .ok_or_else(|| Pm3Error::InvalidInput("empty pm3_global.csv".to_string()))?;
        let (c_i, c_v) = (
            tg.col("index")
                .ok_or_else(|| Pm3Error::MissingParameter("index".into()))?,
            tg.col("value")
                .ok_or_else(|| Pm3Error::MissingParameter("value".into()))?,
        );
        let mut global = vec![0.0; 61];
        for row in &tg.rows {
            let i = tg.f64_at(row, c_i) as usize;
            if i > 0 && i < global.len() {
                global[i] = tg.f64_at(row, c_v);
            }
        }

        // --- lanthanide sparkles: 3+ point cores with no valence orbitals.
        // Represented as zero-orbital elements so the electron–core attraction
        // and core–core repulsion flow through the standard machinery. PM3 has
        // no orbital parameterization for any lanthanide, so all 15 of La-Lu
        // come from the Sparkle table (MOPAC `parameters_for_PM3_Sparkles_C`).
        let sparkles = parse_sparkles(data_tables::PM3_SPARKLES_CSV, &edata);
        for (&z, sp) in &sparkles {
            let rho0 = if sp.g_ss > 0.0 {
                0.5 * PM3_EV / sp.g_ss
            } else {
                0.0
            };
            // The sparkle heat of formation is the gaseous 3+ ion's ΔHf
            // (atomic ΔHf + first three ionization energies). MOPAC bakes this
            // into its sparkle bookkeeping; it is not in the parameter file, so
            // it is taken from isolated-Ln³⁺ MOPAC reference runs.
            let eheat_ev = sp.eheat_ev;
            let mut po = [0.0; 10];
            po[1] = rho0;
            po[7] = rho0;
            po[8] = rho0;
            po[9] = rho0;
            elements.insert(
                z,
                Pm3Element {
                    z,
                    n: 0,
                    n_s: 0,
                    n_p: 0,
                    n_d: 0,
                    u_ss: 0.0,
                    u_pp: 0.0,
                    u_dd: 0.0,
                    zeta_s: 0.0,
                    zeta_p: 0.0,
                    zeta_d: 0.0,
                    beta_s: 0.0,
                    beta_p: 0.0,
                    beta_d: 0.0,
                    g_ss: sp.g_ss,
                    g_sp: 0.0,
                    g_pp: 0.0,
                    g_p2: 0.0,
                    h_sp: 0.0,
                    zsn: 0.0,
                    zpn: 0.0,
                    zdn: 0.0,
                    f0sd: 0.0,
                    g2sd: 0.0,
                    alpha: sp.alpha,
                    poc: 0.0,
                    gauss: sp.gauss.clone(),
                    n_orb: 0,
                    core_charge: sp.core_charge,
                    occ_s: 0.0,
                    occ_p: 0.0,
                    occ_d: 0.0,
                    main_group: true,
                    ndelec: 0,
                    eheat_ev,
                    e_isol: 0.0,
                    mass: sp.mass,
                    dd: 0.0,
                    qq: 0.0,
                    rho0,
                    rho1: 0.0,
                    rho2: 0.0,
                    ddp: [0.0; 7],
                    po,
                    onecenter: None,
                },
            );
        }

        Ok(Self {
            elements,
            sparkles,
            pair,
            global,
        })
    }
}

/// Gaseous Ln³⁺ heat of formation (kcal/mol) for the PM3 sparkle model,
/// oracle-derived from isolated-ion MOPAC v23.2.5 runs (`PM3 CHARGE=3` on a
/// single lanthanide). Used as the sparkle's ΔH_f offset.
fn parse_sparkles(text: &str, edata: &[data_tables::ElementData]) -> HashMap<u8, SparkleElement> {
    let mut out = HashMap::new();
    let Some(t) = CsvTable::parse(text) else {
        return out;
    };
    let Some(c_z) = t.col("z") else { return out };
    let c_gss = t.col("gss");
    let c_alp = t.col("alp");
    let c_eheat = t.col("eheat_kcal");
    let gcols: Vec<(Option<usize>, Option<usize>, Option<usize>)> = (1..=4)
        .map(|i| {
            (
                t.col(&format!("g{i}_k")),
                t.col(&format!("g{i}_l")),
                t.col(&format!("g{i}_m")),
            )
        })
        .collect();
    for row in &t.rows {
        let z = t.f64_at(row, c_z) as u8;
        if z == 0 || z as usize >= edata.len() {
            continue;
        }
        let ed = edata[z as usize];
        let mut gauss = Vec::new();
        for &(k, l, m) in &gcols {
            if let (Some(k), Some(l), Some(m)) = (k, l, m) {
                let (kv, lv, mv) = (t.f64_at(row, k), t.f64_at(row, l), t.f64_at(row, m));
                if kv != 0.0 || lv != 0.0 {
                    gauss.push((kv, lv, mv));
                }
            }
        }
        out.insert(
            z,
            SparkleElement {
                z,
                core_charge: 3.0,
                g_ss: c_gss.map(|c| t.f64_at(row, c)).unwrap_or(0.0),
                alpha: c_alp.map(|c| t.f64_at(row, c)).unwrap_or(0.0),
                gauss,
                eheat_ev: c_eheat
                    .map(|column| t.f64_at(row, column) / EV_TO_KCAL)
                    .unwrap_or(ed.eheat_kcal / EV_TO_KCAL),
                mass: ed.mass,
            },
        );
    }
    out
}

// Average-of-configuration coefficients for the isolated-atom electronic
// energy, in the integer shell occupancies `ios`/`iop` (MOPAC `calpar.F90:75-97`,
// transcribed exactly so the transition-metal path stays correct):
//   gssc = max(ios-1, 0)
//   gspc = ios*iop
//   l    = min(iop, 6-iop)
//   gp2c = (iop*(iop-1))/2 [integer] + (l*(l-1))/4
//   gppc = -(l*(l-1))/4
//   hspc = -iop*ios/2

fn gssc(ios: f64) -> f64 {
    (ios - 1.0).max(0.0)
}
fn gspc(ios: f64, iop: f64) -> f64 {
    ios * iop
}
fn hspc(ios: f64, iop: f64) -> f64 {
    -iop * ios * 0.5
}
fn gp2c(iop: f64) -> f64 {
    let k = iop as i64;
    let l = k.min(6 - k);
    ((k * (k - 1)) / 2) as f64 + (l * (l - 1)) as f64 / 4.0
}
fn gppc(iop: f64) -> f64 {
    let k = iop as i64;
    let l = k.min(6 - k);
    -((l * (l - 1)) as f64) / 4.0
}

/// Multipole principal quantum number `nspqn` used by the two-center multipole charge
/// separations (MOPAC `calpar.F90:44`: `data nspqn/2*1, 8*2, 8*3, 18*4, 18*5, 32*6, 21*0/`).
///
/// This is the **periodic-table row** number, NOT the per-element Slater principal quantum
/// number `n`: PM3 assigns some elements a Slater `n` one higher than their row (e.g. Ne uses a
/// 3s Slater orbital, Ar a 4s), but the `dd`/`qq` multipole formulas always use the row number.
/// For most main-group elements the two coincide (C/N/O = 2), so only the outliers (chiefly the
/// noble gases) were affected.
pub fn multipole_pqn(z: u8) -> f64 {
    match z {
        1..=2 => 1.0,
        3..=10 => 2.0,
        11..=18 => 3.0,
        19..=36 => 4.0,
        37..=54 => 5.0,
        55..=86 => 6.0,
        _ => 0.0,
    }
}

/// Dipole charge separation `dd` (Bohr). MOPAC `calpar.F90`. `qn` is the multipole principal
/// quantum number ([`multipole_pqn`]).
pub fn dd_charge_sep(qn: f64, zs: f64, zp: f64) -> f64 {
    (2.0 * qn + 1.0) * (4.0 * zs * zp).powf(qn + 0.5)
        / (zs + zp).powf(2.0 * qn + 2.0)
        / 3.0_f64.sqrt()
}

/// Quadrupole charge separation `qq` (Bohr). MOPAC `calpar.F90`. `qn` is the multipole principal
/// quantum number ([`multipole_pqn`]).
pub fn qq_charge_sep(qn: f64, zp: f64) -> f64 {
    ((4.0 * qn * qn + 6.0 * qn + 2.0) / 20.0).sqrt() / zp
}

/// Additive term `rho1` reproducing the one-center dipole integral `H_sp` (Bohr).
///
/// Solves `H_sp(au) = ½ d − ½ / √(4 D1² + 1/d²)` for `d`, returning `rho1 = 0.5/d`.
///
/// **Exactly 5 secant iterations**, mirroring MOPAC `calpar.F90:159-171` (`jmax = 5`) — the
/// solver is *not* run to convergence. For most elements it has effectively converged by 5
/// iterations, but for elements whose target integral was floored (`hpp → 0.1` when `gpp < gp2`,
/// e.g. every noble gas) the 5-iteration truncation differs from the converged root, and matching
/// MOPAC's truncation is required for bit-agreement.
pub fn additive_rho1(hsp_ev: f64, dd: f64) -> f64 {
    // MOPAC floors hsp to 1e-7 eV before the secant (calpar.F90:111).
    let hsp = hsp_ev.max(1.0e-7) / PM3_EV;
    let g = |d: f64| 0.5 * d - 0.5 / (4.0 * dd * dd + 1.0 / (d * d)).sqrt();
    let gdd1 = (hsp / (dd * dd)).powf(1.0 / 3.0);
    let (mut d1, mut d2) = (gdd1, gdd1 + 0.04);
    for _ in 0..5 {
        let df = d2 - d1;
        let (h1, h2) = (g(d1), g(d2));
        if (h2 - h1).abs() < 1.0e-25 {
            break;
        }
        let d3 = d1 + df * (hsp - h1) / (h2 - h1);
        d1 = d2;
        d2 = d3;
    }
    0.5 / d2
}

/// Additive term `rho2` reproducing the one-center quadrupole integral `H_pp` (Bohr).
///
/// Solves `H_pp(au) = ¼ q − ½/√(4 D2² + 1/q²) + ¼/√(8 D2² + 1/q²)` for `q`, returning
/// `rho2 = 0.5/q`. **Exactly 5 secant iterations** (MOPAC `calpar.F90:172-185`, `jmax = 5`);
/// see [`additive_rho1`].
pub fn additive_rho2(hpp_ev: f64, qq: f64) -> f64 {
    let hpp = hpp_ev / PM3_EV;
    let g = |q: f64| {
        0.25 * q - 0.5 / (4.0 * qq * qq + 1.0 / (q * q)).sqrt()
            + 0.25 / (8.0 * qq * qq + 1.0 / (q * q)).sqrt()
    };
    // Start point: p4 = 2^4 = 16, matching MOPAC `gqq = (p4*hpp/(ev*48*qq^4))^0.2`.
    let gqq = (16.0 * hpp / (48.0 * qq.powi(4))).powf(0.2);
    let (mut q1, mut q2) = (gqq, gqq + 0.04);
    for _ in 0..5 {
        let qf = q2 - q1;
        let (h1, h2) = (g(q1), g(q2));
        if (h2 - h1).abs() < 1.0e-25 {
            break;
        }
        let q3 = q1 + qf * (hpp - h1) / (h2 - h1);
        q1 = q2;
        q2 = q3;
    }
    0.5 / q2
}

/// Fill the MNDO-d charge separations `ddp` and additive terms `po` on an
/// element (MOPAC `aijm`/`ddpo`/`poij` + the main-group overwrite in `inid`).
/// For a d element this also builds and attaches [`crate::onecenter::OneCenterSpd`]
/// and folds its `eisol_d` into `e_isol`.
fn derive_multipoles(elem: &mut Pm3Element) {
    // One-center spd integrals (d elements): needed for the d additive-term
    // targets (`repd`) and the isolated-atom d-shell energy.
    if elem.has_d() {
        let oc = crate::onecenter::OneCenterSpd::build(elem);
        elem.e_isol += oc.eisol_d;
        // aij (unnormalized-exponent multipole normalizations, MOPAC `aijm`).
        let aij = aijm(elem);
        // po(1)/ss monopole.
        if elem.g_ss > 0.1 {
            elem.po[1] = poij(0, 1.0, elem.g_ss);
        }
        // sp dipole, pp quadrupole.
        let d_sp = aij[2] / 12.0_f64.sqrt();
        elem.ddp[2] = d_sp;
        elem.po[2] = poij(1, d_sp, elem.h_sp);
        elem.po[7] = elem.po[1];
        let d_pp = (aij[3] * 0.1).sqrt();
        elem.ddp[3] = d_pp;
        elem.po[3] = poij(2, d_pp, 0.5 * (elem.g_pp - elem.g_p2));
        // d multipoles.
        let da = (1.0_f64 / 60.0).sqrt();
        let d_sd = (aij[4] * da).sqrt();
        elem.ddp[4] = d_sd;
        elem.po[4] = poij(2, d_sd, oc.repd[19]);
        let d_pd = aij[5] / 20.0_f64.sqrt();
        elem.ddp[5] = d_pd;
        elem.po[5] = poij(1, d_pd, oc.repd[23] - 1.8 * oc.repd[35]);
        let fg_dd = 0.2 * (oc.repd[29] + 2.0 * oc.repd[30] + 2.0 * oc.repd[31]);
        elem.po[8] = if fg_dd > 1e-5 {
            poij(0, 1.0, fg_dd)
        } else {
            1e5
        };
        let d_dd = (aij[6] / 14.0).sqrt();
        elem.ddp[6] = d_dd;
        elem.po[6] = poij(2, d_dd, oc.repd[44] - (20.0 / 35.0) * oc.repd[52]);
        elem.po[9] = elem.po[1];
        elem.onecenter = Some(oc);
    } else if elem.n_orb >= 4 {
        // Main-group sp element: MOPAC `inid` overwrites po(1,2,3,7) and ddp(2,3)
        // from the calpar am/ad/aq/dd/qq path.
        elem.po[1] = elem.rho0;
        elem.po[2] = elem.rho1;
        elem.po[3] = elem.rho2;
        elem.po[7] = elem.rho0;
        elem.po[9] = elem.rho0;
        elem.ddp[2] = elem.dd;
        elem.ddp[3] = elem.qq * 2.0_f64.sqrt();
    } else {
        // Hydrogen / s-only: monopole core term only.
        elem.po[1] = elem.rho0;
        elem.po[7] = elem.rho0;
        elem.po[9] = elem.rho0;
    }
    if elem.z == 102 {
        // MOPAC calls `inid` before assigning am(102)=1e-10.  Consequently
        // Cb's electron-core monopoles retain the ordinary Gss-derived value,
        // while its dipole/quadrupole core terms remain zero.  These `po`
        // values must not be replaced by the huge electron-electron rho terms.
        let core_monopole = 0.5 * PM3_EV / elem.g_ss;
        elem.po = [0.0; 10];
        elem.po[1] = core_monopole;
        elem.po[7] = core_monopole;
        elem.po[9] = core_monopole;
        elem.ddp = [0.0; 7];
    }
    // Core additive-term override (MOPAC `inid`): `po(9) = pocord` when the
    // element defines one (Sc, Fe, Ni, …); otherwise `po(9) = po(1)`. `po(9)`
    // enters BOTH the electron–core attraction and the core–core repulsion,
    // where the two nearly cancel. Our two-center d path seeds the s/p
    // electron-core block from the `rho0`-based s/p path, so `po(9)` currently
    // reaches only the d-core terms; applying `poc` there alone breaks the
    // cancellation and worsens the (already physical) Sc/Fe/Ni energies. Until
    // the s/p electron-core seed is made `po(9)`-consistent, keep `po(9)=po(1)`
    // (self-consistent with the core-core), leaving a ~1.5 kcal/mol residual on
    // those three elements. `elem.poc` is retained for that future rework.
    let _ = elem.poc;
}

/// MOPAC `aijm`/`aijl`: multipole normalization factors from the *valence*
/// (unnormalized) Slater exponents. Returns `aij[1..=6]` (index 0 unused).
fn aijm(elem: &Pm3Element) -> [f64; 7] {
    let mut aij = [0.0f64; 7];
    let (z1, z2, z3) = (elem.zeta_s, elem.zeta_p, elem.zeta_d);
    let nsp = elem.n_s as i32;
    if elem.z < 3 || z1 * z2 < 0.01 {
        return aij;
    }
    aij[2] = aijl(z1, z2, nsp, nsp, 1);
    aij[3] = aijl(z2, z2, nsp, nsp, 2);
    if elem.has_d() {
        let nd = elem.n_d as i32;
        aij[4] = aijl(z1, z3, nsp, nd, 2);
        aij[5] = aijl(z2, z3, nsp, nd, 1);
        aij[6] = aijl(z3, z3, nd, nd, 2);
    }
    aij
}

/// MOPAC `aijl` (mndod.F90:2146). `fx(i) = (i-1)!` (1-based factorial).
fn aijl(z1: f64, z2: f64, n1: i32, n2: i32, l: i32) -> f64 {
    fn fac(n: i32) -> f64 {
        // n! as a plain factorial (n >= 0).
        (1..=n).map(|k| k as f64).product::<f64>().max(1.0)
    }
    let zz = z1 + z2 + 1e-20;
    // MOPAC uses fx(n1+n2+l+1) = (n1+n2+l)! and fx(2*n1+1)=(2*n1)!, fx(2*n2+1)=(2*n2)!.
    fac(n1 + n2 + l) / (fac(2 * n1) * fac(2 * n2)).sqrt()
        * (2.0 * z1 / zz).powi(n1)
        * (2.0 * z1 / zz).sqrt()
        * (2.0 * z2 / zz).powi(n2)
        * (2.0 * z2 / zz).sqrt()
        * 2.0_f64.powi(l)
        / zz.powi(l)
}

/// MOPAC `poij` (mndod.F90:165): additive Klopman term (Bohr) reproducing the
/// one-center multipole integral `fg` (eV). Golden-section minimization for
/// `l = 1, 2`; closed form for the monopole `l = 0`.
fn poij(l: i32, d: f64, fg: f64) -> f64 {
    if l == 0 {
        return 0.5 * PM3_EV / fg;
    }
    let ev4 = PM3_EV / 4.0;
    let ev8 = PM3_EV / 8.0;
    let dsq = d * d;
    let (mut a1, mut a2) = (0.1, 5.0);
    let (mut f1, mut f2) = (0.0, 0.0);
    for _ in 0..100 {
        let delta = a2 - a1;
        if delta < 1e-8 {
            break;
        }
        let y1 = a1 + delta * 0.382;
        let y2 = a1 + delta * 0.618;
        match l {
            1 => {
                f1 = (ev4 * (1.0 / y1 - 1.0 / (y1 * y1 + dsq).sqrt()) - fg).powi(2);
                f2 = (ev4 * (1.0 / y2 - 1.0 / (y2 * y2 + dsq).sqrt()) - fg).powi(2);
            }
            _ => {
                f1 = (ev8
                    * (1.0 / y1 - 2.0 / (y1 * y1 + dsq * 0.5).sqrt()
                        + 1.0 / (y1 * y1 + dsq).sqrt())
                    - fg)
                    .powi(2);
                f2 = (ev8
                    * (1.0 / y2 - 2.0 / (y2 * y2 + dsq * 0.5).sqrt()
                        + 1.0 / (y2 * y2 + dsq).sqrt())
                    - fg)
                    .powi(2);
            }
        }
        if f1 < f2 {
            a2 = y2;
        } else {
            a1 = y1;
        }
    }
    if f1 >= f2 {
        a2
    } else {
        a1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_core_elements() {
        let p = Pm3Parameters::standard().unwrap();
        for z in [1u8, 6, 7, 8] {
            let e = p.element(z).unwrap();
            assert!(e.rho0 > 0.0, "rho0 must be positive for Z={z}");
        }
        // Hydrogen reference values (MOPAC v23.2.5 parameters_for_PM3_C.F90).
        let h = p.element(1).unwrap();
        assert!((h.u_ss - (-13.073321)).abs() < 1e-9);
        assert!((h.beta_s - (-5.626512)).abs() < 1e-9);
        assert!((h.zeta_s - 0.967807).abs() < 1e-9);
        assert!((h.g_ss - 14.794208).abs() < 1e-9);
        assert_eq!(h.n_orb, 1);
        assert_eq!(h.dd, 0.0);
        // Carbon.
        let c = p.element(6).unwrap();
        assert!((c.u_ss - (-47.270320)).abs() < 1e-9);
        assert!(c.dd > 0.0 && c.qq > 0.0 && c.rho1 > 0.0 && c.rho2 > 0.0);
        // H-H pair parameters.
        let nah = p.pair(11, 1).unwrap();
        assert!((nah.alpha - 1.800472).abs() < 1e-9);
        assert!((nah.x - 3.171946).abs() < 1e-9);
    }

    #[test]
    fn pm3_uses_no_d_orbitals() {
        let p = Pm3Parameters::standard().unwrap();
        // Sulfur (Z=16) is a hypervalent PM3 d element (Zn/Cd/Hg are sp-only).
        let s = p.element(16).unwrap();
        assert!(!s.has_d(), "PM3 sulfur must use only s/p orbitals");
        assert_eq!(s.n_orb, 4);
        assert!(s.onecenter.is_none());
        // All additive terms and separations finite and positive.
        for i in 1..=9 {
            assert!(s.po[i].is_finite(), "po[{i}] not finite");
        }
        assert!(s.po[1] > 0.0 && s.po[2] > 0.0 && s.po[3] > 0.0);
        assert!(s.ddp[2] > 0.0 && s.ddp[3] > 0.0);
        // Core rho defaults to the ss additive term.
        assert!((s.po[9] - s.po[1]).abs() < 1e-12);
        // Isolated-atom energy is finite (d-shell eiscor folded in).
        assert!(s.e_isol.is_finite());
    }

    #[test]
    fn sp_element_multipoles_match_calpar_path() {
        let p = Pm3Parameters::standard().unwrap();
        let o = p.element(8).unwrap();
        // MOPAC inid overwrite: po(1)=rho0, po(2)=rho1, po(3)=rho2, ddp(2)=dd,
        // ddp(3)=qq*sqrt(2).
        assert!((o.po[1] - o.rho0).abs() < 1e-12);
        assert!((o.po[2] - o.rho1).abs() < 1e-12);
        assert!((o.po[3] - o.rho2).abs() < 1e-12);
        assert!((o.ddp[2] - o.dd).abs() < 1e-12);
        assert!((o.ddp[3] - o.qq * 2.0_f64.sqrt()).abs() < 1e-12);
    }
}
