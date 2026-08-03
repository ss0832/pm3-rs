// SPDX-License-Identifier: GPL-3.0-or-later

//! Embedded PM3 parameter tables and per-element reference data.
//!
//! All numeric data are extracted from MOPAC v23.2.5 (openmopac/mopac,
//! Apache-2.0) by `tools/extract_pm3_params.py`; every CSV carries its own
//! PROVENANCE header naming the exact Fortran source file. See
//! `THIRD_PARTY_NOTICES.md`.
//!
//! - `pm3_parameters.csv`      — per-element PM3 parameters (`parameters_for_PM3_C.F90`)
//! - `pm3_pair_parameters.csv` — diatomic core-core pairs `alpb`/`xfac` (same file)
//! - `pm3_global.csv`          — zero-filled compatibility vector (PM3 has no `v_par6`)
//! - `pm3_sparkles.csv`        — lanthanide sparkle parameters (`parameters_for_PM3_Sparkles_C.F90`)
//! - `element_data.csv`        — occupancies `ios/iop/iod`, per-shell principal quantum
//!   numbers `npq`, `main_group`, `ndelec`, experimental atomic ΔH_f, masses and core
//!   charges (`parameters_C.F90`)

/// Raw per-element PM3 parameter table.
pub const PM3_PARAM_CSV: &str = include_str!("data/pm3_parameters.csv");
/// Diatomic `alpb`/`xfac` core-core parameters.
pub const PM3_PAIR_CSV: &str = include_str!("data/pm3_pair_parameters.csv");
/// Zero-filled compatibility vector; PM3 does not define PM6's `v_par6` terms.
pub const PM3_GLOBAL_CSV: &str = include_str!("data/pm3_global.csv");
/// Lanthanide sparkle parameters.
pub const PM3_SPARKLES_CSV: &str = include_str!("data/pm3_sparkles.csv");
/// Element reference data (occupancies, shell quantum numbers, ΔH_f, masses).
pub const ELEMENT_DATA_CSV: &str = include_str!("data/element_data.csv");
/// Grimme D3 C6 reference table (`iat, jat, c6, cn_a, cn_b`).
pub const D3_C6_CSV: &str = include_str!("data/d3_c6_reference.csv");
/// Grimme D3 cutoff radii `r0ab` (packed lower triangle, Å).
pub const D3_R0AB_CSV: &str = include_str!("data/d3_r0ab.csv");
/// Grimme D3 per-element `r2r4` and `rcov`.
pub const D3_RADII_CSV: &str = include_str!("data/d3_radii.csv");

/// A parsed CSV: named columns and numeric rows (non-numeric cells become NaN,
/// the `sym` column is skipped by callers via name lookup on the raw fields).
pub struct CsvTable {
    pub header: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

impl CsvTable {
    /// Parse a comma-separated table, skipping `#` provenance/comment lines.
    pub fn parse(text: &str) -> Option<Self> {
        let mut lines = text.lines().filter(|l| {
            let t = l.trim();
            !t.is_empty() && !t.starts_with('#')
        });
        let header = lines
            .next()?
            .split(',')
            .map(|s| s.trim().to_string())
            .collect::<Vec<_>>();
        let rows = lines
            .map(|l| l.split(',').map(|s| s.trim().to_string()).collect())
            .collect();
        Some(Self { header, rows })
    }

    pub fn col(&self, name: &str) -> Option<usize> {
        self.header.iter().position(|c| c == name)
    }

    pub fn f64_at(&self, row: &[String], idx: usize) -> f64 {
        let value = row
            .get(idx)
            .unwrap_or_else(|| panic!("CSV row is missing column index {idx}"));
        if value.is_empty() {
            0.0
        } else {
            value
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("invalid floating-point CSV value `{value}`"))
        }
    }
}

/// Per-element reference data parsed from `element_data.csv` (index = Z, 1..=107).
#[derive(Clone, Copy, Debug, Default)]
pub struct ElementData {
    /// Initial s/p/d shell occupancies (`ios`, `iop`, `iod`).
    pub occ_s: f64,
    pub occ_p: f64,
    pub occ_d: f64,
    /// Principal quantum numbers of the valence s/p/d shells (`npq`).
    pub npq_s: u8,
    pub npq_p: u8,
    pub npq_d: u8,
    /// Whether the one-center integrals derive directly from `Gss…Hsp` (true)
    /// or from `zsn/zpn/zdn` Slater–Condon parameters (false, transition metals).
    pub main_group: bool,
    /// d electrons assigned to the core in MOPAC's `Eisol` bookkeeping (`ndelec`).
    pub ndelec: i32,
    /// Experimental gas-phase atomic ΔH_f (kcal/mol) (`eheat`).
    pub eheat_kcal: f64,
    /// Atomic mass (amu) (`ams`).
    pub mass: f64,
    /// Core charge = number of valence electrons (`tore`).
    pub tore: f64,
}

/// Parse `element_data.csv` into a Z-indexed table (index 0 unused).
pub fn element_data() -> Vec<ElementData> {
    let table = CsvTable::parse(ELEMENT_DATA_CSV).expect("embedded element_data.csv is valid");
    let col = |n: &str| {
        table
            .col(n)
            .unwrap_or_else(|| panic!("element_data.csv missing column {n}"))
    };
    let (c_z, c_ios, c_iop, c_iod) = (col("z"), col("ios"), col("iop"), col("iod"));
    let (c_ns, c_np, c_nd) = (col("npq_s"), col("npq_p"), col("npq_d"));
    let (c_mg, c_nde) = (col("main_group"), col("ndelec"));
    let (c_eh, c_mass, c_tore) = (col("eheat_kcal"), col("mass"), col("tore"));
    let mut out = vec![ElementData::default(); 108];
    for row in &table.rows {
        let z = table.f64_at(row, c_z) as usize;
        if z == 0 || z >= out.len() {
            continue;
        }
        out[z] = ElementData {
            occ_s: table.f64_at(row, c_ios),
            occ_p: table.f64_at(row, c_iop),
            occ_d: table.f64_at(row, c_iod),
            npq_s: table.f64_at(row, c_ns) as u8,
            npq_p: table.f64_at(row, c_np) as u8,
            npq_d: table.f64_at(row, c_nd) as u8,
            main_group: table.f64_at(row, c_mg) != 0.0,
            ndelec: table.f64_at(row, c_nde) as i32,
            eheat_kcal: table.f64_at(row, c_eh),
            mass: table.f64_at(row, c_mass),
            tore: table.f64_at(row, c_tore),
        };
    }
    out
}
