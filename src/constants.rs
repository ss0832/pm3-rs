// SPDX-License-Identifier: GPL-3.0-or-later

//! Physical constants and unit conversions for the PM3 model.
//!
//! PM3 in MOPAC (v23.2.5) uses the **2018 CODATA** fundamental constants by
//! default (`conref_C.F90` `fpcref(1,*)`, selected in `input/readmo.F90` unless
//! `OLDFPC`/`MNDOD` is requested). pm3-rs adopts the same values so that
//! energies match the MOPAC oracle digit-for-digit. Do NOT replace these with
//! newer CODATA revisions: they are model constants, part of the PM3
//! parameterization as shipped by MOPAC.
//!
//! PROVENANCE: openmopac/mopac v23.2.5 `src/conref_C.F90` (Apache-2.0).
//! See THIRD_PARTY_NOTICES.md.
//!
//! Internal unit policy: energies in **eV**,
//! distances in **Bohr** throughout the computational core. The Rust and
//! Python-native API boundaries convert to Hartree/Bohr (atomic units); the
//! ASE calculator converts to eV/Å.

/// One Hartree in eV (2018 CODATA, MOPAC `fpc(4)` = `fpcref(1,4)`).
pub const PM3_EV: f64 = 27.211386245988;

/// Bohr radius in Ångström (2018 CODATA, MOPAC `fpc(3)` = `fpcref(1,3)`).
pub const PM3_A0: f64 = 0.529177210903;

/// One eV in kcal/mol (2018 CODATA derived, MOPAC `fpc(9)` = `fpcref(1,9)`).
pub const EV_TO_KCAL: f64 = 23.060_547_830_619_03;

/// `a0 * ev` — one (elementary charge)²/Å in eV (MOPAC `fpc(2)` = `fpcref(1,2)`).
/// MOPAC's two-electron integral kernels use this product directly.
pub const A0_TIMES_EV: f64 = 14.399645478456;

pub const HARTREE_TO_EV: f64 = PM3_EV;
pub const EV_TO_HARTREE: f64 = 1.0 / PM3_EV;
pub const ANGSTROM_TO_BOHR: f64 = 1.0 / PM3_A0;
pub const BOHR_TO_ANGSTROM: f64 = PM3_A0;
pub const KCAL_TO_EV: f64 = 1.0 / EV_TO_KCAL;
pub const HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM: f64 = HARTREE_TO_EV / BOHR_TO_ANGSTROM;

/// Dipole conversion: one electron·Bohr in Debye.
/// MOPAC (`dipole.F90`) computes point-charge dipoles as `4.803 * charge * Å`;
/// the atomic-unit equivalent used here is `e·a0 → Debye`.
pub const AU_DIPOLE_TO_DEBYE: f64 = 2.541746473;

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the MOPAC 2018-CODATA model constants (extraction checkpoint).
    #[test]
    fn mopac_codata_2018_constants() {
        assert_eq!(PM3_EV, 27.211386245988);
        assert_eq!(PM3_A0, 0.529177210903);
        assert_eq!(EV_TO_KCAL, 23.060_547_830_619_03);
        // fpc(2) is the product a0*ev, tabulated separately by MOPAC.
        assert!((A0_TIMES_EV - PM3_A0 * PM3_EV).abs() < 1e-9);
    }
}
