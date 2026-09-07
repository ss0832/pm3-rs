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

/// A convergence tolerance on a **force**, eV/Å in, eV/Bohr out.
///
/// Named rather than written inline because writing it inline has gone wrong five times in
/// this crate, always the same way and never noisily. A force is an energy *per unit length*,
/// so it scales the **opposite** way to a length: a 1 eV/Å force does 0.529 eV of work over a
/// Bohr, so the factor is [`BOHR_TO_ANGSTROM`], not [`ANGSTROM_TO_BOHR`]. Reaching for the
/// factor whose name reads like the direction of travel gives a tolerance 3.57× *looser* than
/// the caller asked for, the optimizer stops early, and it still reports `converged: true` —
/// there is no symptom except an answer that is slightly wrong.
pub const fn force_tol_to_au(ev_per_angstrom: f64) -> f64 {
    ev_per_angstrom * BOHR_TO_ANGSTROM
}

/// A convergence tolerance on a **stress or pressure**, eV/Å³ in, eV/Bohr³ out.
///
/// Same trap as [`force_tol_to_au`], cubed: a stress is an energy *density*, so the factor is
/// the cube of the length conversion and getting it backwards is 6.75× rather than 3.57×.
pub const fn stress_tol_to_au(ev_per_angstrom3: f64) -> f64 {
    ev_per_angstrom3 * BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM
}

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

    /// A tolerance converted the wrong way is still a number, so pin the direction.
    ///
    /// Both of these must come out **smaller** than they went in: a Bohr is shorter than an
    /// Ångström, so the same physical force is a smaller number per Bohr. Every occurrence of
    /// this bug in this crate has been the inverse, which makes the tolerance looser and the
    /// optimizer stop early while still reporting success.
    #[test]
    fn a_tolerance_in_atomic_units_is_the_smaller_number() {
        assert!(force_tol_to_au(0.02) < 0.02);
        assert!(stress_tol_to_au(0.001) < 0.001);

        // The physical statement, done by hand: a 1 eV/Å force does this much work over a Bohr.
        assert!((force_tol_to_au(1.0) - PM3_A0).abs() < 1e-12);
        // And a stress is that per volume, so the cube of it.
        assert!((stress_tol_to_au(1.0) - PM3_A0.powi(3)).abs() < 1e-12);

        // The two factors are not interchangeable, which is the whole trap: reaching for the
        // other one is off by this much, silently.
        assert!((force_tol_to_au(1.0) * ANGSTROM_TO_BOHR - 1.0).abs() < 1e-12);
        assert!((ANGSTROM_TO_BOHR / BOHR_TO_ANGSTROM - 3.5709).abs() < 1e-3);
    }
}
