// SPDX-License-Identifier: GPL-3.0-or-later

#![forbid(unsafe_code)]

//! # pm3-rs
//!
//! Rust-native implementation of the PM3 semiempirical NDDO method
//! (J. J. P. Stewart, *J. Comput. Chem.* **10**, 209 (1989)): non-periodic
//! energies, fully analytic nuclear gradients (forward-mode dual numbers
//! through the integral kernels) and analytic Hessians (hyper-dual skeleton +
//! iterative CPHF response), L-BFGS geometry optimization, and D3/H4/X post-SCF
//! corrections (PM3-D3, PM3-D3H4, PM3-D3H4X).
//!
//! Parameters are extracted from MOPAC v23.2.5 (openmopac/mopac, Apache-2.0)
//! with provenance recorded in `THIRD_PARTY_NOTICES.md` and in the embedded
//! CSV headers. MOPAC is also the validation oracle for every milestone.
//!
//! Internal units: eV energies, Bohr distances, 2018-CODATA MOPAC model
//! constants ([`constants`]). Public Rust/Python-native APIs speak atomic
//! units (Hartree/Bohr); the ASE calculator speaks eV/Å.

pub mod basis;
pub mod constants;
pub mod corrections;
pub mod data_tables;
pub mod dual;
pub mod dual2;
pub mod error;
pub mod fock;
pub mod frame;
pub mod gradient;
pub mod hamiltonian;
pub mod hessian;
pub mod integrals;
pub mod integrals_d;
pub mod linalg;
pub mod math;
pub mod onecenter;
pub mod optimizer;
pub mod overlap;
pub mod overlap_numeric;
pub mod params;
pub mod repulsion;
pub mod rotations;
pub mod scf;
pub mod system;

#[cfg(feature = "python")]
pub mod python;

pub use corrections::{correction_energy, CorrectionEnergies, Variant};
pub use error::{Pm3Error, Result};
pub use gradient::{analytic_gradient, closed_form_gradient, numerical_gradient, GradientResult};
pub use hessian::{analytic_hessian, numerical_hessian, vibrational_analysis, VibrationalModes};
pub use linalg::Matrix;
pub use math::{Mat3, Vec3};
pub use optimizer::{optimize, OptOptions, OptResult};
pub use params::{PairParams, Pm3Element, Pm3Parameters, SparkleElement};
pub use scf::{run_pm3, Pm3Calculator, Pm3Options, Pm3Result, Reference, ScfAccelerator};
pub use system::{symbol_to_z, z_to_symbol, Atom, Molecule};
