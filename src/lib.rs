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
pub mod cell;
pub mod cli;
pub mod cmatrix;
pub mod constants;
pub mod corrections;
pub mod data_tables;
pub mod dc;
mod densitydiis;
pub mod dipole;
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
pub mod ir;
pub mod linalg;
pub mod math;
pub mod molden;
pub mod neighbor;
pub mod onecenter;
pub mod optimizer;
pub mod overlap;
pub mod overlap_numeric;
pub mod params;
pub mod pbc;
pub mod repulsion;
pub mod rigid;
pub mod rotations;
pub mod scf;
pub mod special;
pub mod system;

#[cfg(feature = "python")]
pub mod python;

pub use cell::Cell;
pub use cmatrix::{hermitian_eigen, CMatrix};
pub use corrections::{correction_energy, CorrectionEnergies, Variant};
pub use dc::{
    dc_gradient, dc_periodic_gradient, partition, run_dc, run_dc_gamma, DcGradient, DcOptions,
    DcPeriodicGradient, DcPeriodicResult, DcResult, Partition, Subsystem,
};
pub use dipole::{centre_of_mass, dipole_from_density, dipole_matrix, field_terms};
pub use error::{Pm3Error, Result};
pub use gradient::{analytic_gradient, closed_form_gradient, numerical_gradient, GradientResult};
pub use hessian::{analytic_hessian, numerical_hessian, vibrational_analysis, VibrationalModes};
pub use ir::{dipole_derivatives, dipole_derivatives_at, ir_spectrum, IrSpectrum};
pub use linalg::Matrix;
pub use math::{Mat3, Vec3};
pub use molden::molden_string;
pub use neighbor::{NeighborList, PairImage};
pub use optimizer::{optimize, optimize_dc, DcOptResult, OptOptions, OptResult};
pub use params::{PairParams, Pm3Element, Pm3Parameters, SparkleElement};
pub use pbc::berry::{berry_polarization, BerryPolarization};
pub use pbc::born::{born_charge_sum_rule_residual, born_charges, enforce_born_sum_rule};
pub use pbc::dfpt::{
    dynamical_matrix, dynamical_matrix_on_mesh, force_constants_at_q, frequencies_at_q,
    frequencies_of, phonon_frequencies, phonon_frequencies_on_mesh, rigid_ion_dynamical_matrix,
    DfptOptions, DfptResult, DynamicalMatrix, LongRange,
};
#[allow(deprecated)]
pub use pbc::dielectric::SOFT_MODE_FLOOR;
pub use pbc::dielectric::{
    dielectric_origin_sensitivity, dielectric_tensor, polarizability, static_dielectric_tensor,
    DielectricTensors, StaticDielectric,
};
pub use pbc::ewald::{ewald, ChargeSite, EwaldOutput, EwaldParams};
pub use pbc::ewald_hessian::ewald_atom_hessian;
pub use pbc::finite_field::{run_finite_field, FiniteFieldOptions, FiniteFieldResult};
pub use pbc::gamma::{run_gamma, PeriodicOptions, PeriodicResult};
pub use pbc::gradient::{periodic_gradient, PeriodicGradient};
pub use pbc::hessian::{periodic_hessian, periodic_phonons, PeriodicPhonons};
pub use pbc::kpoints::{band_path, monkhorst_pack, KPoint, KpointSpec};
pub use pbc::kscf::{
    band_structure, kpoint_gradient, run_kpoints, BandStructure, KpointGradient, KpointOptions,
    KpointResult, Magnetization,
};
pub use pbc::lo_to::{add_non_analytic, non_analytic_term};
pub use pbc::optimize::{relax, CellRelaxation, PeriodicOptOptions, PeriodicOptResult};
pub use pbc::phonon::{build_supercell, cartesian_q, q_path, ForceConstants};
pub use scf::{run_pm3, Pm3Calculator, Pm3Options, Pm3Result, Reference, ScfAccelerator};
pub use system::{symbol_to_z, z_to_symbol, Atom, Molecule};
