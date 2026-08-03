// SPDX-License-Identifier: GPL-3.0-or-later

//! Python bindings (pyo3), built into the `pm3-rs-python` distribution as
//! `pm3_rs._native`.
//!
//! Every function here returns **atomic units (Hartree, Bohr)** — the raw
//! native surface — plus convenience eV and kcal/mol fields. Input coordinates
//! are Ångström (the common Python convention). The eV/Å ASE boundary is
//! applied in the pure-Python `pm3_rs.ase` layer.

use crate::constants::{ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, EV_TO_HARTREE};
use crate::corrections::Variant;
use crate::gradient::closed_form_gradient;
use crate::optimizer::{optimize as opt_geom, OptOptions};
use crate::params::Pm3Parameters;
use crate::scf::{run_pm3, Pm3Options, Reference};
use crate::system::{Atom, Molecule};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn to_py_err(e: crate::error::Pm3Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Parse the `reference` string ("auto" | "rhf" | "uhf", case-insensitive).
fn parse_reference(reference: &str) -> PyResult<Reference> {
    match reference.to_ascii_lowercase().as_str() {
        "auto" | "" => Ok(Reference::Auto),
        "rhf" | "r" | "restricted" => Ok(Reference::Rhf),
        "uhf" | "u" | "unrestricted" => Ok(Reference::Uhf),
        other => Err(PyValueError::new_err(format!(
            "reference must be 'auto', 'rhf' or 'uhf' (got {other:?})"
        ))),
    }
}

/// Parse the `method` string into a correction [`Variant`]
/// ("pm3" | "pm3-d3" | "pm3-d3h4" | "pm3-d3h4x", case-insensitive).
fn parse_variant(method: &str) -> PyResult<Variant> {
    if method.is_empty() {
        return Ok(Variant::Pm3);
    }
    Variant::parse(method).ok_or_else(|| {
        PyValueError::new_err(format!(
            "method must be 'pm3', 'pm3-d3', 'pm3-d3h4' or 'pm3-d3h4x' (got {method:?})"
        ))
    })
}

fn build_molecule(
    numbers: &[u8],
    positions: &[Vec<f64>],
    charge: f64,
    mult: usize,
) -> PyResult<Molecule> {
    if numbers.len() != positions.len() {
        return Err(PyValueError::new_err(
            "numbers and positions length mismatch",
        ));
    }
    let mut atoms = Vec::with_capacity(numbers.len());
    for (z, p) in numbers.iter().zip(positions) {
        if p.len() != 3 {
            return Err(PyValueError::new_err(
                "each position must have 3 components",
            ));
        }
        atoms.push(Atom {
            z: *z,
            position: crate::math::Vec3::new(p[0], p[1], p[2]) * ANGSTROM_TO_BOHR,
        });
    }
    Ok(Molecule {
        atoms,
        charge,
        multiplicity: mult.max(1),
    })
}

fn options(charge: f64, multiplicity: usize, reference: Reference, variant: Variant) -> Pm3Options {
    Pm3Options {
        charge,
        multiplicity,
        reference,
        variant,
        ..Pm3Options::default()
    }
}

/// Single-point PM3. Returns a dict in atomic units (Hartree), plus eV and
/// ΔHf in kcal/mol.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="pm3"))]
fn single_point(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let r = run_pm3(
        &mol,
        &params,
        &options(
            charge,
            multiplicity,
            parse_reference(reference)?,
            parse_variant(method)?,
        ),
    )
    .map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item("energy_hartree", r.total_ev * EV_TO_HARTREE)?;
    d.set_item("energy_ev", r.total_ev)?;
    d.set_item("heat_of_formation_kcal", r.heat_of_formation_kcal)?;
    d.set_item("electronic_ev", r.electronic_ev)?;
    d.set_item("core_ev", r.core_ev)?;
    d.set_item("charges", r.charges)?;
    d.set_item(
        "dipole_debye",
        [r.dipole_debye.x, r.dipole_debye.y, r.dipole_debye.z],
    )?;
    d.set_item("homo_ev", r.homo_ev)?;
    d.set_item("lumo_ev", r.lumo_ev)?;
    d.set_item("converged", r.converged)?;
    d.set_item("unrestricted", r.unrestricted)?;
    Ok(d.into())
}

/// Energy + gradient. Gradient in Hartree/Bohr (atomic units) and eV/Å.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="pm3"))]
fn gradient(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let g = closed_form_gradient(
        &mol,
        &params,
        &options(
            charge,
            multiplicity,
            parse_reference(reference)?,
            parse_variant(method)?,
        ),
    )
    .map_err(to_py_err)?;
    let grad_au: Vec<[f64; 3]> = g
        .gradient
        .iter()
        .map(|v| {
            [
                v.x * EV_TO_HARTREE,
                v.y * EV_TO_HARTREE,
                v.z * EV_TO_HARTREE,
            ]
        })
        .collect();
    // eV/Å = (eV/Bohr) · (Bohr per Å).
    let grad_ev_ang: Vec<[f64; 3]> = g
        .gradient
        .iter()
        .map(|v| {
            [
                v.x * ANGSTROM_TO_BOHR,
                v.y * ANGSTROM_TO_BOHR,
                v.z * ANGSTROM_TO_BOHR,
            ]
        })
        .collect();
    let d = PyDict::new(py);
    d.set_item("energy_hartree", g.energy_ev * EV_TO_HARTREE)?;
    d.set_item("energy_ev", g.energy_ev)?;
    d.set_item("heat_of_formation_kcal", g.scf.heat_of_formation_kcal)?;
    d.set_item("gradient_hartree_per_bohr", grad_au)?;
    d.set_item("gradient_ev_per_angstrom", grad_ev_ang)?;
    Ok(d.into())
}

/// Energy + forces (= −gradient). Returned in Hartree/Bohr and eV/Å.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="pm3"))]
fn forces(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let g = closed_form_gradient(
        &mol,
        &params,
        &options(
            charge,
            multiplicity,
            parse_reference(reference)?,
            parse_variant(method)?,
        ),
    )
    .map_err(to_py_err)?;
    // forces = −gradient.
    let f_au: Vec<[f64; 3]> = g
        .forces
        .iter()
        .map(|v| {
            [
                v.x * EV_TO_HARTREE,
                v.y * EV_TO_HARTREE,
                v.z * EV_TO_HARTREE,
            ]
        })
        .collect();
    let f_ev_ang: Vec<[f64; 3]> = g
        .forces
        .iter()
        .map(|v| {
            [
                v.x * ANGSTROM_TO_BOHR,
                v.y * ANGSTROM_TO_BOHR,
                v.z * ANGSTROM_TO_BOHR,
            ]
        })
        .collect();
    let d = PyDict::new(py);
    d.set_item("energy_hartree", g.energy_ev * EV_TO_HARTREE)?;
    d.set_item("energy_ev", g.energy_ev)?;
    d.set_item("heat_of_formation_kcal", g.scf.heat_of_formation_kcal)?;
    d.set_item("forces_hartree_per_bohr", f_au)?;
    d.set_item("forces_ev_per_angstrom", f_ev_ang)?;
    Ok(d.into())
}

/// L-BFGS geometry optimization. Returns optimized positions in Ångström.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="pm3"))]
fn optimize(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let res = opt_geom(
        &mol,
        &params,
        &options(
            charge,
            multiplicity,
            parse_reference(reference)?,
            parse_variant(method)?,
        ),
        &OptOptions::default(),
    )
    .map_err(to_py_err)?;
    let coords: Vec<[f64; 3]> = res
        .molecule
        .atoms
        .iter()
        .map(|a| {
            let p = a.position * BOHR_TO_ANGSTROM;
            [p.x, p.y, p.z]
        })
        .collect();
    let d = PyDict::new(py);
    d.set_item("positions_angstrom", coords)?;
    d.set_item("energy_hartree", res.scf.total_ev * EV_TO_HARTREE)?;
    d.set_item("heat_of_formation_kcal", res.scf.heat_of_formation_kcal)?;
    d.set_item("converged", res.converged)?;
    d.set_item("iterations", res.iterations)?;
    Ok(d.into())
}

/// Harmonic vibrational frequencies (cm⁻¹) at the given geometry.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="pm3"))]
fn frequencies(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let vib = crate::hessian::vibrational_analysis(
        &mol,
        &params,
        &options(
            charge,
            multiplicity,
            parse_reference(reference)?,
            parse_variant(method)?,
        ),
        1.0e-3,
    )
    .map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item("frequencies_cm", vib.frequencies_cm)?;
    d.set_item("eigenvalues", vib.eigenvalues)?;
    Ok(d.into())
}

/// Analytic Cartesian Hessian (Hartree/Bohr²), row-major `3N × 3N`.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="pm3"))]
fn hessian(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let h = crate::hessian::analytic_hessian(
        &mol,
        &params,
        &options(
            charge,
            multiplicity,
            parse_reference(reference)?,
            parse_variant(method)?,
        ),
        1.0e-3,
    )
    .map_err(to_py_err)?;
    let n = h.rows;
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(n);
        for j in 0..n {
            // eV/Bohr² → Hartree/Bohr².
            row.push(h[(i, j)] * EV_TO_HARTREE);
        }
        rows.push(row);
    }
    let d = PyDict::new(py);
    d.set_item("hessian_hartree_per_bohr2", rows)?;
    d.set_item("ndof", n)?;
    Ok(d.into())
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(single_point, m)?)?;
    m.add_function(wrap_pyfunction!(gradient, m)?)?;
    m.add_function(wrap_pyfunction!(forces, m)?)?;
    m.add_function(wrap_pyfunction!(optimize, m)?)?;
    m.add_function(wrap_pyfunction!(frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(hessian, m)?)?;
    Ok(())
}
