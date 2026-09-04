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

/// Attach a cell to a molecule, converting Ångström to Bohr.
///
/// `cell` is three lattice vectors as rows, in Ångström — the same convention ASE uses and the
/// same one `Molecule::from_xyz_*` uses for positions. `pbc` says which directions are periodic;
/// omitting it means all three.
fn attach_cell(
    molecule: &mut Molecule,
    cell: Option<Vec<Vec<f64>>>,
    pbc: Option<Vec<bool>>,
) -> PyResult<()> {
    let Some(rows) = cell else {
        if pbc.is_some_and(|flags| flags.iter().any(|f| *f)) {
            return Err(PyValueError::new_err(
                "pbc was requested but no cell was given",
            ));
        }
        return Ok(());
    };
    if rows.len() != 3 || rows.iter().any(|row| row.len() != 3) {
        return Err(PyValueError::new_err(
            "cell must be three lattice vectors of three components each",
        ));
    }
    let flags = match pbc {
        Some(values) => {
            if values.len() != 3 {
                return Err(PyValueError::new_err("pbc must have three components"));
            }
            [values[0], values[1], values[2]]
        }
        None => [true; 3],
    };
    let vector = |row: &Vec<f64>| crate::math::Vec3::new(row[0], row[1], row[2]) * ANGSTROM_TO_BOHR;
    molecule.cell = Some(
        crate::cell::Cell::new(vector(&rows[0]), vector(&rows[1]), vector(&rows[2]), flags)
            .map_err(to_py_err)?,
    );
    Ok(())
}

/// Resolve a k-point specification: `None` or `[1, 1, 1]` is the Γ point.
/// How the two spin channels share (or do not share) a Fermi level.
///
/// `"fixed"` holds `n_α − n_β` at the multiplicity's value — the periodic reading of a molecular
/// multiplicity, and the only way to converge onto a chosen spin state. `"free"` gives both
/// spins one Fermi level and lets the moment come out wherever the electronic structure puts it,
/// which is the right convention for a magnetic solid, where the moment is an output.
///
/// `"fixed"` is the default here because it is [`crate::pbc::kscf::KpointOptions`]'s. Until this
/// existed every non-Rust caller was pinned to it, so a magnetic solid could not be asked the
/// one question it exists to answer.
fn parse_magnetization(name: &str) -> PyResult<crate::pbc::kscf::Magnetization> {
    match name.to_ascii_lowercase().as_str() {
        "fixed" => Ok(crate::pbc::kscf::Magnetization::Fixed),
        "free" => Ok(crate::pbc::kscf::Magnetization::Free),
        other => Err(PyValueError::new_err(format!(
            "unknown magnetization: {other} (fixed, free)"
        ))),
    }
}

fn parse_kpoints(kpts: Option<Vec<usize>>) -> PyResult<crate::pbc::kpoints::KpointSpec> {
    match kpts {
        None => Ok(crate::pbc::kpoints::KpointSpec::Gamma),
        Some(values) => {
            if values.len() != 3 {
                return Err(PyValueError::new_err(
                    "kpts must have three components, one per lattice vector",
                ));
            }
            Ok(crate::pbc::kpoints::KpointSpec::mesh([
                values[0], values[1], values[2],
            ]))
        }
    }
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
        cell: None,
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

/// A uniform external field in **volts per Angstrom**, the unit MOPAC's own `FIELD=` keyword
/// takes and the one the rest of this layer's lengths are in.
///
/// Internally the field is eV per Bohr with the sign that makes `E = E₀ + μ·F`, so the conversion
/// is a length unit and a sign. Doing it here rather than asking the caller to is what keeps this
/// surface consistent with the Angstroms it takes everywhere else.
fn parse_field(field: Option<Vec<f64>>) -> PyResult<Option<crate::math::Vec3>> {
    let Some(values) = field else { return Ok(None) };
    if values.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "field must have three components, got {}",
            values.len()
        )));
    }
    let scale = -crate::constants::BOHR_TO_ANGSTROM;
    Ok(Some(crate::math::Vec3::new(
        scale * values[0],
        scale * values[1],
        scale * values[2],
    )))
}

/// [`options`] with an external field attached.
fn options_with_field(
    charge: f64,
    multiplicity: usize,
    reference: Reference,
    variant: Variant,
    field: Option<Vec<f64>>,
) -> PyResult<Pm3Options> {
    Ok(Pm3Options {
        field: parse_field(field)?,
        ..options(charge, multiplicity, reference, variant)
    })
}

/// Single-point PM3. Returns a dict in atomic units (Hartree), plus eV and
/// ΔHf in kcal/mol.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="pm3", field=None))]
#[allow(clippy::too_many_arguments)]
fn single_point(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    field: Option<Vec<f64>>,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let r = run_pm3(
        &mol,
        &params,
        &options_with_field(
            charge,
            multiplicity,
            parse_reference(reference)?,
            parse_variant(method)?,
            field,
        )?,
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
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="pm3", field=None))]
#[allow(clippy::too_many_arguments)]
fn gradient(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    field: Option<Vec<f64>>,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let g = closed_form_gradient(
        &mol,
        &params,
        &options_with_field(
            charge,
            multiplicity,
            parse_reference(reference)?,
            parse_variant(method)?,
            field,
        )?,
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
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="pm3", field=None))]
#[allow(clippy::too_many_arguments)]
fn forces(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    field: Option<Vec<f64>>,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let g = closed_form_gradient(
        &mol,
        &params,
        &options_with_field(
            charge,
            multiplicity,
            parse_reference(reference)?,
            parse_variant(method)?,
            field,
        )?,
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
    // Everything else the SCF already produced. Returning it here means a caller that wants
    // energy *and* forces *and* charges — a molecular-dynamics step, say — converges one SCF
    // rather than two, which is most of the cost of such a step.
    d.set_item("charges", g.scf.charges.clone())?;
    d.set_item(
        "dipole_debye",
        [
            g.scf.dipole_debye.x,
            g.scf.dipole_debye.y,
            g.scf.dipole_debye.z,
        ],
    )?;
    d.set_item("converged", g.scf.converged)?;
    Ok(d.into())
}

/// L-BFGS geometry optimization. Returns optimized positions in Ångström.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="pm3", field=None))]
#[allow(clippy::too_many_arguments)]
fn optimize(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    field: Option<Vec<f64>>,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let res = opt_geom(
        &mol,
        &params,
        &options_with_field(
            charge,
            multiplicity,
            parse_reference(reference)?,
            parse_variant(method)?,
            field,
        )?,
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
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="pm3", field=None))]
#[allow(clippy::too_many_arguments)]
fn frequencies(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    field: Option<Vec<f64>>,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let vib = crate::hessian::vibrational_analysis(
        &mol,
        &params,
        &options_with_field(
            charge,
            multiplicity,
            parse_reference(reference)?,
            parse_variant(method)?,
            field,
        )?,
        1.0e-3,
    )
    .map_err(to_py_err)?;

    // The modes and the masses that de-weight them. `VibrationalModes::modes` is in
    // **mass-weighted** coordinates, so the Cartesian displacement of mode `m` is
    // `l_{im} / √m_i` — which a caller cannot form without the masses, and the crate's
    // isotope-averaged values are not available anywhere else on this surface. Returning the
    // frequencies alone left every use of a normal mode (animating it, projecting onto it,
    // contracting a Cartesian quantity against it) reachable only from Rust.
    let n = vib.modes.rows;
    let modes: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..vib.modes.cols).map(|j| vib.modes[(i, j)]).collect())
        .collect();
    let masses = atomic_masses(&mol, &params)?;

    let d = PyDict::new(py);
    d.set_item("frequencies_cm", vib.frequencies_cm)?;
    d.set_item("eigenvalues", vib.eigenvalues)?;
    d.set_item("modes", modes)?;
    d.set_item("masses", masses)?;
    Ok(d.into())
}

/// Isotope-averaged atomic masses (amu), in the order the geometry lists the atoms.
///
/// The crate's own values, from the PM3 element table — deliberately not `ase`'s, so that a
/// mass-weighted quantity handed out here can be un-weighted with exactly what weighted it.
fn atomic_masses(mol: &crate::system::Molecule, params: &Pm3Parameters) -> PyResult<Vec<f64>> {
    mol.atoms
        .iter()
        .map(|atom| params.element(atom.z).map(|e| e.mass).map_err(to_py_err))
        .collect()
}

/// Analytic Cartesian Hessian (Hartree/Bohr²), row-major `3N × 3N`.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto", method="pm3", field=None))]
#[allow(clippy::too_many_arguments)]
fn hessian(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    field: Option<Vec<f64>>,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let h = crate::hessian::analytic_hessian(
        &mol,
        &params,
        &options_with_field(
            charge,
            multiplicity,
            parse_reference(reference)?,
            parse_variant(method)?,
            field,
        )?,
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

/// Periodic single point. `cell` is three lattice vectors as rows, in Ångström.
///
/// Omitting `kpts` (or passing `[1, 1, 1]`) takes only the Γ point — see `gamma_margin` in the
/// result, and the validity condition in `docs/pbc.md`.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, pbc=None, kpts=None, charge=0.0, multiplicity=1,
                    reference="auto", method="pm3", smearing_ev=0.0, magnetization="fixed"))]
#[allow(clippy::too_many_arguments)]
fn periodic_single_point(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    pbc: Option<Vec<bool>>,
    kpts: Option<Vec<usize>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    smearing_ev: f64,
    magnetization: &str,
) -> PyResult<PyObject> {
    let mut mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    attach_cell(&mut mol, Some(cell), pbc)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let options = options(
        charge,
        multiplicity,
        parse_reference(reference)?,
        parse_variant(method)?,
    );
    let periodic = crate::pbc::gamma::PeriodicOptions::default();
    let spec = parse_kpoints(kpts)?;

    let d = PyDict::new(py);
    if spec == crate::pbc::kpoints::KpointSpec::Gamma && smearing_ev == 0.0 {
        let r =
            crate::pbc::gamma::run_gamma(&mol, &params, &options, &periodic).map_err(to_py_err)?;
        d.set_item("energy_hartree", r.total_ev * EV_TO_HARTREE)?;
        d.set_item("energy_ev", r.total_ev)?;
        d.set_item("heat_of_formation_kcal", r.heat_of_formation_kcal)?;
        d.set_item("electronic_ev", r.electronic_ev)?;
        d.set_item("core_ev", r.core_ev)?;
        d.set_item("correction_ev", r.correction_ev)?;
        d.set_item("ewald_ev", r.ewald_ev)?;
        d.set_item("charges", r.charges)?;
        d.set_item("homo_ev", r.homo_ev)?;
        d.set_item("lumo_ev", r.lumo_ev)?;
        d.set_item("gamma_margin_bohr", r.gamma_margin)?;
        // The two branches return the same keys, differing only in which of them are `None`.
        // Which keys exist used to depend on which branch ran, so a caller reading
        // `result["band_gap_ev"]` got a `KeyError` for exactly the Γ-point calculations that
        // have a gap to report — and the fix for that, in caller code, is a `.get()` that then
        // hides a genuinely absent value too.
        //
        // At Γ the gap is the HOMO–LUMO separation. There is no Fermi level: orbitals are filled
        // by aufbau, so nothing here is a chemical potential and `None` is the honest answer
        // rather than the midpoint dressed up as one.
        let gap = match (r.homo_ev, r.lumo_ev) {
            (Some(homo), Some(lumo)) => Some(lumo - homo),
            _ => None,
        };
        d.set_item("band_gap_ev", gap)?;
        d.set_item("fermi_ev", None::<f64>)?;
        // The Γ branch is reached only with no smearing, so the occupations are a step and the
        // electronic entropy is identically zero. Reported anyway, so that the two branches
        // return the same keys.
        d.set_item("entropy_ts_ev", 0.0)?;
        d.set_item("free_energy_ev", r.total_ev)?;
        d.set_item("converged", r.converged)?;
        d.set_item("unrestricted", r.unrestricted)?;
        d.set_item("n_kpoints", 1)?;
    } else {
        let kopt = crate::pbc::kscf::KpointOptions {
            spec,
            smearing_ev,
            magnetization: parse_magnetization(magnetization)?,
        };
        let r = crate::pbc::kscf::run_kpoints(&mol, &params, &options, &periodic, &kopt)
            .map_err(to_py_err)?;
        d.set_item("energy_hartree", r.total_ev * EV_TO_HARTREE)?;
        d.set_item("energy_ev", r.total_ev)?;
        d.set_item("heat_of_formation_kcal", r.heat_of_formation_kcal)?;
        d.set_item("electronic_ev", r.electronic_ev)?;
        d.set_item("core_ev", r.core_ev)?;
        d.set_item("correction_ev", r.correction_ev)?;
        d.set_item("ewald_ev", r.ewald_ev)?;
        d.set_item("charges", r.charges)?;
        d.set_item("homo_ev", r.homo_ev)?;
        d.set_item("lumo_ev", r.lumo_ev)?;
        d.set_item("band_gap_ev", r.band_gap_ev)?;
        d.set_item("fermi_ev", r.fermi_ev)?;
        d.set_item("entropy_ts_ev", r.entropy_ts_ev)?;
        d.set_item("free_energy_ev", r.free_energy_ev)?;
        // Reported here too, where it is a diagnosis rather than a warning: a negative margin
        // says *why* the mesh was necessary. `KpointResult` has carried it all along.
        d.set_item("gamma_margin_bohr", r.gamma_margin)?;
        d.set_item("converged", r.converged)?;
        d.set_item("unrestricted", r.unrestricted)?;
        d.set_item("n_kpoints", r.kpoints.len())?;
    }
    Ok(d.into())
}

/// Periodic forces and stress.
///
/// Forces are eV/Å and the stress is the 6-component Voigt vector in eV/Å³, the two conventions
/// ASE expects. Every periodic dimensionality reports a stress, with exact zeros in the
/// non-periodic directions; only an isolated cell returns `None`, having no strain at all.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, pbc=None, kpts=None, charge=0.0, multiplicity=1,
                    reference="auto", method="pm3", smearing_ev=0.0, magnetization="fixed"))]
#[allow(clippy::too_many_arguments)]
fn periodic_forces(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    pbc: Option<Vec<bool>>,
    kpts: Option<Vec<usize>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    smearing_ev: f64,
    magnetization: &str,
) -> PyResult<PyObject> {
    let mut mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    attach_cell(&mut mol, Some(cell), pbc)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let options = options(
        charge,
        multiplicity,
        parse_reference(reference)?,
        parse_variant(method)?,
    );
    let periodic = crate::pbc::gamma::PeriodicOptions::default();
    let spec = parse_kpoints(kpts)?;
    let magnetization = parse_magnetization(magnetization)?;

    // One SCF serves the whole calculator. ASE asks for energy, forces and stress in separate
    // calls, so returning only what was named would converge the same SCF two or three times per
    // molecular-dynamics step.
    //
    // The dispatch condition is `periodic_single_point`'s, character for character, and it has
    // to be: this function had no `smearing_ev` at all, so an ASE calculator built with one got
    // its energy and forces from a strictly-filled SCF and its charges from a smeared one — two
    // different self-consistent solutions stitched into one result dictionary, on exactly the
    // metals that smearing exists for.
    let (
        energy_ev,
        gradient,
        stress,
        charges,
        heat_of_formation_kcal,
        gamma_margin,
        converged,
        entropy_ts_ev,
    ) = if spec == crate::pbc::kpoints::KpointSpec::Gamma && smearing_ev == 0.0 {
        let g = crate::pbc::gradient::periodic_gradient(&mol, &params, &options, &periodic)
            .map_err(to_py_err)?;
        (
            g.energy_ev,
            g.gradient,
            g.stress,
            g.scf.charges.clone(),
            g.scf.heat_of_formation_kcal,
            g.scf.gamma_margin,
            g.scf.converged,
            // No smearing on this branch, so the occupations are a step and there is no
            // electronic entropy to subtract.
            0.0,
        )
    } else {
        let kopt = crate::pbc::kscf::KpointOptions {
            spec,
            smearing_ev,
            magnetization,
        };
        let g = crate::pbc::kscf::kpoint_gradient(&mol, &params, &options, &periodic, &kopt)
            .map_err(to_py_err)?;
        (
            g.energy_ev,
            g.gradient,
            g.stress,
            g.scf.charges.clone(),
            g.scf.heat_of_formation_kcal,
            g.scf.gamma_margin,
            g.scf.converged,
            g.scf.entropy_ts_ev,
        )
    };

    // eV/Bohr → eV/Å, and the sign flip from gradient to force.
    let scale = ANGSTROM_TO_BOHR;
    let forces: Vec<Vec<f64>> = gradient
        .iter()
        .map(|g| vec![-g.x * scale, -g.y * scale, -g.z * scale])
        .collect();
    let d = PyDict::new(py);
    d.set_item("energy_ev", energy_ev)?;
    d.set_item("energy_hartree", energy_ev * EV_TO_HARTREE)?;
    d.set_item("forces_ev_per_angstrom", forces)?;
    // As in the molecular path: one SCF serves the whole calculator.
    d.set_item("charges", charges)?;
    // The Mermin electronic free energy, and the entropy term it differs from `energy_ev` by.
    //
    // This is what the forces above are the gradient of: with Fermi–Dirac occupations the
    // variational functional is `A = E − TS`, so a Hellmann–Feynman force is `−dA/dR` and not
    // `−dE/dR`. Reporting only `energy_ev` beside these forces made the pair inconsistent by
    // `∂(TS)/∂R` on exactly the metals that smearing exists for.
    //
    // Not a Gibbs free energy: no zero-point energy, no vibrational partition function, no `pV`,
    // no nuclear entropy. See `KpointResult::free_energy_ev`.
    d.set_item("entropy_ts_ev", entropy_ts_ev)?;
    d.set_item("free_energy_ev", energy_ev - entropy_ts_ev)?;
    d.set_item("heat_of_formation_kcal", heat_of_formation_kcal)?;
    d.set_item("gamma_margin_bohr", gamma_margin)?;
    d.set_item("converged", converged)?;
    match stress {
        Some(s) => d.set_item("stress_ev_per_angstrom3", voigt_ev_per_angstrom3(&s))?,
        None => d.set_item("stress_ev_per_angstrom3", py.None())?,
    }
    Ok(d.into())
}

/// A stress tensor in eV/Bohr³ as the 6-component Voigt vector in eV/Å³ that ASE expects.
///
/// Shared rather than written out at each call site: the order `(xx, yy, zz, yz, xz, xy)`, the
/// symmetrization of the off-diagonal pairs and the *cube* of the length conversion are three
/// separate things to get wrong, and a second copy is a second place for them to drift.
fn voigt_ev_per_angstrom3(stress: &crate::math::Mat3) -> Vec<f64> {
    let per_volume = ANGSTROM_TO_BOHR * ANGSTROM_TO_BOHR * ANGSTROM_TO_BOHR;
    let at = |alpha: usize, beta: usize| stress.col[beta].to_array()[alpha] * per_volume;
    vec![
        at(0, 0),
        at(1, 1),
        at(2, 2),
        0.5 * (at(1, 2) + at(2, 1)),
        0.5 * (at(0, 2) + at(2, 0)),
        0.5 * (at(0, 1) + at(1, 0)),
    ]
}

/// Γ-point phonon frequencies (cm⁻¹) for a periodic cell.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, pbc=None, charge=0.0, multiplicity=1,
                    reference="auto", method="pm3", q=None, kpts=None, smearing_ev=0.0,
                    lo_to_direction=None))]
#[allow(clippy::too_many_arguments)]
fn phonons(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    pbc: Option<Vec<bool>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    q: Option<Vec<f64>>,
    kpts: Option<Vec<usize>>,
    smearing_ev: f64,
    lo_to_direction: Option<Vec<f64>>,
) -> PyResult<PyObject> {
    let mut mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    attach_cell(&mut mol, Some(cell), pbc)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let options = options(
        charge,
        multiplicity,
        parse_reference(reference)?,
        parse_variant(method)?,
    );
    let periodic = crate::pbc::gamma::PeriodicOptions::default();
    let d = PyDict::new(py);

    // `q = None` keeps the Γ-point path, which reports the acoustic residual and the energy
    // alongside the frequencies. A wavevector goes through the dynamical matrix instead, where
    // there is no acoustic sum rule to report away from Γ and no separate SCF energy to attach.
    let Some(q_frac) = q else {
        if lo_to_direction.is_some() {
            return Err(PyValueError::new_err(
                "lo_to_direction needs a q to be the limit of. Pass q=[0, 0, 0] to ask for the \
                 Gamma-point matrix with the non-analytic term added along that direction; the \
                 Gamma-point Hessian path here reports the transverse limit and has no direction \
                 to take.",
            ));
        }
        let result = crate::pbc::hessian::periodic_phonons(&mol, &params, &options, &periodic)
            .map_err(to_py_err)?;
        d.set_item("frequencies_cm", result.frequencies_cm)?;
        d.set_item("acoustic_residual_cm", result.acoustic_residual_cm)?;
        d.set_item("energy_ev", result.scf.total_ev)?;
        return Ok(d.into());
    };
    if q_frac.len() != 3 {
        return Err(PyValueError::new_err(
            "q must have three fractional components, one per reciprocal lattice vector",
        ));
    }
    let q_frac = [q_frac[0], q_frac[1], q_frac[2]];
    // Built as the matrix and then diagonalized, rather than through `phonon_frequencies`, which
    // is exactly that composition with the matrix dropped on the floor. Keeping it costs nothing
    // and carries out the Hermitian defect — the number that moves first if a phase is wrong,
    // and which `dynamical_matrix` and the CLI both report. Reporting it from two of the three
    // entry points is how a caller ends up trusting the one that stayed quiet.
    let matrix = match kpts {
        None => crate::pbc::dfpt::dynamical_matrix(&mol, &params, &options, &periodic, q_frac),
        Some(_) => {
            // `smearing_ev` is here because the response's own refusal names it: a gapless mesh
            // makes the energy denominators singular and the error says to set this. Until it
            // was a parameter, that instruction could not be followed from Python.
            let kopt = crate::pbc::kscf::KpointOptions {
                spec: parse_kpoints(kpts)?,
                smearing_ev,
                ..Default::default()
            };
            crate::pbc::dfpt::dynamical_matrix_on_mesh(
                &mol, &params, &options, &periodic, &kopt, q_frac,
            )
        }
    }
    .map_err(to_py_err)?;
    let mut matrix = matrix;
    // The non-analytic term, if a direction was named. `D(q)` here carries only the microscopic
    // field — the macroscopic `G = 0` member is excluded from the response, because keeping it
    // makes the acoustic sum rule diverge — so this is what restores the longitudinal limit.
    if let Some(direction) = lo_to_direction {
        if direction.len() != 3 {
            return Err(PyValueError::new_err(
                "lo_to_direction takes three Cartesian components",
            ));
        }
        let born = crate::pbc::born::born_charges(&mol, &params, &options, &periodic)
            .map_err(to_py_err)?;
        let tensors = crate::pbc::dielectric::dielectric_tensor(&mol, &params, &options, &periodic)
            .map_err(to_py_err)?;
        crate::pbc::lo_to::add_non_analytic(
            &mut matrix,
            &mol,
            crate::math::Vec3::new(direction[0], direction[1], direction[2]),
            &born,
            tensors.epsilon,
        )
        .map_err(to_py_err)?;
        d.set_item("lo_to_direction", direction)?;
    }
    let frequencies = crate::pbc::dfpt::frequencies_of(&matrix).map_err(to_py_err)?;
    d.set_item("frequencies_cm", frequencies)?;
    d.set_item("masses", matrix.masses.clone())?;
    d.set_item("hermitian_defect", matrix.hermitian_defect)?;
    d.set_item("q", q_frac.to_vec())?;
    Ok(d.into())
}

/// Band energies along a path, in the potential of a converged k-point calculation.
///
/// `path` is fractional k-points, one three-component entry each. Energies come back in eV, one
/// list per point, ascending — the same order [`KpointResult::bands`] uses.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, path, pbc=None, kpts=None, charge=0.0,
                    multiplicity=1, reference="auto", method="pm3", smearing_ev=0.0,
                    magnetization="fixed"))]
#[allow(clippy::too_many_arguments)]
fn bands(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    path: Vec<Vec<f64>>,
    pbc: Option<Vec<bool>>,
    kpts: Option<Vec<usize>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    smearing_ev: f64,
    magnetization: &str,
) -> PyResult<PyObject> {
    let mut mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    attach_cell(&mut mol, Some(cell), pbc)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let options = options(
        charge,
        multiplicity,
        parse_reference(reference)?,
        parse_variant(method)?,
    );
    // The mesh this path is evaluated in the potential of. It is a self-consistent calculation
    // in its own right, so it needs the same occupation controls the energy does: a band
    // structure of a metal converged under strict filling is a different SCF from the energy the
    // caller ran, and its `fermi_ev` is that other calculation's.
    let kopt = crate::pbc::kscf::KpointOptions {
        spec: parse_kpoints(kpts)?,
        smearing_ev,
        magnetization: parse_magnetization(magnetization)?,
    };
    let mut points = Vec::with_capacity(path.len());
    for entry in &path {
        if entry.len() != 3 {
            return Err(PyValueError::new_err(
                "each path point must have three fractional components",
            ));
        }
        points.push(crate::pbc::kpoints::KPoint {
            frac: [entry[0], entry[1], entry[2]],
            weight: 1.0,
        });
    }
    let periodic = crate::pbc::gamma::PeriodicOptions::default();
    let scf = crate::pbc::kscf::run_kpoints(&mol, &params, &options, &periodic, &kopt)
        .map_err(to_py_err)?;
    let result = crate::pbc::kscf::band_structure(&mol, &params, &periodic, &scf, &points)
        .map_err(to_py_err)?;
    // The horizontal axis. `distances` is the cumulative Cartesian path length (1/Bohr), which
    // is what makes segments of different reciprocal length look different on a plot; without it
    // a Python caller can only index the points and gets a band structure whose x-axis is a
    // count. The CLI has printed this since it gained the command — this layer had not.
    let kpoints: Vec<Vec<f64>> = result.kpoints.iter().map(|k| k.frac.to_vec()).collect();
    let d = PyDict::new(py);
    d.set_item("bands_ev", result.bands)?;
    d.set_item("bands_beta_ev", result.bands_beta)?;
    d.set_item("fermi_ev", scf.fermi_ev)?;
    d.set_item("distances_per_bohr", result.distances)?;
    d.set_item("kpoints", kpoints)?;
    Ok(d.into())
}

/// Variable-cell relaxation: the atoms and the lattice vectors that exist.
///
/// Returns the relaxed positions and cell in Ångström. A cell with no periodic direction is
/// refused, having no strain to relax against.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, pbc=None, kpts=None, charge=0.0, multiplicity=1,
                    reference="auto", method="pm3", max_steps=200, force_tol=0.02,
                    stress_tol=0.001, fixed_cell=false))]
#[allow(clippy::too_many_arguments)]
fn relax(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    pbc: Option<Vec<bool>>,
    kpts: Option<Vec<usize>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    max_steps: usize,
    force_tol: f64,
    stress_tol: f64,
    fixed_cell: bool,
) -> PyResult<PyObject> {
    let mut mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    attach_cell(&mut mol, Some(cell), pbc)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let options = options(
        charge,
        multiplicity,
        parse_reference(reference)?,
        parse_variant(method)?,
    );
    // `kpts` is accepted so the signature matches the rest of the periodic surface, and refused
    // rather than ignored: `relax` drives the Γ gradient, so a mesh here would be silently
    // discarded.
    if kpts.is_some() {
        return Err(PyValueError::new_err(
            "relax runs the Γ-point gradient; a k-mesh is not carried through it yet",
        ));
    }
    let opt = crate::pbc::optimize::PeriodicOptOptions {
        max_iter: max_steps,
        // Ångström in, Bohr inside.
        gtol: force_tol * BOHR_TO_ANGSTROM,
        // And the same for the stress, which is a *density*: eV/Å³ → eV/Bohr³ is the cube of the
        // length conversion. This went in raw, so a tolerance documented as eV/Å³ was applied as
        // eV/Bohr³ — 6.75× looser than asked for — and `converged: True` came back for cells
        // whose stress had never reached the threshold. The force line one above is what the
        // missing line should have looked like.
        stress_tol: stress_tol * BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM,
        cell: if fixed_cell {
            crate::pbc::optimize::CellRelaxation::Fixed
        } else {
            crate::pbc::optimize::CellRelaxation::Variable
        },
        ..Default::default()
    };
    let result = crate::pbc::optimize::relax(
        &mol,
        &params,
        &options,
        &crate::pbc::gamma::PeriodicOptions::default(),
        &opt,
    )
    .map_err(to_py_err)?;
    let d = PyDict::new(py);
    let angstrom: Vec<Vec<f64>> = result
        .molecule
        .atoms
        .iter()
        .map(|a| {
            vec![
                a.position.x * BOHR_TO_ANGSTROM,
                a.position.y * BOHR_TO_ANGSTROM,
                a.position.z * BOHR_TO_ANGSTROM,
            ]
        })
        .collect();
    d.set_item("positions", angstrom)?;
    let relaxed = result.molecule.cell.expect("a periodic result has a cell");
    let rows: Vec<Vec<f64>> = relaxed
        .to_rows()
        .iter()
        .map(|row| row.iter().map(|v| v * BOHR_TO_ANGSTROM).collect())
        .collect();
    d.set_item("cell", rows)?;
    d.set_item("energy_ev", result.gradient.energy_ev)?;
    d.set_item("converged", result.converged)?;
    d.set_item("steps", result.iterations)?;
    Ok(d.into())
}

/// Divide-and-conquer single point, molecular or periodic.
///
/// `buffer_radius` is the knob that matters: increasing it must converge the result onto the full
/// diagonalization. `dropped_pairs` and `largest_subsystem` report what the partitioning traded.
///
/// `long_range_cutoff` (Ångström, `None` to leave it off) switches on the linear-scaling near
/// field: pair tables inside the cutoff, a point-charge model outside it. That makes the
/// two-electron work grow with the system rather than with its square, at the cost of the
/// documented Klopman–Ohno switch — a measured 28 µeV per atom at the usual 22 Bohr handover,
/// flat in system size. Molecular runs only.
#[pyfunction]
// The radii are Ångström here and Bohr in `DcOptions`, whose defaults are 6.0 and 9.0 Bohr.
// 3.2 and 4.8 Å are those same radii, which is what the CLI and `pm3_rs.native` already pass.
// Writing 6.0 and 9.0 here made the Python default 1.9× the Rust one — subsystems nearly twice
// the intended size, silently, for anyone who omitted the argument.
#[pyo3(signature = (numbers, positions, cell=None, pbc=None, charge=0.0, multiplicity=1,
                    reference="auto", method="pm3", core_radius=3.2, buffer_radius=4.8,
                    smearing_ev=0.1, long_range_cutoff=None, field=None))]
#[allow(clippy::too_many_arguments)]
fn divide_and_conquer(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Option<Vec<Vec<f64>>>,
    pbc: Option<Vec<bool>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    core_radius: f64,
    buffer_radius: f64,
    smearing_ev: f64,
    long_range_cutoff: Option<f64>,
    field: Option<Vec<f64>>,
) -> PyResult<PyObject> {
    let mut mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let periodic_run = cell.is_some();
    attach_cell(&mut mol, cell, pbc)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    // The field reaches `run_dc`, which carries it (energy, gradient and the nuclear half). The
    // screened and periodic branches refuse it in Rust rather than dropping it, so this passes it
    // in every case and lets those refusals speak.
    let options = options_with_field(
        charge,
        multiplicity,
        parse_reference(reference)?,
        parse_variant(method)?,
        field,
    )?;
    // `run_dc_gamma` does not read `long_range_cutoff` — the screening split belongs to the
    // molecular branch, and the periodic one gets its long range from the lattice sum instead.
    // Setting it and calling the periodic path therefore ran the unscreened `O(N²)` route while
    // the caller believed they had asked for the linear-scaling one. Refused rather than
    // ignored, the way `relax` refuses a `kpts` it cannot carry.
    if periodic_run && long_range_cutoff.is_some() {
        return Err(PyValueError::new_err(
            "long_range_cutoff is the molecular near-field split; the periodic \
             divide-and-conquer path takes its long range from the lattice sum and would ignore \
             it. Drop long_range_cutoff, or drop the cell.",
        ));
    }
    // Radii are Bohr internally; the Python surface is Ångström like every other length here.
    let dc = crate::dc::DcOptions {
        core_radius: core_radius * ANGSTROM_TO_BOHR,
        buffer_radius: buffer_radius * ANGSTROM_TO_BOHR,
        smearing_ev,
        long_range_cutoff: long_range_cutoff.map(|r| r * ANGSTROM_TO_BOHR),
        ..crate::dc::DcOptions::default()
    };

    let d = PyDict::new(py);
    if periodic_run {
        let r = crate::dc::run_dc_gamma(
            &mol,
            &params,
            &options,
            &crate::pbc::gamma::PeriodicOptions::default(),
            &dc,
        )
        .map_err(to_py_err)?;
        d.set_item("energy_ev", r.total_ev)?;
        d.set_item("energy_hartree", r.total_ev * EV_TO_HARTREE)?;
        d.set_item("heat_of_formation_kcal", r.heat_of_formation_kcal)?;
        d.set_item("charges", r.charges)?;
        d.set_item("fermi_ev", r.fermi_ev)?;
        d.set_item("converged", r.converged)?;
        d.set_item("dropped_pairs", r.dropped_pairs)?;
        d.set_item("largest_subsystem", r.largest_subsystem)?;
        d.set_item("n_subsystems", r.n_subsystems)?;
        d.set_item("gamma_margin_bohr", r.gamma_margin)?;
    } else {
        let r = crate::dc::run_dc(&mol, &params, &options, &dc).map_err(to_py_err)?;
        d.set_item("energy_ev", r.total_ev)?;
        d.set_item("energy_hartree", r.total_ev * EV_TO_HARTREE)?;
        d.set_item("heat_of_formation_kcal", r.heat_of_formation_kcal)?;
        d.set_item("charges", r.charges)?;
        d.set_item("fermi_ev", r.fermi_ev)?;
        d.set_item("converged", r.converged)?;
        d.set_item("dropped_pairs", r.dropped_pairs)?;
        d.set_item("largest_subsystem", r.largest_subsystem)?;
        d.set_item("n_subsystems", r.n_subsystems)?;
    }
    Ok(d.into())
}

/// Forces (and, for a cell, stress) from a divide-and-conquer density.
///
/// The partitioned analogue of [`periodic_forces`], and separate from
/// [`divide_and_conquer`] for the same reason that one is separate from
/// `periodic_single_point`: divide and conquer exists for systems large enough that the gradient
/// is worth asking for explicitly rather than producing unasked.
///
/// One caveat the crate states rather than hides: a divide-and-conquer density is not
/// variational — it is assembled, not minimized — so the usual argument that the first-order
/// energy error vanishes does not apply, and the gradient carries the density's own truncation
/// error rather than its square. It converges with the buffer the same way the energy does.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell=None, pbc=None, charge=0.0, multiplicity=1,
                    reference="auto", method="pm3", core_radius=3.2, buffer_radius=4.8,
                    smearing_ev=0.1, long_range_cutoff=None, field=None))]
#[allow(clippy::too_many_arguments)]
fn divide_and_conquer_forces(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Option<Vec<Vec<f64>>>,
    pbc: Option<Vec<bool>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    core_radius: f64,
    buffer_radius: f64,
    smearing_ev: f64,
    long_range_cutoff: Option<f64>,
    field: Option<Vec<f64>>,
) -> PyResult<PyObject> {
    let mut mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let periodic_run = cell.is_some();
    attach_cell(&mut mol, cell, pbc)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let options = options_with_field(
        charge,
        multiplicity,
        parse_reference(reference)?,
        parse_variant(method)?,
        field,
    )?;
    if periodic_run && long_range_cutoff.is_some() {
        return Err(PyValueError::new_err(
            "long_range_cutoff is the molecular near-field split; the periodic \
             divide-and-conquer path takes its long range from the lattice sum and would ignore \
             it. Drop long_range_cutoff, or drop the cell.",
        ));
    }
    let dc = crate::dc::DcOptions {
        core_radius: core_radius * ANGSTROM_TO_BOHR,
        buffer_radius: buffer_radius * ANGSTROM_TO_BOHR,
        smearing_ev,
        long_range_cutoff: long_range_cutoff.map(|r| r * ANGSTROM_TO_BOHR),
        ..crate::dc::DcOptions::default()
    };

    // eV/Bohr → eV/Å, and the sign flip from gradient to force, as everywhere else here.
    let scale = ANGSTROM_TO_BOHR;
    let d = PyDict::new(py);
    if periodic_run {
        let g = crate::dc::dc_periodic_gradient(
            &mol,
            &params,
            &options,
            &crate::pbc::gamma::PeriodicOptions::default(),
            &dc,
        )
        .map_err(to_py_err)?;
        let forces: Vec<Vec<f64>> = g
            .gradient
            .iter()
            .map(|v| vec![-v.x * scale, -v.y * scale, -v.z * scale])
            .collect();
        d.set_item("energy_ev", g.energy_ev)?;
        d.set_item("energy_hartree", g.energy_ev * EV_TO_HARTREE)?;
        d.set_item("forces_ev_per_angstrom", forces)?;
        d.set_item("charges", g.scf.charges.clone())?;
        d.set_item("heat_of_formation_kcal", g.scf.heat_of_formation_kcal)?;
        d.set_item("converged", g.scf.converged)?;
        d.set_item("gamma_margin_bohr", g.scf.gamma_margin)?;
        d.set_item("n_subsystems", g.scf.n_subsystems)?;
        match g.stress {
            Some(s) => d.set_item("stress_ev_per_angstrom3", voigt_ev_per_angstrom3(&s))?,
            None => d.set_item("stress_ev_per_angstrom3", None::<Vec<f64>>)?,
        }
    } else {
        let g = crate::dc::dc_gradient(&mol, &params, &options, &dc).map_err(to_py_err)?;
        let forces: Vec<Vec<f64>> = g
            .gradient
            .iter()
            .map(|v| vec![-v.x * scale, -v.y * scale, -v.z * scale])
            .collect();
        d.set_item("energy_ev", g.energy_ev)?;
        d.set_item("energy_hartree", g.energy_ev * EV_TO_HARTREE)?;
        d.set_item("forces_ev_per_angstrom", forces)?;
        d.set_item("charges", g.scf.charges.clone())?;
        d.set_item("heat_of_formation_kcal", g.scf.heat_of_formation_kcal)?;
        d.set_item("converged", g.scf.converged)?;
        d.set_item("stress_ev_per_angstrom3", None::<Vec<f64>>)?;
        d.set_item("n_subsystems", g.scf.n_subsystems)?;
    }
    Ok(d.into())
}

/// Berry-phase polarization: the modern theory, in `e/Bohr^2`.
///
/// `strings` is the number of k-points along each Brillouin-zone string and is the convergence
/// parameter -- the answer must become independent of it. `kpts` is the transverse sampling.
///
/// Polarization is defined only **modulo** `quantum`, which is `e a/V` along each lattice vector.
/// That is physics, not a defect: a different branch assigns the electrons to a different unit
/// cell. Only *differences* are meaningful, and a plain subtraction of two `total` values is off
/// by exactly one quantum whenever the two landed on different branches -- reduce with
/// `difference` instead, which this returns the ingredients for.
///
/// Three-dimensional cells only. Present as an independent check on `born_charges`: it reaches
/// the same `Z*` through overlaps between neighbouring k-points with no response equation
/// anywhere in it. The two differ by the intra-atomic `s`-`p` moment the phase cannot carry.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, strings=12, kpts=None, pbc=None, charge=0.0,
                    multiplicity=1, reference="auto", method="pm3"))]
#[allow(clippy::too_many_arguments)]
fn berry_polarization(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    strings: usize,
    kpts: Option<Vec<usize>>,
    pbc: Option<Vec<bool>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
) -> PyResult<PyObject> {
    let mut mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    attach_cell(&mut mol, Some(cell), pbc)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let opts = options(
        charge,
        multiplicity,
        parse_reference(reference)?,
        parse_variant(method)?,
    );
    let periodic = crate::pbc::gamma::PeriodicOptions::default();
    let kopt = match kpts {
        Some(divisions) => {
            if divisions.len() != 3 {
                return Err(PyValueError::new_err("kpts takes three divisions"));
            }
            crate::KpointOptions::mesh([divisions[0], divisions[1], divisions[2]])
        }
        None => crate::KpointOptions::mesh([1, 1, 1]),
    };
    let p = crate::pbc::berry::berry_polarization(&mol, &params, &opts, &periodic, &kopt, strings)
        .map_err(to_py_err)?;

    let vector = |v: crate::Vec3| vec![v.x, v.y, v.z];
    let d = PyDict::new(py);
    d.set_item("total", vector(p.total))?;
    d.set_item("electronic", vector(p.electronic))?;
    d.set_item("ionic", vector(p.ionic))?;
    d.set_item("phase", p.phase.to_vec())?;
    d.set_item(
        "quantum",
        p.quantum.iter().map(|q| vector(*q)).collect::<Vec<_>>(),
    )?;
    d.set_item("string_length", p.string_length)?;
    d.set_item("gamma_margin_bohr", gamma_margin(&mol, &periodic))?;
    Ok(d.into())
}

/// Phonon dispersion from supercell force constants: one Hessian for the whole path.
///
/// `supercell` is the replication the force constants are cut out of. `path` is a list of
/// fractional wavevectors; `points` interpolates that many per segment.
///
/// A supercell's Gamma point *is* a mesh of the primitive cell, so what this reproduces is the
/// mesh-sampled response with the matching mesh -- not a Gamma-only one.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, supercell, path, points=12, pbc=None,
                    charge=0.0, multiplicity=1, reference="auto", method="pm3",
                    enforce_asr=false))]
#[allow(clippy::too_many_arguments)]
fn phonon_bands(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    supercell: Vec<usize>,
    path: Vec<Vec<f64>>,
    points: usize,
    pbc: Option<Vec<bool>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    enforce_asr: bool,
) -> PyResult<PyObject> {
    let mut mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    attach_cell(&mut mol, Some(cell), pbc)?;
    if supercell.len() != 3 {
        return Err(PyValueError::new_err("supercell takes three repeats"));
    }
    let corners: Vec<[f64; 3]> = path
        .iter()
        .map(|q| {
            if q.len() != 3 {
                Err(PyValueError::new_err(
                    "each point on the path takes three fractional components",
                ))
            } else {
                Ok([q[0], q[1], q[2]])
            }
        })
        .collect::<PyResult<_>>()?;
    if corners.len() < 2 {
        return Err(PyValueError::new_err(
            "a path needs at least two corners to run between",
        ));
    }
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let opts = options(
        charge,
        multiplicity,
        parse_reference(reference)?,
        parse_variant(method)?,
    );
    let periodic = crate::pbc::gamma::PeriodicOptions::default();
    let mut constants = crate::ForceConstants::from_supercell(
        &mol,
        &params,
        &opts,
        &periodic,
        [supercell[0], supercell[1], supercell[2]],
    )
    .map_err(to_py_err)?;
    // Reported before it can be imposed: it is what the truncation threw away, and flattening it
    // unseen would hide force constants that are simply wrong.
    let residual = constants.acoustic_sum_rule_residual();
    if enforce_asr {
        constants.enforce_acoustic_sum_rule();
    }
    let points_on_path = crate::q_path(&corners, points);
    let bands = constants
        .band_structure(&points_on_path)
        .map_err(to_py_err)?;

    let d = PyDict::new(py);
    d.set_item(
        "q",
        points_on_path
            .iter()
            .map(|q| q.to_vec())
            .collect::<Vec<_>>(),
    )?;
    d.set_item("frequencies_cm", bands)?;
    d.set_item("supercell", constants.supercell().to_vec())?;
    d.set_item("acoustic_sum_rule_residual", residual)?;
    d.set_item("gamma_margin_bohr", gamma_margin(&mol, &periodic))?;
    Ok(d.into())
}

/// A finite electric field along a periodic direction, by the Berry-phase electric enthalpy.
///
/// `field` is in eV per (e*Bohr) and may point anywhere; a component along a lattice vector is
/// what needs this machinery, and one orthogonal to all of them is an ordinary calculation.
/// `kpts` is the mesh, and its division along each field direction is that direction's string
/// length -- the convergence parameter the answer must become independent of.
///
/// Restricted, gapped, three-dimensional cells only. `-grad(F)` is not a force: this returns the
/// state and its polarization, and the derivative of the enthalpy is not implemented.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, field, kpts, pbc=None, charge=0.0,
                    multiplicity=1, reference="auto", method="pm3", tol=1.0e-8,
                    max_iter=60, mixing=0.5))]
#[allow(clippy::too_many_arguments)]
fn finite_field(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    field: Vec<f64>,
    kpts: Vec<usize>,
    pbc: Option<Vec<bool>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    tol: f64,
    max_iter: usize,
    mixing: f64,
) -> PyResult<PyObject> {
    let mut mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    attach_cell(&mut mol, Some(cell), pbc)?;
    if field.len() != 3 {
        return Err(PyValueError::new_err(
            "field takes three Cartesian components",
        ));
    }
    if kpts.len() != 3 {
        return Err(PyValueError::new_err("kpts takes three divisions"));
    }
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let opts = options(
        charge,
        multiplicity,
        parse_reference(reference)?,
        parse_variant(method)?,
    );
    let periodic = crate::pbc::gamma::PeriodicOptions::default();
    let ff = crate::pbc::finite_field::FiniteFieldOptions {
        tol,
        max_iter,
        mixing,
    };
    let result = crate::pbc::finite_field::run_finite_field(
        &mol,
        &params,
        &opts,
        &periodic,
        [kpts[0], kpts[1], kpts[2]],
        crate::Vec3::new(field[0], field[1], field[2]),
        &ff,
    )
    .map_err(to_py_err)?;

    let vector = |v: crate::Vec3| vec![v.x, v.y, v.z];
    let d = PyDict::new(py);
    d.set_item("energy", result.scf.total_ev)?;
    d.set_item("enthalpy_ev", result.enthalpy_ev)?;
    d.set_item("polarization", vector(result.polarization))?;
    d.set_item(
        "electronic_polarization",
        vector(result.electronic_polarization),
    )?;
    d.set_item("ionic_polarization", vector(result.ionic_polarization))?;
    d.set_item("phase", result.phase.to_vec())?;
    d.set_item("field", vector(result.field))?;
    d.set_item("iterations", result.iterations)?;
    d.set_item("converged", result.converged)?;
    // Which axes the mesh had three k-points on. An unresolved axis contributes zero to `phase`
    // and to `electronic_polarization`, which is not the same as its contribution being zero --
    // so this travels beside them rather than being left to be inferred.
    d.set_item("resolved", result.resolved.to_vec())?;
    d.set_item("gamma_margin_bohr", gamma_margin(&mol, &periodic))?;
    Ok(d.into())
}

/// Born effective charges: the dipole a cell acquires per unit displacement of one atom.
///
/// One `3 × 3` tensor per atom in elementary charges, `tensor[alpha][beta] = d(V P_alpha)/du_beta`.
/// Periodic only — an isolated molecule's equivalent is the atomic polar tensor from
/// `dipole(...)["derivatives_e"]`.
///
/// `sum_rule_residual` is the largest `|sum_a Z*_a|`, which is zero for an exact response because
/// translating the whole crystal produces no dipole. It is reported rather than enforced: it is
/// the number that says whether the coupled-perturbed solve converged, and flattening it silently
/// would hide exactly that. Pass `enforce=True` to have the mean violation removed afterwards.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, pbc=None, charge=0.0, multiplicity=1,
                    reference="auto", method="pm3", enforce=false))]
#[allow(clippy::too_many_arguments)]
fn born_charges(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    pbc: Option<Vec<bool>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    enforce: bool,
) -> PyResult<PyObject> {
    let mut mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    attach_cell(&mut mol, Some(cell), pbc)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let opts = options(
        charge,
        multiplicity,
        parse_reference(reference)?,
        parse_variant(method)?,
    );
    let periodic = crate::pbc::gamma::PeriodicOptions::default();
    let mut born =
        crate::pbc::born::born_charges(&mol, &params, &opts, &periodic).map_err(to_py_err)?;
    let residual = crate::pbc::born::born_charge_sum_rule_residual(&born);
    if enforce {
        crate::pbc::born::enforce_born_sum_rule(&mut born);
    }
    let tensors: Vec<Vec<Vec<f64>>> = born
        .iter()
        .map(|z| z.iter().map(|row| row.to_vec()).collect())
        .collect();
    let d = PyDict::new(py);
    d.set_item("born_charges", tensors)?;
    d.set_item("sum_rule_residual", residual)?;
    d.set_item("gamma_margin_bohr", gamma_margin(&mol, &periodic))?;
    Ok(d.into())
}

/// The Γ-point validity margin: the narrowest periodic width minus the short-range cutoff.
///
/// Purely geometric, so it costs nothing to report next to a response that depends on it. It
/// belongs next to one because a negative margin does not make the SCF fail -- it converges
/// cleanly to a well-defined wrong answer, and the response built on that answer inherits it
/// without any other sign. `periodic_single_point` has always returned this; the response
/// entry points return it for the same reason.
fn gamma_margin(mol: &Molecule, periodic: &crate::pbc::gamma::PeriodicOptions) -> f64 {
    mol.cell
        .map(|cell| {
            cell.periodic_widths()
                .into_iter()
                .fold(f64::INFINITY, |narrowest, (_, width)| narrowest.min(width))
                - periodic.short_range_cutoff
        })
        .unwrap_or(f64::NAN)
}

/// Electronic polarizability and, for a fully periodic cell, the dielectric tensor.
///
/// `polarizability` is `alpha` in Bohr^3 and is available in every dimensionality. `epsilon` is
/// `1 + 4*pi*alpha/V` and is `None` for a chain or a slab, where `V` would be a length or an area
/// and dividing by a supercell's vacuum padding would make the answer a statement about the
/// padding rather than about the material.
///
/// `epsilon` is the **electronic** (clamped-ion, high-frequency) response, `ε∞`.
///
/// `include_ionic=True` adds `epsilon_static` -- the static constant `ε₀`, which lets the nuclei
/// relax along each infrared-active mode. It is off by default because it costs a Γ-point phonon
/// calculation and a set of Born charges on top of the field response, and because it is only
/// meaningful at a **relaxed geometry**: `skipped_modes` counts the modes with `ω² ≤ 0` that were
/// left out, and anything above the expected three acoustic ones means the structure is not a
/// minimum and the ionic term is missing whatever those modes would have contributed.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, pbc=None, charge=0.0, multiplicity=1,
                    reference="auto", method="pm3", include_ionic=false))]
#[allow(clippy::too_many_arguments)]
fn dielectric(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    pbc: Option<Vec<bool>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    include_ionic: bool,
) -> PyResult<PyObject> {
    let mut mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    attach_cell(&mut mol, Some(cell), pbc)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let opts = options(
        charge,
        multiplicity,
        parse_reference(reference)?,
        parse_variant(method)?,
    );
    let periodic = crate::pbc::gamma::PeriodicOptions::default();
    let alpha = crate::pbc::dielectric::polarizability(&mol, &params, &opts, &periodic)
        .map_err(to_py_err)?;
    let rows = |m: [[f64; 3]; 3]| -> Vec<Vec<f64>> { m.iter().map(|r| r.to_vec()).collect() };

    let d = PyDict::new(py);
    d.set_item("polarizability", rows(alpha))?;
    d.set_item("gamma_margin_bohr", gamma_margin(&mol, &periodic))?;
    // `None` rather than an error for a low-dimensional cell: the caller asked for both and one
    // of them exists. The Rust `dielectric_tensor` refuses instead, because there the caller
    // named the one that does not.
    match crate::pbc::dielectric::dielectric_tensor(&mol, &params, &opts, &periodic) {
        Ok(tensors) => d.set_item("epsilon", rows(tensors.epsilon))?,
        Err(_) => d.set_item("epsilon", None::<Vec<Vec<f64>>>)?,
    }

    // The static tensor is asked for, so a low-dimensional cell is an error here rather than a
    // `None` -- unlike `epsilon` above, which the caller got by asking for the pair.
    if include_ionic {
        let s = crate::pbc::dielectric::static_dielectric_tensor(&mol, &params, &opts, &periodic)
            .map_err(to_py_err)?;
        d.set_item("epsilon_static", rows(s.epsilon))?;
        d.set_item("epsilon_electronic", rows(s.electronic))?;
        d.set_item("epsilon_ionic", rows(s.ionic))?;
        // Reported next to the tensor it damages rather than left to be inferred from a
        // suspicious number: three is the acoustic branch, more than three is not a minimum.
        d.set_item("skipped_modes", s.skipped_modes)?;
    }
    Ok(d.into())
}

/// Infrared spectrum: frequencies, intensities, and the dipole-derivative tensor.
///
/// Evaluate at a **stationary point** (optimize first), as for `frequencies`. The tensor is
/// returned alongside the per-mode intensities because it is the mode-independent object — an
/// isotope substitution or a different projection reuses it without solving the response again.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto",
                    method="pm3", field=None))]
#[allow(clippy::too_many_arguments)]
fn ir_spectrum(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    field: Option<Vec<f64>>,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let opts = options_with_field(
        charge,
        multiplicity,
        parse_reference(reference)?,
        parse_variant(method)?,
        field,
    )?;
    let s = crate::ir::ir_spectrum(&mol, &params, &opts, 1.0e-3).map_err(to_py_err)?;

    // The tensor is in elementary charges, which is the same number whether the length is Bohr
    // or Angstrom — a dipole derivative is a charge. Reported as e/Angstrom for the unit a
    // spectroscopist expects to see next to a km/mol.
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(3);
    for axis in 0..3 {
        rows.push(
            (0..s.dipole_derivatives.cols)
                .map(|dof| s.dipole_derivatives[(axis, dof)])
                .collect(),
        );
    }
    let d = PyDict::new(py);
    d.set_item("frequencies_cm", s.frequencies_cm)?;
    d.set_item("intensities_km_per_mol", s.intensities_km_per_mol)?;
    d.set_item("dipole_derivatives_e", rows)?;
    d.set_item("ndof", s.modes.frequencies_cm.len())?;
    Ok(d.into())
}

/// The dipole operator and everything derived from it.
///
/// `pm3_rs::dipole` is one matrix used twice — the reported dipole and the external field's
/// coupling are the same `M_α` — so exposing it exposes the thing both rest on rather than two
/// numbers that happen to agree.
///
/// Returns `dipole_debye` and `dipole_e_angstrom` (about the centre of mass, MOPAC's
/// convention), `centre_of_mass_angstrom`, the `3 × 3N` derivative tensor `derivatives_e` in
/// elementary charges, and — when `operator=True` — the three `nao × nao` moment matrices
/// themselves as `operator_bohr`, in Bohr about that same centre.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto",
                    method="pm3", field=None, operator=false))]
#[allow(clippy::too_many_arguments)]
fn dipole(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    field: Option<Vec<f64>>,
    operator: bool,
) -> PyResult<PyObject> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let opts = options_with_field(
        charge,
        multiplicity,
        parse_reference(reference)?,
        parse_variant(method)?,
        field,
    )?;
    let scf = run_pm3(&mol, &params, &opts).map_err(to_py_err)?;
    let basis = crate::basis::Basis::build(&mol, &params).map_err(to_py_err)?;
    let com = crate::dipole::centre_of_mass(&mol, &params).map_err(to_py_err)?;
    let tensor = crate::ir::dipole_derivatives_at(&mol, &params, &opts, &scf).map_err(to_py_err)?;

    let d = PyDict::new(py);
    d.set_item(
        "dipole_debye",
        [scf.dipole_debye.x, scf.dipole_debye.y, scf.dipole_debye.z],
    )?;
    // 1 Debye = 0.2081943 e·Å.
    let to_e_angstrom = 0.2081943;
    d.set_item(
        "dipole_e_angstrom",
        [
            scf.dipole_debye.x * to_e_angstrom,
            scf.dipole_debye.y * to_e_angstrom,
            scf.dipole_debye.z * to_e_angstrom,
        ],
    )?;
    d.set_item(
        "centre_of_mass_angstrom",
        [
            com.x * BOHR_TO_ANGSTROM,
            com.y * BOHR_TO_ANGSTROM,
            com.z * BOHR_TO_ANGSTROM,
        ],
    )?;
    let rows: Vec<Vec<f64>> = (0..3)
        .map(|axis| (0..tensor.cols).map(|dof| tensor[(axis, dof)]).collect())
        .collect();
    d.set_item("derivatives_e", rows)?;
    if operator {
        let matrices =
            crate::dipole::dipole_matrix(&mol, &params, &basis, com).map_err(to_py_err)?;
        let packed: Vec<Vec<Vec<f64>>> = matrices
            .iter()
            .map(|m| {
                (0..m.rows)
                    .map(|i| (0..m.cols).map(|j| m[(i, j)]).collect())
                    .collect()
            })
            .collect();
        d.set_item("operator_bohr", packed)?;
    }
    Ok(d.into())
}

/// The dynamical matrix at a wavevector, as a matrix rather than as frequencies.
///
/// `phonons(q=…)` gives the frequencies; this gives what they are diagonalized from, which is
/// what a caller building a dispersion, applying their own masses, or checking an identity
/// wants. `real` and `imag` are `3N × 3N` in eV/Bohr²; `hermitian_defect` is the largest
/// departure from `D(q)† = D(q)` **before** the matrix was symmetrized, reported so a wrong
/// assembly is visible rather than averaged away.
///
/// `rigid_ion=True` leaves the electronic response out, giving the fixed-density part alone.
#[pyfunction]
#[pyo3(signature = (numbers, positions, cell, q, pbc=None, kpts=None, charge=0.0,
                    multiplicity=1, reference="auto", method="pm3", rigid_ion=false,
                    smearing_ev=0.0))]
#[allow(clippy::too_many_arguments)]
fn dynamical_matrix(
    py: Python<'_>,
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    cell: Vec<Vec<f64>>,
    q: Vec<f64>,
    pbc: Option<Vec<bool>>,
    kpts: Option<Vec<usize>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    rigid_ion: bool,
    smearing_ev: f64,
) -> PyResult<PyObject> {
    let mut mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    attach_cell(&mut mol, Some(cell), pbc)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let opts = options(
        charge,
        multiplicity,
        parse_reference(reference)?,
        parse_variant(method)?,
    );
    if q.len() != 3 {
        return Err(PyValueError::new_err(
            "q must have three fractional components, one per reciprocal lattice vector",
        ));
    }
    let q_frac = [q[0], q[1], q[2]];
    let periodic = crate::pbc::gamma::PeriodicOptions::default();
    let result = if rigid_ion {
        if kpts.is_some() {
            return Err(PyValueError::new_err(
                "the rigid-ion matrix has no electronic response, so a k-mesh would change \
                 nothing; leave kpts unset",
            ));
        }
        crate::pbc::dfpt::rigid_ion_dynamical_matrix(&mol, &params, &opts, &periodic, q_frac)
    } else if kpts.is_some() {
        let kopt = crate::pbc::kscf::KpointOptions {
            spec: parse_kpoints(kpts)?,
            smearing_ev,
            ..Default::default()
        };
        crate::pbc::dfpt::dynamical_matrix_on_mesh(&mol, &params, &opts, &periodic, &kopt, q_frac)
    } else {
        crate::pbc::dfpt::dynamical_matrix(&mol, &params, &opts, &periodic, q_frac)
    }
    .map_err(to_py_err)?;

    let n = result.matrix.rows;
    let real: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| result.matrix[(i, j)].re).collect())
        .collect();
    let imaginary: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| result.matrix[(i, j)].im).collect())
        .collect();
    // The frequencies and the masses that produce them travel with the matrix. Without the
    // masses a Python caller cannot mass-weight `D(q)` at all — the isotope-averaged values are
    // the crate's, not `ase`'s — so returning the matrix alone made the frequencies at a
    // wavevector unreachable from anywhere but Rust.
    let frequencies = crate::pbc::dfpt::frequencies_of(&result).map_err(to_py_err)?;
    let d = PyDict::new(py);
    d.set_item("real", real)?;
    d.set_item("imag", imaginary)?;
    d.set_item("masses", result.masses.clone())?;
    d.set_item("frequencies_cm", frequencies)?;
    d.set_item("hermitian_defect", result.hermitian_defect)?;
    d.set_item("q", q_frac.to_vec())?;
    Ok(d.into())
}

/// The converged wavefunction as a Molden document.
///
/// Returned as a string rather than written to a file: the caller decides where it goes, and a
/// pure function is what the rest of this surface is. Molecular only — a periodic wavefunction
/// has no single set of molecular orbitals to write.
#[pyfunction]
#[pyo3(signature = (numbers, positions, charge=0.0, multiplicity=1, reference="auto",
                    method="pm3", field=None))]
#[allow(clippy::too_many_arguments)]
fn molden(
    numbers: Vec<u8>,
    positions: Vec<Vec<f64>>,
    charge: f64,
    multiplicity: usize,
    reference: &str,
    method: &str,
    field: Option<Vec<f64>>,
) -> PyResult<String> {
    let mol = build_molecule(&numbers, &positions, charge, multiplicity)?;
    let params = Pm3Parameters::standard().map_err(to_py_err)?;
    let result = run_pm3(
        &mol,
        &params,
        &options_with_field(
            charge,
            multiplicity,
            parse_reference(reference)?,
            parse_variant(method)?,
            field,
        )?,
    )
    .map_err(to_py_err)?;
    crate::molden::molden_string(&mol, &params, &result).map_err(to_py_err)
}

/// Run the `pm3-rs` command line, returning its exit code.
///
/// The same [`crate::cli::main_with_args`] the standalone executable calls, so a `pip install`
/// puts the identical interface on the path rather than a Python reimplementation that would
/// drift from it. `argv[0]` is the program name and is skipped, matching `sys.argv`.
#[pyfunction]
fn cli_main(argv: Vec<String>) -> i32 {
    crate::cli::main_with_args(&argv)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(single_point, m)?)?;
    m.add_function(wrap_pyfunction!(gradient, m)?)?;
    m.add_function(wrap_pyfunction!(forces, m)?)?;
    m.add_function(wrap_pyfunction!(optimize, m)?)?;
    m.add_function(wrap_pyfunction!(frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(hessian, m)?)?;
    m.add_function(wrap_pyfunction!(periodic_single_point, m)?)?;
    m.add_function(wrap_pyfunction!(periodic_forces, m)?)?;
    m.add_function(wrap_pyfunction!(phonons, m)?)?;
    m.add_function(wrap_pyfunction!(bands, m)?)?;
    m.add_function(wrap_pyfunction!(relax, m)?)?;
    m.add_function(wrap_pyfunction!(dipole, m)?)?;
    m.add_function(wrap_pyfunction!(dynamical_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(divide_and_conquer, m)?)?;
    m.add_function(wrap_pyfunction!(divide_and_conquer_forces, m)?)?;
    m.add_function(wrap_pyfunction!(born_charges, m)?)?;
    m.add_function(wrap_pyfunction!(dielectric, m)?)?;
    m.add_function(wrap_pyfunction!(finite_field, m)?)?;
    m.add_function(wrap_pyfunction!(berry_polarization, m)?)?;
    m.add_function(wrap_pyfunction!(phonon_bands, m)?)?;
    m.add_function(wrap_pyfunction!(ir_spectrum, m)?)?;
    m.add_function(wrap_pyfunction!(molden, m)?)?;
    m.add_function(wrap_pyfunction!(cli_main, m)?)?;
    Ok(())
}
