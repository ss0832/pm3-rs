// SPDX-License-Identifier: GPL-3.0-or-later

//! Geometry and variable-cell optimization for periodic structures.
//!
//! L-BFGS over the atomic positions and, optionally, the cell. The cell degrees of freedom enter
//! as a **strain** rather than as the lattice vectors themselves: a step `δε` maps
//! `h → (1 + δε) h` and `r → (1 + δε) r` together, so the atoms ride with the cell and a pure
//! cell step never shears the contents out of position. The gradient conjugate to that variable
//! is the virial, which [`crate::pbc::gradient`] already returns.
//!
//! # Why the strain is scaled
//!
//! Strain is dimensionless and a position is a length, so packing the two into one L-BFGS vector
//! puts quantities with different units under a single scalar step length. They are not merely
//! different units but different *magnitudes*: for a water cell the virial runs to tens of eV
//! while the forces are a couple of eV/Bohr, so the initial inverse-Hessian guess — which scales
//! against the largest component of the whole vector — is set by the virial and makes the atomic
//! steps ten times too small, while the strain steps it does take are several percent per
//! iteration. The line search then spends most of its evaluations backtracking.
//!
//! So the cell variable is `u = L·ε` with `L = V^{1/3}` from the starting cell, which is a length
//! like every other variable, and its conjugate gradient is `∂H/∂ε / L`. `L` is deliberately held
//! fixed rather than tracking the current volume: a metric that moved with the variables would
//! change the meaning of the accumulated L-BFGS history at every step.
//!
//! # Pressure
//!
//! At a target pressure the quantity being minimized is the enthalpy `H = E + P V`, whose strain
//! derivative is `∂E/∂ε + P V δ_αβ` — the `V` because `∂V/∂ε_αβ = V δ_αβ`. Setting `pressure` to
//! zero recovers plain energy minimization, where convergence means the stress has vanished.
//!
//! # Reduced dimensionality
//!
//! Every periodic dimensionality now reports a virial, and the strain degrees of freedom follow
//! the cell: a chain relaxes its axis, a slab its two in-plane vectors, a crystal all nine
//! components. The projection is the one
//! [`crate::pbc::gradient::forces_and_stress`] already applies, so a non-periodic direction has
//! stress exactly zero and the optimizer never moves it. An isolated cell has no strain at all,
//! and a variable-cell run there is still refused. Fixed-cell relaxation works everywhere.

use crate::error::{Pm3Error, Result};
use crate::math::{Mat3, Vec3};
use crate::params::Pm3Parameters;
use crate::pbc::gamma::PeriodicOptions;
use crate::pbc::gradient::{periodic_gradient, PeriodicGradient};
use crate::scf::Pm3Options;
use crate::system::Molecule;

/// What is allowed to move.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CellRelaxation {
    /// Atoms only; the cell is held fixed.
    #[default]
    Fixed,
    /// Atoms and every component of the strain.
    Variable,
}

#[derive(Clone, Copy, Debug)]
pub struct PeriodicOptOptions {
    pub max_iter: usize,
    /// Convergence on the largest force component (eV/Bohr).
    pub gtol: f64,
    /// Convergence on the largest stress component (eV/Bohr³). Ignored for a fixed cell.
    pub stress_tol: f64,
    /// L-BFGS history length.
    pub history: usize,
    /// External pressure (eV/Bohr³). Minimizes `E + P V` when non-zero.
    pub pressure: f64,
    pub cell: CellRelaxation,
}

impl Default for PeriodicOptOptions {
    fn default() -> Self {
        Self {
            max_iter: 200,
            gtol: 1.0e-3,
            stress_tol: 1.0e-6,
            history: 8,
            pressure: 0.0,
            cell: CellRelaxation::Fixed,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PeriodicOptResult {
    pub molecule: Molecule,
    pub gradient: PeriodicGradient,
    pub converged: bool,
    pub iterations: usize,
}

/// Relax a periodic structure.
pub fn relax(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    periodic: &PeriodicOptions,
    opt: &PeriodicOptOptions,
) -> Result<PeriodicOptResult> {
    let variable_cell = opt.cell == CellRelaxation::Variable;
    let cell = molecule
        .cell
        .ok_or_else(|| Pm3Error::InvalidInput("relax needs a periodic cell".to_string()))?;
    if variable_cell && cell.n_periodic() == 0 {
        return Err(Pm3Error::InvalidInput(
            "variable-cell relaxation needs a periodic cell: an isolated system has no strain, \
             so there is nothing for the cell degrees of freedom to minimize"
                .to_string(),
        ));
    }

    let nat = molecule.atoms.len();
    let n_strain = if variable_cell { 9 } else { 0 };
    let ndof = 3 * nat + n_strain;
    // The metric that puts the strain on the same footing as a coordinate; see the module note.
    // Taken from the starting cell and then left alone.
    let length_scale = if variable_cell {
        cell.measure().cbrt()
    } else {
        1.0
    };
    let mut current = molecule.clone();
    let mut evaluation = periodic_gradient(&current, params, options, periodic)?;
    let mut g = pack_gradient(
        &evaluation,
        nat,
        variable_cell,
        opt.pressure,
        &current,
        length_scale,
    );
    let mut value = objective(&evaluation, opt.pressure, &current);

    let mut s_history: Vec<Vec<f64>> = Vec::new();
    let mut y_history: Vec<Vec<f64>> = Vec::new();
    let mut rho_history: Vec<f64> = Vec::new();
    let mut converged = is_converged(&evaluation, opt, variable_cell);
    let mut iterations = 0;

    for iteration in 0..opt.max_iter {
        iterations = iteration + 1;
        if converged {
            iterations = iteration;
            break;
        }

        // L-BFGS two-loop recursion.
        let mut q = g.clone();
        let m = s_history.len();
        let mut alpha = vec![0.0; m];
        for i in (0..m).rev() {
            let a = rho_history[i] * dot(&s_history[i], &q);
            alpha[i] = a;
            axpy(&mut q, -a, &y_history[i]);
        }
        let gamma = if m > 0 {
            let sy = dot(&s_history[m - 1], &y_history[m - 1]);
            let yy = dot(&y_history[m - 1], &y_history[m - 1]);
            if yy > 0.0 {
                sy / yy
            } else {
                1.0
            }
        } else {
            0.05 / g.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1.0e-6)
        };
        for v in q.iter_mut() {
            *v *= gamma;
        }
        for i in 0..m {
            let beta = rho_history[i] * dot(&y_history[i], &q);
            axpy(&mut q, alpha[i] - beta, &s_history[i]);
        }
        let mut direction: Vec<f64> = q.iter().map(|v| -v).collect();
        if dot(&direction, &g) > 0.0 {
            direction = g.iter().map(|v| -v).collect();
        }

        // Backtracking Armijo line search on the objective.
        let slope = dot(&g, &direction);
        let mut step = 1.0;
        let mut accepted = None;
        loop {
            let trial = apply_step(&current, &direction, step, nat, variable_cell, length_scale);
            if let Ok(trial_gradient) = periodic_gradient(&trial, params, options, periodic) {
                let trial_value = objective(&trial_gradient, opt.pressure, &trial);
                if trial_value <= value + 1.0e-4 * step * slope {
                    accepted = Some((trial, trial_gradient, trial_value, step));
                    break;
                }
            }
            step *= 0.5;
            if step < 1.0e-9 {
                break;
            }
        }
        let Some((next, next_gradient, next_value, accepted_step)) = accepted else {
            break;
        };

        let s: Vec<f64> = direction.iter().map(|d| accepted_step * d).collect();
        let next_g = pack_gradient(
            &next_gradient,
            nat,
            variable_cell,
            opt.pressure,
            &next,
            length_scale,
        );
        let y: Vec<f64> = (0..ndof).map(|i| next_g[i] - g[i]).collect();
        let sy = dot(&s, &y);
        if sy > 1.0e-12 {
            s_history.push(s);
            y_history.push(y);
            rho_history.push(1.0 / sy);
            if s_history.len() > opt.history {
                s_history.remove(0);
                y_history.remove(0);
                rho_history.remove(0);
            }
        }

        current = next;
        evaluation = next_gradient;
        g = next_g;
        value = next_value;
        converged = is_converged(&evaluation, opt, variable_cell);
    }

    Ok(PeriodicOptResult {
        molecule: current,
        gradient: evaluation,
        converged,
        iterations,
    })
}

/// `E + P V`, the quantity actually minimized.
fn objective(evaluation: &PeriodicGradient, pressure: f64, molecule: &Molecule) -> f64 {
    let volume = molecule.cell.map_or(1.0, |c| c.measure());
    evaluation.energy_ev + pressure * volume
}

/// Forces and, for a variable cell, the enthalpy's strain derivative, in one flat vector.
fn pack_gradient(
    evaluation: &PeriodicGradient,
    nat: usize,
    variable_cell: bool,
    pressure: f64,
    molecule: &Molecule,
    length_scale: f64,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(3 * nat + if variable_cell { 9 } else { 0 });
    for atom in 0..nat {
        out.push(evaluation.gradient[atom].x);
        out.push(evaluation.gradient[atom].y);
        out.push(evaluation.gradient[atom].z);
    }
    if variable_cell {
        let virial = evaluation
            .virial
            .expect("variable-cell relaxation is restricted to 3D, where the virial exists");
        let volume = molecule.cell.map_or(1.0, |c| c.measure());
        for beta in 0..3 {
            for alpha in 0..3 {
                // ∂(E + PV)/∂ε_αβ = virial_αβ + P V δ_αβ, then divided by the metric because the
                // variable is `u = L·ε` rather than `ε` itself.
                let pressure_term = if alpha == beta {
                    pressure * volume
                } else {
                    0.0
                };
                out.push((virial.col[beta].get(alpha) + pressure_term) / length_scale);
            }
        }
    }
    out
}

/// Take a step: atoms move directly, the cell moves through a strain that carries the atoms with
/// it so a pure cell step is a similarity transform of the contents.
fn apply_step(
    molecule: &Molecule,
    direction: &[f64],
    step: f64,
    nat: usize,
    variable_cell: bool,
    length_scale: f64,
) -> Molecule {
    let mut out = molecule.clone();
    if variable_cell {
        let mut strain = Mat3::zero();
        for beta in 0..3 {
            for alpha in 0..3 {
                // The variable is `u = L·ε`, so the strain the step asks for is `δu / L`.
                let value = step * direction[3 * nat + 3 * beta + alpha] / length_scale;
                match alpha {
                    0 => strain.col[beta].x = value,
                    1 => strain.col[beta].y = value,
                    _ => strain.col[beta].z = value,
                }
            }
        }
        if let Some(cell) = out.cell {
            out.cell = Some(cell.strained(&strain));
        }
        for atom in &mut out.atoms {
            atom.position += strain.mul_vec(atom.position);
        }
    }
    for (atom, slot) in out.atoms.iter_mut().enumerate() {
        let base = Vec3::new(
            direction[3 * atom],
            direction[3 * atom + 1],
            direction[3 * atom + 2],
        );
        slot.position += base * step;
    }
    out
}

fn is_converged(
    evaluation: &PeriodicGradient,
    opt: &PeriodicOptOptions,
    variable_cell: bool,
) -> bool {
    if evaluation.max_gradient >= opt.gtol {
        return false;
    }
    if !variable_cell {
        return true;
    }
    match evaluation.stress {
        Some(stress) => (0..3).all(|beta| {
            (0..3).all(|alpha| {
                (stress.col[beta].get(alpha) + opt.pressure * indicator(alpha, beta)).abs()
                    < opt.stress_tol
            })
        }),
        None => false,
    }
}

#[inline]
fn indicator(alpha: usize, beta: usize) -> f64 {
    if alpha == beta {
        1.0
    } else {
        0.0
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for (slot, value) in y.iter_mut().zip(x) {
        *slot += a * value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::Cell;

    /// Cell edges here stay above [`crate::pbc::gamma::DEFAULT_SHORT_RANGE_CUTOFF`] so that no
    /// image sits inside the exchange cutoff — see the Γ-point validity condition in
    /// [`crate::pbc::gamma`]. Below it the energy surface these tests optimize on is dominated by
    /// the Γ-point sampling error, which pulls the cell into an unbounded collapse: not something
    /// the optimizer should be judged against, and slow to grind through besides, since every
    /// contraction pulls more images inside every cutoff.
    fn distorted_water_cell(edge: f64) -> Molecule {
        let mut molecule = Molecule::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.0\nH 1.05 0.0 0.0\nH -0.30 1.02 0.0\n",
            0.0,
        )
        .unwrap();
        molecule.cell = Some(Cell::cubic(edge).unwrap());
        molecule
    }

    /// Fixed-cell relaxation must lower the energy and drive the forces down.
    #[test]
    fn fixed_cell_relaxation_reaches_a_stationary_point() {
        let params = Pm3Parameters::standard().unwrap();
        let start = distorted_water_cell(30.0);
        let initial = periodic_gradient(
            &start,
            &params,
            &Pm3Options::default(),
            &PeriodicOptions::default(),
        )
        .unwrap();
        let result = relax(
            &start,
            &params,
            &Pm3Options::default(),
            &PeriodicOptions::default(),
            &PeriodicOptOptions {
                max_iter: 60,
                ..PeriodicOptOptions::default()
            },
        )
        .unwrap();
        assert!(result.converged, "did not converge in 60 steps");
        assert!(
            result.gradient.energy_ev < initial.energy_ev,
            "the optimizer raised the energy"
        );
        assert!(result.gradient.max_gradient < 1.0e-3);
    }

    /// Variable-cell relaxation has to lower the enthalpy *and* reduce the stress it is
    /// minimizing against.
    ///
    /// Note what this test deliberately does **not** claim. A molecular crystal at zero pressure
    /// genuinely wants to contract — the images attract — so a shrinking cell is the correct
    /// answer, not a bug, and asserting that an isolated molecule leaves its cell alone would be
    /// asserting the wrong physics. What must be true is that the optimizer moves downhill on
    /// the quantity it is given and that the stress driving it gets smaller.
    #[test]
    fn variable_cell_relaxation_lowers_the_enthalpy_and_the_stress() {
        let params = Pm3Parameters::standard().unwrap();
        let start = distorted_water_cell(18.0);
        let initial = periodic_gradient(
            &start,
            &params,
            &Pm3Options::default(),
            &PeriodicOptions::default(),
        )
        .unwrap();
        let largest = |gradient: &PeriodicGradient| {
            gradient
                .stress
                .expect("3D reports a stress")
                .col
                .iter()
                .flat_map(|column| column.to_array())
                .fold(0.0_f64, |m, v| m.max(v.abs()))
        };
        let result = relax(
            &start,
            &params,
            &Pm3Options::default(),
            &PeriodicOptions::default(),
            &PeriodicOptOptions {
                cell: CellRelaxation::Variable,
                max_iter: 25,
                ..PeriodicOptOptions::default()
            },
        )
        .unwrap();
        assert!(
            result.gradient.energy_ev < initial.energy_ev,
            "the optimizer raised the energy: {} -> {}",
            initial.energy_ev,
            result.gradient.energy_ev
        );
        assert!(
            largest(&result.gradient) < largest(&initial),
            "the stress grew: {:.3e} -> {:.3e}",
            largest(&initial),
            largest(&result.gradient)
        );
        assert!(result.gradient.max_gradient < initial.max_gradient);
    }

    /// External pressure has to push the right way: raising it must make the relaxed cell
    /// smaller than the one relaxed at zero pressure.
    #[test]
    fn pressure_compresses_the_cell() {
        let params = Pm3Parameters::standard().unwrap();
        let start = distorted_water_cell(18.0);
        let relaxed_volume = |pressure: f64| -> f64 {
            relax(
                &start,
                &params,
                &Pm3Options::default(),
                &PeriodicOptions::default(),
                &PeriodicOptOptions {
                    cell: CellRelaxation::Variable,
                    max_iter: 12,
                    pressure,
                    ..PeriodicOptOptions::default()
                },
            )
            .unwrap()
            .molecule
            .cell
            .unwrap()
            .measure()
        };
        // 1e-4 eV/Bohr³ is about 2.4 GPa — large enough to move a soft molecular cell within a
        // few steps, small enough not to collapse it.
        let free = relaxed_volume(0.0);
        let squeezed = relaxed_volume(1.0e-4);
        assert!(
            squeezed < free,
            "pressure did not compress the cell ({squeezed} vs {free})"
        );
    }

    /// A slab now has an in-plane stress, so a variable-cell run must relax the two in-plane
    /// vectors and leave the padding direction exactly where it was. Moving the vacuum would be
    /// the visible symptom of an unprojected strain.
    #[test]
    fn a_slab_relaxes_in_plane_and_leaves_the_vacuum_alone() {
        let params = Pm3Parameters::standard().unwrap();
        let mut slab = distorted_water_cell(30.0);
        let start = Cell::new(
            Vec3::new(11.0, 0.0, 0.0),
            Vec3::new(0.0, 11.5, 0.0),
            Vec3::new(0.0, 0.0, 34.0),
            [true, true, false],
        )
        .unwrap();
        slab.cell = Some(start);

        let relaxed = relax(
            &slab,
            &params,
            &Pm3Options::default(),
            &PeriodicOptions::default(),
            &PeriodicOptOptions {
                cell: CellRelaxation::Variable,
                max_iter: 3,
                ..PeriodicOptOptions::default()
            },
        )
        .expect("a slab has an in-plane strain derivative");

        let cell = relaxed.molecule.cell.unwrap();
        assert_eq!(
            cell.vector(2),
            start.vector(2),
            "the non-periodic vector must not move"
        );
        let moved =
            (cell.vector(0) - start.vector(0)).norm() + (cell.vector(1) - start.vector(1)).norm();
        assert!(moved > 1.0e-6, "the in-plane vectors should have relaxed");
    }

    /// An isolated cell has no strain at all, so a variable-cell request must be refused rather
    /// than converging instantly against zeros.
    #[test]
    fn variable_cell_is_refused_without_a_cell() {
        let params = Pm3Parameters::standard().unwrap();
        let mut molecule = distorted_water_cell(30.0);
        molecule.cell = Some(Cell::isolated());
        let error = relax(
            &molecule,
            &params,
            &Pm3Options::default(),
            &PeriodicOptions::default(),
            &PeriodicOptOptions {
                cell: CellRelaxation::Variable,
                ..PeriodicOptOptions::default()
            },
        )
        .expect_err("an isolated system has no strain");
        assert!(error.to_string().contains("periodic cell"));
    }
}
