// SPDX-License-Identifier: GPL-3.0-or-later

//! L-BFGS geometry optimization on the PM3 energy surface (Rust-native), driven by the
//! Hellmann-Feynman nuclear gradient from [`crate::gradient`]. Positions are Bohr
//! internally; gradients are eV/Bohr.

use crate::error::Result;
use crate::gradient::closed_form_gradient;
use crate::math::Vec3;
use crate::params::Pm3Parameters;
use crate::scf::{run_pm3, Pm3Options, Pm3Result};
use crate::system::Molecule;

#[derive(Clone, Debug)]
pub struct OptOptions {
    pub max_iter: usize,
    /// Convergence on the max gradient component (eV/Bohr).
    pub gtol: f64,
    /// Finite-difference step for the gradient (Bohr).
    pub grad_step: f64,
    /// L-BFGS history length.
    pub history: usize,
}

impl Default for OptOptions {
    fn default() -> Self {
        Self {
            max_iter: 200,
            gtol: 1.0e-3,
            grad_step: 5.0e-4,
            history: 8,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OptStep {
    pub energy_ev: f64,
    pub heat_of_formation_kcal: f64,
    pub max_gradient: f64,
    pub positions: Vec<Vec3>,
}

#[derive(Clone, Debug)]
pub struct OptResult {
    pub molecule: Molecule,
    pub scf: Pm3Result,
    pub converged: bool,
    pub iterations: usize,
    pub trajectory: Vec<OptStep>,
}

/// The result of one energy-and-gradient evaluation, whatever produced it.
///
/// `payload` is the caller's own SCF result — a `Pm3Result` for the full diagonalization, a
/// `DcResult` for the partitioned one — carried through untouched so the driver below never has
/// to know which it is.
struct Evaluated<T> {
    energy_ev: f64,
    gradient: Vec<Vec3>,
    max_gradient: f64,
    heat_of_formation_kcal: f64,
    payload: T,
}

/// L-BFGS with a backtracking Armijo line search, over whatever produces the energy and gradient.
///
/// Written once and used twice. The full-diagonalization path and the divide-and-conquer path
/// differ only in which function evaluates a geometry; the two-loop recursion, the curvature
/// filter, the uphill guard and the step halving are the same optimizer, and a second copy of
/// them is a second place for the line search to drift.
///
/// `energy_only` is the line search's cheaper question — it needs the energy at a trial point and
/// not the gradient. `None` means that point could not be evaluated, which the search treats as a
/// rejected step rather than as an error, since the usual cause is a trial geometry far enough
/// off that the SCF does not settle.
fn drive<T, E, L>(
    molecule: &Molecule,
    opt: &OptOptions,
    mut evaluate: E,
    mut energy_only: L,
) -> Result<(Molecule, T, bool, usize, Vec<OptStep>)>
where
    E: FnMut(&Molecule) -> Result<Evaluated<T>>,
    L: FnMut(&Molecule) -> Option<f64>,
{
    let nat = molecule.atoms.len();
    let ndof = 3 * nat;
    let mut mol = molecule.clone();

    let mut x = flatten(&mol);
    let first = evaluate(&mol)?;
    let mut g = flatten_grad(&first.gradient);
    let mut energy = first.energy_ev;
    let mut heat = first.heat_of_formation_kcal;
    let mut payload = first.payload;
    let mut max_grad = first.max_gradient;

    let mut s_hist: Vec<Vec<f64>> = Vec::new();
    let mut y_hist: Vec<Vec<f64>> = Vec::new();
    let mut rho_hist: Vec<f64> = Vec::new();

    let mut trajectory = vec![OptStep {
        energy_ev: energy,
        heat_of_formation_kcal: heat,
        max_gradient: max_grad,
        positions: unflatten(&x),
    }];

    let mut converged = max_grad < opt.gtol;
    let mut iterations = 0;

    for iter in 0..opt.max_iter {
        iterations = iter + 1;
        if converged {
            break;
        }

        // L-BFGS two-loop recursion -> search direction d = -H*g.
        let mut q = g.clone();
        let m = s_hist.len();
        let mut alpha = vec![0.0; m];
        for i in (0..m).rev() {
            let a = rho_hist[i] * dot(&s_hist[i], &q);
            alpha[i] = a;
            axpy(&mut q, -a, &y_hist[i]);
        }
        // Initial Hessian scaling.
        let gamma = if m > 0 {
            let sy = dot(&s_hist[m - 1], &y_hist[m - 1]);
            let yy = dot(&y_hist[m - 1], &y_hist[m - 1]);
            if yy > 0.0 {
                sy / yy
            } else {
                1.0
            }
        } else {
            // Cautious first step.
            0.1 / max_grad.max(1.0e-6)
        };
        for v in q.iter_mut() {
            *v *= gamma;
        }
        for i in 0..m {
            let beta = rho_hist[i] * dot(&y_hist[i], &q);
            axpy(&mut q, alpha[i] - beta, &s_hist[i]);
        }
        let mut d: Vec<f64> = q.iter().map(|v| -v).collect();
        // Guard against uphill directions.
        if dot(&d, &g) > 0.0 {
            d = g.iter().map(|v| -v).collect();
        }

        // Backtracking Armijo line search.
        let g_dot_d = dot(&g, &d);
        let mut step = 1.0;
        let c1 = 1.0e-4;
        let mut x_new;
        let mut ok = false;
        loop {
            x_new = x.clone();
            axpy(&mut x_new, step, &d);
            set_positions(&mut mol, &x_new);
            if let Some(trial) = energy_only(&mol) {
                if trial <= energy + c1 * step * g_dot_d {
                    ok = true;
                    break;
                }
            }
            step *= 0.5;
            if step < 1.0e-8 {
                break;
            }
        }
        if !ok {
            // Could not make progress; stop at the current point.
            set_positions(&mut mol, &x);
            break;
        }

        let next = evaluate(&mol)?;
        let g_new = flatten_grad(&next.gradient);

        // Update L-BFGS memory.
        let s: Vec<f64> = (0..ndof).map(|i| x_new[i] - x[i]).collect();
        let y: Vec<f64> = (0..ndof).map(|i| g_new[i] - g[i]).collect();
        let sy = dot(&s, &y);
        if sy > 1.0e-10 {
            s_hist.push(s);
            y_hist.push(y);
            rho_hist.push(1.0 / sy);
            if s_hist.len() > opt.history {
                s_hist.remove(0);
                y_hist.remove(0);
                rho_hist.remove(0);
            }
        }

        x = x_new;
        g = g_new;
        energy = next.energy_ev;
        heat = next.heat_of_formation_kcal;
        payload = next.payload;
        max_grad = next.max_gradient;
        converged = max_grad < opt.gtol;

        trajectory.push(OptStep {
            energy_ev: energy,
            heat_of_formation_kcal: heat,
            max_gradient: max_grad,
            positions: unflatten(&x),
        });
    }

    set_positions(&mut mol, &x);
    Ok((mol, payload, converged, iterations, trajectory))
}

/// A charged molecule in a uniform field has no minimum: the net force `−Q f` never vanishes, so
/// the whole thing accelerates down the field forever and the optimizer runs to its iteration
/// limit against a gradient that never falls. Warn rather than refuse — the run is legitimate if
/// what is wanted is the trajectory rather than a stationary point.
fn warn_if_no_minimum(scf_options: &Pm3Options) {
    if scf_options.field.is_some() && scf_options.charge != 0.0 {
        eprintln!(
            "warning: a net charge of {} in a uniform field feels a constant force, so this \
             geometry has no minimum to find and the optimization will not converge",
            scf_options.charge
        );
    }
}

pub fn optimize(
    molecule: &Molecule,
    params: &Pm3Parameters,
    scf_options: &Pm3Options,
    opt: &OptOptions,
) -> Result<OptResult> {
    warn_if_no_minimum(scf_options);
    let (molecule, scf, converged, iterations, trajectory) = drive(
        molecule,
        opt,
        |mol| {
            let g = closed_form_gradient(mol, params, scf_options)?;
            Ok(Evaluated {
                energy_ev: g.energy_ev,
                gradient: g.gradient,
                max_gradient: g.max_gradient,
                heat_of_formation_kcal: g.scf.heat_of_formation_kcal,
                payload: g.scf,
            })
        },
        |mol| run_pm3(mol, params, scf_options).ok().map(|r| r.total_ev),
    )?;
    Ok(OptResult {
        molecule,
        scf,
        converged,
        iterations,
        trajectory,
    })
}

/// A geometry optimization on the **divide-and-conquer** gradient.
///
/// The same L-BFGS, over `dc_gradient` instead of the full diagonalization. This exists for the
/// case `--dc` exists for: a system large enough that a full diagonalization per line-search
/// trial is not affordable, which is also the case where a geometry optimization is most
/// expensive and most wanted.
///
/// **The density is not variational**, so the usual argument that the first-order energy error
/// vanishes at the SCF solution does not apply and the gradient carries the partitioning's own
/// truncation error rather than its square. It converges with `DcOptions::buffer_radius` the same
/// way the energy does, so a geometry optimized at one buffer should be checked at a wider one
/// before it is believed — the same caveat `dc_gradient` carries, and it compounds over the
/// several hundred gradients an optimization takes.
pub fn optimize_dc(
    molecule: &Molecule,
    params: &Pm3Parameters,
    scf_options: &Pm3Options,
    dc: &crate::dc::DcOptions,
    opt: &OptOptions,
) -> Result<DcOptResult> {
    warn_if_no_minimum(scf_options);
    let (molecule, scf, converged, iterations, trajectory) = drive(
        molecule,
        opt,
        |mol| {
            let g = crate::dc::dc_gradient(mol, params, scf_options, dc)?;
            Ok(Evaluated {
                energy_ev: g.energy_ev,
                gradient: g.gradient,
                max_gradient: g.max_gradient,
                heat_of_formation_kcal: g.scf.heat_of_formation_kcal,
                payload: g.scf,
            })
        },
        |mol| {
            crate::dc::run_dc(mol, params, scf_options, dc)
                .ok()
                .map(|r| r.total_ev)
        },
    )?;
    Ok(DcOptResult {
        molecule,
        scf,
        converged,
        iterations,
        trajectory,
    })
}

/// [`OptResult`] for a partitioned optimization: the same fields, carrying a [`crate::dc::DcResult`].
#[derive(Clone, Debug)]
pub struct DcOptResult {
    pub molecule: Molecule,
    pub scf: crate::dc::DcResult,
    pub converged: bool,
    pub iterations: usize,
    pub trajectory: Vec<OptStep>,
}

fn flatten(mol: &Molecule) -> Vec<f64> {
    let mut v = Vec::with_capacity(3 * mol.atoms.len());
    for a in &mol.atoms {
        v.push(a.position.x);
        v.push(a.position.y);
        v.push(a.position.z);
    }
    v
}
fn flatten_grad(g: &[Vec3]) -> Vec<f64> {
    let mut v = Vec::with_capacity(3 * g.len());
    for gi in g {
        v.push(gi.x);
        v.push(gi.y);
        v.push(gi.z);
    }
    v
}
fn unflatten(x: &[f64]) -> Vec<Vec3> {
    x.chunks(3).map(|c| Vec3::new(c[0], c[1], c[2])).collect()
}
fn set_positions(mol: &mut Molecule, x: &[f64]) {
    for (i, a) in mol.atoms.iter_mut().enumerate() {
        a.position = Vec3::new(x[3 * i], x[3 * i + 1], x[3 * i + 2]);
    }
}
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn axpy(y: &mut [f64], a: f64, x: &[f64]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimizes_water() {
        // MOPAC v23.2.5 PM3 optimization of the same distorted water reaches
        // dHf = -54.30653 kcal/mol.
        let xyz = "3\nwater\nO 0.0 0.0 0.0\nH 1.05 0.0 0.0\nH -0.30 1.02 0.0\n";
        let mol = Molecule::from_xyz_str(xyz, 0.0).unwrap();
        let params = Pm3Parameters::standard().unwrap();
        let res = optimize(
            &mol,
            &params,
            &Pm3Options::default(),
            &OptOptions::default(),
        )
        .unwrap();
        eprintln!(
            "opt H2O: converged={} iters={} dHf={:.3} kcal/mol maxgrad={:.2e}",
            res.converged,
            res.iterations,
            res.scf.heat_of_formation_kcal,
            res.trajectory.last().unwrap().max_gradient
        );
        assert!(res.converged);
        assert!((res.scf.heat_of_formation_kcal - (-53.4330121104622)).abs() < 0.005);
        assert!(res.trajectory.last().unwrap().energy_ev < res.trajectory[0].energy_ev);
    }
}
