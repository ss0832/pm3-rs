// SPDX-License-Identifier: GPL-3.0-or-later

//! Molden-format wavefunction export.
//!
//! # What is written, and what that means
//!
//! PM3 is an NDDO method: it assumes the valence AO basis is orthonormal and neglects
//! differential overlap between atoms. The Slater orbitals it names are therefore not a basis it
//! ever integrates over — they set the parameters and then step out of the way. Writing the MO
//! coefficients into a Molden file and letting a viewer draw them treats that orthonormal basis
//! as if it were the real, non-orthogonal Slater one.
//!
//! That is the conventional semiempirical visualization — it is what MOPAC's `VECTORS` and
//! `GRAPHF` produce, and writing the same thing is what makes the output comparable against
//! MOPAC — but it is a convention, not an identity, and it is worth knowing which one is being
//! looked at. Löwdin-orthogonalizing back (`C → S^{-1/2} C`) would give the orbitals of the real
//! basis and would no longer match MOPAC.
//!
//! # Slater orbitals, written as Gaussians
//!
//! Molden's format has an `[STO]` section, and PM3's orbitals are exactly what it wants: one
//! uncontracted Slater function per shell, with the element's own `ζ`. Almost no viewer reads
//! it. So the shells are expanded into Gaussians and written as `[GTO]`, which every viewer does
//! read.
//!
//! The expansion is **derived here rather than transcribed**. The published STO-nG coefficient
//! tables would need their provenance recorded and would fix the accuracy at whatever the table
//! chose; instead the exponents are laid on a documented even-tempered geometric series and only
//! the *coefficients* are fitted, which makes the fit a linear least-squares problem — solved
//! once per `(n, l)` at `ζ = 1` and then scaled, since a Slater orbital of exponent `ζ` is the
//! `ζ = 1` one with every Gaussian exponent multiplied by `ζ²`.
//!
//! Linear rather than nonlinear is the whole point: optimizing exponents too would fit better and
//! would be a fragile numerical dependency inside a file writer. The price is more primitives for
//! the same accuracy, and a Molden file does not care how many it is given.
//!
//! Accuracy is stated as the **overlap deficit** `1 − ⟨fit|STO⟩` between normalized functions,
//! because that is what a rendered isosurface responds to; an L² residual would report the same
//! fit as a smaller number and mean less. `the_expansion_reproduces_the_slater_orbitals` measures
//! it.

use crate::basis::Basis;
use crate::error::{Pm3Error, Result};
use crate::linalg::Matrix;
use crate::params::Pm3Parameters;
use crate::scf::Pm3Result;
use crate::system::Molecule;

/// Primitives per shell. Chosen by measurement — see the test — not by analogy with STO-6G.
const PRIMITIVES: usize = 14;

/// The even-tempered series `α_k = α_min · ratio^k`, in units of `ζ²`.
///
/// A Slater orbital decays as `e^{−ζr}` and a Gaussian as `e^{−αr²}`, so reproducing both the
/// cusp and the tail needs exponents spanning several orders of magnitude. `1s` sets the upper
/// bound: it is the only shell with a cusp at the origin, and by far the hardest to fit.
///
/// Measured overlap deficits with fourteen primitives: `1.9e-5` for `1s`, and between `1.7e-7`
/// and `1.9e-5` for every other shell PM3 uses. STO-6G's `1s` deficit is of order `1e-3`.
const ALPHA_MIN: f64 = 0.008;
const ALPHA_MAX: f64 = 30000.0;
/// The upper bound for every shell that has no cusp at the origin, which is every `n >= 2`.
const ALPHA_SMOOTH: f64 = 300.0;

/// Gaussian exponents (in units of `ζ²`) and the contraction coefficients fitting one
/// `(n, l)` Slater shell.
struct Expansion {
    exponents: [f64; PRIMITIVES],
    coefficients: [f64; PRIMITIVES],
}

/// The radial part of a normalized Slater orbital, `N r^{n−1} e^{−r}` at `ζ = 1`.
fn slater_radial(n: usize, r: f64) -> f64 {
    // N = (2ζ)^n sqrt(2ζ / (2n)!) with ζ = 1; the normalization cancels in the overlap deficit
    // but is kept so the coefficients are the ones a reader expects.
    let factorial = (1..=2 * n).map(|k| k as f64).product::<f64>();
    let norm = (2.0_f64).powi(n as i32) * (2.0 / factorial).sqrt();
    norm * r.powi(n as i32 - 1) * (-r).exp()
}

/// The radial part of a normalized primitive Gaussian of angular momentum `l`, `N r^l e^{−αr²}`.
fn gaussian_radial(l: usize, alpha: f64, r: f64) -> f64 {
    // ∫ (r^l e^{−αr²})² r² dr = Γ(l + 3/2) / (2 (2α)^{l + 3/2}).
    let half = l as f64 + 1.5;
    let gamma = gamma_half_integer(l);
    let norm = (2.0 * (2.0 * alpha).powf(half) / gamma).sqrt();
    norm * r.powi(l as i32) * (-alpha * r * r).exp()
}

/// `Γ(l + 3/2)` for integer `l`, by the half-integer recurrence from `Γ(3/2) = √π/2`.
fn gamma_half_integer(l: usize) -> f64 {
    let mut value = std::f64::consts::PI.sqrt() / 2.0;
    for k in 0..l {
        value *= k as f64 + 1.5;
    }
    value
}

/// Fit `PRIMITIVES` Gaussians to the `(n, l)` Slater shell at `ζ = 1`.
///
/// The exponents are fixed by the geometric series, so this is a linear least-squares problem in
/// the coefficients: minimize `‖Σ_k c_k g_k − s‖²` under the radial measure `r² dr`. The normal
/// equations are solved through their eigendecomposition rather than by Gaussian elimination —
/// an even-tempered Gaussian set is famously ill-conditioned (neighbouring primitives overlap by
/// more than 0.99), and truncating the small eigenvalues is what keeps the fit from being a
/// large cancellation between huge coefficients.
fn fit(n: usize, l: usize) -> Result<Expansion> {
    // The upper bound is per shell; the lower bound is not.
    //
    // At `ζ = 1` every shell decays as `e^{−r}` whatever `n` is — the `r^{n−1}` prefactor moves
    // the peak outwards but does not change the tail — so the smallest exponent needed is the
    // same for all of them. What differs is the origin: `1s` has a cusp there and needs
    // exponents four orders of magnitude higher to resolve it, while every `n ≥ 2` orbital
    // vanishes at `r = 0` and does not.
    //
    // Scaling *both* bounds with `n` was tried first, on the reasoning that the orbital's radius
    // grows as `n²`. It fits `1s` to 1.9e-5 and destroys everything else — 9.5e-2 at `n = 3` and
    // 8.2e-1 at `n = 4` — because it drags the lower bound below where the tail lives and spends
    // the primitives on exponents nothing needs.
    let high = if n == 1 { ALPHA_MAX } else { ALPHA_SMOOTH };
    let low = ALPHA_MIN;
    let ratio = (high / low).powf(1.0 / (PRIMITIVES - 1) as f64);
    let mut exponents = [0.0; PRIMITIVES];
    for (k, slot) in exponents.iter_mut().enumerate() {
        *slot = low * ratio.powi(k as i32);
    }

    // Overlaps by Gauss–Legendre-free quadrature: the integrands decay exponentially, so a fine
    // uniform grid out to where the Slater orbital has died is both simple and ample.
    let (points, cut) = (20_000, 40.0);
    let step = cut / points as f64;
    let mut overlap = Matrix::zeros(PRIMITIVES, PRIMITIVES);
    let mut target = [0.0; PRIMITIVES];
    for index in 0..points {
        let r = (index as f64 + 0.5) * step;
        let weight = r * r * step;
        let s = slater_radial(n, r);
        let g: Vec<f64> = exponents
            .iter()
            .map(|a| gaussian_radial(l, *a, r))
            .collect();
        for i in 0..PRIMITIVES {
            target[i] += weight * g[i] * s;
            for j in 0..PRIMITIVES {
                overlap[(i, j)] += weight * g[i] * g[j];
            }
        }
    }

    // Solve `S c = t` through the eigenbasis, dropping directions the grid cannot resolve.
    let (values, vectors) = crate::linalg::symmetric_eigen(&overlap)?;
    let largest = values.iter().cloned().fold(0.0_f64, f64::max);
    if largest <= 0.0 {
        return Err(Pm3Error::InvalidInput(
            "the Gaussian overlap matrix is singular; the exponent range is wrong".to_string(),
        ));
    }
    let floor = largest * 1.0e-12;
    let mut coefficients = [0.0; PRIMITIVES];
    for mode in 0..PRIMITIVES {
        if values[mode] < floor {
            continue;
        }
        let projection: f64 = (0..PRIMITIVES)
            .map(|i| vectors[(i, mode)] * target[i])
            .sum();
        let scale = projection / values[mode];
        for (i, slot) in coefficients.iter_mut().enumerate() {
            *slot += scale * vectors[(i, mode)];
        }
    }
    Ok(Expansion {
        exponents,
        coefficients,
    })
}

/// Molden's AO ordering within a shell, as this crate's own indices.
///
/// This crate follows MOPAC: `s, px, py, pz, d(x²−y²), d(xz), d(z²), d(yz), d(xy)`. Molden's
/// Cartesian `d` order is `xx, yy, zz, xy, xz, yz`, and its `p` order is `x, y, z` — the same as
/// ours. There is no way to write MOPAC's five real `d` functions as Molden Cartesians without
/// re-expanding them, so a `d` element is refused rather than written in an order no viewer
/// would read correctly. No PM3 element has a `d` shell, so this is a guard, not a limitation.
fn shell_order(n_orb: usize) -> Result<&'static [usize]> {
    match n_orb {
        0 => Ok(&[]),
        1 => Ok(&[0]),
        4 => Ok(&[0, 1, 2, 3]),
        _ => Err(Pm3Error::InvalidInput(
            "Molden export covers s and sp elements; a d shell would need MOPAC's five real \
             d functions re-expanded into Molden's six Cartesian ones, which is a conversion \
             rather than a reordering. No PM3 element has a d shell."
                .to_string(),
        )),
    }
}

/// Write a converged wavefunction as a Molden file.
///
/// Pure: it returns the document rather than writing it, matching how the rest of the crate
/// separates computation from output (see `cli::write_xyz`).
pub fn molden_string(
    molecule: &Molecule,
    params: &Pm3Parameters,
    result: &Pm3Result,
) -> Result<String> {
    let basis = Basis::build(molecule, params)?;
    let mut out = String::from("[Molden Format]\n");
    out.push_str("[Title]\n pm3-rs wavefunction (NDDO orbitals in a Slater basis; see docs)\n");

    // Coordinates in Bohr, which is what the calculation used — no conversion to get wrong.
    out.push_str("[Atoms] AU\n");
    for (index, atom) in molecule.atoms.iter().enumerate() {
        let symbol = crate::system::z_to_symbol(atom.z).unwrap_or("X");
        // A sparkle or point charge has no basis function but is still an atom in the geometry.
        out.push_str(&format!(
            " {symbol:<3} {:>5} {:>4} {:>18.10} {:>18.10} {:>18.10}\n",
            index + 1,
            atom.z,
            atom.position.x,
            atom.position.y,
            atom.position.z
        ));
    }

    out.push_str("[GTO]\n");
    for (index, atom) in molecule.atoms.iter().enumerate() {
        let element = params.element(atom.z)?;
        shell_order(element.n_orb)?;
        out.push_str(&format!(" {:>4} 0\n", index + 1));
        if element.n_orb >= 1 {
            write_shell(
                &mut out,
                "s",
                element.n_s.max(1) as usize,
                0,
                element.zeta_s,
            )?;
        }
        if element.n_orb >= 4 {
            write_shell(
                &mut out,
                "p",
                element.n_p.max(2) as usize,
                1,
                element.zeta_p,
            )?;
        }
        // Molden separates atoms with a blank line.
        out.push('\n');
    }

    let restricted = result.mo_coeff_beta.is_none();
    out.push_str("[MO]\n");
    write_orbitals(
        &mut out,
        &basis,
        &result.mo_coeff,
        &result.mo_energies,
        result.n_occ,
        "Alpha",
        if restricted { 2.0 } else { 1.0 },
        molecule,
        params,
    )?;
    if let (Some(coefficients), Some(energies)) = (&result.mo_coeff_beta, &result.mo_energies_beta)
    {
        write_orbitals(
            &mut out,
            &basis,
            coefficients,
            energies,
            result.n_beta,
            "Beta",
            1.0,
            molecule,
            params,
        )?;
    }
    Ok(out)
}

/// One shell's primitives, with the `ζ²` scaling applied.
fn write_shell(out: &mut String, label: &str, n: usize, l: usize, zeta: f64) -> Result<()> {
    if zeta <= 0.0 {
        // Hydrogen carries a `zeta_p` of zero; that channel does not exist and writing it would
        // put a Gaussian of exponent zero into the file.
        return Ok(());
    }
    let expansion = fit(n, l)?;
    out.push_str(&format!(" {label} {PRIMITIVES:>4} 1.00\n"));
    let scale = zeta * zeta;
    for (alpha, coefficient) in expansion.exponents.iter().zip(&expansion.coefficients) {
        out.push_str(&format!(
            " {:>20.10e} {:>20.10e}\n",
            alpha * scale,
            coefficient
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_orbitals(
    out: &mut String,
    basis: &Basis,
    coefficients: &Matrix,
    energies: &[f64],
    n_occupied: usize,
    spin: &str,
    occupancy: f64,
    molecule: &Molecule,
    params: &Pm3Parameters,
) -> Result<()> {
    // The AO index Molden expects, given ours. Built once so the inner loop is a lookup.
    let mut order = Vec::with_capacity(basis.nao);
    for (index, atom) in molecule.atoms.iter().enumerate() {
        let element = params.element(atom.z)?;
        let offset = basis.atom_offset[index];
        for local in shell_order(element.n_orb)? {
            order.push(offset + local);
        }
    }

    for mo in 0..basis.nao {
        out.push_str(&format!(
            " Sym= {}\n Ene= {:>18.10}\n Spin= {spin}\n Occup= {:>10.6}\n",
            mo + 1,
            // Molden wants Hartree; this crate works in eV.
            energies[mo] * crate::constants::EV_TO_HARTREE,
            if mo < n_occupied { occupancy } else { 0.0 }
        ));
        for (row, &ao) in order.iter().enumerate() {
            out.push_str(&format!(
                " {:>5} {:>18.10}\n",
                row + 1,
                coefficients[(ao, mo)]
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scf::{run_pm3, Pm3Options};

    fn water() -> Molecule {
        Molecule::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
            0.0,
        )
        .unwrap()
    }

    /// How well the Gaussians stand in for the Slater orbitals, measured as the overlap deficit
    /// between normalized functions — the quantity a rendered isosurface responds to.
    ///
    /// This is what fixes [`PRIMITIVES`] and the exponent range, both by measurement. Fourteen
    /// primitives on a geometric series from `0.008 ζ²` to `3·10⁴ ζ²` reach 2·10⁻⁵ for the
    /// 1s shell and better for every other one PM3 uses -- about fifty times closer than STO-6G,
    /// whose 1s deficit is of order 10⁻³, and far below the error already accepted by drawing
    /// an orthonormal basis as a Slater one.
    #[test]
    fn the_expansion_reproduces_the_slater_orbitals() {
        let mut measured = Vec::new();
        for (n, l) in [
            (1usize, 0usize),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
            (4, 0),
            (4, 1),
        ] {
            let expansion = fit(n, l).unwrap();
            let (points, cut) = (40_000, 60.0);
            let step = cut / points as f64;
            let (mut ss, mut gg, mut sg) = (0.0, 0.0, 0.0);
            for index in 0..points {
                let r = (index as f64 + 0.5) * step;
                let weight = r * r * step;
                let s = slater_radial(n, r);
                let g: f64 = expansion
                    .exponents
                    .iter()
                    .zip(&expansion.coefficients)
                    .map(|(a, c)| c * gaussian_radial(l, *a, r))
                    .sum();
                ss += weight * s * s;
                gg += weight * g * g;
                sg += weight * s * g;
            }
            let deficit = 1.0 - sg / (ss * gg).sqrt();
            measured.push((n, l, deficit));
        }

        // Report every shell, not just the first bad one: which shells fail is the whole
        // diagnostic, and stopping at the first hides whether the range is wrong at one end or
        // spread too thin across the middle.
        let worst = measured.iter().map(|(_, _, d)| *d).fold(0.0_f64, f64::max);
        let report: Vec<String> = measured
            .iter()
            .map(|(n, l, d)| format!("(n={n}, l={l}) {d:.3e}"))
            .collect();
        assert!(
            measured.iter().all(|(_, _, d)| *d >= 0.0) && worst < 3.0e-5,
            "overlap deficits: {}",
            report.join("  ")
        );
    }

    /// The document has to be readable as Molden, which means the sections a viewer looks for,
    /// one coefficient block per AO per MO, and occupations that add up to the electron count.
    #[test]
    fn the_document_has_the_sections_a_viewer_needs() {
        let molecule = water();
        let params = Pm3Parameters::standard().unwrap();
        let result = run_pm3(&molecule, &params, &Pm3Options::default()).unwrap();
        let text = molden_string(&molecule, &params, &result).unwrap();

        for section in ["[Molden Format]", "[Atoms] AU", "[GTO]", "[MO]"] {
            assert!(text.contains(section), "missing {section}");
        }
        // Six AOs for water (4 on O, 1 per H), so six MOs, each with six coefficients.
        assert_eq!(text.matches(" Spin= Alpha").count(), 6);
        assert!(
            !text.contains("Spin= Beta"),
            "a closed shell has one orbital set"
        );

        let occupations: f64 = text
            .lines()
            .filter_map(|line| line.strip_prefix(" Occup="))
            .map(|value| value.trim().parse::<f64>().unwrap())
            .sum();
        assert!(
            (occupations - 8.0).abs() < 1.0e-9,
            "water has eight valence electrons, the file says {occupations}"
        );
        // Every number has to be finite, or a viewer will read the file and draw nothing.
        for line in text.lines() {
            for token in line.split_whitespace() {
                if let Ok(value) = token.parse::<f64>() {
                    assert!(value.is_finite(), "non-finite value in: {line}");
                }
            }
        }
    }

    /// An open-shell system writes both spin sets, which is the whole reason `Pm3Result` keeps
    /// the β orbitals rather than recomputing them.
    #[test]
    fn an_open_shell_writes_both_spin_sets() {
        let molecule = Molecule::from_xyz_str(
            "4\nmethyl\nC 0.0 0.0 0.0\nH 0.0 1.078 0.0\nH 0.9336 -0.539 0.0\nH -0.9336 -0.539 0.0\n",
            0.0,
        )
        .unwrap()
        .with_multiplicity(2);
        let params = Pm3Parameters::standard().unwrap();
        let options = Pm3Options {
            multiplicity: 2,
            ..Pm3Options::default()
        };
        let result = run_pm3(&molecule, &params, &options).unwrap();
        assert!(result.unrestricted);
        let text = molden_string(&molecule, &params, &result).unwrap();

        let alpha = text.matches(" Spin= Alpha").count();
        let beta = text.matches(" Spin= Beta").count();
        assert_eq!(alpha, 7, "seven AOs for the methyl radical");
        assert_eq!(beta, alpha, "both spins get a full set");

        let occupations: f64 = text
            .lines()
            .filter_map(|line| line.strip_prefix(" Occup="))
            .map(|value| value.trim().parse::<f64>().unwrap())
            .sum();
        assert!(
            (occupations - 7.0).abs() < 1.0e-9,
            "the methyl radical has seven valence electrons, the file says {occupations}"
        );
    }
}
