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

/// Which radial form the basis section carries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MoldenBasis {
    /// A Gaussian expansion of each Slater shell, written as `[GTO]`. **The default**, and what
    /// every viewer reads.
    #[default]
    Gto,
    /// The Slater functions themselves, written as `[STO]`.
    ///
    /// Kept for compatibility and **not** the default. PM3's orbitals are exactly one
    /// uncontracted Slater function per shell, so this is the more faithful description of what
    /// the model actually names — and almost no viewer implements the section, which is why the
    /// Gaussian expansion exists at all. The layout mirrors `[GTO]`'s (an atom index line, then
    /// one shell line per shell, then its exponent), because that is the parallel Molden's own
    /// documentation draws; a reader that does not implement `[STO]` will skip it, and one that
    /// does may expect a different field order. Use `Gto` unless you know your viewer.
    Sto,
}

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
#[derive(Clone, Copy)]
struct Expansion {
    exponents: [f64; PRIMITIVES],
    coefficients: [f64; PRIMITIVES],
}

/// [`fit`], memoized on `(n, l)`.
///
/// The fit is a 20 000-point quadrature and a `14 × 14` eigendecomposition, and it depends on
/// nothing but the shell: `ζ` enters afterwards, by scaling. It was being redone for every shell
/// of every atom, so a hundred carbons paid for the same two fits a hundred times. PM3 reaches at
/// most eleven distinct `(n, l)` pairs.
fn cached_fit(n: usize, l: usize) -> Result<Expansion> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    static CACHE: OnceLock<Mutex<HashMap<(usize, usize), Expansion>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    // A poisoned lock means another thread panicked mid-fit. The table is a pure function of its
    // key, so whatever is in it is still correct and the entry can simply be recomputed.
    if let Ok(table) = cache.lock() {
        if let Some(hit) = table.get(&(n, l)) {
            return Ok(*hit);
        }
    }
    let made = fit(n, l)?;
    if let Ok(mut table) = cache.lock() {
        table.insert((n, l), made);
    }
    Ok(made)
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
    molden_string_with(molecule, params, result, MoldenBasis::Gto)
}

/// [`molden_string`], choosing which radial form the basis section carries.
///
/// See [`MoldenBasis`]. `Gto` is what [`molden_string`] writes and what a viewer will read.
pub fn molden_string_with(
    molecule: &Molecule,
    params: &Pm3Parameters,
    result: &Pm3Result,
    form: MoldenBasis,
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

    out.push_str(match form {
        MoldenBasis::Gto => "[GTO]\n",
        MoldenBasis::Sto => "[STO]\n",
    });
    for (index, atom) in molecule.atoms.iter().enumerate() {
        let element = params.element(atom.z)?;
        shell_order(element.n_orb)?;
        out.push_str(&format!(" {:>4} 0\n", index + 1));
        if element.n_orb >= 1 {
            write_shell(
                &mut out,
                form,
                "s",
                element.n_s.max(1) as usize,
                0,
                element.zeta_s,
            )?;
        }
        if element.n_orb >= 4 {
            write_shell(
                &mut out,
                form,
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

/// One shell, either as its Gaussian expansion or as the Slater function itself.
///
/// A zero `ζ` on a shell the element *declares* is an error rather than a silent skip. It cannot
/// happen with the shipped parameters — `n_orb` is 4 only when `zeta_p > 0` — but if it ever did,
/// skipping would write one basis function into the section while `[MO]` wrote four coefficient
/// rows for the atom, and **every AO index after it would be off by three**. That is the exact
/// shape of "the occupied orbitals do not line up with the molecular orbitals", and it would be
/// invisible in the file: a viewer would draw the wrong thing rather than refuse.
fn write_shell(
    out: &mut String,
    form: MoldenBasis,
    label: &str,
    n: usize,
    l: usize,
    zeta: f64,
) -> Result<()> {
    if zeta <= 0.0 {
        return Err(Pm3Error::InvalidInput(format!(
            "the element declares a {label} shell but its Slater exponent is {zeta}. Writing no \
             basis function for a shell the [MO] section has coefficients for would shift every \
             AO index after this atom, so the file is refused rather than written misaligned."
        )));
    }
    match form {
        MoldenBasis::Sto => {
            // One uncontracted Slater function, which is what PM3 actually names.
            out.push_str(&format!(" {label}    1 1.00\n"));
            out.push_str(&format!(" {} {}\n", fortran(zeta), fortran(1.0)));
        }
        MoldenBasis::Gto => {
            let expansion = cached_fit(n, l)?;
            out.push_str(&format!(" {label} {PRIMITIVES:>4} 1.00\n"));
            let scale = zeta * zeta;
            for (alpha, coefficient) in expansion.exponents.iter().zip(&expansion.coefficients) {
                out.push_str(&format!(
                    " {} {}\n",
                    fortran(alpha * scale),
                    fortran(*coefficient)
                ));
            }
        }
    }
    Ok(())
}

/// A float in the `1.2345678901E+02` form Fortran list-directed input expects.
///
/// Rust's `{:e}` gives `1.2345678901e2` — no sign on the exponent, no padding, and a lowercase
/// `e`. Most readers cope; the ones that do not are Fortran, which is most of the programs that
/// read this format. Writing the conventional form costs nothing and removes the question.
fn fortran(value: f64) -> String {
    let formatted = format!("{value:>20.10E}");
    // Rust writes `E2` and `E-2`; Fortran readers expect `E+02` and `E-02`.
    match formatted.rsplit_once('E') {
        Some((mantissa, exponent)) => {
            let (sign, digits) = match exponent.strip_prefix('-') {
                Some(rest) => ('-', rest),
                None => ('+', exponent),
            };
            format!("{mantissa}E{sign}{digits:0>2}")
        }
        None => formatted,
    }
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
        // `Sym=` is a *label*, not an index. PM3 carries no point group, so there is no
        // irreducible representation to name and every orbital gets the placeholder `a` — which
        // is what MOPAC writes for the same reason. It used to be the bare integer `mo + 1`, and
        // a reader that parses the label as `<number><irrep>` reads that as a symmetry species
        // with an empty name.
        let label = format!("{}a", mo + 1);
        out.push_str(&format!(
            " Sym= {label}\n Ene= {:>18.10}\n Spin= {spin}\n Occup= {:>10.6}\n",
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

    /// Read the file back and check that the basis section and the orbital section agree.
    ///
    /// This is the test for "the occupied orbitals do not line up with the molecular orbitals".
    /// That symptom has exactly one mechanical cause in a Molden file — the number of basis
    /// functions the basis section declares disagreeing with the number of coefficient rows each
    /// `[MO]` block writes — and it is invisible from the inside, because both halves are
    /// individually well formed. So the check parses the document as a viewer would and compares
    /// the two counts, rather than asserting on how it was generated.
    ///
    /// Run over **both** basis forms. `[STO]` is reachable from the CLI and from Python, and a
    /// shell that is written into one section and not the other is exactly the mismatch above —
    /// the coefficients would be right and the basis they index into wrong.
    #[test]
    fn the_basis_section_and_the_orbital_section_agree_on_the_basis_size() {
        for (form, section) in [(MoldenBasis::Gto, "[GTO]"), (MoldenBasis::Sto, "[STO]")] {
            check_basis_and_orbitals_agree(form, section);
        }
    }

    fn check_basis_and_orbitals_agree(form: MoldenBasis, section: &str) {
        for (label, molecule, options) in [
            ("water", water(), Pm3Options::default()),
            (
                "methyl radical",
                Molecule::from_xyz_str(
                    "4\nmethyl\nC 0.0 0.0 0.0\nH 0.0 1.078 0.0\nH 0.9336 -0.539 0.0\n\
                     H -0.9336 -0.539 0.0\n",
                    0.0,
                )
                .unwrap()
                .with_multiplicity(2),
                Pm3Options {
                    multiplicity: 2,
                    ..Pm3Options::default()
                },
            ),
        ] {
            let params = Pm3Parameters::standard().unwrap();
            let result = run_pm3(&molecule, &params, &options).unwrap();
            let basis = Basis::build(&molecule, &params).unwrap();
            let text = molden_string_with(&molecule, &params, &result, form).unwrap();
            let label = &format!("{label} {section}");

            // Count basis functions the way a reader does: walk the basis section, and for each
            // shell line add the number of functions that shell carries.
            let mut declared = 0usize;
            let mut in_gto = false;
            for line in text.lines() {
                if line.starts_with('[') {
                    in_gto = line.starts_with(section);
                    continue;
                }
                if !in_gto {
                    continue;
                }
                let fields: Vec<&str> = line.split_whitespace().collect();
                // A shell line is `<label> <nprim> <scale>`; anything else is an atom index line,
                // a primitive, or the blank separator.
                if fields.len() == 3 && fields[1].parse::<usize>().is_ok() {
                    declared += match fields[0] {
                        "s" => 1,
                        "p" => 3,
                        other => panic!("{label}: unexpected shell label {other}"),
                    };
                }
            }
            assert_eq!(
                declared, basis.nao,
                "{label}: the basis section declares {declared} basis functions, the calculation \
                 has {}. Every AO index after the first mismatch is shifted and the viewer draws \
                 the wrong orbital.",
                basis.nao
            );

            // And each [MO] block writes exactly that many coefficient rows.
            let mut blocks = 0usize;
            let mut rows = 0usize;
            let mut in_mo = false;
            for line in text.lines() {
                if line.starts_with('[') {
                    in_mo = line.starts_with("[MO]");
                    continue;
                }
                if !in_mo {
                    continue;
                }
                if line.starts_with(" Sym=") {
                    if blocks > 0 {
                        assert_eq!(
                            rows, declared,
                            "{label}: an [MO] block has {rows} coefficients against {declared} \
                             declared basis functions"
                        );
                    }
                    blocks += 1;
                    rows = 0;
                } else if line.starts_with(" Ene=")
                    || line.starts_with(" Spin=")
                    || line.starts_with(" Occup=")
                {
                    continue;
                } else if !line.trim().is_empty() {
                    rows += 1;
                }
            }
            assert_eq!(rows, declared, "{label}: the last [MO] block is short");
            let spins = if result.unrestricted { 2 } else { 1 };
            assert_eq!(
                blocks,
                spins * declared,
                "{label}: expected {spins} × {declared} orbitals"
            );

            // The occupied block is the *leading* one, which is what `Occup=` claims: energies
            // ascending, and no occupied orbital above an empty one.
            let energies: Vec<f64> = text
                .lines()
                .filter_map(|l| l.strip_prefix(" Ene="))
                .map(|v| v.trim().parse::<f64>().unwrap())
                .collect();
            let occupations: Vec<f64> = text
                .lines()
                .filter_map(|l| l.strip_prefix(" Occup="))
                .map(|v| v.trim().parse::<f64>().unwrap())
                .collect();
            assert_eq!(energies.len(), occupations.len());
            // Per spin block, ascending and aufbau-filled.
            for chunk in energies.chunks(declared).zip(occupations.chunks(declared)) {
                let (e, f) = chunk;
                for pair in e.windows(2) {
                    assert!(
                        pair[0] <= pair[1] + 1e-12,
                        "{label}: energies not ascending"
                    );
                }
                let mut seen_empty = false;
                for value in f {
                    if *value == 0.0 {
                        seen_empty = true;
                    } else {
                        assert!(
                            !seen_empty,
                            "{label}: an occupied orbital sits above an empty one, so `Occup=` \
                             and the energy order disagree"
                        );
                    }
                }
            }
        }
    }

    /// The `[STO]` form is opt-in, and describes the same wavefunction.
    #[test]
    fn the_slater_form_is_available_and_is_not_the_default() {
        let molecule = water();
        let params = Pm3Parameters::standard().unwrap();
        let result = run_pm3(&molecule, &params, &Pm3Options::default()).unwrap();

        let gto = molden_string(&molecule, &params, &result).unwrap();
        assert!(
            gto.contains("[GTO]") && !gto.contains("[STO]"),
            "GTO is the default"
        );

        let sto = molden_string_with(&molecule, &params, &result, MoldenBasis::Sto).unwrap();
        assert!(sto.contains("[STO]") && !sto.contains("[GTO]"));
        // One Slater function per shell rather than fourteen primitives.
        assert_eq!(
            sto.matches(" s    1 1.00").count(),
            3,
            "one s shell per atom"
        );
        assert_eq!(sto.matches(" p    1 1.00").count(), 1, "oxygen alone has p");
        // The orbital section is the same either way: only the radial description changed.
        let orbitals = |text: &str| text[text.find("[MO]").unwrap()..].to_string();
        assert_eq!(orbitals(&gto), orbitals(&sto));
    }

    /// Numbers come out in the exponent form Fortran list-directed input expects.
    #[test]
    fn exponents_are_written_in_the_conventional_form() {
        assert_eq!(fortran(240.0).trim(), "2.4000000000E+02");
        assert_eq!(fortran(0.0024).trim(), "2.4000000000E-03");
        assert_eq!(fortran(-1.5).trim(), "-1.5000000000E+00");
        let molecule = water();
        let params = Pm3Parameters::standard().unwrap();
        let result = run_pm3(&molecule, &params, &Pm3Options::default()).unwrap();
        let text = molden_string(&molecule, &params, &result).unwrap();

        // Only the primitive lines of the basis section, since the title is prose and the atom
        // lines carry element symbols.
        let mut in_gto = false;
        let mut primitives = 0usize;
        for line in text.lines() {
            if line.starts_with('[') {
                in_gto = line.starts_with("[GTO]");
                continue;
            }
            if !in_gto || line.trim().is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() == 2 && fields[0].contains('E') {
                primitives += 1;
                for field in fields {
                    assert!(
                        !field.contains('e'),
                        "a lowercase exponent survived: {line}"
                    );
                    let (_, exponent) = field.rsplit_once('E').unwrap();
                    assert!(
                        exponent.len() >= 3
                            && (exponent.starts_with('+') || exponent.starts_with('-')),
                        "exponent {exponent} is not the signed two-digit form: {line}"
                    );
                    field
                        .replace('E', "e")
                        .parse::<f64>()
                        .expect("still a number");
                }
            }
        }
        // Water: an s and a p shell on oxygen, an s on each hydrogen, at PRIMITIVES each.
        assert_eq!(primitives, 4 * PRIMITIVES);
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
