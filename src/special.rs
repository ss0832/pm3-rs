// SPDX-License-Identifier: GPL-3.0-or-later

//! Error functions, which the Rust standard library does not provide.
//!
//! The Ewald sum needs `erfc` in its real-space term and — for the reduced-dimensionality
//! variants, where the reciprocal-space term contains products like `exp(+G z)·erfc(αz + G/2α)`
//! whose factors overflow long before the product does — the scaled complementary error
//! function `erfcx(x) = exp(x²)·erfc(x)`.
//!
//! Both branches below are derived rather than transcribed from a coefficient table, so their
//! accuracy can be argued from the series themselves:
//!
//! * For small argument, the **all-positive** series
//!   `erf(x) = (2x/√π)·e^{−x²}·Σ_n (2x²)^n / (1·3·5···(2n+1))`
//!   (Abramowitz & Stegun 7.1.6). Every term is positive, so unlike the alternating Maclaurin
//!   series it suffers no cancellation, and the truncation error is bounded by the first
//!   omitted term.
//! * For large argument, the continued fraction
//!   `erfcx(x) = 1/(√π) · 1/(x + ½/(x + 1/(x + 3/2/(x + …))))`
//!   evaluated by the modified Lentz algorithm, which converges quickly once `x ≳ 4`.
//!
//! The two branches overlap, and `branches_agree_where_they_meet` pins that they agree there
//! to the precision the series can support — the sharpest available check that neither is
//! mis-derived, since the two share no algebra. Measured against `scipy.special` over 157
//! points (`tools/oracle/check_erf.py`), the worst relative error is 8.9e-16 for `erf` and
//! 3.9e-14 for `erfc`/`erfcx`.

use std::f64::consts::PI;

/// Argument above which `erf` switches from the series to `1 − erfc`. `erf` itself is O(1)
/// here, so the switch point only has to be where the continued fraction is converged.
const BRANCH: f64 = 4.0;

/// Argument above which `erfc` must use the continued fraction rather than `1 − erf`.
///
/// This is much lower than [`BRANCH`] and the difference matters: `erfc` is the *small*
/// quantity, so forming it as `1 − erf` cancels away roughly `−log10(erfc)` digits. At
/// x = 3.9, where `erfc ≈ 1.1e-8`, that subtraction keeps only 8 of the 16 digits — measured
/// against scipy the error was 1.2e-8 relative before this branch existed. Below 2 the
/// subtraction costs under a digit and the continued fraction has not yet converged, so 2 is
/// where the two errors cross.
const ERFC_BRANCH: f64 = 2.0;

/// Beyond this, `erfc` underflows a double (`erfc(27) ≈ 5e-319`).
const ERFC_UNDERFLOW: f64 = 27.0;

/// Error function `erf(x) = (2/√π) ∫₀ˣ e^{−t²} dt`.
pub fn erf(x: f64) -> f64 {
    if x.abs() < BRANCH {
        erf_series(x)
    } else {
        // erfc is tiny here, so erf = 1 − erfc loses nothing.
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        sign * (1.0 - erfc_large(x.abs()))
    }
}

/// Complementary error function `erfc(x) = 1 − erf(x)`.
///
/// Accurate in the tail, where `1 − erf(x)` would cancel to nothing: `erfc(6)` is 2e-17, which
/// the subtraction cannot represent at all but the continued fraction gives to full precision.
pub fn erfc(x: f64) -> f64 {
    if x < -ERFC_BRANCH {
        2.0 - erfc(-x)
    } else if x < ERFC_BRANCH {
        1.0 - erf_series(x)
    } else {
        erfc_large(x)
    }
}

/// Scaled complementary error function `erfcx(x) = e^{x²}·erfc(x)`.
///
/// For positive `x` this decays like `1/(x√π)` instead of `e^{−x²}`, which is what makes the
/// reduced-dimension Ewald kernels evaluable: `e^{Gz}·erfc(αz + G/2α)` is rewritten as
/// `e^{Gz − (αz + G/2α)²}·erfcx(αz + G/2α)`, where the exponent is bounded even though each
/// original factor is not.
pub fn erfcx(x: f64) -> f64 {
    if x >= ERFC_BRANCH {
        erfcx_cf(x)
    } else if x >= 0.0 {
        (x * x).exp() * (1.0 - erf_series(x))
    } else {
        // erfcx(−x) = 2e^{x²} − erfcx(x); for very negative x the first term overflows,
        // which is the correct answer (erfcx does diverge there).
        2.0 * (x * x).exp() - erfcx(-x)
    }
}

/// `erf` by the all-positive series. Valid everywhere but only used for `|x| < BRANCH`,
/// where it converges in a few tens of terms.
fn erf_series(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let x2 = x * x;
    let mut term = 1.0; // (2x²)^n / (1·3·5···(2n+1)), starting at n = 0
    let mut sum = 1.0;
    for n in 1..200 {
        term *= 2.0 * x2 / (2.0 * n as f64 + 1.0);
        sum += term;
        if term < 1.0e-18 * sum {
            break;
        }
    }
    2.0 * x / PI.sqrt() * (-x2).exp() * sum
}

/// `erfc` for `x >= BRANCH`, via the scaled form so the exponential is applied once.
fn erfc_large(x: f64) -> f64 {
    if x > ERFC_UNDERFLOW {
        return 0.0;
    }
    (-x * x).exp() * erfcx_cf(x)
}

/// `erfcx` by the continued fraction `1/√π · 1/(x + ½/(x + 1/(x + 3/2/(x + …))))`,
/// evaluated with modified Lentz. Converges for `x ≳ 2`; used from `BRANCH` up.
fn erfcx_cf(x: f64) -> f64 {
    const TINY: f64 = 1.0e-300;
    let mut f = TINY;
    let mut c = f;
    let mut d = 0.0_f64;
    // b0 = x, then a_n = n/2, b_n = x.
    for n in 0..300 {
        let a = if n == 0 { 1.0 } else { n as f64 / 2.0 };
        let b = x;
        d = b + a * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + a / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = c * d;
        f *= delta;
        if (delta - 1.0).abs() < 1.0e-16 {
            break;
        }
    }
    f / PI.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two branches must agree where both are valid — the sharpest check that neither
    /// the series nor the continued fraction is mis-derived, since they share no algebra.
    ///
    /// The tolerance has to follow the cancellation in `1 − erf_series`, which is the whole
    /// reason [`ERFC_BRANCH`] exists: at argument `x` that subtraction retains only
    /// `eps / erfc(x)` relative accuracy. Comparing against that bound tests the continued
    /// fraction as sharply as the series allows at each point, instead of picking one loose
    /// tolerance that would let a real error through at small `x`.
    #[test]
    fn branches_agree_where_they_meet() {
        for k in 0..40 {
            let x = 1.0 + 0.1 * k as f64; // 1.0 .. 4.9, spanning ERFC_BRANCH and BRANCH
            let series = 1.0 - erf_series(x);
            let fraction = erfc_large(x);
            let relative = (series - fraction).abs() / fraction;
            let cancellation_bound = 20.0 * f64::EPSILON / fraction;
            assert!(
                relative < cancellation_bound.max(1.0e-15),
                "x={x}: series {series:.17e} vs continued fraction {fraction:.17e} \
                 (rel {relative:.2e}, cancellation bound {cancellation_bound:.2e})"
            );
        }
    }

    /// `erf` must be the antiderivative it claims to be: `d/dx erf(x) = (2/√π) e^{−x²}`.
    ///
    /// The comparison is absolute, not relative: a central difference of a function whose
    /// values are O(1) carries `~eps/h` of noise regardless of how small the derivative
    /// itself has become, so a relative test would be measuring the finite difference rather
    /// than `erf`.
    #[test]
    fn derivative_matches_the_defining_integrand() {
        let h = 1.0e-5;
        let noise = f64::EPSILON / h; // ~2e-11, the floor a central difference can resolve
        for k in 0..60 {
            let x = 0.05 + 0.1 * k as f64;
            let numeric = (erf(x + h) - erf(x - h)) / (2.0 * h);
            let exact = 2.0 / PI.sqrt() * (-x * x).exp();
            let tolerance = 1.0e-9 * exact + 10.0 * noise;
            assert!(
                (numeric - exact).abs() < tolerance,
                "x={x}: d/dx erf = {numeric:.12e}, expected {exact:.12e}"
            );
        }
    }

    #[test]
    fn identities_hold() {
        assert_eq!(erf(0.0), 0.0);
        assert!((erfc(0.0) - 1.0).abs() < 1.0e-16);
        for k in 0..50 {
            let x = 0.05 + 0.2 * k as f64;
            assert!((erf(-x) + erf(x)).abs() < 1.0e-15, "erf is not odd at {x}");
            assert!(
                (erfc(x) + erf(x) - 1.0).abs() < 1.0e-15,
                "erfc + erf != 1 at {x}"
            );
            assert!(
                (erfc(-x) - (2.0 - erfc(x))).abs() < 1.0e-15,
                "erfc(-x) != 2 - erfc(x) at {x}"
            );
            let scaled = erfcx(x);
            let plain = erfc(x);
            if plain > 0.0 {
                let relative = (scaled * (-x * x).exp() - plain).abs() / plain;
                assert!(
                    relative < 1.0e-13,
                    "erfcx inconsistent at {x}: {relative:.2e}"
                );
            }
        }
    }

    /// A handful of reference values, to catch a systematically wrong normalisation that the
    /// self-consistency checks above would not notice.
    #[test]
    fn reference_values() {
        let cases = [
            (0.5_f64, 0.520_499_877_813_046_5_f64),
            (1.0, 0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            (3.0, 0.999_977_909_503_001_4),
        ];
        for (x, expected) in cases {
            let got = erf(x);
            assert!(
                (got - expected).abs() < 1.0e-14,
                "erf({x}) = {got:.17e}, expected {expected:.17e}"
            );
        }
        // Tail values, where only the continued fraction can reach them at all.
        assert!((erfc(5.0) / 1.537_459_794_428_035_1e-12 - 1.0).abs() < 1.0e-12);
        assert!((erfc(10.0) / 2.088_487_583_762_545e-45 - 1.0).abs() < 1.0e-11);
        // erfcx decays algebraically, so it stays representable where erfc has underflowed.
        assert_eq!(erfc(40.0), 0.0);
        assert!(erfcx(40.0) > 0.0 && erfcx(40.0) < 0.02);
        // Leading asymptotic behaviour erfcx(x) -> 1/(x sqrt(pi)).
        assert!((erfcx(1000.0) * 1000.0 * PI.sqrt() - 1.0).abs() < 1.0e-6);
    }

    /// The series must stay monotone and bounded — a runaway term count or an overflow in the
    /// recurrence would show up here rather than as a wrong digit somewhere.
    ///
    /// Monotonicity is only testable where `erf` is still resolvable: beyond |x| ≈ 5.9 it has
    /// reached ±1 to the last bit and consecutive samples are exactly equal, so the check is
    /// "never decreasing", tightened to "strictly increasing" wherever the values differ.
    #[test]
    fn erf_is_monotone_and_bounded() {
        let mut previous = f64::NEG_INFINITY;
        let mut strict_increases = 0;
        for k in 0..500 {
            let x = -6.0 + 0.024 * k as f64;
            let value = erf(x);
            assert!(value.is_finite());
            assert!(
                (-1.0..=1.0).contains(&value),
                "erf({x}) = {value} out of range"
            );
            assert!(value >= previous, "erf decreased at {x}");
            if value > previous {
                strict_increases += 1;
            }
            previous = value;
        }
        // Saturation should only account for the far tails, not most of the range.
        assert!(
            strict_increases > 400,
            "erf was flat over too much of the range ({strict_increases} strict increases)"
        );
    }
}
