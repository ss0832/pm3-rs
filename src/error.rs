// SPDX-License-Identifier: GPL-3.0-or-later

use std::fmt;

pub type Result<T> = std::result::Result<T, Pm3Error>;

/// Errors raised across the PM3 pipeline.
///
/// The method is molecular NDDO: there is no periodic cell, and SCF
/// non-convergence is reported on the density residual.
#[derive(Debug)]
pub enum Pm3Error {
    Io(std::io::Error),
    Parse {
        line: usize,
        message: String,
    },
    InvalidInput(String),
    /// No PM3 parameter block exists for this atomic number.
    MissingElement(u8),
    /// A code path was asked to handle a valence d shell that it does not support.
    /// The main PM3 pipeline supports d orbitals; this remains for deliberately
    /// s/p-only low-level helpers.
    UnsupportedDOrbitals(u8),
    /// A named per-element or derived parameter is absent.
    MissingParameter(String),
    LinearAlgebra(String),
    /// A requested calculation would exceed its configured soft workspace limit.
    ResourceLimit {
        operation: &'static str,
        required_mb: usize,
        limit_mb: usize,
    },
    /// The SCF loop hit `max_scf` without reaching the density/energy tolerance.
    ScfNotConverged {
        iterations: usize,
        error: f64,
        /// What the iteration was actually doing when it ran out, when the loop can tell.
        ///
        /// "did not converge (error=3.7e-5)" says a calculation failed and nothing about what to
        /// do next, and the three ways a periodic SCF fails want three different remedies —
        /// charge sloshing wants preconditioning, band crossing wants smearing, and a slow tail
        /// just wants iterations. The k-point loop measures which one it is and says so; paths
        /// that do not yet distinguish them leave this `None` rather than guess.
        diagnosis: Option<String>,
    },
}

impl fmt::Display for Pm3Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::Parse { line, message } => write!(f, "parse error at line {line}: {message}"),
            Self::InvalidInput(msg) => write!(f, "{msg}"),
            Self::MissingElement(z) => write!(f, "missing PM3 parameter block for Z={z}"),
            Self::UnsupportedDOrbitals(z) => {
                write!(f, "this calculation path does not support PM3 d orbitals for Z={z}")
            }
            Self::MissingParameter(key) => write!(f, "missing PM3 parameter `{key}`"),
            Self::LinearAlgebra(msg) => write!(f, "linear algebra error: {msg}"),
            Self::ResourceLimit {
                operation,
                required_mb,
                limit_mb,
            } => write!(
                f,
                "{operation} requires approximately {required_mb} MiB, exceeding the configured {limit_mb} MiB workspace limit"
            ),
            Self::ScfNotConverged {
                iterations,
                error,
                diagnosis,
            } => {
                write!(
                    f,
                    "PM3 SCF did not converge after {iterations} iterations (error={error:.3e})"
                )?;
                match diagnosis {
                    Some(text) => write!(f, ". {text}"),
                    None => Ok(()),
                }
            }
        }
    }
}

impl std::error::Error for Pm3Error {}

impl From<std::io::Error> for Pm3Error {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}
