// SPDX-License-Identifier: GPL-3.0-or-later

//! Atomic geometry and XYZ I/O, for both molecules and periodic systems.
//!
//! Positions are stored in **Bohr** internally. `from_xyz_*` reads Ångström by
//! default (the XYZ convention) and converts on input.
//!
//! A [`Molecule`] carries an optional [`Cell`]. `cell = None` is the molecular case and
//! takes exactly the code path it always did; `Some(cell)` makes the same structure
//! periodic in 1, 2 or 3 directions. Keeping one type — rather than a separate periodic
//! one — is what lets the neighbour list, the classical corrections and the derivative
//! machinery serve both cases from a single implementation.

use crate::cell::Cell;
use crate::constants::ANGSTROM_TO_BOHR;
use crate::error::{Pm3Error, Result};
use crate::math::Vec3;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Atom {
    pub z: u8,
    /// Position in Bohr.
    pub position: Vec3,
}

#[derive(Clone, Debug, Default)]
pub struct Molecule {
    pub atoms: Vec<Atom>,
    /// Total charge (electrons removed = positive). For a periodic system this is the
    /// net charge **per unit cell**; a non-zero value is neutralized by a uniform
    /// background in the Ewald sum, which makes the absolute energy convention-dependent.
    pub charge: f64,
    /// Spin multiplicity (2S+1). 1 = closed-shell singlet.
    pub multiplicity: usize,
    /// Periodic cell, or `None` for a molecule. `None` reproduces the molecular path
    /// exactly; the periodic path is selected purely by this field being present.
    pub cell: Option<Cell>,
}

impl Molecule {
    pub fn new(atoms: Vec<Atom>) -> Self {
        Self {
            atoms,
            charge: 0.0,
            multiplicity: 1,
            cell: None,
        }
    }

    pub fn with_charge(mut self, charge: f64) -> Self {
        self.charge = charge;
        self
    }

    pub fn with_multiplicity(mut self, multiplicity: usize) -> Self {
        self.multiplicity = multiplicity.max(1);
        self
    }

    /// Make the structure periodic in the directions `cell` marks as periodic.
    pub fn with_cell(mut self, cell: Cell) -> Self {
        self.cell = Some(cell);
        self
    }

    /// `true` when this structure is periodic in at least one direction.
    pub fn is_periodic(&self) -> bool {
        self.cell.is_some_and(|c| c.n_periodic() > 0)
    }

    pub fn len(&self) -> usize {
        self.atoms.len()
    }
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }

    /// Sum of atomic numbers (used to derive the electron count).
    pub fn total_nuclear_charge(&self) -> u32 {
        self.atoms.iter().map(|a| a.z as u32).sum()
    }

    pub fn from_xyz_file(path: impl AsRef<Path>, charge: f64) -> Result<Self> {
        Self::from_xyz_str(&fs::read_to_string(path)?, charge)
    }

    /// Parse a standard XYZ block. Coordinates are Ångström and converted to Bohr.
    pub fn from_xyz_str(text: &str, charge: f64) -> Result<Self> {
        // Windows text editors and PowerShell's `-Encoding utf8` both start a UTF-8 file with a
        // byte-order mark, which is invisible everywhere except in front of the atom count, where
        // it turns a perfectly good file into "invalid XYZ atom count: 3".
        let mut lines = text.strip_prefix('\u{feff}').unwrap_or(text).lines();
        let natoms_line = lines
            .next()
            .ok_or_else(|| Pm3Error::InvalidInput("empty XYZ".to_string()))?;
        let natoms = natoms_line.trim().parse::<usize>().map_err(|_| {
            Pm3Error::InvalidInput(format!("invalid XYZ atom count: {natoms_line}"))
        })?;
        let _comment = lines.next().unwrap_or_default();
        // Reserve for what the file can actually hold, not for what its first line claims.
        //
        // The count is the first token of an untrusted file and it used to size the allocation
        // directly, so a corrupted or hostile header ("99999999999") aborted the process on an
        // allocation failure instead of returning `Pm3Error::Parse`. The loop below already
        // refuses a file that ends early, so the honest bound is the number of lines left: it
        // costs one pass over a string that has just been read, and it turns an abort into the
        // error the caller is prepared for.
        let remaining = text.lines().count().saturating_sub(2);
        let mut atoms = Vec::with_capacity(natoms.min(remaining));
        for idx in 0..natoms {
            let line_no = idx + 3;
            let line = lines
                .next()
                .ok_or_else(|| Pm3Error::InvalidInput(format!("XYZ ended before atom {idx}")))?;
            let parts = line.split_whitespace().collect::<Vec<_>>();
            if parts.len() < 4 {
                return Err(Pm3Error::InvalidInput(format!(
                    "XYZ line {line_no} has fewer than 4 fields"
                )));
            }
            let z = symbol_to_z(parts[0]).ok_or_else(|| {
                Pm3Error::InvalidInput(format!("unknown element on line {line_no}: {}", parts[0]))
            })?;
            let position = Vec3::new(
                parse_f64(parts[1], line_no)?,
                parse_f64(parts[2], line_no)?,
                parse_f64(parts[3], line_no)?,
            ) * ANGSTROM_TO_BOHR;
            atoms.push(Atom { z, position });
        }
        Ok(Self {
            atoms,
            charge,
            multiplicity: 1,
            cell: None,
        })
    }
}

fn parse_f64(token: &str, line: usize) -> Result<f64> {
    token.parse::<f64>().map_err(|_| Pm3Error::Parse {
        line,
        message: format!("invalid floating point value: {token}"),
    })
}

pub fn symbol_to_z(sym: &str) -> Option<u8> {
    if let Ok(z) = sym.parse::<u8>() {
        if (1..=107).contains(&z) {
            return Some(z);
        }
    }
    let s = normalize_symbol(sym);
    ELEMENTS
        .iter()
        .position(|&x| x == s.as_str())
        .map(|i| i as u8)
}

pub fn z_to_symbol(z: u8) -> Option<&'static str> {
    ELEMENTS.get(z as usize).copied().filter(|s| !s.is_empty())
}

fn normalize_symbol(sym: &str) -> String {
    let mut chars = sym.chars();
    match chars.next() {
        Some(first) => {
            let mut out = String::new();
            out.push(first.to_ascii_uppercase());
            for c in chars {
                out.push(c.to_ascii_lowercase());
            }
            out
        }
        None => String::new(),
    }
}

pub const ELEMENTS: [&str; 108] = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Mi", "XX", "+3", "-3", "Cb", "++", "+", "--", "-", "Tv",
];

#[cfg(test)]
mod tests {
    use super::*;

    const WATER: &str = "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n";

    /// Saving an XYZ file from Notepad, or from PowerShell's `Set-Content -Encoding utf8`, puts a
    /// byte-order mark in front of the atom count. The file looks identical in every editor and
    /// used to be rejected with a message naming a number that was plainly a number.
    #[test]
    fn a_byte_order_mark_does_not_make_a_file_unreadable() {
        let plain = Molecule::from_xyz_str(WATER, 0.0).unwrap();
        let marked = Molecule::from_xyz_str(&format!("\u{feff}{WATER}"), 0.0).unwrap();

        assert_eq!(marked.atoms.len(), plain.atoms.len());
        for (a, b) in marked.atoms.iter().zip(&plain.atoms) {
            assert_eq!(a.z, b.z);
            assert!((a.position - b.position).norm() < 1.0e-15);
        }
    }

    /// One mark is stripped, not any number of them: a second is a genuinely malformed file and
    /// should still say so rather than being silently tidied away.
    #[test]
    fn only_the_leading_mark_is_forgiven() {
        assert!(Molecule::from_xyz_str(&format!("\u{feff}\u{feff}{WATER}"), 0.0).is_err());
    }
}
