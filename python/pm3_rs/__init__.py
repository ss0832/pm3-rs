# SPDX-License-Identifier: GPL-3.0-or-later
"""pm3-rs: Rust-native PM3 semiempirical method with a Python API.

Top-level convenience re-exports the native (atomic-unit) functions. The ASE
calculator lives in :mod:`pm3_rs.ase` and is imported lazily so that ASE remains
an optional dependency (``pip install pm3-rs-python[ase]``).
"""

from . import native

single_point = native.single_point
gradient = native.gradient
forces = native.forces
optimize = native.optimize
frequencies = native.frequencies
orbitals = native.orbitals
hessian = native.hessian
periodic_single_point = native.periodic_single_point
periodic_forces = native.periodic_forces
phonons = native.phonons
divide_and_conquer = native.divide_and_conquer
divide_and_conquer_optimize = native.divide_and_conquer_optimize
divide_and_conquer_forces = native.divide_and_conquer_forces
born_charges = native.born_charges
dielectric = native.dielectric
finite_field = native.finite_field
berry_polarization = native.berry_polarization
phonon_bands = native.phonon_bands
molden = native.molden
ir_spectrum = native.ir_spectrum
bands = native.bands
relax = native.relax
dipole = native.dipole
dynamical_matrix = native.dynamical_matrix

__all__ = [
    "native",
    "single_point",
    "gradient",
    "forces",
    "optimize",
    "frequencies",
    "orbitals",
    "hessian",
    "periodic_single_point",
    "periodic_forces",
    "phonons",
    "divide_and_conquer",
    "divide_and_conquer_optimize",
    "divide_and_conquer_forces",
    "born_charges",
    "dielectric",
    "finite_field",
    "berry_polarization",
    "phonon_bands",
    "molden",
    "ir_spectrum",
    "bands",
    "relax",
    "dipole",
    "dynamical_matrix",
]

def _installed_version() -> str:
    """The version of the installed distribution, rather than a literal that can drift.

    This used to be a hard-coded string, and by 0.2.5 it still said ``0.2.3``: it had
    survived two releases without anyone noticing, because nothing reads it on the way to
    an answer. Asking the package metadata means it cannot be wrong; the fallback covers
    running straight from the source tree with nothing installed.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("pm3-rs-python")
    except PackageNotFoundError:
        return "0+unknown"


__version__ = _installed_version()
