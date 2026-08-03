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
hessian = native.hessian

__all__ = [
    "native",
    "single_point",
    "gradient",
    "forces",
    "optimize",
    "frequencies",
    "hessian",
]
__version__ = "0.1.2"
