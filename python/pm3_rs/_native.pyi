# SPDX-License-Identifier: GPL-3.0-or-later
"""Type stubs for the compiled extension.

The extension returns plain dicts rather than typed objects, so these signatures
are what a type checker has to go on. Keep them in step with `src/python.rs`:
the arguments here are positional in the same order the Rust `#[pyo3(signature)]`
declares them, because `pm3_rs.native` calls them positionally.
"""

from typing import Any, Sequence

Positions = Sequence[Sequence[float]]
Cell = Sequence[Sequence[float]]

def single_point(
    numbers: Sequence[int],
    positions: Positions,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
    field: Sequence[float] | None = ...,
) -> dict[str, Any]: ...
def gradient(
    numbers: Sequence[int],
    positions: Positions,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
    field: Sequence[float] | None = ...,
) -> dict[str, Any]: ...
def forces(
    numbers: Sequence[int],
    positions: Positions,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
    field: Sequence[float] | None = ...,
) -> dict[str, Any]: ...
def optimize(
    numbers: Sequence[int],
    positions: Positions,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
    field: Sequence[float] | None = ...,
    max_steps: int = ...,
    force_tol: float | None = ...,
) -> dict[str, Any]: ...
def frequencies(
    numbers: Sequence[int],
    positions: Positions,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
    field: Sequence[float] | None = ...,
    cphf_max_iter: int | None = ...,
) -> dict[str, Any]: ...
def orbitals(
    numbers: Sequence[int],
    positions: Positions,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
    field: Sequence[float] | None = ...,
) -> dict[str, Any]: ...
def hessian(
    numbers: Sequence[int],
    positions: Positions,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
    field: Sequence[float] | None = ...,
    cphf_max_iter: int | None = ...,
) -> dict[str, Any]: ...
def periodic_single_point(
    numbers: Sequence[int],
    positions: Positions,
    cell: Cell,
    pbc: Sequence[bool] | None = ...,
    kpts: Sequence[int] | None = ...,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
    smearing_ev: float = ...,
    magnetization: str = ...,
) -> dict[str, Any]: ...
def periodic_forces(
    numbers: Sequence[int],
    positions: Positions,
    cell: Cell,
    pbc: Sequence[bool] | None = ...,
    kpts: Sequence[int] | None = ...,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
    smearing_ev: float = ...,
    magnetization: str = ...,
) -> dict[str, Any]: ...
def phonons(
    numbers: Sequence[int],
    positions: Positions,
    cell: Cell,
    pbc: Sequence[bool] | None = ...,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
    q: Sequence[float] | None = ...,
    kpts: Sequence[int] | None = ...,
    smearing_ev: float = ...,
    lo_to_direction: Sequence[float] | None = ...,
    cphf_max_iter: int | None = ...,
) -> dict[str, Any]: ...
def bands(
    numbers: Sequence[int],
    positions: Positions,
    cell: Cell,
    path: Sequence[Sequence[float]],
    pbc: Sequence[bool] | None = ...,
    kpts: Sequence[int] | None = ...,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
    smearing_ev: float = ...,
    magnetization: str = ...,
) -> dict[str, Any]: ...
def relax(
    numbers: Sequence[int],
    positions: Positions,
    cell: Cell,
    pbc: Sequence[bool] | None = ...,
    kpts: Sequence[int] | None = ...,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
    max_steps: int = ...,
    force_tol: float = ...,
    stress_tol: float = ...,
    fixed_cell: bool = ...,
) -> dict[str, Any]: ...
def divide_and_conquer(
    numbers: Sequence[int],
    positions: Positions,
    cell: Cell | None = ...,
    pbc: Sequence[bool] | None = ...,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
    core_radius: float = ...,
    buffer_radius: float = ...,
    smearing_ev: float = ...,
    long_range_cutoff: float | None = ...,
    field: Sequence[float] | None = ...,
) -> dict[str, Any]: ...
def divide_and_conquer_optimize(
    numbers: Sequence[int],
    positions: Positions,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
    core_radius: float = ...,
    buffer_radius: float = ...,
    smearing_ev: float = ...,
    long_range_cutoff: float | None = ...,
    field: Sequence[float] | None = ...,
    max_steps: int = ...,
    force_tol: float = ...,
) -> dict[str, Any]: ...
def divide_and_conquer_forces(
    numbers: Sequence[int],
    positions: Positions,
    cell: Cell | None = ...,
    pbc: Sequence[bool] | None = ...,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
    core_radius: float = ...,
    buffer_radius: float = ...,
    smearing_ev: float = ...,
    long_range_cutoff: float | None = ...,
    field: Sequence[float] | None = ...,
) -> dict[str, Any]: ...
def born_charges(
    numbers: Sequence[int],
    positions: Positions,
    cell: Cell,
    pbc: Sequence[bool] | None = ...,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
    enforce: bool = ...,
    cphf_max_iter: int | None = ...,
) -> dict[str, Any]: ...
def dielectric(
    numbers: Sequence[int],
    positions: Positions,
    cell: Cell,
    pbc: Sequence[bool] | None = ...,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
    include_ionic: bool = ...,
    cphf_max_iter: int | None = ...,
) -> dict[str, Any]: ...
def berry_polarization(
    numbers: Sequence[int],
    positions: Positions,
    cell: Cell,
    strings: int = ...,
    kpts: Sequence[int] | None = ...,
    pbc: Sequence[bool] | None = ...,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
) -> dict[str, Any]: ...
def phonon_bands(
    numbers: Sequence[int],
    positions: Positions,
    cell: Cell,
    supercell: Sequence[int],
    path: Sequence[Sequence[float]],
    points: int = ...,
    pbc: Sequence[bool] | None = ...,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
    enforce_asr: bool = ...,
) -> dict[str, Any]: ...
def finite_field(
    numbers: Sequence[int],
    positions: Positions,
    cell: Cell,
    field: Sequence[float],
    kpts: Sequence[int],
    pbc: Sequence[bool] | None = ...,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
    tol: float = ...,
    max_iter: int = ...,
    mixing: float = ...,
) -> dict[str, Any]: ...
def ir_spectrum(
    numbers: Sequence[int],
    positions: Positions,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
    field: Sequence[float] | None = ...,
    cphf_max_iter: int | None = ...,
) -> dict[str, Any]: ...
def molden(
    numbers: Sequence[int],
    positions: Positions,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
    field: Sequence[float] | None = ...,
    basis: str = ...,
) -> str: ...
def dipole(
    numbers: Sequence[int],
    positions: Positions,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
    field: Sequence[float] | None = ...,
    operator: bool = ...,
) -> dict[str, Any]: ...
def dynamical_matrix(
    numbers: Sequence[int],
    positions: Positions,
    cell: Cell,
    q: Sequence[float],
    pbc: Sequence[bool] | None = ...,
    kpts: Sequence[int] | None = ...,
    charge: float = ...,
    multiplicity: int = ...,
    reference: str = ...,
    method: str = ...,
    rigid_ion: bool = ...,
    smearing_ev: float = ...,
) -> dict[str, Any]: ...
def cli_main(argv: Sequence[str]) -> int: ...
