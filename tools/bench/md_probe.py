#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""What the MD trajectories actually do, so the test bounds come from data.

Prints conservation, temperature, pressure and geometry statistics for the same
runs ``tests/test_ase_npt.py`` performs. Not a test — a measurement, kept so the
numbers in that file can be re-derived rather than taken on trust.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms, units
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from pm3_rs.ase import PM3

BOHR = 0.52917721


def water_cell(edge_bohr=20.0):
    edge = edge_bohr * BOHR
    atoms = Atoms(
        numbers=[8, 1, 1],
        positions=[[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.24, 0.9278, 0.0]],
    )
    atoms.set_cell([edge, edge, edge])
    atoms.set_pbc(True)
    return atoms


def bond_lengths(atoms):
    positions = atoms.get_positions()
    return [
        float(np.linalg.norm(positions[1] - positions[0])),
        float(np.linalg.norm(positions[2] - positions[0])),
    ]


def summarize(label, series):
    array = np.asarray(series, dtype=float)
    finite = np.all(np.isfinite(array))
    print(
        f"  {label:24} min {array.min():12.5f}  max {array.max():12.5f}  "
        f"mean {array.mean():12.5f}  finite {finite}"
    )


def run_nve(timestep_fs, steps, method="pm3"):
    atoms = water_cell()
    atoms.calc = PM3(method=method)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=np.random.default_rng(7))
    dynamics = VelocityVerlet(atoms, timestep_fs * units.fs)

    total, temperature, bonds = [], [], []

    def sample():
        total.append(atoms.get_potential_energy() + atoms.get_kinetic_energy())
        temperature.append(atoms.get_temperature())
        bonds.extend(bond_lengths(atoms))

    dynamics.attach(sample, interval=1)
    dynamics.run(steps)
    return np.asarray(total), np.asarray(temperature), np.asarray(bonds)


def run_npt(pressure_gpa, steps):
    atoms = water_cell()
    atoms.calc = PM3()
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=np.random.default_rng(11))
    dynamics = NPTBerendsen(
        atoms,
        timestep=0.5 * units.fs,
        temperature_K=300.0,
        pressure_au=pressure_gpa * units.GPa,
        taut=100 * units.fs,
        taup=500 * units.fs,
        compressibility_au=0.46 / units.GPa,
    )
    volume, temperature, pressure, bonds = [], [], [], []

    def sample():
        volume.append(atoms.get_volume())
        temperature.append(atoms.get_temperature())
        # Internal pressure from the virial: -tr(sigma)/3, in GPa.
        stress = atoms.get_stress(voigt=False)
        pressure.append(-np.trace(stress) / 3.0 / units.GPa)
        bonds.extend(bond_lengths(atoms))

    dynamics.attach(sample, interval=1)
    dynamics.run(steps)
    return (
        np.asarray(volume),
        np.asarray(temperature),
        np.asarray(pressure),
        np.asarray(bonds),
    )


def main():
    print("NVE, 300 K start:")
    for timestep, steps in [(1.0, 60), (0.5, 120), (0.25, 240)]:
        total, temperature, bonds = run_nve(timestep, steps)
        drift = total.max() - total.min()
        relative = drift / abs(total.mean())
        print(f"  dt = {timestep:4.2f} fs, {steps:3} steps")
        print(f"    conserved drift  {drift:.3e} eV   relative {relative:.3e}")
        summarize("temperature (K)", temperature)
        summarize("O-H bond (A)", bonds)

    print("NVE with pm3-d3h4x:")
    total, temperature, bonds = run_nve(0.5, 120, method="pm3-d3h4x")
    print(f"    conserved drift  {total.max() - total.min():.3e} eV")
    summarize("temperature (K)", temperature)
    summarize("O-H bond (A)", bonds)

    print("NPT (Berendsen), 300 K:")
    for pressure_gpa in [0.0, 2.0, 5.0]:
        volume, temperature, pressure, bonds = run_npt(pressure_gpa, 80)
        print(f"  target {pressure_gpa:4.1f} GPa")
        print(f"    volume {volume[0]:10.2f} -> {volume[-1]:10.2f} A^3")
        summarize("temperature (K)", temperature)
        summarize("internal P (GPa)", pressure)
        summarize("O-H bond (A)", bonds)


if __name__ == "__main__":
    main()
