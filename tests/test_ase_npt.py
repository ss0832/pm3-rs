# SPDX-License-Identifier: GPL-3.0-or-later
"""Molecular dynamics through ASE, as an integration test of the periodic path.

# Why dynamics rather than another finite-difference check

A finite-difference test says the forces are the derivative of *the energy this
code computes*. It cannot say that the energy, the forces and the stress are the
same physical quantity — a consistent triple of wrong expressions passes it just
as well as a right one.

Dynamics cannot be fooled that way. In NVE the conserved quantity drifts unless
the forces really are the gradient of the energy, and the drift falls as the
square of the timestep *only* for an exact gradient. Under a barostat the cell
responds to the **stress** while the integrator conserves the **energy**, so a
stress in the wrong units, the wrong sign, or the wrong Voigt order moves the
cell by the wrong amount or in the wrong direction.

# What is asserted

Three things, in increasing strength:

1. **Nothing diverges.** Every energy, force, position, cell and stress sampled
   along the trajectory is finite, the molecule does not dissociate, the cell
   neither collapses nor explodes, and the temperature stays bounded.
2. **The physics is right qualitatively.** Pressure compresses; more pressure
   compresses more; the conserved quantity is conserved.
3. **The physics is right quantitatively.** Halving the timestep quarters the
   NVE drift, and the barostat moves the cell by the amount the Berendsen
   equation predicts from the stress this code reports.

The numbers the bounds are set against were measured with
``tools/bench/md_probe.py``, which can be re-run to re-derive them.

# Why the systems are small

These run in CI. Each is a few atoms for a few dozen steps: enough for a
divergence or a drift to show, not enough to be a production trajectory.
Nothing here claims the sampling is converged.
"""

from __future__ import annotations

import numpy as np
import pytest

ase = pytest.importorskip("ase")

from ase import Atoms, units  # noqa: E402
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution  # noqa: E402
from ase.md.verlet import VelocityVerlet  # noqa: E402

from pm3_rs.ase import PM3  # noqa: E402

BOHR = 0.52917721

# Equilibrium O–H in PM3 water is 0.958 A. Anything outside this band is a
# molecule coming apart or collapsing, not a molecule vibrating.
BOND_MIN, BOND_MAX = 0.80, 1.15


def water_cell(edge_bohr=20.0):
    """One water in a cube wide enough to stay inside the Gamma-point validity
    condition (see docs/pbc.md)."""
    edge = edge_bohr * BOHR
    atoms = Atoms(
        numbers=[8, 1, 1],
        positions=[[0.0, 0.0, 0.0], [0.9584, 0.0, 0.0], [-0.24, 0.9278, 0.0]],
    )
    atoms.set_cell([edge, edge, edge])
    atoms.set_pbc(True)
    return atoms


class Trajectory:
    """Everything sampled along a run, and the sanity checks that apply to all of them."""

    def __init__(self):
        self.total = []
        self.potential = []
        self.temperature = []
        self.volume = []
        self.bonds = []
        self.pressure = []

    def sample(self, atoms, with_stress=False):
        potential = atoms.get_potential_energy()
        kinetic = atoms.get_kinetic_energy()
        self.potential.append(potential)
        self.total.append(potential + kinetic)
        self.temperature.append(atoms.get_temperature())
        self.volume.append(atoms.get_volume())
        positions = atoms.get_positions()
        self.bonds.append(float(np.linalg.norm(positions[1] - positions[0])))
        self.bonds.append(float(np.linalg.norm(positions[2] - positions[0])))
        # Forces are sampled for their finiteness, not stored.
        assert np.all(np.isfinite(atoms.get_forces())), "non-finite force"
        assert np.all(np.isfinite(positions)), "non-finite position"
        assert np.all(np.isfinite(np.asarray(atoms.get_cell()))), "non-finite cell"
        if with_stress:
            stress = atoms.get_stress(voigt=False)
            assert np.all(np.isfinite(stress)), "non-finite stress"
            self.pressure.append(-np.trace(stress) / 3.0 / units.GPa)

    def assert_did_not_diverge(self, label):
        for name, series in (
            ("total energy", self.total),
            ("potential energy", self.potential),
            ("temperature", self.temperature),
            ("volume", self.volume),
            ("bond length", self.bonds),
            ("pressure", self.pressure),
        ):
            array = np.asarray(series, dtype=float)
            if array.size == 0:
                continue
            assert np.all(np.isfinite(array)), f"{label}: {name} went non-finite"

        bonds = np.asarray(self.bonds)
        assert bonds.min() > BOND_MIN and bonds.max() < BOND_MAX, (
            f"{label}: the molecule left the bound region — O-H ranged "
            f"{bonds.min():.3f}-{bonds.max():.3f} A"
        )
        temperature = np.asarray(self.temperature)
        assert temperature.max() < 2000.0, (
            f"{label}: the run heated to {temperature.max():.0f} K, which is a "
            "runaway rather than a fluctuation"
        )
        volume = np.asarray(self.volume)
        assert volume.min() > 0.25 * volume[0] and volume.max() < 4.0 * volume[0], (
            f"{label}: the cell went from {volume[0]:.0f} to "
            f"{volume.min():.0f}-{volume.max():.0f} A^3"
        )

    @property
    def drift(self):
        array = np.asarray(self.total)
        return float(array.max() - array.min())


def run_nve(timestep_fs, steps, method="pm3", seed=7):
    atoms = water_cell()
    atoms.calc = PM3(method=method)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=np.random.default_rng(seed))
    dynamics = VelocityVerlet(atoms, timestep_fs * units.fs)
    trajectory = Trajectory()
    dynamics.attach(lambda: trajectory.sample(atoms), interval=1)
    dynamics.run(steps)
    return trajectory


def run_npt(pressure_gpa, steps, timestep_fs=0.5, taup_fs=500.0, compressibility=0.46):
    from ase.md.nptberendsen import NPTBerendsen

    atoms = water_cell()
    atoms.calc = PM3()
    MaxwellBoltzmannDistribution(atoms, temperature_K=300.0, rng=np.random.default_rng(11))
    dynamics = NPTBerendsen(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=300.0,
        pressure_au=pressure_gpa * units.GPa,
        taut=100 * units.fs,
        taup=taup_fs * units.fs,
        compressibility_au=compressibility / units.GPa,
    )
    trajectory = Trajectory()
    dynamics.attach(lambda: trajectory.sample(atoms, with_stress=True), interval=1)
    dynamics.run(steps)
    return trajectory


class TestNve:
    """The forces must be the gradient of the energy, and exactly so."""

    def test_nothing_diverges_and_the_energy_is_conserved(self):
        trajectory = run_nve(0.5, 120)
        trajectory.assert_did_not_diverge("NVE")

        mean = abs(np.mean(trajectory.total))
        relative = trajectory.drift / mean
        assert relative < 1.0e-4, (
            f"NVE conserved quantity drifted by {trajectory.drift:.3e} eV "
            f"({relative:.3e} of the total), which is not conservation"
        )

    def test_the_drift_falls_as_the_square_of_the_timestep(self):
        """The sharpest statement available about the forces.

        Velocity Verlet's energy error is `O(dt^2)` when the force is an exact
        gradient and `O(dt)` when it is not, so halving the step must **quarter**
        the drift rather than halve it. A force that is merely *close* to the
        gradient — the failure mode a single finite-difference check can miss —
        shows up here as a ratio near two.

        Measured: 4.22 and 4.06 across the two halvings.
        """
        drifts = [run_nve(dt, steps).drift for dt, steps in ((1.0, 60), (0.5, 120), (0.25, 240))]
        ratios = [drifts[0] / drifts[1], drifts[1] / drifts[2]]
        for ratio in ratios:
            assert ratio > 3.0, (
                f"halving the timestep reduced the drift by only {ratio:.2f}x "
                f"(drifts {drifts}); an exact gradient gives about 4x, an "
                "inexact one about 2x"
            )

    def test_the_temperature_settles_by_equipartition(self):
        """Started at 300 K, a bound system should end near half of it.

        All the energy starts as kinetic; at equilibrium a harmonic system shares
        it equally with the potential, so the temperature halves. This is not a
        tight test — three atoms fluctuate hugely — but a run that stayed at
        300 K would mean the potential was not coupling to the motion at all, and
        one that ran to thousands of Kelvin would mean it was coupling wrongly.
        """
        trajectory = run_nve(0.5, 120)
        mean = float(np.mean(trajectory.temperature))
        assert 40.0 < mean < 260.0, (
            f"mean temperature {mean:.0f} K is not equipartition from a 300 K start"
        )

    def test_the_corrected_variant_conserves_too(self):
        """The D3/H4/X forces are analytic and lattice summed; an error in either
        shows up as a drift the plain variant would not have."""
        trajectory = run_nve(0.5, 120, method="pm3-d3h4x")
        trajectory.assert_did_not_diverge("NVE with corrections")
        relative = trajectory.drift / abs(np.mean(trajectory.total))
        assert relative < 1.0e-4, f"drift with corrections {trajectory.drift:.3e} eV"


class TestBarostat:
    """The stress must be the strain derivative of the same energy, in ASE's units."""

    def test_pressure_compresses_and_more_pressure_compresses_more(self):
        """The sign test, plus monotonicity. A stress with the wrong sign expands."""
        volumes = {}
        for pressure in (0.0, 2.0, 5.0):
            trajectory = run_npt(pressure, 80)
            trajectory.assert_did_not_diverge(f"NPT at {pressure} GPa")
            volumes[pressure] = trajectory.volume[-1]
        assert volumes[5.0] < volumes[2.0] < volumes[0.0], (
            f"volume should fall with pressure, got {volumes}"
        )

    def test_the_cell_moves_by_the_amount_the_barostat_equation_predicts(self):
        """The quantitative stress check.

        A Berendsen barostat scales the cell by
        `d(ln V)/dt = -kappa (P_target - P_internal) / tau_p`, so over `n` steps

            ln(V_final / V_initial) = -(dt/tau_p) * kappa * (P_target - <P>) * n

        with `<P>` the internal pressure **this code reports**. Everything on the
        right is either a setting or a measurement, so agreement ties the stress
        to an observable rather than to another of our own numbers. A stress out
        by a factor, or in Pascals instead of eV/A^3, misses by that factor.
        """
        pressure_gpa, steps, timestep_fs, taup_fs, compressibility = 5.0, 80, 0.5, 500.0, 0.46
        trajectory = run_npt(
            pressure_gpa,
            steps,
            timestep_fs=timestep_fs,
            taup_fs=taup_fs,
            compressibility=compressibility,
        )
        trajectory.assert_did_not_diverge("NPT prediction")

        internal = float(np.mean(trajectory.pressure))
        predicted = (
            -(timestep_fs / taup_fs) * compressibility * (pressure_gpa - internal) * steps
        )
        observed = float(np.log(trajectory.volume[-1] / trajectory.volume[0]))
        assert predicted < 0.0, "the setup should compress"
        ratio = observed / predicted
        assert 0.7 < ratio < 1.3, (
            f"the cell moved by ln(V/V0) = {observed:.4f} where the barostat "
            f"equation predicts {predicted:.4f} from our own stress "
            f"(<P> = {internal:.3f} GPa); ratio {ratio:.2f}"
        )

    def test_a_relaxed_cell_reports_near_zero_pressure(self):
        """At a zero-pressure target the internal pressure must stay near zero.

        A dilute water vapour at this density genuinely has almost no pressure,
        so this is a units check with a known answer: a stress off by orders of
        magnitude cannot produce a fraction of a GPa here.
        """
        trajectory = run_npt(0.0, 60)
        trajectory.assert_did_not_diverge("NPT at zero pressure")
        internal = np.asarray(trajectory.pressure)
        assert abs(internal.mean()) < 0.2, (
            f"a nearly ideal gas should show little pressure, not {internal.mean():.3f} GPa"
        )
        assert np.abs(internal).max() < 1.0, (
            f"pressure spiked to {np.abs(internal).max():.3f} GPa"
        )


class TestVariableCell:
    """The energy and the stress must agree about where the minimum is."""

    def test_a_cell_filter_relaxes_towards_zero_stress(self):
        from ase.filters import FrechetCellFilter
        from ase.optimize import BFGS

        atoms = water_cell(edge_bohr=16.0)
        atoms.calc = PM3()
        before = np.abs(atoms.get_stress()).max()
        BFGS(FrechetCellFilter(atoms), logfile=None).run(fmax=0.05, steps=12)
        after = np.abs(atoms.get_stress()).max()
        assert np.all(np.isfinite(atoms.get_stress()))
        assert after < before, (
            f"variable-cell relaxation raised the stress: {before:.3e} -> {after:.3e}"
        )


class TestReducedDimensionality:
    """A slab strains in its plane and nowhere else."""

    def test_a_slab_reports_only_an_in_plane_stress(self):
        atoms = water_cell()
        atoms.set_pbc([True, True, False])
        atoms.calc = PM3()
        assert np.isfinite(atoms.get_potential_energy())
        assert np.all(np.isfinite(atoms.get_forces()))

        stress = atoms.get_stress()
        assert stress.shape == (6,)
        assert np.all(np.isfinite(stress))
        # Exactly zero, not merely small: a masked barostat is entitled to move any
        # direction whose stress is nonzero, and the vacuum thickness is not a
        # degree of freedom.
        for index, label in ((2, "zz"), (3, "yz"), (4, "xz")):
            assert stress[index] == 0.0, f"slab stress {label} should be absent, got {stress[index]}"
        assert abs(stress[0]) + abs(stress[1]) > 0.0, "the in-plane stress should be live"

    def test_a_molecule_still_refuses_a_stress(self):
        """No cell means no strain, and that is reported as absence, not as zero."""
        atoms = water_cell()
        atoms.set_pbc([False, False, False])
        atoms.calc = PM3()
        assert np.isfinite(atoms.get_potential_energy())
        with pytest.raises(Exception):
            atoms.get_stress()

    def test_the_stress_tensor_is_symmetric(self):
        atoms = water_cell()
        atoms.calc = PM3()
        full = atoms.get_stress(voigt=False)
        assert full.shape == (3, 3)
        assert np.allclose(full, full.T, atol=1e-10)
        assert atoms.get_stress().shape == (6,)
