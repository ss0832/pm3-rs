# Divide and conquer

Diagonalizing an `N × N` Fock matrix costs `O(N³)`. Diagonalizing `N/m` matrices
of size `m` costs `O(N m²)`, which is linear in `N`. Divide and conquer cuts the
system into pieces, solves each in the presence of a buffer, and reassembles the
density.

## What makes it work

The density matrix of a system with a gap is **near-sighted**: `P_μν` decays
exponentially with the distance between the two orbitals. A subsystem large
enough to contain that decay length reproduces its own core's density to within
the truncation, and the pieces reassemble exactly.

The buffer radius is therefore the one knob that matters. Increasing it must
converge the result onto the full diagonalization. **If it does not, the system
is not near-sighted** — it is metallic, or the gap has closed — and divide and
conquer is the wrong method rather than a poorly converged one.

## How the pieces fit together

Every atom belongs to exactly **one** core; buffers overlap freely. A density
element is assembled from the subsystems that can see both of its orbitals:

```text
D^α_μν = 1     both μ and ν in core α
       = 1/2   one in core α, the other in α's buffer
       = 0     otherwise
```

These sum to exactly 1 over `α` for every pair that any subsystem covers. Take
`μ ∈ core α` and `ν ∈ core β` with `α ≠ β`: if each core reaches into the other's
buffer, both contribute a half. If neither does, the element is set to **zero** —
and that, rather than any weighting subtlety, is the entire approximation.

`Partition::weight_sum` reports the actual sums so a test can assert they are
exactly one or exactly zero. A pair scaled by anything in between would give a
plausible but wrong energy.

## One chemical potential, not one per subsystem

Subsystems are diagonalized independently but they are not independent systems:
electrons flow between them. Filling each to its own electron count would freeze
whatever charge distribution the partitioning happened to imply.

Instead every subsystem is occupied from a **single global Fermi level**,
bisected until the assembled density holds the right number of electrons. That is
what lets charge redistribute, and it is why a charged system needs no special
handling here: the constraint is the electron count, and a charged system simply
has a different one.

The local eigenvalues are occupied with a small Fermi–Dirac broadening rather
than a step. This is not optional: subsystems are diagonalized independently, so
a hard cutoff would let an orbital jump between "occupied here, empty there" from
one iteration to the next and the electron count would oscillate.

## Derivatives

Forces and stress come from the divide-and-conquer density through the same
expressions the full paths use. ZDO removes the Pulay term, so the gradient is
`Tr[P ∂H]` with no overlap-constraint piece and no energy-weighted density
matrix — which means the gradient expression never asks where the density came
from.

One caveat, stated rather than hidden: a divide-and-conquer density is not
variational. It minimizes nothing; it is assembled. So the usual argument that
the first-order energy error vanishes at the SCF solution does not apply, and the
gradient carries the density's own truncation error rather than its square. In
practice it converges with the buffer the same way the energy does, which the
tests check directly.

## Using it

```rust
use pm3_rs::{run_dc, dc_gradient, DcOptions};

let dc = DcOptions {
    core_radius: 6.0,      // Bohr
    buffer_radius: 9.0,    // Bohr — the knob that matters
    ..DcOptions::default()
};
let result = run_dc(&molecule, &params, &options, &dc)?;
println!("{} eV, {} subsystems", result.total_ev, result.n_subsystems);
```

`run_dc_gamma` is the periodic form. A subsystem does not know or care whether
the potential it sits in came from a lattice sum, so only the Fock build differs
— and the Γ-point validity condition from [`pbc.md`](pbc.md) applies unchanged.
`DcPeriodicResult::gamma_margin` reports it.

From Python (radii in Ångström there, like every other length):

```python
import pm3_rs

r = pm3_rs.divide_and_conquer(numbers, positions, buffer_radius=5.0)
print(r["energy_ev"], r["n_subsystems"], r["dropped_pairs"])
```

Adding `long_range_cutoff` switches on the linear-scaling near field. It is an
Ångström length like every other distance on the Python surface, and widening it
converges back onto the dense answer:

```python
r = pm3_rs.divide_and_conquer(
    numbers, positions, buffer_radius=5.0, long_range_cutoff=12.0
)
```

## Validation

- Widening the buffer converges the energy onto the full diagonalization
  (`5e-4` eV) and the forces onto the full gradient (`1e-4` eV/Bohr).
- A buffer reaching the whole system reproduces the full result **exactly**
  (`1e-6` eV), which separates the partitioning from the truncation: a failure
  there is wiring, not approximation. The same holds for the gradient, and for
  the periodic path including the stress (`1e-9`).
- The electron count is exact at `Q = 0`, `+1` and `−1`.
- Two different core radii agree, so the answer does not depend on where the cuts
  fall.
- An open-shell system polarizes to exactly the requested moment.
- The classical corrections are bit-identical across partitionings.

One thing the tests deliberately do **not** assert: that the error decreases at
every buffer radius. It does not have to. Widening the buffer changes which atoms
fall in which subsystem, so a term that happened to cancel at one radius need not
cancel at the next. The trend is downward and the wide-buffer limit is right;
step-by-step monotonicity is not a property of the method.

## Performance, honestly

On water chains, measured on the same machine in one run, release profile,
360 to 2880 atoms:

| | 2880 atoms | log-log slope |
|---|---|---|
| full diagonalization | 680.3 s | 2.76 |
| divide and conquer | 39.3 s (17.3×) | 1.88 |
| …with `long_range_cutoff` | 7.9 s (**86.4×**) | **1.16** |

Per size, for the linear-scaling path:

| atoms | full | dc | linear | speed-up | error |
|---|---|---|---|---|---|
| 360 | 2.58 s | 1.06 s | 0.73 s | 3.5× | 27.6 µeV/atom |
| 720 | 11.62 s | 3.79 s | 1.57 s | 7.4× | 28.1 µeV/atom |
| 1440 | 126.72 s | 34.69 s | 3.88 s | 32.7× | 28.3 µeV/atom |
| 2880 | 680.27 s | 39.26 s | 7.88 s | 86.4× | 28.4 µeV/atom |

The error column is the point of the table as much as the times are: the
partitioning costs about 28 µeV per atom and that number does **not** grow with
the system. An approximation whose per-atom error were size-dependent would be
buying its speed from accuracy rather than from structure.

**Slope 1.16, against an acceptance criterion of 1.15.** Missed, narrowly, and
recorded as missed. The previous measurement of this table read 1.27 and
predated the multipole tree for the far field; that tree is what took it to
1.16, and the last three points alone give 1.17, so the figure is not still
falling with size.

**Not slope 1**, either. The method removes the `O(N³)` diagonalization, which
was never what dominated at these sizes, and four separate things had to follow
before even 1.3 was reachable:

- **The density algebra.** A partitioned density is zero wherever no subsystem
  holds both orbitals, and storing it densely made DIIS, damping and the RMS
  change all `O(N²)` on numbers that are structurally absent. `dc::pattern`
  records the live set once; the DIIS history alone fell from 7.17 s to 0.19 s
  at 960 atoms.
- **Allocation.** Every matrix the SCF loop touches is now allocated once. A
  fresh `nao × nao` per iteration is thirty megabytes at a thousand atoms,
  allocated, zeroed and discarded.
- **The near field.** `DcOptions::long_range_cutoff` splits the Coulomb term the
  way the periodic path does — full pair tables inside the cutoff, a
  point-charge model outside — so the two-electron work grows with the system
  rather than with its square.

- **The far field's own units.** The point-charge sum outside the cutoff ran over
  *sites*, and a heavy atom carries twenty-five of them, so two distant atoms cost
  six hundred and twenty-five pair evaluations to answer a question their net
  charge, dipole and quadrupole already answer. Past 80 Bohr each atom's cloud is
  now collapsed onto those moments and the potential is carried back out to its
  sites by a Taylor expansion of the same order. The error is measured, not
  assumed: 8 µeV over forty-eight atoms in `pbc::ewald`'s own test, and no change
  at all in the 28 µeV/atom below. Together with dropping the site gradients the
  self-consistent field never reads, this took the Fock build at 960 atoms from
  2.52 s to 1.22 s.

- **The matrices themselves.** The eight `nao × nao` arrays the loop held — two
  densities, two Fock matrices and four workspaces — are stored on the pattern
  now, not dense. Every one of them was already read back only through the
  pattern, and the subsystem gather looks exactly where some subsystem holds both
  orbitals, which is what the pattern *is*; so nothing outside it was ever
  consulted and nothing changes numerically. The Fock build writes into the
  pattern directly rather than filling an array to have most of it ignored.
  Measured on a chain of waters:

  | waters | nao | dense | on the pattern | ratio |
  |---:|---:|---:|---:|---:|
  | 8 | 48 | 0.14 MiB | 0.141 MiB | 1.0× |
  | 32 | 192 | 2.25 | 1.122 | 2.0× |
  | 128 | 768 | 36.00 | 5.059 | 7.1× |

  The dense column quadruples per doubling — slope exactly 2. The sparse one
  grows by 2.08 over the last doubling: **slope 1.06**. Below about thirty atoms
  the pattern is nearly dense and there is nothing to save; what grows is the
  ratio.

- **The far field's own scaling.** The point-multipole sum outside the handover
  was quadratic in atoms. It now runs over an octree whose nodes carry the
  combined moments of everything beneath them, so a distant group is one term
  rather than many. Two things make it safe rather than merely fast:

  - A node is taken whole only when `d − s > MULTIPOLE_RADIUS`, so by the
    triangle inequality **every** cloud inside it was already far by the pairwise
    rule. The near field is therefore exactly the set it was, to the bit.
  - The acceptance test is an **absolute** error bound, not an opening angle.
    Barnes–Hut's `s/d < θ` is relative, and what has to be matched here is
    absolute: a cloud is taken whole past 80 Bohr, where the first term it drops
    is `Σ|q| s³/d⁴ ≈ 1e-7`. A node is accepted when the same estimate of its own
    first neglected term is below that. A fixed angle is looser where it matters:
    `θ = 0.3` moved the energy by 1.5 meV against a 50 µeV tolerance.

  Counted rather than timed — this machine runs other work, so a wall-clock slope
  would measure the load as much as the algorithm, while the number of source
  terms each target sums is a property of the tree and the geometry alone. Over a
  compact block growing 125 → 1000 clouds, the far-field term count grows with a
  log-log slope of **1.58** against the pairwise sum's exact 2.00, and at 1000
  clouds it does 18% of the pairwise work.

  It is not `O(N log N)`, and what limits it is the accuracy criterion rather
  than the tree. Accepting a node of `M` clouds needs
  `Σ|q| s³/d⁴ < 1e-7`; with `Σ|q| ∝ M` and `s ∝ M^⅓` in three dimensions that is
  `d ∝ √M`, so large nodes are only taken from far away. Reaching `O(N log N)`
  means carrying the octupole so the same accuracy is bought at smaller `d/s` —
  a term the expansion does not have, and a derivation rather than a tuning.

  There is a size below which none of this applies. The handover is 80 Bohr, so a
  system narrower than about 160 Bohr across has no far field at all — every pair
  is near. At liquid density that is some tens of thousands of atoms.

What is still quadratic is the unscreened molecular path's pair cache:
`core.pairs` holds every pair, which is what `DcOptions::long_range_cutoff`
exists to replace. No `nao × nao` array is held across an iteration on either
path any more — the core Hamiltonian goes onto the pattern once and the dense
copy is released, and the periodic Fock build has a sparse form of its own. Only
the density a caller asked for is materialized dense, at the end, because that is
what the public result holds.

The plan's wall-clock target of 1.15 is still not met, and the memory and the
far-field count it blamed are now linear and 1.58 respectively.

`long_range_cutoff` is `None` by default, because it is not free: the
Klopman–Ohno switch costs a measured **28 µeV per atom** at a 22 Bohr handover,
roughly a hundred times the divide-and-conquer truncation error per atom at the
default buffer. It saturates rather than accumulating — 25.9, 27.2, 27.9 and
28.2 µeV/atom at 120, 240, 480 and 960 atoms — so it is a fixed price, and
widening the cutoff shrinks it as `r⁻³`.

```bash
cargo run --release --example scaling -- 10,20,40,80,160,320
```

prints the table and fits the slopes, so the claim above can be re-measured
rather than taken on trust. Setting `PM3_DC_PROFILE=1` adds a line per run
splitting the loop into the Fock build, the subsystem solves and the density
assembly, which is how the paragraphs above were written: guessing which of the
three dominated was wrong twice.
