# Periodic boundary conditions

pm3-rs runs PM3 under 1D, 2D, and 3D periodic boundary conditions. This page
covers what is computed, the approximations that are made, and — most
importantly — the conditions under which a periodic result is trustworthy.

Everything here is per unit cell. Units are eV and Bohr unless stated.

## Attaching a cell

A `Molecule` becomes periodic when it carries a `Cell`:

```rust
use pm3_rs::{Cell, Molecule};

let mut molecule = Molecule::from_xyz_str(xyz, 0.0)?;
molecule.cell = Some(Cell::cubic(20.0)?);          // 20 Bohr, periodic in x, y, z
```

`Cell::new(a, b, c, pbc)` takes three lattice vectors in Bohr and a
`[bool; 3]` saying which directions are periodic. The non-periodic directions
of a slab or a chain are ignored throughout: they do not enter the volume, the
reciprocal basis, or the stress.

| `pbc` | Dimension | `measure()` returns |
|---|---|---|
| `[true, true, true]` | 3D crystal | volume |
| `[true, true, false]` | 2D slab | area |
| `[true, false, false]` | 1D chain | length |

A slab's vacuum thickness therefore never leaks into any result. That is worth
stating explicitly because the common "pad with vacuum and use a 3D method"
recipe does let it leak, and the answer then depends on how much vacuum was
added.

## What is computed

`run_gamma` returns a `PeriodicResult` with the Γ-point SCF; `periodic_gradient`
adds analytic forces and the analytic stress; `relax` optimizes the geometry and,
optionally, the cell.

```rust
use pm3_rs::pbc::gamma::{run_gamma, PeriodicOptions};

let result = run_gamma(&molecule, &params, &options, &PeriodicOptions::default())?;
println!("{} eV per cell", result.total_ev);
```

RHF and UHF are both available and are selected exactly as in the molecular
path, from the charge and multiplicity, or forced with `Reference::Uhf`.

## When Γ alone is enough

**Read this before trusting a periodic number.**

One k-point cannot resolve the density matrix by image. Bloch theory gives

```text
P(0, T) = Σ_k w_k e^{−ik·T} P(k)
```

and with `k = Γ` the only term available, `P(0, T) = P(Γ)` for *every* `T`.

This costs nothing in the Coulomb terms, which depend on the on-site density
and are lattice summed properly. It is fatal in the exchange, whose true weight
`P(0, T)` decays exponentially with `|T|` while the substitute `P(Γ)` does not
decay at all.

The condition is sharp, purely geometric, and has nothing to do with
convergence:

> **Every periodic width must exceed `PeriodicOptions::short_range_cutoff`**
> (14 Bohr by default).

`PeriodicResult::gamma_margin` reports the narrowest periodic width minus that
cutoff. Positive is fine; negative is not. Measured on a cubic lattice of water,
comparing one cell against a `2×2×2` supercell of the same crystal:

| Cell edge | `gamma_margin` | Supercell disagreement |
|---|---|---|
| 22 Bohr | +8 | `< 1e-8` eV |
| 18 Bohr | +4 | `5.7e-11` eV |
| 16 Bohr | +2 | `1.4e-4` eV |
| 14 Bohr | 0 | **`3.9e1` eV** |

Nothing in the SCF reacts to a violation. At 14 Bohr it converges cleanly, in
the usual number of iterations, to an answer that is wrong by 38 eV per cell.

This is not a defect peculiar to NDDO — it is the ordinary reason a Γ-only
calculation needs a supercell. A crystal whose bonding runs through its images
(graphene, a metal, a chain) has a permanently negative margin and genuinely
requires a k-mesh rather than a bigger cell.

## k-points

`run_kpoints` samples the Brillouin zone instead of taking only Γ, which lifts
the restriction above: with more than one `k`, `P(0, T)` can decay with `T`, so
the cell no longer has to be wider than the exchange range.

```rust
use pm3_rs::pbc::kscf::{run_kpoints, KpointOptions};

let result = run_kpoints(&molecule, &params, &options, &periodic,
                         &KpointOptions::mesh([4, 4, 4]))?;
println!("{} eV per cell, gap {:?}", result.total_ev, result.band_gap_ev);
```

Meshes are **Γ-centred** Monkhorst–Pack. That is a deliberate choice: the
Γ-centred mesh is exactly the set of `k` allowed by an `N₁×N₂×N₃` supercell's
periodicity, so it — and not the zone-centred original — satisfies
`E(mesh) == E(supercell at Γ)/N`. `KpointSpec::Mesh { shift, .. }` gives the
other convention, in units of half a mesh spacing.

Non-periodic directions must have a division of 1. A slab has no dispersion
along its normal, so a request to sample it is rejected rather than quietly
reduced — quietly correcting it would hide a mistake in the caller's setup.

**Time-reversal reduction** halves the work exactly, not approximately. With a
real Hamiltonian `H(−k) = H(k)*`, so `−k` has the same eigenvalues and a
conjugate density; only one member of each `±k` pair is diagonalized and its
weight is doubled. Points that are their own negative (Γ and the zone-boundary
points) keep their weight.

**Occupations** are global. Bands at different `k` interleave, so filling each
k-point to its own electron count would be a different — and wrong —
calculation; there is one Fermi level across the whole mesh.
`smearing_ev = 0` fills strictly by energy (exact for an insulator, capable of
oscillating for a metal); a positive value bisects a Fermi–Dirac distribution
onto the electron count. For unrestricted runs, `Magnetization::Fixed` holds
`n_α − n_β` at what the multiplicity asks for (two Fermi levels), while
`Magnetization::Free` shares one Fermi level and lets the moment come out as an
output.

**Corrections do not depend on `k`.** D3/H4/X are classical and post-SCF, so
they are lattice summed once per cell, outside the k loop, and reported in a
field of their own. A test asserts they are invariant under the mesh, since the
natural bug is to add them once per k-point.

### Band structure

`band_structure` evaluates a path in the converged potential, without
re-converging — a path is not a quadrature and its points carry no meaningful
integration weight.

```rust
use pm3_rs::pbc::kpoints::band_path;
use pm3_rs::pbc::kscf::band_structure;

let path = band_path(&cell, &[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], 40)?;
let bands = band_structure(&molecule, &params, &periodic, &result, &path)?;
```

`BandStructure::distances` is the cumulative Cartesian path length, which is the
right horizontal axis for a plot.

### Forces and stress at k-points

`kpoint_gradient` returns the same quantities `periodic_gradient` does, from the
same automatic differentiation. The only difference is where the inter-atomic
density comes from: `P_ij(T)` for the image being differentiated, rather than one
`P(Gamma)` shared by every image. Only two terms read one — the resonance and the
exchange — and both are gated on the same cutoff.

The forces are checked against finite differences of the k-point energy, fold
onto the equivalent supercell atom for atom, and sum to zero; the stress is
checked against finite differences of the energy under strain.

## Long-range electrostatics

The NDDO two-electron kernel is Klopman–Ohno screened,
`q_i q_j / sqrt(r² + a)`, which is a point-charge interaction plus a
`−a/2r³` tail. Under a 3D lattice sum that tail diverges logarithmically, and
it is an artefact of the interpolation rather than physics: a real pair of
spherical charge distributions interacts exactly as `q_A q_B / r` once their
overlap is gone.

So the electrostatics is split exactly:

```text
E_ES = Ewald[ point-charge model over the lattice ]
     + Σ_{|T| < r_off}  f(r) · [ W_KO(r) − W_point(r) ]
```

The point-charge model is the same Dewar–Thiel multipole configuration the
molecular kernel already uses, so the two halves match term by term. `f(r)` is
a `C²` quintic switch that hands the Klopman–Ohno form over to the point limit
between `r_on = 18` and `r_off = 22` Bohr, removing the spurious tail. Dipole
and higher configurations need no switch: their difference already decays as
`1/r⁴` and converges absolutely.

Exchange is treated differently and deliberately: it uses the **full** NDDO
integral inside a cutoff and gets no lattice sum. Subtracting the point-charge
model from the integral table would remove it from the exchange as well, while
the Ewald sum only puts the Coulomb part back — for H₂ that showed up as a
9.73 eV error, exactly half the point-charge pair energy.

### Dimensionality

The short-range half is dimension-independent. All the dimension dependence is
confined to the Ewald reciprocal sum:

| Dimension | Reciprocal treatment |
|---|---|
| 3D | Standard Ewald, tinfoil boundary, uniform neutralizing background |
| 2D | Exact Parry/Heyes slab sum (`erfcx` where the naive form overflows) |
| 1D | Cell-grouped direct sum with a Euler–Maclaurin tail |

Each is validated against a brute-force direct lattice sum in its own
dimension, which is the only sound way to check an Ewald implementation.

### Charged cells

For a periodic system `Molecule::charge` means the **net charge per unit cell**.
A charged cell's Ewald sum diverges, so a uniform neutralizing background is
added. Its energy is `−πQ²/(2α²V)`, and — the part that is easy to omit — its
potential `−πQ/(α²V)` is added to every multipole site so the SCF stays
consistent with the background rather than merely being corrected afterwards.

The background does not depend on atomic positions, so it contributes nothing to
the forces; it scales as `1/V`, so it does contribute
`πQ²/(2α²V²) δ_αβ` to the stress and always matters for the pressure.

Absolute energies of cells with different `Q` are not comparable — that is a
property of the convention, not of this implementation.

A charged **1D** system takes a uniform neutralizing **line charge** `−Q/L`
instead of a background. The three pieces — charges with charges, charges with
line, line with line — carry `+Q²H_N/L`, `−2Q²H_N/L` and `+Q²H_N/L` for a sum
truncated at `N` images, and cancel for any `N`.

What is subtracted is `H_N + ln(L/L₀)`, not `H_N`, and the difference matters.
The sum grows like the logarithm of the *extent* reached, `N·L`, not of the image
count alone, so subtracting `Q²H_N/L` leaves a remainder carrying a `ln L`. Each
cell then gets a finite, stable, entirely plausible energy while a chain and its
own doubled cell disagree — by 9.4 eV on a six-Bohr cell. Fixing `L₀` at one Bohr
rather than at the cell length is what makes the convention size-consistent, and
`tests` in `pbc::ewald` check exactly that, on bare point charges where nothing
but the electrostatics can differ.

The residual dependence on the image count is asserted to shrink **fourfold per
doubling** — the truncated dipole tail, which a charged cell cannot correct
analytically because a charged distribution's dipole depends on the origin. A
tolerance on the energy could be met by a logarithm that had not grown much yet;
a ratio of successive differences cannot.

### A neutral chain works; its two halves individually do not

This is worth spelling out, because it looks like a contradiction. The SCF splits
the electrostatics into a **core** half and an **electron** half, so that one can
enter `H_core` (linear in the density) and the other the Fock matrix (quadratic).
Each half carries the full nuclear charge with opposite sign, so *each is a
charged lattice sum on its own*, even for a perfectly neutral cell.

In 3D and 2D that is harmless — a neutralizing background makes each half finite
and the two backgrounds cancel in the total. In 1D there is none, and a charged
chain's potential grows logarithmically with transverse distance.

What saves it is that the divergence cancels **exactly** between the three
pieces, provided they share one truncation. Writing `H_N = Σ_{n≤N} 1/n` for the
divergent lattice sum and `Q` for the nuclear charge, the monopole part of each
piece is

```text
½Σ_cc → +Q²H_N/L      Σ_ce → −2Q²H_N/L      ½Σ_ee → +Q²H_N/L
```

which sums to zero for any `N`. So a chain is summed with charged halves allowed
and a common image count, and the total is finite and `N`-independent even though
no single term is.

Each half now also carries its own neutralizing line charge, which makes each
finite on its own. For a neutral cell that changes nothing — the line terms
cancel across the three pieces exactly as the divergences did — and it is what
lets a genuinely charged cell work at all, where the halves no longer balance and
the leftover *is* the physical background term.

One consequence remains: the dipole tail correction is skipped for a charged
half, because the dipole of a charged distribution is origin-dependent. What is
lost is the *total's* tail, of order `1e-6` eV at the default image count.

`tests/crystals.rs` checks a 1D polyacetylene chain against the same chain in a
3D cell with vacuum — two treatments that share none of this machinery — and they
agree to better than `1e-3` eV.

## Corrections under PBC

D3, H4, X, and the simple hydrogen-bond term are all classical, short-range, and
absolutely convergent, so they need a lattice sum but no Ewald. They are
evaluated once per cell, outside the SCF, and reported in
`PeriodicResult::correction_ev`.

Two things about the D3 lattice sum are worth knowing:

- The **coordination number** is computed over images too. Computing it from the
  bare cell contents gives a systematic error at any surface or interface.
- `CorrectionCutoffs::coordination` is a **model parameter, not a convergence
  parameter**. The D3 counting function saturates at `1/(1+e¹⁶) ≈ 1.1e-7` rather
  than decaying, so the coordination lattice sum grows with the number of
  enclosed atoms and never converges. The cutoff defines the model; it cannot be
  increased to convergence.
- An image's coordination number is **its own where the cluster is wide enough
  around it, and its parent's beyond**. The parent's is exact at `Γ`, where every
  copy moves alike, and only there: under the phased displacement a phonon at
  finite `q` applies, an image's neighbours move by different amounts than its
  parent's do. The honest region is set by a triangle inequality — an entry
  within `cluster radius − coordination cutoff` of the cell has its whole
  coordination ball inside the cluster — and the residual the copy leaves beyond
  it is a relative `4e-7` on the force constants, falling with the radius.

## Forces and stress

Forces and stress are analytic, from the same forward-mode `Dual` derivatives the
molecular gradient uses. Because NDDO assumes an orthogonal AO basis,
`S(T) = δ`, there is no Pulay term anywhere — the derivative is `Tr[P dH]` alone.

The stress comes out of the per-pair derivatives at no extra cost. A `Dual`
carries the derivative with respect to the pair displacement `d`, an image pair
has displacement `d + T`, and strain maps it to `(1 + ε)(d + T)`, so

```text
σ_αβ = (1/V) Σ_pairs (∂E/∂d_α) (d + T)_β
```

One correction is needed and is easy to miss: the multipole site offsets are
fixed lengths in the local frame and do **not** scale under strain, so the Ewald
virial has `Σ_sites g_site,α · offset_β` subtracted. Omitting it left the virial
10% out (35.70 against a finite-difference 39.25).

The stress is reported only where it is defined, and the two reduced
dimensionalities differ.

A **chain** has an axial stress. 1D is summed directly rather than through
reciprocal space, so its virial is the ordinary pair virial and comes free with
the gradients. Only the axial component exists — a chain has no cross-section to
squeeze — and every component touching a non-periodic direction is zeroed. That
projection is applied once, for every dimensionality, rather than being left to
each lattice sum: `∂E/∂ε` is a well-defined derivative in all nine components,
since deforming a chain transversally really does move its atoms apart, but a
chain has no transverse cell vector to relax.

A **slab** reports the two in-plane components and exact zeros elsewhere. The
Parry sum's strain derivative turns on one observation: the phase `G·d` is
strain-invariant, because `G` transforms as `(1 − εᵀ)G` while the in-plane part
of `d` transforms as `(1 + ε)d` and the two cancel exactly. What is left is the
`1/(A|G|)` prefactor and the kernel's own dependence on `|G|`,

```text
∂(1/A)/∂ε_βγ = −P_βγ / A        ∂|G|/∂ε_βγ = −G_β G_γ / |G|
```

with `P = 1 − n̂ ⊗ n̂` standing in for the identity, since only in-plane strains
are the slab's own. The `z` extent is padding, and
`the_slab_virial_ignores_the_vacuum_thickness` says the in-plane stress does not
depend on how much of it there is.

Only an **isolated** cell returns `None` now, and `relax` refuses a variable-cell
run there — not because the derivative is missing but because there is no strain
for it to be taken with respect to.

## The dynamical matrix at finite `q`

`pbc::dfpt::rigid_ion_dynamical_matrix` assembles

```text
D_{κα,κ′β}(q) = Σ_T e^{iq·T} Φ_{κα,κ′β}(0, T)
```

from the primitive cell at any wavevector, at a cost that does not depend on `q`.
For a pair term with `3×3` second derivative `H`, the two *diagonal* blocks take
`+H` with **no** phase and the two mixed blocks take `−H e^{±iq·T}`: the
`(κ, κ)` self-force-constant sums over every image unphased, because both its
indices sit in the reference cell. Reusing the phased lattice sum for the
diagonal blocks — the natural shortcut — puts a `q`-dependence into the self term
and breaks the acoustic sum rule by exactly that amount.

The Coulomb part needs `pbc::phased`, which sums `Σ_T e^{iq·T}/|d + T|`. Away
from `q = 0` that is the *easier* sum: `|G + q|` never vanishes, so there is no
divergent term, no background and no special case. The reduction test is
therefore not `q == 0` but "`q` is a reciprocal lattice vector" — every one of
them phases the sum by one, and a phonon commensurate with an `n×n×n` mesh sits
at a supercell reciprocal lattice vector.

The phased sum runs in every dimensionality, each the `q`-shifted form of its
unphased machinery. 2D shifts the Parry sum onto the full `G + q` set with
prefactor `π/(A|G+q|)` — the ±G folding and its factor of two are a `q = 0`
symmetry — and restores the smeared-sheet term only where `q` is a reciprocal
lattice vector, since the shifted set has no `G + q = 0` member anywhere else.
1D keeps the chain path's direct-summation character: at `q ≠ 0` the
oscillation itself converges the sum and the truncated tail is summed by
repeated Abel transformation (the naive tail is `O(1/N)` there, not the
`O(1/N³)` the unphased grouping enjoys), while the neutralizing line charge and
the dipole tail are `q = 0` artefacts gated on the reciprocal lattice exactly
as the 3D background is. `q` must lie in the periodic subspace — a slab has no
perpendicular wavevector — and is refused rather than projected otherwise.

The reciprocal-space phase is `e^{−i(G+q)·d}`, and the minus sign has a test of
its own: re-indexing the lattice sum gives the exact identity
`Φ_q(d + T₀) = e^{−iq·T₀} Φ_q(d)`, which a conjugated smooth half fails. The 3D
sum shipped with `e^{+i(G+q)·d}` for months — the oracle contraction is
identically real over its ±d pairs, the folding comparison sits at `q = b/2`
where the shifted set is symmetric under negation, and every internal identity
conjugates both halves together — until the 2D splitting-independence test
compared `Im Φ_q` across two values of α and watched it move in the fourth
digit.

Validated by folding: a doubled cell's Γ-point force constants hold the
primitive cell's `D(0)` and `D(zone boundary)` between them, compared as
eigenvalue *sets* so the comparison does not depend on atom ordering.

**The response runs on the microscopic kernel.** The `G = 0` member of the
phased reciprocal sum — the macroscopic field, `4π/(Ωq²)` — is excluded from the
coupled-perturbed solve and restored analytically by `pbc::lo_to`. That is the
standard decomposition, and here it is also a bug fix: keeping it inside a
self-consistent response makes the bare perturbation `O(1/q)` per atom and the
induced density `O(1/q)` in reply, so their contraction is `O(1/q²)` — and the
`Σ_a Q_a = 0` that saves the fixed-charge sum cannot save a product. Through
0.2.1 the acoustic sum rule of `D(q)` grew as `1/q²` in a polar cell (3.69 at
`q = 0.1`, 220 at `q = 0.0125`), putting water's lowest frequency at −4705 cm⁻¹
by `q = 0.01`. It now falls with `q`, as the rigid-ion half always did. No test
saw it because every wavevector they use is moderate or commensurate.

The **rigid-ion** matrix keeps the macroscopic member, and correctly: there is no
response there to amplify it, and charge neutrality gives the fixed-charge sum a
finite limit. That is also what makes the non-analytic term's prefactor
checkable — `D_rigid(q) − D_rigid(0)` approaches the closed form as `q → 0`,
which is a comparison against something other than the same formula written
twice.

**The electronic response is included.** A displacement moves the electrons too,
and that relaxation is most of the answer: on water it turns a rigid-ion force
constant of 5.96 eV/Bohr² into 0.56. It is solved as a self-consistent linear
response in the band basis of the coupled pair `(Γ, q)`, with both the
occupied–empty and empty–occupied blocks — the second is the response of the bra
at `k + q`, and dropping it halves the answer.
`rigid_ion_dynamical_matrix` gives the fixed-density part alone;
`phonon_frequencies` gives cm⁻¹ at any `q`, imaginary modes as negatives.

Every piece is checked against something built differently — the perturbation
against `kernel::fock_derivative_pairs` and against finite differences of the
Ewald sum, the two-electron kernel against `PeriodicKernel`, the second
derivatives against `ewald_atom_hessian` and `hessian::skeleton`, `D(0)` against
`periodic_hessian`, and `D(q)` by folding. The piecewise tests exist because the
folding test says *that* something is wrong and never *where*: it found a lattice
sum that skipped same-atom images, a phased kernel with no self term, and an
exchange handed a total density where it wanted a spin one.

The response samples whatever k-mesh it is handed
(`dynamical_matrix_on_mesh`), pairing each point with `q`; handed none it samples
`Γ` alone, which is the sampling a `Γ` ground state supports and carries the same
`gamma_margin` condition. The mesh it runs is the **unreduced** one even where the
SCF reduced it by time reversal — TR maps the coupled pair `(k, k+q)` onto a pair
at `−q`, so the irreducible sum with doubled weights is a different number
wherever `q ≠ −q`.

Open and closed shell, plain PM3 and every corrected variant, any periodic
dimensionality — a slab takes in-plane `q` and a chain axial `q`, checked and
refused otherwise. The open-shell response is one channel per spin, coupled
through the Coulomb kernel and validated against central differences of the
periodic UHF force; the corrections are the same forward-mode arithmetic with
each image's displacement scaled by its own `e^{iq·T}`, validated by folding a
hydrogen-bonded chain onto its own doubled cell. The Γ-point analytic Hessian
reaches 1D and 2D the same way, its Ewald block being the phased sum at `q = 0`.

## Response properties

Everything in this section is a contraction of the same first-order density the
dynamical matrix already solves for, so none of it costs a new kind of
calculation — only a different thing read off the answer.

All of them are Γ-point paths, and all of them are only as good as the ground
state underneath. **Check the `gamma_margin_bohr` they return.** Below zero, one
k-point cannot represent the cell and the SCF converges cleanly to a
well-defined wrong answer; the response inherits that with no other symptom.

### Born effective charges

```python
z = pm3_rs.born_charges(numbers, positions, cell)["born_charges"]  # [nat][3][3]
```

`Z*_a[α][β] = ∂(cell dipole)_α / ∂u_{aβ}`, in electrons. `sum_rule_residual`
comes back beside them: translating the crystal produces no dipole, so
`Σ_a Z*_a` vanishes for the exact response and the residual measures everything
the calculation approximated. It is reported rather than imposed — pass
`enforce=True`, or call `enforce_born_sum_rule`, to flatten it once you have
looked at it.

### Dielectric tensors

```python
d = pm3_rs.dielectric(numbers, positions, cell, include_ionic=True)
d["epsilon"]          # eps_inf, electronic, clamped ions
d["epsilon_static"]   # eps_0 = eps_inf + ionic
d["skipped_modes"]    # how much of the ionic sum is missing
```

`epsilon` is the clamped-ion response and is `None` for a chain or a slab, where
the `4π/Ω` wants a volume and the answer would otherwise be a statement about
the supercell's vacuum padding. `polarizability` is defined in every
dimensionality and is always returned.

`include_ionic=True` lets the nuclei relax along each infrared-active mode and
costs a Γ-point phonon run plus a set of Born charges. It is only meaningful at
a **relaxed geometry**: the sum weights each mode by `1/ω²`, so `skipped_modes`
counting more than the three acoustic ones means the structure is not a minimum
and `ε₀` is missing whatever those modes carried.

### LO–TO splitting

```python
pm3_rs.phonons(numbers, positions, cell, q=[0, 0, 0], lo_to_direction=[1, 0, 0])
```

Adds `(4π/Ω)(q̂·Z*)(q̂·Z*)/(q̂·ε∞·q̂)` to the dynamical matrix, which is what
lifts the longitudinal branch above the transverse one in a polar crystal. 3D
only, and it needs a direction because the limit is direction-dependent —
`q = 0` exactly has no direction, so it is refused without one.

The coefficient here was not copied from anywhere. This crate has the full
`D(q)` at arbitrary `q`, so the closed form can be checked against the `q → 0`
limit of the rigid-ion matrix, which keeps the macroscopic term: the ratio goes
0.56 → 0.89 → 0.973 as `q²`, fixing both the `4π/Ω` and the Hartree conversion.

### Supercell force constants

```bash
pm3_rs_cli phonon-bands crystal.xyz --cell 6.0 --supercell 2,2,2
```

One Hessian of the supercell gives `Φ(T)`, and Fourier interpolation gives the
frequencies at any `q` — a whole dispersion for the price of one calculation,
where direct DFPT pays per wavevector.

A supercell's Γ point *is* a mesh of the primitive cell, so what this reproduces
is the **mesh**-sampled response with the matching mesh, not a Γ-only one. That
was measured rather than assumed: at a cell where the Γ margin is positive the
two agree to 0.011 cm⁻¹, while the Γ-only comparison does not.

### Berry-phase polarization

```rust
let p = pm3_rs::berry_polarization(&molecule, &params, &options, &periodic, &kopts, 12)?;
let dp = before.difference(&after);   // never `after.total - before.total`
```

Present as an independent check rather than as a feature. It reaches the same
`Z*` through a product of overlaps between neighbouring k points, with no
response equation anywhere in it, so a sign or a spin factor that is invisible
inside the CPHF route shows up across the pair.

Polarization is defined only **modulo** `e a_α/Ω`, which is physics rather than
a defect: a different branch assigns the electrons to a different unit cell.
Always take differences through `difference`, which reduces onto the nearest
branch — a plain subtraction is off by exactly one quantum whenever the two
landed differently, which for a finite displacement is common.

The two routes differ by the intra-atomic `s`–`p` moment the phase omits, since
it places every orbital at its atom's centre. Measured: `0.147 e` on HF, falling
to `1.9e-4` when that term is removed from the CPHF side.

### A finite field along a periodic direction

```python
d = pm3_rs.finite_field(numbers, positions, cell, field=[1e-3, 0, 0], kpts=[6, 1, 1])
d["polarization"], d["enthalpy_ev"], d["resolved"]
```

`𝓔·R` shifts by `𝓔·T` under a lattice translation, so it is periodic only when
`𝓔·T = 0` for every lattice vector. Along a periodic direction the potential is
unbounded and the ground state of `H − 𝓔·R` does not exist. A field orthogonal
to *every* lattice vector needs none of this and goes through `PM3(field=...)`.

What replaces it is the Nunes–Gonze electric enthalpy `F = E − Ω 𝓔·P`, with `P`
the Berry-phase polarization. Because `P` is built from overlaps between
neighbouring k-points, its derivative couples them: the k-points can no longer
be solved one at a time, and the SCF carries an extra term per k-point held
fixed while the inner loop converges.

`kpts` along each field direction is that direction's string length and is the
convergence parameter. At least 3 — two points cannot resolve a winding.
`resolved` says which axes the mesh could see: an unresolved axis contributes
zero to `electronic_polarization`, which is not the same as its contribution
being zero. Restricted, gapped, 3D only, and **there is no force** — the
derivative of the enthalpy with respect to the nuclei is not implemented.

It reproduces the CPHF polarizability to 1 part in 10⁴ on hydrogen, where both
routes carry the same position operator. On a cell with `dd` they legitimately
differ, by the term the Berry phase cannot hold.

## Where this stops

One limit found by running real crystals rather than molecules in boxes, recorded
so it is known rather than discovered:

- **An even k-mesh can fail on a cubic cell.** Rocksalt NaCl at its experimental 5.64 Å runs out
  of SCF iterations under a `2×2×2` mesh, with or without smearing, with damping to 0.95, with
  level shifts to 5 eV, and at a thousand iterations. Two eV of smearing does converge — to a
  *different* solution 200 eV away.

  The chemical potential is what gives it away: it swings by an electronvolt every iteration as
  occupations flip. An even Γ-centred mesh on a cubic cell puts every one of its points on a
  zone-boundary symmetry point, where bands meet; a Fermi level inside a degenerate manifold has
  no stable filling to find, and the bisection reassigns it each pass.

  Odd meshes do not sit on those points. `3×3×3` converges in eighteen iterations, and the answer
  is right: energies agree to 0.02 eV and charges to 0.02 electrons across `3×3×3`, `4×4×4` and
  `5×5×5`. **If a k-point SCF will not settle on a symmetric cell, try an odd mesh before
  anything else.** `tests/crystals.rs` pins both the failure and the remedy.

  Worth knowing separately: PM3 puts ±0.87 electrons on the ions at 8 Å and ±1.00 at 22.56 Å, but
  only ±0.16 compressed to 5.64 Å, while binding the cell more strongly there. That is the
  method's statement about dense NaCl, not the sampling's.

## Physical validation

Beyond the internal-consistency checks below, four real crystals, each chosen for
the one feature a molecule in a box cannot exercise:

| system | what it establishes |
|---|---|
| NaCl rocksalt | binding carried by the lattice sum, with ionic charges |
| graphene, 2D | the exact Parry sum agrees with 3D-with-vacuum to `< 1e-3` eV |
| graphene, Γ | the margin is negative and stays negative at any cell size |
| polyacetylene | Peierls dimerization lowers the energy and opens a gap |

The graphene comparison is the sharpest of these: Parry's slab sum with an
analytic `z` dependence and the textbook 3D sum with a neutralizing background
are genuinely different derivations, and agreement between them is a check of the
2D branch against something other than itself.

## Validation

There is no external oracle for periodic PM3, so correctness rests on internal
consistency. The checks that carry the most weight:

- **Supercell folding.** One cell of edge `L` and a `2×2×2` supercell of edge
  `2L` describe the same crystal, so the energies must differ by exactly a
  factor of eight. This is sensitive to every counting convention — the
  ordered-visit halving, the unique half in the core–core term, the self-image
  block, the Ewald owner exclusion — none of which the isolated-molecule limit
  probes, since there the miscounted terms are all zero anyway. Agreement is
  `5.7e-11` eV per cell.
- **k↔supercell folding.** An `n×n×n` mesh on one cell against the Γ point of an
  `n×n×n` supercell, on a cell narrow enough that Γ alone would be badly wrong.
  Essentially nothing survives this by accident: it pins the Bloch phase
  convention and its sign, the direction of the density back-transform, the
  weights, the time-reversal reduction, and the global Fermi level. Checked
  neutral and at `Q = +1`, where it additionally pins the neutralizing
  background's `1/V` scaling against the cell's charge.
- **Γ against Γ.** The k-point path at `k = 0` reproduces the dedicated
  Γ-point path — two independent implementations agreeing to `1e-8` eV, one of
  which is validated against MOPAC in the isolated limit. This is what caught a
  factor-of-two in the restricted occupations that every k-point-only test
  passed, being self-consistently wrong on both sides of the comparison.
- **Time-reversal reduction** against the unreduced mesh, to `1e-8` eV.
- **Madelung constant.** Rocksalt reproduces 1.747565 to `1e-10`.
- **Ewald against direct summation**, independently in 1D, 2D, and 3D.
- **Isolated limit.** A molecule in a large cell reproduces the molecular
  energy: H₂ to `2.3e-9` eV at every cell size, and water converging as `1/L³`
  (`−1.02e-3 → −1.27e-4 → −1.58e-5` eV as the edge doubles), with the exponent
  asserted rather than eyeballed.
- **Analytic against finite difference** for forces, for all nine virial
  components under strain, and with corrections enabled as well as without.
- **Translation invariance** and forces summing to zero.
- **Periodic UHF**: a closed shell forced through the unrestricted path returns
  the RHF energy with exactly zero spin density; a methyl radical's spin density
  integrates to 1.
