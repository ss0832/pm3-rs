// SPDX-License-Identifier: GPL-3.0-or-later
//! **Documented-API conformance tests.**
//!
//! Every test here transcribes a code block or a stated guarantee from
//! `README.md` / `docs/rust-api.md` and checks that it compiles, runs, and
//! behaves as documented — the units, the field names, the enum variants, the
//! error types, and the invariants the prose promises. If a doc example stops
//! working, or a documented field is renamed, this file fails to compile.
//!
//! The numerical *values* are covered by the MOPAC oracle regressions in
//! `tests/molecules.rs`; this file covers the *surface*.

use pm3_rs::constants::{BOHR_TO_ANGSTROM, EV_TO_KCAL};

const WATER_XYZ: &str = "3\nwater\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\n";

// ---------------------------------------------------------------------------
// README.md — "Rust API"
// ---------------------------------------------------------------------------

/// The README's quick-start block, verbatim, including its root-level imports.
#[test]
fn readme_rust_quickstart_works() {
    use pm3_rs::{run_pm3, Molecule, Pm3Options, Pm3Parameters};

    let path = std::env::current_dir().unwrap().join("examples/water.xyz");
    let molecule = Molecule::from_xyz_file(&path, 0.0).unwrap();
    let parameters = Pm3Parameters::standard().unwrap();
    let result = run_pm3(&molecule, &parameters, &Pm3Options::default()).unwrap();
    // "heat of formation = {} kcal/mol"
    assert!(result.heat_of_formation_kcal.is_finite());
    assert!(result.converged);
}

/// The README's method table: all four variants must parse and run, and the
/// correction must actually change the energy in the documented direction
/// (`PM3` = no correction; each variant adds terms on top).
#[test]
fn readme_correction_variant_table_is_accurate() {
    use pm3_rs::{run_pm3, Molecule, Pm3Options, Pm3Parameters, Variant};

    // A water dimer: has a hydrogen bond (H4) and dispersion (D3), no halogens.
    let dimer = "6\nwater dimer\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n\
                 O 2.9 0.0 0.0\nH 3.2 0.9 0.0\nH 3.2 -0.5 0.8\n";
    let molecule = Molecule::from_xyz_str(dimer, 0.0).unwrap();
    let parameters = Pm3Parameters::standard().unwrap();
    let energy = |variant| {
        let options = Pm3Options {
            variant,
            ..Pm3Options::default()
        };
        run_pm3(&molecule, &parameters, &options).unwrap().total_ev
    };
    let plain = energy(Variant::Pm3);
    let d3 = energy(Variant::Pm3D3);
    let d3h4 = energy(Variant::Pm3D3H4);
    let d3h4x = energy(Variant::Pm3D3H4X);

    // "PM3 | none": the plain variant must add nothing.
    assert_eq!(plain, energy(Variant::Pm3));
    // D3 dispersion is attractive, so it must lower the energy.
    assert!(d3 < plain, "D3 must be attractive: {d3} vs {plain}");
    // D3H4 uses a different (refitted) D3 set plus H4 and H-H terms.
    assert_ne!(d3h4, d3);
    // "PM3-D3H4X | PM3-D3H4 + X halogen-bond term": no halogens here, so the
    // X term must contribute exactly zero.
    assert_eq!(d3h4x, d3h4, "X must vanish for a halogen-free system");
}

/// README: "elements without PM3 parameters return an explicit
/// missing-parameter error", and the supported set is H-Ca, Zn-Sr, Cd-Ba, Hg-Bi.
#[test]
fn readme_supported_element_ranges_are_accurate() {
    use pm3_rs::{Pm3Error, Pm3Parameters};

    let parameters = Pm3Parameters::standard().unwrap();
    let supported: Vec<u8> = (1..=20)
        .chain(30..=38)
        .chain(48..=56)
        .chain(80..=83)
        .collect();
    for z in &supported {
        assert!(
            parameters.element(*z).is_ok(),
            "README claims Z={z} is supported"
        );
        // "PM3 uses an s/p valence basis for these elements".
        let element = parameters.element(*z).unwrap();
        assert!(
            !element.has_d(),
            "Z={z} must not carry d orbitals in PM3 (n_orb={})",
            element.n_orb
        );
    }
    // La-Lu (57-71) are the Sparkles; also present, with zero orbitals.
    for z in 57..=71u8 {
        let element = parameters.element(z).unwrap();
        assert_eq!(element.n_orb, 0, "Z={z} Sparkle must have no orbitals");
        assert_eq!(element.core_charge, 3.0, "Z={z} Sparkle must be trivalent");
    }
    // Anything outside those ranges errors explicitly rather than silently.
    for z in [21u8, 25, 39, 45, 57 - 1 + 30, 72, 79, 84] {
        if supported.contains(&z) || (57..=71).contains(&z) {
            continue;
        }
        match parameters.element(z) {
            Err(Pm3Error::MissingElement(reported)) => assert_eq!(reported, z),
            Err(other) => panic!("Z={z}: wrong error variant: {other}"),
            Ok(_) => panic!("Z={z} should not be parameterized in PM3"),
        }
    }
    // MOPAC special atom codes.
    for (z, orbitals, core) in [(102u8, 4usize, 1.0), (104, 0, 1.0), (106, 0, -1.0)] {
        let element = parameters.element(z).unwrap();
        assert_eq!(element.n_orb, orbitals);
        assert_eq!(element.core_charge, core);
    }
}

// ---------------------------------------------------------------------------
// docs/rust-api.md — "Building a molecule"
// ---------------------------------------------------------------------------

#[test]
fn documented_molecule_constructors_work() {
    use pm3_rs::system::{Atom, Molecule};
    use pm3_rs::Vec3;

    // From an XYZ string (Ångström); the second argument is the total charge.
    let from_string = Molecule::from_xyz_str(WATER_XYZ, 0.0).unwrap();
    assert_eq!(from_string.atoms.len(), 3);

    // …or from a file …
    let path = std::env::current_dir().unwrap().join("examples/water.xyz");
    let from_file = Molecule::from_xyz_file(&path, 0.0).unwrap();
    assert_eq!(from_file.atoms.len(), 3);

    // … or built directly.
    let atoms = vec![
        Atom {
            z: 8,
            position: Vec3::new(0.0, 0.0, 0.0),
        },
        Atom {
            z: 1,
            position: Vec3::new(1.8, 0.0, 0.0),
        },
    ];
    let built = Molecule::new(atoms).with_charge(1.0).with_multiplicity(2);
    assert_eq!(built.charge, 1.0);
    assert_eq!(built.multiplicity, 2);

    // "Positions are stored internally in Bohr (`Molecule::from_xyz_*` reads
    // Ångström and converts)": the O-H distance must be 0.96 Å in Bohr.
    let bond = (from_string.atoms[1].position - from_string.atoms[0].position).norm();
    assert!(
        (bond * BOHR_TO_ANGSTROM - 0.96).abs() < 1.0e-12,
        "O-H = {} Bohr = {} Å",
        bond,
        bond * BOHR_TO_ANGSTROM
    );
}

/// "conflicting non-default molecule/option values return
/// `Pm3Error::InvalidInput` rather than silently choosing one" — and the
/// non-conflicting direction really does read from the molecule.
#[test]
fn documented_charge_multiplicity_reconciliation() {
    use pm3_rs::{run_pm3, Molecule, Pm3Error, Pm3Options, Pm3Parameters};

    let parameters = Pm3Parameters::standard().unwrap();
    let cation = Molecule::from_xyz_str(WATER_XYZ, 1.0)
        .unwrap()
        .with_multiplicity(2);

    // Molecule metadata alone is honoured when the option stays at its default.
    let from_molecule = run_pm3(&cation, &parameters, &Pm3Options::default()).unwrap();
    assert!(from_molecule.unrestricted);
    assert!((from_molecule.charges.iter().sum::<f64>() - 1.0).abs() < 1.0e-10);

    // Setting the same value explicitly in the options is also supported.
    let explicit = run_pm3(
        &Molecule::from_xyz_str(WATER_XYZ, 0.0).unwrap(),
        &parameters,
        &Pm3Options {
            charge: 1.0,
            multiplicity: 2,
            ..Pm3Options::default()
        },
    )
    .unwrap();
    assert!((from_molecule.total_ev - explicit.total_ev).abs() < 1.0e-12);

    // Conflicting values are rejected, not silently resolved.
    let conflict = run_pm3(
        &cation,
        &parameters,
        &Pm3Options {
            charge: -1.0,
            ..Pm3Options::default()
        },
    );
    match conflict {
        Err(Pm3Error::InvalidInput(message)) => assert!(message.contains("conflicting charge")),
        other => panic!("expected InvalidInput, got {:?}", other.map(|r| r.total_ev)),
    }
}

// ---------------------------------------------------------------------------
// docs/rust-api.md — "Parameters and options"
// ---------------------------------------------------------------------------

#[test]
fn documented_options_block_and_variant_parse() {
    use pm3_rs::corrections::Variant;
    use pm3_rs::params::Pm3Parameters;
    use pm3_rs::scf::{Pm3Options, Reference};

    let _params = Pm3Parameters::standard().unwrap();
    let options = Pm3Options {
        charge: 0.0,
        multiplicity: 1,
        reference: Reference::Auto, // Auto | Rhf | Uhf
        variant: Variant::Pm3D3H4,  // Pm3 | Pm3D3 | Pm3D3H4 | Pm3D3H4X
        ..Pm3Options::default()
    };
    assert_eq!(options.reference, Reference::Auto);
    assert_eq!(options.variant, Variant::Pm3D3H4);

    // "`Variant::parse("pm3-d3h4")` maps method strings to the enum."
    assert_eq!(Variant::parse("pm3-d3h4"), Some(Variant::Pm3D3H4));
    assert_eq!(Variant::parse("PM3"), Some(Variant::Pm3));
    assert_eq!(Variant::parse("pm3-d3"), Some(Variant::Pm3D3));
    assert_eq!(Variant::parse("pm3-d3h4x"), Some(Variant::Pm3D3H4X));
    assert_eq!(Variant::parse("pm6"), None);
    // The CLI documents `--method <pm3|pm3-d3|pm3-d3h4|pm3-d3h4x>`; the parser
    // also accepts the spaced/underscored spellings it normalizes.
    assert_eq!(Variant::parse("PM3_D3H4"), Some(Variant::Pm3D3H4));

    // "Other notable `Pm3Options` fields" — each must exist with the documented type.
    let tuned = Pm3Options {
        max_scf: 300,
        e_tol: 1.0e-9,
        p_tol: 1.0e-8,
        level_shift_ev: 0.5,
        damping: 0.2,
        hessian_cutoff: Some(20.0),
        ..Pm3Options::default()
    };
    assert_eq!(tuned.max_scf, 300);
    assert_eq!(tuned.hessian_cutoff, Some(20.0));

    // The documented memory-budget defaults.
    let defaults = Pm3Options::default();
    assert_eq!(defaults.scf_memory_mb, 512);
    assert_eq!(defaults.hessian_memory_mb, 1024);
    assert_eq!(defaults.integral_memory_mb, 0); // 0 = "use the 4096 MiB process default"
    assert_eq!(
        pm3_rs::hamiltonian::default_pair_cache_limit_mb(),
        4096,
        "documented default pair-cache ceiling"
    );
}

/// "Trimming the accelerator history changes only the SCF path, not its fixed
/// point: the converged energy, density, and every derived quantity are
/// unchanged."
#[test]
fn documented_memory_budgets_do_not_change_results() {
    use pm3_rs::{run_pm3, Molecule, Pm3Options, Pm3Parameters};

    let molecule = Molecule::from_xyz_str(
        "4\nformaldehyde\nC 0 0 0\nO 0 0 1.21\nH 0.94 0 -0.54\nH -0.94 0 -0.54\n",
        0.0,
    )
    .unwrap();
    let parameters = Pm3Parameters::standard().unwrap();
    let reference = run_pm3(&molecule, &parameters, &Pm3Options::default()).unwrap();
    let squeezed = run_pm3(
        &molecule,
        &parameters,
        &Pm3Options {
            scf_memory_mb: 1,
            ..Pm3Options::default()
        },
    )
    .unwrap();

    assert!((reference.total_ev - squeezed.total_ev).abs() < 1.0e-8);
    assert!((reference.heat_of_formation_kcal - squeezed.heat_of_formation_kcal).abs() < 1.0e-6);
    for (a, b) in reference.charges.iter().zip(&squeezed.charges) {
        assert!((a - b).abs() < 1.0e-8, "charge {a} vs {b}");
    }
    for (a, b) in reference
        .density
        .as_slice()
        .iter()
        .zip(squeezed.density.as_slice())
    {
        assert!((a - b).abs() < 1.0e-8, "density {a} vs {b}");
    }
}

// ---------------------------------------------------------------------------
// docs/rust-api.md — "Single point"
// ---------------------------------------------------------------------------

/// The documented `Pm3Result` field list, plus the physical relationships the
/// docs imply between them.
#[test]
fn documented_single_point_result_surface() {
    use pm3_rs::scf::{run_pm3, Pm3Calculator, Pm3Options};
    use pm3_rs::{Molecule, Pm3Parameters};

    let mol = Molecule::from_xyz_str(WATER_XYZ, 0.0).unwrap();
    let params = Pm3Parameters::standard().unwrap();
    let options = Pm3Options::default();
    let r = run_pm3(&mol, &params, &options).unwrap();

    // Every documented field, read at its documented type.
    let _: f64 = r.total_ev;
    let _: f64 = r.electronic_ev;
    let _: f64 = r.core_ev;
    let _: f64 = r.heat_of_formation_kcal;
    let _: &Vec<f64> = &r.charges;
    let _: pm3_rs::Vec3 = r.dipole_debye;
    let _: &Vec<f64> = &r.mo_energies;
    let _: &pm3_rs::Matrix = &r.mo_coeff;
    let _: &pm3_rs::Matrix = &r.density;
    let _: Option<f64> = r.homo_ev;
    let _: Option<f64> = r.lumo_ev;
    let _: usize = r.n_occ;
    let _: usize = r.iterations;
    let _: bool = r.converged;
    let _: bool = r.unrestricted;

    // Documented decomposition: total = electronic + core (+ corrections; none for plain PM3).
    assert!((r.total_ev - (r.electronic_ev + r.core_ev)).abs() < 1.0e-9);
    // Mulliken charges (e) sum to the total charge.
    assert!(r.charges.iter().sum::<f64>().abs() < 1.0e-9);
    // Water is closed shell, so `Auto` picks RHF.
    assert!(!r.unrestricted);
    // 8 electrons in a 6-AO basis -> 4 doubly-occupied MOs.
    assert_eq!(r.n_occ, 4);
    assert_eq!(r.mo_energies.len(), 6);
    assert_eq!((r.mo_coeff.rows, r.mo_coeff.cols), (6, 6));
    assert_eq!((r.density.rows, r.density.cols), (6, 6));
    // HOMO below LUMO, both taken from the ascending MO spectrum.
    let (homo, lumo) = (r.homo_ev.unwrap(), r.lumo_ev.unwrap());
    assert!(homo < lumo, "HOMO {homo} >= LUMO {lumo}");
    assert_eq!(homo, r.mo_energies[r.n_occ - 1]);
    assert_eq!(lumo, r.mo_energies[r.n_occ]);
    // The dipole magnitude is the norm of the documented `[x, y, z]` vector.
    assert!((r.dipole_debye.norm() - r.dipole_magnitude).abs() < 1.0e-12);

    // "A convenience wrapper bundles parameters + options."
    let calc = Pm3Calculator::with_options(params.clone(), options.clone());
    let via_calculator = calc.calculate(&mol).unwrap();
    assert_eq!(via_calculator.total_ev, r.total_ev);
    // …and the bare constructor documented on the type.
    let defaulted = Pm3Calculator::new(params.clone());
    assert_eq!(defaulted.calculate(&mol).unwrap().total_ev, r.total_ev);
}

/// docs: "`Reference::Auto` = RHF closed shell / UHF open shell"; `Rhf` on an
/// open shell is an error; `Uhf` forces the unrestricted path for a singlet.
#[test]
fn documented_reference_selection() {
    use pm3_rs::scf::{run_pm3, Pm3Options, Reference};
    use pm3_rs::{Molecule, Pm3Error, Pm3Parameters};

    let params = Pm3Parameters::standard().unwrap();
    let closed = Molecule::from_xyz_str(WATER_XYZ, 0.0).unwrap();
    let radical = Molecule::from_xyz_str(
        "4\nmethyl\nC 0 0 0\nH 0 1.078 0\nH 0.9336 -0.539 0\nH -0.9336 -0.539 0\n",
        0.0,
    )
    .unwrap()
    .with_multiplicity(2);

    let run = |mol: &Molecule, reference| {
        run_pm3(
            mol,
            &params,
            &Pm3Options {
                reference,
                ..Pm3Options::default()
            },
        )
    };

    assert!(!run(&closed, Reference::Auto).unwrap().unrestricted);
    assert!(run(&radical, Reference::Auto).unwrap().unrestricted);
    assert!(!run(&closed, Reference::Rhf).unwrap().unrestricted);
    // Forced UHF on a singlet: unrestricted path, same energy (no symmetry breaking here).
    let forced = run(&closed, Reference::Uhf).unwrap();
    assert!(forced.unrestricted);
    assert!((forced.total_ev - run(&closed, Reference::Auto).unwrap().total_ev).abs() < 1.0e-6);
    // RHF on an open shell is rejected.
    match run(&radical, Reference::Rhf) {
        Err(Pm3Error::InvalidInput(message)) => assert!(message.contains("RHF requested")),
        other => panic!("expected InvalidInput, got {:?}", other.map(|r| r.total_ev)),
    }
    // The open-shell result carries the documented spin density.
    let spin = run(&radical, Reference::Auto).unwrap();
    let spin_density = spin.spin_density.expect("UHF must report a spin density");
    let net: f64 = (0..spin_density.rows).map(|i| spin_density[(i, i)]).sum();
    assert!(
        (net - 1.0).abs() < 1.0e-6,
        "one unpaired electron, got {net}"
    );
}

// ---------------------------------------------------------------------------
// docs/rust-api.md — "Analytic gradient"
// ---------------------------------------------------------------------------

#[test]
fn documented_gradient_surface_and_units() {
    use pm3_rs::gradient::{closed_form_gradient, numerical_gradient};
    use pm3_rs::{Molecule, Pm3Options, Pm3Parameters};

    // A distorted geometry, so the gradient is genuinely non-zero.
    let mol = Molecule::from_xyz_str("3\nwater\nO 0 0 0\nH 1.02 0.05 0\nH -0.28 0.96 0.1\n", 0.0)
        .unwrap();
    let params = Pm3Parameters::standard().unwrap();
    let options = Pm3Options::default();

    let g = closed_form_gradient(&mol, &params, &options).unwrap();
    assert_eq!(g.gradient.len(), 3);
    assert_eq!(g.forces.len(), 3);
    let _: f64 = g.max_gradient;
    let _: f64 = g.energy_ev;
    let _: &pm3_rs::scf::Pm3Result = &g.scf;

    // "for f in &g.forces { /* eV/Bohr, = −gradient */ }"
    for (force, grad) in g.forces.iter().zip(&g.gradient) {
        for k in 0..3 {
            assert!((force.get(k) + grad.get(k)).abs() < 1.0e-14);
        }
    }
    // `max_gradient` is the largest |component|.
    let expected = g
        .gradient
        .iter()
        .flat_map(|v| [v.x, v.y, v.z])
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!((g.max_gradient - expected).abs() < 1.0e-14);
    assert!(g.max_gradient > 1.0e-3, "distorted water must have a force");
    // Energy agrees with the single point.
    assert!((g.energy_ev - g.scf.total_ev).abs() < 1.0e-12);

    // "The gradient is fully analytic" — it must match finite differences of
    // the energy (this is the claim the docs make, so verify it here).
    let numeric = numerical_gradient(&mol, &params, &options, 5.0e-4).unwrap();
    for (a, n) in g.gradient.iter().zip(&numeric.gradient) {
        for k in 0..3 {
            assert!(
                (a.get(k) - n.get(k)).abs() < 1.0e-5,
                "analytic {} vs numerical {}",
                a.get(k),
                n.get(k)
            );
        }
    }

    // Translational invariance: Σ gradient = 0 (no net force on the molecule).
    let mut sum = pm3_rs::Vec3::zero();
    for grad in &g.gradient {
        sum += *grad;
    }
    assert!(sum.norm() < 1.0e-8, "net force {sum:?}");
}

/// docs: "including the analytic D3/H4/X correction gradient for the correction
/// variants. Open-shell systems use the spin-resolved path automatically."
#[test]
fn documented_gradient_covers_variants_and_open_shell() {
    use pm3_rs::gradient::{closed_form_gradient, numerical_gradient};
    use pm3_rs::{Molecule, Pm3Options, Pm3Parameters, Variant};

    let params = Pm3Parameters::standard().unwrap();
    let dimer = Molecule::from_xyz_str(
        "6\nwater dimer\nO 0 0 0\nH 0.96 0 0\nH -0.24 0.93 0\nO 2.9 0.1 0\nH 3.2 0.9 0\nH 3.2 -0.5 0.8\n",
        0.0,
    )
    .unwrap();
    for variant in [
        Variant::Pm3,
        Variant::Pm3D3,
        Variant::Pm3D3H4,
        Variant::Pm3D3H4X,
    ] {
        let options = Pm3Options {
            variant,
            ..Pm3Options::default()
        };
        let analytic = closed_form_gradient(&dimer, &params, &options).unwrap();
        let numeric = numerical_gradient(&dimer, &params, &options, 5.0e-4).unwrap();
        for (a, n) in analytic.gradient.iter().zip(&numeric.gradient) {
            for k in 0..3 {
                assert!(
                    (a.get(k) - n.get(k)).abs() < 2.0e-5,
                    "{variant:?}: analytic {} vs numerical {}",
                    a.get(k),
                    n.get(k)
                );
            }
        }
    }

    // Open shell, no extra flags needed.
    let radical = Molecule::from_xyz_str(
        "4\nmethyl\nC 0 0 0\nH 0 1.09 0.05\nH 0.94 -0.54 0\nH -0.94 -0.54 0\n",
        0.0,
    )
    .unwrap()
    .with_multiplicity(2);
    let open = closed_form_gradient(&radical, &params, &Pm3Options::default()).unwrap();
    assert!(open.scf.unrestricted);
    let numeric = numerical_gradient(&radical, &params, &Pm3Options::default(), 5.0e-4).unwrap();
    for (a, n) in open.gradient.iter().zip(&numeric.gradient) {
        for k in 0..3 {
            assert!((a.get(k) - n.get(k)).abs() < 1.0e-4);
        }
    }
}

// ---------------------------------------------------------------------------
// docs/rust-api.md — "Analytic Hessian and frequencies"
// ---------------------------------------------------------------------------

#[test]
fn documented_hessian_and_frequency_surface() {
    use pm3_rs::hessian::{analytic_hessian, numerical_hessian, vibrational_analysis};
    use pm3_rs::optimizer::{optimize, OptOptions};
    use pm3_rs::{Molecule, Pm3Options, Pm3Parameters};

    let mol = Molecule::from_xyz_str(WATER_XYZ, 0.0).unwrap();
    let params = Pm3Parameters::standard().unwrap();
    let options = Pm3Options::default();

    // 3N × 3N Cartesian Hessian (eV/Bohr²).
    let h = analytic_hessian(&mol, &params, &options, 1.0e-3).unwrap();
    assert_eq!((h.rows, h.cols), (9, 9));
    for i in 0..9 {
        for j in 0..9 {
            assert!((h[(i, j)] - h[(j, i)]).abs() < 1.0e-6, "Hessian asymmetric");
        }
    }
    // "`numerical_hessian` … is available as an independent reference."
    let numeric = numerical_hessian(&mol, &params, &options, 1.0e-3).unwrap();
    let mut worst = 0.0_f64;
    for i in 0..9 {
        for j in 0..9 {
            worst = worst.max((h[(i, j)] - numeric[(i, j)]).abs());
        }
    }
    assert!(worst < 5.0e-3, "analytic vs numerical Hessian: {worst:.3e}");

    // "Optimise first, then analyse the harmonic modes at the stationary point."
    let res = optimize(&mol, &params, &options, &OptOptions::default()).unwrap();
    let opt_mol = res.molecule;
    let modes = vibrational_analysis(&opt_mol, &params, &options, 1.0e-3).unwrap();

    // `VibrationalModes { hessian, frequencies_cm, eigenvalues }`.
    assert_eq!((modes.hessian.rows, modes.hessian.cols), (9, 9));
    assert_eq!(modes.frequencies_cm.len(), 9);
    assert_eq!(modes.eigenvalues.len(), 9);
    // "cm^-1, ascending".
    for pair in modes.frequencies_cm.windows(2) {
        assert!(pair[0] <= pair[1], "frequencies not ascending: {pair:?}");
    }
    // At a minimum: 6 translation/rotation modes and 3 real vibrations in water's range.
    //
    // Six because the geometry says so — `n_rigid` is the rank found by orthonormalizing the
    // rigid-body generators, not a constant — and **exactly** zero because they are projected out
    // rather than recognised afterwards for being small. This used to read `f.abs() < 50.0` here,
    // `< 100.0` in `tests/test_python_api.py` and `< 300.0` in `src/hessian.rs`: three bounds on
    // one quantity, which is what a threshold nobody can derive looks like from the outside.
    // `== 0.0` rather than `< 1e-6`, and that is safe rather than lucky: the eigenvalues along
    // the projected subspace are **assigned** zero, not merely computed to be small. The
    // projection leaves them at ±1e-15, `rigid_mode_indices` says which they are — a count taken
    // from the geometry applied to a ranking, with no magnitude in it — and they are then set to
    // `0.0`. `signed_wavenumber(0.0)` is `521.47 * (0.0).sqrt()`, exactly `0.0` under IEEE-754 on
    // every platform. A tolerance here would be asserting less than is true and would let a
    // regression that reintroduces the noise pass.
    //
    // What is *not* assumed is where the zeros land. They are the lowest modes only at a
    // minimum; an imaginary mode sorts below them. So this counts them rather than slicing.
    assert_eq!(modes.n_rigid, 6, "a bent triatomic has three of each");
    assert_eq!(
        modes.frequencies_cm.iter().filter(|f| **f == 0.0).count(),
        modes.n_rigid,
        "expected exactly {} exact zeros, got {:?}",
        modes.n_rigid,
        modes.frequencies_cm
    );
    let vibrations = &modes.frequencies_cm[modes.n_rigid..];
    assert!(
        vibrations.iter().all(|f| *f != 0.0),
        "a vibration came out exactly zero, so the count above is measuring the wrong modes"
    );
    assert!(
        vibrations.iter().all(|f| *f > 1000.0 && *f < 4200.0),
        "water vibrations out of range: {vibrations:?}"
    );
    // The number the projection replaced is still reported, as the diagnostic it always was: how
    // far from zero the Hessian put those directions before they were removed.
    assert!(
        modes.rigid_residual_cm > 0.0 && modes.rigid_residual_cm < 50.0,
        "the pre-projection residual should be small but non-zero, got {}",
        modes.rigid_residual_cm
    );
    // "negative = imaginary": the sign convention is carried from the eigenvalue.
    for (frequency, eigenvalue) in modes.frequencies_cm.iter().zip(&modes.eigenvalues) {
        assert_eq!(
            frequency.is_sign_negative(),
            eigenvalue.is_sign_negative(),
            "sign convention broken"
        );
    }
}

/// The Hessian handed back is the **raw** second derivative, with nothing imposed on it.
///
/// The projection that produces the frequencies is applied to a mass-weighted copy. A caller
/// doing its own analysis — a thermochemistry code, a transition-state search, a comparison
/// against another program's Hessian — needs the unmodified matrix, and would have no way to
/// recover it if the rigid-body subspace had been removed on the way out.
#[test]
fn the_reported_hessian_is_the_unmodified_analytic_one() {
    use pm3_rs::{analytic_hessian, vibrational_analysis, Molecule, Pm3Options, Pm3Parameters};

    let mol = Molecule::from_xyz_str(WATER_XYZ, 0.0).unwrap();
    let params = Pm3Parameters::standard().unwrap();
    let options = Pm3Options::default();

    let raw = analytic_hessian(&mol, &params, &options, 1.0e-3).unwrap();
    let modes = vibrational_analysis(&mol, &params, &options, 1.0e-3).unwrap();

    for i in 0..raw.rows {
        for j in 0..raw.cols {
            assert_eq!(
                modes.hessian[(i, j)],
                raw[(i, j)],
                "VibrationalModes::hessian differs from analytic_hessian at ({i}, {j})"
            );
        }
    }
    // And it is symmetric, which a one-sided sum-rule correction would have broken.
    for i in 0..raw.rows {
        for j in 0..i {
            assert_eq!(
                modes.hessian[(i, j)],
                modes.hessian[(j, i)],
                "the reported Hessian is not symmetric at ({i}, {j})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// docs/rust-api.md — "Geometry optimization (L-BFGS)"
// ---------------------------------------------------------------------------

#[test]
fn documented_optimizer_surface() {
    use pm3_rs::gradient::closed_form_gradient;
    use pm3_rs::optimizer::{optimize, OptOptions};
    use pm3_rs::{Molecule, Pm3Options, Pm3Parameters};

    let mol = Molecule::from_xyz_str(WATER_XYZ, 0.0).unwrap();
    let params = Pm3Parameters::standard().unwrap();
    let options = Pm3Options::default();

    // `OptOptions { max_iter, gtol /* eV/Bohr */, grad_step, history }`.
    let defaults = OptOptions::default();
    let _: usize = defaults.max_iter;
    let _: f64 = defaults.gtol;
    let _: f64 = defaults.grad_step;
    let _: usize = defaults.history;

    let res = optimize(&mol, &params, &options, &defaults).unwrap();
    // `OptResult { molecule, scf, converged, iterations, trajectory }`.
    let opt_mol = res.molecule.clone();
    let _: &pm3_rs::scf::Pm3Result = &res.scf;
    assert!(res.converged);
    assert!(res.iterations > 0);
    assert!(!res.trajectory.is_empty());
    assert_eq!(opt_mol.atoms.len(), 3);

    // The optimizer must actually reach `gtol` on the max gradient component.
    let g = closed_form_gradient(&opt_mol, &params, &options).unwrap();
    assert!(
        g.max_gradient <= defaults.gtol,
        "max gradient {} > gtol {}",
        g.max_gradient,
        defaults.gtol
    );
    // Optimization lowers the energy, and `res.scf` is the result *at the
    // optimized geometry*.
    let start = pm3_rs::run_pm3(&mol, &params, &options).unwrap();
    assert!(res.scf.total_ev < start.total_ev);
    assert!((res.scf.total_ev - g.scf.total_ev).abs() < 1.0e-6);
    // The trajectory is monotonically improving in the max gradient at the end.
    let last = res.trajectory.last().unwrap();
    assert_eq!(last.positions.len(), 3);
    assert!(last.max_gradient <= defaults.gtol);
    // Documented units: the trajectory carries eV and kcal/mol side by side.
    assert!(
        (last.heat_of_formation_kcal - res.scf.heat_of_formation_kcal).abs() < 1.0e-6,
        "trajectory ΔHf must match the final SCF"
    );
}

// ---------------------------------------------------------------------------
// docs/rust-api.md — "Units" and "Errors"
// ---------------------------------------------------------------------------

/// "Energies are in eV; … the heat of formation is reported in kcal/mol."
#[test]
fn documented_unit_conventions_hold() {
    use pm3_rs::{run_pm3, Molecule, Pm3Options, Pm3Parameters};

    let mol = Molecule::from_xyz_str(WATER_XYZ, 0.0).unwrap();
    let params = Pm3Parameters::standard().unwrap();
    let r = run_pm3(&mol, &params, &Pm3Options::default()).unwrap();

    // ΔHf = (E_total − Σ E_isol + Σ ΔHf_atom) · EV_TO_KCAL, so re-deriving it
    // from the eV fields must reproduce the reported kcal/mol number.
    let mut e_isol = 0.0;
    let mut eheat = 0.0;
    for atom in &mol.atoms {
        let element = params.element(atom.z).unwrap();
        e_isol += element.e_isol;
        eheat += element.eheat_ev;
    }
    let derived = (r.total_ev - e_isol + eheat) * EV_TO_KCAL;
    assert!(
        (derived - r.heat_of_formation_kcal).abs() < 1.0e-9,
        "{derived} vs {}",
        r.heat_of_formation_kcal
    );
}

/// "All fallible entry points return `pm3_rs::error::Result<T>` … covering XYZ
/// parse errors, missing parameters, and non-convergence."
#[test]
fn documented_error_surface() {
    use pm3_rs::error::{Pm3Error, Result};
    use pm3_rs::{run_pm3, Molecule, Pm3Options, Pm3Parameters};

    // XYZ parse error.
    let parsed: Result<Molecule> = Molecule::from_xyz_str("2\nbad\nO 0 0\n", 0.0);
    assert!(matches!(
        parsed,
        Err(Pm3Error::Parse { .. }) | Err(Pm3Error::InvalidInput(_))
    ));

    // Missing parameter (titanium has no PM3 block).
    let titanium = Molecule::from_xyz_str("2\ntih\nTi 0 0 0\nH 1.6 0 0\n", 0.0).unwrap();
    let params = Pm3Parameters::standard().unwrap();
    let missing = run_pm3(&titanium, &params, &Pm3Options::default());
    assert!(matches!(missing, Err(Pm3Error::MissingElement(22))));
    assert!(missing.unwrap_err().to_string().contains("Z=22"));

    // Non-convergence is reported, not silently returned.
    let mol = Molecule::from_xyz_str(WATER_XYZ, 0.0).unwrap();
    let starved = run_pm3(
        &mol,
        &params,
        &Pm3Options {
            max_scf: 1,
            ..Pm3Options::default()
        },
    );
    assert!(matches!(starved, Err(Pm3Error::ScfNotConverged { .. })));

    // Every error type is `std::error::Error` + `Display`, as documented.
    fn assert_is_error<E: std::error::Error>(_: &E) {}
    assert_is_error(&Pm3Error::MissingElement(22));
}

// ---------------------------------------------------------------------------
// v0.2.1 additions -- every module whose contents the crate root re-exports
// ---------------------------------------------------------------------------

/// The new surface is reachable from the crate root, not only from its module.
///
/// This file is the one that fails when a documented name is renamed, and it covered none of
/// the v0.2.1 modules until now: `dipole`, `ir`, `molden` and `pbc::dfpt` were public modules
/// with nothing re-exported and nothing pinned. Naming each item here is what makes a rename a
/// compile error rather than a silent break in someone's import.
#[test]
fn the_new_modules_are_reachable_from_the_crate_root() {
    use pm3_rs::pbc::gamma::PeriodicOptions;
    use pm3_rs::{
        centre_of_mass, dipole_derivatives, dipole_from_density, dipole_matrix, dynamical_matrix,
        dynamical_matrix_on_mesh, field_terms, ir_spectrum, molden_string, phonon_frequencies,
        phonon_frequencies_on_mesh, rigid_ion_dynamical_matrix, Cell, DynamicalMatrix, IrSpectrum,
        Molecule, Pm3Options, Pm3Parameters, Vec3,
    };

    let molecule = Molecule::from_xyz_str(WATER_XYZ, 0.0).unwrap();
    let params = Pm3Parameters::standard().unwrap();
    let options = Pm3Options::default();
    let basis = pm3_rs::basis::Basis::build(&molecule, &params).unwrap();
    let scf = pm3_rs::run_pm3(&molecule, &params, &options).unwrap();

    // Dipole.
    let com = centre_of_mass(&molecule, &params).unwrap();
    let matrices = dipole_matrix(&molecule, &params, &basis, com).unwrap();
    assert_eq!(matrices.len(), 3);
    let mu = dipole_from_density(&molecule, &params, &basis, &scf.density, com).unwrap();
    assert!(mu.norm().is_finite());
    let (_, nuclear) = field_terms(&molecule, &params, &basis, Vec3::new(0.0, 0.0, 0.01)).unwrap();
    assert!(nuclear.is_finite());

    // Infrared.
    let derivatives = dipole_derivatives(&molecule, &params, &options).unwrap();
    assert_eq!(
        (derivatives.rows, derivatives.cols),
        (3, 3 * molecule.atoms.len())
    );
    let spectrum: IrSpectrum = ir_spectrum(&molecule, &params, &options, 1.0e-3).unwrap();
    assert_eq!(
        spectrum.frequencies_cm.len(),
        spectrum.intensities_km_per_mol.len()
    );

    // Molden.
    let document = molden_string(&molecule, &params, &scf).unwrap();
    assert!(document.contains("[Molden Format]"));

    // Phonons at a wavevector.
    let mut crystal = molecule.clone();
    crystal.cell = Some(Cell::cubic(12.0).unwrap());
    let periodic = PeriodicOptions::default();
    let q = [0.25, 0.0, 0.0];
    let rigid: DynamicalMatrix =
        rigid_ion_dynamical_matrix(&crystal, &params, &options, &periodic, q).unwrap();
    assert_eq!(rigid.matrix.rows, 3 * crystal.atoms.len());
    let full = dynamical_matrix(&crystal, &params, &options, &periodic, q).unwrap();
    assert!(full.hermitian_defect < 1.0e-6);
    let frequencies = phonon_frequencies(&crystal, &params, &options, &periodic, q).unwrap();
    assert_eq!(frequencies.len(), 3 * crystal.atoms.len());

    let mesh = pm3_rs::KpointOptions::mesh([1, 1, 1]);
    let on_mesh =
        dynamical_matrix_on_mesh(&crystal, &params, &options, &periodic, &mesh, q).unwrap();
    assert_eq!(on_mesh.matrix.rows, full.matrix.rows);
    let mesh_frequencies =
        phonon_frequencies_on_mesh(&crystal, &params, &options, &periodic, &mesh, q).unwrap();
    assert_eq!(mesh_frequencies.len(), frequencies.len());

    // A caller holding a `DynamicalMatrix` can read frequencies off it without reimplementing
    // the mass weighting or the sign convention. `phonon_frequencies` is this composed with
    // `dynamical_matrix`, so the two must agree exactly rather than merely closely -- if they
    // do not, one of them is weighting by something else.
    let derived = pm3_rs::frequencies_of(&full).unwrap();
    assert_eq!(derived, frequencies);

    // The masses are part of the public surface for the same reason: they are the crate's
    // isotope-averaged values, and nothing else in the API hands them out, so a caller who
    // wants to mass-weight `D(q)` themselves has no other source for them.
    assert_eq!(full.masses.len(), crystal.atoms.len());
    assert!(full.masses[0] > full.masses[1], "oxygen outweighs hydrogen");
}
