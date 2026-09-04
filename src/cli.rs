// SPDX-License-Identifier: GPL-3.0-or-later

//! The command-line interface, as a library entry point.
//!
//! It lives here rather than in `src/bin` so that the same implementation backs both the
//! `pm3-rs` executable and the console script installed by `pip`. A PyO3 wrapper hands over
//! `sys.argv`; the binary hands over `std::env::args`. Neither knows anything the other does
//! not.
use crate::constants::ANGSTROM_TO_BOHR;
use crate::{
    analytic_hessian, closed_form_gradient, optimize, run_pm3, vibrational_analysis, Cell,
    Molecule, OptOptions, Pm3Options, Pm3Parameters,
};
use std::env;
use std::path::Path;

fn cli_usage() -> &'static str {
    r#"Usage: pm3_rs_cli <command> file.xyz [options]

PM3 calculations on a standard XYZ geometry (Angstrom).

Commands
  energy       total and heat-of-formation energies
  gradient     energy and Cartesian gradient (eV/Bohr)
  charges      Mulliken charges and dipole
  optimize     optimize geometry and write <input>.pm3opt.xyz
  frequencies  harmonic frequencies (cm^-1)
  hessian      Cartesian Hessian (eV/Bohr^2)
  molden       write <input>.molden: orbitals, energies and occupations for a viewer
  ir           harmonic frequencies with infrared intensities (km/mol)
  stress       stress tensor (eV/Bohr^3); needs --cell
  phonons      phonon frequencies; Gamma-point, or at --q by DFPT; needs --cell
  bands        electronic band structure along a default path; needs --cell
  phonon-bands phonon dispersion from supercell force constants; needs --cell
               and --supercell
  born         Born effective charges Z* and their acoustic sum rule; needs --cell
  dielectric   polarizability and the dielectric tensor; needs --cell
  berry        Berry-phase polarization, modulo its quantum; needs --cell
  finite-field a field along a periodic direction, by the Berry-phase electric
               enthalpy; needs --cell, --field and --kpts

Options
  --charge <q>            total charge; per unit cell for a periodic system
  --multiplicity <m>      spin multiplicity (2S+1)
  --reference <name>      auto | rhf | uhf; auto is RHF closed shell, UHF open.
                          uhf on a singlet is how a broken-symmetry solution is
                          asked for; rhf on an open shell is refused
  --method <name>         pm3 | pm3-d3 | pm3-d3h4 | pm3-d3h4x
  --no-diis               disable SCF acceleration
  --cell <a[,b,c[,...]]>  cell in Angstrom: one number (cubic), three
                          (orthorhombic), or nine (lattice vectors as rows)
  --pbc <x,y,z>           which directions are periodic (1 or 0); default 1,1,1
  --kpts <n1,n2,n3>       Gamma-centred Monkhorst-Pack mesh; default is Gamma only
  --q <h,k,l>             phonon wavevector, in fractions of the reciprocal
                          lattice vectors; makes `phonons` a DFPT run at that
                          wavevector instead of the Gamma-point Hessian. With
                          --kpts the response is summed over that mesh, each
                          point paired with k+q
  --rigid-ion             with --q, leave the electronic response out: the
                          fixed-density half of D(q) alone
  --supercell <n1,n2,n3>  replication for `phonon-bands`: the force constants
                          are cut out of that supercell's Gamma-point Hessian
                          and Fourier interpolated onto the path. Costs one
                          Hessian for the whole dispersion
  --lo-to <x,y,z>         with `phonons --q 0,0,0`, add the non-analytic term
                          along this Cartesian direction: the longitudinal
                          `q -> 0` limit instead of the transverse one. 3D only,
                          and it needs a direction because the limit has one
  --strings <n>           k-points per Berry-phase string (default 12); the
                          convergence parameter for `berry` and the string
                          length `finite-field` reads from --kpts
  --static                with `dielectric`, also report the static tensor eps_0,
                          which lets the nuclei relax along each infrared-active
                          mode. Costs a Gamma-point phonon run and a set of Born
                          charges, and is only meaningful at a relaxed geometry
  --dc <radius>           divide-and-conquer buffer radius in Angstrom
  --dc-core <radius>      divide-and-conquer core radius in Angstrom (default 3.2)
  --field <x,y,z>         uniform electric field in V/Angstrom; molecular only

A Gamma-point periodic run reports its Gamma margin: the cell width minus the
exchange cutoff. It must be positive, or one k-point is not enough and the answer
is wrong regardless of how cleanly the SCF converged. See docs/pbc.md.
"#
}

#[derive(Debug)]
struct Cli {
    command: String,
    path: String,
    charge: f64,
    multiplicity: usize,
    use_diis: bool,
    /// Which SCF reference to use. `Auto` unless `--reference` says otherwise — without the flag
    /// a broken-symmetry UHF singlet could not be asked for from here at all.
    reference: crate::Reference,
    variant: crate::Variant,
    /// Whatever the caller wrote after `--cell`: one number (cubic), three (orthorhombic edges)
    /// or nine (lattice vectors as rows), all in Angstrom.
    cell: Option<Vec<f64>>,
    /// Which directions are periodic; all three unless `--pbc` says otherwise.
    pbc: Option<[bool; 3]>,
    /// Monkhorst-Pack divisions; `None` means the Gamma point alone.
    kpts: Option<[usize; 3]>,
    /// The phonon wavevector in fractional reciprocal coordinates; `None` keeps `phonons` on the
    /// Γ-point Hessian rather than sending it through DFPT.
    q: Option<[f64; 3]>,
    /// With `q`, drop the electronic response and report the rigid-ion matrix.
    rigid_ion: bool,
    /// Supercell replication for `phonon-bands`; `None` means the command was not asked for.
    supercell: Option<[usize; 3]>,
    /// Cartesian direction for the LO–TO non-analytic term; `None` leaves it off.
    lo_to: Option<[f64; 3]>,
    /// k-points per Berry-phase string.
    strings: usize,
    /// With `dielectric`, add the ionic term and report `ε₀` as well as `ε∞`.
    static_dielectric: bool,
    /// Divide-and-conquer buffer radius in Angstrom; `None` leaves the method off.
    dc_buffer: Option<f64>,
    dc_core: f64,
    /// A uniform external field in volts per Angstrom; None leaves it off. Molecular only.
    field: Option<crate::Vec3>,
}

/// Turn `--cell` into a [`Cell`], in Angstrom like the XYZ input.
fn build_cell(values: &[f64], pbc: [bool; 3]) -> Result<Cell, String> {
    let rows: [[f64; 3]; 3] = match values.len() {
        1 => [
            [values[0], 0.0, 0.0],
            [0.0, values[0], 0.0],
            [0.0, 0.0, values[0]],
        ],
        3 => [
            [values[0], 0.0, 0.0],
            [0.0, values[1], 0.0],
            [0.0, 0.0, values[2]],
        ],
        9 => [
            [values[0], values[1], values[2]],
            [values[3], values[4], values[5]],
            [values[6], values[7], values[8]],
        ],
        other => {
            return Err(format!(
                "--cell takes 1 (cubic), 3 (orthorhombic) or 9 (general) numbers, got {other}"
            ))
        }
    };
    let scaled = rows.map(|row| row.map(|v| v * ANGSTROM_TO_BOHR));
    Cell::from_rows(scaled, pbc).map_err(|e| e.to_string())
}

fn parse_numbers(value: &str, flag: &str) -> Result<Vec<f64>, String> {
    value
        .split([',', ' '])
        .filter(|piece| !piece.is_empty())
        .map(|piece| {
            piece
                .parse::<f64>()
                .map_err(|_| format!("invalid number in {flag}: {piece}"))
        })
        .collect()
}

fn parse_args(argv: &[String]) -> Result<Cli, String> {
    let mut args = argv.iter().skip(1).cloned();
    let command = args.next().ok_or_else(|| cli_usage().to_owned())?;
    if command == "--help" || command == "-h" {
        return Err(String::new());
    }
    let path = args
        .next()
        .ok_or_else(|| format!("missing XYZ file\n\n{}", cli_usage()))?;
    let mut charge = 0.0;
    let mut multiplicity = 1;
    let mut use_diis = true;
    let mut reference = crate::Reference::Auto;
    let mut variant = crate::Variant::Pm3;
    let mut cell = None;
    let mut pbc = None;
    let mut kpts = None;
    let mut dc_buffer = None;
    let mut dc_core = 3.2;
    let mut field = None;
    let mut q = None;
    let mut rigid_ion = false;
    let mut supercell = None;
    let mut static_dielectric = false;
    let mut lo_to = None;
    let mut strings = 12usize;
    while let Some(flag) = args.next() {
        if flag == "--no-diis" {
            use_diis = false;
            continue;
        }
        if flag == "--rigid-ion" {
            rigid_ion = true;
            continue;
        }
        if flag == "--static" {
            static_dielectric = true;
            continue;
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--charge" => {
                charge = value
                    .parse()
                    .map_err(|_| format!("invalid charge: {value}"))?
            }
            "--multiplicity" => {
                multiplicity = value
                    .parse()
                    .map_err(|_| format!("invalid multiplicity: {value}"))?;
                if multiplicity == 0 {
                    return Err("multiplicity must be >= 1".to_owned());
                }
            }
            "--reference" => {
                reference = match value.to_ascii_lowercase().as_str() {
                    "auto" => crate::Reference::Auto,
                    "rhf" => crate::Reference::Rhf,
                    "uhf" => crate::Reference::Uhf,
                    other => return Err(format!("unknown reference: {other} (auto, rhf, uhf)")),
                };
            }
            "--method" => {
                variant = crate::Variant::parse(&value).ok_or_else(|| {
                    format!("unknown method: {value} (PM3, PM3-D3, PM3-D3H4, PM3-D3H4X)")
                })?;
            }
            "--field" => {
                let components = parse_numbers(&value, "--field")?;
                if components.len() != 3 {
                    return Err("--field takes three components in volts per Angstrom".to_owned());
                }
                // Volts per Angstrom in, eV per Bohr with the sign that makes `E = E₀ + μ·F`
                // inside — the same conversion the Python layer applies, and MOPAC's own unit.
                let scale = -crate::constants::BOHR_TO_ANGSTROM;
                field = Some(crate::Vec3::new(
                    scale * components[0],
                    scale * components[1],
                    scale * components[2],
                ));
            }
            "--cell" => cell = Some(parse_numbers(&value, "--cell")?),
            "--pbc" => {
                let flags = parse_numbers(&value, "--pbc")?;
                if flags.len() != 3 {
                    return Err("--pbc takes three values (1 = periodic, 0 = not)".to_owned());
                }
                pbc = Some([flags[0] != 0.0, flags[1] != 0.0, flags[2] != 0.0]);
            }
            "--kpts" => {
                // Parsed as integers, not as floats cast to `usize`. The cast saturated a
                // negative to zero, so `--kpts -1,1,1` produced "k-point division along axis 0
                // is zero; use 1 for no sampling" — a message about a value the caller never
                // typed — and it truncated `2.9` to 2 without comment.
                let divisions: Vec<usize> = value
                    .split([',', ' '])
                    .filter(|piece| !piece.is_empty())
                    .map(|piece| {
                        piece.parse::<usize>().map_err(|_| {
                            format!("invalid division in --kpts: {piece} (a positive whole number)")
                        })
                    })
                    .collect::<Result<_, _>>()?;
                if divisions.len() != 3 {
                    return Err("--kpts takes three divisions".to_owned());
                }
                kpts = Some([divisions[0], divisions[1], divisions[2]]);
            }
            "--supercell" => {
                // Whole numbers for the same reason `--kpts` is: a replication is a count, and a
                // float cast would turn `2.9` into two copies without saying so.
                let repeats: Vec<usize> = value
                    .split([',', ' '])
                    .filter(|piece| !piece.is_empty())
                    .map(|piece| {
                        piece.parse::<usize>().map_err(|_| {
                            format!(
                                "invalid repeat in --supercell: {piece} (a positive whole number)"
                            )
                        })
                    })
                    .collect::<Result<_, _>>()?;
                if repeats.len() != 3 {
                    return Err("--supercell takes three repeats".to_owned());
                }
                supercell = Some([repeats[0], repeats[1], repeats[2]]);
            }
            "--lo-to" => {
                let components = parse_numbers(&value, "--lo-to")?;
                if components.len() != 3 {
                    return Err(
                        "--lo-to takes three Cartesian components naming a direction".to_owned(),
                    );
                }
                lo_to = Some([components[0], components[1], components[2]]);
            }
            "--strings" => {
                strings = value
                    .parse()
                    .map_err(|_| format!("invalid --strings: {value} (a whole number)"))?;
            }
            "--q" => {
                let components = parse_numbers(&value, "--q")?;
                if components.len() != 3 {
                    return Err(
                        "--q takes three fractional components, one per reciprocal lattice vector"
                            .to_owned(),
                    );
                }
                q = Some([components[0], components[1], components[2]]);
            }
            "--dc" => {
                dc_buffer = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid divide-and-conquer buffer: {value}"))?,
                )
            }
            "--dc-core" => {
                dc_core = value
                    .parse()
                    .map_err(|_| format!("invalid divide-and-conquer core radius: {value}"))?
            }
            _ => return Err(format!("unknown option: {flag}\n\n{}", cli_usage())),
        }
    }
    match command.as_str() {
        "energy" | "gradient" | "charges" | "optimize" | "frequencies" | "hessian" | "stress"
        | "phonons" | "bands" | "molden" | "ir" | "phonon-bands" | "born" | "dielectric"
        | "berry" | "finite-field" => {}
        _ => return Err(format!("unknown command: {command}\n\n{}", cli_usage())),
    }
    if cell.is_none()
        && matches!(
            command.as_str(),
            "stress"
                | "phonons"
                | "bands"
                | "phonon-bands"
                | "born"
                | "dielectric"
                | "berry"
                | "finite-field"
        )
    {
        return Err(format!("{command} needs a periodic cell; pass --cell"));
    }
    // `--lo-to` adds the non-analytic term to a dynamical matrix, and only the `q = 0` limit of
    // one has a direction for it to be taken along.
    if lo_to.is_some() && !matches!(command.as_str(), "phonons" | "frequencies") {
        return Err(format!(
            "--lo-to adds the non-analytic term to a dynamical matrix and only `phonons` builds \
             one; {command} would ignore it"
        ));
    }
    if lo_to.is_some() && q.is_none() {
        return Err(
            "--lo-to is the direction the `q -> 0` limit is taken along, so it needs a --q to be \
             the limit of. Pass --q 0,0,0 for the limit itself."
                .to_owned(),
        );
    }
    if command == "finite-field" {
        if field.is_none() {
            return Err(
                "finite-field needs a --field to apply. A field orthogonal to every lattice \
                 vector does not need this command at all."
                    .to_owned(),
            );
        }
        if kpts.is_none() {
            return Err(
                "finite-field needs a --kpts: the division along each field direction is that \
                 direction's Berry-phase string length, and at least 3 are needed to resolve a \
                 winding."
                    .to_owned(),
            );
        }
    } else if field.is_some() && cell.is_some() {
        // Every other periodic path refuses a field, which is what `pbc::refuse_field` is for.
        // Naming the command that does accept one turns a refusal into a redirection.
        return Err(format!(
            "a field is not lattice-periodic along a periodic direction, so {command} refuses \
             one. Use `finite-field`, which minimizes the Berry-phase electric enthalpy instead."
        ));
    }
    // `--supercell` is what `phonon-bands` interpolates from, and nothing else reads it.
    if supercell.is_some() && command != "phonon-bands" {
        return Err(format!(
            "--supercell is the replication `phonon-bands` interpolates from; {command} would \
             ignore it"
        ));
    }
    if command == "phonon-bands" && supercell.is_none() {
        return Err(
            "phonon-bands interpolates from a supercell's force constants, so it needs a \
             --supercell to cut them out of. A repeat above 1 along an axis is what resolves \
             dispersion along that axis, so `--supercell 2,1,1` is enough for one direction and \
             `2,2,2` covers all three."
                .to_owned(),
        );
    }
    // `--static` adds the ionic term to a dielectric tensor. There is no other tensor here for it
    // to be added to, and a command that quietly ignored it would look like it had honoured it.
    if static_dielectric && command != "dielectric" {
        return Err(format!(
            "--static asks for the ionic half of a dielectric tensor and only `dielectric` \
             reports one; {command} would ignore it"
        ));
    }
    if cell.is_none() && (pbc.is_some() || kpts.is_some()) {
        return Err("--pbc and --kpts need a --cell".to_owned());
    }
    // `--q` names a phonon wavevector, so it belongs to the one command that has phonons. Any
    // other command would silently ignore it, which is the failure mode worth refusing.
    if q.is_some() && !matches!(command.as_str(), "phonons" | "frequencies") {
        return Err(format!(
            "--q is a phonon wavevector and only `phonons` takes one; {command} would ignore it"
        ));
    }
    // `frequencies` is the one command that runs either way: with a cell it is the periodic path
    // and takes a wavevector, without one it is the molecular Hessian, which has no lattice for a
    // wavevector to be measured against. Refuse rather than compute molecular frequencies and
    // drop the `--q` the caller asked for.
    if q.is_some() && cell.is_none() {
        return Err(
            "--q is measured in fractions of the reciprocal lattice vectors, so it needs a \
             --cell. Without one this is the molecular Hessian, which has no wavevector."
                .to_owned(),
        );
    }
    if rigid_ion && q.is_none() {
        return Err(
            "--rigid-ion drops the electronic response from D(q), so it needs a --q to drop it \
             from. The Gamma-point Hessian has no rigid-ion form here."
                .to_owned(),
        );
    }
    if rigid_ion && kpts.is_some() {
        return Err(
            "the rigid-ion matrix has no electronic response, so a k-mesh would change nothing; \
             drop --kpts or drop --rigid-ion"
                .to_owned(),
        );
    }
    // `--kpts` is a Brillouin-zone mesh, and only the paths that integrate over one can use it.
    // Three here cannot, and each used to parse the flag and then drop it: the caller got a
    // Γ-point answer under the impression it had been sampled. Refusing is what the Python
    // layer already does for the same combination on `relax`.
    if kpts.is_some() {
        if dc_buffer.is_some() {
            return Err(
                "--dc runs a Gamma-point partitioned SCF and has no k-mesh to sample; drop \
                 --kpts, or drop --dc and run the mesh on the whole cell"
                    .to_owned(),
            );
        }
        if matches!(command.as_str(), "optimize" | "hessian") {
            return Err(format!(
                "{command} is a Gamma-point path for a periodic cell, so --kpts would be ignored \
                 rather than honoured. Check the Gamma margin `energy` reports before trusting \
                 it, and use a larger supercell if the margin is negative."
            ));
        }
        if matches!(command.as_str(), "phonons" | "frequencies") && q.is_none() {
            return Err(
                "Gamma-point phonons come from the Gamma-point Hessian, which has no k-mesh. \
                 Pass --q to run DFPT, where the response is summed over --kpts."
                    .to_owned(),
            );
        }
        // These are Γ-point paths for the same reason `hessian` is: each runs a Γ-point response
        // and would parse the mesh without ever sampling it. `berry` and `finite-field` are not
        // in this list -- they read the mesh as their string length.
        if matches!(command.as_str(), "phonon-bands" | "born" | "dielectric") {
            return Err(format!(
                "{command} is a Gamma-point path, so --kpts would be ignored rather than \
                 honoured. Check the Gamma margin `energy` reports before trusting it, and use a \
                 larger cell if the margin is negative."
            ));
        }
    }
    Ok(Cli {
        command,
        path,
        charge,
        multiplicity,
        use_diis,
        reference,
        variant,
        cell,
        pbc,
        kpts,
        q,
        rigid_ion,
        supercell,
        lo_to,
        strings,
        static_dielectric,
        dc_buffer,
        dc_core,
        field,
    })
}

fn write_xyz(path: &Path, molecule: &Molecule) -> std::io::Result<()> {
    let mut out = format!(
        "{}\npm3-rs optimized geometry; coordinates in Angstrom\n",
        molecule.len()
    );
    for atom in &molecule.atoms {
        let symbol = crate::z_to_symbol(atom.z).unwrap_or("X");
        let pos = atom.position / ANGSTROM_TO_BOHR;
        out.push_str(&format!(
            "{symbol:2} {:+.10} {:+.10} {:+.10}\n",
            pos.x, pos.y, pos.z
        ));
    }
    std::fs::write(path, out)
}

/// The periodic and divide-and-conquer commands.
///
/// Split out rather than folded into the molecular `match` because the two answer different
/// questions: a periodic run reports energies **per unit cell** and adds a stress. Printing a
/// per-cell energy under the same label as a total one invites reading it as a total.
fn run_extended(
    molecule: &Molecule,
    params: &Pm3Parameters,
    options: &Pm3Options,
    cli: &Cli,
) -> Result<(), Box<dyn std::error::Error>> {
    let periodic = crate::PeriodicOptions::default();

    if let Some(buffer) = cli.dc_buffer {
        let dc = crate::DcOptions {
            core_radius: cli.dc_core * ANGSTROM_TO_BOHR,
            buffer_radius: buffer * ANGSTROM_TO_BOHR,
            ..crate::DcOptions::default()
        };
        if molecule.cell.is_some() {
            let r = crate::run_dc_gamma(molecule, params, options, &periodic, &dc)?;
            println!("Total energy per cell: {:.12} eV", r.total_ev);
            println!(
                "Heat of formation:     {:.8} kcal/mol",
                r.heat_of_formation_kcal
            );
            println!("Fermi level:           {:.8} eV", r.fermi_ev);
            println!("Subsystems:            {}", r.n_subsystems);
            println!("Largest subsystem:     {} orbitals", r.largest_subsystem);
            println!("Dropped pairs:         {}", r.dropped_pairs);
            println!("Gamma margin:          {:.4} Bohr", r.gamma_margin);
        } else {
            let r = crate::run_dc(molecule, params, options, &dc)?;
            println!("Total energy:          {:.12} eV", r.total_ev);
            println!(
                "Heat of formation:     {:.8} kcal/mol",
                r.heat_of_formation_kcal
            );
            println!("Fermi level:           {:.8} eV", r.fermi_ev);
            println!("Subsystems:            {}", r.n_subsystems);
            println!("Largest subsystem:     {} orbitals", r.largest_subsystem);
            println!("Dropped pairs:         {}", r.dropped_pairs);
        }
        return Ok(());
    }

    let kopt = cli.kpts.map(|divisions| crate::KpointOptions {
        spec: crate::KpointSpec::mesh(divisions),
        ..crate::KpointOptions::default()
    });

    match cli.command.as_str() {
        "energy" | "charges" => match &kopt {
            Some(mesh) => {
                let r = crate::run_kpoints(molecule, params, options, &periodic, mesh)?;
                println!("Total energy per cell: {:.12} eV", r.total_ev);
                println!("Electronic energy:     {:.12} eV", r.electronic_ev);
                println!("Core-core energy:      {:.12} eV", r.core_ev);
                println!("Correction energy:     {:.12} eV", r.correction_ev);
                println!("Ewald energy:          {:.12} eV", r.ewald_ev);
                println!(
                    "Heat of formation:     {:.8} kcal/mol",
                    r.heat_of_formation_kcal
                );
                println!("k-points (irreducible): {}", r.kpoints.len());
                println!("Fermi level:           {:.8} eV", r.fermi_ev);
                match r.band_gap_ev {
                    Some(gap) => println!("Band gap:              {gap:.8} eV"),
                    None => println!("Band gap:              (none: no gap in the sampled bands)"),
                }
                if cli.command == "charges" {
                    println!("Mulliken charges:      {:?}", r.charges);
                }
            }
            None => {
                let r = crate::run_gamma(molecule, params, options, &periodic)?;
                println!("Total energy per cell: {:.12} eV", r.total_ev);
                println!("Electronic energy:     {:.12} eV", r.electronic_ev);
                println!("Core-core energy:      {:.12} eV", r.core_ev);
                println!("Correction energy:     {:.12} eV", r.correction_ev);
                println!("Ewald energy:          {:.12} eV", r.ewald_ev);
                println!(
                    "Heat of formation:     {:.8} kcal/mol",
                    r.heat_of_formation_kcal
                );
                println!("Gamma margin:          {:.4} Bohr", r.gamma_margin);
                if r.gamma_margin <= 0.0 {
                    eprintln!(
                        "warning: the cell is narrower than the exchange cutoff, so one k-point \
                         is not enough here and this energy is wrong however cleanly the SCF \
                         converged. Use --kpts or a larger supercell; see docs/pbc.md."
                    );
                }
                if cli.command == "charges" {
                    println!("Mulliken charges:      {:?}", r.charges);
                }
            }
        },
        "gradient" | "stress" => {
            let (energy, gradient, stress) = match &kopt {
                Some(mesh) => {
                    let g = crate::kpoint_gradient(molecule, params, options, &periodic, mesh)?;
                    (g.energy_ev, g.gradient, g.stress)
                }
                None => {
                    let g = crate::periodic_gradient(molecule, params, options, &periodic)?;
                    (g.energy_ev, g.gradient, g.stress)
                }
            };
            println!("Total energy per cell: {energy:.12} eV");
            if cli.command == "gradient" {
                println!("Gradient (eV/Bohr):");
                for (index, g) in gradient.iter().enumerate() {
                    println!("  {index:4} {:+.10} {:+.10} {:+.10}", g.x, g.y, g.z);
                }
            }
            match stress {
                Some(s) => {
                    println!("Stress (eV/Bohr^3):");
                    for alpha in 0..3 {
                        println!(
                            "  {:+.10} {:+.10} {:+.10}",
                            s.col[0].to_array()[alpha],
                            s.col[1].to_array()[alpha],
                            s.col[2].to_array()[alpha]
                        );
                    }
                }
                None => println!(
                    "Stress: not defined here. A slab or a chain has no strain derivative along \
                     its non-periodic directions, and zeros would be a different claim."
                ),
            }
        }
        "phonons" | "frequencies" => match cli.q {
            // A wavevector goes through DFPT. There is no acoustic sum rule to report away from
            // Γ, and no separate SCF energy to attach, so what is printed is the dispersion at
            // that point plus the Hermitian defect the matrix was assembled at — the number that
            // moves first if a phase is wrong.
            Some(q_frac) => {
                let matrix = if cli.rigid_ion {
                    crate::rigid_ion_dynamical_matrix(molecule, params, options, &periodic, q_frac)?
                } else if let Some(mesh) = &kopt {
                    crate::pbc::dfpt::dynamical_matrix_on_mesh(
                        molecule, params, options, &periodic, mesh, q_frac,
                    )?
                } else {
                    crate::dynamical_matrix(molecule, params, options, &periodic, q_frac)?
                };
                let mut matrix = matrix;
                if let Some(direction) = cli.lo_to {
                    // The Born charges and `ε∞` the term is built from, computed here rather
                    // than asked for, because a caller who has to supply them separately can
                    // supply ones from a different geometry.
                    let born = crate::born_charges(molecule, params, options, &periodic)?;
                    let epsilon =
                        crate::dielectric_tensor(molecule, params, options, &periodic)?.epsilon;
                    crate::add_non_analytic(
                        &mut matrix,
                        molecule,
                        crate::Vec3::new(direction[0], direction[1], direction[2]),
                        &born,
                        epsilon,
                    )?;
                }
                let frequencies = crate::pbc::dfpt::frequencies_of(&matrix)?;
                println!(
                    "Wavevector (fractional): {:+.6} {:+.6} {:+.6}",
                    q_frac[0], q_frac[1], q_frac[2]
                );
                if let Some(direction) = cli.lo_to {
                    println!(
                        "LO-TO direction:         {:+.4} {:+.4} {:+.4}",
                        direction[0], direction[1], direction[2]
                    );
                }
                if cli.rigid_ion {
                    println!("Electronic response:     omitted (--rigid-ion)");
                }
                println!("Frequencies (cm^-1):");
                for (index, frequency) in frequencies.iter().enumerate() {
                    println!("  {index:4} {frequency:+12.4}");
                }
                println!(
                    "Hermitian defect: {:.3e} eV/Bohr^2 (should vanish)",
                    matrix.hermitian_defect
                );
            }
            None => {
                let modes = crate::periodic_phonons(molecule, params, options, &periodic)?;
                println!("Total energy per cell: {:.12} eV", modes.scf.total_ev);
                println!("Gamma-point frequencies (cm^-1):");
                for (index, frequency) in modes.frequencies_cm.iter().enumerate() {
                    println!("  {index:4} {frequency:+12.4}");
                }
                println!(
                    "Largest acoustic residual: {:.4} cm^-1 (should vanish)",
                    modes.acoustic_residual_cm
                );
            }
        },
        "bands" => {
            let mesh = kopt.unwrap_or_default();
            let scf = crate::run_kpoints(molecule, params, options, &periodic, &mesh)?;
            let cell = molecule.cell.expect("checked while parsing the arguments");
            // A path along the three axes of the reciprocal cell. Naming high-symmetry points
            // would mean classifying the lattice, which this tool does not do.
            let corners = [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 0.5],
            ];
            let path = crate::band_path(&cell, &corners, 12)?;
            let bands = crate::band_structure(molecule, params, &periodic, &scf, &path)?;
            println!("Fermi level: {:.8} eV", bands.fermi_ev);
            println!("distance(1/Bohr)  bands (eV)");
            for (distance, energies) in bands.distances.iter().zip(&bands.bands) {
                let row: Vec<String> = energies.iter().map(|e| format!("{e:.4}")).collect();
                println!("  {distance:10.5}  {}", row.join(" "));
            }
        }
        "phonon-bands" => {
            let repeats = cli.supercell.expect("checked while parsing the arguments");
            let constants = crate::ForceConstants::from_supercell(
                molecule, params, options, &periodic, repeats,
            )?;
            // The same corners `bands` uses, for the same reason: naming high-symmetry points
            // would mean classifying the lattice, which this tool does not do.
            let corners = [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 0.5],
            ];
            let path = crate::q_path(&corners, 12);
            let bands = constants.band_structure(&path)?;
            println!(
                "Supercell: {} x {} x {}",
                repeats[0], repeats[1], repeats[2]
            );
            // Printed before the dispersion rather than after: it says how much of the answer is
            // the truncation, and a reader who sees it first knows what to make of what follows.
            println!(
                "Acoustic sum-rule residual: {:.4e} eV/Bohr^2 (not imposed)",
                constants.acoustic_sum_rule_residual()
            );
            println!("q(fractional)              frequencies (cm^-1)");
            for (point, row) in path.iter().zip(&bands) {
                let text: Vec<String> = row.iter().map(|f| format!("{f:+.2}")).collect();
                println!(
                    "  {:+.3} {:+.3} {:+.3}   {}",
                    point[0],
                    point[1],
                    point[2],
                    text.join(" ")
                );
            }
        }
        "born" => {
            let born = crate::born_charges(molecule, params, options, &periodic)?;
            println!("Born effective charges Z* (electrons), per atom:");
            for (index, tensor) in born.iter().enumerate() {
                let z = molecule.atoms[index].z;
                let symbol = crate::z_to_symbol(z).unwrap_or("X");
                println!("  {index:4} {symbol:<2}");
                for row in tensor.iter() {
                    println!("       {:+10.5} {:+10.5} {:+10.5}", row[0], row[1], row[2]);
                }
            }
            // The sum rule is a property of the exact response, so the residual is the size of
            // everything the calculation approximated. Reported, not imposed.
            println!(
                "Acoustic sum rule |sum_a Z*_a|: {:.3e} e (should vanish)",
                crate::born_charge_sum_rule_residual(&born)
            );
        }
        "dielectric" => {
            let alpha = crate::polarizability(molecule, params, options, &periodic)?;
            println!("Polarizability alpha (Bohr^3):");
            for row in alpha.iter() {
                println!("  {:+12.5} {:+12.5} {:+12.5}", row[0], row[1], row[2]);
            }
            if cli.static_dielectric {
                let s = crate::static_dielectric_tensor(molecule, params, options, &periodic)?;
                println!("Electronic tensor eps_inf (clamped ions):");
                for row in s.electronic.iter() {
                    println!("  {:+12.5} {:+12.5} {:+12.5}", row[0], row[1], row[2]);
                }
                println!("Ionic term (4 pi / V) sum_m Z* Z* / omega_m^2:");
                for row in s.ionic.iter() {
                    println!("  {:+12.5} {:+12.5} {:+12.5}", row[0], row[1], row[2]);
                }
                println!("Static tensor eps_0 = eps_inf + ionic:");
                for row in s.epsilon.iter() {
                    println!("  {:+12.5} {:+12.5} {:+12.5}", row[0], row[1], row[2]);
                }
                // Three is the acoustic branch. More than three is a structure that is not a
                // minimum, and then the ionic term is missing whatever those modes carried.
                println!(
                    "Modes left out (omega^2 <= 0): {} (three acoustic ones are expected; more \
                     than three means this geometry is not a minimum and eps_0 is incomplete)",
                    s.skipped_modes
                );
            } else {
                match crate::dielectric_tensor(molecule, params, options, &periodic) {
                    Ok(tensors) => {
                        println!("Electronic tensor eps_inf (clamped ions):");
                        for row in tensors.epsilon.iter() {
                            println!("  {:+12.5} {:+12.5} {:+12.5}", row[0], row[1], row[2]);
                        }
                        println!("Pass --static to add the ionic term and report eps_0.");
                    }
                    // A chain or a slab has a length or an area where the formula wants a volume.
                    // The polarizability above is still the answer to the part that is defined.
                    Err(error) => println!("Dielectric tensor: not available -- {error}"),
                }
            }
        }
        "berry" => {
            let kopt = kopt.clone().unwrap_or_default();
            let p = crate::berry_polarization(
                molecule,
                params,
                options,
                &periodic,
                &kopt,
                cli.strings,
            )?;
            println!("Strings: {} k-points each", p.string_length);
            println!(
                "Electronic (e/Bohr^2): {:+.8} {:+.8} {:+.8}",
                p.electronic.x, p.electronic.y, p.electronic.z
            );
            println!(
                "Ionic      (e/Bohr^2): {:+.8} {:+.8} {:+.8}",
                p.ionic.x, p.ionic.y, p.ionic.z
            );
            println!(
                "Total      (e/Bohr^2): {:+.8} {:+.8} {:+.8}",
                p.total.x, p.total.y, p.total.z
            );
            println!(
                "Phase (turns):         {:+.8} {:+.8} {:+.8}",
                p.phase[0], p.phase[1], p.phase[2]
            );
            // Printed because the total above is only defined against it: two polarizations
            // differing by an integer combination of these are the same physical state.
            println!("Quantum (e/Bohr^2), one per lattice vector:");
            for q in &p.quantum {
                println!("  {:+.8} {:+.8} {:+.8}", q.x, q.y, q.z);
            }
            println!(
                "The total is defined modulo those. Only differences between two calculations \
                 are meaningful, reduced onto the nearest branch."
            );
        }
        "finite-field" => {
            // `cli.field` is stored **negated**: `--field` converts volts per Angstrom into the
            // sign convention `Pm3Options::field` wants, which is the one making `E = E₀ + μ·F`
            // hold for the molecular `−𝓔·r` coupling. `run_finite_field` takes the field itself,
            // so the convention has to be undone here rather than silently inverting the
            // response. The magnitude conversion is shared and correct.
            let stored = cli.field.expect("checked while parsing the arguments");
            let strength = crate::Vec3::new(-stored.x, -stored.y, -stored.z);
            let mesh = cli.kpts.expect("checked while parsing the arguments");
            let ff = crate::FiniteFieldOptions::default();
            let result =
                crate::run_finite_field(molecule, params, options, &periodic, mesh, strength, &ff)?;
            println!(
                "Field (eV per e.Bohr): {:+.6e} {:+.6e} {:+.6e}",
                result.field.x, result.field.y, result.field.z
            );
            println!("Energy:               {:.10} eV", result.scf.total_ev);
            println!(
                "Electric enthalpy:    {:.10} eV  (E - V.field.P)",
                result.enthalpy_ev
            );
            println!(
                "Polarization (e/Bohr^2): {:+.8} {:+.8} {:+.8}",
                result.polarization.x, result.polarization.y, result.polarization.z
            );
            println!(
                "Outer iterations: {} ({})",
                result.iterations,
                if result.converged {
                    "converged"
                } else {
                    "not converged"
                }
            );
            // An unresolved axis contributes zero, which is not the same as its contribution
            // being zero -- so which axes the mesh could see is printed beside the vector.
            println!(
                "Axes resolved by the mesh: {} {} {}  (an unresolved axis contributes zero to \
                 the electronic half, which is not its value)",
                result.resolved[0], result.resolved[1], result.resolved[2]
            );
        }
        "optimize" => {
            let result = crate::relax(
                molecule,
                params,
                options,
                &periodic,
                &crate::PeriodicOptOptions::default(),
            )?;
            let out = Path::new(&cli.path).with_extension("pm3opt.xyz");
            write_xyz(&out, &result.molecule)?;
            println!("Converged:             {}", result.converged);
            println!("Iterations:            {}", result.iterations);
            println!(
                "Total energy per cell: {:.12} eV",
                result.gradient.energy_ev
            );
            println!("Wrote {}", out.display());
        }
        "hessian" => {
            let h = crate::periodic_hessian(molecule, params, options, &periodic)?;
            println!(
                "Cartesian Hessian per cell (eV/Bohr^2), {} x {}:",
                h.rows, h.rows
            );
            for i in 0..h.rows {
                let row: Vec<String> = (0..h.rows).map(|j| format!("{:+.8}", h[(i, j)])).collect();
                println!("  {}", row.join(" "));
            }
        }
        other => return Err(format!("{other} is not available for a periodic system").into()),
    }
    Ok(())
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let mut molecule =
        Molecule::from_xyz_file(&cli.path, cli.charge)?.with_multiplicity(cli.multiplicity);
    if let Some(values) = &cli.cell {
        molecule.cell = Some(build_cell(values, cli.pbc.unwrap_or([true; 3]))?);
    }
    let params = Pm3Parameters::standard()?;
    let options = Pm3Options {
        charge: cli.charge,
        multiplicity: cli.multiplicity,
        use_diis: cli.use_diis,
        reference: cli.reference,
        variant: cli.variant,
        // `finite-field` is the one command whose `--field` is *not* an `-𝓔·r` term in the
        // Hamiltonian: it is the field the Berry-phase enthalpy is minimized against, and it is
        // handed to `run_finite_field` separately. Putting it here as well would be the same
        // perturbation applied twice, which that function refuses -- correctly, and confusingly,
        // since the caller passed it only once.
        field: if cli.command == "finite-field" {
            None
        } else {
            cli.field
        },
        ..Pm3Options::default()
    };
    if molecule.cell.is_some() || cli.dc_buffer.is_some() {
        return run_extended(&molecule, &params, &options, &cli);
    }
    match cli.command.as_str() {
        "energy" => {
            let result = run_pm3(&molecule, &params, &options)?;
            println!("Total energy:          {:.12} eV", result.total_ev);
            println!("Electronic energy:     {:.12} eV", result.electronic_ev);
            println!("Core-core energy:      {:.12} eV", result.core_ev);
            println!(
                "Heat of formation:     {:.8} kcal/mol",
                result.heat_of_formation_kcal
            );
            println!("SCF iterations:        {}", result.iterations);
            // `tools/oracle/all_element_validation.py` parses this line with
            // `^MO energies \(eV\):\s*\[(.*)\]\s*$`. Changing the label, the brackets or the
            // debug formatting breaks the exhaustive element sweep silently — it raises
            // "did not print MO energies" rather than a mismatch, which reads like a broken
            // harness rather than a broken format. [`tests::the_oracle_can_still_read_the_mo_line`]
            // holds the shape.
            println!("MO energies (eV):       {:?}", result.mo_energies);
        }
        "gradient" => {
            let result = closed_form_gradient(&molecule, &params, &options)?;
            println!("Energy: {:.12} eV", result.energy_ev);
            println!("# atom       dE/dx              dE/dy              dE/dz       (eV/Bohr)");
            for (i, g) in result.gradient.iter().enumerate() {
                println!(
                    "{:>5} {:>18.10e} {:>18.10e} {:>18.10e}",
                    i + 1,
                    g.x,
                    g.y,
                    g.z
                );
            }
        }
        "charges" => {
            let result = run_pm3(&molecule, &params, &options)?;
            println!("# atom  element    Mulliken charge (e)");
            for (i, (atom, charge)) in molecule.atoms.iter().zip(&result.charges).enumerate() {
                println!(
                    "{:>5} {:>4} {:>20.10}",
                    i + 1,
                    crate::z_to_symbol(atom.z).unwrap_or("X"),
                    charge
                );
            }
            println!(
                "Dipole: {:.8} {:.8} {:.8} D; |mu| = {:.8} D",
                result.dipole_debye.x,
                result.dipole_debye.y,
                result.dipole_debye.z,
                result.dipole_magnitude
            );
        }
        "optimize" => {
            let result = optimize(&molecule, &params, &options, &OptOptions::default())?;
            let input = Path::new(&cli.path);
            let stem = input
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("optimized");
            let output = input.with_file_name(format!("{stem}.pm3opt.xyz"));
            write_xyz(&output, &result.molecule)?;
            println!(
                "Converged: {} after {} optimization steps",
                result.converged, result.iterations
            );
            println!("Final energy: {:.12} eV", result.scf.total_ev);
            println!(
                "Heat of formation: {:.8} kcal/mol",
                result.scf.heat_of_formation_kcal
            );
            println!("Optimized geometry: {}", output.display());
        }
        "molden" => {
            let result = run_pm3(&molecule, &params, &options)?;
            let text = crate::molden::molden_string(&molecule, &params, &result)?;
            let input = Path::new(&cli.path);
            let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("pm3");
            let output = input.with_file_name(format!("{stem}.molden"));
            std::fs::write(&output, text)?;
            println!("Wrote {}", output.display());
            println!("SCF iterations:        {}", result.iterations);
            println!("Orbitals:              {}", result.mo_energies.len());
            if result.unrestricted {
                println!("Spin sets:             2 (alpha and beta)");
            } else {
                println!("Spin sets:             1 (restricted; doubly occupied)");
            }
        }
        "ir" => {
            let s = crate::ir::ir_spectrum(&molecule, &params, &options, 1.0e-3)?;
            println!("# mode  frequency (cm^-1)     intensity (km/mol)");
            for (i, (freq, intensity)) in s
                .frequencies_cm
                .iter()
                .zip(&s.intensities_km_per_mol)
                .enumerate()
            {
                println!("{:>5} {:>20.8} {:>22.6}", i + 1, freq, intensity);
            }
        }
        "frequencies" => {
            let result = vibrational_analysis(&molecule, &params, &options, 1.0e-3)?;
            println!("# mode  frequency (cm^-1)");
            for (i, freq) in result.frequencies_cm.iter().enumerate() {
                println!("{:>5} {:>20.8}", i + 1, freq);
            }
        }
        "hessian" => {
            let result = analytic_hessian(&molecule, &params, &options, 1.0e-3)?;
            println!("# Cartesian Hessian (eV/Bohr^2)");
            for i in 0..result.rows {
                for j in 0..result.cols {
                    if j > 0 {
                        print!(" ");
                    }
                    print!("{:.10e}", result[(i, j)]);
                }
                println!();
            }
        }
        _ => unreachable!("command was validated during parsing"),
    }
    Ok(())
}

/// Run the command-line interface over an explicit argument vector, returning a process exit
/// code.
///
/// `argv[0]` is the program name and is skipped, so this takes exactly what `std::env::args`
/// or Python's `sys.argv` hands over. Taking the vector rather than reading the process
/// environment is what lets the same code back both the `pm3-rs` executable and the console
/// script a `pip install` puts on the path — one implementation, not two that drift.
pub fn main_with_args(argv: &[String]) -> i32 {
    if matches!(
        argv.get(1).map(String::as_str),
        Some("--version") | Some("-V")
    ) {
        println!("pm3-rs {}", env!("CARGO_PKG_VERSION"));
        return 0;
    }
    match parse_args(argv) {
        Ok(cli) => match run(cli) {
            Ok(()) => 0,
            Err(err) => {
                eprintln!("pm3-rs: {err}");
                1
            }
        },
        Err(message) if message.is_empty() => {
            print!("{}", cli_usage());
            0
        }
        Err(message) => {
            eprintln!("pm3-rs: {message}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `energy` command still prints the line the element sweep parses.
    ///
    /// `tools/oracle/all_element_validation.py:275` reads the molecular orbital energies out of
    /// this command's stdout with a regular expression. Nothing else pins that format, and when
    /// it breaks the harness reports "did not print MO energies" -- which reads as a broken
    /// oracle rather than a broken CLI, and costs an afternoon. This asserts the shape without
    /// asserting the numbers, so it survives a parameter change and not a formatting one.
    #[test]
    fn the_oracle_can_still_read_the_mo_line() {
        let params = Pm3Parameters::standard().unwrap();
        let molecule = crate::system::Molecule::from_xyz_str(
            "3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n",
            0.0,
        )
        .unwrap();
        let result = run_pm3(&molecule, &params, &Pm3Options::default()).unwrap();
        let line = format!("MO energies (eV):       {:?}", result.mo_energies);

        // The oracle's own pattern, character for character.
        let pattern =
            regex_lite_mo(&line).expect("the MO line no longer matches the oracle's regex");
        let count = pattern.split(',').count();
        assert_eq!(
            count,
            result.mo_energies.len(),
            "the oracle would read {count} energies out of {}",
            result.mo_energies.len()
        );
    }

    fn argv(pieces: &[&str]) -> Vec<String> {
        std::iter::once("pm3_rs_cli")
            .chain(pieces.iter().copied())
            .map(str::to_owned)
            .collect()
    }

    /// `--q` sends `phonons` through DFPT, and every way of asking for it wrongly is refused
    /// rather than dropped.
    ///
    /// A dropped flag is the failure mode that matters here. `phonons --q 0.25,0,0` that quietly
    /// computed the Γ-point answer would print a plausible table of frequencies for the wrong
    /// wavevector, and nothing in the output would say so.
    #[test]
    fn the_wavevector_flag_is_honoured_or_refused_but_never_ignored() {
        let good = parse_args(&argv(&[
            "phonons", "c.xyz", "--cell", "6.0", "--q", "0.25,0,0",
        ]))
        .expect("a wavevector on a periodic cell is the DFPT path");
        assert_eq!(good.q, Some([0.25, 0.0, 0.0]));
        assert!(!good.rigid_ion);

        // `frequencies` runs either way: with a cell it is the periodic path, without one it is
        // the molecular Hessian, which has no lattice for a wavevector to be measured against.
        // Before this check that second case parsed happily and threw the `--q` away.
        let error = parse_args(&argv(&["frequencies", "m.xyz", "--q", "0.25,0,0"]))
            .expect_err("a wavevector without a lattice has nothing to be a fraction of");
        assert!(error.contains("--cell"), "unhelpful message: {error}");

        let error = parse_args(&argv(&[
            "energy", "c.xyz", "--cell", "6.0", "--q", "0.5,0,0",
        ]))
        .expect_err("only the phonon commands have a wavevector");
        assert!(error.contains("--q"), "unhelpful message: {error}");

        let error = parse_args(&argv(&[
            "phonons", "c.xyz", "--cell", "6.0", "--q", "0.5,0",
        ]))
        .expect_err("a wavevector has three fractional components");
        assert!(error.contains("three"), "unhelpful message: {error}");
    }

    /// `--rigid-ion` names the half of `D(q)` that has no electronic response, so it needs a `q`
    /// to take that half of, and a k-mesh would change nothing it computes.
    #[test]
    fn the_rigid_ion_flag_refuses_the_combinations_that_would_do_nothing() {
        let good = parse_args(&argv(&[
            "phonons",
            "c.xyz",
            "--cell",
            "6.0",
            "--q",
            "0.5,0,0",
            "--rigid-ion",
        ]))
        .expect("the fixed-density half at a wavevector");
        assert!(good.rigid_ion);

        let error = parse_args(&argv(&["phonons", "c.xyz", "--cell", "6.0", "--rigid-ion"]))
            .expect_err("there is no rigid-ion form of the Gamma-point Hessian here");
        assert!(error.contains("--q"), "unhelpful message: {error}");

        let error = parse_args(&argv(&[
            "phonons",
            "c.xyz",
            "--cell",
            "6.0",
            "--q",
            "0.5,0,0",
            "--rigid-ion",
            "--kpts",
            "2,2,2",
        ]))
        .expect_err("a mesh cannot change a matrix with no response in it");
        assert!(error.contains("k-mesh"), "unhelpful message: {error}");
    }

    /// `--kpts` takes whole numbers, and says so when it does not get them.
    ///
    /// It parsed floats and cast them: `-1` saturated to `0` and produced an error message about
    /// a zero division the caller never typed, and `2.9` became `2` in silence.
    #[test]
    fn a_k_mesh_is_read_as_whole_numbers() {
        let good = parse_args(&argv(&[
            "energy", "c.xyz", "--cell", "6.0", "--kpts", "2,3,4",
        ]))
        .expect("three whole divisions");
        assert_eq!(good.kpts, Some([2, 3, 4]));

        for bad in ["-1,1,1", "2.9,1,1", "a,1,1"] {
            let error = parse_args(&argv(&["energy", "c.xyz", "--cell", "6.0", "--kpts", bad]))
                .err()
                .unwrap_or_else(|| panic!("`--kpts {bad}` was accepted"));
            assert!(
                error.contains("--kpts"),
                "`--kpts {bad}` was refused, but the message does not name the flag: {error}"
            );
        }
    }

    /// A k-mesh handed to a path that cannot sample one is refused, not dropped.
    ///
    /// The periodic optimizer, the periodic Hessian, the Γ-point phonons and the
    /// divide-and-conquer SCF are all Γ-point paths. Each of them parsed `--kpts` and then never
    /// looked at it, so `optimize crystal.xyz --cell 6 --kpts 4,4,4` returned a Γ-point
    /// relaxation and said nothing — which is indistinguishable, in the output, from a relaxation
    /// that really had been sampled on a 4×4×4 mesh.
    #[test]
    fn a_k_mesh_is_refused_where_it_would_be_ignored() {
        let refused: [(Vec<&str>, &str); 4] = [
            (
                vec!["optimize", "c.xyz", "--cell", "6.0", "--kpts", "4,4,4"],
                "Gamma-point path",
            ),
            (
                vec!["hessian", "c.xyz", "--cell", "6.0", "--kpts", "4,4,4"],
                "Gamma-point path",
            ),
            (
                vec!["phonons", "c.xyz", "--cell", "6.0", "--kpts", "4,4,4"],
                "--q",
            ),
            (
                vec![
                    "energy", "c.xyz", "--cell", "6.0", "--kpts", "4,4,4", "--dc", "5.0",
                ],
                "partitioned SCF",
            ),
        ];
        for (pieces, expected) in refused {
            let joined = pieces.join(" ");
            let error = parse_args(&argv(&pieces))
                .err()
                .unwrap_or_else(|| panic!("`{joined}` was accepted and would drop --kpts"));
            assert!(
                error.contains(expected),
                "`{joined}` was refused, but the message does not say why: {error}"
            );
        }

        // And the combination that *does* sample the mesh is still accepted.
        let good = parse_args(&argv(&[
            "phonons", "c.xyz", "--cell", "6.0", "--q", "0.25,0,0", "--kpts", "2,2,2",
        ]))
        .expect("DFPT at a wavevector is the path that sums the response over a mesh");
        assert_eq!(good.kpts, Some([2, 2, 2]));
    }

    /// The v0.2.2 response commands are reachable from here, and need the cell they measure.
    ///
    /// `born`, `dielectric` and `phonon-bands` are all periodic responses. Each was added to the
    /// Rust API and the Python layer first, and this asserts the third layer arrived with them —
    /// the omission this release exists partly to stop repeating.
    #[test]
    fn the_response_commands_are_reachable_and_need_a_cell() {
        for command in ["born", "dielectric", "phonon-bands"] {
            let error = parse_args(&argv(&[command, "c.xyz"]))
                .err()
                .unwrap_or_else(|| panic!("`{command}` was accepted without a cell to respond in"));
            assert!(
                !error.contains("unknown command"),
                "`{command}` is not wired into the command list at all: {error}"
            );
            assert!(
                error.contains("--cell"),
                "`{command}` was refused, but not for the missing cell: {error}"
            );
        }

        // And each one is accepted with what it needs, so the refusals above are about the cell
        // rather than about the command being unreachable by any route.
        assert!(parse_args(&argv(&["born", "c.xyz", "--cell", "6.0"])).is_ok());
        assert!(parse_args(&argv(&["dielectric", "c.xyz", "--cell", "6.0", "--static"])).is_ok());
        let good = parse_args(&argv(&[
            "phonon-bands",
            "c.xyz",
            "--cell",
            "6.0",
            "--supercell",
            "2,1,1",
        ]))
        .expect("a supercell is all phonon-bands needs beyond a cell");
        assert_eq!(good.supercell, Some([2, 1, 1]));
    }

    /// `--supercell` and `--static` are refused where they would be dropped, like every flag here.
    ///
    /// The pattern the rest of this parser already follows: a flag a command will never read is
    /// an error, because honouring it and ignoring it produce output that looks the same.
    #[test]
    fn the_new_flags_are_refused_where_they_would_be_ignored() {
        let refused: [(Vec<&str>, &str); 5] = [
            (
                vec!["energy", "c.xyz", "--cell", "6.0", "--supercell", "2,1,1"],
                "would ignore it",
            ),
            (
                vec!["phonons", "c.xyz", "--cell", "6.0", "--supercell", "2,1,1"],
                "would ignore it",
            ),
            (
                vec!["energy", "c.xyz", "--cell", "6.0", "--static"],
                "would ignore it",
            ),
            // A supercell is what `phonon-bands` interpolates from; without one there is nothing
            // to transform, so this is a missing argument rather than an ignored flag.
            (
                vec!["phonon-bands", "c.xyz", "--cell", "6.0"],
                "--supercell",
            ),
            // Γ-point paths, for the same reason `hessian` is one.
            (
                vec!["born", "c.xyz", "--cell", "6.0", "--kpts", "2,2,2"],
                "Gamma-point path",
            ),
        ];
        for (pieces, expected) in refused {
            let joined = pieces.join(" ");
            let error = parse_args(&argv(&pieces))
                .err()
                .unwrap_or_else(|| panic!("`{joined}` was accepted and would drop a flag"));
            assert!(
                error.contains(expected),
                "`{joined}` was refused, but the message does not say why: {error}"
            );
        }

        // Whole numbers, for the reason `--kpts` learned: a cast turns `2.9` into two copies in
        // silence and saturates `-1` to a zero the caller never typed.
        for bad in ["-1,1,1", "2.9,1,1", "a,1,1", "2,1"] {
            let error = parse_args(&argv(&[
                "phonon-bands",
                "c.xyz",
                "--cell",
                "6.0",
                "--supercell",
                bad,
            ]))
            .err()
            .unwrap_or_else(|| panic!("`--supercell {bad}` was accepted"));
            assert!(
                error.contains("--supercell"),
                "`--supercell {bad}` was refused without naming the flag: {error}"
            );
        }
    }

    /// `finite-field` and `berry` are reachable, and `--field` reaches the right one.
    ///
    /// `--field` is stored negated, because `Pm3Options::field` is defined with the sign that
    /// makes `E = E₀ + μ·F` hold for the molecular `−𝓔·r` coupling. `run_finite_field` takes the
    /// field itself. Handing the stored value straight through inverts the response and nothing
    /// says so — the enthalpy still converges, the polarization still looks reasonable, and the
    /// sign of `∂P/∂𝓔` is backwards. This pins the storage convention so the dispatch that undoes
    /// it cannot drift away from it.
    #[test]
    fn the_field_commands_are_reachable_and_the_field_sign_is_pinned() {
        for command in ["berry", "finite-field"] {
            let error = parse_args(&argv(&[command, "c.xyz"]))
                .err()
                .unwrap_or_else(|| panic!("`{command}` was accepted without a cell"));
            assert!(
                !error.contains("unknown command"),
                "`{command}` is not wired into the command list: {error}"
            );
        }

        // `berry` needs only a cell.
        let berry = parse_args(&argv(&[
            "berry",
            "c.xyz",
            "--cell",
            "8.0",
            "--strings",
            "8",
        ]))
        .expect("a cell and a string length are all `berry` needs");
        assert_eq!(berry.strings, 8);

        // `finite-field` needs both a field and a mesh, and says which is missing.
        let error = parse_args(&argv(&["finite-field", "c.xyz", "--cell", "8.0"]))
            .err()
            .unwrap_or_else(|| panic!("accepted with no field to apply"));
        assert!(error.contains("--field"), "{error}");
        let error = parse_args(&argv(&[
            "finite-field",
            "c.xyz",
            "--cell",
            "8.0",
            "--field",
            "0.001,0,0",
        ]))
        .err()
        .unwrap_or_else(|| panic!("accepted with no mesh to build strings on"));
        assert!(error.contains("--kpts"), "{error}");

        // The storage convention itself: positive volts per Angstrom in, negative out.
        let cli = parse_args(&argv(&[
            "finite-field",
            "c.xyz",
            "--cell",
            "8.0",
            "--field",
            "1.0,0,0",
            "--kpts",
            "6,1,1",
        ]))
        .expect("a field and a mesh are what this command needs");
        let stored = cli.field.expect("--field was given");
        assert!(
            stored.x < 0.0,
            "`--field 1.0,0,0` stored {stored:?}. This is stored negated on purpose; if that \
             changes, the negation in the `finite-field` dispatch must go with it, or the \
             polarization response silently inverts."
        );
        assert!(
            (stored.x.abs() - crate::constants::BOHR_TO_ANGSTROM).abs() < 1.0e-12,
            "the magnitude conversion moved: {stored:?}"
        );

        // A field on any *other* periodic command is a redirection, not a silent drop.
        let error = parse_args(&argv(&[
            "energy",
            "c.xyz",
            "--cell",
            "8.0",
            "--field",
            "0.001,0,0",
        ]))
        .err()
        .unwrap_or_else(|| panic!("a periodic energy accepted a field it cannot apply"));
        assert!(error.contains("finite-field"), "{error}");
    }

    /// `--lo-to` is refused where it would be dropped, and needs a limit to be taken along.
    #[test]
    fn the_lo_to_direction_needs_a_wavevector() {
        let good = parse_args(&argv(&[
            "phonons", "c.xyz", "--cell", "8.0", "--q", "0,0,0", "--lo-to", "1,0,0",
        ]))
        .expect("the `q -> 0` limit along a direction is exactly what this is for");
        assert_eq!(good.lo_to, Some([1.0, 0.0, 0.0]));

        // Without a `--q` there is no limit for the direction to belong to.
        let error = parse_args(&argv(&[
            "phonons", "c.xyz", "--cell", "8.0", "--lo-to", "1,0,0",
        ]))
        .err()
        .unwrap_or_else(|| panic!("accepted a direction with no limit to take"));
        assert!(error.contains("--q"), "{error}");

        // And a command with no dynamical matrix would ignore it.
        let error = parse_args(&argv(&[
            "energy", "c.xyz", "--cell", "8.0", "--lo-to", "1,0,0",
        ]))
        .err()
        .unwrap_or_else(|| panic!("`energy` accepted a term it has nothing to add to"));
        assert!(error.contains("would ignore it"), "{error}");
    }

    /// The SCF reference is selectable from here.
    ///
    /// Every other layer exposed it and this one did not, which left the CLI permanently on
    /// `Auto`. That is not a missing convenience: `Auto` is RHF for a closed shell, so a
    /// broken-symmetry UHF singlet — the case the option exists for — could not be asked for at
    /// all, and neither could the "RHF requested for an open-shell system" guard.
    #[test]
    fn the_scf_reference_is_selectable() {
        let default = parse_args(&argv(&["energy", "m.xyz"])).unwrap();
        assert_eq!(default.reference, crate::Reference::Auto);

        for (text, want) in [
            ("auto", crate::Reference::Auto),
            ("rhf", crate::Reference::Rhf),
            ("uhf", crate::Reference::Uhf),
            ("UHF", crate::Reference::Uhf),
        ] {
            let cli = parse_args(&argv(&["energy", "m.xyz", "--reference", text]))
                .unwrap_or_else(|e| panic!("--reference {text} was refused: {e}"));
            assert_eq!(cli.reference, want, "--reference {text}");
        }

        let error = parse_args(&argv(&["energy", "m.xyz", "--reference", "mp2"]))
            .expect_err("an unknown reference is a typo, not a silent Auto");
        assert!(
            error.contains("auto, rhf, uhf"),
            "unhelpful message: {error}"
        );
    }

    /// `^MO energies \(eV\):\s*\[(.*)\]\s*$` without pulling in a regex crate: the label, then
    /// whitespace, then a bracketed body.
    fn regex_lite_mo(line: &str) -> Option<&str> {
        let rest = line.strip_prefix("MO energies (eV):")?;
        let rest = rest.trim_start_matches([' ', '\t']);
        let rest = rest.strip_prefix('[')?;
        rest.strip_suffix(']')
    }
}
