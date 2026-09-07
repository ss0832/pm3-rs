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
use std::path::{Path, PathBuf};

/// Which Cartesian directions are periodic.
///
/// A cell is periodic per direction — `Cell::pbc` is a `[bool; 3]`, not a dimensionality — so a
/// chain, a slab and a crystal are all expressible, and which axes carry the lattice is the
/// caller's to say. The spellings below all mean the same thing because a reader writing "x and
/// z" should not have to remember whether this flag wants `1,0,1` or `101` or `xz`:
///
/// ```text
/// --pbc 1,0,1      --pbc 101      --pbc xz      --pbc x,z      --pbc true,false,true
/// ```
///
/// An empty selection (`--pbc 0,0,0` or `--pbc none`) is an isolated cell, which is legal — it is
/// the zero-dimensional case of the same machinery — but almost always a mistake to write with a
/// `--cell` beside it, so it is accepted and the commands that need a lattice refuse it by name
/// rather than silently producing a molecular answer.
fn parse_pbc(value: &str) -> Result<[bool; 3], String> {
    let text = value.trim().to_ascii_lowercase();
    if text == "none" {
        return Ok([false; 3]);
    }
    if text == "all" || text == "xyz" {
        return Ok([true; 3]);
    }

    // Axis letters, with or without separators: `xz`, `x,z`, `x z`.
    let letters: Vec<char> = text
        .chars()
        .filter(|c| !c.is_whitespace() && *c != ',')
        .collect();
    if !letters.is_empty() && letters.iter().all(|c| matches!(c, 'x' | 'y' | 'z')) {
        let mut flags = [false; 3];
        for c in &letters {
            flags[match c {
                'x' => 0,
                'y' => 1,
                _ => 2,
            }] = true;
        }
        return Ok(flags);
    }

    // Words, or numbers with or without separators: `true,false,true`, `1,0,1`, `101`.
    let fields: Vec<String> = if text.contains(',') || text.contains(char::is_whitespace) {
        text.split([',', ' ', '\t'])
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect()
    } else {
        text.chars().map(|c| c.to_string()).collect()
    };
    if fields.len() != 3 {
        return Err(format!(
            "--pbc takes three directions, got {} from `{value}`. Write it as `1,0,1`, `101`, \
             `xz`, or `true,false,true` — all four mean periodic along x and z",
            fields.len()
        ));
    }
    let mut flags = [false; 3];
    for (axis, field) in fields.iter().enumerate() {
        flags[axis] = match field.as_str() {
            "1" | "t" | "true" | "yes" | "on" => true,
            "0" | "f" | "false" | "no" | "off" => false,
            other => {
                return Err(format!(
                    "--pbc: `{other}` is not a direction flag. Use 1/0, true/false, or the axis \
                     letters (`--pbc xz`)"
                ))
            }
        };
    }
    Ok(flags)
}

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
  orbitals     molecular orbital energies and occupations, with HOMO and LUMO
               marked; --coefficients adds the coefficient matrix
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
  --method <name>         pm3 | pm3-d3 | pm3-d3h4 | pm3-d3h4x, each optionally +mmok
  --no-diis               disable SCF acceleration
  --cell <a[,b,c[,...]]>  cell in Angstrom: one number (cubic), three
                          (orthorhombic), or nine (lattice vectors as rows)
  --pbc <axes>            which directions are periodic; default all three.
                          `1,0,1`, `101`, `xz`, `x,z` and `true,false,true` all
                          mean the same thing. `none` is an isolated cell
  --kpts <n1,n2,n3>       Gamma-centred Monkhorst-Pack mesh; default is Gamma only
  --q <h,k,l>             phonon wavevector, in fractions of the reciprocal
                          lattice vectors; makes `phonons` a DFPT run at that
                          wavevector instead of the Gamma-point Hessian. With
                          --kpts the response is summed over that mesh, each
                          point paired with k+q
  --coefficients          with `orbitals`, print the MO coefficient matrix as
                          well as the energies, labelled by atom and orbital
  --sto                   with `molden`, write [STO] rather than [GTO]: the
                          Slater exponents PM3 actually uses, instead of the
                          even-tempered Gaussian fit to them. [GTO] is the
                          default because it is what viewers read
  --output <path>         with `molden`, where to write the file; the default is
                          <input stem>.molden beside the input
  --relax-cell            with `optimize` on a periodic cell, relax the lattice
                          vectors as well as the atoms. Default is atoms only;
                          --fixed-cell says so explicitly. A slab relaxes its two
                          in-plane vectors and leaves the vacuum alone
  --fixed-cell            the default, stated
  --max-steps <n>         optimizer iteration cap
  --force-tol <f>         convergence on the largest force, eV/Angstrom
  --stress-tol <s>        convergence on the largest stress, eV/Angstrom^3
  --pressure <p>          external pressure in eV/Angstrom^3; minimizes E + PV
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
    /// MOPAC's MMOK amide correction, from a +mmok suffix on --method. Off by default.
    mmok: bool,
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
    /// Print the MO coefficient matrix as well as the energies (`orbitals` only).
    coefficients: bool,
    /// Write `[STO]` rather than `[GTO]` in the Molden file (`molden` only).
    ///
    /// PM3 *is* a Slater basis, so `[STO]` is the honest transcription and `[GTO]` is a fitted
    /// approximation to it. `[GTO]` is nevertheless the default because it is what viewers
    /// actually read; this flag is for the ones that take `[STO]`, and for anyone who wants the
    /// exponents the model really uses rather than a 14-primitive fit to them.
    sto: bool,
    /// Where the Molden file goes; `None` puts it beside the input as `<stem>.molden`.
    output: Option<String>,
    /// `optimize` on a periodic cell: whether the lattice vectors move too.
    ///
    /// `None` means the caller did not say, and the default is atoms-only — which is what this
    /// path has always done, and is kept so that a command that worked keeps meaning the same
    /// thing. Whichever applies is printed, because "optimized" without saying what moved is the
    /// kind of output someone reads once and misremembers.
    relax_cell: Option<bool>,
    max_steps: Option<usize>,
    force_tol: Option<f64>,
    stress_tol: Option<f64>,
    pressure: Option<f64>,
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
    let mut mmok = false;
    let mut cell = None;
    let mut pbc = None;
    let mut kpts = None;
    let mut dc_buffer = None;
    let mut dc_core = 3.2;
    let mut field = None;
    let mut q = None;
    let mut rigid_ion = false;
    let mut coefficients = false;
    let mut sto = false;
    let mut output: Option<String> = None;
    // `--dc-core` has a default, so its value cannot say whether the caller set it.
    let mut cli_saw_dc_core = false;
    let mut relax_cell: Option<bool> = None;
    let mut max_steps: Option<usize> = None;
    let mut force_tol: Option<f64> = None;
    let mut stress_tol: Option<f64> = None;
    let mut pressure: Option<f64> = None;
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
        if flag == "--coefficients" {
            coefficients = true;
            continue;
        }
        if flag == "--relax-cell" {
            relax_cell = Some(true);
            continue;
        }
        if flag == "--fixed-cell" {
            relax_cell = Some(false);
            continue;
        }
        if flag == "--static" {
            static_dielectric = true;
            continue;
        }
        if flag == "--sto" {
            sto = true;
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
                // A `+mmok` suffix switches on MOPAC's molecular-mechanics amide correction,
                // the same spelling the Python `method=` string takes. It is off by default
                // here and on by default in MOPAC, which is the difference `--method
                // pm3+mmok` exists to close when reproducing a MOPAC run.
                let lower = value.to_ascii_lowercase().replace([' ', '_'], "-");
                let base = match lower.strip_suffix("+mmok") {
                    Some(rest) => {
                        mmok = true;
                        rest.to_owned()
                    }
                    None => lower,
                };
                variant = crate::Variant::parse(&base).ok_or_else(|| {
                    format!(
                        "unknown method: {value} (PM3, PM3-D3, PM3-D3H4, PM3-D3H4X, each \
                         optionally with a +MMOK suffix)"
                    )
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
            "--pbc" => pbc = Some(parse_pbc(&value)?),
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
            "--output" => {
                if value.is_empty() {
                    return Err("--output needs a path".to_owned());
                }
                output = Some(value.clone());
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
            "--max-steps" => {
                max_steps = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --max-steps: {value} (a whole number)"))?,
                )
            }
            "--force-tol" => {
                force_tol = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --force-tol: {value} (eV/Angstrom)"))?,
                )
            }
            "--stress-tol" => {
                stress_tol = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --stress-tol: {value} (eV/Angstrom^3)"))?,
                )
            }
            "--pressure" => {
                pressure = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid --pressure: {value} (eV/Angstrom^3)"))?,
                )
            }
            "--dc-core" => {
                cli_saw_dc_core = true;
                dc_core = value
                    .parse()
                    .map_err(|_| format!("invalid divide-and-conquer core radius: {value}"))?
            }
            _ => return Err(format!("unknown option: {flag}\n\n{}", cli_usage())),
        }
    }
    match command.as_str() {
        "energy" | "gradient" | "charges" | "optimize" | "frequencies" | "hessian" | "stress"
        | "phonons" | "bands" | "molden" | "ir" | "orbitals" | "phonon-bands" | "born"
        | "dielectric" | "berry" | "finite-field" => {}
        _ => return Err(format!("unknown command: {command}\n\n{}", cli_usage())),
    }
    // `--dc` covers the energy, the charges and the geometry — the paths a partitioned density
    // actually has. It used to be accepted with *every* command and then answered a single point
    // regardless: `optimize big.xyz --dc 4.8` printed an energy, wrote no `.pm3opt.xyz`,
    // optimized nothing, and exited 0. That is the failure mode the rest of this parser exists to
    // prevent — a flag that silently replaces the command — so the ones it cannot serve are
    // refused by name, and `optimize` is no longer one of them.
    if dc_buffer.is_some() && !matches!(command.as_str(), "energy" | "charges" | "optimize") {
        return Err(format!(
            "--dc has no partitioned {command}: the second derivatives a Hessian, frequencies or \
             an infrared spectrum need are not defined by the current partitioning, and the \
             wavefunction a Molden file draws is assembled per subsystem rather than globally. \
             It would print an energy and silently do none of what {command} was asked for. Drop \
             --dc, or use `energy`, `charges` or `optimize`"
        ));
    }
    if dc_buffer.is_none() && cli_saw_dc_core {
        return Err(
            "--dc-core sets the core radius of a divide-and-conquer partition, and there is no \
             partition without --dc"
                .to_owned(),
        );
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
    // `--sto` and `--output` both name something only the Molden writer has: a basis-set section
    // to choose the form of, and a single file to put somewhere. Every other command prints to
    // stdout, so `--output` on one would look honoured and write nothing.
    if sto && command != "molden" {
        return Err(format!(
            "--sto chooses the basis-set section of a Molden file and only `molden` writes one; \
             {command} would ignore it"
        ));
    }
    if output.is_some() && command != "molden" {
        return Err(format!(
            "--output names where a Molden file goes and only `molden` writes a file to name; \
             {command} prints to stdout, so redirect it with `>` instead"
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
        mmok,
        cell,
        pbc,
        kpts,
        q,
        rigid_ion,
        coefficients,
        sto,
        output,
        relax_cell,
        max_steps,
        force_tol,
        stress_tol,
        pressure,
        supercell,
        lo_to,
        strings,
        static_dielectric,
        dc_buffer,
        dc_core,
        field,
    })
}

/// Write an XYZ, carrying the cell in the comment line when there is one.
///
/// The `Lattice="..."` key is the extended-XYZ convention: nine numbers, the three lattice
/// vectors as rows, in Ångström, which is the same order `--cell` accepts. Without it a relaxed
/// periodic structure came back as a bare list of atoms and the cell it was relaxed *in* was
/// gone — harmless while the CLI could only hold the cell fixed, and silent data loss the moment
/// `--relax-cell` exists, since the whole answer is then in the vectors that were dropped.
fn write_xyz(path: &Path, molecule: &Molecule) -> std::io::Result<()> {
    let comment = match molecule.cell {
        Some(cell) => {
            let rows = cell.to_rows();
            let numbers: Vec<String> = rows
                .iter()
                .flat_map(|row| row.iter())
                .map(|v| format!("{:.10}", v / ANGSTROM_TO_BOHR))
                .collect();
            format!(
                "Lattice=\"{}\" pbc=\"{} {} {}\" pm3-rs optimized geometry; Angstrom",
                numbers.join(" "),
                if cell.pbc[0] { "T" } else { "F" },
                if cell.pbc[1] { "T" } else { "F" },
                if cell.pbc[2] { "T" } else { "F" },
            )
        }
        None => "pm3-rs optimized geometry; coordinates in Angstrom".to_string(),
    };
    let mut out = format!("{}\n{comment}\n", molecule.len());
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
        // The command is looked at *before* the single point runs. It used to be looked at after,
        // which is how `optimize --dc` came to print an energy and do nothing else.
        if cli.command == "optimize" {
            if molecule.cell.is_some() {
                return Err(
                    "a partitioned geometry optimization is molecular so far: the periodic \
                     partitioned gradient exists but is not wired to the optimizer. Drop --cell, \
                     or drop --dc"
                        .into(),
                );
            }
            let defaults = crate::OptOptions::default();
            let opt = crate::OptOptions {
                max_iter: cli.max_steps.unwrap_or(defaults.max_iter),
                gtol: cli
                    .force_tol
                    .map(crate::constants::force_tol_to_au)
                    .unwrap_or(defaults.gtol),
                ..defaults
            };
            let result = crate::optimizer::optimize_dc(molecule, params, options, &dc, &opt)?;
            let input = Path::new(&cli.path);
            let stem = input
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("optimized");
            let output = input.with_file_name(format!("{stem}.pm3opt.xyz"));
            write_xyz(&output, &result.molecule)?;
            println!(
                "Converged: {} after {} optimization steps (divide and conquer)",
                result.converged, result.iterations
            );
            println!("Final energy: {:.12} eV", result.scf.total_ev);
            println!(
                "Heat of formation: {:.8} kcal/mol",
                result.scf.heat_of_formation_kcal
            );
            println!("Subsystems: {}", result.scf.n_subsystems);
            // The buffer is the knob that matters and the gradient inherits its truncation
            // rather than its square, so the number to widen is named beside the answer.
            println!(
                "Buffer radius: {:.3} Angstrom -- widen it and re-run before believing this \
                 geometry",
                buffer
            );
            println!("Optimized geometry: {}", output.display());
            return Ok(());
        }
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
            let defaults = crate::PeriodicOptOptions::default();
            // Atoms only unless asked, which is what this path has always done. Changing the
            // default would change the answer of a command that already worked.
            let variable = cli.relax_cell.unwrap_or(false);
            let opt = crate::PeriodicOptOptions {
                max_iter: cli.max_steps.unwrap_or(defaults.max_iter),
                // eV/Angstrom in, eV/Bohr inside -- see the helpers, which exist because
                // writing these three lines by hand has gone wrong every time it was tried.
                gtol: cli
                    .force_tol
                    .map(crate::constants::force_tol_to_au)
                    .unwrap_or(defaults.gtol),
                stress_tol: cli
                    .stress_tol
                    .map(crate::constants::stress_tol_to_au)
                    .unwrap_or(defaults.stress_tol),
                pressure: cli
                    .pressure
                    .map(crate::constants::stress_tol_to_au)
                    .unwrap_or(defaults.pressure),
                cell: if variable {
                    crate::CellRelaxation::Variable
                } else {
                    crate::CellRelaxation::Fixed
                },
                ..defaults
            };
            let result = crate::relax(molecule, params, options, &periodic, &opt)?;
            let out = Path::new(&cli.path).with_extension("pm3opt.xyz");
            write_xyz(&out, &result.molecule)?;
            println!(
                "Relaxed:               {}",
                if variable {
                    "atoms and lattice vectors (--relax-cell)"
                } else {
                    "atoms only; the cell was held fixed (--relax-cell to relax it)"
                }
            );
            println!("Converged:             {}", result.converged);
            println!("Iterations:            {}", result.iterations);
            println!(
                "Total energy per cell: {:.12} eV",
                result.gradient.energy_ev
            );
            println!(
                "Heat of formation:     {:.8} kcal/mol",
                result.gradient.scf.heat_of_formation_kcal
            );
            println!(
                "Largest force:         {:.8} eV/Angstrom",
                result.gradient.max_gradient / ANGSTROM_TO_BOHR
            );
            if let Some(stress) = result.gradient.stress {
                let worst = stress
                    .col
                    .iter()
                    .flat_map(|c| c.to_array())
                    .map(f64::abs)
                    .fold(0.0_f64, f64::max);
                println!(
                    "Largest stress:        {:.8} eV/Angstrom^3",
                    worst / ANGSTROM_TO_BOHR.powi(3)
                );
            }
            if let Some(cell) = result.molecule.cell {
                println!("Relaxed cell (Angstrom, lattice vectors as rows):");
                for row in cell.to_rows() {
                    println!(
                        "  {:>14.8} {:>14.8} {:>14.8}",
                        row[0] / ANGSTROM_TO_BOHR,
                        row[1] / ANGSTROM_TO_BOHR,
                        row[2] / ANGSTROM_TO_BOHR
                    );
                }
            }
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
        mmok: cli.mmok,
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
        "orbitals" => {
            let result = run_pm3(&molecule, &params, &options)?;
            let basis = crate::basis::Basis::build(&molecule, &params)?;
            let restricted = result.mo_coeff_beta.is_none();

            // AO row labels, so a coefficient is readable without reconstructing the basis
            // ordering from the element table.
            let labels: Vec<String> = basis
                .aos
                .iter()
                .map(|ao| {
                    let symbol = crate::system::z_to_symbol(ao.z).unwrap_or("X");
                    let name = match ao.orb {
                        0 => "s",
                        1 => "px",
                        2 => "py",
                        3 => "pz",
                        4 => "dx2-y2",
                        5 => "dxz",
                        6 => "dz2",
                        7 => "dyz",
                        _ => "dxy",
                    };
                    format!("{}{} {}", symbol, ao.atom + 1, name)
                })
                .collect();

            let print_set = |spin: &str,
                             energies: &[f64],
                             coeff: &crate::linalg::Matrix,
                             n_occ: usize,
                             occupancy: f64| {
                println!();
                println!("{spin} orbitals ({} occupied of {})", n_occ, energies.len());
                println!(
                    "{:>5} {:>8} {:>16} {:>16}   ",
                    "index", "occ", "energy (eV)", "energy (Hartree)"
                );
                for (index, energy) in energies.iter().enumerate() {
                    let occupation = if index < n_occ { occupancy } else { 0.0 };
                    // The frontier is what a reader is looking for, so it is marked rather
                    // than left to be counted off against the occupation column.
                    let mark = if index + 1 == n_occ {
                        "  <- HOMO"
                    } else if index == n_occ {
                        "  <- LUMO"
                    } else {
                        ""
                    };
                    println!(
                        "{:>5} {occupation:>8.4} {energy:>16.8} {:>16.10}{mark}",
                        index + 1,
                        energy * crate::constants::EV_TO_HARTREE
                    );
                }
                if cli.coefficients {
                    println!();
                    println!("{spin} coefficients (rows = atomic orbitals, columns = MOs)");
                    print!("{:>12}", "");
                    for mo in 0..coeff.cols {
                        print!(" {:>11}", format!("MO {}", mo + 1));
                    }
                    println!();
                    for (row, label) in labels.iter().enumerate() {
                        print!("{label:>12}");
                        for mo in 0..coeff.cols {
                            print!(" {:>11.6}", coeff[(row, mo)]);
                        }
                        println!();
                    }
                }
            };

            print_set(
                if restricted { "Restricted" } else { "Alpha" },
                &result.mo_energies,
                &result.mo_coeff,
                result.n_occ,
                if restricted { 2.0 } else { 1.0 },
            );
            if let (Some(coeff), Some(energies)) = (&result.mo_coeff_beta, &result.mo_energies_beta)
            {
                print_set("Beta", energies, coeff, result.n_beta, 1.0);
            }

            println!();
            // Across both spin channels: a radical's beta LUMO sits below its alpha one, so the
            // alpha spectrum alone names the wrong frontier.
            match (result.homo_ev, result.lumo_ev) {
                (Some(homo), Some(lumo)) => {
                    println!("HOMO:  {homo:.8} eV");
                    println!("LUMO:  {lumo:.8} eV");
                    println!("Gap:   {:.8} eV", lumo - homo);
                }
                _ => println!("No frontier orbital: the valence shell is empty or full."),
            }
        }
        "molden" => {
            let result = run_pm3(&molecule, &params, &options)?;
            let form = if cli.sto {
                crate::molden::MoldenBasis::Sto
            } else {
                crate::molden::MoldenBasis::Gto
            };
            let text = crate::molden::molden_string_with(&molecule, &params, &result, form)?;
            let output = match &cli.output {
                Some(path) => PathBuf::from(path),
                None => {
                    let input = Path::new(&cli.path);
                    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("pm3");
                    input.with_file_name(format!("{stem}.molden"))
                }
            };
            std::fs::write(&output, text)?;
            println!("Wrote {}", output.display());
            println!("SCF iterations:        {}", result.iterations);
            println!("Orbitals:              {}", result.mo_energies.len());
            // Which basis section was written, because the two are not interchangeable and a
            // viewer that silently ignores the one it cannot read shows an empty orbital.
            match form {
                crate::molden::MoldenBasis::Gto => println!(
                    "Basis section:         [GTO] (even-tempered fit to the Slater functions; \
                     what viewers read)"
                ),
                crate::molden::MoldenBasis::Sto => println!(
                    "Basis section:         [STO] (the Slater exponents PM3 actually uses; \
                     fewer viewers read this)"
                ),
            }
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

    /// Every command the parser accepts. Kept here rather than derived, so that adding a command
    /// without adding it to the matrix below is a compile-time-visible omission in one place.
    const COMMANDS: &[&str] = &[
        "energy",
        "gradient",
        "charges",
        "optimize",
        "frequencies",
        "hessian",
        "orbitals",
        "molden",
        "ir",
        "stress",
        "phonons",
        "bands",
        "phonon-bands",
        "born",
        "dielectric",
        "berry",
        "finite-field",
    ];

    /// Flag groups, crossed against every command.
    ///
    /// Each entry is one option as a caller would type it. The point is not that every pairing is
    /// meaningful — most are not — but that every pairing produces either a run or a sentence,
    /// and never a panic or a silent drop.
    const FLAG_SETS: &[&[&str]] = &[
        &[],
        &["--charge", "1"],
        &["--charge", "-1", "--multiplicity", "2"],
        &["--multiplicity", "3"],
        &["--reference", "uhf"],
        &["--reference", "rhf"],
        &["--method", "pm3-d3h4x"],
        &["--no-diis"],
        &["--field", "0.1,0,0"],
        &["--cell", "9.0"],
        &["--cell", "9.0", "--pbc", "xz"],
        &["--cell", "9.0", "--pbc", "1,0,0"],
        &["--cell", "9.0", "--kpts", "2,2,2"],
        &["--cell", "9.0", "--q", "0.25,0,0"],
        &["--cell", "9.0", "--q", "0,0,0", "--rigid-ion"],
        &["--cell", "9.0", "--q", "0,0,0", "--lo-to", "1,0,0"],
        &["--cell", "9.0", "--supercell", "2,1,1"],
        &["--cell", "9.0", "--static"],
        &["--cell", "9.0", "--strings", "6"],
        &["--cell", "9.0", "--field", "0.001,0,0", "--kpts", "3,1,1"],
        &["--dc", "4.5"],
        &["--dc", "4.5", "--dc-core", "3.0"],
        &["--coefficients"],
        &["--sto"],
        &["--output", "out.molden"],
        &["--sto", "--output", "out.molden"],
        &["--relax-cell"],
        &["--fixed-cell", "--max-steps", "5"],
        &["--force-tol", "0.01", "--stress-tol", "1e-4"],
        &["--pressure", "0.001"],
        &["--cell", "9.0", "--relax-cell", "--pressure", "0.001"],
    ];

    /// **Every command against every flag set: a run or a sentence, never a panic.**
    ///
    /// The parser is a hand-rolled loop with about a hundred and fifty lines of cross-flag
    /// validation, and its design rule is that a flag a command would ignore is an error rather
    /// than a silent drop. That rule was enforced by tests written one combination at a time, so
    /// it held wherever someone had thought to look — and `--dc` silently replaced *seven*
    /// commands with a single point because nobody had crossed those two.
    ///
    /// This crosses them mechanically. It cannot say whether a rejection is the *right* one, but
    /// it can say that every pairing was considered: a refusal has to name something the caller
    /// typed, so a message that mentions neither the command nor a flag is one written without a
    /// particular combination in mind.
    ///
    /// It would **not** have caught the `--dc` bug on its own, and that is worth being clear
    /// about: `optimize --dc 4.8` parsed cleanly, so this test would have counted it as accepted
    /// and moved on. Catching that needs the command to actually run and be held to what it
    /// promised, which is
    /// [`tests::every_molecular_command_produces_what_it_promises`].
    #[test]
    fn every_command_and_flag_combination_parses_or_explains_itself() {
        let mut accepted = 0usize;
        let mut refused = 0usize;
        for command in COMMANDS {
            for flags in FLAG_SETS {
                let mut pieces = vec![*command, "m.xyz"];
                pieces.extend_from_slice(flags);
                match parse_args(&argv(&pieces)) {
                    Ok(cli) => {
                        assert_eq!(&cli.command, command);
                        accepted += 1;
                    }
                    Err(message) => {
                        assert!(
                            !message.trim().is_empty(),
                            "{pieces:?} was refused with an empty message"
                        );
                        // A refusal that names nothing the caller typed is a refusal written
                        // without this combination in mind.
                        let names_something = message.contains(command)
                            || flags
                                .iter()
                                .any(|f| f.starts_with("--") && message.contains(f))
                            || message.contains("Usage:");
                        assert!(
                            names_something,
                            "{pieces:?} was refused without naming the command or any flag: \
                             {message}"
                        );
                        refused += 1;
                    }
                }
            }
        }
        // A sanity floor on the matrix itself: if a refactor made everything parse, or nothing,
        // the loop above would still pass and would be testing nothing.
        assert!(
            accepted > 100 && refused > 40,
            "the matrix accepted {accepted} and refused {refused}; one of those is degenerate"
        );
    }

    /// **Every molecular command runs, and the ones that promise a file write it.**
    ///
    /// This is the layer that catches a command being silently replaced. `optimize --dc 4.8`
    /// parsed cleanly, printed an energy, wrote no `.pm3opt.xyz` and exited zero, because the
    /// divide-and-conquer branch returned before the command was ever looked at — and a parse
    /// test cannot see that, because nothing about the parse was wrong. What sees it is holding
    /// the command to its output.
    ///
    /// Molecular only, and a three-atom molecule: this runs a real SCF per command, and the
    /// periodic commands cost seconds to minutes each. Their arms have their own tests.
    #[test]
    fn every_molecular_command_produces_what_it_promises() {
        let directory = std::env::temp_dir().join(format!(
            "pm3_cli_matrix_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&directory).expect("a scratch directory");
        let xyz = directory.join("water.xyz");
        std::fs::write(
            &xyz,
            "3\nwater\nO 0.0 0.0 0.0\nH 0.9584 0.0 0.0\nH -0.24 0.9278 0.0\n",
        )
        .expect("the fixture writes");
        let path = xyz.to_str().expect("a utf-8 path").to_owned();

        // command -> the file it documents itself as writing, if any.
        let cases: &[(&str, Option<&str>)] = &[
            ("energy", None),
            ("gradient", None),
            ("charges", None),
            ("orbitals", None),
            ("frequencies", None),
            ("hessian", None),
            ("ir", None),
            ("optimize", Some("water.pm3opt.xyz")),
            ("molden", Some("water.molden")),
        ];

        for (command, artifact) in cases {
            if let Some(name) = artifact {
                let _ = std::fs::remove_file(directory.join(name));
            }
            let cli = parse_args(&argv(&[command, &path]))
                .unwrap_or_else(|e| panic!("`{command}` did not parse: {e}"));
            run(cli).unwrap_or_else(|e| panic!("`{command}` failed to run: {e}"));
            if let Some(name) = artifact {
                let written = directory.join(name);
                assert!(
                    written.exists(),
                    "`{command}` documents itself as writing {name} and did not. That is what a \
                     command silently replaced by another looks like from outside."
                );
                let bytes = std::fs::metadata(&written).map(|m| m.len()).unwrap_or(0);
                assert!(bytes > 0, "`{command}` wrote an empty {name}");
            }
        }

        // `--coefficients` is only meaningful for `orbitals`, and has to survive the round trip.
        let cli = parse_args(&argv(&["orbitals", &path, "--coefficients"])).unwrap();
        assert!(cli.coefficients);
        run(cli).expect("orbitals with coefficients runs");

        // `--output` has to put the file where it was told, not beside the input. A flag that is
        // accepted and then ignored writes the default path and reports success, which is
        // indistinguishable from working until someone looks for the file they asked for.
        let elsewhere = directory.join("named-by-hand.molden");
        let cli = parse_args(&argv(&[
            "molden",
            &path,
            "--output",
            elsewhere.to_str().expect("a utf-8 path"),
        ]))
        .unwrap();
        run(cli).expect("molden with --output runs");
        assert!(
            elsewhere.exists(),
            "`molden --output` wrote somewhere else: the flag was accepted and ignored"
        );

        // `--sto` has to change the document, not just the message about it.
        let gto_path = directory.join("as-gto.molden");
        let sto_path = directory.join("as-sto.molden");
        for (flag_set, out) in [
            (vec!["molden", &path], &gto_path),
            (vec!["molden", &path, "--sto"], &sto_path),
        ] {
            let mut args = flag_set;
            args.push("--output");
            let out_str = out.to_str().expect("a utf-8 path").to_owned();
            args.push(&out_str);
            let cli = parse_args(&argv(&args)).unwrap();
            run(cli).expect("molden runs");
        }
        let gto = std::fs::read_to_string(&gto_path).expect("the GTO document");
        let sto = std::fs::read_to_string(&sto_path).expect("the STO document");
        assert!(
            gto.contains("[GTO]") && !gto.contains("[STO]"),
            "default is not [GTO]"
        );
        assert!(
            sto.contains("[STO]") && !sto.contains("[GTO]"),
            "--sto did not write [STO]"
        );
        // The wavefunction is the same either way -- only its basis section differs -- so the
        // orbital block has to be identical. If it is not, one of the two is describing
        // coefficients against a basis it did not write, which is the misalignment this
        // release set out to rule out.
        let orbitals_of = |text: &str| {
            text.split("[MO]")
                .nth(1)
                .map(str::to_owned)
                .expect("a [MO] section")
        };
        assert_eq!(
            orbitals_of(&gto),
            orbitals_of(&sto),
            "the [MO] blocks differ between the two basis sections; the coefficients belong to \
             one wavefunction and cannot depend on how the basis was written down"
        );

        let _ = std::fs::remove_dir_all(&directory);
    }

    /// **Every accepted command reaches an implementation.**
    ///
    /// `run` dispatches on a `match` whose fallback is `unreachable!("command was validated
    /// during parsing")`, and `run_extended`'s is a refusal naming the command. Both are correct
    /// only while the parser's whitelist and the two match arms agree, and nothing was checking
    /// that they do — a command added to the whitelist and not to the molecular arm is a panic in
    /// release, reached by a caller typing a documented command.
    ///
    /// So: parse each command, then look it up in the dispatch. Molecular commands run here
    /// because they are fast; the periodic ones are covered by their own tests, and what is
    /// asserted for them is that the parser and the periodic dispatch agree about which they are.
    #[test]
    fn every_command_reaches_an_implementation() {
        // The `--dc` branch of `run_extended`, which is checked before the cell is.
        const PARTITIONED: &[&str] = &["energy", "charges", "optimize"];
        for command in COMMANDS {
            let accepted = parse_args(&argv(&[command, "m.xyz", "--dc", "4.5"])).is_ok();
            assert_eq!(
                accepted,
                PARTITIONED.contains(command),
                "`{command} --dc` parses = {accepted}, but the partitioned dispatch {} handle it",
                if PARTITIONED.contains(command) {
                    "does"
                } else {
                    "does not"
                }
            );
        }

        // The molecular arm of `run`.
        const MOLECULAR: &[&str] = &[
            "energy",
            "gradient",
            "charges",
            "optimize",
            "molden",
            "ir",
            "orbitals",
            "frequencies",
            "hessian",
        ];
        // The arms of `run_extended`. `charges` shares `energy`'s.
        const PERIODIC: &[&str] = &[
            "energy",
            "charges",
            "gradient",
            "stress",
            "phonons",
            "frequencies",
            "bands",
            "phonon-bands",
            "born",
            "dielectric",
            "berry",
            "finite-field",
            "optimize",
            "hessian",
        ];

        for command in COMMANDS {
            let handled = MOLECULAR.contains(command) || PERIODIC.contains(command);
            assert!(
                handled,
                "`{command}` is accepted by the parser and has no arm in either dispatch, so it \
                 reaches `unreachable!` or a generic refusal"
            );
            // And a command that only the periodic dispatch handles must require a cell, or the
            // molecular arm's `unreachable!` is live.
            if !MOLECULAR.contains(command) {
                let error = parse_args(&argv(&[command, "m.xyz"]))
                    .err()
                    .unwrap_or_else(|| {
                        panic!(
                            "`{command}` has no molecular implementation but parses without a \
                             cell, so `run` would reach its `unreachable!`"
                        )
                    });
                assert!(
                    error.contains("--cell"),
                    "`{command}` needs a cell and the message does not say so: {error}"
                );
            }
        }
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
