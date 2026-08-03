// SPDX-License-Identifier: GPL-3.0-or-later

//! Small dependency-free command-line frontend for the native PM3 API.
//!
//! Coordinates in XYZ input and output are Angstrom; energies and gradients printed by
//! this program use the native PM3 units (eV and eV/Bohr).  Keeping this parser in-tree
//! avoids a CLI dependency for the core library while providing the validation entry point
//! specified by the project plan.

use pm3_rs::constants::ANGSTROM_TO_BOHR;
use pm3_rs::{
    analytic_hessian, closed_form_gradient, optimize, run_pm3, vibrational_analysis, Molecule,
    OptOptions, Pm3Options, Pm3Parameters,
};
use std::env;
use std::path::Path;
use std::process::ExitCode;

#[allow(dead_code)]
fn usage() -> &'static str {
    "Usage: pm3_rs_cli <energy|gradient|charges|optimize|frequencies|hessian> file.xyz \\\n+    [--charge <q>] [--multiplicity <m>]\n\
\n\
PM3 calculations on a standard XYZ geometry (Angstrom).\n\
  energy       print total and heat-of-formation energies\n\
  gradient     print energy and Cartesian gradient (eV/Bohr)\n\
  charges      print Mulliken charges and dipole\n\
  optimize     optimize geometry and write <input>.pm3opt.xyz\n\
  frequencies  print harmonic frequencies (cm^-1)\n\
  hessian      print the Cartesian Hessian (eV/Bohr^2)\n"
}

fn cli_usage() -> &'static str {
    r#"Usage: pm3_rs_cli <energy|gradient|charges|optimize|frequencies|hessian> file.xyz [--charge <q>] [--multiplicity <m>] [--no-diis]

PM3 calculations on a standard XYZ geometry (Angstrom).
  energy       print total and heat-of-formation energies
  gradient     print energy and Cartesian gradient (eV/Bohr)
  charges      print Mulliken charges and dipole
  optimize     optimize geometry and write <input>.pm3opt.xyz
  frequencies  print harmonic frequencies (cm^-1)
  hessian      print the Cartesian Hessian (eV/Bohr^2)
"#
}

#[derive(Debug)]
struct Cli {
    command: String,
    path: String,
    charge: f64,
    multiplicity: usize,
    use_diis: bool,
    variant: pm3_rs::Variant,
}

fn parse_args() -> Result<Cli, String> {
    let mut args = env::args().skip(1);
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
    let mut variant = pm3_rs::Variant::Pm3;
    while let Some(flag) = args.next() {
        if flag == "--no-diis" {
            use_diis = false;
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
            "--method" => {
                variant = pm3_rs::Variant::parse(&value).ok_or_else(|| {
                    format!("unknown method: {value} (PM3, PM3-D3, PM3-D3H4, PM3-D3H4X)")
                })?;
            }
            _ => return Err(format!("unknown option: {flag}\n\n{}", cli_usage())),
        }
    }
    match command.as_str() {
        "energy" | "gradient" | "charges" | "optimize" | "frequencies" | "hessian" => {}
        _ => return Err(format!("unknown command: {command}\n\n{}", cli_usage())),
    }
    Ok(Cli {
        command,
        path,
        charge,
        multiplicity,
        use_diis,
        variant,
    })
}

fn write_xyz(path: &Path, molecule: &Molecule) -> std::io::Result<()> {
    let mut out = format!(
        "{}\npm3-rs optimized geometry; coordinates in Angstrom\n",
        molecule.len()
    );
    for atom in &molecule.atoms {
        let symbol = pm3_rs::z_to_symbol(atom.z).unwrap_or("X");
        let pos = atom.position / ANGSTROM_TO_BOHR;
        out.push_str(&format!(
            "{symbol:2} {:+.10} {:+.10} {:+.10}\n",
            pos.x, pos.y, pos.z
        ));
    }
    std::fs::write(path, out)
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    let molecule =
        Molecule::from_xyz_file(&cli.path, cli.charge)?.with_multiplicity(cli.multiplicity);
    let params = Pm3Parameters::standard()?;
    let options = Pm3Options {
        charge: cli.charge,
        multiplicity: cli.multiplicity,
        use_diis: cli.use_diis,
        variant: cli.variant,
        ..Pm3Options::default()
    };
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
                    pm3_rs::z_to_symbol(atom.z).unwrap_or("X"),
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

fn main() -> ExitCode {
    if matches!(
        env::args().nth(1).as_deref(),
        Some("--version") | Some("-V")
    ) {
        println!("pm3-rs {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }
    match parse_args() {
        Ok(cli) => match run(cli) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("pm3-rs: {err}");
                ExitCode::FAILURE
            }
        },
        Err(message) if message.is_empty() => {
            print!("{}", cli_usage());
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("pm3-rs: {message}");
            ExitCode::FAILURE
        }
    }
}
