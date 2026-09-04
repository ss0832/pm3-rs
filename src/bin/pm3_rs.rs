// SPDX-License-Identifier: GPL-3.0-or-later

//! The `pm3-rs` executable. Everything it does lives in [`pm3_rs::cli`], so that a `pip install`
//! can put the identical interface on the path through a console script.

fn main() -> std::process::ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    match pm3_rs::cli::main_with_args(&argv) {
        0 => std::process::ExitCode::SUCCESS,
        _ => std::process::ExitCode::FAILURE,
    }
}
