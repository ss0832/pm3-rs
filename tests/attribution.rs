// SPDX-License-Identifier: GPL-3.0-or-later

//! **What this project owes upstream, checked rather than asserted in prose.**
//!
//! `THIRD_PARTY_NOTICES.md` and `third_party/*/NOTICE` make specific, checkable claims:
//! that no upstream source is vendored, that MOPAC's Apache-2.0 text is retained, that the
//! oracle binary never ships. Prose claims decay silently -- a file gets added, an exclude
//! list gets edited, and the notice keeps saying what used to be true. These tests hold the
//! notices to what the tree actually contains.
//!
//! The gap that prompted the file: `cargo package` carried every notice, and the **wheel**
//! carried only `LICENSE`. `_native.pyd` has the MOPAC-derived PM3 parameter tables
//! `include_str!`-ed into it, so the wheel redistributes Apache-2.0 material -- to the
//! people most likely never to see this repository. Nothing was checking the two
//! distributions agreed.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    let path = root().join(rel);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("{} is missing: {e}", path.display()))
}

/// Files git actually tracks. The distinction matters everywhere in this file: a MOPAC
/// checkout sitting in `tools/oracle/` is a developer's local tool, and the same bytes
/// tracked by git would be redistribution.
fn tracked() -> Vec<String> {
    let out = Command::new("git")
        .arg("ls-files")
        .current_dir(root())
        .output()
        .expect("git ls-files");
    assert!(out.status.success(), "git ls-files failed");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .map(str::to_owned)
        .collect()
}

/// Every upstream this project draws on has a directory saying what is owed.
#[test]
fn every_third_party_directory_states_its_terms() {
    let dir = root().join("third_party");
    let mut found = BTreeSet::new();
    for entry in fs::read_dir(&dir).expect("third_party/") {
        let entry = entry.expect("entry");
        if !entry.file_type().expect("file type").is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        assert!(
            entry.path().join("NOTICE").is_file(),
            "third_party/{name}/ has no NOTICE: an upstream with no stated terms is one \
             nobody can check"
        );
        found.insert(name);
    }
    let expected: BTreeSet<String> = ["dftd3", "h_bonds4", "mopac", "pyseqm", "rust-crates"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(
        found, expected,
        "the set of upstreams changed; THIRD_PARTY_NOTICES.md and third_party/README.md \
         name each one and must be updated in the same commit"
    );

    // The index has to actually index them.
    let readme = read("third_party/README.md");
    let notices = read("THIRD_PARTY_NOTICES.md");
    for name in &expected {
        assert!(
            readme.contains(name),
            "third_party/README.md never mentions {name}"
        );
        assert!(
            notices.contains(name),
            "THIRD_PARTY_NOTICES.md never mentions {name}"
        );
    }
}

/// MOPAC is Apache-2.0, and section 4(a) is satisfied by retaining the text, not citing it.
#[test]
fn the_mopac_license_is_the_apache_text_itself() {
    let text = read("third_party/mopac/LICENSE");
    for phrase in [
        "Apache License",
        "Version 2.0, January 2004",
        "Redistribution.",
        "You must give any other recipients of the Work or",
        "Derivative Works a copy of this License",
    ] {
        assert!(
            text.contains(phrase),
            "third_party/mopac/LICENSE does not contain {phrase:?} -- it is not the Apache-2.0 \
             text, so the retention obligation is not met by having the file"
        );
    }

    // Apache-2.0 section 4(d) only bites if upstream ships a NOTICE. MOPAC v23.2.5 does not,
    // which is why this repository's own third_party/mopac/NOTICE is written as attribution
    // rather than as a copy of one. Recorded here so the reasoning survives.
    let notice = read("third_party/mopac/NOTICE");
    assert!(
        notice.contains("Apache-2.0") && notice.contains("v23.2.5"),
        "third_party/mopac/NOTICE must name the license and the exact upstream version"
    );
    assert!(
        notice.contains("23.2.5"),
        "the MOPAC version is the provenance; a notice without it cannot be checked"
    );
}

/// The claim "no third-party source code is vendored" -- held to the tracked tree.
#[test]
fn no_upstream_source_is_actually_vendored() {
    let notices = read("THIRD_PARTY_NOTICES.md");
    assert!(
        notices.contains("No third-party source code is vendored"),
        "this test exists to hold that sentence true; if the policy changed, change the test \
         deliberately rather than deleting the sentence"
    );

    let vendored: Vec<String> = tracked()
        .into_iter()
        .filter(|p| {
            let lower = p.to_lowercase();
            lower.ends_with(".f90") || lower.ends_with(".f") || lower.ends_with(".for")
        })
        .collect();
    assert!(
        vendored.is_empty(),
        "Fortran source is tracked by git, so it ships in `cargo package`: {vendored:?}. The \
         MOPAC oracle checkout under tools/oracle/mopac/ is gitignored precisely so that \
         running it locally is not redistributing it."
    );
}

/// The MOPAC binary is a build-time tool, and `tools/**` is how that stays true.
#[test]
fn the_oracle_checkout_is_excluded_from_both_distributions() {
    let tracked_tools: Vec<String> = tracked()
        .into_iter()
        .filter(|p| p.starts_with("tools/oracle/mopac"))
        .collect();
    assert!(
        tracked_tools.is_empty(),
        "the MOPAC checkout is tracked by git: {tracked_tools:?}"
    );

    let pyproject = read("pyproject.toml");
    assert!(
        pyproject.contains("\"tools/**\""),
        "pyproject.toml no longer excludes tools/** from the wheel and sdist"
    );
}

/// The wheel must carry what `cargo package` carries. This is the gap that prompted the file.
#[test]
fn the_wheel_carries_the_same_attribution_the_crate_does() {
    let pyproject = read("pyproject.toml");
    let block = pyproject
        .split("license-files")
        .nth(1)
        .and_then(|rest| rest.split(']').next())
        .expect("pyproject.toml has no license-files entry");

    for required in ["LICENSE", "THIRD_PARTY_NOTICES.md", "third_party/*/NOTICE"] {
        assert!(
            block.contains(required),
            "pyproject.toml license-files does not cover {required:?}. The wheel is the copy \
             most users install, and `_native.pyd` has MOPAC-derived parameter tables compiled \
             into it -- shipping it without the notices is the whole problem this test exists \
             for. license-files entries land in dist-info/licenses/."
        );
    }
    assert!(
        block.contains("third_party/*/LICENSE") || block.contains("third_party/mopac/LICENSE"),
        "the Apache-2.0 text itself is not in license-files, so the wheel names MOPAC without \
         carrying the license it is under"
    );
}

/// **Every crate linked into the binaries is in the notice.**
///
/// This is the obligation the source tree's own notices do not cover. `_native.pyd` and the
/// `pm3-rs` executable are statically linked, so `faer`, `rayon`, `pyo3` and their transitive
/// closure are compiled into everything this project ships. Most are MIT, which asks for the
/// copyright notice to be included in "all copies or substantial portions of the Software" --
/// and a static binary is one. Naming the crates in prose is not the point either; the
/// copyright lines are in `LICENSES.txt`.
///
/// It fails when a dependency is added and `tools/collect_rust_notices.py` is not re-run,
/// which is the only way this file goes stale.
#[test]
fn every_linked_crate_appears_in_the_dependency_notice() {
    let notice = read("third_party/rust-crates/NOTICE");
    let texts = read("third_party/rust-crates/LICENSES.txt");

    let out = Command::new("cargo")
        .args([
            "tree",
            "-e",
            "normal",
            "--all-features",
            "--prefix",
            "none",
            "-f",
            "{p}",
        ])
        .current_dir(root())
        .output()
        .expect("cargo tree");
    assert!(out.status.success(), "cargo tree failed");
    let listing = String::from_utf8_lossy(&out.stdout);

    let mut absent = Vec::new();
    let mut counted = 0usize;
    for line in listing.lines() {
        let line = line.trim().trim_end_matches("(*)").trim();
        let line = line.replace(" (proc-macro)", "");
        let mut fields = line.split_whitespace();
        let (Some(name), Some(version)) = (fields.next(), fields.next()) else {
            continue;
        };
        let Some(version) = version.strip_prefix('v') else {
            continue;
        };
        if name == "pm3-rs" {
            continue;
        }
        counted += 1;
        // The index writes them in fixed-width columns, so match on the pair rather than on
        // the exact spacing: a crate is covered if its name and version both appear on a line.
        let listed = notice
            .lines()
            .any(|l| l.split_whitespace().take(2).eq([name, version]));
        if !listed {
            absent.push(format!("{name} {version}"));
        }
    }

    assert!(
        counted > 20,
        "cargo tree returned only {counted} crates; the check is not running"
    );
    assert!(
        absent.is_empty(),
        "these crates are linked into the binaries and are not in \
         third_party/rust-crates/NOTICE: {absent:#?}\n\
         Run `python tools/collect_rust_notices.py` and commit the result."
    );
    // And the texts are actually there, not just the index. A notice listing sixty MIT crates
    // with no copyright line in it satisfies nothing.
    assert!(
        texts.len() > 100_000,
        "third_party/rust-crates/LICENSES.txt is {} bytes, which is too small to hold {counted} \
         licence texts -- it was probably truncated or regenerated with an empty registry",
        texts.len()
    );
    assert!(
        texts
            .matches("Permission is hereby granted, free of charge")
            .count()
            > 20,
        "LICENSES.txt does not carry the MIT permission notice for most of the graph"
    );
}

/// The dependency notice has to reach the wheel, like the rest of the attribution.
#[test]
fn the_dependency_licences_ship_in_the_wheel() {
    let pyproject = read("pyproject.toml");
    let block = pyproject
        .split("license-files")
        .nth(1)
        .and_then(|rest| rest.split(']').next())
        .expect("pyproject.toml has no license-files entry");
    assert!(
        block.contains("third_party/rust-crates/LICENSES.txt"),
        "the Rust dependency licence texts are not in license-files, so the wheel ships sixty \
         statically linked MIT crates with none of their copyright notices. \
         `third_party/*/NOTICE` catches the index but not the texts."
    );
}

/// Every tracked source file says what it is licensed under.
#[test]
fn every_source_file_carries_its_spdx_header() {
    let mut missing = Vec::new();
    for path in tracked() {
        let is_source = path.ends_with(".rs") || (path.ends_with(".py") && !path.contains("data"));
        if !is_source {
            continue;
        }
        let full = root().join(&path);
        let Ok(text) = fs::read_to_string(&full) else {
            continue;
        };
        // The header is in the first few lines, before any prose.
        let head: String = text.lines().take(5).collect::<Vec<_>>().join("\n");
        if !head.contains("SPDX-License-Identifier: GPL-3.0-or-later") {
            missing.push(path);
        }
    }
    assert!(
        missing.is_empty(),
        "these tracked source files have no SPDX header: {missing:#?}"
    );
}

/// No file in this tree has been through a cp932 round-trip.
///
/// Windows PowerShell reads a UTF-8 file as the system ANSI codepage unless told otherwise
/// and writes it back the same way, so `Get-Content x | ... | Set-Content x` silently bakes
/// the display error into the file: an em dash becomes two CJK ideographs, `Γ` becomes a
/// halfwidth katakana. Seventeen lines across three files were committed that way, in
/// comments explaining the physics, where a reader would read them as encoding noise and
/// not as a lost minus sign in a formula.
///
/// The tell is a character that has no business in this codebase: this project has no
/// Japanese text, so a CJK ideograph or a halfwidth katakana anywhere in it is wreckage.
#[test]
fn no_file_has_been_through_a_codepage_round_trip() {
    fn wreckage(c: char) -> bool {
        let o = c as u32;
        (0xFF61..=0xFF9F).contains(&o)  // halfwidth katakana
            || o == 0x30FB              // katakana middle dot
            || (0x3040..=0x30FA).contains(&o) // hiragana and katakana
            || (0x4E00..=0x9FFF).contains(&o) // CJK unified ideographs
    }

    let mut damaged: Vec<String> = Vec::new();
    for path in tracked() {
        let Ok(text) = fs::read_to_string(root().join(&path)) else {
            continue; // binary, or not valid UTF-8; not this test's subject
        };
        for (n, line) in text.lines().enumerate() {
            if line.chars().any(wreckage) {
                damaged.push(format!("{path}:{}", n + 1));
            }
        }
    }
    assert!(
        damaged.is_empty(),
        "these lines contain characters that only appear here as encoding damage: {damaged:#?}\n\
         Repair by reversing the round-trip (encode the line back to cp932, decode as UTF-8) \
         rather than by retyping it, and never round-trip a UTF-8 source file through \
         PowerShell's Get-Content/Set-Content without -Encoding utf8."
    );
}

/// The notices name files; the files have to exist.
#[test]
fn the_notices_do_not_point_at_files_that_are_gone() {
    // Markdown wraps paths in backticks, brackets and parentheses, sometimes several at once
    // (``[`third_party/mopac/NOTICE`](third_party/mopac/NOTICE)``), so scan for the prefix and
    // take the run of path characters after it rather than splitting on whitespace.
    let mut checked = 0;
    for source in ["THIRD_PARTY_NOTICES.md", "third_party/README.md"] {
        let text = read(source);
        for (start, _) in text.match_indices("third_party/") {
            let candidate: String = text[start..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || "/._-".contains(*c))
                .collect();
            // `third_party/<name>/NOTICE` is prose describing the convention, not a path; it
            // stops at the `<` and so never reaches the NOTICE/LICENSE suffix test below.
            if !(candidate.ends_with("NOTICE") || candidate.ends_with("LICENSE")) {
                continue;
            }
            // One path is named as where a file *would* go, not where one is: PySEQM's
            // license is deliberately absent because nothing of PySEQM is redistributed, and
            // both documents say so. Its absence is checked below, which is the stronger
            // statement -- a LICENSE appearing there without a vendoring commit is the error.
            if candidate == "third_party/pyseqm/LICENSE" {
                continue;
            }
            checked += 1;
            assert!(
                Path::new(&root().join(&candidate)).is_file(),
                "{source} points at {candidate}, which does not exist"
            );
        }
    }
    assert!(
        checked > 0,
        "the link check matched nothing; it is not testing anything"
    );
}

/// PySEQM has no LICENSE file here, on purpose, and that has to stay a stated decision.
///
/// BSD-3-Clause requires the copyright notice to be retained in *redistributions*, and
/// nothing of PySEQM's is redistributed -- only its published working equations are
/// followed. A license text sitting beside them would describe an obligation that does not
/// exist and imply a vendoring that has not happened. If that ever changes, the file has to
/// arrive in the same commit as the vendored source, which is what this checks.
#[test]
fn the_absent_pyseqm_license_stays_a_decision_rather_than_an_oversight() {
    let notice = read("third_party/pyseqm/NOTICE");
    let readme = read("third_party/README.md");
    assert!(
        notice.contains("no LICENSE file in this directory"),
        "third_party/pyseqm/NOTICE no longer explains why it has no LICENSE; without that, \
         the absence reads as something nobody got round to"
    );
    assert!(
        readme.contains("has no `LICENSE` file"),
        "third_party/README.md no longer explains pyseqm's missing LICENSE"
    );
    assert!(
        !root().join("third_party/pyseqm/LICENSE").exists(),
        "a LICENSE appeared under third_party/pyseqm/. If PySEQM source was vendored, say so \
         in the NOTICE and in THIRD_PARTY_NOTICES.md -- which currently claims that no \
         third-party source code is vendored -- and delete this test."
    );
}
