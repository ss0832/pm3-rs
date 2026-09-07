#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Collect the licence texts of every Rust crate that is linked into the binaries.

`_native.pyd` and the `pm3-rs` command are statically linked: the crates in the normal
dependency graph are compiled *into* them. MIT -- which is most of the graph -- requires
the copyright notice and the permission notice to travel with "copies or substantial
portions of the Software", and a static binary is exactly that. Naming the crate is not
enough; the copyright line is the thing the licence asks for, so this reads the actual
LICENSE files out of the local registry checkout rather than printing an SPDX id.

Run after changing a dependency, and commit the result:

    python tools/collect_rust_notices.py

It writes `third_party/rust-crates/NOTICE` (the index: crate, version, SPDX expression,
which files were found) and `third_party/rust-crates/LICENSES.txt` (every text, verbatim,
concatenated). `tests/attribution.rs` fails if the graph and the index disagree, so a new
dependency that skips this step is a test failure rather than a silent omission.

A crate whose licence text cannot be found is reported and listed in the NOTICE as
missing rather than quietly dropped: an incomplete notice that says so is honest, and one
that does not is worse than no notice at all.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "third_party" / "rust-crates"

# Files a crate uses to carry its terms. Ordered so the index reads predictably.
LICENSE_PATTERNS = ("LICENSE*", "LICENCE*", "COPYING*", "NOTICE*", "COPYRIGHT*")


def linked_crates() -> list[tuple[str, str, str]]:
    """(name, version, spdx) for every crate in the normal dependency graph.

    `-e normal` drops dev-dependencies, which are compiled only for the test binaries and
    never shipped. Build-dependencies and proc-macros are kept: they do not link in, but
    they generate code that does, and listing them costs nothing next to the risk of
    reasoning wrongly about which side of that line a crate falls on.
    """
    out = subprocess.run(
        ["cargo", "tree", "-e", "normal", "--all-features", "--prefix", "none",
         "-f", "{p}|{l}"],
        # `text=True` alone decodes with the locale encoding, which on a Japanese Windows
        # install is cp932. Cargo emits UTF-8, so a copyright holder called Quiñones came
        # out of the manifest with their name corrupted -- in a legal notice, which is the
        # one place a mangled name is not cosmetic. `tests/attribution.rs` caught it.
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout

    seen: dict[tuple[str, str], str] = {}
    for line in out.splitlines():
        line = line.strip().removesuffix(" (*)").strip()
        if not line or "|" not in line:
            continue
        package, _, spdx = line.partition("|")
        package = package.replace(" (proc-macro)", "").strip()
        match = re.match(r"^(\S+) v(\S+)", package)
        if not match:
            continue
        name, version = match.group(1), match.group(2)
        if name == "pm3-rs":
            continue
        seen[(name, version)] = spdx.strip() or "UNSPECIFIED"
    return sorted((n, v, s) for (n, v), s in seen.items())


def crate_metadata() -> dict[tuple[str, str], dict]:
    """`authors` and `repository` per package, for crates that ship no licence file.

    A crate can declare `license = "MIT"` in its manifest and ship no LICENSE text at all,
    which several in this graph do. The grant is still made -- the manifest field is the
    licensing statement -- but there is no copyright line in the crate to reproduce. The
    manifest's `authors` is who claims it, so that is what gets recorded, with the
    repository where the text lives upstream.
    """
    data = json.loads(
        subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--all-features"],
            # `text=True` alone decodes with the locale encoding, which on a Japanese Windows
        # install is cp932. Cargo emits UTF-8, so a copyright holder called Quiñones came
        # out of the manifest with their name corrupted -- in a legal notice, which is the
        # one place a mangled name is not cosmetic. `tests/attribution.rs` caught it.
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True,
        ).stdout
    )
    return {
        (p["name"], p["version"]): {
            "authors": p.get("authors") or [],
            "repository": p.get("repository") or "",
        }
        for p in data["packages"]
    }


MIT_CANONICAL = """\
Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""


def crate_source(name: str, version: str) -> Path | None:
    """Where cargo unpacked this crate, across however many registry indexes exist."""
    registry = Path.home() / ".cargo" / "registry" / "src"
    for index in sorted(registry.glob("*")):
        candidate = index / f"{name}-{version}"
        if candidate.is_dir():
            return candidate
    return None


def license_files(directory: Path) -> list[Path]:
    found: list[Path] = []
    for pattern in LICENSE_PATTERNS:
        for path in sorted(directory.glob(pattern)):
            if path.is_file() and path not in found:
                found.append(path)
    return found


# A phrase that appears in the body of each licence family, used to check that the files a
# crate ships actually cover the licence it declares.
FAMILY_MARKERS = {
    "MIT": "Permission is hereby granted, free of charge",
    "Apache-2.0": "Apache License",
    "BSD-2-Clause": "Redistributions of source code must retain",
    "BSD-3-Clause": "Redistributions of source code must retain",
    "Zlib": "altered source versions must be plainly marked",
    "MPL-2.0": "Mozilla Public License",
    "Unicode-3.0": "Unicode",
}


def declared_families(spdx: str) -> list[str]:
    """The licence families named in an SPDX expression, in the order they appear."""
    return [name for name in FAMILY_MARKERS if re.search(rf"\b{re.escape(name)}\b", spdx)]


def uncovered_families(spdx: str, bodies: str) -> list[str]:
    """Families the crate declares but whose text is nowhere in the files it ships.

    `faer` is the case this exists for: it declares MIT and ships four `COPYING.*` files
    that are its *upstream* attributions (Eigen, LAPACK, SuiteSparse) and no text for its
    own MIT grant. Counting files rather than checking what is in them would have called
    that covered.
    """
    lowered = bodies.lower()
    missing = []
    for family in declared_families(spdx):
        if FAMILY_MARKERS[family].lower() not in lowered:
            missing.append(family)
    # An `OR` expression is satisfied by any one of its alternatives.
    if " OR " in spdx.upper() and len(missing) < len(declared_families(spdx)):
        return []
    return missing


def main() -> int:
    crates = linked_crates()
    if not crates:
        print("cargo tree returned nothing; not overwriting the notices", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = crate_metadata()
    index: list[str] = []
    texts: list[str] = []
    missing: list[str] = []
    partial: list[str] = []

    for name, version, spdx in crates:
        source = crate_source(name, version)
        files = license_files(source) if source else []
        if not files:
            # No text in the crate, but the manifest still makes the grant. Record who
            # claims it and where the text lives, and carry the canonical permission
            # notice so the obligation is met with the information that exists.
            info = meta.get((name, version), {})
            holders = "; ".join(info.get("authors", [])) or "not stated in the manifest"
            repo = info.get("repository") or "not stated in the manifest"
            missing.append(f"{name} {version} ({spdx})")
            index.append(f"  {name:<28} {version:<12} {spdx:<40} (manifest grant only)")
            stanza = (
                "=" * 79
                + f"\n{name} {version} -- no licence file in the published crate\n"
                + f"SPDX (from the manifest): {spdx}\n"
                + "=" * 79
                + "\n\nCopyright holders, as declared in the crate manifest:\n"
                + f"  {holders}\n"
                + f"Upstream repository: {repo}\n\n"
                + "The published crate ships no licence text, so there is no copyright line\n"
                + "in it to reproduce. The manifest field above is the grant.\n"
            )
            if spdx.strip() == "MIT":
                stanza += "\nThe canonical terms it names:\n\n" + MIT_CANONICAL
            texts.append(stanza)
            continue
        bodies = []
        for path in files:
            body = path.read_text(encoding="utf-8", errors="replace").strip()
            bodies.append(body)
            texts.append(
                "=" * 79
                + f"\n{name} {version} -- {path.name}\nSPDX: {spdx}\n"
                + "=" * 79
                + f"\n\n{body}\n"
            )

        # Shipping *a* licence file is not the same as shipping the one that was declared.
        gaps = uncovered_families(spdx, "\n".join(bodies))
        note = ", ".join(f.name for f in files)
        if gaps:
            info = meta.get((name, version), {})
            holders = "; ".join(info.get("authors", [])) or "not stated in the manifest"
            repo = info.get("repository") or "not stated in the manifest"
            partial.append(f"{name} {version}: declares {spdx}, ships no {'/'.join(gaps)} text")
            note += f"  [no {'/'.join(gaps)} text]"
            stanza = (
                "=" * 79
                + f"\n{name} {version} -- declared {'/'.join(gaps)}, not shipped\n"
                + "=" * 79
                + "\n\nThe files above are this crate's own upstream attributions and do not\n"
                + f"carry its {'/'.join(gaps)} grant. The manifest declares "
                + f"`license = \"{spdx}\"`,\nwhich is the grant.\n\n"
                + "Copyright holders, as declared in the crate manifest:\n"
                + f"  {holders}\nUpstream repository: {repo}\n"
            )
            if "MIT" in gaps:
                stanza += "\nThe canonical MIT terms:\n\n" + MIT_CANONICAL
            texts.append(stanza)
        index.append(f"  {name:<28} {version:<12} {spdx:<40} {note}")

    header = f"""\
Rust dependencies statically linked into pm3-rs
===============================================

`_native.pyd` (the Python extension) and the `pm3-rs` command are statically linked, so
every crate below is compiled into the distributed binaries. Most are MIT, which requires
the copyright notice and the permission notice to be included in "all copies or
substantial portions of the Software" -- a static binary being a substantial portion.
The verbatim texts, with their copyright lines, are in LICENSES.txt beside this file.

pm3-rs itself is GPL-3.0-or-later. MIT, Apache-2.0, BSD-2-Clause, Zlib and
Apache-2.0-WITH-LLVM-exception are all one-way compatible with it in the direction used
here: this project incorporates them, not the reverse. `unicode-ident` is the one entry
whose expression is conjunctive -- `(MIT OR Apache-2.0) AND Unicode-3.0` -- so its
Unicode terms apply on top of whichever of the first pair is chosen, and its text is
included for that reason rather than as a courtesy.

Dev-dependencies are excluded: they are compiled only into the test binaries and are not
distributed. Build-dependencies and proc-macros are included -- they do not link in, but
they generate code that does.

**Generated by tools/collect_rust_notices.py. Do not edit by hand.** Re-run it after
changing a dependency; `tests/attribution.rs` fails if this file and the dependency graph
disagree.

{len(crates)} crates.

  {'CRATE':<28} {'VERSION':<12} {'SPDX':<40} FILES
"""
    notice = header + "\n".join(index) + "\n"
    if missing:
        notice += (
            "\n\nThe following publish no licence file in the crate itself. The manifest's\n"
            "`license` field is still the grant, so LICENSES.txt records the copyright\n"
            "holders it declares, the upstream repository, and the canonical terms named.\n"
            "Listed here rather than passed over, because a notice that says where it is\n"
            "thin can be completed and one that hides it cannot.\n\n"
            + "".join(f"  {m}\n" for m in missing)
        )
    if partial:
        notice += (
            "\n\nThe following ship licence files that do not cover the licence they\n"
            "declare -- typically their own upstream attributions and nothing for their\n"
            "own grant. Handled the same way, in LICENSES.txt.\n\n"
            + "".join(f"  {p}\n" for p in partial)
        )

    (OUT_DIR / "NOTICE").write_text(notice, encoding="utf-8", newline="\n")
    (OUT_DIR / "LICENSES.txt").write_text(
        "Verbatim licence texts of the Rust crates linked into pm3-rs.\n"
        "See NOTICE beside this file for the index. Generated; do not edit.\n\n"
        + "\n".join(texts),
        encoding="utf-8",
        newline="\n",
    )

    print(f"{len(crates)} crates -> {OUT_DIR.relative_to(ROOT)}/NOTICE and LICENSES.txt")
    if missing:
        print(f"  {len(missing)} ship no licence file (manifest grant recorded):", file=sys.stderr)
        for m in missing:
            print(f"    {m}", file=sys.stderr)
    if partial:
        print(f"  {len(partial)} ship files that do not cover what they declare:", file=sys.stderr)
        for p in partial:
            print(f"    {p}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
