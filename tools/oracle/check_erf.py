# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare pm3-rs's error functions against `scipy.special`.

    cargo run --release --example erf_probe | python tools/oracle/check_erf.py

Reads `x erf erfc erfcx` rows on stdin and reports the worst relative disagreement for each
function. scipy is an optional development dependency rather than a runtime one, which is why
this is a script instead of a unit test; the Rust suite pins the same functions against their
own identities, the agreement of their two independent branches, and a few reference values.

This is the comparison that found the original `erfc` defect: forming `erfc` as `1 - erf`
cancels away roughly `-log10(erfc(x))` digits, so at x = 3.9 it was returning only 8 correct
digits (1.2e-8 relative). The identity tests could not see that, because `erf + erfc = 1` holds
exactly for a subtraction that has lost precision.
"""

import sys

try:
    from scipy.special import erf, erfc, erfcx
except ImportError:  # pragma: no cover
    sys.exit("this comparison needs scipy: pip install scipy")

# Far tighter than anything the Ewald sums need, and far looser than the ~1e-14 the
# implementation actually achieves, so it only fires on a real regression.
THRESHOLD = 1.0e-12

worst = {name: (0.0, None) for name in ("erf", "erfc", "erfcx")}
rows = 0
for line in sys.stdin:
    # Windows shells happily insert a BOM when the probe output is routed through a file.
    fields = line.lstrip("﻿").split()
    if len(fields) != 4:
        continue
    rows += 1
    x, got_erf, got_erfc, got_erfcx = (float(value) for value in fields)
    for name, got, reference in (
        ("erf", got_erf, erf(x)),
        ("erfc", got_erfc, erfc(x)),
        ("erfcx", got_erfcx, erfcx(x)),
    ):
        error = abs(got - reference)
        if reference != 0.0:
            error /= abs(reference)
        if error > worst[name][0]:
            worst[name] = (error, x)

if rows == 0:
    sys.exit("no rows on stdin - did the probe run?")

print(f"compared {rows} points")
status = 0
for name, (error, x) in worst.items():
    print(f"  {name:<6s} worst relative error {error:.3e} at x = {x:.4f}")
    if error > THRESHOLD:
        status = 1
        print(f"  {name}: FAILED (above the {THRESHOLD:.0e} threshold)")
sys.exit(status)
