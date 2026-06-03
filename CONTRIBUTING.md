# Contributing

`cross-lang-verifier` accepts contributions that preserve the core honesty
contract: a confirmed divergence must be backed by a real witness, and unsupported
constructs must abstain loudly rather than becoming silent success.

## Fast local checks

```bash
python3 -m pip install -e .
python3 scripts/validate_launch_readiness.py
python3 -m pytest tests/test_launch_readiness.py -q
```

For oracle or target-pack changes, also run the narrow check named by the code
you touched, for example `make soundness-check`, `make c2rust-corpus-check`, or
the matching `tests/test_*.py` file. Do not replace targeted checks with a broad
claim that "all tests probably pass."

## Good first contributions

- Add a minimized manifest unit for a translated function in
  `examples/units_manifest.json` or a relevant corpus.
- Improve a docs page by replacing a prose claim with a command that reproduces
  it.
- Add a safe negative control for an existing oracle.
- Add a target-semantics-pack conformance case before adding a new target.
- Start a plugin from `examples/plugins/float_cast_overflow_oracle.py` and wire a
  positive witness plus a safe control.

## Pull request standard

Every PR should say:

- which language pair and divergence class it affects;
- **Positive witness:** which witness proves the reported case;
- **Safe negative control:** which safe control proves the oracle does not
  over-report;
- which targeted command was run; and
- whether any construct remains `UNKNOWN`, `CANDIDATE`, or `NOT-COVERED`.

Avoid vendoring third-party vulnerable source. The historical-CVE corpus uses
from-scratch weakness-class reproductions, not copied vulnerable projects.
