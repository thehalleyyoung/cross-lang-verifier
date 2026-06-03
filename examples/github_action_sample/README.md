# Translation Equivalence Guard sample

This fixture is a minimal consumer repository for the published GitHub Action.
The workflow runs only when translated C/Rust files or the manifest change, emits
SARIF for code scanning, and fails on symbolic `CANDIDATE` findings in
`--no-confirm` mode so it is deterministic without local compilers.
