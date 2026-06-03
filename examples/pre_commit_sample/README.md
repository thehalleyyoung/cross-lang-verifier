# Pre-commit fixture

This fixture is a minimal consumer repository for the `cross-lang-verify`
pre-commit hook. The hook runs in deterministic `--no-confirm` mode, blocks on
symbolic `CANDIDATE` findings, and ignores unrelated files that do not resolve
to a verifier manifest.
