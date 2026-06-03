# Pre-commit hook

`cross-lang-verifier` ships a `pre-commit` hook for teams that want the oracle
to run before translated code lands in a commit.

```yaml
repos:
  - repo: https://github.com/thehalleyyoung/cross-lang-verifier
    rev: v1.0.0
    hooks:
      - id: cross-lang-verify
        args: [--no-confirm, --fail-on, candidate]
```

By default the hook is fast and deterministic: it resolves staged C/Rust/Go/Zig
or manifest files to the nearest `units_manifest.json`, runs the same
`cross-lang-verify` CLI in `--no-confirm` mode, and blocks on symbolic
`CANDIDATE` findings. Use `--confirm` when local compiler re-execution is cheap
enough for your commit path.

The checked fixture in `examples/pre_commit_sample/` proves the hook on a real
consumer-style repo: a staged overflow pair blocks the commit, a safe-control
manifest passes, unrelated files no-op, and an explicit missing manifest is an
operational error rather than a silent success.
