# LSP diagnostics

`cross-lang-verifier` ships a small Language Server Protocol adapter:

```bash
cross-lang-verify-lsp --stdio --manifest units_manifest.json
```

Editors pass the same manifest used by `cross-lang-verify`, the GitHub Action,
and the pre-commit hook, usually through `initializationOptions.manifest`.
Diagnostics are published only for manifest units that declare a physical
`source_file` or `target_file`; the server does not invent file locations and it
does not claim to parse unsaved buffers.

By default the server runs in deterministic `--no-confirm` mode, so symbolic
witnesses are `CANDIDATE` warnings. If `initializationOptions.confirm` or
`--confirm` is enabled, confirmed real-compiler divergences are surfaced as
errors. Clean files receive an empty diagnostic set so editors clear stale
findings.
