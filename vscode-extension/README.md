# cross-lang-verifier for VS Code

Thin editor wrapper for the repository's real `cross-lang-verify-lsp` server.
The extension starts `python -m ub_oracle.lsp --stdio`, passes the configured
translation-unit manifest, and lets the LSP publish diagnostics for opened C,
Rust, Go, Swift, Zig, WebAssembly text, and manifest files.

```bash
npm install
npm run smoke
```

`npm run smoke` compiles the TypeScript against the real VS Code API typings and
builds a `.vsix` package. The generated package is ignored by git.
