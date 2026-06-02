// Thin VS Code extension for cross-lang-verifier.
//
// It does not re-implement any analysis: it shells out to the real
// `cross-lang-verify` CLI (the same oracle the test-suite proves), parses its
// JSON, and surfaces each divergent translation unit as an in-editor
// Diagnostic. The whole point is to be a faithful, thin surface over the proven
// oracle — never a second, drifting implementation.

import * as cp from "child_process";
import * as path from "path";
import * as vscode from "vscode";

interface UnitReport {
  label: string;
  verdict: string;
  pair: string;
  detail: string;
  suppressed: boolean;
}

interface CliResult {
  summary: Record<string, number>;
  units: UnitReport[];
}

const SEVERITY: Record<string, vscode.DiagnosticSeverity> = {
  divergent: vscode.DiagnosticSeverity.Error,
  spec_gap: vscode.DiagnosticSeverity.Warning,
  abstain: vscode.DiagnosticSeverity.Information,
  equivalent: vscode.DiagnosticSeverity.Hint,
};

function runCli(
  python: string,
  manifest: string,
  confirm: boolean,
  cwd: string
): Promise<CliResult> {
  return new Promise((resolve, reject) => {
    const args = ["-m", "ub_oracle.cli", "--units", manifest, "--format", "json"];
    if (!confirm) {
      args.push("--no-confirm");
    }
    cp.execFile(
      python,
      args,
      { cwd, maxBuffer: 16 * 1024 * 1024 },
      (err, stdout, stderr) => {
        // The CLI exits non-zero when it finds a blocking divergence; that is a
        // successful run for our purposes, so parse stdout regardless and only
        // reject when there is no parseable payload.
        const text = stdout.trim();
        if (!text) {
          reject(new Error(stderr || (err ? err.message : "no output")));
          return;
        }
        try {
          resolve(JSON.parse(text) as CliResult);
        } catch (e) {
          reject(new Error(`could not parse CLI output: ${String(e)}`));
        }
      }
    );
  });
}

function publish(
  result: CliResult,
  manifestUri: vscode.Uri,
  collection: vscode.DiagnosticCollection
): number {
  const diags: vscode.Diagnostic[] = [];
  for (const u of result.units) {
    if (u.verdict === "equivalent" || u.suppressed) {
      continue;
    }
    const sev = SEVERITY[u.verdict] ?? vscode.DiagnosticSeverity.Warning;
    const d = new vscode.Diagnostic(
      new vscode.Range(0, 0, 0, 0),
      `[${u.pair}] ${u.label}: ${u.verdict} — ${u.detail}`,
      sev
    );
    d.source = "cross-lang-verifier";
    diags.push(d);
  }
  collection.set(manifestUri, diags);
  return diags.length;
}

export function activate(context: vscode.ExtensionContext): void {
  const collection = vscode.languages.createDiagnosticCollection(
    "cross-lang-verifier"
  );
  context.subscriptions.push(collection);

  const cmd = vscode.commands.registerCommand(
    "crossLangVerifier.verify",
    async () => {
      const folders = vscode.workspace.workspaceFolders;
      if (!folders || folders.length === 0) {
        void vscode.window.showErrorMessage(
          "cross-lang-verifier: open a workspace folder first."
        );
        return;
      }
      const cfg = vscode.workspace.getConfiguration("crossLangVerifier");
      const python = cfg.get<string>("pythonPath", "python");
      const manifest = cfg.get<string>("unitsManifest", "units.json");
      const confirm = cfg.get<boolean>("confirm", true);
      const cwd = folders[0].uri.fsPath;
      const manifestUri = vscode.Uri.file(path.join(cwd, manifest));

      try {
        const result = await vscode.window.withProgress(
          {
            location: vscode.ProgressLocation.Window,
            title: "cross-lang-verifier: verifying divergences…",
          },
          () => runCli(python, manifest, confirm, cwd)
        );
        const n = publish(result, manifestUri, collection);
        void vscode.window.showInformationMessage(
          n === 0
            ? "cross-lang-verifier: no divergences found."
            : `cross-lang-verifier: surfaced ${n} divergence(s).`
        );
      } catch (e) {
        void vscode.window.showErrorMessage(
          `cross-lang-verifier failed: ${String(e)}`
        );
      }
    }
  );
  context.subscriptions.push(cmd);
}

export function deactivate(): void {
  // nothing to clean up; the DiagnosticCollection is disposed via subscriptions.
}
