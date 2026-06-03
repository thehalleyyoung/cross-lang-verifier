// Thin VS Code extension for cross-lang-verifier.
//
// The extension is intentionally only an editor surface: all analysis runs in
// the repository's real Language Server Protocol adapter (`ub_oracle.lsp`).

import * as path from "path";
import * as vscode from "vscode";
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
} from "vscode-languageclient/node";

const CLIENT_ID = "cross-lang-verifier";
const CLIENT_NAME = "cross-lang-verifier";
const LSP_REVERIFY_COMMAND = "cross-lang-verifier.reverify";

let client: LanguageClient | undefined;

interface ReverifyResult {
  reverified?: number;
  reason?: string;
}

function primaryWorkspaceFolder(): vscode.WorkspaceFolder | undefined {
  const folders = vscode.workspace.workspaceFolders;
  return folders && folders.length > 0 ? folders[0] : undefined;
}

function extensionConfig(): vscode.WorkspaceConfiguration {
  return vscode.workspace.getConfiguration("crossLangVerifier");
}

function configuredManifest(folder: vscode.WorkspaceFolder): string {
  const cfg = extensionConfig();
  const manifest = cfg.get<string>("unitsManifest", "units_manifest.json");
  return path.isAbsolute(manifest)
    ? manifest
    : path.join(folder.uri.fsPath, manifest);
}

function serverOptions(folder: vscode.WorkspaceFolder): ServerOptions {
  const cfg = extensionConfig();
  const python = cfg.get<string>("pythonPath", "python");
  const moduleName = cfg.get<string>("lspModule", "ub_oracle.lsp");
  const manifest = configuredManifest(folder);
  const confirm = cfg.get<boolean>("confirm", false);
  const args = ["-m", moduleName, "--stdio", "--manifest", manifest];
  if (confirm) {
    args.push("--confirm");
  }
  return {
    command: python,
    args,
    options: { cwd: folder.uri.fsPath },
  };
}

function clientOptions(
  folder: vscode.WorkspaceFolder,
  outputChannel: vscode.OutputChannel
): LanguageClientOptions {
  const manifest = configuredManifest(folder);
  const confirm = extensionConfig().get<boolean>("confirm", false);
  return {
    documentSelector: [
      { scheme: "file", language: "c" },
      { scheme: "file", language: "cpp" },
      { scheme: "file", language: "rust" },
      { scheme: "file", language: "go" },
      { scheme: "file", language: "swift" },
      { scheme: "file", language: "zig" },
      { scheme: "file", pattern: "**/*.{c,h,cc,cpp,cxx,hpp,rs,go,swift,zig,wat,json}" },
    ],
    initializationOptions: {
      manifest,
      confirm,
    },
    outputChannel,
    workspaceFolder: folder,
  };
}

async function stopClient(): Promise<void> {
  const current = client;
  client = undefined;
  if (current) {
    await current.stop();
  }
}

async function startClient(
  outputChannel: vscode.OutputChannel
): Promise<LanguageClient | undefined> {
  const folder = primaryWorkspaceFolder();
  if (!folder) {
    outputChannel.appendLine("cross-lang-verifier: no workspace folder open.");
    return undefined;
  }
  const next = new LanguageClient(
    CLIENT_ID,
    CLIENT_NAME,
    serverOptions(folder),
    clientOptions(folder, outputChannel)
  );
  client = next;
  await next.start();
  outputChannel.appendLine(
    `cross-lang-verifier: LSP started with manifest ${configuredManifest(folder)}`
  );
  return next;
}

async function restartClient(outputChannel: vscode.OutputChannel): Promise<void> {
  await stopClient();
  await startClient(outputChannel);
}

async function reverifyOpenDocuments(
  outputChannel: vscode.OutputChannel
): Promise<void> {
  if (!client) {
    await startClient(outputChannel);
  }
  if (!client) {
    void vscode.window.showErrorMessage(
      "cross-lang-verifier: open a workspace folder first."
    );
    return;
  }

  const result = await client.sendRequest<ReverifyResult>(
    "workspace/executeCommand",
    { command: LSP_REVERIFY_COMMAND, arguments: [] }
  );
  if (typeof result.reason === "string") {
    void vscode.window.showWarningMessage(
      `cross-lang-verifier: ${result.reason}`
    );
    return;
  }
  const count = typeof result.reverified === "number" ? result.reverified : 0;
  void vscode.window.showInformationMessage(
    `cross-lang-verifier: reverified ${count} open document(s).`
  );
}

export function activate(context: vscode.ExtensionContext): void {
  const outputChannel =
    vscode.window.createOutputChannel("cross-lang-verifier");
  context.subscriptions.push(outputChannel);

  context.subscriptions.push(
    vscode.commands.registerCommand("crossLangVerifier.verify", () =>
      reverifyOpenDocuments(outputChannel)
    )
  );
  context.subscriptions.push(
    vscode.commands.registerCommand("crossLangVerifier.restartLanguageServer", () =>
      restartClient(outputChannel)
    )
  );
  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("crossLangVerifier")) {
        void restartClient(outputChannel);
      }
    })
  );

  void startClient(outputChannel);
}

export async function deactivate(): Promise<void> {
  await stopClient();
}
