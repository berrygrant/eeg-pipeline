const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const path = require("node:path");
const { spawn } = require("node:child_process");

const repoRoot = path.resolve(__dirname, "..", "..");
let backendProcess = null;
let backendUrl = null;

function startBackend() {
  return new Promise((resolve, reject) => {
    const python = process.env.EEG_PIPELINE_PYTHON || "python3";
    backendProcess = spawn(
      python,
      ["-m", "eeg_pipeline.oneclick.backend", "--port", "8765"],
      { cwd: repoRoot, env: { ...process.env, PYTHONUNBUFFERED: "1" } }
    );

    let settled = false;
    backendProcess.stdout.on("data", (chunk) => {
      const lines = chunk.toString().split(/\r?\n/).filter(Boolean);
      for (const line of lines) {
        try {
          const message = JSON.parse(line);
          if (message.event === "ready") {
            backendUrl = `http://${message.host}:${message.port}`;
            settled = true;
            resolve(backendUrl);
          }
        } catch {
          console.log(`[backend] ${line}`);
        }
      }
    });
    backendProcess.stderr.on("data", (chunk) => console.error(`[backend] ${chunk}`));
    backendProcess.on("exit", (code) => {
      backendProcess = null;
      if (!settled) {
        reject(new Error(`Backend exited before startup with code ${code}`));
      }
    });
  });
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1220,
    height: 820,
    minWidth: 980,
    minHeight: 680,
    title: "eeg-pipeline OneClick",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  win.loadFile(path.join(__dirname, "renderer", "index.html"));
}

app.whenReady().then(async () => {
  await startBackend();
  createWindow();
});

ipcMain.handle("backend-url", () => backendUrl);

ipcMain.handle("pick-file", async (_event, options = {}) => {
  const result = await dialog.showOpenDialog({
    properties: ["openFile"],
    filters: options.filters || [{ name: "Config", extensions: ["yaml", "yml", "json"] }],
  });
  return result.canceled ? null : result.filePaths[0];
});

ipcMain.handle("pick-directory", async () => {
  const result = await dialog.showOpenDialog({ properties: ["openDirectory"] });
  return result.canceled ? null : result.filePaths[0];
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  if (backendProcess) {
    backendProcess.kill();
  }
});

