const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("oneclick", {
  backendUrl: () => ipcRenderer.invoke("backend-url"),
  pickConfig: () => ipcRenderer.invoke("pick-file"),
  pickDirectory: () => ipcRenderer.invoke("pick-directory"),
});

