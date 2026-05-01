const state = {
  backendUrl: null,
  currentJobId: null,
  pollTimer: null,
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  if (!state.backendUrl) {
    state.backendUrl = await window.oneclick.backendUrl();
  }
  const response = await fetch(`${state.backendUrl}${path}`, {
    headers: { "content-type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json();
  if (!data.ok) {
    throw new Error(data.error || "Request failed.");
  }
  return data;
}

function configPath() {
  return $("configPath").value.trim() || "config.yaml";
}

function stages() {
  return {
    processData: $("stageProcess").checked,
    getMetrics: $("stageMetrics").checked,
    plotFigures: $("stageFigures").checked,
  };
}

function renderSummary(summary) {
  const entries = Object.entries(summary || {});
  $("summary").innerHTML = entries
    .map(([key, value]) => `<dt>${key}</dt><dd>${value == null ? "" : String(value)}</dd>`)
    .join("");
}

function renderWarnings(warnings) {
  $("warnings").innerHTML = (warnings || []).map((item) => `<p>${item}</p>`).join("");
}

function renderRecordings(items) {
  $("recordings").innerHTML = (items || [])
    .map(
      (item) => `<tr>
        <td>${item.subject || ""}</td>
        <td>${item.session || ""}</td>
        <td>${item.task || ""}</td>
        <td>${item.run || ""}</td>
        <td>${item.behaviorKind || ""}</td>
      </tr>`
    )
    .join("");
}

function setLog(lines) {
  $("logs").textContent = (lines || []).join("\n");
  $("logs").scrollTop = $("logs").scrollHeight;
}

async function validate() {
  const data = await api("/api/config/validate", {
    method: "POST",
    body: JSON.stringify({ configPath: configPath() }),
  });
  renderSummary(data.summary);
  renderWarnings(data.warnings);
}

async function discover() {
  const data = await api("/api/recordings/discover", {
    method: "POST",
    body: JSON.stringify({ configPath: configPath() }),
  });
  $("recordingCount").textContent = `${data.count} recording${data.count === 1 ? "" : "s"} found.`;
  renderRecordings(data.recordings);
}

async function runPipeline() {
  const data = await api("/api/run", {
    method: "POST",
    body: JSON.stringify({
      configPath: configPath(),
      stages: stages(),
      erpCore: $("erpCore").checked,
      useGpu: $("useGpu").checked,
    }),
  });
  state.currentJobId = data.job.id;
  $("jobStatus").textContent = data.job.status;
  setLog(data.job.logs);
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
  }
  state.pollTimer = setInterval(pollJob, 1200);
  await pollJob();
}

async function pollJob() {
  if (!state.currentJobId) return;
  const data = await api(`/api/jobs/${state.currentJobId}`);
  $("jobStatus").textContent = data.job.status;
  setLog(data.job.logs);
  if (["succeeded", "failed"].includes(data.job.status) && state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

async function init() {
  state.backendUrl = await window.oneclick.backendUrl();
  const health = await api("/api/health");
  $("backendStatus").textContent = `Backend ready: ${health.version}`;
}

$("pickConfigBtn").addEventListener("click", async () => {
  const path = await window.oneclick.pickConfig();
  if (path) $("configPath").value = path;
});
$("validateBtn").addEventListener("click", () => validate().catch(showError));
$("discoverBtn").addEventListener("click", () => discover().catch(showError));
$("runBtn").addEventListener("click", () => runPipeline().catch(showError));

function showError(error) {
  $("warnings").innerHTML = `<p class="error">${error.message}</p>`;
}

init().catch(showError);

