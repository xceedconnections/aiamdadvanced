const state = {
  token: localStorage.getItem("openamd_token") || "",
  username: localStorage.getItem("openamd_user") || "",
  liveTimer: null,
  audioObjectUrl: null,
  cdrPage: 1,
  cdrTotal: 0,
  cdrPageSize: 50,
  chartDonut: null,
  chartTrend: null,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const routePages = {
  "/": "dashboard",
  "/dashboard.php": "dashboard",
  "/livecalls.php": "live",
  "/cdr.php": "cdr",
  "/vicidialservers.php": "servers",
  "/reports.php": "reports",
  "/training.php": "training",
  "/training-history.php": "training-history",
  "/wipe.php": "wipe",
  "/audio.php": "audio",
  "/cronjob.php": "cronjob",
  "/settings.php": "settings",
};

async function api(path, options = {}) {
  const headers = options.headers || {};
  if (state.token) headers["Authorization"] = `Bearer ${state.token}`;
  if (!(options.body instanceof FormData) && options.json) {
    headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(options.json);
  }
  const res = await fetch(path, { ...options, headers });
  if (res.status === 401) {
    logout(false);
    throw new Error("Unauthorized");
  }
  if (options.raw) return res;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data.detail;
    const msg = Array.isArray(detail)
      ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
      : detail || data.message || "Request failed";
    throw new Error(msg);
  }
  return data;
}

function showApp(show) {
  $("#login-view").classList.toggle("hidden", show);
  $("#app-view").classList.toggle("hidden", !show);
  closeNav();
  if (show) {
    $("#whoami").textContent = state.username || "admin";
    // Relink any on-disk WAVs to call rows (by call id)
    api("/api/recordings/repair", { method: "POST" }).catch(() => {});
    loadHealth();
    openPage(routePages[window.location.pathname] || "dashboard");
    startLiveTimer();
  } else {
    stopLiveTimer();
    stopPlayer();
  }
}

function logout(clear = true) {
  if (clear) {
    localStorage.removeItem("openamd_token");
    localStorage.removeItem("openamd_user");
  }
  state.token = "";
  state.username = "";
  showApp(false);
}

function openNav() {
  $("#app-view").classList.add("nav-open");
}
function closeNav() {
  $("#app-view").classList.remove("nav-open");
}
$("#menu-toggle")?.addEventListener("click", () => {
  $("#app-view").classList.toggle("nav-open");
});
$("#sidebar-backdrop")?.addEventListener("click", closeNav);

$("#login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#login-error").textContent = "";
  try {
    const data = await api("/api/login", {
      method: "POST",
      json: {
        username: $("#login-user").value.trim(),
        password: $("#login-pass").value,
      },
    });
    state.token = data.access_token;
    state.username = data.username;
    localStorage.setItem("openamd_token", state.token);
    localStorage.setItem("openamd_user", state.username);
    showApp(true);
  } catch (err) {
    $("#login-error").textContent = err.message || "Login failed";
  }
});

$("#logout-btn").addEventListener("click", () => logout(true));

function openPage(page) {
  const link = $(`.nav[data-page="${page}"]`) || $(".nav[data-page='dashboard']");
  $$(".nav").forEach((item) => item.classList.remove("active"));
  link.classList.add("active");
  $$(".page").forEach((item) => item.classList.add("hidden"));
  $(`#page-${page}`).classList.remove("hidden");
  const title = link.dataset.title || link.textContent.replace(/^[^\w]+/, "").trim();
  $("#page-title").textContent = title;
  closeNav();
  if (page === "servers") loadServers();
  if (page === "reports") loadReports();
  if (page === "live") {
    fillServerFilters().then(() => loadLive());
  }
  if (page === "cdr") {
    state.cdrPage = 1;
    const sizeSel = $("#cdr-page-size");
    if (sizeSel) state.cdrPageSize = Number(sizeSel.value) || 50;
    fillServerFilters().finally(() => loadCdr());
  }
  if (page === "training") {
    fillServerFilters().then(() => loadTraining());
  }
  if (page === "training-history") {
    loadTrainingHistory();
  }
  if (page === "wipe") loadWipePage();
  if (page === "audio") loadAudioPage();
  if (page === "cronjob") loadCronPage();
  if (page === "settings") {
    loadHealth();
    loadAmdSettings();
  }
  if (page === "dashboard") loadDashboard();
}

$$(".nav").forEach((link) => {
  link.addEventListener("click", (event) => {
    if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    history.pushState({}, "", link.href);
    openPage(link.dataset.page);
  });
});

document.addEventListener("click", (event) => {
  const jump = event.target.closest("a.panel-link[data-page]");
  if (!jump) return;
  if (event.button !== 0 || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  history.pushState({}, "", jump.href);
  openPage(jump.dataset.page);
});

window.addEventListener("popstate", () => {
  openPage(routePages[window.location.pathname] || "dashboard");
});

function statusBadge(status) {
  const labels = {
    HUMAN: "LIVE HUMAN",
    MACHINE: "ANSWERING MACHINE",
    IVR: "IVR / MENU",
    SIT: "CANCELLED",
    FAX: "FAX",
    ERROR: "ERROR",
  };
  const label = labels[status] || status;
  return `<span class="badge ${status}">${label}</span>`;
}

function confClass(conf) {
  const pct = (conf || 0) * 100;
  if (pct >= 90) return "conf-hi";
  if (pct >= 70) return "conf-mid";
  return "conf-lo";
}

function actionChip(status) {
  if (status === "HUMAN") return `<span class="action-chip to-agent">→ TO AGENT</span>`;
  return `<span class="action-chip disposed">✕ DISPOSED</span>`;
}

function waveformHtml(status) {
  const heights = [8, 14, 20, 11, 18, 9, 16, 12, 19, 7, 15, 10];
  return `<span class="wave ${status || "HUMAN"}">${heights
    .map((h, i) => `<i style="height:${h}px;animation-delay:${i * 0.07}s"></i>`)
    .join("")}</span>`;
}

function sparklineSvg(values, color) {
  const w = 140;
  const h = 34;
  const nums = values.length ? values : [0, 0];
  const max = Math.max(...nums, 1);
  const min = Math.min(...nums, 0);
  const span = Math.max(max - min, 1);
  const step = w / Math.max(nums.length - 1, 1);
  const pts = nums
    .map((v, i) => {
      const x = i * step;
      const y = h - ((v - min) / span) * (h - 4) - 2;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return `<svg class="hero-spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">
    <polyline fill="none" stroke="${color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" points="${pts}" />
  </svg>`;
}

function teachButtons(r) {
  const id = r.id;
  const opts = [
    ["HUMAN", "Human"],
    ["MACHINE", "VM"],
    ["IVR", "IVR"],
    ["SIT", "SIT"],
    ["FAX", "Fax"],
  ];
  return `<div class="teach-actions" title="Correct this call disposition (logged; does not force future AMD)">
    ${opts
      .map(
        ([v, label]) =>
          `<button type="button" class="teach-btn ${r.status === v ? "active" : ""}" data-teach-id="${id}" data-teach-status="${v}">${label}</button>`
      )
      .join("")}
  </div>`;
}

function dashLiveRowHtml(r) {
  const secs = Number(r.audio_seconds || 0);
  const dur = secs > 0 ? `${secs.toFixed(1)}s` : `${r.processing_ms || 0}ms`;
  const confPct = ((r.confidence || 0) * 100).toFixed(1);
  return `<tr>
    <td class="callid-cell" title="${escapeHtml(r.call_id)}">${escapeHtml(r.call_id)}</td>
    <td>${escapeHtml(r.caller_id || "—")}</td>
    <td>${dur}</td>
    <td>${statusBadge(r.status)}</td>
    <td><span class="${confClass(r.confidence)}">${confPct}%</span></td>
    <td>${actionChip(r.status)}</td>
    <td>${waveformHtml(r.status)}</td>
    <td class="audio-cell">${audioButtons(r)}</td>
    <td>${teachButtons(r)}</td>
  </tr>`;
}

function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString();
}

function pct(n, total) {
  if (!total) return "0%";
  return `${((n / total) * 100).toFixed(1)}%`;
}

function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function recordingUrl(analysisId, kind) {
  const token = encodeURIComponent(state.token || "");
  return `/api/recordings/${analysisId}/${kind}?token=${token}`;
}

function audioButtons(r) {
  const tip = escapeHtml(r.recording_filename || r.call_id || `id-${r.id}`);
  const callAttr = encodeURIComponent(r.call_id || "");
  // Always offer play/download — backend resolves by id / callid / path
  const missing = !r.has_recording;
  return `<div class="audio-actions">
    <button type="button" class="btn-icon play" title="${missing ? "Recording may be missing — try play" : "Play " + tip}" data-play-id="${r.id}" data-call-id="${callAttr}">▶ Play</button>
    <a class="btn-icon download" title="Download ${tip}" href="${recordingUrl(r.id, "download")}">⬇</a>
    ${missing ? `<span class="audio-missing">relink…</span>` : ""}
  </div>`;
}

function gatedNote(r) {
  const raw = (r.raw_status || "").toUpperCase();
  if (raw && raw !== r.status && raw === "HUMAN") {
    return `<div class="hint" style="margin-top:.15rem;color:#fbbf24">AI: HUMAN (below min)</div>`;
  }
  return "";
}

function callRowHtml(r) {
  return `<tr>
    <td class="audio-cell">${audioButtons(r)}</td>
    <td class="callid-cell" title="${escapeHtml(r.call_id)}">${escapeHtml(r.call_id)}</td>
    <td>${fmtTime(r.created_at)}</td>
    <td>${escapeHtml(r.server_name || "—")}</td>
    <td>${escapeHtml(r.called_number || "—")}</td>
    <td>${escapeHtml(r.caller_id || "—")}</td>
    <td>${escapeHtml(r.campaign || "—")}</td>
    <td>${statusBadge(r.status)}${gatedNote(r)}</td>
    <td>${(r.confidence * 100).toFixed(1)}%</td>
    <td>${r.processing_ms}</td>
    <td>${teachButtons(r)}</td>
  </tr>`;
}

function callCardHtml(r) {
  return `<article class="call-card">
    <div class="call-card-top">
      ${statusBadge(r.status)}
      <span class="hint">${fmtTime(r.created_at)}</span>
    </div>
    <div class="call-card-grid">
      <div><div class="k">Call ID</div><div class="v">${escapeHtml(r.call_id)}</div></div>
      <div><div class="k">Server</div><div class="v">${escapeHtml(r.server_name || "—")}</div></div>
      <div><div class="k">Called</div><div class="v">${escapeHtml(r.called_number || "—")}</div></div>
      <div><div class="k">Caller ID</div><div class="v">${escapeHtml(r.caller_id || "—")}</div></div>
      <div><div class="k">Campaign</div><div class="v">${escapeHtml(r.campaign || "—")}</div></div>
      <div><div class="k">Confidence</div><div class="v">${(r.confidence * 100).toFixed(1)}% · ${r.processing_ms} ms</div></div>
    </div>
    <div class="call-card-actions">${audioButtons(r)}</div>
    <div class="call-card-actions" style="margin-top:.4rem">${teachButtons(r)}</div>
  </article>`;
}

function renderCallLists(rows, tableId, cardsId) {
  const table = $(tableId);
  const cards = $(cardsId);
  if (table) table.innerHTML = rows.map(callRowHtml).join("");
  if (cards) cards.innerHTML = rows.map(callCardHtml).join("");
}

function stopPlayer() {
  const audio = $("#player-audio");
  const bar = $("#player-bar");
  if (audio) {
    audio.pause();
    audio.removeAttribute("src");
    audio.load();
  }
  if (state.audioObjectUrl) {
    URL.revokeObjectURL(state.audioObjectUrl);
    state.audioObjectUrl = null;
  }
  bar?.classList.add("hidden");
}

window.playRecording = async (analysisId, callId) => {
  const url = recordingUrl(analysisId, "play");
  const bar = $("#player-bar");
  const audio = $("#player-audio");
  $("#player-label").textContent = callId || `call #${analysisId}`;

  if (!bar || !audio) return;
  bar.classList.remove("hidden");
  if (audio.src !== new URL(url, window.location.origin).href) {
    audio.pause();
    audio.src = url;
  }
  try {
    await audio.play();
  } catch (err) {
    alert(err.message || "Play failed — recording may be missing on disk");
  }
};

window.downloadRecording = async (analysisId, callId) => {
  // Native link preferred; keep helper for programmatic use
  window.location.href = recordingUrl(analysisId, "download");
};

document.addEventListener("click", (e) => {
  const playBtn = e.target.closest("[data-play-id]");
  if (playBtn) {
    e.preventDefault();
    const id = Number(playBtn.dataset.playId);
    const callId = decodeURIComponent(playBtn.dataset.callId || "");
    playRecording(id, callId);
    return;
  }
  const dlBtn = e.target.closest("[data-dl-id]");
  if (dlBtn) {
    e.preventDefault();
    const id = Number(dlBtn.dataset.dlId);
    const callId = decodeURIComponent(dlBtn.dataset.callId || "");
    downloadRecording(id, callId);
  }
});

$("#player-close")?.addEventListener("click", stopPlayer);

async function loadDashboard() {
  const [s, system, live, hourly] = await Promise.all([
    api("/api/dashboard/stats"),
    api("/api/system/stats"),
    api("/api/live?limit=12"),
    api("/api/reports/hourly").catch(() => []),
  ]);

  updateSidebarStatus(system);
  updateHealthPill(system);

  const total = s.total_calls_today || 0;
  const humanPct = total ? ((s.human / total) * 100).toFixed(1) : "0.0";
  const machinePct = total ? ((s.machine / total) * 100).toFixed(1) : "0.0";
  const accuracy = ((s.avg_confidence || 0) * 100).toFixed(2);
  const sparkHuman = sparkFromHourly(hourly, "HUMAN");
  const sparkMachine = sparkFromHourly(hourly, "MACHINE");
  const sparkAll = sparkFromHourly(hourly, null);
  const sparkConf = confSparkFromHourly(hourly);

  $("#hero-grid").innerHTML = `
    <div class="hero-card blue">
      <div class="hero-top"><div class="hero-label">Live Calls</div><div class="hero-ico">☎</div></div>
      <div class="hero-value">${total}</div>
      <div class="hero-sub">Detections today · ${s.active_servers}/${s.total_servers} VICIdial active</div>
      ${sparklineSvg(sparkAll, "#60a5fa")}
    </div>
    <div class="hero-card green">
      <div class="hero-top"><div class="hero-label">Human Passed</div><div class="hero-ico">☺</div></div>
      <div class="hero-value">${s.human} <span style="font-size:.85rem;color:#86efac;font-weight:600">(${humanPct}%)</span></div>
      <div class="hero-sub">Routed to agent</div>
      ${sparklineSvg(sparkHuman, "#4ade80")}
    </div>
    <div class="hero-card red">
      <div class="hero-top"><div class="hero-label">Machines Blocked</div><div class="hero-ico">⚙</div></div>
      <div class="hero-value">${s.machine} <span style="font-size:.85rem;color:#fda4af;font-weight:600">(${machinePct}%)</span></div>
      <div class="hero-sub">Answering machines / AMD</div>
      ${sparklineSvg(sparkMachine, "#fb7185")}
    </div>
    <div class="hero-card purple">
      <div class="hero-top"><div class="hero-label">Avg Confidence</div><div class="hero-ico">◎</div></div>
      <div class="hero-value">${accuracy}%</div>
      <div class="hero-sub">Avg processing ${s.avg_processing_ms} ms</div>
      ${sparklineSvg(sparkConf, "#c084fc")}
    </div>`;

  const engine = system.amd_engine || {};
  const engineName =
    engine.name || system.amd_engine_name || "OpenAMD Hybrid (Heuristic + Silero)";
  const engineModel =
    engine.model ||
    system.amd_engine_model ||
    "Rule-based acoustic features + Silero VAD ONNX";
  const engineVersion = engine.version || system.amd_engine_version || "4.0.0";
  const engineRuntime =
    engine.runtime ||
    system.amd_engine_runtime ||
    "NumPy + SoundFile + ONNX Runtime (Silero)";
  const appVersion = system.application_version || system.version || "1.0.0";

  $("#system-info-list").innerHTML = `
    <li><span class="k">AI Model</span><span class="v">${escapeHtml(engineName)}</span></li>
    <li><span class="k">Model detail</span><span class="v">${escapeHtml(engineModel)}</span></li>
    <li><span class="k">Engine</span><span class="v">v${escapeHtml(engineVersion)}</span></li>
    <li><span class="k">Runtime</span><span class="v">${escapeHtml(engineRuntime)}</span></li>
    <li><span class="k">Portal</span><span class="v">v${escapeHtml(String(appVersion))}</span></li>
    <li><span class="k">Integration</span><span class="v">Vicidial / AGI</span></li>
    <li><span class="k">Status</span><span class="v ok">${(system.status || "ok").toUpperCase()}</span></li>`;

  $("#conn-list").innerHTML = `
    <div class="conn-row"><span>Vicidial</span><span class="conn-badge">${s.active_servers ? "Connected" : "Idle"}</span></div>
    <div class="conn-row"><span>Database</span><span class="conn-badge">Connected</span></div>
    <div class="conn-row"><span>API / Portal</span><span class="conn-badge">Online</span></div>`;

  $("#recent-mini").innerHTML = (live || [])
    .slice(0, 5)
    .map(
      (r) => `<div class="rm">
        <span class="id">${escapeHtml(r.call_id)}</span>
        ${statusBadge(r.status)}
        <span class="${confClass(r.confidence)}">${((r.confidence || 0) * 100).toFixed(0)}%</span>
      </div>`
    )
    .join("") || `<div class="hint">No recent calls yet</div>`;

  $("#dash-live-body").innerHTML = (live || []).map(dashLiveRowHtml).join("");
  $("#dash-live-cards").innerHTML = (live || []).map(callCardHtml).join("");

  renderDonutChart(s);
  renderTrendChart(hourly);

  const systemCards = [
    ["Server status", (system.status || "ok").toUpperCase()],
    ["AMD engine", engineName],
    ["AMD model", engineModel],
    ["Engine version", engineVersion],
    ["Engine runtime", engineRuntime],
    ["Portal/API version", appVersion],
    ["CPU", `${system.cpu_percent}% / ${system.cpu_count} cores`],
    ["Load", `${system.load_1} / ${system.load_5} / ${system.load_15}`],
    ["RAM used", `${system.ram_used_gb} GB`],
    ["RAM available", `${system.ram_available_gb} GB`],
    ["RAM total", `${system.ram_total_gb} GB (${system.ram_percent}%)`],
    ["Disk used", `${system.disk_used_gb} GB`],
    ["Disk available", `${system.disk_available_gb || system.disk_free_gb} GB`],
    ["Disk total", `${system.disk_total_gb} GB (${system.disk_percent}%)`],
    ["Uptime", formatUptime(system.uptime_seconds)],
  ];
  $("#system-stat-grid").innerHTML = systemCards
    .map(
      ([label, value]) =>
        `<div class="stat"><div class="label">${label}</div><div class="value">${value}</div></div>`
    )
    .join("");
  const healthOpen = $("#system-health-toggle")?.getAttribute("aria-expanded") === "true";
  $("#system-updated").textContent = healthOpen
    ? `Updated ${new Date().toLocaleTimeString()}`
    : "Hidden — click to expand";
}

function updateSidebarStatus(system) {
  const status = (system.status || "ok").toLowerCase();
  const label = status === "ok" ? "Healthy" : status.toUpperCase();
  const el = $("#ss-status-label");
  if (el) el.textContent = label;
  const cpu = Number(system.cpu_percent || 0);
  const ram = Number(system.ram_percent || 0);
  const disk = Number(system.disk_percent || 0);
  $("#ss-cpu").textContent = `${cpu}%`;
  $("#ss-ram").textContent = `${ram}%`;
  $("#ss-disk").textContent = `${disk}%`;
  $("#ss-cpu-fill").style.width = `${Math.min(100, cpu)}%`;
  $("#ss-ram-fill").style.width = `${Math.min(100, ram)}%`;
  $("#ss-disk-fill").style.width = `${Math.min(100, disk)}%`;
  $("#ss-uptime").textContent = formatUptime(system.uptime_seconds || 0);
}

function updateHealthPill(system) {
  const pill = $("#health-pill");
  if (!pill) return;
  const status = (system.status || "ok").toLowerCase();
  pill.classList.remove("warn", "critical");
  if (status === "warning") {
    pill.classList.add("warn");
    pill.textContent = "SYSTEM WARNING";
  } else if (status === "critical") {
    pill.classList.add("critical");
    pill.textContent = "SYSTEM CRITICAL";
  } else {
    pill.textContent = "SYSTEM ONLINE";
  }
}

function sparkFromHourly(hourly, status) {
  const buckets = new Array(12).fill(0);
  const now = Date.now();
  (hourly || []).forEach((row) => {
    if (status && row.status !== status) return;
    if (!row.hour) return;
    const t = new Date(row.hour).getTime();
    const hoursAgo = Math.floor((now - t) / 3600000);
    if (hoursAgo >= 0 && hoursAgo < 12) {
      buckets[11 - hoursAgo] += Number(row.count || 0);
    }
  });
  return buckets;
}

function confSparkFromHourly(hourly) {
  // Approximate confidence trend from volume mix (human share)
  const human = sparkFromHourly(hourly, "HUMAN");
  const all = sparkFromHourly(hourly, null);
  return all.map((t, i) => (t ? Math.round((human[i] / t) * 100) : 90));
}

function renderDonutChart(s) {
  const canvas = $("#chart-donut");
  if (!canvas || typeof Chart === "undefined") return;
  const parts = [
    { label: "Human", value: s.human || 0, color: "#22c55e" },
    { label: "Machine", value: s.machine || 0, color: "#f43f5e" },
    { label: "IVR", value: s.ivr || 0, color: "#a855f7" },
    { label: "Cancelled", value: s.sit || 0, color: "#f59e0b" },
    { label: "Fax", value: s.fax || 0, color: "#38bdf8" },
    { label: "Errors", value: s.errors || 0, color: "#94a3b8" },
  ].filter((p) => p.value > 0);
  const dataParts = parts.length
    ? parts
    : [{ label: "No data", value: 1, color: "#334155" }];
  const total = dataParts.reduce((a, b) => a + b.value, 0) || 1;

  $("#donut-legend").innerHTML = dataParts
    .map(
      (p) => `<div class="lg"><span><span class="dot" style="background:${p.color}"></span>${p.label}</span>
      <b>${p.value} · ${((p.value / total) * 100).toFixed(0)}%</b></div>`
    )
    .join("");

  if (state.chartDonut) state.chartDonut.destroy();
  state.chartDonut = new Chart(canvas, {
    type: "doughnut",
    data: {
      labels: dataParts.map((p) => p.label),
      datasets: [
        {
          data: dataParts.map((p) => p.value),
          backgroundColor: dataParts.map((p) => p.color),
          borderWidth: 0,
          hoverOffset: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "68%",
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.label}: ${ctx.raw}`,
          },
        },
      },
    },
    plugins: [
      {
        id: "centerText",
        afterDraw(chart) {
          const { ctx, chartArea } = chart;
          if (!chartArea) return;
          const sum = chart.data.datasets[0].data.reduce((a, b) => a + b, 0);
          ctx.save();
          ctx.fillStyle = "#e2e8f0";
          ctx.font = "700 18px Outfit, sans-serif";
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          const x = (chartArea.left + chartArea.right) / 2;
          const y = (chartArea.top + chartArea.bottom) / 2;
          ctx.fillText(String(sum), x, y - 6);
          ctx.fillStyle = "#8b9bb3";
          ctx.font = "500 10px Outfit, sans-serif";
          ctx.fillText("TODAY", x, y + 12);
          ctx.restore();
        },
      },
    ],
  });
}

function renderTrendChart(hourly) {
  const canvas = $("#chart-trend");
  if (!canvas || typeof Chart === "undefined") return;
  const labels = [];
  const values = [];
  for (let i = 11; i >= 0; i--) {
    const d = new Date(Date.now() - i * 3600000);
    labels.push(`${String(d.getHours()).padStart(2, "0")}:00`);
    values.push(0);
  }
  const now = Date.now();
  const bucket = {};
  (hourly || []).forEach((row) => {
    if (!row.hour) return;
    const t = new Date(row.hour).getTime();
    const hoursAgo = Math.floor((now - t) / 3600000);
    if (hoursAgo < 0 || hoursAgo > 11) return;
    const idx = 11 - hoursAgo;
    if (!bucket[idx]) bucket[idx] = { human: 0, total: 0 };
    bucket[idx].total += Number(row.count || 0);
    if (row.status === "HUMAN") bucket[idx].human += Number(row.count || 0);
  });
  Object.keys(bucket).forEach((idx) => {
    const b = bucket[idx];
    values[idx] = b.total ? Math.round((b.human / b.total) * 1000) / 10 : null;
  });

  if (state.chartTrend) state.chartTrend.destroy();
  state.chartTrend = new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Human %",
          data: values,
          borderColor: "#a855f7",
          backgroundColor: "rgba(168,85,247,0.15)",
          fill: true,
          tension: 0.35,
          pointRadius: 3,
          pointBackgroundColor: "#c084fc",
          spanGaps: true,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          ticks: { color: "#7f91a8", maxRotation: 0, font: { size: 10 } },
          grid: { color: "rgba(148,163,184,0.08)" },
        },
        y: {
          min: 0,
          max: 100,
          ticks: { color: "#7f91a8", callback: (v) => `${v}%`, font: { size: 10 } },
          grid: { color: "rgba(148,163,184,0.08)" },
        },
      },
      plugins: {
        legend: { display: false },
      },
    },
  });
}

$("#system-health-toggle")?.addEventListener("click", () => {
  const btn = $("#system-health-toggle");
  const grid = $("#system-stat-grid");
  const open = btn.getAttribute("aria-expanded") === "true";
  const next = !open;
  btn.setAttribute("aria-expanded", String(next));
  grid.classList.toggle("hidden", !next);
  $("#system-updated").textContent = next
    ? `Updated ${new Date().toLocaleTimeString()}`
    : "Hidden — click to expand";
});

function liveQueryParams(limit = 100) {
  const params = new URLSearchParams();
  params.set("limit", String(limit));
  const serverId = $("#live-server-filter")?.value || "";
  const status = $("#live-status-filter")?.value || "";
  if (serverId) params.set("server_id", serverId);
  if (status) params.set("status", status);
  return params.toString();
}

function cdrQueryParams() {
  const params = new URLSearchParams();
  params.set("page", String(state.cdrPage || 1));
  params.set("page_size", String(state.cdrPageSize || 50));
  const serverId = $("#cdr-server-filter")?.value || "";
  const status = $("#cdr-status-filter")?.value || "";
  const q = ($("#cdr-search")?.value || "").trim();
  if (serverId) params.set("server_id", serverId);
  if (status) params.set("status", status);
  if (q) params.set("q", q);
  return params.toString();
}

async function fillServerFilters() {
  try {
    const servers = await api("/api/servers");
    const options =
      `<option value="">ALL</option>` +
      servers
        .map((s) => `<option value="${s.id}">${escapeHtml(s.name)}</option>`)
        .join("");
    const liveSel = $("#live-server-filter");
    const cdrSel = $("#cdr-server-filter");
    const trainSel = $("#train-server-filter");
    const liveVal = liveSel?.value || "";
    const cdrVal = cdrSel?.value || "";
    const trainVal = trainSel?.value || "";
    if (liveSel) {
      liveSel.innerHTML = options;
      liveSel.value = liveVal;
    }
    if (cdrSel) {
      cdrSel.innerHTML = options;
      cdrSel.value = cdrVal;
    }
    if (trainSel) {
      trainSel.innerHTML = options;
      trainSel.value = trainVal;
    }
  } catch (e) {
    /* ignore */
  }
}

function trainQueryParams() {
  const params = new URLSearchParams();
  params.set("limit", "80");
  const serverId = $("#train-server-filter")?.value || "";
  const status = $("#train-status-filter")?.value || "";
  const q = ($("#train-search")?.value || "").trim();
  if (serverId) params.set("server_id", serverId);
  if (status) params.set("status", status);
  if (q) params.set("q", q);
  return params.toString();
}

function cdrExportParams(format) {
  const params = new URLSearchParams();
  params.set("format", format);
  const serverId = $("#cdr-server-filter")?.value || "";
  const status = $("#cdr-status-filter")?.value || "";
  const q = ($("#cdr-search")?.value || "").trim();
  if (serverId) params.set("server_id", serverId);
  if (status) params.set("status", status);
  if (q) params.set("q", q);
  return params.toString();
}

async function downloadCdrExport(format) {
  const url = `/api/cdr/export?${cdrExportParams(format)}`;
  const res = await fetch(url, {
    headers: state.token ? { Authorization: `Bearer ${state.token}` } : {},
  });
  if (!res.ok) {
    let detail = `Export failed (${res.status})`;
    try {
      const j = await res.json();
      detail = j.detail || detail;
    } catch (e) {
      /* ignore */
    }
    throw new Error(detail);
  }
  const blob = await res.blob();
  const cd = res.headers.get("Content-Disposition") || "";
  const match = cd.match(/filename=\"?([^\";]+)\"?/i);
  const filename =
    (match && match[1]) ||
    (format === "csv" ? "openamd_cdr.csv" : "openamd_cdr.xls");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(a.href);
}

function formatUptime(seconds) {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  return `${days}d ${hours}h ${minutes}m`;
}

async function loadLive(silent = false) {
  try {
    const live = await api(`/api/live?${liveQueryParams(100)}`);
    renderCallLists(live, "#live-body", "#live-cards");
    if (!silent && $("#page-dashboard") && !$("#page-dashboard").classList.contains("hidden")) {
      $("#dash-live-body").innerHTML = live.slice(0, 12).map(dashLiveRowHtml).join("");
      $("#dash-live-cards").innerHTML = live.slice(0, 12).map(callCardHtml).join("");
    }
  } catch (e) {
    /* ignore transient */
  }
}

async function loadCdr() {
  const meta = $("#cdr-meta");
  const prevBtn = $("#cdr-prev");
  const nextBtn = $("#cdr-next");
  try {
    if (meta) meta.textContent = "Loading…";
    const data = await api(`/api/cdr?${cdrQueryParams()}`);
    // Support both new paginated shape and legacy array (older servers)
    const rows = Array.isArray(data) ? data : data.rows || [];
    const total = Array.isArray(data) ? rows.length : Number(data.total || 0);
    const page = Array.isArray(data) ? 1 : Number(data.page || state.cdrPage || 1);
    const pageSize = Array.isArray(data)
      ? rows.length || state.cdrPageSize
      : Number(data.page_size || state.cdrPageSize || 50);

    state.cdrTotal = total;
    state.cdrPage = page;
    state.cdrPageSize = pageSize;

    renderCallLists(rows, "#cdr-body", "#cdr-cards");

    const pages = Math.max(1, Math.ceil(total / pageSize) || 1);
    const q = ($("#cdr-search")?.value || "").trim();
    const start = total === 0 ? 0 : (page - 1) * pageSize + 1;
    const end = Math.min(page * pageSize, total);
    if (meta) {
      meta.textContent = q
        ? `${start}–${end} of ${total} for “${q}” (page ${page}/${pages})`
        : `${start}–${end} of ${total} calls (page ${page}/${pages})`;
    }
    if (prevBtn) prevBtn.disabled = page <= 1;
    if (nextBtn) nextBtn.disabled = page >= pages || total === 0;
  } catch (err) {
    renderCallLists([], "#cdr-body", "#cdr-cards");
    if (meta) meta.textContent = err.message || "Failed to load CDR";
    if (prevBtn) prevBtn.disabled = true;
    if (nextBtn) nextBtn.disabled = true;
  }
}

function resetCdrToFirstPage() {
  state.cdrPage = 1;
  loadCdr();
}

$("#live-filter-apply")?.addEventListener("click", () => loadLive());
$("#live-server-filter")?.addEventListener("change", () => loadLive());
$("#live-status-filter")?.addEventListener("change", () => loadLive());
$("#cdr-refresh")?.addEventListener("click", () => loadCdr());
$("#cdr-search-btn")?.addEventListener("click", () => resetCdrToFirstPage());
$("#cdr-server-filter")?.addEventListener("change", () => resetCdrToFirstPage());
$("#cdr-status-filter")?.addEventListener("change", () => resetCdrToFirstPage());
$("#cdr-export-csv")?.addEventListener("click", async () => {
  try {
    await downloadCdrExport("csv");
  } catch (err) {
    alert(err.message || "CSV export failed");
  }
});
$("#cdr-export-excel")?.addEventListener("click", async () => {
  try {
    await downloadCdrExport("xls");
  } catch (err) {
    alert(err.message || "Excel export failed");
  }
});
$("#cdr-page-size")?.addEventListener("change", () => {
  state.cdrPageSize = Number($("#cdr-page-size").value) || 50;
  resetCdrToFirstPage();
});
$("#cdr-prev")?.addEventListener("click", () => {
  if (state.cdrPage > 1) {
    state.cdrPage -= 1;
    loadCdr();
  }
});
$("#cdr-next")?.addEventListener("click", () => {
  const pages = Math.max(1, Math.ceil(state.cdrTotal / (state.cdrPageSize || 50)));
  if (state.cdrPage < pages) {
    state.cdrPage += 1;
    loadCdr();
  }
});
$("#cdr-search")?.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    resetCdrToFirstPage();
  }
});

async function loadCronPage() {
  const msg = $("#cron-msg");
  msg.textContent = "";
  msg.className = "hint";
  try {
    const cfg = await api("/api/cron/recording-retention");
    $("#cron-enabled").checked = !!cfg.enabled;
    $("#cron-days").value = cfg.retention_days ?? 7;
    $("#cron-path").textContent = cfg.path
      ? `Settings file: ${cfg.path} · daily timer ~02:15`
      : "Daily timer ~02:15";
  } catch (err) {
    msg.className = "error";
    msg.textContent = err.message;
  }
}

$("#cron-form")?.addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = $("#cron-msg");
  msg.className = "hint";
  msg.textContent = "Saving…";
  try {
    const data = await api("/api/cron/recording-retention", {
      method: "PUT",
      json: {
        enabled: $("#cron-enabled").checked,
        retention_days: Number($("#cron-days").value),
      },
    });
    msg.className = "ok";
    msg.textContent = `Saved: ${data.enabled ? "enabled" : "disabled"}, delete after ${data.retention_days} day(s).`;
    $("#cron-path").textContent = data.path
      ? `Settings file: ${data.path} · daily timer ~02:15`
      : "Daily timer ~02:15";
  } catch (err) {
    msg.className = "error";
    msg.textContent = err.message;
  }
});

$("#cron-run-now")?.addEventListener("click", async () => {
  const msg = $("#cron-msg");
  if (!confirm("Run recording cleanup now using the saved retention days?")) return;
  msg.className = "hint";
  msg.textContent = "Running cleanup…";
  try {
    const data = await api("/api/cron/recording-retention/run", { method: "POST" });
    if (data.skipped) {
      msg.className = "hint";
      msg.textContent = data.message || "Skipped — cron disabled";
      return;
    }
    msg.className = "ok";
    msg.textContent = `Deleted ${data.deleted_files} files, freed ${data.freed_mb} MB (older than ${data.retention_days} day(s)).`;
  } catch (err) {
    msg.className = "error";
    msg.textContent = err.message;
  }
});

async function loadAmdSettings() {
  const msg = $("#amd-msg");
  if (msg) {
    msg.textContent = "";
    msg.className = "hint";
  }
  try {
    const cfg = await api("/api/settings/amd");
    $("#amd-enabled").checked = !!cfg.enabled;
    $("#amd-min").value = cfg.min_human_confidence_percent ?? 70;
    if ($("#amd-action")) $("#amd-action").value = cfg.below_threshold_action || "MACHINE";
    if ($("#amd-blank-as-machine")) {
      $("#amd-blank-as-machine").checked = cfg.blank_as_machine !== false;
    }
    if ($("#amd-path")) {
      $("#amd-path").textContent = cfg.path ? `Settings file: ${cfg.path}` : "";
    }
  } catch (err) {
    if (msg) {
      msg.className = "error";
      msg.textContent = err.message;
    }
  }
}

$("#amd-form")?.addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = $("#amd-msg");
  msg.className = "hint";
  msg.textContent = "Saving…";
  try {
    const data = await api("/api/settings/amd", {
      method: "PUT",
      json: {
        enabled: $("#amd-enabled").checked,
        min_human_confidence_percent: Number($("#amd-min").value),
        below_threshold_action: $("#amd-action").value,
        blank_as_machine: $("#amd-blank-as-machine")
          ? $("#amd-blank-as-machine").checked
          : true,
      },
    });
    msg.className = "ok";
    const blankNote = data.blank_as_machine
      ? " Blank/silent → MACHINE."
      : " Blank/silent may pass as HUMAN.";
    msg.textContent = data.enabled
      ? `Saved: HUMAN calls need ≥ ${data.min_human_confidence_percent}% confidence; below that → ${data.below_threshold_action}.${blankNote}`
      : `Saved: confidence gate disabled — all HUMAN calls pass to agents.${blankNote}`;
    if ($("#amd-path")) {
      $("#amd-path").textContent = data.path ? `Settings file: ${data.path}` : "";
    }
  } catch (err) {
    msg.className = "error";
    msg.textContent = err.message;
  }
});

async function loadServers() {
  const servers = await api("/api/servers");
  window.__openamd_servers = servers;
  $("#servers-body").innerHTML = servers
    .map((s) => {
      const gate = s.confidence_gate_enabled
        ? `Gate ≥${s.min_human_confidence_percent}% → ${s.below_threshold_action}`
        : "Gate: global";
      const loc = s.locale_pack_enabled
        ? `Locale: ${(s.locale_pack || "usa").toUpperCase()}`
        : "Locale: default";
      return `<tr>
      <td><strong>${escapeHtml(s.name)}</strong><div class="hint">${escapeHtml(s.description || "")}</div>
        <div class="hint">${escapeHtml(s.timezone || "")}</div>
        <div class="hint">${escapeHtml(gate)} · ${escapeHtml(loc)}</div></td>
      <td><span class="hint">${escapeHtml(s.ip_whitelist || "(any)")}</span></td>
      <td>${s.calls_today}</td>
      <td>${s.total_calls}</td>
      <td>${fmtTime(s.last_seen)}</td>
      <td>${s.is_active ? "Yes" : "No"}</td>
      <td style="white-space:nowrap">
        <button class="ghost" type="button" onclick="editServer(${s.id})">Edit</button>
        ${
          s.is_active
            ? `<button class="ghost" type="button" onclick="deactivateServer(${s.id})">Disable</button>`
            : `<button class="ghost" type="button" onclick="activateServer(${s.id})">Enable</button>`
        }
      </td>
    </tr>`;
    })
    .join("");

  $("#key-server-select").innerHTML = servers
    .filter((s) => s.is_active)
    .map((s) => `<option value="${s.id}">${escapeHtml(s.name)}</option>`)
    .join("");
}

function resetServerForm() {
  const form = $("#server-form");
  form.reset();
  $("#server-edit-id").value = "";
  $("#server-timezone").value = "America/New_York";
  $("#server-gate-enabled").checked = false;
  $("#server-gate-min").value = "70";
  $("#server-gate-action").value = "MACHINE";
  $("#server-locale-enabled").checked = false;
  $("#server-locale-pack").value = "usa";
  $("#server-form-title").textContent = "Add VICIdial server";
  $("#server-submit-btn").textContent = "Create server";
  $("#server-cancel-edit").classList.add("hidden");
  $("#server-msg").textContent = "";
}

window.editServer = (id) => {
  const servers = window.__openamd_servers || [];
  const s = servers.find((x) => x.id === id);
  if (!s) return;
  $("#server-edit-id").value = String(s.id);
  $("#server-name").value = s.name || "";
  $("#server-description").value = s.description || "";
  $("#server-timezone").value = s.timezone || "UTC";
  $("#server-ip-whitelist").value = s.ip_whitelist || "";
  $("#server-gate-enabled").checked = !!s.confidence_gate_enabled;
  $("#server-gate-min").value = String(s.min_human_confidence_percent ?? 70);
  $("#server-gate-action").value = s.below_threshold_action || "MACHINE";
  $("#server-locale-enabled").checked = !!s.locale_pack_enabled;
  $("#server-locale-pack").value = s.locale_pack || "usa";
  $("#server-form-title").textContent = "Edit VICIdial server";
  $("#server-submit-btn").textContent = "Save changes";
  $("#server-cancel-edit").classList.remove("hidden");
  $("#server-msg").textContent = `Editing #${s.id} — update server, confidence gate, or locale pack.`;
  $("#server-name").focus();
};

window.deactivateServer = async (id) => {
  if (!confirm("Disable this VICIdial server?")) return;
  await api(`/api/servers/${id}`, { method: "DELETE" });
  resetServerForm();
  loadServers();
};

window.activateServer = async (id) => {
  await api(`/api/servers/${id}`, {
    method: "PATCH",
    json: { is_active: true },
  });
  loadServers();
};

$("#server-cancel-edit").addEventListener("click", () => resetServerForm());

$("#server-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const id = (fd.get("id") || "").toString().trim();
  const payload = {
    name: (fd.get("name") || "").toString().trim(),
    description: (fd.get("description") || "").toString(),
    timezone: (fd.get("timezone") || "UTC").toString(),
    ip_whitelist: (fd.get("ip_whitelist") || "").toString().trim(),
    confidence_gate_enabled: !!$("#server-gate-enabled")?.checked,
    min_human_confidence_percent: Number($("#server-gate-min")?.value || 70),
    below_threshold_action: ($("#server-gate-action")?.value || "MACHINE").toString(),
    locale_pack_enabled: !!$("#server-locale-enabled")?.checked,
    locale_pack: ($("#server-locale-pack")?.value || "usa").toString(),
  };
  try {
    if (id) {
      await api(`/api/servers/${id}`, { method: "PATCH", json: payload });
      $("#server-msg").textContent = "Server updated.";
    } else {
      await api("/api/servers", { method: "POST", json: payload });
      $("#server-msg").textContent = "Server created.";
    }
    resetServerForm();
    loadServers();
  } catch (err) {
    $("#server-msg").textContent = err.message;
  }
});

$("#key-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  try {
    const data = await api("/api/servers/api-keys", {
      method: "POST",
      json: {
        server_id: Number(fd.get("server_id")),
        name: fd.get("name") || "default",
        notes: "",
      },
    });
    $("#key-msg").textContent = "Copy this key now — it will not be shown again.";
    $("#key-reveal").classList.remove("hidden");
    $("#key-reveal").textContent = data.api_key;
  } catch (err) {
    $("#key-msg").textContent = err.message;
  }
});

async function loadReports() {
  const days = $("#report-days").value;
  const rows = await api(`/api/reports/servers?days=${days}`);
  $("#reports-body").innerHTML = rows
    .map(
      (r) => `<tr>
      <td>${escapeHtml(r.server_name)}</td>
      <td>${r.total}</td>
      <td>${r.human} <span class="hint">${pct(r.human, r.total)}</span></td>
      <td>${r.machine} <span class="hint">${pct(r.machine, r.total)}</span></td>
      <td>${r.ivr}</td>
      <td>${r.sit}</td>
      <td>${r.fax}</td>
      <td>${r.errors}</td>
      <td>${r.avg_processing_ms}</td>
      <td>${(r.avg_confidence * 100).toFixed(1)}%</td>
      <td>${r.accuracy == null ? "—" : r.accuracy + "%"}</td>
    </tr>`
    )
    .join("");
}

$("#report-days").addEventListener("change", loadReports);

async function loadTraining() {
  const meta = $("#train-meta");
  try {
    if (meta) meta.textContent = "Loading…";
    const rows = await api(`/api/training/calls?${trainQueryParams()}`);
    const q = ($("#train-search")?.value || "").trim();
    if (meta) {
      meta.textContent = q
        ? `${rows.length} call(s) matching “${q}”`
        : `${rows.length} recent call(s) — search by called number or caller ID to train specific leads`;
    }

    $("#train-body").innerHTML = rows
      .map(
        (r) => `<tr>
      <td>${audioButtons(r)}</td>
      <td class="callid-cell" title="${escapeHtml(r.call_id)}">${escapeHtml(r.call_id)}</td>
      <td>${fmtTime(r.created_at)}</td>
      <td>${escapeHtml(r.server_name || "—")}</td>
      <td>${escapeHtml(r.called_number || "—")}</td>
      <td>${escapeHtml(r.caller_id || "—")}</td>
      <td>${statusBadge(r.status)}</td>
      <td>
        <select id="corr-${r.id}">
          <option value="HUMAN">HUMAN</option>
          <option value="MACHINE">MACHINE</option>
          <option value="IVR">IVR</option>
          <option value="SIT">CANCELLED</option>
          <option value="FAX">FAX</option>
        </select>
      </td>
      <td><button class="ghost" type="button" onclick="saveCorrection(${r.id})">Save</button></td>
    </tr>`
      )
      .join("");

    $("#train-cards").innerHTML = rows
      .map(
        (r) => `<article class="call-card">
      <div class="call-card-top">
        ${statusBadge(r.status)}
        <span class="hint">${escapeHtml(r.server_name || "—")}</span>
      </div>
      <div class="call-card-grid">
        <div><div class="k">Call ID</div><div class="v">${escapeHtml(r.call_id)}</div></div>
        <div><div class="k">Called</div><div class="v">${escapeHtml(r.called_number || "—")}</div></div>
        <div><div class="k">Caller ID</div><div class="v">${escapeHtml(r.caller_id || "—")}</div></div>
        <div><div class="k">Time</div><div class="v">${fmtTime(r.created_at)}</div></div>
      </div>
      <div class="call-card-actions" style="margin-top:.65rem;flex-direction:column;align-items:stretch">
        ${audioButtons(r)}
        <label class="hint" style="margin:0">Correct to</label>
        <select id="corr-m-${r.id}">
          <option value="HUMAN">HUMAN</option>
          <option value="MACHINE">MACHINE</option>
          <option value="IVR">IVR</option>
          <option value="SIT">CANCELLED</option>
          <option value="FAX">FAX</option>
        </select>
        <button class="ghost" type="button" onclick="saveCorrectionMobile(${r.id})">Save correction</button>
      </div>
    </article>`
      )
      .join("");
  } catch (err) {
    if (meta) meta.textContent = err.message || "Failed to load training calls";
    $("#train-body").innerHTML = "";
    $("#train-cards").innerHTML = "";
  }
}

$("#train-search-btn")?.addEventListener("click", () => loadTraining());
$("#train-refresh")?.addEventListener("click", () => loadTraining());
$("#train-server-filter")?.addEventListener("change", () => loadTraining());
$("#train-status-filter")?.addEventListener("change", () => loadTraining());
$("#train-search")?.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    loadTraining();
  }
});

window.saveCorrection = async (id) => {
  const status = $(`#corr-${id}`).value;
  const data = await api("/api/training/correct", {
    method: "POST",
    json: { call_analysis_id: id, corrected_status: status, notes: "" },
  });
  alert(data.message || "Correction logged for this call. Next dial is judged fresh from audio.");
  loadTraining();
};

window.saveCorrectionMobile = async (id) => {
  const status = $(`#corr-m-${id}`).value;
  const data = await api("/api/training/correct", {
    method: "POST",
    json: { call_analysis_id: id, corrected_status: status, notes: "" },
  });
  alert(data.message || "Correction logged for this call. Next dial is judged fresh from audio.");
  loadTraining();
};

window.quickTeach = async (id, status) => {
  try {
    const data = await api("/api/training/correct", {
      method: "POST",
      json: {
        call_analysis_id: Number(id),
        corrected_status: status,
        notes: "Quick teach from live/dashboard",
      },
    });
    const msg = data.message || `Taught → ${status}`;
    if ($("#page-live") && !$("#page-live").classList.contains("hidden")) loadLive(true);
    if ($("#page-dashboard") && !$("#page-dashboard").classList.contains("hidden")) {
      refreshDashboardLive();
    }
    if ($("#page-cdr") && !$("#page-cdr").classList.contains("hidden")) loadCdr();
    if ($("#page-training") && !$("#page-training").classList.contains("hidden")) loadTraining();
    alert(msg);
  } catch (err) {
    alert(err.message || "Teach failed");
  }
};

document.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-teach-id]");
  if (!btn) return;
  e.preventDefault();
  const id = btn.getAttribute("data-teach-id");
  const status = btn.getAttribute("data-teach-status");
  if (id && status) quickTeach(id, status);
});

async function loadTrainingHistory() {
  const phone = ($("#thist-phone")?.value || "").trim();
  const params = new URLSearchParams({ limit: "150" });
  if (phone) params.set("phone", phone);
  try {
    const [hist, ovs, stats] = await Promise.all([
      api(`/api/training/history?${params}`),
      api("/api/training/overrides?limit=200"),
      api("/api/training/stats"),
    ]);
    if ($("#thist-meta")) {
      $("#thist-meta").textContent =
        `${hist.total || 0} history rows (corrections do not force future AMD)`;
    }
    $("#thist-body").innerHTML = (hist.items || [])
      .map(
        (r) => `<tr>
        <td>${fmtTime(r.created_at)}</td>
        <td>${escapeHtml(r.phone_number || "—")}</td>
        <td>${statusBadge(r.ai_status || "ERROR")}</td>
        <td>${statusBadge(r.corrected_status || "ERROR")}</td>
        <td>${escapeHtml(r.previous_taught_status || "—")}</td>
        <td>${escapeHtml(r.action || "teach")}</td>
        <td>${escapeHtml(r.corrected_by || "—")}</td>
        <td class="hint">${escapeHtml(r.notes || "")}</td>
      </tr>`
      )
      .join("") || `<tr><td colspan="8" class="hint">No training history yet</td></tr>`;

    $("#thist-overrides").innerHTML = (ovs.items || [])
      .map(
        (o) => `<tr>
        <td>${escapeHtml(o.phone_number)}</td>
        <td>${statusBadge(o.taught_status)}</td>
        <td>${o.hit_count || 0}</td>
        <td>${escapeHtml(o.taught_by || "—")}</td>
        <td>${fmtTime(o.updated_at)}</td>
        <td><button type="button" class="ghost" onclick="revertOverride(${o.id})">Revert</button></td>
      </tr>`
      )
      .join("") || `<tr><td colspan="6" class="hint">No active overrides</td></tr>`;
  } catch (err) {
    if ($("#thist-meta")) $("#thist-meta").textContent = err.message;
  }
}

window.revertOverride = async (id) => {
  if (!confirm("Mark this legacy override inactive? (AMD already judges every call from audio.)")) return;
  try {
    await api(`/api/training/overrides/${id}/revert`, { method: "POST" });
    loadTrainingHistory();
  } catch (err) {
    alert(err.message);
  }
};

$("#thist-refresh")?.addEventListener("click", () => loadTrainingHistory());
$("#thist-phone")?.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    loadTrainingHistory();
  }
});

$("#train-backup-btn")?.addEventListener("click", async () => {
  const msg = $("#train-backup-msg");
  try {
    const res = await api("/api/training/backup", { raw: true });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `openamd_training_backup_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    if (msg) msg.textContent = "Backup downloaded.";
  } catch (err) {
    if (msg) msg.textContent = err.message;
  }
});

$("#train-import-file")?.addEventListener("change", async (e) => {
  const file = e.target.files && e.target.files[0];
  const msg = $("#train-backup-msg");
  if (!file) return;
  const replace = !!$("#train-import-replace")?.checked;
  if (replace && !confirm("Replace ALL existing training with this backup?")) {
    e.target.value = "";
    return;
  }
  const fd = new FormData();
  fd.append("file", file);
  try {
    const data = await api(`/api/training/backup/import?replace=${replace ? "true" : "false"}`, {
      method: "POST",
      body: fd,
    });
    if (msg) {
      msg.textContent =
        `Imported overrides +${data.imported_overrides} / updated ${data.updated_overrides}, ` +
        `corrections +${data.imported_corrections}.`;
    }
  } catch (err) {
    if (msg) msg.textContent = err.message;
  }
  e.target.value = "";
});

$("#train-wipe-form")?.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const confirmText = String(fd.get("confirm") || "").trim();
  if (!confirm("Delete ALL training history? This only clears the audit log — AMD already judges every call from audio.")) return;
  const msg = $("#train-wipe-msg");
  try {
    const data = await api("/api/training/wipe", {
      method: "POST",
      json: { confirm: confirmText },
    });
    if (msg) {
      msg.textContent =
        `Wiped ${data.deleted_overrides} overrides and ${data.deleted_corrections} history rows.`;
    }
    e.target.reset();
  } catch (err) {
    if (msg) msg.textContent = err.message;
  }
});


async function loadHealth() {
  const h = await fetch("/api/health").then((r) => r.json());
  let rec = null;
  try {
    rec = await api("/api/recordings/status");
  } catch (e) {
    rec = null;
  }
  const payload = rec ? { ...h, recordings: rec } : h;
  const el = $("#health-json");
  if (el) el.textContent = JSON.stringify(payload, null, 2);
  updateHealthPill({ status: h.status === "ok" ? "ok" : "warning" });
}

$("#password-form")?.addEventListener("submit", async (e) => {
  e.preventDefault();
  const msg = $("#password-msg");
  msg.textContent = "";
  msg.className = "hint";
  const fd = new FormData(e.target);
  try {
    const data = await api("/api/change-password", {
      method: "POST",
      json: {
        current_password: fd.get("current_password"),
        new_password: fd.get("new_password"),
        confirm_password: fd.get("confirm_password"),
      },
    });
    msg.className = "ok";
    msg.textContent = data.message || "Password updated.";
    e.target.reset();
  } catch (err) {
    msg.className = "error";
    msg.textContent = err.message;
  }
});

$("#repair-recordings-btn")?.addEventListener("click", async () => {
  const msg = $("#repair-msg");
  msg.textContent = "Scanning recordings…";
  msg.className = "hint";
  try {
    const data = await api("/api/recordings/repair", { method: "POST" });
    msg.className = "ok";
    msg.textContent = `Linked ${data.linked}/${data.checked} calls. WAV files on disk: ${data.wav_files_on_disk}. Dir: ${data.recordings_dir}`;
    loadLive(true);
    loadDashboard().catch(() => {});
  } catch (err) {
    msg.className = "error";
    msg.textContent = err.message;
  }
});

function renderMaintStats(targetId, s) {
  const cards = [
    ["Call analyses", s.call_analyses],
    ["Training corrections", s.training_corrections],
    ["Oldest call", s.oldest_call ? fmtTime(s.oldest_call) : "—"],
    ["Newest call", s.newest_call ? fmtTime(s.newest_call) : "—"],
    ["Audio files", s.audio_files],
    ["Audio size", `${s.audio_mb} MB (${s.audio_gb} GB)`],
    ["Recordings path", s.recordings_dir],
    ["Database", s.database || "postgresql"],
  ];
  $(targetId).innerHTML = cards
    .map(
      ([label, value]) =>
        `<div class="stat"><div class="label">${label}</div><div class="value">${value}</div></div>`
    )
    .join("");
}

async function loadWipePage() {
  $("#wipe-msg").textContent = "";
  const s = await api("/api/maintenance/stats");
  renderMaintStats("#wipe-stats", s);
}

async function loadAudioPage() {
  $("#audio-msg").textContent = "";
  const s = await api("/api/maintenance/stats");
  renderMaintStats("#audio-stats", s);
}

$("#wipe-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const confirmText = String(fd.get("confirm") || "").trim();
  const daysRaw = String(fd.get("older_than_days") || "").trim();
  const older = daysRaw === "" ? null : Number(daysRaw);
  const scope = older == null ? "ALL detection logs" : `logs older than ${older} day(s)`;
  if (!confirm(`Permanently wipe ${scope}? This cannot be undone.`)) return;
  $("#wipe-msg").textContent = "Wiping…";
  try {
    const data = await api("/api/maintenance/wipe-logs", {
      method: "POST",
      json: { confirm: confirmText, older_than_days: older },
    });
    $("#wipe-msg").textContent =
      `Deleted ${data.deleted_call_analyses} call logs, ${data.deleted_training_corrections} corrections` +
      (data.deleted_audio_files != null
        ? `, and ${data.deleted_audio_files} audio file(s) (${data.freed_mb || 0} MB).`
        : ".");
    if (data.failed_audio_files) {
      $("#wipe-msg").textContent += ` Warning: ${data.failed_audio_files} audio file(s) could not be removed.`;
    }
    e.target.reset();
    loadWipePage();
  } catch (err) {
    $("#wipe-msg").textContent = err.message;
  }
});

$("#audio-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const confirmText = String(fd.get("confirm") || "").trim();
  const daysRaw = String(fd.get("older_than_days") || "").trim();
  const older = daysRaw === "" ? null : Number(daysRaw);
  const scope = older == null ? "ALL audio files" : `audio older than ${older} day(s)`;
  if (!confirm(`Permanently delete ${scope} from the AI AMD server?`)) return;
  $("#audio-msg").textContent = "Deleting audio…";
  try {
    const data = await api("/api/maintenance/delete-audio", {
      method: "POST",
      json: { confirm: confirmText, older_than_days: older },
    });
    $("#audio-msg").textContent =
      `Deleted ${data.deleted_files} files, freed ${data.freed_mb} MB (${data.freed_gb} GB).` +
      (data.failed_files ? ` Failed: ${data.failed_files}.` : "");
    e.target.reset();
    loadAudioPage();
  } catch (err) {
    $("#audio-msg").textContent = err.message;
  }
});

function tickClock() {
  const el = $("#clock");
  if (!el) return;
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  el.textContent = `Server Time ${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

async function refreshDashboardLive() {
  try {
    const [system, live] = await Promise.all([
      api("/api/system/stats"),
      api("/api/live?limit=12"),
    ]);
    updateSidebarStatus(system);
    updateHealthPill(system);
    $("#dash-live-body").innerHTML = (live || []).map(dashLiveRowHtml).join("");
    $("#dash-live-cards").innerHTML = (live || []).map(callCardHtml).join("");
    $("#recent-mini").innerHTML = (live || [])
      .slice(0, 5)
      .map(
        (r) => `<div class="rm">
          <span class="id">${escapeHtml(r.call_id)}</span>
          ${statusBadge(r.status)}
          <span class="${confClass(r.confidence)}">${((r.confidence || 0) * 100).toFixed(0)}%</span>
        </div>`
      )
      .join("") || `<div class="hint">No recent calls yet</div>`;
  } catch (e) {
    /* ignore */
  }
}

function startLiveTimer() {
  stopLiveTimer();
  tickClock();
  state.liveTimer = setInterval(() => {
    if (!$("#page-live").classList.contains("hidden") && $("#live-auto").checked) {
      loadLive(true);
    }
    if (!$("#page-dashboard").classList.contains("hidden")) {
      refreshDashboardLive();
    }
    tickClock();
  }, 5000);
}

function stopLiveTimer() {
  if (state.liveTimer) clearInterval(state.liveTimer);
  state.liveTimer = null;
}

if (state.token) {
  showApp(true);
} else {
  showApp(false);
}
