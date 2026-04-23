const TAU_BUCKETS = [
  { label: "0-2", min: 0, max: 2 },
  { label: "2-4", min: 2, max: 4 },
  { label: "4-6", min: 4, max: 6 },
  { label: "6-8", min: 6, max: 8 },
  { label: "8-10", min: 8, max: 10 },
  { label: "10-12", min: 10, max: 12 },
  { label: "12-15", min: 12, max: 15, inclusiveMax: true },
];

const COMPARISON_COLUMNS = [
  { key: "model", label: "Model", sortable: false },
  { key: "paperObjective", label: "Objective", sortable: true, defaultDir: "desc" },
  { key: "realized", label: "Realized PnL", sortable: true, defaultDir: "desc" },
  { key: "equity", label: "Equity", sortable: true, defaultDir: "desc" },
  { key: "settled", label: "Settled", sortable: true, defaultDir: "desc" },
  { key: "claimedYN", label: "Claimed Y / N", sortable: false },
  { key: "winRate", label: "Win Rate", sortable: true, defaultDir: "desc" },
  { key: "expectancy", label: "Expectancy", sortable: true, defaultDir: "desc" },
  { key: "avgWin", label: "Avg Win", sortable: false },
  { key: "avgLoss", label: "Avg Loss", sortable: false },
  { key: "largestWin", label: "Largest Win", sortable: false },
  { key: "largestLoss", label: "Largest Loss", sortable: false },
  { key: "maxDd", label: "Max Drawdown", sortable: true, defaultDir: "asc" },
  { key: "open", label: "Open", sortable: false },
  { key: "logs", label: "Logs", sortable: false },
];

const state = {
  rootHandle: null,
  snapshotFiles: [],
  folderRootHandle: null,
  runHandleMap: new Map(),
  snapshotRunMap: new Map(),
  selectedRun: "",
  sourceKind: "idle",
  sourceLabel: "",
  environment: "demo",
  refreshSeconds: 5,
  lookbackDays: 7,
  autoRefresh: true,
  timer: null,
  loading: false,
  selectedModel: "",
  selectedTab: "overview",
  comparisonSort: { key: "paperObjective", dir: "desc" },
  lastLoaded: null,
  lastLoadedAt: null,
};

const $ = (id) => document.getElementById(id);
const setTextIfPresent = (id, value) => {
  const node = $(id);
  if (node) node.textContent = value;
};

const fmtMoney = (v) => `${v < 0 ? "-" : ""}$${Math.abs(Number(v || 0)).toFixed(2)}`;
const fmtPct = (v, d = 1) => `${(Number(v || 0) * 100).toFixed(d)}%`;
const fmtCount = (v) => new Intl.NumberFormat().format(Number(v || 0));
const fmtRatio = (v, d = 2) => {
  if (v === Number.POSITIVE_INFINITY) return "inf";
  return Number(v || 0).toFixed(d);
};
const fmtDate = (v) => (v ? new Date(v).toLocaleString() : "n/a");
const num = (v, f = 0) => (Number.isFinite(Number(v)) ? Number(v) : f);

function calculateKalshiFeeDollars(entryPrice, contracts) {
  const rawFee = 0.07 * num(contracts, 0) * num(entryPrice, 0) * (1 - num(entryPrice, 0));
  return Math.ceil(rawFee * 100) / 100;
}

function calculateRealizedCashMetrics(entryPriceCents, contracts) {
  const entryPrice = num(entryPriceCents, 0) / 100;
  const entryCost = entryPrice * num(contracts, 0);
  const fees = calculateKalshiFeeDollars(entryPrice, contracts);
  return {
    entryCost,
    fees,
    cashRequired: entryCost + fees,
  };
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]
  ));
}

function sync(label, kind = "warn") {
  $("sync-label").textContent = label;
  $("sync-dot").className = `dot${kind === "ok" ? " ok" : kind === "bad" ? " bad" : ""}`;
}

function setTopMeta({
  source = "idle",
  run = "none",
  mode = "unknown",
  root = "none",
  loader = "folder or snapshot",
  files = 0,
  focus = "auto",
} = {}) {
  $("source-pill").textContent = source;
  $("run-pill").textContent = run;
  $("mode-pill").textContent = mode;
  $("root-pill").textContent = root;
  $("loader-pill").textContent = loader;
  $("file-pill").textContent = fmtCount(files);
  $("focus-meta-pill").textContent = focus;
}

function lines(text) {
  return text
    .split(/\r?\n/)
    .map((x) => x.trim())
    .filter(Boolean)
    .map((line) => {
      try {
        return JSON.parse(line);
      } catch {
        return null;
      }
    })
    .filter(Boolean);
}

function parseDateFolder(name) {
  const d = new Date(`${name}T00:00:00`);
  return Number.isNaN(d.getTime()) ? null : d;
}

function inLookback(name) {
  const d = parseDateFolder(name);
  if (!d) return true;
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - state.lookbackDays);
  cutoff.setHours(0, 0, 0, 0);
  return d >= cutoff;
}

async function getDir(parent, name) {
  try {
    return await parent.getDirectoryHandle(name);
  } catch {
    return null;
  }
}

async function looksLikeRunRoot(handle) {
  if (!handle) return false;
  const [signalDir, executionDir] = await Promise.all([
    getDir(handle, "signal"),
    getDir(handle, "execution"),
  ]);
  return Boolean(signalDir || executionDir);
}

async function discoverRunHandles(baseHandle) {
  const runs = new Map();
  if (await looksLikeRunRoot(baseHandle)) {
    runs.set(baseHandle.name || "selected", baseHandle);
    return runs;
  }
  for await (const [name, handle] of baseHandle.entries()) {
    if (handle.kind !== "directory") continue;
    if (await looksLikeRunRoot(handle)) runs.set(name, handle);
  }
  return runs;
}

function preferredRunKey(keys) {
  return [...keys].sort().at(-1) || "";
}

function updateRunSelector(runKeys) {
  const select = $("run-select");
  if (!runKeys.length) {
    select.innerHTML = `<option value="">auto</option>`;
    state.selectedRun = "";
    return;
  }
  if (!runKeys.includes(state.selectedRun)) state.selectedRun = preferredRunKey(runKeys);
  select.innerHTML = runKeys.map((runKey) => `<option value="${escapeHtml(runKey)}">${escapeHtml(runKey)}</option>`).join("");
  select.value = state.selectedRun;
}

function snapshotDescriptor(file) {
  const rel = (file.webkitRelativePath || file.name).replace(/\\/g, "/");
  const parts = rel.split("/").filter(Boolean);
  const areaIndex = parts.findIndex((part) => part === "execution" || part === "signal");
  if (areaIndex === -1) return null;
  const area = parts[areaIndex];
  const runKey = areaIndex > 0 ? parts[areaIndex - 1] : "selected";
  const tail = parts.slice(areaIndex);
  let model = "single";
  let env = null;
  let dateKey = null;
  if (tail.length >= 4 && (tail[1] === "demo" || tail[1] === "production")) {
    env = tail[1];
    dateKey = tail[2];
  } else if (tail.length >= 5 && (tail[2] === "demo" || tail[2] === "production")) {
    model = tail[1];
    env = tail[2];
    dateKey = tail[3];
  } else {
    return null;
  }
  if (tail.at(-1) !== "events.jsonl") return null;
  return { rel, runKey, area, model, env, dateKey };
}

function buildSnapshotRunMap(files) {
  const grouped = new Map();
  for (const file of files) {
    const descriptor = snapshotDescriptor(file);
    if (!descriptor) continue;
    if (!grouped.has(descriptor.runKey)) grouped.set(descriptor.runKey, []);
    grouped.get(descriptor.runKey).push({ file, descriptor });
  }
  return grouped;
}

async function dateHandlesFromEnvDir(envDir) {
  const out = [];
  for await (const [name, h] of envDir.entries()) {
    if (h.kind !== "directory" || !inLookback(name)) continue;
    try {
      const f = await h.getFileHandle("events.jsonl");
      out.push({ handle: f, dateKey: name });
    } catch {
      continue;
    }
  }
  return out.sort((a, b) => a.dateKey.localeCompare(b.dateKey));
}

async function areaHandlesByModel(root, area, env) {
  const areaDir = await getDir(root, area);
  if (!areaDir) return new Map();
  const modelMap = new Map();
  for await (const [name, h] of areaDir.entries()) {
    if (h.kind !== "directory") continue;
    if (name === "demo" || name === "production") {
      if (name !== env) continue;
      const handles = await dateHandlesFromEnvDir(h);
      if (handles.length) modelMap.set("single", handles);
      continue;
    }
    const envDir = await getDir(h, env);
    if (!envDir) continue;
    const handles = await dateHandlesFromEnvDir(envDir);
    if (handles.length) modelMap.set(name, handles);
  }
  return modelMap;
}

async function readHandleRows(entry) {
  return lines(await (await entry.handle.getFile()).text());
}

function pruneLegacySingleModel(modelData) {
  const modelNames = Object.keys(modelData || {});
  const namedModels = modelNames.filter((model) => model !== "single");
  if (!namedModels.length || !modelData.single) return modelData;
  const next = { ...modelData };
  delete next.single;
  return next;
}

function normOpen(raw) {
  if (!Array.isArray(raw)) return [];
  return raw.map((x) => (typeof x === "string"
    ? { ticker: x, side: "?", contracts: 1, cash: 0 }
    : {
        ticker: x.ticker || "unknown",
        side: x.side || "?",
        contracts: num(x.contracts, 1),
        cash: num(x.cash_required_dollars, 0),
      }));
}

function exposure(raw) {
  const map = new Map();
  for (const p of normOpen(raw)) {
    const key = `${p.ticker}|${p.side}`;
    const cur = map.get(key) || { ticker: p.ticker, side: p.side, count: 0, contracts: 0, cash: 0 };
    cur.count += 1;
    cur.contracts += p.contracts;
    cur.cash += p.cash;
    map.set(key, cur);
  }
  return [...map.values()].sort((a, b) => b.cash - a.cash);
}

function tableHtml(cols, rows, empty, minWidth = 860) {
  if (!rows.length) return `<div class="empty">${escapeHtml(empty)}</div>`;
  return `<div class="table-wrap"><table style="min-width:${minWidth}px"><thead><tr>${cols.map((c) => `<th>${c.label}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${cols.map((c) => `<td>${c.render(row)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function statsGridHtml(items, className = "stats-grid") {
  const filtered = items.filter(Boolean);
  if (!filtered.length) return `<div class="empty">No stats available yet.</div>`;
  return `<div class="${className}">${filtered.map((item) => `
    <div class="stat-card">
      <div class="stat-label">${escapeHtml(item.label)}</div>
      <div class="stat-value ${item.tone || ""}">${item.value}</div>
      ${item.sub ? `<div class="stat-sub">${item.sub}</div>` : ""}
    </div>
  `).join("")}</div>`;
}

function panelSection(title, subtitle, body) {
  return `<article class="subpanel"><div class="panel-head"><div><h3>${escapeHtml(title)}</h3><p class="section-copy">${escapeHtml(subtitle)}</p></div></div>${body}</article>`;
}

function moneyTone(v) {
  return v > 0 ? "good" : v < 0 ? "bad" : "";
}

function ratioTone(v) {
  return v > 0 ? "good" : v < 0 ? "bad" : "";
}

function normalizeSide(side) {
  return side === "YES" || side === "NO" ? side : "UNKNOWN";
}

function tauBucketLabel(tau) {
  const value = Number(tau);
  if (!Number.isFinite(value)) return null;
  for (const bucket of TAU_BUCKETS) {
    const upperOk = bucket.inclusiveMax ? value <= bucket.max : value < bucket.max;
    if (value >= bucket.min && upperOk) return bucket.label;
  }
  return null;
}

function createTauRows() {
  return TAU_BUCKETS.map((bucket) => ({
    label: bucket.label,
    approved: 0,
    claimed: 0,
    settled: 0,
    wins: 0,
    losses: 0,
    realized: 0,
    expectancy: 0,
  }));
}

function createSideStats() {
  return {
    YES: { approved: 0, claimed: 0, settled: 0, wins: 0, losses: 0, realized: 0 },
    NO: { approved: 0, claimed: 0, settled: 0, wins: 0, losses: 0, realized: 0 },
    UNKNOWN: { approved: 0, claimed: 0, settled: 0, wins: 0, losses: 0, realized: 0 },
  };
}

function syntheticApprovalDecisionId(index) {
  return `approved:${index}`;
}

function isSyntheticApprovalDecisionId(decisionId) {
  return String(decisionId || "").startsWith("approved:");
}

function approvalDecisionKey(ticker, side) {
  return `${ticker || "unknown"}|${normalizeSide(side)}`;
}

function applySignalPayloadToDecision(decision, payload, loggedAt) {
  decision.ticker = payload.ticker || decision.ticker;
  decision.side = normalizeSide(payload.side || decision.side);
  decision.approvedAt = decision.approvedAt || loggedAt;
  decision.status = decision.claimedAt ? decision.status : "approved";
  decision.tau = payload.tau_minutes ?? decision.tau;
  decision.predictedYesProbability = payload.predicted_yes_probability ?? decision.predictedYesProbability;
  decision.predictedNoProbability = payload.predicted_no_probability ?? (
    payload.predicted_yes_probability == null ? decision.predictedNoProbability : 1 - payload.predicted_yes_probability
  );
  decision.featureBasisMarketProb = payload.feature_basis_market_prob ?? payload.market_prob ?? decision.featureBasisMarketProb;
  decision.rawEdge = payload.raw_model_edge ?? decision.rawEdge;
  decision.postEdge = payload.post_cost_edge ?? decision.postEdge;
  decision.yesEdge = payload.yes_post_cost_edge ?? decision.yesEdge;
  decision.noEdge = payload.no_post_cost_edge ?? decision.noEdge;
  decision.refPx = payload.reference_price_cents ?? decision.refPx;
  decision.maxPx = payload.max_acceptable_entry_price_cents ?? decision.maxPx;
  decision.lastYesPx = payload.last_yes_price_cents ?? decision.lastYesPx;
  decision.yesBid = payload.yes_bid_cents ?? decision.yesBid;
  decision.yesAsk = payload.yes_ask_cents ?? decision.yesAsk;
  decision.buyYesPx = payload.buy_yes_price_cents ?? decision.buyYesPx;
  decision.buyNoPx = payload.buy_no_price_cents ?? decision.buyNoPx;
  decision.quoteMid = payload.quote_mid_prob ?? decision.quoteMid;
  decision.quoteSpread = payload.quote_spread_cents ?? decision.quoteSpread;
  decision.quoteAge = payload.quote_age_seconds ?? decision.quoteAge;
}

function applyApprovalSnapshotToDecision(decision, snapshot) {
  if (!snapshot?.payload) return;
  applySignalPayloadToDecision(decision, snapshot.payload, snapshot.loggedAt || decision.approvedAt);
}

function closeApprovalEpisodes(approvalEpisodes, ticker, side = null) {
  for (const key of [...approvalEpisodes.keys()]) {
    const [episodeTicker, episodeSide] = key.split("|");
    if (episodeTicker !== ticker) continue;
    if (side && episodeSide !== side) continue;
    approvalEpisodes.delete(key);
  }
}

function findPendingApprovalDecisionId(decisions, ticker, side, claimTime) {
  let chosenId = null;
  let chosenAt = null;
  for (const [decisionId, decision] of decisions.entries()) {
    if (!isSyntheticApprovalDecisionId(decisionId)) continue;
    if (decision.claimedAt || decision.settledAt) continue;
    if ((decision.ticker || "unknown") !== ticker) continue;
    if (normalizeSide(decision.side) !== side) continue;
    if (!decision.approvedAt) continue;
    const approvedAt = new Date(decision.approvedAt);
    if (Number.isNaN(approvedAt.getTime())) continue;
    if (claimTime && approvedAt > claimTime) continue;
    if (!chosenAt || approvedAt > chosenAt) {
      chosenAt = approvedAt;
      chosenId = decisionId;
    }
  }
  return chosenId;
}

function discardPendingApprovalDecisions(decisions, ticker, side, claimTime, keepDecisionId = null) {
  for (const [decisionId, decision] of decisions.entries()) {
    if (!isSyntheticApprovalDecisionId(decisionId)) continue;
    if (decisionId === keepDecisionId) continue;
    if (decision.claimedAt || decision.settledAt) continue;
    if ((decision.ticker || "unknown") !== ticker) continue;
    if (normalizeSide(decision.side) !== side) continue;
    if (!decision.approvedAt) continue;
    const approvedAt = new Date(decision.approvedAt);
    if (claimTime && !Number.isNaN(approvedAt.getTime()) && approvedAt > claimTime) continue;
    decisions.delete(decisionId);
  }
}

function promoteApprovalDecision(decisions, sourceId, targetId, seed = {}) {
  if (!sourceId || sourceId === targetId) return ensureDecision(decisions, targetId, seed);
  const source = decisions.get(sourceId);
  const target = ensureDecision(decisions, targetId, seed);
  if (source) {
    Object.assign(target, source, { decisionId: targetId });
    decisions.delete(sourceId);
  }
  return target;
}

function tickersFromRawPositions(raw) {
  if (!Array.isArray(raw)) return [];
  return raw.map((entry) => (typeof entry === "string" ? entry : entry?.ticker || "unknown"));
}

function tallyTickers(items) {
  const counts = new Map();
  for (const item of items) counts.set(item, (counts.get(item) || 0) + 1);
  return counts;
}

function addedTickers(prevRaw, nextRaw) {
  const prior = tallyTickers(tickersFromRawPositions(prevRaw));
  const current = tallyTickers(tickersFromRawPositions(nextRaw));
  const added = [];
  for (const [ticker, count] of current.entries()) {
    const delta = count - (prior.get(ticker) || 0);
    for (let index = 0; index < Math.max(0, delta); index += 1) added.push(ticker);
  }
  return added;
}

function removedTickers(prevRaw, nextRaw) {
  const prior = tallyTickers(tickersFromRawPositions(prevRaw));
  const current = tallyTickers(tickersFromRawPositions(nextRaw));
  const removed = [];
  for (const [ticker, count] of prior.entries()) {
    const delta = count - (current.get(ticker) || 0);
    for (let index = 0; index < Math.max(0, delta); index += 1) removed.push(ticker);
  }
  return removed;
}

function assignSnapshotDeltasToClaims(decisions, previousSnapshot, currentSnapshot) {
  if (!previousSnapshot) {
    const initialPending = tickersFromRawPositions(currentSnapshot.pendingRaw);
    if (!initialPending.length || !(num(currentSnapshot.deployed, 0) > 0)) return;
    const share = num(currentSnapshot.deployed, 0) / initialPending.length;
    for (const ticker of initialPending) {
    const candidates = [...decisions.values()]
      .filter((decision) => decision.claimedAt
        && !decision.settledAt
        && (decision.ticker || "unknown") === ticker)
      .sort((a, b) => new Date(b.claimedAt) - new Date(a.claimedAt));
    const match = candidates.find((decision) => decision.cashRequired == null);
    if (!match) continue;
    match.cashRequired = share;
    match.entryCost = share;
    match.contracts = match.contracts ?? 1;
  }
    return;
  }
  const added = addedTickers(previousSnapshot.openRaw, currentSnapshot.openRaw);
  const pendingAdded = addedTickers(previousSnapshot.pendingRaw, currentSnapshot.pendingRaw);
  const removed = removedTickers(previousSnapshot.openRaw, currentSnapshot.openRaw);
  if ((!added.length && !pendingAdded.length) || removed.length) return;
  const delta = num(currentSnapshot.deployed, 0) - num(previousSnapshot.deployed, 0);
  if (!(delta > 0)) return;
  const targets = pendingAdded.length ? pendingAdded : added;
  const share = delta / targets.length;
  const snapshotTime = currentSnapshot.time;
  for (const ticker of targets) {
    const candidates = [...decisions.values()]
      .filter((decision) => decision.claimedAt
        && !decision.settledAt
        && (decision.ticker || "unknown") === ticker
        && (!snapshotTime || new Date(decision.claimedAt) <= snapshotTime))
      .sort((a, b) => new Date(b.claimedAt) - new Date(a.claimedAt));
    const match = candidates.find((decision) => decision.cashRequired == null);
    if (!match) continue;
    match.cashRequired = share;
    match.entryCost = share;
    match.contracts = match.contracts ?? 1;
  }
}

function buildOpenExposure(raw, decisions) {
  const enriched = [];
  if (Array.isArray(raw) && raw.some((entry) => typeof entry !== "string")) {
    return exposure(raw);
  }

  const openDecisionGroups = new Map();
  for (const decision of decisions.values()) {
    if (!decision.claimedAt || decision.settledAt) continue;
    if (["cancelled", "rejected", "error"].includes(String(decision.status || "").toLowerCase())) continue;
    const side = normalizeSide(decision.side);
    const key = `${decision.ticker}|${side}`;
    const group = openDecisionGroups.get(key) || {
      ticker: decision.ticker || "unknown",
      side,
      remaining: 0,
      avgCash: 0,
      totalCash: 0,
    };
    group.remaining += 1;
    group.totalCash += num(decision.cashRequired, 0);
    group.avgCash = group.remaining ? group.totalCash / group.remaining : 0;
    openDecisionGroups.set(key, group);
  }

  for (const ticker of tickersFromRawPositions(raw)) {
    const matches = [...openDecisionGroups.values()].filter((group) => group.ticker === ticker && group.remaining > 0);
    if (matches.length === 1) {
      const match = matches[0];
      enriched.push({
        ticker,
        side: match.side,
        contracts: 1,
        cash_required_dollars: match.avgCash,
      });
      match.remaining -= 1;
      continue;
    }
    enriched.push({ ticker, side: "?", contracts: 1, cash_required_dollars: 0 });
  }

  if (!enriched.length && openDecisionGroups.size) {
    for (const group of openDecisionGroups.values()) {
      for (let index = 0; index < group.remaining; index += 1) {
        enriched.push({
          ticker: group.ticker,
          side: group.side,
          contracts: 1,
          cash_required_dollars: group.avgCash,
        });
      }
    }
  }

  return exposure(enriched);
}

function isDecisionOpen(decision) {
  const status = String(decision.status || "").toLowerCase();
  return Boolean(
    decision.claimedAt
      && !decision.settledAt
      && !["cancelled", "rejected", "error"].includes(status),
  );
}

function deriveOpenDecisionPositions(decisions) {
  return decisions
    .filter((decision) => isDecisionOpen(decision))
    .sort((a, b) => new Date(a.claimedAt || 0) - new Date(b.claimedAt || 0))
    .map((decision) => ({
      ticker: decision.ticker || "unknown",
      side: normalizeSide(decision.side),
      contracts: Math.max(1, num(decision.contracts, 1)),
      cash_required_dollars: num(decision.cashRequired, 0),
    }));
}

function deriveLatestPortfolio(decisions, realized, initialEquity, fallbackSnapshot = null) {
  const openPositions = deriveOpenDecisionPositions(decisions);
  const deployed = openPositions.reduce((sum, row) => sum + num(row.cash_required_dollars, 0), 0);
  const equity = num(initialEquity, 0) + num(realized, 0);
  const cash = equity - deployed;
  const latestDecisionTime = decisions
    .map((decision) => [decision.settledAt, decision.claimedAt].find(Boolean))
    .filter(Boolean)
    .map((value) => new Date(value))
    .filter((value) => !Number.isNaN(value.getTime()))
    .sort((a, b) => b - a)[0] || null;
  return {
    time: fallbackSnapshot?.time || latestDecisionTime,
    label: fallbackSnapshot?.label || (latestDecisionTime ? latestDecisionTime.toISOString() : null),
    cash,
    deployed,
    equity,
    open: openPositions.length,
    pending: 0,
    openRaw: openPositions,
    pendingRaw: [],
  };
}

function ensureDecision(decisions, decisionId, seed = {}) {
  const existing = decisions.get(decisionId);
  if (existing) return existing;
  const created = {
    decisionId,
    ticker: seed.ticker || "unknown",
    side: seed.side || "UNKNOWN",
    approvedAt: null,
    claimedAt: null,
    settledAt: null,
    status: "observed",
    tau: null,
    predictedYesProbability: null,
    predictedNoProbability: null,
    featureBasisMarketProb: null,
    rawEdge: null,
    postEdge: null,
    yesEdge: null,
    noEdge: null,
    refPx: null,
    maxPx: null,
    lastYesPx: null,
    yesBid: null,
    yesAsk: null,
    buyYesPx: null,
    buyNoPx: null,
    quoteMid: null,
    quoteSpread: null,
    quoteAge: null,
    result: null,
    pnl: null,
    cumPnl: null,
    cashRequired: null,
    contracts: null,
    orderId: null,
  };
  decisions.set(decisionId, created);
  return created;
}

function analyze(signalRows, executionRows) {
  const signals = signalRows
    .map((r) => ({ ...r, t: new Date(r.logged_at) }))
    .filter((r) => !Number.isNaN(r.t.getTime()))
    .sort((a, b) => a.t - b.t);
  const execs = executionRows
    .map((r) => ({ ...r, t: new Date(r.logged_at) }))
    .filter((r) => !Number.isNaN(r.t.getTime()))
    .sort((a, b) => a.t - b.t);

  const signalStart = signals.filter((r) => r.event_type === "signal_started").at(-1)?.payload || {};
  const decisions = new Map();
  const approvalEpisodes = new Map();
  const latestApprovedByKey = new Map();
  const blockCounts = new Map();
  let blocked = 0;
  let latestSignal = null;
  let nextSyntheticApprovalId = 0;

  for (const row of signals) {
    if (row.event_type !== "signal_decision") continue;
    const p = row.payload || {};
    const ticker = p.ticker || "unknown";
    const side = normalizeSide(p.side);
    latestSignal = {
      loggedAt: row.logged_at,
      ticker,
      side: p.side || "UNKNOWN",
      approved: Boolean(p.approved),
      predicted_yes_probability: p.predicted_yes_probability ?? null,
      predicted_no_probability: p.predicted_no_probability ?? (p.predicted_yes_probability == null ? null : 1 - p.predicted_yes_probability),
      feature_basis_market_prob: p.feature_basis_market_prob ?? p.market_prob ?? null,
      raw_model_edge: p.raw_model_edge ?? null,
      post_cost_edge: p.post_cost_edge ?? null,
      yes_post_cost_edge: p.yes_post_cost_edge ?? null,
      no_post_cost_edge: p.no_post_cost_edge ?? null,
      tau_minutes: p.tau_minutes ?? null,
      reference_price_cents: p.reference_price_cents ?? null,
      max_acceptable_entry_price_cents: p.max_acceptable_entry_price_cents ?? null,
      last_yes_price_cents: p.last_yes_price_cents ?? null,
      yes_bid_cents: p.yes_bid_cents ?? null,
      yes_ask_cents: p.yes_ask_cents ?? null,
      buy_yes_price_cents: p.buy_yes_price_cents ?? null,
      buy_no_price_cents: p.buy_no_price_cents ?? null,
      quote_mid_prob: p.quote_mid_prob ?? null,
      quote_spread_cents: p.quote_spread_cents ?? null,
      quote_age_seconds: p.quote_age_seconds ?? null,
      block_reason: p.block_reason || null,
    };

    if (p.approved) {
      const decisionKey = approvalDecisionKey(ticker, side);
      const decisionId = p.decision_id || approvalEpisodes.get(decisionKey) || syntheticApprovalDecisionId(nextSyntheticApprovalId++);
      approvalEpisodes.set(decisionKey, decisionId);
      latestApprovedByKey.set(decisionKey, { loggedAt: row.logged_at, payload: { ...p } });
      const decision = ensureDecision(decisions, decisionId, { ticker, side });
      applySignalPayloadToDecision(decision, p, row.logged_at);
      continue;
    }

    blocked += 1;
    const reason = p.block_reason || "unknown";
    blockCounts.set(reason, (blockCounts.get(reason) || 0) + 1);
    closeApprovalEpisodes(approvalEpisodes, ticker);
  }

  let mode = null;
  let executionActive = false;
  let executionStartedPayload = null;
  let previousSnapshot = null;
  for (const row of execs) {
    const p = row.payload || {};
    if (row.event_type === "execution_started") {
      mode = p.mode || mode;
      executionStartedPayload = p;
      executionActive = true;
      continue;
    }
    if (row.event_type === "execution_stopped") {
      executionActive = false;
      continue;
    }
    if (row.event_type === "intent_claimed") {
      const claimTime = row.t;
      const ticker = p.ticker || "unknown";
      const side = normalizeSide(p.side);
      const decisionKey = approvalDecisionKey(ticker, side);
      const approvalSnapshot = latestApprovedByKey.get(decisionKey) || null;
      const pendingApprovalId = approvalEpisodes.get(decisionKey)
        || findPendingApprovalDecisionId(decisions, ticker, side, claimTime);
      const d = pendingApprovalId
        ? promoteApprovalDecision(decisions, pendingApprovalId, p.decision_id, { ticker, side })
        : ensureDecision(decisions, p.decision_id, { ticker, side });
      d.ticker = p.ticker || d.ticker;
      d.side = side;
      d.claimedAt = row.logged_at;
      d.status = mode === "live" ? "claimed" : "simulated";
      d.contracts = num(
        p.contracts ?? p.remaining_contracts_before_submit ?? p.desired_contracts,
        d.contracts ?? 1,
      );
      applyApprovalSnapshotToDecision(d, approvalSnapshot);
      if (d.cashRequired == null && p.limit_price_cents != null) {
        d.cashRequired = calculateRealizedCashMetrics(p.limit_price_cents, d.contracts).cashRequired;
      }
      d.refPx = p.model_limit_price_cents ?? d.refPx;
      d.maxPx = p.limit_price_cents ?? d.maxPx;
      closeApprovalEpisodes(approvalEpisodes, ticker, side);
      discardPendingApprovalDecisions(decisions, ticker, side, claimTime, p.decision_id);
      continue;
    }
    if (row.event_type === "submit_response" || row.event_type === "submit_retry_response") {
      const decisionId = p.decision_id;
      if (!decisionId) continue;
      const d = ensureDecision(decisions, decisionId, { ticker: p.ticker, side: normalizeSide(p.side) });
      const order = p.response?.order || {};
      d.status = order.status || "submitted";
      d.orderId = order.order_id || null;
      continue;
    }
    if (row.event_type === "simulated_position_settled" || row.event_type === "live_position_settled") {
      const d = ensureDecision(decisions, p.decision_id, { ticker: p.ticker, side: normalizeSide(p.side) });
      d.ticker = p.ticker || d.ticker;
      d.side = normalizeSide(p.side);
      d.status = "settled";
      d.settledAt = row.logged_at;
      d.result = p.settlement_result || null;
      d.contracts = num(p.contracts, 0);
      d.cashRequired = num(p.cash_required_dollars, 0);
      d.pnl = num(p.realized_pnl_dollars, 0);
      d.cumPnl = num(p.cumulative_realized_pnl_dollars, 0);
    }
  }

  const portfolio = execs
    .filter((r) => r.event_type === "portfolio_snapshot_full"
      || r.event_type === "portfolio_snapshot_balance"
      || r.event_type.startsWith("simulated_portfolio_")
      || r.event_type.startsWith("shadow_portfolio_")
      || r.event_type.startsWith("paper_portfolio_"))
    .map((r) => {
      const p = r.payload || {};
      const cash = num(p.available_cash_dollars, 0);
      const deployed = num(p.deployed_capital_dollars, 0);
      const snapshot = {
        time: r.t,
        label: r.logged_at,
        cash,
        deployed,
        equity: cash + deployed,
        open: normOpen(p.open_positions).length,
        pending: Array.isArray(p.pending_reservations) ? p.pending_reservations.length : 0,
        openRaw: p.open_positions || [],
        pendingRaw: p.pending_reservations || [],
      };
      assignSnapshotDeltasToClaims(decisions, previousSnapshot, snapshot);
      previousSnapshot = snapshot;
      return snapshot;
    });

  const settled = [...decisions.values()]
    .filter((d) => d.settledAt)
    .sort((a, b) => new Date(b.settledAt) - new Date(a.settledAt));
  const wins = settled.filter((d) => num(d.pnl, 0) > 0);
  const losses = settled.filter((d) => num(d.pnl, 0) < 0);
  const flats = settled.filter((d) => num(d.pnl, 0) === 0);
  const grossWin = wins.reduce((sum, d) => sum + num(d.pnl, 0), 0);
  const grossLoss = losses.reduce((sum, d) => sum + Math.abs(num(d.pnl, 0)), 0);
  const realized = settled.reduce((sum, d) => sum + num(d.pnl, 0), 0);
  const expectancy = settled.length ? realized / settled.length : 0;
  const avgWin = wins.length ? grossWin / wins.length : 0;
  const avgLoss = losses.length ? losses.reduce((sum, d) => sum + num(d.pnl, 0), 0) / losses.length : 0;
  const bestTrade = wins.reduce((best, d) => (!best || num(d.pnl, 0) > num(best.pnl, 0) ? d : best), null);
  const worstTrade = losses.reduce((worst, d) => (!worst || num(d.pnl, 0) < num(worst.pnl, 0) ? d : worst), null);
  const profitFactor = grossLoss > 0 ? grossWin / grossLoss : grossWin > 0 ? Number.POSITIVE_INFINITY : 0;
  const initialEquity = portfolio[0]?.equity ?? num(signalStart.starting_cash_dollars, 0);
  const tickerMap = new Map();
  const sideStats = createSideStats();
  const tauRows = createTauRows();
  const tauMap = new Map(tauRows.map((row) => [row.label, row]));
  const timeline = [];

  for (const d of decisions.values()) {
    const tickerKey = d.ticker || "unknown";
    const tickerStats = tickerMap.get(tickerKey) || {
      ticker: tickerKey,
      approved: 0,
      claimed: 0,
      settled: 0,
      wins: 0,
      losses: 0,
      open: 0,
      realized: 0,
    };

    const side = normalizeSide(d.side);
    const sideRow = sideStats[side] || sideStats.UNKNOWN;
    const tauLabel = tauBucketLabel(d.tau);
    const tauRow = tauLabel ? tauMap.get(tauLabel) : null;

    if (d.approvedAt) {
      tickerStats.approved += 1;
      sideRow.approved += 1;
      if (tauRow) tauRow.approved += 1;
    }
    if (d.claimedAt) {
      tickerStats.claimed += 1;
      sideRow.claimed += 1;
      if (tauRow) tauRow.claimed += 1;
      if (!d.settledAt) {
        timeline.push({
          time: d.claimedAt,
          type: "claimed",
          ticker: d.ticker,
          side: d.side,
          status: d.status,
          result: null,
          pnl: null,
        });
      }
    }
    if (d.settledAt) {
      tickerStats.settled += 1;
      tickerStats.realized += num(d.pnl, 0);
      sideRow.settled += 1;
      sideRow.realized += num(d.pnl, 0);
      if (tauRow) {
        tauRow.settled += 1;
        tauRow.realized += num(d.pnl, 0);
      }
      if (num(d.pnl, 0) > 0) {
        tickerStats.wins += 1;
        sideRow.wins += 1;
        if (tauRow) tauRow.wins += 1;
      } else if (num(d.pnl, 0) < 0) {
        tickerStats.losses += 1;
        sideRow.losses += 1;
        if (tauRow) tauRow.losses += 1;
      }
      timeline.push({
        time: d.settledAt,
        type: "settled",
        ticker: d.ticker,
        side: d.side,
        status: d.status,
        result: d.result,
        pnl: d.pnl,
      });
    }

    tickerMap.set(tickerKey, tickerStats);
  }

  for (const row of tauRows) {
    row.winRate = row.settled ? row.wins / row.settled : 0;
    row.expectancy = row.settled ? row.realized / row.settled : 0;
  }

  const decisionValues = [...decisions.values()];
  const approved = decisionValues.filter((d) => d.approvedAt).length;
  const claimCount = decisionValues.filter((d) => d.claimedAt).length;
  const latestSnapshot = portfolio.at(-1) || null;
  const shouldUseDerivedLatest = ["paper", "shadow"].includes(String(mode || "").toLowerCase());
  const derivedLatest = deriveLatestPortfolio(decisionValues, realized, initialEquity, latestSnapshot);
  const latest = shouldUseDerivedLatest
    ? derivedLatest
    : (latestSnapshot || derivedLatest);
  const portfolioForDisplay = (() => {
    if (!shouldUseDerivedLatest) return portfolio;
    if (!latestSnapshot) return [latest];
    const materiallyDifferent = (
      latestSnapshot.open !== latest.open
      || Math.abs(num(latestSnapshot.deployed, 0) - num(latest.deployed, 0)) > 1e-9
      || Math.abs(num(latestSnapshot.equity, 0) - num(latest.equity, 0)) > 1e-9
    );
    return materiallyDifferent ? [...portfolio, latest] : portfolio;
  })();

  let peak = portfolioForDisplay[0]?.equity || 0;
  let maxDd = 0;
  let maxDdPct = 0;
  for (const pt of portfolioForDisplay) {
    peak = Math.max(peak, pt.equity);
    if (peak > 0) {
      const dd = peak - pt.equity;
      maxDd = Math.max(maxDd, dd);
      maxDdPct = Math.max(maxDdPct, dd / peak);
    }
  }
  const paperObjective = realized / Math.max(1, maxDd);
  const latestEquity = latest.equity ?? initialEquity;
  const equityDelta = latestEquity - initialEquity;

  for (const row of exposure(latest.openRaw)) {
    const tickerStats = tickerMap.get(row.ticker) || {
      ticker: row.ticker,
      approved: 0,
      claimed: 0,
      settled: 0,
      wins: 0,
      losses: 0,
      open: 0,
      realized: 0,
    };
    tickerStats.open = row.count;
    tickerMap.set(row.ticker, tickerStats);
  }

  const sideSummary = ["YES", "NO"].map((side) => {
    const stats = sideStats[side];
    return {
      side,
      approved: stats.approved,
      claimed: stats.claimed,
      settled: stats.settled,
      wins: stats.wins,
      losses: stats.losses,
      winRate: stats.settled ? stats.wins / stats.settled : 0,
      realized: stats.realized,
      expectancy: stats.settled ? stats.realized / stats.settled : 0,
    };
  });

  timeline.sort((a, b) => new Date(b.time) - new Date(a.time));

  return {
    signalStart,
    latestSignal,
    mode,
    executionActive,
    executionStartedPayload,
    approved,
    blocked,
    claimCount,
    blockCounts: [...blockCounts.entries()].sort((a, b) => b[1] - a[1]),
    blockedSpread: num(blockCounts.get("spread_destroyed_edge"), 0),
    blockedMissingQuote: num(blockCounts.get("missing_quote"), 0),
    blockedStaleQuote: num(blockCounts.get("stale_quote"), 0),
    blockedCrossedQuote: num(blockCounts.get("crossed_quote"), 0),
    recent: decisionValues
      .sort((a, b) => new Date(b.claimedAt || b.approvedAt || 0) - new Date(a.claimedAt || a.approvedAt || 0))
      .slice(0, 40),
    settled,
    wins,
    losses,
    flats,
    grossWin,
    grossLoss,
    profitFactor,
    realized,
    paperObjective,
    expectancy,
    avgWin,
    avgLoss,
    initialEquity,
    equityDelta,
    bestTrade,
    worstTrade,
    maxDd,
    maxDdPct,
    portfolio: portfolioForDisplay,
    latest,
    openExposure: buildOpenExposure(latest.openRaw, decisions),
    tickerBreakdown: [...tickerMap.values()].sort((a, b) => b.realized - a.realized),
    timeline: timeline.slice(0, 80),
    sideSummary,
    tauBreakdown: tauRows,
  };
}

function aggregateTradingSummary(summaryMap) {
  const entries = Object.entries(summaryMap);
  if (!entries.length) return null;

  let realized = 0;
  let equity = 0;
  let open = 0;
  let claimed = 0;
  let settled = 0;
  let worstDd = 0;
  let worstDdModel = "n/a";
  const blockCounts = new Map();

  for (const [model, summary] of entries) {
    realized += num(summary.realized, 0);
    equity += num(summary.latest?.equity, 0);
    open += num(summary.latest?.open, 0);
    claimed += num(summary.claimCount, 0);
    settled += summary.settled.length;
    if (num(summary.maxDd, 0) >= worstDd) {
      worstDd = num(summary.maxDd, 0);
      worstDdModel = model;
    }
    for (const [reason, count] of summary.blockCounts || []) {
      blockCounts.set(reason, (blockCounts.get(reason) || 0) + num(count, 0));
    }
  }

  const [topBlockReason, topBlockCount] = [...blockCounts.entries()].sort((a, b) => b[1] - a[1])[0] || [null, 0];
  return {
    modelCount: entries.length,
    realized,
    equity,
    open,
    claimed,
    settled,
    worstDd,
    worstDdModel,
    topBlockReason,
    topBlockCount,
  };
}

function renderTradingRunOverview(aggregate) {
  if (!aggregate) {
    setTextIfPresent("run-overview-realized", "$0.00");
    setTextIfPresent("run-overview-realized-note", "No loaded trades yet");
    setTextIfPresent("run-overview-equity", "$0.00");
    setTextIfPresent("run-overview-equity-note", "Waiting for portfolio snapshots");
    setTextIfPresent("run-overview-drawdown", "$0.00");
    setTextIfPresent("run-overview-drawdown-note", "Worst model drawdown");
    setTextIfPresent("run-overview-open", "0");
    setTextIfPresent("run-overview-open-note", "Open slots in latest snapshots");
    setTextIfPresent("run-overview-flow", "0 / 0");
    setTextIfPresent("run-overview-flow-note", "Execution flow through the run");
    setTextIfPresent("run-overview-block", "n/a");
    setTextIfPresent("run-overview-block-note", "No blocked signals yet");
    return;
  }

  setTextIfPresent("run-overview-realized", fmtMoney(aggregate.realized));
  setTextIfPresent("run-overview-realized-note", `Across ${fmtCount(aggregate.modelCount)} model lanes`);
  setTextIfPresent("run-overview-equity", fmtMoney(aggregate.equity));
  setTextIfPresent("run-overview-equity-note", "Summed latest equity snapshots");
  setTextIfPresent("run-overview-drawdown", fmtMoney(aggregate.worstDd));
  setTextIfPresent("run-overview-drawdown-note", `${aggregate.worstDdModel} worst max DD`);
  setTextIfPresent("run-overview-open", fmtCount(aggregate.open));
  setTextIfPresent("run-overview-open-note", "Open slots in latest snapshots");
  setTextIfPresent("run-overview-flow", `${fmtCount(aggregate.claimed)} / ${fmtCount(aggregate.settled)}`);
  setTextIfPresent("run-overview-flow-note", "Claimed vs settled intents");
  setTextIfPresent("run-overview-block", aggregate.topBlockReason || "n/a");
  setTextIfPresent(
    "run-overview-block-note",
    aggregate.topBlockReason ? `${fmtCount(aggregate.topBlockCount)} blocked decisions` : "No blocked signals yet",
  );
}

function equitySvg(points) {
  if (!points.length) {
    return `<text x="50%" y="50%" text-anchor="middle" fill="rgba(22,34,43,.45)" font-size="18">No portfolio snapshots yet</text>`;
  }
  const w = 800;
  const h = 260;
  const p = { t: 18, r: 18, b: 28, l: 54 };
  const values = points.map((x) => x.equity);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(0.0001, max - min);
  const path = points.map((pt, i) => {
    const x = p.l + (i / Math.max(1, points.length - 1)) * (w - p.l - p.r);
    const y = p.t + (1 - ((pt.equity - min) / span)) * (h - p.t - p.b);
    return `${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
  }).join(" ");
  const area = `${path} L ${w - p.r} ${h - p.b} L ${p.l} ${h - p.b} Z`;
  let grid = "";
  for (let i = 0; i <= 4; i += 1) {
    const y = p.t + (i / 4) * (h - p.t - p.b);
    const value = max - (i / 4) * span;
    grid += `<line x1="${p.l}" y1="${y}" x2="${w - p.r}" y2="${y}" stroke="rgba(22,34,43,.12)" stroke-dasharray="4 6"></line>`;
    grid += `<text x="${p.l - 8}" y="${y + 4}" text-anchor="end" fill="rgba(22,34,43,.55)" font-size="12">${fmtMoney(value)}</text>`;
  }
  return `<defs><linearGradient id="eqFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stop-color="rgba(15,118,110,.3)"></stop><stop offset="100%" stop-color="rgba(15,118,110,.03)"></stop></linearGradient></defs>${grid}<path d="${area}" fill="url(#eqFill)"></path><path d="${path}" fill="none" stroke="#0f766e" stroke-width="3"></path>`;
}

function blockSvg(items) {
  if (!items.length) {
    return `<text x="50%" y="50%" text-anchor="middle" fill="rgba(22,34,43,.45)" font-size="18">No blocked signals yet</text>`;
  }
  const top = items.slice(0, 6);
  const w = 420;
  const p = { t: 18, r: 12, l: 132 };
  const barH = 22;
  const gap = 12;
  const max = Math.max(...top.map((x) => x[1]), 1);
  return top.map(([reason, count], i) => {
    const y = p.t + i * (barH + gap);
    const barWidth = ((w - p.l - p.r) * count) / max;
    return `<text x="${p.l - 10}" y="${y + 15}" text-anchor="end" fill="rgba(22,34,43,.72)" font-size="12">${escapeHtml(reason)}</text><rect x="${p.l}" y="${y}" width="${barWidth}" height="${barH}" rx="11" fill="rgba(154,103,0,.72)"></rect><text x="${p.l + barWidth + 8}" y="${y + 15}" fill="rgba(22,34,43,.72)" font-size="12">${count}</text>`;
  }).join("");
}

function comparisonRows(summaryMap, metaMap) {
  return Object.entries(summaryMap).map(([model, summary]) => ({
    model,
    summary,
    meta: metaMap[model] || { execFiles: 0, signalFiles: 0 },
    isLegacy: model === "single",
  }));
}

function comparisonValue(row, key) {
  switch (key) {
    case "model":
      return row.model;
    case "paperObjective":
      return row.summary.paperObjective;
    case "realized":
      return row.summary.realized;
    case "equity":
      return row.summary.latest.equity;
    case "settled":
      return row.summary.settled.length;
    case "winRate":
      return row.summary.settled.length ? row.summary.wins.length / row.summary.settled.length : 0;
    case "expectancy":
      return row.summary.expectancy;
    case "maxDd":
      return row.summary.maxDd;
    default:
      return 0;
  }
}

function sortRows(rows) {
  const { key, dir } = state.comparisonSort;
  return rows.sort((a, b) => {
    const av = comparisonValue(a, key);
    const bv = comparisonValue(b, key);
    let delta = 0;
    if (typeof av === "string" || typeof bv === "string") {
      delta = String(av).localeCompare(String(bv));
    } else {
      delta = num(av, 0) - num(bv, 0);
    }
    if (delta === 0) delta = a.model.localeCompare(b.model);
    return dir === "desc" ? -delta : delta;
  });
}

function rankedRows(summaryMap, metaMap) {
  return sortRows(comparisonRows(summaryMap, metaMap));
}

function rankedModelNames(summaryMap, metaMap) {
  return rankedRows(summaryMap, metaMap).map((row) => row.model);
}

function renderComparison(summaryMap, metaMap) {
  const rows = rankedRows(summaryMap, metaMap);
  const wrap = $("model-comparison-wrap");
  if (!rows.length) {
    wrap.innerHTML = `<div class="empty">No model logs found yet. Start the paper runner, then choose the root output folder.</div>`;
    return;
  }

  const head = COMPARISON_COLUMNS.map((column) => {
    if (!column.sortable) return `<th>${escapeHtml(column.label)}</th>`;
    const active = state.comparisonSort.key === column.key;
    const arrow = active ? (state.comparisonSort.dir === "desc" ? "↓" : "↑") : "";
    return `<th><button type="button" class="sort-button" data-sort-key="${column.key}">${escapeHtml(column.label)} ${arrow}</button></th>`;
  }).join("");

  const body = rows.map((row) => {
    const summary = row.summary;
    const yesClaimed = summary.sideSummary.find((item) => item.side === "YES")?.claimed || 0;
    const noClaimed = summary.sideSummary.find((item) => item.side === "NO")?.claimed || 0;
    const winRate = summary.settled.length ? summary.wins.length / summary.settled.length : 0;
    const selected = row.model === state.selectedModel ? " selected-row" : "";
    return `<tr class="row-selectable${selected}" data-model-row="${escapeHtml(row.model)}">
      <td><span class="tag">${escapeHtml(row.model)}</span>${row.isLegacy ? ' <span class="tag legacy">legacy</span>' : ""}</td>
      <td><span class="${ratioTone(summary.paperObjective)}">${fmtRatio(summary.paperObjective)}</span></td>
      <td><span class="${moneyTone(summary.realized)}">${fmtMoney(summary.realized)}</span></td>
      <td><span class="${moneyTone(summary.equityDelta)}">${fmtMoney(summary.latest.equity)}</span></td>
      <td>${fmtCount(summary.settled.length)}</td>
      <td>Y ${fmtCount(yesClaimed)} / N ${fmtCount(noClaimed)}</td>
      <td>${fmtPct(winRate)}</td>
      <td><span class="${moneyTone(summary.expectancy)}">${fmtMoney(summary.expectancy)}</span></td>
      <td><span class="${moneyTone(summary.avgWin)}">${fmtMoney(summary.avgWin)}</span></td>
      <td><span class="${moneyTone(summary.avgLoss)}">${fmtMoney(summary.avgLoss)}</span></td>
      <td><span class="${moneyTone(summary.bestTrade ? summary.bestTrade.pnl : 0)}">${fmtMoney(summary.bestTrade ? summary.bestTrade.pnl : 0)}</span></td>
      <td><span class="${moneyTone(summary.worstTrade ? summary.worstTrade.pnl : 0)}">${fmtMoney(summary.worstTrade ? summary.worstTrade.pnl : 0)}</span></td>
      <td><span class="${moneyTone(-summary.maxDd)}">${fmtMoney(summary.maxDd)}</span></td>
      <td>${fmtCount(summary.latest.open)}</td>
      <td>${fmtCount(row.meta.signalFiles)} signal / ${fmtCount(row.meta.execFiles)} exec</td>
    </tr>`;
  }).join("");

  wrap.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  wrap.querySelectorAll("[data-sort-key]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.sortKey;
      const column = COMPARISON_COLUMNS.find((item) => item.key === key);
      if (!column) return;
      if (state.comparisonSort.key === key) {
        state.comparisonSort.dir = state.comparisonSort.dir === "desc" ? "asc" : "desc";
      } else {
        state.comparisonSort = { key, dir: column.defaultDir || "desc" };
      }
      rerender();
    });
  });
  wrap.querySelectorAll("[data-model-row]").forEach((rowEl) => {
    rowEl.addEventListener("click", () => {
      state.selectedModel = rowEl.dataset.modelRow || "";
      $("model-select").value = state.selectedModel;
      rerender();
    });
  });
}

function renderModelCards(summaryMap, metaMap) {
  const rows = rankedRows(summaryMap, metaMap);
  const wrap = $("model-summary-grid");
  if (!rows.length) {
    wrap.innerHTML = "";
    return;
  }
  wrap.innerHTML = rows.map((row) => {
    const summary = row.summary;
    const selected = row.model === state.selectedModel ? " selected" : "";
    const yesClaimed = summary.sideSummary.find((item) => item.side === "YES")?.claimed || 0;
    const noClaimed = summary.sideSummary.find((item) => item.side === "NO")?.claimed || 0;
    const winRate = summary.settled.length ? summary.wins.length / summary.settled.length : 0;
    return `<div class="model-card${selected}" data-model-card="${escapeHtml(row.model)}">
      <div class="model-card-head">
        <div>
          <h3>${escapeHtml(row.model)}</h3>
          <div class="muted">${fmtCount(summary.settled.length)} settled | ${fmtCount(summary.latest.open)} open</div>
        </div>
        <div class="badge-row">
          ${row.isLegacy ? '<span class="badge legacy">legacy</span>' : ""}
          <span class="badge">${fmtCount(row.meta.signalFiles)} signal / ${fmtCount(row.meta.execFiles)} exec</span>
        </div>
      </div>
      <div>
        <div class="big ${moneyTone(summary.realized)}">${fmtMoney(summary.realized)}</div>
        <div class="muted">Realized PnL</div>
      </div>
      ${statsGridHtml([
        { label: "Objective", value: fmtRatio(summary.paperObjective), tone: ratioTone(summary.paperObjective) },
        { label: "Claimed Y / N", value: `Y ${fmtCount(yesClaimed)} / N ${fmtCount(noClaimed)}` },
        { label: "Win Rate", value: fmtPct(winRate) },
        { label: "Expectancy", value: fmtMoney(summary.expectancy), tone: moneyTone(summary.expectancy) },
        { label: "Avg Win", value: fmtMoney(summary.avgWin), tone: moneyTone(summary.avgWin) },
        { label: "Avg Loss", value: fmtMoney(summary.avgLoss), tone: moneyTone(summary.avgLoss) },
        { label: "Largest Win", value: fmtMoney(summary.bestTrade ? summary.bestTrade.pnl : 0), tone: moneyTone(summary.bestTrade ? summary.bestTrade.pnl : 0) },
        { label: "Largest Loss", value: fmtMoney(summary.worstTrade ? summary.worstTrade.pnl : 0), tone: moneyTone(summary.worstTrade ? summary.worstTrade.pnl : 0) },
        { label: "Max DD", value: fmtMoney(summary.maxDd), tone: moneyTone(-summary.maxDd) },
        { label: "Equity", value: fmtMoney(summary.latest.equity), tone: moneyTone(summary.equityDelta), sub: `vs start ${fmtMoney(summary.initialEquity)}` },
      ])}
    </div>`;
  }).join("");
  wrap.querySelectorAll("[data-model-card]").forEach((card) => {
    card.addEventListener("click", () => {
      state.selectedModel = card.dataset.modelCard || "";
      $("model-select").value = state.selectedModel;
      rerender();
    });
  });
}

function updateModelSelector(summaryMap, metaMap) {
  const names = rankedModelNames(summaryMap, metaMap);
  const select = $("model-select");
  if (!names.length) {
    select.innerHTML = `<option value="">auto</option>`;
    state.selectedModel = "";
    return null;
  }
  if (!names.includes(state.selectedModel)) state.selectedModel = names[0];
  select.innerHTML = names.map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("");
  select.value = state.selectedModel;
  return state.selectedModel;
}

function renderFocusedHeader(modelName, summary, meta) {
  const winRate = summary.settled.length ? summary.wins.length / summary.settled.length : 0;
  $("focused-model-title").textContent = `Focused Model: ${modelName}`;
  $("focused-model-subtitle").textContent = `${modelName} selected-model detail. The tabs below show overview, side and TAU breakdowns, trade logs, and live diagnostics for this model only.`;
  $("focused-model-badges").innerHTML = [
    `<span class="badge ${ratioTone(summary.paperObjective)}">objective ${fmtRatio(summary.paperObjective)}</span>`,
    `<span class="badge ${moneyTone(summary.realized)}">realized ${fmtMoney(summary.realized)}</span>`,
    `<span class="badge">win rate ${fmtPct(winRate)}</span>`,
    `<span class="badge">logs ${fmtCount(meta.signalFiles)} signal / ${fmtCount(meta.execFiles)} exec</span>`,
    modelName === "single" ? '<span class="badge legacy">legacy mixed-layout data</span>' : "",
  ].filter(Boolean).join("");
}

function sideBreakdownRows(summary) {
  const totalApproved = summary.sideSummary.reduce((sum, row) => sum + row.approved, 0);
  const totalClaimed = summary.sideSummary.reduce((sum, row) => sum + row.claimed, 0);
  const totalSettled = summary.sideSummary.reduce((sum, row) => sum + row.settled, 0);
  const totalWins = summary.sideSummary.reduce((sum, row) => sum + row.wins, 0);
  const totalLosses = summary.sideSummary.reduce((sum, row) => sum + row.losses, 0);
  const totalRealized = summary.sideSummary.reduce((sum, row) => sum + row.realized, 0);
  const totalExpectancy = totalSettled ? totalRealized / totalSettled : 0;
  return [
    ...summary.sideSummary,
    {
      side: "TOTAL",
      approved: totalApproved,
      claimed: totalClaimed,
      settled: totalSettled,
      wins: totalWins,
      losses: totalLosses,
      winRate: totalSettled ? totalWins / totalSettled : 0,
      realized: totalRealized,
      expectancy: totalExpectancy,
    },
  ];
}

function overviewHtml(modelName, summary, meta) {
  const yesSide = summary.sideSummary.find((row) => row.side === "YES") || { claimed: 0, settled: 0, realized: 0 };
  const noSide = summary.sideSummary.find((row) => row.side === "NO") || { claimed: 0, settled: 0, realized: 0 };
  const overviewMetrics = statsGridHtml([
    { label: "Realized PnL", value: fmtMoney(summary.realized), tone: moneyTone(summary.realized), sub: `${fmtCount(summary.settled.length)} settled trades` },
    { label: "Objective", value: fmtRatio(summary.paperObjective), tone: ratioTone(summary.paperObjective), sub: "Realized PnL / max(1, drawdown)" },
    { label: "Equity", value: fmtMoney(summary.latest.equity), tone: moneyTone(summary.equityDelta), sub: `Started at ${fmtMoney(summary.initialEquity)}` },
    { label: "Win Rate", value: fmtPct(summary.settled.length ? summary.wins.length / summary.settled.length : 0), sub: `${fmtCount(summary.wins.length)} wins / ${fmtCount(summary.losses.length)} losses` },
    { label: "Max Drawdown", value: fmtMoney(summary.maxDd), tone: moneyTone(-summary.maxDd), sub: `${fmtPct(summary.maxDdPct, 2)} of peak` },
    { label: "Approved", value: fmtCount(summary.approved), sub: `${fmtCount(summary.blocked)} blocked signals` },
    { label: "Claimed", value: fmtCount(summary.claimCount), sub: "Execution-picked trade intents" },
    { label: "Open", value: fmtCount(summary.latest.open), sub: `${fmtCount(summary.openExposure.length)} exposure buckets` },
  ], "detail-metrics");

  const sideCards = statsGridHtml([
    { label: "Claimed YES", value: fmtCount(yesSide.claimed), sub: `${fmtCount(yesSide.settled)} settled YES trades` },
    { label: "Claimed NO", value: fmtCount(noSide.claimed), sub: `${fmtCount(noSide.settled)} settled NO trades` },
    { label: "YES Realized PnL", value: fmtMoney(yesSide.realized), tone: moneyTone(yesSide.realized) },
    { label: "NO Realized PnL", value: fmtMoney(noSide.realized), tone: moneyTone(noSide.realized) },
  ]);

  const riskCards = statsGridHtml([
    { label: "Expectancy", value: fmtMoney(summary.expectancy), tone: moneyTone(summary.expectancy), sub: "Average realized PnL per settled trade" },
    { label: "Profit Factor", value: fmtRatio(summary.profitFactor), tone: ratioTone(summary.profitFactor - 1), sub: `${fmtMoney(summary.grossWin)} gross wins / ${fmtMoney(-summary.grossLoss)} gross losses` },
    { label: "Average Win", value: fmtMoney(summary.avgWin), tone: moneyTone(summary.avgWin) },
    { label: "Average Loss", value: fmtMoney(summary.avgLoss), tone: moneyTone(summary.avgLoss) },
    { label: "Largest Win", value: fmtMoney(summary.bestTrade ? summary.bestTrade.pnl : 0), tone: moneyTone(summary.bestTrade ? summary.bestTrade.pnl : 0), sub: summary.bestTrade ? `${summary.bestTrade.ticker} • ${fmtDate(summary.bestTrade.settledAt)}` : "No winning settlements yet" },
    { label: "Largest Loss", value: fmtMoney(summary.worstTrade ? summary.worstTrade.pnl : 0), tone: moneyTone(summary.worstTrade ? summary.worstTrade.pnl : 0), sub: summary.worstTrade ? `${summary.worstTrade.ticker} • ${fmtDate(summary.worstTrade.settledAt)}` : "No losing settlements yet" },
    { label: "Available Cash", value: fmtMoney(summary.latest.cash), tone: moneyTone(summary.latest.cash - summary.initialEquity), sub: `Snapshot ${fmtDate(summary.latest.label)}` },
    { label: "Pending Reservations", value: fmtCount(summary.latest.pending), sub: `${modelName} current pending reservations` },
  ]);

  return `
    <div class="detail-grid-two">
      ${panelSection("Overview KPIs", `${modelName} top-line paper performance and position state.`, overviewMetrics)}
      ${panelSection("Side Summary", `${modelName} claimed and settled side mix.`, sideCards)}
    </div>
    <div class="detail-grid-two">
      ${panelSection("Equity Curve", `${modelName} cash plus deployed capital over time.`, `<div class="chart"><svg viewBox="0 0 800 260" preserveAspectRatio="none">${equitySvg(summary.portfolio)}</svg></div><div class="legend-note">${escapeHtml(modelName)} latest equity ${fmtMoney(summary.latest.equity)} • logs ${fmtCount(meta.signalFiles)} signal / ${fmtCount(meta.execFiles)} exec</div>`)}
      ${panelSection("Risk & Trade Quality", `${modelName} average wins, losses, and drawdown-aware quality metrics.`, riskCards)}
    </div>
  `;
}

function breakdownsHtml(modelName, summary) {
  const sideTable = tableHtml([
    { label: "Side", render: (row) => row.side === "TOTAL" ? "<strong>TOTAL</strong>" : `<span class="tag">${escapeHtml(row.side)}</span>` },
    { label: "Approved", render: (row) => fmtCount(row.approved) },
    { label: "Claimed", render: (row) => fmtCount(row.claimed) },
    { label: "Settled", render: (row) => fmtCount(row.settled) },
    { label: "Wins", render: (row) => fmtCount(row.wins) },
    { label: "Losses", render: (row) => fmtCount(row.losses) },
    { label: "Win Rate", render: (row) => fmtPct(row.winRate) },
    { label: "Realized PnL", render: (row) => `<span class="${moneyTone(row.realized)}">${fmtMoney(row.realized)}</span>` },
    { label: "Expectancy", render: (row) => `<span class="${moneyTone(row.expectancy)}">${fmtMoney(row.expectancy)}</span>` },
  ], sideBreakdownRows(summary), `No side breakdown available yet for ${modelName}.`);

  const tauActivity = tableHtml([
    { label: "TAU Bucket", render: (row) => `<span class="tag">${escapeHtml(row.label)}</span>` },
    { label: "Approved", render: (row) => fmtCount(row.approved) },
    { label: "Claimed", render: (row) => fmtCount(row.claimed) },
    { label: "Claim Rate", render: (row) => row.approved ? fmtPct(row.claimed / row.approved) : "0.0%" },
  ], summary.tauBreakdown, `No TAU activity yet for ${modelName}.`);

  const tauSettled = tableHtml([
    { label: "TAU Bucket", render: (row) => `<span class="tag">${escapeHtml(row.label)}</span>` },
    { label: "Settled", render: (row) => fmtCount(row.settled) },
    { label: "Wins", render: (row) => fmtCount(row.wins) },
    { label: "Losses", render: (row) => fmtCount(row.losses) },
    { label: "Win Rate", render: (row) => fmtPct(row.winRate) },
    { label: "Realized PnL", render: (row) => `<span class="${moneyTone(row.realized)}">${fmtMoney(row.realized)}</span>` },
    { label: "Expectancy", render: (row) => `<span class="${moneyTone(row.expectancy)}">${fmtMoney(row.expectancy)}</span>` },
  ], summary.tauBreakdown, `No settled TAU buckets yet for ${modelName}.`);

  return `
    <div class="detail-grid-two">
      ${panelSection("YES / NO Breakdown", `${modelName} approved, claimed, and settled performance by side.`, sideTable)}
      ${panelSection("Activity by TAU", `${modelName} approved and claimed trade activity in fixed 2-minute TAU buckets.`, tauActivity)}
    </div>
    ${panelSection("Settled PnL by TAU", `${modelName} realized performance across TAU buckets, ordered chronologically.`, tauSettled)}
  `;
}

function tradesHtml(modelName, summary) {
  const recentTrades = tableHtml([
    { label: "Time", render: (row) => fmtDate(row.claimedAt || row.approvedAt) },
    { label: "Ticker", render: (row) => `<span class="mono">${escapeHtml(row.ticker || "n/a")}</span>` },
    { label: "Side", render: (row) => `<span class="tag">${escapeHtml(row.side || "?")}</span>` },
    { label: "TAU", render: (row) => row.tau == null ? "n/a" : `${num(row.tau, 0).toFixed(2)}m` },
    { label: "Status", render: (row) => `<span class="tag">${escapeHtml(row.status || "approved")}</span>` },
    { label: "YES / NO Edge", render: (row) => `${row.yesEdge == null ? "n/a" : fmtPct(row.yesEdge, 2)} / ${row.noEdge == null ? "n/a" : fmtPct(row.noEdge, 2)}` },
    { label: "Ref Px", render: (row) => row.refPx == null ? "n/a" : `${row.refPx}c` },
    { label: "PnL", render: (row) => row.pnl == null ? "open" : `<span class="${moneyTone(row.pnl)}">${fmtMoney(row.pnl)}</span>` },
  ], summary.recent, `No approved or claimed trades yet for ${modelName}.`, 980);

  const openExposure = tableHtml([
    { label: "Ticker", render: (row) => `<span class="mono">${escapeHtml(row.ticker)}</span>` },
    { label: "Side", render: (row) => `<span class="tag">${escapeHtml(row.side)}</span>` },
    { label: "Slots", render: (row) => fmtCount(row.count) },
    { label: "Contracts", render: (row) => fmtCount(row.contracts) },
    { label: "Cash", render: (row) => fmtMoney(row.cash) },
  ], summary.openExposure, `No open exposure in the latest snapshot for ${modelName}.`);

  const settlements = tableHtml([
    { label: "Settled", render: (row) => fmtDate(row.settledAt) },
    { label: "Ticker", render: (row) => `<span class="mono">${escapeHtml(row.ticker || "n/a")}</span>` },
    { label: "Side", render: (row) => `<span class="tag">${escapeHtml(row.side || "?")}</span>` },
    { label: "Result", render: (row) => row.result ? `<span class="tag">${escapeHtml(row.result)}</span>` : "n/a" },
    { label: "Contracts", render: (row) => fmtCount(row.contracts) },
    { label: "Cash Required", render: (row) => fmtMoney(row.cashRequired) },
    { label: "PnL", render: (row) => `<span class="${moneyTone(row.pnl)}">${fmtMoney(row.pnl)}</span>` },
    { label: "Cum PnL", render: (row) => row.cumPnl == null ? "n/a" : `<span class="${moneyTone(row.cumPnl)}">${fmtMoney(row.cumPnl)}</span>` },
  ], summary.settled, `No settled trades yet for ${modelName}.`, 1040);

  const tickerBreakdown = tableHtml([
    { label: "Ticker", render: (row) => `<span class="mono">${escapeHtml(row.ticker)}</span>` },
    { label: "Approved", render: (row) => fmtCount(row.approved) },
    { label: "Claimed", render: (row) => fmtCount(row.claimed) },
    { label: "Settled", render: (row) => fmtCount(row.settled) },
    { label: "Wins", render: (row) => fmtCount(row.wins) },
    { label: "Losses", render: (row) => fmtCount(row.losses) },
    { label: "Open", render: (row) => fmtCount(row.open) },
    { label: "Realized PnL", render: (row) => `<span class="${moneyTone(row.realized)}">${fmtMoney(row.realized)}</span>` },
  ], summary.tickerBreakdown, `No per-ticker trade data yet for ${modelName}.`);

  const timeline = tableHtml([
    { label: "Time", render: (row) => fmtDate(row.time) },
    { label: "Type", render: (row) => `<span class="tag">${escapeHtml(row.type)}</span>` },
    { label: "Ticker", render: (row) => `<span class="mono">${escapeHtml(row.ticker || "n/a")}</span>` },
    { label: "Side", render: (row) => `<span class="tag">${escapeHtml(row.side || "?")}</span>` },
    { label: "Result", render: (row) => row.result ? `<span class="tag">${escapeHtml(row.result)}</span>` : "n/a" },
    { label: "PnL", render: (row) => row.pnl == null ? "n/a" : `<span class="${moneyTone(row.pnl)}">${fmtMoney(row.pnl)}</span>` },
  ], summary.timeline, `No trade timeline yet for ${modelName}.`, 960);

  return `
    <div class="detail-grid-two">
      ${panelSection("Recent Trades", `${modelName} approved and claimed decisions, newest first.`, recentTrades)}
      ${panelSection("Open Exposure", `${modelName} latest portfolio exposure by ticker and side.`, openExposure)}
    </div>
    <div class="detail-grid-two">
      ${panelSection("Settlements", `${modelName} completed paper trades with realized PnL.`, settlements)}
      ${panelSection("Per-Ticker Breakdown", `${modelName} trade counts and realized PnL by contract ticker.`, tickerBreakdown)}
    </div>
    ${panelSection("Trade Timeline", `${modelName} claimed and settled timeline, newest first.`, timeline)}
  `;
}

function diagnosticsHtml(modelName, summary, meta) {
  const latestSignal = summary.latestSignal;
  const startMeta = summary.executionStartedPayload || {};
  const signalHealth = statsGridHtml([
    { label: "Execution Mode", value: summary.mode || "unknown", tone: summary.mode === "live" ? "warn" : "good", sub: summary.executionActive ? "Execution loop active in logs" : "Latest observed execution mode" },
    { label: "Subaccount", value: startMeta.subaccount == null ? "n/a" : String(startMeta.subaccount), sub: "Configured execution subaccount" },
    { label: "Edge Threshold", value: `${num(summary.signalStart.edge_threshold_cents, 0).toFixed(1)}c` },
    { label: "Tau Gate", value: `${num(summary.signalStart.min_tau_minutes, 0)}-${num(summary.signalStart.max_tau_minutes, 0)}m` },
    { label: "Stacking", value: summary.signalStart.allow_stacking ? "on" : "off" },
    { label: "Spread Blocks", value: fmtCount(summary.blockedSpread), sub: "Edge destroyed by executable spread" },
    { label: "Missing Quotes", value: fmtCount(summary.blockedMissingQuote) },
    { label: "Stale Quotes", value: fmtCount(summary.blockedStaleQuote) },
    { label: "Crossed Quotes", value: fmtCount(summary.blockedCrossedQuote) },
    { label: "Log Files", value: `${fmtCount(meta.signalFiles)} / ${fmtCount(meta.execFiles)}`, sub: "signal / execution" },
    modelName === "single"
      ? { label: "Compatibility", value: "legacy", tone: "warn", sub: "Mixed-layout logs detected under output/live/kalshi/*/demo" }
      : { label: "Compatibility", value: "multi-model", tone: "good", sub: "Per-model log layout detected" },
    { label: "Top Block", value: summary.blockCounts[0] ? escapeHtml(summary.blockCounts[0][0]) : "n/a", sub: summary.blockCounts[0] ? `${fmtCount(summary.blockCounts[0][1])} blocked decisions` : "No blocked signals yet" },
  ]);

  const latestSignalStats = latestSignal && Object.keys(latestSignal).length
    ? statsGridHtml([
        { label: "Last Signal", value: fmtDate(latestSignal.loggedAt) },
        { label: "Chosen Side", value: latestSignal.side || latestSignal.block_reason || "n/a" },
        { label: "Pred YES / NO", value: `${latestSignal.predicted_yes_probability == null ? "n/a" : fmtPct(latestSignal.predicted_yes_probability, 2)} / ${latestSignal.predicted_no_probability == null ? "n/a" : fmtPct(latestSignal.predicted_no_probability, 2)}` },
        { label: "Bid / Ask", value: latestSignal.yes_bid_cents == null || latestSignal.yes_ask_cents == null ? "n/a" : `${latestSignal.yes_bid_cents}c / ${latestSignal.yes_ask_cents}c` },
        { label: "Buy YES / NO", value: latestSignal.buy_yes_price_cents == null || latestSignal.buy_no_price_cents == null ? "n/a" : `${latestSignal.buy_yes_price_cents}c / ${latestSignal.buy_no_price_cents}c` },
        { label: "Spread / Age", value: latestSignal.quote_spread_cents == null ? "n/a" : `${latestSignal.quote_spread_cents}c / ${latestSignal.quote_age_seconds == null ? "n/a" : `${num(latestSignal.quote_age_seconds, 0).toFixed(2)}s`}` },
        { label: "Quote Mid", value: latestSignal.quote_mid_prob == null ? "n/a" : fmtPct(latestSignal.quote_mid_prob, 2) },
        { label: "LTP / Basis Prob", value: `${latestSignal.last_yes_price_cents == null ? "n/a" : `${latestSignal.last_yes_price_cents}c`} / ${latestSignal.feature_basis_market_prob == null ? "n/a" : fmtPct(latestSignal.feature_basis_market_prob, 2)}` },
        { label: "YES Edge", value: latestSignal.yes_post_cost_edge == null ? "n/a" : fmtPct(latestSignal.yes_post_cost_edge, 2), tone: latestSignal.yes_post_cost_edge == null ? "" : ratioTone(latestSignal.yes_post_cost_edge) },
        { label: "NO Edge", value: latestSignal.no_post_cost_edge == null ? "n/a" : fmtPct(latestSignal.no_post_cost_edge, 2), tone: latestSignal.no_post_cost_edge == null ? "" : ratioTone(latestSignal.no_post_cost_edge) },
      ])
    : `<div class="empty">No signal decisions yet for ${escapeHtml(modelName)}.</div>`;

  const blockTable = tableHtml([
    { label: "Block Reason", render: (row) => `<span class="tag">${escapeHtml(row.reason)}</span>` },
    { label: "Count", render: (row) => fmtCount(row.count) },
    { label: "Share", render: (row) => summary.blocked ? fmtPct(row.count / summary.blocked) : "0.0%" },
  ], summary.blockCounts.map(([reason, count]) => ({ reason, count })), `No blocked signal reasons yet for ${modelName}.`);

  return `
    <div class="detail-grid-two">
      ${panelSection("Block Reasons", `${modelName} most common reasons for rejected signals.`, `<div class="chart"><svg viewBox="0 0 420 260" preserveAspectRatio="none">${blockSvg(summary.blockCounts)}</svg></div><div class="legend-note">${escapeHtml(modelName)} blocked ${fmtCount(summary.blocked)} signals in the current lookback window.</div>`)}
      ${panelSection("Signal Health", `${modelName} live policy, quote-health, and compatibility diagnostics.`, signalHealth)}
    </div>
    <div class="detail-grid-two">
      ${panelSection("Latest Signal & Quote Snapshot", `${modelName} latest decision payload and executable quote context.`, latestSignalStats)}
      ${panelSection("Blocked Signal Breakdown", `${modelName} blocked-signal count and share by reason.`, blockTable)}
    </div>
  `;
}

function updateDashboardChrome(modelName, summary, loadedMeta = {}, modelCount = 0) {
  $("env-pill").textContent = state.environment;
  $("refresh-pill").textContent = `${state.refreshSeconds}s`;
  $("lookback-pill").textContent = `${state.lookbackDays} days`;
  $("last-load").textContent = fmtDate(state.lastLoadedAt || new Date());
  setTopMeta({
    source: loadedMeta.sourceKind || state.sourceKind || "idle",
    run: loadedMeta.runName || state.selectedRun || "none",
    mode: summary?.mode || "unknown",
    root: loadedMeta.rootLabel || state.sourceLabel || "none",
    loader: loadedMeta.loaderLabel || "folder or snapshot",
    files: loadedMeta.fileCount || 0,
    focus: modelName || "auto",
  });
  $("source-caption").textContent = loadedMeta.caption
    || "Choose a run folder or snapshot first. The dashboard can read both shadow and live execution logs, and it now distinguishes runs instead of assuming one fixed folder shape.";
  const syncLabel = summary
    ? loadedMeta.sourceKind === "snapshot"
      ? `Snapshot loaded | ${fmtCount(modelCount)} models | ${summary.mode || "unknown"} mode | re-upload snapshot to see new writes`
      : `Dashboard synced | ${fmtCount(modelCount)} models | ${summary.mode || "unknown"} mode | focused ${modelName}`
    : "Waiting for logs";
  sync(syncLabel, summary ? "ok" : "warn");
}

function renderSelectedModel(modelName, summary, meta, modelCount, loadedMeta, aggregate) {
  updateDashboardChrome(modelName, summary, loadedMeta, modelCount);
  renderTradingRunOverview(aggregate);
  renderFocusedHeader(modelName, summary, meta);
  $("tab-overview").innerHTML = overviewHtml(modelName, summary, meta);
  $("tab-breakdowns").innerHTML = breakdownsHtml(modelName, summary);
  $("tab-trades").innerHTML = tradesHtml(modelName, summary);
  $("tab-diagnostics").innerHTML = diagnosticsHtml(modelName, summary, meta);
  updateTabVisibility();
}

function renderEmptyFocus(message, loadedMeta = {}, aggregate = null) {
  updateDashboardChrome("", null, loadedMeta, 0);
  renderTradingRunOverview(aggregate);
  $("focused-model-title").textContent = "Focused Model";
  $("focused-model-subtitle").textContent = message;
  $("focused-model-badges").innerHTML = "";
  $("tab-overview").innerHTML = `<div class="empty">${escapeHtml(message)}</div>`;
  $("tab-breakdowns").innerHTML = `<div class="empty">${escapeHtml(message)}</div>`;
  $("tab-trades").innerHTML = `<div class="empty">${escapeHtml(message)}</div>`;
  $("tab-diagnostics").innerHTML = `<div class="empty">${escapeHtml(message)}</div>`;
  updateTabVisibility();
}

function updateTabVisibility() {
  document.querySelectorAll("[data-tab]").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === state.selectedTab);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${state.selectedTab}`);
  });
}

function renderAll(loaded) {
  const summaryMap = {};
  const metaMap = {};
  for (const [model, payload] of Object.entries(loaded.modelData || {})) {
    summaryMap[model] = analyze(payload.signalRows || [], payload.executionRows || []);
    metaMap[model] = payload.meta || { execFiles: 0, signalFiles: 0 };
  }
  const aggregate = aggregateTradingSummary(summaryMap);
  const selectedModel = updateModelSelector(summaryMap, metaMap);
  renderComparison(summaryMap, metaMap);
  renderModelCards(summaryMap, metaMap);
  if (!selectedModel) {
    renderEmptyFocus("No logs found yet. Choose a live folder or snapshot first.", loaded.meta, aggregate);
    return;
  }
  renderSelectedModel(
    selectedModel,
    summaryMap[selectedModel],
    metaMap[selectedModel],
    Object.keys(summaryMap).length,
    loaded.meta,
    aggregate,
  );
}

async function loadFromHandles() {
  if (!state.rootHandle) return null;
  const [execByModel, signalByModel] = await Promise.all([
    areaHandlesByModel(state.rootHandle, "execution", state.environment),
    areaHandlesByModel(state.rootHandle, "signal", state.environment),
  ]);
  const modelNames = [...new Set([...execByModel.keys(), ...signalByModel.keys()])].sort();
  const modelData = {};
  for (const model of modelNames) {
    const execHandles = execByModel.get(model) || [];
    const signalHandles = signalByModel.get(model) || [];
    const [execRows, signalRows] = await Promise.all([
      Promise.all(execHandles.map(readHandleRows)),
      Promise.all(signalHandles.map(readHandleRows)),
    ]);
    modelData[model] = {
      executionRows: execRows.flat(),
      signalRows: signalRows.flat(),
      meta: { execFiles: execHandles.length, signalFiles: signalHandles.length },
    };
  }
  const fileCount = Object.values(modelData).reduce((sum, item) => sum + item.meta.execFiles + item.meta.signalFiles, 0);
  return {
    modelData: pruneLegacySingleModel(modelData),
    meta: {
      sourceKind: "folder",
      runName: state.selectedRun || state.rootHandle.name || "selected",
      rootLabel: state.sourceLabel || state.rootHandle.name || "selected",
      loaderLabel: "live folder access",
      fileCount,
      caption: `Reading ${fmtCount(fileCount)} JSONL log files from the selected live folder. Current run: ${state.selectedRun || state.rootHandle.name || "selected"}.`,
    },
  };
}

async function loadFromSnapshot() {
  const grouped = state.snapshotRunMap.get(state.selectedRun) || [];
  if (!grouped.length) return null;
  const modelData = {};
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - state.lookbackDays);
  cutoff.setHours(0, 0, 0, 0);

  for (const entry of grouped) {
    const { file, descriptor } = entry;
    const { area, model, env, dateKey } = descriptor;
    if (env !== state.environment) continue;
    const d = parseDateFolder(dateKey);
    if (d && d < cutoff) continue;
    const rows = lines(await file.text());
    if (!modelData[model]) {
      modelData[model] = { executionRows: [], signalRows: [], meta: { execFiles: 0, signalFiles: 0 } };
    }
    if (area === "execution") {
      modelData[model].executionRows.push(...rows);
      modelData[model].meta.execFiles += 1;
    } else {
      modelData[model].signalRows.push(...rows);
      modelData[model].meta.signalFiles += 1;
    }
  }
  const fileCount = Object.values(modelData).reduce((sum, item) => sum + item.meta.execFiles + item.meta.signalFiles, 0);
  return {
    modelData: pruneLegacySingleModel(modelData),
    meta: {
      sourceKind: "snapshot",
      runName: state.selectedRun || "selected",
      rootLabel: state.sourceLabel || "snapshot",
      loaderLabel: "offline snapshot",
      fileCount,
      caption: `Reading ${fmtCount(fileCount)} JSONL log files from uploaded snapshot files. Current snapshot run: ${state.selectedRun || "selected"}.`,
    },
  };
}

async function refresh() {
  if (state.loading) return;
  state.loading = true;
  sync("Refreshing logs...");
  try {
    const loaded = state.rootHandle
      ? await loadFromHandles()
      : state.snapshotFiles.length
        ? await loadFromSnapshot()
        : null;
    if (!loaded) {
      state.lastLoaded = null;
      renderEmptyFocus("Choose a live folder or snapshot first.", {
        sourceKind: state.sourceKind,
        runName: state.selectedRun || "none",
        rootLabel: state.sourceLabel || "none",
        loaderLabel: state.sourceKind === "snapshot" ? "offline snapshot" : "live folder access",
        fileCount: 0,
      });
      sync("Choose a live folder or snapshot first");
      return;
    }
    state.lastLoaded = loaded;
    state.lastLoadedAt = new Date();
    renderAll(loaded);
  } catch (err) {
    console.error(err);
    sync(`Load failed: ${err.message || err}`, "bad");
  } finally {
    state.loading = false;
  }
}

function rerender() {
  if (!state.lastLoaded) return;
  renderAll(state.lastLoaded);
}

async function populateEnvs() {
  const envs = new Set();
  if (state.rootHandle) {
    for (const area of ["execution", "signal"]) {
      const areaDir = await getDir(state.rootHandle, area);
      if (!areaDir) continue;
      for await (const [name, h] of areaDir.entries()) {
        if (h.kind !== "directory") continue;
        if (name === "demo" || name === "production") {
          envs.add(name);
          continue;
        }
        const demoDir = await getDir(h, "demo");
        if (demoDir) envs.add("demo");
        const prodDir = await getDir(h, "production");
        if (prodDir) envs.add("production");
      }
    }
  } else if (state.snapshotRunMap.size) {
    for (const entry of state.snapshotRunMap.get(state.selectedRun) || []) {
      envs.add(entry.descriptor.env);
    }
  }
  if (!envs.size) return;
  $("environment-select").innerHTML = [...envs].sort().map((x) => `<option value="${x}">${x}</option>`).join("");
  if (!envs.has(state.environment)) state.environment = [...envs].sort()[0];
  $("environment-select").value = state.environment;
}

function restartTimer() {
  if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
  if (!state.autoRefresh) return;
  if (state.sourceKind === "snapshot") return;
  state.timer = setInterval(refresh, state.refreshSeconds * 1000);
}

async function chooseFolder() {
  if (!window.showDirectoryPicker) {
    sync("Folder picker unsupported here; use snapshot loader", "bad");
    return;
  }
  try {
    const picked = await window.showDirectoryPicker({ mode: "read" });
    const runs = await discoverRunHandles(picked);
    state.folderRootHandle = picked;
    state.runHandleMap = runs;
    state.selectedRun = preferredRunKey(runs.keys());
    state.rootHandle = state.runHandleMap.get(state.selectedRun) || null;
    state.snapshotFiles = [];
    state.snapshotRunMap = new Map();
    state.sourceKind = "folder";
    state.sourceLabel = picked.name || "selected";
    updateRunSelector([...runs.keys()].sort());
    await populateEnvs();
    restartTimer();
    await refresh();
  } catch {
    return;
  }
}

$("folder-button").addEventListener("click", chooseFolder);
$("manual-refresh").addEventListener("click", refresh);
$("environment-select").addEventListener("change", async (e) => {
  state.environment = e.target.value;
  await refresh();
});
$("run-select").addEventListener("change", async (e) => {
  state.selectedRun = e.target.value;
  if (state.sourceKind === "folder") {
    state.rootHandle = state.runHandleMap.get(state.selectedRun) || null;
  }
  await populateEnvs();
  await refresh();
});
$("refresh-seconds").addEventListener("change", (e) => {
  state.refreshSeconds = Math.max(1, num(e.target.value, 5));
  restartTimer();
});
$("lookback-days").addEventListener("change", async (e) => {
  state.lookbackDays = Math.max(1, num(e.target.value, 7));
  await refresh();
});
$("auto-refresh").addEventListener("change", (e) => {
  state.autoRefresh = e.target.value === "on";
  restartTimer();
});
$("model-select").addEventListener("change", (e) => {
  state.selectedModel = e.target.value;
  rerender();
});
$("snapshot-input").addEventListener("change", async (e) => {
  state.snapshotFiles = [...(e.target.files || [])];
  state.snapshotRunMap = buildSnapshotRunMap(state.snapshotFiles);
  state.selectedRun = preferredRunKey(state.snapshotRunMap.keys());
  updateRunSelector([...state.snapshotRunMap.keys()].sort());
  state.rootHandle = null;
  state.folderRootHandle = null;
  state.runHandleMap = new Map();
  state.sourceKind = "snapshot";
  state.sourceLabel = state.snapshotFiles.length ? "uploaded snapshot" : "snapshot";
  state.autoRefresh = false;
  $("auto-refresh").value = "off";
  await populateEnvs();
  restartTimer();
  await refresh();
});
document.querySelectorAll("[data-tab]").forEach((button) => {
  button.addEventListener("click", () => {
    state.selectedTab = button.dataset.tab || "overview";
    updateTabVisibility();
  });
});

restartTimer();
setTopMeta();
renderEmptyFocus("Choose a live folder or snapshot first.");
sync("Waiting for logs");
