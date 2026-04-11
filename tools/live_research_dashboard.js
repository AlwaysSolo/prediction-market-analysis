const TAU_BUCKETS = ["2-4", "4-6", "6-8", "8-10", "10-12", "12-14"];
const PRICE_BUCKETS = Array.from({ length: 10 }, (_, i) => `${i * 10}-${i * 10 + 10}`);
const PROB_BUCKETS = Array.from({ length: 10 }, (_, i) => `${i * 10}-${i * 10 + 10}`);
const EDGE_BUCKETS = ["<0", "0-5", "5-10", "10-20", "20-40", "40-60", "60+"];

const COMPARISON_COLUMNS = [
  { key: "model", label: "Model", sortable: false },
  { key: "recorded", label: "Recorded", sortable: true, defaultDir: "desc" },
  { key: "settled", label: "Settled", sortable: true, defaultDir: "desc" },
  { key: "open", label: "Open", sortable: true, defaultDir: "desc" },
  { key: "winRate", label: "Win Rate", sortable: true, defaultDir: "desc" },
  { key: "realized", label: "Realized PnL", sortable: true, defaultDir: "desc" },
  { key: "investment", label: "Invested", sortable: true, defaultDir: "desc" },
  { key: "pnlPct", label: "PnL %", sortable: true, defaultDir: "desc" },
  { key: "maxDrawdown", label: "Max DD", sortable: true, defaultDir: "desc" },
  { key: "expectancy", label: "Expectancy", sortable: true, defaultDir: "desc" },
  { key: "avgEdgeCents", label: "Avg Edge", sortable: true, defaultDir: "desc" },
  { key: "avgProb", label: "Avg Prob", sortable: true, defaultDir: "desc" },
  { key: "skipped", label: "Skipped", sortable: true, defaultDir: "desc" },
  { key: "logs", label: "Logs", sortable: false },
];

const state = {
  rootHandle: null,
  snapshotFiles: [],
  environment: "demo",
  refreshSeconds: 5,
  lookbackDays: 7,
  autoRefresh: true,
  timer: null,
  loading: false,
  selectedModel: "",
  selectedTab: "overview",
  comparisonSort: { key: "settled", dir: "desc" },
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
const fmtEdge = (v) => `${Number(v || 0).toFixed(1)}c`;
const fmtDate = (v) => (v ? new Date(v).toLocaleString() : "n/a");
const num = (v, f = 0) => (Number.isFinite(Number(v)) ? Number(v) : f);

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]
  ));
}

function tagHtml(label, tone = "") {
  const classes = ["tag"];
  if (tone) classes.push(tone);
  return `<span class="${classes.join(" ")}">${escapeHtml(label ?? "n/a")}</span>`;
}

function tickerLabel(ticker) {
  return `<span class="mono">${escapeHtml(ticker || "n/a")}</span>`;
}

function modelTag(model, isLegacy = false) {
  return `${tagHtml(model, "tag-model")}${isLegacy ? ` ${tagHtml("legacy", "tag-legacy")}` : ""}`;
}

function sideTag(side) {
  if (side === "YES") return tagHtml("YES", "tag-yes");
  if (side === "NO") return tagHtml("NO", "tag-no");
  return tagHtml(side || "n/a");
}

function statusTag(status) {
  if (status === "open") return tagHtml("open", "tag-open");
  if (status === "settled") return tagHtml("settled", "tag-settled");
  return tagHtml(status || "n/a");
}

function resultTag(result) {
  if (result === "YES" || result === "NO") return sideTag(result);
  return result ? tagHtml(result) : "n/a";
}

function reasonTag(reason) {
  return tagHtml(reason || "unknown", "warn");
}

function eventTypeTag(type) {
  if (type === "recorded") return tagHtml("recorded", "tag-model");
  if (type === "settled") return tagHtml("settled", "tag-settled");
  if (type === "skipped") return tagHtml("skipped", "warn");
  return tagHtml(type || "n/a");
}

function bucketTag(label, kind = "generic") {
  if (kind === "edge") {
    if (label === "<0") return tagHtml(label, "bad");
    if (label === "0-5") return tagHtml(label, "warn");
  }
  if (kind === "probability") return tagHtml(label, "tag-settled");
  return tagHtml(label, "tag-model");
}

function sync(label, kind = "warn") {
  $("sync-label").textContent = label;
  $("sync-dot").className = `dot${kind === "ok" ? " ok" : kind === "bad" ? " bad" : ""}`;
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

async function researchHandlesByModel(root, env) {
  const areaDir = await getDir(root, "research");
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

function tableHtml(cols, rows, empty, minWidth = 720) {
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

function pnlPct(realized, invested) {
  return invested > 0 ? realized / invested : 0;
}

function ensureBucketMap(order) {
  const map = new Map();
  for (const label of order) {
    map.set(label, {
      label,
      recorded: 0,
      settled: 0,
      wins: 0,
      losses: 0,
      realized: 0,
      recordedInvestment: 0,
      settledInvestment: 0,
      winRate: 0,
      expectancy: 0,
      pnlPct: 0,
    });
  }
  return map;
}

function ensureSideMap() {
  return {
    YES: { side: "YES", recorded: 0, settled: 0, wins: 0, losses: 0, realized: 0, recordedInvestment: 0, settledInvestment: 0, winRate: 0, expectancy: 0, pnlPct: 0 },
    NO: { side: "NO", recorded: 0, settled: 0, wins: 0, losses: 0, realized: 0, recordedInvestment: 0, settledInvestment: 0, winRate: 0, expectancy: 0, pnlPct: 0 },
  };
}

function ensureSideBucketMaps(order) {
  return {
    YES: ensureBucketMap(order),
    NO: ensureBucketMap(order),
  };
}

function maxDrawdownSummary(settledSamples) {
  const ordered = [...settledSamples].sort((a, b) => new Date(a.settledAt) - new Date(b.settledAt));
  let cumulative = 0;
  let peak = 0;
  let maxDrawdown = 0;
  let maxDrawdownPct = null;
  for (const sample of ordered) {
    cumulative += num(sample.pnl, 0);
    if (cumulative > peak) peak = cumulative;
    const drawdown = peak - cumulative;
    if (drawdown > maxDrawdown) {
      maxDrawdown = drawdown;
      maxDrawdownPct = peak > 0 ? drawdown / peak : null;
    }
  }
  return { maxDrawdown, maxDrawdownPct };
}

function summarizeRows(rows) {
  const research = rows
    .map((r) => ({ ...r, t: new Date(r.logged_at) }))
    .filter((r) => !Number.isNaN(r.t.getTime()))
    .sort((a, b) => a.t - b.t);

  const startConfig = {};
  const samples = new Map();
  const skippedCounts = new Map();
  const summaryPoints = [];
  const sideSummary = ensureSideMap();
  const tauMap = ensureBucketMap(TAU_BUCKETS);
  const priceMap = ensureBucketMap(PRICE_BUCKETS);
  const probMap = ensureBucketMap(PROB_BUCKETS);
  const edgeMap = ensureBucketMap(EDGE_BUCKETS);
  const tauSideMaps = ensureSideBucketMaps(TAU_BUCKETS);
  const priceSideMaps = ensureSideBucketMaps(PRICE_BUCKETS);
  const probSideMaps = ensureSideBucketMaps(PROB_BUCKETS);
  const edgeSideMaps = ensureSideBucketMaps(EDGE_BUCKETS);
  const tickerMap = new Map();
  let latestRecorded = null;
  let latestSkipped = null;
  const timeline = [];
  let skipCount = 0;

  for (const row of research) {
    const p = row.payload || {};
    if (row.event_type === "research_started") {
      Object.assign(startConfig, p);
      continue;
    }
    if (row.event_type === "research_summary_snapshot") {
      summaryPoints.push({
        time: row.t,
        label: row.logged_at,
        open: num(p.open_sample_count, 0),
        settled: num(p.settled_sample_count, 0),
        wins: num(p.win_count, 0),
        losses: num(p.loss_count, 0),
        cumPnl: num(p.cumulative_realized_pnl_dollars, 0),
      });
      continue;
    }
    if (row.event_type === "research_sample_skipped") {
      const reason = p.reason || "unknown";
      skippedCounts.set(reason, (skippedCounts.get(reason) || 0) + 1);
      latestSkipped = {
        loggedAt: row.logged_at,
        ticker: p.ticker || "unknown",
        reason,
        side: p.side || "n/a",
        predictedYesProbability: p.predicted_yes_probability ?? null,
        predictedNoProbability: p.predicted_no_probability ?? null,
        featureBasisMarketProb: p.feature_basis_market_prob ?? null,
        chosenPostCostEdge: p.chosen_post_cost_edge ?? null,
        tauMinutes: p.tau_minutes ?? null,
        yesBidCents: p.yes_bid_cents ?? null,
        yesAskCents: p.yes_ask_cents ?? null,
        buyYesPriceCents: p.buy_yes_price_cents ?? null,
        buyNoPriceCents: p.buy_no_price_cents ?? null,
        quoteMidProb: p.quote_mid_prob ?? null,
        quoteSpreadCents: p.quote_spread_cents ?? null,
        quoteAgeSeconds: p.quote_age_seconds ?? null,
      };
      skipCount += 1;
      continue;
    }
    if (row.event_type === "research_sample_recorded") {
      const sample = {
        sampleId: p.sample_id,
        ticker: p.ticker || "unknown",
        side: p.side || "UNKNOWN",
        status: "open",
        recordedAt: row.logged_at,
        settledAt: null,
        settlementResult: null,
        isWin: null,
        pnl: null,
        cumPnl: null,
        contracts: num(p.contracts, 0),
        referencePriceCents: p.reference_price_cents ?? null,
        maxAcceptablePriceCents: p.max_acceptable_entry_price_cents ?? null,
        predictedYesProbability: p.predicted_yes_probability ?? null,
        predictedNoProbability: p.predicted_no_probability ?? null,
        chosenSideProbability: p.chosen_side_probability ?? null,
        featureBasisMarketProb: p.feature_basis_market_prob ?? null,
        rawModelEdge: p.raw_model_edge ?? null,
        yesPostCostEdge: p.yes_post_cost_edge ?? null,
        noPostCostEdge: p.no_post_cost_edge ?? null,
        chosenPostCostEdge: p.chosen_post_cost_edge ?? null,
        chosenEdgeCents: p.chosen_post_cost_edge == null ? null : p.chosen_post_cost_edge * 100,
        tauMinutes: p.tau_minutes ?? null,
        tauBucket: p.tau_bucket || "unknown",
        priceBucket: p.price_bucket || "unknown",
        probabilityBucket: p.chosen_side_probability_bucket || "unknown",
        edgeBucket: p.chosen_side_edge_bucket || "unknown",
        quoteSpreadCents: p.quote_spread_cents ?? null,
        quoteAgeSeconds: p.quote_age_seconds ?? null,
        cashRequired: num(p.estimated_cash_required_dollars, 0),
        investmentAmount: ((p.reference_price_cents ?? 0) * num(p.contracts, 0)) / 100,
      };
      samples.set(sample.sampleId, sample);
      latestRecorded = {
        loggedAt: row.logged_at,
        ticker: sample.ticker,
        side: sample.side,
        predictedYesProbability: sample.predictedYesProbability,
        predictedNoProbability: sample.predictedNoProbability,
        chosenSideProbability: sample.chosenSideProbability,
        featureBasisMarketProb: sample.featureBasisMarketProb,
        rawModelEdge: sample.rawModelEdge,
        yesPostCostEdge: sample.yesPostCostEdge,
        noPostCostEdge: sample.noPostCostEdge,
        chosenPostCostEdge: sample.chosenPostCostEdge,
        referencePriceCents: sample.referencePriceCents,
        tauMinutes: sample.tauMinutes,
        tauBucket: sample.tauBucket,
        priceBucket: sample.priceBucket,
        probabilityBucket: sample.probabilityBucket,
        edgeBucket: sample.edgeBucket,
        yesBidCents: p.yes_bid_cents ?? null,
        yesAskCents: p.yes_ask_cents ?? null,
        buyYesPriceCents: p.buy_yes_price_cents ?? null,
        buyNoPriceCents: p.buy_no_price_cents ?? null,
        quoteSpreadCents: sample.quoteSpreadCents,
        quoteAgeSeconds: sample.quoteAgeSeconds,
        cashRequired: sample.cashRequired,
      };
      if (sideSummary[sample.side]) {
        sideSummary[sample.side].recorded += 1;
        sideSummary[sample.side].recordedInvestment += sample.investmentAmount;
      }
      if (tauMap.has(sample.tauBucket)) {
        tauMap.get(sample.tauBucket).recorded += 1;
        tauMap.get(sample.tauBucket).recordedInvestment += sample.investmentAmount;
      }
      if (priceMap.has(sample.priceBucket)) {
        priceMap.get(sample.priceBucket).recorded += 1;
        priceMap.get(sample.priceBucket).recordedInvestment += sample.investmentAmount;
      }
      if (probMap.has(sample.probabilityBucket)) {
        probMap.get(sample.probabilityBucket).recorded += 1;
        probMap.get(sample.probabilityBucket).recordedInvestment += sample.investmentAmount;
      }
      if (edgeMap.has(sample.edgeBucket)) {
        edgeMap.get(sample.edgeBucket).recorded += 1;
        edgeMap.get(sample.edgeBucket).recordedInvestment += sample.investmentAmount;
      }
      for (const [sideMaps, label] of [
        [tauSideMaps, sample.tauBucket],
        [priceSideMaps, sample.priceBucket],
        [probSideMaps, sample.probabilityBucket],
        [edgeSideMaps, sample.edgeBucket],
      ]) {
        if (!sideMaps[sample.side]?.has(label)) continue;
        const rowData = sideMaps[sample.side].get(label);
        rowData.recorded += 1;
        rowData.recordedInvestment += sample.investmentAmount;
      }
      const tickerStats = tickerMap.get(sample.ticker) || { ticker: sample.ticker, recorded: 0, settled: 0, wins: 0, losses: 0, open: 0, realized: 0 };
      tickerStats.recorded += 1;
      tickerMap.set(sample.ticker, tickerStats);
      timeline.push({ time: row.logged_at, type: "recorded", ticker: sample.ticker, side: sample.side, result: null, pnl: null });
      continue;
    }
    if (row.event_type === "research_sample_settled") {
      const sampleId = p.sample_id;
      const sample = samples.get(sampleId);
      if (!sample || sample.status === "settled") continue;
      sample.status = "settled";
      sample.settledAt = row.logged_at;
      sample.settlementResult = p.settlement_result || null;
      sample.isWin = Boolean(p.is_win);
      sample.pnl = num(p.realized_pnl_dollars, 0);
      sample.cumPnl = num(p.cumulative_realized_pnl_dollars, 0);
      if (sideSummary[sample.side]) {
        sideSummary[sample.side].settled += 1;
        sideSummary[sample.side].realized += sample.pnl;
        sideSummary[sample.side].settledInvestment += sample.investmentAmount;
        if (sample.isWin) sideSummary[sample.side].wins += 1;
        else sideSummary[sample.side].losses += 1;
      }
      for (const [bucketMap, label] of [
        [tauMap, sample.tauBucket],
        [priceMap, sample.priceBucket],
        [probMap, sample.probabilityBucket],
        [edgeMap, sample.edgeBucket],
      ]) {
        if (!bucketMap.has(label)) continue;
        const rowData = bucketMap.get(label);
        rowData.settled += 1;
        rowData.realized += sample.pnl;
        rowData.settledInvestment += sample.investmentAmount;
        if (sample.isWin) rowData.wins += 1;
        else rowData.losses += 1;
      }
      for (const [sideMaps, label] of [
        [tauSideMaps, sample.tauBucket],
        [priceSideMaps, sample.priceBucket],
        [probSideMaps, sample.probabilityBucket],
        [edgeSideMaps, sample.edgeBucket],
      ]) {
        if (!sideMaps[sample.side]?.has(label)) continue;
        const rowData = sideMaps[sample.side].get(label);
        rowData.settled += 1;
        rowData.realized += sample.pnl;
        rowData.settledInvestment += sample.investmentAmount;
        if (sample.isWin) rowData.wins += 1;
        else rowData.losses += 1;
      }
      const tickerStats = tickerMap.get(sample.ticker) || { ticker: sample.ticker, recorded: 0, settled: 0, wins: 0, losses: 0, open: 0, realized: 0 };
      tickerStats.settled += 1;
      tickerStats.realized += sample.pnl;
      if (sample.isWin) tickerStats.wins += 1;
      else tickerStats.losses += 1;
      tickerMap.set(sample.ticker, tickerStats);
      timeline.push({ time: row.logged_at, type: "settled", ticker: sample.ticker, side: sample.side, result: sample.settlementResult, pnl: sample.pnl });
    }
  }

  const recorded = [...samples.values()].sort((a, b) => new Date(b.recordedAt) - new Date(a.recordedAt));
  const settled = recorded.filter((sample) => sample.status === "settled").sort((a, b) => new Date(b.settledAt) - new Date(a.settledAt));
  const openSamples = recorded.filter((sample) => sample.status !== "settled");
  const wins = settled.filter((sample) => sample.isWin);
  const losses = settled.filter((sample) => !sample.isWin);
  const realized = settled.reduce((sum, sample) => sum + num(sample.pnl, 0), 0);
  const expectancy = settled.length ? realized / settled.length : 0;
  const avgEdgeCents = recorded.length ? recorded.reduce((sum, sample) => sum + num(sample.chosenEdgeCents, 0), 0) / recorded.length : 0;
  const avgProb = recorded.length ? recorded.reduce((sum, sample) => sum + num(sample.chosenSideProbability, 0), 0) / recorded.length : 0;
  const totalInvestment = recorded.reduce((sum, sample) => sum + num(sample.investmentAmount, 0), 0);
  const settledInvestment = settled.reduce((sum, sample) => sum + num(sample.investmentAmount, 0), 0);
  const openInvestment = openSamples.reduce((sum, sample) => sum + num(sample.investmentAmount, 0), 0);
  const realizedPnlPct = pnlPct(realized, settledInvestment);
  const { maxDrawdown, maxDrawdownPct } = maxDrawdownSummary(settled);

  for (const side of Object.values(sideSummary)) {
    side.winRate = side.settled ? side.wins / side.settled : 0;
    side.expectancy = side.settled ? side.realized / side.settled : 0;
    side.pnlPct = pnlPct(side.realized, side.settledInvestment);
  }
  for (const bucketMap of [tauMap, priceMap, probMap, edgeMap, tauSideMaps.YES, tauSideMaps.NO, priceSideMaps.YES, priceSideMaps.NO, probSideMaps.YES, probSideMaps.NO, edgeSideMaps.YES, edgeSideMaps.NO]) {
    for (const row of bucketMap.values()) {
      row.winRate = row.settled ? row.wins / row.settled : 0;
      row.expectancy = row.settled ? row.realized / row.settled : 0;
      row.pnlPct = pnlPct(row.realized, row.settledInvestment);
    }
  }
  for (const tickerStats of tickerMap.values()) tickerStats.open = tickerStats.recorded - tickerStats.settled;

  const topSkipped = [...skippedCounts.entries()].sort((a, b) => b[1] - a[1]);
  const latestSummary = summaryPoints.at(-1) || {
    open: openSamples.length,
    settled: settled.length,
    wins: wins.length,
    losses: losses.length,
    cumPnl: realized,
    label: null,
  };

  return {
    startConfig,
    latestRecorded,
    latestSkipped,
    recorded,
    settled,
    openSamples,
    wins,
    losses,
    realized,
    totalInvestment,
    settledInvestment,
    openInvestment,
    realizedPnlPct,
    maxDrawdown,
    maxDrawdownPct,
    expectancy,
    avgEdgeCents,
    avgProb,
    skipCount,
    skipCounts: topSkipped,
    summaryPoints,
    latestSummary,
    sideSummary: [sideSummary.YES, sideSummary.NO],
    tauBreakdown: TAU_BUCKETS.map((label) => tauMap.get(label)),
    priceBreakdown: PRICE_BUCKETS.map((label) => priceMap.get(label)),
    probabilityBreakdown: PROB_BUCKETS.map((label) => probMap.get(label)),
    edgeBreakdown: EDGE_BUCKETS.map((label) => edgeMap.get(label)),
    tauBreakdownBySide: { YES: TAU_BUCKETS.map((label) => tauSideMaps.YES.get(label)), NO: TAU_BUCKETS.map((label) => tauSideMaps.NO.get(label)) },
    priceBreakdownBySide: { YES: PRICE_BUCKETS.map((label) => priceSideMaps.YES.get(label)), NO: PRICE_BUCKETS.map((label) => priceSideMaps.NO.get(label)) },
    probabilityBreakdownBySide: { YES: PROB_BUCKETS.map((label) => probSideMaps.YES.get(label)), NO: PROB_BUCKETS.map((label) => probSideMaps.NO.get(label)) },
    edgeBreakdownBySide: { YES: EDGE_BUCKETS.map((label) => edgeSideMaps.YES.get(label)), NO: EDGE_BUCKETS.map((label) => edgeSideMaps.NO.get(label)) },
    tickerBreakdown: [...tickerMap.values()].sort((a, b) => b.realized - a.realized),
    timeline: timeline.sort((a, b) => new Date(b.time) - new Date(a.time)).slice(0, 120),
  };
}

function aggregateResearchSummary(summaryMap) {
  const entries = Object.entries(summaryMap);
  if (!entries.length) return null;

  let recorded = 0;
  let settled = 0;
  let open = 0;
  let invested = 0;
  let realized = 0;
  let settledInvestment = 0;
  let worstDd = 0;
  let worstDdModel = "n/a";
  const skipCounts = new Map();

  for (const [model, summary] of entries) {
    recorded += summary.recorded.length;
    settled += summary.settled.length;
    open += summary.openSamples.length;
    invested += num(summary.totalInvestment, 0);
    realized += num(summary.realized, 0);
    settledInvestment += num(summary.settledInvestment, 0);
    if (num(summary.maxDrawdown, 0) >= worstDd) {
      worstDd = num(summary.maxDrawdown, 0);
      worstDdModel = model;
    }
    for (const [reason, count] of summary.skipCounts || []) {
      skipCounts.set(reason, (skipCounts.get(reason) || 0) + num(count, 0));
    }
  }

  const [topSkipReason, topSkipCount] = [...skipCounts.entries()].sort((a, b) => b[1] - a[1])[0] || [null, 0];
  return {
    modelCount: entries.length,
    recorded,
    settled,
    open,
    invested,
    realized,
    pnlPct: pnlPct(realized, settledInvestment),
    worstDd,
    worstDdModel,
    topSkipReason,
    topSkipCount,
  };
}

function renderResearchRunOverview(aggregate) {
  if (!aggregate) {
    setTextIfPresent("research-overview-recorded", "0");
    setTextIfPresent("research-overview-recorded-note", "No loaded samples yet");
    setTextIfPresent("research-overview-settled", "0");
    setTextIfPresent("research-overview-settled-note", "No settled samples yet");
    setTextIfPresent("research-overview-open", "0");
    setTextIfPresent("research-overview-open-note", "Waiting on resolution");
    setTextIfPresent("research-overview-invested", "$0.00");
    setTextIfPresent("research-overview-invested-note", "Recorded notional");
    setTextIfPresent("research-overview-realized", "$0.00");
    setTextIfPresent("research-overview-realized-note", "Settled-only realized");
    setTextIfPresent("research-overview-pnlpct", "0.00%");
    setTextIfPresent("research-overview-pnlpct-note", "Realized / settled investment");
    setTextIfPresent("research-overview-drawdown", "$0.00");
    setTextIfPresent("research-overview-drawdown-note", "Worst model drawdown");
    setTextIfPresent("research-overview-skip", "n/a");
    setTextIfPresent("research-overview-skip-note", "No skipped samples yet");
    return;
  }

  setTextIfPresent("research-overview-recorded", fmtCount(aggregate.recorded));
  setTextIfPresent("research-overview-recorded-note", `Across ${fmtCount(aggregate.modelCount)} model lanes`);
  setTextIfPresent("research-overview-settled", fmtCount(aggregate.settled));
  setTextIfPresent("research-overview-settled-note", "Resolved observation-only samples");
  setTextIfPresent("research-overview-open", fmtCount(aggregate.open));
  setTextIfPresent("research-overview-open-note", "Still waiting on settlement");
  setTextIfPresent("research-overview-invested", fmtMoney(aggregate.invested));
  setTextIfPresent("research-overview-invested-note", "Recorded notional across the run");
  setTextIfPresent("research-overview-realized", fmtMoney(aggregate.realized));
  setTextIfPresent("research-overview-realized-note", "Settled-only realized");
  setTextIfPresent("research-overview-pnlpct", fmtPct(aggregate.pnlPct, 2));
  setTextIfPresent("research-overview-pnlpct-note", "Realized / settled investment");
  setTextIfPresent("research-overview-drawdown", fmtMoney(aggregate.worstDd));
  setTextIfPresent("research-overview-drawdown-note", `${aggregate.worstDdModel} worst max DD`);
  setTextIfPresent("research-overview-skip", aggregate.topSkipReason || "n/a");
  setTextIfPresent(
    "research-overview-skip-note",
    aggregate.topSkipReason ? `${fmtCount(aggregate.topSkipCount)} skipped candidates` : "No skipped samples yet",
  );
}

function updateResearchChrome(modelName, loadedMeta = {}, modelCount = 0) {
  setTextIfPresent("env-pill", state.environment);
  setTextIfPresent("refresh-pill", `${state.refreshSeconds}s`);
  setTextIfPresent("lookback-pill", `${state.lookbackDays} days`);
  setTextIfPresent("last-load", fmtDate(state.lastLoadedAt || new Date()));
  setTextIfPresent("research-root-pill", loadedMeta.rootLabel || "none");
  setTextIfPresent("research-focus-pill", modelName || "auto");
  setTextIfPresent("research-source-pill", loadedMeta.sourceKind || "idle");
  setTextIfPresent("research-files-pill", fmtCount(loadedMeta.fileCount || 0));
  sync(
    modelName
      ? `Research dashboard synced | ${fmtCount(modelCount)} models | focused ${modelName}`
      : "Waiting for research logs",
    modelName ? "ok" : "warn",
  );
}

function pnlSvg(points) {
  if (!points.length) {
    return `<text x="50%" y="50%" text-anchor="middle" fill="rgba(237,245,251,.42)" font-size="18">No summary snapshots yet</text>`;
  }
  const w = 800;
  const h = 260;
  const p = { t: 18, r: 18, b: 28, l: 54 };
  const values = points.map((x) => x.cumPnl);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(0.0001, max - min);
  const coords = points.map((pt, i) => {
    const x = p.l + (i / Math.max(1, points.length - 1)) * (w - p.l - p.r);
    const y = p.t + (1 - ((pt.cumPnl - min) / span)) * (h - p.t - p.b);
    return { x, y };
  });
  const path = coords.map((pt, i) => `${i === 0 ? "M" : "L"} ${pt.x.toFixed(2)} ${pt.y.toFixed(2)}`).join(" ");
  const area = `${path} L ${w - p.r} ${h - p.b} L ${p.l} ${h - p.b} Z`;
  const latest = points.at(-1);
  const latestCoord = coords.at(-1);
  const positiveFinish = num(latest?.cumPnl, 0) >= 0;
  const lineColor = positiveFinish ? "#53d0ff" : "#ff6b81";
  const fillId = `researchFill-${points.length}`;
  let grid = "";
  for (let i = 0; i <= 4; i += 1) {
    const y = p.t + (i / 4) * (h - p.t - p.b);
    const value = max - (i / 4) * span;
    grid += `<line x1="${p.l}" y1="${y}" x2="${w - p.r}" y2="${y}" stroke="rgba(170,196,220,.12)" stroke-dasharray="4 6"></line>`;
    grid += `<text x="${p.l - 8}" y="${y + 4}" text-anchor="end" fill="rgba(237,245,251,.56)" font-size="12">${fmtMoney(value)}</text>`;
  }
  if (min < 0 && max > 0) {
    const zeroY = p.t + (1 - ((0 - min) / span)) * (h - p.t - p.b);
    grid += `<line x1="${p.l}" y1="${zeroY}" x2="${w - p.r}" y2="${zeroY}" stroke="rgba(244,191,79,.28)" stroke-dasharray="5 5"></line>`;
  }
  return `
    <defs>
      <linearGradient id="${fillId}" x1="0" x2="0" y1="0" y2="1">
        <stop offset="0%" stop-color="${positiveFinish ? "rgba(83,208,255,.28)" : "rgba(255,107,129,.26)"}"></stop>
        <stop offset="100%" stop-color="rgba(83,208,255,.02)"></stop>
      </linearGradient>
    </defs>
    ${grid}
    <path d="${area}" fill="url(#${fillId})"></path>
    <path d="${path}" fill="none" stroke="${lineColor}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"></path>
    ${latestCoord ? `<circle cx="${latestCoord.x}" cy="${latestCoord.y}" r="4.5" fill="${lineColor}" stroke="rgba(7,16,24,.9)" stroke-width="2"></circle>` : ""}
  `;
}

function skipSvg(items) {
  if (!items.length) {
    return `<text x="50%" y="50%" text-anchor="middle" fill="rgba(237,245,251,.42)" font-size="18">No skipped samples yet</text>`;
  }
  const top = items.slice(0, 6);
  const w = 420;
  const p = { t: 18, r: 12, l: 150 };
  const barH = 22;
  const gap = 12;
  const max = Math.max(...top.map((x) => x[1]), 1);
  return top.map(([reason, count], i) => {
    const y = p.t + i * (barH + gap);
    const barWidth = ((w - p.l - p.r) * count) / max;
    return `
      <text x="${p.l - 10}" y="${y + 15}" text-anchor="end" fill="rgba(237,245,251,.62)" font-size="12">${escapeHtml(reason)}</text>
      <rect x="${p.l}" y="${y}" width="${w - p.l - p.r}" height="${barH}" rx="11" fill="rgba(255,255,255,.035)"></rect>
      <rect x="${p.l}" y="${y}" width="${barWidth}" height="${barH}" rx="11" fill="rgba(244,191,79,.82)"></rect>
      <text x="${p.l + barWidth + 8}" y="${y + 15}" fill="rgba(237,245,251,.82)" font-size="12">${count}</text>
    `;
  }).join("");
}

function comparisonRows(summaryMap, metaMap) {
  return Object.entries(summaryMap).map(([model, summary]) => ({
    model,
    summary,
    meta: metaMap[model] || { researchFiles: 0 },
    isLegacy: model === "single",
  }));
}

function comparisonValue(row, key) {
  switch (key) {
    case "model": return row.model;
    case "recorded": return row.summary.recorded.length;
    case "settled": return row.summary.settled.length;
    case "open": return row.summary.openSamples.length;
    case "winRate": return row.summary.settled.length ? row.summary.wins.length / row.summary.settled.length : 0;
    case "realized": return row.summary.realized;
    case "investment": return row.summary.totalInvestment;
    case "pnlPct": return row.summary.realizedPnlPct;
    case "maxDrawdown": return row.summary.maxDrawdown;
    case "expectancy": return row.summary.expectancy;
    case "avgEdgeCents": return row.summary.avgEdgeCents;
    case "avgProb": return row.summary.avgProb;
    case "skipped": return row.summary.skipCount;
    default: return 0;
  }
}

function sortRows(rows) {
  const { key, dir } = state.comparisonSort;
  return rows.sort((a, b) => {
    const av = comparisonValue(a, key);
    const bv = comparisonValue(b, key);
    let delta = 0;
    if (typeof av === "string" || typeof bv === "string") delta = String(av).localeCompare(String(bv));
    else delta = num(av, 0) - num(bv, 0);
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
    wrap.innerHTML = `<div class="empty">No research logs found yet. Start the research runner, then choose the root output folder.</div>`;
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
    const winRate = summary.settled.length ? summary.wins.length / summary.settled.length : 0;
    const topSkip = summary.skipCounts[0] ? summary.skipCounts[0][0] : "n/a";
    const selected = row.model === state.selectedModel ? " selected-row" : "";
    return `<tr class="row-selectable${selected}" data-model-row="${escapeHtml(row.model)}">
      <td>${modelTag(row.model, row.isLegacy)}</td>
      <td>${fmtCount(summary.recorded.length)}</td>
      <td>${fmtCount(summary.settled.length)}</td>
      <td>${fmtCount(summary.openSamples.length)}</td>
      <td>${fmtPct(winRate)}</td>
      <td><span class="${moneyTone(summary.realized)}">${fmtMoney(summary.realized)}</span></td>
      <td>${fmtMoney(summary.totalInvestment)}</td>
      <td><span class="${moneyTone(summary.realizedPnlPct)}">${fmtPct(summary.realizedPnlPct, 2)}</span></td>
      <td><span class="${moneyTone(-summary.maxDrawdown)}">${fmtMoney(summary.maxDrawdown)}</span></td>
      <td><span class="${moneyTone(summary.expectancy)}">${fmtMoney(summary.expectancy)}</span></td>
      <td><span class="${ratioTone(summary.avgEdgeCents)}">${fmtEdge(summary.avgEdgeCents)}</span></td>
      <td>${fmtPct(summary.avgProb, 2)}</td>
      <td>${reasonTag(topSkip)}</td>
      <td><span class="mono">${fmtCount(row.meta.researchFiles)} research</span></td>
    </tr>`;
  }).join("");

  wrap.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
  wrap.querySelectorAll("[data-sort-key]").forEach((button) => {
    button.addEventListener("click", () => {
      const key = button.dataset.sortKey;
      const column = COMPARISON_COLUMNS.find((item) => item.key === key);
      if (!column) return;
      if (state.comparisonSort.key === key) state.comparisonSort.dir = state.comparisonSort.dir === "desc" ? "asc" : "desc";
      else state.comparisonSort = { key, dir: column.defaultDir || "desc" };
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
    const winRate = summary.settled.length ? summary.wins.length / summary.settled.length : 0;
    return `<div class="model-card${selected}" data-model-card="${escapeHtml(row.model)}">
      <div class="model-card-head">
        <div>
          <h3>${escapeHtml(row.model)}</h3>
          <div class="muted">${fmtCount(summary.recorded.length)} recorded | ${fmtCount(summary.settled.length)} settled | ${fmtCount(summary.openSamples.length)} open</div>
        </div>
        <div class="badge-row">
          ${modelTag(row.model, row.isLegacy)}
          <span class="badge">${fmtCount(row.meta.researchFiles)} research files</span>
        </div>
      </div>
      <div>
        <div class="big ${moneyTone(summary.realized)}">${fmtMoney(summary.realized)}</div>
        <div class="muted">Settled realized PnL</div>
      </div>
      ${statsGridHtml([
        { label: "Win Rate", value: fmtPct(winRate) },
        { label: "Invested", value: fmtMoney(summary.totalInvestment) },
        { label: "PnL %", value: fmtPct(summary.realizedPnlPct, 2), tone: moneyTone(summary.realizedPnlPct) },
        { label: "Max DD", value: fmtMoney(summary.maxDrawdown), tone: moneyTone(-summary.maxDrawdown) },
        { label: "Expectancy", value: fmtMoney(summary.expectancy), tone: moneyTone(summary.expectancy) },
        { label: "Avg Edge", value: fmtEdge(summary.avgEdgeCents), tone: ratioTone(summary.avgEdgeCents) },
        { label: "Avg Prob", value: fmtPct(summary.avgProb, 2) },
        { label: "Skipped", value: fmtCount(summary.skipCount) },
        { label: "Open Samples", value: fmtCount(summary.openSamples.length) },
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
  $("focused-model-subtitle").textContent = `${modelName} inside the research-only sampler, with bucket coverage, settlement quality, and current open exposure in one place.`;
  $("focused-model-badges").innerHTML = [
    `<span class="badge ${moneyTone(summary.realized)}">realized ${fmtMoney(summary.realized)}</span>`,
    `<span class="badge">invested ${fmtMoney(summary.totalInvestment)}</span>`,
    `<span class="badge ${moneyTone(summary.realizedPnlPct)}">pnl ${fmtPct(summary.realizedPnlPct, 2)}</span>`,
    `<span class="badge ${moneyTone(-summary.maxDrawdown)}">max dd ${fmtMoney(summary.maxDrawdown)}</span>`,
    `<span class="badge">settled ${fmtCount(summary.settled.length)}</span>`,
    `<span class="badge">open ${fmtCount(summary.openSamples.length)}</span>`,
    `<span class="badge">win rate ${fmtPct(winRate)}</span>`,
    `<span class="badge">avg edge ${fmtEdge(summary.avgEdgeCents)}</span>`,
    `<span class="badge">logs ${fmtCount(meta.researchFiles)} research</span>`,
  ].join("");
}

function overviewHtml(modelName, summary, meta) {
  const yesSide = summary.sideSummary.find((row) => row.side === "YES") || { recorded: 0, settled: 0, realized: 0 };
  const noSide = summary.sideSummary.find((row) => row.side === "NO") || { recorded: 0, settled: 0, realized: 0 };
  const config = summary.startConfig || {};

  const overviewMetrics = statsGridHtml([
    { label: "Recorded", value: fmtCount(summary.recorded.length), sub: `${fmtCount(summary.openSamples.length)} still open` },
    { label: "Settled", value: fmtCount(summary.settled.length), sub: `${fmtCount(summary.wins.length)} wins / ${fmtCount(summary.losses.length)} losses` },
    { label: "Realized PnL", value: fmtMoney(summary.realized), tone: moneyTone(summary.realized) },
    { label: "Total Investment", value: fmtMoney(summary.totalInvestment), sub: `${fmtMoney(summary.openInvestment)} still open notional` },
    { label: "Settled Investment", value: fmtMoney(summary.settledInvestment), sub: "Basis for realized PnL %" },
    { label: "Realized PnL %", value: fmtPct(summary.realizedPnlPct, 2), tone: moneyTone(summary.realizedPnlPct), sub: "Realized PnL / settled investment" },
    { label: "Max Drawdown", value: fmtMoney(summary.maxDrawdown), tone: moneyTone(-summary.maxDrawdown), sub: summary.maxDrawdownPct == null ? "No positive equity peak yet" : `${fmtPct(summary.maxDrawdownPct, 2)} from prior peak` },
    { label: "Expectancy", value: fmtMoney(summary.expectancy), tone: moneyTone(summary.expectancy), sub: "Average settled PnL per sample" },
    { label: "Win Rate", value: fmtPct(summary.settled.length ? summary.wins.length / summary.settled.length : 0) },
    { label: "Avg Edge", value: fmtEdge(summary.avgEdgeCents), tone: ratioTone(summary.avgEdgeCents) },
    { label: "Avg Probability", value: fmtPct(summary.avgProb, 2) },
    { label: "Skipped", value: fmtCount(summary.skipCount), sub: summary.skipCounts[0] ? `Top skip: ${summary.skipCounts[0][0]}` : "No skipped samples yet" },
  ], "detail-metrics");

  const configCards = statsGridHtml([
    { label: "Min Edge", value: config.min_edge_cents == null ? "n/a" : `${num(config.min_edge_cents, 0).toFixed(1)}c` },
    { label: "Tau Gate", value: config.min_tau_minutes == null ? "n/a" : `${num(config.min_tau_minutes, 0)}-${num(config.max_tau_minutes, 0)}m` },
    { label: "Price Band", value: config.price_band_min_cents == null ? "n/a" : `${num(config.price_band_min_cents, 0)}-${num(config.price_band_max_cents, 0)}c` },
    { label: "Quote Max Age", value: config.quote_max_age_seconds == null ? "n/a" : `${num(config.quote_max_age_seconds, 0).toFixed(1)}s` },
    { label: "Contracts / Sample", value: config.contracts_per_sample == null ? "n/a" : fmtCount(config.contracts_per_sample) },
    { label: "Slippage", value: config.slippage_pct == null ? "n/a" : `${num(config.slippage_pct, 0).toFixed(1)}%` },
  ]);

  const sideCards = statsGridHtml([
    { label: "YES Lane", value: `${sideTag("YES")} ${fmtCount(yesSide.recorded)} / ${fmtCount(yesSide.settled)}`, sub: "Recorded / Settled" },
    { label: "NO Lane", value: `${sideTag("NO")} ${fmtCount(noSide.recorded)} / ${fmtCount(noSide.settled)}`, sub: "Recorded / Settled" },
    { label: "YES Realized", value: fmtMoney(yesSide.realized), tone: moneyTone(yesSide.realized) },
    { label: "NO Realized", value: fmtMoney(noSide.realized), tone: moneyTone(noSide.realized) },
    { label: "YES PnL %", value: fmtPct(yesSide.pnlPct, 2), tone: moneyTone(yesSide.pnlPct) },
    { label: "NO PnL %", value: fmtPct(noSide.pnlPct, 2), tone: moneyTone(noSide.pnlPct) },
  ]);

  return `
    <div class="detail-grid-two">
      ${panelSection("Overview KPIs", `${modelName} top-line research coverage and settled outcome quality.`, overviewMetrics)}
      ${panelSection("Sampler Config", `${modelName} latest recorded research sampler settings.`, configCards)}
    </div>
    <div class="detail-grid-two">
      ${panelSection("Cumulative Settled PnL", `${modelName} research ledger realized PnL over time.`, `<div class="chart"><svg viewBox="0 0 800 260" preserveAspectRatio="none">${pnlSvg(summary.summaryPoints)}</svg></div><div class="legend-note">${escapeHtml(modelName)} latest cumulative PnL ${fmtMoney(summary.latestSummary.cumPnl)} • logs ${fmtCount(meta.researchFiles)} research</div>`)}
      ${panelSection("Side Summary", `${modelName} chosen-side research mix and settled side outcomes.`, sideCards)}
    </div>
  `;
}

function bucketTable(rows, empty, bucketLabel = "Bucket", bucketKind = "generic") {
  return tableHtml([
    { label: bucketLabel, render: (row) => bucketTag(row.label, bucketKind) },
    { label: "Recorded", render: (row) => fmtCount(row.recorded) },
    { label: "Settled", render: (row) => fmtCount(row.settled) },
    { label: "Wins", render: (row) => fmtCount(row.wins) },
    { label: "Losses", render: (row) => fmtCount(row.losses) },
    { label: "Win Rate", render: (row) => fmtPct(row.winRate) },
    { label: "Realized PnL", render: (row) => `<span class="${moneyTone(row.realized)}">${fmtMoney(row.realized)}</span>` },
    { label: "Expectancy", render: (row) => `<span class="${moneyTone(row.expectancy)}">${fmtMoney(row.expectancy)}</span>` },
  ], rows, empty);
}

function bucketSideSplitTable(totalRows, sideRows, empty, bucketLabel = "Bucket", bucketKind = "generic") {
  const rows = totalRows.map((totalRow, index) => ({
    total: totalRow,
    yes: sideRows.YES[index] || { recorded: 0, settled: 0, winRate: 0, realized: 0 },
    no: sideRows.NO[index] || { recorded: 0, settled: 0, winRate: 0, realized: 0 },
  }));
  return tableHtml([
    { label: bucketLabel, render: (row) => bucketTag(row.total.label, bucketKind) },
    { label: "Total Rec / Set", render: (row) => `${fmtCount(row.total.recorded)} / ${fmtCount(row.total.settled)}` },
    { label: "Total PnL", render: (row) => `<span class="${moneyTone(row.total.realized)}">${fmtMoney(row.total.realized)}</span>` },
    { label: "YES Rec / Set", render: (row) => `${fmtCount(row.yes.recorded)} / ${fmtCount(row.yes.settled)}` },
    { label: "YES Win", render: (row) => fmtPct(row.yes.winRate) },
    { label: "YES PnL", render: (row) => `<span class="${moneyTone(row.yes.realized)}">${fmtMoney(row.yes.realized)}</span>` },
    { label: "NO Rec / Set", render: (row) => `${fmtCount(row.no.recorded)} / ${fmtCount(row.no.settled)}` },
    { label: "NO Win", render: (row) => fmtPct(row.no.winRate) },
    { label: "NO PnL", render: (row) => `<span class="${moneyTone(row.no.realized)}">${fmtMoney(row.no.realized)}</span>` },
  ], rows, empty, 980);
}

function bucketsHtml(modelName, summary) {
  return `
    <div class="detail-grid-two">
      ${panelSection("Tau Buckets", `${modelName} bucket coverage by tau, with YES and NO side splits in the same view.`, bucketSideSplitTable(summary.tauBreakdown, summary.tauBreakdownBySide, `No tau buckets yet for ${modelName}.`, "Tau", "tau"))}
      ${panelSection("Price Buckets", `${modelName} chosen contract price buckets, split by YES and NO outcomes.`, bucketSideSplitTable(summary.priceBreakdown, summary.priceBreakdownBySide, `No price buckets yet for ${modelName}.`, "Price", "price"))}
    </div>
    <div class="detail-grid-two">
      ${panelSection("Probability Buckets", `${modelName} chosen-side probability buckets, split by YES and NO performance.`, bucketSideSplitTable(summary.probabilityBreakdown, summary.probabilityBreakdownBySide, `No probability buckets yet for ${modelName}.`, "Probability", "probability"))}
      ${panelSection("Edge Buckets", `${modelName} post-cost edge buckets, split by YES and NO performance.`, bucketSideSplitTable(summary.edgeBreakdown, summary.edgeBreakdownBySide, `No edge buckets yet for ${modelName}.`, "Edge", "edge"))}
    </div>
  `;
}

function samplesHtml(modelName, summary) {
  const recentSamples = tableHtml([
    { label: "Recorded", render: (row) => fmtDate(row.recordedAt) },
    { label: "Ticker", render: (row) => tickerLabel(row.ticker) },
    { label: "Side", render: (row) => sideTag(row.side) },
    { label: "Tau", render: (row) => row.tauMinutes == null ? "n/a" : `${num(row.tauMinutes, 0).toFixed(2)}m` },
    { label: "Price", render: (row) => row.referencePriceCents == null ? "n/a" : `${row.referencePriceCents}c` },
    { label: "Prob", render: (row) => row.chosenSideProbability == null ? "n/a" : fmtPct(row.chosenSideProbability, 2) },
    { label: "Edge", render: (row) => row.chosenEdgeCents == null ? "n/a" : fmtEdge(row.chosenEdgeCents) },
    { label: "Status", render: (row) => statusTag(row.status) },
  ], summary.recorded.slice(0, 60), `No recorded samples yet for ${modelName}.`, 820);

  const settledSamples = tableHtml([
    { label: "Settled", render: (row) => fmtDate(row.settledAt) },
    { label: "Ticker", render: (row) => tickerLabel(row.ticker) },
    { label: "Side", render: (row) => sideTag(row.side) },
    { label: "Result", render: (row) => resultTag(row.settlementResult) },
    { label: "Price", render: (row) => row.referencePriceCents == null ? "n/a" : `${row.referencePriceCents}c` },
    { label: "Cash", render: (row) => fmtMoney(row.cashRequired) },
    { label: "PnL", render: (row) => `<span class="${moneyTone(row.pnl)}">${fmtMoney(row.pnl)}</span>` },
    { label: "Cum PnL", render: (row) => row.cumPnl == null ? "n/a" : `<span class="${moneyTone(row.cumPnl)}">${fmtMoney(row.cumPnl)}</span>` },
  ], summary.settled.slice(0, 60), `No settled samples yet for ${modelName}.`, 860);

  const openSamples = tableHtml([
    { label: "Recorded", render: (row) => fmtDate(row.recordedAt) },
    { label: "Ticker", render: (row) => tickerLabel(row.ticker) },
    { label: "Side", render: (row) => sideTag(row.side) },
    { label: "Tau Bucket", render: (row) => bucketTag(row.tauBucket, "tau") },
    { label: "Price Bucket", render: (row) => bucketTag(row.priceBucket, "price") },
    { label: "Prob Bucket", render: (row) => bucketTag(row.probabilityBucket, "probability") },
    { label: "Edge Bucket", render: (row) => bucketTag(row.edgeBucket, "edge") },
    { label: "Cash", render: (row) => fmtMoney(row.cashRequired) },
  ], summary.openSamples.slice(0, 60), `No open research samples for ${modelName}.`, 860);

  return `
    <div class="detail-grid-two">
      ${panelSection("Recent Samples", `${modelName} chosen-side research samples, newest first.`, recentSamples)}
      ${panelSection("Open Samples", `${modelName} recorded samples waiting for market resolution.`, openSamples)}
    </div>
    <div class="detail-grid-two">
      ${panelSection("Settlements", `${modelName} resolved research samples with observation-only PnL.`, settledSamples)}
      ${panelSection("Per-Ticker Breakdown", `${modelName} research sample counts and settled PnL by ticker.`, tableHtml([
        { label: "Ticker", render: (row) => `<span class="mono">${escapeHtml(row.ticker)}</span>` },
        { label: "Recorded", render: (row) => fmtCount(row.recorded) },
        { label: "Settled", render: (row) => fmtCount(row.settled) },
        { label: "Wins", render: (row) => fmtCount(row.wins) },
        { label: "Losses", render: (row) => fmtCount(row.losses) },
        { label: "Open", render: (row) => fmtCount(row.open) },
        { label: "Realized", render: (row) => `<span class="${moneyTone(row.realized)}">${fmtMoney(row.realized)}</span>` },
      ], summary.tickerBreakdown, `No per-ticker sample data yet for ${modelName}.`))}
    </div>
  `;
}

function diagnosticsHtml(modelName, summary, meta) {
  const config = summary.startConfig || {};
  const latestRecorded = summary.latestRecorded;
  const latestSkipped = summary.latestSkipped;

  const healthCards = statsGridHtml([
    { label: "Research Files", value: fmtCount(meta.researchFiles), sub: "JSONL files loaded in the current lookback" },
    { label: "Top Skip", value: summary.skipCounts[0] ? reasonTag(summary.skipCounts[0][0]) : "n/a", sub: summary.skipCounts[0] ? `${fmtCount(summary.skipCounts[0][1])} skipped samples` : "No skipped samples yet" },
    { label: "Latest Snapshot", value: fmtDate(summary.latestSummary.label), sub: `Open ${fmtCount(summary.latestSummary.open)} | Settled ${fmtCount(summary.latestSummary.settled)}` },
    { label: "Latest Cum PnL", value: fmtMoney(summary.latestSummary.cumPnl), tone: moneyTone(summary.latestSummary.cumPnl) },
    { label: "Min Edge", value: config.min_edge_cents == null ? "n/a" : `${num(config.min_edge_cents, 0).toFixed(1)}c` },
    { label: "Price Band", value: config.price_band_min_cents == null ? "n/a" : `${num(config.price_band_min_cents, 0)}-${num(config.price_band_max_cents, 0)}c` },
    { label: "Tau Gate", value: config.min_tau_minutes == null ? "n/a" : `${num(config.min_tau_minutes, 0)}-${num(config.max_tau_minutes, 0)}m` },
    { label: "Quote Max Age", value: config.quote_max_age_seconds == null ? "n/a" : `${num(config.quote_max_age_seconds, 0).toFixed(1)}s` },
  ], "detail-metrics");

  const latestRecordedStats = latestRecorded
    ? statsGridHtml([
        { label: "Recorded At", value: fmtDate(latestRecorded.loggedAt) },
        { label: "Ticker / Side", value: `${tickerLabel(latestRecorded.ticker)} ${sideTag(latestRecorded.side)}` },
        { label: "Pred YES / NO", value: `${fmtPct(latestRecorded.predictedYesProbability, 2)} / ${fmtPct(latestRecorded.predictedNoProbability, 2)}` },
        { label: "Chosen Prob", value: latestRecorded.chosenSideProbability == null ? "n/a" : fmtPct(latestRecorded.chosenSideProbability, 2) },
        { label: "Basis Prob", value: latestRecorded.featureBasisMarketProb == null ? "n/a" : fmtPct(latestRecorded.featureBasisMarketProb, 2) },
        { label: "Raw Model Edge", value: latestRecorded.rawModelEdge == null ? "n/a" : fmtPct(latestRecorded.rawModelEdge, 2), tone: latestRecorded.rawModelEdge == null ? "" : ratioTone(latestRecorded.rawModelEdge) },
        { label: "Chosen Edge", value: latestRecorded.chosenPostCostEdge == null ? "n/a" : fmtEdge(latestRecorded.chosenPostCostEdge * 100), tone: latestRecorded.chosenPostCostEdge == null ? "" : ratioTone(latestRecorded.chosenPostCostEdge) },
        { label: "Price / Tau", value: `${latestRecorded.referencePriceCents == null ? "n/a" : `${latestRecorded.referencePriceCents}c`} / ${latestRecorded.tauMinutes == null ? "n/a" : `${num(latestRecorded.tauMinutes, 0).toFixed(2)}m`}` },
        { label: "Buckets", value: `${bucketTag(latestRecorded.tauBucket, "tau")} ${bucketTag(latestRecorded.priceBucket, "price")} ${bucketTag(latestRecorded.probabilityBucket, "probability")} ${bucketTag(latestRecorded.edgeBucket, "edge")}` },
        { label: "Quote Inputs", value: `bid ${latestRecorded.yesBidCents == null ? "?" : `${latestRecorded.yesBidCents}c`} / ask ${latestRecorded.yesAskCents == null ? "?" : `${latestRecorded.yesAskCents}c`} / buy YES ${latestRecorded.buyYesPriceCents == null ? "?" : `${latestRecorded.buyYesPriceCents}c`} / buy NO ${latestRecorded.buyNoPriceCents == null ? "?" : `${latestRecorded.buyNoPriceCents}c`}` },
        { label: "Quote Spread / Age", value: `${latestRecorded.quoteSpreadCents == null ? "n/a" : `${latestRecorded.quoteSpreadCents}c`} / ${latestRecorded.quoteAgeSeconds == null ? "n/a" : `${num(latestRecorded.quoteAgeSeconds, 0).toFixed(2)}s`}` },
        { label: "Estimated Cash", value: fmtMoney(latestRecorded.cashRequired) },
      ], "detail-metrics")
    : `<div class="empty">No recorded sample payload yet for ${escapeHtml(modelName)}.</div>`;

  const latestSkippedStats = latestSkipped
    ? statsGridHtml([
        { label: "Skipped At", value: fmtDate(latestSkipped.loggedAt) },
        { label: "Ticker / Side", value: `${tickerLabel(latestSkipped.ticker)} ${sideTag(latestSkipped.side)}` },
        { label: "Reason", value: reasonTag(latestSkipped.reason) },
        { label: "Pred YES / NO", value: `${latestSkipped.predictedYesProbability == null ? "n/a" : fmtPct(latestSkipped.predictedYesProbability, 2)} / ${latestSkipped.predictedNoProbability == null ? "n/a" : fmtPct(latestSkipped.predictedNoProbability, 2)}` },
        { label: "Basis Prob", value: latestSkipped.featureBasisMarketProb == null ? "n/a" : fmtPct(latestSkipped.featureBasisMarketProb, 2) },
        { label: "Chosen Edge", value: latestSkipped.chosenPostCostEdge == null ? "n/a" : fmtEdge(latestSkipped.chosenPostCostEdge * 100), tone: latestSkipped.chosenPostCostEdge == null ? "" : ratioTone(latestSkipped.chosenPostCostEdge) },
        { label: "Tau", value: latestSkipped.tauMinutes == null ? "n/a" : `${num(latestSkipped.tauMinutes, 0).toFixed(2)}m` },
        { label: "Quote Mid", value: latestSkipped.quoteMidProb == null ? "n/a" : fmtPct(latestSkipped.quoteMidProb, 2) },
        { label: "Quote Inputs", value: `bid ${latestSkipped.yesBidCents == null ? "?" : `${latestSkipped.yesBidCents}c`} / ask ${latestSkipped.yesAskCents == null ? "?" : `${latestSkipped.yesAskCents}c`} / buy YES ${latestSkipped.buyYesPriceCents == null ? "?" : `${latestSkipped.buyYesPriceCents}c`} / buy NO ${latestSkipped.buyNoPriceCents == null ? "?" : `${latestSkipped.buyNoPriceCents}c`}` },
        { label: "Quote Spread / Age", value: `${latestSkipped.quoteSpreadCents == null ? "n/a" : `${latestSkipped.quoteSpreadCents}c`} / ${latestSkipped.quoteAgeSeconds == null ? "n/a" : `${num(latestSkipped.quoteAgeSeconds, 0).toFixed(2)}s`}` },
      ], "detail-metrics")
    : `<div class="empty">No skipped sample payload yet for ${escapeHtml(modelName)}.</div>`;

  const skipTable = tableHtml([
    { label: "Skip Reason", render: (row) => reasonTag(row.reason) },
    { label: "Count", render: (row) => fmtCount(row.count) },
    { label: "Share", render: (row) => summary.skipCount ? fmtPct(row.count / summary.skipCount) : "0.0%" },
  ], summary.skipCounts.map(([reason, count]) => ({ reason, count })), `No skipped sample reasons yet for ${modelName}.`, 640);

  const timelineTable = tableHtml([
    { label: "Time", render: (row) => fmtDate(row.time) },
    { label: "Type", render: (row) => eventTypeTag(row.type) },
    { label: "Ticker", render: (row) => tickerLabel(row.ticker) },
    { label: "Side", render: (row) => sideTag(row.side || "n/a") },
    { label: "Result", render: (row) => resultTag(row.result) },
    { label: "PnL", render: (row) => row.pnl == null ? "n/a" : `<span class="${moneyTone(row.pnl)}">${fmtMoney(row.pnl)}</span>` },
  ], summary.timeline, `No research timeline events yet for ${modelName}.`, 760);

  return `
    <div class="detail-grid-two">
      ${panelSection("Research Health", `${modelName} research-safe gating, log coverage, and latest settled snapshot.`, healthCards)}
      ${panelSection("Skip Reasons", `${modelName} most common reasons a research sample was skipped instead of recorded.`, `<div class="chart"><svg viewBox="0 0 420 260" preserveAspectRatio="none">${skipSvg(summary.skipCounts)}</svg></div><div class="legend-note">${escapeHtml(modelName)} skipped ${fmtCount(summary.skipCount)} candidate samples in the current lookback.</div>${skipTable}`)}
    </div>
    <div class="detail-grid-two">
      ${panelSection("Latest Recorded Sample", `${modelName} latest chosen-side research sample payload.`, latestRecordedStats)}
      ${panelSection("Latest Skipped Candidate", `${modelName} latest skipped candidate and its rejection context.`, latestSkippedStats)}
    </div>
    <div class="detail-grid-two">
      ${panelSection("Recent Timeline", `${modelName} most recent recorded and settled research events.`, timelineTable)}
      ${panelSection("Side Performance", `${modelName} settled outcomes split by chosen side.`, tableHtml([
        { label: "Side", render: (row) => sideTag(row.side) },
        { label: "Recorded", render: (row) => fmtCount(row.recorded) },
        { label: "Settled", render: (row) => fmtCount(row.settled) },
        { label: "Invested", render: (row) => fmtMoney(row.recordedInvestment) },
        { label: "Wins", render: (row) => fmtCount(row.wins) },
        { label: "Losses", render: (row) => fmtCount(row.losses) },
        { label: "Win Rate", render: (row) => fmtPct(row.winRate) },
        { label: "Realized", render: (row) => `<span class="${moneyTone(row.realized)}">${fmtMoney(row.realized)}</span>` },
        { label: "PnL %", render: (row) => `<span class="${moneyTone(row.pnlPct)}">${fmtPct(row.pnlPct, 2)}</span>` },
        { label: "Expectancy", render: (row) => `<span class="${moneyTone(row.expectancy)}">${fmtMoney(row.expectancy)}</span>` },
      ], summary.sideSummary, `No side summary yet for ${modelName}.`, 820))}
    </div>
  `;
}

function renderSelectedModel(modelName, summary, meta, modelCount, loadedMeta, aggregate) {
  updateResearchChrome(modelName, loadedMeta, modelCount);
  renderResearchRunOverview(aggregate);
  renderFocusedHeader(modelName, summary, meta);
  $("tab-overview").innerHTML = overviewHtml(modelName, summary, meta);
  $("tab-buckets").innerHTML = bucketsHtml(modelName, summary);
  $("tab-samples").innerHTML = samplesHtml(modelName, summary);
  $("tab-diagnostics").innerHTML = diagnosticsHtml(modelName, summary, meta);
  updateTabVisibility();
}

function renderEmptyFocus(message, loadedMeta = {}, aggregate = null) {
  updateResearchChrome("", loadedMeta, 0);
  renderResearchRunOverview(aggregate);
  $("focused-model-title").textContent = "Focused Model";
  $("focused-model-subtitle").textContent = message;
  $("focused-model-badges").innerHTML = "";
  $("tab-overview").innerHTML = `<div class="empty">${escapeHtml(message)}</div>`;
  $("tab-buckets").innerHTML = `<div class="empty">${escapeHtml(message)}</div>`;
  $("tab-samples").innerHTML = `<div class="empty">${escapeHtml(message)}</div>`;
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
    summaryMap[model] = summarizeRows(payload.researchRows || []);
    metaMap[model] = payload.meta || { researchFiles: 0 };
  }
  const aggregate = aggregateResearchSummary(summaryMap);
  const selectedModel = updateModelSelector(summaryMap, metaMap);
  renderComparison(summaryMap, metaMap);
  renderModelCards(summaryMap, metaMap);
  if (!selectedModel) {
    renderEmptyFocus("No research logs found yet. Choose a research run folder or snapshot first.", loaded.meta, aggregate);
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
  const researchByModel = await researchHandlesByModel(state.rootHandle, state.environment);
  const modelData = {};
  for (const model of [...researchByModel.keys()].sort()) {
    const researchHandles = researchByModel.get(model) || [];
    const researchRows = await Promise.all(researchHandles.map(readHandleRows));
    modelData[model] = {
      researchRows: researchRows.flat(),
      meta: { researchFiles: researchHandles.length },
    };
  }
  const fileCount = Object.values(modelData).reduce((sum, item) => sum + item.meta.researchFiles, 0);
  return {
    modelData: pruneLegacySingleModel(modelData),
    meta: {
      sourceKind: "folder",
      rootLabel: state.rootHandle?.name || "selected",
      fileCount,
    },
  };
}

async function loadFromSnapshot() {
  const modelData = {};
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - state.lookbackDays);
  cutoff.setHours(0, 0, 0, 0);

  for (const file of state.snapshotFiles) {
    const rel = (file.webkitRelativePath || file.name).replace(/\\/g, "/");
    let model = "single";
    let env = null;
    let dateKey = null;
    let match = rel.match(/\/research\/([^/]+)\/(demo|production)\/(\d{4}-\d{2}-\d{2})\/events\.jsonl$/);
    if (match) {
      model = match[1];
      env = match[2];
      dateKey = match[3];
    } else {
      match = rel.match(/\/research\/(demo|production)\/(\d{4}-\d{2}-\d{2})\/events\.jsonl$/);
      if (match) {
        env = match[1];
        dateKey = match[2];
      }
    }
    if (!match || env !== state.environment) continue;
    const d = parseDateFolder(dateKey);
    if (d && d < cutoff) continue;
    const rows = lines(await file.text());
    if (!modelData[model]) modelData[model] = { researchRows: [], meta: { researchFiles: 0 } };
    modelData[model].researchRows.push(...rows);
    modelData[model].meta.researchFiles += 1;
  }
  const fileCount = Object.values(modelData).reduce((sum, item) => sum + item.meta.researchFiles, 0);
  return {
    modelData: pruneLegacySingleModel(modelData),
    meta: {
      sourceKind: "snapshot",
      rootLabel: state.snapshotFiles.length ? "uploaded snapshot" : "snapshot",
      fileCount,
    },
  };
}

async function refresh() {
  if (state.loading) return;
  state.loading = true;
  sync("Refreshing research logs...");
  try {
    const loaded = state.rootHandle
      ? await loadFromHandles()
      : state.snapshotFiles.length
        ? await loadFromSnapshot()
        : null;
    if (!loaded) {
      state.lastLoaded = null;
      renderEmptyFocus("Choose a research run folder or snapshot first.", {
        sourceKind: state.snapshotFiles.length ? "snapshot" : "idle",
        rootLabel: state.snapshotFiles.length ? "uploaded snapshot" : "none",
        fileCount: 0,
      });
      sync("Choose a research run folder or snapshot first");
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
  if (!state.rootHandle) return;
  const envs = new Set();
  const areaDir = await getDir(state.rootHandle, "research");
  if (!areaDir) return;
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
  state.timer = setInterval(refresh, state.refreshSeconds * 1000);
}

async function chooseFolder() {
  if (!window.showDirectoryPicker) {
    sync("Folder picker unsupported here; use snapshot loader", "bad");
    return;
  }
  try {
    state.rootHandle = await window.showDirectoryPicker({ mode: "read" });
    state.snapshotFiles = [];
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
  state.rootHandle = null;
  await refresh();
});
document.querySelectorAll("[data-tab]").forEach((button) => {
  button.addEventListener("click", () => {
    state.selectedTab = button.dataset.tab || "overview";
    updateTabVisibility();
  });
});

restartTimer();
renderEmptyFocus("Choose a research run folder or snapshot first.");
sync("Waiting for research logs");
