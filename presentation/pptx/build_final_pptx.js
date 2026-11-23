#!/usr/bin/env node
/*
 * Build the final-presentation deck as a PowerPoint file in UNC colors and type.
 *
 *   npm install --prefix <dir> pptxgenjs@4.0.1
 *   NODE_PATH=<dir>/node_modules node presentation/pptx/build_final_pptx.js [output.pptx] [font]
 *
 * On the Mac mini, install into /Volumes/MacDev/Developer/build/comp690-pptx with ~/.local/bin/macdev npm.
 * Defaults: presentation/final.pptx and the font "Open Sans", a UNC brand typeface for digital and print use
 * (identity.unc.edu/brand/typography); install it on the presenting computer. Passing a wider font such as
 * Verdana gives a stricter check of text fit. Colors are the UNC palette (its-commons.unc.edu/brand/colors).
 * Every number on a slide is read from results/ (the frozen analysis, the benchmark and the supplementary studies);
 * the console prints the key values so the talk script can be checked against them.
 * Speaker notes come from the talk script presentation/final_script.md: for each "## N" section, its "> " lines
 * (what to say) and its "If asked:" line. Edit the script, then rebuild. The script's timing is printed as a check.
 * Slide titles also come from the script's "## N · Title（start time）" headings; backup sections have no time.
 * To add a slide, write its section in the script (renumber the later sections and refresh their start times) and
 * add its block below at the same position; the build stops if a block and its script section disagree.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..", "..");
const OUT = path.resolve(process.argv[2] || path.join(ROOT, "presentation", "final.pptx"));
const FONT = process.argv[3] || "Open Sans";

// UNC palette. Carolina Blue as text only at 18 pt and above (or 14 pt bold).
const UNC = {
  carolina: "4B9CD3", navy: "13294B", white: "FFFFFF", bolin: "2C5080", cloud: "EDF5FB",
  basin: "5B6670", grey: "A2AAAD", cornerstone: "CFD3D5",
};

// ---------- reading results ----------
const rel = (p) => path.join(ROOT, p);
const readJson = (p) => JSON.parse(fs.readFileSync(rel(p), "utf8"));
function readCsv(p) { // RFC 4180 subset: quoted fields may hold commas
  const s = fs.readFileSync(rel(p), "utf8");
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (quoted) {
      if (ch === "\"" && s[i + 1] === "\"") { field += "\""; i++; } else if (ch === "\"") quoted = false; else field += ch;
    } else if (ch === "\"") quoted = true;
    else if (ch === ",") { row.push(field); field = ""; } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && s[i + 1] === "\n") i++;
      row.push(field); rows.push(row); row = []; field = "";
    } else field += ch;
  }
  if (field !== "" || row.length) { row.push(field); rows.push(row); }
  const [head, ...rest] = rows.filter((r) => r.length > 1 || r[0] !== "");
  return rest.map((r) => Object.fromEntries(head.map((k, i) => [k, r[i]])));
}
function find(list, pred, what) {
  const hit = list.find(pred);
  if (!hit) throw new Error(`not found: ${what}`);
  return hit;
}
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const MINUS = "−";
const sgn = (v, d = 1) => (v < 0 ? MINUS : "+") + Math.abs(v).toFixed(d);
const neg = (v, d = 1) => (v < 0 ? MINUS : "") + Math.abs(v).toFixed(d);
const ci = (lo, hi, d = 1) => `[${sgn(lo, d)}, ${sgn(hi, d)}]`;
const int = (v) => Math.round(v).toLocaleString("en-US");

// BCI IV 2a, frozen analysis (three seeds)
const primary = readCsv("results/analysis/conditions_primary.csv");
const primaryRow = (c, p = "fp32") => find(primary, (r) => r.condition === c && r.precision === p, `${c} ${p} primary`);
const acc = (c, p = "fp32") => 100 * Number(primaryRow(c, p).acc_mean);
const randomMean = (k) => mean([1, 2, 3].map((i) => acc(`random${i}-${k}`)));
const conditions2a = [...new Set(primary.map((r) => r.condition))];
const int8MaxChange = Math.max(...conditions2a.map((c) => Math.abs(acc(c, "int8") - acc(c))));
const nPeople2a = new Set(readCsv("results/analysis/per_participant_primary.csv").map((r) => r.subject)).size;

// ten seeds per electrode count
const curveRows = readCsv("results/supplementary/electrode_count/electrode_count_curve.csv")
  .filter((r) => r.seeds === "0-9" && r.precision === "fp32");
const COUNTS = [22, 20, 18, 16, 14, 12, 10, 8];
const curve = COUNTS.map((k) => Number(find(curveRows,
  (r) => r.condition === (k === 22 ? "full-22" : `relevance-${k}`), `curve ${k}`).acc_mean));
const ec = readJson("results/supplementary/electrode_count/analysis.json");
const cmp10 = (a, b) => find(ec.comparisons, (c) => c.a === a && c.b === b && c.seeds === "0-9" && c.precision === "fp32",
  `${a} vs ${b}, ten seeds`);
const c16 = cmp10("relevance-16", "full-22"), c8 = cmp10("relevance-8", "full-22");
const group16 = find(ec.by_baseline_group, (r) => r.condition === "relevance-16", "group relevance-16");
const amp = find(ec.amplitude_what_if, (r) => r.scaling === "protocol" && r.a === "relevance-16" && r.b === "full-22",
  "amplitude protocol row");
const kept16 = ec.selection.kept_at_16_count_per_electrode;
const byRow = ec.selection.by_scalp_row;
const frontalDrops = Object.entries(byRow).filter(([k]) => k.startsWith("front")).reduce((s, [, v]) => s + v.dropped_at_16, 0);
const allDrops = Object.values(byRow).reduce((s, v) => s + v.dropped_at_16, 0);
const selByP = readCsv("results/supplementary/electrode_count/selection_by_participant.csv");
const fisherKept = mean(selByP.map((r) => Number(r.fisher_day1_kept)));
const fisherDropped = mean(selByP.map((r) => Number(r.fisher_day1_dropped)));
const lowGroup = ec.groups.below, threshold = ec.groups.threshold_pct;

// redundancy
const red = readJson("results/supplementary/redundancy_check/summary.json");
const meanCorr = mean(red.participants.map((p) => p.original.mean_abs_correlation));
function sameSubsets(k) {
  let n = 0;
  for (let s = 1; s <= nPeople2a; s++) {
    const o = readJson(`results/final/rankings/sub-${String(s).padStart(2, "0")}_seed-0.json`).orders;
    const a = new Set(o.proposed.slice(0, k)), b = o.relevance.slice(0, k);
    if (b.every((e) => a.has(e))) n++;
  }
  return n;
}
const same16 = sameSubsets(16), same8 = sameSubsets(8);

// brain regions
const reg = readJson("results/supplementary/region_study/analysis.json");
const REGIONS = reg.regions.fine;
const shap = reg.shapley_pp.cv;
const nRegionSets = 2 ** Object.keys(REGIONS).length - 1;
const regionModels = reg.records.cv + reg.records.test;
const regionVsRel = reg.relevance_comparison.filter((r) => r.method === "own_cv").map((r) => r.minus_relevance_pp);
const regionOf = Object.fromEntries(Object.entries(REGIONS).flatMap(([r, es]) => es.map((e) => [e, r])));
const topRegion = (cls) => Object.keys(REGIONS).reduce((best, r) =>
  (shap[r].recall[cls].mean > shap[best].recall[cls].mean ? r : best));

// joint selection (BCI IV 2a, exploratory)
const sel = readJson("results/supplementary/selection_study/analysis.json");
const SEL_COUNTS = [4, 6, 8, 10, 12, 16];
const selAcc = (m, k) => find(sel.accuracy, (r) => r.method === m && r.k === k, `selection ${m} ${k}`);
const riem8 = selAcc("riemann", 8);
const riemCtrl = find(sel.hardware_control, (r) => r.kind === "minus retrained relevance" && r.method === "riemann" && r.k === 8,
  "riemann control");
const selFull = selAcc("full-22", 22).acc_mean_pct;
const avgOverCounts = (m) => mean(SEL_COUNTS.map((k) => selAcc(m, k).acc_mean_pct));
const selMethods = [...new Set(sel.accuracy.map((r) => r.method))].filter((m) => m !== "full-22");
const lowestMethod = selMethods.reduce((lo, m) => (avgOverCounts(m) < avgOverCounts(lo) ? m : lo));

// Lee 2019 replication (registered before scoring)
const rep = readJson("results/supplementary/replication_study/analysis.json");
const repCsv = readCsv("results/supplementary/replication_study/hypotheses.csv");
const repR = readJson("results/supplementary/replication_riemann_study/analysis.json");
if (rep.hypotheses.length !== repCsv.length) throw new Error("replication hypotheses: JSON and CSV differ");
const repAll = repCsv.map((r, i) => { // the CSV carries each row's condition and alignment
  if (rep.hypotheses[i].hypothesis !== r.hypothesis) throw new Error(`replication row ${i}: JSON and CSV differ`);
  return Object.assign({}, rep.hypotheses[i], { condition: r.condition, alignment: r.alignment });
});
const H = (id, cond = "") => {
  const hits = repAll.filter((h) => h.hypothesis === id && h.condition === cond);
  if (hits.length !== 1) throw new Error(`hypothesis ${id} ${cond}: ${hits.length} rows`);
  return hits[0];
};
const R = (id) => find(repR.hypotheses, (h) => h.hypothesis === id, id);
const repConditions = readCsv("results/supplementary/replication_study/conditions.csv");
const fullLee = find(repConditions, (r) => r.condition.startsWith("full-"), "full Lee condition");
const leeElectrodes = Number(fullLee.condition.split("-")[1]);
const fullLeeAcc = Number(fullLee.accuracy);
const nPeopleLee = H("H1").n;
const bitsLee = Math.max(...["full-62", "relevance-45", "relevance-16"].map((c) => Math.abs(H("H4", c).estimate_pp)),
  ...["full-62", "relevance-16"].map((c) => Math.abs(H("H4b", c).estimate_pp)));

// bit width (simulated integer grids, all archived models)
const qs = readJson("results/supplementary/quant_study/analysis.json");
const ptq = (s) => find(qs.ptq_pooled, (r) => r.condition === "all conditions" && r.setting === s, `ptq ${s}`);
const qr = readJson("results/supplementary/quant_rule_study/analysis.json");
const rule = (s) => find(qr.settings, (r) => r.condition === "all conditions" && r.setting === s, `rule ${s}`);
const bitMap = rule("pooled-0.5"), bit8 = rule("8/8");
const byteSaving = 100 * (1 - bitMap.weight_bytes / bit8.weight_bytes);
const nModels = qs.records.ptq;

// INT8 speed
const bench = readJson("results/benchmark/benchmark.json");
const benchRow = (c) => find(bench.rows, (r) => r.condition === c && r.precision === "fp32", `benchmark ${c}`);
function layerTimes(key) { // [temporal conv, grouped spatial conv, everything else], microseconds
  const prof = bench.op_profile_us[key];
  const conv = (id) => Object.entries(prof)
    .filter(([name]) => new RegExp(`^(QLinear)?Conv:node_${id}(?!\\d)`).test(name))
    .reduce((sum, [, us]) => sum + us, 0);
  const total = Object.values(prof).reduce((sum, us) => sum + us, 0);
  const temporal = conv("Conv_90"), spatial = conv("Conv_92");
  return [temporal, spatial, total - temporal - spatial];
}
const t32 = layerTimes("full-22 fp32"), t8 = layerTimes("full-22 int8");
const fullCost = bench.layer_costs["full-22"];
const firstLayerShare = 100 * fullCost[0].macs / fullCost.reduce((s, l) => s + l.macs, 0);
const i8 = readJson("results/supplementary/int8_study/analysis.json");
const lat = i8.latency_full22_ms;
const costs = readCsv("results/supplementary/int8_study/costs.csv");
const costOf = (k) => find(costs, (r) => Number(r.electrodes) === k && Number(r.F2) === 16, `cost ${k}`);
const macsSaved = (col) => 100 * (1 - Number(costOf(8)[col]) / Number(costOf(22)[col]));
const machines = readCsv("results/supplementary/int8_portability/machines.csv");
const m4 = find(machines, (r) => r.cpu === "Apple M4", "M4 machine");
const latency = readCsv("results/supplementary/hw_latency/latency.csv");
const latUs = (machine, cond, variant) => Number(find(latency,
  (r) => r.machine === machine && r.timed_condition === cond && r.variant === variant, `${machine} ${cond} ${variant}`).median_us);
const FINAL = "relevance-16-prune50";
const timedCpus = [...new Set(latency.map((r) => r.cpu))];
const slowerCpus = timedCpus.filter((cpu) => latency.filter((r) => r.cpu === cpu && r.timed_condition === FINAL)
  .some((r) => r.variant === "int8" && Number(r.ratio_to_fp32) > 1));
const cpuName = (s) => s.replace(/\((R|TM)\)/g, "").replace(/ @.*$/, "").replace(/\bCPU\b/g, "")
  .replace(/ \d+-Core Processor$/, "").replace(/\s+/g, " ").trim();
const extName = (e) => (e.includes("VNNI") ? "AVX-512 VNNI" : e.includes("AVX-512") ? "AVX-512, no VNNI"
  : e.includes("AVX2") ? "AVX2 only" : "Arm dot product");
const cpuTypes = [...new Set(machines.map((r) => r.cpu))].map((cpu) => {
  const rows = machines.filter((r) => r.cpu === cpu);
  const rr = rows.find((r) => r.rr_acc_change_vs_fp32_pp !== "");
  return { cpu: cpuName(cpu), ext: extName(rows[0].integer_extensions), change: Number(rows[0].int8_acc_change_pp),
    changed: Number(rows[0].int8_models_changed), rr: rr ? Number(rr.rr_acc_change_vs_fp32_pp) : null };
});
const changedModels = Math.max(...cpuTypes.map((c) => c.changed));
const rrCpus = cpuTypes.filter((c) => c.rr !== null);
const nAccel = i8.n_models;

// pruning
const mb = readJson("results/supplementary/matched_budget/analysis.json");
const matched = find(mb.comparisons, (c) => c.base === "relevance-16" && c.precision === "int8" && c.a === "50" && c.b === "0",
  "matched-budget comparison");
const pruneSaving = 100 * (1 - benchRow(FINAL).macs / benchRow("relevance-16").macs);
const swappedPruned = latUs("m4-int8-study", FINAL, "int8-spatial-first");
const swappedUnpruned = latUs("m4-int8-study", "relevance-16", "int8-spatial-first");

// aligning the two days
const al = readJson("results/supplementary/alignment_study/analysis.json");
const align = (c) => find(al.alignment, (r) => r.condition === c && r.variant === "ea-all", `alignment ${c}`);

// tools: package versions from the locked environment, toolchain versions from the study records
const LOCK = fs.readFileSync(rel("uv.lock"), "utf8");
const lockVersion = (name) => {
  const m = LOCK.match(new RegExp(`\\nname = "${name}"\\nversion = "([^"]+)"`));
  if (!m) throw new Error(`uv.lock: no package ${name}`);
  return m[1];
};
const V = Object.fromEntries(["torch", "braindecode", "onnxruntime", "moabb", "pyriemann"].map((p) => [p, lockVersion(p)]));

// the missing INT8 kernel (ONNX Runtime custom operator, Apple M4)
const ok = readJson("results/supplementary/ort_kernel_study/analysis.json");
const okProtocol = readJson("results/supplementary/ort_kernel_study/protocol.json");
const okLat = readCsv("results/supplementary/ort_kernel_study/latency.csv");
const okLayers = readCsv("results/supplementary/ort_kernel_study/layers.csv");
const kUs = (c, v) => Number(find(okLat, (r) => r.condition === c && r.variant === v, `kernel latency ${c} ${v}`).us);
const kLayer = (c, v) => { // [temporal conv, grouped spatial conv, everything else], microseconds (ONNX Runtime profiler)
  const r = find(okLayers, (x) => x.condition === c && x.variant === v, `kernel layers ${c} ${v}`);
  return [Number(r.temporal), Number(r.spatial), Number(r.rest)];
};
if (ok.environment.onnxruntime !== V.onnxruntime) throw new Error("kernel study and uv.lock name different ONNX Runtime versions");
const usHalfEven = (v) => { // whole microseconds, rounded half to even like the study's own tables
  const f = Math.floor(v), d = v - f;
  return d > 0.5 || (d === 0.5 && f % 2 === 1) ? f + 1 : f;
};
const spatialMacs = fullCost[1].macs, temporalMacs = fullCost[0].macs;
if (fullCost[1].layer !== "conv_spatial" || fullCost[0].layer !== "conv_temporal") throw new Error("layer_costs order changed");
const spatialShare = 100 * spatialMacs / fullCost.reduce((s, l) => s + l.macs, 0);
const k32 = kLayer("full-22", "fp32"), k8 = kLayer("full-22", "int8"), kK = kLayer("full-22", "int8-kernel");
const nsPerMac = (us, macs) => us * 1000 / macs;
const kernelVsFp32 = kUs("full-22", "fp32") / kUs("full-22", "int8-kernel");
const swapVsFp32 = kUs("full-22", "fp32") / kUs("full-22", "int8-spatial-first");
const swapKernelVsFp32 = kUs("full-22", "fp32") / kUs("full-22", "int8-spatial-first-kernel");
const eqGrouped = ok.equivalence.grouped, eqSwapped = ok.equivalence.swapped;
const PYTHON = fs.readFileSync(rel(".python-version"), "utf8").trim();

// INT8 on emulated microcontrollers (CMSIS-NN in QEMU)
const mcu = readJson("results/supplementary/mcu_study/analysis.json");
const mcuProtocol = readJson("results/supplementary/mcu_study/protocol.json");
const mcuIns = readCsv("results/supplementary/mcu_study/instructions.csv");
const mcuMem = readCsv("results/supplementary/mcu_study/memory.csv");
const mcuLayers = readCsv("results/supplementary/mcu_study/layers.csv");
const CORES = [["m0", "Cortex-M0"], ["m4", "Cortex-M4"], ["m55", "Cortex-M55"]];
const mIns = (c, o, p, core) => Number(find(mcuIns, (r) => r.condition === c && r.order === o && r.precision === p,
  `instructions ${c} ${o} ${p}`)[core]);
const mMem = (c, o, p, core) => find(mcuMem, (r) => r.condition === c && r.order === o && r.precision === p && r.core === core,
  `memory ${c} ${o} ${p} ${core}`);
const mLayer = (c, o, core, layer) => find(mcuLayers, (r) => r.condition === c && r.order === o && r.precision === "int8"
  && r.core === core && r.layer === layer, `mcu layer ${c} ${o} ${core} ${layer}`);
const mIpm = (c, o, core, layer) => Number(mLayer(c, o, core, layer).instructions_per_mac);
const mRatio = (c, o, core) => mIns(c, o, "f32", core) / mIns(c, o, "int8", core);
const mcuAccMax = Math.max(...mcu.accuracy.map((a) => Math.abs(a.int8_minus_fp32)));
const mcuLosing = Math.max(...mcu.accuracy.map((a) => a.losing_2pp));
const gccName = mcu.tools.gcc.match(/\((Arm GNU Toolchain [^ ]+)/)[1], gccVersion = mcu.tools.gcc.match(/\) (\d+\.\d+\.\d+)/)[1];
const qemuVersion = mcu.tools.qemu.match(/(\d+\.\d+\.\d+)/)[1];
const cmsisCommit = mcu.tools.cmsis_nn_commit.slice(0, 8);
const boardOf = (core) => mcuProtocol.cores[core].board;
const cpuOf = (core) => mcuProtocol.cores[core].flags[0].replace("-mcpu=cortex-", "").toUpperCase();
const spatialIpmM4 = mIpm("full-22", "original", "m4", "l2"), temporalIpmM4 = mIpm("full-22", "original", "m4", "l1");
// the FIR filter-bank kernel for the temporal layer (scripts/mcu_kernel_study.py), the same builds with one layer replaced
const mk = readCsv("results/supplementary/mcu_kernel_study/builds.csv");
const mkChecks = readJson("results/supplementary/mcu_kernel_study/analysis.json").checks;
const mkRow = (o, core) => find(mk, (r) => r.subject === "1" && r.condition === "full-22" && r.order === o && r.core === core,
  `mcu kernel ${o} ${core}`);
const mkRatios = mk.map((r) => Number(r.total_ratio));

// what pruning removes: the last 1x1 convolution's share of the multiply-accumulates (16 electrodes, both orders)
const LAYER_NAMES = { original: { l1: "Temporal conv", l2: "Spatial conv" }, swapped: { l1: "Spatial conv", l2: "Temporal conv" } };
const macShares = (o) => {
  const rows = mcuLayers.filter((r) => r.condition === "relevance-16" && r.order === o && r.precision === "int8" && r.core === "m4"
    && Number(r.macs) > 0);
  const total = rows.reduce((s, r) => s + Number(r.macs), 0);
  const name = (l) => LAYER_NAMES[o][l] || { conv_d: "Depthwise conv", conv_p: "Last 1×1 conv (pruned)", fc: "Classifier" }[l];
  return Object.fromEntries(rows.map((r) => [name(r.layer), 100 * Number(r.macs) / total]));
};
const sharesOriginal = macShares("original"), sharesSwapped = macShares("swapped");
const totalMacs = (c, o) => mcuLayers.filter((r) => r.condition === c && r.order === o && r.precision === "int8" && r.core === "m4")
  .reduce((sum, r) => sum + Number(r.macs), 0);
const pruneSavingSwapped = 100 * (1 - totalMacs(FINAL, "swapped") / totalMacs("relevance-16", "swapped"));
const PRUNED = "Last 1×1 conv (pruned)";
const mcuPruneBefore = mIns("relevance-16", "swapped", "int8", "m4"), mcuPruneAfter = mIns(FINAL, "swapped", "int8", "m4");
const cpuPruneBefore = kUs("relevance-16", "int8-kernel"), cpuPruneAfter = kUs(FINAL, "int8-kernel");

// Longleaf: models trained and CPU-hours (summed training time) from each study's run records
function studyRuns(name) {
  const dir = rel(`results/supplementary/${name}/runs`);
  let models = 0, seconds = 0;
  const walk = (d) => fs.readdirSync(d, { withFileTypes: true }).forEach((e) => {
    const p = path.join(d, e.name);
    if (e.isDirectory()) walk(p);
    else if (e.name.endsWith(".json")) {
      const t = JSON.parse(fs.readFileSync(p, "utf8")).train_time_s;
      if (t) { models++; seconds += t; } // records of models not trained here (e.g. reference re-measurements) are skipped, as in scripts/analyze_compute.py
    }
  });
  if (fs.existsSync(dir)) walk(dir);
  return { models, hours: seconds / 3600 };
}
const ranOn = (name) => [...new Set(fs.readdirSync(rel(`results/supplementary/${name}/environments`))
  .map((f) => readJson(`results/supplementary/${name}/environments/${f}`).os.split("-")[0]))];
// The totals come from compute_summary.json (scripts/analyze_compute.py); each Longleaf study's count and hours
// are checked against its own run records, so a study pulled back without rerunning that script stops the build.
const computeSummary = readJson("results/supplementary/compute_summary.json");
const LONGLEAF_STUDIES = computeSummary.studies.filter((r) => r.where === "Longleaf" && r.models_trained > 0).map((r) => {
  const runs = studyRuns(r.study), where = ranOn(r.study);
  if (runs.models !== r.models_trained || Math.abs(runs.hours - r.cpu_hours) > 0.051) { // it rounds hours per study
    throw new Error(`${r.study}: ${runs.models} run records and ${runs.hours.toFixed(2)} h, compute_summary.json says ${r.models_trained} and ${r.cpu_hours} h`);
  }
  if (where.length !== 1 || where[0] !== "Linux") throw new Error(`${r.study} did not run only on Longleaf: ${where}`);
  return { name: r.study, models: runs.models, hours: runs.hours };
});
const qatModels = fs.readdirSync(rel("results/supplementary/quant_study/qat")).filter((f) => f.endsWith(".json")).length;
if (find(computeSummary.studies, (r) => r.study === "quant_study", "quant_study compute").fine_tunes !== qatModels) {
  throw new Error("quant_study fine-tunes disagree with compute_summary.json");
}
const longleafTrainings = computeSummary.longleaf_models, longleafHours = computeSummary.longleaf_cpu_hours;
if (LONGLEAF_STUDIES.reduce((s, r) => s + r.models, 0) !== longleafTrainings) throw new Error("compute_summary.json: Longleaf total is not the sum of its studies");
const llModels = (name) => find(LONGLEAF_STUDIES, (r) => r.name === name, `${name} runs`).models;
const longleafCpus = cpuTypes.filter((c) => c.cpu !== "Apple M4");
const seedModels = studyRuns("electrode_count").models + studyRuns("electrode_count_wave2").models;
const ecProtocol = readJson("results/supplementary/electrode_count/protocol.json");
const seedsBefore = ecProtocol.historical_seeds.length, seedsAfter = seedsBefore + ecProtocol.new_seeds.length;
if (!ranOn("electrode_count").every((os) => os === "macOS")) throw new Error("the ten-seed study did not run on the Mac");

// why 16 electrodes are enough: noise or unrelated EEG in place of the dropped electrodes, same hardware
const ov = readJson("results/supplementary/overfit_study/analysis.json");
const ovProtocol = readJson("results/supplementary/overfit_study/protocol.json");
const ovC = (ds, name) => find(ov.contrasts, (c) => c.dataset === ds && c.contrast === name, `overfit ${ds}: ${name}`);
const OVC = { info: "information in the dropped electrodes: full minus reduced plus noise", cost: "P3: reduced plus noise minus reduced",
  net: "reduced minus full, next day, all of the training day", shuffled: "reduced plus shuffled minus reduced",
  dropout: "full with channel dropout minus full" };
const OV_SETS = [["bnci2a", "BCI IV 2a"], ["lee2019", "Lee 2019"]].map(([ds, name]) =>
  ({ ds, name, full: ovProtocol.datasets[ds].full, reduced: ovProtocol.datasets[ds].reduced, seeds: ovProtocol.datasets[ds].seeds.length }));
const ovLevel = (ds, level) => find(ov.levels, (l) => l.dataset === ds && l.level === level && l.condition.startsWith("full-"),
  `overfit level ${ds} ${level}`);
const trainFits = ["final", "fold", "rfold"].map((lv) => ovLevel("bnci2a", lv).train);
if (!ov.tests_lee2019.P3.holds) throw new Error("the registered P3 test no longer holds: reword the Result 3 slide");
const ovModels = llModels("overfit_study");

// where accuracy levels off: the electrode-count curve with part of the training day
const edA = readJson("results/supplementary/electrode_data_study/analysis.json");
const edCurve = (ds, level) => edA.curves.filter((c) => c.dataset === ds && c.level === level).sort((a, b) => a.electrodes - b.electrodes);
const edData = (ds, level) => find(edA.peaks, (pk) => pk.dataset === ds && pk.level === level, `peak ${ds} ${level}`).data;
const ED_LEVELS = { bnci2a: ["final", "fold", "rfold"], lee2019: ["final", "fold"] };
const PLATEAU = 16; // the flat part of every curve starts here
const plateauDev = Math.max(...Object.entries(ED_LEVELS).flatMap(([ds, lvs]) => lvs.map((lv) => {
  const ys = edCurve(ds, lv).filter((c) => c.electrodes >= PLATEAU).map((c) => c.next_day);
  return Math.max(...ys.map((y) => Math.abs(y - mean(ys))));
})));
const bestCounts = [...new Set(edA.peaks.map((pk) => pk.best_count))].sort((a, b) => a - b);
if (edA.tests_lee2019.D1.holds || edA.tests_lee2019.D2.holds) throw new Error("a registered D1/D2 test now holds: reword the Result 4 slide");
const edModels = llModels("electrode_data_study");

// what each extra input costs: k noise inputs appended to the 16 most relevant electrodes
const ic = readJson("results/supplementary/input_cost_study/analysis.json");
if (!ic.tests_lee2019.C1.holds || ic.tests_lee2019.C2.holds) throw new Error("a registered C1/C2 outcome changed: reword Result 3 and What Did Not Work");
const icSlope = (ds, level) => find(ic.slopes, (r) => r.dataset === ds && r.level === level, `input-cost slope ${ds} ${level}`).slope;
const icRows = (ds, level) => ic.costs.filter((r) => r.dataset === ds && r.level === level).sort((a, b) => a.added - b.added);
const icData = (ds, level) => icRows(ds, level)[0].data;
ic.costs.filter((r) => r.within_cost !== null).forEach((r) => { // "as much within the training day"
  if (Math.abs(r.within_cost) < 0.8 * Math.abs(r.cost)) throw new Error(`input cost ${r.dataset} ${r.level} +${r.added}: much smaller within the training day`);
});
const c2Shares = Object.values(ic.tests_lee2019.C2.share_of_margin).map((v) => Math.round(100 * v));
if (new Set(c2Shares).size !== 1) throw new Error(`C2 shares of the margin now differ (${c2Shares}): reword What Did Not Work`);
const C2_ADDED = readJson("results/supplementary/input_cost_study/protocol.json").predictions.C2.contrast.match(/\((\d+) added minus none\)/)[1];
const icModels = llModels("input_cost_study");

// where the cost comes from: the noise weights after training (held-out runs of day 1) and eight noise draws (2a, next day)
const NM = "results/supplementary/noise_mechanism_study";
const nm = readJson(`${NM}/analysis.json`);
if (nm.reproduced[0] !== nm.reproduced[1] || nm.references_reproduced[0] !== nm.references_reproduced[1]) {
  throw new Error("noise-mechanism study: rescored models no longer equal their records");
}
const SHARE_K = OV_SETS[0].full - OV_SETS[0].reduced; // as many noise inputs as dropped electrodes
const share6 = find(nm.shares, (r) => r.dataset === "bnci2a" && r.added === SHARE_K, "noise share, 2a");
const drawCosts = nm.draws.map((d) => d.cost);
if (drawCosts.length !== 8) throw new Error(`expected 8 versions of the 16 + ${SHARE_K} noise condition, found ${drawCosts.length}`);
const drawCost = mean(drawCosts); // every version, both placements of each moved draw
const info2a = -ovC("bnci2a", OVC.net).mean - drawCost; // all minus fewer electrodes, minus the cost of the extra inputs
const nmTests = Object.fromEntries(fs.readdirSync(rel(`${NM}/test`)).filter((f) => f.endsWith(".json"))
  .map((f) => [f, readJson(`${NM}/test/${f}`)]));
const personMean = (rows, k) => { // seeds averaged within each participant, then participants
  const by = {};
  rows.forEach((r) => (by[r.subject] = by[r.subject] || []).push(r[k]));
  return mean(Object.values(by).map(mean));
};
const ref16 = Object.values(nmTests).filter((r) => r.arm === "relevance-16");
const nextDayRecovered = nm.draws.map((d) => { // noise weights zeroed, batch norms re-estimated: share of the cost won back
  const t = Object.values(nmTests).filter((r) => r.arm === d.arm);
  return (personMean(t, "noise_removed_recalibrated") - personMean(t, "recalibrated")) /
    (personMean(ref16, "recalibrated") - personMean(t, "recalibrated"));
});
const recoveredMax = Math.max(...nm.mechanism.map((r) => r.share_recovered), ...nextDayRecovered);
// runs that fit day 1 worse do worse on day 2: training loss (last 10 epochs) against next-day accuracy,
// within each participant and draw, over the newly trained draw models
const nmRuns = fs.readdirSync(rel(`${NM}/runs`)).filter((f) => f.endsWith(".json")).map((f) => {
  const r = readJson(`${NM}/runs/${f}`);
  return { subject: r.subject, arm: r.arm, loss: mean(r.train_loss.slice(-10)), acc: nmTests[f].as_trained };
});
function withinR(rows) { // Pearson correlation after removing each participant-and-draw mean
  const cells = {};
  rows.forEach((r) => (cells[`${r.subject} ${r.arm}`] = cells[`${r.subject} ${r.arm}`] || []).push(r));
  const d = Object.values(cells).flatMap((c) => {
    const mx = mean(c.map((r) => r.loss)), my = mean(c.map((r) => r.acc));
    return c.map((r) => [r.loss - mx, r.acc - my]);
  });
  const sum = (f) => d.reduce((s, p) => s + f(p), 0);
  return sum(([x, y]) => x * y) / Math.sqrt(sum(([x]) => x * x) * sum(([, y]) => y * y));
}
const lossR = withinR(nmRuns);
const nmPeople = [...new Set(nmRuns.map((r) => r.subject))];
const lossNeg = nmPeople.filter((p) => withinR(nmRuns.filter((r) => r.subject === p)) < 0).length;
const nmModels = llModels("noise_mechanism_study");
if (nmRuns.length !== nmModels) throw new Error(`noise-mechanism study: ${nmRuns.length} runs, ${nmModels} trained`);

// what kind of extra input costs: copies, one shared noise waveform, half noise, independent noise (input_kind_study)
const IK = "results/supplementary/input_kind_study";
const ik = readJson(`${IK}/analysis.json`);
if (!["K1", "K2", "K3"].every((k) => ik.tests_lee2019[k].holds)) throw new Error("a registered K1-K3 test no longer holds: reword Result 3");
const ikRows = readCsv(`${IK}/conditions.csv`);
const ikCost = (ds, cond) => Number(find(ikRows, (r) => r.dataset === ds && r.condition === cond, `input kind ${ds} ${cond}`).next_day_cost);
const IK_KINDS = [["dup", "Copies"], ["shared", "One noise, repeated"], ["mix50", "Half noise"], ["noise", "Independent noise"]];
const IK_EXTRA = { bnci2a: 6, lee2019: 46 };
const predictable = (ds) => 1 - ik.predictability[ds].mean; // share of a dropped electrode the kept 16 predict
const realCost = (ds) => -ik.a_equivalent[ds].next_day_cost_at_a; // a mix that predictable, read off the mixes' curve
const realInfo = (ds) => realCost(ds) - (ds === "bnci2a" ? ovC("bnci2a", OVC.net).mean : H("H1b").estimate_pp);
const ikModels = llModels("input_kind_study");

// the dropped electrodes' cost and information measured directly (residual_surrogate_study): S1-S4 on Lee 2019
const rs = readJson("results/supplementary/residual_surrogate_study/analysis.json");
if (!["S1", "S2", "S3"].every((k) => rs.tests_lee2019[k].holds) || rs.tests_lee2019.S4.holds) throw new Error("the registered S1-S4 outcomes changed: reword Result 3");
const rsTerm = (ds, term) => find(rs.decomposition, (r) => r.dataset === ds && r.level === "final" && r.term === term, `surrogate ${ds} ${term}`).mean;
const rsModels = llModels("residual_surrogate_study");

// the same noise drawn afresh in every epoch (fresh_noise_study): F1-F3 on Lee 2019
const fn = readJson("results/supplementary/fresh_noise_study/analysis.json");
if (!fn.tests_lee2019.F1.holds || fn.tests_lee2019.F2.holds || !fn.tests_lee2019.F3.holds) throw new Error("the registered F1-F3 outcomes changed: reword Result 3 and the takeaways");
const fnCost = (label) => find(fn.contrasts, (r) => r.dataset === "lee2019" && r.added === 46 && r.level === "final" && r.a === label && r.b === "none", `fresh noise ${label}`).mean;
const fnRemoved = Math.round(100 * (1 - fnCost("fresh") / fnCost("fixed")));
const fnModels = llModels("fresh_noise_study");

// why alignment hurts on Lee 2019 (alignment_mechanism_study): exploratory readings M1-M3
const am = readJson("results/supplementary/alignment_mechanism_study/analysis.json").readings;
if (!am.M1.holds || !am.M3.holds || am.M2.variants["ea-floor"].share_of_harm_removed < 0.8) throw new Error("the alignment readings changed: reword What Did Not Work");
const amSpec = readCsv("results/supplementary/alignment_mechanism_study/spectra.csv").filter((r) => r.dataset === "lee2019" && r.montage === "full-62");
const amBelow = Math.round(mean(amSpec.map((r) => Number(r.below_1pct))));

// training four times longer (long_training_study): L1-L3 on Lee 2019
const lt = readJson("results/supplementary/long_training_study/analysis.json");
if (["L1", "L2", "L3"].some((k) => lt.tests_lee2019[k].holds)) throw new Error("a registered L1-L3 test now holds: reword What Did Not Work");

// channel dropout (chdrop_study): batch norms re-estimated, smaller p, a failed electrode
const cd = readJson("results/supplementary/chdrop_study/analysis.json");
if (cd.tests_lee2019.B1.holds || !cd.tests_lee2019.B3.holds) throw new Error("registered B1/B3 outcome changed: reword the channel-dropout card");
const cdFailRow = (p, version) => find(cd.failure, (r) => r.dataset === "bnci2a" && r.montage === "full" && r.p === p && r.version === version, `chdrop failure ${p} ${version}`);
const cdFail = { plain: cdFailRow(0, "as_trained").single_mean, drop: cdFailRow(cd.tests_lee2019.B3.p, "recalibrated").single_mean };

// a penalty no rescaling can lower (scalefree_penalty_study): N1-N3 on Lee 2019
const sf = readJson("results/supplementary/scalefree_penalty_study/analysis.json");
const sfFailed = ["N1", "N2", "N3"].filter((k) => sf.tests_lee2019[k].holds === false).length;
if (sfFailed !== 3) throw new Error("a registered N1-N3 test now holds: reword What Did Not Work and the takeaways");

// pretraining on 45 other people, then fine-tuning (pretrain_study): does the network then use all electrodes?
const pt = readJson("results/supplementary/pretrain_study/analysis.json");
if (!["T1", "T2", "T3"].every((k) => pt.tests_lee2019[k].holds)) throw new Error("a registered T1-T3 test no longer holds: reword the Levels Off slide");
const ptAllVs16 = pt.tests_lee2019.T3.mean;

// where the non-VNNI difference starts: integer tensors of the graph each CPU actually runs (int8_capture_study)
const icap = readJson("results/supplementary/int8_capture_study/analysis.json").summaries[0];
const icapFirst = Object.entries(icap.first_differing);
if (icapFirst.length !== 1 || icap.share_of_differing_predicted < 0.95) throw new Error("int8_capture_study: the first differing layer or the saturation match changed: reword the INT8 Across Processors slide");
const icapLayer = icapFirst[0][0].replace(" conv", " convolution");

// a sparsity penalty on each input's spatial weights (group lasso), its strength chosen on day 1 only
const ip = readJson("results/supplementary/input_penalty_study/analysis.json");
const ipFailed = ["G1", "G2"].filter((g) => ip.tests_lee2019[g].holds === false).length;
if (ipFailed !== 2 || ip.tests_lee2019.G3.holds !== null) throw new Error("the registered G1-G3 outcomes changed: reword What Did Not Work and the takeaways");
const ipRatios = ip.norms.filter((r) => r.noise_norm !== null).map((r) => r.noise_norm / r.electrode_norm);
if (Math.min(...ipRatios) < 0.5) throw new Error("the penalty pushed a noise input below half the electrodes' weight: reword \"shrank them all together\"");
const penaltyModels = llModels("input_penalty_study");
const whyModels = ovModels + edModels + icModels + nmModels + penaltyModels + ikModels;
// the final-sprint tests of that account: measured cost and information, redrawn noise, longer training, a
// scale-free penalty, pretraining on other people
const whyModelsAll = whyModels + rsModels + fnModels + llModels("long_training_study") + llModels("scalefree_penalty_study") + llModels("pretrain_study");

console.log([
  `2a: ${nPeople2a} people; full-22 ${acc("full-22").toFixed(2)}; max INT8 change ${int8MaxChange.toFixed(2)}`,
  `ten seeds: curve ${curve.map((v) => v.toFixed(2)).join(" ")}; 16-22 ${sgn(c16.mean_diff_pp, 2)} ${ci(c16.ci_low_pp, c16.ci_high_pp, 2)} se ${c16.seed_se_pp.toFixed(2)}; 8-22 ${sgn(c8.mean_diff_pp, 2)}`,
  `groups at 16: below ${sgn(group16.below_minus_full_pp)} above ${sgn(group16.above_minus_full_pp)}; fisher ${fisherDropped.toFixed(3)} vs ${fisherKept.toFixed(3)}; drops ${frontalDrops}/${allDrops}`,
  `redundancy: mean |rho| ${meanCorr.toFixed(2)}; same subsets ${same16}/9 at 16, ${same8}/9 at 8`,
  `regions: ${nRegionSets} sets, ${regionModels} models; vs relevance ${neg(Math.min(...regionVsRel))} to ${sgn(Math.max(...regionVsRel))}; top: ${["left_hand", "right_hand", "feet"].map((c) => `${c} ${topRegion(c)}`).join(", ")}`,
  `selection: riemann-8 ${riem8.acc_mean_pct.toFixed(2)} (${sgn(riem8.vs_relevance_diff_pp, 2)}, ${riem8.vs_relevance_n_better}/${riem8.vs_relevance_n_worse}); control ${sgn(riemCtrl.diff_pp, 2)} ${riemCtrl.n_better}/${riemCtrl.n_worse}; lowest ${lowestMethod}`,
  `Lee: ${nPeopleLee} people, ${leeElectrodes} electrodes; R1 ${sgn(R("R1").diff_pp, 2)} R2 ${sgn(R("R2").diff_pp, 2)} R3 ${sgn(R("R3").diff_pp, 2)}; H5 ${sgn(H("H5", "full-62").estimate_pp, 2)}; bits within ${bitsLee.toFixed(2)}`,
  `bits: INT16 ${sgn(ptq("16/16").vs_float_diff_pp, 2)} INT8 ${sgn(ptq("8/8").vs_float_diff_pp, 2)} 4/8 ${sgn(ptq("4/8").vs_float_diff_pp, 2)}; map ${sgn(bitMap.vs_float_diff_pp, 2)} saves ${byteSaving.toFixed(1)}%`,
  `speed: first layer ${firstLayerShare.toFixed(1)}% of MACs; spatial INT8/FP32 ${(t8[1] / t32[1]).toFixed(1)}x; swapped INT8 ${lat["int8-spatial-first"]} ms vs FP32 ${lat.fp32} (${(lat.fp32 / lat["int8-spatial-first"]).toFixed(1)}x); 22->8 saves ${macsSaved("original_macs").toFixed(1)}% exported, ${macsSaved("spatial-first_macs").toFixed(1)}% swapped`,
  `cpus: ${cpuTypes.length} types; ${changedModels}/${nAccel} changed; 7-bit on ${rrCpus.length}: ${rrCpus[0].rr.toFixed(2)}; slower INT8 on ${slowerCpus.length}/${timedCpus.length} timed CPUs`,
  `pruning: ${sgn(matched.difference_pp, 2)} ${ci(matched.ci_low_pp, matched.ci_high_pp, 2)}; saves ${pruneSaving.toFixed(2)}% MACs; swapped ${swappedPruned.toFixed(1)} vs ${swappedUnpruned.toFixed(1)} us; final ${acc(FINAL, "int8").toFixed(2)}`,
  `pruned layer: ${sharesOriginal[PRUNED].toFixed(2)}% of MACs as exported, ${sharesSwapped[PRUNED].toFixed(2)}% swapped; MCU m4 ${mcuPruneBefore.toFixed(2)} -> ${mcuPruneAfter.toFixed(2)} M; CPU kernel ${cpuPruneBefore.toFixed(1)} -> ${cpuPruneAfter.toFixed(1)} us`,
  `kernel: spatial ${spatialMacs} MACs (${spatialShare.toFixed(1)}%); FP32 ${k32[1].toFixed(1)} INT8 ${k8[1].toFixed(1)} kernel ${kK[1].toFixed(1)} us; ns/MAC ${nsPerMac(k8[1], spatialMacs).toFixed(3)} vs ${nsPerMac(k32[1], spatialMacs).toFixed(3)}; temporal INT8 ${(k32[0] / k8[0]).toFixed(2)}x faster`,
  `kernel whole: full-22 ${["fp32", "int8", "int8-kernel", "int8-spatial-first", "fp32-spatial-first"].map((v) => `${v} ${usHalfEven(kUs("full-22", v))}`).join(" ")}; kernel ${kernelVsFp32.toFixed(2)}x, swap ${swapVsFp32.toFixed(2)}x vs FP32; frozen ${["fp32", "int8", "int8-kernel"].map((v) => usHalfEven(kUs(FINAL, v))).join("/")}; swapped+kernel ${usHalfEven(kUs("full-22", "int8-spatial-first-kernel"))} (${swapKernelVsFp32.toFixed(2)}x); identical ${eqGrouped.logits_identical}/${eqGrouped.models} and ${eqSwapped.logits_identical}/${eqSwapped.models}, ${eqGrouped.predictions_equal_archived}/${eqGrouped.windows}`,
  `mcu: ratios orig ${CORES.map(([c]) => mRatio("full-22", "original", c).toFixed(2)).join("/")}, swapped ${CORES.map(([c]) => mRatio("full-22", "swapped", c).toFixed(2)).join("/")}; spatial ${spatialIpmM4.toFixed(1)} vs temporal ${temporalIpmM4.toFixed(1)} instr/MAC (M4); bit-exact ${mcu.int8_bit_exact_builds}/${mcu.int8_builds}; acc within ${mcuAccMax.toFixed(2)}; RAM m4 ${mMem("full-22", "original", "f32", "m4").ram_bytes} -> ${mMem("full-22", "swapped", "int8", "m4").ram_bytes}`,
  `longleaf: ${LONGLEAF_STUDIES.map((r) => `${r.name} ${r.models}/${r.hours.toFixed(1)}h`).join(", ")}; trainings ${longleafTrainings} + QAT ${qatModels}; ${longleafHours.toFixed(1)} h; ${longleafCpus.length} CPU types; seeds ${seedsBefore}->${seedsAfter}, ${seedModels} models on the Mac`,
  `direct tests: ${OV_SETS.map((d) => `${d.ds} ${d.full}->${d.reduced} info ${sgn(ovC(d.ds, OVC.info).mean, 2)} cost ${sgn(ovC(d.ds, OVC.cost).mean, 2)} net ${sgn(ovC(d.ds, OVC.net).mean, 2)} shuffled ${sgn(ovC(d.ds, OVC.shuffled).mean, 2)} dropout ${sgn(ovC(d.ds, OVC.dropout).mean, 2)}`).join("; ")}; fits ${trainFits.map((v) => v.toFixed(1)).join("/")}`,
  `levels: 2a fold ${["train", "within", "next_day"].map((k) => ovLevel("bnci2a", "fold")[k].toFixed(1)).join("/")}, lee fold ${["train", "within", "next_day"].map((k) => ovLevel("lee2019", "fold")[k].toFixed(1)).join("/")}; curves flat from ${PLATEAU} within ${plateauDev.toFixed(2)}; best counts ${bestCounts.join("/")}`,
  `input cost: slopes ${ic.slopes.map((r) => `${r.dataset} ${r.level} ${r.slope.toFixed(2)}`).join(", ")}; C2 ${sgn(ic.tests_lee2019.C2.mean, 2)}, shares ${c2Shares.join("/")}%`,
  `mechanism: noise share 2a +${SHARE_K} ${(100 * share6.noise_share_init).toFixed(1)} -> ${(100 * share6.noise_share_trained).toFixed(1)}%; draws ${drawCosts.map((v) => v.toFixed(2)).join(" ")} mean ${drawCost.toFixed(2)}, info ${sgn(info2a, 2)}; recovered max ${(100 * recoveredMax).toFixed(1)}% (next day ${nextDayRecovered.map((v) => (100 * v).toFixed(1)).join(" ")}); loss r ${lossR.toFixed(3)}, negative in ${lossNeg}/${nmPeople.length}`,
  `input kind: ${IK_KINDS.map(([c]) => `${c} ${sgn(ikCost("bnci2a", c), 2)}/${sgn(ikCost("lee2019", c), 2)}`).join(" ")}; predictable ${(100 * predictable("bnci2a")).toFixed(1)}/${(100 * predictable("lee2019")).toFixed(1)}%; real cost ${realCost("bnci2a").toFixed(2)}/${realCost("lee2019").toFixed(2)}, info ${realInfo("bnci2a").toFixed(2)}/${realInfo("lee2019").toFixed(2)}; models ${ikModels}`,
  `surrogate: cost ${sgn(rsTerm("bnci2a", "cost"), 2)}/${sgn(rsTerm("lee2019", "cost"), 2)}, information ${sgn(rsTerm("bnci2a", "information"), 2)}/${sgn(rsTerm("lee2019", "information"), 2)}, S4 rho ${rs.tests_lee2019.S4.rho.toFixed(3)}; fresh noise ${sgn(fnCost("fixed"), 2)} -> ${sgn(fnCost("fresh"), 2)} (${fnRemoved}% removed); models ${rsModels}+${fnModels}`,
  `penalty: G1 ${sgn(ip.tests_lee2019.G1.mean, 2)}, G2 ${sgn(ip.tests_lee2019.G2.mean, 2)} ${ci(...ip.tests_lee2019.G2.ci, 2)}, ${ipFailed} failed, G3 ${ip.tests_lee2019.G3.holds}; noise/electrode norms ${Math.min(...ipRatios).toFixed(2)}-${Math.max(...ipRatios).toFixed(2)}; models ${ovModels}+${edModels}+${icModels}+${nmModels}+${penaltyModels} = ${whyModels}`,
].join("\n"));

// ---------- speaker notes and timing from the talk script ----------
const SCRIPT = path.join(ROOT, "presentation", "final_script.md");
const WPM = 125;
const NOTES = (() => {
  const md = fs.readFileSync(SCRIPT, "utf8");
  const heads = [...md.matchAll(/^## (\d+) · (.+?)(?:（(\d+):(\d\d)）)?$/gm)];
  if (heads.some((m, i) => Number(m[1]) !== i + 1)) throw new Error(`${SCRIPT}: number the "## N" sections 1, 2, 3, ... in order`);
  const timed = heads.map((m) => m[3] !== undefined); // main slides carry a start time; backup slides come last, without
  const mainSlides = timed.includes(false) ? timed.indexOf(false) : timed.length;
  if (timed.slice(mainSlides).some(Boolean)) throw new Error(`${SCRIPT}: timed sections must come before the backup sections`);
  let clock = 0, words = 0, wordsShort = 0;
  const out = heads.map((m, i) => {
    const lines = md.slice(m.index + m[0].length, heads[i + 1] ? heads[i + 1].index : undefined).split(/\r?\n/);
    const say = lines.filter((l) => l.startsWith("> ")).map((l) => l.slice(2).trim()).join(" ");
    if (!say) throw new Error(`slide ${m[1]} has no "> " script lines in ${SCRIPT}`);
    const count = (t) => t.split(/\s+/).filter((w) => /[A-Za-z0-9]/.test(w)).length;
    if (i < mainSlides) {
      const written = Number(m[3]) * 60 + Number(m[4]);
      if (Math.abs(written - clock) > 10) console.warn(`slide ${m[1]}: script says ${m[3]}:${m[4]}, words give ${Math.floor(clock / 60)}:${String(Math.round(clock % 60)).padStart(2, "0")}`);
      clock += count(say) * 60 / WPM;
      words += count(say);
      wordsShort += count(say.replace(/\[[^\]]*\]/g, " "));
    }
    const asked = lines.find((l) => l.startsWith("If asked: "));
    return { title: m[2], note: asked ? `${say}\n\n${asked.trim()}` : say };
  });
  const mmss = (sec) => `${Math.floor(sec / 60)}:${String(Math.round(sec % 60)).padStart(2, "0")}`;
  console.log(`script: ${mainSlides} main slides, ${words} words, ${mmss(words * 60 / WPM)} at ${WPM} wpm; without brackets ${wordsShort} words, ${mmss(wordsShort * 60 / WPM)}; ${heads.length - mainSlides} backup slides`);
  return out;
})();
// The slide code below runs in script order. Each slide takes the next script section and checks its title,
// so a slide is added by writing its "## N" section in the script and its block here, in the same place.
let slideNo = 0;
function nextNote(expect) {
  const note = NOTES[slideNo++];
  if (!note || !note.title.includes(expect)) throw new Error(`slide ${slideNo}: script section is "${note && note.title}", code expects "${expect}"`);
  return note;
}
const resultNo = (needle) => { // "Result N" numbers, read from the script's titles
  const m = find(NOTES, (n) => n.title.includes(needle), `script title with "${needle}"`).title.match(/^Result (\d+):/);
  if (!m) throw new Error(`"${needle}" is not a "Result N:" title`);
  return m[1];
};

// ---------- deck frame ----------
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.author = "Peiye Li";
pres.company = "UNC-Chapel Hill";
pres.title = "EEG Channel Selection and Model Compression: final presentation";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
const S = pres.ShapeType;
const W = 13.333, M = 0.6, CW = W - 2 * M;

pres.defineSlideMaster({
  title: "UNC_CONTENT",
  background: { color: UNC.white },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.42, w: CW, h: 0.8, fontFace: FONT, fontSize: 30,
      bold: true, color: UNC.navy, align: "left", valign: "top", margin: 0 }, text: "" } },
    { text: { text: "Peiye Li · COMP 690 · Final presentation · 8 December 2026",
      options: { x: M, y: 6.98, w: 8, h: 0.3, fontFace: FONT, fontSize: 10, color: UNC.basin, margin: 0, valign: "middle" } } },
  ],
  slideNumber: { x: W - M - 0.6, y: 6.98, w: 0.6, h: 0.3, fontFace: FONT, fontSize: 10, color: UNC.basin, align: "right" },
});
pres.defineSlideMaster({ title: "UNC_COVER", background: { color: UNC.carolina }, objects: [] });
pres.defineSlideMaster({ title: "UNC_CLOSING", background: { color: UNC.navy }, objects: [] });

// ---------- helpers ----------
function text(slide, t, o) {
  slide.addText(t, Object.assign({ isTextBox: true, margin: 0, fontFace: FONT, fontSize: 16, color: UNC.navy, valign: "top" }, o));
}
function eyebrow(slide, t, x, y, w, color = UNC.bolin) {
  text(slide, t.toUpperCase(), { x, y, w, h: 0.32, fontSize: 13, bold: true, color, charSpacing: 1.5 });
}
function tag(slide, t) { eyebrow(slide, t, M, 1.2, CW); }
function whyLine(slide, t, y, o = {}) { // the short reason under a result
  text(slide, [{ text: "Why: ", options: { bold: true, color: UNC.bolin } }, { text: t }],
    Object.assign({ x: M, y, w: CW, h: 0.62, fontSize: 15 }, o));
}
function card(slide, x, y, w, h, color) {
  slide.addShape(S.roundRect, { x, y, w, h, rectRadius: 0.12, fill: { color }, line: { color, width: 0.75 } });
}
function bigStat(slide, value, label, x, y, w, o = {}) {
  text(slide, value, { x, y, w, h: o.valueH || 0.7, fontSize: o.size || 34, bold: true, color: o.color || UNC.carolina });
  text(slide, label, { x, y: y + (o.valueH || 0.7), w, h: o.labelH || 0.6, fontSize: o.labelSize || 15,
    color: o.labelColor || UNC.navy });
}
function numberDot(slide, n, x, y, d = 0.6, fill = UNC.carolina) {
  slide.addShape(S.ellipse, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill, width: 0.75 } });
  text(slide, String(n), { x, y, w: d, h: d, fontSize: d * 34, bold: true, align: "center", valign: "middle" });
}
function pill(slide, t, x, y, w, held) {
  const color = held ? UNC.carolina : UNC.cornerstone;
  slide.addShape(S.roundRect, { x, y, w, h: 0.36, rectRadius: 0.18, fill: { color }, line: { color, width: 0.75 } });
  text(slide, t, { x, y, w, h: 0.36, fontSize: 12, bold: true, align: "center", valign: "middle", color: UNC.navy });
}
const CHART_TEXT = {
  catAxisLabelFontFace: FONT, valAxisLabelFontFace: FONT, legendFontFace: FONT,
  dataLabelFontFace: FONT, titleFontFace: FONT, catAxisTitleFontFace: FONT, valAxisTitleFontFace: FONT,
  catAxisLabelColor: UNC.navy, valAxisLabelColor: UNC.basin, legendColor: UNC.navy, titleColor: UNC.navy,
  catAxisTitleColor: UNC.basin, valAxisTitleColor: UNC.basin, dataLabelColor: UNC.navy,
  catAxisLabelFontSize: 13, valAxisLabelFontSize: 12, legendFontSize: 13, titleFontSize: 14,
  catAxisTitleFontSize: 12, valAxisTitleFontSize: 12, dataLabelFontSize: 13,
};
const chartOpts = (o) => Object.assign({}, CHART_TEXT, o); // a fresh object per chart
function table(slide, rows, o) { // o.aligns: one alignment per column (default: first left, the rest right)
  const { aligns, ...opts } = o;
  const cell = (v, i, header) => ({ text: String(v), options: { fontFace: FONT, fontSize: o.fontSize || 12,
    color: UNC.navy, bold: header, fill: { color: header ? UNC.cloud : UNC.white }, valign: "middle",
    align: aligns ? aligns[i] : (i === 0 ? "left" : "right"), margin: [2, 6, 2, 6] } });
  slide.addTable(rows.map((r, j) => r.map((v, i) => cell(v, i, j === 0))), Object.assign({ fontFace: FONT,
    border: { type: "solid", pt: 0.5, color: UNC.cornerstone } }, opts));
}
function mix(a, b, t) { // blend two hex colors
  const p = (h, i) => parseInt(h.slice(i, i + 2), 16);
  return [0, 2, 4].map((i) => Math.round(p(a, i) + (p(b, i) - p(a, i)) * t).toString(16).padStart(2, "0")).join("").toUpperCase();
}
const ramp = (t) => { // 0 = Cloud, 0.5 = Carolina, 1 = Navy
  const u = Math.max(0, Math.min(1, t));
  return u < 0.5 ? mix(UNC.cloud, UNC.carolina, u / 0.5) : mix(UNC.carolina, UNC.navy, (u - 0.5) / 0.5);
};

// 22 BCI IV 2a electrodes on a grid: [name, column, row]
const MONTAGE = [["Fz", 0, 0], ["FC3", -2, 1], ["FC1", -1, 1], ["FCz", 0, 1], ["FC2", 1, 1], ["FC4", 2, 1],
  ["C5", -3, 2], ["C3", -2, 2], ["C1", -1, 2], ["Cz", 0, 2], ["C2", 1, 2], ["C4", 2, 2], ["C6", 3, 2],
  ["CP3", -2, 3], ["CP1", -1, 3], ["CPz", 0, 3], ["CP2", 1, 3], ["CP4", 2, 3],
  ["P1", -1, 4], ["Pz", 0, 4], ["P2", 1, 4], ["POz", 0, 5]];
function montage(slide, o) {
  const rx = 3.9 * o.step, ry = 3.35 * o.step, nose = 0.36 * o.step;
  slide.addShape(S.triangle, { x: o.cx - nose * 0.75, y: o.cy - ry - nose + 0.02, w: nose * 1.5, h: nose,
    fill: { color: o.outline }, line: { color: o.outline, width: 1 } });
  slide.addShape(S.ellipse, { x: o.cx - rx, y: o.cy - ry, w: 2 * rx, h: 2 * ry, fill: { type: "none" },
    line: { color: o.outline, width: 2.25 } });
  for (const [name, gx, gy] of MONTAGE) {
    const fill = o.fillFor ? o.fillFor(name) : o.dotFill;
    const x = o.cx + gx * o.step - o.dot / 2, y = o.cy + (gy - 2.5) * o.step - o.dot / 2;
    slide.addShape(S.ellipse, { x, y, w: o.dot, h: o.dot, fill: { color: fill },
      line: { color: o.dotLine || UNC.navy, width: 0.75 }, objectName: `electrode ${name}` });
    if (o.labels) {
      text(slide, name, { x: x - 0.1, y, w: o.dot + 0.2, h: o.dot, fontSize: o.labelSize || 8, bold: true, align: "center",
        valign: "middle", color: o.textFor ? o.textFor(name) : UNC.navy });
    }
  }
}

// ---------- 1 cover ----------
pres.addSection({ title: "Problem and approach" });
let s = pres.addSlide({ masterName: "UNC_COVER", sectionTitle: "Problem and approach" });
eyebrow(s, "COMP 690-193 · Final project presentation · 8 December 2026", 0.8, 0.75, 9, UNC.navy);
text(s, [{ text: "EEG Channel Selection", options: { breakLine: true } },
  { text: "and Model Compression", options: { breakLine: true } }, { text: "for Motor-Imagery BCI" }],
{ x: 0.8, y: 1.85, w: 7.8, h: 2.1, fontSize: 38, bold: true });
text(s, "Which electrodes, how many bits, and does it hold in new people?", { x: 0.8, y: 4.2, w: 7.4, h: 1.0, fontSize: 20 });
text(s, `Two datasets · ${nPeople2a + nPeopleLee} people · two recording days each`,
  { x: 0.8, y: 5.25, w: 7.4, h: 0.4, fontSize: 16, bold: true });
text(s, "Peiye Li", { x: 0.8, y: 6.15, w: 5, h: 0.5, fontSize: 20, bold: true });
montage(s, { cx: 10.75, cy: 3.85, step: 0.5, dot: 0.27, outline: UNC.navy, dotFill: UNC.white });
s.addNotes(nextNote("Cover").note);

// ---------- 2 problem ----------
function contentSlide(expect, section) {
  const note = nextNote(expect);
  const slide = pres.addSlide({ masterName: "UNC_CONTENT", sectionTitle: section });
  slide.addText(note.title, { placeholder: "title" });
  slide.addNotes(note.note);
  return slide;
}
s = contentSlide("Why Fewer Electrodes", "Problem and approach");
const halfW = (CW - 0.4) / 2;
[["1", "Hardware cost", "Each electrode needs a contact, an amplifier channel, power and setup time"],
  ["2", "Compute cost", "In the standard design, the first layer runs once per electrode"]].forEach(([n, head, body], i) => {
  const x = M + i * (halfW + 0.4), y = 1.75;
  card(s, x, y, halfW, 3.15, UNC.cloud);
  numberDot(s, n, x + 0.45, y + 0.45, 0.75);
  text(s, head, { x: x + 0.45, y: y + 1.5, w: halfW - 0.9, h: 0.55, fontSize: 24, bold: true });
  text(s, body, { x: x + 0.45, y: y + 2.1, w: halfW - 0.9, h: 0.75, fontSize: 16, color: UNC.basin });
});
text(s, [{ text: "Fewer electrodes save both. ", options: { bold: true } },
  { text: "Pruning the network saves only compute." }], { x: M, y: 5.35, w: CW, h: 0.6, fontSize: 22 });

// ---------- 3 questions ----------
s = contentSlide("Research Questions", "Problem and approach");
[["Electrodes", "Which can a decoder lose on a new day? Does avoiding redundancy help?"],
  ["Bits", "How many does it need? Does 8-bit arithmetic make it faster?"],
  ["People", `Do the answers hold in ${nPeopleLee} new people?`]].forEach(([head, body], i) => {
  const y = 1.75 + i * 1.5;
  numberDot(s, i + 1, M, y + 0.05, 0.7);
  text(s, head, { x: M + 1.0, y, w: 6.3, h: 0.5, fontSize: 22, bold: true });
  text(s, body, { x: M + 1.0, y: y + 0.52, w: 6.3, h: 0.75, fontSize: 16, color: UNC.basin });
});
const qx = 8.05, qw = W - M - qx;
card(s, qx, 1.6, qw, 5.05, UNC.navy);
eyebrow(s, "Midterm review questions", qx + 0.4, 1.85, qw - 0.8, UNC.carolina);
[["Why can 16 electrodes beat all 22?", `Results ${resultNo("Why 16")}–${resultNo("Levels Off")}`],
  ["How are the electrodes chosen? Could brain regions guide it?", `Method · Results ${resultNo("Brain Regions")}–${resultNo("Replication")}`],
  ["Why was INT8 slower on the M4? Would INT16 help?", `Results ${resultNo("Bit Width")}–${resultNo("Microcontrollers")}`],
  ["Why didn't pruning make the model faster?", `Result ${resultNo("Pruning")}`]].forEach(([q, where], i) => {
  const y = 2.28 + i * 1.08;
  text(s, q, { x: qx + 0.4, y, w: qw - 0.8, h: 0.62, fontSize: 15, bold: true, color: UNC.white });
  text(s, where, { x: qx + 0.4, y: y + 0.63, w: qw - 0.8, h: 0.3, fontSize: 13, bold: true, color: UNC.carolina });
});

// ---------- 4 related work ----------
s = contentSlide("Related Work", "Problem and approach");
eyebrow(s, "Already shown", M, 1.6, 6);
[["EEGNet", "2018", "Compact CNN for EEG; my decoder"],
  ["Q-EEGNet", "2020", "8-bit EEGNet on a microcontroller"],
  ["MI-BMInet", "2024", "Weight-based electrode selection, then 8-bit"],
  ["mRMR", "2005", "Relevance minus redundancy; basis of my rule"],
  ["Riemannian selection", "2011", "Chooses electrodes as a group by covariance distance"]].forEach(([name, year, what], i) => {
  const y = 2.05 + i * 0.86;
  s.addShape(S.ellipse, { x: M, y: y + 0.1, w: 0.22, h: 0.22, fill: { color: UNC.carolina },
    line: { color: UNC.carolina, width: 0.75 } });
  text(s, [{ text: name, options: { bold: true } }, { text: "   " + year, options: { color: UNC.basin } }],
    { x: M + 0.45, y, w: 6.0, h: 0.4, fontSize: 18 });
  text(s, what, { x: M + 0.45, y: y + 0.4, w: 6.0, h: 0.38, fontSize: 14, color: UNC.basin });
});
const px = 7.3, pw = W - M - px;
card(s, px, 1.6, pw, 4.75, UNC.carolina);
eyebrow(s, "This project adds", px + 0.45, 1.95, pw - 0.9, UNC.navy);
text(s, [{ text: "One factor at a time, same training budget", options: { bullet: true, breakLine: true } },
  { text: "Test days locked until each plan froze", options: { bullet: true, breakLine: true } },
  { text: `Main tests repeated in ${nPeopleLee} new people`, options: { bullet: true, breakLine: true } },
  { text: `Same INT8 files on ${cpuTypes.length} processor types`, options: { bullet: true, breakLine: true } },
  { text: "The missing INT8 kernels, written", options: { bullet: true, breakLine: true } },
  { text: `INT8 on ${CORES.length} emulated Cortex-M cores`, options: { bullet: true } }],
{ x: px + 0.45, y: 2.3, w: pw - 0.9, h: 3.55, fontSize: 15, bold: true, paraSpaceAfter: 5 });
text(s, "Not a new algorithm: a controlled comparison", { x: px + 0.45, y: 5.9, w: pw - 0.9, h: 0.35, fontSize: 13 });
text(s, "Lawhern et al. 2018 · Schneider et al. 2020 · Wang et al., arXiv 2203.14592 · Peng et al. 2005 · Barachant & Bonnet 2011 · Lee et al. 2019",
  { x: M, y: 6.55, w: CW, h: 0.3, fontSize: 10, color: UNC.basin });

// ---------- 5 datasets ----------
s = contentSlide("Datasets and Evaluation", "Problem and approach");
tag(s, `Two public datasets · ${nPeople2a + nPeopleLee} people · each recorded on two days`);
[["BCI Competition IV 2a · primary",[[nPeople2a, "people"], [22, "electrodes"], [4, "movements"], [2, "days each"]]],
  ["Lee et al. 2019", [[nPeopleLee, "people"], [leeElectrodes, "electrodes"], [2, "movements"], [2, "days each"]]]]
  .forEach(([name, stats], i) => {
    const x = M + i * (halfW + 0.4), y = 1.6, sw = (halfW - 0.7) / 4;
    card(s, x, y, halfW, 1.85, UNC.cloud);
    eyebrow(s, name, x + 0.35, y + 0.25, halfW - 0.7);
    stats.forEach(([v, l], j) => bigStat(s, String(v), l, x + 0.35 + j * sw, y + 0.65, sw - 0.1,
      { size: 30, valueH: 0.6, labelH: 0.4, labelSize: 14 }));
  });
const fW = 3.55, fGap = (CW - 3 * fW) / 2;
[["Day 1", "Train, validate and make every choice"],
  ["Freeze the plan", [{ text: "3 Nov for 2a", options: { breakLine: true } }, { text: "21 Nov for Lee 2019" }]],
  ["Day 2", "Score once"]].forEach(([head, body], i) => {
  const x = M + i * (fW + fGap), y = 3.8, h = 1.55, dark = i === 1;
  card(s, x, y, fW, h, dark ? UNC.navy : UNC.cloud);
  text(s, head, { x: x + 0.35, y: y + 0.28, w: fW - 0.7, h: 0.5, fontSize: 20, bold: true, color: dark ? UNC.white : UNC.navy });
  text(s, body, { x: x + 0.35, y: y + 0.82, w: fW - 0.7, h: 0.6, fontSize: 15, color: dark ? UNC.white : UNC.navy });
  if (i < 2) {
    s.addShape(S.rightArrow, { x: x + fW + (fGap - 0.42) / 2, y: y + h / 2 - 0.2, w: 0.42, h: 0.4,
      fill: { color: UNC.carolina }, line: { color: UNC.carolina, width: 0.75 } });
  }
});
text(s, "Studies added after the 2a test results are exploratory unless registered in advance on Lee 2019 · chance level: 25% (2a), 50% (Lee 2019)",
  { x: M, y: 5.75, w: CW, h: 0.35, fontSize: 14, color: UNC.basin });

// ---------- 6 systems view ----------
s = contentSlide("Systems View", "Problem and approach");
const full = benchRow("full-22");
[[int(full.params), `parameters, ${(full.parameter_bytes / 1000).toFixed(1)} KB in FP32`],
  [`${(full.macs / 1e6).toFixed(1)} M`, "multiply-accumulates per decision"],
  [`${firstLayerShare.toFixed(0)}%`, "of them in the first layer, which runs on every electrode"]].forEach(([n, l], i) => {
  bigStat(s, n, l, M, 1.65 + i * 1.45, 5.3);
});
s.addChart(pres.ChartType.bar, [{ name: "Multiply-accumulates (millions)",
  labels: ["22 electrodes", "16 electrodes", "8 electrodes"],
  values: [22, 16, 8].map((k) => Number(costOf(k).original_macs) / 1e6) }],
chartOpts({ x: 6.4, y: 1.5, w: W - M - 6.4, h: 4.45, barDir: "col", chartColors: [UNC.carolina], barGapWidthPct: 70,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelFontSize: 16, dataLabelFontBold: true,
  valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 7, valGridLine: { style: "none" }, catGridLine: { style: "none" },
  catAxisLabelFontSize: 14, showLegend: false, showTitle: true, title: "Multiply-accumulates per decision (millions)" }));
text(s, `As exported, compute follows the electrode count: 22 → 8 saves ${macsSaved("original_macs").toFixed(0)}%.`,
  { x: M, y: 6.2, w: CW, h: 0.45, fontSize: 20, bold: true });

// ---------- 7 selection method ----------
s = contentSlide("Electrode Selection", "Problem and approach");
const leftW = 7.2;
card(s, M, 1.6, leftW, 0.95, UNC.navy);
text(s, [{ text: "score = ", options: { color: UNC.white } }, { text: "relevance", options: { color: UNC.carolina, bold: true } },
  { text: " − 0.5 × ", options: { color: UNC.white } }, { text: "redundancy", options: { color: UNC.carolina, bold: true } }],
{ x: M + 0.35, y: 1.6, w: leftW - 0.7, h: 0.95, fontSize: 20, valign: "middle" });
text(s, [{ text: "Relevance  ", options: { bold: true } },
  { text: "how well an electrode's mu and beta power separate the movements on day 1", options: { breakLine: true } },
  { text: "Redundancy  ", options: { bold: true } }, { text: "mean absolute correlation with electrodes already chosen" }],
{ x: M, y: 2.75, w: leftW, h: 1.3, fontSize: 16, paraSpaceAfter: 8 });
eyebrow(s, "Compared at the same count", M, 4.15, leftW);
const chipW = (leftW - 0.25) / 2;
["Relevance only (frozen rule)", "Weight norm", "Random × 3", "All electrodes"].forEach((c, i) => {
  const x = M + (i % 2) * (chipW + 0.25), y = 4.55 + Math.floor(i / 2) * 0.62;
  card(s, x, y, chipW, 0.48, i === 0 ? UNC.carolina : UNC.cloud);
  text(s, c, { x, y, w: chipW, h: 0.48, fontSize: 15, bold: true, align: "center", valign: "middle" });
});
text(s, "After the midterm: Riemannian, wrapper, CSP, learned selector and brain regions",
  { x: M, y: 5.95, w: leftW, h: 0.35, fontSize: 14, color: UNC.basin });
montage(s, { cx: 10.4, cy: 3.55, step: 0.52, dot: 0.44, outline: UNC.basin, labels: true, labelSize: 8,
  fillFor: (e) => ramp(kept16[e] / nPeople2a), textFor: (e) => (kept16[e] / nPeople2a > 0.6 ? UNC.white : UNC.navy) });
text(s, `How many of the ${nPeople2a} people kept each electrode at 16 (darker = more)`,
  { x: 8.1, y: 5.55, w: 4.6, h: 0.55, fontSize: 13, color: UNC.basin, align: "center" });

// ---------- process after the midterm ----------
pres.addSection({ title: "How the work was done" });
s = contentSlide("Process After the Midterm", "How the work was done");
tag(s, "After the 9 Nov midterm review: each step is a question, a tool and a result");
const STEPS = [
  ["12 Nov", "Is the 16-vs-22 gap real?", `${seedsAfter} seeds instead of ${seedsBefore}: ${int(seedModels)} new models on the M4 Mac`,
    `${sgn(c16.mean_diff_pp, 2)} pts, seed error ${c16.seed_se_pp.toFixed(2)}; a same-machine retrain later gave ${sgn(ovC("bnci2a", OVC.net).mean, 2)}`],
  ["12 Nov", "Why is INT8 slower?", `ONNX Runtime ${V.onnxruntime} profiler and source code`,
    "One small layer, the grouped spatial convolution, runs on a generic path"],
  ["20–21 Nov", "How should electrodes be chosen? Does it hold in new people?",
    `Longleaf: all ${nRegionSets} region sets, ${selMethods.length} selection methods, a registered test in ${nPeopleLee} new people`,
    "Choosing electrodes as a group beat one at a time, also in the new people"],
  ["20–21 Nov", "Is INT8 the same on every CPU?", `Longleaf: the same INT8 files on ${longleafCpus.length} server CPU types`,
    `Without VNNI, ${changedModels} of ${nAccel} models change; 7-bit weights fix it`],
  ["21–22 Nov", "Can INT8 be faster, also on microcontrollers?", "Two ONNX Runtime kernels written in C; CMSIS-NN in QEMU",
    `INT8 ${kernelVsFp32.toFixed(1)}× faster than FP32 on the M4 CPU; ${mRatio("full-22", "original", "m0").toFixed(0)}× fewer instructions on the Cortex-M0`],
  ["21–22 Nov", "Why are 16 electrodes enough?", `Longleaf: ${int(whyModelsAll)} models: noise inputs, copies, less data, penalties, longer training, pretraining`,
    "An input costs what the others cannot predict: the network memorizes trials with it, even with a penalty"],
];
const PX = [M, M + 1.4, M + 4.6, M + 8.05], PW = [1.3, 2.85, 3.05, CW - 8.05];
["When", "Question", "Tool", "Result"].forEach((h, j) => eyebrow(s, h, PX[j], 1.62, PW[j]));
const stepH = 0.78;
STEPS.forEach((row, i) => {
  const y = 1.98 + i * stepH;
  s.addShape(S.line, { x: M, y: y - 0.08, w: CW, h: 0, line: { color: UNC.cornerstone, width: 0.75 } });
  row.forEach((t, j) => text(s, t, { x: PX[j], y, w: PW[j], h: stepH - 0.12, fontSize: j === 0 ? 15 : 13,
    bold: j === 0 || j === 3, color: j === 2 ? UNC.basin : UNC.navy }));
  [1, 2].forEach((j) => s.addShape(S.rightArrow, { x: PX[j + 1] - 0.32, y: y + 0.1, w: 0.2, h: 0.2,
    fill: { color: UNC.carolina }, line: { color: UNC.carolina, width: 0.5 } }));
});

// ---------- tools and compute ----------
s = contentSlide("Tools and Compute", "How the work was done");
const TOOLS = [
  ["Software", [
    [`PyTorch ${V.torch} · Braindecode ${V.braindecode} · MOABB ${V.moabb}`, "Trained EEGNet on both datasets"],
    [`ONNX Runtime ${V.onnxruntime}`, "INT8 on CPUs; its profiler found the slow layer, and our two C kernels fixed it"],
    [`CMSIS-NN v8.0.0 · QEMU ${qemuVersion} · Arm GCC ${gccVersion}`,
      `INT8 and FP32 on emulated Cortex-${CORES.map((c) => cpuOf(c[0])).join(", ")}`]]],
  ["Compute", [
    ["Apple M4 Mac mini", `${int(seedModels)} trainings for ${seedsAfter} seeds instead of ${seedsBefore}; all timing, the kernels and the emulation`],
    ["UNC Longleaf cluster", `${int(longleafTrainings)} trainings, ${int(qatModels)} fine-tunes, ${int(longleafHours)} CPU-hours: ${nRegionSets} region sets, the ${nPeopleLee}-person replication, why 16 electrodes suffice, INT8 on ${longleafCpus.length} server CPUs`]]],
];
let ty = 1.45;
TOOLS.forEach(([group, rows]) => {
  eyebrow(s, group, M, ty, CW);
  ty += 0.4;
  rows.forEach(([tool, use]) => {
    const h = use.length > 110 ? 1.1 : 0.76;
    card(s, M, ty, CW, h - 0.1, UNC.cloud);
    text(s, tool, { x: M + 0.3, y: ty, w: 5.6, h: h - 0.1, fontSize: 15, bold: true, valign: "middle" });
    text(s, use, { x: M + 6.1, y: ty, w: CW - 6.4, h: h - 0.1, fontSize: 15, valign: "middle" });
    ty += h;
  });
  ty += 0.16;
});

// ---------- 8 accuracy vs count ----------
pres.addSection({ title: "Results: electrodes" });
s = contentSlide("Accuracy vs. Electrode Count", "Results: electrodes");
s.addChart(pres.ChartType.line, [{ name: "Relevance order, ten seeds", labels: COUNTS.map(String), values: curve }],
  chartOpts({ x: M, y: 1.5, w: 8.3, h: 4.5, chartColors: [UNC.carolina], lineSize: 3, lineDataSymbol: "circle",
    lineDataSymbolSize: 10, valAxisMinVal: 55, valAxisMaxVal: 70, valAxisMajorUnit: 5, valAxisLabelFormatCode: "0",
    showValAxisTitle: true, valAxisTitle: "Day-2 accuracy (%)", showCatAxisTitle: true, catAxisTitle: "Electrodes kept",
    valGridLine: { color: "DDE3E8", size: 1 }, catGridLine: { style: "none" }, showLegend: false,
    showValue: true, dataLabelPosition: "t", dataLabelFormatCode: "0.0", dataLabelFontSize: 12 }));
[[`${curve[0].toFixed(1)}%`, "all 22 electrodes; chance is 25%"],
  [`${sgn(c16.mean_diff_pp)} pts`, `at 16 electrodes, interval ${ci(c16.ci_low_pp, c16.ci_high_pp)}`],
  [`${sgn(c8.mean_diff_pp)} pts`, `at 8 electrodes, interval ${ci(c8.ci_low_pp, c8.ci_high_pp)}`]].forEach(([n, l], i) => {
  bigStat(s, n, l, 9.35, 1.6 + i * 1.4, W - M - 9.35, { size: 28, valueH: 0.55, labelH: 0.7, labelSize: 14 });
});
whyLine(s, "the rule drops the least informative electrodes first; down to about 16, what they carry is offset by the cost of having them as inputs (Results 3–4).", 6.15);

// ---------- 9 redundancy ----------
s = contentSlide("Redundancy-Aware Selection", "Results: electrodes");
tag(s, "The proposal's main hypothesis · frozen analysis, three seeds");
const vs = ["vs random (mean of 3)", "vs weight norm", "vs relevance only"]; // bars draw bottom to top
const round1 = (v) => Math.round(v * 10) / 10 || 0; // labels show one decimal; avoids "-0.0"
const diffs = (k) => [acc(`proposed-${k}`) - randomMean(k), acc(`proposed-${k}`) - acc(`weightnorm-${k}`),
  acc(`proposed-${k}`) - acc(`relevance-${k}`)].map(round1);
s.addChart(pres.ChartType.bar, [{ name: "16 electrodes", labels: vs, values: diffs(16) },
  { name: "8 electrodes", labels: vs, values: diffs(8) }],
chartOpts({ x: M, y: 1.6, w: 7.4, h: 5.05, barDir: "bar", barGrouping: "clustered", barGapWidthPct: 60,
  chartColors: [UNC.carolina, UNC.navy], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.0;-0.0;0.0",
  valAxisMinVal: -2, valAxisMaxVal: 4, valAxisMajorUnit: 1, valAxisLabelFormatCode: "0", showValAxisTitle: true,
  valAxisTitle: "Proposed minus control (accuracy points)", valGridLine: { color: "DDE3E8", size: 1 },
  catGridLine: { style: "none" }, catAxisLabelFontSize: 14, catAxisLabelPos: "low", showLegend: true, legendPos: "b" }));
const rx0 = 8.35, rw = W - M - rx0;
card(s, rx0, 1.7, rw, 4.8, UNC.cloud);
eyebrow(s, "Why so little", rx0 + 0.35, 2.0, rw - 0.7);
bigStat(s, meanCorr.toFixed(2), "mean correlation between electrodes' band power", rx0 + 0.35, 2.4, rw - 0.7);
bigStat(s, `${Math.min(same16, same8)}–${Math.max(same16, same8)} of ${nPeople2a}`,
  "people got the same electrodes from both rules", rx0 + 0.35, 3.85, rw - 0.7);
text(s, "Band power rises and falls together across the scalp, so every candidate looks equally redundant.",
  { x: rx0 + 0.35, y: 5.35, w: rw - 0.7, h: 0.95, fontSize: 13, color: UNC.basin });

// ---------- why 16 electrodes are enough ----------
s = contentSlide("Why 16 Electrodes Are Enough", "Results: electrodes");
tag(s, `Answers a midterm question · ${int(ovModels + icModels + nmModels + ikModels + rsModels + fnModels)} models on Longleaf · Lee 2019 tests registered`);
// what each kind of extra input costs next to the 16 kept electrodes alone, both datasets
s.addChart(pres.ChartType.bar, [["bnci2a", "BCI IV 2a"], ["lee2019", "Lee 2019"]].map(([ds, name]) => ({
  name: `${name}, ${IK_EXTRA[ds]} extra inputs`, labels: IK_KINDS.map((k) => k[1]), values: IK_KINDS.map(([c]) => ikCost(ds, c)) })),
chartOpts({ x: M, y: 1.5, w: 7.45, h: 4.2, barDir: "col", barGrouping: "clustered", barGapWidthPct: 55,
  chartColors: [UNC.navy, UNC.carolina], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.0;-0.0;0.0",
  dataLabelFontSize: 11, valAxisMinVal: -16, valAxisMaxVal: 2, valAxisMajorUnit: 4, valAxisLabelFormatCode: "0",
  showValAxisTitle: true, valAxisTitle: "Day-2 change vs 16 alone (points)", valGridLine: { color: "DDE3E8", size: 1 },
  catGridLine: { style: "none" }, catAxisLabelFontSize: 12, catAxisLabelPos: "low", showLegend: true, legendPos: "b",
  legendFontSize: 11, showTitle: true, titleFontSize: 13, title: "What costs is what the other inputs cannot predict" }));
const ox = 8.15, ow = W - M - ox;
[[`${Math.round(100 * predictable("bnci2a"))}%`, `of the ${SHARE_K} dropped electrodes' signal is predictable from the 16 kept ones (Lee 2019: ${Math.round(100 * predictable("lee2019"))}%)`],
  [`${neg(rsTerm("lee2019", "cost"), 1)} / ${sgn(rsTerm("lee2019", "information"), 1)}`, `points: what the dropped electrodes' unpredictable part costs / adds on Lee 2019, measured; they cancel (2a: ${neg(rsTerm("bnci2a", "cost"), 1)} / ${sgn(rsTerm("bnci2a", "information"), 1)})`],
  [`${neg(fnCost("fixed"), 1)} → ${neg(fnCost("fresh"), 1)}`, `points: 46 noise inputs, recorded vs redrawn every epoch so they cannot be memorized (Lee 2019)`]]
  .forEach(([n, l], i) => bigStat(s, n, l, ox, 1.55 + i * 1.37, ow, { size: 28, valueH: 0.52, labelH: 0.8, labelSize: 13 }));
whyLine(s, `the network memorizes its training trials with what the other inputs cannot predict, so that part costs accuracy. The dropped electrodes are mostly predictable, and what they add cancels what they cost.`, 5.75, { h: 0.8 });
text(s, `Registered on Lee 2019, all held: copies cost less than noise (K1), half noise lies between (K2), one repeated waveform costs less than independent noise (K3). Pure noise costs about ${Math.round(-icSlope("bnci2a", "final"))} pt per input on 2a, as much within the training day, and more data does not shrink it.`,
  { x: M, y: 6.55, w: CW, h: 0.45, fontSize: 11, color: UNC.basin });

// ---------- accuracy levels off ----------
const ED_COLORS = [UNC.navy, UNC.carolina, UNC.grey]; // training amounts, as on the electrode-count slides
s = contentSlide("Levels Off", "Results: electrodes");
tag(s, `${int(edModels)} more models on Longleaf, part of day 1 · Lee 2019 tests registered`);
[["bnci2a", "BCI IV 2a", 44, 68], ["lee2019", "Lee 2019", 58, 70]].forEach(([ds, name, lo, hi], i) => {
  const lvs = ED_LEVELS[ds], xs = edCurve(ds, lvs[0]).map((c) => c.electrodes);
  s.addChart(pres.ChartType.line, lvs.map((lv) => ({ name: `${edData(ds, lv)} of day 1`, labels: xs.map(String),
    values: xs.map((k) => find(edCurve(ds, lv), (c) => c.electrodes === k, `${ds} ${lv} ${k}`).next_day) })),
  chartOpts({ x: M + i * 6.18, y: 1.55, w: 5.95, h: 4.0, chartColors: ED_COLORS.slice(0, lvs.length), lineSize: 3,
    lineDataSymbol: "circle", lineDataSymbolSize: 7, valAxisMinVal: lo, valAxisMaxVal: hi, valAxisMajorUnit: 4,
    valAxisLabelFormatCode: "0", showValAxisTitle: true, valAxisTitle: "Day-2 accuracy (%)", showCatAxisTitle: true,
    catAxisTitle: "Electrodes kept", valGridLine: { color: "DDE3E8", size: 1 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 11, showTitle: true, title: name }));
});
text(s, `From ${PLATEAU} electrodes on, every curve stays within ±${plateauDev.toFixed(1)} pt of its level; halving the training data does not move that point (registered D1, D2: not held).`,
  { x: M, y: 5.62, w: CW, h: 0.55, fontSize: 14, bold: true });
whyLine(s, `with one day of a person's data, each electrode past about ${PLATEAU} adds about as much as it costs (Result ${resultNo("Why 16")}). Pretrained on 45 other people, all 62 beat a shared 16 by ${sgn(ptAllVs16, 1)} pts (registered).`, 6.2);

// ---------- 11 regions ----------
s = contentSlide("Brain Regions", "Results: electrodes");
tag(s, `Answers a midterm question · all ${nRegionSets} region sets, day-1 cross-validation`);
const CLASSES = [["left_hand", "Left hand"], ["right_hand", "Right hand"], ["feet", "Feet"]];
const vmax = Math.max(...CLASSES.flatMap(([c]) => Object.keys(REGIONS).map((r) => shap[r].recall[c].mean)));
const hx = [2.35, 6.67, 10.98];
CLASSES.forEach(([cls, name], i) => {
  text(s, name, { x: hx[i] - 1.9, y: 1.65, w: 3.8, h: 0.45, fontSize: 20, bold: true, align: "center" });
  montage(s, { cx: hx[i], cy: 3.5, step: 0.36, dot: 0.3, outline: UNC.basin, dotLine: UNC.basin,
    fillFor: (e) => ramp(Math.max(0, shap[regionOf[e]].recall[cls].mean) / vmax) });
  const top = topRegion(cls);
  text(s, `Strongest: ${top} ${sgn(shap[top].recall[cls].mean)} pts`,
    { x: hx[i] - 2.1, y: 4.85, w: 4.2, h: 0.35, fontSize: 13, color: UNC.basin, align: "center" });
});
const legendX = M, legendY = 5.45;
text(s, "Recall each region adds (points):", { x: legendX, y: legendY, w: 3.4, h: 0.3, fontSize: 12, color: UNC.basin });
[0, 0.25, 0.5, 0.75, 1].forEach((t, j) => {
  s.addShape(S.rect, { x: legendX + 3.45 + j * 0.62, y: legendY + 0.02, w: 0.6, h: 0.26, fill: { color: ramp(t) },
    line: { color: UNC.basin, width: 0.5 } });
  text(s, (t * vmax).toFixed(0), { x: legendX + 3.45 + j * 0.62, y: legendY + 0.3, w: 0.6, h: 0.25, fontSize: 11,
    color: UNC.basin, align: "center" });
});
text(s, `${int(regionModels)} models trained`, { x: 8.3, y: 5.45, w: W - M - 8.3, h: 0.3, fontSize: 14, bold: true,
  align: "right", color: UNC.bolin });
text(s, [{ text: "Each hand comes from the opposite hemisphere, the feet from the midline. ", options: { bold: true } },
  { text: `As a selection rule, whole regions did not beat single electrodes (${neg(Math.min(...regionVsRel))} to ${sgn(Math.max(...regionVsRel))} pts).` }],
{ x: M, y: 6.05, w: CW, h: 0.65, fontSize: 16 });

// ---------- 12 smarter selection ----------
s = contentSlide("Smarter Selection", "Results: electrodes");
tag(s, "Exploratory · chosen after the 2a test results");
const SEL_SERIES = [["relevance", "Frozen rule (one at a time)", UNC.navy], ["riemann", "Riemannian (as a group)", UNC.carolina],
  ["wrapper", "Wrapper (fast decoder)", UNC.basin], ["learned", "Learned selector", UNC.grey]];
s.addChart(pres.ChartType.line, SEL_SERIES.map(([m, name]) => ({ name, labels: SEL_COUNTS.map(String),
  values: SEL_COUNTS.map((k) => selAcc(m, k).acc_mean_pct) })),
chartOpts({ x: M, y: 1.55, w: 8.3, h: 4.5, chartColors: SEL_SERIES.map((r) => r[2]), lineSize: 3, lineDataSymbol: "circle",
  lineDataSymbolSize: 8, valAxisMinVal: 50, valAxisMaxVal: 70, valAxisMajorUnit: 5, valAxisLabelFormatCode: "0",
  showValAxisTitle: true, valAxisTitle: "Day-2 accuracy (%)", showCatAxisTitle: true, catAxisTitle: "Electrodes kept",
  valGridLine: { color: "DDE3E8", size: 1 }, catGridLine: { style: "none" }, showLegend: true, legendPos: "b", legendFontSize: 12 }));
[[`${sgn(riem8.vs_relevance_diff_pp)} pts`, `Riemannian vs frozen rule at 8 electrodes; ${riem8.vs_relevance_n_better} of ${nPeople2a} people better`],
  [`${sgn(riemCtrl.diff_pp)} pts`, `vs references retrained on the same hardware; ${riemCtrl.n_better} of ${nPeople2a} better`],
  [`${riem8.acc_mean_pct.toFixed(1)}%`, `with 8 Riemannian electrodes; all 22 give ${selFull.toFixed(1)}%`]].forEach(([n, l], i) => {
  bigStat(s, n, l, 9.35, 1.6 + i * 1.45, W - M - 9.35, { size: 30, valueH: 0.55, labelH: 0.8, labelSize: 14 });
});
whyLine(s, "a set chosen as a whole can keep electrodes that help only in combination; one-at-a-time scores cannot see that.", 6.15);

// ---------- 13 replication ----------
s = contentSlide("Replication", "Results: electrodes");
tag(s, `Registered before scoring · plans frozen 21 Nov · ${nPeopleLee} people, chance 50%`);
const h1 = H("H1"), h1b = H("H1b"), h2 = H("H2");
const repRows = [
  [true, "Fewer electrodes not worse on average", `45/${leeElectrodes}: ${sgn(h1.estimate_pp)} · 16/${leeElectrodes}: ${sgn(h1b.estimate_pp)} pts`],
  [true, "Relevance beats random at 16", `${sgn(h2.estimate_pp)} pts ${ci(h2.low_pp, h2.high_pp)}`],
  [true, "8 bits lose nothing; the 2a bit map carries over", `every change within ±${bitsLee.toFixed(1)} pts`],
  [true, "Riemannian beats the frozen rule", ["R1", "R2", "R3"].map((id) => `${R(id).electrodes}: ${sgn(R(id).diff_pp)}`).join(" · ")],
  [false, "Worst decoders gain most", `correlation ${neg(H("H3").rho, 2)}`],
  [false, "Aligning the two days helps", `${sgn(H("H5", "full-62").estimate_pp)} pts; ${H("H5", "full-62").worse} of ${nPeopleLee} worse`],
];
const checks = [H("H1").met && H("H1b").met, H("H2").met,
  ["full-62", "relevance-45", "relevance-16"].every((c) => H("H4", c).met) && ["full-62", "relevance-16"].every((c) => H("H4b", c).met),
  ["R1", "R2", "R3"].every((id) => R(id).met), H("H3").met || H("H3b").met, H("H5", "full-62").met];
repRows.forEach(([held, claim, nums], i) => {
  if (held !== checks[i]) throw new Error(`replication row ${i + 1}: slide says ${held}, results say ${checks[i]}`);
  const y = 1.7 + i * 0.8;
  pill(s, held ? "Held" : "Not held", M, y + 0.05, 1.3, held);
  text(s, claim, { x: M + 1.55, y, w: 4.45, h: 0.65, fontSize: 17, bold: true, valign: "middle" });
  text(s, nums, { x: M + 6.1, y, w: 3.05, h: 0.65, fontSize: 14, color: UNC.basin, valign: "middle" });
});
const bx = 9.9, bw = W - M - bx;
card(s, bx, 1.7, bw, 4.6, UNC.cloud);
bigStat(s, `${repR.mean_accuracy_pct["riemann-8"].toFixed(1)}%`, "with 8 Riemannian electrodes", bx + 0.3, 2.0, bw - 0.6,
  { size: 32, labelH: 0.7 });
bigStat(s, `${fullLeeAcc.toFixed(1)}%`, `with all ${leeElectrodes} electrodes`, bx + 0.3, 3.5, bw - 0.6,
  { size: 32, color: UNC.navy, labelH: 0.7 });
text(s, "Descriptive: the plan did not test this pair.", { x: bx + 0.3, y: 5.15, w: bw - 0.6, h: 0.8, fontSize: 13, color: UNC.basin });

// ---------- 14 bit width ----------
pres.addSection({ title: "Results: bits and speed" });
s = contentSlide("Bit Width", "Results: bits and speed");
tag(s, `Answers a midterm question · simulated integer rounding of all ${nModels} models`);
const BIT_BARS = [["Float", ptq("float/float").acc_mean], ["16-bit", ptq("16/16").acc_mean], ["8-bit", ptq("8/8").acc_mean],
  ["6-bit weights", ptq("6/8").acc_mean], ["Per-layer bits", bitMap.accuracy], ["4-bit weights", ptq("4/8").acc_mean],
  ["4-bit all", ptq("4/4").acc_mean]];
s.addChart(pres.ChartType.bar, [{ name: "Day-2 accuracy (%)", labels: BIT_BARS.map((b) => b[0]), values: BIT_BARS.map((b) => b[1]) }],
  chartOpts({ x: M, y: 1.55, w: 7.9, h: 4.5, barDir: "col", chartColors: [UNC.carolina], barGapWidthPct: 45,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelFontSize: 13, valAxisHidden: true,
    valAxisMinVal: 0, valAxisMaxVal: 72, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    catAxisLabelFontSize: 12, showLegend: false, showTitle: true, title: "Day-2 accuracy (%), mean of all models" }));
[[`${sgn(ptq("8/8").vs_float_diff_pp)} pts`, `INT8 vs float; ${ptq("8/8").vs_float_n_losing_over_2pp} of ${nPeople2a} people lose over 2 pts`],
  ["No gain", `from INT16 (${sgn(ptq("16/16").vs_float_diff_pp, 2)} pts)`],
  [`${MINUS}${byteSaving.toFixed(0)}%`, `weight bytes with per-layer bits, ${sgn(bitMap.vs_float_diff_pp)} pts`]].forEach(([n, l], i) => {
  bigStat(s, n, l, 8.9, 1.65 + i * 1.45, W - M - 8.9, { size: 30, valueH: 0.6, labelH: 0.7, labelSize: 14 });
});
whyLine(s, "below 8 bits, the raw input and the first layer's outputs break first: 16 levels cannot hold their amplitude range. Below 8 bits is simulated, not run on low-bit hardware.", 6.15);

// ---------- 15 why INT8 was slower ----------
s = contentSlide("Why INT8 Was Slower", "Results: bits and speed");
tag(s, `Answers a midterm question · Apple M4, ONNX Runtime ${V.onnxruntime}, one thread, 22 electrodes`);
const layers = ["Temporal conv", "Spatial conv (grouped)", "Everything else"];
s.addChart(pres.ChartType.bar, [{ name: "FP32", labels: layers, values: k32 }, { name: "INT8 (ONNX Runtime)", labels: layers, values: k8 }],
  chartOpts({ x: M, y: 1.55, w: 6.9, h: 4.45, barDir: "col", barGrouping: "clustered", barGapWidthPct: 55,
    chartColors: [UNC.basin, UNC.carolina], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0",
    valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 280, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    catAxisLabelFontSize: 13, showLegend: true, legendPos: "t", showTitle: true,
    title: "Time per decision by layer (µs), ONNX Runtime profiler" }));
const wx = 7.85, ww = W - M - wx;
[[`${spatialShare.toFixed(0)}% of the work`, `the grouped spatial layer: ${int(spatialMacs)} multiply-accumulates`],
  [`${usHalfEven(k8[1])} vs ${usHalfEven(k32[1])} µs`, `that layer in INT8 vs FP32: ${nsPerMac(k8[1], spatialMacs).toFixed(1)} vs ${nsPerMac(k32[1], spatialMacs).toFixed(3)} ns per multiply-accumulate`],
  [`${(k32[0] / k8[0]).toFixed(1)}× faster`, `the temporal layer in INT8, so the M4's 8-bit units work (${m4.integer_extensions})`]]
  .forEach(([n, l], i) => bigStat(s, n, l, wx, 1.6 + i * 1.45, ww, { size: 28, valueH: 0.55, labelH: 0.75, labelSize: 14 }));
whyLine(s, "ONNX Runtime has fast INT8 kernels only for ordinary and depthwise convolutions. EEGNet's 8 groups × 2 filters take a generic path: copy each group's input, then multiply by a matrix 2 columns wide.", 6.12);

// ---------- the missing INT8 kernel ----------
s = contentSlide("The Missing INT8 Kernel", "Results: bits and speed");
tag(s, "Exploratory · ONNX Runtime custom operator, C with ARM NEON · Apple M4");
const KMODELS = [["full-22", "22 electrodes"], ["relevance-16", "16 electrodes"], ["relevance-8", "8 electrodes"], [FINAL, "Frozen final"]];
s.addChart(pres.ChartType.bar, [["fp32", "FP32"], ["int8", "INT8 as exported"], ["int8-kernel", "INT8 + new kernel"]].map(([v, name]) =>
  ({ name, labels: KMODELS.map((m) => m[1]), values: KMODELS.map(([c]) => usHalfEven(kUs(c, v))) })),
chartOpts({ x: M, y: 1.55, w: 7.1, h: 4.45, barDir: "col", barGrouping: "clustered", barGapWidthPct: 45,
  chartColors: [UNC.basin, UNC.cornerstone, UNC.carolina], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0",
  dataLabelFontSize: 12, valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 400, valGridLine: { style: "none" },
  catGridLine: { style: "none" }, catAxisLabelFontSize: 13, showLegend: true, legendPos: "t", legendFontSize: 12, showTitle: true,
  title: `Whole model, µs per decision (one thread, ${okProtocol.timing.rounds} × ${okProtocol.timing.runs} runs)` }));
const kx = 8.0, kw = W - M - kx;
[[`${usHalfEven(k8[1])} → ${usHalfEven(kK[1])} µs`, `the spatial layer, now faster than FP32 (${usHalfEven(k32[1])} µs)`],
  [`${kernelVsFp32.toFixed(1)}× faster`, `than FP32 for the whole 22-electrode model: ${usHalfEven(kUs("full-22", "int8-kernel"))} vs ${usHalfEven(kUs("full-22", "fp32"))} µs`],
  [`${eqGrouped.logits_identical} of ${eqGrouped.models}`, "models give bit-identical outputs, so accuracy cannot change"]]
  .forEach(([n, l], i) => bigStat(s, n, l, kx, 1.6 + i * 1.45, kw, { size: 28, valueH: 0.55, labelH: 0.75, labelSize: 14 }));
whyLine(s, "the kernel reads each input once and computes all 16 filter outputs in NEON registers, with ONNX Runtime's own rounding: no per-group copy, no 2-column matrix multiply.", 6.12);

// ---------- 16 layer swap ----------
s = contentSlide("Swapping the First Two Layers", "Results: bits and speed");
tag(s, "Exact rewrite, no retraining · Apple M4, 22 electrodes");
[["As exported", ["Temporal filter on every electrode", "Spatial mix"]],
  ["Swapped", ["Spatial mix", "Temporal filter on 16 mixtures"]]].forEach(([name, boxes], i) => {
  const y = 1.6 + i * 1.0;
  text(s, name, { x: M, y: y + 0.1, w: 1.55, h: 0.6, fontSize: 15, bold: true, color: UNC.bolin, valign: "middle" });
  boxes.forEach((b, j) => {
    const bwid = j === (i === 0 ? 0 : 1) ? 2.35 : 1.35, x = M + 1.6 + (j === 0 ? 0 : (i === 0 ? 2.35 : 1.35) + 0.45);
    card(s, x, y + 0.05, bwid, 0.7, j === (i === 0 ? 0 : 1) ? UNC.carolina : UNC.cloud);
    text(s, b, { x: x + 0.1, y: y + 0.05, w: bwid - 0.2, h: 0.7, fontSize: 13, bold: true, align: "center", valign: "middle" });
    if (j === 0) {
      s.addShape(S.rightArrow, { x: x + bwid + 0.08, y: y + 0.25, w: 0.3, h: 0.3, fill: { color: UNC.basin },
        line: { color: UNC.basin, width: 0.5 } });
    }
  });
});
[[`${swapVsFp32.toFixed(1)}×`, `faster than FP32 as exported (${usHalfEven(kUs("full-22", "int8-spatial-first"))} vs ${usHalfEven(kUs("full-22", "fp32"))} µs); ${swapKernelVsFp32.toFixed(1)}× with a second kernel`],
  [`${macsSaved("spatial-first_macs").toFixed(0)}%`, `of compute saved by 22 → 8 electrodes after the swap (${macsSaved("original_macs").toFixed(0)}% before)`]]
  .forEach(([n, l], i) => bigStat(s, n, l, M, 3.6 + i * 1.2, 5.6, { size: 30, valueH: 0.55, labelH: 0.6, labelSize: 14 }));
const SWAP = [["INT8 as exported", "int8"], ["FP32 as exported", "fp32"], ["INT8 + new kernel", "int8-kernel"],
  ["FP32, layers swapped", "fp32-spatial-first"], ["INT8, layers swapped", "int8-spatial-first"],
  ["INT8, swapped + new kernel", "int8-spatial-first-kernel"]].reverse();
s.addChart(pres.ChartType.bar, [{ name: "Time per decision (µs)", labels: SWAP.map((r) => r[0]),
  values: SWAP.map((r) => usHalfEven(kUs("full-22", r[1]))) }],
chartOpts({ x: 6.55, y: 1.55, w: W - M - 6.55, h: 4.45, barDir: "bar", chartColors: [UNC.carolina], barGapWidthPct: 45,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", dataLabelFontSize: 14, valAxisHidden: true,
  valAxisMinVal: 0, valAxisMaxVal: 430, valGridLine: { style: "none" }, catGridLine: { style: "none" },
  catAxisLabelFontSize: 13, showLegend: false, showTitle: true, title: "Time per decision (µs), one timing run" }));
whyLine(s, `both layers are linear, so swapping them keeps every prediction of all ${nAccel} models; the costly temporal filter then runs on 16 mixtures instead of every electrode.`, 6.12);

// ---------- 17 INT8 across processors ----------
s = contentSlide("INT8 Across Processors", "Results: bits and speed");
tag(s, `The same INT8 files of all ${nAccel} models, scored on each processor`);
const CPUS = [...cpuTypes].sort((a, b) => b.change - a.change || a.cpu.localeCompare(b.cpu)).reverse();
s.addChart(pres.ChartType.bar, [{ name: "INT8 accuracy change (points)", labels: CPUS.map((c) => `${c.cpu} · ${c.ext}`),
  values: CPUS.map((c) => c.change) }],
chartOpts({ x: M, y: 1.55, w: 7.7, h: 4.6, barDir: "bar", chartColors: [UNC.navy], barGapWidthPct: 45, showValue: true,
  dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.0;-0.0;0.0", dataLabelFontSize: 13, valAxisMinVal: -5,
  valAxisMaxVal: 1, valAxisMajorUnit: 1, valAxisLabelFormatCode: "0", showValAxisTitle: true,
  valAxisTitle: "INT8 accuracy change vs the M4 (points)", valGridLine: { color: "DDE3E8", size: 1 },
  catGridLine: { style: "none" }, catAxisLabelFontSize: 11, catAxisLabelPos: "low", showLegend: false }));
[[`${changedModels} of ${nAccel}`, "models changed predictions on processors without VNNI"],
  [`${sgn(Math.min(...cpuTypes.map((c) => c.change)))} pts`, "INT8 accuracy on those processors"],
  ["7-bit weights", `identical on all ${rrCpus.length} processors tested, ${sgn(rrCpus[0].rr, 2)} pts vs FP32`]].forEach(([n, l], i) => {
  bigStat(s, n, l, 8.75, 1.65 + i * 1.45, W - M - 8.75, { size: 28, valueH: 0.55, labelH: 0.7, labelSize: 14 });
});
whyLine(s, `without VNNI, x86 CPUs sum pairs of 8-bit products in 16 bits. Measured layer by layer: the first difference is in the ${icapLayer}, and ${Math.round(100 * icap.share_of_differing_predicted)}% of the outputs that differ are predicted 16-bit overflows.`, 6.2);

// ---------- INT8 on microcontrollers ----------
s = contentSlide("INT8 on Microcontrollers", "Results: bits and speed");
tag(s, "Exploratory · CMSIS-NN in QEMU · 22 electrodes, participant 1, per 2-s window");
const MCU_SERIES = [["original", "f32", "FP32 as exported", UNC.basin], ["original", "int8", "INT8 as exported", UNC.carolina],
  ["swapped", "f32", "FP32, layers swapped", UNC.cornerstone], ["swapped", "int8", "INT8, layers swapped", UNC.navy]];
const MCU_FIR = ["INT8, swapped + our FIR kernel", UNC.bolin];
s.addChart(pres.ChartType.bar, [...MCU_SERIES.map(([o, p, name]) => ({ name, labels: CORES.map((c) => c[1]),
  values: CORES.map(([c]) => mIns("full-22", o, p, c)) })),
{ name: MCU_FIR[0], labels: CORES.map((c) => c[1]), values: CORES.map(([c]) => Number(mkRow("swapped", c).fir_total) / 1e6) }],
chartOpts({ x: M, y: 1.55, w: 6.5, h: 4.4, barDir: "col", barGrouping: "clustered", barGapWidthPct: 40,
  chartColors: [...MCU_SERIES.map((r) => r[3]), MCU_FIR[1]], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "#,##0.0",
  dataLabelFontSize: 10, valAxisLogScaleBase: 10, valAxisMinVal: 1, valAxisMaxVal: 10000, valAxisLabelFormatCode: "#,##0",
  showValAxisTitle: true, valAxisTitle: "Million instructions (log scale)", valGridLine: { color: "DDE3E8", size: 1 },
  catGridLine: { style: "none" }, catAxisLabelFontSize: 13, showLegend: true, legendPos: "t", legendFontSize: 11 }));
const mx = 7.35, mw = W - M - mx;
table(s, [["FP32 ÷ INT8", ...CORES.map((c) => c[1].replace("Cortex-", ""))],
  ["As exported", ...CORES.map(([c]) => `${mRatio("full-22", "original", c).toFixed(1)}×`)],
  ["Layers swapped", ...CORES.map(([c]) => `${mRatio("full-22", "swapped", c).toFixed(1)}×`)]],
{ x: mx, y: 1.6, w: mw, colW: [1.88, 1.15, 1.15, 1.203], fontSize: 13, rowH: 0.36 });
[[`${mcu.int8_bit_exact_builds} + ${mkChecks.bit_exact}`, `INT8 builds bit-exact (library alone; with our kernel); day-2 accuracy within ${mcuAccMax.toFixed(2)} pts of FP32`],
  [`${(Number(mkRow("swapped", "m4").library_total) / 1e6).toFixed(1)} → ${(Number(mkRow("swapped", "m4").fir_total) / 1e6).toFixed(1)} M`,
    "M4 instructions per window, INT8 swapped, before and after our kernel"]]
  .forEach(([n, l], i) => bigStat(s, n, l, mx, 3.0 + i * 1.4, mw, { size: 28, valueH: 0.55, labelH: 0.7, labelSize: 14 }));
whyLine(s, `CMSIS-NN runs the temporal filters as a general depthwise layer (${Number(mkRow("swapped", "m4").library_per_mac).toFixed(1)} instructions per multiply-accumulate on the M4). Our FIR kernel takes ${Number(mkRow("swapped", "m4").fir_per_mac).toFixed(1)}: ${Math.min(...mkRatios).toFixed(1)}–${Math.max(...mkRatios).toFixed(1)}× fewer instructions per window, outputs unchanged.`, 6.0);
text(s, "QEMU counts executed instructions; it models no timing, so these are not cycles, time or energy.",
  { x: M, y: 6.6, w: CW, h: 0.3, fontSize: 11, color: UNC.basin });

// ---------- 18 pruning ----------
s = contentSlide("Pruning", "Results: bits and speed");
tag(s, "Answers a midterm question · 16 electrodes · equal-training control added post-test");
const SHARE_LAYERS = [["Temporal conv", UNC.navy], ["Spatial conv", UNC.bolin], ["Depthwise conv", UNC.grey],
  [PRUNED, UNC.carolina], ["Classifier", UNC.cornerstone]];
s.addChart(pres.ChartType.bar, SHARE_LAYERS.map(([name]) => ({ name, labels: ["As exported", "Layers swapped"],
  values: [sharesOriginal[name], sharesSwapped[name]] })),
chartOpts({ x: M, y: 1.55, w: 6.3, h: 4.45, barDir: "bar", barGrouping: "stacked", barGapWidthPct: 50,
  chartColors: SHARE_LAYERS.map((r) => r[1]), showValue: false, valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 25,
  valAxisLabelFormatCode: "0\"%\"", valGridLine: { color: "DDE3E8", size: 1 }, catGridLine: { style: "none" },
  catAxisLabelFontSize: 13, showLegend: true, legendPos: "b", legendFontSize: 11, showTitle: true,
  title: "Share of multiply-accumulates by layer" }));
const px2 = 7.3, pw2 = W - M - px2;
[[`${mcuPruneBefore.toFixed(2)} → ${mcuPruneAfter.toFixed(2)} M`, "instructions on the emulated Cortex-M4, INT8 with the layers swapped"],
  [`${usHalfEven(cpuPruneBefore)} → ${usHalfEven(cpuPruneAfter)} µs`, "on the Apple M4 CPU, INT8 with the new kernel"],
  [`${sgn(matched.difference_pp, 2)} pts`, `pruned vs unpruned at equal training, INT8, 95% CI ${ci(matched.ci_low_pp, matched.ci_high_pp, 2)}`],
  [`${acc(FINAL, "int8").toFixed(1)}%`, `frozen final model, the same mean as all 22; ${primaryRow(FINAL, "int8").n_losing_more_than_2pts} of ${nPeople2a} people lost over 2 pts`]]
  .forEach(([n, l], i) => bigStat(s, n, l, px2, 1.55 + i * 1.12, pw2, { size: 24, valueH: 0.48, labelH: 0.6, labelSize: 13 }));
whyLine(s, `pruning halves the last 1×1 convolution, which holds ${sharesOriginal[PRUNED].toFixed(2)}% of the multiply-accumulates as exported and ${sharesSwapped[PRUNED].toFixed(1)}% after the swap, so it cannot save more.`, 6.15);

// ---------- 19 demo ----------
pres.addSection({ title: "Demo and conclusions" });
s = contentSlide("Demo", "Demo and conclusions");
const VIDEO = rel("results/demo/replay_participant3.mp4"), STILL = rel("results/demo/replay_participant3_still.png");
const vh = 4.9, vw = vh * 1320 / 704;
s.addMedia({ type: "video", path: VIDEO, cover: "data:image/png;base64," + fs.readFileSync(STILL).toString("base64"),
  x: (W - vw) / 2, y: 1.35, w: vw, h: vh });
text(s, "Participant 3's recorded day-2 EEG at 250 samples per second · 16 electrodes, INT8 · a replay of recorded data, not online closed-loop control",
  { x: M, y: 6.35, w: CW, h: 0.5, fontSize: 13, color: UNC.basin, align: "center" });

// ---------- 20 what did not work ----------
s = contentSlide("What Did Not Work", "Demo and conclusions");
const p16 = cmp10("proposed-16", "relevance-16");
const NOT = [
  ["Redundancy penalty", `${sgn(p16.mean_diff_pp)} pts over relevance alone at 16 electrodes, ten seeds`],
  ["Learned selector", `lowest of ${selMethods.length} selection methods, ${avgOverCounts("learned").toFixed(1)}% averaged over ${SEL_COUNTS.length} counts`],
  ["Whole brain regions", `${neg(Math.min(...regionVsRel))} to ${sgn(Math.max(...regionVsRel))} pts against single electrodes`],
  ["Aligning the two days", `${sgn(H("H5", "full-62").estimate_pp)} pts on Lee 2019: whitening blows up ${amBelow} of 62 near-empty directions; not amplifying them removes the harm`],
  ["INT16 and CoreML", `INT16 ${(lat.int16 / lat.fp32).toFixed(2)}× the FP32 time; CoreML never beat the swapped CPU graph`],
  ["Pruning", `${sgn(matched.difference_pp)} pts at equal training`],
  ["Channel dropout", `${sgn(ovC("bnci2a", OVC.dropout).mean)} pts on 2a (p = ${ovProtocol.dropout_p}), not a batch-norm mismatch; but a failed electrode then costs ${cdFail.drop.toFixed(1)} pts, not ${cdFail.plain.toFixed(1)}`],
  ["Less data, longer training", `registered, not held: with less data inputs cost no more; 4× the epochs changed the noise cost by ${sgn(lt.tests_lee2019.L1.mean)} pts`],
  ["Sparsity penalties", `group lasso and a scale-free version: all ${ipFailed + sfFailed} registered tests failed; the noise helps it memorize the training trials`],
];
if (lowestMethod !== "learned") throw new Error(`lowest selection method is now ${lowestMethod}`);
const COLS = 3, gGap = 0.22, gW = (CW - (COLS - 1) * gGap) / COLS, gH = 1.45;
NOT.forEach(([head, body], i) => { // heading and text in one box, so a wrapped heading pushes the text down
  const x = M + (i % COLS) * (gW + gGap), y = 1.5 + Math.floor(i / COLS) * (gH + 0.16);
  card(s, x, y, gW, gH, UNC.cloud);
  text(s, [{ text: head, options: { bold: true, fontSize: 16, color: UNC.navy, breakLine: true, paraSpaceAfter: 3 } },
    { text: body, options: { fontSize: 13, color: UNC.basin } }], { x: x + 0.22, y: y + 0.13, w: gW - 0.44, h: gH - 0.22 });
});
text(s, "Reported with the same intervals and tests as the positive results.", { x: M, y: 6.38, w: CW, h: 0.4, fontSize: 16, color: UNC.basin });

// ---------- takeaways ----------
s = pres.addSlide({ masterName: "UNC_CLOSING", sectionTitle: "Demo and conclusions" });
eyebrow(s, "Takeaways", 0.8, 0.8, 6, UNC.carolina);
[`With one day of data, choose ${PLATEAU}–${bestCounts[bestCounts.length - 1]} electrodes before the network, as a group: EEGNet uses useless inputs to memorize trials, even with sparsity penalties. Pretraining on many people changes that`,
  `8 bits are enough. INT8 was slower only because a kernel was missing: with it, INT8 is ${kernelVsFp32.toFixed(1)}× faster than FP32, and ${swapKernelVsFp32.toFixed(1)}× with the layer swap and a second kernel`,
  `Compression pays only where the compute is: INT8 helps most without a floating-point unit, and pruning a layer with ${sharesOriginal[PRUNED].toFixed(0)}% of the work saves about that much`]
  .forEach((t, i) => {
    const y = 1.3 + i * 1.6;
    text(s, String(i + 1), { x: 0.8, y, w: 0.8, h: 0.9, fontSize: 44, bold: true, color: UNC.carolina });
    text(s, t, { x: 1.75, y: y + 0.08, w: 10.8, h: 1.25, fontSize: 22, color: UNC.white });
  });
text(s, "Thank you · questions welcome", { x: 0.8, y: 6.15, w: 8, h: 0.5, fontSize: 20, bold: true, color: UNC.carolina });
s.addNotes(nextNote("Takeaways").note);

if (slideNo !== NOTES.length) throw new Error(`built ${slideNo} slides, but the script has ${NOTES.length} sections`);
pres.writeFile({ fileName: OUT }).then(async (file) => {
  // pptxgenjs also writes folder entries into the ZIP; some cloud services refuse such files.
  const JSZip = require("jszip");
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  for (const [name, entry] of Object.entries(zip.files)) if (entry.dir) delete zip.files[name];
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  console.log(`wrote ${file}`);
});
