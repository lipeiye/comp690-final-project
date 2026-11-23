#!/usr/bin/env node
/*
 * Build the 14-slide midterm-review deck as a PowerPoint file in UNC colors and type.
 *
 *   npm install --prefix <dir> pptxgenjs@4.0.1
 *   NODE_PATH=<dir>/node_modules node presentation/pptx/build_midterm_pptx.js [output.pptx] [font]
 *
 * Defaults: presentation/midterm.pptx and the font "Open Sans", a UNC brand typeface for digital
 * and print use (identity.unc.edu/brand/typography). Passing a wider font such as Verdana gives a
 * stricter check of text fit. Colors are the UNC palette (its-commons.unc.edu/brand/colors).
 * Accuracy, MACs, layer timings and the matched-budget difference are read from results/; the
 * remaining numbers are quoted from the full deck and README, with their source noted.
 * Speaker notes come from the talk script presentation/midterm_script.md: for each "## N" section,
 * its "> " lines (what to say) and its "If asked:" line. Edit the script, then rebuild.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..", "..");
const OUT = path.resolve(process.argv[2] || path.join(ROOT, "presentation", "midterm.pptx"));
const FONT = process.argv[3] || "Open Sans";

// UNC palette. Carolina Blue as text only at 18 pt and above (or 14 pt bold).
const UNC = {
  carolina: "4B9CD3", navy: "13294B", white: "FFFFFF", bolin: "2C5080", cloud: "EDF5FB",
  basin: "5B6670", grey: "A2AAAD", cornerstone: "CFD3D5",
};

// ---------- numbers from results/ ----------
function readCsv(file) {
  const [head, ...lines] = fs.readFileSync(file, "utf8").trim().split(/\r?\n/);
  const keys = head.split(",");
  return lines.map((line) => Object.fromEntries(line.split(",").map((v, i) => [keys[i], v])));
}
const primary = readCsv(path.join(ROOT, "results/analysis/conditions_primary.csv"));
function acc(condition, precision = "fp32") {
  const row = primary.find((r) => r.condition === condition && r.precision === precision);
  if (!row) throw new Error(`no ${condition} ${precision} in conditions_primary.csv`);
  return 100 * Number(row.acc_mean);
}
const randomMean = (k) => (acc(`random1-${k}`) + acc(`random2-${k}`) + acc(`random3-${k}`)) / 3;
const bench = JSON.parse(fs.readFileSync(path.join(ROOT, "results/benchmark/benchmark.json"), "utf8"));
function macs(condition) {
  const row = bench.rows.find((r) => r.condition === condition && r.precision === "fp32");
  if (!row) throw new Error(`no ${condition} in benchmark.json`);
  return row.macs;
}
function layerTimes(key) { // [temporal conv, grouped spatial conv, everything else], microseconds
  const prof = bench.op_profile_us[key];
  const conv = (id) => Object.entries(prof)
    .filter(([name]) => new RegExp(`^(QLinear)?Conv:node_${id}(?!\\d)`).test(name))
    .reduce((sum, [, us]) => sum + us, 0);
  const total = Object.values(prof).reduce((sum, us) => sum + us, 0);
  const temporal = conv("Conv_90"), spatial = conv("Conv_92");
  return [temporal, spatial, total - temporal - spatial];
}
const matched = JSON.parse(fs.readFileSync(
  path.join(ROOT, "results/supplementary/matched_budget/analysis.json"), "utf8")).comparisons
  .find((c) => c.base === "relevance-16" && c.precision === "int8" && c.a === "50" && c.b === "0");
if (!matched) throw new Error("matched-budget comparison not found");

const FULL = acc("full-22");
const t32 = layerTimes("full-22 fp32"), t8 = layerTimes("full-22 int8");
const pruneSaving = 100 * (1 - macs("relevance-16-prune50") / macs("relevance-16"));
const sign = (v) => (v < 0 ? "−" : "+") + Math.abs(v).toFixed(2);
console.log([
  `full-22 FP32 ${FULL.toFixed(2)}%`,
  `16: proposed ${acc("proposed-16").toFixed(2)}, relevance ${acc("relevance-16").toFixed(2)}, weightnorm ${acc("weightnorm-16").toFixed(2)}, random ${randomMean(16).toFixed(2)}`,
  `8: proposed ${acc("proposed-8").toFixed(2)}, relevance ${acc("relevance-8").toFixed(2)}, weightnorm ${acc("weightnorm-8").toFixed(2)}, random ${randomMean(8).toFixed(2)}`,
  `layer us FP32 ${t32.map((v) => v.toFixed(1))} | INT8 ${t8.map((v) => v.toFixed(1))}`,
  `MACs 22/16/8 ${[macs("full-22"), macs("proposed-16"), macs("proposed-8")].join("/")}; prune50 saves ${pruneSaving.toFixed(2)}%`,
  `matched ${sign(matched.difference_pp)} [${sign(matched.ci_low_pp)}, ${sign(matched.ci_high_pp)}]`,
].join("\n"));

// ---------- speaker notes from the talk script ----------
const SCRIPT = path.join(ROOT, "presentation", "midterm_script.md");
const NOTES = (() => {
  const md = fs.readFileSync(SCRIPT, "utf8");
  const heads = [...md.matchAll(/^## (\d+)\b.*$/gm)];
  if (heads.map((m) => m[1]).join() !== Array.from({ length: 14 }, (_, i) => i + 1).join()) {
    throw new Error(`${SCRIPT} needs sections "## 1" to "## 14" in order`);
  }
  return heads.map((m, i) => {
    const lines = md.slice(m.index + m[0].length, heads[i + 1] ? heads[i + 1].index : undefined).split(/\r?\n/);
    const say = lines.filter((l) => l.startsWith("> ")).map((l) => l.slice(2).trim()).join(" ");
    if (!say) throw new Error(`slide ${m[1]} has no "> " script lines in ${SCRIPT}`);
    const asked = lines.find((l) => l.startsWith("If asked: "));
    return asked ? `${say}\n\n${asked.trim()}` : say;
  });
})();

// ---------- deck frame ----------
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.author = "Peiye Li";
pres.company = "UNC-Chapel Hill";
pres.title = "EEG Channel Selection and Model Compression: midterm review";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
const S = pres.ShapeType;
const W = 13.333, M = 0.6, CW = W - 2 * M;

pres.defineSlideMaster({
  title: "UNC_CONTENT",
  background: { color: UNC.white },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.42, w: CW, h: 1.0, fontFace: FONT, fontSize: 30,
      bold: true, color: UNC.navy, align: "left", valign: "top", margin: 0 }, text: "" } },
    { text: { text: "Peiye Li · COMP 690 · Midterm review · 9 November 2026",
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
function card(slide, x, y, w, h, color) {
  slide.addShape(S.roundRect, { x, y, w, h, rectRadius: 0.12, fill: { color }, line: { color, width: 0.75 } });
}
function bigStat(slide, value, label, x, y, w, o = {}) {
  text(slide, value, { x, y, w, h: o.valueH || 0.7, fontSize: o.size || 34, bold: true, color: o.color || UNC.carolina });
  text(slide, label, { x, y: y + (o.valueH || 0.7), w, h: o.labelH || 0.6, fontSize: o.labelSize || 15,
    color: o.labelColor || UNC.navy });
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
    const hi = (o.highlight || []).includes(name);
    slide.addShape(S.ellipse, {
      x: o.cx + gx * o.step - o.dot / 2, y: o.cy + (gy - 2.5) * o.step - o.dot / 2, w: o.dot, h: o.dot,
      fill: { color: hi ? o.hiFill : o.dotFill }, line: { color: hi ? o.hiLine : o.dotLine, width: 1 },
      objectName: `electrode ${name}`,
    });
  }
}

// ---------- 1 cover ----------
pres.addSection({ title: "Problem and novelty" });
let s = pres.addSlide({ masterName: "UNC_COVER", sectionTitle: "Problem and novelty" });
eyebrow(s, "COMP 690-193 · Midterm project review · 9 November 2026", 0.8, 0.75, 9, UNC.navy);
text(s, [{ text: "EEG Channel Selection", options: { breakLine: true } },
  { text: "and Model Compression", options: { breakLine: true } }, { text: "for Motor-Imagery BCI" }],
{ x: 0.8, y: 1.85, w: 7.8, h: 2.1, fontSize: 38, bold: true });
text(s, "How many electrodes, and how many bits, can a decoder lose on a new recording day?",
  { x: 0.8, y: 4.2, w: 7.4, h: 1.0, fontSize: 20 });
text(s, "Peiye Li", { x: 0.8, y: 6.15, w: 5, h: 0.5, fontSize: 20, bold: true });
montage(s, { cx: 10.75, cy: 3.85, step: 0.5, dot: 0.27, outline: UNC.navy, dotFill: UNC.white, dotLine: UNC.navy });
s.addNotes(NOTES[0]);

// ---------- 2 problem ----------
s = pres.addSlide({ masterName: "UNC_CONTENT", sectionTitle: "Problem and novelty" });
s.addText("Why Fewer Electrodes?", { placeholder: "title" });
const halfW = (CW - 0.4) / 2;
[["1", "Hardware cost", "Each electrode needs a contact, an amplifier channel and setup time"],
  ["2", "Compute cost", "The model's first layer runs once per electrode"]].forEach(([n, head, body], i) => {
  const x = M + i * (halfW + 0.4), y = 1.75;
  card(s, x, y, halfW, 3.15, UNC.cloud);
  s.addShape(S.ellipse, { x: x + 0.45, y: y + 0.45, w: 0.75, h: 0.75, fill: { color: UNC.carolina },
    line: { color: UNC.carolina, width: 0.75 } });
  text(s, n, { x: x + 0.45, y: y + 0.45, w: 0.75, h: 0.75, fontSize: 24, bold: true, align: "center", valign: "middle" });
  text(s, head, { x: x + 0.45, y: y + 1.5, w: halfW - 0.9, h: 0.55, fontSize: 24, bold: true });
  text(s, body, { x: x + 0.45, y: y + 2.1, w: halfW - 0.9, h: 0.75, fontSize: 16, color: UNC.basin });
});
text(s, [{ text: "Fewer electrodes save both. ", options: { bold: true } },
  { text: "Pruning the network saves only compute." }], { x: M, y: 5.35, w: CW, h: 0.6, fontSize: 22 });
s.addNotes(NOTES[1]);

// ---------- 3 landscape ----------
s = pres.addSlide({ masterName: "UNC_CONTENT", sectionTitle: "Problem and novelty" });
s.addText("Related Work and Novelty", { placeholder: "title" });
eyebrow(s, "Already shown", M, 1.72, 6);
[["EEGNet", "2018", "Compact CNN for EEG; my decoder"],
  ["Q-EEGNet", "2020", "8-bit EEGNet on a microcontroller"],
  ["MI-BMInet", "2024", "Weight-based electrode selection, then 8-bit"],
  ["mRMR", "2005", "Relevance minus redundancy; basis of my rule"]].forEach(([name, year, what], i) => {
  const y = 2.2 + i * 1.02;
  s.addShape(S.ellipse, { x: M, y: y + 0.1, w: 0.22, h: 0.22, fill: { color: UNC.carolina },
    line: { color: UNC.carolina, width: 0.75 } });
  text(s, [{ text: name, options: { bold: true } }, { text: "   " + year, options: { color: UNC.basin } }],
    { x: M + 0.45, y, w: 5.9, h: 0.42, fontSize: 18 });
  text(s, what, { x: M + 0.45, y: y + 0.44, w: 5.9, h: 0.4, fontSize: 15, color: UNC.basin });
});
const px = 7.3, pw = W - M - px;
card(s, px, 1.7, pw, 4.7, UNC.carolina);
eyebrow(s, "This project adds", px + 0.45, 2.05, pw - 0.9, UNC.navy);
text(s, [{ text: "One factor at a time, same budget", options: { bullet: true, breakLine: true } },
  { text: "Test day locked until the plan froze", options: { bullet: true, breakLine: true } },
  { text: "A negative result, explained", options: { bullet: true } }],
{ x: px + 0.45, y: 2.55, w: pw - 0.9, h: 2.6, fontSize: 21, bold: true, paraSpaceAfter: 14 });
text(s, "Not a new algorithm: an mRMR-style rule", { x: px + 0.45, y: 5.65, w: pw - 0.9, h: 0.4, fontSize: 14 });
text(s, "Lawhern et al., J. Neural Eng. 2018 · Schneider et al., SMARTCOMP 2020 · Wang et al., arXiv 2203.14592 · Peng et al., IEEE TPAMI 2005",
  { x: M, y: 6.55, w: CW, h: 0.3, fontSize: 10, color: UNC.basin });
s.addNotes(NOTES[2]);

// ---------- 4 question ----------
s = pres.addSlide({ masterName: "UNC_CONTENT", sectionTitle: "Problem and novelty" });
s.addText("Research Question", { placeholder: "title" });
card(s, M, 1.7, CW, 2.75, UNC.cloud);
s.addShape(S.ellipse, { x: M + 0.5, y: 2.52, w: 1.1, h: 1.1, fill: { color: UNC.carolina },
  line: { color: UNC.carolina, width: 0.75 } });
text(s, "?", { x: M + 0.5, y: 2.52, w: 1.1, h: 1.1, fontSize: 44, bold: true, align: "center", valign: "middle" });
text(s, "At a fixed electrode count, does accounting for redundancy beat relevance-only and weight-based selection on a new recording day? Does any advantage survive INT8?",
  { x: M + 2.0, y: 1.95, w: CW - 2.45, h: 2.25, fontSize: 24, bold: true, valign: "middle" });
eyebrow(s, "Hypothesis", M, 4.85, 6);
text(s, "A task-aware subset beats a random one, but correlated electrodes are not necessarily dispensable. No gain in accuracy or speed is assumed.",
  { x: M, y: 5.25, w: CW, h: 1.0, fontSize: 18 });
s.addNotes(NOTES[3]);

// ---------- 5 data ----------
pres.addSection({ title: "Approach" });
s = pres.addSlide({ masterName: "UNC_CONTENT", sectionTitle: "Approach" });
s.addText("Dataset and Evaluation", { placeholder: "title" });
const qW = (CW - 3 * 0.3) / 4;
[["9", "people"], ["2", "recording days each"], ["22", "EEG electrodes"], ["4", "imagined movements"]].forEach(([n, l], i) => {
  const x = M + i * (qW + 0.3);
  card(s, x, 1.7, qW, 1.6, UNC.cloud);
  bigStat(s, n, l, x + 0.35, 1.8, qW - 0.7, { size: 40, valueH: 0.75, labelH: 0.6 });
});
const fW = 3.55, fGap = (CW - 3 * fW) / 2;
[["Day 1", "Validate in 3 folds; make every decision"], ["Freeze the plan", "3 Nov, before day 2 was read"],
  ["Day 2", "Score once, FP32 and INT8"]].forEach(([head, body], i) => {
  const x = M + i * (fW + fGap), y = 3.8, h = 1.65, dark = i === 1;
  card(s, x, y, fW, h, dark ? UNC.navy : UNC.cloud);
  text(s, head, { x: x + 0.35, y: y + 0.3, w: fW - 0.7, h: 0.5, fontSize: 20, bold: true, color: dark ? UNC.white : UNC.navy });
  text(s, body, { x: x + 0.35, y: y + 0.85, w: fW - 0.7, h: 0.7, fontSize: 15, color: dark ? UNC.white : UNC.navy });
  if (i < 2) {
    s.addShape(S.rightArrow, { x: x + fW + (fGap - 0.42) / 2, y: y + h / 2 - 0.2, w: 0.42, h: 0.4,
      fill: { color: UNC.carolina }, line: { color: UNC.carolina, width: 0.75 } });
  }
});
text(s, "BCI Competition IV 2a · 250 Hz · 2-s window after each cue · left hand, right hand, feet, tongue",
  { x: M, y: 5.85, w: CW, h: 0.35, fontSize: 13, color: UNC.basin });
s.addNotes(NOTES[4]);

// ---------- 6 cost ----------
s = pres.addSlide({ masterName: "UNC_CONTENT", sectionTitle: "Approach" });
s.addText("Systems View: Model Size and Compute", { placeholder: "title" });
[["2,420", "parameters, about 9.4 KB in FP32"], ["5.9 M", "multiply-accumulates per decision"],
  ["96%", "of them in the first layer, which runs once per electrode"]].forEach(([n, l], i) => {
  bigStat(s, n, l, M, 1.72 + i * 1.45, 5.3);
});
s.addChart(pres.ChartType.bar, [{ name: "Multiply-accumulates (millions)",
  labels: ["22 electrodes", "16 electrodes", "8 electrodes"],
  values: [macs("full-22"), macs("proposed-16"), macs("proposed-8")].map((v) => v / 1e6) }],
chartOpts({ x: 6.4, y: 1.6, w: W - M - 6.4, h: 4.35, barDir: "col", chartColors: [UNC.carolina], barGapWidthPct: 70,
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", dataLabelFontSize: 16, dataLabelFontBold: true,
  valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 7, valGridLine: { style: "none" }, catGridLine: { style: "none" },
  catAxisLabelFontSize: 14, showLegend: false, showTitle: true, title: "Multiply-accumulates per decision (millions)" }));
text(s, "Weights are tiny; compute follows the electrode count.", { x: M, y: 6.2, w: CW, h: 0.45, fontSize: 20, bold: true });
s.addNotes(NOTES[5]);

// ---------- 7 selection ----------
s = pres.addSlide({ masterName: "UNC_CONTENT", sectionTitle: "Approach" });
s.addText("Method: Electrode Selection", { placeholder: "title" });
const leftW = 7.35;
card(s, M, 1.7, leftW, 1.0, UNC.navy);
text(s, [{ text: "next = argmax ", options: { color: UNC.white } }, { text: "relevance", options: { color: UNC.carolina, bold: true } },
  { text: " − 0.5 × ", options: { color: UNC.white } }, { text: "redundancy", options: { color: UNC.carolina, bold: true } }],
{ x: M + 0.35, y: 1.7, w: leftW - 0.7, h: 1.0, fontSize: 18, valign: "middle" });
text(s, [{ text: "Relevance  ", options: { bold: true } },
  { text: "how well an electrode's μ and β power separate the four movements", options: { breakLine: true } },
  { text: "Redundancy  ", options: { bold: true } }, { text: "mean |ρ| with the electrodes already chosen" }],
{ x: M, y: 3.0, w: leftW, h: 1.35, fontSize: 16, paraSpaceAfter: 10 });
eyebrow(s, "Controls at the same count, 16 or 8", M, 4.55, leftW);
const chipW = (leftW - 0.25) / 2;
["Relevance only", "Weight norm", "Random × 3", "All 22"].forEach((c, i) => {
  const x = M + (i % 2) * (chipW + 0.25), y = 4.95 + Math.floor(i / 2) * 0.65;
  card(s, x, y, chipW, 0.5, UNC.cloud);
  text(s, c, { x, y, w: chipW, h: 0.5, fontSize: 15, bold: true, align: "center", valign: "middle" });
});
text(s, "Day-1 data only; each subset retrains EEGNet from scratch", { x: M, y: 6.3, w: leftW, h: 0.35, fontSize: 13, color: UNC.basin });
montage(s, { cx: 10.45, cy: 3.75, step: 0.48, dot: 0.27, outline: UNC.basin, dotFill: UNC.cornerstone, dotLine: UNC.grey,
  highlight: ["C3", "C4", "CP3", "CP4"], hiFill: UNC.carolina, hiLine: UNC.navy });
text(s, "Every rule favoured C3, C4, CP3 and CP4 (blue), over the hand areas",
  { x: 8.35, y: 5.75, w: 4.2, h: 0.6, fontSize: 13, color: UNC.basin, align: "center" });
s.addNotes(NOTES[6]);

// ---------- 8 accuracy ----------
pres.addSection({ title: "Results so far" });
s = pres.addSlide({ masterName: "UNC_CONTENT", sectionTitle: "Results so far" });
s.addText("Result 1: Accuracy vs. Electrode Count", { placeholder: "title" });
const k = ["8", "16", "22"];
s.addChart(pres.ChartType.line, [
  { name: "Proposed (redundancy-aware)", labels: k, values: [acc("proposed-8"), acc("proposed-16"), FULL] },
  { name: "Relevance only", labels: k, values: [acc("relevance-8"), acc("relevance-16"), FULL] },
  { name: "Weight norm", labels: k, values: [acc("weightnorm-8"), acc("weightnorm-16"), FULL] },
  { name: "Random (mean of 3)", labels: k, values: [randomMean(8), randomMean(16), FULL] },
], chartOpts({ x: M, y: 1.6, w: 8.3, h: 5.05, chartColors: [UNC.carolina, UNC.navy, UNC.basin, UNC.grey],
  lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 9, valAxisMinVal: 55, valAxisMaxVal: 70, valAxisMajorUnit: 5,
  valAxisLabelFormatCode: "0", showValAxisTitle: true, valAxisTitle: "Day-2 accuracy (%)", showCatAxisTitle: true,
  catAxisTitle: "Electrodes kept", valGridLine: { color: "DDE3E8", size: 1 }, catGridLine: { style: "none" },
  showLegend: true, legendPos: "b", legendFontSize: 12 }));
[["65.0%", "all 22 electrodes; chance is 25%"], ["66–67%", "16 chosen by any task-aware rule"],
  ["3–5 pts", "lost with 8 electrodes"], ["≤ 0.5 pts", "change after INT8"]].forEach(([n, l], i) => {
  bigStat(s, n, l, 9.35, 1.7 + i * 1.22, W - M - 9.35, { size: 28, valueH: 0.55, labelH: 0.5, labelSize: 14 });
});
s.addNotes(NOTES[7]);

// ---------- 9 redundancy ----------
s = pres.addSlide({ masterName: "UNC_CONTENT", sectionTitle: "Results so far" });
s.addText("Result 2: Comparing Selection Rules", { placeholder: "title" });
const vs = ["vs random (mean of 3)", "vs weight norm", "vs relevance only"]; // bars draw bottom to top
const diffs = (kk) => [acc(`proposed-${kk}`) - randomMean(kk), acc(`proposed-${kk}`) - acc(`weightnorm-${kk}`),
  acc(`proposed-${kk}`) - acc(`relevance-${kk}`)];
s.addChart(pres.ChartType.bar, [{ name: "16 electrodes", labels: vs, values: diffs(16) },
  { name: "8 electrodes", labels: vs, values: diffs(8) }],
chartOpts({ x: M, y: 1.6, w: 7.4, h: 5.0, barDir: "bar", barGrouping: "clustered", barGapWidthPct: 60,
  chartColors: [UNC.carolina, UNC.navy], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0.0;-0.0;0.0",
  valAxisMinVal: -2, valAxisMaxVal: 4, valAxisMajorUnit: 1, valAxisLabelFormatCode: "0", showValAxisTitle: true,
  valAxisTitle: "Proposed minus control (accuracy points)", valGridLine: { color: "DDE3E8", size: 1 },
  catGridLine: { style: "none" }, catAxisLabelFontSize: 14, catAxisLabelPos: "low", showLegend: true, legendPos: "b" }));
const rx0 = 8.35, rw = W - M - rx0;
card(s, rx0, 1.7, rw, 4.8, UNC.cloud);
eyebrow(s, "Why so little", rx0 + 0.35, 2.0, rw - 0.7);
bigStat(s, "0.74", "mean |ρ| between electrodes' band power", rx0 + 0.35, 2.4, rw - 0.7);
bigStat(s, "6–7 of 9", "people got the same electrodes from both rules", rx0 + 0.35, 3.85, rw - 0.7);
text(s, "Band power swings together across the scalp, so every candidate looks equally redundant.",
  { x: rx0 + 0.35, y: 5.35, w: rw - 0.7, h: 0.95, fontSize: 13, color: UNC.basin });
s.addNotes(NOTES[8]);

// ---------- 10 INT8 ----------
s = pres.addSlide({ masterName: "UNC_CONTENT", sectionTitle: "Results so far" });
s.addText("Result 3: INT8 Quantization", { placeholder: "title" });
[["≤ 0.5 pts", "accuracy change in all 17 conditions"], ["2.9×", "fewer weight bytes (9,576 → 3,260)"],
  ["+28%", "time per decision (0.36 vs 0.28 ms)"]].forEach(([n, l], i) => {
  bigStat(s, n, l, M, 1.72 + i * 1.45, 3.9);
});
const layers = ["Temporal conv", "Spatial conv (grouped)", "Everything else"];
s.addChart(pres.ChartType.bar, [{ name: "FP32", labels: layers, values: t32 }, { name: "INT8", labels: layers, values: t8 }],
  chartOpts({ x: 4.85, y: 1.6, w: W - M - 4.85, h: 4.4, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: [UNC.basin, UNC.carolina], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0",
    valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 270, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    catAxisLabelFontSize: 14, showLegend: true, legendPos: "t", showTitle: true,
    title: "Time per decision by layer (µs), 22 electrodes" }));
text(s, [{ text: "Fewer bytes is not faster: ", options: { bold: true } },
  { text: "the quantized grouped conv ran 14× slower here" }], { x: M, y: 6.2, w: CW, h: 0.45, fontSize: 20 });
s.addNotes(NOTES[9]);

// ---------- 11 pruning ----------
s = pres.addSlide({ masterName: "UNC_CONTENT", sectionTitle: "Results so far" });
s.addText("Result 4: Pruning", { placeholder: "title" });
eyebrow(s, "Added after the test results · independent M5 reproduction", M, 1.55, CW);
const tW = (CW - 2 * 0.3) / 3;
const matchedText = `${(matched.difference_pp < 0 ? "−" : "+") + Math.abs(matched.difference_pp).toFixed(2)} pts`;
const ciText = `95% CI ${(matched.ci_low_pp < 0 ? "−" : "+") + Math.abs(matched.ci_low_pp).toFixed(2)} to ${(matched.ci_high_pp < 0 ? "−" : "+") + Math.abs(matched.ci_high_pp).toFixed(2)}`;
[[matchedText, "50% pruned vs. unpruned, both given 100 extra epochs", `16 electrodes, INT8 · ${ciText}`, true],
  [`${pruneSaving.toFixed(1)}%`, "of multiply-accumulates saved by pruning half the last layer", "", false],
  [`${acc("relevance-16-prune50", "int8").toFixed(1)}%`, "frozen final choice on day 2, the same mean as all 22",
    "4 of 9 people lost more than 2 points", false]].forEach(([n, l, sub, hi], i) => {
  const x = M + i * (tW + 0.3), y = 2.05, h = 3.75;
  card(s, x, y, tW, h, hi ? UNC.carolina : UNC.cloud);
  text(s, n, { x: x + 0.35, y: y + 0.35, w: tW - 0.7, h: 0.8, fontSize: 36, bold: true, color: hi ? UNC.navy : UNC.carolina });
  text(s, l, { x: x + 0.35, y: y + 1.3, w: tW - 0.7, h: 1.2, fontSize: 16 });
  if (sub) text(s, sub, { x: x + 0.35, y: y + 2.65, w: tW - 0.7, h: 0.8, fontSize: 13, color: hi ? UNC.navy : UNC.basin });
});
text(s, "Pruned models first got 100 extra epochs; equal training removes that head start.",
  { x: M, y: 6.15, w: CW, h: 0.45, fontSize: 16 });
s.addNotes(NOTES[10]);

// ---------- 12 progress ----------
pres.addSection({ title: "Progress and next steps" });
s = pres.addSlide({ masterName: "UNC_CONTENT", sectionTitle: "Progress and next steps" });
s.addText("Progress vs. Schedule", { placeholder: "title" });
const phases = [["Sep", "Baseline and export check", true], ["Early–mid Oct", "Selection rules and validation", true],
  ["Late Oct – early Nov", "INT8, pruning; plan frozen 3 Nov", true], ["Nov", "Evaluation, statistics, demo", true],
  ["Late Nov – early Dec", "Report and final talk", false]];
const colW = CW / phases.length;
s.addShape(S.line, { x: M + colW / 2, y: 3.05, w: CW - colW, h: 0, line: { color: UNC.cornerstone, width: 3 } });
phases.forEach(([when, what, done], i) => {
  const cx = M + colW * (i + 0.5);
  text(s, when, { x: cx - colW / 2 + 0.1, y: 1.8, w: colW - 0.2, h: 0.8, fontSize: 15, bold: true, align: "center", valign: "bottom" });
  s.addShape(S.ellipse, { x: cx - 0.28, y: 2.77, w: 0.56, h: 0.56, fill: { color: done ? UNC.carolina : UNC.white },
    line: { color: UNC.carolina, width: 3 } });
  text(s, what, { x: cx - colW / 2 + 0.1, y: 3.6, w: colW - 0.2, h: 0.9, fontSize: 15, align: "center" });
  text(s, done ? "Done" : "In progress", { x: cx - colW / 2 + 0.1, y: 4.55, w: colW - 0.2, h: 0.4, fontSize: 14,
    bold: true, color: done ? UNC.bolin : UNC.navy, align: "center" });
});
card(s, M, 5.35, CW, 0.95, UNC.cloud);
text(s, [{ text: "Beyond the plan: ", options: { bold: true } },
  { text: "an equal-fine-tuning pruning control (post-test) and public code with every model on GitHub" }],
{ x: M + 0.35, y: 5.35, w: CW - 0.7, h: 0.95, fontSize: 16, valign: "middle" });
s.addNotes(NOTES[11]);

// ---------- 13 next steps ----------
s = pres.addSlide({ masterName: "UNC_CONTENT", sectionTitle: "Progress and next steps" });
s.addText("Next Steps", { placeholder: "title" });
[["Required", "Finish the report and slides", "Revise from today's feedback; the frozen-plan results stay the main claims", "Late Nov – early Dec", true],
  ["Follow-up A · first", "Why was INT8 slower?", "Keep only the grouped conv in FP32; time FP32, INT8 and the mix on the same M4", "No retraining needed", false],
  ["Follow-up B · if time", "A better redundancy measure", "Remove the band power all electrodes share; retest at 8 electrodes; confirm on a second two-day dataset", "Needs new training runs", false]]
  .forEach(([eb, head, body, foot, dark], i) => {
    const x = M + i * (tW + 0.3), y = 1.7, h = 4.2;
    card(s, x, y, tW, h, dark ? UNC.navy : UNC.cloud);
    eyebrow(s, eb, x + 0.35, y + 0.35, tW - 0.7, dark ? UNC.carolina : UNC.bolin);
    text(s, head, { x: x + 0.35, y: y + 0.8, w: tW - 0.7, h: 1.1, fontSize: 20, bold: true, color: dark ? UNC.white : UNC.navy });
    text(s, body, { x: x + 0.35, y: y + 2.1, w: tW - 0.7, h: 1.35, fontSize: 15, color: dark ? UNC.white : UNC.navy });
    text(s, foot, { x: x + 0.35, y: y + 3.55, w: tW - 0.7, h: 0.4, fontSize: 13, bold: true, color: dark ? UNC.carolina : UNC.basin });
  });
s.addNotes(NOTES[12]);

// ---------- 14 takeaways ----------
s = pres.addSlide({ masterName: "UNC_CLOSING", sectionTitle: "Progress and next steps" });
eyebrow(s, "Takeaways", 0.8, 0.8, 6, UNC.carolina);
["Electrode count and choice matter more than how the weights are stored",
  "16 chosen electrodes match all 22; the redundancy term did not beat relevance only",
  "INT8 saves bytes, not M4 time; at equal training, pruning costs accuracy"].forEach((t, i) => {
  const y = 1.55 + i * 1.35;
  text(s, String(i + 1), { x: 0.8, y, w: 0.8, h: 0.9, fontSize: 44, bold: true, color: UNC.carolina });
  text(s, t, { x: 1.75, y: y + 0.08, w: 10.6, h: 1.05, fontSize: 24, color: UNC.white });
});
text(s, "Thank you · questions welcome", { x: 0.8, y: 6.15, w: 8, h: 0.5, fontSize: 20, bold: true, color: UNC.carolina });
s.addNotes(NOTES[13]);

pres.writeFile({ fileName: OUT }).then(async (file) => {
  // pptxgenjs also writes folder entries into the ZIP; some cloud services refuse such files.
  const JSZip = require("jszip");
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  for (const [name, entry] of Object.entries(zip.files)) if (entry.dir) delete zip.files[name];
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  console.log(`wrote ${file}`);
});
