#!/usr/bin/env node
/*
 * Build the final-presentation talk script as a Word file from presentation/final_script.md.
 *
 *   npm install --prefix <dir> docx@9.13.0
 *   NODE_PATH=<dir>/node_modules node presentation/pptx/build_final_script_docx.js [output.docx] [chinese-font]
 *
 * Default output: presentation/final_script.docx, US Letter. The Markdown file stays the source:
 * its "- " lines before slide 1 become the how-to list, and each "## N · Title（time）" section gives
 * (backup sections have no time)
 * the English to say ("> " lines), the Chinese meaning ("意思："), and, where present, the
 * "If asked:" line with its Chinese gloss ("（补充：…）"). Text in [brackets] may be skipped; it is
 * set in grey italics. English is Arial, a UNC-approved substitute for Open Sans; Chinese defaults
 * to PingFang SC, which Word replaces with its default Chinese font on computers without it.
 * Headless LibreOffice uses its bundled fontconfig and sees no Chinese font; to check the layout,
 * set FONTCONFIG_FILE to a config that adds /System/Library/Fonts, its Supplemental folder and the
 * folder holding PingFang.ttc (fc-list shows it), then convert to PDF.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, Footer, Header, HeadingLevel, LevelFormat, Packer, PageNumber,
  Paragraph, ShadingType, Tab, TabStopType, TextRun,
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "..");
const SRC = path.join(ROOT, "presentation", "final_script.md");
const OUT = path.resolve(process.argv[2] || path.join(ROOT, "presentation", "final_script.docx"));
const CJK = process.argv[3] || "PingFang SC";

const UNC = { carolina: "4B9CD3", navy: "13294B", bolin: "2C5080", basin: "5B6670", cloud: "EDF5FB", cornerstone: "CFD3D5" };
const FONT = { ascii: "Arial", hAnsi: "Arial", cs: "Arial", eastAsia: CJK };
const pt = (n) => Math.round(n * 2); // run sizes are in half-points
const PAGE = { width: 12240, height: 15840, side: 1296 }; // US Letter and side margins, in twips
const TEXT_WIDTH = PAGE.width - 2 * PAGE.side;
const INDENT = 220; // twips; the English, its meaning and the "If asked" box share this left edge

// ---------- read the script ----------
const lines = fs.readFileSync(SRC, "utf8").split(/\r?\n/);
const first = lines.findIndex((l) => /^## \d+\b/.test(l));
if (first < 0) throw new Error(`no "## 1" section in ${SRC}`);
const title = (lines.find((l) => l.startsWith("# ")) || "# Talk script").slice(2).trim();
const howTo = lines.slice(0, first).filter((l) => l.startsWith("- ")).map((l) => l.slice(2).trim());
const slides = [];
for (const l of lines.slice(first)) {
  const head = l.match(/^## (\d+) · (.+?)(?:（(.+?)）)?\s*$/);
  if (head) {
    slides.push({ n: Number(head[1]), title: head[2], time: head[3] || "", say: [], meaning: "", asked: "", gloss: "" });
    continue;
  }
  const s = slides[slides.length - 1];
  if (l.startsWith("> ")) s.say.push(l.slice(2).trim());
  else if (l.startsWith("意思：")) s.meaning = l.slice("意思：".length).trim();
  else if (l.startsWith("If asked: ")) s.asked = l.slice("If asked: ".length).trim();
  else if (l.startsWith("（补充：") && l.trim().endsWith("）")) s.gloss = l.trim().slice("（补充：".length, -1);
}
if (!slides.length || slides.some((s, i) => s.n !== i + 1)) {
  throw new Error(`${SRC} needs its sections numbered "## 1", "## 2", ... in order`);
}
for (const s of slides) {
  if (!s.say.length || !s.meaning) throw new Error(`slide ${s.n} needs "> " lines and a "意思：" line`);
  if (!s.asked !== !s.gloss) throw new Error(`slide ${s.n}: an "If asked:" line needs its "（补充：…）" line, and the reverse`);
}

// ---------- text helpers ----------
// Straight quotes become curly ones: every apostrophe, and double quotes in alternating pairs.
function curly(text) {
  let open = false;
  return text.replace(/'/g, "’").replace(/"/g, () => ((open = !open) ? "“" : "”"));
}
// One paragraph's runs: "**bold**" spans become bold, "[skippable]" spans take the `skip` style.
function runs(text, base = {}, skip = { italics: true, color: UNC.basin }) {
  const t = curly(text), out = [];
  let last = 0;
  for (const m of t.matchAll(/\*\*([^*]+)\*\*|\[[^\]]*\]/g)) {
    if (m.index > last) out.push(new TextRun({ ...base, text: t.slice(last, m.index) }));
    out.push(new TextRun(m[1] !== undefined ? { ...base, text: m[1], bold: true } : { ...base, ...skip, text: m[0] }));
    last = m.index + m[0].length;
  }
  if (last < t.length) out.push(new TextRun({ ...base, text: t.slice(last) }));
  return out;
}
const label = (text, size) => new TextRun({ text: `${text}  `, bold: true, color: UNC.bolin, size });
// The "If asked" lines sit in a pale box; equal borders on both paragraphs make Word draw one box.
const boxBorder = () => Object.fromEntries(["top", "bottom", "left", "right"].map((side) =>
  [side, { style: BorderStyle.SINGLE, size: 4, color: UNC.cloud, space: 4 }]));
const box = () => ({ shading: { type: ShadingType.CLEAR, color: "auto", fill: UNC.cloud }, border: boxBorder(),
  indent: { left: INDENT + 80, right: 80 }, spacing: { line: 240 } });

// ---------- document ----------
const children = [
  new Paragraph({ heading: HeadingLevel.TITLE, children: runs(title) }),
  new Paragraph({ children: [new TextRun({ text: "EEG Channel Selection and Model Compression for Motor-Imagery BCI",
    size: pt(10.5), color: UNC.basin })] }),
  new Paragraph({ spacing: { after: 220 }, children: [new TextRun({ text: "Peiye Li · COMP 690 · 8 December 2026",
    size: pt(10.5), color: UNC.basin })] }),
  ...howTo.map((t) => new Paragraph({ numbering: { reference: "how-to", level: 0 }, spacing: { after: 60, line: 240 },
    children: runs(t, { size: pt(10.5) }, { color: UNC.basin }) })),
];
for (const s of slides) {
  children.push(new Paragraph({
    heading: HeadingLevel.HEADING_1,
    border: { top: { style: BorderStyle.SINGLE, size: 6, color: UNC.cornerstone, space: 10 } },
    tabStops: [{ type: TabStopType.RIGHT, position: TEXT_WIDTH }], // start time flush with the right margin
    children: [
      new TextRun({ text: String(s.n), color: UNC.carolina }),
      new TextRun({ text: `   ${curly(s.title)}` }),
      new TextRun({ children: [new Tab(), s.time], color: UNC.bolin, size: pt(12) }),
    ],
  }));
  children.push(new Paragraph({
    keepNext: true, keepLines: true, spacing: { after: 100, line: 300 }, indent: { left: INDENT },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: UNC.carolina, space: 10 } },
    children: runs(s.say.join(" "), { size: pt(13) }),
  }));
  children.push(new Paragraph({
    keepNext: Boolean(s.asked), keepLines: true, spacing: { after: s.asked ? 100 : 0, line: 240 }, indent: { left: INDENT },
    children: [label("意思", pt(10.5)), ...runs(s.meaning, { size: pt(10.5), color: UNC.basin }, { color: UNC.basin })],
  }));
  if (s.asked) {
    children.push(new Paragraph({ ...box(), keepNext: true, keepLines: true,
      children: [label("If asked", pt(9.5)), ...runs(s.asked, { size: pt(9.5) })] }));
    children.push(new Paragraph({ ...box(), keepLines: true,
      children: [label("补充", pt(9.5)), ...runs(s.gloss, { size: pt(9.5), color: UNC.basin }, { color: UNC.basin })] }));
  }
}

const small = (text) => new TextRun({ text, size: pt(9), color: UNC.basin });
const doc = new Document({
  creator: "Peiye Li",
  title: "Final presentation talk script",
  description: "Per-slide script for the COMP 690 final presentation, built from presentation/final_script.md",
  styles: {
    default: {
      document: { run: { font: FONT, size: pt(11), color: UNC.navy }, paragraph: { spacing: { after: 0, line: 276 } } },
      title: { run: { font: FONT, size: pt(22), bold: true, color: UNC.navy },
        paragraph: { spacing: { before: 0, after: 60 }, keepNext: true } },
      heading1: { run: { font: FONT, size: pt(15), bold: true, color: UNC.navy },
        paragraph: { spacing: { before: 240, after: 100 }, keepNext: true, keepLines: true } },
    },
  },
  numbering: { config: [{ reference: "how-to", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
    alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 360, hanging: 240 } }, run: { color: UNC.carolina } } }] }] },
  sections: [{
    properties: { page: { size: { width: PAGE.width, height: PAGE.height },
      margin: { top: 1224, bottom: 1224, left: PAGE.side, right: PAGE.side, header: 620, footer: 620 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [small("COMP 690 · Final presentation · Peiye Li")] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT, " / ", PageNumber.TOTAL_PAGES], size: pt(9), color: UNC.basin })] })] }) },
    children,
  }],
});

Packer.toBuffer(doc).then(async (buffer) => {
  // Drop ZIP folder entries, which Word has refused when a file is opened through a cloud service.
  const JSZip = require("jszip");
  const zip = await JSZip.loadAsync(buffer);
  for (const [name, entry] of Object.entries(zip.files)) if (entry.dir) delete zip.files[name];
  fs.writeFileSync(OUT, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
  console.log(`wrote ${OUT} (${slides.length} slides, ${howTo.length} how-to items)`);
});
