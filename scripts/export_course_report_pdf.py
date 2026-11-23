"""Typeset docs/COURSE_REPORT.md as the course's PDF: 12-point body, 1-inch margins, US letter.

The course requires 10-12 pages of main text (Introduction through Conclusion) in a 12-point font with
1-inch margins; References, Artifacts, Appendices and the Ethics section do not count. Tables and figure
captions are set at 10 points. References start on a new page, and the script prints the page on which
every section starts, so the main-text length can be checked at a glance.

Needs ReportLab and Pillow, which the project environment does not include; run it in a throwaway
environment so that uv.lock stays as it is:

    uv run --offline --no-project --with reportlab --with pillow python scripts/export_course_report_pdf.py
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "COURSE_REPORT.md"
TARGET = ROOT / "docs" / "COURSE_REPORT.pdf"
MARGIN = 72  # 1 inch
WIDTH = letter[0] - 2 * MARGIN
NAVY = colors.HexColor("#13294B")
BODY, SMALL = 12, 10
FONT_DIR = Path("/System/Library/Fonts/Supplemental")
FONTS = {"Body": "Times New Roman.ttf", "BodyBold": "Times New Roman Bold.ttf",
         "BodyItalic": "Times New Roman Italic.ttf", "BodyBoldItalic": "Times New Roman Bold Italic.ttf"}  # fmt: skip


def fonts() -> tuple[str, str, str]:
    if all((FONT_DIR / f).exists() for f in FONTS.values()):
        for name, file in FONTS.items():
            pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
        pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="BodyItalic",
                                      boldItalic="BodyBoldItalic")  # fmt: skip
        return "Body", "BodyBold", "BodyItalic"
    return "Times-Roman", "Times-Bold", "Times-Italic"


def styles() -> dict[str, ParagraphStyle]:
    normal, bold, _ = fonts()
    base = {"fontName": normal, "textColor": colors.black}
    return {
        "title": ParagraphStyle("title", fontName=bold, fontSize=16, leading=20, spaceAfter=10, textColor=NAVY),
        "h2": ParagraphStyle("h2", fontName=bold, fontSize=13.5, leading=17, spaceBefore=10, spaceAfter=4,
                             keepWithNext=True, textColor=NAVY),
        "h3": ParagraphStyle("h3", fontName=bold, fontSize=BODY, leading=15, spaceBefore=7, spaceAfter=2,
                             keepWithNext=True, textColor=NAVY),
        "body": ParagraphStyle("body", **base, fontSize=BODY, leading=14.6, spaceAfter=6, alignment=4),
        "item": ParagraphStyle("item", **base, fontSize=BODY, leading=14.6, spaceAfter=2, alignment=4),
        "cell": ParagraphStyle("cell", **base, fontSize=SMALL, leading=12),
        "head": ParagraphStyle("head", fontName=bold, fontSize=SMALL, leading=12, textColor=colors.white),
        "caption": ParagraphStyle("caption", fontName=normal, fontSize=SMALL, leading=12.2, spaceAfter=8,
                                  textColor=colors.HexColor("#333333")),
    }  # fmt: skip


def inline(text: str) -> str:
    text = escape(text.replace("–", "-").replace("—", "-").replace("−", "-"))
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier" size="10.5">\1</font>', text)
    return re.sub(r"(https?://\S+)", r'<link href="\1" color="#2E75B6">\1</link>', text)


def table(lines: list[str], st: dict) -> Table:
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines]
    rows = [r for r in rows if not all(re.fullmatch(r"[:\- ]+", c) for c in r)]
    longest = [max(len(re.sub(r"[*`]", "", r[i])) for r in rows) for i in range(len(rows[0]))]
    widths = [WIDTH * min(max(n, 6), 60) / sum(min(max(m, 6), 60) for m in longest) for n in longest]
    cells = [[Paragraph(inline(c), st["head" if i == 0 else "cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
    stripes = [colors.white, colors.HexColor("#F1F4F8")]
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                           ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), stripes)]))  # fmt: skip
    return t


def figure(line: str, st: dict) -> list:
    label, path = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line).groups()
    image = Image(str((SOURCE.parent / path).resolve()))
    scale = min(WIDTH / image.imageWidth, 280 / image.imageHeight)
    image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
    return [Spacer(1, 2), image, Paragraph(inline(label), st["caption"])]


class Report(SimpleDocTemplate):
    """Records the page on which each section heading lands."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sections: list[tuple[str, int]] = []

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in ("title", "h2"):
            self.sections.append((flowable.getPlainText(), self.page))


def story(lines: list[str], st: dict) -> list:
    out, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
        elif line.startswith("|"):
            group = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                group.append(lines[i])
                i += 1
            out += [table(group, st), Spacer(1, 8)]
        elif line.startswith("!["):
            out += figure(line, st)
            i += 1
        elif line.startswith("# "):
            out.append(Paragraph(inline(line[2:]), st["title"]))
            i += 1
        elif line.startswith("## "):
            if line[3:].startswith("References"):
                out.append(PageBreak())
            out.append(Paragraph(inline(line[3:]), st["h2"]))
            i += 1
        elif line.startswith("### "):
            out.append(Paragraph(inline(line[4:]), st["h3"]))
            i += 1
        elif re.match(r"^(\d+\.|-) ", line):
            flowables, i = bullets(lines, i, st)
            out += flowables
        else:
            group = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(BLOCK, lines[i].strip()):
                group.append(lines[i].strip())
                i += 1
            out.append(Paragraph(inline(" ".join(group)), st["body"]))
    return out


BLOCK = r"^(#|\||!\[|\d+\. |- )"  # lines that start something other than a paragraph


def bullets(lines: list[str], i: int, st: dict) -> tuple[list, int]:
    """A numbered or bulleted list starting at line i, and the index after it."""
    numbered = bool(re.match(r"^\d+\. ", lines[i].strip()))
    items = []
    while i < len(lines) and re.match(r"^(\d+\.|-) ", lines[i].strip()):
        text = re.sub(r"^(\d+\.|-) ", "", lines[i].strip())
        items.append(ListItem(Paragraph(inline(text), st["item"]), leftIndent=16))
        i += 1
    kind = {"bulletType": "1", "bulletFormat": "%s."} if numbered else {"bulletType": "bullet", "start": "-"}
    flow = ListFlowable(items, leftIndent=16, bulletFontSize=BODY, bulletFontName=st["body"].fontName, **kind)
    return [flow, Spacer(1, 4)], i


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    st = styles()
    normal = st["body"].fontName

    def page(canvas, doc):
        canvas.saveState()
        canvas.setFont(normal, 10)
        canvas.drawCentredString(letter[0] / 2, 36, str(doc.page))
        canvas.restoreState()

    fd, temporary = tempfile.mkstemp(prefix=".COURSE_REPORT.", suffix=".pdf", dir=TARGET.parent)
    os.close(fd)
    try:
        title = SOURCE.read_text().splitlines()[0].lstrip("# ")
        doc = Report(temporary, pagesize=letter, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN,
                     bottomMargin=MARGIN, title=title, author="Peiye Li")  # fmt: skip
        doc.build(story(SOURCE.read_text().splitlines(), st), onFirstPage=page, onLaterPages=page)
        Path(temporary).replace(TARGET)
    finally:
        Path(temporary).unlink(missing_ok=True)
    for name, number in doc.sections:
        print(f"page {number:>2}  {name[:70]}")
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
