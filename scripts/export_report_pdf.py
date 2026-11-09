"""Render the reviewed Markdown report as a portable, typeset PDF.

Use the bundled artifact Python runtime (reportlab + Pillow); training dependencies
and uv.lock are not changed. Regenerate after build_report.py updates the evidence.
"""

from __future__ import annotations

from pathlib import Path
import os
import re
import tempfile
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle,getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"docs/REPORT.md"
TARGET=ROOT/"docs/REPORT.pdf"
WIDTH=516
NAVY=colors.HexColor("#14213D")
BLUE=colors.HexColor("#2E75B6")


def clean(text):
    return text.replace("\u2013","-").replace("\u2014","-").replace("\u2011","-").replace("\u2212","-")


def inline(text):
    text=escape(clean(text))
    def link(match):
        label,target=match.groups()
        if not target.startswith(("http:","https:")):
            location=target.removeprefix("../") if target.startswith("../") else "docs/"+target
            return label+f" <font size='8' color='#666666'>({location})</font>"
        return f'<link href="{target}" color="#2E75B6">{label}</link>'
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,text)
    text=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'`([^`]+)`',r'<font name="Courier">\1</font>',text)
    return text


def setup_styles():
    fonts=Path("/System/Library/Fonts/Supplemental")
    names={"Report":"Arial.ttf","ReportBold":"Arial Bold.ttf","ReportItalic":"Arial Italic.ttf"}
    if all((fonts/name).exists() for name in names.values()):
        for name,file in names.items():pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
        pdfmetrics.registerFontFamily("Report",normal="Report",bold="ReportBold",italic="ReportItalic",boldItalic="ReportBold")
        normal,bold="Report","ReportBold"
    else:normal,bold="Helvetica","Helvetica-Bold"
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportTitle",fontName=bold,fontSize=23,leading=28,textColor=NAVY,spaceAfter=18))
    styles.add(ParagraphStyle("ReportBody",fontName=normal,fontSize=10,leading=14.4,textColor=colors.HexColor("#253247"),spaceAfter=9,allowWidows=0,allowOrphans=0))
    styles.add(ParagraphStyle("ReportH2",fontName=bold,fontSize=14,leading=18,textColor=NAVY,spaceBefore=17,spaceAfter=8,keepWithNext=True))
    styles.add(ParagraphStyle("ReportH3",fontName=bold,fontSize=11.5,leading=15,textColor=NAVY,spaceBefore=12,spaceAfter=6,keepWithNext=True))
    styles.add(ParagraphStyle("ReportCell",fontName=normal,fontSize=8.1,leading=10.5,spaceAfter=0))
    styles.add(ParagraphStyle("ReportHeader",fontName=bold,fontSize=8.1,leading=10.5,textColor=colors.white))
    styles.add(ParagraphStyle("ReportCaption",fontName=normal,fontSize=8,leading=10.5,textColor=colors.HexColor("#526177"),spaceAfter=12))
    return styles,normal


def make_table(lines,styles):
    values=[[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    values=[row for row in values if not all(re.fullmatch(r'[:\- ]+',cell) for cell in row)]
    count=len(values[0])
    if count==6:weights=[1.55,.95,.9,.9,1.1,1.65]
    elif count==5:weights=[.85,.8,1.35,1.4,1.6]
    elif count==4:weights=[1.5,1.1,1.1,2.3] if values[0][0]=="Quantity" else [1.2,1.4,1.4,1.2]
    else:weights=[1]*count
    widths=[WIDTH*v/sum(weights) for v in weights]
    cells=[[Paragraph(inline(cell),styles["ReportHeader" if i==0 else "ReportCell"]) for cell in row] for i,row in enumerate(values)]
    result=Table(cells,colWidths=widths,repeatRows=1,hAlign="LEFT",splitByRow=1)
    result.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY),("VALIGN",(0,0),(-1,-1),"TOP"),
        ("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F1F4F8")]),
        ("LINEBELOW",(0,0),(-1,0),.6,NAVY),("LINEBELOW",(0,-1),(-1,-1),.4,colors.HexColor("#C8CFD9"))]))
    return result


def main():
    styles,font=setup_styles()
    lines=SOURCE.read_text().splitlines();story=[];i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith("|"):
            group=[]
            while i<len(lines) and lines[i].strip().startswith("|"):
                group.append(lines[i]);i+=1
            story.extend([make_table(group,styles),Spacer(1,12)]);continue
        if line.startswith("!["):
            match=re.match(r'!\[([^\]]*)\]\(([^)]+)\)',line)
            label,path=match.groups();path=(SOURCE.parent/path).resolve()
            image=Image(str(path));ratio=min(WIDTH/image.imageWidth,330/image.imageHeight)
            image.drawWidth=image.imageWidth*ratio;image.drawHeight=image.imageHeight*ratio
            image.hAlign="CENTER"
            story.extend([Spacer(1,5),image,Paragraph(inline(label),styles["ReportCaption"])]);i+=1;continue
        if line.startswith("# "):story.append(Paragraph(inline(line[2:]),styles["ReportTitle"]));i+=1;continue
        if line.startswith("## "):
            if line.startswith("## 5. "):story.append(PageBreak())
            story.append(Paragraph(inline(line[3:]),styles["ReportH2"]));i+=1;continue
        if line.startswith("### "):story.append(Paragraph(inline(line[4:]),styles["ReportH3"]));i+=1;continue
        group=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(("#","|","![")) and not re.match(r'^\d+\. ',lines[i]):
            group.append(lines[i].strip());i+=1
        story.append(Paragraph(inline(" ".join(group)),styles["ReportBody"]))
    def page(canvas,doc):
        canvas.saveState();canvas.setFont(font,8);canvas.setFillColor(colors.HexColor("#65748A"))
        canvas.drawString(48,letter[1]-29,"EEG channel selection and model compression")
        canvas.drawRightString(letter[0]-48,28,f"Peiye Li | COMP 690-193 | {doc.page}")
        canvas.restoreState()
    fd,temporary=tempfile.mkstemp(prefix=".REPORT.",suffix=".pdf",dir=TARGET.parent)
    os.close(fd)
    try:
        document=SimpleDocTemplate(temporary,pagesize=letter,leftMargin=48,rightMargin=48,topMargin=49,bottomMargin=43,
            title="EEG Channel Selection and Model Compression",author="Peiye Li",subject="Source-backed technical report with supplementary matched-budget experiment")
        document.build(story,onFirstPage=page,onLaterPages=page)
        os.replace(temporary,TARGET)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f"Wrote {TARGET}")


if __name__=="__main__":main()
