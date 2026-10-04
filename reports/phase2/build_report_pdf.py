"""Build the report PDF from its Markdown source using ReportLab."""
from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

ROOT = Path(__file__).parent
SOURCE = ROOT / "PHASE2_FINAL_EVALUATION_REPORT.md"
DEST = ROOT / "PHASE2_FINAL_EVALUATION_REPORT.pdf"


class Architecture(Flowable):
    def __init__(self, width=510, height=125):
        super().__init__(); self.width=width; self.height=height
    def draw(self):
        c=self.canv
        boxes={"api":(8,47,82,34,"FastAPI"),"embed":(121,82,100,34,"Qwen3\nembeddings"),"qdrant":(265,82,102,34,"Qdrant\ndense"),"bm25":(121,15,100,34,"SQLite FTS5\nBM25"),"rrf":(265,47,92,34,"RRF fusion"),"top":(414,47,88,34,"Top five")}
        for x,y,w,h,label in boxes.values():
            c.setFillColor(colors.HexColor("#EAF1F8")); c.setStrokeColor(colors.HexColor("#6B829B")); c.roundRect(x,y,w,h,5,fill=1,stroke=1)
            for i,line in enumerate(label.split("\n")):
                c.setFillColor(colors.HexColor("#1B3148")); c.setFont("Helvetica-Bold",8)
                c.drawCentredString(x+w/2,y+h/2+(4 if len(label.split("\n"))>1 else -3)+(1-i*10),line)
        c.setStrokeColor(colors.HexColor("#536A81")); c.setLineWidth(1.2)
        self._arrow(c,90,66,121,99); self._arrow(c,90,62,121,32); self._arrow(c,221,99,265,99)
        self._arrow(c,221,32,265,61); self._arrow(c,367,99,300,81); self._arrow(c,357,64,414,64)
        c.setFillColor(colors.HexColor("#536A81")); c.setFont("Helvetica",7)
        c.drawString(5,4,"MS MARCO passages feed the embedding and lexical indexes; search fuses ranked results.")
    @staticmethod
    def _arrow(c,x1,y1,x2,y2):
        c.line(x1,y1,x2,y2); import math
        a=math.atan2(y2-y1,x2-x1); size=5
        for delta in (2.65,-2.65): c.line(x2,y2,x2+size*math.cos(a+delta),y2+size*math.sin(a+delta))


def ascii_text(value: str) -> str:
    value=value.replace("\u2011","-").replace("\u2013","-").replace("\u2014","-")
    return unicodedata.normalize("NFKD",value).encode("ascii","replace").decode("ascii")


def inline_markup(value: str) -> str:
    value=html.escape(ascii_text(value),quote=False)
    value=re.sub(r"\*\*(.+?)\*\*",r"<b>\1</b>",value)
    value=re.sub(r"`([^`]+)`",r'<font name="Courier" size="8">\1</font>',value)
    value=re.sub(r"\[([^\]]+)\]\(([^)]+)\)",r"\1",value)
    return value


def parse_markdown(text: str):
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle",parent=styles["Title"],fontName="Helvetica-Bold",fontSize=21,leading=26,textColor=colors.HexColor("#17324D"),alignment=TA_CENTER,spaceAfter=10))
    styles.add(ParagraphStyle(name="ReportH1",parent=styles["Heading1"],fontName="Helvetica-Bold",fontSize=15,leading=19,textColor=colors.HexColor("#17324D"),spaceBefore=12,spaceAfter=6,keepWithNext=True))
    styles.add(ParagraphStyle(name="ReportH2",parent=styles["Heading2"],fontName="Helvetica-Bold",fontSize=11,leading=14,textColor=colors.HexColor("#24577A"),spaceBefore=8,spaceAfter=4,keepWithNext=True))
    styles.add(ParagraphStyle(name="ReportBody",parent=styles["BodyText"],fontName="Helvetica",fontSize=8.7,leading=12.4,textColor=colors.HexColor("#273747"),spaceAfter=5))
    styles.add(ParagraphStyle(name="ReportSmall",parent=styles["BodyText"],fontName="Helvetica",fontSize=7.4,leading=9.4,textColor=colors.HexColor("#273747")))
    styles.add(ParagraphStyle(name="ReportCode",parent=styles["Code"],fontName="Courier",fontSize=7.1,leading=9,textColor=colors.HexColor("#26384A"),backColor=colors.HexColor("#F2F5F8"),borderPadding=5))
    lines=text.splitlines(); story=[]; i=0; first_heading=True
    while i<len(lines):
        line=lines[i].strip()
        if not line or line in {"---","***","___"}: i+=1; continue
        if line.startswith("```mermaid"):
            i+=1
            while i<len(lines) and not lines[i].strip().startswith("```"): i+=1
            story.extend([Architecture(),Spacer(1,5)]); i+=1; continue
        if line.startswith("```"):
            i+=1; block=[]
            while i<len(lines) and not lines[i].strip().startswith("```"): block.append(ascii_text(lines[i])); i+=1
            story.append(Preformatted("\n".join(block),styles["ReportCode"])); story.append(Spacer(1,5)); i+=1; continue
        if line.startswith("|"):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith("|"):
                cells=[cell.strip() for cell in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"\s*:?-{3,}:?\s*",cell) for cell in cells): rows.append([Paragraph(inline_markup(cell),styles["ReportSmall"]) for cell in cells])
                i+=1
            if rows:
                count=max(len(row) for row in rows); rows=[row+[Paragraph("",styles["ReportSmall"])]*(count-len(row)) for row in rows]
                widths=[(A4[0]-38*mm)/count]*count
                table=Table(rows,colWidths=widths,repeatRows=1,hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#DCE7F1")),("TEXTCOLOR",(0,0),(-1,0),colors.HexColor("#17324D")),
                    ("GRID",(0,0),(-1,-1),0.35,colors.HexColor("#B7C4D1")),("VALIGN",(0,0),(-1,-1),"TOP"),
                    ("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
                    ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
                    ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F6F8FA")]),
                ])); story.extend([table,Spacer(1,7)])
            continue
        heading=re.match(r"^(#{1,3})\s+(.*)$",line)
        if heading:
            level=len(heading.group(1)); title=inline_markup(heading.group(2))
            if level==1 and first_heading:
                story.append(Paragraph(title,styles["ReportTitle"])); first_heading=False
            elif level<=2: story.append(Paragraph(title,styles["ReportH1" if level==2 else "ReportH2"]))
            else: story.append(Paragraph(title,styles["ReportH2"]))
            i+=1; continue
        if line.startswith("- ") or re.match(r"^\d+\.\s",line):
            bullet="-" if line.startswith("- ") else re.match(r"^(\d+)\.",line).group(1)+"."
            content=line[2:] if line.startswith("- ") else re.sub(r"^\d+\.\s","",line)
            pstyle=ParagraphStyle("list",parent=styles["ReportBody"],leftIndent=12,firstLineIndent=-10,spaceAfter=2)
            story.append(Paragraph(inline_markup(content),pstyle,bulletText=bullet)); i+=1; continue
        paragraph=[line]; i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(("#","|","```","- ")) and not re.match(r"^\d+\.\s",lines[i].strip()):
            paragraph.append(lines[i].strip()); i+=1
        story.append(Paragraph(inline_markup(" ".join(paragraph)),styles["ReportBody"]))
    return story


def page_chrome(canvas, doc):
    canvas.saveState(); width,height=A4
    canvas.setStrokeColor(colors.HexColor("#D4DEE7")); canvas.line(19*mm,height-16*mm,width-19*mm,height-16*mm)
    canvas.setFont("Helvetica",7); canvas.setFillColor(colors.HexColor("#647588"))
    canvas.drawString(19*mm,height-12*mm,"ADROSONIC PHASE 2 | FINAL EVALUATION")
    canvas.drawRightString(width-19*mm,11*mm,f"2026-10-03 | Page {doc.page}")
    canvas.restoreState()


def main():
    text=SOURCE.read_text(encoding="utf-8")
    doc=SimpleDocTemplate(str(DEST),pagesize=A4,rightMargin=19*mm,leftMargin=19*mm,topMargin=22*mm,bottomMargin=18*mm,title="Adrosonic Phase 2 Final Evaluation Report",author="Adrosonic Phase 2 Evaluation")
    doc.build(parse_markdown(text),onFirstPage=page_chrome,onLaterPages=page_chrome)
    print(f"created={DEST} bytes={DEST.stat().st_size}")

if __name__=="__main__": main()
