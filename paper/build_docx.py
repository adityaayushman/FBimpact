"""Build paper/fbimpact_ieee.docx from the same source as the LaTeX and PDF versions.

    python paper/build_docx.py

IEEE journal geometry: US Letter, two 3.5 in columns with a 0.25 in gutter, Times New Roman 10 pt.
Title and the full-width figures and tables sit in single-column sections joined to the two-column
body by continuous section breaks, so Word balances the columns around them.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_paper as bp  # noqa: E402
from paperlib import Doc  # noqa: E402

FONT = "Times New Roman"
COL_W, GUTTER, PAGE_W = 3.5, 0.25, 7.16


# ------------------------------------------------------------------------------------------ inline runs
class RunParser(HTMLParser):
    """Turn the HTML produced by `Doc.inline` into (text, style) runs."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.runs: list[tuple[str, dict]] = []
        self.stack: list[str] = []

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class", "")
        if tag in ("i", "b", "sub", "sup", "code"):
            self.stack.append(tag)
        elif tag == "span":
            self.stack.append({"m": "m", "rm": "rm", "cal": "cal"}.get(cls, "span"))
        elif tag == "a":
            self.stack.append("a")

    def handle_endtag(self, tag):
        if tag in ("i", "b", "sub", "sup", "code", "span", "a") and self.stack:
            self.stack.pop()

    def handle_data(self, data):
        st = self.stack
        italic = ("i" in st) or ("m" in st and "rm" not in st) or ("cal" in st and "rm" not in st)
        self.runs.append((data, {"italic": italic, "bold": "b" in st, "sub": "sub" in st, "sup": "sup" in st, "mono": "code" in st}))


def style_run(run, size: float, style: dict | None = None, bold=False, italic=False) -> None:
    run.font.name = FONT
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for att in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(att), FONT)
    run.font.size = Pt(size)
    style = style or {}
    run.font.bold = bold or style.get("bold", False)
    run.font.italic = italic or style.get("italic", False)
    if style.get("sub"):
        run.font.subscript = True
    if style.get("sup"):
        run.font.superscript = True
    if style.get("mono"):
        run.font.name = "Courier New"
        rfonts.set(qn("w:ascii"), "Courier New")
        rfonts.set(qn("w:hAnsi"), "Courier New")
        run.font.size = Pt(size * 0.88)


def add_runs(par, html_text: str, size: float, bold=False, italic=False) -> None:
    parser = RunParser()
    parser.feed(html_text)
    for text, st in parser.runs:
        if text:
            style_run(par.add_run(text), size, st, bold, italic)


# ------------------------------------------------------------------------------------------ page / styles
def set_cols(section, num: int) -> None:
    sect = section._sectPr
    cols = sect.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        sect.append(cols)
    for att in list(cols.attrib):
        del cols.attrib[att]
    cols.set(qn("w:num"), str(num))
    cols.set(qn("w:space"), str(int(GUTTER * 1440)))


def setup_page(section) -> None:
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin, section.bottom_margin = Inches(0.75), Inches(1.0)
    section.left_margin = section.right_margin = Inches(0.625)


def style_font(style, size: float, bold=False, italic=False, caps=False) -> None:
    style.font.name = FONT
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = italic
    style.font.all_caps = caps
    style.font.color.rgb = None
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    for att in list(rfonts.attrib):
        del rfonts.attrib[att]
    for att in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rfonts.set(qn(att), FONT)


def add_page_number(section) -> None:
    par = section.footer.paragraphs[0]
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = par.add_run()
    style_run(run, 8)
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._element.append(el)


# ------------------------------------------------------------------------------------------ paragraphs
def body_par(doc, html_text: str, size=10, first_indent=0.17, align=WD_ALIGN_PARAGRAPH.JUSTIFY, bold=False, space_after=0):
    par = doc.add_paragraph()
    par.alignment = align
    pf = par.paragraph_format
    pf.first_line_indent = Inches(first_indent)
    pf.space_after, pf.space_before = Pt(space_after), Pt(0)
    pf.line_spacing = 1.0
    pf.widow_control = True
    add_runs(par, html_text, size, bold=bold)
    return par


def heading(doc, text: str, level: int, label: str | None = None):
    style = "Heading 1" if level == 1 else "Heading 2"
    par = doc.add_paragraph(style=style)
    if level == 1:
        run = par.add_run((f"{label}. " if label else "") + text.upper())
        style_run(run, 10)
    else:
        run = par.add_run((f"{label}. " if label else "") + text)
        style_run(run, 10, italic=True)
    return par


def border(cell_or_tbl_pr, **sides) -> None:
    """Set borders on a table (tblPr) or cell (tcPr): sides like top=8 sets a single line of 8/8 pt."""
    tag = "w:tblBorders" if cell_or_tbl_pr.tag.endswith("tblPr") else "w:tcBorders"
    el = cell_or_tbl_pr.find(qn(tag))
    if el is None:
        el = OxmlElement(tag)
        cell_or_tbl_pr.append(el)
    for side, size in sides.items():
        b = OxmlElement(f"w:{side}")
        b.set(qn("w:val"), "single" if size else "nil")
        if size:
            b.set(qn("w:sz"), str(size))
            b.set(qn("w:color"), "000000")
        el.append(b)


def table_widths(header, rows, total: float) -> list[float]:
    cols = len(header)
    weights = []
    for i in range(cols):
        cells = [re.sub(r"<[^>]+>", "", header[i])] + [re.sub(r"<[^>]+>", "", r[i]) for r in rows]
        weights.append(min(max(sum(len(c) for c in cells) / len(cells), 6), 46))
    scale = total / sum(weights)
    return [w * scale for w in weights]


def add_table(doc, d, header, rows, total: float, caption_label: str, caption: str, H) -> None:
    for text, kw in ((caption_label, {}), (caption, {})):
        par = doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        par.paragraph_format.space_before, par.paragraph_format.space_after = Pt(4 if text == caption_label else 0), Pt(2)
        par.paragraph_format.keep_with_next = True
        if text == caption_label:
            style_run(par.add_run(text), 8)
        else:
            add_runs(par, H(text), 8)
    widths = table_widths(header, rows, total)
    tbl = doc.add_table(rows=1 + len(rows), cols=len(header))
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    tblpr = tbl._tbl.tblPr
    border(tblpr, top=8, bottom=8)
    mar = OxmlElement("w:tblCellMar")
    for side in ("left", "right"):
        m = OxmlElement(f"w:{side}")
        m.set(qn("w:w"), "60")
        m.set(qn("w:type"), "dxa")
        mar.append(m)
    tblpr.append(mar)
    for r, row in enumerate([header] + rows):
        for c, text in enumerate(row):
            cell = tbl.cell(r, c)
            cell.width = Inches(widths[c])
            par = cell.paragraphs[0]
            par.paragraph_format.space_after = par.paragraph_format.space_before = Pt(0)
            par.paragraph_format.line_spacing = 1.0
            par.paragraph_format.keep_with_next = r < len(rows)
            add_runs(par, H(text), 8, bold=(r == 0))
            if r == 0:
                border(cell._tc.get_or_add_tcPr(), bottom=4)
    for c, w in enumerate(widths):
        tbl.columns[c].width = Inches(w)
    gap = doc.add_paragraph()
    gap.paragraph_format.space_after = gap.paragraph_format.space_before = Pt(0)
    gap.paragraph_format.line_spacing = Pt(4)


def add_figure(doc, f, number: str, width: float, H) -> None:
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    par.paragraph_format.keep_with_next = True
    par.paragraph_format.space_before, par.paragraph_format.space_after = Pt(4), Pt(2)
    par.add_run().add_picture(str(HERE / "figures" / f"{f['file']}.png"), width=Inches(width))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    cap.paragraph_format.space_after = Pt(6)
    style_run(cap.add_run(f"Fig. {number}. "), 8)
    add_runs(cap, H(f["caption"]), 8)


def add_equation(doc, html_eq: str, number: str, width: float) -> None:
    par = doc.add_paragraph()
    pf = par.paragraph_format
    pf.space_before, pf.space_after = Pt(4), Pt(4)
    pf.tab_stops.add_tab_stop(Inches(width / 2), WD_TAB_ALIGNMENT.CENTER)
    pf.tab_stops.add_tab_stop(Inches(width), WD_TAB_ALIGNMENT.RIGHT)
    style_run(par.add_run("\t"), 10)
    add_runs(par, html_eq, 10)
    style_run(par.add_run(f"\t({number})"), 10)


ALGO_WORDS = {"for": ("for", "do"), "while": ("while", "do"), "if": ("if", "then"), "elsif": ("else if", "then")}


def add_algorithm(doc, a: dict, number: str, H, width: float = COL_W) -> None:
    lines = a["lines"]
    tbl = doc.add_table(rows=1 + len(lines), cols=2)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    border(tbl._tbl.tblPr, top=8, bottom=8)
    widths = (0.3, width - 0.3)
    head = tbl.cell(0, 0).merge(tbl.cell(0, 1))
    hp = head.paragraphs[0]
    hp.paragraph_format.space_after = Pt(1)
    hp.paragraph_format.keep_with_next = True
    style_run(hp.add_run(f"Algorithm {number} "), 8, bold=True)
    add_runs(hp, H(a["caption"]), 8)
    border(head._tc.get_or_add_tcPr(), bottom=4)
    depth, n = 0, 0
    for i, (kind, text) in enumerate(lines, start=1):
        if kind in ("endfor", "endwhile", "endif", "else", "elsif"):
            depth = max(depth - 1, 0)
        comment = ""
        if " // " in text:
            text, comment = text.split(" // ", 1)
        numbered = kind not in ("req", "ens")
        if numbered:
            n += 1
        num_cell, text_cell = tbl.cell(i, 0), tbl.cell(i, 1)
        num_cell.width, text_cell.width = Inches(widths[0]), Inches(widths[1])
        np_, tp = num_cell.paragraphs[0], text_cell.paragraphs[0]
        for p in (np_, tp):
            p.paragraph_format.space_after = p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.keep_with_next = i < len(lines)
        np_.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        if numbered:
            style_run(np_.add_run(str(n)), 8)
        tp.paragraph_format.left_indent = Inches(0.1 + 0.11 * depth)
        if kind == "req":
            style_run(tp.add_run("Require: "), 8, bold=True)
            add_runs(tp, H(text), 8)
        elif kind == "ens":
            style_run(tp.add_run("Ensure: "), 8, bold=True)
            add_runs(tp, H(text), 8)
        elif kind == "return":
            style_run(tp.add_run("return "), 8, bold=True)
            add_runs(tp, H(text), 8)
        elif kind in ALGO_WORDS:
            style_run(tp.add_run(ALGO_WORDS[kind][0] + " "), 8, bold=True)
            add_runs(tp, H(text), 8)
            style_run(tp.add_run(" " + ALGO_WORDS[kind][1]), 8, bold=True)
        elif kind == "else":
            style_run(tp.add_run("else"), 8, bold=True)
        elif kind.startswith("end"):
            style_run(tp.add_run("end " + kind[3:]), 8, bold=True)
        else:
            add_runs(tp, H(text), 8)
        if comment:
            style_run(tp.add_run("   // "), 8, italic=True)
            add_runs(tp, H(comment), 8, italic=True)
        if kind in ("for", "while", "if", "else", "elsif"):
            depth += 1
    for c, w in enumerate(widths):
        tbl.columns[c].width = Inches(w)
    gap = doc.add_paragraph()
    gap.paragraph_format.space_after = gap.paragraph_format.space_before = Pt(0)
    gap.paragraph_format.line_spacing = Pt(4)


# ------------------------------------------------------------------------------------------ main build
def build(out: Path) -> None:
    d = Doc(bp.META, bp.BLOCKS, bp.REFS)
    H = lambda s: d.inline(s, "html")  # noqa: E731
    doc = Document()
    style_font(doc.styles["Normal"], 10)
    style_font(doc.styles["Heading 1"], 10, caps=False)
    style_font(doc.styles["Heading 2"], 10, italic=True)
    for name, align, before, after in (("Heading 1", WD_ALIGN_PARAGRAPH.CENTER, 10, 4), ("Heading 2", WD_ALIGN_PARAGRAPH.LEFT, 6, 2)):
        pf = doc.styles[name].paragraph_format
        pf.alignment, pf.space_before, pf.space_after, pf.keep_with_next = align, Pt(before), Pt(after), True
    now = datetime.now()
    doc.core_properties.author = "Aditya Ayushman Sahoo"
    doc.core_properties.title = bp.META["title_plain"]
    doc.core_properties.last_modified_by = "Aditya Ayushman Sahoo"
    doc.core_properties.comments = ""
    doc.core_properties.created = now
    doc.core_properties.modified = now
    doc.core_properties.revision = 1

    first = doc.sections[0]
    setup_page(first)
    set_cols(first, 1)
    add_page_number(first)
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.paragraph_format.space_after = Pt(10)
    style_run(t.add_run(bp.META["title_plain"]), 24)
    authors = [
        ("Aditya Ayushman Sahoo",
         ["Department of Computational Intelligence", "School of Computing,"],
         ["SRM Institute of Science and Technology,", "Kattankulathur – 603203", "adityaasahoo@gmail.com"]),
        ("Kothai G",
         ["Department of Computational Intelligence", "School of Computing,", "Faculty of Engineering and Technology,"],
         ["SRM Institute of Science and Technology,", "Kattankulathur – 603203"]),
    ]
    atbl = doc.add_table(rows=1, cols=len(authors))
    atbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    atbl.autofit = False
    acol_w = PAGE_W / len(authors)
    for c, (name, pre_lines, post_lines) in enumerate(authors):
        cell = atbl.cell(0, c)
        cell.width = Inches(acol_w)
        first = True
        for text, size, space_after in (
            [(name, 11, 2)] + [(t, 10, 0) for t in pre_lines] + [("", 4, 0)] + [(t, 10, 0) for t in post_lines]
        ):
            par = cell.paragraphs[0] if first else cell.add_paragraph()
            first = False
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            par.paragraph_format.space_after, par.paragraph_format.space_before = Pt(space_after), Pt(0)
            par.paragraph_format.line_spacing = 1.0
            if text:
                style_run(par.add_run(text), size)
    for c in range(len(authors)):
        atbl.columns[c].width = Inches(acol_w)
    gap = doc.add_paragraph()
    gap.paragraph_format.space_after = gap.paragraph_format.space_before = Pt(10)

    def to_columns(n: int) -> None:
        doc.add_section(WD_SECTION.CONTINUOUS)
        set_cols(doc.sections[-1], n)

    to_columns(2)
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    par.paragraph_format.space_after = Pt(4)
    style_run(par.add_run("Abstract"), 9, bold=True, italic=True)
    style_run(par.add_run("—"), 9, bold=True)
    add_runs(par, H(bp.META["abstract"]), 9, bold=True)
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    par.paragraph_format.space_after = Pt(4)
    style_run(par.add_run("Index Terms"), 9, bold=True, italic=True)
    style_run(par.add_run("—"), 9, bold=True)
    add_runs(par, H(bp.META["keywords"]), 9, bold=True)

    for b in bp.BLOCKS:
        kind = b[0]
        if kind == "sec":
            heading(doc, b[1], 1, d.labels[b[2]])
        elif kind == "sub":
            heading(doc, b[1], 2, d.labels[b[2]].split("-")[-1])
        elif kind == "p":
            body_par(doc, H(b[1]))
        elif kind == "enum":
            counter = 0
            for item in b[1]:
                par = doc.add_paragraph()
                par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                pf = par.paragraph_format
                pf.left_indent, pf.first_line_indent = Inches(0.25), Inches(-0.25)
                pf.space_after, pf.line_spacing = Pt(1), 1.0
                if item.startswith("<b>RQ"):
                    pf.left_indent, pf.first_line_indent = Inches(0.0), Inches(0.0)
                else:
                    counter += 1
                    style_run(par.add_run(f"{counter})\t"), 10)
                    pf.tab_stops.add_tab_stop(Inches(0.25))
                add_runs(par, H(item), 10)
        elif kind == "eq":
            add_equation(doc, b[3], d.labels[b[1]], COL_W)
        elif kind == "algo":
            to_columns(1)
            add_algorithm(doc, b[1], d.labels[b[1]["label"]], H, PAGE_W)
            to_columns(2)
        elif kind == "fig":
            f = b[1]
            if f.get("wide"):
                to_columns(1)
                add_figure(doc, f, d.labels[f["label"]], PAGE_W, H)
                to_columns(2)
            else:
                add_figure(doc, f, d.labels[f["label"]], COL_W, H)
        elif kind == "table":
            t_ = b[1]
            label = f"TABLE {d.labels[t_['label']]}"
            if t_.get("wide"):
                to_columns(1)
                add_table(doc, d, t_["header"], t_["rows"], PAGE_W, label, t_["caption"], H)
                to_columns(2)
            else:
                add_table(doc, d, t_["header"], t_["rows"], COL_W, label, t_["caption"], H)

    if bp.META.get("acknowledgment"):
        heading(doc, "Acknowledgment", 1)
        body_par(doc, H(bp.META["acknowledgment"]))
    heading(doc, "References", 1)
    for i, key in enumerate(d.cite_order, 1):
        par = doc.add_paragraph()
        par.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf = par.paragraph_format
        pf.left_indent, pf.first_line_indent = Inches(0.3), Inches(-0.3)
        pf.space_after, pf.line_spacing = Pt(2), 1.0
        pf.tab_stops.add_tab_stop(Inches(0.3))
        style_run(par.add_run(f"[{i}]\t"), 8)
        add_runs(par, H(bp.REFS[key]), 8)
    fix_ooxml(doc)
    doc.save(out)
    fix_app_properties(out)


TBLPR_ORDER = ["tblStyle", "tblpPr", "tblOverlap", "bidiVisual", "tblStyleRowBandSize", "tblStyleColBandSize", "tblW", "jc", "tblCellSpacing",
               "tblInd", "tblBorders", "shd", "tblLayout", "tblCellMar", "tblLook", "tblCaption", "tblDescription"]


def fix_app_properties(path: Path) -> None:
    """python-docx's default template leaves docProps/app.xml claiming <Application>Microsoft Macintosh Word</Application>;
    replace it with a plain, accurate value so the saved file doesn't carry that tell."""
    import zipfile

    with zipfile.ZipFile(path) as z:
        items = {i.filename: z.read(i.filename) for i in z.infolist()}
    app = items["docProps/app.xml"].decode("utf-8").replace("Microsoft Macintosh Word", "Microsoft Office Word")
    items["docProps/app.xml"] = app.encode("utf-8")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in items.items():
            z.writestr(name, data)


def fix_ooxml(doc) -> None:
    """Put table properties in schema order and give the zoom setting the attribute the schema requires."""
    for tbl in doc.tables:
        pr = tbl._tbl.tblPr
        children = list(pr)
        children.sort(key=lambda el: TBLPR_ORDER.index(el.tag.split("}")[1]) if el.tag.split("}")[1] in TBLPR_ORDER else len(TBLPR_ORDER))
        for el in children:
            pr.remove(el)
        for el in children:
            pr.append(el)
    zoom = doc.settings.element.find(qn("w:zoom"))
    if zoom is not None and zoom.get(qn("w:percent")) is None:
        zoom.set(qn("w:percent"), "100")


if __name__ == "__main__":
    target = HERE / "fbimpact_ieee.docx"
    build(target)
    print("wrote", target)
