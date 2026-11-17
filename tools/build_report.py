"""Сборка отчётов: labN/report.md  ->  reports/ЛР<N>_<Фамилия>.docx и .pdf

Шаблон отчёта – Markdown (pandoc) с подстановками:
  {{meta.KEY}}                – поле из report_meta.json (ФИО, группа, ...);
  {{res.a.b.c}} / {{res.a:.3g}} – число/строка из labN/results.json (результаты расчёта);
  {{code:путь/к/файлу.py:имя}} – исходный текст функции/класса «имя» из программы
                                (так выдержки кода в отчёте всегда совпадают с программой).

Запуск:  python tools/build_report.py 0 1 2 3 4     (без аргументов – все отчёты)
Требуются pandoc и LibreOffice (для PDF).
"""
import ast
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
OUT = os.path.join(ROOT, "reports")


def load_meta():
    with open(os.path.join(ROOT, "report_meta.json"), encoding="utf-8") as f:
        return json.load(f)


def get_code(path, name):
    src = open(os.path.join(ROOT, path), encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            seg = ast.get_source_segment(src, node)
            return f"```python\n{seg}\n```"
    raise KeyError(f"{name} не найдено в {path}")


def lookup(d, dotted):
    """Поиск по пути a.b.c; ключи с точкой (например «80.0») склеиваются жадно."""
    parts = dotted.split(".")
    cur = d
    i = 0
    while i < len(parts):
        if isinstance(cur, list):
            cur = cur[int(parts[i])]
            i += 1
            continue
        for j in range(len(parts), i, -1):
            key = ".".join(parts[i:j])
            if key in cur:
                cur = cur[key]
                i = j
                break
        else:
            raise KeyError(dotted)
    return cur


def substitute(text, meta, res):
    text = re.sub(r"\{\{code:([^:}]+):(\w+)\}\}", lambda m: get_code(m.group(1), m.group(2)), text)

    def rep(m):
        kind, key, fmt = m.group(1), m.group(2), m.group(3)
        val = lookup(meta if kind == "meta" else res, key)
        if fmt:
            val = format(val, fmt)
        return str(val)

    return re.sub(r"\{\{(meta|res)\.([\w.]+)(?::([^}]+))?\}\}", rep, text)


def title_page(meta, lab_no, lab_title):
    pb = '```{=openxml}\n<w:p><w:r><w:br w:type="page"/></w:r></w:p>\n```'
    group = meta.get("group") or "__________"
    student = meta["student"]
    return f"""::: {{custom-style="Center"}}
{meta['ministry']}

{meta['university']}

{meta['faculty']}

{meta['department']}
:::

&nbsp;

&nbsp;

&nbsp;

::: {{custom-style="CenterBig"}}
ОТЧЁТ

по лабораторной работе № {lab_no}

«{lab_title}»
:::

::: {{custom-style="Center"}}
по дисциплине «{meta['discipline']}»
:::

&nbsp;

&nbsp;

&nbsp;

::: {{custom-style="Right"}}
Выполнил: студент группы {group}

{student}

&nbsp;

Проверил: {meta['teacher']}
:::

&nbsp;

&nbsp;

&nbsp;

::: {{custom-style="Center"}}
{meta['city']} {meta['year']}
:::

{pb}
"""


def make_reference_docx(path):
    """Стили: Times New Roman 12 pt, поля 2 см, стили Center/CenterBig/Right для титула."""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.style import WD_STYLE_TYPE
    from docx.shared import Pt, Cm
    from docx.oxml.ns import qn

    raw = subprocess.run(["pandoc", "--print-default-data-file", "reference.docx"],
                         capture_output=True, check=True).stdout
    open(path, "wb").write(raw)
    doc = Document(path)

    def set_font(style, name="Times New Roman", size=12, bold=None, color_black=True):
        style.font.name = name
        style.font.size = Pt(size)
        if bold is not None:
            style.font.bold = bold
        if color_black:
            from docx.shared import RGBColor
            style.font.color.rgb = RGBColor(0, 0, 0)
        rpr = style.element.get_or_add_rPr()
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = rpr.makeelement(qn("w:rFonts"), {})
            rpr.append(rfonts)
        for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rfonts.set(qn(a), name)
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
            rfonts.attrib.pop(qn(a), None)

    st0 = doc.styles
    st = {s.name: s for s in st0}
    for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
        if name in st:
            set_font(st[name], size=12)
            pf = st[name].paragraph_format
            pf.space_after = Pt(4)
            pf.line_spacing = 1.15
            if name in ("Body Text", "First Paragraph"):
                pf.first_line_indent = Cm(1.0)
                pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for name, size in (("Title", 16), ("Heading 1", 14), ("Heading 2", 13), ("Heading 3", 12)):
        if name in st:
            set_font(st[name], size=size, bold=True)
    for name in ("Source Code", "Verbatim Char"):
        if name in st:
            set_font(st[name], name="Courier New", size=9, color_black=False)
    if "Image Caption" in st:
        set_font(st["Image Caption"], size=11)
        st["Image Caption"].font.italic = False
        st["Image Caption"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    if "Table Caption" in st:
        set_font(st["Table Caption"], size=11)
        st["Table Caption"].font.italic = False

    for nm, align, size, bold in (("Center", WD_ALIGN_PARAGRAPH.CENTER, 12, False),
                                  ("CenterBig", WD_ALIGN_PARAGRAPH.CENTER, 16, True),
                                  ("Right", WD_ALIGN_PARAGRAPH.RIGHT, 12, False)):
        s = st0.add_style(nm, WD_STYLE_TYPE.PARAGRAPH)
        s.base_style = st["Normal"]
        set_font(s, size=size, bold=bold)
        s.paragraph_format.alignment = align
        s.paragraph_format.first_line_indent = Cm(0)
        s.paragraph_format.space_after = Pt(2)

    for sec in doc.sections:
        sec.left_margin = Cm(2.5)
        sec.right_margin = Cm(1.5)
        sec.top_margin = Cm(2)
        sec.bottom_margin = Cm(2)
    doc.save(path)


def build(lab_no, meta, ref):
    lab_dir = os.path.join(ROOT, f"lab{lab_no}")
    md = open(os.path.join(lab_dir, "report.md"), encoding="utf-8").read()
    res_path = os.path.join(lab_dir, "results.json")
    res = json.load(open(res_path, encoding="utf-8")) if os.path.exists(res_path) else {}
    m = re.match(r"<!--\s*title:\s*(.+?)\s*-->", md)
    lab_title = m.group(1) if m else ""
    body = substitute(md, meta, res)
    full = title_page(meta, lab_no, lab_title) + "\n" + body
    os.makedirs(BUILD, exist_ok=True)
    md_out = os.path.join(BUILD, f"lab{lab_no}.md")
    open(md_out, "w", encoding="utf-8").write(full)

    surname = meta["student"].split()[0]
    docx = os.path.join(OUT, f"ЛР{lab_no}_{surname}.docx")
    subprocess.run(["pandoc", md_out, "-f", "markdown+tex_math_dollars+pipe_tables+implicit_figures",
                    "-o", docx, f"--reference-doc={ref}", f"--resource-path={lab_dir}",
                    "--highlight-style=tango"], check=True)
    print("  ->", os.path.relpath(docx, ROOT))
    if shutil.which("soffice"):
        subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", OUT, docx],
                       check=True, capture_output=True, timeout=600)
        print("  ->", os.path.relpath(docx[:-5] + ".pdf", ROOT))


def main():
    labs = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3, 4]
    meta = load_meta()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(BUILD, exist_ok=True)
    ref = os.path.join(BUILD, "reference.docx")
    make_reference_docx(ref)
    for n in labs:
        print(f"Отчёт по ЛР{n}")
        build(n, meta, ref)


if __name__ == "__main__":
    main()
