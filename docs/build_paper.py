"""Build docs/paper.tex and docs/paper.pdf (two-column) from docs/PAPER.md, the single source of the paper.

    PATH=<env with pandoc and tectonic>:$PATH python docs/build_paper.py [--strict]

Needs pandoc and tectonic on PATH (e.g. `conda install -c conda-forge pandoc tectonic`).
The Markdown keeps figure and table captions as italic paragraphs under the figure or table;
this script turns them into LaTeX captions, then adapts pandoc's output to two columns:
- longtable (not allowed in two-column mode) becomes table or table* (wide tables span both columns),
- Figure 1 uses the single-column diagram (fig_pipeline_col) at the top of the right column,
- the other figures span both columns.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WIDE_TABLES = ("Controllers where", "Every hack")  # caption starts -> table*
COLUMN_FIGURES = {"figures/fig_pipeline.png": "figures/fig_pipeline_col.png",
                  "figures/fig_envelope.png": "figures/fig_envelope_col.png"}  # others -> figure*

md = (HERE / "PAPER.md").read_text()

# Title block: first H1 is the title; the next line is "**Author**, affiliation"; then the links line.
title = re.search(r"^# (.+)$", md, re.M).group(1)
author_line = re.search(r"^\*\*(.+?)\*\*, (.+)$", md, re.M)
author, affil = author_line.group(1), author_line.group(2)
links = re.search(r"^Project page: (.+)$", md, re.M).group(1)
body = md[md.index("## Abstract"):]

# Abstract into metadata, not a section.
abstract = re.search(r"## Abstract\n\n(.+?)\n\n## ", body, re.S).group(1)
body = body[body.index("## 1. Introduction"):]

# "![alt](path)\n\n*Figure N: caption*"  ->  "![caption](path)"
body = re.sub(r"!\[[^\]]*\]\(([^)]+)\)\n\n\*Figure \d+: (.+?)\*\n",
              lambda m: f"![{m.group(2)}]({m.group(1)})\n", body, flags=re.S)
# "*Table N: caption*" right after a table  ->  pandoc "Table: caption"
body = re.sub(r"\n\*Table \d+: (.+?)\*\n", lambda m: f"\nTable: {m.group(1)}\n", body, flags=re.S)
# Section numbers come from LaTeX.
body = re.sub(r"^## \d+\. ", "## ", body, flags=re.M)
body = re.sub(r"^### \d+\.\d+ ", "### ", body, flags=re.M)
for h in ("Acknowledgments", "References", "Appendix"):
    body = body.replace(f"## {h}\n", f"## {h} {{-}}\n")

meta = f"""---
title: "{title}"
author: |
  {author}\\
  {affil}\\
  {links}
abstract: |
  {abstract}
documentclass: article
classoption: [10pt, twocolumn]
geometry: [margin=0.75in, columnsep=0.25in]
numbersections: true
colorlinks: true
header-includes:
  - \\usepackage{{caption}}
  - \\captionsetup{{font=small}}
---

"""


def tables_to_tabular(tex: str) -> str:
    """Replace each pandoc longtable with a float: table (one column) or table* (both columns)."""
    out, pos = [], 0
    for m in re.finditer(r"\\begin\{longtable\}\[\]\{(.*?)\n\\caption\{", tex, re.S):
        start = m.start()
        end = tex.index("\\end{longtable}", start) + len("\\end{longtable}")
        block = tex[start:end]
        spec = m.group(1).rstrip()
        assert spec.endswith("}"), spec[-20:]
        spec = spec[:-1]
        cap_start = block.index("\\caption{") + len("\\caption{")
        cap_end = block.index("}\\tabularnewline", cap_start)
        caption = block[cap_start:cap_end]
        head = block[cap_end + len("}\\tabularnewline"):block.index("\\endfirsthead")]
        rows = block[block.index("\\endlastfoot") + len("\\endlastfoot"):block.index("\\end{longtable}")]
        env = "table*" if caption.lstrip("\\textbf{").startswith(WIDE_TABLES) else "table"
        out.append(tex[pos:start])
        out.append(f"\\begin{{{env}}}[t]\n\\centering\\small\n\\caption{{{caption}}}\n"
                   f"\\begin{{tabular}}{{{spec}}}{head}{rows}\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}")
        pos = end
    out.append(tex[pos:])
    return "".join(out)


def place_figures(tex: str) -> str:
    def fix(m):
        block = m.group(0)
        path = re.search(r"\]\{(figures/[^}]+)\}", block).group(1)
        if path in COLUMN_FIGURES:
            return block.replace(path, COLUMN_FIGURES[path]).replace("\\begin{figure}", "\\begin{figure}[t]", 1)
        return (block.replace("\\begin{figure}", "\\begin{figure*}[t]", 1)
                     .replace("\\end{figure}", "\\end{figure*}"))
    return re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", fix, tex, flags=re.S)


src = HERE / "_paper_build.md"
src.write_text(meta + body)
try:
    subprocess.run(["pandoc", str(src), "-s", "-o", str(HERE / "paper.tex"), "--resource-path", str(HERE),
                    "--shift-heading-level-by=-1", "--pdf-engine=xelatex",
                    "-V", "mainfont=Times New Roman", "-V", "mathfont=STIX Two Math"], check=True)
finally:
    src.unlink(missing_ok=True)

tex = (HERE / "paper.tex").read_text()
tex = place_figures(tables_to_tabular(tex))
assert "longtable}[]" not in tex.split("\\begin{document}")[1], "a longtable survived"
(HERE / "paper.tex").write_text(tex)
subprocess.run(["tectonic", "--keep-logs", "--outdir", str(HERE), str(HERE / "paper.tex")], check=True)
print("wrote", HERE / "paper.tex", "and", HERE / "paper.pdf")

if "--strict" in sys.argv:
    log = (HERE / "paper.log").read_text(errors="replace")
    problems = re.findall(r"Missing character: There is no (.) ", log)
    overfull = re.findall(r"Overfull \\hbox \((\d+\.\d+)pt too wide\)", log)
    big = [float(x) for x in overfull if float(x) > 10]
    if problems or big:
        sys.exit(f"missing glyphs: {sorted(set(problems))}; overfull boxes > 10pt: {big}")
