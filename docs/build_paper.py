"""Build docs/paper.tex and docs/paper.pdf from docs/PAPER.md (the single source of the paper).

    PATH=<env with pandoc and tectonic>:$PATH python docs/build_paper.py

Needs pandoc and tectonic on PATH (e.g. `conda install -c conda-forge pandoc tectonic`).
The Markdown keeps figure and table captions as italic paragraphs under the figure or table;
this script turns them into real LaTeX captions.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
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
# Section titles: drop the roman numerals (LaTeX numbers sections itself).
body = re.sub(r"^## \d+\. ", "## ", body, flags=re.M)
body = re.sub(r"^### \d+\.\d+ ", "### ", body, flags=re.M)
# The checklist and appendix go unnumbered.
for h in ("Acknowledgments", "References", "Appendix", "Paper checklist"):
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
classoption: [10pt]
geometry: margin=0.9in
numbersections: true
colorlinks: true
header-includes:
  - \\usepackage{{float}}
  - \\floatplacement{{figure}}{{tbp}}
  - \\usepackage{{caption}}
  - \\captionsetup{{font=small}}
---

"""
src = HERE / "_paper_build.md"
src.write_text(meta + body)
try:
    subprocess.run(["pandoc", str(src), "-s", "-o", str(HERE / "paper.tex"), "--resource-path", str(HERE), "--shift-heading-level-by=-1",
                    "--pdf-engine=xelatex", "-V", "mainfont=Times New Roman", "-V", "mathfont=STIX Two Math"],
                   check=True)
    subprocess.run(["tectonic", "--keep-logs", "--outdir", str(HERE), str(HERE / "paper.tex")], check=True)
finally:
    src.unlink(missing_ok=True)
print("wrote", HERE / "paper.tex", "and", HERE / "paper.pdf")
if "--strict" in sys.argv:
    log = (HERE / "paper.log").read_text(errors="replace")
    missing = re.findall(r"Missing character: There is no (.) ", log)
    if missing:
        sys.exit(f"missing glyphs: {sorted(set(missing))}")
