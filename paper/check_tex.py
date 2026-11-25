"""Static checks on paper/fbimpact_ieee.tex (no LaTeX installation needed).

    python paper/check_tex.py

It cannot replace a real compile, but it catches the errors that most often break one: unbalanced braces,
mismatched environments, references or citations that point nowhere, and characters pdfLaTeX cannot read.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

tex = (Path(__file__).resolve().parent / "fbimpact_ieee.tex").read_text(encoding="utf-8")
problems: list[str] = []

body = re.sub(r"(?<!\\)%.*", "", tex)                       # drop comments
depth = 0
for n, line in enumerate(body.splitlines(), 1):
    for i, ch in enumerate(line):
        if ch in "{}" and (i == 0 or line[i - 1] != "\\"):
            depth += 1 if ch == "{" else -1
            if depth < 0:
                problems.append(f"line {n}: closing brace without an opening one")
                depth = 0
if depth:
    problems.append(f"{depth} unclosed brace(s) at end of file")

stack: list[str] = []
for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", body):
    if m.group(1) == "begin":
        stack.append(m.group(2))
    elif not stack or stack.pop() != m.group(2):
        problems.append(f"environment mismatch near \\end{{{m.group(2)}}}")
if stack:
    problems.append(f"unclosed environments: {stack}")

labels = set(re.findall(r"\\label\{([^}]+)\}", body))
for ref in re.findall(r"\\ref\{([^}]+)\}", body):
    if ref not in labels:
        problems.append(f"\\ref to undefined label {ref}")
bib = set(re.findall(r"\\bibitem\{([^}]+)\}", body))
cited = set()
for group in re.findall(r"\\cite\{([^}]+)\}", body):
    cited.update(k.strip() for k in group.split(","))
for key in sorted(cited - bib):
    problems.append(f"\\cite of undefined key {key}")
for key in sorted(bib - cited):
    problems.append(f"bibitem never cited: {key}")

for n, line in enumerate(body.splitlines(), 1):
    if line.count("~") > 8 and not line.lstrip().startswith("\\bibitem"):
        problems.append(f"line {n}: {line.count('~')} non-breaking spaces (ordinary spaces turned into ~?)")

bad = sorted({c for c in tex if ord(c) > 127})
if bad:
    problems.append(f"non-ASCII characters: {bad}")

for f in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", body):
    if not (Path(__file__).resolve().parent / f).exists():
        problems.append(f"missing figure file {f}")

# tabular rows: the number of & separators must match the column count of each table
for table in re.finditer(r"\\begin\{tabular\}\{((?:[^{}]|\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})*)\}(.*?)\\end\{tabular\}", body, re.S):
    spec, content = table.group(1), table.group(2)
    cols = len(re.findall(r"[lcrXS]|p\{[^}]*\}", re.sub(r"p\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "p{}", spec)))
    cols = len(re.findall(r"[lcr]|p\{\}", re.sub(r"p\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", "p{}", spec)))
    for row in re.split(r"\\\\", content):
        row = row.replace("\\hline", "").strip()
        if row and row.count("&") + 1 != cols:
            problems.append(f"table row has {row.count('&') + 1} cells, spec has {cols}: {row[:50]!r}")

print(f"{len(labels)} labels, {len(bib)} references, {len(cited)} cited keys, {len(re.findall(r'includegraphics', body))} figures")
if problems:
    print("PROBLEMS:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
print("static checks passed")
