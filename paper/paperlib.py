"""A small renderer: one source of paper content, two outputs (IEEEtran LaTeX and an IEEE-style HTML/PDF).

Inline markup in the content: <i> <b> <sub> <sup> <code> <m>latex</m>, citations [@key,key], and
cross-references [#fig:x] [#tab:x] [#eq:x] [#sec:x]. Cite keys are numbered in order of first citation, and the
bibliography is emitted in that order, as IEEE requires.
"""

from __future__ import annotations

import html
import re

UNICODE_LATEX = {
    "τ": r"$\tau$", "λ": r"$\lambda$", "Δ": r"$\Delta$", "×": r"$\times$", "±": r"$\pm$", "−": r"$-$", "≥": r"$\geq$",
    "≤": r"$\leq$", "→": r"$\rightarrow$", "·": r"$\cdot$", "≈": r"$\approx$", "–": "--", "—": "---", "’": "'",
    "‘": "`", "“": "``", "”": "''", "é": r"\'e", "í": r"\'{\i}", "ñ": r"\~n", "ú": r"\'u", "ö": r'\"o', "ü": r'\"u', "ó": r"\'o", "ã": r"\~a", "ç": r"\c{c}", "º": r"$^{\circ}$",
    "σ": r"$\sigma$", "×": r"$\times$", " ": "~", "…": r"\ldots{}", "α": r"$\alpha$", "∈": r"$\in$",
}
LATEX_SPECIAL = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_", "$": r"\$", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}
MATH_HTML = {r"\delta": "δ", r"\mathbb{R}": "ℝ", r"\kappa": "κ", r"\theta": "θ", r"\dots": "…", r"\leftarrow": "←", r"\emptyset": "∅", r"\wedge": "∧",
             r"\big": "", r"\!": "", r"\quad": "&emsp;", r"\rho": "ρ", r"\pi": "π", r"\ell": "ℓ", r"\{": "{", r"\}": "}", r"\tau": "τ", r"\lambda": "λ", r"\Delta": "Δ", r"\times": "×", r"\pm": "±", r"\geq": "≥", r"\ge": "≥", r"\leq": "≤",
             r"\le": "≤", r"\rightarrow": "→", r"\cdot": "·", r"\approx": "≈", r"\sigma": "σ", r"\in": "∈", r"\sum": "Σ", r"\max": "max",
             r"\min": "min", r"\exp": "exp", r"\log": "log", r"\,": "&thinsp;", r"\;": "&ensp;", r"\ ": " "}
TOKEN = re.compile(r"(\[@[^\]]+\]|\[#[^\]]+\]|</?(?:i|b|sub|sup|m|code)>)")


def roman(n: int) -> str:
    out = ""
    for value, sym in ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= value:
            out += sym
            n -= value
    return out


def latex_text(s: str) -> str:
    out = []
    for ch in s:
        if ch in LATEX_SPECIAL:
            out.append(LATEX_SPECIAL[ch])
        elif ch in UNICODE_LATEX:
            out.append(UNICODE_LATEX[ch])
        elif ord(ch) < 128:
            out.append(ch)
        else:
            raise ValueError(f"unmapped character {ch!r} (U+{ord(ch):04X}) in: {s[:60]}")
    return "".join(out)


def math_html(s: str) -> str:
    s = re.sub(r"\\dot(?![A-Za-z])\s*(\\[A-Za-z]+|[A-Za-z])", lambda m: m.group(1) + "\u0307", s)
    s = re.sub(r"\\bar\s*([A-Za-z])", lambda m: m.group(1) + "\u0304", s)
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", lambda m: f'<span class="rm">{html.escape(m.group(1))}</span>', s)
    s = re.sub(r"\\text\{([^}]*)\}", lambda m: f'<span class="rm">{html.escape(m.group(1))}</span>', s)
    s = re.sub(r"\\mathcal\{([^}]*)\}", lambda m: f'<span class="cal">{html.escape(m.group(1))}</span>', s)
    for k in sorted(MATH_HTML, key=len, reverse=True):
        s = s.replace(k, MATH_HTML[k])
    s = re.sub(r"_\{([^{}]*)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"\^\{([^{}]*)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"_([A-Za-z0-9])", r"<sub>\1</sub>", s)
    s = re.sub(r"\^([A-Za-z0-9*])", r"<sup>\1</sup>", s)
    return f'<span class="m">{s}</span>'


class Doc:
    def __init__(self, meta: dict, blocks: list, refs: dict[str, str]):
        self.meta, self.blocks, self.refs = meta, blocks, refs
        self.cite_order: list[str] = []
        self.labels: dict[str, str] = {}
        self._number()

    # -------------------------------------------------------------------------- numbering
    def _scan(self, text: str) -> None:
        for m in re.finditer(r"\[@([^\]]+)\]", text):
            for key in (k.strip() for k in m.group(1).split(",")):
                if key not in self.refs:
                    raise KeyError(f"unknown citation key {key}")
                if key not in self.cite_order:
                    self.cite_order.append(key)

    def _number(self) -> None:
        sec = sub = fig = tab = eq = alg = 0
        texts = [self.meta.get("abstract", "")]
        for b in self.blocks:
            kind = b[0]
            if kind == "sec":
                sec, sub = sec + 1, 0
                self.labels[b[2]] = roman(sec) if len(b) > 2 and b[2] else ""
                texts.append(b[1])
            elif kind == "sub":
                sub += 1
                if len(b) > 2 and b[2]:
                    self.labels[b[2]] = f"{roman(sec)}-{chr(64 + sub)}"
                texts.append(b[1])
            elif kind == "p":
                texts.append(b[1])
            elif kind == "enum":
                texts.extend(b[1])
            elif kind == "eq":
                eq += 1
                self.labels[b[1]] = str(eq)
            elif kind == "fig":
                fig += 1
                self.labels[b[1]["label"]] = str(fig)
                texts.append(b[1]["caption"])
            elif kind == "algo":
                alg += 1
                self.labels[b[1]["label"]] = str(alg)
                texts.append(b[1]["caption"])
                texts.extend(line[1] for line in b[1]["lines"])
            elif kind == "table":
                tab += 1
                self.labels[b[1]["label"]] = roman(tab)
                texts.append(b[1]["caption"])
                texts.extend(c for row in [b[1]["header"]] + b[1]["rows"] for c in row)
        for t in texts:
            self._scan(t)
        unused = [k for k in self.refs if k not in self.cite_order]
        if unused:
            raise ValueError(f"references never cited: {unused}")

    def num(self, key: str) -> int:
        return self.cite_order.index(key) + 1

    # -------------------------------------------------------------------------- inline
    def inline(self, text: str, target: str) -> str:
        out, in_math, in_code = [], False, False
        for tok in TOKEN.split(text):
            if not tok:
                continue
            if tok.startswith("[@"):
                keys = [k.strip() for k in tok[2:-1].split(",")]
                if target == "tex":
                    out.append("~\\cite{" + ",".join(keys) + "}")
                else:
                    nums = sorted(self.num(k) for k in keys)
                    runs, i = [], 0
                    while i < len(nums):
                        j = i
                        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                            j += 1
                        runs.append((nums[i], nums[j]))
                        i = j + 1
                    parts = []
                    for a, b in runs:
                        if b - a >= 2:
                            parts.append(f'<a href="#ref{a}">[{a}]</a>&ndash;<a href="#ref{b}">[{b}]</a>')
                        else:
                            parts.extend(f'<a href="#ref{n}">[{n}]</a>' for n in range(a, b + 1))
                    out.append("&nbsp;" + ", ".join(parts))
            elif tok.startswith("[#"):
                kind, name = tok[2:-1].split(":", 1)
                label = f"{kind}:{name}"
                if label not in self.labels:
                    raise KeyError(f"unknown reference {label}")
                n = self.labels[label]
                if target == "tex":
                    out.append({"fig": "Fig.~", "tab": "Table~", "eq": "(", "sec": "", "alg": "Algorithm~"}[kind] + "\\ref{" + label + "}" + (")" if kind == "eq" else ""))
                else:
                    word = {"fig": "Fig.&nbsp;", "tab": "Table&nbsp;", "eq": "(", "sec": "", "alg": "Algorithm&nbsp;"}[kind]
                    out.append(f'<a href="#{label.replace(":", "-")}">{word}{n}{")" if kind == "eq" else ""}</a>')
            elif tok in ("<i>", "</i>", "<b>", "</b>", "<sub>", "</sub>", "<sup>", "</sup>"):
                if target == "tex":
                    mapping = {"<i>": r"\textit{", "<b>": r"\textbf{", "<sub>": r"\textsubscript{", "<sup>": r"\textsuperscript{"}
                    out.append(mapping.get(tok, "}"))
                else:
                    out.append(tok)
            elif tok == "<m>":
                in_math = True
                out.append("$" if target == "tex" else "")
            elif tok == "</m>":
                in_math = False
                out.append("$" if target == "tex" else "")
            elif tok == "<code>":
                in_code = True
                out.append(r"\texttt{" if target == "tex" else "<code>")
            elif tok == "</code>":
                in_code = False
                out.append("}" if target == "tex" else "</code>")
            else:
                if in_math:
                    out.append(tok if target == "tex" else math_html(tok))
                elif target == "tex":
                    out.append(latex_text(tok))
                else:
                    out.append(html.escape(tok, quote=False))
        return "".join(out)

    # -------------------------------------------------------------------------- LaTeX
    def tex(self) -> str:
        m = self.meta
        T = lambda s: self.inline(s, "tex")  # noqa: E731
        o: list[str] = []
        o.append(r"""% IEEE journal manuscript. Compile with pdflatex (Overleaf: select "IEEEtran" class, compiler pdfLaTeX).
% Run pdflatex twice so the cross-references resolve. Figures are in ./figures/.
\documentclass[journal]{IEEEtran}
\usepackage[T1]{fontenc}
\usepackage{cite}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{array}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{textcomp}
\usepackage[hidelinks]{hyperref}
\hyphenation{op-tical net-works semi-conduc-tor}
\begin{document}
""")
        o.append("\\title{" + T(m["title"]) + "}\n")
        o.append("\\author{" + m["authors_tex"] + "}\n")
        o.append("\\markboth{" + m["running_head"] + "}{" + m["running_head"] + "}\n\\maketitle\n")
        o.append("\\begin{abstract}\n" + T(m["abstract"]) + "\n\\end{abstract}\n")
        o.append("\\begin{IEEEkeywords}\n" + T(m["keywords"]) + "\n\\end{IEEEkeywords}\n\\IEEEpeerreviewmaketitle\n")
        for b in self.blocks:
            k = b[0]
            if k == "sec":
                o.append("\n\\section{" + T(b[1]) + "}" + (f"\\label{{{b[2]}}}" if b[2] else ""))
            elif k == "sub":
                o.append("\n\\subsection{" + T(b[1]) + "}" + (f"\\label{{{b[2]}}}" if len(b) > 2 and b[2] else ""))
            elif k == "p":
                o.append("\n" + T(b[1]) + "\n")
            elif k == "enum":
                o.append("\\begin{enumerate}\n" + "\n".join("\\item " + T(i) for i in b[1]) + "\n\\end{enumerate}\n")
            elif k == "eq":
                o.append("\\begin{equation}\n\\label{" + b[1] + "}\n" + b[2] + "\n\\end{equation}\n")
            elif k == "fig":
                f = b[1]
                star = "*" if f.get("wide") else ""
                width = "\\textwidth" if f.get("wide") else "\\columnwidth"
                o.append(f"\\begin{{figure{star}}}[!t]\n\\centering\n\\includegraphics[width={width}]{{figures/{f['file']}.pdf}}\n"
                         f"\\caption{{{T(f['caption'])}}}\n\\label{{{f['label']}}}\n\\end{{figure{star}}}\n")
            elif k == "algo":
                a = b[1]
                o.append("\\begin{algorithm*}[!t]\n\\caption{" + T(a["caption"]) + "}\n\\label{" + a["label"] + "}\n\\begin{algorithmic}[1]\n"
                         + "\n".join(self._algo_tex_line(kind, text) for kind, text in a["lines"]) + "\n\\end{algorithmic}\n\\end{algorithm*}\n")
            elif k == "table":
                t = b[1]
                star = "*" if t.get("wide") else ""
                header = " & ".join(T(c) for c in t["header"]) + " \\\\\n\\hline\n"
                rows = " \\\\\n".join(" & ".join(T(c) for c in r) for r in t["rows"]) + " \\\\\n"
                o.append(f"\\begin{{table{star}}}[!t]\n\\renewcommand{{\\arraystretch}}{{1.15}}\n\\caption{{{T(t['caption'])}}}\n\\label{{{t['label']}}}\n\\centering\n\\footnotesize\n"
                         f"\\begin{{tabular}}{{{t['spec']}}}\n\\hline\n{header}{rows}\\hline\n\\end{{tabular}}\n\\end{{table{star}}}\n")
        if m.get("acknowledgment"):
            o.append("\n\\section*{Acknowledgment}\n" + T(m["acknowledgment"]) + "\n")
        o.append("\n\\begin{thebibliography}{" + str(len(self.refs)) + "}\n")
        for key in self.cite_order:
            o.append("\\bibitem{" + key + "} " + T(self.refs[key]) + "\n")
        o.append("\\end{thebibliography}\n\\end{document}\n")
        return "\n".join(o)

    # -------------------------------------------------------------------------- algorithms
    _OPEN = {"for", "while", "if", "else", "elsif"}
    _CLOSE = {"endfor", "endwhile", "endif", "else", "elsif"}

    def _algo_tex_line(self, kind: str, text: str) -> str:
        T = lambda s: self.inline(s, "tex")  # noqa: E731
        comment = ""
        if " // " in text:
            text, comment = text.split(" // ", 1)
            comment = "\\COMMENT{" + T(comment) + "}"
        simple = {"req": "\\REQUIRE ", "ens": "\\ENSURE ", "state": "\\STATE ", "return": "\\RETURN "}
        braced = {"for": "\\FOR", "while": "\\WHILE", "if": "\\IF", "elsif": "\\ELSIF"}
        bare = {"endfor": "\\ENDFOR", "endwhile": "\\ENDWHILE", "endif": "\\ENDIF", "else": "\\ELSE"}
        if kind in simple:
            return simple[kind] + T(text) + (" " + comment if comment else "")
        if kind in braced:
            return braced[kind] + "{" + T(text) + "}"
        return bare[kind]

    def _algo_html(self, a: dict) -> str:
        H = lambda s: self.inline(s, "html")  # noqa: E731
        depth, n, rows = 0, 0, []
        words = {"for": ("for", "do"), "while": ("while", "do"), "if": ("if", "then"), "elsif": ("else if", "then")}
        for kind, text in a["lines"]:
            if kind in self._CLOSE:
                depth = max(depth - 1, 0)
            comment = ""
            if " // " in text:
                text, comment = text.split(" // ", 1)
                comment = f'<span class="cm">// {H(comment)}</span>'
            numbered = kind not in ("req", "ens")
            if numbered:
                n += 1
            if kind == "req":
                inner = f"<b>Require:</b> {H(text)}"
            elif kind == "ens":
                inner = f"<b>Ensure:</b> {H(text)}"
            elif kind == "state":
                inner = H(text)
            elif kind == "return":
                inner = f"<b>return</b> {H(text)}"
            elif kind in words:
                inner = f"<b>{words[kind][0]}</b> {H(text)} <b>{words[kind][1]}</b>"
            elif kind == "else":
                inner = "<b>else</b>"
            else:
                inner = "<b>end " + kind[3:] + "</b>"
            rows.append(f'<div class="al"><span class="ln">{n if numbered else ""}</span><span class="at" style="padding-left:{depth * 10}pt">{inner} {comment}</span></div>')
            if kind in self._OPEN:
                depth += 1
        return (f'<div class="alg" id="{a["label"].replace(":", "-")}"><div class="algt"><b>Algorithm {self.labels[a["label"]]}</b> {H(a["caption"])}</div>'
                + "".join(rows) + "</div>")

    # -------------------------------------------------------------------------- HTML
    def html(self, css: str) -> str:
        m = self.meta
        H = lambda s: self.inline(s, "html")  # noqa: E731
        body: list[str] = []
        body.append('<div class="abs"><b><i>Abstract</i>&mdash;' + H(m["abstract"]) + '</b></div>')
        body.append('<div class="abs"><b><i>Index Terms</i>&mdash;' + H(m["keywords"]) + "</b></div>")
        for b in self.blocks:
            k = b[0]
            if k == "sec":
                n = self.labels.get(b[2], "") if b[2] else ""
                body.append(f'<h2 id="{b[2].replace(":", "-") if b[2] else ""}">{n}. {H(b[1])}</h2>')
            elif k == "sub":
                label = self.labels.get(b[2], "") if len(b) > 2 and b[2] else ""
                letter = label.split("-")[-1] if label else ""
                body.append(f'<h3 id="{b[2].replace(":", "-") if len(b) > 2 and b[2] else ""}">{letter}. {H(b[1])}</h3>')
            elif k == "p":
                body.append("<p>" + H(b[1]) + "</p>")
            elif k == "enum":
                body.append("<ol>" + "".join(f"<li>{H(i)}</li>" for i in b[1]) + "</ol>")
            elif k == "eq":
                body.append(f'<div class="eq" id="{b[1].replace(":", "-")}"><span class="eqbody">{b[3]}</span><span class="eqn">({self.labels[b[1]]})</span></div>')
            elif k == "fig":
                f = b[1]
                cls = "fig wide" if f.get("wide") else "fig"
                body.append(f'<figure class="{cls}" id="{f["label"].replace(":", "-")}"><img src="figures/{f["file"]}.png" alt="">'
                            f'<figcaption>Fig.&nbsp;{self.labels[f["label"]]}. {H(f["caption"])}</figcaption></figure>')
            elif k == "algo":
                body.append(self._algo_html(b[1]))
            elif k == "table":
                t = b[1]
                cls = "tbl wide" if t.get("wide") else "tbl"
                head = "".join(f"<th>{H(c)}</th>" for c in t["header"])
                rows = "".join("<tr>" + "".join(f"<td>{H(c)}</td>" for c in r) + "</tr>" for r in t["rows"])
                body.append(f'<div class="{cls}" id="{t["label"].replace(":", "-")}"><div class="tcap">TABLE {self.labels[t["label"]]}</div>'
                            f'<div class="tttl">{H(t["caption"])}</div><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>')
        if m.get("acknowledgment"):
            body.append("<h2>Acknowledgment</h2><p>" + H(m["acknowledgment"]) + "</p>")
        refs = "".join(f'<li id="ref{i}"><span class="rn">[{i}]</span> {H(self.refs[key])}</li>' for i, key in enumerate(self.cite_order, 1))
        body.append(f'<h2 class="refh">References</h2><ol class="refs">{refs}</ol>')
        return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(m["title_plain"])}</title><style>{css}</style></head><body>'
                f'<header><h1>{H(m["title"])}</h1><div class="au">{m["authors_html"]}</div></header><main>{"".join(body)}</main></body></html>')
