"""Presentation normalization only; never claim general mathematical equivalence."""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


TOKEN = re.compile(r"\\[a-zA-Z]+|\\.|[^\s]")
# Unknown commands are retained and block source-based acceptance; no macro execution.
STANDARD = set("alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi pi varpi rho varrho sigma varsigma tau upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega frac sqrt sum prod int iint iiint oint lim min max arg log ln exp sin cos tan cot sec csc sinh cosh tanh det dim ker Pr inf sup argmin argmax mathbf mathbb mathcal mathrm mathit mathsf mathtt boldsymbol bm vec hat widehat tilde widetilde bar overline underline dot ddot acute grave breve check text textrm textbf textit operatorname operatornamewithlimits overset underset overbrace underbrace binom dbinom tbinom substack mathop mathrel mathbin mathord mathopen mathclose mathpunct displaystyle textstyle scriptstyle scriptscriptstyle limits nolimits begin end hline cline ldots cdots vdots ddots dots times cdot ast star circ bullet otimes oplus odot pm mp div le leq ge geq neq ne approx sim simeq equiv propto in notin ni subset subseteq supset supseteq cup cap setminus emptyset varnothing forall exists nexists neg land lor wedge vee to mapsto rightarrow leftarrow Rightarrow Leftarrow leftrightarrow Leftrightarrow longrightarrow Longrightarrow gets iff implies partial nabla infty ell hbar Re Im top bot perp parallel angle triangle square checkmark lVert rVert lvert rvert vert Vert langle rangle lbrace rbrace lceil rceil lfloor rfloor backslash percent mod bmod pmod underbracket overbracket color phantom vphantom hphantom split aligned alignedat gathered cases matrix pmatrix bmatrix Bmatrix vmatrix Vmatrix smallmatrix array eqno middle big Big bigg Bigg bigl bigr Bigl Bigr biggl biggr Biggl Biggr label tag nonumber notag left right dfrac tfrac quad qquad enspace thinspace hspace vspace textnormal mathnormal ensuremath mbox null rm bf it cal tt sf cr noalign displaylimits mathstrut textcolor eqref ref mathscr mathfrak xrightarrow xleftarrow stackrel choose atop colon coloneqq triangleq lnot lnotin smallsetminus dotsc dotsb dotsm dotso dotsi prime dagger ddagger wr uplus sqcup sqcap amalg oslash bigcup bigcap bigoplus bigotimes bigvee bigwedge varepsilon probability expectation restrict restriction diag tr trace rank softmax sigmoid relu sgn sign ln limsup liminf gcd lcm degree epsilon omicron leqslant geqslant lesssim gtrsim lhd rhd unlhd unrhd multicolumn hdotsfor cfrac genfrac protect relax displaybreak allowbreak intertext shortintertext eqdot".split())


def strip_comments(text: str) -> str:
    return re.sub(r"(\\\\)*%[^\n]*|\\%", lambda m: m[0] if m[0] == r"\%" else (m[1] or ""), text)


def normalize_latex(text: str) -> str:
    value = strip_comments(text).strip()
    # Unicode mathematical alphabets carry scientific style; never reduce to plain Latin.
    def unicode_math(character: str) -> str:
        name = unicodedata.name(character, "")
        if name.startswith(("MATHEMATICAL BOLD CAPITAL ", "MATHEMATICAL BOLD SMALL ")):
            return r"\mathbf{" + unicodedata.normalize("NFKC", character) + "}"
        if name.startswith(("MATHEMATICAL BOLD ITALIC CAPITAL ", "MATHEMATICAL BOLD ITALIC SMALL ")):
            return r"\boldsymbol{" + unicodedata.normalize("NFKC", character) + "}"
        return character
    value = "".join(unicode_math(c) for c in value)
    value = re.sub(r"\\bm\b", lambda _: r"\boldsymbol", value)
    # Combining accents are explicit vector/hat/bar marks, never plain characters.
    for mark, command in [("\u20d7", "vec"), ("\u0302", "hat"), ("\u0304", "bar")]:
        value = re.sub(r"([A-Za-z])" + mark, lambda m: "\\" + command + "{" + m[1] + "}", value)
    for left, right in [("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$")]:
        if value.startswith(left) and value.endswith(right):
            value = value[len(left):-len(right)].strip()
            break
    value = re.sub(r"(\\[a-zA-Z]+)\s+(?=\{)", r"\1", value)
    value = re.sub(r"\\(?:label|tag)\*?\{[^{}]*\}", "", value)
    value = re.sub(r"\\(?:nonumber|notag)\b", "", value)
    value = re.sub(r"\\(?:left|right)\b", "", value)
    value = re.sub(r"\\(?:dfrac|tfrac)\b", lambda _: r"\frac", value)
    value = re.sub(r"\\[,!;:]|\\(?:quad|qquad|enspace|thinspace)\b", "", value)
    # Preserve words and spaces inside text groups. Do not collapse 'a b' into 'ab'.
    value = re.sub(r"([_^])(\\[a-zA-Z]+|[a-zA-Z0-9])", lambda m: m[1] + "{" + m[2] + "}", value)
    tokens = re.findall(r"\\(?:text|textrm|mbox)\{[^{}]*\}|\\[a-zA-Z]+|\\.|[^\s]", value)
    # TeX permits x_i^j and x^j_i: canonicalize this one syntax equivalence.
    def script_end(start: int) -> int:
        if start + 1 >= len(tokens):
            return start
        if tokens[start + 1] != "{":
            return start + 2
        depth = 0
        for end in range(start + 1, len(tokens)):
            depth += (tokens[end] == "{") - (tokens[end] == "}")
            if depth == 0:
                return end + 1
        return start
    index = 0
    while index < len(tokens):
        if tokens[index] == "^":
            end = script_end(index)
            if end > index and end < len(tokens) and tokens[end] == "_":
                last = script_end(end)
                if last > end:
                    tokens[index:last] = tokens[end:last] + tokens[index:end]
                    index = last
                    continue
        index += 1
    return " ".join(tokens)


def structure(text: str) -> dict:
    tokens = TOKEN.findall(normalize_latex(text))
    depth = 0
    balanced = True
    for token in tokens:
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth < 0:
                balanced = False
    environments = re.findall(r"\\(begin|end)\s*\{\s*([^{}]+?)\s*\}", text)
    stack = []
    for kind, env in environments:
        if kind == "begin":
            stack.append(env)
        elif not stack or stack.pop() != env:
            balanced = False
    commands = {t[1:] for t in tokens if re.fullmatch(r"\\[a-zA-Z]+", t)}
    return {"balanced": balanced and depth == 0 and not stack,
            "subscripts": tokens.count("_"), "superscripts": tokens.count("^"),
            "fractions": tokens.count(r"\frac"), "environments": environments,
            "unknown_commands": sorted(commands - STANDARD)}


def compare_latex(left: str, right: str) -> dict:
    a, b = normalize_latex(left), normalize_latex(right)
    sa, sb = structure(left), structure(right)
    ta, tb = TOKEN.findall(a), TOKEN.findall(b)
    diff = [{"operation": op, "left": ta[i:j], "right": tb[k:l]}
            for op, i, j, k, l in SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes() if op != "equal"]
    return {"normalized_exact_match": bool(a) and a == b,
            "token_similarity": SequenceMatcher(None, ta, tb, autojunk=False).ratio(),
            "left_structure": sa, "right_structure": sb, "token_differences": diff,
            "symbol_diagnostics": symbol_differences(left, right),
            "semantic_equivalence": "not_evaluated"}


def without_bold(text: str) -> str:
    """Diagnostic alignment only. This must never be used as exact-match scoring."""
    from .macros import group
    value = text
    pattern = re.compile(r"\\(?:mathbf|boldsymbol|bm)\s*")
    for _ in range(128):
        match = pattern.search(value)
        if not match:
            break
        try:
            body, end = group(value, match.end())
        except ValueError:
            break
        value = value[:match.start()] + " " + body + " " + value[end:]
    return value


def symbol_differences(left: str, right: str) -> dict:
    def styles(value):
        return {kind: len(re.findall(r"\\" + kind + r"\b", value))
                for kind in ["mathbf", "boldsymbol", "bm", "vec", "overrightarrow", "hat", "bar"]}
    a, b = styles(left), styles(right)
    return {"parser_styles": a, "source_styles": b,
            "bold_only_alignment": bool(normalize_latex(left)) and
                normalize_latex(without_bold(left)) == normalize_latex(without_bold(right)),
            "style_difference": a != b,
            "warning": "bold/arrow distinctions are preserved; alignment is not mathematical equivalence"}
