"""Bounded static substitution, never a TeX interpreter.

Conflicting, scoped, conditional and executable definitions fail closed. Only
literal math/text bodies, positional parameters and one optional argument work.
"""
from __future__ import annotations

import re

from .formulas import STANDARD, strip_comments

COMMAND = re.compile(r"\\[A-Za-z]+|\\.")
DECL = re.compile(r"\\(newcommand|renewcommand|providecommand|DeclareMathOperator|def|gdef|edef|xdef)\*?\s*")


def group(text: str, offset: int, left="{", right="}") -> tuple[str, int]:
    while offset < len(text) and text[offset].isspace():
        offset += 1
    if offset >= len(text) or text[offset] != left:
        raise ValueError("missing grouped argument")
    start, depth = offset + 1, 1
    offset += 1
    while offset < len(text):
        if text[offset] == "\\":
            match = COMMAND.match(text, offset)
            offset = match.end() if match else offset + 1
            continue
        depth += (text[offset] == left) - (text[offset] == right)
        if depth == 0:
            return text[start:offset], offset + 1
        offset += 1
    raise ValueError("unbalanced grouped argument")


def collect_macros(files: list[dict]) -> dict:
    definitions, blocked = {}, {}
    for item in files:
        text = strip_comments(item["text"])
        depth, conditional, scope, cursor = 0, 0, 0, 0
        for match in DECL.finditer(text):
            # Definition bodies are not scanned as new declarations.
            if match.start() < cursor:
                continue
            prefix = text[cursor:match.start()]
            depth += len(re.findall(r"(?<!\\)\{", prefix)) - len(re.findall(r"(?<!\\)\}", prefix))
            for command in COMMAND.findall(prefix):
                if command in {r"\begingroup", r"\bgroup"}:
                    scope += 1
                elif command in {r"\endgroup", r"\egroup"}:
                    scope = max(0, scope - 1)
                if command.startswith(r"\if"):
                    conditional += 1
                elif command == r"\fi":
                    conditional = max(0, conditional - 1)
            offset, name = match.end(), None
            try:
                if text[offset:offset + 1] == "{":
                    name, offset = group(text, offset)
                else:
                    token = COMMAND.match(text, offset)
                    if token is None:
                        raise ValueError("missing macro name")
                    name, offset = token[0], token.end()
                if not re.fullmatch(r"\\[A-Za-z]+", name):
                    raise ValueError("unsupported macro name")
                nargs, default = 0, None
                if match[1] in {"def", "gdef", "edef", "xdef"}:
                    body_start = text.index("{", offset)
                    parameters = text[offset:body_start].strip()
                    nargs = len(re.findall(r"#[1-9]", parameters))
                    if parameters != "".join(f"#{i}" for i in range(1, nargs + 1)):
                        raise ValueError("delimited parameters unsupported")
                    offset = body_start
                else:
                    offset += len(text[offset:]) - len(text[offset:].lstrip())
                    if text[offset:offset + 1] == "[":
                        count, offset = group(text, offset, "[", "]")
                        nargs = int(count)
                        offset += len(text[offset:]) - len(text[offset:].lstrip())
                        if text[offset:offset + 1] == "[":
                            default, offset = group(text, offset, "[", "]")
                body, offset = group(text, offset)
                if depth != 0 or conditional or scope or match[1] in {"gdef", "edef", "xdef"}:
                    raise ValueError("scoped, conditional or dynamic definition")
                if not 0 <= nargs <= 9 or len(body) > 4096 or len(definitions) >= 2000:
                    raise ValueError("macro definition limit")
                if re.search(r"#(?![1-9])", body) or any(int(i) > nargs for i in re.findall(r"#([1-9])", body)):
                    raise ValueError("invalid positional parameter")
                if default is not None and nargs == 0:
                    raise ValueError("optional argument requires a parameter")
                if match[1] == "DeclareMathOperator":
                    body = r"\operatorname" + ("*" if "*" in match[0] else "") + "{" + body + "}"
                definition = {"body": body, "nargs": nargs, "default": default,
                              "member": item["member"], "line": text.count("\n", 0, match.start()) + 1}
                previous = definitions.get(name)
                if previous and any(previous[k] != definition[k] for k in ("body", "nargs", "default")):
                    raise ValueError("conflicting definitions; include order not executed")
                definitions[name] = definition
            except (ValueError, IndexError) as error:
                if name:
                    blocked[name] = str(error)
            cursor = max(match.end(), offset)
    # Unknown control flow or package machinery is not executed, even indirectly.
    allowed = {"\\" + c for c in STANDARD} | set(definitions)
    for name, definition in definitions.items():
        unknown = {c for c in COMMAND.findall(definition["body"]) if c[1:].isalpha() and c not in allowed}
        if unknown:
            blocked[name] = "unsupported body commands: " + ", ".join(sorted(unknown))
    return {"definitions": definitions, "blocked": blocked}


def expand_macros(text: str, registry: dict, *, max_steps=512, max_depth=16, max_chars=20000) -> dict:
    definitions, blocked = registry["definitions"], registry["blocked"]
    used, errors, steps = set(), {}, 0

    def expand(value: str, stack: tuple = ()) -> str:
        nonlocal steps
        output, cursor = [], 0
        for token in COMMAND.finditer(value):
            if token.start() < cursor:
                continue
            output.append(value[cursor:token.start()])
            name, cursor = token[0], token.end()
            if name not in definitions and name not in blocked:
                output.append(name)
                continue
            try:
                if name in blocked:
                    raise ValueError(blocked[name])
                if name in stack or len(stack) >= max_depth or steps >= max_steps:
                    raise ValueError("recursive macro or expansion limit")
                steps += 1
                spec, args, end = definitions[name], [], cursor
                if spec["default"] is not None:
                    start = end + len(value[end:]) - len(value[end:].lstrip())
                    if value[start:start + 1] == "[":
                        arg, end = group(value, start, "[", "]")
                    else:
                        arg = spec["default"]
                    args.append(arg)
                for _ in range(spec["nargs"] - len(args)):
                    end += len(value[end:]) - len(value[end:].lstrip())
                    if value[end:end + 1] == "{":
                        arg, end = group(value, end)
                    else:
                        part = COMMAND.match(value, end)
                        if end >= len(value):
                            raise ValueError("missing macro parameter")
                        arg, end = (part[0], part.end()) if part else (value[end], end + 1)
                    args.append(arg)
                body = re.sub(r"#([1-9])", lambda m: args[int(m[1]) - 1], spec["body"])
                replacement = expand(body, (*stack, name))
                if re.search(r"\\[A-Za-z]+$", replacement) and value[end:end + 1].isalpha():
                    replacement += " "
                if output and re.search(r"\\[A-Za-z]+$", output[-1]) and replacement[:1].isalpha():
                    replacement = " " + replacement
                if sum(map(len, output)) + len(replacement) + len(value[end:]) > max_chars:
                    raise ValueError("expanded character limit")
                output.append(replacement)
                cursor = end
                used.add(name)
            except ValueError as error:
                errors[name] = str(error)
                output.append(name)
        output.append(value[cursor:])
        return "".join(output)

    if len(text) > max_chars:
        return {"latex": text, "expanded_macros": [], "errors": {"input": "character limit"}, "complete": False}
    value = expand(text)
    remaining = sorted(set(COMMAND.findall(value)) & (set(definitions) | set(blocked)))
    return {"latex": value, "expanded_macros": sorted(used), "unexpanded_macros": remaining,
            "errors": errors, "complete": not errors and not remaining, "steps": steps}
