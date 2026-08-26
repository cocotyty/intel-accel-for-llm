#!/usr/bin/env python3
"""Cut long docstrings back to their opening summary paragraph.

Used by dev/gen_slim.sh on the slim tree only. A docstring whose body
continues past the first blank line is replaced by the part before it;
a summary that then fits on one line is written on one line. Every
rewrite is validated with ast.parse before being written, and inline
comments are never touched (collapsing them would truncate sentences
mid-thought).
"""
import ast
import pathlib
import re
import sys


def first_para(doc: str) -> str:
    out = []
    for line in doc.splitlines():
        if not line.strip() and out:
            break
        out.append(line.rstrip())
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def trim(path: str) -> None:
    p = pathlib.Path(path)
    lines = p.read_text().splitlines()
    drop = set()
    repl = {}

    tree = ast.parse(p.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        d = body[0]
        if not (isinstance(d, ast.Expr) and isinstance(d.value, ast.Constant)
                and isinstance(d.value.value, str)):
            continue
        raw = d.value.value
        if raw.count("\n") + 1 < 4:
            continue
        short = first_para(raw)
        if short == raw.rstrip():
            continue
        a, b = d.lineno - 1, d.end_lineno - 1
        indent = " " * (len(lines[a]) - len(lines[a].lstrip()))
        s = short.strip("\n")
        if "\n" in s:
            txt = "\n".join(
                (indent + ln.strip()) if i else (indent + '"""' + ln.strip())
                for i, ln in enumerate(s.splitlines()))
            new = txt + "\n" + indent + '"""'
        else:
            new = f'{indent}"""{s.strip()}"""'
        repl[a] = new
        for i in range(a + 1, b + 1):
            drop.add(i)

    out = []
    i = 0
    while i < len(lines):
        if i in drop:
            i += 1
            continue
        line = repl.get(i, lines[i])
        # A multi-line form holding a single summary line -> one line.
        m = re.match(r'^(\s*)"""(.*)$', line)
        if (m and not line.rstrip().endswith('"""')
                and i + 1 < len(lines) and lines[i + 1].strip() == '"""'):
            text = m.group(2).strip()
            if text and len(m.group(1)) + 6 + len(text) <= 79:
                line = f'{m.group(1)}"""{text}"""'
                i += 1
        out.append(line)
        i += 1

    new_src = "\n".join(out) + "\n"
    ast.parse(new_src)  # refuse to write anything unparsable
    p.write_text(new_src)


for f in sys.argv[1:]:
    trim(f)
