"""Deterministic repair-and-parse routine for LLM JSON outputs (quote normalisation,
newline escaping, trailing-comma removal, missing-comma insertion, bracket balancing)."""

import json
import re


def _extract_block(text):
    lb, rb = text.find("{"), text.rfind("}")
    la, ra = text.find("["), text.rfind("]")
    if lb != -1 and rb > lb:
        return text[lb:rb + 1]
    if la != -1 and ra > la:
        return text[la:ra + 1]
    return text


def _escape_newlines_in_strings(text):
    out, in_str, esc = [], False, False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            elif ch == "\n":
                out.append("\\n")
                continue
            elif ch == "\r":
                continue
        elif ch == '"':
            in_str = True
        out.append(ch)
    return "".join(out)


def _balance(text):
    stack, out, in_str, esc = [], [], False, False
    for ch in text:
        out.append(ch)
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack and stack[-1] == ch:
            stack.pop()
    if in_str:
        out.append('"')
    out.extend(reversed(stack))
    return "".join(out)


def repair_json(text):
    t = _extract_block(text.strip())
    t = t.replace("“", '"').replace("”", '"')
    t = _escape_newlines_in_strings(t)
    t = re.sub(r",\s*([}\]])", r"\1", t)
    t = re.sub(r'([}\]"])\s*\n\s*(")', r"\1,\2", t)
    t = _balance(t)
    return re.sub(r",\s*([}\]])", r"\1", t)


def parse_json(text):
    """Return (obj, repaired_flag). Raises ValueError if unrecoverable."""
    try:
        return json.loads(text), False
    except Exception:
        pass
    try:
        return json.loads(repair_json(text)), True
    except Exception as e:
        raise ValueError(f"unparseable JSON: {e}; head={text[:200]!r}")
