#!/usr/bin/env python3
"""The Good Earth (Pearl S. Buck, 1931) is still under copyright, so the released files do not contain the
text of the excerpt (chapter 26) used in the study. Every passage of 25 or more characters taken from it
(narration sentences and quoted speech, also inside extracted dialogue, scene initialisations, session
openings, cached LLM responses and the source-reference table) is replaced by a placeholder of the form
[GE-OMITTED:<fragment>:<encoding>]. Generated dialogue is not affected.

Anyone with a copy of the book can restore the files exactly:

    python texts/good_earth_release.py restore --excerpt good_earth_ch26.txt --root .

where good_earth_ch26.txt holds the excerpt of chapter 26 that was used in the study (from the first to
the last words listed in texts/good_earth_manifest.json; curly or straight quotation marks and any line
breaks are accepted). The script checks the excerpt against the SHA-256 checksums in the manifest before
it writes anything.

The authors' pipeline calls `redact()` when the release archives are built.
"""

import argparse
import hashlib
import json
import os
import re
import sys

MIN_LEN = 25
TAG = re.compile(r"\[GE-OMITTED:(\d+):([a-z]+)\]")
TEXT_EXT = (".json", ".jsonl", ".csv", ".txt", ".md", ".tex", ".html", ".log")


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def fragments(excerpt):
    """Sentences of the narration and of the quoted speech, in order of appearance (deterministic)."""
    out = []
    for chunk in re.split(r"[\"“”]", excerpt):
        for part in re.split(r"(?<=[.!?;:,])\s+", chunk):
            part = part.strip()
            if len(part) >= MIN_LEN and part not in out:
                out.append(part)
    return out


def forms(f):
    """The ways a fragment is written in the released files: raw, JSON (with and without ASCII escapes),
    JSON inside JSON (cached LLM responses), CSV (doubled quotes) and with straight apostrophes."""
    j = json.dumps(f, ensure_ascii=False)[1:-1]
    a = json.dumps(f, ensure_ascii=True)[1:-1]
    out = {"r": f, "j": j, "a": a, "jj": json.dumps(j, ensure_ascii=False)[1:-1], "aa": json.dumps(a, ensure_ascii=True)[1:-1],
           "c": f.replace('"', '""')}
    if "’" in f:
        out["q"] = f.replace("’", "'")
    return out


def redact(text, frags):
    """Replace every occurrence of every fragment (longest first) by its placeholder."""
    order = sorted(range(len(frags)), key=lambda i: -len(frags[i]))
    for i in order:
        fs = forms(frags[i])
        for enc in ("aa", "jj", "a", "j", "c", "q", "r"):  # escaped forms first: they contain backslashes
            v = fs.get(enc)
            if v and v in text:
                text = text.replace(v, f"[GE-OMITTED:{i}:{enc}]")
    return text


def restore(text, frags):
    return TAG.sub(lambda m: forms(frags[int(m.group(1))])[m.group(2)], text)


def normalize(text):
    """Straight quotes and single spaces, as in the excerpt file of the study."""
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return " ".join(text.split())


def whitespace_map(excerpt):
    """Offsets (in the normalized text) of whitespace that is not a single space, e.g. paragraph breaks."""
    out = []
    for m in re.finditer(r"\s+", excerpt):
        out_pos = len(" ".join(excerpt[:m.start()].split()))
        if m.group(0) != " ":
            out.append([out_pos, m.group(0)])
    return out


def rebuild(normalized, ws_map):
    chars = list(normalized)
    for off, ws in ws_map:
        chars[off] = ws
    return "".join(chars)


def manifest(excerpt):
    frags = fragments(excerpt)
    words = excerpt.split()
    return {"work": "Pearl S. Buck, The Good Earth (1931), chapter 26", "n_words": len(words),
            "first_words": " ".join(words[:4]), "last_words": " ".join(words[-4:]),
            "excerpt_sha256": sha(excerpt), "whitespace": whitespace_map(excerpt), "min_fragment_length": MIN_LEN,
            "n_fragments": len(frags), "fragment_sha256": [sha(f) for f in frags]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["restore"])
    ap.add_argument("--excerpt", required=True, help="text file with the excerpt of chapter 26 as used in the study")
    ap.add_argument("--root", default=".", help="folder that contains texts/, runs/ and results/")
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    man = json.load(open(os.path.join(here, "good_earth_manifest.json"), encoding="utf-8"))
    excerpt = rebuild(normalize(open(args.excerpt, encoding="utf-8").read()), man["whitespace"])
    if sha(excerpt) != man["excerpt_sha256"]:
        frags = fragments(excerpt)
        ok = sum(sha(f) == h for f, h in zip(frags, man["fragment_sha256"]))
        sys.exit(f"The excerpt does not match the one used in the study ({ok}/{man['n_fragments']} fragments match). "
                 f"It must run from '{man['first_words']} ...' to '... {man['last_words']}' ({man['n_words']} words).")
    frags = fragments(excerpt)
    n_files = n_tags = 0
    for base, _, files in os.walk(args.root):
        for fn in files:
            if not fn.endswith(TEXT_EXT):
                continue
            p = os.path.join(base, fn)
            s = open(p, encoding="utf-8", newline="").read()
            k = len(TAG.findall(s))
            if k:
                open(p, "w", encoding="utf-8", newline="").write(restore(s, frags))
                n_files, n_tags = n_files + 1, n_tags + k
    print(f"restored {n_tags} passages in {n_files} files")


if __name__ == "__main__":
    main()
