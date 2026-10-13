#!/usr/bin/env python3
"""Evaluate the Director's dialogue extraction against manually attributed gold quotations.

For every (text, model) scene: quote recall (gold quotations recovered), speaker attribution
accuracy on recovered quotations (after alias mapping to the verified cast), and the share
of extracted utterances that cannot be matched to any quotation of the source (fabricated).
"""

import argparse
import glob
import json
import os
import re

import pandas as pd


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower().replace("’", "'")).split()


def containment(gold, ext):
    g, e = norm(gold), set(norm(ext))
    return sum(w in e for w in g) / max(len(g), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--excerpts", default=os.path.join(os.path.dirname(__file__), "..", "texts", "excerpts.json"))
    args = ap.parse_args()
    texts = {t["id"]: t for t in json.load(open(args.excerpts, encoding="utf-8"))}
    rows = []
    for f in sorted(glob.glob(os.path.join(args.runs, "scenes", "*.json"))):
        tid, mslug = os.path.basename(f)[:-5].split("__")
        sc = json.load(open(f, encoding="utf-8"))
        gold = texts[tid]["gold_quotes"]
        kept = sc["dialogue"]
        raw = sc.get("dialogue_raw", kept)
        hit = correct = 0
        for g in gold:
            best = max(kept, key=lambda e: containment(g["text"], e["text"]), default=None)
            if best is not None and containment(g["text"], best["text"]) >= 0.8:
                hit += 1
                correct += int(best["speaker"] == g["speaker"])
        fabricated = sum(max((containment(e["text"], g["text"]) for g in gold), default=0) < 0.5 for e in raw)
        m = sc.get("cast_mapping", {})
        rows.append(dict(text_id=tid, model=mslug, gold_quotes=len(gold), extracted_raw=len(raw), kept=len(kept),
                         dropped_unmapped=m.get("dropped_unmapped", 0), quote_recall=hit / len(gold),
                         attribution_acc=correct / hit if hit else float("nan"),
                         fabricated_rate=fabricated / max(len(raw), 1), raw_labels="; ".join(m.get("raw_labels", []))))
    df = pd.DataFrame(rows)
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "table_extraction.csv"), index=False)
    print(df.to_string())
    print(df.groupby("model")[["quote_recall", "attribution_acc", "fabricated_rate"]].mean())


if __name__ == "__main__":
    main()
