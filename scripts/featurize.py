#!/usr/bin/env python3
"""Featurise all generated sessions.

Outputs (in --out):
  utterances.csv   one row per generated turn (all conditions, plus the rule-based
                   'mechanical' transposition of every first-person turn of the main runs)
  sessions.csv     one row per session (session means of utterance measures + session measures)
  source_reference.csv  measures for quoted speech vs. narration in the six source excerpts
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from narrsim.metrics import Featurizer, session_measures, transpose_to_third  # noqa: E402

UTT_MEANS = ["n_words", "n_content", "fp_ratio", "fp_rate", "tp_rate", "self_name", "polarity", "intensity",
             "subjectivity", "vader", "distilbert", "word_len_all", "word_len", "zipf", "rare_rate", "question"]


def featurise_session(fz, rec, scene, mode_label=None, transform=None):
    turns = rec["turns"]
    if transform:
        turns = [dict(t, text=transform(t)) for t in turns]
    names = list(scene["characters"])
    feats = fz.utterances(turns, names) if turns else []
    meta = dict(rec["meta"])
    if mode_label:
        meta["mode"] = mode_label
    rows = []
    for t, f in zip(turns, feats):
        r = {k: v for k, v in f.items() if k not in ("content_tokens", "tokens")}
        r.update(meta, speaker=t["speaker"], turn=t.get("turn"), text=t["text"],
                 content_tokens=" ".join(f["content_tokens"]), residual_violation=bool(t.get("residual_violation", False)))
        rows.append(r)
    sess = dict(meta, n_characters=len(names))
    for m in UTT_MEANS:
        vals = [f[m] for f in feats if f[m] == f[m]]
        sess[m] = float(np.mean(vals)) if vals else np.nan
    for c in ("pos", "neu", "neg"):
        sess[f"share_{c}"] = float(np.mean([f["sent_class"] == c for f in feats])) if feats else np.nan
    opening_last = rec["opening"][-1] if rec.get("opening") else None
    sess.update(session_measures(turns, feats, names, opening_last))
    cs = rec.get("call_stats", {})
    sess.update({f"calls_{k}": v for k, v in cs.items()}, termination=rec.get("termination"), seconds=rec.get("seconds"))
    return rows, sess


def split_source(text):
    """Separate quoted direct speech from narration in a prose excerpt."""
    quotes = re.findall(r"[“\"]([^”\"]+)[”\"]", text)
    narration = re.sub(r"[“\"][^”\"]+[”\"]", " ", text)
    return [q for q in quotes if len(q.split()) >= 3], [s for s in re.split(r"(?<=[.!?])\s+", narration) if len(s.split()) >= 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--excerpts", default=os.path.join(os.path.dirname(__file__), "..", "texts", "excerpts.json"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fz = Featurizer()

    scenes = {}
    for f in glob.glob(os.path.join(args.runs, "scenes", "*.json")):
        tid, mslug = os.path.basename(f)[:-5].split("__")
        scenes[(tid, mslug)] = json.load(open(f, encoding="utf-8"))

    utt, ses = [], []
    files = sorted(glob.glob(os.path.join(args.runs, "sessions", "*", "*", "*", "*.json")))
    for i, f in enumerate(files):
        rec = json.load(open(f, encoding="utf-8"))
        mslug = rec["meta"]["model"].replace(":", "_").replace("/", "_")
        scene = scenes[(rec["meta"]["text_id"], mslug)]
        rows, sess = featurise_session(fz, rec, scene)
        sess["file"] = os.path.relpath(f, args.runs)
        utt += rows
        ses.append(sess)
        if rec["meta"]["condition"] == "full" and rec["meta"]["mode"] == "first_person":
            def tr(t, sc=scene):
                g = sc["characters"].get(t["speaker"], {}).get("gender", "unknown")
                return transpose_to_third(t["text"], t["speaker"], g, t.get("speak_to"))
            rows, sess = featurise_session(fz, rec, scene, mode_label="mechanical_third", transform=tr)
            sess["file"] = os.path.relpath(f, args.runs) + "#mechanical"
            utt += rows
            ses.append(sess)
        if i % 50 == 0:
            print(f"{i}/{len(files)} sessions featurised", flush=True)

    pd.DataFrame(utt).to_csv(os.path.join(args.out, "utterances.csv"), index=False)
    pd.DataFrame(ses).to_csv(os.path.join(args.out, "sessions.csv"), index=False)

    # Human-authored reference: direct speech vs. narration in the source excerpts
    ref = []
    for t in json.load(open(args.excerpts, encoding="utf-8")):
        if t["id"] == "earnest":
            continue  # dramatic text: no narration
        speech, narr = split_source(t["text"])
        for kind, segs in (("speech", speech), ("narration", narr)):
            feats = fz.utterances([dict(text=s, speaker="") for s in segs], [])
            for s, fe in zip(segs, feats):
                ref.append(dict(text_id=t["id"], kind=kind, text=s,
                                **{k: v for k, v in fe.items() if k not in ("content_tokens", "tokens", "sent_class")}))
    pd.DataFrame(ref).to_csv(os.path.join(args.out, "source_reference.csv"), index=False)
    print(f"wrote {len(utt)} utterances, {len(ses)} sessions, {len(ref)} source segments")


if __name__ == "__main__":
    main()
