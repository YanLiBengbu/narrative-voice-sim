#!/usr/bin/env python3
"""Collect the numbers quoted in the manuscript, the response letter and the project page
into results/facts.json (single source of truth for every reported value)."""

import argparse
import json
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    args = ap.parse_args()
    R = args.res
    S = json.load(open(os.path.join(R, "summary.json"), encoding="utf-8"))
    F = {}
    F["design"] = S["design"]
    comp = pd.DataFrame(S["mode_comparisons"])
    for _, r in comp.iterrows():
        k = f"{r.metric}|{r.a}|{r.b}"
        F[k] = {c: (None if pd.isna(r[c]) else float(r[c])) for c in ["n", "mean_a", "mean_b", "diff", "ci_low", "ci_high", "dz", "r_rb", "p", "p_holm"]}
    F["cells"] = {r["metric"]: {"n_pos": int(r["n_pos"]), "n_neg": int(r["n_neg"]), "n_cells": int(r["n_cells"])} for r in S["cell_consistency"]}
    F["mixed"] = S["mixed_models"]
    F["by_group"] = S["effects_by_group"]
    F["json_stats"] = S["json_stats"]
    rel = pd.DataFrame(S["reliability"])
    F["reliability_full"] = rel[rel.condition == "full"].to_dict(orient="records")
    F["reliability_all"] = rel.groupby(["condition", "mode"])[["compliant_first", "regenerated", "director_rewrite", "residual_violation", "n_turns"]].mean().reset_index().to_dict(orient="records")
    ext = pd.read_csv(os.path.join(R, "table_extraction.csv"))
    F["extraction_by_model"] = ext.groupby("model")[["quote_recall", "attribution_acc", "fabricated_rate"]].mean().reset_index().to_dict(orient="records")
    F["extraction_overall"] = ext[["quote_recall", "attribution_acc", "fabricated_rate", "dropped_unmapped"]].mean().to_dict()
    F["identity"] = S.get("identity")
    F["attribution"] = S.get("attribution")
    F["attribution_exemplars"] = S.get("attribution_exemplars")
    for k in ("voice", "narrated"):  # improvement experiments (voice-preserving instruction, speak-then-narrate)
        F[k] = S.get(k)
    he = os.path.join(R, "human_eval_summary.json")  # expert ratings (scripts/analyze_human_eval.py), once available
    if os.path.exists(he):
        H = json.load(open(he, encoding="utf-8"))
        for k in ("speaker_human_judge_agreement", "speaker_id_judges_same_items"):
            if isinstance(H.get(k), dict):
                H[k] = [dict(judge=j, value=v) for j, v in H[k].items()]
        H["means_by_mode"] = {m["mode"]: {} for m in H.get("means", [])}
        for m in H.get("means", []):
            H["means_by_mode"][m["mode"]][m["metric"]] = m["mean"]
        H["alpha_range"] = dict(min=float(min(H["alpha"].values())), max=float(max(H["alpha"].values()))) if H.get("alpha") else None
        hj = [r["rho"] for r in H.get("human_vs_judges", []) if r["rho"] == r["rho"]]  # skip undefined (constant) correlations
        H["judge_rho_range"] = dict(min=float(min(hj)), max=float(max(hj))) if hj else None
        F["human_eval"] = H
    F["ablation"] = S.get("ablation")
    F["ablation_means"] = S.get("ablation_means")
    for k in ("judge_by_condition_mode", "judge_mode_tests", "judge_perspective_acc", "judge_agreement", "judge_auto_corr",
              "speaker_id", "speaker_id_tests", "speaker_subj_corr", "speaker_agreement", "source_reference", "source_reference_by_text",
              "speaker_id_exemplars"):
        F[k] = S.get(k)
    for k in ("judge_auto_corr", "speaker_subj_corr"):  # judge names contain dots: store as lists
        if isinstance(F.get(k), dict):
            F[k] = [dict(judge=j, **v) for j, v in F[k].items()]
    if F.get("judge_by_condition_mode"):
        jb = pd.DataFrame(F["judge_by_condition_mode"])
        crit = ["coherence", "character_consistency", "scene_fidelity", "subjectivity", "emotional_intensity"]
        F["judge_pooled"] = jb.groupby(["condition", "mode"])[crit].mean().reset_index().to_dict(orient="records")
        pa = pd.DataFrame(F.get("judge_perspective_acc") or [])
        if len(pa):
            F["judge_perspective_pooled"] = pa.groupby(["condition", "mode"]).ok.mean().reset_index().to_dict(orient="records")
    jp = os.path.join(R, "judge_session_all.csv")
    if os.path.exists(jp):  # judge ratings of the ablation conditions vs. matched full-framework sessions
        J = pd.read_csv(jp)
        J = J[(J.model == "llama3.1:8b") & J.seed.isin([0, 1, 2]) & J["mode"].isin(["first_person", "third_person"])]
        crit = ["coherence", "character_consistency", "scene_fidelity", "subjectivity", "emotional_intensity"]
        F["judge_ablation"] = J.groupby(["judge", "condition"])[crit].mean().reset_index().to_dict(orient="records")
    if F.get("speaker_id"):
        sid = pd.DataFrame(F["speaker_id"])
        F["speaker_id_pooled"] = sid.groupby("mode")[["acc", "chance"]].mean().reset_index().to_dict(orient="records")
    if F.get("judge_agreement"):
        al = [v["alpha"] for v in F["judge_agreement"].values()]
        F["judge_alpha_range"] = dict(min=float(min(al)), max=float(max(al)))
    json.dump(F, open(os.path.join(R, "facts.json"), "w", encoding="utf-8"), indent=1, default=float)
    print("facts written:", len(F), "keys")


if __name__ == "__main__":
    main()
