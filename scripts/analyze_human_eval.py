#!/usr/bin/env python3
"""Analyse the completed human-evaluation kits (one Excel file per rater, made by
submission_tools/make_human_eval_kit.py) and compare the experts with the LLM judges.

Usage: python scripts/analyze_human_eval.py --key HumanEval_Key.xlsx --kits rater1.xlsx rater2.xlsx [...] --res results

Writes results/human_eval_summary.json (also merged into results/facts.json under "human_eval" by
make_facts.py) and results/table_human_eval_*.csv:
  * inter-rater agreement (Krippendorff's alpha, ordinal) for every scale;
  * mean ratings per condition (items averaged over raters) with bootstrap CIs and paired Wilcoxon tests
    over the six excerpts (first vs third person, first person vs speak-then-narrate, third person vs
    speak-then-narrate);
  * accuracy of the narrative-form classification per condition;
  * Spearman correlations of the expert means with the LLM judges and with the automatic measures;
  * speaker identification: accuracy per mode against chance (binomial test), per rater and pooled, and
    agreement with the LLM judges on the same masked turns.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats as st

CRIT = {"Coherence (1-5)": "coherence", "Character consistency (1-5)": "character_consistency",
        "Scene fidelity (1-5)": "scene_fidelity", "Subjectivity (1-5)": "subjectivity", "Emotional intensity (1-5)": "emotional_intensity"}
CRIT_ZH = {"连贯性\n(1-5)": "coherence", "人物像不像\n(1-5)": "character_consistency", "个人色彩\n(1-5)": "subjectivity",
           "情绪强度\n(1-5)": "emotional_intensity"}  # short kit (make_human_eval_kit_short.py)
FORM = {"first": "first_person_direct_speech", "third": "third_person_narration", "mixed": "mixed",
        "第一人称": "first_person_direct_speech", "第三人称": "third_person_narration", "混合": "mixed"}
TARGET = {"first_person": "first_person_direct_speech", "third_person": "third_person_narration",
          "third_person_narrated": "third_person_narration"}
PAIRS = [("first_person", "third_person"), ("first_person", "third_person_narrated"), ("third_person", "third_person_narrated"),
         ("first_person", "neutral")]


def boot_ci(x, n=10000, seed=1):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return [np.nan, np.nan]
    r = np.random.default_rng(seed)
    m = r.choice(x, (n, len(x))).mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--kits", nargs="+", required=True)
    ap.add_argument("--res", default="results")
    args = ap.parse_args()
    ka = pd.read_excel(args.key, sheet_name="A_Sessions")
    kb = pd.read_excel(args.key, sheet_name="B_Speakers")
    A, B, raters = [], [], []
    for k, path in enumerate(args.kits):
        rater = f"R{k + 1}"
        sheets = pd.read_excel(path, sheet_name=None)
        sa = next(v for n, v in sheets.items() if n.startswith("A"))
        sb = next(v for n, v in sheets.items() if n.startswith("B"))
        sa, sb = (x.rename(columns={x.columns[0]: "ID"}) for x in (sa, sb))  # item IDs are always in the first column
        a = sa.rename(columns={**CRIT, **CRIT_ZH, "编号": "ID"})
        fcol = next(c for c in a.columns if str(c).startswith(("Narrative form", "叙述形式")))
        a["form"] = a[fcol].astype(str).str.strip().str.lower().map(FORM)
        crit = [c for c in CRIT.values() if c in a.columns]
        A.append(a[["ID", *crit, "form"]].assign(rater=rater))
        b = sb.rename(columns={"编号": "ID", "Speaker": "guess", "说话人（下拉选择）": "guess", "Subjectivity (1-5)": "subjectivity"})
        if "subjectivity" not in b:
            b["subjectivity"] = np.nan
        B.append(b[["ID", "guess", "subjectivity"]].assign(rater=rater))
        info = next((v for n, v in sheets.items() if n.startswith("评分者信息")), None)  # rater information (short kit)
        if info is not None:
            vals = dict(zip(info.iloc[:, 0].astype(str), info.iloc[:, 1]))
            decl = next((str(v).strip() for q, v in vals.items() if "独立完成" in q), "")
            raters.append(dict(rater=rater, degree=str(next((v for q, v in vals.items() if "学位" in q), "")), independent_no_ai=decl))
            if not decl.startswith(("是", "确认")):
                print(f"WARNING: {rater} ({path}) did not declare independent rating without AI tools: {decl!r}")
    A = pd.concat(A).merge(ka, on="ID")
    B = pd.concat(B).merge(kb, on="ID")
    CRIT_USED = [c for c in CRIT.values() if c in A.columns]
    for c in CRIT_USED:
        A[c] = pd.to_numeric(A[c], errors="coerce")
    B["subjectivity"] = pd.to_numeric(B["subjectivity"], errors="coerce")
    B["correct"] = B.guess.astype(str).str.strip().str.lower() == B.speaker.astype(str).str.strip().str.lower()
    out = dict(n_raters=len(args.kits), n_sessions=int(ka.shape[0]), n_turns=int(kb.shape[0]), raters=raters, scales=CRIT_USED)

    # agreement between raters
    if len(args.kits) >= 2:
        import krippendorff
        out["alpha"] = {c: float(krippendorff.alpha(reliability_data=A.pivot_table(index="rater", columns="ID", values=c).to_numpy(),
                                                     level_of_measurement="ordinal")) for c in CRIT_USED}
        sp = B.pivot_table(index="ID", columns="rater", values="guess", aggfunc="first")
        out["speaker_raw_agreement"] = float(np.mean([len(set(r.dropna())) == 1 for _, r in sp.iterrows()]))

    # condition means (items averaged over raters) and paired tests over excerpts
    item = A.groupby(["ID", "text_id", "mode"])[CRIT_USED].mean().reset_index()
    means, tests = [], []
    for mode, g in item.groupby("mode"):
        for c in CRIT_USED:
            means.append(dict(mode=mode, metric=c, mean=float(g[c].mean()), ci=boot_ci(g[c]), n=int(g[c].notna().sum())))
    for a, b in PAIRS:
        for c in CRIT_USED:
            w = item.pivot_table(index="text_id", columns="mode", values=c)
            if a in w and b in w:
                d = (w[a] - w[b]).dropna()
                p = float(st.wilcoxon(d).pvalue) if len(d) >= 5 and not np.allclose(d, 0) else None
                tests.append(dict(metric=c, a=a, b=b, n=len(d), mean_a=float(w[a].mean()), mean_b=float(w[b].mean()),
                                  diff=float(d.mean()), ci=boot_ci(d), p=p))
    out["means"], out["tests"] = means, tests
    fa = A[A["mode"].isin(TARGET)]
    out["form_accuracy"] = {m: float((g.form == TARGET[m]).mean()) for m, g in fa.groupby("mode")}

    # experts vs LLM judges and automatic measures (session level)
    jp = os.path.join(args.res, "judge_session_all.csv")
    if os.path.exists(jp):
        J = pd.read_csv(jp)
        J["file"] = J.key.str.replace("\\\\", "/", regex=True)
        cor = []
        for judge, g in J.groupby("judge"):
            m = item.merge(ka[["ID", "file"]], on="ID").merge(g, on="file", suffixes=("_human", "_judge"))
            for c in CRIT_USED:
                r = st.spearmanr(m[f"{c}_human"], m[f"{c}_judge"], nan_policy="omit")
                cor.append(dict(judge=judge, metric=c, n=len(m), rho=float(r.correlation), p=float(r.pvalue)))
        out["human_vs_judges"] = cor
    sp_ = os.path.join(args.res, "sessions.csv")
    if os.path.exists(sp_):
        ses = pd.read_csv(sp_)
        ses["file"] = ses.file.str.replace("\\\\", "/", regex=True)
        m = item.merge(ka[["ID", "file"]], on="ID").merge(ses[["file", "subjectivity", "intensity"]], on="file", suffixes=("_human", "_auto"))
        out["human_vs_auto"] = {"subjectivity": float(st.spearmanr(m["subjectivity_human"], m["subjectivity_auto"], nan_policy="omit").correlation),
                                "intensity": float(st.spearmanr(m["emotional_intensity"], m["intensity"], nan_policy="omit").correlation)}

    # speaker identification
    sid = []
    for (mode,), g in B.groupby(["mode"]):
        k, n = int(g.correct.sum()), int(g.correct.notna().sum())
        sid.append(dict(mode=mode, acc=k / n if n else np.nan, n=n, chance=float(g.chance.mean()),
                        p_above_chance=float(st.binomtest(k, n, float(g.chance.mean()), alternative="greater").pvalue) if n else None,
                        per_rater={r: float(h.correct.mean()) for r, h in g.groupby("rater")}))
    out["speaker_id"] = sid
    # raters' answers are not independent: also test the per-item majority vote (ties count as wrong)
    maj = []
    for (mode,), g in B.groupby(["mode"]):
        votes = g.groupby("ID").agg(correct=("correct", "sum"), n=("correct", "size"), chance=("chance", "first"))
        k, n = int((votes.correct * 2 > votes.n).sum()), len(votes)
        maj.append(dict(mode=mode, acc=k / n, k=k, n=n, chance=float(votes.chance.mean()),
                        p_above_chance=float(st.binomtest(k, n, float(votes.chance.mean()), alternative="greater").pvalue)))
    out["speaker_id_majority"] = maj
    if len(args.kits) >= 2:
        import krippendorff
        codes = {v: i for i, v in enumerate(sorted(B.guess.astype(str).unique()))}
        out["speaker_alpha_nominal"] = float(krippendorff.alpha(
            reliability_data=B.assign(code=B.guess.astype(str).map(codes)).pivot_table(index="rater", columns="ID", values="code").to_numpy(),
            level_of_measurement="nominal"))
    sjp = os.path.join(args.res, "judge_speaker_all.csv")
    if os.path.exists(sjp):
        SJ = pd.read_csv(sjp)
        SJ["judge_key"] = SJ.key.str.replace("\\\\", "/", regex=True)
        m = B.merge(SJ[["judge_key", "judge", "guess"]], on="judge_key", suffixes=("", "_judge"))
        out["speaker_human_judge_agreement"] = {j: float((g.guess.str.lower() == g.guess_judge.str.lower()).mean()) for j, g in m.groupby("judge")}
        out["speaker_id_judges_same_items"] = {j: float(g.drop_duplicates("judge_key").pipe(lambda d: (d.guess_judge.str.lower() == d.speaker.str.lower()).mean()))
                                               for j, g in m.groupby("judge")}
    # anonymized item-level ratings (rater codes only) for the data release
    A[["rater", "ID", "text_id", "model", "mode", "file", *CRIT_USED, "form"]].sort_values(["rater", "ID"]).to_csv(
        os.path.join(args.res, "human_eval_session_ratings.csv"), index=False)
    B[["rater", "ID", "text_id", "mode", "judge_key", "speaker", "guess", "correct"]].sort_values(["rater", "ID"]).to_csv(
        os.path.join(args.res, "human_eval_speaker_guesses.csv"), index=False)
    pd.DataFrame(means).to_csv(os.path.join(args.res, "table_human_eval_means.csv"), index=False)
    pd.DataFrame(tests).to_csv(os.path.join(args.res, "table_human_eval_tests.csv"), index=False)
    pd.DataFrame(sid).to_csv(os.path.join(args.res, "table_human_eval_speaker.csv"), index=False)
    json.dump(out, open(os.path.join(args.res, "human_eval_summary.json"), "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
