#!/usr/bin/env python3
"""Statistical analysis of the featurised sessions; writes CSV tables and summary.json."""

import argparse
import itertools
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from narrsim.stats import cell_consistency, comparison_table, holm, jsd, mixed_model, paired_test  # noqa: E402

MANIP = ["fp_ratio", "fp_rate", "tp_rate", "self_name"]
OUTCOME = ["subjectivity", "intensity", "polarity", "vader", "distilbert", "n_words", "word_len", "zipf", "rare_rate", "mattr", "mtld"]
INTERACT = ["turn_balance", "responsiveness", "self_succession", "repetition", "distinct2", "n_turns"]
COMPLY = ["compliant_first", "regenerated", "director_rewrite", "residual_violation"]
ZIPF_BANDS = [0, 2, 3, 4, 5, 6, 99]


def jdump(obj, path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, pd.DataFrame):
            return o.to_dict(orient="records")
        raise TypeError(type(o))
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=1, default=conv)


def read_jsonl(path):
    """Robust JSON-lines reader (pandas' ujson parser rejects some valid lines)."""
    with open(path, encoding="utf-8") as fh:
        return pd.DataFrame([json.loads(line) for line in fh if line.strip()])


# ----------------------------------------------------------------------------- identity
def char_dists(tokens, sents, zipf_of):
    L = Counter(min(len(w), 15) for w in tokens)
    F = Counter(np.digitize([zipf_of(w) for w in tokens], ZIPF_BANDS[1:-1]).tolist())
    S = Counter(sents)
    U = Counter(tokens)
    norm = lambda c: {k: v / sum(c.values()) for k, v in c.items()} if c else {}
    return norm(L), norm(F), norm(S), norm(U)


def consistency(a, b):
    js_len, js_freq, js_sent, js_lex = (jsd(a[i], b[i]) for i in range(4))
    return dict(js_len=js_len, js_freq=js_freq, js_sent=js_sent, js_lex=js_lex,
                score=1 - (0.3 * js_len + 0.4 * js_freq + 0.3 * js_sent))


def subsampled(ta, sa, tb, sb, zipf_of, rng, n_tok=120, n_utt=8, draws=20):
    vals = []
    for _ in range(draws):
        A = char_dists(list(rng.choice(ta, n_tok, replace=False)), list(rng.choice(sa, min(n_utt, len(sa)), replace=False)), zipf_of)
        B = char_dists(list(rng.choice(tb, n_tok, replace=False)), list(rng.choice(sb, min(n_utt, len(sb)), replace=False)), zipf_of)
        vals.append(consistency(A, B))
    return {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}


def identity_analysis(utt, modes=("first_person", "third_person"), min_tok=120):
    from wordfreq import zipf_frequency
    zcache = {}

    def zipf_of(w):
        if w not in zcache:
            zcache[w] = zipf_frequency(w, "en")
        return zcache[w]

    rng = np.random.default_rng(7)
    u = utt[(utt.condition == "full") & utt["mode"].isin(modes)].copy()
    u["content_tokens"] = u["content_tokens"].fillna("").astype(str)
    rows = []
    for (tid, model), g in u.groupby(["text_id", "model"]):
        pools = {}
        for (spk, mode), h in g.groupby(["speaker", "mode"]):
            toks = " ".join(h.content_tokens).split()
            pools[(spk, mode)] = (np.array(toks), np.array(h.sent_class.tolist()))
        chars = sorted({s for s, m in pools if all(len(pools.get((s, mm), ([], []))[0]) >= min_tok for mm in modes)})
        for c in chars:
            within = subsampled(*pools[(c, modes[0])], *pools[(c, modes[1])], zipf_of, rng, n_tok=min_tok)
            betw = []
            for m in modes:
                for c2 in chars:
                    if c2 != c:
                        betw.append(subsampled(*pools[(c, m)], *pools[(c2, m)], zipf_of, rng, n_tok=min_tok))
            if not betw:
                continue
            row = dict(text_id=tid, model=model, character=c, n_chars=len(chars))
            for k in within:
                row[f"within_{k}"] = within[k]
                row[f"between_{k}"] = float(np.mean([b[k] for b in betw]))
            rows.append(row)
    df = pd.DataFrame(rows)
    summ = {}
    if len(df):
        for k in ["score", "js_len", "js_freq", "js_sent", "js_lex"]:
            w, b = df[f"within_{k}"], df[f"between_{k}"]
            test = st.wilcoxon(w, b) if len(df) >= 5 else None
            summ[k] = dict(n=len(df), within=float(w.mean()), between=float(b.mean()),
                           frac_within_better=float(np.mean(w > b) if k == "score" else np.mean(w < b)),
                           p=float(test.pvalue) if test else None)
    return df, summ


# ----------------------------------------------------------------------------- attribution
FUNCTION_WORDS = ("the a an and or but if then than so as of in on at to for with by from about into over after before "
                  "not no never very too more most much many some any all each every only just even still yet already "
                  "is are was were be been being am have has had do does did will would shall should can could may might must "
                  "this that these those there here what which who whom whose when where why how because while though although "
                  "up down out off again once now soon always").split()


def speaker_attribution(utt, modes=("first_person", "third_person")):
    """Can a character's turns be attributed to the right speaker when a classifier trained on
    one narrative mode is applied to the other? Balanced accuracy vs. chance (1/K)."""
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    u = utt[(utt.condition == "full") & utt["mode"].isin(modes)].copy()
    u["content_tokens"] = u["content_tokens"].fillna("").astype(str)
    u["fw_text"] = u["text"].fillna("").str.lower()
    feats = {
        "content": lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        "function": lambda: CountVectorizer(vocabulary=FUNCTION_WORDS, token_pattern=r"(?u)\b[a-z']+\b"),
    }
    rows = []
    for (tid, model), g in u.groupby(["text_id", "model"]):
        counts = g.groupby("speaker").size()
        keep = counts[counts >= 10].index
        g = g[g.speaker.isin(keep)]
        if g.speaker.nunique() < 2:
            continue
        k = g.speaker.nunique()
        for fname, mk in feats.items():
            col = "content_tokens" if fname == "content" else "fw_text"

            def model_():
                vec = mk()
                clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
                if fname == "function":
                    return make_pipeline(vec, StandardScaler(with_mean=False), clf)
                return make_pipeline(vec, clf)

            res = dict(text_id=tid, model=model, features=fname, n_chars=k, chance=1 / k)
            for m in modes:  # within mode: leave-one-seed-out
                d = g[g["mode"] == m]
                preds, gold = [], []
                for s in sorted(d.seed.unique()):
                    tr, te = d[d.seed != s], d[d.seed == s]
                    if tr.speaker.nunique() < 2 or te.empty:
                        continue
                    try:
                        clf = model_().fit(tr[col], tr.speaker)
                        preds += list(clf.predict(te[col]))
                        gold += list(te.speaker)
                    except ValueError:
                        pass
                res[f"within_{m}"] = balanced_accuracy_score(gold, preds) if gold else np.nan
            for a, b in (modes, modes[::-1]):
                tr, te = g[g["mode"] == a], g[g["mode"] == b]
                try:
                    clf = model_().fit(tr[col], tr.speaker)
                    res[f"cross_{a}_to_{b}"] = balanced_accuracy_score(te.speaker, clf.predict(te[col]))
                except ValueError:
                    res[f"cross_{a}_to_{b}"] = np.nan
            rows.append(res)
    df = pd.DataFrame(rows)
    summ = {}
    for fname, d in df.groupby("features"):
        cross = d[[c for c in d.columns if c.startswith("cross_")]].mean(axis=1)
        within = d[[c for c in d.columns if c.startswith("within_")]].mean(axis=1)
        summ[fname] = dict(n_cells=len(d), chance=float(d.chance.mean()), within=float(within.mean()), cross=float(cross.mean()),
                           **{c: float(d[c].mean()) for c in d.columns if c.startswith(("within_", "cross_"))},
                           p_cross_vs_chance=float(st.wilcoxon(cross - d.chance, alternative="greater").pvalue) if len(d) >= 5 else None,
                           p_cross_vs_within=float(st.wilcoxon(cross - within).pvalue) if len(d) >= 5 else None,
                           cells_cross_above_chance=int((cross > d.chance).sum()))
    return df, summ


# ----------------------------------------------------------------------------- improvement experiment
def voice_analysis(ses, utt, J=None, S=None, cond="voice", V="third_person_voice"):
    """Improvement experiments (condition `cond`, mode `V`: voice-preserving instruction or speak-then-narrate)
    against the standard third-person mode; both are compared with the same first-person sessions
    (same excerpt, model and seed)."""
    FP, TP = "first_person", "third_person"
    vs = ses[(ses.condition == cond) & (ses["mode"] == V)]
    if vs.empty:
        return None
    keys = ["text_id", "model", "seed"]
    idx = pd.MultiIndex.from_frame(vs[keys])
    base = ses[(ses.condition == "full") & ses["mode"].isin([FP, TP])]
    both = pd.concat([base, vs.assign(condition="full")])
    both = both[pd.MultiIndex.from_frame(both[keys]).isin(idx)]
    out = dict(n_sessions=int(len(vs)), n_turns=int((utt.condition == cond).sum()),
               models=sorted(vs.model.unique().tolist()), seeds=sorted(vs.seed.unique().tolist()))
    out["comparisons"] = comparison_table(both, MANIP + OUTCOME + INTERACT + COMPLY, [(FP, V), (TP, V), (FP, TP)])
    out["means"] = both.groupby("mode")[MANIP + OUTCOME + INTERACT + COMPLY].mean().reset_index()
    out["reliability_by_model"] = both.groupby(["model", "mode"])[COMPLY].mean().reset_index()
    # does voice preservation shrink the first/third gap? paired on |FP - TP| vs |FP - TPvoice| per block
    gaps = []
    for m in ["subjectivity", "intensity", "polarity", "word_len", "zipf", "mattr", "n_words"]:
        w = both.pivot_table(index=keys, columns="mode", values=m).dropna()
        g_std, g_voice = (w[FP] - w[TP]).abs(), (w[FP] - w[V]).abs()
        r = dict(metric=m, n=len(w), gap_standard=float(g_std.mean()), gap_voice=float(g_voice.mean()),
                 signed_standard=float((w[FP] - w[TP]).mean()), signed_voice=float((w[FP] - w[V]).mean()),
                 frac_smaller=float(np.mean(g_voice < g_std)))
        r["p"] = float(st.wilcoxon(g_std, g_voice).pvalue) if len(w) >= 5 and not np.allclose(g_std, g_voice) else 1.0
        gaps.append(r)
    gaps = pd.DataFrame(gaps)
    gaps["p_holm"] = holm(gaps.p.tolist())
    out["gap_change"] = gaps

    # character identity and cross-mode speaker attribution
    u = utt[utt.condition.isin(["full", cond]) & utt["mode"].isin([FP, TP, V])].assign(condition="full")
    u = u[pd.MultiIndex.from_frame(u[keys]).isin(idx)]
    a_std, s_std = speaker_attribution(u, modes=(FP, TP))
    a_voi, s_voi = speaker_attribution(u, modes=(FP, V))
    out["attribution"] = {"standard": s_std, "voice": s_voi}
    cell = ["text_id", "model", "features"]
    a_std["cross"] = a_std[[c for c in a_std.columns if c.startswith("cross_")]].mean(axis=1)
    a_voi["cross"] = a_voi[[c for c in a_voi.columns if c.startswith("cross_")]].mean(axis=1)
    a_std["within_tp"], a_voi["within_tp"] = a_std[f"within_{TP}"], a_voi[f"within_{V}"]
    mg = a_std[cell + ["cross", "within_tp", "chance"]].merge(a_voi[cell + ["cross", "within_tp"]], on=cell, suffixes=("_std", "_voice"))
    tests = {}
    for f, d in mg.groupby("features"):
        d = d.dropna()
        tests[f] = dict(n_cells=len(d), chance=float(d.chance.mean()),
                        cross_std=float(d.cross_std.mean()), cross_voice=float(d.cross_voice.mean()),
                        within_tp_std=float(d.within_tp_std.mean()), within_tp_voice=float(d.within_tp_voice.mean()),
                        cells_voice_better=int((d.cross_voice > d.cross_std).sum()),
                        p=float(st.wilcoxon(d.cross_voice, d.cross_std).pvalue) if len(d) >= 5 else None)
    out["attribution_paired"] = tests
    i_std, si_std = identity_analysis(u, modes=(FP, TP))
    i_voi, si_voi = identity_analysis(u, modes=(FP, V))
    out["identity"] = {"standard": si_std, "voice": si_voi}
    if len(i_std) and len(i_voi):
        mi = i_std.merge(i_voi, on=["text_id", "model", "character"], suffixes=("_std", "_voice"))
        if len(mi) >= 5:
            out["identity_paired"] = dict(n=len(mi), within_std=float(mi.within_score_std.mean()), within_voice=float(mi.within_score_voice.mean()),
                                          p=float(st.wilcoxon(mi.within_score_voice, mi.within_score_std).pvalue))
    rows = pd.concat([a_std.assign(variant="standard"), a_voi.assign(variant=cond)])
    # robustness: the same attribution test without turns that still violate the third-person form
    if "residual_violation" in u.columns:
        uc = u[~u.residual_violation.fillna(False).astype(bool)]
        out["attribution_compliant_only"] = {"standard": speaker_attribution(uc, modes=(FP, TP))[1], "variant": speaker_attribution(uc, modes=(FP, V))[1]}

    # LLM judges: session ratings and masked speaker identification
    crit = ["coherence", "character_consistency", "scene_fidelity", "subjectivity", "emotional_intensity"]
    if J is not None and (J.condition == cond).any():
        Jb = pd.concat([J[(J.condition == "full") & J["mode"].isin([FP, TP])], J[J.condition == cond].assign(condition="full")])
        jt = []
        for judge, g in Jb.groupby("judge"):
            for m in crit:
                for a, b in [(TP, V), (FP, V)]:
                    r = paired_test(g, m, a, b)
                    r["judge"] = judge
                    jt.append(r)
        out["judge_tests"] = pd.DataFrame(jt)
        jv = J[J.condition == cond]
        out["judge_perspective"] = [dict(judge=j, third_share=float((g.perspective == "third_person_narration").mean()), n=len(g))
                                    for j, g in jv.groupby("judge")]
    if S is not None and (S.condition == cond).any():
        sv = S[S.condition == cond]
        st_ = S[(S.condition == "full") & (S["mode"] == TP) & S.seed.isin(sv.seed.unique())]
        sid = []
        for judge in sorted(sv.judge.unique()):
            a, b = st_[st_.judge == judge], sv[sv.judge == judge]
            tab = [[int(a.correct.sum()), int(len(a) - a.correct.sum())], [int(b.correct.sum()), int(len(b) - b.correct.sum())]]
            sid.append(dict(judge=judge, acc_standard=float(a.correct.mean()), n_standard=len(a), acc_voice=float(b.correct.mean()),
                            n_voice=len(b), chance_voice=float(b.chance.mean()), p_fisher=float(st.fisher_exact(tab).pvalue)))
        out["speaker_id"] = pd.DataFrame(sid)
    return out, rows


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    args = ap.parse_args()
    R = args.res
    ses = pd.read_csv(os.path.join(R, "sessions.csv"))
    utt = pd.read_csv(os.path.join(R, "utterances.csv"))
    out = {}

    main_s = ses[ses.condition == "full"]
    out["design"] = dict(
        n_sessions_total=int(len(ses[~ses["mode"].eq("mechanical_third")])),
        n_turns_total=int(utt[~utt["mode"].eq("mechanical_third")].shape[0]),
        main=main_s.groupby("mode").size().to_dict(),
        texts=sorted(main_s.text_id.unique().tolist()), models=sorted(main_s.model.unique().tolist()),
        seeds=sorted(main_s.seed.unique().tolist()),
        by_condition=ses.groupby(["condition", "mode"]).size().rename("n").reset_index().to_dict(orient="records"))

    # 1) mode comparisons (paired within text x model x seed)
    pairs = [("first_person", "third_person"), ("first_person", "neutral"), ("third_person", "neutral"),
             ("first_person", "mechanical_third"), ("mechanical_third", "third_person")]
    comp = comparison_table(main_s, MANIP + OUTCOME, pairs)
    comp.to_csv(os.path.join(R, "table_mode_comparisons.csv"), index=False)
    out["mode_comparisons"] = comp

    means = main_s.groupby("mode")[MANIP + OUTCOME + INTERACT + COMPLY].agg(["mean", "std"])
    means.columns = [f"{a}_{b}" for a, b in means.columns]
    means.reset_index().to_csv(os.path.join(R, "table_mode_means.csv"), index=False)
    out["mode_means"] = means.reset_index()

    # 2) mixed models and cell-level direction consistency
    mm = []
    for m in OUTCOME:
        try:
            mm += mixed_model(main_s[main_s["mode"].isin(["first_person", "third_person", "neutral"])], m)
        except Exception as e:
            mm.append(dict(metric=m, term="error", coef=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, err=str(e)))
    out["mixed_models"] = pd.DataFrame(mm)
    pd.DataFrame(mm).to_csv(os.path.join(R, "table_mixed_models.csv"), index=False)
    cc = [cell_consistency(main_s, m, "first_person", "third_person") for m in MANIP + OUTCOME]
    out["cell_consistency"] = pd.DataFrame(cc)

    # per-text and per-model FP-TP effects (for forest plots)
    eff = []
    for keyname in ("text_id", "model"):
        for key, g in main_s.groupby(keyname):
            for m in ["subjectivity", "intensity", "polarity", "n_words", "word_len", "zipf", "mattr", "fp_ratio"]:
                r = paired_test(g, m, "first_person", "third_person")
                r.update(group=keyname, level=key)
                eff.append(r)
    eff = pd.DataFrame(eff)
    eff.to_csv(os.path.join(R, "table_effects_by_group.csv"), index=False)
    out["effects_by_group"] = eff

    # 3) reliability / compliance of the pipeline
    rel = ses[ses["mode"] != "mechanical_third"].groupby(["condition", "model", "mode"])[
        COMPLY + ["calls_calls", "calls_repaired", "calls_parse_fail", "n_turns"]].mean().reset_index()
    rel.to_csv(os.path.join(R, "table_reliability.csv"), index=False)
    out["reliability"] = rel
    tot = ses[ses["mode"] != "mechanical_third"][["calls_calls", "calls_repaired", "calls_parse_fail"]].sum()
    out["json_stats"] = dict(calls=int(tot.calls_calls), repaired=int(tot.calls_repaired), parse_fail=int(tot.calls_parse_fail),
                             repaired_rate=float(tot.calls_repaired / max(tot.calls_calls, 1)),
                             fail_rate=float(tot.calls_parse_fail / max(tot.calls_calls, 1)),
                             termination=ses[ses["mode"] != "mechanical_third"].termination.value_counts().to_dict())

    # 4) character identity across modes
    idf, isum = identity_analysis(utt)
    idf.to_csv(os.path.join(R, "table_identity.csv"), index=False)
    out["identity"] = isum
    try:
        adf, asum = speaker_attribution(utt)
        adf.to_csv(os.path.join(R, "table_attribution.csv"), index=False)
        out["attribution"] = asum
    except ImportError as e:
        out["attribution"] = {"error": str(e)}

    # 5) ablations: each condition is paired with the full-framework sessions of the same
    #    excerpt, model, seed and mode (models/seeds differ between conditions)
    two = ses[ses["mode"].isin(["first_person", "third_person"])]
    arows, amrows = [], []
    for cond in ["no_memory", "uniform", "round_robin", "no_enforce", "single_prompt", "exemplars"]:
        sc = two[two.condition == cond]
        if sc.empty:
            continue
        sub = two[two.condition.isin(["full", cond]) & two.model.isin(sc.model.unique()) & two.seed.isin(sc.seed.unique())]
        for c in ["full", cond]:
            r = sub[sub.condition == c][INTERACT + COMPLY + ["subjectivity", "fp_ratio", "mattr", "n_words"]].mean().to_dict()
            amrows.append(dict(condition=c, reference_for=cond, models=",".join(sorted(sc.model.unique())), **r))
        for m in INTERACT + COMPLY + ["subjectivity", "fp_ratio", "mattr"]:
            r = paired_test(sub, m, "full", cond, by="condition", block=("text_id", "model", "seed", "mode"))
            arows.append(r)
        # does the ablation change the size of the FP-TP contrast?
        for m in ["subjectivity", "fp_ratio", "n_words"]:
            gaps = {}
            for c in ["full", cond]:
                w = sub[sub.condition == c].pivot_table(index=["text_id", "model", "seed"], columns="mode", values=m)
                gaps[c] = (w["first_person"] - w["third_person"]).rename(c)
            g = pd.concat(gaps.values(), axis=1).dropna()
            if len(g) >= 3:
                arows.append(dict(metric=f"gapFPminusTP_{m}", a="full", b=cond, n=len(g), mean_a=g["full"].mean(), mean_b=g[cond].mean(),
                                  diff=(g["full"] - g[cond]).mean(), p=st.wilcoxon(g["full"], g[cond]).pvalue if not np.allclose(g["full"], g[cond]) else 1.0))
    abl_t = pd.DataFrame(arows)
    abl_t.to_csv(os.path.join(R, "table_ablation.csv"), index=False)
    out["ablation"] = abl_t
    am = pd.DataFrame(amrows)
    am.to_csv(os.path.join(R, "table_ablation_means.csv"), index=False)
    out["ablation_means"] = am
    # identity under the voice-exemplar extension vs. the matched full-framework sessions
    if (utt.condition == "exemplars").any():
        ex_models = utt[utt.condition == "exemplars"].model.unique()
        ex_seeds = utt[utt.condition == "exemplars"].seed.unique()
        base = utt[(utt.condition == "full") & utt.model.isin(ex_models) & utt.seed.isin(ex_seeds)]
        exu = utt[utt.condition == "exemplars"].assign(condition="full")  # reuse the same routine
        out["attribution_exemplars"] = {"full_matched": speaker_attribution(base)[1], "exemplars": speaker_attribution(exu)[1]}
        pd.concat([speaker_attribution(base)[0].assign(variant="full_matched"), speaker_attribution(exu)[0].assign(variant="exemplars")]).to_csv(
            os.path.join(R, "table_attribution_exemplars.csv"), index=False)

    # 6) judges
    jfiles = [f for f in os.listdir(R) if f.startswith("judge_session_") and f.endswith(".jsonl")]
    if jfiles:
        J = pd.concat([read_jsonl(os.path.join(R, f)) for f in jfiles])
        J.to_csv(os.path.join(R, "judge_session_all.csv"), index=False)
        crit = ["coherence", "character_consistency", "scene_fidelity", "subjectivity", "emotional_intensity"]
        out["judge_by_condition_mode"] = J.groupby(["judge", "condition", "mode"])[crit].mean().reset_index()
        jm = J[J.condition == "full"]
        jt = []
        for judge, g in jm.groupby("judge"):
            for m in crit:
                for a, b in [("first_person", "third_person"), ("first_person", "neutral"), ("third_person", "neutral")]:
                    r = paired_test(g, m, a, b)
                    r["judge"] = judge
                    jt.append(r)
        out["judge_mode_tests"] = pd.DataFrame(jt)
        # perspective classification accuracy (manipulation check by an independent judge)
        target = {"first_person": "first_person_direct_speech", "third_person": "third_person_narration"}
        pj = J[J["mode"].isin(target)]
        out["judge_perspective_acc"] = pj.assign(ok=pj.apply(lambda r: r.perspective == target[r["mode"]], axis=1)).groupby(
            ["judge", "condition", "mode"]).ok.mean().reset_index()
        # inter-judge agreement
        if J.judge.nunique() >= 2:
            import krippendorff
            agree = {}
            piv = {m: J.pivot_table(index="key", columns="judge", values=m) for m in crit}
            for m, p in piv.items():
                p = p.dropna()
                agree[m] = dict(n=len(p), alpha=float(krippendorff.alpha(reliability_data=p.T.to_numpy(), level_of_measurement="ordinal")),
                                spearman=float(st.spearmanr(p.iloc[:, 0], p.iloc[:, 1]).correlation))
            out["judge_agreement"] = agree
        # validity of the automatic subjectivity measure at session level
        mrg = J.merge(ses[["file", "subjectivity", "intensity"]].rename(columns={"file": "key", "subjectivity": "auto_subj", "intensity": "auto_int"}), on="key")
        out["judge_auto_corr"] = {jd: dict(subj=float(st.spearmanr(g.subjectivity, g.auto_subj, nan_policy="omit").correlation),
                                          intensity=float(st.spearmanr(g.emotional_intensity, g.auto_int, nan_policy="omit").correlation), n=len(g))
                                  for jd, g in mrg.groupby("judge")}
    sfiles = [f for f in os.listdir(R) if f.startswith("judge_speaker_") and f.endswith(".jsonl")]
    if sfiles:
        S_all = pd.concat([read_jsonl(os.path.join(R, f)) for f in sfiles])
        S_all.to_csv(os.path.join(R, "judge_speaker_all.csv"), index=False)
        if (S_all.condition == "exemplars").any():  # extension vs. matched full-framework sessions (seeds 0-2)
            ex = S_all[S_all.condition.isin(["full", "exemplars"]) & S_all["mode"].isin(["first_person", "third_person"])]
            out["speaker_id_exemplars"] = ex.groupby(["judge", "condition", "mode"]).agg(
                acc=("correct", "mean"), chance=("chance", "mean"), n=("correct", "size")).reset_index()
        S = S_all[S_all.condition == "full"]
        out["speaker_id"] = S.groupby(["judge", "mode"]).agg(acc=("correct", "mean"), chance=("chance", "mean"), n=("correct", "size")).reset_index()
        tests = []
        for (judge, mode), g in S.groupby(["judge", "mode"]):
            b = st.binomtest(int(g.correct.sum()), len(g), float(g.chance.mean()), alternative="greater")
            tests.append(dict(judge=judge, mode=mode, k=int(g.correct.sum()), n=len(g), p_above_chance=b.pvalue))
        out["speaker_id_tests"] = pd.DataFrame(tests)
        u2 = utt[utt.condition == "full"].copy()
        u2["key"] = u2.apply(lambda r: f"sessions/full/{r.model.replace(':', '_')}/{r.text_id}/{r['mode']}_s{r.seed}.json#{int(r.turn) - 1}", axis=1)
        mrg = S.merge(u2[["key", "subjectivity"]].rename(columns={"subjectivity": "auto_subj"}), on="key", suffixes=("", "_auto"))
        out["speaker_subj_corr"] = {jd: dict(rho=float(st.spearmanr(g.subjectivity, g.auto_subj, nan_policy="omit").correlation), n=len(g))
                                    for jd, g in mrg.groupby("judge")}
        if S.judge.nunique() >= 2:
            p = S.pivot_table(index="key", columns="judge", values="guess", aggfunc="first").dropna()
            out["speaker_agreement"] = dict(n=len(p), raw_agreement=float((p.iloc[:, 0] == p.iloc[:, 1]).mean()))

    # 6b) improvement experiments: voice-preserving instruction and speak-then-narrate rendering
    for cond, V in (("voice", "third_person_voice"), ("narrated", "third_person_narrated")):
        vres = voice_analysis(ses, utt, J if jfiles else None, S_all if sfiles else None, cond=cond, V=V)
        if vres:
            out[cond], vrows = vres
            vrows.to_csv(os.path.join(R, f"table_attribution_{cond}.csv"), index=False)
            out[cond]["comparisons"].to_csv(os.path.join(R, f"table_{cond}_comparisons.csv"), index=False)

    # 7) human-authored reference (source excerpts)
    ref_path = os.path.join(R, "source_reference.csv")
    if os.path.exists(ref_path):
        ref = pd.read_csv(ref_path)
        out["source_reference"] = ref.groupby("kind")[["subjectivity", "intensity", "polarity", "n_words", "word_len", "zipf", "fp_ratio"]].mean().reset_index()
        out["source_reference_by_text"] = ref.groupby(["text_id", "kind"])[["subjectivity", "intensity", "n_words"]].mean().reset_index()

    jdump(out, os.path.join(R, "summary.json"))
    print("analysis written to", R)


if __name__ == "__main__":
    main()
