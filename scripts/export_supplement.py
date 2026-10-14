#!/usr/bin/env python3
"""Build the Supplemental Information document (LaTeX) from the analysis outputs:
extended statistical tables and example transcripts for every excerpt."""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

MODEL_LAB = {"llama3.1:8b": "Llama 3.1 8B", "qwen2.5:14b": "Qwen2.5 14B", "gemma3:12b": "Gemma 3 12B"}
MODE_LAB = {"first_person": "First", "third_person": "Third", "neutral": "Neutral", "mechanical_third": "Rule-based"}
TEXT_LAB = {"good_earth": "The Good Earth", "pride_prejudice": "Pride and Prejudice", "great_expectations": "Great Expectations",
            "scandal_bohemia": "A Scandal in Bohemia", "earnest": "The Importance of Being Earnest", "home_world": "The Home and the World"}


def esc(s):
    s = str(s)
    for a, b in [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"),
                 ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}"),
                 ("\u201c", "``"), ("\u201d", "''"), ("\u2019", "'"), ("\u2018", "`"), ("\u2014", "---"), ("\u2013", "--")]:
        s = s.replace(a, b)
    return s


def f(x, d=3):
    try:
        return "--" if x is None or not np.isfinite(float(x)) else f"{float(x):.{d}f}"
    except (TypeError, ValueError):
        return esc(x)


def table(df, cols, heads, caption, align=None, size="\\scriptsize", digits=None):
    digits = digits or {}
    align = align or "l" * len(cols)
    lines = [f"\\begin{{longtable}}{{{align}}}", f"\\caption{{{caption}}}\\\\", "\\toprule", " & ".join(heads) + " \\\\", "\\midrule", "\\endhead"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f(v, digits.get(c, 3)) if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool) else esc(v))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{longtable}"]
    return "{" + size + "\n" + "\n".join(lines) + "\n}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/supplement.tex")
    args = ap.parse_args()
    R = args.res
    S = json.load(open(os.path.join(R, "summary.json"), encoding="utf-8"))
    parts = []

    comp = pd.DataFrame(S["mode_comparisons"])
    comp["pair"] = comp.a.map(MODE_LAB) + " vs " + comp.b.map(MODE_LAB)
    parts.append("\\section*{S1. Paired comparisons of narrative modes (main experiment)}")
    parts.append("Session-level means were compared within excerpt--model--seed blocks (n = number of blocks). Diff = mean of A$-$B with 95\\% bootstrap CI; $p$: Wilcoxon signed-rank; $p_{Holm}$: Holm-adjusted within each pair of modes; $d_z$: Cohen's $d_z$; $r_{rb}$: matched-pairs rank-biserial correlation.")
    parts.append(table(comp, ["pair", "metric", "n", "mean_a", "mean_b", "diff", "ci_low", "ci_high", "dz", "r_rb", "p", "p_holm"],
                       ["Pair (A vs B)", "Measure", "n", "Mean A", "Mean B", "Diff", "CI low", "CI high", "$d_z$", "$r_{rb}$", "$p$", "$p_{Holm}$"],
                       "Paired comparisons between narrative modes.", "llrrrrrrrrrr", digits={"n": 0}))

    mm = pd.DataFrame(S["mixed_models"])
    parts.append("\\section*{S2. Linear mixed models}")
    parts.append("Measure $\\sim$ mode + generator model + (1 $|$ excerpt), REML. Coefficients are relative to the third-person mode.")
    parts.append(table(mm, ["metric", "term", "coef", "ci_low", "ci_high", "p"], ["Measure", "Mode", "Coef.", "CI low", "CI high", "$p$"],
                       "Fixed effects of narrative mode in linear mixed models.", "llrrrr"))

    eff = pd.DataFrame(S["effects_by_group"])
    eff["level"] = eff.level.map(lambda x: TEXT_LAB.get(x, MODEL_LAB.get(x, x)))
    parts.append("\\section*{S3. First$-$third differences by excerpt and by generator}")
    parts.append(table(eff, ["group", "level", "metric", "n", "mean_a", "mean_b", "diff", "ci_low", "ci_high", "dz", "p"],
                       ["Grouping", "Level", "Measure", "n", "First", "Third", "Diff", "CI low", "CI high", "$d_z$", "$p$"],
                       "First$-$third differences within each excerpt and each generator (unadjusted $p$).", "lllrrrrrrrr", digits={"n": 0}))

    ext = pd.read_csv(os.path.join(R, "table_extraction.csv"))
    parts.append("\\section*{S4. Director extraction quality}")
    parts.append(table(ext, ["text_id", "model", "gold_quotes", "extracted_raw", "kept", "quote_recall", "attribution_acc", "fabricated_rate", "raw_labels"],
                       ["Excerpt", "Model", "Gold", "Extracted", "Kept", "Recall", "Attribution", "Fabricated", "Raw speaker labels"],
                       "Dialogue extraction by the Director against manually attributed quotations.", "llrrrrrrp{4cm}",
                       digits={"gold_quotes": 0, "extracted_raw": 0, "kept": 0}))

    rel = pd.DataFrame(S["reliability"])
    parts.append("\\section*{S5. Pipeline reliability and narrative-mode enforcement}")
    parts.append(table(rel, ["condition", "model", "mode", "compliant_first", "regenerated", "director_rewrite", "residual_violation", "calls_calls", "calls_repaired", "calls_parse_fail", "n_turns"],
                       ["Condition", "Model", "Mode", "Compliant", "Regen.", "Rewritten", "Residual", "Calls", "Repaired", "Failed", "Turns"],
                       "Mean per session: share of turns compliant at the first attempt, regenerated, rewritten by the Director, and still violating the mode; LLM calls, JSON repairs and unrecoverable parse failures.",
                       "lllrrrrrrrr"))

    for name, title in (("table_identity.csv", "S6. Distributional character consistency"), ("table_attribution.csv", "S7. Cross-mode speaker attribution")):
        p = os.path.join(R, name)
        if os.path.exists(p):
            d = pd.read_csv(p)
            parts.append(f"\\section*{{{title}}}")
            cols = list(d.columns)
            parts.append(table(d, cols, [esc(c) for c in cols], esc(title), "l" * 3 + "r" * (len(cols) - 3)))

    for key, title in (("judge_agreement", "S8. Agreement between the LLM judges"), ("judge_auto_corr", "S9. Correlation between judge ratings and automatic measures")):
        if key in S:
            d = pd.DataFrame(S[key]).T.reset_index().rename(columns={"index": "item"})
            parts.append(f"\\section*{{{title}}}")
            parts.append(table(d, list(d.columns), [esc(c) for c in d.columns], esc(title), "l" + "r" * (len(d.columns) - 1)))

    # example transcripts: one block per excerpt (Qwen2.5 14B, seed 0, first vs third person)
    parts.append("\\section*{S10. Example transcripts}")
    parts.append("For every excerpt, the first-person and third-person sessions generated by Qwen2.5 14B with seed 0 are reproduced in full. All other transcripts are provided in the data files.")
    for tid, lab in TEXT_LAB.items():
        for mode in ("first_person", "third_person"):
            fp = os.path.join(args.runs, "sessions", "full", "qwen2.5_14b", tid, f"{mode}_s0.json")
            if not os.path.exists(fp):
                continue
            r = json.load(open(fp, encoding="utf-8"))
            parts.append(f"\\subsection*{{{esc(lab)} --- {MODE_LAB[mode]} person}}")
            parts.append("\\begin{enumerate}\\setlength{\\itemsep}{1pt}\\small")
            for t in r["turns"]:
                parts.append(f"\\item \\textbf{{{esc(t['speaker'])}}}: {esc(t['text'])}")
            parts.append("\\end{enumerate}")

    # improvement experiments: paired comparisons and example transcripts
    imp = [(c, V, lab) for c, V, lab in (("voice", "third_person_voice", "voice-preserving instruction"),
                                         ("narrated", "third_person_narrated", "speak-then-narrate")) if S.get(c)]
    if imp:
        parts.append("\\section*{S11. Improvement experiments: preserving character voice in third-person narration}")
        parts.append("Improvement 1 adds a voice-preserving instruction to the third-person mode; improvement 2 lets each Actor speak in its own "
                     "voice and has the Director narrate the speech in free indirect style (prompts in Appendix A.7 of the article). Both are compared "
                     "with the first-person and the standard third-person sessions of the same excerpts, models and seeds.")
        keep = ["fp_ratio", "subjectivity", "intensity", "polarity", "distilbert", "n_words", "word_len", "zipf", "mattr", "responsiveness", "compliant_first", "residual_violation"]
        for c, V, lab in imp:
            d = pd.DataFrame(S[c]["comparisons"])
            d = d[d.metric.isin(keep)]
            d["a"], d["b"] = d.a.map(lambda x: MODE_LAB.get(x, x)), d.b.map(lambda x: {"third_person_voice": "Voice-pres.", "third_person_narrated": "Narrated"}.get(x, MODE_LAB.get(x, x)))
            parts.append(table(d, ["metric", "a", "b", "n", "mean_a", "mean_b", "diff", "ci_low", "ci_high", "dz", "p_holm"],
                               ["Measure", "A", "B", "n", "Mean A", "Mean B", "A$-$B", "CI low", "CI high", "$d_z$", "$p_\\mathrm{Holm}$"],
                               f"Paired comparisons for the {lab} experiment (blocks: excerpt $\\times$ model $\\times$ seed).", "lllrrrrrrrr"))
        for c, V, lab in imp:
            for tid in ("good_earth", "pride_prejudice"):
                fp = os.path.join(args.runs, "sessions", c, "qwen2.5_14b", tid, f"{V}_s0.json")
                if not os.path.exists(fp):
                    continue
                r = json.load(open(fp, encoding="utf-8"))
                parts.append(f"\\subsection*{{{esc(TEXT_LAB[tid])} --- {lab} (Qwen2.5 14B, seed 0)}}")
                parts.append("\\begin{enumerate}\\setlength{\\itemsep}{1pt}\\small")
                for t in r["turns"]:
                    sp = f" \\emph{{(Actor's speech: {esc(t['speech'])})}}" if t.get("speech") else ""
                    parts.append(f"\\item \\textbf{{{esc(t['speaker'])}}}: {esc(t['text'])}{sp}")
                parts.append("\\end{enumerate}")

    # expert evaluation (scripts/analyze_human_eval.py)
    hp = os.path.join(R, "human_eval_summary.json")
    if os.path.exists(hp):
        H = json.load(open(hp, encoding="utf-8"))
        lab = {"first_person": "First person", "third_person": "Third person", "third_person_narrated": "Speak-then-narrate"}
        parts.append("\\section*{S12. Expert evaluation}")
        parts.append(f"{H['n_raters']} raters with master's degrees in English literature or English drama rated {H['n_sessions']} sessions "
                     "(four excerpts $\\times$ first-person, standard third-person and speak-then-narrate condition; seed 0, first eight turns) "
                     f"and identified the speakers of {H['n_turns']} masked turns (turns also rated by the LLM judges), blind to the generating model and the condition. "
                     "The responses were entered into the rating sheets by one of the authors, who was not blind to the conditions. "
                     "Condition means are averages over the four excerpts of the per-session means over raters (95\\% bootstrap CIs over excerpts). "
                     "Item-level ratings are provided in the data files (human\\_eval\\_session\\_ratings.csv, human\\_eval\\_speaker\\_guesses.csv).")
        m = pd.DataFrame(H["means"])
        m["mode"] = m["mode"].map(lambda x: lab.get(x, x))
        m["ci_low"], m["ci_high"] = m.ci.map(lambda c: c[0]), m.ci.map(lambda c: c[1])
        parts.append(table(m, ["mode", "metric", "n", "mean", "ci_low", "ci_high"], ["Condition", "Scale", "Excerpts", "Mean", "CI low", "CI high"],
                           "Expert ratings by condition (1--5 scales).", "llrrrr", digits={"n": 0, "mean": 2, "ci_low": 2, "ci_high": 2}))
        rows = [dict(item=f"Krippendorff's alpha (ordinal), {k}", value=v) for k, v in H.get("alpha", {}).items()]
        rows.append(dict(item="Krippendorff's alpha (nominal), speaker choice", value=H.get("speaker_alpha_nominal")))
        rows += [dict(item=f"Narrative form classified as instructed, {lab.get(k, k)}", value=v) for k, v in H.get("form_accuracy", {}).items()]
        rows += [dict(item=f"Spearman rho with {r['judge']}, {r['metric']} (n = {r['n']})", value=r["rho"]) for r in H.get("human_vs_judges", [])]
        rows += [dict(item=f"Spearman rho with automatic {k} (n = {H['n_sessions']})", value=v) for k, v in H.get("human_vs_auto", {}).items()]
        parts.append(table(pd.DataFrame(rows), ["item", "value"], ["Statistic", "Value"],
                           "Agreement between raters, narrative-form classification and correlations with the LLM judges and the automatic measures (undefined correlations: constant judge ratings).",
                           "lr", digits={"value": 2}))
        sp = pd.DataFrame(H["speaker_id"]).merge(pd.DataFrame(H["speaker_id_majority"])[["mode", "acc", "k", "n", "p_above_chance"]], on="mode", suffixes=("", "_maj"))
        sp["mode"] = sp["mode"].map(lambda x: lab.get(x, x))
        parts.append(table(sp, ["mode", "n", "acc", "chance", "p_above_chance", "k", "n_maj", "p_above_chance_maj"],
                           ["Condition", "Judgements", "Accuracy", "Chance", "$p$", "Majority correct", "Turns", "$p$ (majority)"],
                           "Masked speaker identification by the experts: all rater judgements pooled, and the majority vote per turn (one-sided binomial tests against chance).",
                           "lrrrrrrr", digits={"n": 0, "k": 0, "n_maj": 0, "acc": 2, "chance": 2}))
        J = H.get("speaker_id_judges_same_items") or {}
        if J:
            parts.append(f"Accuracy of the LLM judges on the same {H['n_turns']} masked turns: " + "; ".join(f"{esc(j)}: {v:.2f}" for j, v in J.items()) + ".")

    doc = ("\\documentclass[10pt]{article}\n\\usepackage[margin=2cm]{geometry}\n\\usepackage{booktabs,longtable,array}\n\\usepackage[T1]{fontenc}\n"
           "\\usepackage{times}\n\\usepackage[hidelinks]{hyperref}\n\\setlength{\\LTcapwidth}{\\textwidth}\n"
           "\\title{Supplemental Information\\\\[4pt]\\large An LLM-driven multi-agent simulation framework for quantitative analysis of narrative voice shifts in literary texts}\n"
           "\\author{Lili Zhao and Hui Shi}\n\\date{}\n\\begin{document}\n\\maketitle\n" + "\n\n".join(parts) + "\n\\end{document}\n")
    open(args.out, "w", encoding="utf-8").write(doc)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
