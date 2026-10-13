#!/usr/bin/env python3
"""Publication figures for the revised manuscript (PNG, 300 dpi, no titles inside images).

Colour encodes narrative mode consistently in every figure (Okabe-Ito, checked for
colour-vision deficiency); marker shape is a redundant second encoding.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

MODE_COL = {"first_person": "#0072B2", "third_person": "#D55E00", "neutral": "#009E73", "mechanical_third": "#CC79A7"}
MODE_MK = {"first_person": "o", "third_person": "s", "neutral": "^", "mechanical_third": "D"}
MODE_LAB = {"first_person": "First person", "third_person": "Third person", "neutral": "Neutral (no instruction)",
            "mechanical_third": "Rule-based transposition"}
TEXT_LAB = {"good_earth": "The Good Earth", "pride_prejudice": "Pride and Prejudice", "great_expectations": "Great Expectations",
            "scandal_bohemia": "A Scandal in Bohemia", "earnest": "The Importance of Being Earnest", "home_world": "The Home and the World"}
TEXT_ORDER = list(TEXT_LAB)
TEXT_MK = dict(zip(TEXT_ORDER, ["o", "s", "^", "D", "v", "P"]))
TEXT_COL = dict(zip(TEXT_ORDER, ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]))
MODEL_LAB = {"llama3.1:8b": "Llama 3.1 8B", "qwen2.5:14b": "Qwen2.5 14B", "gemma3:12b": "Gemma 3 12B"}
METRIC_LAB = {"subjectivity": "Subjectivity", "intensity": "Emotional intensity", "polarity": "Polarity (TextBlob)",
              "vader": "Polarity (VADER)", "distilbert": "Polarity (DistilBERT)", "n_words": "Utterance length",
              "word_len": "Content-word length", "zipf": "Content-word frequency (Zipf)", "rare_rate": "Rare-word rate",
              "mattr": "Lexical diversity (MATTR)", "mtld": "Lexical diversity (MTLD)", "fp_ratio": "First-person ratio"}
INK, MUTED, GRID = "#222222", "#666666", "#DDDDDD"
DIVERGE = LinearSegmentedColormap.from_list("fp_tp", ["#D55E00", "#F2F2F2", "#0072B2"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})


def panel_label(ax, s):
    ax.text(-0.02, 1.02, f"({s})", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right", color=INK)


def save(fig, out, name):
    fig.savefig(os.path.join(out, name))
    plt.close(fig)
    print("saved", name)


# --------------------------------------------------------------------------- Fig 3
def fig_effects(summ, out):
    comp = pd.DataFrame(summ["mode_comparisons"])
    metrics = ["subjectivity", "intensity", "polarity", "vader", "distilbert", "n_words", "word_len", "zipf", "rare_rate", "mattr", "mtld"]
    pairs = [("first_person", "third_person", "First − Third", "first_person"),
             ("first_person", "neutral", "First − Neutral", "neutral"),
             ("third_person", "neutral", "Third − Neutral", "third_person"),
             ("first_person", "mechanical_third", "First − Rule-based transposition", "mechanical_third")]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y0 = np.arange(len(metrics))[::-1]
    for k, (a, b, lab, ck) in enumerate(pairs):
        sub = comp[(comp.a == a) & (comp.b == b)].set_index("metric")
        off = (k - 1.5) * 0.18
        for yi, m in zip(y0, metrics):
            if m not in sub.index or not np.isfinite(sub.loc[m, "dz"]):
                continue
            r = sub.loc[m]
            sd = abs(r["diff"] / r["dz"]) if r["dz"] else np.nan
            lo, hi = (r.ci_low / sd, r.ci_high / sd) if sd and np.isfinite(sd) else (np.nan, np.nan)
            ax.plot([lo, hi], [yi + off] * 2, color=MODE_COL[ck], lw=1.6, solid_capstyle="round")
            ax.plot(r["dz"], yi + off, MODE_MK[ck], color=MODE_COL[ck], ms=5.5, mec="white", mew=0.8,
                    label=lab if yi == y0[0] else None)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y0)
    ax.set_yticklabels([METRIC_LAB[m] for m in metrics])
    ax.set_xlabel("Standardized paired difference (Cohen's $d_z$) with 95% bootstrap CI")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=2, fontsize=8, handletextpad=0.3)
    save(fig, out, "Fig3.png")


# --------------------------------------------------------------------------- Fig 4
def fig_cells(ses, out):
    m = ses[(ses.condition == "full") & ses["mode"].isin(["first_person", "third_person"])]
    metrics = ["fp_ratio", "subjectivity", "intensity", "polarity", "n_words", "word_len", "zipf", "mattr"]
    g = m.groupby(["text_id", "model", "mode"])[metrics].mean().unstack("mode")
    rows, labels = [], []
    for t in TEXT_ORDER:
        for mod in MODEL_LAB:
            if (t, mod) in g.index:
                rows.append([(g.loc[(t, mod), (k, "first_person")] - g.loc[(t, mod), (k, "third_person")]) for k in metrics])
                labels.append(f"{TEXT_LAB[t]} · {MODEL_LAB[mod]}")
    if not rows:
        return
    D = np.array(rows, dtype=float)
    sd = m.groupby("mode")[metrics].std().mean().to_numpy()
    Z = D / sd
    fig, ax = plt.subplots(figsize=(7.2, 0.26 * len(rows) + 1.4))
    lim = 2.5  # colour saturates at +-2.5 SD; exact values are printed in the cells
    im = ax.imshow(np.clip(Z, -lim, lim), cmap=DIVERGE, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if np.isfinite(D[i, j]):
                ax.text(j, i, f"{D[i, j]:+.2f}" if abs(D[i, j]) < 10 else f"{D[i, j]:+.0f}", ha="center", va="center",
                        fontsize=6.5, color="white" if abs(Z[i, j]) > 0.65 * lim else INK)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([METRIC_LAB[k] for k in metrics], rotation=35, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.grid(False)
    for i in range(3, len(labels), 3):
        ax.axhline(i - 0.5, color="white", lw=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, extend="both")
    cb.set_label("First − Third (in pooled SD units)\nblue: higher in first person", fontsize=7.5)
    save(fig, out, "Fig4.png")


# --------------------------------------------------------------------------- Fig 5
def fig_character_distributions(utt, out, text_id="good_earth"):
    from wordfreq import zipf_frequency
    u = utt[(utt.condition == "full") & (utt.text_id == text_id) & utt["mode"].isin(["first_person", "third_person"])].copy()
    u["content_tokens"] = u["content_tokens"].fillna("").astype(str)
    chars = [c for c in ["O-lan", "Wang Lung", "the eldest son", "the maid"] if c in set(u.speaker)]
    if not chars:
        return
    fig, axes = plt.subplots(2, len(chars), figsize=(7.4, 4.4), sharey="row")
    axes = np.atleast_2d(axes)
    letters = iter("abcdefgh")
    bands = ["<2", "2–3", "3–4", "4–5", "5–6", "≥6"]
    for j, c in enumerate(chars):
        for mode in ["first_person", "third_person"]:
            toks = " ".join(u[(u.speaker == c) & (u["mode"] == mode)].content_tokens).split()
            if not toks:
                continue
            L = np.array([min(len(w), 14) for w in toks])
            h = np.bincount(L, minlength=15)[1:] / len(L)
            off = -0.2 if mode == "first_person" else 0.2
            axes[0, j].bar(np.arange(1, 15) + off, h, width=0.38, color=MODE_COL[mode], label=MODE_LAB[mode])
            axes[0, j].axvline(L.mean(), color=MODE_COL[mode], ls="--", lw=1)
            zb = np.digitize([zipf_frequency(w, "en") for w in toks], [2, 3, 4, 5, 6])
            hz = np.bincount(zb, minlength=6) / len(zb)
            axes[1, j].bar(np.arange(6) + off, hz, width=0.38, color=MODE_COL[mode])
        axes[0, j].set_title(c if c[0].isupper() else c[0].upper() + c[1:], fontsize=9, color=INK)
        axes[0, j].set_xlabel("Length (characters)", fontsize=8)
        axes[1, j].set_xticks(range(6))
        axes[1, j].set_xticklabels(bands, fontsize=7)
        axes[1, j].set_xlabel("Zipf frequency band", fontsize=8)
        panel_label(axes[0, j], next(letters))
    for j in range(len(chars)):
        panel_label(axes[1, j], next(letters))
    axes[0, 0].set_ylabel("Proportion of content words")
    axes[1, 0].set_ylabel("Proportion of content words")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, "Fig5.png")


# --------------------------------------------------------------------------- Fig 6
def fig_subjectivity(ses, utt, out):
    m = ses[(ses.condition == "full")]
    modes = ["first_person", "third_person", "neutral"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw=dict(width_ratios=[1, 1.25]))
    data = [m[m["mode"] == k].subjectivity.dropna().to_numpy() for k in modes]
    parts = a1.violinplot(data, showmedians=True, widths=0.8)
    for pc, k in zip(parts["bodies"], modes):
        pc.set_facecolor(MODE_COL[k])
        pc.set_edgecolor(MODE_COL[k])
        pc.set_alpha(0.35)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color(MUTED)
        parts[key].set_linewidth(1)
    rng = np.random.default_rng(1)
    for i, (k, d) in enumerate(zip(modes, data)):
        a1.scatter(i + 1 + rng.uniform(-0.12, 0.12, len(d)), d, s=7, color=MODE_COL[k], marker=MODE_MK[k], lw=0, alpha=0.8)
    a1.set_xticks([1, 2, 3])
    a1.set_xticklabels(["First\nperson", "Third\nperson", "Neutral"])
    a1.set_ylabel("Session mean subjectivity (TextBlob)")
    panel_label(a1, "a")
    u = utt[(utt.condition == "full") & utt["mode"].isin(["first_person", "third_person"])]
    g = u.groupby(["text_id", "model", "speaker", "mode"]).subjectivity.mean().unstack("mode").dropna()
    mean, diff = (g.first_person + g.third_person) / 2, g.first_person - g.third_person
    for t in TEXT_ORDER:
        idx = g.index.get_level_values(0) == t
        a2.scatter(mean[idx], diff[idx], s=16, marker=TEXT_MK[t], color=TEXT_COL[t], label=TEXT_LAB[t], lw=0.4, edgecolor="white")
    md, sd = diff.mean(), diff.std()
    for v, ls in ((md, "-"), (md + 1.96 * sd, "--"), (md - 1.96 * sd, "--")):
        a2.axhline(v, color=MUTED, ls=ls, lw=1)
    a2.text(a2.get_xlim()[1], md, f" mean {md:+.3f}", va="bottom", ha="right", fontsize=7, color=MUTED)
    a2.set_xlabel("Mean subjectivity of a character (First and Third)")
    a2.set_ylabel("First − Third")
    a2.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0), handletextpad=0.2)
    panel_label(a2, "b")
    fig.tight_layout()
    save(fig, out, "Fig6.png")


# --------------------------------------------------------------------------- Fig 7
def fig_identity(res, out):
    p = os.path.join(res, "table_identity.csv")
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    if df.empty:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw=dict(width_ratios=[1, 1.35]))
    k = "score"
    for t in TEXT_ORDER:
        s = df[df.text_id == t]
        a1.scatter(s[f"between_{k}"], s[f"within_{k}"], marker=TEXT_MK[t], color=TEXT_COL[t], s=22, label=TEXT_LAB[t], lw=0.4, edgecolor="white")
    lo = min(df[f"between_{k}"].min(), df[f"within_{k}"].min())
    hi = max(df[f"between_{k}"].max(), df[f"within_{k}"].max())
    pad = 0.05 * (hi - lo)
    a1.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=1, ls="--")
    a1.set_xlim(lo - pad, hi + pad)
    a1.set_ylim(lo - pad, hi + pad)
    a1.set_xlabel("Consistency with other characters\n(same mode)")
    a1.set_ylabel("Consistency of the same character\n(first vs third person)")
    panel_label(a1, "a")
    ap = os.path.join(res, "table_attribution.csv")
    if os.path.exists(ap):
        A = pd.read_csv(ap)
        A = A[A.features == "content"].copy()
        A["within"] = A[["within_first_person", "within_third_person"]].mean(axis=1)
        A["cross"] = A[["cross_first_person_to_third_person", "cross_third_person_to_first_person"]].mean(axis=1)
        A["order"] = A.text_id.map({t: i for i, t in enumerate(TEXT_ORDER)})
        A = A.sort_values(["order", "model"]).reset_index(drop=True)
        x = np.arange(len(A))
        a2.vlines(x, A.chance, A.cross, color=GRID, lw=1)
        a2.scatter(x, A.within, marker="o", s=22, color=INK, label="Within mode", zorder=3)
        a2.scatter(x, A.cross, marker="s", s=22, color="#CC79A7", label="Across modes", zorder=3)
        a2.scatter(x, A.chance, marker="_", s=90, color=MUTED, label="Chance", zorder=3)
        short = {"good_earth": "Good Earth", "pride_prejudice": "Pride", "great_expectations": "Great Exp.",
                 "scandal_bohemia": "Scandal", "earnest": "Earnest", "home_world": "Home/World"}
        for i, t in enumerate(TEXT_ORDER):
            idx = np.where(A.text_id == t)[0]
            if len(idx):
                a2.text(idx.mean(), 0.03, short[t], ha="center", fontsize=6.3, color=MUTED)
                if idx.min() > 0:
                    a2.axvline(idx.min() - 0.5, color=GRID, lw=1.2)
        a2.set_xticks(x)
        a2.set_xticklabels([MODEL_LAB.get(m, m).split(" ")[0] for m in A.model], rotation=90, fontsize=6)
        a2.set_ylim(0, 1.02)
        a2.set_ylabel("Balanced accuracy of speaker attribution")
        a2.legend(fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, handletextpad=0.2, columnspacing=0.8)
        a2.grid(axis="x", visible=False)
        panel_label(a2, "b")
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, out, "Fig7.png")


# --------------------------------------------------------------------------- Fig 8
def fig_trajectory(utt, out):
    u = utt[(utt.condition == "full")]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for k in ["first_person", "third_person", "neutral"]:
        s = u[u["mode"] == k].groupby("turn").subjectivity
        mu, se, n = s.mean(), s.std() / np.sqrt(s.count()), s.count()
        mu, se = mu[n >= 20], se[n >= 20]
        ax.fill_between(mu.index, mu - 1.96 * se, mu + 1.96 * se, color=MODE_COL[k], alpha=0.15, lw=0)
        ax.plot(mu.index, mu, color=MODE_COL[k], lw=2, marker=MODE_MK[k], ms=4, label=MODE_LAB[k])
    ax.set_xlabel("Turn")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylabel("Mean subjectivity (95% CI)")
    ax.legend(ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    save(fig, out, "Fig8.png")


# --------------------------------------------------------------------------- Fig 9
def fig_ablation(ses, out):
    conds = ["full", "no_memory", "uniform", "round_robin", "no_enforce", "single_prompt"]
    lab = {"full": "Full", "no_memory": "k=1", "uniform": "Unif.", "round_robin": "Round\nrobin",
           "no_enforce": "No\nenf.", "single_prompt": "Single\nprompt"}
    abl_only = ses[ses.condition.isin(conds[1:])]
    seeds = sorted(abl_only.seed.unique())
    models = abl_only.model.unique()
    a = ses[ses.condition.isin(conds) & ses.seed.isin(seeds) & ses.model.isin(models) & ses["mode"].isin(["first_person", "third_person"])]
    conds = [c for c in conds if c in set(a.condition)]
    if len(conds) < 2:
        return
    panels = [("responsiveness", "Responsiveness"), ("turn_balance", "Turn balance"), ("self_succession", "Self-succession"),
              ("repetition", "Repetition"), ("compliant_first", "Compliant (1st attempt)"), ("residual_violation", "Residual violations")]
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.8))
    for ax, (m, title), letter in zip(axes.flat, panels, "abcdef"):
        for i, c in enumerate(conds):
            for j, mode in enumerate(["first_person", "third_person"]):
                v = a[(a.condition == c) & (a["mode"] == mode)][m].dropna()
                if v.empty:
                    continue
                x = i + (j - 0.5) * 0.36
                ax.bar(x, v.mean(), width=0.34, color=MODE_COL[mode], label=MODE_LAB[mode] if i == 0 else None)
                ci = 1.96 * v.std() / np.sqrt(len(v)) if len(v) > 1 else 0
                ax.plot([x, x], [v.mean() - ci, v.mean() + ci], color=INK, lw=0.9)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([lab[c] for c in conds], fontsize=7, rotation=0)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.grid(axis="x", visible=False)
        panel_label(ax, letter)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, "Fig9.png")


# --------------------------------------------------------------------------- Fig 10
def fig_judges(res, out):
    sj = os.path.join(res, "judge_session_all.csv")
    sp = os.path.join(res, "judge_speaker_all.csv")
    if not (os.path.exists(sj) and os.path.exists(sp)):
        return
    J = pd.read_csv(sj)
    S = pd.read_csv(sp)
    J = J[J.condition == "full"]
    S = S[S.condition == "full"]
    crit = ["coherence", "character_consistency", "scene_fidelity", "subjectivity", "emotional_intensity"]
    clab = ["Coherence", "Character\nconsistency", "Scene\nfidelity", "Subjectivity", "Emotional\nintensity"]
    modes = ["first_person", "third_person", "neutral"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.6), gridspec_kw=dict(width_ratios=[1.7, 1]))
    for j, k in enumerate(modes):
        v = J[J["mode"] == k][crit]
        mu = v.mean().to_numpy()
        ci = 1.96 * v.std().to_numpy() / np.sqrt(v.count().to_numpy())
        x = np.arange(len(crit)) + (j - 1) * 0.26
        a1.bar(x, mu, width=0.24, color=MODE_COL[k], label=MODE_LAB[k])
        a1.errorbar(x, mu, yerr=ci, fmt="none", ecolor=INK, elinewidth=0.9)
    a1.set_xticks(range(len(crit)))
    a1.set_xticklabels(clab, fontsize=7)
    a1.set_ylim(1, 5)
    a1.set_ylabel("Mean rating (both judges, 95% CI)")
    a1.grid(axis="x", visible=False)
    a1.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    panel_label(a1, "a")
    g = S.groupby("mode").agg(acc=("correct", "mean"), chance=("chance", "mean"), n=("correct", "size")).reindex(modes).dropna()
    for i, (k, r) in enumerate(g.iterrows()):
        se = np.sqrt(r.acc * (1 - r.acc) / r.n)
        a2.bar(i, r.acc, width=0.6, color=MODE_COL[k])
        a2.plot([i, i], [r.acc - 1.96 * se, r.acc + 1.96 * se], color=INK, lw=0.9)
        a2.plot([i - 0.35, i + 0.35], [r.chance] * 2, color=INK, ls="--", lw=1)
    a2.legend([plt.Line2D([], [], color=INK, ls="--", lw=1)], ["Chance level"], fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    a2.set_xticks(range(len(g)))
    a2.set_xticklabels([{"first_person": "First\nperson", "third_person": "Third\nperson", "neutral": "Neutral"}[k] for k in g.index])
    a2.set_ylim(0, 1)
    a2.set_ylabel("Speaker identification accuracy")
    a2.grid(axis="x", visible=False)
    panel_label(a2, "b")
    fig.tight_layout()
    save(fig, out, "Fig10.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results")
    args = ap.parse_args()
    out = os.path.join(args.res, "figures")
    os.makedirs(out, exist_ok=True)
    ses = pd.read_csv(os.path.join(args.res, "sessions.csv"))
    utt = pd.read_csv(os.path.join(args.res, "utterances.csv"))
    summ = json.load(open(os.path.join(args.res, "summary.json"), encoding="utf-8"))
    for f in (lambda: fig_effects(summ, out), lambda: fig_cells(ses, out), lambda: fig_character_distributions(utt, out),
              lambda: fig_subjectivity(ses, utt, out), lambda: fig_identity(args.res, out), lambda: fig_trajectory(utt, out),
              lambda: fig_ablation(ses, out), lambda: fig_judges(args.res, out)):
        try:
            f()
        except Exception as e:  # keep going so that one missing input does not block the others
            print("figure failed:", repr(e))


if __name__ == "__main__":
    main()
