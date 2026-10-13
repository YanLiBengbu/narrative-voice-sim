"""Statistical procedures used in the paper.

Unit of analysis: one generated session (utterance-level values are first averaged within a
session to avoid pseudo-replication). Narrative modes are compared within blocks
(text x model x seed) that share the same Director initialisation and seed.
"""

import numpy as np
import pandas as pd
from scipy import stats as st


def paired_frame(df, metric, a, b, by="mode", block=("text_id", "model", "seed")):
    block = list(block)
    w = df.pivot_table(index=block, columns=by, values=metric, aggfunc="mean")
    if a not in w.columns or b not in w.columns:  # measure undefined in one condition
        return np.array([]), np.array([]), pd.DataFrame()
    w = w[[a, b]].dropna()
    return w[a].to_numpy(), w[b].to_numpy(), w.reset_index()


def bootstrap_ci(diff, n_boot=10000, seed=12345, alpha=0.05):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_test(df, metric, a, b, by="mode", block=("text_id", "model", "seed")):
    x, y, _ = paired_frame(df, metric, a, b, by, block)
    d = x - y
    n = len(d)
    res = dict(metric=metric, a=a, b=b, n=n, mean_a=float(np.mean(x)) if n else np.nan,
               mean_b=float(np.mean(y)) if n else np.nan, diff=float(np.mean(d)) if n else np.nan)
    if n < 3 or np.allclose(d, 0):
        res.update(ci_low=np.nan, ci_high=np.nan, p=1.0 if n else np.nan, dz=0.0, r_rb=0.0, frac_pos=np.nan)
        return res
    lo, hi = bootstrap_ci(d)
    w = st.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    nz = d[d != 0]
    ranks = st.rankdata(np.abs(nz))
    r_rb = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
    sd = np.std(d, ddof=1)
    res.update(ci_low=lo, ci_high=hi, p=float(w.pvalue), dz=float(np.mean(d) / sd) if sd > 0 else np.nan,
               r_rb=float(r_rb), frac_pos=float(np.mean(d > 0)))
    return res


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def comparison_table(df, metrics, pairs, by="mode", block=("text_id", "model", "seed")):
    rows = [paired_test(df, m, a, b, by, block) for a, b in pairs for m in metrics]
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    for (a, b), g in t.groupby(["a", "b"]):  # Holm correction within each pairwise family
        t.loc[g.index, "p_holm"] = holm(g["p"].fillna(1.0).to_numpy())
    return t


def mixed_model(df, metric, ref="third_person"):
    """metric ~ mode + model + (1 | text); returns fixed effects of mode vs. ref."""
    import statsmodels.formula.api as smf
    import warnings
    cols = ["text_id", "model", "mode", metric] + (["seed"] if "seed" in df.columns else [])
    d = df[cols].dropna().rename(columns={metric: "y"})
    d["y_s"] = (d.y - d.y.mean()) / (d.y.std() or 1.0)  # standardize for numerical stability
    scale = d.y.std() or 1.0
    formula = f"y_s ~ C(mode, Treatment('{ref}')) + C(model)"
    fit, method = None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for opt in ("lbfgs", "powell", "nm"):
            try:
                fit = smf.mixedlm(formula, d, groups=d["text_id"]).fit(reml=True, method=opt)
                if np.all(np.isfinite(fit.bse)):
                    method = f"mixedlm-{opt}"
                    break
            except Exception:
                fit = None
        if method is None:  # fallback: excerpt fixed effects, cluster-robust SE by excerpt-model-seed block
            d["block"] = d.text_id + "|" + d.model + "|" + d.get("seed", 0).astype(str)
            fit = smf.ols(formula + " + C(text_id)", d).fit(cov_type="cluster", cov_kwds={"groups": d["block"]})
            method = "ols-fe-cluster"
    out = []
    ci = fit.conf_int()
    params = fit.fe_params if hasattr(fit, "fe_params") else fit.params
    for name, coef in params.items():
        if name.startswith("C(mode,"):
            lo, hi = ci.loc[name]
            out.append(dict(metric=metric, term=name.split("[T.")[1].rstrip("]"), coef=coef * scale, ci_low=lo * scale,
                            ci_high=hi * scale, p=fit.pvalues[name], method=method))
    return out


def cell_consistency(df, metric, a, b, cells=("text_id", "model")):
    """Direction of the a-b difference in every text x model cell (generalisation check)."""
    g = df.groupby(list(cells) + ["mode"])[metric].mean().unstack("mode")
    diff = (g[a] - g[b]).dropna()
    return dict(metric=metric, n_cells=len(diff), n_pos=int((diff > 0).sum()), n_neg=int((diff < 0).sum()))


def jsd(p, q, eps=1e-9):
    keys = sorted(set(p) | set(q))
    a = np.array([p.get(k, 0.0) for k in keys]) + eps
    b = np.array([q.get(k, 0.0) for k in keys]) + eps
    a, b = a / a.sum(), b / b.sum()
    m = (a + b) / 2
    return float(0.5 * (st.entropy(a, m, base=2) + st.entropy(b, m, base=2)))
