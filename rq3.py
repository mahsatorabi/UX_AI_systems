"""
RQ3 — Emerging trends & future research directions

Research question:
    What emerging trends and future research directions can be identified in user
    experience studies related to generative AI technologies?

Methods (query-bias controlled via shared term omission in rq_common):
  1) Log-odds ratio (recent vs past) with informative Dirichlet prior
  2) Bursty phrase acceleration over years
  3) NMF topic share growth in the recent window

Outputs: CSV/JSON reports + journal figures in outputs/figures/
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from rq_common import (
    apply_journal_style,
    fig_dir,
    filter_terms_for_display,
    is_omitted,
    make_omit_analyzer,
    save_figure,
    PALETTE,
)


@dataclass
class Config:
    input_path: Path = Path("outputs") / "corpus_preprocessed.csv"
    outdir: Path = Path("outputs")
    text_col: str = "text_processed"
    year_col: str = "Year"
    title_col: str = "Title"

    recent_years: int = 3
    min_term_total: int = 20
    top_n_terms: int = 20
    top_n_phrases: int = 20

    min_df: int = 5
    max_df: float = 0.55
    max_features: int = 9000
    nmf_k: int = 10
    nmf_max_iter: int = 400
    random_state: int = 42


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="RQ3: emerging trends & future directions")
    p.add_argument("--input", type=Path, default=Config.input_path)
    p.add_argument("--outdir", type=Path, default=Config.outdir)
    p.add_argument("--recent-years", type=int, default=Config.recent_years)
    p.add_argument("--nmf-k", type=int, default=Config.nmf_k)
    a = p.parse_args()
    return Config(
        input_path=a.input,
        outdir=a.outdir,
        recent_years=a.recent_years,
        nmf_k=a.nmf_k,
    )


def load_df(cfg: Config) -> pd.DataFrame:
    if not cfg.input_path.exists():
        raise FileNotFoundError(f"Missing input: {cfg.input_path}. Run preprocess.py first.")
    df = pd.read_csv(cfg.input_path, usecols=[cfg.text_col, cfg.year_col, cfg.title_col])
    df[cfg.text_col] = df[cfg.text_col].fillna("").astype(str)
    df[cfg.year_col] = pd.to_numeric(df[cfg.year_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=[cfg.year_col]).copy()
    df[cfg.year_col] = df[cfg.year_col].astype(int)
    return df


def count_terms(texts: Iterable[str], analyzer) -> Counter:
    c: Counter = Counter()
    for t in texts:
        c.update(analyzer(t))
    return c


def log_odds_ratio_with_prior(counts_a: dict, counts_b: dict, alpha: dict) -> dict[str, float]:
    vocab = set(counts_a) | set(counts_b) | set(alpha)
    n_a = sum(counts_a.get(w, 0) for w in vocab)
    n_b = sum(counts_b.get(w, 0) for w in vocab)
    alpha0 = sum(alpha.get(w, 0.0) for w in vocab)
    out: dict[str, float] = {}
    for w in vocab:
        a, b = counts_a.get(w, 0), counts_b.get(w, 0)
        aw = alpha.get(w, 0.0)
        logit_a = math.log((a + aw) / max(1e-9, (n_a + alpha0) - (a + aw)))
        logit_b = math.log((b + aw) / max(1e-9, (n_b + alpha0) - (b + aw)))
        delta = logit_a - logit_b
        var = 1.0 / (a + aw + 1e-9) + 1.0 / (b + aw + 1e-9)
        out[w] = float(delta / math.sqrt(var))
    return out


def _filter_emergence_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty:
        return df
    mask = ~df[col].astype(str).str.lower().apply(is_omitted)
    return df.loc[mask].reset_index(drop=True)


def emerging_terms(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    analyzer = make_omit_analyzer()
    y_max = int(df[cfg.year_col].max())
    recent_set = set(range(y_max - cfg.recent_years + 1, y_max + 1))
    recent = df[df[cfg.year_col].isin(recent_set)]
    past = df[~df[cfg.year_col].isin(recent_set)]

    c_recent = count_terms(recent[cfg.text_col], analyzer)
    c_past = count_terms(past[cfg.text_col], analyzer)
    pooled = Counter()
    pooled.update(c_recent)
    pooled.update(c_past)
    alpha = {w: float(pooled[w] * 0.01 + 0.1) for w in pooled}

    z = log_odds_ratio_with_prior(c_recent, c_past, alpha)
    rows = []
    for w, score in z.items():
        total = pooled.get(w, 0)
        if total < cfg.min_term_total or is_omitted(w):
            continue
        rows.append(
            {
                "term": w,
                "z_log_odds_recent_vs_past": score,
                "count_recent": int(c_recent.get(w, 0)),
                "count_past": int(c_past.get(w, 0)),
                "count_total": int(total),
                "recent_share": float(c_recent.get(w, 0)) / max(1, total),
            }
        )
    out = pd.DataFrame(rows).sort_values("z_log_odds_recent_vs_past", ascending=False)
    return _filter_emergence_table(out, "term")


def validate_with_classifier(df: pd.DataFrame, cfg: Config, top_k: int = 40) -> pd.DataFrame:
    y_max = int(df[cfg.year_col].max())
    recent_set = set(range(y_max - cfg.recent_years + 1, y_max + 1))
    y = df[cfg.year_col].isin(recent_set).astype(int).to_numpy()

    vec = TfidfVectorizer(
        analyzer=make_omit_analyzer(),
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        max_features=cfg.max_features,
    )
    X = vec.fit_transform(df[cfg.text_col].tolist())
    clf = LogisticRegression(max_iter=400, solver="liblinear", random_state=cfg.random_state)
    clf.fit(X, y)
    coefs = clf.coef_[0]
    vocab = np.array(vec.get_feature_names_out())
    idx = np.argsort(coefs)[::-1]
    rows = []
    for i in idx:
        t = str(vocab[i])
        if is_omitted(t):
            continue
        rows.append({"term": t, "coef_recent": float(coefs[i])})
        if len(rows) >= top_k:
            break
    return pd.DataFrame(rows)


def bursty_phrases(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    analyzer = make_omit_analyzer()
    years = sorted(df[cfg.year_col].unique().tolist())
    y_max = max(years)
    recent_years = list(range(y_max - cfg.recent_years + 1, y_max + 1))
    prior_years = [y for y in years if y not in recent_years]

    yearly = {y: count_terms(df.loc[df[cfg.year_col] == y, cfg.text_col], analyzer) for y in years}
    pooled = Counter()
    for c in yearly.values():
        pooled.update(c)

    rows = []
    for term, total in pooled.items():
        if total < cfg.min_term_total or is_omitted(term):
            continue
        prior_vals = np.array([yearly[y].get(term, 0) for y in prior_years], dtype=float) if prior_years else np.array([0.0])
        recent_vals = np.array([yearly[y].get(term, 0) for y in recent_years], dtype=float)
        prior_mean = float(prior_vals.mean()) if prior_vals.size else 0.0
        recent_mean = float(recent_vals.mean())
        score = (recent_mean - prior_mean) / (float(prior_vals.std()) + 1.0)
        rows.append(
            {
                "phrase": term,
                "burst_score": float(score),
                "prior_mean_per_year": prior_mean,
                "recent_mean_per_year": recent_mean,
                "count_total": int(total),
            }
        )
    out = pd.DataFrame(rows).sort_values("burst_score", ascending=False)
    return _filter_emergence_table(out, "phrase")


def topic_trends(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    vec = TfidfVectorizer(
        analyzer=make_omit_analyzer(),
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        max_features=cfg.max_features,
        norm="l2",
    )
    X = vec.fit_transform(df[cfg.text_col].tolist())
    nmf = NMF(
        n_components=cfg.nmf_k,
        init="nndsvda",
        max_iter=cfg.nmf_max_iter,
        random_state=cfg.random_state,
    )
    W = nmf.fit_transform(X)
    H = nmf.components_
    vocab = list(vec.get_feature_names_out())

    topic_labels = []
    for k in range(cfg.nmf_k):
        ranked = [vocab[i] for i in np.argsort(H[k])[::-1]]
        label_terms = filter_terms_for_display(ranked, max_n=4)
        topic_labels.append(", ".join(label_terms) if label_terms else f"Topic {k}")

    dom = W.argmax(axis=1)
    df2 = df[[cfg.year_col]].copy()
    df2["topic_id"] = dom
    by_year = df2.groupby([cfg.year_col, "topic_id"]).size().unstack(fill_value=0)
    share = by_year.div(by_year.sum(axis=1), axis=0)

    years = share.index.to_numpy(dtype=int)
    y_max = int(years.max())
    recent_mask = years >= (y_max - cfg.recent_years + 1)
    yrs_recent = years[recent_mask]

    rows = []
    for tid in share.columns:
        s = share[tid].to_numpy(dtype=float)
        s_recent = s[recent_mask]
        Xr = np.c_[np.ones_like(yrs_recent, dtype=float), yrs_recent.astype(float)]
        beta, *_ = np.linalg.lstsq(Xr, s_recent, rcond=None)
        rows.append(
            {
                "topic_id": int(tid),
                "topic_label": topic_labels[int(tid)],
                "slope_recent_share_per_year": float(beta[1]),
                "share_recent_mean": float(s_recent.mean()),
            }
        )
    df_trend = pd.DataFrame(rows).sort_values("slope_recent_share_per_year", ascending=False)
    share_out = share.reset_index().rename(columns={cfg.year_col: "Year"})
    return share_out, df_trend


def plot_rq3_figures(
    cfg: Config,
    terms: pd.DataFrame,
    bursts: pd.DataFrame,
    topic_trend: pd.DataFrame,
    topic_share: pd.DataFrame,
) -> None:
    apply_journal_style()
    out = fig_dir(cfg.outdir)

    tplot = terms.head(15)
    if not tplot.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.barh(
            range(len(tplot)),
            tplot["z_log_odds_recent_vs_past"].astype(float),
            color=PALETTE["accent"],
            alpha=0.9,
        )
        ax.set_yticks(range(len(tplot)))
        ax.set_yticklabels(tplot["term"].astype(str), fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Log-odds z-score (recent vs past)")
        ax.set_title("Fig. RQ3a. Emerging discriminative terms (2024–2026)")
        save_figure(fig, out / "rq3_fig_emerging_terms")

    bplot = bursts.head(15)
    if not bplot.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.barh(
            range(len(bplot)),
            bplot["burst_score"].astype(float),
            color=PALETTE["secondary"],
            alpha=0.9,
        )
        ax.set_yticks(range(len(bplot)))
        ax.set_yticklabels(bplot["phrase"].astype(str), fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Burst score (acceleration)")
        ax.set_title("Fig. RQ3b. Bursty phrases in the recent period")
        save_figure(fig, out / "rq3_fig_bursty_phrases")

    growth = topic_trend.sort_values("slope_recent_share_per_year", ascending=False).head(8)
    if not growth.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        x = np.arange(len(growth))
        ax.bar(
            x,
            growth["slope_recent_share_per_year"].astype(float),
            color=PALETTE["primary"],
            alpha=0.88,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(growth["topic_label"].astype(str), rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Slope of topic share (per year)")
        ax.set_title("Fig. RQ3c. Fast-growing research themes")
        save_figure(fig, out / "rq3_fig_topic_growth")

    # Topic share trajectories for top growing topics (2018+)
    if not topic_share.empty:
        ts = topic_share[topic_share["Year"] >= 2018].copy()
        if not ts.empty and not growth.empty:
            top_ids = [int(i) for i in growth["topic_id"].head(4)]
            cols = [c for c in top_ids if c in ts.columns]
            if cols:
                fig, ax = plt.subplots(figsize=(7.2, 4.2))
                for c in cols:
                    lab = growth.loc[growth["topic_id"] == c, "topic_label"].iloc[0]
                    ax.plot(ts["Year"], ts[c], marker="o", label=lab)
                ax.set_xlabel("Year")
                ax.set_ylabel("Topic share")
                ax.set_title("Fig. RQ3d. Trajectories of emerging themes")
                ax.legend(fontsize=7, frameon=True)
                save_figure(fig, out / "rq3_fig_theme_trajectories")


def write_report(
    cfg: Config,
    df: pd.DataFrame,
    terms: pd.DataFrame,
    clf_terms: pd.DataFrame,
    bursts: pd.DataFrame,
    topic_share: pd.DataFrame,
    topic_trend: pd.DataFrame,
) -> dict:
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    terms.to_csv(cfg.outdir / "rq3_emerging_terms.csv", index=False, encoding="utf-8")
    bursts.to_csv(cfg.outdir / "rq3_bursty_phrases.csv", index=False, encoding="utf-8")
    topic_share.to_csv(cfg.outdir / "rq3_topic_share_by_year.csv", index=False, encoding="utf-8")
    topic_trend.to_csv(cfg.outdir / "rq3_topic_growth.csv", index=False, encoding="utf-8")

    y_max = int(df[cfg.year_col].max())
    recent_years = list(range(y_max - cfg.recent_years + 1, y_max + 1))

    head = {
        "n_documents": int(len(df)),
        "year_max": y_max,
        "recent_years": recent_years,
        "top_emerging_terms": terms.head(cfg.top_n_terms)["term"].tolist(),
        "top_bursty_phrases": bursts.head(cfg.top_n_phrases)["phrase"].tolist(),
        "top_growing_topics": topic_trend.head(8).to_dict(orient="records"),
    }

    md = [
        "# RQ3 Report — Emerging Trends & Future Directions",
        "",
        f"**Corpus:** {head['n_documents']} documents | **Recent window:** {recent_years[0]}–{recent_years[-1]}",
        "",
        "_Query-related and generic terms (e.g., language, model, user experience) were omitted to reveal substantive trends._",
        "",
        "## Emerging terms",
        "",
    ]
    for r in terms.head(cfg.top_n_terms).itertuples():
        md.append(f"- **{r.term}** (z={r.z_log_odds_recent_vs_past:.2f})")
    md += ["", "## Bursty phrases", ""]
    for r in bursts.head(cfg.top_n_phrases).itertuples():
        md.append(f"- **{r.phrase}** (burst={r.burst_score:.2f})")
    md += ["", "## Fast-growing themes", ""]
    for r in topic_trend.head(8).itertuples():
        md.append(
            f"- **{r.topic_label}** (slope={r.slope_recent_share_per_year:+.4f}/year)"
        )
    md += [
        "",
        "## Figures",
        "",
        f"- `{fig_dir(cfg.outdir) / 'rq3_fig_emerging_terms.png'}`",
        f"- `{fig_dir(cfg.outdir) / 'rq3_fig_bursty_phrases.png'}`",
        f"- `{fig_dir(cfg.outdir) / 'rq3_fig_topic_growth.png'}`",
        "",
    ]

    (cfg.outdir / "rq3_report.md").write_text("\n".join(md), encoding="utf-8")
    (cfg.outdir / "rq3_report.json").write_text(json.dumps(head, indent=2), encoding="utf-8")
    return head


def main() -> None:
    cfg = parse_args()
    df = load_df(cfg)

    terms = emerging_terms(df, cfg)
    clf_terms = validate_with_classifier(df, cfg)
    bursts = bursty_phrases(df, cfg)
    topic_share, topic_trend = topic_trends(df, cfg)

    head = write_report(cfg, df, terms, clf_terms, bursts, topic_share, topic_trend)
    plot_rq3_figures(cfg, terms, bursts, topic_trend, topic_share)
    print(json.dumps(head, indent=2))


if __name__ == "__main__":
    main()
