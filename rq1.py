"""
RQ1 — Evolution over time

Research question:
    How has the scientific production on user experience in generative AI and
    conversational AI evolved over time?

This script uses your *preprocessed corpus* (created by `preprocess.py`) and
produces:
  - yearly publication counts (and shares)
  - growth rates + smoothed trends (EMA + rolling mean)
  - change-point detection (modern: PELT-style dynamic programming with robust loss)
  - piecewise linear trend fits (interpretable growth phases)
  - an auto-written Markdown report with the key numbers

Run:
    python rq1.py
    python rq1.py --input outputs/corpus_preprocessed.csv --outdir outputs
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rq_common import PALETTE, apply_journal_style, fig_dir, save_figure


@dataclass
class Config:
    input_path: Path = Path("outputs") / "corpus_preprocessed.csv"
    outdir: Path = Path("outputs")
    year_col: str = "Year"
    min_year: int | None = None
    max_year: int | None = None
    # smoothing
    ema_span: int = 3
    rolling_window: int = 3
    # change-point detection
    # Keep phases interpretable: avoid 2-year "phases" with extreme slopes.
    max_changepoints: int = 3  # phases = cp + 1
    min_segment_years: int = 3
    penalty: float | None = None  # if None: data-driven heuristic


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="RQ1: scientific production over time")
    p.add_argument("--input", type=Path, default=Config.input_path)
    p.add_argument("--outdir", type=Path, default=Config.outdir)
    p.add_argument("--min-year", type=int, default=None)
    p.add_argument("--max-year", type=int, default=None)
    p.add_argument("--ema-span", type=int, default=Config.ema_span)
    p.add_argument("--rolling-window", type=int, default=Config.rolling_window)
    p.add_argument("--max-changepoints", type=int, default=Config.max_changepoints)
    p.add_argument("--min-segment-years", type=int, default=Config.min_segment_years)
    p.add_argument("--penalty", type=float, default=None)
    a = p.parse_args()
    return Config(
        input_path=a.input,
        outdir=a.outdir,
        min_year=a.min_year,
        max_year=a.max_year,
        ema_span=a.ema_span,
        rolling_window=a.rolling_window,
        max_changepoints=a.max_changepoints,
        min_segment_years=a.min_segment_years,
        penalty=a.penalty,
    )


def load_years(cfg: Config) -> pd.Series:
    if not cfg.input_path.exists():
        raise FileNotFoundError(
            f"Input not found: {cfg.input_path}. Run preprocess.py first."
        )
    df = pd.read_csv(cfg.input_path, usecols=[cfg.year_col])
    years = pd.to_numeric(df[cfg.year_col], errors="coerce").dropna().astype(int)
    if cfg.min_year is not None:
        years = years[years >= cfg.min_year]
    if cfg.max_year is not None:
        years = years[years <= cfg.max_year]
    if years.empty:
        raise ValueError("No valid years after filtering.")
    return years


def yearly_counts(years: pd.Series) -> pd.DataFrame:
    vc = years.value_counts().sort_index()
    all_years = pd.Index(range(int(vc.index.min()), int(vc.index.max()) + 1), name="Year")
    counts = vc.reindex(all_years, fill_value=0).astype(int)
    dfy = counts.to_frame("n_pubs")
    dfy["share"] = dfy["n_pubs"] / dfy["n_pubs"].sum()
    dfy["yoy_growth"] = dfy["n_pubs"].pct_change().replace([np.inf, -np.inf], np.nan)
    dfy["cumulative"] = dfy["n_pubs"].cumsum()
    return dfy.reset_index()


def add_smoothing(dfy: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    s = dfy["n_pubs"].astype(float)
    dfy["ema"] = s.ewm(span=max(2, cfg.ema_span), adjust=False).mean()
    w = max(2, cfg.rolling_window)
    dfy["roll_mean"] = s.rolling(window=w, center=True, min_periods=1).mean()
    return dfy


def _robust_segment_cost(y: np.ndarray) -> float:
    """
    Robust cost for a segment: Huber-like on deviations from the median.
    This is stable for sparse early years (lots of zeros).
    """
    if y.size == 0:
        return 0.0
    med = np.median(y)
    r = y - med
    scale = np.median(np.abs(r)) + 1e-9
    u = r / scale
    # pseudo-Huber
    delta = 1.5
    return float(np.sum(delta**2 * (np.sqrt(1.0 + (u / delta) ** 2) - 1.0))) * float(
        scale
    )


def detect_changepoints(
    y: np.ndarray, cfg: Config
) -> list[int]:
    """
    Modern-ish, dependency-free change-point detection:
    - dynamic programming to minimize (segment_cost + penalty * #segments)
    - robust segment cost
    Returns change-point indices as *end positions* (exclusive), excluding final n.
    """
    n = int(len(y))
    min_len = int(cfg.min_segment_years)
    if n < 2 * min_len:
        return []

    # Precompute costs for all segments [i:j)
    cost = np.full((n + 1, n + 1), np.inf, dtype=float)
    for i in range(0, n):
        for j in range(i + min_len, n + 1):
            cost[i, j] = _robust_segment_cost(y[i:j])

    # penalty heuristic: scale with robust variance and log(n)
    if cfg.penalty is not None:
        pen = float(cfg.penalty)
    else:
        mad = np.median(np.abs(y - np.median(y))) + 1e-9
        pen = float((2.0 * mad) * math.log(n + 1))

    # DP: best[k][t] = best score up to t with k segments
    max_cps = int(cfg.max_changepoints)
    max_segs = max_cps + 1
    best = np.full((max_segs + 1, n + 1), np.inf, dtype=float)
    prev = np.full((max_segs + 1, n + 1), -1, dtype=int)
    best[0, 0] = 0.0

    for k in range(1, max_segs + 1):
        for t in range(k * min_len, n + 1):
            # last segment starts at s
            s_min = (k - 1) * min_len
            s_max = t - min_len
            if s_max < s_min:
                continue
            scores = best[k - 1, s_min : s_max + 1] + cost[s_min : s_max + 1, t]
            s_rel = int(np.argmin(scores))
            s = s_min + s_rel
            best[k, t] = float(scores[s_rel]) + pen
            prev[k, t] = s

    # choose k with best score at n, but not more than max_cps
    k_star = int(np.nanargmin(best[1 : max_segs + 1, n])) + 1
    # backtrack
    cps: list[int] = []
    t = n
    k = k_star
    while k > 0:
        s = int(prev[k, t])
        if s <= 0:
            break
        cps.append(s)
        t = s
        k -= 1
    cps = sorted([cp for cp in cps if 0 < cp < n])

    # Enforce max changepoints and min lengths (already), but also avoid trivial cps
    cps = cps[-max_cps:]
    return cps


def piecewise_linear_slopes(years: np.ndarray, y: np.ndarray, cps: list[int]) -> list[dict]:
    """
    Fit y = a + b*year within each segment (least squares),
    return slope per year and R^2.
    """
    idx = [0] + cps + [len(y)]
    out: list[dict] = []
    for a, b in zip(idx[:-1], idx[1:]):
        xs = years[a:b].astype(float)
        ys = y[a:b].astype(float)
        if len(xs) < 2 or np.allclose(xs, xs[0]):
            continue
        X = np.c_[np.ones_like(xs), xs]
        beta, *_ = np.linalg.lstsq(X, ys, rcond=None)
        yhat = X @ beta
        ss_res = float(np.sum((ys - yhat) ** 2))
        ss_tot = float(np.sum((ys - np.mean(ys)) ** 2)) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        out.append(
            {
                "start_year": int(xs[0]),
                "end_year": int(xs[-1]),
                "slope_per_year": float(beta[1]),
                "intercept": float(beta[0]),
                "r2": float(r2),
                "n_years": int(len(xs)),
                "sum_pubs": int(np.sum(ys)),
            }
        )
    return out


def write_outputs(dfy: pd.DataFrame, phases: list[dict], cfg: Config) -> dict:
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    dfy_path = cfg.outdir / "rq1_yearly_production.csv"
    dfy.to_csv(dfy_path, index=False, encoding="utf-8")

    phases_path = cfg.outdir / "rq1_growth_phases.json"
    phases_path.write_text(json.dumps(phases, indent=2), encoding="utf-8")

    # Basic headline numbers
    n_total = int(dfy["n_pubs"].sum())
    nonzero_years = int((dfy["n_pubs"] > 0).sum())
    peak_row = dfy.loc[dfy["n_pubs"].idxmax()]
    first_year = int(dfy["Year"].min())
    last_year = int(dfy["Year"].max())
    peak_year = int(peak_row["Year"])
    peak_n = int(peak_row["n_pubs"])

    # “recent momentum”: last 3 years share and average
    tail = dfy.tail(3)
    recent_share = float(tail["n_pubs"].sum() / max(1, n_total))
    recent_avg = float(tail["n_pubs"].mean())

    report = {
        "n_total_publications": n_total,
        "year_range": [first_year, last_year],
        "nonzero_years": nonzero_years,
        "peak_year": peak_year,
        "peak_year_publications": peak_n,
        "recent_3y_share": recent_share,
        "recent_3y_avg": recent_avg,
        "n_phases": len(phases),
    }

    report_path = cfg.outdir / "rq1_report.md"
    report_path.write_text(render_report(dfy, phases, report, cfg), encoding="utf-8")

    report_json_path = cfg.outdir / "rq1_report.json"
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def plot_rq1_figure(dfy: pd.DataFrame, phases: list[dict], cfg: Config) -> None:
    """Journal figure: annual production with growth phases (2018+ main panel)."""
    apply_journal_style()
    out = fig_dir(cfg.outdir)

    df_plot = dfy[dfy["Year"] >= 2018].copy()
    if df_plot.empty:
        df_plot = dfy.copy()

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(
        df_plot["Year"],
        df_plot["n_pubs"],
        color=PALETTE["primary"],
        alpha=0.75,
        width=0.72,
        label="Publications",
        zorder=2,
    )
    ax.plot(
        df_plot["Year"],
        df_plot["ema"],
        color=PALETTE["secondary"],
        linewidth=2.2,
        marker="o",
        markersize=4,
        label="Exponential moving average",
        zorder=3,
    )

    for i, ph in enumerate(phases):
        if ph["end_year"] < df_plot["Year"].min():
            continue
        start = max(ph["start_year"], int(df_plot["Year"].min()))
        end = min(ph["end_year"], int(df_plot["Year"].max()))
        ax.axvspan(
            start - 0.5,
            end + 0.5,
            color=PALETTE["phase"][i % len(PALETTE["phase"])],
            alpha=0.12,
            label=f"Phase {i+1} ({start}–{end})" if i < 4 else None,
        )

    ax.set_xlabel("Publication year")
    ax.set_ylabel("Number of publications")
    ax.set_title("Fig. RQ1. Scientific production on UX in generative AI (2018–2026)")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    save_figure(fig, out / "rq1_fig_production")

    # Cumulative curve (full timeline)
    fig2, ax2 = plt.subplots(figsize=(7.2, 4.0))
    ax2.plot(dfy["Year"], dfy["cumulative"], color=PALETTE["neutral"], linewidth=2.2)
    ax2.fill_between(dfy["Year"], dfy["cumulative"], alpha=0.15, color=PALETTE["primary"])
    ax2.set_xlabel("Publication year")
    ax2.set_ylabel("Cumulative publications")
    ax2.set_title("Fig. RQ1 (suppl.). Cumulative growth of the corpus")
    save_figure(fig2, out / "rq1_fig_cumulative")


def render_report(dfy: pd.DataFrame, phases: list[dict], headline: dict, cfg: Config) -> str:
    # Small “sparkline-like” bar text (unicode blocks)
    blocks = "▁▂▃▄▅▆▇█"
    y = dfy["n_pubs"].to_numpy(dtype=float)
    if np.max(y) > 0:
        scaled = (y / np.max(y) * (len(blocks) - 1)).astype(int)
        spark = "".join(blocks[i] for i in scaled)
    else:
        spark = ""

    # Top 10 years by volume
    top = dfy.sort_values("n_pubs", ascending=False).head(10)[["Year", "n_pubs"]]
    top_lines = "\n".join(f"- {int(r.Year)}: **{int(r.n_pubs)}**" for r in top.itertuples())

    # Phase lines
    if phases:
        phase_lines = []
        for i, ph in enumerate(phases, start=1):
            phase_lines.append(
                f"- Phase {i}: {ph['start_year']}–{ph['end_year']} | "
                f"slope **{ph['slope_per_year']:+.2f} pubs/year** | "
                f"R²={ph['r2']:.2f} | total={ph['sum_pubs']}"
            )
        phases_text = "\n".join(phase_lines)
    else:
        phases_text = "- (Not enough years to detect phases.)"

    return "\n".join(
        [
            "# RQ1 Report — Scientific Production Over Time",
            "",
            f"**Input:** `{cfg.input_path}`  ",
            f"**Year range:** **{headline['year_range'][0]}–{headline['year_range'][1]}**  ",
            f"**Total publications (final corpus):** **{headline['n_total_publications']}**  ",
            "",
            "## Visual summary (count per year)",
            "",
            spark,
            "",
            "## Key numbers",
            "",
            f"- Peak year: **{headline['peak_year']}** with **{headline['peak_year_publications']}** publications",
            f"- Share of publications in the last 3 years: **{headline['recent_3y_share']*100:.1f}%** "
            f"(avg **{headline['recent_3y_avg']:.1f}** per year)",
            "",
            "## Top 10 years by volume",
            "",
            top_lines,
            "",
            "## Detected growth phases (change-points + piecewise trends)",
            "",
            phases_text,
            "",
            "## Files written",
            "",
            f"- `{cfg.outdir / 'rq1_yearly_production.csv'}`",
            f"- `{cfg.outdir / 'rq1_growth_phases.json'}`",
            f"- `{cfg.outdir / 'rq1_report.json'}`",
            f"- `{fig_dir(cfg.outdir) / 'rq1_fig_production.png'}`",
            f"- `{fig_dir(cfg.outdir) / 'rq1_fig_cumulative.png'}`",
            "",
        ]
    )


def main() -> None:
    cfg = parse_args()

    years = load_years(cfg)
    dfy = yearly_counts(years)
    dfy = add_smoothing(dfy, cfg)

    # change-points on smoothed series to reduce noise
    y = dfy["ema"].to_numpy(dtype=float)
    yrs = dfy["Year"].to_numpy(dtype=int)
    cps = detect_changepoints(y, cfg)
    phases = piecewise_linear_slopes(yrs, dfy["n_pubs"].to_numpy(dtype=float), cps)

    # Attach cps (as years) for interpretability
    cp_years = [int(yrs[i]) for i in cps]
    meta = {"changepoint_years": cp_years, "penalty_used": cfg.penalty}
    for ph in phases:
        ph["changepoint_years"] = cp_years
        ph["method_meta"] = meta

    headline = write_outputs(dfy, phases, cfg)
    plot_rq1_figure(dfy, phases, cfg)

    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()

