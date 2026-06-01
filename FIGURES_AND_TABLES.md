# Figures and tables for the manuscript

Mapping between analysis outputs and suggested placement in the paper. All figures are in `outputs/figures/` (PNG 300 dpi and PDF vector).

## RQ1 — Evolution of scientific production

| Manuscript item | File | Description |
|-----------------|------|-------------|
| **Figure (main)** | `rq1_fig_production.png` / `.pdf` | Annual publications 2018–2026, EMA trend, growth phases |
| **Figure (suppl.)** | `rq1_fig_cumulative.png` / `.pdf` | Cumulative corpus size 2006–2026 |
| **Table** | `rq1_yearly_production.csv` | Year, count, share, YoY growth, cumulative, EMA |
| **Table** | `rq1_growth_phases.json` | Phase boundaries, slopes, R² |

## RQ2 — Thematic and intellectual structure

| Manuscript item | File | Description |
|-----------------|------|-------------|
| **Figure** | `rq2_fig_topics.png` | Top 8 NMF topics by document count |
| **Figure** | `rq2_fig_keyword_network.png` | Keyword co-occurrence network (top 90 terms) |
| **Figure** | `rq2_fig_topic_evolution.png` | Stacked topic share by year (2018–2026) |
| **Table** | `rq2_topics.csv` | 14 topics with top terms |
| **Table** | `rq2_topic_doc_assignments.csv` | Document–topic mapping |
| **Table** | `rq2_keyword_communities.csv` | Keywords, frequency, community ID |
| **Table** | `rq2_keyword_network_edges.csv` | Co-occurrence edges |

## RQ3 — Emerging trends and future directions

| Manuscript item | File | Description |
|-----------------|------|-------------|
| **Figure** | `rq3_fig_emerging_terms.png` | Top discriminative terms (log-odds z-score) |
| **Figure** | `rq3_fig_bursty_phrases.png` | Top bursty phrases |
| **Figure** | `rq3_fig_topic_growth.png` | Topic-share slopes in 2024–2026 |
| **Figure (suppl.)** | `rq3_fig_theme_trajectories.png` | Line plots for fastest-growing themes |
| **Table** | `rq3_emerging_terms.csv` | Full ranked emerging terms |
| **Table** | `rq3_bursty_phrases.csv` | Full ranked bursty phrases |
| **Table** | `rq3_topic_growth.csv` | Topic growth metrics |
| **Table** | `rq3_topic_share_by_year.csv` | Annual topic shares |

## Narrative reports

- `outputs/rq1_report.md`, `outputs/rq2_report.md`, `outputs/rq3_report.md` — human-readable summaries
- `outputs/rq*_report.json` — machine-readable headline statistics

## Suggested figure numbering in the article

| # | RQ | File |
|---|-----|------|
| 1 | RQ1 | `rq1_fig_production` |
| 2 | RQ1 (opt.) | `rq1_fig_cumulative` |
| 3 | RQ2 | `rq2_fig_topics` |
| 4 | RQ2 | `rq2_fig_keyword_network` |
| 5 | RQ2 (opt.) | `rq2_fig_topic_evolution` |
| 6 | RQ3 | `rq3_fig_emerging_terms` |
| 7 | RQ3 | `rq3_fig_bursty_phrases` |
| 8 | RQ3 | `rq3_fig_topic_growth` |
| 9 | RQ3 (opt.) | `rq3_fig_theme_trajectories` |
