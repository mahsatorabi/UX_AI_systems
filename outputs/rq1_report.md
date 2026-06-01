# RQ1 Report — Scientific Production Over Time

**Input:** `outputs\corpus_preprocessed.csv`  
**Year range:** **2006–2026**  
**Total publications (final corpus):** **3714**  

## Visual summary (count per year)

▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▄█▃

## Key numbers

- Peak year: **2025** with **1601** publications
- Share of publications in the last 3 years: **82.3%** (avg **1019.3** per year)

## Top 10 years by volume

- 2025: **1601**
- 2024: **882**
- 2026: **575**
- 2023: **270**
- 2021: **110**
- 2022: **108**
- 2020: **66**
- 2019: **60**
- 2018: **21**
- 2017: **16**

## Detected growth phases (change-points + piecewise trends)

- Phase 1: 2006–2017 | slope **+0.58 pubs/year** | R²=0.21 | total=21
- Phase 2: 2018–2020 | slope **+22.50 pubs/year** | R²=0.85 | total=147
- Phase 3: 2021–2023 | slope **+80.00 pubs/year** | R²=0.74 | total=488
- Phase 4: 2024–2026 | slope **-153.50 pubs/year** | R²=0.08 | total=3058

## Files written

- `outputs\rq1_yearly_production.csv`
- `outputs\rq1_growth_phases.json`
- `outputs\rq1_report.json`
- `outputs\figures\rq1_fig_production.png`
- `outputs\figures\rq1_fig_cumulative.png`
