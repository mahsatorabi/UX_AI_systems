# Evolution of User Experience Research in AI-Powered Systems

**A bibliometric and topic-modelling analysis of UX research in generative and conversational AI**

[![GitHub](https://img.shields.io/badge/GitHub-mahsatorabi%2FUX__AI__systems-blue)](https://github.com/mahsatorabi/UX_AI_systems)

This repository contains the **data**, **code**, **figures**, and **supplementary tables** for reproducing the analyses reported in the manuscript on the evolution of user experience (UX) research at the intersection of generative AI, large language models, and conversational systems.

---

## Overview

| Item | Value |
|------|-------|
| Data source | Scopus |
| Final corpus | **N = 3,714** publications (2006–2026) |
| Research questions | **RQ1** temporal production · **RQ2** thematic/intellectual structure · **RQ3** emerging trends |
| Implementation | Python 3.11+ |

**RQ1** — How has scientific production evolved over time?  
**RQ2** — What are the main thematic (NMF) and intellectual (keyword network) structures?  
**RQ3** — What emerging terms, bursty phrases, and growing themes indicate future research directions?

Full methodological detail: [`METHODOLOGY.md`](METHODOLOGY.md)  
Figure/table mapping for the paper: [`FIGURES_AND_TABLES.md`](FIGURES_AND_TABLES.md)

---

## Repository structure

```
UX_AI_systems/
├── README.md                 # This file
├── METHODOLOGY.md            # Methods section (for the article)
├── FIGURES_AND_TABLES.md     # Manuscript figure & table index
├── requirements.txt          # Python dependencies
├── run_all.py                # One-command reproduction
├── data.csv                  # Raw Scopus export (3,721 records)
├── data/
│   └── README.md             # Scopus query & data notes
├── preprocess.py             # Text preprocessing pipeline
├── rq1.py                    # RQ1: production over time
├── rq2.py                    # RQ2: NMF topics + keyword network
├── rq3.py                    # RQ3: emerging trends (3 signals)
├── rq_common.py              # Shared term filters & figure style
└── outputs/
    ├── corpus_preprocessed.csv
    ├── preprocess_report.md
    ├── rq1_yearly_production.csv
    ├── rq1_growth_phases.json
    ├── rq1_report.md
    ├── rq2_topics.csv
    ├── rq2_topic_doc_assignments.csv
    ├── rq2_keyword_communities.csv
    ├── rq2_keyword_network_edges.csv
    ├── rq2_report.md
    ├── rq3_emerging_terms.csv
    ├── rq3_bursty_phrases.csv
    ├── rq3_topic_growth.csv
    ├── rq3_topic_share_by_year.csv
    ├── rq3_report.md
    └── figures/              # PNG (300 dpi) + PDF for the paper
        ├── rq1_fig_production.{png,pdf}
        ├── rq1_fig_cumulative.{png,pdf}
        ├── rq2_fig_topics.{png,pdf}
        ├── rq2_fig_keyword_network.{png,pdf}
        ├── rq2_fig_topic_evolution.{png,pdf}
        ├── rq3_fig_emerging_terms.{png,pdf}
        ├── rq3_fig_bursty_phrases.{png,pdf}
        ├── rq3_fig_topic_growth.{png,pdf}
        └── rq3_fig_theme_trajectories.{png,pdf}
```

---

## Quick start (reproduce results)

### 1. Clone and install

```bash
git clone https://github.com/mahsatorabi/UX_AI_systems.git
cd UX_AI_systems

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 2. Run the full pipeline

```bash
python run_all.py
```

This runs, in order: `preprocess.py` → `rq1.py` → `rq2.py` → `rq3.py`.

If `outputs/corpus_preprocessed.csv` already exists:

```bash
python run_all.py --skip-preprocess
```

### 3. Run individual steps

```bash
python preprocess.py --input data.csv --output-dir outputs
python rq1.py --input outputs/corpus_preprocessed.csv --outdir outputs
python rq2.py --input outputs/corpus_preprocessed.csv --outdir outputs
python rq3.py --input outputs/corpus_preprocessed.csv --outdir outputs
```

---

## Data

- **Raw export:** `data.csv` at the repository root (see [`data/README.md`](data/README.md) for the Scopus query).
- **Analytical corpus:** `outputs/corpus_preprocessed.csv` (records without abstracts excluded).
- **Scopus licensing:** Do not redistribute the raw export beyond what your Elsevier agreement allows. This repo is intended for **research reproducibility** linked to the published paper.

---

## Key outputs for the article

### Figures (`outputs/figures/`)

Use **PDF** in LaTeX/Word for vector quality; **PNG** for review systems that require raster images.

| RQ | Recommended figures |
|----|---------------------|
| RQ1 | `rq1_fig_production`, `rq1_fig_cumulative` (optional) |
| RQ2 | `rq2_fig_topics`, `rq2_fig_keyword_network`, `rq2_fig_topic_evolution` (optional) |
| RQ3 | `rq3_fig_emerging_terms`, `rq3_fig_bursty_phrases`, `rq3_fig_topic_growth`, `rq3_fig_theme_trajectories` (optional) |

### Tables (CSV / JSON)

Build manuscript tables from:

- `outputs/rq1_yearly_production.csv`, `outputs/rq1_growth_phases.json`
- `outputs/rq2_topics.csv`, `outputs/rq2_keyword_communities.csv`
- `outputs/rq3_emerging_terms.csv`, `outputs/rq3_bursty_phrases.csv`, `outputs/rq3_topic_growth.csv`

See [`FIGURES_AND_TABLES.md`](FIGURES_AND_TABLES.md) for the full mapping.

### Summary reports

- `outputs/rq1_report.md` — peak year, growth phases, recent-period share  
- `outputs/rq2_report.md` — topic labels, keyword communities  
- `outputs/rq3_report.md` — top emerging terms, bursty phrases, growing themes  

---

## Methods summary

1. **Preprocessing** (`preprocess.py`): merge title, abstract, and keywords; clean boilerplate; lemmatise (spaCy / NLTK fallback); export token lists and `text_processed`.
2. **Search-bias control** (`rq_common.py`): omit query-umbrella terms (e.g., *user experience*, *generative ai*, *llm*) and generic academic tokens from TF–IDF and keyword networks so latent themes are visible.
3. **RQ1**: annual counts, EMA, change-point detection, piecewise growth phases.
4. **RQ2**: NMF topic modelling (*k* = 14) + keyword co-occurrence network with community detection.
5. **RQ3**: log-odds emerging terms, burst scores, and recent-window topic-share slopes (2024–2026 vs 2006–2023).

Details: [`METHODOLOGY.md`](METHODOLOGY.md).

---

## Dependencies

| Package | Role |
|---------|------|
| pandas, numpy | Data handling |
| scikit-learn | NMF, TF–IDF, classification (RQ3 validation) |
| spacy (`en_core_web_sm`) | English lemmatisation |
| nltk | Stopwords, WordNet fallback |
| matplotlib | Figures (300 dpi PNG + PDF) |
| networkx | Keyword network layout (RQ2) |

---

## Citation

If you use this repository, please cite the associated article (it will be updateed when published):

```bibtex
@article{torabi2026uxai,
  title   = {Evolution of User Experience Research in AI-Powered Systems: A Bibliometric and Topic Modelling Analysis},
  author  = {Sangari, Mahmood and Torabi, Mahsa},
  journal = {},
  year    = {2026},
  note    = {Code and data: \url{https://github.com/mahsatorabi/UX_AI_systems}}
}
```

---

## Authors & contact

- **Repository:** [https://github.com/mahsatorabi/UX_AI_systems](https://github.com/mahsatorabi/UX_AI_systems)
- **Issues:** use GitHub Issues for questions about reproduction or file paths.

---

## License

- **Code** (`.py` files): University of Birjand License.
- **Data** (`data.csv`, derived CSVs): subject to Scopus/Elsevier terms; provided for reproducibility of the published study only.
