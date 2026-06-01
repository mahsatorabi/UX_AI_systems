# Methodology

This section describes the data source, preprocessing pipeline, and analytical procedures used to address each research question (RQ1–RQ3). All analyses were implemented in Python 3.11 and applied to the final bibliometric corpus derived from Scopus.

---

## 3.1 Data source and corpus construction

### 3.1.1 Retrieval strategy

Records were retrieved from Scopus using the following search string applied to title, abstract, and keywords:

> **TITLE-ABS-KEY** ( ( "user experience" OR UX ) AND ( "generative AI" OR "large language model*" OR LLM* OR ChatGPT OR chatbot* OR "conversational AI" ) )

The initial export contained **3,721** documents. Records with missing or placeholder abstracts (e.g., *[No abstract available]*) were excluded (**n = 7**), yielding a final analytical corpus of **N = 3,714** publications (2006–2026).

### 3.1.2 Text preprocessing

Bibliographic text was preprocessed prior to thematic and trend analyses (`preprocess.py`). For each record, the following fields were retained and cleaned: *Title*, *Abstract*, *Author Keywords*, and *Index Keywords*.

The preprocessing pipeline comprised:

1. **Encoding normalisation** (Unicode NFKC; mojibake correction where applicable).
2. **Text cleaning**: removal of publisher boilerplate, copyright notices, DOIs, URLs, and HTML entities; normalisation of punctuation and whitespace; light contraction expansion.
3. **Document assembly**: concatenation of cleaned title, abstract, and keyword fields into a single document representation.
4. **Tokenisation and lemmatisation**:
   - English documents (*n* ≈ 3,664): spaCy (`en_core_web_sm`) with POS-filtered lemmatisation (nouns, verbs, adjectives, proper nouns, adverbs).
   - Non-English documents (*n* ≈ 50): NLTK WordNet lemmatisation as a fallback.
5. **Stopword removal**: NLTK English stopwords supplemented with a manual bibliometric stoplist (e.g., *study*, *paper*, *results*). Domain-relevant terms (e.g., *ux*, *generative ai*, *chatbot*) were **protected** during general preprocessing but handled separately in RQ2/RQ3 (see §3.1.3).

The output corpus (`corpus_preprocessed.csv`) includes lemmatised token lists and a space-joined `text_processed` field used in subsequent text mining.

### 3.1.3 Control for search-formula bias (RQ2 and RQ3)

Because the retrieval query explicitly targets UX and generative/conversational AI, those terms appear ubiquitously across the corpus and can dominate unsupervised models. To reveal *latent* thematic and emergent structures, a shared omission list (`rq_common.py`) was applied during TF–IDF vectorisation and keyword-network construction. This list comprises:

- **Query/umbrella terms** (e.g., *user experience*, *generative ai*, *llm*, *chatbot*, *conversational ai*, *language model*);
- **Generic academic and technical tokens** (e.g., *study*, *method*, *system*, *language*, *model*, *enhance*, *integrate*).

Substantive emerging concepts (e.g., *rag*, *hallucination*, *multimodal*, *agentic*, *trust*, *privacy*) were explicitly **protected** and retained. Topic labels and emergence rankings were computed on the highest-weighted non-omitted terms.

---

## 3.2 RQ1 — Evolution of scientific production over time

**Research question:** *How has the scientific production on user experience in generative AI and conversational AI evolved over time?*

### 3.2.1 Unit of analysis and temporal aggregation

The unit of analysis was the publication year (*Year* field in Scopus). Annual publication counts were computed for all years from the first to the last publication year in the corpus, including years with zero publications, to preserve a continuous time series.

### 3.2.2 Descriptive and smoothing indicators

For each year *t*, we computed:

- **Annual count** (*n<sub>t</sub>*): number of publications;
- **Annual share**: *n<sub>t</sub>* / Σ *n<sub>t</sub>*;
- **Year-on-year growth rate**: relative change in *n<sub>t</sub>* versus *t* − 1;
- **Cumulative count**: running sum of publications;
- **Exponential moving average (EMA)**: span = 3 years, applied to smooth short-term fluctuations while preserving recent dynamics.

### 3.2.3 Change-point detection and growth phases

To identify structural shifts in publication intensity, change-points were detected on the EMA-smoothed series using a **dynamic programming** procedure that minimises a penalised sum of segment costs. Segment homogeneity was measured with a **robust pseudo-Huber cost** around the segment median, which limits the influence of outlier years.

Model parameters:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Maximum change-points | 3 | Interpretable number of phases |
| Minimum segment length | 3 years | Avoid trivial 1–2 year “phases” |
| Penalty | Data-driven (MAD × log *n*) | Balance fit vs. model complexity |

Change-point locations partition the timeline into contiguous **growth phases**. Within each phase, a **piecewise linear regression** of raw annual counts on year was fitted to estimate the slope (publications per year) and coefficient of determination (*R*²).

### 3.2.4 Recent-period indicators

Supplementary indicators characterise recent concentration:

- **Peak year**: year with maximum annual output;
- **Three-year share**: proportion of all publications appearing in the last three corpus years;
- **Three-year average**: mean annual publications in that window.

### 3.2.5 Outputs and visualisation

Results were exported as yearly statistics (`rq1_yearly_production.csv`), phase metadata (`rq1_growth_phases.json`), and a narrative report. Figures include: (a) annual production with EMA and shaded growth phases (2018–2026 focus), and (b) cumulative growth over the full timeline (PNG/PDF, 300 dpi).

---

## 3.3 RQ2 — Intellectual and thematic structures

**Research question:** *What are the main intellectual and thematic structures of user experience research in generative AI systems?*

A **dual-structure** approach was adopted, combining full-text topic modelling with keyword co-occurrence analysis.

### 3.3.1 Text-based thematic structure (NMF topic modelling)

**Input text:** lemmatised `text_processed` field per document.

**Vectorisation:** Term frequency–inverse document frequency (TF–IDF) with a custom analyser that excludes query/umbrella and generic terms (§3.1.3), retaining unigrams and bigrams.

| Parameter | Value |
|-----------|-------|
| min_df | 5 |
| max_df | 0.60 |
| max_features | 7,000 |
| Normalisation | L2 |

**Topic model:** Non-negative Matrix Factorisation (NMF; `init='nndsvda'`, max_iter = 400). The number of topics *k* was selected from *k* ∈ {6, …, 14} by maximising a composite score combining:

1. Scaled reconstruction error (lower is better);
2. Topic sparsity (higher is better);
3. Topic diversity of top terms across topics (higher is better).

**Document–topic assignment:** Each document was assigned to its **dominant topic** (highest topic weight in the document–topic matrix **W**).

**Topic interpretation:** For each topic, the top 15 discriminative terms were extracted from the topic–term matrix **H**, excluding omitted tokens. Topics were labelled by their leading non-omitted terms.

### 3.3.2 Intellectual structure (keyword co-occurrence network)

**Input:** cleaned *Author Keywords* and *Index Keywords* (semicolon/comma-separated), lowercased and de-duplicated per document.

**Network construction:**

- **Nodes:** keywords occurring in ≥ 8 documents;
- **Edges:** undirected co-occurrence links when two keywords appear in the same record; edge weight = co-occurrence frequency;
- **Pruning:** edges with weight < 4 were removed;
- **Filtering:** query/umbrella keywords removed (§3.1.3).

**Community detection:** A greedy modularity-style heuristic iteratively reassigns keywords to neighbouring communities when within-community edge weight increases (max 20 iterations). Communities represent intellectually clustered keyword groups.

### 3.3.3 Integration of textual and keyword structures

Each NMF topic was linked to keyword communities by **top-term overlap**: the community sharing the largest number of top topic terms with a keyword node set. This mapping supports triangulation between full-text themes and author/index keyword structures.

### 3.3.4 Temporal extension (thematic prevalence)

Topic prevalence over time was computed as the annual proportion of documents assigned to each dominant topic (2018–2026), visualised as a stacked area chart to show shifting thematic emphasis.

### 3.3.5 Outputs and visualisation

Outputs include topic term lists (`rq2_topics.csv`), document–topic assignments (`rq2_topic_doc_assignments.csv`), network nodes/edges (`rq2_keyword_communities.csv`, `rq2_keyword_network_edges.csv`), and integrated reports. Figures: topic distribution bar chart, keyword co-occurrence network (top 90 nodes), and thematic evolution (stacked area).

---

## 3.4 RQ3 — Emerging trends and future research directions

**Research question:** *What emerging trends and future research directions can be identified in user experience studies related to generative AI technologies?*

Emergence was operationalised along three complementary signals, all applied to the preprocessed corpus with search-bias control (§3.1.3).

### 3.4.1 Temporal windows

Let *Y*<sub>max</sub> denote the latest publication year in the corpus. The **recent window** comprises the last three years: [*Y*<sub>max</sub> − 2, *Y*<sub>max</sub>]. All earlier years constitute the **past window**. In the current corpus, this corresponds to **2024–2026** (recent) versus **2006–2023** (past).

### 3.4.2 Signal 1 — Discriminative emerging terms (log-odds ratio)

Term frequencies were counted separately in recent and past windows using the same omitted-token analyser as RQ2. For each term *w*, we computed the **log-odds ratio** with an **informative Dirichlet prior** (Monroe et al., 2008), using pooled counts to define the prior. The test statistic is reported as a **z-score**; higher values indicate stronger association with the recent period.

Inclusion criteria:

- Minimum total corpus frequency ≥ 20;
- Term not in the omission list (§3.1.3).

**Validation:** A logistic regression classifier (L2, liblinear) was trained on TF–IDF features to predict recent vs. past documents; top positive coefficients were inspected for consistency with log-odds rankings.

### 3.4.3 Signal 2 — Bursty phrases (temporal acceleration)

For each term/phrase, yearly occurrence counts were computed across the full timeline. **Burst score** was defined as:

> (mean frequency in recent years − mean frequency in prior years) / (SD of prior-year frequencies + 1)

This highlights concepts with accelerating presence in the recent window relative to historical baseline variability. The same minimum frequency and omission filters as Signal 1 were applied.

### 3.4.4 Signal 3 — Growing thematic themes (topic share slopes)

An NMF model (*k* = 10 topics) was fitted on TF–IDF features (same omission rules; min_df = 5, max_df = 0.55, max_features = 9,000). For each topic, annual **topic share** (proportion of documents assigned to that dominant topic per year) was computed.

Within the recent window, the **slope** of topic share regressed on year (ordinary least squares) estimates the rate of thematic growth. Topics with the highest positive slopes were interpreted as fast-growing research themes. Topic labels were derived from top non-omitted terms.

### 3.4.5 Synthesis of future research directions

Future directions were inferred by **triangulating** the three signals: terms and phrases with high recent-discriminative or burst scores, and topics with positive recent share slopes. Directional themes (e.g., RAG/information-seeking UX, trust and safety, multimodal interaction, health and education applications) were articulated when supported by multiple indicators.

### 3.4.6 Outputs and visualisation

Outputs include ranked emerging terms (`rq3_emerging_terms.csv`), bursty phrases (`rq3_bursty_phrases.csv`), topic share by year (`rq3_topic_share_by_year.csv`), topic growth metrics (`rq3_topic_growth.csv`), and summary reports. Figures: horizontal bar charts of top emerging terms and bursty phrases, bar chart of topic growth slopes, and line plots of trajectories for leading emerging themes.

---

## 3.5 Software and reproducibility

| Component | Implementation |
|-----------|----------------|
| Preprocessing | `preprocess.py` |
| RQ1 | `rq1.py` |
| RQ2 | `rq2.py` |
| RQ3 | `rq3.py` |
| Shared term filters & figure style | `rq_common.py` |
| Dependencies | pandas, NumPy, scikit-learn, spaCy, NLTK, matplotlib, NetworkX |

All scripts can be re-executed from the project root after installing requirements (`requirements.txt`) and generating the preprocessed corpus. Figures are written to `outputs/figures/` in PNG and PDF formats (300 dpi).

---

## References (methods)

- Monroe, B., Colaresi, M. P., & Quinn, K. M. (2008). Fightin’ words: Lexical feature selection and evaluation for identifying the content of political conflict. *Computational Statistics*, *23*(4), 491–503.
- Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. *Nature*, *401*(6755), 788–791.
