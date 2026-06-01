# Preprocessing Report — UX × Generative AI Corpus

**Input:** `data.csv`  
**Output directory:** `outputs`  

## Record counts

| Stage | Count |
|-------|------:|
| Records in Scopus export | 3721 |
| Excluded (no abstract) | 7 |
| **Final corpus** | **3714** |

## Preprocessing steps

1. Load CSV (UTF-8)
2. **Drop** rows with missing/placeholder abstract (`[No abstract available]`)
3. Per-field text cleaning (encoding, boilerplate, URLs/DOI)
4. Merge Title + Abstract + Author Keywords + Index Keywords
5. Language detection + English spaCy lemmatization (non-English: NLTK fallback)
6. NLTK + manual stopword removal (domain terms protected)

## Token statistics (final corpus)

- Mean tokens per document: **165.1**
- Median tokens per document: **159**
- Documents flagged as short (<10 tokens): **0**

## Language / lemmatization

- spaCy (`en_core_web_sm`): **3664** documents
- NLTK fallback: **50** documents

## Excluded titles (no abstract)

- (2025) The use of generative AI in enhancing customer experience
- (2025) Factors influencing user experience in AI chat systems – a satisfaction study based on factor analysis and linear regres
- (2025) Generative AI in aviation: Transforming passenger experience and operational efficiency with ethical AI
- (2024) Bridging the gap between conversation technology and conversation analysis
- (2023) Unleashing the Potential: Integrating ChatGPT and the Internet of Things for Enhanced User Experiences and Automation
- (2023) The IBM natural conversation framework: a new paradigm for conversational UX design
- (2020) How to design and evaluate intuitive conversational user interfaces (USABLEBOTS)
