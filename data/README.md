# Data

## Scopus retrieval

Records were exported from [Scopus](https://www.scopus.com) with the following search string (title, abstract, keywords):

```
TITLE-ABS-KEY(
  ( "user experience" OR UX )
  AND
  ( "generative AI" OR "large language model*" OR LLM* OR ChatGPT OR chatbot* OR "conversational AI" )
)
```

**Export settings:** CSV format, all available bibliographic fields (at minimum: *Year*, *Title*, *Abstract*, *Author Keywords*, *Index Keywords*, *DOI*).

## Files

| File | Location | Description |
|------|----------|-------------|
| Raw export | `../data.csv` (repository root) | 3,721 Scopus records |
| Preprocessed corpus | `../outputs/corpus_preprocessed.csv` | 3,714 records after excluding missing abstracts |
| Excluded records | `../outputs/excluded_no_abstract.csv` | 7 records without usable abstracts |

## Licensing note

Scopus data are subject to [Elsevier’s terms of use](https://www.elsevier.com/legal/elsevier-website-terms-and-conditions). This repository includes the export and derived tables **for reproducibility of the published analysis**. Redistribution for purposes other than verifying this study may require a separate agreement with Elsevier.

If you clone the repository without `data.csv`, run `preprocess.py` after placing your own Scopus export at the project root (see main [README](../README.md)).
