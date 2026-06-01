"""
Shared utilities for RQ1–RQ3: term omission (search-bias control) and journal figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Terms to omit from topic / trend modelling (query + generic noise)
# ---------------------------------------------------------------------------

QUERY_UMBRELLA_TERMS = frozenset(
    """
    ux user users experience experiences usability
    user experience user experiences user interface user interfaces
    human computer interaction hci
    ai artificial intelligence generative generative ai genai gai
    llm llms gpt chatgpt
    large language model large language models language model language models
    language large large language language natural natural language
    chatbot chatbots conversational conversational ai conversation dialogue
    assistant virtual assistant virtual assistants
    """.split()
)

# Bigrams (must be listed explicitly)
QUERY_UMBRELLA_BIGRAMS = frozenset(
    """
    user experience user experiences user interface user interfaces
    human computer interaction artificial intelligence generative ai
    generative artificial large language language model language models
    large language natural language language processing language process
    conversational ai virtual assistant chat bot
    """.split()
)

GENERIC_ACADEMIC_TERMS = frozenset(
    """
    study studies paper papers article articles journal research researcher
    result results finding findings method methods methodology approach
    framework model models system systems tool tools technology technologies
    analysis factor factors effect effects influence impact purpose aim
    objective context significant significantly demonstrate show provide
    propose develop design based use used using include including
    need may can work recent new various different important overall
    paper-based review reviews survey surveys
    ensure enable often remain high low diverse advanced traditional
    practical quality decision feedback tailor enable offer highlight
    """.split()
)

GENERIC_TECH_VERBS = frozenset(
    """
    language languages natural large processing process processed nlp
    speech voice text image intelligence artificial human interaction
    interface computer datum data information network content base
    enhance enhanced integrate integration generate generation generative
    offer drive power real time leverage highlight challenge optimize
    dynamic insight engineering augment aware engagement accuracy
    efficiency semantic personalize multi learn learning machine deep
    neural recommender recommendation prompt prompts fine tune tuning
    performance task tasks application applications platform service
    digital online web mobile cloud edge
    """.split()
)

# Substantive terms we deliberately KEEP (even if similar to omitted ones)
PROTECTED_SUBSTANTIVE = frozenset(
    """
    rag retrieval hallucination agentic multimodal metaverse vr ar
    trust privacy security bias fairness transparency explainability explanation
    governance risk safety misinformation alignment align ethical ethics
    health mental healthcare patient clinical therapy wellbeing depression
    anxiety education student teacher university classroom tutoring
    customer commerce banking marketing satisfaction intention adoption
    avatar immersive museum cultural heritage robot humanoid hri
    sentiment emotion cognitive load accessibility usability testing
    personalization recommend recommender proactive context-aware
    gemini llama openai copilot
    """.split()
)

# Union used for vectorizer / keyword filtering
OMIT_TERMS: frozenset[str] = (
    (QUERY_UMBRELLA_TERMS | QUERY_UMBRELLA_BIGRAMS | GENERIC_ACADEMIC_TERMS | GENERIC_TECH_VERBS)
    - PROTECTED_SUBSTANTIVE
)


def is_omitted(term: str) -> bool:
    t = term.strip().lower()
    if not t:
        return True
    if t in PROTECTED_SUBSTANTIVE:
        return False
    if t in OMIT_TERMS:
        return True
    if " " in t:
        return any(p in OMIT_TERMS for p in t.split())
    return False


def _tokenize_basic(text: str) -> list[str]:
    return [t for t in str(text).lower().split() if t]


def make_omit_analyzer(omit_terms: frozenset[str] | None = None) -> Callable[[str], list[str]]:
    omit = omit_terms or OMIT_TERMS
    omit_unigrams = {t for t in omit if " " not in t}
    omit_ngrams = {t for t in omit if " " in t}

    def analyzer(doc: str) -> list[str]:
        kept = [t for t in _tokenize_basic(doc) if t not in omit_unigrams]
        out: list[str] = []
        for i, t in enumerate(kept):
            out.append(t)
            if i + 1 < len(kept):
                bg = f"{t} {kept[i + 1]}"
                if bg in omit_ngrams:
                    continue
                if kept[i + 1] in omit_unigrams:
                    continue
                out.append(bg)
        return out

    return analyzer


def filter_terms_for_display(terms: list[str], max_n: int = 15) -> list[str]:
    """Return up to max_n terms that are not omitted (for labels / reports)."""
    out: list[str] = []
    for t in terms:
        if is_omitted(t):
            continue
        out.append(t)
        if len(out) >= max_n:
            break
    return out


def top_weighted_terms(
    weights: list[float] | None,
    vocab_terms: list[str],
    n: int = 15,
) -> list[str]:
    """Pick top terms by weight, skipping omitted tokens."""
    if weights is None:
        ordered = vocab_terms
    else:
        idx = sorted(range(len(vocab_terms)), key=lambda i: weights[i], reverse=True)
        ordered = [vocab_terms[i] for i in idx]
    return filter_terms_for_display(ordered, max_n=n)


# ---------------------------------------------------------------------------
# Journal-quality matplotlib style (Q1/Q2)
# ---------------------------------------------------------------------------

PALETTE = {
    "primary": "#2E5A88",
    "secondary": "#C44E52",
    "accent": "#55A868",
    "neutral": "#4C4C4C",
    "grid": "#CCCCCC",
    "phase": ["#8DA0CB", "#FC8D62", "#66C2A5", "#E78AC3"],
}


def apply_journal_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "axes.titleweight": "normal",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "-",
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
        }
    )


def fig_dir(outdir: Path) -> Path:
    d = outdir / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    try:
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)
