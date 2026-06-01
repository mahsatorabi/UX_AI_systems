"""
RQ2 — Intellectual and thematic structures

Research question:
    What are the main intellectual and thematic structures of user experience
    research in generative AI systems (incl. conversational AI)?

This script consumes the *preprocessed* corpus (`outputs/corpus_preprocessed.csv`)
and performs a modern multi-view mapping of the field:

  1. Text-side topic modelling on `text_processed`
     - TF–IDF features
     - non-negative matrix factorisation (NMF) for several k
     - automatic selection of k based on a combined score:
         * reconstruction quality (lower error is better)
         * topic sparsity / distinctiveness (higher is better)
         * topic diversity (fraction of unique top-n terms)
  2. Keyword co-occurrence structure
     - build co-word network from `Author Keywords_clean` + `Index Keywords_clean`
     - filter to UX/AI-related terms
     - community detection via Louvain-style greedy modularity (simple heuristic)
  3. Integrated report
     - ranked topic list with top terms
     - mapping between NMF topics and keyword communities (by word overlap)

Outputs in `outputs/`:
  - `rq2_topics.csv`              — NMF topics and top terms
  - `rq2_topic_doc_assignments.csv` — dominant topic per document
  - `rq2_keyword_network_edges.csv` — co-occurrence edges
  - `rq2_keyword_communities.csv`   — keyword clusters
  - `rq2_report.md` / `rq2_report.json`

Run:
    python rq2.py
    python rq2.py --input outputs/corpus_preprocessed.csv --outdir outputs
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

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
    doi_col: str = "DOI"
    title_col: str = "Title"
    author_kw_col: str = "Author Keywords_clean"
    index_kw_col: str = "Index Keywords_clean"

    # topic model
    min_df: int = 5          # ignore ultra-rare terms
    max_df: float = 0.6      # ignore very common terms
    max_features: int = 7000
    topic_k_range: tuple[int, int] = (6, 14)  # inclusive bounds
    nmf_max_iter: int = 400
    nmf_random_state: int = 42
    top_terms_per_topic: int = 15

    # keyword network
    min_keyword_freq: int = 8
    min_edge_weight: int = 4


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="RQ2: thematic structures (topics + co-word)")
    p.add_argument("--input", type=Path, default=Config.input_path)
    p.add_argument("--outdir", type=Path, default=Config.outdir)
    p.add_argument("--min-df", type=int, default=Config.min_df)
    p.add_argument("--max-df", type=float, default=Config.max_df)
    p.add_argument("--max-features", type=int, default=Config.max_features)
    p.add_argument("--k-min", type=int, default=Config.topic_k_range[0])
    p.add_argument("--k-max", type=int, default=Config.topic_k_range[1])
    args = p.parse_args()
    return Config(
        input_path=args.input,
        outdir=args.outdir,
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
        topic_k_range=(args.k_min, args.k_max),
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def load_corpus(cfg: Config) -> pd.DataFrame:
    if not cfg.input_path.exists():
        raise FileNotFoundError(
            f"Input not found: {cfg.input_path}. Run preprocess.py first."
        )
    cols = [
        cfg.text_col,
        cfg.year_col,
        cfg.doi_col,
        cfg.title_col,
        cfg.author_kw_col,
        cfg.index_kw_col,
    ]
    df = pd.read_csv(cfg.input_path, usecols=[c for c in cols if c is not None])
    df[cfg.text_col] = df[cfg.text_col].fillna("").astype(str)
    return df


def build_tfidf_matrix(texts: Iterable[str], cfg: Config) -> tuple[TfidfVectorizer, np.ndarray]:
    analyzer = make_omit_analyzer()
    vec = TfidfVectorizer(
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        max_features=cfg.max_features,
        analyzer=analyzer,
        norm="l2",
    )
    X = vec.fit_transform(texts)
    return vec, X


def topic_diversity(components: np.ndarray, top_n: int, vocab: list[str]) -> float:
    """
    Topic diversity = fraction of unique terms among all top-n words per topic.
    """
    m, _ = components.shape
    all_terms: list[str] = []
    for k in range(m):
        idx = np.argsort(components[k])[::-1][:top_n]
        all_terms.extend(vocab[i] for i in idx)
    return len(set(all_terms)) / max(1, len(all_terms))


def topic_sparsity(components: np.ndarray) -> float:
    """
    Sparsity proxy: fraction of weights that are below a small threshold.
    """
    thr = np.percentile(components, 75)  # keep only strongest 25% as "non-sparse"
    return float(np.mean(components < thr))


def evaluate_nmf_models(
    X, k_values: list[int], cfg: Config, vocab: list[str]
) -> tuple[NMF, np.ndarray, int]:
    """
    Fit NMF for several k, pick the best by a composite score:
      score = -reconstruction_error (scaled) + alpha*sparsity + beta*diversity
    """
    best_score = -np.inf
    best_model: NMF | None = None
    best_W: np.ndarray | None = None
    best_k: int = -1

    errs: list[float] = []
    models: dict[int, NMF] = {}
    Ws: dict[int, np.ndarray] = {}

    for k in k_values:
        model = NMF(
            n_components=k,
            init="nndsvda",
            max_iter=cfg.nmf_max_iter,
            random_state=cfg.nmf_random_state,
            alpha_W=0.0,
            alpha_H=0.0,
        )
        W = model.fit_transform(X)
        H = model.components_
        err = float(model.reconstruction_err_)
        errs.append(err)
        models[k] = model
        Ws[k] = W

        div = topic_diversity(H, top_n=min(20, len(vocab)), vocab=vocab)
        spars = topic_sparsity(H)
        models[k].__dict__["_topic_diversity"] = div
        models[k].__dict__["_topic_sparsity"] = spars

    # scale errors to [0,1]
    e_arr = np.array(errs)
    e_scaled = (e_arr.max() - e_arr) / (e_arr.max() - e_arr.min() + 1e-9)
    alpha, beta = 0.4, 0.6

    for i, k in enumerate(k_values):
        model = models[k]
        div = model.__dict__["_topic_diversity"]
        spars = model.__dict__["_topic_sparsity"]
        score = float(e_scaled[i] + alpha * spars + beta * div)
        if score > best_score:
            best_score = score
            best_model = model
            best_W = Ws[k]
            best_k = k

    assert best_model is not None and best_W is not None
    return best_model, best_W, best_k


def nmf_topics(X, cfg: Config, vocab: list[str]) -> tuple[pd.DataFrame, np.ndarray, NMF]:
    k_min, k_max = cfg.topic_k_range
    k_values = list(range(k_min, k_max + 1))
    model, W, best_k = evaluate_nmf_models(X, k_values, cfg, vocab)
    H = model.components_

    records: list[dict] = []
    for k in range(best_k):
        row = H[k]
        ranked = [vocab[i] for i in np.argsort(row)[::-1]]
        terms = filter_terms_for_display(ranked, max_n=cfg.top_terms_per_topic)
        weights = [float(row[vocab.index(t)]) for t in terms if t in vocab]
        records.append(
            {
                "topic_id": k,
                "top_terms": ", ".join(terms),
                "top_terms_list": "; ".join(terms),
                "topic_diversity": getattr(model, "_topic_diversity", np.nan),
                "topic_sparsity": getattr(model, "_topic_sparsity", np.nan),
            }
        )
    df_topics = pd.DataFrame.from_records(records)
    return df_topics, W, model


# ---------------------------------------------------------------------------
# Keyword co-occurrence network
# ---------------------------------------------------------------------------


def parse_keywords(cell: str) -> list[str]:
    if not isinstance(cell, str):
        return []
    # Split on semicolons or commas; lowercased; strip
    parts: list[str] = []
    for chunk in cell.split(";"):
        for sub in chunk.split(","):
            t = sub.strip().lower()
            if t:
                parts.append(t)
    return parts


def build_keyword_network(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_kw: list[list[str]] = []
    for _, row in df.iterrows():
        kws: list[str] = []
        if cfg.author_kw_col in row:
            kws.extend(parse_keywords(row[cfg.author_kw_col]))
        if cfg.index_kw_col in row:
            kws.extend(parse_keywords(row[cfg.index_kw_col]))
        # de-duplicate per document
        kws = sorted(set(kws))
        if kws:
            all_kw.append(kws)

    all_kw = [[k for k in doc if not is_omitted(k)] for doc in all_kw]

    freq = Counter([k for doc in all_kw for k in doc])
    # filter to reasonably frequent keywords
    keep = {k for k, c in freq.items() if c >= cfg.min_keyword_freq}

    edges = Counter()
    for kws in all_kw:
        filtered = [k for k in kws if k in keep]
        for i in range(len(filtered)):
            for j in range(i + 1, len(filtered)):
                a, b = sorted((filtered[i], filtered[j]))
                edges[(a, b)] += 1

    edge_rows = [
        {"source": a, "target": b, "weight": int(w)}
        for (a, b), w in edges.items()
        if w >= cfg.min_edge_weight
    ]
    df_edges = pd.DataFrame(edge_rows)
    df_nodes = pd.DataFrame(
        [{"keyword": k, "freq": int(freq[k])} for k in keep]
    )
    return df_nodes, df_edges


def louvain_like_communities(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """
    Simple greedy modularity-style clustering without external deps.
    This is *not* full Louvain, but a lightweight approximation:
      - start with each node in its own community
      - repeatedly move nodes to neighbour communities if weighted degree
        inside community increases
    """
    if edges.empty or nodes.empty:
        nodes["community"] = np.arange(len(nodes))
        return nodes

    # adjacency
    nbrs: dict[str, dict[str, float]] = defaultdict(dict)
    for _, r in edges.iterrows():
        a = str(r["source"])
        b = str(r["target"])
        w = float(r["weight"])
        nbrs[a][b] = nbrs[a].get(b, 0.0) + w
        nbrs[b][a] = nbrs[b].get(a, 0.0) + w

    comm = {k: i for i, k in enumerate(nodes["keyword"])}

    improved = True
    iters = 0
    max_iter = 20
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for k in nodes["keyword"]:
            k = str(k)
            cur_c = comm[k]
            neigh = nbrs.get(k, {})
            if not neigh:
                continue
            # total weight to each neighbouring community
            weight_per_comm: dict[int, float] = defaultdict(float)
            for nb, w in neigh.items():
                weight_per_comm[comm[nb]] += w
            # choose best community
            best_c = cur_c
            best_w = weight_per_comm.get(cur_c, 0.0)
            for c, w in weight_per_comm.items():
                if w > best_w + 1e-9:
                    best_w = w
                    best_c = c
            if best_c != cur_c:
                comm[k] = best_c
                improved = True

    nodes = nodes.copy()
    nodes["community"] = [comm[str(k)] for k in nodes["keyword"]]
    return nodes


# ---------------------------------------------------------------------------
# Integration + reporting
# ---------------------------------------------------------------------------


def map_topics_to_keyword_communities(
    df_topics: pd.DataFrame, kw_nodes: pd.DataFrame
) -> dict[int, dict]:
    """
    For each topic, see which keyword communities share the most top terms.
    """
    # build community -> set(keywords)
    comm_terms: dict[int, set[str]] = defaultdict(set)
    for _, r in kw_nodes.iterrows():
        comm_terms[int(r["community"])].add(str(r["keyword"]))

    mapping: dict[int, dict] = {}
    for _, r in df_topics.iterrows():
        tid = int(r["topic_id"])
        raw = r["top_terms_list"]
        if isinstance(raw, str):
            terms = [t.strip().lower() for t in raw.replace(";", ",").split(",") if t.strip()]
        else:
            terms = [str(t).strip().lower() for t in raw]
        best_comm = None
        best_overlap = 0
        for c, kws in comm_terms.items():
            ov = len(set(terms) & kws)
            if ov > best_overlap:
                best_overlap = ov
                best_comm = c
        if best_comm is not None and best_overlap > 0:
            mapping[tid] = {"community": best_comm, "overlap": int(best_overlap)}
        else:
            mapping[tid] = {"community": None, "overlap": 0}
    return mapping


def write_outputs(
    df: pd.DataFrame,
    cfg: Config,
    df_topics: pd.DataFrame,
    W: np.ndarray,
    kw_nodes: pd.DataFrame,
    kw_edges: pd.DataFrame,
    topic_comm_map: dict[int, dict],
) -> dict:
    cfg.outdir.mkdir(parents=True, exist_ok=True)

    # topic table
    topics_path = cfg.outdir / "rq2_topics.csv"
    df_topics.to_csv(topics_path, index=False, encoding="utf-8")

    # doc-topic assignment (dominant topic)
    doc_topic = np.asarray(W)
    dom_topic = doc_topic.argmax(axis=1)
    max_weight = doc_topic.max(axis=1)
    df_assign = pd.DataFrame(
        {
            "topic_id": dom_topic,
            "topic_weight": max_weight,
        }
    )
    if cfg.year_col in df.columns:
        df_assign[cfg.year_col] = df[cfg.year_col].values
    if cfg.doi_col in df.columns:
        df_assign[cfg.doi_col] = df[cfg.doi_col].values
    if cfg.title_col in df.columns:
        df_assign[cfg.title_col] = df[cfg.title_col].values

    assign_path = cfg.outdir / "rq2_topic_doc_assignments.csv"
    df_assign.to_csv(assign_path, index=False, encoding="utf-8")

    # keyword network
    kw_nodes_path = cfg.outdir / "rq2_keyword_communities.csv"
    kw_edges_path = cfg.outdir / "rq2_keyword_network_edges.csv"
    kw_nodes.to_csv(kw_nodes_path, index=False, encoding="utf-8")
    kw_edges.to_csv(kw_edges_path, index=False, encoding="utf-8")

    # High-level headline
    n_topics = int(df_topics["topic_id"].nunique())
    by_topic = df_assign["topic_id"].value_counts().sort_index()
    topic_sizes = {int(k): int(v) for k, v in by_topic.items()}

    def _terms_list(cell) -> list[str]:
        if isinstance(cell, str):
            return [t.strip() for t in cell.replace(";", ",").split(",") if t.strip()]
        return list(cell)[:3]

    topic_labels = {
        int(r["topic_id"]): ", ".join(_terms_list(r["top_terms_list"])[:3])
        for _, r in df_topics.iterrows()
    }

    # community sizes
    comm_sizes = kw_nodes["community"].value_counts().to_dict()

    headline = {
        "n_documents": int(len(df)),
        "n_topics": n_topics,
        "topic_sizes": topic_sizes,
        "topic_labels": topic_labels,
        "n_keyword_nodes": int(len(kw_nodes)),
        "n_keyword_edges": int(len(kw_edges)),
        "community_sizes": {int(k): int(v) for k, v in comm_sizes.items()},
        "topic_to_community": {
            int(t): {"community": m["community"], "overlap": m["overlap"]}
            for t, m in topic_comm_map.items()
        },
    }

    # write JSON + Markdown narrative
    json_path = cfg.outdir / "rq2_report.json"
    json_path.write_text(json.dumps(headline, indent=2), encoding="utf-8")

    md_path = cfg.outdir / "rq2_report.md"
    md_path.write_text(render_report(cfg, df_topics, headline), encoding="utf-8")

    return headline


def render_report(cfg: Config, df_topics: pd.DataFrame, headline: dict) -> str:
    # top 8 topics by size
    topic_sizes = headline["topic_sizes"]
    labels = headline["topic_labels"]
    top_topic_ids = sorted(topic_sizes, key=topic_sizes.get, reverse=True)[:8]
    lines: list[str] = [
        "# RQ2 Report — Thematic Structures",
        "",
        f"**Input:** `{cfg.input_path}`  ",
        f"**Documents:** **{headline['n_documents']}**  ",
        f"**NMF topics:** **{headline['n_topics']}**  ",
        "",
        "## Major topics (NMF on full text)",
        "",
    ]
    for tid in top_topic_ids:
        label = labels.get(tid, f"Topic {tid}")
        size = topic_sizes[tid]
        row = df_topics[df_topics["topic_id"] == tid].iloc[0]
        terms = row["top_terms"]
        lines.append(f"- **T{tid} — {label}** (n={size} docs)")
        lines.append(f"  - Top terms: {terms}")
    lines.append("")

    # keyword communities
    lines.append("## Keyword communities (co-word structure)")
    lines.append("")
    comm_sizes = headline["community_sizes"]
    for cid, size in sorted(comm_sizes.items(), key=lambda x: x[1], reverse=True)[:8]:
        # show up to 10 keywords from each community
        lines.append(f"- **C{cid}** (n={size} keywords)")
    lines.append("")

    return "\n".join(lines)


def _parse_topic_terms(cell) -> list[str]:
    if isinstance(cell, str):
        return [t.strip() for t in cell.replace(";", ",").split(",") if t.strip()]
    return list(cell)


def plot_rq2_figures(
    cfg: Config,
    df_topics: pd.DataFrame,
    df_assign: pd.DataFrame,
    kw_nodes: pd.DataFrame,
    kw_edges: pd.DataFrame,
    headline: dict,
) -> None:
    apply_journal_style()
    out = fig_dir(cfg.outdir)

    # Fig 1: topic prevalence (horizontal bar, top 8)
    sizes = headline["topic_sizes"]
    labels = headline["topic_labels"]
    top_ids = sorted(sizes, key=sizes.get, reverse=True)[:8]
    counts = [sizes[t] for t in top_ids]
    ylabels = [labels.get(t, f"Topic {t}") for t in top_ids]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y_pos = np.arange(len(top_ids))
    ax.barh(y_pos, counts, color=PALETTE["primary"], alpha=0.88, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of documents (dominant topic)")
    ax.set_title("Fig. RQ2a. Major thematic topics in the corpus")
    save_figure(fig, out / "rq2_fig_topics")

    # Fig 2: keyword co-occurrence network (top nodes)
    if not kw_edges.empty and not kw_nodes.empty:
        top_n = 90
        top_kws = (
            kw_nodes.sort_values("freq", ascending=False)
            .head(top_n)["keyword"]
            .astype(str)
            .tolist()
        )
        sub_n = kw_nodes[kw_nodes["keyword"].astype(str).isin(top_kws)].copy()
        sub_e = kw_edges[
            kw_edges["source"].astype(str).isin(top_kws)
            & kw_edges["target"].astype(str).isin(top_kws)
        ].copy()

        G = nx.Graph()
        for _, r in sub_n.iterrows():
            G.add_node(
                str(r["keyword"]),
                freq=int(r["freq"]),
                community=int(r["community"]),
            )
        for _, r in sub_e.iterrows():
            G.add_edge(str(r["source"]), str(r["target"]), weight=float(r["weight"]))

        pos = nx.spring_layout(G, seed=42, k=0.45, iterations=200)
        comms = sorted({G.nodes[n]["community"] for n in G.nodes()})
        cmap = plt.get_cmap("tab10")
        comm_color = {c: cmap(i % 10) for i, c in enumerate(comms)}

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        max_w = max((d["weight"] for _, _, d in G.edges(data=True)), default=1.0)
        for u, v, d in G.edges(data=True):
            w = d["weight"]
            ax.plot(
                [pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                color="#B0B0B0",
                alpha=0.12 + 0.4 * (w / max(1.0, max_w)),
                linewidth=0.6,
            )
        freqs = np.array([G.nodes[n]["freq"] for n in G.nodes()], dtype=float)
        sizes = 30 + 180 * (freqs / max(1.0, freqs.max()))
        for n in G.nodes():
            ax.scatter(
                pos[n][0],
                pos[n][1],
                s=sizes[list(G.nodes()).index(n)],
                c=[comm_color[G.nodes[n]["community"]]],
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
        for n in sorted(G.nodes(), key=lambda x: G.nodes[x]["freq"], reverse=True)[:18]:
            ax.text(pos[n][0], pos[n][1], n, fontsize=6.5, ha="center", va="center")
        ax.set_title("Fig. RQ2b. Keyword co-occurrence structure (top terms)")
        ax.axis("off")
        save_figure(fig, out / "rq2_fig_keyword_network")

    # Fig 3: topic prevalence over time (stacked area, 2018+)
    if cfg.year_col in df_assign.columns:
        dfa = df_assign.copy()
        dfa = dfa[dfa[cfg.year_col] >= 2018]
        if not dfa.empty:
            share = (
                dfa.groupby([cfg.year_col, "topic_id"])
                .size()
                .unstack(fill_value=0)
                .div(dfa.groupby(cfg.year_col).size(), axis=0)
            )
            top_t = top_ids[:6]
            share = share[[c for c in top_t if c in share.columns]]

            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            ax.stackplot(
                share.index.astype(int),
                [share[c].values for c in share.columns],
                labels=[labels.get(int(c), f"T{c}") for c in share.columns],
                alpha=0.85,
            )
            ax.set_xlabel("Year")
            ax.set_ylabel("Topic share")
            ax.set_title("Fig. RQ2c. Evolution of thematic prevalence (2018–2026)")
            ax.legend(loc="upper left", fontsize=7, frameon=True)
            save_figure(fig, out / "rq2_fig_topic_evolution")


def main() -> None:
    cfg = parse_args()
    df = load_corpus(cfg)

    vec, X = build_tfidf_matrix(df[cfg.text_col].tolist(), cfg)
    vocab = list(vec.get_feature_names_out())
    df_topics, W, _ = nmf_topics(X, cfg, vocab)

    kw_nodes, kw_edges = build_keyword_network(df, cfg)
    kw_nodes = louvain_like_communities(kw_nodes, kw_edges)

    topic_comm_map = map_topics_to_keyword_communities(df_topics, kw_nodes)

    headline = write_outputs(df, cfg, df_topics, W, kw_nodes, kw_edges, topic_comm_map)

    df_assign = pd.read_csv(cfg.outdir / "rq2_topic_doc_assignments.csv")
    plot_rq2_figures(cfg, df_topics, df_assign, kw_nodes, kw_edges, headline)

    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()

