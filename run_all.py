"""
Reproduce the full analysis pipeline (preprocessing + RQ1–RQ3).

Usage:
    python run_all.py
    python run_all.py --skip-preprocess   # if outputs/corpus_preprocessed.csv exists
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"\n>>> {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def main() -> None:
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Run full UX × GenAI bibliometric pipeline")
    p.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocess.py (use existing outputs/corpus_preprocessed.csv)",
    )
    p.add_argument("--input", default=str(root / "data.csv"), help="Scopus export CSV")
    p.add_argument("--outdir", default=str(root / "outputs"), help="Output directory")
    args = p.parse_args()

    py = sys.executable
    out = args.outdir

    if not args.skip_preprocess:
        if not Path(args.input).is_file():
            print(
                f"ERROR: {args.input} not found.\n"
                "Place your Scopus export as data.csv (see data/README.md) or pass --input."
            )
            sys.exit(1)
        run([py, str(root / "preprocess.py"), "--input", args.input, "--output-dir", out])

    corpus = Path(out) / "corpus_preprocessed.csv"
    if not corpus.is_file():
        print(f"ERROR: {corpus} missing. Run preprocessing first.")
        sys.exit(1)

    run([py, str(root / "rq1.py"), "--input", str(corpus), "--outdir", out])
    run([py, str(root / "rq2.py"), "--input", str(corpus), "--outdir", out])
    run([py, str(root / "rq3.py"), "--input", str(corpus), "--outdir", out])
    print("\nDone. Figures: outputs/figures/  |  Reports: outputs/rq*_report.md")


if __name__ == "__main__":
    main()
