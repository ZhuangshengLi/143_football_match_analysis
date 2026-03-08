"""Football Match Analysis: process | train

CLI commands:
  process  - Build L1/L2/L3 features from European Soccer DB, save to .npz
  train    - Run A/B/C/D ablation (LR on incremental feature sets), print metrics
"""
import argparse
import sys
from pathlib import Path

# Default paths (override via CLI args)
ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "inputs" / "raw" / "database.sqlite"
DATA_PATH = ROOT / "outputs" / "clean" / "processed_dataset.npz"
sys.path.insert(0, str(ROOT))


def process(args):
    """Build L1/L2/L3 features from DB, save to .npz."""
    from inputs.match_features import build_match_dataset
    db = Path(args.db) if args.db else DB_PATH
    out = Path(args.out) if args.out else DATA_PATH
    min_date = getattr(args, "min_date", None)  # e.g. 2010-02-22 for valid L2 (Team_Attributes)
    build_match_dataset(db, out, min_date=min_date)


def train(args):
    """Run A/B/C/D ablation, print metrics. Requires processed_dataset.npz from process."""
    from inputs.match_features import load_match_dataset
    from models.ablation import run_ablation
    path = Path(args.data) if args.data else DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"Run process first: {path}")
    df = load_match_dataset(path)
    results = run_ablation(df)
    print(results.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description="Football Match Analysis")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("process", help="Feature engineering")
    sp.add_argument("--db", "-d", help="Database path")
    sp.add_argument("--out", "-o", help="Output path")
    sp.add_argument("--min-date", help="Min match date for valid L2 (e.g. 2010-02-22)")
    sp.set_defaults(func=process)

    st = sub.add_parser("train", help="A/B/C/D ablation")
    st.add_argument("--data", help="Data path")
    st.set_defaults(func=train)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
