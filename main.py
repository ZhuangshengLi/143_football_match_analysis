"""
Football Match Analysis - Main Entry

Usage:
    python main.py process   # Feature engineering -> processed_dataset.npz
    python main.py train     # Model A/B/C/D ablation
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "inputs" / "raw" / "database.sqlite"
DATA_PATH = ROOT / "outputs" / "clean" / "processed_dataset.npz"
sys.path.insert(0, str(ROOT))


def process(args):
    from inputs.match_features import build_match_dataset
    db = Path(args.db) if args.db else DB_PATH
    out = Path(args.out) if args.out else DATA_PATH
    print("Feature engineering (L1/L2/L3)...")
    df = build_match_dataset(db, out)
    print(f"Done shape={df.shape}")


def train(args):
    from inputs.match_features import load_match_dataset
    from models.ablation import run_ablation
    path = Path(args.data) if args.data else DATA_PATH
    if not path.exists():
        print(f"Error: {path} does not exist. Run python main.py process first.")
        return
    df = load_match_dataset(path)
    print(f"Data {df.shape}, split 80/20 by time\n")
    results = run_ablation(df)
    print("--- Model A/B/C/D ---")
    print(results.to_string(index=False))


def main():
    p = argparse.ArgumentParser(description="Football Match Analysis")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("process", help="Feature engineering")
    sp.add_argument("--db", "-d", help="Database path")
    sp.add_argument("--out", "-o", help="Output path")
    sp.set_defaults(func=process)

    st = sub.add_parser("train", help="A/B/C/D ablation")
    st.add_argument("--data", help="Data path")
    st.set_defaults(func=train)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
