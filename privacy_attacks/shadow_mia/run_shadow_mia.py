from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from privacy_attacks.common import distance_features, encode_mixed


def run_shadow_mia(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    seed: int,
    k: int = 5,
) -> float:
    encoded = encode_mixed(train_df=train_df, test_df=test_df, synth_df=synth_df)
    x_train = np.asarray(encoded.X_train, dtype=np.float32)
    x_test = np.asarray(encoded.X_test, dtype=np.float32)
    x_synth = np.asarray(encoded.X_synth, dtype=np.float32)

    member_feat = distance_features(x_train, x_synth, k=k)
    nonmember_feat = distance_features(x_test, x_synth, k=k)

    x = np.vstack([member_feat, nonmember_feat])
    y = np.concatenate([np.ones(len(member_feat)), np.zeros(len(nonmember_feat))])

    if len(np.unique(y)) < 2:
        raise RuntimeError("Shadow-MIA labels must contain both member and non-member classes.")

    bincount = np.bincount(y.astype(int))
    min_count = int(bincount.min()) if bincount.size > 0 else 0
    stratify = y if min_count >= 2 else None

    x_train_att, x_eval_att, y_train_att, y_eval_att = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=seed,
        stratify=stratify,
    )

    if len(np.unique(y_train_att)) < 2:
        x_train_att = x
        y_train_att = y
        x_eval_att = x
        y_eval_att = y

    attacker = LogisticRegression(max_iter=2000, random_state=seed)
    attacker.fit(x_train_att, y_train_att)

    y_prob = attacker.predict_proba(x_eval_att)[:, 1]
    auc = roc_auc_score(y_eval_att, y_prob)
    return float(auc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run shadow-model MIA on synthetic data.")
    parser.add_argument("--real_train_csv", required=True)
    parser.add_argument("--real_test_csv", required=True)
    parser.add_argument("--synthetic_csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "shadow_auc.json"

    try:
        train_df = pd.read_csv(args.real_train_csv)
        test_df = pd.read_csv(args.real_test_csv)
        synth_df = pd.read_csv(args.synthetic_csv)

        auc = run_shadow_mia(
            train_df=train_df,
            test_df=test_df,
            synth_df=synth_df,
            seed=args.seed,
            k=args.k,
        )

        payload = {
            "status": "ok",
            "shadow_auc": auc,
            "k": args.k,
            "error": None,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        payload = {
            "status": "fail",
            "shadow_auc": None,
            "k": args.k,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
