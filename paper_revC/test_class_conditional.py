from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .baselines_wrappers import generate_synth
from .datasets import load_and_split_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test class-conditional synthesis.")
    parser.add_argument("--config", "-c", default="paper_revC/configs/revC.yaml")
    parser.add_argument("--method", "-m", default="tabddpm_single")
    parser.add_argument("--n-samples", "-n", type=int, default=1000)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    method = args.method
    datasets = cfg.get("datasets", [])

    # Override n_samples for the requested method only.
    cfg = dict(cfg)
    cfg.setdefault("method_params", {})
    cfg["method_params"].setdefault(method, {})
    cfg["method_params"][method]["n_samples"] = args.n_samples

    for entry in datasets:
        name = entry if isinstance(entry, str) else entry.get("name")
        split = load_and_split_dataset(name=name)
        synth_df, _ = generate_synth(split, method, seed=0, config=cfg)
        counts = synth_df[split.label_col].value_counts(dropna=False)
        print(f"{name} [{method}] label counts:\n{counts}\n")


if __name__ == "__main__":
    main()
