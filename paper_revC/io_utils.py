from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable


def make_run_dir(run_tag: str | None = None) -> tuple[Path, Path, Path, Path]:
    base_dir = Path('paper_revC') / 'results_revC'
    base_dir.mkdir(parents=True, exist_ok=True)
    if run_tag is None:
        run_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = base_dir / run_tag
    tables_dir = run_dir / 'tables'
    figures_dir = run_dir / 'figures'
    logs_dir = run_dir / 'logs'
    for folder in (tables_dir, figures_dir, logs_dir):
        folder.mkdir(parents=True, exist_ok=True)
    return run_dir, tables_dir, figures_dir, logs_dir


def append_csv_row(
    path: Path,
    row: dict[str, object],
    fieldnames: Iterable[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(fieldnames) if fieldnames is not None else list(row.keys())
    existed = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        if not existed:
            writer.writeheader()
        writer.writerow({k: row.get(k, '') for k in header})
