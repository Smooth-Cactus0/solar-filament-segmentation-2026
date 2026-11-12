"""Build the cross-validation folds.

The one rule that matters: **group on `file_stem`, never `image_id`**. The JSON
has 1,154 annotation passes covering only 707 exposures, so splitting on
`image_id` puts the same physical Sun in both train and validation and
contaminates 57.8% of the validation set.

Rows are annotation passes (all 1,154 are training samples -- see the design
doc on why inconsistent labels give us consensus targets for free), grouped by
exposure and stratified on filament-count bin x solar-cycle era, because
coverage across the 2011-2022 span is very uneven.

Usage:
    PYTHONPATH=src python scripts/make_folds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg import data  # noqa: E402
from filseg.paths import PROJECT_ROOT  # noqa: E402

OUT = PROJECT_ROOT / "data" / "folds.csv"
N_FOLDS = 5
SEED = 20260818

# Count bins follow the observed per-pass quartiles (q25 4, median 7, q75 10).
COUNT_BINS = [0, 4, 7, 10, np.inf]
COUNT_LABELS = ["<=4", "5-7", "8-10", ">10"]


def era_of(year: int) -> str:
    """Solar-cycle phase. Cycle 24 peaked ~2014; cycle 25 began Dec 2019."""
    if year <= 2016:
        return "c24_active"
    if year <= 2020:
        return "minimum"
    return "c25_rising"


def build() -> pd.DataFrame:
    ann = data.annotation_table()
    img = data.image_table()

    counts = ann.groupby("image_id").size().rename("n_filaments")
    df = img.merge(counts, left_on="image_id", right_index=True, how="left")
    df["n_filaments"] = df.n_filaments.fillna(0).astype(int)
    df["era"] = df.year.map(era_of)
    df["count_bin"] = pd.cut(df.n_filaments, bins=COUNT_BINS, labels=COUNT_LABELS)
    df["stratum"] = df.era.astype(str) + "|" + df.count_bin.astype(str)

    # Strata too small to appear in every fold are collapsed, first into their
    # era and then into a shared bucket, so no singleton class remains.
    rare = df.stratum.value_counts()
    df.loc[df.stratum.isin(rare[rare < N_FOLDS].index), "stratum"] = df.era
    rare = df.stratum.value_counts()
    df.loc[df.stratum.isin(rare[rare < N_FOLDS].index), "stratum"] = "rare"

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    df["fold"] = -1
    for k, (_, val) in enumerate(sgkf.split(df, df.stratum, groups=df.file_stem)):
        df.iloc[val, df.columns.get_loc("fold")] = k
    return df


def verify(df: pd.DataFrame) -> None:
    """Assert the properties the whole competition depends on."""
    assert (df.fold >= 0).all(), "some rows were not assigned a fold"

    # No exposure may appear in more than one fold.
    spread = df.groupby("file_stem").fold.nunique()
    leaked = spread[spread > 1]
    assert leaked.empty, f"{len(leaked)} exposures span multiple folds"

    # For the record: share of rows that are a repeat labelling of an exposure
    # already present elsewhere in the table. This is NOT the 57.8% figure from
    # the EDA -- that one is the Monte-Carlo probability that a validation pass
    # has a sibling sitting in train under an image_id split.
    dup_share = 1.0 - df.file_stem.nunique() / len(df)
    print(f"\n  exposures spanning folds:      0  (grouped correctly)")
    print(f"  repeat-labelling share of rows: {dup_share:.1%}")


def main() -> None:
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["image_id", "file_stem", "file_name", "fold", "n_filaments",
            "era", "count_bin", "station", "year", "timestamp"]
    df[cols].to_csv(OUT, index=False)

    print(f"wrote {OUT}")
    print(f"  {len(df)} annotation passes over {df.file_stem.nunique()} exposures\n")
    summary = df.groupby("fold").agg(
        passes=("image_id", "size"),
        exposures=("file_stem", "nunique"),
        filaments=("n_filaments", "sum"),
        mean_per_pass=("n_filaments", "mean"),
    ).round(2)
    print(summary.to_string())
    print("\n  era balance per fold (%):")
    print((pd.crosstab(df.fold, df.era, normalize="index") * 100).round(1).to_string())
    verify(df)


if __name__ == "__main__":
    main()
