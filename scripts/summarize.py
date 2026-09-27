import argparse
import os

import pandas as pd

from common import REPORTED_METRICS, RESULTS_DIR, ROOT

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

TABLE_NAMES = ["coverage", "baselines", "maxseverity", "degradation",
               "overall", "bycategory", "bytype"]

IDENTITY = 0


def load(path):
    if not os.path.exists(path):
        raise SystemExit(
            f"no results at {path}\n"
            f"Run the experiment first (python scripts/run_all.py), or point "
            f"--input at another CSV.")
    frame = pd.read_csv(path)
    frame["ok"] = frame["status"] == "ok"
    return frame


def _max_severity_rows(frame):
    top = frame.groupby("perturbation")["severity_index"].transform("max")
    return frame[frame["severity_index"] == top]


def _identity_rows(frame):
    return frame[frame["severity_index"] == IDENTITY]


def _with_degradation(frame):
    keys = ["category", "method", "seed", "perturbation"]
    identity = _identity_rows(frame)[keys + REPORTED_METRICS]
    identity = identity.rename(columns={m: f"{m}__identity" for m in REPORTED_METRICS})
    merged = frame.merge(identity, on=keys, how="left")
    for metric in REPORTED_METRICS:
        merged[f"delta_{metric}"] = merged[metric] - merged[f"{metric}__identity"]
    return merged


def table_coverage(frame, full):
    print(f"rows: {len(frame)}   ok: {int(frame['ok'].sum())}   "
          f"not ok: {int((~frame['ok']).sum())}")

    units = frame.groupby(["category", "method", "seed"]).size()
    n_perturbations = frame["perturbation"].nunique()
    print(f"units (category x method x seed) present: {len(units)}")
    print(f"perturbations seen: {n_perturbations}")

    if not frame["ok"].all():
        bad = frame[~frame["ok"]]
        print("\nnon-ok rows by status:")
        print(bad["status"].value_counts().to_string())
        print("\naffected units:")
        print(bad.groupby(["category", "method", "status"]).size()
              .rename("rows").reset_index().to_string(index=False))

    expected = frame.groupby("perturbation")["severity_index"].nunique().sum()
    short = units[units < expected]
    if len(short):
        print(f"\nincomplete units (fewer than {expected} rows):")
        print(short.rename("rows").reset_index().to_string(index=False))
    else:
        print(f"\nevery present unit has its full {expected} rows")
    return None


def table_baselines(frame, full):
    identity = _identity_rows(frame[frame["ok"]])
    per_unit = (identity.groupby(["method", "category", "seed"])[REPORTED_METRICS]
                .mean().reset_index())
    out = (per_unit.groupby("method")[REPORTED_METRICS]
           .agg(["mean", "std"]).round(4))
    if full:
        detail = (per_unit.groupby(["method", "category"])[REPORTED_METRICS]
                  .mean().round(4))
        print(detail.to_string())
        print()
    return out


def table_maxseverity(frame, full):
    rows = _max_severity_rows(frame[frame["ok"]])
    per_unit = rows.groupby(["method", "perturbation", "category", "seed"])[
        "image_AUROC"].mean().reset_index()
    return (per_unit.groupby(["method", "perturbation"])["image_AUROC"]
            .mean().unstack().round(3))


def table_degradation(frame, full):
    rows = _max_severity_rows(_with_degradation(frame[frame["ok"]]))
    per_unit = rows.groupby(["method", "perturbation", "category", "seed"])[
        "delta_image_AUROC"].mean().reset_index()
    table = (per_unit.groupby(["method", "perturbation"])["delta_image_AUROC"]
             .mean().unstack().round(3))
    if full:
        spread = (per_unit.groupby(["method", "perturbation"])["delta_image_AUROC"]
                  .std().unstack().round(3))
        print("std across categories and seeds:")
        print(spread.to_string())
        print()
    return table


def table_overall(frame, full):
    scored = _with_degradation(frame[frame["ok"]])
    perturbed = scored[scored["severity_index"] != IDENTITY]
    per_unit = perturbed.groupby(["method", "category", "seed"])[
        "delta_image_AUROC"].mean().reset_index()
    out = per_unit.groupby("method")["delta_image_AUROC"].agg(
        robustness_score="mean", std_across_units="std").round(4)

    clean = (_identity_rows(frame[frame["ok"]])
             .groupby("method")["image_AUROC"].mean().round(4))
    out.insert(0, "clean_AUROC", clean)
    return out.sort_values("robustness_score", ascending=False)


def table_bycategory(frame, full):
    rows = _max_severity_rows(_with_degradation(frame[frame["ok"]]))
    per_unit = rows.groupby(["category", "method", "seed"])[
        "delta_image_AUROC"].mean().reset_index()
    table = (per_unit.groupby(["category", "method"])["delta_image_AUROC"]
             .mean().unstack().round(3))
    table["mean"] = table.mean(axis=1).round(3)
    return table.sort_values("mean")


def table_bytype(frame, full):
    rows = _max_severity_rows(_with_degradation(frame[frame["ok"]]))
    per_unit = rows.groupby(["category_type", "perturbation", "category",
                             "method", "seed"])["delta_image_AUROC"].mean().reset_index()
    return (per_unit.groupby(["category_type", "perturbation"])["delta_image_AUROC"]
            .mean().unstack().round(3))


TABLES = {
    "coverage": (table_coverage, "Coverage and failures"),
    "baselines": (table_baselines, "Clean-data baselines (identity rows)"),
    "maxseverity": (table_maxseverity, "Image AUROC at maximum severity"),
    "degradation": (table_degradation, "AUROC degradation at maximum severity"),
    "overall": (table_overall, "Overall robustness per method"),
    "bycategory": (table_bycategory, "Degradation by category"),
    "bytype": (table_bytype, "Degradation by category type"),
}


def main():
    parser = argparse.ArgumentParser(
        description='Print summary tables of a results CSV for all five methods '
                    '(a completeness check; the paper tables come from export_latex.py).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", default=os.path.join(RESULTS_DIR, "stress_all.csv"),
                        help="results CSV (default: results/stress_all.csv)")
    parser.add_argument("--tables", nargs="+", default=TABLE_NAMES,
                        choices=TABLE_NAMES, metavar="NAME",
                        help=f"which tables to print ({' '.join(TABLE_NAMES)})")
    parser.add_argument("--full", action="store_true",
                        help="also show per-category detail and spreads")
    parser.add_argument("--export", action="store_true",
                        help="write each table to results/tables/<name>.csv")
    args = parser.parse_args()

    frame = load(args.input)
    print(f"source: {args.input}\n")

    export_dir = os.path.join(RESULTS_DIR, "tables")
    if args.export:
        os.makedirs(export_dir, exist_ok=True)

    for name in args.tables:
        function, title = TABLES[name]
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
        table = function(frame, args.full)
        if table is not None:
            print(table.to_string())
            if args.export:
                path = os.path.join(export_dir, f"{name}.csv")
                table.to_csv(path)
                print(f"\n-> {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
