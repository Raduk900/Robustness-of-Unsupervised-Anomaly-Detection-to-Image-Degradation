import argparse
import os

import pandas as pd

from common import MAIN_SWEEP, NATIVE_SWEEP, RESULTS_DIR, ROOT

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)

# the native-resolution sweep (historical folder name)
LEGACY_DIR = os.path.join(RESULTS_DIR, "legacy_protocol")
TABLE_NAMES = ["summary", "detail", "ranking", "worst", "baseline"]

NOT_SCALE_INVARIANT = {"gaussian", "salt_pepper"}

KEYS = ["category", "method", "seed", "perturbation"]

WORKING = ["PaDiM", "PatchCore", "ReverseDistillation", "EfficientAD"]
TIE = 0.0005


def load_pair():
    legacy = pd.read_csv(NATIVE_SWEEP)
    current = pd.read_csv(MAIN_SWEEP)

    for name, frame in (("native-resolution", legacy), ("input-referred", current)):
        bad = int((frame.status != "ok").sum())
        if bad:
            print(f"  warning: {bad} non-ok rows in the {name} sweep, dropped")
    return legacy[legacy.status == "ok"], current[current.status == "ok"]


def _degradation(df):
    top = df.groupby("perturbation")["severity_index"].transform("max")
    peak = df[df.severity_index == top].set_index(KEYS)["image_AUROC"]
    identity = df[df.severity_index == 0].set_index(KEYS)["image_AUROC"]
    return (peak - identity).reset_index(name="delta")


def _overall(df):
    identity = df[df.severity_index == 0].set_index(KEYS)["image_AUROC"]
    perturbed = df[df.severity_index != 0].set_index(KEYS)["image_AUROC"]
    return ((perturbed - identity).reset_index(name="delta")
            .groupby("method")["delta"].mean())


def table_summary(legacy, current):
    def mean(df):
        d = _degradation(df[df.method.isin(WORKING)])
        return (d.groupby(["category", "perturbation"])["delta"].mean()
                .groupby("perturbation").mean())
    old, new = mean(legacy), mean(current)
    out = pd.DataFrame({"native": old, "input": new})
    out["diff"] = out["input"] - out["native"]
    out["scale_invariant"] = [p not in NOT_SCALE_INVARIANT for p in out.index]
    return out.round(3).sort_values("diff")


def table_detail(legacy, current):
    old = _degradation(legacy).groupby(["method", "perturbation"])["delta"].mean()
    new = _degradation(current).groupby(["method", "perturbation"])["delta"].mean()
    out = pd.DataFrame({"native": old, "input": new})
    out["diff"] = out["input"] - out["native"]
    out["factor"] = (out["input"] / out["native"]).where(out["native"].abs() > 0.02)
    return out.round(3)


def table_ranking(legacy, current):
    out = pd.DataFrame({"native": _overall(legacy), "input": _overall(current)})
    out["rank_native"] = out["native"].rank(ascending=False).astype(int)
    out["rank_input"] = out["input"].rank(ascending=False).astype(int)
    out["rank_differs"] = out.rank_native != out.rank_input
    return out.round(4).sort_values("input", ascending=False)


def table_worst(legacy, current):
    def worst(df):
        g = (_degradation(df)
             .groupby(["method", "perturbation"])["delta"].mean().unstack())
        return pd.Series({m: " / ".join(sorted(
            p for p in g.columns if g.loc[m, p] <= g.loc[m].min() + TIE))
            for m in g.index})

    out = pd.DataFrame({"native": worst(legacy), "input": worst(current)})
    out["differs"] = out["native"] != out["input"]
    return out


def table_baseline(legacy, current):
    path = os.path.join(LEGACY_DIR, "baseline_all.csv")
    if not os.path.exists(path):
        print(f"  (skipped: {path} not found)")
        return None
    separate = (pd.read_csv(path)
                .set_index(["category", "method", "seed"])["image_AUROC"])
    identity = (legacy[legacy.severity_index == 0]
                .groupby(["category", "method", "seed"])["image_AUROC"].mean())
    both = pd.DataFrame({"separate_run": separate,
                         "identity_rows": identity}).dropna()
    both["abs_diff"] = (both.identity_rows - both.separate_run).abs()

    out = both.groupby(level="method")["abs_diff"].agg(
        mean_diff="mean", max_diff="max", units="count")
    out["units_over_0_05"] = both.groupby(level="method")["abs_diff"].apply(
        lambda s: int((s > 0.05).sum()))
    return out.round(4).sort_values("mean_diff")


TABLES = {
    "summary": (table_summary, "Mean degradation per perturbation"),
    "detail": (table_detail, "Degradation per method and perturbation"),
    "ranking": (table_ranking, "Overall robustness ranking"),
    "worst": (table_worst, "Worst perturbation per method"),
    "baseline": (table_baseline, "Separate baseline vs matched identity rows"),
}


def main():
    parser = argparse.ArgumentParser(
        description='Compare the native-resolution sweep with the input-referred one.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--tables", nargs="+", default=TABLE_NAMES,
                        choices=TABLE_NAMES, metavar="NAME",
                        help=f"which tables to print ({' '.join(TABLE_NAMES)})")
    parser.add_argument("--export", action="store_true",
                        help="write each table to results/tables/protocol_<name>.csv")
    args = parser.parse_args()

    legacy, current = load_pair()
    print(f"native-resolution sweep : {len(legacy)} rows")
    print(f"input-referred sweep    : {len(current)} rows")

    export_dir = os.path.join(RESULTS_DIR, "tables")
    if args.export:
        os.makedirs(export_dir, exist_ok=True)

    for name in args.tables:
        function, title = TABLES[name]
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
        table = function(legacy, current)
        if table is not None:
            print(table.to_string())
            if args.export:
                path = os.path.join(export_dir, f"protocol_{name}.csv")
                table.to_csv(path)
                print(f"\n-> {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
