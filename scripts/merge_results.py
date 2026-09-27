import argparse
import glob
import os

import pandas as pd

from common import RESULTS_DIR

KEY_COLUMNS = {
    "stress": ["category", "method", "seed", "perturbation", "severity_index"],
    "baseline": ["category", "method", "seed"],
}


def main():
    parser = argparse.ArgumentParser(
        description='Merge per-method result CSVs into one file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", choices=["stress", "baseline"], default="stress")
    parser.add_argument("--inputs", nargs="+", default=None,
                        help="explicit input CSVs (default: results/<task>_*.csv, "
                             "excluding the merged output itself)")
    parser.add_argument("--output", default=None,
                        help="default: results/<task>_all.csv")
    parser.add_argument("--native", action="store_true",
                        help="use results/legacy_protocol/ instead of results/")
    args = parser.parse_args()

    out_dir = os.path.join(RESULTS_DIR, "legacy_protocol") if args.native else RESULTS_DIR
    output = args.output or os.path.join(out_dir, f"{args.task}_all.csv")
    inputs = args.inputs
    if inputs is None:
        pattern = os.path.join(out_dir, f"{args.task}_*.csv")
        inputs = [p for p in sorted(glob.glob(pattern))
                  if os.path.abspath(p) != os.path.abspath(output)]
    if not inputs:
        raise SystemExit(f"no input CSVs found for task '{args.task}'")

    frames = []
    for path in inputs:
        frame = pd.read_csv(path)
        print(f"  {os.path.basename(path):<40} {len(frame):>6} rows")
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    keys = [c for c in KEY_COLUMNS[args.task] if c in merged.columns]
    merged = merged.drop_duplicates(subset=keys, keep="last")
    merged = merged.sort_values(keys).reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    merged.to_csv(output, index=False)

    print(f"\nmerged {len(inputs)} files: {before} rows -> {len(merged)} "
          f"after de-duplication")
    print(f"written to {output}")

    failed = merged[merged.get("status", "ok") != "ok"] if "status" in merged else []
    if len(failed):
        print(f"\n!! {len(failed)} row(s) have a non-ok status:")
        print(failed[[*keys, "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
