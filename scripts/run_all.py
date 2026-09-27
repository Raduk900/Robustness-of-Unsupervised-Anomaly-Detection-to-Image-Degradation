import os
import subprocess
import sys
import time

from common import RESULTS_DIR, ROOT, ensure_repo_cwd, make_arg_parser
from registry import METHODS, MODULE_FILES, select

LOG_DIR = os.path.join(ROOT, "logs")


def parse_args():
    parser = make_arg_parser('Orchestrator -- run the experiment for every method.')
    parser.add_argument("--mode", choices=["joint", "separate"], default="joint",
                        help="all methods in one process, or one subprocess each")
    parser.add_argument("--task", choices=["stress", "baseline"], default="stress",
                        help="stress: full perturbation sweep; "
                             "baseline: clean-data results only")
    parser.add_argument("--output", default=None,
                        help="joint mode only: results CSV "
                             "(default: results/<task>_all.csv, under "
                             "results/legacy_protocol/ with --native)")
    return parser.parse_args()


def run_joint(args):
    from runner import run_baselines, run_stress

    suffix = "_quick" if args.quick else ""
    out_dir = os.path.join(RESULTS_DIR, "legacy_protocol") if args.native else RESULTS_DIR
    output = args.output or os.path.join(out_dir, f"{args.task}_all{suffix}.csv")
    specs = select(args.methods)
    print(f"=== joint run | {len(specs)} methods x {len(args.categories)} "
          f"categories x {len(args.seeds)} seeds ===")
    print(f"    output: {output}\n")

    for spec in specs:
        print(f"\n{'#' * 70}\n# {spec.name} ({spec.paradigm})\n{'#' * 70}")
        if args.task == "baseline":
            run_baselines(spec, args, output)
        else:
            run_stress(spec, args, output)

    print(f"\nALL METHODS DONE. Results in {output}")


def run_separate(args):
    os.makedirs(LOG_DIR, exist_ok=True)
    specs = select(args.methods)
    print(f"=== separate run | {len(specs)} methods, one subprocess each ===\n")

    failures = []
    for spec in specs:
        module = MODULE_FILES[spec.name]
        log_path = os.path.join(LOG_DIR, f"{args.task}_{spec.name.lower()}.log")
        cmd = [
            sys.executable, module,
            "--mode", args.task,
            "--data-root", args.data_root,
            "--workers", str(args.workers),
            "--categories", *args.categories,
            "--seeds", *[str(s) for s in args.seeds],
        ]
        if args.quick:
            cmd += ["--quick", str(args.quick)]
        if args.native:
            cmd += ["--native"]
        print(f"--- {spec.name} --- (log: {log_path})")
        print("+ " + " ".join(cmd))
        t0 = time.time()
        with open(log_path, "a", encoding="utf-8") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        mins = (time.time() - t0) / 60
        if result.returncode == 0:
            print(f"    ok ({mins:.0f} min)")
        else:
            print(f"    FAILED with exit code {result.returncode} "
                  f"after {mins:.0f} min -- see {log_path}")
            failures.append(spec.name)

    out_dir = os.path.join(RESULTS_DIR, "legacy_protocol") if args.native else RESULTS_DIR
    native = " --native" if args.native else ""
    print(f"\nSeparate run finished. Results in {out_dir}")
    print(f"Merge them with: python scripts/merge_results.py --task {args.task}{native}")
    if failures:
        print(f"\n!! {len(failures)} method(s) failed: {', '.join(failures)}")
        raise SystemExit(1)


def main():
    args = parse_args()
    ensure_repo_cwd()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    unknown = set(args.methods) - set(METHODS)
    if unknown:
        raise SystemExit(f"unknown method(s): {', '.join(sorted(unknown))}")

    if args.mode == "joint":
        run_joint(args)
    else:
        run_separate(args)


if __name__ == "__main__":
    main()
