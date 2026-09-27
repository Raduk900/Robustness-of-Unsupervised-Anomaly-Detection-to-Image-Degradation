import _paths  # puts scripts/ on sys.path

import os
from dataclasses import dataclass
from typing import Callable

from common import RESULTS_DIR, ensure_repo_cwd, make_arg_parser


@dataclass(frozen=True)
class MethodSpec:

    name: str
    paradigm: str
    build: Callable
    train_batch_size: int
    eval_batch_size: int
    max_epochs: int
    # False: no anomaly map, so the pixel columns stay empty and no seed-0
    # checkpoint is saved
    pixel_output: bool = True
    notes: str = ""


def method_main(spec):
    parser = make_arg_parser(
        f"{spec.name} ({spec.paradigm}) -- robustness experiment for this "
        f"method only.",
        methods=False,
    )
    parser.add_argument("--mode", choices=["stress", "baseline"],
                        default="stress",
                        help="stress: train + full perturbation sweep; "
                             "baseline: train + clean test only")
    parser.add_argument("--output", default=None,
                        help="results CSV (default: results/<mode>_<method>.csv, "
                             "under results/legacy_protocol/ with --native)")
    args = parser.parse_args()

    ensure_repo_cwd()
    suffix = "_quick" if args.quick else ""
    out_dir = os.path.join(RESULTS_DIR, "legacy_protocol") if args.native else RESULTS_DIR
    output = args.output or os.path.join(
        out_dir, f"{args.mode}_{spec.name.lower()}{suffix}.csv")

    from runner import run_baselines, run_stress

    print(f"=== {spec.name} ({spec.paradigm}) | mode={args.mode} ===")
    print(f"    categories: {len(args.categories)}  seeds: {args.seeds}")
    print(f"    output: {output}")
    if spec.notes:
        print(f"    note: {spec.notes}")

    if args.mode == "baseline":
        run_baselines(spec, args, output)
    else:
        run_stress(spec, args, output)
