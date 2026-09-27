import sys

from common import (ensure_repo_cwd, make_arg_parser, make_datamodule,
                    resolve_category_root, results_to_metrics)
from methods.padim import SPEC
from perturbations import PERTURBATIONS, make_transform
from runner import train_model

SMOKE_CATEGORIES = ["wood", "metal_nut", "toothbrush"]

IDENTITY_TOL = 1e-6


def main():
    parser = make_arg_parser('Fast pre-flight check -- run this FIRST on a new machine.', methods=False)
    parser.set_defaults(categories=SMOKE_CATEGORIES, seeds=[0])
    args = parser.parse_args()
    ensure_repo_cwd()

    print(f"=== SMOKE TEST: {args.categories} x {SPEC.name} x "
          f"{len(PERTURBATIONS)} perturbations ===")
    problems = []

    for category in args.categories:
        category_root = resolve_category_root(args.data_root, category)
        if category_root is None:
            problems.append(f"{category}: not found under {args.data_root}")
            continue

        print(f"\n--- {category} ---")
        try:
            model, engine = train_model(SPEC, category, category_root, 0, args.workers)
        except Exception as e:
            problems.append(f"{category}: training failed: {e}")
            continue

        def evaluate(test_aug=None):
            dm = make_datamodule(category, category_root, SPEC.train_batch_size,
                                 SPEC.eval_batch_size, num_workers=args.workers,
                                 test_aug=test_aug)
            metrics = results_to_metrics(
                engine.test(model=model, datamodule=dm, verbose=False))
            return metrics.get("image_AUROC")

        baseline = evaluate()
        print(f"  clean baseline image_AUROC = {baseline}")

        moved = False
        for pert_name, (_family, severities) in PERTURBATIONS.items():
            row = {}
            for label, severity in [("IDENTITY", severities[0]),
                                    ("MAX", severities[-1])]:
                try:
                    row[label] = evaluate(make_transform(pert_name, severity))
                except Exception as e:
                    problems.append(f"{category}/{pert_name}/{label}: {e}")
                    row[label] = None

            identity, maximum = row["IDENTITY"], row["MAX"]
            drift = (abs(identity - baseline)
                     if identity is not None and baseline is not None else None)
            mark = ""
            if drift is not None and drift > IDENTITY_TOL:
                mark = f"   !! identity differs from baseline by {drift:.6f}"
                problems.append(
                    f"{category}/{pert_name}: identity AUROC {identity} != "
                    f"baseline {baseline}")
            if (maximum is not None and baseline is not None
                    and abs(maximum - baseline) > IDENTITY_TOL):
                moved = True
            print(f"  {pert_name:<12} identity={identity}  max={maximum}{mark}")

        if not moved:
            problems.append(
                f"{category}: no perturbation changed the AUROC at maximum "
                f"severity -- transforms are probably not being applied")

    print("\n" + "=" * 60)
    if problems:
        print("SMOKE TEST FAILED:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("SMOKE TEST PASSED.")
    print("  identity rows reproduce the clean baseline in every category")
    print("  perturbations measurably move the metric at maximum severity")


if __name__ == "__main__":
    main()

