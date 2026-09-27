import os
import time
import traceback

from lightning.pytorch import seed_everything

from common import (CHECKPOINT_DIR, METRIC_KEYS, append_csv_row,
                    category_type, make_datamodule, make_engine,
                    metric_columns, read_completed_keys, resolve_category_root,
                    results_to_metrics)
from perturbations import PERTURBATIONS, make_transform

STRESS_HEADER = ["category", "category_type", "method", "seed", "perturbation",
                 "family", "severity_index", "severity_value", *METRIC_KEYS,
                 "status"]
STRESS_KEYS = ["category", "method", "seed", "perturbation", "severity_index"]

BASELINE_HEADER = ["category", "category_type", "method", "seed",
                   *METRIC_KEYS, "status"]
BASELINE_KEYS = ["category", "method", "seed"]

# written as the perturbation name when training fails, so resume skips it
TRAIN_FAILED = "TRAIN"


def train_model(spec, category, category_root, seed, workers, quick=0,
                native=False):
    seed_everything(seed, workers=True)
    model = spec.build()
    dm = make_datamodule(category, category_root, spec.train_batch_size,
                         spec.eval_batch_size, num_workers=workers,
                         native=native)
    engine = make_engine(spec.max_epochs, quick=quick)
    engine.fit(model=model, datamodule=dm)
    return model, engine


def _save_seed0_checkpoint(engine, category, method_name):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(CHECKPOINT_DIR, f"{category}__{method_name}.ckpt")
    if os.path.exists(path):
        return
    try:
        engine.trainer.save_checkpoint(path)
    except Exception as e:
        print(f"  (checkpoint save failed: {e})")


def run_baselines(spec, args, output):
    done = read_completed_keys(output, BASELINE_KEYS)
    if done:
        print(f"Resuming: {len(done)} rows already in {output}")

    for category in args.categories:
        for seed in args.seeds:
            head = f"{category} / {spec.name} / seed {seed}"
            if (category, spec.name, str(seed)) in done:
                print(f"{head} -- already done, skipping")
                continue
            category_root = resolve_category_root(args.data_root, category)
            if category_root is None:
                print(f"{head} -- not found under {args.data_root}, skipping")
                continue

            print(f"\n--- {head} ---")
            t0 = time.time()
            try:
                model, engine = train_model(spec, category, category_root,
                                            seed, args.workers, args.quick,
                                            args.native)
                dm = make_datamodule(category, category_root,
                                     spec.train_batch_size,
                                     spec.eval_batch_size,
                                     num_workers=args.workers,
                                     native=args.native)
                metrics = results_to_metrics(
                    engine.test(model=model, datamodule=dm, verbose=False))
                status = "ok"
                print(f"  image_AUROC={metrics.get('image_AUROC')}  "
                      f"({time.time() - t0:.0f}s)")
            except Exception as e:
                traceback.print_exc()
                metrics, status = {}, f"failed:{type(e).__name__}"
                print(f"  FAILED: {e}")

            append_csv_row(output, BASELINE_HEADER,
                           [category, category_type(category), spec.name, seed,
                            *metric_columns(metrics), status])


def run_stress(spec, args, output):
    done = read_completed_keys(output, STRESS_KEYS)
    if done:
        print(f"Resuming: {len(done)} rows already in {output}")

    expected = [(p, str(i)) for p, (_f, sevs) in PERTURBATIONS.items()
                for i in range(len(sevs))]
    units = [(c, s) for c in args.categories for s in args.seeds]

    for n, (category, seed) in enumerate(units, 1):
        head = f"[{n}/{len(units)}] {category} / {spec.name} / seed {seed}"
        key = (category, spec.name, str(seed))

        if (*key, TRAIN_FAILED, "-1") in done:
            print(f"{head} -- training failed previously, skipping "
                  f"(delete its rows in {output} to retry)")
            continue
        if all((*key, p, i) in done for p, i in expected):
            print(f"{head} -- already complete, skipping")
            continue
        category_root = resolve_category_root(args.data_root, category)
        if category_root is None:
            print(f"{head} -- not found under {args.data_root}, skipping")
            continue

        print(f"\n{'=' * 70}\n{head}\n{'=' * 70}")
        _run_stress_unit(spec, args, output, category, category_root, seed, done)

    print(f"\nDONE. Results in {output}")


def _run_stress_unit(spec, args, output, category, category_root, seed, done):
    def record(pert, family, idx, sev, metrics, status):
        append_csv_row(output, STRESS_HEADER,
                       [category, category_type(category), spec.name, seed,
                        pert, family, idx, sev,
                        *metric_columns(metrics), status])

    t0 = time.time()
    try:
        model, engine = train_model(spec, category, category_root, seed,
                                    args.workers, args.quick, args.native)
    except Exception as e:
        traceback.print_exc()
        record(TRAIN_FAILED, "", -1, -1, {}, f"train_failed:{type(e).__name__}")
        return
    print(f"  trained in {time.time() - t0:.0f}s")

    # no checkpoint in quick mode: a junk checkpoint would be kept by the real
    # run, which skips existing checkpoints
    if seed == 0 and not args.quick and not args.native and spec.pixel_output:
        _save_seed0_checkpoint(engine, category, spec.name)

    for pert_name, (family, severities) in PERTURBATIONS.items():
        for idx, severity in enumerate(severities):
            if (category, spec.name, str(seed), pert_name, str(idx)) in done:
                continue
            dm_test = make_datamodule(
                category, category_root, spec.train_batch_size,
                spec.eval_batch_size, num_workers=args.workers,
                test_aug=make_transform(pert_name, severity),
                native=args.native,
            )
            try:
                metrics = results_to_metrics(
                    engine.test(model=model, datamodule=dm_test, verbose=False))
                record(pert_name, family, idx, severity, metrics, "ok")
                flag = "  (identity)" if idx == 0 else ""
                print(f"  {pert_name:<12} sev={severity:<5} "
                      f"image_AUROC={metrics.get('image_AUROC')}{flag}")
            except Exception as e:
                record(pert_name, family, idx, severity, {},
                       f"failed:{type(e).__name__}")
                print(f"  !! {pert_name} sev={severity} FAILED: {e}")


