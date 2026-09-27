import argparse
import os
import time
import traceback

import torch
from anomalib.engine import Engine
from lightning.pytorch.callbacks import Callback

from common import (ALL_CATEGORIES, CHECKPOINT_DIR, DEFAULT_DATA_ROOT,
                    DEFAULT_WORKERS, RESULTS_DIR, RUN_DIR, append_csv_row,
                    make_datamodule, read_completed_keys,
                    resolve_category_root)
from perturbations import PERTURBATIONS, make_transform
from raw_score_check import METHODS_WITH_CKPT, native_datamodule
from registry import METHODS

OUTPUT = os.path.join(RESULTS_DIR, "image_scores.csv")
HEADER = ["category", "method", "seed", "placement", "perturbation",
          "severity_index", "severity_value", "image_index", "label",
          "raw_score", "threshold"]
KEYS = ["category", "method", "placement", "perturbation", "severity_index"]
# placement matters only for the pixel-unit perturbations
NATIVE_PERTURBATIONS = ("gaussian", "salt_pepper")


class ScoreRecorder(Callback):

    def __init__(self):
        self.reset()

    def reset(self):
        self.scores, self.labels = [], []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch,
                          batch_idx, dataloader_idx=0):
        self.scores.append(outputs.pred_score.detach().float().cpu().reshape(-1))
        self.labels.append(outputs.gt_label.detach().cpu().reshape(-1))


def conditions():
    for pert, (family, severities) in PERTURBATIONS.items():
        for idx, sev in enumerate(severities):
            yield "input", pert, idx, sev
    for pert in NATIVE_PERTURBATIONS:
        for idx, sev in enumerate(PERTURBATIONS[pert][1]):
            if idx > 0:
                yield "native", pert, idx, sev


def run_unit(spec, category, category_root, workers, done, output):
    ckpt = os.path.join(CHECKPOINT_DIR, f"{category}__{spec.name}.ckpt")
    if not os.path.exists(ckpt):
        print(f"  no checkpoint at {ckpt}, skipping")
        return
    model = spec.build()
    model.post_processor.enable_normalization = False
    recorder = ScoreRecorder()
    engine = Engine(accelerator="auto", devices=1, default_root_dir=RUN_DIR,
                    callbacks=[recorder], logger=False)
    loaded = False
    for placement, pert, idx, sev in conditions():
        if (category, spec.name, placement, pert, str(idx)) in done:
            continue
        transform = make_transform(pert, sev)
        if placement == "input":
            dm = make_datamodule(category, category_root, spec.train_batch_size,
                                 spec.eval_batch_size, num_workers=workers,
                                 test_aug=transform)
        else:
            dm = native_datamodule(category, category_root,
                                   spec.train_batch_size,
                                   spec.eval_batch_size, workers, transform)
        try:
            recorder.reset()
            engine.test(model=model, datamodule=dm,
                        ckpt_path=None if loaded else ckpt, verbose=False)
            loaded = True
            scores = torch.cat(recorder.scores).tolist()
            labels = torch.cat(recorder.labels).int().tolist()
            t = float(model.post_processor.image_threshold)
            for i, (score, label) in enumerate(zip(scores, labels)):
                append_csv_row(output, HEADER,
                               [category, spec.name, 0, placement, pert, idx,
                                sev, i, label, f"{score:.6g}", f"{t:.6g}"])
            frr = sum(sc > t for sc, lb in zip(scores, labels) if lb == 0) / max(1, labels.count(0))
            miss = sum(sc <= t for sc, lb in zip(scores, labels) if lb == 1) / max(1, labels.count(1))
            print(f"  {placement:<6} {pert:<12} sev={sev:<5} FRR={frr:.2f} miss={miss:.2f}")
        except Exception:
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Per-image raw scores and thresholds of the saved seed-0 models.")
    parser.add_argument("--categories", nargs="+", default=ALL_CATEGORIES)
    parser.add_argument("--methods", nargs="+", default=METHODS_WITH_CKPT,
                        choices=METHODS_WITH_CKPT)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--output", default=OUTPUT,
                        help="CSV to append to; give each parallel process its own")
    parser.add_argument("--merge", nargs="+", metavar="CSV",
                        help="merge these per-process CSVs into results/image_scores.csv and exit")
    args = parser.parse_args()
    if args.merge:
        import pandas as pd
        merged = pd.concat([pd.read_csv(f) for f in args.merge], ignore_index=True)
        merged = merged.drop_duplicates(subset=KEYS + ["image_index"], keep="last")
        merged.to_csv(OUTPUT, index=False)
        print(f"{len(merged)} rows -> {OUTPUT}")
        return
    done = read_completed_keys(args.output, KEYS)
    for method in args.methods:
        spec = METHODS[method]
        for category in args.categories:
            print(f"\n=== {category} / {method} ===")
            t0 = time.time()
            run_unit(spec, category, resolve_category_root(args.data_root, category),
                     args.workers, done, args.output)
            print(f"  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
