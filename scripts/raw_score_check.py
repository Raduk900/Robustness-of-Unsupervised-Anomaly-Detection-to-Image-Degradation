import argparse
import os
import time
import traceback

import torch
from anomalib.data import Folder
from anomalib.data.utils import ValSplitMode
from anomalib.engine import Engine
from lightning.pytorch.callbacks import Callback
from sklearn.metrics import roc_auc_score

from common import (ALL_CATEGORIES, CHECKPOINT_DIR, DEFAULT_WORKERS,
                    RESULTS_DIR, RUN_DIR, SPLIT_SEED, append_csv_row,
                    category_type, discover_defects, make_datamodule,
                    read_completed_keys, resolve_category_root,
                    DEFAULT_DATA_ROOT)
from perturbations import PERTURBATIONS, make_transform
from registry import METHODS

OUTPUT = os.path.join(RESULTS_DIR, "raw_score_check.csv")
METHODS_WITH_CKPT = ["PaDiM", "PatchCore", "ReverseDistillation", "EfficientAD"]
PLACEMENTS = ["input", "native"]
HEADER = ["category", "category_type", "method", "seed", "placement",
          "perturbation", "family", "severity_index", "severity_value",
          "image_AUROC_raw", "pixel_AUROC_raw", "image_AUROC_clamped",
          "pixel_AUROC_clamped", "sat_normal", "sat_anomalous", "status"]
KEYS = ["category", "method", "placement", "perturbation", "severity_index"]


# AUROC is computed here with sklearn: torchmetrics treats scores outside
# [0, 1] as logits and applies a sigmoid, which saturates raw scores.
class ScoreRecorder(Callback):

    def __init__(self):
        self.reset()

    def reset(self):
        self.scores, self.labels, self.maps, self.masks = [], [], [], []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch,
                          batch_idx, dataloader_idx=0):
        self.scores.append(outputs.pred_score.detach().float().cpu().reshape(-1))
        self.labels.append(outputs.gt_label.detach().cpu().reshape(-1))
        if outputs.anomaly_map is not None and outputs.gt_mask is not None:
            self.maps.append(outputs.anomaly_map.detach().float().cpu()
                             .reshape(-1))
            self.masks.append(outputs.gt_mask.detach().cpu().reshape(-1).bool())


# As common.make_datamodule, but the perturbation acts on the native-resolution
# image and anomalib's model resize follows it.
def native_datamodule(category, category_root, train_bs, eval_bs, workers,
                      perturbation):
    defects = discover_defects(category_root)
    return Folder(
        name=category, root=category_root,
        normal_dir="train/good", normal_test_dir="test/good",
        abnormal_dir=[f"test/{d}" for d in defects],
        mask_dir=[f"ground_truth/{d}" for d in defects],
        train_batch_size=train_bs, eval_batch_size=eval_bs,
        num_workers=workers, seed=SPLIT_SEED,
        val_split_mode=ValSplitMode.SAME_AS_TEST,
        test_augmentations=perturbation,
    )


# anomalib's normalization (s - t) / (max - min) + 0.5 with the F1 threshold t
# and the unperturbed validation min and max, clamped to [0, 1]; the main
# sweep's AUROC was computed on these scores
def clamped_scores(raw, post):
    return post._normalize(raw, post.image_min.cpu(), post.image_max.cpu(),
                           post.image_threshold.cpu())


def clamped_maps(raw, post):
    return post._normalize(raw, post.pixel_min.cpu(), post.pixel_max.cpu(),
                           post.pixel_threshold.cpu())


def evaluate(engine, model, dm, recorder, ckpt_path=None):
    recorder.reset()
    engine.test(model=model, datamodule=dm, ckpt_path=ckpt_path, verbose=False)
    post = model.post_processor
    raw = torch.cat(recorder.scores)
    labels = torch.cat(recorder.labels).numpy().astype(int)
    norm = clamped_scores(raw, post).numpy()
    sat = norm >= 1.0
    out = {
        "image_AUROC_raw": roc_auc_score(labels, raw.numpy()),
        "image_AUROC_clamped": roc_auc_score(labels, norm),
        "sat_normal": float(sat[labels == 0].mean()),
        "sat_anomalous": float(sat[labels == 1].mean()),
        "pixel_AUROC_raw": None,
        "pixel_AUROC_clamped": None,
    }
    if recorder.maps:
        maps = torch.cat(recorder.maps).float()
        masks = torch.cat(recorder.masks).numpy()
        out["pixel_AUROC_raw"] = roc_auc_score(masks, maps.numpy())
        out["pixel_AUROC_clamped"] = roc_auc_score(
            masks, clamped_maps(maps, post).numpy())
    return out


def run_unit(spec, category, category_root, workers, done, output):
    ckpt = os.path.join(CHECKPOINT_DIR, f"{category}__{spec.name}.ckpt")
    if not os.path.exists(ckpt):
        print(f"  no checkpoint at {ckpt}, skipping")
        return
    model = spec.build()
    recorder = ScoreRecorder()
    engine = Engine(accelerator="auto", devices=1, default_root_dir=RUN_DIR,
                    callbacks=[recorder], logger=False)
    loaded = False

    for placement in PLACEMENTS:
        for pert, (family, severities) in PERTURBATIONS.items():
            for idx, sev in enumerate(severities):
                key = (category, spec.name, placement, pert, str(idx))
                if key in done:
                    continue
                transform = make_transform(pert, sev)
                if placement == "input":
                    dm = make_datamodule(category, category_root,
                                         spec.train_batch_size,
                                         spec.eval_batch_size,
                                         num_workers=workers,
                                         test_aug=transform)
                else:
                    dm = native_datamodule(category, category_root,
                                           spec.train_batch_size,
                                           spec.eval_batch_size, workers,
                                           transform)
                try:
                    if not loaded:
                        # the checkpoint's post-processor statistics are loaded with the weights;
                        # normalization is switched off so the recorded scores are raw
                        model.post_processor.enable_normalization = False
                        r = evaluate(engine, model, dm, recorder, ckpt_path=ckpt)
                        loaded = True
                    else:
                        r = evaluate(engine, model, dm, recorder)
                    fmt = (lambda v: "" if v is None else f"{v:.6f}")
                    row = [fmt(r["image_AUROC_raw"]),
                           fmt(r["pixel_AUROC_raw"]),
                           fmt(r["image_AUROC_clamped"]),
                           fmt(r["pixel_AUROC_clamped"]),
                           f"{r['sat_normal']:.4f}",
                           f"{r['sat_anomalous']:.4f}", "ok"]
                    print(f"  {placement:<6} {pert:<12} sev={sev:<5} "
                          f"raw={r['image_AUROC_raw']:.3f} "
                          f"clamped={r['image_AUROC_clamped']:.3f} "
                          f"sat={r['sat_normal']:.2f}/{r['sat_anomalous']:.2f}")
                except Exception as e:
                    traceback.print_exc()
                    row = ["", "", "", "", "", "", f"failed:{type(e).__name__}"]
                append_csv_row(output, HEADER,
                               [category, category_type(category), spec.name,
                                0, placement, pert, family, idx, sev, *row])


def main():
    parser = argparse.ArgumentParser(description='Re-evaluate the saved seed-0 models on raw, unclamped anomaly scores.')
    parser.add_argument("--categories", nargs="+", default=ALL_CATEGORIES)
    parser.add_argument("--methods", nargs="+", default=METHODS_WITH_CKPT,
                        choices=METHODS_WITH_CKPT)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--output", default=OUTPUT,
                        help="CSV to append to; give each parallel process its own")
    parser.add_argument("--merge", nargs="+", metavar="CSV",
                        help="merge these per-process CSVs into results/raw_score_check.csv and exit")
    args = parser.parse_args()

    if args.merge:
        import pandas as pd
        merged = pd.concat([pd.read_csv(f) for f in args.merge], ignore_index=True)
        merged = merged.drop_duplicates(subset=KEYS, keep="last")
        merged.to_csv(OUTPUT, index=False)
        print(f"{len(merged)} rows -> {OUTPUT}")
        return

    done = read_completed_keys(args.output, KEYS)
    for method in args.methods:
        spec = METHODS[method]
        for category in args.categories:
            category_root = resolve_category_root(args.data_root, category)
            print(f"\n=== {category} / {method} ===")
            t0 = time.time()
            run_unit(spec, category, category_root, args.workers, done,
                     args.output)
            print(f"  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
