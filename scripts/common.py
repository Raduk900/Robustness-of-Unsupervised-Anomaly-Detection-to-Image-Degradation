import argparse
import csv
import os
import platform

import torch
from anomalib.callbacks.checkpoint import ModelCheckpoint
from anomalib.data import Folder
from anomalib.data.utils import ValSplitMode
from anomalib.engine import Engine
from torchvision.transforms import v2

torch.set_float32_matmul_precision("high")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_ROOT = os.path.join(ROOT, "dataset")
RESULTS_DIR = os.path.join(ROOT, "results")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")

RUN_DIR = os.path.join(ROOT, "runs")
MAIN_SWEEP = os.path.join(RESULTS_DIR, "stress_all.csv")
NATIVE_SWEEP = os.path.join(RESULTS_DIR, "legacy_protocol", "stress_all.csv")

TEXTURE_CATEGORIES = ["carpet", "grid", "leather", "tile", "wood"]
OBJECT_CATEGORIES = [
    "bottle", "cable", "capsule", "hazelnut", "metal_nut",
    "pill", "screw", "toothbrush", "transistor", "zipper",
]
ALL_CATEGORIES = TEXTURE_CATEGORIES + OBJECT_CATEGORIES

INPUT_SIZE = [256, 256]

SPLIT_SEED = 42

METRIC_KEYS = ["image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score"]
REPORTED_METRICS = ["image_AUROC", "pixel_AUROC"]
DEFAULT_SEEDS = [0, 1, 2]

# Windows starts each dataloader worker by re-importing torch; with a new
# datamodule per severity that startup dominates, so load in-process there.
DEFAULT_WORKERS = 0 if platform.system() == "Windows" else 4


def category_type(category):
    return "texture" if category in TEXTURE_CATEGORIES else "object"


def discover_defects(category_root):
    test_dir = os.path.join(category_root, "test")
    return sorted(
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d)) and d != "good"
    )


def ensure_repo_cwd():
    os.chdir(ROOT)


def make_resize():
    return v2.Resize(INPUT_SIZE, interpolation=v2.InterpolationMode.BILINEAR,
                     antialias=True)


def make_test_augmentation(perturbation):
    # anomalib appends the model's own resize after test_augmentations; resizing
    # first makes the perturbation act on the 256 px model input.
    return v2.Compose([make_resize(), perturbation])


def make_datamodule(category, category_root, train_bs, eval_bs,
                    num_workers=DEFAULT_WORKERS, test_aug=None, native=False):
    defects = discover_defects(category_root)
    if native:
        # native-resolution sweep: perturb the native-resolution image before
        # the model's resize, and hold out half of the test set (anomalib's
        # defaults)
        split = dict(val_split_mode=ValSplitMode.FROM_TEST, val_split_ratio=0.5)
        test_augmentations = test_aug
    else:
        # main sweep: evaluate on the full test set; the unperturbed validation
        # copy supplies anomalib's threshold and normalization statistics
        split = dict(val_split_mode=ValSplitMode.SAME_AS_TEST)
        test_augmentations = (make_test_augmentation(test_aug)
                              if test_aug is not None else None)
    # Folder rather than MVTecAD: the MVTecAD validation split produced empty
    # subsets in anomalib 2.5.0.
    return Folder(
        name=category, root=category_root,
        normal_dir="train/good", normal_test_dir="test/good",
        abnormal_dir=[f"test/{d}" for d in defects],
        mask_dir=[f"ground_truth/{d}" for d in defects],
        train_batch_size=train_bs, eval_batch_size=eval_bs,
        num_workers=num_workers, seed=SPLIT_SEED,
        **split,
        test_augmentations=test_augmentations,
    )


def make_engine(max_epochs, validate_last_only=True, quick=0):
    kwargs = {}
    # validation only feeds the post-processor statistics (and EfficientAD's
    # map quantiles), so the last epoch is enough
    if validate_last_only and max_epochs > 1:
        kwargs["check_val_every_n_epoch"] = max_epochs
    if quick:
        max_epochs = 1
        kwargs.pop("check_val_every_n_epoch", None)
        kwargs.update(limit_train_batches=quick, limit_val_batches=quick,
                      limit_test_batches=quick)
    return Engine(
        max_epochs=max_epochs, accelerator="auto", devices=1,
        default_root_dir=RUN_DIR,
        # no automatic checkpoint per fit; runner.py saves the seed-0 models
        callbacks=[ModelCheckpoint(dirpath=os.path.join(RUN_DIR, "unused"),
                                   save_top_k=0)],
        **kwargs,
    )


def results_to_metrics(results):
    metrics = {}
    for key, value in (results[0] if results else {}).items():
        try:
            metrics[key] = float(value)
        except (TypeError, ValueError):
            pass
    return metrics


def metric_columns(metrics):
    return [f"{metrics[k]:.6f}" if k in metrics else "" for k in METRIC_KEYS]


def append_csv_row(path, header, row):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(header)
        writer.writerow(row)


def read_completed_keys(path, key_cols):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {tuple(row[c] for c in key_cols) for row in csv.DictReader(f)}


def make_arg_parser(description, methods=True):
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                        help="MVTec AD root: <data-root>/<category>/{train,test,ground_truth}")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="dataloader workers (0 is the fast choice on "
                             "Windows -- see DEFAULT_WORKERS in common.py)")
    parser.add_argument("--categories", nargs="+", default=ALL_CATEGORIES,
                        choices=ALL_CATEGORIES, metavar="CAT",
                        help="subset of the 15 categories")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="random seeds")
    parser.add_argument("--quick", type=int, nargs="?", const=2, default=0,
                        metavar="N",
                        help="rehearsal mode: cap every stage at N batches "
                             "(default 2) and train one epoch, so the whole "
                             "pipeline runs in minutes. The resulting numbers "
                             "are meaningless -- this only proves nothing "
                             "crashes. Writes to a *_quick.csv by default")
    parser.add_argument("--native", action="store_true",
                        help="native-resolution sweep: perturb before the "
                             "resize and hold out half of each test set; "
                             "writes to results/legacy_protocol/")
    if methods:
        from registry import METHOD_NAMES
        parser.add_argument("--methods", nargs="+", default=list(METHOD_NAMES),
                            choices=list(METHOD_NAMES), metavar="METHOD",
                            help=f"subset of methods ({' '.join(METHOD_NAMES)})")
    return parser


def load_sweep(path):
    import pandas as pd
    if not os.path.exists(path):
        raise SystemExit(f"results not found: {path}")
    df = pd.read_csv(path)
    return df[df.status == "ok"]


def resolve_category_root(data_root, category):
    path = os.path.join(data_root, category)
    return path if os.path.isdir(path) else None
