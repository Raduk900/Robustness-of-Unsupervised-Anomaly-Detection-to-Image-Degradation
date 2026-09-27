# Robustness of Unsupervised Anomaly Detection to Image Degradation

Code and results for the paper. Five methods (GANomaly, PaDiM, PatchCore,
Reverse Distillation, EfficientAD) are trained on clean MVTec AD images and
tested under seven perturbations at five severities, 15 categories, 3 seeds.

## Requirements

- NVIDIA GPU with at least 6 GB of memory, CUDA driver ≥ 12.6
- conda, Python 3.11 (installed by the setup script)
- about 6 GB for the dataset, 17 GB for the seed-0 checkpoints

## 1. Environment

```bash
python setup/setup_env.py        # creates the conda env "anomaly"
conda activate anomaly
```

`requirements.txt` pins the versions of the direct dependencies, including the
CUDA build of torch.
Do not run `anomalib install`.

## 2. Data

Download MVTec AD from https://www.mvtec.com/company/research/datasets/mvtec-ad
and extract it so that each category is a folder under `dataset/`:

```
dataset/<category>/train/good, test/<defect>, ground_truth/<defect>
```

```bash
python scripts/check_dataset.py   # checks the layout
python setup/run_setup_test.py    # ~10 min smoke test; must pass
```

EfficientAD downloads ImageNette into `datasets/` on its first run.

## 3. Experiments

The released `results/` already contain every row, so a run in place would
skip everything. To reproduce, move `results/` aside first.

```bash
python scripts/run_all.py                          # main sweep -> results/stress_all.csv
python scripts/run_all.py --native                 # -> results/legacy_protocol/stress_all.csv
python scripts/run_all.py --native --task baseline # -> results/legacy_protocol/baseline_all.csv
python scripts/raw_score_check.py                  # -> results/raw_score_check.csv
python scripts/threshold_check.py                  # -> results/image_scores.csv
```

- Main sweep: perturbation applied at the 256 px model input, full test set
  (7875 rows, several days on one GPU). It also saves the seed-0 models of
  the four methods with an anomaly map (all but GANomaly) to `checkpoints/`.
- `--native`: perturbation applied before the resize, half of each test set
  held out.
- `raw_score_check.py`: re-evaluates the saved seed-0 models on unclamped
  scores under both placements; no training.
- `threshold_check.py`: records every test image's raw score and the model's
  threshold for the saved seed-0 models (all conditions at the model input,
  the two noise perturbations also at native resolution), from which the
  false-reject and miss rates at a fixed threshold follow; no training.

Runs can be interrupted and resumed with the same command. `run_all.py`
accepts `--categories`, `--methods`, `--seeds`, and `--quick` for a
minutes-long rehearsal whose numbers are meaningless.

## 4. Tables, figures and numbers in the paper

These read the CSVs above; the category table (native resolution column) and
the image figures also read `dataset/`.
Output goes to `paper/tables`, `paper/figures` and `results/tables`.

```bash
python scripts/export_latex.py
python scripts/make_figures.py
python scripts/text_numbers.py
python scripts/compare_protocols.py --export
python scripts/measure_kappa.py
```

## Notes

- Anomalib maps every score s to (s − t)/(max − min) + 0.5, with the F1
  threshold t and the extremes of the unperturbed validation images, and
  clamps the result to [0, 1] before computing AUROC. When a perturbation
  pushes all scores above this range, the image AUROC is exactly 0.5. The
  main sweep is evaluated this way; `raw_score_check.py` measures the effect.
- F1 columns are diagnostic only: their threshold is fitted on the evaluated
  images.
- GPU training is not bit-reproducible, and noise perturbations are random.
  Re-running can give slightly different values from the released CSVs;
  PaDiM and PatchCore are otherwise deterministic. The checkpoints used for
  `raw_score_check.csv` and `image_scores.csv` are not included (17 GB).

## Released results

| File | Contents |
|---|---|
| `results/stress_all.csv` | main sweep |
| `results/legacy_protocol/stress_all.csv` | native-resolution sweep |
| `results/legacy_protocol/baseline_all.csv` | separately trained clean baselines |
| `results/raw_score_check.csv` | raw vs clamped AUROC, seed-0 models |
| `results/image_scores.csv` | per-image raw score and threshold, seed-0 models |
| `results/tables/*.csv` | placement comparison and numbers quoted in the text |
| `results/training_times.csv` | training time per model, main sweep (from the run logs) |
