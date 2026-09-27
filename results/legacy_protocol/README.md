# Native-resolution sweep

Produced with `python scripts/run_all.py --native`: perturbations applied at
native resolution before the resize to 256 px, half of each test set held out
for validation, separately trained models. Same 15 categories, 5 methods,
3 seeds and severities as the main sweep (7875 rows).

`baseline_all.csv` holds separately trained clean baselines (225 rows).

Kept in a subfolder so that `merge_results.py` never mixes it with the main
sweep. Compare the two with `python scripts/compare_protocols.py`.
