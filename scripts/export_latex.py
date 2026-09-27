import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

from common import (MAIN_SWEEP, NATIVE_SWEEP, RESULTS_DIR, ROOT,
                    TEXTURE_CATEGORIES, load_sweep)

# GANomaly is at chance on clean data, so aggregates use these four
WORKING = ["PaDiM", "PatchCore", "ReverseDistillation", "EfficientAD"]
# mean degradations closer than this are reported as tied
TIE = 0.0005


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def pfmt(p, bold=None):
    s = "<0.001" if p < 0.001 else f"{p:.3f}"
    bold = p < 0.05 if bold is None else bold
    return f"$\\mathbf{{{s}}}$" if bold else f"${s}$"

OUT = os.path.join(ROOT, "paper", "tables")
KEYS = ["category", "method", "seed", "perturbation"]

METHODS = ["PaDiM", "PatchCore", "ReverseDistillation", "EfficientAD", "GANomaly"]
NAMES = {"PaDiM": "PaDiM", "PatchCore": "PatchCore",
         "ReverseDistillation": "Rev.\\ Dist.", "EfficientAD": "EfficientAD",
         "GANomaly": "GANomaly"}
PERTS = ["brightness", "contrast", "saturation", "gaussian", "salt_pepper",
         "rotation", "translation"]
PNAMES = {"brightness": "Bright.", "contrast": "Contr.", "saturation": "Satur.",
          "gaussian": "Gauss.", "salt_pepper": "S\\&P", "rotation": "Rot.",
          "translation": "Transl."}


def with_delta(df, metric="image_AUROC"):
    ident = df[df.severity_index == 0].set_index(KEYS)[metric].rename("ident")
    out = df.join(ident, on=KEYS)
    out["delta"] = out[metric] - out["ident"]
    return out


def at_max(df):
    top = df.groupby("perturbation")["severity_index"].transform("max")
    return df[df.severity_index == top]


def seed_stats(df, by, val):
    per_seed = df.groupby(by + ["seed"])[val].mean()
    return per_seed.groupby(by).agg(["mean", "std"])


def num(v, bold=False):
    if pd.isna(v):
        return "--"
    s = f"{v:.3f}"
    if s == "-0.000":
        s = "0.000"
    return f"$\\mathbf{{{s}}}$" if bold else f"${s}$"


def write(name, body):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name + ".tex"), "w", encoding="utf-8") as f:
        f.write(body)
    print("->", os.path.relpath(os.path.join(OUT, name + ".tex"), ROOT))


def pert_matrix(df, caption, label, star=True):
    t = seed_stats(df, ["method", "perturbation"], "delta")["mean"].unstack()
    rows = []
    for m in [m for m in METHODS if m in t.index]:
        vals = t.loc[m, PERTS]
        worst = ({p for p in PERTS if vals[p] <= vals.min() + TIE}
                 if m in WORKING else set())
        rows.append(NAMES[m] + " & " + " & ".join(
            num(vals[p], p in worst) for p in PERTS) + " \\\\")
    env = "table*" if star else "table"
    head = " & ".join(PNAMES[p] for p in PERTS)
    return (f"\\begin{{{env}}}[t]\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
            "\\centering\n\\footnotesize\n"
            "\\begin{tabular}{l" + "r" * len(PERTS) + "}\n\\toprule\n"
            f"Method & {head} \\\\\n\\midrule\n" + "\n".join(rows) +
            f"\n\\bottomrule\n\\end{{tabular}}\n\\end{{{env}}}\n")


def main():
    new = load_sweep(MAIN_SWEEP)
    old = load_sweep(NATIVE_SWEEP)
    d_new = at_max(with_delta(new))
    d_old = at_max(with_delta(old))

    wn = d_new[d_new.method.isin(WORKING)]
    wo = d_old[d_old.method.isin(WORKING)]
    cat_new = wn.groupby(["category", "perturbation"])["delta"].mean().unstack()
    cat_old = wo.groupby(["category", "perturbation"])["delta"].mean().unstack()
    p_proto = holm([wilcoxon(cat_new[p], cat_old[p]).pvalue for p in PERTS])
    rows = []
    for p, pv in zip(PERTS, p_proto):
        inv = "no" if p in ("gaussian", "salt_pepper") else "yes"
        lo, co = cat_old[p].mean(), cat_new[p].mean()
        rows.append(f"{PNAMES[p]} & {inv} & {num(lo)} & {num(co)} & "
                    f"{num(co - lo, pv < 0.05)} & {pfmt(pv)} \\\\")
    write("protocol", "\\begin{table}[t]\n"
          "\\caption{Mean image-AUROC degradation at maximum severity with the "
          "perturbation applied at native resolution, before the resize (half the "
          "test set), and at the model input, after the resize (full test set), "
          "averaged over 15 categories, the four above-chance methods and 3 seeds. "
          "Diff.: input minus native. $p$: Wilcoxon signed-rank test paired over "
          "categories, Holm-adjusted over the seven perturbations.}\n"
          "\\label{tab:protocol}\n\\centering\n\\footnotesize\n"
          "\\setlength{\\tabcolsep}{4pt}\n"
          "\\begin{tabular}{lcrrrr}\n\\toprule\n"
          "Perturbation & Scale-inv. & Native & Input & Diff. & $p$ \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    allp4 = with_delta(new[new.method.isin(WORKING)])
    allp4 = allp4[allp4.severity_index != 0]
    r_cat = allp4.groupby(["category", "method"])["delta"].mean().unstack()
    pairs = [(a, b) for i, a in enumerate(WORKING) for b in WORKING[i + 1:]]
    p_pairs = holm([wilcoxon(r_cat[a], r_cat[b]).pvalue for a, b in pairs])
    rows = []
    for (a, b), pv in zip(pairs, p_pairs):
        diff = (r_cat[a] - r_cat[b]).median()
        wins = int((r_cat[a] > r_cat[b]).sum())
        rows.append(f"{NAMES[a]} -- {NAMES[b]} & {num(diff)} & {wins}/15 & "
                    f"{pfmt(pv)} \\\\")
    write("pairs", "\\begin{table}[t]\n"
          "\\caption{Pairwise comparison of overall robustness $R$ over the 15 "
          "categories. Median difference (first minus second method), number of "
          "categories in which the first method is more robust, and Wilcoxon "
          "signed-rank $p$, Holm-adjusted over the six pairs.}\n"
          "\\label{tab:pairs}\n\\centering\n\\footnotesize\n"
          "\\begin{tabular}{lrrr}\n\\toprule\n"
          "Pair & Median diff. & More robust & $p$ \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    def worst(df):
        g = df.groupby(["method", "perturbation"])["delta"].mean().unstack()
        return {m: " / ".join(PNAMES[p] for p in PERTS
                              if g.loc[m, p] <= g.loc[m].min() + TIE)
                for m in g.index}
    wo, wn = worst(d_old), worst(d_new)
    rows = [f"{NAMES[m]} & {wo[m]} & {wn[m]} \\\\"
            for m in METHODS if m != "GANomaly"]
    write("worst", "\\begin{table}[t]\n"
          "\\caption{Perturbation with the largest image-AUROC degradation, per "
          "method, with the perturbation applied at native resolution and at the "
          "model input, on Anomalib's clamped scores. Two entries: tied within "
          "0.0005.}\n"
          "\\label{tab:worst}\n\\centering\n"
          "\\footnotesize\n\\begin{tabular}{lll}\n\\toprule\n"
          "Method & Native res. & Model input \\\\\n\\midrule\n" + "\n".join(rows) +
          "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    ident = new[new.severity_index == 0].groupby(
        ["category", "method", "seed"])[["image_AUROC", "pixel_AUROC"]].mean().reset_index()
    img = seed_stats(ident, ["method"], "image_AUROC")
    pix = seed_stats(ident, ["method"], "pixel_AUROC")
    allp = with_delta(new)
    allp = allp[allp.severity_index != 0]
    rob = seed_stats(allp, ["method"], "delta")
    rows = []
    for m in METHODS:
        p = ("--" if pd.isna(pix.loc[m, "mean"])
             else f"${pix.loc[m, 'mean']:.3f} \\pm {pix.loc[m, 'std']:.3f}$")
        rows.append(f"{NAMES[m]} & ${img.loc[m, 'mean']:.3f} \\pm {img.loc[m, 'std']:.3f}$ "
                    f"& {p} & ${rob.loc[m, 'mean']:.3f} \\pm {rob.loc[m, 'std']:.3f}$ \\\\")
    write("overall", "\\begin{table}[t]\n"
          "\\caption{Clean image- and pixel-level AUROC and overall robustness $R$ "
          "(Eq.~\\eqref{eq:robustness}) over all 28 perturbed conditions. Mean "
          "$\\pm$ standard deviation over three seeds.}\n\\label{tab:overall}\n"
          "\\centering\n\\footnotesize\n\\begin{tabular}{lrrr}\n\\toprule\n"
          "Method & Image AUROC & Pixel AUROC & $R$ \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    sd = seed_stats(d_new, ["method", "perturbation"], "delta")["std"]
    write("degradation", pert_matrix(
        d_new, "Image-AUROC degradation $\\Delta$ (Eq.~\\eqref{eq:degradation}) "
        "at maximum severity, averaged over 15 categories and 3 seeds, on "
        "Anomalib's clamped scores (Section~\\ref{sec:raw}). The "
        f"largest degradation per above-chance method is in bold, both values "
        f"where two are tied within 0.0005. Seed standard deviation "
        f"$\\le {sd.max():.3f}$ in every cell.", "tab:degradation"))
    dp = at_max(with_delta(new[new.method != "GANomaly"], "pixel_AUROC"))
    write("pixel", pert_matrix(
        dp, "Pixel-AUROC degradation at maximum severity, on Anomalib's clamped "
        "scores. GANomaly produces no "
        "anomaly map and is omitted.", "tab:pixel"))

    bc = d_new.groupby(["category", "method"])["delta"].mean().unstack()
    working = [m for m in METHODS if m != "GANomaly"]
    bc["mean"] = bc[working].mean(axis=1)
    res = {}
    for c in bc.index:
        f = sorted(glob.glob(os.path.join(ROOT, "dataset", c, "train", "good", "*")))[0]
        from PIL import Image
        res[c] = Image.open(f).size[0]
    rows = []
    for c in bc.sort_values("mean").index:
        typ = "T" if c in TEXTURE_CATEGORIES else "O"
        rows.append(f"{c.replace('_', ' ')} & {typ} & {res[c]} & " +
                    " & ".join(num(bc.loc[c, m]) for m in working) +
                    f" & {num(bc.loc[c, 'mean'], True)} \\\\")
    write("category", "\\begin{table}[t]\n"
          "\\caption{Image-AUROC degradation per category at maximum severity, "
          "averaged over the seven perturbations and three seeds, sorted by the "
          "mean over the four methods. T: texture, O: object.}\n"
          "\\label{tab:category}\n\\centering\n\\footnotesize\n"
          "\\setlength{\\tabcolsep}{3pt}\n"
          "\\begin{tabular}{lcrrrrrr}\n\\toprule\n"
          "Category & & Res. & PaDiM & PatchC. & RD & EffAD & Mean \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    d4 = d_new[d_new.method != "GANomaly"].copy()
    d4["type"] = d4.category.map(lambda c: "texture" if c in TEXTURE_CATEGORIES else "object")
    tt = d4.groupby(["type", "perturbation"])["delta"].mean().unstack()
    per_cat = d4.groupby(["category", "perturbation"])["delta"].mean().unstack()
    tex = [c for c in per_cat.index if c in TEXTURE_CATEGORIES]
    obj = [c for c in per_cat.index if c not in TEXTURE_CATEGORIES]
    p_type = holm([mannwhitneyu(per_cat.loc[tex, p], per_cat.loc[obj, p],
                                alternative="two-sided").pvalue for p in PERTS])
    rows = [f"{PNAMES[p]} & {num(tt.loc['texture', p])} & {num(tt.loc['object', p])} & "
            f"{num(tt.loc['object', p] - tt.loc['texture', p], pv < 0.05)} & {pfmt(pv)} \\\\"
            for p, pv in zip(PERTS, p_type)]
    write("type", "\\begin{table}[t]\n"
          "\\caption{Image-AUROC degradation at maximum severity for texture (5) "
          "and object (10) categories, averaged over the four above-chance "
          "methods and three seeds. Difference: object minus texture. $p$: "
          "Mann--Whitney $U$ test over categories, Holm-adjusted over the seven "
          "perturbations.}\n\\label{tab:type}\n"
          "\\centering\n\\footnotesize\n\\begin{tabular}{lrrrr}\n\\toprule\n"
          "Perturbation & Texture & Object & Difference & $p$ \\\\\n\\midrule\n"
          + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    raw_path = os.path.join(RESULTS_DIR, "raw_score_check.csv")
    if os.path.exists(raw_path):
        write("raw", raw_table(pd.read_csv(raw_path)))
    scores_path = os.path.join(RESULTS_DIR, "image_scores.csv")
    if os.path.exists(scores_path):
        write("threshold", threshold_table(pd.read_csv(scores_path)))


def threshold_rates(scores):
    scores = scores[scores.method.isin(WORKING)].copy()
    scores["reject"] = scores.raw_score > scores.threshold
    keys = ["category", "method", "placement", "perturbation", "severity_index"]
    frr = scores[scores.label == 0].groupby(keys)["reject"].mean().rename("frr")
    miss = (1 - scores[scores.label == 1].groupby(keys)["reject"].mean()).rename("miss")
    rates = pd.concat([frr, miss], axis=1).reset_index()
    return (rates.groupby(["method", "placement", "perturbation", "severity_index"])
            [["frr", "miss"]].mean().reset_index())


def threshold_table(scores):
    rates = threshold_rates(scores)
    mean = rates.groupby(["placement", "perturbation", "severity_index"])[["frr", "miss"]].mean()
    pct = lambda v: f"{100 * v:.0f}"
    clean = mean.loc[("input", "brightness", 0)]
    rows = [f"Clean & \\multicolumn{{2}}{{c}}{{{pct(clean.frr)}}} & "
            f"\\multicolumn{{2}}{{c}}{{{pct(clean.miss)}}} \\\\", "\\midrule"]
    for placement in ("input", "native"):
        for p in PERTS:
            if (placement, p, 1) not in mean.index:
                continue
            lo, hi = mean.loc[(placement, p, 1)], mean.loc[(placement, p, 4)]
            name = PNAMES[p] + (" (native)" if placement == "native" else "")
            rows.append(f"{name} & {pct(lo.frr)} & {pct(hi.frr)} & "
                        f"{pct(lo.miss)} & {pct(hi.miss)} \\\\")
    return ("\\begin{table}[t]\n"
            "\\caption{Operation at a fixed threshold: false-reject rate (FRR, "
            "normal images flagged) and miss rate (defective images passed), in "
            "percent, at the lowest and the highest severity. Seed-0 models, raw "
            "scores, Anomalib's image threshold fitted on the unperturbed test "
            "images; averaged over 15 categories and the four above-chance "
            "methods. Perturbation at the model input unless marked native.}\n"
            "\\label{tab:threshold}\n\\centering\n\\footnotesize\n"
            "\\begin{tabular}{lrrrr}\n\\toprule\n"
            " & \\multicolumn{2}{c}{FRR} & \\multicolumn{2}{c}{Miss rate} \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
            "Perturbation & Lowest & Highest & Lowest & Highest \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def raw_table(raw):
    raw = raw[(raw.status == "ok") & raw.method.isin(WORKING)]
    keys = ["category", "method", "placement", "perturbation"]
    cells = {}
    for col in ("image_AUROC_clamped", "image_AUROC_raw"):
        ident = raw[raw.severity_index == 0].set_index(keys)[col].rename("ident")
        d = raw.join(ident, on=keys)
        d["delta"] = d[col] - d["ident"]
        d = d[d.severity_index == d.groupby("perturbation")["severity_index"].transform("max")]
        for pl in ("input", "native"):
            cells[(pl, col)] = (d[d.placement == pl]
                                .groupby(["category", "perturbation"])["delta"].mean()
                                .unstack())
    ri, rn = cells[("input", "image_AUROC_raw")], cells[("native", "image_AUROC_raw")]
    p_adj = holm([wilcoxon(ri[p], rn[p]).pvalue for p in PERTS])
    rows = []
    for p, pv in zip(PERTS, p_adj):
        vals = [cells[(pl, c)][p].mean() for pl in ("input", "native")
                for c in ("image_AUROC_clamped", "image_AUROC_raw")]
        diff = ri[p].mean() - rn[p].mean()
        rows.append(f"{PNAMES[p]} & " + " & ".join(num(v) for v in vals)
                    + f" & {num(diff, pv < 0.05)} & {pfmt(pv)} \\\\")
    return ("\\begin{table}[t]\n"
            "\\caption{Image-AUROC degradation at maximum severity of the seed-0 "
            "models, evaluated on clamped scores (as in the main sweep) and on raw "
            "scores, with the perturbation at the model input and at native "
            "resolution. Same models and full test set for both placements; "
            "averaged over 15 categories and the four above-chance methods. Diff.: "
            "input minus native, raw scores. $p$: Wilcoxon signed-rank test over "
            "categories, Holm-adjusted over the seven perturbations.}\n"
            "\\label{tab:raw}\n\\centering\n\\footnotesize\n"
            "\\setlength{\\tabcolsep}{3pt}\n"
            "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
            " & \\multicolumn{2}{c}{Input} & \\multicolumn{2}{c}{Native} & & \\\\\n"
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
            "Perturbation & Clamped & Raw & Clamped & Raw & Diff. & $p$ \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


if __name__ == "__main__":
    main()
