import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from anomalib.data.utils import read_image

from common import (MAIN_SWEEP, NATIVE_SWEEP, RESULTS_DIR, ROOT, load_sweep,
                    make_resize, make_test_augmentation)
from perturbations import PERTURBATIONS, AddGaussianNoise, make_transform

OUT = os.path.join(ROOT, "paper", "figures")
DATA = os.path.join(ROOT, "dataset")

METHODS = ["PaDiM", "PatchCore", "ReverseDistillation", "EfficientAD", "GANomaly"]
LABEL = {"ReverseDistillation": "Reverse Distillation"}
COLOR = dict(zip(METHODS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
MARKER = dict(zip(METHODS, ["o", "s", "^", "D", "v"]))
PTITLE = {"brightness": "Brightness", "contrast": "Contrast",
          "saturation": "Saturation", "gaussian": "Gaussian noise",
          "salt_pepper": "Salt-and-pepper", "rotation": "Rotation",
          "translation": "Translation"}
PUNIT = {"brightness": "factor", "contrast": "factor", "saturation": "factor",
         "gaussian": r"$\sigma$", "salt_pepper": "fraction", "rotation": "degrees",
         "translation": "fraction of size"}

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6, "axes.edgecolor": "#666666",
    "xtick.color": "#444444", "ytick.color": "#444444",
    "pdf.fonttype": 42,
})


def first_image(category, defect):
    return sorted(glob.glob(os.path.join(DATA, category, "test", defect, "*.png")))[0]


def to_np(t):
    return t.clamp(0, 1).permute(1, 2, 0).numpy()


def fig_ordering(sigma=0.30):
    cases = [("wood", "good"), ("metal_nut", "good")]
    fig, axes = plt.subplots(2, 3, figsize=(3.45, 2.55))
    resize = make_resize()
    noise = AddGaussianNoise(sigma)
    c = 64

    for r, (cat, sub) in enumerate(cases):
        x = read_image(first_image(cat, sub), as_tensor=True)
        n_native = x.shape[-1]
        clean = resize(x)
        torch.manual_seed(0)
        before = resize(noise(x))
        torch.manual_seed(0)
        after = noise(resize(x))
        s_before = float((before - clean).std())
        s_after = float((after - clean).std())
        pred = (2 / 3) * sigma * 256 / n_native

        h = clean.shape[-1] // 2
        crop = lambda t: t[:, h - c // 2:h + c // 2, h - c // 2:h + c // 2]
        panels = [(clean, "clean"),
                  (before, f"before resize\n$\\hat\\sigma$={s_before:.3f} (pred. {pred:.3f})"),
                  (after, f"after resize\n$\\hat\\sigma$={s_after:.3f}")]
        for k, (img, txt) in enumerate(panels):
            ax = axes[r, k]
            ax.imshow(to_np(crop(img)), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xlabel(txt, fontsize=6.5, labelpad=2)
        axes[r, 0].set_ylabel(f"{cat.replace('_', ' ')}\n{n_native} px", fontsize=7)

    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.3)
    path = os.path.join(OUT, "fig_ordering.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print("->", os.path.relpath(path, ROOT))


def fig_perturbations(category="hazelnut", defect="crack"):
    x = read_image(first_image(category, defect), as_tensor=True)
    perts = list(PERTURBATIONS)
    fig, axes = plt.subplots(len(perts), 5, figsize=(3.45, 4.9))
    torch.manual_seed(0)
    for r, p in enumerate(perts):
        for j, sev in enumerate(PERTURBATIONS[p][1]):
            img = make_test_augmentation(make_transform(p, sev))(x)
            ax = axes[r, j]
            ax.imshow(to_np(img))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(f"{j}", fontsize=7, pad=2)
            if j == 0:
                ax.set_ylabel(PTITLE[p], fontsize=6.5, rotation=0, ha="right",
                              va="center", labelpad=2)
    fig.suptitle("severity index", fontsize=7, y=0.995)
    fig.tight_layout(pad=0.2, h_pad=0.15, w_pad=0.15, rect=(0, 0, 1, 0.985))
    path = os.path.join(OUT, "fig_perturbations.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print("->", os.path.relpath(path, ROOT))


def fig_curves():
    df = load_sweep(MAIN_SWEEP)
    curve = df.groupby(["perturbation", "method", "severity_index"])["image_AUROC"].mean()

    perts = list(PERTURBATIONS)
    fig, axes = plt.subplots(2, 4, figsize=(7.1, 3.2), sharey=True)
    axes = axes.ravel()
    for i, p in enumerate(perts):
        ax = axes[i]
        sev = PERTURBATIONS[p][1]
        xs = range(len(sev))
        ax.axhline(0.5, color="#999999", lw=0.7, ls="--", zorder=0)
        for m in METHODS:
            ys = [curve[(p, m, j)] for j in xs]
            ax.plot(xs, ys, color=COLOR[m], marker=MARKER[m], ms=3.2, lw=1.1,
                    markeredgewidth=0, label=LABEL.get(m, m))
        ax.set_title(PTITLE[p], pad=3)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"{v:g}" for v in sev])
        ax.set_xlabel(PUNIT[p], labelpad=1)
        ax.set_ylim(0.40, 1.0)
        ax.grid(axis="y", color="#e6e6e6", lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[0::4]:
        ax.set_ylabel("image AUROC")
    axes[-1].axis("off")
    h, l = axes[0].get_legend_handles_labels()
    axes[-1].legend(h, l, loc="center", frameon=False, title="Method",
                    title_fontsize=7, handlelength=1.8)
    fig.tight_layout(pad=0.3, h_pad=0.8, w_pad=0.5)
    path = os.path.join(OUT, "fig_curves.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("->", os.path.relpath(path, ROOT))


# attenuation constant per native size, measured by measure_kappa.py
KAPPA = {700: 0.668, 800: 0.668, 840: 0.669, 900: 0.668, 1000: 0.667, 1024: 0.657}
# native side of the categories that are not 1024 px
NATIVE = {"metal_nut": 700, "pill": 800, "tile": 840, "bottle": 900, "capsule": 1000}
SIGMAS = [0.0, 0.05, 0.10, 0.20, 0.30]
KEYS = ["category", "method", "seed", "perturbation"]


def _gauss_delta(frame):
    ident = frame[frame.severity_index == 0].set_index(KEYS)["image_AUROC"].rename("i")
    d = frame.join(ident, on=KEYS)
    d["d"] = d["image_AUROC"] - d["i"]
    d = d[(d.perturbation == "gaussian") & d.method.isin(METHODS[:4])]
    return d.groupby(["category", "method", "severity_value"])["d"].mean()


def mechanism_points():
    g_new = _gauss_delta(load_sweep(MAIN_SWEEP))
    g_old = _gauss_delta(load_sweep(NATIVE_SWEEP))
    rows = []
    for (c, m, s), obs in g_old.items():
        if s == 0:
            continue
        n_nat = NATIVE.get(c, 1024)
        curve = [g_new[(c, m, x)] for x in SIGMAS]
        s_eff = KAPPA[n_nat] * s * 256 / n_nat
        rows.append(dict(N=n_nat, obs=obs,
                         mech=float(np.interp(s_eff, SIGMAS, curve)),
                         null=float(np.interp(s, SIGMAS, curve))))
    return pd.DataFrame(rows)


def fig_mechanism():
    pts = mechanism_points()
    groups = [("1024 px", pts.N == 1024, "#2a78d6", "o"),
              ("700–1000 px", pts.N < 1024, "#eb6834", "^")]
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.95), sharey=True)
    lo = -0.8
    hi = max(0.05, float(pts[["obs", "mech", "null"]].max().max()) + 0.02)
    for ax, col, title, where in ((axes[0], "null", "no attenuation", (0.97, 0.04, "right", "bottom")),
                                  (axes[1], "mech", "with attenuation", (0.04, 0.96, "left", "top"))):
        ax.plot([lo, hi], [lo, hi], color="#999999", lw=0.7, ls="--", zorder=0)
        for label, mask, color, marker in groups:
            sub = pts[mask]
            ax.scatter(sub[col], sub.obs, s=8, color=color, marker=marker,
                       edgecolors="white", linewidths=0.3, label=label, zorder=2)
        mae = (pts.obs - pts[col]).abs().mean()
        r = np.corrcoef(pts.obs, pts[col])[0, 1]
        x, y, ha, va = where
        ax.text(x, y, f"MAE {mae:.3f}\n$r$ = {r:.2f}", transform=ax.transAxes,
                ha=ha, va=va, fontsize=6.5)
        ax.set_title(title, pad=3)
        ax.set_xlabel("predicted $\\Delta$", labelpad=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(color="#eeeeee", lw=0.5); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("observed $\\Delta$, native res.")
    axes[1].legend(loc="lower right", fontsize=6, frameon=False, handletextpad=0.1,
                   borderaxespad=0.1, labelspacing=0.3, markerscale=1.4)
    fig.tight_layout(pad=0.3, w_pad=0.6)
    path = os.path.join(OUT, "fig_mechanism.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("->", os.path.relpath(path, ROOT))


CATEGORIES = ["carpet", "grid", "leather", "tile", "wood",
              "bottle", "cable", "capsule", "hazelnut", "metal_nut",
              "pill", "screw", "toothbrush", "transistor", "zipper"]
TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}
WORKING = METHODS[:4]
PERTS = ["brightness", "contrast", "saturation", "gaussian", "salt_pepper",
         "rotation", "translation"]
PSHORT = {"brightness": "Bright.", "contrast": "Contr.", "saturation": "Satur.",
          "gaussian": "Gauss.", "salt_pepper": "S&P", "rotation": "Rot.",
          "translation": "Transl."}


def _max_delta(frame):
    frame = frame[(frame.status == "ok") & frame.method.isin(WORKING)]
    ident = frame[frame.severity_index == 0].set_index(KEYS)["image_AUROC"].rename("i")
    d = frame.join(ident, on=KEYS)
    d["d"] = d["image_AUROC"] - d["i"]
    top = d.groupby("perturbation")["severity_index"].transform("max")
    return d[d.severity_index == top]


def _load_new():
    return load_sweep(MAIN_SWEEP)


def fig_dataset():
    fig, axes = plt.subplots(3, 5, figsize=(7.1, 4.55))
    for ax, cat in zip(axes.ravel(), CATEGORIES):
        test = os.path.join(DATA, cat, "test")
        defect = sorted(d for d in os.listdir(test) if d != "good")[0]
        img_path = sorted(glob.glob(os.path.join(test, defect, "*.png")))[0]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(DATA, cat, "ground_truth", defect, f"{stem}_mask.png")
        img = read_image(img_path, as_tensor=True)
        n_nat = img.shape[-1]
        img = make_resize()(img)
        ax.imshow(to_np(img))
        if os.path.exists(mask_path):
            mask = make_resize()(read_image(mask_path, as_tensor=True))[0].numpy()
            ax.contour(mask > 0.5, levels=[0.5], colors="#ff2a2a", linewidths=0.7)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{cat.replace('_', ' ')}", fontsize=7.5, pad=2)
        ax.set_xlabel(f"{n_nat} px, {defect.replace('_', ' ')}", fontsize=6.3, labelpad=2)
    fig.tight_layout(pad=0.3, h_pad=1.1, w_pad=0.3)
    path = os.path.join(OUT, "fig_dataset.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("->", os.path.relpath(path, ROOT))


def fig_protocol():
    new = _max_delta(_load_new())
    old = _max_delta(load_sweep(NATIVE_SWEEP))
    lo = old.groupby("perturbation")["d"].mean()
    co = new.groupby("perturbation")["d"].mean()
    from scipy.stats import wilcoxon
    from export_latex import holm
    cat_old = old.groupby(["category", "perturbation"])["d"].mean().unstack()
    cat_new = new.groupby(["category", "perturbation"])["d"].mean().unstack()
    p_adj = dict(zip(PERTS, holm([wilcoxon(cat_new[p], cat_old[p]).pvalue
                                  for p in PERTS])))
    fig, ax = plt.subplots(figsize=(3.45, 2.1))
    ys = list(range(len(PERTS)))[::-1]
    for y, p in zip(ys, PERTS):
        ax.plot([lo[p], co[p]], [y, y], color="#b0b0b0", lw=1.2, zorder=1)
        if p_adj[p] < 0.05:
            ax.text(min(lo[p], co[p]) - 0.012, y, "*", ha="right", va="center", fontsize=9)
    ax.scatter([lo[p] for p in PERTS], ys, s=26, facecolor="white", edgecolor="#2a78d6",
               linewidth=1.3, marker="o", label="native res. (before resize)", zorder=2)
    ax.scatter([co[p] for p in PERTS], ys, s=26, color="#eb6834", marker="s",
               label="model input (after resize)", zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([PTITLE[p] for p in PERTS])
    ax.axvline(0, color="#999999", lw=0.6)
    ax.set_xlabel("image-AUROC degradation $\\Delta$")
    ax.set_xlim(-0.52, 0.02)
    ax.grid(axis="x", color="#eeeeee", lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", fontsize=6.3, frameon=False, handletextpad=0.3)
    fig.tight_layout(pad=0.3)
    path = os.path.join(OUT, "fig_protocol.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("->", os.path.relpath(path, ROOT))


def fig_heatmap():
    from matplotlib.colors import LinearSegmentedColormap
    d = _max_delta(_load_new())
    m = d.groupby(["category", "perturbation"])["d"].mean().unstack()[PERTS]
    tex = [c for c in CATEGORIES if c in TEXTURES]
    obj = [c for c in CATEGORIES if c not in TEXTURES]
    order = (m.loc[tex].mean(axis=1).sort_values(ascending=False).index.tolist()
             + m.loc[obj].mean(axis=1).sort_values(ascending=False).index.tolist())
    m = m.loc[order]
    cmap = LinearSegmentedColormap.from_list(
        "loss", ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb", "#f4f8fd"])
    fig, ax = plt.subplots(figsize=(3.45, 4.1))
    im = ax.imshow(m.values, cmap=cmap, vmin=-0.55, vmax=0, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            label = ("0" if abs(v) < 0.005 else
                     f"{v:.2f}".replace("-0.", "−.").replace("0.", "."))
            ax.text(j, i, label, ha="center", va="center", fontsize=5.6,
                    color="white" if -v > 0.3 else "#1a1a1a")
    ax.axhline(len(tex) - 0.5, color="white", lw=2.5)
    ax.set_xticks(range(len(PERTS)))
    ax.set_xticklabels([PSHORT[p] for p in PERTS], rotation=40, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([c.replace("_", " ") for c in order])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("image-AUROC degradation $\\Delta$", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    fig.tight_layout(pad=0.3)
    path = os.path.join(OUT, "fig_heatmap.pdf")
    fig.savefig(path)
    plt.close(fig)
    print("->", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_dataset()
    fig_ordering()
    fig_perturbations()
    fig_curves()
    fig_mechanism()
    fig_protocol()
    fig_heatmap()
