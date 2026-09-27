import os

import numpy as np
import pandas as pd

import make_figures as mf
from common import MAIN_SWEEP, NATIVE_SWEEP, RESULTS_DIR, load_sweep

OUT = os.path.join(RESULTS_DIR, "tables", "text_numbers.csv")
WORKING = ["PaDiM", "PatchCore", "ReverseDistillation", "EfficientAD"]
BOOT = 10_000
BOOT_SEED = 0
UNIT = ["category", "method", "seed"]
rows = []


def put(name, value, note=""):
    rows.append({"name": name, "value": value, "note": note})
    print(f"{name:<48} {value:>10.4f}  {note}" if isinstance(value, float)
          else f"{name:<48} {value!s:>10}  {note}")


def load_main():
    return load_sweep(MAIN_SWEEP)


def load_native():
    return load_sweep(NATIVE_SWEEP)


def with_delta(d):
    ident = d[d.severity_index == 0].set_index(mf.KEYS)["image_AUROC"].rename("i")
    d = d.join(ident, on=mf.KEYS)
    d["delta"] = d.image_AUROC - d.i
    return d


def bootstrap_r(main):
    d = with_delta(main[main.method.isin(WORKING)])
    d = d[d.severity_index != 0]
    r_cat = (d.groupby(UNIT)["delta"].mean()
             .groupby(["category", "method"]).mean().unstack())
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(r_cat), size=(BOOT, len(r_cat)))
    for m in WORKING:
        v = r_cat[m].to_numpy()
        boots = v[idx].mean(axis=1)
        put(f"R_{m}", float(v.mean()))
        put(f"R_{m}_ci_low", float(np.percentile(boots, 2.5)), "95% percentile bootstrap")
        put(f"R_{m}_ci_high", float(np.percentile(boots, 97.5)))


def mechanism(main, native):
    g_new = mf._gauss_delta(main)
    g_old = mf._gauss_delta(native)
    for label, kappa in (("measured", None), ("2/3", 2 / 3), ("1", 1.0)):
        obs, pred, s_eff = [], [], []
        for (c, m, s), o in g_old.items():
            if s == 0:
                continue
            n_nat = mf.NATIVE.get(c, 1024)
            k = mf.KAPPA[n_nat] if kappa is None else kappa
            se = k * s * 256 / n_nat
            curve = [g_new[(c, m, x)] for x in mf.SIGMAS]
            obs.append(o)
            pred.append(float(np.interp(se, mf.SIGMAS, curve)))
            s_eff.append(se)
        obs, pred = np.array(obs), np.array(pred)
        put(f"mechanism_MAE_kappa_{label}", float(np.abs(obs - pred).mean()))
        put(f"mechanism_r_kappa_{label}", float(np.corrcoef(obs, pred)[0, 1]))
        put(f"mechanism_n_points_kappa_{label}", len(obs))
        put(f"mechanism_n_sigma_eff_le_0.05_kappa_{label}",
            int((np.array(s_eff) <= 0.05 + 1e-12).sum()))
    null = [float(np.interp(s, mf.SIGMAS, [g_new[(c, m, x)] for x in mf.SIGMAS]))
            for (c, m, s) in g_old.index if s != 0]
    obs = np.array([o for (c, m, s), o in g_old.items() if s != 0])
    put("mechanism_MAE_no_attenuation", float(np.abs(obs - np.array(null)).mean()))
    put("mechanism_r_no_attenuation", float(np.corrcoef(obs, null)[0, 1]))
    for group, cats in (("1024px", [c for c in mf.CATEGORIES if c not in mf.NATIVE]),
                        ("metal_nut", ["metal_nut"])):
        p, o = [], []
        for c in cats:
            n_nat = mf.NATIVE.get(c, 1024)
            se = mf.KAPPA[n_nat] * 0.30 * 256 / n_nat
            for m in WORKING:
                p.append(np.interp(se, mf.SIGMAS, [g_new[(c, m, x)] for x in mf.SIGMAS]))
                o.append(g_old[(c, m, 0.30)])
        put(f"sigma0.30_{group}_predicted_native", float(np.mean(p)))
        put(f"sigma0.30_{group}_observed_native", float(np.mean(o)))


def gauss_1024(main, native):
    cats = [c for c in mf.CATEGORIES if c not in mf.NATIVE]
    for label, d in (("input", main), ("native", native)):
        g = mf._gauss_delta(d)
        for s in (0.05, 0.30):
            v = np.mean([g[(c, m, s)] for c in cats for m in WORKING])
            put(f"gauss_1024px_sigma{s:.2f}_{label}", float(v))


def agreement(main, native):
    ident = main[main.severity_index == 0]
    spread = ident.groupby(UNIT)["image_AUROC"].agg(lambda x: x.max() - x.min())
    for m, v in spread.groupby("method").max().items():
        put(f"identity_spread_max_{m}", float(v))
    base = pd.read_csv(os.path.join(RESULTS_DIR, "legacy_protocol", "baseline_all.csv"))
    oi = native[native.severity_index == 0].groupby(UNIT)["image_AUROC"].mean()
    j = pd.concat([base.set_index(UNIT)["image_AUROC"].rename("b"),
                   oi.rename("i")], axis=1).dropna()
    diff = (j.b - j.i).abs()
    for m, g in diff.groupby("method"):
        put(f"baseline_vs_identity_mean_{m}", float(g.mean()))
        put(f"baseline_vs_identity_max_{m}", float(g.max()))
    top = main[main.method.isin(WORKING) & (main.severity_index > 0)]
    put("share_perturbed_rows_exactly_0.5", float((top.image_AUROC == 0.5).mean()),
        "main sweep, four methods")


def raw_check():
    path = os.path.join(RESULTS_DIR, "raw_score_check.csv")
    if not os.path.exists(path):
        return
    r = pd.read_csv(path)
    r = r[(r.status == "ok") & r.method.isin(WORKING)]
    keys = ["category", "method", "placement", "perturbation"]
    j = r[r.placement == "input"].merge(
        load_main().query("seed == 0"),
        on=["category", "method", "perturbation", "severity_index"])
    nn = j[~j.perturbation.isin(["gaussian", "salt_pepper"])]
    put("raw_reload_max_diff_image", float((nn.image_AUROC_clamped - nn.image_AUROC).abs().max()),
        "clamped seed-0 re-evaluation vs main sweep, non-noise")
    put("raw_reload_max_diff_pixel", float((nn.pixel_AUROC_clamped - nn.pixel_AUROC).abs().max()))
    noisy = j[j.perturbation.isin(["gaussian", "salt_pepper"]) & (j.severity_index > 0)]
    put("raw_reload_noise_mean_abs_diff", float((noisy.image_AUROC_clamped - noisy.image_AUROC).abs().mean()))
    put("raw_reload_noise_max_abs_diff", float((noisy.image_AUROC_clamped - noisy.image_AUROC).abs().max()))
    inp0 = r[r.placement == "input"]
    clean = inp0[inp0.severity_index == 0].groupby("method").image_AUROC_raw.mean()
    mild = inp0[(inp0.perturbation == "gaussian") & (inp0.severity_index == 1)].groupby("method").image_AUROC_raw.mean()
    for m in WORKING:
        put(f"raw_retention_gauss0.05_{m}", float(mild[m] / clean[m]))
    for pl in ("input", "native"):
        q = r[r.placement == pl]
        deltas = {}
        for col in ("pixel_AUROC_raw", "pixel_AUROC_clamped"):
            ident = q[q.severity_index == 0].set_index(keys)[col].rename("i")
            d = q.join(ident, on=keys)
            d["d"] = d[col] - d.i
            deltas[col] = d[d.severity_index == 4].groupby(["method", "perturbation"]).d.mean()
        change = (deltas["pixel_AUROC_raw"] - deltas["pixel_AUROC_clamped"]).abs()
        put(f"raw_{pl}_pixel_change_max_per_method", float(change.max()), str(change.idxmax()))
    for pl in ("input", "native"):
        p = r[(r.placement == pl) & (r.severity_index > 0)]
        c = p[p.image_AUROC_clamped == 0.5]
        put(f"raw_{pl}_share_clamped_exactly_0.5", len(c) / len(p))
        put(f"raw_{pl}_clamped_0.5_raw_mean", float(c.image_AUROC_raw.mean()))
        put(f"raw_{pl}_clamped_0.5_share_raw_below_0.55", float((c.image_AUROC_raw < 0.55).mean()))
        put(f"raw_{pl}_clamped_0.5_share_raw_above_0.6", float((c.image_AUROC_raw > 0.6).mean()))
        put(f"raw_{pl}_max_abs_raw_minus_clamped", float((p.image_AUROC_raw - p.image_AUROC_clamped).abs().max()),
            "max over conditions")
        put(f"raw_{pl}_mean_abs_raw_minus_clamped_mean", float((p.image_AUROC_raw - p.image_AUROC_clamped).abs().mean()))
    inp = r[r.placement == "input"]
    top = inp[inp.severity_index == 4].groupby(["perturbation", "method"])[
        ["image_AUROC_raw", "image_AUROC_clamped"]].mean()
    for p in ("gaussian", "salt_pepper", "rotation"):
        put(f"raw_maxsev_{p}_raw_min", float(top.loc[p, "image_AUROC_raw"].min()))
        put(f"raw_maxsev_{p}_raw_max", float(top.loc[p, "image_AUROC_raw"].max()))
    low = inp[inp.severity_index == 1].groupby(["perturbation", "method"])[
        ["image_AUROC_raw", "image_AUROC_clamped"]].mean()
    for m in WORKING:
        put(f"raw_rotation5_{m}_raw", float(low.loc[("rotation", m), "image_AUROC_raw"]))
        put(f"raw_rotation5_{m}_clamped", float(low.loc[("rotation", m), "image_AUROC_clamped"]))
    for col in ("image_AUROC_raw", "image_AUROC_clamped", "pixel_AUROC_raw", "pixel_AUROC_clamped"):
        ident = inp[inp.severity_index == 0].set_index(keys)[col].rename("i")
        d = inp.join(ident, on=keys)
        d["delta"] = d[col] - d.i
        pert = d[d.severity_index > 0]
        for m, v in pert.groupby("method").delta.mean().items():
            put(f"raw_R_{col}_{m}", float(v), "seed 0, input")
        t = d[d.severity_index == 4].groupby(["method", "perturbation"]).delta.mean().unstack()
        for m in WORKING:
            put(f"raw_worst_{col}_{m}", t.loc[m].idxmin())
        per_cat = d[d.severity_index == 4].groupby(["category", "perturbation"]).delta.mean().unstack()
        tex = [c for c in per_cat.index if c in {"carpet", "grid", "leather", "tile", "wood"}]
        obj = [c for c in per_cat.index if c not in tex]
        put(f"raw_translation_{col}_object", float(per_cat.loc[obj, "translation"].mean()))
        put(f"raw_translation_{col}_texture", float(per_cat.loc[tex, "translation"].mean()))
        eff = t.loc["EfficientAD"]
        for p in ("brightness", "contrast", "saturation"):
            put(f"raw_EfficientAD_{p}_{col}", float(eff[p]))
        mean_cat = per_cat.mean()
        for p in mean_cat.index:
            put(f"raw_maxsev_mean_{col}_{p}", float(mean_cat[p]))


def retention_ci(main):
    d = main[main.method.isin(WORKING)]
    cat = d.groupby(["method", "category", "perturbation", "severity_index"]).image_AUROC.mean()
    rng = np.random.default_rng(BOOT_SEED)
    for m in WORKING:
        clean = cat.loc[m].xs(0, level="severity_index").groupby("category").mean()
        for p in ("gaussian", "salt_pepper", "rotation", "translation"):
            low = cat.xs((m, p, 1), level=["method", "perturbation", "severity_index"])
            c, l = clean.to_numpy(), low.reindex(clean.index).to_numpy()
            idx = rng.integers(0, len(c), size=(BOOT, len(c)))
            boots = l[idx].mean(1) / c[idx].mean(1)
            put(f"retention_sev1_{p}_{m}", float(l.mean() / c.mean()))
            put(f"retention_sev1_{p}_{m}_ci_high", float(np.percentile(boots, 97.5)),
                "95% percentile bootstrap over categories")


def threshold_numbers():
    path = os.path.join(RESULTS_DIR, "image_scores.csv")
    if not os.path.exists(path):
        return
    from export_latex import threshold_rates
    scores = pd.read_csv(path)
    rates = threshold_rates(scores)
    clean = rates[(rates.placement == "input") & (rates.severity_index == 0)]
    by_method = clean.groupby("method")[["frr", "miss"]].mean()
    for m, row in by_method.iterrows():
        put(f"thr_clean_frr_{m}", float(row.frr))
        put(f"thr_clean_miss_{m}", float(row.miss))
    put("thr_clean_frr_mean", float(by_method.frr.mean()))
    put("thr_clean_miss_mean", float(by_method.miss.mean()))
    for placement in ("input", "native"):
        sub = rates[(rates.placement == placement) & (rates.severity_index.isin([1, 4]))]
        mean = sub.groupby(["perturbation", "severity_index"])[["frr", "miss"]].mean()
        for (p, idx), row in mean.iterrows():
            put(f"thr_{placement}_{p}_sev{idx}_frr", float(row.frr))
            put(f"thr_{placement}_{p}_sev{idx}_miss", float(row.miss))
    low = rates[(rates.placement == "input") & (rates.severity_index == 1)]
    for (p, m), v in low.set_index(["perturbation", "method"]).frr.items():
        put(f"thr_input_{p}_sev1_frr_{m}", float(v))
    scores["reject"] = scores.raw_score > scores.threshold
    keys = ["category", "method", "placement", "perturbation", "severity_index"]
    frr = scores[(scores.label == 0) & (scores.severity_index > 0)].groupby(keys).reject.mean()
    full = (frr >= 0.999).groupby(level=["placement", "perturbation"]).mean()
    for (pl, p), v in full.items():
        put(f"thr_share_all_normals_rejected_{pl}_{p}", float(v))


def main():
    main_d, native_d = load_main(), load_native()
    bootstrap_r(main_d)
    mechanism(main_d, native_d)
    gauss_1024(main_d, native_d)
    agreement(main_d, native_d)
    raw_check()
    retention_ci(main_d)
    threshold_numbers()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("->", OUT)


if __name__ == "__main__":
    main()
