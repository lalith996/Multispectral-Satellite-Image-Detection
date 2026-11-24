"""
run_experiments.py
Top-Tier Q1 Journal Experimental Suite (IEEE TGRS / RSE Caliber) on 1,000 Real EuroSAT-MS GeoTIFFs.

Covers all 6 top-tier Q1 reviewer criteria:
  1. 4 Independent PyTorch Neural Multispectral Generators (ConvAE, RGB->13B, DDPM-UNet, ResNet-GAN)
  2. Cross-Atmospheric (L1C TOA -> L2A BOA) & Cross-Sensor (Sentinel-2 -> Landsat-8 OLI 7-Band) transfer
  3. Adaptive White-Box Physics-Aware Attackers (WCT Covariance Alignment, NDVI Matching, Combined)
  4. Head-to-Head Published Baselines (Durall-DFT [9], Horvath ViT [3], Qi SFNet [2]) + TPR@1%/5% FPR & ECE
  5. Explicit Failure-Mode & Error Analysis (actual FP/FN inspection & physical root-cause metrics)
  6. 5-Fold Stratified CV (Mean +/- Std) + DeLong Paired ROC-AUC Tests + Sensitivity + Multi-Scale Localization
"""

from __future__ import annotations
import glob
import json
import os
import sys
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import (
    EUROSAT_CLASSES,
    SENTINEL2_BANDS,
    apply_perturbation,
    read_geotiff_13band,
    synthesize_from_real_cube,
)
from cg_unmixing import compute_unmixing_and_sam, extract_endmembers_vca
from spectral_consistency import (
    compute_sample_correlation,
    extract_dataset_features,
    extract_features_single,
    fit_authentic_priors,
    get_feature_names,
)
from cg_optimizer import (
    RegularizedHuberHingeObjective,
    RegularizedLogisticObjective,
    delong_roc_test,
    solve_lbfgs_benchmark,
    solve_nonlinear_cg,
)
from neural_generators import (
    train_and_generate_all_neural_fakes,
    train_eval_pytorch_detector,
)


def compute_ece(y_bin: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    """Compute 10-bin Expected Calibration Error (ECE) and reliability curve points."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_out = []
    N = len(y_bin)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs <= hi if i == n_bins - 1 else probs < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            acc = float(np.mean(y_bin[mask]))
            conf = float(np.mean(probs[mask]))
            ece += (cnt / N) * abs(acc - conf)
            bins_out.append({"conf": round(conf, 4), "acc": round(acc, 4), "count": cnt})
    return float(ece), bins_out


def tpr_at_fpr(y_bin: np.ndarray, probs: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_bin, probs)
    idx = np.where(fpr <= target_fpr)[0]
    return float(tpr[idx[-1]]) if len(idx) > 0 else 0.0


def compute_metrics(y_true_pm1: np.ndarray, y_prob: np.ndarray, thresh: float = 0.5) -> dict:
    y_bin = (y_true_pm1 == 1).astype(int)
    y_pred = (y_prob >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_bin, y_pred, labels=[0, 1]).ravel()
    ece, rel_bins = compute_ece(y_bin, y_prob, n_bins=10)
    return {
        "accuracy": float(accuracy_score(y_bin, y_pred)),
        "precision": float(precision_score(y_bin, y_pred, zero_division=0)),
        "recall": float(recall_score(y_bin, y_pred, zero_division=0)),
        "f1": float(f1_score(y_bin, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_bin, y_prob)),
        "ap": float(average_precision_score(y_bin, y_prob)),
        "tpr_at_5fpr": tpr_at_fpr(y_bin, y_prob, 0.05),
        "tpr_at_1fpr": tpr_at_fpr(y_bin, y_prob, 0.01),
        "ece": ece,
        "brier": float(brier_score_loss(y_bin, np.clip(y_prob, 0.0, 1.0))),
        "reliability_bins": rel_bins,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def standardize(X_tr: np.ndarray, *others: np.ndarray):
    mu = np.mean(X_tr, axis=0)
    sd = np.std(X_tr, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    res = [(X_tr - mu) / sd]
    for arr in others:
        res.append((arr - mu) / sd)
    return tuple(res), mu, sd


def apply_l2a_boa_correction(cube: np.ndarray) -> np.ndarray:
    """
    Simulate Sentinel-2 Level-1C (TOA) -> Level-2A (BOA) atmospheric correction via
    physical Dark Object Subtraction (DOS1) + wavelength-dependent Rayleigh path-radiance removal.
    """
    x = cube.astype(np.float64).copy()
    # Rayleigh scattering ~ lambda^-4 strongest in Coastal/Blue/Green, negligible in SWIR
    rayleigh_path = np.array([0.042, 0.032, 0.020, 0.012, 0.009, 0.007, 0.005, 0.004, 0.003, 0.008, 0.003, 0.002, 0.001])
    transmittance = np.array([0.82, 0.86, 0.90, 0.93, 0.94, 0.95, 0.95, 0.96, 0.96, 0.85, 0.80, 0.95, 0.94])
    x = (x - rayleigh_path[None, None, :]) / transmittance[None, None, :]
    return np.clip(x, 0.0005, 0.995).astype(np.float32)


def apply_landsat8_sensor_shift(cube: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate cross-sensor projection to Landsat-8/9 OLI 7 overlapping bands
    ([B1, B2, B3, B4, B8A, B11, B12]) with 30 m spatial PSF smoothing + OLI relative spectral response gain.
    """
    x = cube.astype(np.float64).copy()
    # 30 m GSD spatial low-pass PSF relative to Sentinel-2 10-20 m
    for b in range(x.shape[2]):
        x[:, :, b] = gaussian_filter(x[:, :, b], sigma=0.85)
    # Small OLI-vs-MSI spectral bandpass gain/offset shift
    gains = rng.uniform(0.96, 1.04, size=(1, 1, x.shape[2]))
    x = x * gains + rng.normal(0.0, 0.002, size=x.shape)
    return np.clip(x, 0.0005, 0.995).astype(np.float32)


def apply_wct_covariance_attack(fake_cube: np.ndarray, target_Sigma: np.ndarray) -> np.ndarray:
    """
    White-Box Adaptive Attack A (Whitening-and-Coloring Transform):
    Explicitly aligns the fake cube's inter-band covariance Sigma_fake to the authentic
    target covariance Sigma_real to neutralize Module A!
    """
    H, W, B = fake_cube.shape
    Y = fake_cube.reshape(-1, B).T.astype(np.float64)  # (B, P)
    mu = np.mean(Y, axis=1, keepdims=True)
    Yc = Y - mu
    S_f = (Yc @ Yc.T) / max(Y.shape[1] - 1, 1) + 1e-6 * np.eye(B)
    S_t = 0.5 * (target_Sigma + target_Sigma.T) + 1e-6 * np.eye(B)

    ef, Qf = np.linalg.eigh(S_f)
    et, Qt = np.linalg.eigh(S_t)
    inv_sqrt_f = (Qf * (1.0 / np.sqrt(np.maximum(ef, 1e-6)))[None, :]) @ Qf.T
    sqrt_t = (Qt * np.sqrt(np.maximum(et, 1e-6))[None, :]) @ Qt.T

    Y_aligned = sqrt_t @ (inv_sqrt_f @ Yc) + mu
    return np.clip(Y_aligned.T.reshape(H, W, B), 0.0005, 0.995).astype(np.float32)


def apply_ndvi_matching_attack(fake_cube: np.ndarray, target_ndvi_mean: float) -> np.ndarray:
    """
    White-Box Adaptive Attack B: Adjusts Band 8 (NIR) gain so patch mean NDVI matches
    the authentic class mean NDVI, targeting Module B!
    """
    x = fake_cube.astype(np.float64).copy()
    B4, B8 = x[:, :, 3], x[:, :, 7]
    # Solve (g*B8 - B4)/(g*B8 + B4) ~ target_ndvi -> g ~ (1+t)/(1-t) * mean(B4)/mean(B8)
    t = np.clip(target_ndvi_mean, -0.8, 0.85)
    g = np.clip(((1.0 + t) / (1.0 - t)) * (np.mean(B4) / (np.mean(B8) + 1e-6)), 0.65, 1.45)
    x[:, :, 7] = x[:, :, 7] * (0.5 + 0.5 * g)
    return np.clip(x, 0.0005, 0.995).astype(np.float32)


def apply_cirrus_cloud_haze(cube: np.ndarray, optical_depth: float, rng: np.random.Generator) -> np.ndarray:
    H, W, _ = cube.shape
    cloud = gaussian_filter(rng.standard_normal((H, W)), sigma=6.0)
    cloud = (cloud - cloud.min()) / (cloud.max() - cloud.min() + 1e-8)
    tau = optical_depth * (0.4 + 0.6 * cloud)[:, :, None]
    out = cube.astype(np.float64) * (1.0 - 0.5 * tau) + 0.35 * tau
    out[:, :, 10] += 0.25 * optical_depth * (0.5 + 0.5 * cloud)
    return np.clip(out, 0.0005, 0.995).astype(np.float32)


def multiscale_anomaly_map(cube, priors, M_endmembers, mu, sd, w):
    H, W, _ = cube.shape
    hmap_total = np.zeros((H, W), dtype=np.float64)
    for win, stride, weight in [(16, 4, 0.50), (24, 6, 0.50)]:
        prob_acc = np.zeros((H, W), dtype=np.float64)
        cnt_acc = np.zeros((H, W), dtype=np.float64)
        for r in range(0, H - win + 1, stride):
            for c in range(0, W - win + 1, stride):
                sub = cube[r:r + win, c:c + win, :]
                phi, _ = extract_features_single(sub, priors, M_endmembers)
                z = float(np.dot(w[:-1], (phi - mu) / sd) + w[-1])
                p = 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))
                prob_acc[r:r + win, c:c + win] += p
                cnt_acc[r:r + win, c:c + win] += 1.0
        hmap_total += weight * (prob_acc / np.maximum(cnt_acc, 1.0))

    lmm_res, sam_map, _ = compute_unmixing_and_sam(cube, M_endmembers)
    res_norm = (lmm_res - lmm_res.min()) / (lmm_res.max() - lmm_res.min() + 1e-8)
    sam_norm = (sam_map - sam_map.min()) / (sam_map.max() - sam_map.min() + 1e-8)
    return 0.55 * hmap_total + 0.25 * res_norm + 0.20 * sam_norm, sam_map


def one_class_mahalanobis_scores(Phi_tr_auth: np.ndarray, Phi_te: np.ndarray) -> np.ndarray:
    mu = np.mean(Phi_tr_auth, axis=0)
    cov = np.cov(Phi_tr_auth, rowvar=False) + 0.15 * np.eye(Phi_tr_auth.shape[1])
    inv_cov = np.linalg.inv(cov)
    diff = Phi_te - mu[None, :]
    d2 = np.sum((diff @ inv_cov) * diff, axis=1)
    med = np.median(d2); mad = np.median(np.abs(d2 - med)) + 1e-6
    return 1.0 / (1.0 + np.exp(-np.clip((d2 - med) / (2.0 * mad), -20.0, 20.0)))


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eurosat_dir = os.path.join(base_dir, "data", "EuroSAT_MS")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    rng = np.random.default_rng(42)
    print(f"[1/11] Loading 1,000 REAL EuroSAT-MS 64x64x13 GeoTIFFs from {eurosat_dir}...")

    pools = {k: [] for k in ["auth", "ae", "rgb2ms", "ddpm", "resgan"]}
    cls_pools = {k: [] for k in pools}
    for cls in EUROSAT_CLASSES:
        files = sorted(glob.glob(os.path.join(eurosat_dir, cls, "*.tif")))[:100]
        rng.shuffle(files)
        # Disjoint split per class: 50 authentic, 13 AE, 13 RGB2MS, 12 DDPM-UNet, 12 ResNet-GAN
        splits = [
            ("auth", files[:50]),
            ("ae", files[50:63]),
            ("rgb2ms", files[63:76]),
            ("ddpm", files[76:88]),
            ("resgan", files[88:100]),
        ]
        for key, flist in splits:
            for fp in flist:
                pools[key].append(read_geotiff_13band(fp))
                cls_pools[key].append(cls)

    auth_cubes = np.stack(pools["auth"], axis=0)
    print("[2/11] Training 4 Independent PyTorch Neural Multispectral Generators (ConvAE, RGB->13B, DDPM-UNet, ResNet-GAN)...")
    f_ae, f_rgb2ms, f_ddpm, f_rgan = train_and_generate_all_neural_fakes(
        auth_cubes[:300],
        np.stack(pools["ae"], axis=0),
        np.stack(pools["rgb2ms"], axis=0),
        np.stack(pools["ddpm"], axis=0),
        np.stack(pools["resgan"], axis=0),
        epochs=8,
        seed=42,
    )

    X_all = np.concatenate([auth_cubes, f_ae, f_rgb2ms, f_ddpm, f_rgan], axis=0)
    y_all = np.concatenate([
        np.full(len(auth_cubes), -1, dtype=np.int32),
        np.full(len(f_ae) + len(f_rgb2ms) + len(f_ddpm) + len(f_rgan), +1, dtype=np.int32),
    ])
    tiers_all = np.array(
        ["authentic"] * len(auth_cubes)
        + ["neural_ae"] * len(f_ae)
        + ["neural_rgb2ms"] * len(f_rgb2ms)
        + ["ddpm_unet"] * len(f_ddpm)
        + ["resnet_gan"] * len(f_rgan)
    )
    classes_all = np.array(
        cls_pools["auth"] + cls_pools["ae"] + cls_pools["rgb2ms"] + cls_pools["ddpm"] + cls_pools["resgan"]
    )
    N = len(y_all)
    print(f"    Built 1,000-tile REAL EuroSAT-MS + 4-Neural-Generator corpus: Auth={len(auth_cubes)}, AE={len(f_ae)}, RGB2MS={len(f_rgb2ms)}, DDPM={len(f_ddpm)}, ResGAN={len(f_rgan)}")

    # =========================================================================
    # [3/11] 5-Fold Stratified CV across 10 Detectors (Published + Proposed)
    # =========================================================================
    print("[3/11] Running 5-Fold Stratified CV across 10 Detectors (incl. Durall-DFT, Horvath-ViT, Qi-SFNet)...")
    strat_key = np.array([f"{c}_{t}" for c, t in zip(classes_all, tiers_all)])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    detector_names = [
        "Spectral-CG (HZ, 13-Band)",
        "Spectral-Huber-CG (13-Band)",
        "Qi et al. SFNet (13-Band) [2]",
        "Horvath et al. ViT (13-Band) [3]",
        "Durall et al. DFT (13-Band) [9]",
        "PyTorch 13-Band 2D-CNN",
        "SVM-RBF (13-Band)",
        "Random Forest (13-Band)",
        "One-Class Mahalanobis (Zero-Shot)",
        "PyTorch RGB 2D-CNN (3-Band)",
        "RGB-Only Spectral-CG (3-Band)",
    ]
    cv_metrics = {n: {"accuracy": [], "f1": [], "roc_auc": [], "ap": [], "tpr_at_1fpr": [], "tpr_at_5fpr": [], "ece": []} for n in detector_names}
    oof_probs = {n: np.zeros(N, dtype=np.float64) for n in detector_names}
    fold0 = {}

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_all, strat_key)):
        auth_tr = (y_all[tr_idx] == -1)
        priors_f = fit_authentic_priors(X_all[tr_idx][auth_tr], classes_all[tr_idx][auth_tr])
        M_f = extract_endmembers_vca(X_all[tr_idx][auth_tr], n_endmembers=6, seed=42 + fold)

        Phi_tr_raw = extract_dataset_features(X_all[tr_idx], priors_f, M_f)
        Phi_te_raw = extract_dataset_features(X_all[te_idx], priors_f, M_f)
        (Phi_tr, Phi_te), mu_f, sd_f = standardize(Phi_tr_raw, Phi_te_raw)
        y_tr, y_te = y_all[tr_idx], y_all[te_idx]

        # Proposed Spectral-CG (HZ)
        obj_hz = RegularizedLogisticObjective(Phi_tr, y_tr, lam=1e-3)
        res_hz = solve_nonlinear_cg(obj_hz, method="HZ", tol=1e-6, max_iter=400)
        p_hz = obj_hz.predict_proba(res_hz["w"], Phi_te)

        # Proposed Spectral-Huber-CG
        obj_hub = RegularizedHuberHingeObjective(Phi_tr, y_tr, lam=1e-3, delta=0.5)
        res_hub = solve_nonlinear_cg(obj_hub, method="HZ", tol=1e-6, max_iter=400)
        p_hub = obj_hub.predict_proba(res_hub["w"], Phi_te)

        # Published Baseline: Durall et al. [9] (Module D 2D Azimuthal DFT only)
        ob_dur = RegularizedLogisticObjective(Phi_tr[:, 51:72], y_tr, lam=1e-3)
        rs_dur = solve_nonlinear_cg(ob_dur, method="HZ", tol=1e-6, max_iter=400)
        p_dur = ob_dur.predict_proba(rs_dur["w"], Phi_te[:, 51:72])

        # Published Baseline: Qi et al. SFNet [2] (13-Band Spatial-Frequency Fusion CNN)
        p_sfnet = train_eval_pytorch_detector("sfnet", X_all[tr_idx], y_tr, X_all[te_idx], list(range(13)), epochs=6, seed=42 + fold)

        # Published Baseline: Horvath et al. ViT-Patch [3] (13-Band Vision Transformer)
        p_vit = train_eval_pytorch_detector("vit", X_all[tr_idx], y_tr, X_all[te_idx], list(range(13)), epochs=6, seed=42 + fold)

        # PyTorch 13-Band 2D-CNN
        p_cnn13 = train_eval_pytorch_detector("cnn", X_all[tr_idx], y_tr, X_all[te_idx], list(range(13)), epochs=6, seed=42 + fold)

        # SVM-RBF & Random Forest
        svm = SVC(kernel="rbf", C=1.5, probability=True, random_state=42 + fold)
        svm.fit(Phi_tr, (y_tr == 1).astype(int))
        p_svm = svm.predict_proba(Phi_te)[:, 1]

        rf = RandomForestClassifier(n_estimators=200, max_depth=14, random_state=42 + fold)
        rf.fit(Phi_tr, (y_tr == 1).astype(int))
        p_rf = rf.predict_proba(Phi_te)[:, 1]

        # One-Class Mahalanobis (Zero-Shot)
        p_oc = one_class_mahalanobis_scores(Phi_tr[y_tr == -1], Phi_te)

        # RGB-Only Baselines
        rgb_idx = [1, 2, 3]
        p_cnn3 = train_eval_pytorch_detector("cnn", X_all[tr_idx], y_tr, X_all[te_idx], rgb_idx, epochs=6, seed=42 + fold)
        Phi_tr_rgb_raw = extract_dataset_features(X_all[tr_idx], priors_f, M_f, band_indices=rgb_idx)
        Phi_te_rgb_raw = extract_dataset_features(X_all[te_idx], priors_f, M_f, band_indices=rgb_idx)
        (Phi_tr_rgb, Phi_te_rgb), _, _ = standardize(Phi_tr_rgb_raw, Phi_te_rgb_raw)
        ob_rgb = RegularizedLogisticObjective(Phi_tr_rgb, y_tr, lam=1e-3)
        rs_rgb = solve_nonlinear_cg(ob_rgb, method="HZ", tol=1e-6, max_iter=400)
        p_rgb = ob_rgb.predict_proba(rs_rgb["w"], Phi_te_rgb)

        fold_preds = {
            "Spectral-CG (HZ, 13-Band)": p_hz,
            "Spectral-Huber-CG (13-Band)": p_hub,
            "Qi et al. SFNet (13-Band) [2]": p_sfnet,
            "Horvath et al. ViT (13-Band) [3]": p_vit,
            "Durall et al. DFT (13-Band) [9]": p_dur,
            "PyTorch 13-Band 2D-CNN": p_cnn13,
            "SVM-RBF (13-Band)": p_svm,
            "Random Forest (13-Band)": p_rf,
            "One-Class Mahalanobis (Zero-Shot)": p_oc,
            "PyTorch RGB 2D-CNN (3-Band)": p_cnn3,
            "RGB-Only Spectral-CG (3-Band)": p_rgb,
        }
        for name, pr in fold_preds.items():
            m = compute_metrics(y_te, pr)
            oof_probs[name][te_idx] = pr
            for k in ("accuracy", "f1", "roc_auc", "ap", "tpr_at_1fpr", "tpr_at_5fpr", "ece"):
                cv_metrics[name][k].append(m[k])

        print(f"    Fold {fold+1}/5 complete (Spectral-CG AUC={cv_metrics['Spectral-CG (HZ, 13-Band)']['roc_auc'][-1]:.4f}, SFNet={cv_metrics['Qi et al. SFNet (13-Band) [2]']['roc_auc'][-1]:.4f}, ViT={cv_metrics['Horvath et al. ViT (13-Band) [3]']['roc_auc'][-1]:.4f})")
        if fold == 0:
            fold0 = {
                "tr_idx": tr_idx, "te_idx": te_idx, "priors": priors_f, "M": M_f,
                "Phi_tr_raw": Phi_tr_raw, "Phi_te_raw": Phi_te_raw,
                "Phi_tr": Phi_tr, "Phi_te": Phi_te, "mu": mu_f, "sd": sd_f,
                "Phi_tr_rgb_raw": Phi_tr_rgb_raw, "Phi_te_rgb_raw": Phi_te_rgb_raw,
                "w_star": res_hz["w"], "obj_main": obj_hz,
            }

    y_bin_all = (y_all == 1).astype(int)
    cv_summary, delong_tests, roc_curves = {}, {}, {}
    for name in detector_names:
        oof_m = compute_metrics(y_all, oof_probs[name])
        cv_summary[name] = {
            "accuracy_mean": float(np.mean(cv_metrics[name]["accuracy"])),
            "accuracy_std": float(np.std(cv_metrics[name]["accuracy"])),
            "f1_mean": float(np.mean(cv_metrics[name]["f1"])),
            "f1_std": float(np.std(cv_metrics[name]["f1"])),
            "roc_auc_mean": float(np.mean(cv_metrics[name]["roc_auc"])),
            "roc_auc_std": float(np.std(cv_metrics[name]["roc_auc"])),
            "ap_mean": float(np.mean(cv_metrics[name]["ap"])),
            "tpr_at_5fpr_mean": float(np.mean(cv_metrics[name]["tpr_at_5fpr"])),
            "tpr_at_1fpr_mean": float(np.mean(cv_metrics[name]["tpr_at_1fpr"])),
            "ece_mean": float(np.mean(cv_metrics[name]["ece"])),
            "brier": oof_m["brier"],
            "reliability_bins": oof_m["reliability_bins"],
            "accuracy": oof_m["accuracy"],
            "f1": oof_m["f1"],
            "roc_auc": oof_m["roc_auc"],
            "ap": oof_m["ap"],
            "confusion_matrix": oof_m["confusion_matrix"],
        }
        fpr, tpr, _ = roc_curve(y_bin_all, oof_probs[name])
        step = max(1, len(fpr) // 120)
        roc_curves[name] = {"fpr": np.round(fpr[::step], 4).tolist(), "tpr": np.round(tpr[::step], 4).tolist()}
        if name != "Spectral-CG (HZ, 13-Band)":
            delong_tests[name] = delong_roc_test(y_bin_all, oof_probs["Spectral-CG (HZ, 13-Band)"], oof_probs[name])

    per_class_perf = {cls: compute_metrics(y_all[classes_all == cls], oof_probs["Spectral-CG (HZ, 13-Band)"][classes_all == cls]) for cls in EUROSAT_CLASSES}

    # Unpack Fold 0
    tr_idx, te_idx = fold0["tr_idx"], fold0["te_idx"]
    priors, M_endmembers = fold0["priors"], fold0["M"]
    Phi_tr_raw, Phi_te_raw = fold0["Phi_tr_raw"], fold0["Phi_te_raw"]
    Phi_tr, Phi_te = fold0["Phi_tr"], fold0["Phi_te"]
    mu_full, sd_full = fold0["mu"], fold0["sd"]
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]
    w_star, obj_main = fold0["w_star"], fold0["obj_main"]

    grad_rel_err = obj_main.verify_gradient(np.random.default_rng(7).normal(0.0, 0.1, size=Phi_tr.shape[1] + 1))
    cond_init = obj_main.conditioning_stats(np.zeros(Phi_tr.shape[1] + 1))
    cond_opt = obj_main.conditioning_stats(w_star)

    t_inf0 = time.perf_counter()
    for i in range(25):
        p1, _ = extract_features_single(X_all[te_idx[i]], priors, M_endmembers)
        _ = 1.0 / (1.0 + np.exp(-float(np.dot(w_star[:-1], (p1 - mu_full) / sd_full) + w_star[-1])))
    cpu_latency_ms = (time.perf_counter() - t_inf0) * 1000.0 / 25.0

    # =========================================================================
    # [4/11] Adaptive White-Box Physics-Aware Attackers (Gap 3)
    # =========================================================================
    print("[4/11] Running Adaptive White-Box Physics-Aware Adversarial Attacks (WCT Covariance, NDVI Match, Combined)...")
    # Compute class authentic target covariance & mean NDVI from training fold
    auth_cls_Sigma = {}
    auth_cls_ndvi = {}
    for cls in EUROSAT_CLASSES:
        c_tiles = X_all[tr_idx][(y_tr == -1) & (classes_all[tr_idx] == cls)]
        auth_cls_Sigma[cls] = np.mean([compute_sample_correlation(c)[0] for c in c_tiles], axis=0)
        auth_cls_ndvi[cls] = float(np.mean([(c[:, :, 7] - c[:, :, 3]) / (c[:, :, 7] + c[:, :, 3] + 1e-6) for c in c_tiles]))

    # Train Module-A-only and Module-(C+D)-only reference models for adaptive comparison
    ob_modA = RegularizedLogisticObjective(Phi_tr[:, 0:25], y_tr, lam=1e-3)
    w_modA = solve_nonlinear_cg(ob_modA, method="HZ", tol=1e-6, max_iter=400)["w"]
    ob_modCD = RegularizedLogisticObjective(Phi_tr[:, 41:72], y_tr, lam=1e-3)
    w_modCD = solve_nonlinear_cg(ob_modCD, method="HZ", tol=1e-6, max_iter=400)["w"]

    adaptive_attacks = {}
    for atk_key, atk_label in [
        ("clean", "No Attack (Clean Test Set)"),
        ("wct", "Adaptive WCT Covariance Alignment (Targets Mod A)"),
        ("ndvi", "Adaptive NDVI Histogram Matching (Targets Mod B)"),
        ("combined", "Adaptive Combined WCT + NDVI + Smoothing"),
    ]:
        X_atk = X_all[te_idx].copy()
        for ii, idx_g in enumerate(te_idx):
            if y_all[idx_g] == 1:  # Attacker only modifies fake tiles!
                cls = classes_all[idx_g]
                if atk_key == "wct":
                    X_atk[ii] = apply_wct_covariance_attack(X_atk[ii], auth_cls_Sigma[cls])
                elif atk_key == "ndvi":
                    X_atk[ii] = apply_ndvi_matching_attack(X_atk[ii], auth_cls_ndvi[cls])
                elif atk_key == "combined":
                    tmp = apply_wct_covariance_attack(X_atk[ii], auth_cls_Sigma[cls])
                    tmp = apply_ndvi_matching_attack(tmp, auth_cls_ndvi[cls])
                    for b in range(13):
                        tmp[:, :, b] = 0.85 * tmp[:, :, b] + 0.15 * gaussian_filter(tmp[:, :, b], sigma=0.7)
                    X_atk[ii] = np.clip(tmp, 0.0005, 0.995).astype(np.float32)

        Phi_atk = (extract_dataset_features(X_atk, priors, M_endmembers) - mu_full) / sd_full
        adaptive_attacks[atk_label] = {
            "Full Spectral-CG (A+B+C+D)": compute_metrics(y_te, obj_main.predict_proba(w_star, Phi_atk))["roc_auc"],
            "Module A Only": compute_metrics(y_te, ob_modA.predict_proba(w_modA, Phi_atk[:, 0:25]))["roc_auc"],
            "Modules C+D Defense": compute_metrics(y_te, ob_modCD.predict_proba(w_modCD, Phi_atk[:, 41:72]))["roc_auc"],
        }

    # =========================================================================
    # [5/11] Cross-Atmospheric (L1C->L2A) & Cross-Sensor (S2->Landsat-8 OLI) (Gap 2)
    # =========================================================================
    print("[5/11] Running Cross-Atmospheric (L1C TOA -> L2A BOA) & Cross-Sensor (Sentinel-2 -> Landsat-8 OLI 7-Band)...")
    X_te_l2a = np.stack([apply_l2a_boa_correction(c) for c in X_all[te_idx]], axis=0)
    Phi_te_l2a = (extract_dataset_features(X_te_l2a, priors, M_endmembers) - mu_full) / sd_full
    m_l2a = compute_metrics(y_te, obj_main.predict_proba(w_star, Phi_te_l2a))

    # Cross-sensor Landsat-8 OLI 7 overlapping bands: B1(0), B2(1), B3(2), B4(3), B8A(8), B11(11), B12(12)
    l8_idx = [0, 1, 2, 3, 8, 11, 12]
    tr_l8_raw = extract_dataset_features(X_all[tr_idx], priors, M_endmembers, band_indices=l8_idx)
    X_te_l8_shifted = np.stack([apply_landsat8_sensor_shift(c, np.random.default_rng(77 + i)) for i, c in enumerate(X_all[te_idx])], axis=0)
    te_l8_raw = extract_dataset_features(X_te_l8_shifted, priors, M_endmembers, band_indices=l8_idx)
    (tr_l8s, te_l8s), _, _ = standardize(tr_l8_raw, te_l8_raw)
    ob_l8 = RegularizedLogisticObjective(tr_l8s, y_tr, lam=1e-3)
    w_l8 = solve_nonlinear_cg(ob_l8, method="HZ", tol=1e-6, max_iter=400)["w"]
    m_l8 = compute_metrics(y_te, ob_l8.predict_proba(w_l8, te_l8s))

    cross_domain = {
        "In-Domain Sentinel-2 L1C (13-Band)": compute_metrics(y_te, obj_main.predict_proba(w_star, Phi_te)),
        "Cross-Atmospheric L1C -> L2A BOA (Zero-Shot)": m_l2a,
        "Cross-Sensor Sentinel-2 -> Landsat-8 OLI (7-Band, 30m)": m_l8,
    }

    # =========================================================================
    # [6/11] Zero-Shot Cross-Generator Generalization Across 4 Independent Neural Models
    # =========================================================================
    print("[6/11] Running Zero-Shot Cross-Generator Transfer (Unseen DDPM-UNet & ResNet-GAN)...")
    cross_gen = {}
    cg_pairs = [
        (["neural_ae", "neural_rgb2ms"], ["ddpm_unet"], "Train (AE+RGB2MS) -> Unseen DDPM-UNet"),
        (["neural_ae", "neural_rgb2ms"], ["resnet_gan"], "Train (AE+RGB2MS) -> Unseen ResNet-GAN"),
        (["ddpm_unet", "resnet_gan"], ["neural_ae"], "Train (DDPM+ResGAN) -> Unseen ConvAE"),
        (["ddpm_unet", "resnet_gan"], ["neural_rgb2ms"], "Train (DDPM+ResGAN) -> Unseen RGB->13B"),
    ]
    for tr_f, te_f, lbl in cg_pairs:
        tr_m = (tiers_all[tr_idx] == "authentic") | np.isin(tiers_all[tr_idx], tr_f)
        te_m = (tiers_all[te_idx] == "authentic") | np.isin(tiers_all[te_idx], te_f)
        (Xc_tr, Xc_te), _, _ = standardize(Phi_tr_raw[tr_m], Phi_te_raw[te_m])
        ob_c = RegularizedLogisticObjective(Xc_tr, y_tr[tr_m], lam=1e-3)
        w_c = solve_nonlinear_cg(ob_c, method="HZ", tol=1e-6, max_iter=400)["w"]
        p_c = ob_c.predict_proba(w_c, Xc_te)

        # Also compare against SFNet and RGB-Only on the exact same zero-shot transfer!
        p_sf_cg = train_eval_pytorch_detector("sfnet", X_all[tr_idx][tr_m], y_tr[tr_m], X_all[te_idx][te_m], list(range(13)), epochs=6, seed=42)
        (Xr_tr, Xr_te), _, _ = standardize(fold0["Phi_tr_rgb_raw"][tr_m], fold0["Phi_te_rgb_raw"][te_m])
        ob_r = RegularizedLogisticObjective(Xr_tr, y_tr[tr_m], lam=1e-3)
        w_r = solve_nonlinear_cg(ob_r, method="HZ", tol=1e-6, max_iter=400)["w"]
        p_r = ob_r.predict_proba(w_r, Xr_te)

        cross_gen[lbl] = {
            "Spectral-CG (13-Band)": compute_metrics(y_te[te_m], p_c),
            "SFNet (13-Band) [2]": compute_metrics(y_te[te_m], p_sf_cg),
            "RGB-Only Baseline": compute_metrics(y_te[te_m], p_r),
        }

    # =========================================================================
    # [7/11] Optimizer Benchmark + Band/Module Ablations
    # =========================================================================
    print("[7/11] Benchmarking Dual-CG Optimizers & Spectral Band / Module Ablations...")
    opt_benchmark, main_traces = {}, {}
    for lam in [1e-4, 1e-3, 1e-2]:
        lk = f"{lam:.0e}"
        obj = RegularizedLogisticObjective(Phi_tr, y_tr, lam=lam)
        c0 = obj.conditioning_stats(np.zeros(Phi_tr.shape[1] + 1))
        j_star = solve_lbfgs_benchmark(obj, tol=1e-12, max_iter=1000)["final_loss"]
        lres = {"lambda": lam, "kappa_w0": c0["kappa_exact"], "j_star": j_star, "solvers": {}}
        for m in ["GD", "FR", "PRP+", "DY", "HZ", "PCG-HZ", "L-BFGS"]:
            rs = solve_lbfgs_benchmark(obj, tol=1e-6, max_iter=400) if m == "L-BFGS" else solve_nonlinear_cg(obj, method=m, tol=1e-6, max_iter=400)
            lres["solvers"][m] = {
                "iterations": rs["iterations"], "wall_time_ms": round(rs["wall_time_ms"], 3),
                "n_feval": rs["n_feval"], "final_grad_norm": rs["final_grad_norm"],
                "final_gap": max(rs["final_loss"] - j_star, 1e-16), "converged": rs["converged"],
            }
            if lam == 1e-3:
                main_traces[m] = {
                    "grad_history": [float(v) for v in rs["grad_history"][:200]],
                    "gap_history": [float(max(f - j_star, 1e-16)) for f in rs["loss_history"][:200]],
                }
        opt_benchmark[lk] = lres

    band_ablation = {}
    for label, b_idx in [
        ("RGB Only (3 Bands)", [1, 2, 3]),
        ("RGB + NIR (4 Bands)", [1, 2, 3, 7]),
        ("RGB + NIR + Red-Edge (8 Bands)", [1, 2, 3, 4, 5, 6, 7, 8]),
        ("Full Sentinel-2 MSI (13 Bands)", list(range(13))),
    ]:
        if len(b_idx) == 13:
            band_ablation[label] = compute_metrics(y_te, obj_main.predict_proba(w_star, Phi_te))
        else:
            tr_r = extract_dataset_features(X_all[tr_idx], priors, M_endmembers, band_indices=b_idx)
            te_r = extract_dataset_features(X_all[te_idx], priors, M_endmembers, band_indices=b_idx)
            (tr_s, te_s), _, _ = standardize(tr_r, te_r)
            ob = RegularizedLogisticObjective(tr_s, y_tr, lam=1e-3)
            rs = solve_nonlinear_cg(ob, method="HZ", tol=1e-6, max_iter=400)
            band_ablation[label] = compute_metrics(y_te, ob.predict_proba(rs["w"], te_s))

    module_ablation = {}
    for mlabel, cols in [
        ("Module A (Cov. + Riemannian SPD, 25-D)", list(range(0, 25))),
        ("Module B (Biophysical Indices, 16-D)", list(range(25, 41))),
        ("Module C (LMM + SAM Unmixing, 10-D)", list(range(41, 51))),
        ("Module D (2D DFT + Unsigned Dev., 21-D)", list(range(51, 72))),
        ("Modules A + B (41-D)", list(range(0, 41))),
        ("Modules A + B + C (51-D)", list(range(0, 51))),
        ("Full A + B + C + D (72-D)", list(range(0, 72))),
    ]:
        ob = RegularizedLogisticObjective(Phi_tr[:, cols], y_tr, lam=1e-3)
        rs = solve_nonlinear_cg(ob, method="HZ", tol=1e-6, max_iter=400)
        module_ablation[mlabel] = compute_metrics(y_te, ob.predict_proba(rs["w"], Phi_te[:, cols]))

    # =========================================================================
    # [8/11] Robustness, Cirrus Cloud & Hyperparameter Sensitivity
    # =========================================================================
    print("[8/11] Running Robustness, Cirrus Cloud Stress Test & Sensitivity Sweeps...")
    robustness = {"Clean (No Perturbation)": compute_metrics(y_te, obj_main.predict_proba(w_star, Phi_te))}
    pert_rng = np.random.default_rng(999)
    for pkey, plabel in [
        ("noise_40", "Gaussian Noise (SNR 40 dB)"),
        ("noise_30", "Gaussian Noise (SNR 30 dB)"),
        ("noise_20", "Gaussian Noise (SNR 20 dB)"),
        ("quant_8bit", "8-Bit Lossy Quantization"),
        ("downsample_2x", "2x Spatial Down/Upsampling"),
        ("cirrus_thin", "Thin Cirrus Cloud (OD=0.10)"),
        ("cirrus_mod", "Moderate Cirrus Cloud (OD=0.25)"),
    ]:
        if pkey == "cirrus_thin":
            Xp = np.stack([apply_cirrus_cloud_haze(c, 0.10, pert_rng) for c in X_all[te_idx]], axis=0)
        elif pkey == "cirrus_mod":
            Xp = np.stack([apply_cirrus_cloud_haze(c, 0.25, pert_rng) for c in X_all[te_idx]], axis=0)
        else:
            Xp = np.stack([apply_perturbation(c, pkey, pert_rng) for c in X_all[te_idx]], axis=0)
        Phi_p = (extract_dataset_features(Xp, priors, M_endmembers) - mu_full) / sd_full
        robustness[plabel] = compute_metrics(y_te, obj_main.predict_proba(w_star, Phi_p))

    lambda_sweep = []
    for lv in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        ob_l = RegularizedLogisticObjective(Phi_tr, y_tr, lam=lv)
        rs_l = solve_nonlinear_cg(ob_l, method="HZ", tol=1e-6, max_iter=400)
        ml = compute_metrics(y_te, ob_l.predict_proba(rs_l["w"], Phi_te))
        lambda_sweep.append({"lambda": lv, "roc_auc": ml["roc_auc"], "f1": ml["f1"], "iters": rs_l["iterations"]})

    k_sweep = []
    auth_tr_cubes = X_all[tr_idx][y_tr == -1]
    for Kv in [4, 6, 8, 10]:
        Mk = extract_endmembers_vca(auth_tr_cubes, n_endmembers=Kv, seed=42)
        trk, tek = Phi_tr_raw.copy(), Phi_te_raw.copy()
        for ii, ig in enumerate(tr_idx):
            rc, sc, _ = compute_unmixing_and_sam(X_all[ig], Mk, lambda_u=1e-3)
            trk[ii, 41:51] = [float(np.mean(rc)), float(np.std(rc)), float(np.percentile(rc, 95)), float(np.max(rc)), float(np.mean(sc)), float(np.std(sc)), float(np.percentile(sc, 95)), float(np.max(sc)), float(np.mean(rc > 0.035)), float(np.mean(sc > 0.08))]
        for ii, ig in enumerate(te_idx):
            rc, sc, _ = compute_unmixing_and_sam(X_all[ig], Mk, lambda_u=1e-3)
            tek[ii, 41:51] = [float(np.mean(rc)), float(np.std(rc)), float(np.percentile(rc, 95)), float(np.max(rc)), float(np.mean(sc)), float(np.std(sc)), float(np.percentile(sc, 95)), float(np.max(sc)), float(np.mean(rc > 0.035)), float(np.mean(sc > 0.08))]
        (trks, teks), _, _ = standardize(trk, tek)
        ob_k = RegularizedLogisticObjective(trks, y_tr, lam=1e-3)
        rs_k = solve_nonlinear_cg(ob_k, method="HZ", tol=1e-6, max_iter=400)
        mk = compute_metrics(y_te, ob_k.predict_proba(rs_k["w"], teks))
        k_sweep.append({"K": Kv, "roc_auc": mk["roc_auc"], "f1": mk["f1"]})

    # =========================================================================
    # [9/11] Explicit Failure-Mode & Error Analysis (Gap 5)
    # =========================================================================
    print("[9/11] Running Explicit Failure-Mode & Error Analysis (FP / FN Root Causes)...")
    probs_oof = oof_probs["Spectral-CG (HZ, 13-Band)"]
    preds_oof = (probs_oof >= 0.5).astype(int)
    fp_idx = np.where((y_bin_all == 0) & (preds_oof == 1))[0]
    fn_idx = np.where((y_bin_all == 1) & (preds_oof == 0))[0]
    tp_idx = np.where((y_bin_all == 1) & (preds_oof == 1))[0]
    tn_idx = np.where((y_bin_all == 0) & (preds_oof == 0))[0]

    def _tile_stats(idxs):
        if len(idxs) == 0:
            return {"count": 0, "mean_spatial_std": 0.0, "mean_brightness": 0.0, "top_classes": {}}
        stds = [float(np.std(X_all[i])) for i in idxs]
        brts = [float(np.mean(X_all[i])) for i in idxs]
        cc = {}
        for i in idxs:
            c = str(classes_all[i])
            cc[c] = cc.get(c, 0) + 1
        top_c = dict(sorted(cc.items(), key=lambda kv: kv[1], reverse=True)[:4])
        return {"count": int(len(idxs)), "mean_spatial_std": round(float(np.mean(stds)), 4), "mean_brightness": round(float(np.mean(brts)), 4), "top_classes": top_c}

    # Tier breakdown of False Negatives (which generator is hardest?)
    fn_by_tier = {}
    for i in fn_idx:
        t = str(tiers_all[i])
        fn_by_tier[t] = fn_by_tier.get(t, 0) + 1

    failure_analysis = {
        "fp_summary": _tile_stats(fp_idx),
        "fn_summary": _tile_stats(fn_idx),
        "tn_summary": _tile_stats(tn_idx),
        "tp_summary": _tile_stats(tp_idx),
        "fn_by_generator": fn_by_tier,
        "worst_fp_prob": round(float(np.max(probs_oof[fp_idx])), 4) if len(fp_idx) else 0.0,
        "worst_fn_prob": round(float(np.min(probs_oof[fn_idx])), 4) if len(fn_idx) else 1.0,
    }

    # =========================================================================
    # [10/11] Multi-Scale Tier-3 Spatial Anomaly Localization
    # =========================================================================
    print("[10/11] Evaluating Multi-Scale Tier-3 Spatial Anomaly Localization...")
    loc_rng = np.random.default_rng(2026)
    loc_scenes, sample_spliced = [], None
    for idx_s, cls in enumerate(["Forest", "AnnualCrop", "Residential", "River", "Industrial"]):
        cand = [i for i in te_idx if classes_all[i] == cls and tiers_all[i] == "authentic"]
        cube_sp, mask_sp = synthesize_from_real_cube(X_all[cand[0]], "spliced", loc_rng)
        hmap, sam_m = multiscale_anomaly_map(cube_sp, priors, M_endmembers, mu_full, sd_full, w_star)
        auc_px = float(roc_auc_score(mask_sp.ravel(), hmap.ravel()))
        loc_scenes.append({"class": cls, "pixel_auc": round(auc_px, 4)})
        if idx_s == 0:
            rgb_vis = cube_sp[::2, ::2, [3, 2, 1]]
            rgb_scale = float(np.percentile(rgb_vis, 98)) + 1e-6
            sample_spliced = {
                "class": cls,
                "rgb": np.clip(rgb_vis / rgb_scale, 0.0, 1.0).tolist(),
                "ground_truth_mask": mask_sp[::2, ::2].tolist(),
                "anomaly_heatmap": np.round(hmap[::2, ::2], 4).tolist(),
                "sam_map": np.round(sam_m[::2, ::2], 4).tolist(),
                "pixel_auc": round(auc_px, 4),
            }

    # =========================================================================
    # [11/11] Export Complete Top-Tier Q1 Results Payload
    # =========================================================================
    print("[11/11] Exporting Top-Tier Q1 results JSON...")
    feat_names = get_feature_names()
    w_weights = w_star[:-1]
    importance_ranked = sorted(
        [{"feature": feat_names[j], "weight": float(w_weights[j]), "abs_weight": float(abs(w_weights[j])), "module": feat_names[j].split(":")[0]} for j in range(len(feat_names))],
        key=lambda x: x["abs_weight"],
        reverse=True,
    )
    total_abs = sum(x["abs_weight"] for x in importance_ranked) + 1e-12
    module_weight_share = {m: round(100.0 * sum(x["abs_weight"] for x in importance_ranked if x["module"] == m) / total_abs, 2) for m in ["A", "B", "C", "D"]}

    def _mean_spec(tname):
        idxs = np.where((classes_all == "Forest") & (tiers_all == tname))[0][:12]
        return [round(float(v), 4) for v in np.mean(X_all[idxs], axis=(0, 1, 2))]

    _, R_ex = compute_sample_correlation(X_all[np.where((classes_all == "Forest") & (tiers_all == "ddpm_unet"))[0][0]])
    delta_R_ex = np.abs(R_ex - priors["R_global"])

    dashboard_samples = []
    for s_cls, s_tier in [
        ("Forest", "authentic"), ("Forest", "ddpm_unet"), ("Forest", "neural_rgb2ms"),
        ("Residential", "authentic"), ("AnnualCrop", "neural_ae"), ("River", "resnet_gan")
    ]:
        cand = [i for i in te_idx if classes_all[i] == s_cls and tiers_all[i] == s_tier]
        if not cand:
            cand = np.where((classes_all == s_cls) & (tiers_all == s_tier))[0].tolist()
        c_cube = X_all[cand[0]]
        phi_c, mods_c = extract_features_single(c_cube, priors, M_endmembers)
        prob_fake = float(obj_main.predict_proba(w_star, ((phi_c - mu_full) / sd_full)[None, :])[0])
        _, R_c = compute_sample_correlation(c_cube)
        dR_c = np.abs(R_c - priors["R_global"])
        _, _, cg_it = compute_unmixing_and_sam(c_cube, M_endmembers)
        dashboard_samples.append({
            "id": f"{s_cls}_{s_tier}", "land_class": s_cls, "tier": s_tier,
            "ground_truth": "Authentic" if s_tier == "authentic" else "AI-Generated",
            "prob_fake": round(prob_fake, 4),
            "prediction": "AI-Generated" if prob_fake >= 0.5 else "Authentic",
            "mean_spectrum": [round(float(v), 4) for v in np.mean(c_cube, axis=(0, 1))],
            "delta_R_frob": round(float(mods_c["A"][0]), 4),
            "delta_R_spec": round(float(mods_c["A"][1]), 4),
            "riem_spd_dist": round(float(mods_c["A"][5]), 4),
            "ndvi_mean": round(float(mods_c["B"][0]), 4),
            "ndwi_mean": round(float(mods_c["B"][2]), 4),
            "ndbi_mean": round(float(mods_c["B"][4]), 4),
            "ndre_mean": round(float(mods_c["B"][6]), 4),
            "lmm_res_mean": round(float(mods_c["C"][0]), 5),
            "sam_mean_deg": round(float(np.degrees(mods_c["C"][4])), 3),
            "cg_unmix_iters": int(cg_it),
            "hf_vis": round(float(mods_c["D"][0]), 4),
            "hf_nir": round(float(mods_c["D"][6]), 4),
            "hf_swir": round(float(mods_c["D"][9]), 4),
            "delta_R_matrix": np.round(dR_c, 3).tolist(),
        })

    payload = {
        "dataset": {
            "source_label": "Real Copernicus Sentinel-2 EuroSAT-MS 64x64x13 GeoTIFF (Zenodo DOI: 10.5281/zenodo.7711810)",
            "total_patches": int(N), "n_authentic": int(len(auth_cubes)),
            "n_neural_ae": int(len(f_ae)), "n_neural_rgb2ms": int(len(f_rgb2ms)),
            "n_ddpm_unet": int(len(f_ddpm)), "n_resnet_gan": int(len(f_rgan)),
            "cv_folds": 5, "patch_size": 64, "num_bands": 13, "num_features": 72,
            "num_endmembers": 6, "classes": EUROSAT_CLASSES, "bands": SENTINEL2_BANDS,
        },
        "verification": {
            "gradient_relative_error": float(grad_rel_err),
            "conditioning_w0_lam1e-3": cond_init,
            "conditioning_wstar_lam1e-3": cond_opt,
            "cpu_inference_ms_per_tile": round(float(cpu_latency_ms), 2),
            "model_memory_kb": round((len(w_star) * 8 + 13 * 13 * 8 * 11 + 13 * 6 * 8) / 1024.0, 2),
        },
        "detection_comparison": cv_summary,
        "delong_tests": delong_tests,
        "per_class_performance": per_class_perf,
        "roc_curves": roc_curves,
        "band_ablation": band_ablation,
        "module_ablation": module_ablation,
        "cross_generator": cross_gen,
        "adaptive_attacks": adaptive_attacks,
        "cross_domain": cross_domain,
        "robustness": robustness,
        "sensitivity": {"lambda_sweep": lambda_sweep, "k_sweep": k_sweep},
        "failure_analysis": failure_analysis,
        "optimizer_benchmark": opt_benchmark,
        "convergence_traces_lam1e-3": main_traces,
        "localization": {
            "mean_pixel_auc": round(float(np.mean([s["pixel_auc"] for s in loc_scenes])), 4),
            "per_scene": loc_scenes,
            "sample_spliced": sample_spliced,
        },
        "feature_importance": {"top_15": importance_ranked[:15], "module_weight_share_pct": module_weight_share},
        "spectral_profiles": {
            "band_ids": [b["id"] for b in SENTINEL2_BANDS],
            "wavelengths_nm": [b["wavelength_nm"] for b in SENTINEL2_BANDS],
            "authentic_forest": _mean_spec("authentic"),
            "neural_ae_forest": _mean_spec("neural_ae"),
            "neural_rgb2ms_forest": _mean_spec("neural_rgb2ms"),
            "ddpm_unet_forest": _mean_spec("ddpm_unet"),
            "resnet_gan_forest": _mean_spec("resnet_gan"),
            "delta_R_example": np.round(delta_R_ex, 4).tolist(),
        },
        "dashboard_samples": dashboard_samples,
    }

    out_json = os.path.join(results_dir, "experiment_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== TOP-TIER Q1 5-FOLD CV SUMMARY (1,000 REAL EUROSAT-MS + 4 INDEPENDENT NEURAL GENERATORS) ===")
    for name in detector_names:
        s = cv_summary[name]
        p_str = f"p={delong_tests[name]['p_value']:.1e}" if name in delong_tests else "Ref"
        print(f"  {name:34s}: AUC={s['roc_auc_mean']:.4f}±{s['roc_auc_std']:.4f}, TPR@1%={s['tpr_at_1fpr_mean']*100:5.1f}%, ECE={s['ece_mean']:.3f}, {p_str}")
    print("\n--- ADAPTIVE WHITE-BOX ATTACKS ---")
    for k, v in adaptive_attacks.items():
        print(f"  {k:48s}: Full={v['Full Spectral-CG (A+B+C+D)']:.4f} | ModA={v['Module A Only']:.4f} | ModC+D={v['Modules C+D Defense']:.4f}")
    print("\n--- CROSS-DOMAIN (L1C->L2A & SENTINEL-2->LANDSAT-8) ---")
    for k, v in cross_domain.items():
        print(f"  {k:52s}: Acc={v['accuracy']*100:5.2f}%, AUC={v['roc_auc']:.4f}")
    print("\n--- ZERO-SHOT CROSS-GENERATOR TRANSFER ---")
    for k, v in cross_gen.items():
        print(f"  {k:44s}: Spectral-CG={v['Spectral-CG (13-Band)']['roc_auc']:.4f} | SFNet={v['SFNet (13-Band) [2]']['roc_auc']:.4f} | RGB={v['RGB-Only Baseline']['roc_auc']:.4f}")
    print(f"\nFailure Analysis: FP={failure_analysis['fp_summary']['count']}, FN={failure_analysis['fn_summary']['count']} (by tier: {failure_analysis['fn_by_generator']})")
    print(f"Saved Top-Tier Q1 results to: {out_json}")


if __name__ == "__main__":
    main()
