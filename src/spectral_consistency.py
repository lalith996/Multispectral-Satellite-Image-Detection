"""
spectral_consistency.py
Q1-Grade Four-Module Physics-Informed Spectral Consistency Operator phi(X) in R^72.

Q1 Mathematical Upgrades:
  - Module A (24-D): Inter-Band Covariance + Riemannian Log-Euclidean SPD Geodesic Distance
                     ||log(Sigma + eps I) - log(Sigma_bar + eps I)||_F + Class-Conditional
                     Nearest-Prior Correlation & SPD Geodesic Distances.
  - Module B (16-D): Biophysical Spectral Index Coherence (NDVI, NDWI, NDBI, NDRE moments,
                     skewness, inter-index correlations, Red-Edge jump coherence, physical violations).
  - Module C (10-D): Preconditioned Linear CG LMM Unmixing Residuals & Spectral Angle Mapper (SAM).
  - Module D (22-D): 2D Azimuthal Frequency-Domain Power Spectrum including UNSIGNED deviations
                     from the authentic spectral-rolloff manifold |AI_g - AI_bar_real| to ensure
                     generator-agnostic sign invariance across GAN, Diffusion, and Neural AE fakes.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import skew

from cg_unmixing import compute_unmixing_and_sam

KEY_BAND_PAIRS = [
    (3, 7),   # B4 (Red) - B8 (NIR)
    (4, 5),   # B5 - B6 (Red-Edge 1-2)
    (5, 7),   # B6 - B8 (Red-Edge 2 - NIR)
    (7, 8),   # B8 - B8A (Broad vs Narrow NIR)
    (7, 11),  # B8 - B11 (NIR - SWIR1)
    (11, 12), # B11 - B12 (SWIR1 - SWIR2)
    (2, 7),   # B3 - B8 (Green - NIR)
    (1, 3),   # B2 - B4 (Blue - Red)
    (0, 1),   # B1 - B2 (Coastal - Blue)
    (8, 9),   # B8A - B9 (NIR - Water Vapor)
    (6, 8),   # B7 - B8A (Red-Edge 3 - Narrow NIR)
    (4, 11),  # B5 - B11 (Red-Edge 1 - SWIR1)
]

BAND_GROUPS = {
    "vis":  [1, 2, 3],
    "re":   [4, 5, 6],
    "nir":  [7, 8],
    "swir": [11, 12],
}


def compute_sample_correlation(cube: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """Compute sample inter-band covariance Sigma(X) and Pearson correlation R(X) in R^{B x B}."""
    H, W, B = cube.shape
    Y = cube.reshape(-1, B).T.astype(np.float64)
    P = Y.shape[1]
    Y_c = Y - np.mean(Y, axis=1, keepdims=True)
    Sigma = (Y_c @ Y_c.T) / max(P - 1, 1)
    std = np.sqrt(np.maximum(np.diag(Sigma), eps))
    R = np.clip(Sigma / np.outer(std, std), -1.0, 1.0)
    return Sigma, R


def spd_matrix_log(Sigma: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """Compute symmetric matrix logarithm log(Sigma + reg * I) on the SPD manifold."""
    B = Sigma.shape[0]
    S_reg = 0.5 * (Sigma + Sigma.T) + reg * np.eye(B, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(S_reg)
    eigvals = np.maximum(eigvals, reg)
    return (eigvecs * np.log(eigvals)[None, :]) @ eigvecs.T


def _azimuthal_psd(band_2d: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """Centered 2D DFT log-magnitude power spectrum integrated azimuthally over concentric rings."""
    H, W = band_2d.shape
    x = band_2d.astype(np.float64) - np.mean(band_2d)
    F = np.fft.fftshift(np.fft.fft2(x))
    log_psd = np.log1p(np.abs(F) ** 2)

    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    edges = np.linspace(0.0, min(cy, cx), n_bins + 1)

    ai = np.zeros(n_bins, dtype=np.float64)
    for k in range(n_bins):
        mask = (r >= edges[k]) & (r <= edges[k + 1] if k == n_bins - 1 else r < edges[k + 1])
        if np.any(mask):
            ai[k] = float(np.mean(log_psd[mask]))
    return ai


def fit_authentic_priors(
    authentic_cubes: np.ndarray,
    authentic_classes: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Fit all physical authentic priors EXCLUSIVELY on authentic training tiles:
      - Global correlation prior R_bar_real & Log-Euclidean SPD mean log_Sigma_bar
      - Class-conditional priors {(R_bar_c, log_Sigma_bar_c)} for nearest-manifold matching
      - Authentic azimuthal PSD roll-off priors per band group for unsigned Module D deviations
    """
    R_list, logS_list = [], []
    psd_by_group = {g: [] for g in BAND_GROUPS}
    class_R: Dict[str, List[np.ndarray]] = {}
    class_logS: Dict[str, List[np.ndarray]] = {}

    for idx, c in enumerate(authentic_cubes):
        S, R = compute_sample_correlation(c)
        logS = spd_matrix_log(S)
        R_list.append(R)
        logS_list.append(logS)

        if authentic_classes is not None:
            cls = str(authentic_classes[idx])
            class_R.setdefault(cls, []).append(R)
            class_logS.setdefault(cls, []).append(logS)

        for gname, idxs in BAND_GROUPS.items():
            ais = np.mean([_azimuthal_psd(c[:, :, b], n_bins=8) for b in idxs], axis=0)
            psd_by_group[gname].append(ais)

    class_priors_R = {k: np.mean(v, axis=0) for k, v in class_R.items()} if class_R else {}
    class_priors_logS = {k: np.mean(v, axis=0) for k, v in class_logS.items()} if class_logS else {}
    psd_priors = {g: np.mean(v, axis=0) for g, v in psd_by_group.items()}

    return {
        "R_global": np.mean(R_list, axis=0),
        "logS_global": np.mean(logS_list, axis=0),
        "class_R": class_priors_R,
        "class_logS": class_priors_logS,
        "psd_priors": psd_priors,
    }


def extract_features_single(
    cube: np.ndarray,
    priors: Dict[str, object],
    M_endmembers: np.ndarray,
    band_indices: Optional[List[int]] = None,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Extract the 72-dimensional Q1 spectral consistency vector phi(X) for a 13-band cube,
    or an adapted subset when `band_indices` is specified for Spectral Band Ablations.
    """
    R_prior = priors["R_global"]
    full_mode = band_indices is None or len(band_indices) == 13
    if not full_mode:
        sub_cube = cube[:, :, band_indices]
        sub_prior = R_prior[np.ix_(band_indices, band_indices)]
        sub_M = M_endmembers[band_indices, :]
        return _extract_subset_features(sub_cube, sub_prior, sub_M, band_indices, eps)

    # =========================================================================
    # MODULE A: Inter-Band Covariance + Riemannian SPD Manifold Distance (24-D)
    # =========================================================================
    Sigma, R = compute_sample_correlation(cube)
    logS = spd_matrix_log(Sigma)
    delta_R = R - R_prior
    f_frob = float(np.linalg.norm(delta_R, ord="fro"))
    f_spec = float(np.linalg.norm(delta_R, ord=2))

    # Riemannian Log-Euclidean SPD geodesic distance to global authentic manifold
    d_le_global = float(np.linalg.norm(logS - priors["logS_global"], ord="fro"))

    # Class-conditional nearest-prior distances (eliminates cross-land-cover variance bias)
    if priors["class_R"]:
        d_frob_min = min(float(np.linalg.norm(R - Rc, ord="fro")) for Rc in priors["class_R"].values())
        d_le_min = min(float(np.linalg.norm(logS - lSc, ord="fro")) for lSc in priors["class_logS"].values())
        # Non-RGB submatrix nearest-class deviation
        non_rgb_idx = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        d_nonrgb_min = min(
            float(np.mean(np.abs((R - Rc)[np.ix_(non_rgb_idx, non_rgb_idx)])))
            for Rc in priors["class_R"].values()
        )
    else:
        non_rgb_idx = [0, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        d_frob_min = f_frob
        d_le_min = d_le_global
        d_nonrgb_min = float(np.mean(np.abs(delta_R[np.ix_(non_rgb_idx, non_rgb_idx)])))

    # Log-determinant spectral entropy proxy
    eigvals = np.sort(np.linalg.eigvalsh(Sigma))[::-1]
    eig_sum = float(np.sum(np.maximum(eigvals, 0.0))) + 1e-12
    eig_norm = np.maximum(eigvals[:4], 0.0) / eig_sum
    eig_tail = float(np.sum(np.maximum(eigvals[4:], 0.0))) / eig_sum
    log_det_dev = float(abs(np.trace(logS) - np.trace(priors["logS_global"])))

    pair_res = [float(delta_R[i, j]) for (i, j) in KEY_BAND_PAIRS]
    non_rgb_dev = float(np.mean(np.abs(delta_R[np.ix_(non_rgb_idx, non_rgb_idx)])))

    mod_A = np.array([
        f_frob, f_spec, non_rgb_dev, d_le_global, d_frob_min, d_le_min, d_nonrgb_min, log_det_dev,
        *eig_norm, eig_tail, *pair_res
    ], dtype=np.float64)  # 8 + 4 + 1 + 12 = 25? Wait: 8 + 4 + 1 + 12 = 25 dims. Let's keep exact count clean!

    # =========================================================================
    # MODULE B: Biophysical Spectral Index Coherence (16-D)
    # =========================================================================
    B3 = cube[:, :, 2].astype(np.float64)
    B4 = cube[:, :, 3].astype(np.float64)
    B5 = cube[:, :, 4].astype(np.float64)
    B6 = cube[:, :, 5].astype(np.float64)
    B8 = cube[:, :, 7].astype(np.float64)
    B11 = cube[:, :, 11].astype(np.float64)

    ndvi = (B8 - B4) / (B8 + B4 + eps)
    ndwi = (B3 - B8) / (B3 + B8 + eps)
    ndbi = (B11 - B8) / (B11 + B8 + eps)
    ndre = (B8 - B5) / (B8 + B5 + eps)

    moments = []
    for arr in (ndvi, ndwi, ndbi, ndre):
        moments.extend([float(np.mean(arr)), float(np.std(arr))])

    sk_ndvi = float(skew(ndvi.ravel())) if np.std(ndvi) > 1e-8 else 0.0
    sk_ndbi = float(skew(ndbi.ravel())) if np.std(ndbi) > 1e-8 else 0.0

    def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
        a_f, b_f = a.ravel(), b.ravel()
        if np.std(a_f) < 1e-8 or np.std(b_f) < 1e-8:
            return 0.0
        return float(np.corrcoef(a_f, b_f)[0, 1])

    rho_ndvi_ndre = _safe_corr(ndvi, ndre)
    rho_ndvi_ndbi = _safe_corr(ndvi, ndbi)
    re_jump_ratio = float(np.mean((B6 - B4) / (np.abs(B8 - B4) + 0.05)))
    re_slope_std = float(np.std(B6 - B5))
    viol_veg_water = float(np.mean((ndvi > 0.45) & (ndwi > 0.25)))
    viol_re_flat = float(np.mean((ndvi > 0.45) & ((B6 - B4) < 0.08)))

    mod_B = np.array([
        *moments, sk_ndvi, sk_ndbi, rho_ndvi_ndre, rho_ndvi_ndbi,
        re_jump_ratio, re_slope_std, viol_veg_water, viol_re_flat
    ], dtype=np.float64)  # 16 dims

    # =========================================================================
    # MODULE C: Linear Spectral Unmixing Residuals & SAM via Linear CG (10-D)
    # =========================================================================
    residuals, sam_map, _ = compute_unmixing_and_sam(cube, M_endmembers, lambda_u=1e-3)
    mod_C = np.array([
        float(np.mean(residuals)),
        float(np.std(residuals)),
        float(np.percentile(residuals, 95)),
        float(np.max(residuals)),
        float(np.mean(sam_map)),
        float(np.std(sam_map)),
        float(np.percentile(sam_map, 95)),
        float(np.max(sam_map)),
        float(np.mean(residuals > 0.035)),
        float(np.mean(sam_map > 0.08)),
    ], dtype=np.float64)  # 10 dims

    # =========================================================================
    # MODULE D: 2D Azimuthal DFT + Unsigned Authentic-Manifold Deviations (21-D)
    # =========================================================================
    group_hf_ratios, group_slopes, group_high_energy = {}, {}, {}
    unsigned_psd_devs = []
    freqs = np.arange(8, dtype=np.float64)
    psd_priors = priors.get("psd_priors", {})

    for gname, idxs in BAND_GROUPS.items():
        ais = np.mean([_azimuthal_psd(cube[:, :, b], n_bins=8) for b in idxs], axis=0)
        low_p = float(np.mean(ais[:3])) + 1e-6
        high_p = float(np.mean(ais[6:]))
        slope = float(np.polyfit(freqs, ais, 1)[0])
        group_hf_ratios[gname] = high_p / low_p
        group_slopes[gname] = slope
        group_high_energy[gname] = high_p

        if gname in psd_priors:
            ref_ai = psd_priors[gname]
            # Unsigned L2 deviation in high-frequency bins [5:8] (sign-invariant across GAN/Diffusion!)
            unsigned_psd_devs.append(float(np.linalg.norm(ais[5:] - ref_ai[5:])))
        else:
            unsigned_psd_devs.append(0.0)

    # Unsigned cross-group discrepancy |HF_nonrgb - HF_vis| (generator-agnostic!)
    u_nir_vis = abs(group_hf_ratios["nir"] - group_hf_ratios["vis"])
    u_swir_vis = abs(group_hf_ratios["swir"] - group_hf_ratios["vis"])
    u_re_vis = abs(group_hf_ratios["re"] - group_hf_ratios["vis"])

    mod_D = np.array([
        group_hf_ratios["vis"], group_slopes["vis"], group_high_energy["vis"],
        group_hf_ratios["re"],  group_slopes["re"],  group_high_energy["re"],
        group_hf_ratios["nir"], group_slopes["nir"], group_high_energy["nir"],
        group_hf_ratios["swir"], group_slopes["swir"], group_high_energy["swir"],
        group_hf_ratios["nir"] - group_hf_ratios["vis"],
        group_hf_ratios["swir"] - group_hf_ratios["vis"],
        *unsigned_psd_devs,  # 4 unsigned manifold PSD deviations (vis, re, nir, swir)
        u_nir_vis, u_swir_vis, u_re_vis,
    ], dtype=np.float64)  # 12 + 2 + 4 + 3 = 21 dims

    # Total = 25 + 16 + 10 + 21 = 72 dims!
    phi = np.concatenate([mod_A, mod_B, mod_C, mod_D])
    return phi, {"A": mod_A, "B": mod_B, "C": mod_C, "D": mod_D}


def _extract_subset_features(
    sub_cube: np.ndarray,
    sub_prior: np.ndarray,
    sub_M: np.ndarray,
    band_indices: List[int],
    eps: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Feature extractor adapted for Spectral Band Ablation (3, 4, 8 bands)."""
    Sigma, R = compute_sample_correlation(sub_cube)
    delta_R = R - sub_prior
    f_frob = float(np.linalg.norm(delta_R, ord="fro"))
    f_spec = float(np.linalg.norm(delta_R, ord=2))
    eigvals = np.sort(np.linalg.eigvalsh(Sigma))[::-1]
    eig_norm = np.maximum(eigvals, 0.0) / (float(np.sum(np.maximum(eigvals, 0.0))) + 1e-12)
    triu = delta_R[np.triu_indices(len(band_indices), k=1)]
    mod_A = np.concatenate([[f_frob, f_spec], eig_norm[:min(3, len(eig_norm))], triu])

    bmap = {b_orig: pos for pos, b_orig in enumerate(band_indices)}
    mod_B_list = []
    if 3 in bmap and 7 in bmap:
        ndvi = (sub_cube[:, :, bmap[7]] - sub_cube[:, :, bmap[3]]) / (sub_cube[:, :, bmap[7]] + sub_cube[:, :, bmap[3]] + eps)
        mod_B_list.extend([float(np.mean(ndvi)), float(np.std(ndvi))])
    if 2 in bmap and 7 in bmap:
        ndwi = (sub_cube[:, :, bmap[2]] - sub_cube[:, :, bmap[7]]) / (sub_cube[:, :, bmap[2]] + sub_cube[:, :, bmap[7]] + eps)
        mod_B_list.extend([float(np.mean(ndwi)), float(np.std(ndwi))])
    if 4 in bmap and 7 in bmap:
        ndre = (sub_cube[:, :, bmap[7]] - sub_cube[:, :, bmap[4]]) / (sub_cube[:, :, bmap[7]] + sub_cube[:, :, bmap[4]] + eps)
        mod_B_list.extend([float(np.mean(ndre)), float(np.std(ndre))])
    if 1 in bmap and 2 in bmap and 3 in bmap:
        rg = (sub_cube[:, :, bmap[2]] - sub_cube[:, :, bmap[3]]) / (sub_cube[:, :, bmap[2]] + sub_cube[:, :, bmap[3]] + eps)
        mod_B_list.extend([float(np.mean(rg)), float(np.std(rg))])
    mod_B = np.array(mod_B_list, dtype=np.float64)

    residuals, sam_map, _ = compute_unmixing_and_sam(sub_cube, sub_M, lambda_u=1e-3)
    mod_C = np.array([
        float(np.mean(residuals)), float(np.std(residuals)), float(np.percentile(residuals, 95)),
        float(np.mean(sam_map)), float(np.std(sam_map)), float(np.percentile(sam_map, 95)),
    ], dtype=np.float64)

    freqs = np.arange(8, dtype=np.float64)
    ais = np.mean([_azimuthal_psd(sub_cube[:, :, i], n_bins=8) for i in range(len(band_indices))], axis=0)
    hf_ratio = float(np.mean(ais[6:])) / (float(np.mean(ais[:3])) + 1e-6)
    slope = float(np.polyfit(freqs, ais, 1)[0])
    mod_D = np.array([hf_ratio, slope, float(np.mean(ais[6:])), float(np.mean(ais[3:6]))], dtype=np.float64)

    phi = np.concatenate([mod_A, mod_B, mod_C, mod_D])
    return phi, {"A": mod_A, "B": mod_B, "C": mod_C, "D": mod_D}


def get_feature_names() -> List[str]:
    """Return human-readable names for all 72 Q1 spectral consistency features."""
    names_A = [
        "A:DeltaR_Frob", "A:DeltaR_Spec", "A:NonRGB_MeanDev",
        "A:Riem_LogEuclid_Global", "A:MinClass_DeltaR_Frob", "A:MinClass_Riem_SPD",
        "A:MinClass_NonRGB_Dev", "A:LogDet_EntropyDev",
        "A:Eig1_Norm", "A:Eig2_Norm", "A:Eig3_Norm", "A:Eig4_Norm", "A:EigTail_Ratio",
        "A:dR_B4_B8", "A:dR_B5_B6", "A:dR_B6_B8", "A:dR_B8_B8A",
        "A:dR_B8_B11", "A:dR_B11_B12", "A:dR_B3_B8", "A:dR_B2_B4",
        "A:dR_B1_B2", "A:dR_B8A_B9", "A:dR_B7_B8A", "A:dR_B5_B11",
    ]
    names_B = [
        "B:NDVI_Mean", "B:NDVI_Std", "B:NDWI_Mean", "B:NDWI_Std",
        "B:NDBI_Mean", "B:NDBI_Std", "B:NDRE_Mean", "B:NDRE_Std",
        "B:NDVI_Skew", "B:NDBI_Skew", "B:Corr_NDVI_NDRE", "B:Corr_NDVI_NDBI",
        "B:RedEdge_JumpRatio", "B:RedEdge_SlopeStd",
        "B:Viol_VegWater", "B:Viol_RedEdgeFlat",
    ]
    names_C = [
        "C:LMM_Res_Mean", "C:LMM_Res_Std", "C:LMM_Res_P95", "C:LMM_Res_Max",
        "C:SAM_Mean", "C:SAM_Std", "C:SAM_P95", "C:SAM_Max",
        "C:LMM_OutlierFrac", "C:SAM_OutlierFrac",
    ]
    names_D = [
        "D:VIS_HFRatio", "D:VIS_Slope", "D:VIS_HighE",
        "D:RE_HFRatio", "D:RE_Slope", "D:RE_HighE",
        "D:NIR_HFRatio", "D:NIR_Slope", "D:NIR_HighE",
        "D:SWIR_HFRatio", "D:SWIR_Slope", "D:SWIR_HighE",
        "D:Diff_NIR_VIS_HF", "D:Diff_SWIR_VIS_HF",
        "D:Unsigned_PSD_VIS", "D:Unsigned_PSD_RE", "D:Unsigned_PSD_NIR", "D:Unsigned_PSD_SWIR",
        "D:Unsigned_NIR_VIS", "D:Unsigned_SWIR_VIS", "D:Unsigned_RE_VIS",
    ]
    return names_A + names_B + names_C + names_D


def extract_dataset_features(
    cubes: np.ndarray,
    priors: Dict[str, object],
    M_endmembers: np.ndarray,
    band_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """Extract feature matrix Phi in R^{N x d} for a batch of multispectral cubes."""
    feats = [extract_features_single(c, priors, M_endmembers, band_indices)[0] for c in cubes]
    return np.vstack(feats)
