"""
evaluate_and_visualize.py
Generates 8 Top-Tier Q1 IEEE TGRS publication figures (300 DPI PNG + vector PDF):
  - Fig 1: Technical System Architecture Diagram (engineering schematic)
  - Fig 2: 13-Band Sentinel-2 Profiles (Authentic vs. 4 Independent Neural Generators) & |Delta R| Heatmap
  - Fig 3: Dual-CG & Preconditioned PCG-HZ Convergence Curves
  - Fig 4: 5-Fold OOF ROC Curves (incl. SFNet, ViT, Durall-DFT, One-Class, RGB-CNN) & 10-Bin Calibration Reliability
  - Fig 5: Adaptive White-Box WCT/NDVI Attacker Defense & Cross-Domain (L2A BOA / Landsat-8 OLI) Transfer
  - Fig 6: Hyperparameter Sensitivity (lambda) & Noise / Quantization / Resampling / Cirrus Cloud Robustness
  - Fig 7: Spectral Band Ablation (3 -> 4 -> 8 -> 13 Bands) & Top-12 Learned Forensic Weights |w*_j|
  - Fig 8: Multi-Scale Tier-3 Splicing Localization across 5 Real EuroSAT-MS Scenes
"""

from __future__ import annotations
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.titlesize": 9.2,
    "axes.labelsize": 8.3,
    "xtick.labelsize": 7.3,
    "ytick.labelsize": 7.3,
    "legend.fontsize": 6.6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
})


def draw_fig1_architecture(out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")
    NAVY, SLATE, TEAL, AMBER, BG_LIGHT = "#1B365D", "#334E68", "#1F6F78", "#8C5A1E", "#F8FAFC"

    def box(x, y, w, h, title, lines, header_color=NAVY, body_size=6.3):
        ax.add_patch(patches.Rectangle((x, y), w, h, lw=1.0, edgecolor=header_color, facecolor="#FFFFFF", zorder=2))
        hh = 3.2
        ax.add_patch(patches.Rectangle((x, y + h - hh), w, hh, lw=0.8, edgecolor=header_color, facecolor=header_color, zorder=3))
        ax.text(x + w / 2, y + h - hh / 2, title, color="white", fontsize=7.6, fontweight="bold", ha="center", va="center", zorder=4)
        top = y + h - hh - 1.3
        step = (h - hh - 1.8) / max(len(lines), 1)
        for i, ln in enumerate(lines):
            ax.text(x + 0.9, top - i * step, ln, color="#1E293B", fontsize=body_size, ha="left", va="top", zorder=4)

    def arrow(x1, y1, x2, y2, label=None, lx=None, ly=None, style="-|>", lw=1.1, color="#1E293B"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=10, lw=lw, color=color, zorder=5))
        if label:
            ax.text(lx if lx else 0.5*(x1+x2), ly if ly else 0.5*(y1+y2)+1.1, label, fontsize=6.4, color="#334155",
                    ha="center", va="center", bbox=dict(boxstyle="square,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85), zorder=6)

    for sx, sy, sw, sh, slabel in [
        (0.8, 7.5, 16.5, 34.5, "STAGE I: INGESTION"),
        (19.5, 7.5, 34.5, 34.5, "STAGE II: SPECTRAL CONSISTENCY OPERATOR $\\phi(\\mathcal{X})$"),
        (56.2, 7.5, 22.8, 34.5, "STAGE III: DUAL CG SOLVER"),
        (81.2, 7.5, 18.0, 34.5, "STAGE IV: FORENSIC OUTPUT"),
    ]:
        ax.add_patch(patches.Rectangle((sx, sy), sw, sh, lw=0.8, ls="--", edgecolor="#94A3B8", facecolor=BG_LIGHT, zorder=1))
        ax.text(sx + sw / 2, sy + sh - 1.3, slabel, fontsize=6.7, fontweight="bold", color="#334155", ha="center", va="center", zorder=2)

    box(2.0, 23.5, 14.1, 14.5, "EuroSAT-MS GeoTIFF", [r"$\mathcal{X} \in \mathbb{R}^{64 \times 64 \times 13}$", "VIS: B2, B3, B4 (10 m)", "RE: B5, B6, B7 (20 m)", "NIR: B8, B8A (10/20 m)", "SWIR: B11, B12 (20 m)", "Atm: B1, B9, B10 (60 m)"], NAVY)
    box(2.0, 9.2, 14.1, 11.2, "Radiometric Prep.", [r"$\rho = \mathrm{DN}/10^4 \in [0,1]$", "Wavelength Alignment", r"Unfold: $\mathbf{Y} \in \mathbb{R}^{B \times P}$", "5-Fold Disjoint Split"], SLATE)
    arrow(9.05, 23.5, 9.05, 20.4)

    box(21.0, 30.8, 15.2, 8.2, "Mod A: Cov. & SPD", [r"$\Delta\mathbf{R},\ d_{\mathrm{LE}}(\mathbf{\Sigma},\bar{\mathbf{\Sigma}}_c)$", r"$\|\Delta\mathbf{R}\|_F,\ \|\Delta\mathbf{R}\|_2,\ \lambda_i(\Sigma)$", "25 Riemannian/Cov. Dims"], TEAL, 6.1)
    box(37.5, 30.8, 15.2, 8.2, "Mod B: Bio-Indices", ["NDVI, NDWI, NDBI, NDRE", r"$\rho(\mathrm{NDVI},\mathrm{NDRE})$, RE Jump", "16 Physical Coherence Dims"], TEAL, 6.1)
    box(21.0, 19.8, 15.2, 9.0, "Mod C: LMM + SAM", [r"$(\mathbf{M}^\top\mathbf{M}+\lambda_u\mathbf{I})\hat{\mathbf{a}}_p=\mathbf{M}^\top\mathbf{y}_p$", "Linear CG + Simplex Proj.", r"$r_p,\ \mathrm{SAM}(\mathbf{y}_p,\hat{\mathbf{y}}_p)$ (10-D)"], TEAL, 6.1)
    box(37.5, 19.8, 15.2, 9.0, "Mod D: 2D Azim. DFT", [r"$F_b(u,v) = \mathcal{F}_{2\mathrm{D}}\{\mathcal{X}_b\}$", r"Unsigned $|AI_g - \overline{AI}_{\mathrm{real}}|$", "21 Spectral-Freq. Dims"], TEAL, 6.1)
    box(21.0, 9.2, 31.7, 7.6, "Feature Concatenation & Z-Score Standardization", [r"$\phi(\mathcal{X}) = [\phi_A;\ \phi_B;\ \phi_C;\ \phi_D] \in \mathbb{R}^{72}$", "Standardized via Authentic Training Fold Statistics"], SLATE, 6.2)

    ax.plot([16.1, 18.4, 18.4], [14.8, 14.8, 34.9], color="#1E293B", lw=1.1, zorder=4)
    arrow(18.4, 34.9, 21.0, 34.9); arrow(18.4, 24.3, 21.0, 24.3)
    arrow(28.6, 19.8, 28.6, 16.8); arrow(45.1, 19.8, 45.1, 16.8)
    ax.plot([28.6, 28.6], [30.8, 29.6], color="#475569", lw=0.9, zorder=4)
    ax.plot([45.1, 45.1], [30.8, 29.6], color="#475569", lw=0.9, zorder=4)
    ax.plot([28.6, 45.1], [29.6, 29.6], color="#475569", lw=0.9, zorder=4)
    ax.plot([36.8, 36.8], [29.6, 17.5], color="#475569", lw=0.9, zorder=4)
    arrow(36.8, 18.2, 36.8, 16.8, color="#475569", lw=0.9)

    box(57.6, 26.2, 20.0, 12.8, "Strongly Convex Objectives", [r"$\min_{\mathbf{w}} J_{\mathrm{log/Huber}}(\mathbf{w}) + \frac{\lambda}{2}\|\mathbf{w}\|_2^2$", r"$\mu$-Strongly Convex ($\mu \geq \lambda$)", r"$L$-Smooth, $\kappa = L/\mu \gg 1$", r"Exact $\nabla J(\mathbf{w})$ & $\nabla^2 J(\mathbf{w})$"], NAVY, 6.2)
    box(57.6, 9.2, 20.0, 14.2, "Nonlinear CG (HZ / PCG-HZ)", [r"$\mathbf{d}_{k+1} = -\mathbf{z}_{k+1} + \beta_k^{\mathrm{HZ}}\mathbf{d}_k$", "Strong Wolfe Zoom Line Search", r"($c_1=10^{-4},\ c_2=0.1$)", "Powell Restart & Descent Guard", r"$O(d)$ Memory, 15.7 KB Model"], NAVY, 6.2)
    arrow(52.7, 13.0, 57.6, 13.0, label=r"$\mathbf{\Phi}$", lx=55.1, ly=14.4)
    arrow(67.6, 26.2, 67.6, 23.4)

    box(82.4, 24.5, 15.6, 14.5, "Calibrated Inference", [r"$P(\mathrm{Fake}\mid\phi(\mathcal{X}))$", "ECE = 0.010, TPR@1%", "6.69 ms / Tile (CPU)", "L2A & Landsat-8 Transfer", r"Linear Attribution $|w_j^*|$"], AMBER, 6.2)
    box(82.4, 9.2, 15.6, 12.5, "Multi-Scale Localization", ["16x16 + 24x24 Windows", "LMM + SAM Pixel Fusion", "Adaptive WCT Defense", "Tier-3 Anomaly Heatmap"], AMBER, 6.2)
    arrow(77.6, 16.3, 80.0, 16.3, style="-")
    ax.plot([80.0, 80.0], [15.4, 31.7], color="#1E293B", lw=1.1, zorder=4)
    arrow(80.0, 31.7, 82.4, 31.7); arrow(80.0, 15.4, 82.4, 15.4)

    ax.add_patch(patches.Rectangle((0.8, 1.0), 98.4, 4.8, lw=0.9, edgecolor=NAVY, facecolor="#EFF6FF", zorder=2))
    ax.text(50.0, 3.4, "Theoretical Foundations:  [Thm. 1] Strong Convexity & L-Smoothness  |  [Thm. 2] O(sqrt(kappa)) Linear CG Unmixing  |  [Thm. 3] Zoutendijk Global Convergence",
            fontsize=5.9, color=NAVY, fontweight="bold", ha="center", va="center", zorder=3)
    plt.savefig(os.path.join(out_dir, "fig1_architecture.png"))
    plt.savefig(os.path.join(out_dir, "fig1_architecture.pdf"))
    plt.close(fig)


def draw_fig2_spectral_covariance(data: dict, out_dir: str) -> None:
    sp = data["spectral_profiles"]
    bands = sp["band_ids"]; x = np.arange(len(bands))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8), gridspec_kw={"width_ratios": [1.18, 1.0]})
    ax1.axvspan(0.5, 3.5, color="#E0F2FE", alpha=0.55, label="Visible RGB (B2–B4)")
    ax1.axvspan(3.5, 8.5, color="#DCFCE7", alpha=0.50, label="Red-Edge & NIR (B5–B8A)")
    ax1.axvspan(10.5, 12.5, color="#FEF3C7", alpha=0.55, label="SWIR (B11–B12)")

    ax1.plot(x, sp["authentic_forest"], "o-", color="#1B365D", lw=1.9, ms=4.5, label="Authentic EuroSAT-MS (Forest)")
    ax1.plot(x, sp["ddpm_unet_forest"], "s--", color="#C0392B", lw=1.5, ms=4.0, label="PyTorch 13-B DDPM-UNet")
    ax1.plot(x, sp["neural_rgb2ms_forest"], "D-.", color="#7C3AED", lw=1.5, ms=4.0, label="PyTorch RGB->13-B Net")
    ax1.plot(x, sp["resnet_gan_forest"], "^:", color="#D97706", lw=1.5, ms=4.2, label="PyTorch ResNet-GAN")
    ax1.set_xticks(x); ax1.set_xticklabels(bands, rotation=35, ha="right")
    ax1.set_ylabel("TOA Surface Reflectance ($\\rho$)"); ax1.set_xlabel("Sentinel-2 MSI Spectral Band")
    ax1.set_title("(a) Mean 13-Band Profile on Real EuroSAT-MS (Forest)")
    ax1.grid(True, ls=":", alpha=0.5); ax1.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=6.0)

    dR = np.array(sp["delta_R_example"])
    im = ax2.imshow(dR, cmap="YlOrRd", vmin=0.0, vmax=max(0.28, float(dR.max())), aspect="equal")
    ax2.set_xticks(x); ax2.set_yticks(x); ax2.set_xticklabels(bands, rotation=45, ha="right", fontsize=6.7); ax2.set_yticklabels(bands, fontsize=6.7)
    ax2.set_title(r"(b) Inter-Band Correlation Deviation $|\Delta \mathbf{R}|$ (DDPM-UNet)")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04).set_label(r"$|\Delta R_{ij}|$", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig2_spectral_covariance.png")); plt.savefig(os.path.join(out_dir, "fig2_spectral_covariance.pdf")); plt.close(fig)


def draw_fig3_convergence(data: dict, out_dir: str) -> None:
    traces = data["convergence_traces_lam1e-3"]
    kappa = data["optimizer_benchmark"]["1e-03"]["kappa_w0"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.6))
    styles = {
        "GD":     {"color": "#64748B", "ls": ":",  "lw": 1.6, "label": "Steepest Descent (GD)"},
        "FR":     {"color": "#D97706", "ls": "-.", "lw": 1.4, "label": "Fletcher–Reeves (FR-CG)"},
        "PRP+":   {"color": "#059669", "ls": "--", "lw": 1.4, "label": "Polak–Ribière+ (PRP+-CG)"},
        "DY":     {"color": "#7C3AED", "ls": "-",  "lw": 1.4, "label": "Dai–Yuan (DY-CG)"},
        "HZ":     {"color": "#1B365D", "ls": "-",  "lw": 2.0, "label": "Hager–Zhang (CG_DESCENT)"},
        "PCG-HZ": {"color": "#0284C7", "ls": "-.", "lw": 1.6, "label": "Preconditioned PCG-HZ"},
        "L-BFGS": {"color": "#DC2626", "ls": "--", "lw": 1.5, "label": "L-BFGS (Quasi-Newton)"},
    }
    for m, st in styles.items():
        if m not in traces: continue
        ax1.semilogy(np.arange(len(traces[m]["grad_history"])), traces[m]["grad_history"], color=st["color"], ls=st["ls"], lw=st["lw"], label=st["label"])
        ax2.semilogy(np.arange(len(traces[m]["gap_history"])), traces[m]["gap_history"], color=st["color"], ls=st["ls"], lw=st["lw"], label=st["label"])
    ax1.axhline(1e-6, color="#0F172A", ls=":", lw=0.9, alpha=0.7)
    ax1.set_xlabel("Iteration ($k$)"); ax1.set_ylabel(r"Gradient Norm $\|\nabla J(\mathbf{w}_k)\|_2$")
    ax1.set_title(rf"(a) Gradient Norm Convergence ($\lambda=10^{{-3}}, \kappa={kappa:.0f}$)"); ax1.grid(True, which="both", ls=":", alpha=0.45); ax1.legend(loc="upper right", fontsize=6.0)
    ax2.set_xlabel("Iteration ($k$)"); ax2.set_ylabel(r"Optimality Gap $J(\mathbf{w}_k) - J^*$")
    ax2.set_title(r"(b) Objective Suboptimality Decay vs. Exact $J^*$"); ax2.grid(True, which="both", ls=":", alpha=0.45); ax2.legend(loc="upper right", fontsize=6.0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig3_convergence.png")); plt.savefig(os.path.join(out_dir, "fig3_convergence.pdf")); plt.close(fig)


def draw_fig4_roc_calibration(data: dict, out_dir: str) -> None:
    """Fig 4: (a) 5-Fold OOF ROC Curves incl. Published Baselines & (b) 10-Bin Calibration Reliability Diagram."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.7))
    rocs = data["roc_curves"]; dc = data["detection_comparison"]
    for name, col, ls, lw in [
        ("Spectral-CG (HZ, 13-Band)", "#1B365D", "-", 2.2),
        ("Qi et al. SFNet (13-Band) [2]", "#059669", "--", 1.5),
        ("Durall et al. DFT (13-Band) [9]", "#0284C7", "-.", 1.5),
        ("Horvath et al. ViT (13-Band) [3]", "#7C3AED", ":", 1.6),
        ("One-Class Mahalanobis (Zero-Shot)", "#D97706", "-.", 1.5),
        ("RGB-Only Spectral-CG (3-Band)", "#64748B", ":", 1.6),
        ("PyTorch RGB 2D-CNN (3-Band)", "#DC2626", ":", 1.5),
    ]:
        if name not in rocs: continue
        ax1.plot(rocs[name]["fpr"], rocs[name]["tpr"], color=col, ls=ls, lw=lw,
                 label=f"{name.replace(' (13-Band)','').replace(' (3-Band)','')} ({dc[name]['roc_auc_mean']:.3f})")
    ax1.plot([0, 1], [0, 1], color="#94A3B8", ls="--", lw=0.8)
    ax1.set_xlabel("False Positive Rate (FPR)"); ax1.set_ylabel("True Positive Rate (TPR)")
    ax1.set_title("(a) 5-Fold OOF ROC Curves (1,000 Real EuroSAT-MS)")
    ax1.grid(True, ls=":", alpha=0.5); ax1.legend(loc="lower right", fontsize=5.9, frameon=True)

    # (b) Calibration Reliability Diagram
    ax2.plot([0, 1], [0, 1], color="#94A3B8", ls="--", lw=1.0, label="Perfect Calibration")
    for name, col, mk in [
        ("Spectral-CG (HZ, 13-Band)", "#1B365D", "o"),
        ("Qi et al. SFNet (13-Band) [2]", "#059669", "s"),
        ("Horvath et al. ViT (13-Band) [3]", "#7C3AED", "^"),
        ("Durall et al. DFT (13-Band) [9]", "#D97706", "D"),
    ]:
        bins = dc[name]["reliability_bins"]
        confs = [b["conf"] for b in bins]; accs = [b["acc"] for b in bins]
        ax2.plot(confs, accs, marker=mk, color=col, lw=1.6, ms=4.5,
                 label=f"{name.split(' (')[0]} (ECE={dc[name]['ece_mean']:.3f})")
    ax2.set_xlabel("Mean Predicted Probability"); ax2.set_ylabel("Empirical Fraction of Positives")
    ax2.set_title("(b) Probability Calibration Reliability Diagram (10-Bin)")
    ax2.grid(True, ls=":", alpha=0.5); ax2.legend(loc="upper left", fontsize=6.2, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig4_roc_calibration.png")); plt.savefig(os.path.join(out_dir, "fig4_roc_calibration.pdf")); plt.close(fig)


def draw_fig5_adaptive_crossdomain(data: dict, out_dir: str) -> None:
    """Fig 5: (a) Adaptive White-Box WCT/NDVI Attacker Defense & (b) Cross-Domain L2A / Landsat-8 Transfer."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.6), gridspec_kw={"width_ratios": [1.25, 1.0]})

    atk = data["adaptive_attacks"]
    keys = list(atk.keys())
    short = ["Clean", "WCT Cov.\nAttack", "NDVI\nMatch", "Combined\nWCT+NDVI"]
    full_v = [atk[k]["Full Spectral-CG (A+B+C+D)"] * 100 for k in keys]
    modA_v = [atk[k]["Module A Only"] * 100 for k in keys]
    modCD_v = [atk[k]["Modules C+D Defense"] * 100 for k in keys]
    x = np.arange(len(keys)); w = 0.25
    ax1.bar(x - w, full_v, w, color="#1B365D", edgecolor="#0F172A", lw=0.5, label="Full Spectral-CG (A+B+C+D)")
    ax1.bar(x, modA_v, w, color="#DC2626", edgecolor="#0F172A", lw=0.5, label="Module A Only (Covariance)")
    ax1.bar(x + w, modCD_v, w, color="#059669", edgecolor="#0F172A", lw=0.5, label="Modules C+D Defense (LMM+DFT)")
    for i in range(len(keys)):
        ax1.text(x[i] + w, modCD_v[i] + 1.2, f"{modCD_v[i]:.1f}", ha="center", fontsize=6.5, fontweight="bold", color="#059669")
        ax1.text(x[i], modA_v[i] + 1.2, f"{modA_v[i]:.1f}", ha="center", fontsize=6.3, color="#DC2626")
    ax1.set_xticks(x); ax1.set_xticklabels(short, fontsize=7.0); ax1.set_ylim(30, 112)
    ax1.set_ylabel("ROC-AUC (%)"); ax1.set_title("(a) White-Box Adaptive Attacker vs. Multi-Module Defense")
    ax1.grid(True, axis="y", ls=":", alpha=0.5); ax1.legend(loc="lower left", fontsize=6.2, frameon=True)

    cd = data["cross_domain"]
    cd_lbls = ["In-Domain\nS2 L1C (13B)", "Cross-Atm.\nS2 L2A (13B)", "Cross-Sensor\nLandsat-8 (7B)"]
    cd_keys = list(cd.keys())
    cd_auc = [cd[k]["roc_auc"] * 100 for k in cd_keys]
    cd_acc = [cd[k]["accuracy"] * 100 for k in cd_keys]
    x2 = np.arange(len(cd_keys)); w2 = 0.34
    b1 = ax2.bar(x2 - w2/2, cd_auc, w2, color="#1B365D", edgecolor="#0F172A", lw=0.5, label="ROC-AUC (%)")
    b2 = ax2.bar(x2 + w2/2, cd_acc, w2, color="#1F6F78", edgecolor="#0F172A", lw=0.5, label="Accuracy (%)")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height(); ax2.text(bar.get_x() + bar.get_width()/2, h + 1.0, f"{h:.1f}", ha="center", fontsize=6.7)
    ax2.set_xticks(x2); ax2.set_xticklabels(cd_lbls, fontsize=7.0); ax2.set_ylim(50, 110)
    ax2.set_ylabel("Score (%)"); ax2.set_title("(b) Cross-Atmospheric & Cross-Sensor Generalization")
    ax2.grid(True, axis="y", ls=":", alpha=0.5); ax2.legend(loc="lower left", fontsize=6.6, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig5_adaptive_crossdomain.png")); plt.savefig(os.path.join(out_dir, "fig5_adaptive_crossdomain.pdf")); plt.close(fig)


def draw_fig6_sensitivity_robustness(data: dict, out_dir: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.5), gridspec_kw={"width_ratios": [1.0, 1.25]})
    ls = data["sensitivity"]["lambda_sweep"]
    ax1.semilogx([x["lambda"] for x in ls], [x["roc_auc"] for x in ls], "o-", color="#1B365D", lw=1.8, ms=5)
    ax1.set_xlabel(r"L2 Regularization $\lambda$ (log)"); ax1.set_ylabel("ROC-AUC", color="#1B365D"); ax1.set_ylim(0.95, 1.01); ax1.grid(True, ls=":", alpha=0.5)
    ax1b = ax1.twinx()
    ax1b.semilogx([x["lambda"] for x in ls], [x["iters"] for x in ls], "s--", color="#D97706", lw=1.5, ms=4.5)
    ax1b.set_ylabel("HZ-CG Iterations", color="#D97706")
    ax1.set_title(r"(a) Sensitivity to Regularization $\lambda$")

    rb = data["robustness"]
    r_keys = [("Clean (No Perturbation)", "Clean"), ("Gaussian Noise (SNR 40 dB)", "SNR 40dB"), ("8-Bit Lossy Quantization", "8-Bit Quant"),
              ("Thin Cirrus Cloud (OD=0.10)", "Cirrus Thin"), ("2x Spatial Down/Upsampling", "2x Down/Up"), ("Moderate Cirrus Cloud (OD=0.25)", "Cirrus Mod"),
              ("Gaussian Noise (SNR 30 dB)", "SNR 30dB"), ("Gaussian Noise (SNR 20 dB)", "SNR 20dB")]
    r_lbls = [lbl for k, lbl in r_keys if k in rb]; r_vals = [rb[k]["roc_auc"] * 100.0 for k, _ in r_keys if k in rb]
    cols = ["#1B365D" if v >= 95 else ("#1F6F78" if v >= 85 else "#D97706") for v in r_vals]
    yp = np.arange(len(r_lbls))[::-1]
    ax2.barh(yp, r_vals, color=cols, edgecolor="#1E293B", lw=0.5, height=0.65)
    for y0, v in zip(yp, r_vals): ax2.text(v + 0.8, y0, f"{v:.1f}%", va="center", fontsize=6.7)
    ax2.set_yticks(yp); ax2.set_yticklabels(r_lbls, fontsize=7.0); ax2.set_xlim(40, 109)
    ax2.set_xlabel("ROC-AUC (%)"); ax2.set_title("(b) Robustness Under Sensor Noise, Resampling & Cirrus Haze"); ax2.grid(True, axis="x", ls=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig6_sensitivity_robustness.png")); plt.savefig(os.path.join(out_dir, "fig6_sensitivity_robustness.pdf")); plt.close(fig)


def draw_fig7_ablation_importance(data: dict, out_dir: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ba = data["band_ablation"]
    labels = ["RGB\n(3 B)", "RGB+NIR\n(4 B)", "RGB+NIR\n+RE (8 B)", "Full MSI\n(13 B)"]
    keys = ["RGB Only (3 Bands)", "RGB + NIR (4 Bands)", "RGB + NIR + Red-Edge (8 Bands)", "Full Sentinel-2 MSI (13 Bands)"]
    aucs = [ba[k]["roc_auc"] * 100 for k in keys]; f1s = [ba[k]["f1"] * 100 for k in keys]
    x = np.arange(len(labels)); w = 0.35
    b1 = ax1.bar(x - w/2, aucs, w, color="#1B365D", edgecolor="#0F172A", lw=0.5, label="ROC-AUC (%)")
    b2 = ax1.bar(x + w/2, f1s, w, color="#1F6F78", edgecolor="#0F172A", lw=0.5, label="F1-Score (%)")
    for bars in (b1, b2):
        for bar in bars: h = bar.get_height(); ax1.text(bar.get_x() + bar.get_width()/2, h + 1.0, f"{h:.1f}", ha="center", fontsize=6.6)
    ax1.set_xticks(x); ax1.set_xticklabels(labels); ax1.set_ylim(40, 110); ax1.set_ylabel("Score (%)")
    ax1.set_title("(a) Spectral Band Ablation on 1,000 Real EuroSAT-MS"); ax1.grid(True, axis="y", ls=":", alpha=0.5); ax1.legend(loc="upper left", fontsize=6.6)

    top12 = data["feature_importance"]["top_15"][:12][::-1]
    mcol = {"A": "#1B365D", "B": "#1F6F78", "C": "#D97706", "D": "#7C3AED"}
    y2 = np.arange(len(top12))
    ax2.barh(y2, [i["abs_weight"] for i in top12], color=[mcol[i["module"]] for i in top12], edgecolor="#1E293B", lw=0.5, height=0.68)
    ax2.set_yticks(y2); ax2.set_yticklabels([i["feature"] for i in top12], fontsize=6.5)
    ax2.set_xlabel(r"Learned Weight Magnitude $|w_j^*|$"); ax2.set_title("(b) Top-12 Forensic Feature Weights (72-D Operator)")
    ax2.grid(True, axis="x", ls=":", alpha=0.5)
    ax2.legend(handles=[patches.Patch(color=mcol[m], label=f"Mod {m}") for m in ["A", "B", "C", "D"]], loc="lower right", fontsize=6.1, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig7_ablation_importance.png")); plt.savefig(os.path.join(out_dir, "fig7_ablation_importance.pdf")); plt.close(fig)


def draw_fig8_localization(data: dict, out_dir: str) -> None:
    sp = data["localization"]["sample_spliced"]; scenes = data["localization"]["per_scene"]
    fig = plt.figure(figsize=(9.8, 3.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.15, 1.45], wspace=0.28)
    ax0 = fig.add_subplot(gs[0, 0]); ax0.imshow(np.array(sp["rgb"])); ax0.set_title("(a) Spliced Real EuroSAT\nRGB (Visually Seamless)", fontsize=7.6); ax0.set_xticks([]); ax0.set_yticks([])
    ax1 = fig.add_subplot(gs[0, 1]); ax1.imshow(np.array(sp["ground_truth_mask"]), cmap="gray"); ax1.set_title("(b) Ground-Truth\nForgery Mask", fontsize=7.6); ax1.set_xticks([]); ax1.set_yticks([])
    ax2 = fig.add_subplot(gs[0, 2]); im = ax2.imshow(np.array(sp["anomaly_heatmap"]), cmap="inferno", vmin=0.0, vmax=1.0); ax2.set_title(f"(c) Multi-Scale Heatmap\n(Pixel AUC = {sp['pixel_auc']:.3f})", fontsize=7.6); ax2.set_xticks([]); ax2.set_yticks([])
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6.3)
    ax3 = fig.add_subplot(gs[0, 3])
    s_names = [s["class"] for s in scenes]; s_aucs = [s["pixel_auc"] * 100.0 for s in scenes]; yp = np.arange(len(s_names))[::-1]
    ax3.barh(yp, s_aucs, color="#1B365D", edgecolor="#0F172A", lw=0.5, height=0.6)
    for y0, v in zip(yp, s_aucs): ax3.text(v + 0.8, y0, f"{v:.1f}%", va="center", fontsize=6.7)
    ax3.axvline(data["localization"]["mean_pixel_auc"] * 100.0, color="#DC2626", ls="--", lw=1.1, label=f"Mean ({data['localization']['mean_pixel_auc']*100:.1f}%)")
    ax3.set_yticks(yp); ax3.set_yticklabels(s_names, fontsize=7.0); ax3.set_xlim(65, 108); ax3.set_xlabel("Pixel ROC-AUC (%)"); ax3.set_title("(d) Per-Scene Localization AUC", fontsize=7.6)
    ax3.grid(True, axis="x", ls=":", alpha=0.5); ax3.legend(loc="lower right", fontsize=6.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig8_localization.png")); plt.savefig(os.path.join(out_dir, "fig8_localization.pdf")); plt.close(fig)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base_dir, "results", "experiment_results.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    fig_dir = os.path.join(base_dir, "figures"); os.makedirs(fig_dir, exist_ok=True)
    draw_fig1_architecture(fig_dir)
    draw_fig2_spectral_covariance(data, fig_dir)
    draw_fig3_convergence(data, fig_dir)
    draw_fig4_roc_calibration(data, fig_dir)
    draw_fig5_adaptive_crossdomain(data, fig_dir)
    draw_fig6_sensitivity_robustness(data, fig_dir)
    draw_fig7_ablation_importance(data, fig_dir)
    draw_fig8_localization(data, fig_dir)
    print(f"Generated all 8 Top-Tier Q1 publication figures (PNG + PDF) in {fig_dir}")


if __name__ == "__main__":
    main()
