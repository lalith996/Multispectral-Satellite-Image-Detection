# AI-Generated Multispectral Satellite Image Detection via Spectral Consistency Analysis and Dual Conjugate Gradient Optimization

**Author:** Thunga Sai Geethika  
**Affiliation:** Department of Computer Science, Amrita Vishwa Vidyapeetham, Bengaluru, India  

## Overview
This repository contains the complete implementation, experimental pipeline, interactive forensic dashboard, figures, and IEEE manuscript for detecting AI-generated 13-band Sentinel-2 multispectral satellite imagery using physics-grounded spectral consistency features (Modules A–D) and Dual Conjugate Gradient optimization (Linear CG spectral unmixing + Preconditioned Hager–Zhang Nonlinear CG classification).

## Repository Structure
- `src/` — Complete Python pipeline:
  - `download_eurosat_ms.py` — Downloads 1,000 real Copernicus Sentinel-2 EuroSAT-MS 64×64×13 GeoTIFF tiles from [Zenodo (DOI: 10.5281/zenodo.7711810)](https://zenodo.org/records/7711810).
  - `data_loader.py` — Pure-Python 13-band GeoTIFF reader with wavelength-ordered Sentinel-2 bands (`B1–B8, B8A, B9–B12`).
  - `cg_unmixing.py` — Linear Conjugate Gradient spectral unmixing and SAM residual extraction.
  - `spectral_consistency.py` — Modules A–D (Riemannian SPD inter-band covariance geodesics, biophysical indices, linear CG unmixing, and 2D azimuthal DFT spectral-spatial features).
  - `cg_optimizer.py` — Nonlinear CG (Hager–Zhang, Polak–Ribière+, Fletcher–Reeves), Preconditioned CG, Huber–Hinge loss, and Nyström Kernel CG solvers.
  - `neural_generators.py` — Four independent PyTorch multispectral generators (`ConvAE`, `RGB→13B`, `DDPM-UNet`, `ResNet-GAN`) and published baseline detectors (`SFNet`, `ViT`, `Durall-DFT`, 13-band/RGB CNNs).
  - `run_experiments.py` — End-to-end 5-fold stratified CV, zero-shot cross-generator transfer, white-box adaptive attacks (WCT/NDVI), cross-atmospheric (L1C→L2A), cross-sensor (Landsat-8 OLI), calibration (ECE, TPR@1%FPR), and DeLong statistical tests.
  - `evaluate_and_visualize.py` — Generates publication-grade figures (`fig1`–`fig8` in PNG + vector PDF).
  - `build_ieee_report.py` — Compiles the two-column IEEE PDF report and Overleaf-ready `IEEEtran` `.tex` source.
  - `build_dashboard.py` — Builds the standalone interactive HTML/JS forensic dashboard.
- `dashboard/index.html` — Offline interactive forensic dashboard (open directly in any browser).
- `report/` — Compiled 6-page IEEE PDF (`IEEE_Multispectral_Satellite_Detection_Report.pdf`) and Overleaf LaTeX source (`IEEE_Multispectral_Satellite_Detection_Report.tex`).
- `figures/` — All 8 publication-ready figures (`fig1`–`fig8`, 300 DPI PNG and vector PDF).
- `results/experiment_results.json` — Full quantitative experimental outputs across all 1,000 EuroSAT-MS tiles.
- `data/EuroSAT_MS/` — Local dataset directory (excluded from Git via `.gitignore`; reproduce anytime with `python3 src/download_eurosat_ms.py`).

## Quick Start
```bash
pip install -r requirements.txt

# 1. Download 1,000 real EuroSAT-MS 13-band GeoTIFFs (if not already present)
python3 src/download_eurosat_ms.py

# 2. Run full experimental benchmark suite
python3 src/run_experiments.py

# 3. Regenerate figures, dashboard, and IEEE report
python3 src/evaluate_and_visualize.py
python3 src/build_dashboard.py
python3 src/build_ieee_report.py
```

Open `dashboard/index.html` in a web browser to explore the interactive Spectral Inspector, CG Optimizer Lab, Benchmarks & Ablations, and Anomaly Localization panels.
