"""
cg_unmixing.py
Linear Conjugate Gradient (Hestenes-Stiefel) Spectral Unmixing & Probability Simplex Projection.

Solves the L2-regularized Linear Mixture Model (LMM) normal equations:
    (M^T M + lambda_u I) a_p = M^T y_p
using from-scratch Hestenes-Stiefel Linear CG with optional Jacobi diagonal preconditioning,
followed by exact Euclidean projection onto the probability simplex Delta^{K-1}
(Duchi et al., O(K log K)) enforcing ANC (a_p >= 0) and ASC (1^T a_p = 1).
"""

from __future__ import annotations
from typing import Tuple
import numpy as np


def extract_endmembers_vca(
    authentic_cubes: np.ndarray,
    n_endmembers: int = 6,
    max_pixels: int = 12000,
    seed: int = 42,
) -> np.ndarray:
    """
    Extract K pure physical endmembers M in R^{B x K} from authentic training cubes
    using Vertex Component Analysis (VCA) / Orthogonal Subspace Projection.
    """
    rng = np.random.default_rng(seed)
    n, h, w, b = authentic_cubes.shape
    Y_all = authentic_cubes.reshape(-1, b).T  # (B, N_total)
    total_px = Y_all.shape[1]
    if total_px > max_pixels:
        idx = rng.choice(total_px, size=max_pixels, replace=False)
        Y = Y_all[:, idx].astype(np.float64)
    else:
        Y = Y_all.astype(np.float64)

    # Project onto top K SVD subspace for numerical stability
    mean_y = np.mean(Y, axis=1, keepdims=True)
    Y_centered = Y - mean_y
    U, _, _ = np.linalg.svd(Y_centered, full_matrices=False)
    Ud = U[:, :n_endmembers]
    Y_proj = Ud.T @ Y  # (K, P)

    # Iterative extreme simplex vertex selection (VCA core)
    A = np.zeros((n_endmembers, n_endmembers), dtype=np.float64)
    A[-1, 0] = 1.0
    indices = np.zeros(n_endmembers, dtype=np.int64)

    for i in range(n_endmembers):
        w_vec = rng.standard_normal((n_endmembers, 1))
        # Project orthogonal to previously selected vertices
        if i > 0:
            pinv = np.linalg.pinv(A[:, :i])
            f = w_vec - A[:, :i] @ (pinv @ w_vec)
        else:
            f = w_vec
        norm_f = np.linalg.norm(f)
        if norm_f > 1e-12:
            f /= norm_f
        v = np.abs((f.T @ Y_proj).ravel())
        idx_max = int(np.argmax(v))
        indices[i] = idx_max
        A[:, i] = Y_proj[:, idx_max]

    M = Y[:, indices]  # (B, K)
    return M


def project_simplex_batch(V: np.ndarray) -> np.ndarray:
    """
    Exact Euclidean projection of each column of V in R^{K x P} onto the probability simplex
    Delta^{K-1} = {a >= 0, 1^T a = 1} using Duchi et al. (2008) O(K log K) algorithm.
    """
    K, P = V.shape
    U = np.sort(V, axis=0)[::-1, :]  # Descending along K
    cssv = np.cumsum(U, axis=0) - 1.0
    ind = np.arange(1, K + 1, dtype=np.float64)[:, None]
    cond = U - cssv / ind > 0
    # Largest index rho where condition holds
    rho = np.sum(cond, axis=0) - 1
    rho = np.clip(rho, 0, K - 1)
    theta = cssv[rho, np.arange(P)] / (rho + 1.0)
    return np.maximum(V - theta[None, :], 0.0)


def linear_cg_unmix_batch(
    Y: np.ndarray,
    M: np.ndarray,
    lambda_u: float = 1e-3,
    tol: float = 1e-7,
    max_iter: int = 20,
    precondition: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Solve (M^T M + lambda_u I) A = M^T Y for all pixels simultaneously via
    Hestenes-Stiefel Linear Conjugate Gradient (with optional Jacobi preconditioning),
    then project onto the probability simplex Delta^{K-1}.

    Args:
        Y: (B, P) spectral observation matrix for P pixels across B bands.
        M: (B, K) physical endmember dictionary matrix.
        lambda_u: Tikhonov regularization parameter ensuring SPD matrix A_u.
        tol: Residual norm stopping tolerance.
        max_iter: Maximum CG iterations (theoretically converges in <= K steps).
        precondition: Whether to apply Jacobi diagonal preconditioning P = diag(A_u).

    Returns:
        A_proj: (K, P) simplex-constrained abundance vectors.
        iters: Number of CG iterations executed.
    """
    B, K = M.shape
    P = Y.shape[1]

    A_u = M.T @ M + lambda_u * np.eye(K, dtype=np.float64)  # (K, K) SPD
    B_u = M.T @ Y  # (K, P)

    # Initialize a_0 = 0 -> r_0 = b_u
    A_hat = np.zeros((K, P), dtype=np.float64)
    R = B_u.copy()

    if precondition:
        inv_diag = 1.0 / np.diag(A_u)[:, None]
        Z = inv_diag * R
    else:
        Z = R.copy()

    D = Z.copy()
    rz_old = np.sum(R * Z, axis=0)  # (P,)

    iters = 0
    for k in range(min(max_iter, K + 4)):
        iters = k + 1
        AD = A_u @ D  # (K, P)
        dAd = np.sum(D * AD, axis=0)
        alpha = rz_old / np.maximum(dAd, 1e-18)

        A_hat = A_hat + D * alpha[None, :]
        R = R - AD * alpha[None, :]

        if float(np.max(np.linalg.norm(R, axis=0))) < tol:
            break

        if precondition:
            Z = inv_diag * R
        else:
            Z = R

        rz_new = np.sum(R * Z, axis=0)
        beta = rz_new / np.maximum(rz_old, 1e-18)
        D = Z + D * beta[None, :]
        rz_old = rz_new

    # Simplex projection enforcing ANC (a >= 0) and ASC (1^T a = 1)
    A_proj = project_simplex_batch(A_hat)
    return A_proj, iters


def compute_unmixing_and_sam(
    cube: np.ndarray,
    M: np.ndarray,
    lambda_u: float = 1e-3,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute per-pixel LMM Euclidean unmixing residual r_p = ||y_p - M a_p||_2
    and Spectral Angle Mapper SAM(y_p, y_hat_p) in [0, pi/2].

    Returns:
        residuals: (H, W) Euclidean unmixing residual map.
        sam_map: (H, W) Spectral Angle Mapper (radians) map.
        cg_iters: Number of Linear CG iterations performed.
    """
    H, W, B = cube.shape
    Y = cube.reshape(-1, B).T.astype(np.float64)  # (B, P)
    A_proj, cg_iters = linear_cg_unmix_batch(Y, M, lambda_u=lambda_u)
    Y_hat = M @ A_proj  # (B, P)

    # Euclidean unmixing residual
    diff = Y - Y_hat
    residuals = np.linalg.norm(diff, axis=0).reshape(H, W)

    # Illumination-invariant Spectral Angle Mapper (SAM)
    dot = np.sum(Y * Y_hat, axis=0)
    norm_y = np.linalg.norm(Y, axis=0)
    norm_yhat = np.linalg.norm(Y_hat, axis=0)
    cos_theta = np.clip(dot / (norm_y * norm_yhat + eps), -1.0, 1.0)
    sam_map = np.arccos(cos_theta).reshape(H, W)

    return residuals, sam_map, cg_iters
