"""
cg_optimizer.py
Q1-Grade Nonlinear Conjugate Gradient Optimization Suite & Statistical Testing.

Includes:
  - RegularizedLogisticObjective (mu-strongly convex, L-smooth)
  - RegularizedHuberHingeObjective (smooth robust loss for outlier/cloud resilience)
  - Strong Wolfe Zoom Line Search (Nocedal & Wright)
  - Nonlinear CG solvers: GD, FR, PRP+, DY, HZ (CG_DESCENT), and Diagonal-Preconditioned PCG-HZ
  - Nystrom RBF-Kernelized Nonlinear CG (K-CG) at O(m) memory
  - SciPy L-BFGS benchmark wrapper
  - DeLong's Paired ROC-AUC Significance Test (z-stat and two-sided p-value)
"""

from __future__ import annotations
import time
from typing import Dict, List, Tuple
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


class RegularizedLogisticObjective:
    """Strictly convex L2-regularized binary logistic objective."""

    def __init__(self, Phi: np.ndarray, y: np.ndarray, lam: float = 1e-3):
        self.N, self.d = Phi.shape
        self.Phi_aug = np.hstack([Phi.astype(np.float64), np.ones((self.N, 1), dtype=np.float64)])
        self.y = y.astype(np.float64)
        self.lam = float(lam)
        self.n_feval = 0
        self.n_geval = 0

    def reset_counts(self) -> None:
        self.n_feval = 0
        self.n_geval = 0

    def value(self, w: np.ndarray) -> float:
        self.n_feval += 1
        margins = self.y * (self.Phi_aug @ w)
        return float(np.mean(np.logaddexp(0.0, -margins))) + 0.5 * self.lam * float(np.dot(w, w))

    def gradient(self, w: np.ndarray) -> np.ndarray:
        self.n_geval += 1
        margins = self.y * (self.Phi_aug @ w)
        probs_neg = _sigmoid(-margins)
        return -(self.Phi_aug.T @ (self.y * probs_neg)) / self.N + self.lam * w

    def value_and_gradient(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        self.n_feval += 1
        self.n_geval += 1
        margins = self.y * (self.Phi_aug @ w)
        loss = float(np.mean(np.logaddexp(0.0, -margins))) + 0.5 * self.lam * float(np.dot(w, w))
        probs_neg = _sigmoid(-margins)
        grad = -(self.Phi_aug.T @ (self.y * probs_neg)) / self.N + self.lam * w
        return loss, grad

    def hessian(self, w: np.ndarray) -> np.ndarray:
        margins = self.y * (self.Phi_aug @ w)
        s = _sigmoid(margins)
        D = s * (1.0 - s)
        return (self.Phi_aug.T * D[None, :]) @ self.Phi_aug / self.N + self.lam * np.eye(self.d + 1, dtype=np.float64)

    def diag_hessian(self, w: np.ndarray) -> np.ndarray:
        margins = self.y * (self.Phi_aug @ w)
        s = _sigmoid(margins)
        D = s * (1.0 - s)
        return np.sum((self.Phi_aug ** 2) * D[:, None], axis=0) / self.N + self.lam

    def conditioning_stats(self, w: np.ndarray) -> Dict[str, float]:
        H = self.hessian(w)
        eigs = np.linalg.eigvalsh(H)
        mu_exact, L_exact = float(eigs[0]), float(eigs[-1])
        sv_max = float(np.linalg.svd(self.Phi_aug, compute_uv=False)[0])
        L_bound = (sv_max ** 2) / (4.0 * self.N) + self.lam
        return {
            "mu_exact": mu_exact,
            "L_exact": L_exact,
            "kappa_exact": L_exact / max(mu_exact, 1e-15),
            "mu_bound": self.lam,
            "L_bound": L_bound,
            "kappa_bound": L_bound / max(self.lam, 1e-15),
        }

    def verify_gradient(self, w: np.ndarray, eps: float = 1e-6) -> float:
        g_ana = self.gradient(w)
        g_num = np.zeros_like(w)
        for j in range(len(w)):
            e_j = np.zeros_like(w)
            e_j[j] = eps
            g_num[j] = (self.value(w + e_j) - self.value(w - e_j)) / (2.0 * eps)
        return float(np.linalg.norm(g_ana - g_num) / max(np.linalg.norm(g_ana), 1e-12))

    def predict_proba(self, w: np.ndarray, Phi_test: np.ndarray) -> np.ndarray:
        Phi_aug = np.hstack([Phi_test.astype(np.float64), np.ones((Phi_test.shape[0], 1), dtype=np.float64)])
        return _sigmoid(Phi_aug @ w)


class RegularizedHuberHingeObjective:
    """
    Smooth L2-regularized Huber-Hinge loss (robust to cloud/sensor outliers,
    preserving mu-strong convexity and L-Lipschitz continuous gradients):
      h(m) = 0                         if m >= 1
           = (1 - m)^2 / (2 * delta)   if 1 - delta < m < 1
           = 1 - m - delta / 2         if m <= 1 - delta
    """

    def __init__(self, Phi: np.ndarray, y: np.ndarray, lam: float = 1e-3, delta: float = 0.5):
        self.N, self.d = Phi.shape
        self.Phi_aug = np.hstack([Phi.astype(np.float64), np.ones((self.N, 1), dtype=np.float64)])
        self.y = y.astype(np.float64)
        self.lam = float(lam)
        self.delta = float(delta)
        self.n_feval = 0
        self.n_geval = 0

    def reset_counts(self) -> None:
        self.n_feval = 0
        self.n_geval = 0

    def value_and_gradient(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        self.n_feval += 1
        self.n_geval += 1
        m = self.y * (self.Phi_aug @ w)
        d = self.delta
        loss_vec = np.where(
            m >= 1.0,
            0.0,
            np.where(m > 1.0 - d, ((1.0 - m) ** 2) / (2.0 * d), 1.0 - m - 0.5 * d),
        )
        dloss_dm = np.where(
            m >= 1.0,
            0.0,
            np.where(m > 1.0 - d, -(1.0 - m) / d, -1.0),
        )
        loss = float(np.mean(loss_vec)) + 0.5 * self.lam * float(np.dot(w, w))
        grad = (self.Phi_aug.T @ (self.y * dloss_dm)) / self.N + self.lam * w
        return loss, grad

    def diag_hessian(self, w: np.ndarray) -> np.ndarray:
        m = self.y * (self.Phi_aug @ w)
        d2 = np.where((m > 1.0 - self.delta) & (m < 1.0), 1.0 / self.delta, 0.0)
        return np.sum((self.Phi_aug ** 2) * d2[:, None], axis=0) / self.N + self.lam

    def predict_proba(self, w: np.ndarray, Phi_test: np.ndarray) -> np.ndarray:
        Phi_aug = np.hstack([Phi_test.astype(np.float64), np.ones((Phi_test.shape[0], 1), dtype=np.float64)])
        return _sigmoid(Phi_aug @ w)


def strong_wolfe_line_search(
    obj,
    w: np.ndarray,
    d: np.ndarray,
    f0: float,
    g0: np.ndarray,
    c1: float = 1e-4,
    c2: float = 0.1,
    alpha_init: float = 1.0,
    alpha_max: float = 100.0,
    max_ls: int = 30,
) -> Tuple[float, float, np.ndarray]:
    """Strong Wolfe Zoom Line Search (Nocedal & Wright Algorithms 3.5 & 3.6)."""
    phi0 = f0
    dphi0 = float(np.dot(g0, d))
    if dphi0 >= 0.0:
        return 1e-4, f0, g0

    def zoom(a_lo: float, a_hi: float, phi_lo: float) -> Tuple[float, float, np.ndarray]:
        for _ in range(max_ls):
            a_j = 0.5 * (a_lo + a_hi)
            f_j, g_j = obj.value_and_gradient(w + a_j * d)
            dphi_j = float(np.dot(g_j, d))
            if (f_j > phi0 + c1 * a_j * dphi0) or (f_j >= phi_lo):
                a_hi = a_j
            else:
                if abs(dphi_j) <= -c2 * dphi0:
                    return a_j, f_j, g_j
                if dphi_j * (a_hi - a_lo) >= 0.0:
                    a_hi = a_lo
                a_lo = a_j
                phi_lo = f_j
            if abs(a_hi - a_lo) < 1e-12:
                break
        f_j, g_j = obj.value_and_gradient(w + a_lo * d)
        return a_lo, f_j, g_j

    a_prev = 0.0
    phi_prev = phi0
    a_curr = max(min(alpha_init, alpha_max), 1e-8)

    for i in range(max_ls):
        f_curr, g_curr = obj.value_and_gradient(w + a_curr * d)
        dphi_curr = float(np.dot(g_curr, d))
        if (f_curr > phi0 + c1 * a_curr * dphi0) or (i > 0 and f_curr >= phi_prev):
            return zoom(a_prev, a_curr, phi_prev)
        if abs(dphi_curr) <= -c2 * dphi0:
            return a_curr, f_curr, g_curr
        if dphi_curr >= 0.0:
            return zoom(a_curr, a_prev, f_curr)
        a_prev = a_curr
        phi_prev = f_curr
        a_curr = min(2.0 * a_curr, alpha_max)

    return a_curr, f_curr, g_curr


def solve_nonlinear_cg(
    obj,
    method: str = "HZ",
    tol: float = 1e-6,
    max_iter: int = 400,
    c1: float = 1e-4,
    c2: float = 0.1,
    eta: float = 0.01,
    powell_restart: float = 0.5,
) -> Dict[str, object]:
    """
    Solve min_w J(w) via GD, FR, PRP+, DY, HZ (CG_DESCENT), or PCG-HZ
    (Diagonal-Preconditioned Hager-Zhang CG) with Strong Wolfe Zoom Line Search.
    """
    obj.reset_counts()
    t0 = time.perf_counter()

    w = np.zeros(obj.d + 1, dtype=np.float64)
    f_k, g_k = obj.value_and_gradient(w)

    precond = (method == "PCG-HZ")
    inv_P = (1.0 / np.maximum(obj.diag_hessian(w), 1e-6)) if precond else np.ones_like(w)
    z_k = inv_P * g_k
    d_k = -z_k.copy()

    f_hist: List[float] = [f_k]
    g_hist: List[float] = [float(np.linalg.norm(g_k))]

    alpha_prev = 1.0
    dphi_prev = float(np.dot(g_k, d_k))

    k = 0
    while g_hist[-1] > tol and k < max_iter:
        dphi_curr = float(np.dot(g_k, d_k))
        if k == 0:
            alpha_init = min(1.0, 1.0 / max(g_hist[-1], 1.0))
        else:
            alpha_init = np.clip(alpha_prev * (dphi_prev / min(dphi_curr, -1e-18)), 1e-4, 25.0)

        alpha_k, f_next, g_next = strong_wolfe_line_search(
            obj, w, d_k, f_k, g_k, c1=c1, c2=c2, alpha_init=alpha_init
        )
        w = w + alpha_k * d_k
        alpha_prev = alpha_k
        dphi_prev = dphi_curr

        g_next_norm = float(np.linalg.norm(g_next))
        f_hist.append(f_next)
        g_hist.append(g_next_norm)
        k += 1

        if g_next_norm <= tol:
            f_k, g_k = f_next, g_next
            break

        z_next = inv_P * g_next

        if method == "GD":
            beta_k = 0.0
        else:
            y_k = g_next - g_k
            denom = float(np.dot(d_k, y_k))
            g_k_sq = float(np.dot(g_k, z_k))
            g_next_sq = float(np.dot(g_next, z_next))

            need_restart = (
                abs(denom) < 1e-14
                or (abs(float(np.dot(g_next, z_k))) / max(g_next_sq, 1e-18) >= powell_restart)
                or (k % (obj.d + 1) == 0)
            )
            if need_restart:
                beta_k = 0.0
            elif method == "FR":
                beta_k = g_next_sq / max(g_k_sq, 1e-18)
            elif method == "PRP+":
                beta_k = max(0.0, float(np.dot(z_next, y_k)) / max(g_k_sq, 1e-18))
            elif method == "DY":
                beta_k = g_next_sq / denom
            elif method in ("HZ", "PCG-HZ"):
                Py_k = inv_P * y_k
                yPy = float(np.dot(y_k, Py_k))
                beta_N = float(np.dot(Py_k - 2.0 * d_k * (yPy / denom), g_next)) / denom
                eta_k = -1.0 / (max(float(np.linalg.norm(d_k)), 1e-14) * min(eta, float(np.linalg.norm(g_k))))
                beta_k = max(beta_N, eta_k)
            else:
                raise ValueError(f"Unknown CG method: {method}")

        d_next = -z_next + beta_k * d_k
        if float(np.dot(g_next, d_next)) >= 0.0:
            d_next = -z_next

        f_k, g_k, z_k, d_k = f_next, g_next, z_next, d_next

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "method": method,
        "w": w,
        "iterations": k,
        "wall_time_ms": elapsed_ms,
        "n_feval": obj.n_feval,
        "n_geval": obj.n_geval,
        "final_loss": f_hist[-1],
        "final_grad_norm": g_hist[-1],
        "converged": g_hist[-1] <= tol,
        "loss_history": f_hist,
        "grad_history": g_hist,
    }


def solve_lbfgs_benchmark(
    obj: RegularizedLogisticObjective,
    tol: float = 1e-6,
    max_iter: int = 400,
) -> Dict[str, object]:
    """Benchmark SciPy L-BFGS-B on the exact same regularized logistic objective."""
    obj.reset_counts()
    t0 = time.perf_counter()
    f_hist, g_hist = [], []

    w0 = np.zeros(obj.d + 1, dtype=np.float64)
    f0, g0 = obj.value_and_gradient(w0)
    f_hist.append(f0)
    g_hist.append(float(np.linalg.norm(g0)))

    def cb(wk: np.ndarray) -> None:
        fk, gk = obj.value_and_gradient(wk)
        f_hist.append(fk)
        g_hist.append(float(np.linalg.norm(gk)))

    res = minimize(
        obj.value_and_gradient,
        w0,
        method="L-BFGS-B",
        jac=True,
        callback=cb,
        options={"gtol": tol, "ftol": 1e-15, "maxiter": max_iter},
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "method": "L-BFGS",
        "w": res.x,
        "iterations": int(res.nit),
        "wall_time_ms": elapsed_ms,
        "n_feval": obj.n_feval,
        "n_geval": obj.n_geval,
        "final_loss": float(res.fun),
        "final_grad_norm": float(np.linalg.norm(res.jac)),
        "converged": float(np.linalg.norm(res.jac)) <= tol,
        "loss_history": f_hist,
        "grad_history": g_hist,
    }


def nystrom_rbf_transform(
    Phi_tr: np.ndarray,
    Phi_te: np.ndarray,
    n_landmarks: int = 96,
    gamma: float | None = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Nystrom RBF kernel approximation mapping R^d -> R^m for Kernelized Nonlinear CG (K-CG)."""
    rng = np.random.default_rng(seed)
    m = min(n_landmarks, Phi_tr.shape[0])
    idx = rng.choice(Phi_tr.shape[0], size=m, replace=False)
    L = Phi_tr[idx]
    if gamma is None:
        gamma = 1.0 / Phi_tr.shape[1]

    def rbf(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        sq = np.sum(A ** 2, axis=1, keepdims=True) + np.sum(B ** 2, axis=1)[None, :] - 2.0 * (A @ B.T)
        return np.exp(-gamma * np.maximum(sq, 0.0))

    K_mm = rbf(L, L) + 1e-6 * np.eye(m)
    eigvals, eigvecs = np.linalg.eigh(K_mm)
    inv_sqrt = (eigvecs * (1.0 / np.sqrt(np.maximum(eigvals, 1e-8)))[None, :]) @ eigvecs.T
    Z_tr = rbf(Phi_tr, L) @ inv_sqrt
    Z_te = rbf(Phi_te, L) @ inv_sqrt
    return Z_tr, Z_te


def delong_roc_test(y_true_bin: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, float]:
    """
    Paired DeLong's test comparing ROC-AUC of classifier A vs. classifier B on identical test labels.
    Returns auc_a, auc_b, z_stat, and two-sided p_value.
    """
    y = np.asarray(y_true_bin, dtype=int)
    pos = (y == 1)
    neg = (y == 0)
    m, n = int(np.sum(pos)), int(np.sum(neg))

    def _structural(scores: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        sp = scores[pos][:, None]  # (m, 1)
        sn = scores[neg][None, :]  # (1, n)
        psi = (sp > sn).astype(np.float64) + 0.5 * (sp == sn).astype(np.float64)
        v10 = np.mean(psi, axis=1)  # (m,)
        v01 = np.mean(psi, axis=0)  # (n,)
        auc = float(np.mean(v10))
        return auc, v10, v01

    auc_a, v10_a, v01_a = _structural(np.asarray(pred_a, dtype=np.float64))
    auc_b, v10_b, v01_b = _structural(np.asarray(pred_b, dtype=np.float64))

    s10_aa = np.var(v10_a, ddof=1)
    s10_bb = np.var(v10_b, ddof=1)
    s10_ab = np.cov(v10_a, v10_b, ddof=1)[0, 1]

    s01_aa = np.var(v01_a, ddof=1)
    s01_bb = np.var(v01_b, ddof=1)
    s01_ab = np.cov(v01_a, v01_b, ddof=1)[0, 1]

    var_diff = (s10_aa + s10_bb - 2.0 * s10_ab) / m + (s01_aa + s01_bb - 2.0 * s01_ab) / n
    if var_diff <= 1e-15:
        z_stat = 0.0
        p_val = 1.0
    else:
        z_stat = float((auc_a - auc_b) / np.sqrt(var_diff))
        p_val = float(2.0 * (1.0 - norm.cdf(abs(z_stat))))

    return {"auc_a": auc_a, "auc_b": auc_b, "z_stat": z_stat, "p_value": p_val}
