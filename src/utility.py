"""Step 2 任务 D/F — 解析 utility 与对标准化 evidence 的完整梯度。

  u_s^raw = tanh(m_s / s_s),m_s = g_tilde - c_+^2/(2 b_ref)
  u_e^raw = tanh(m_e / s_e),m_e = z_e - v_ref * horizon
同构:u_k = tanh(physical margin_k / physical scale_k),margin=0 <-> u=0,
值域 (-1,1)。F-A(intrinsic physics-based scaling):T_k = identity。

标准化 evidence(任务 E):
  zeta_s = (g_tilde/a_g, c_tilde/a_c),zeta_e = (z_e - v_ref*T)/a_e
完整链式梯度(G_k 用;a_* 因子必须显式保留,即使当前 a_e/s_e = 1):
  du_s/d(zeta_g) = a_g * (1-u_s^2)/s_s
  du_s/d(zeta_c) = a_c * (-(1-u_s^2)/s_s) * c_+ * sigmoid(beta*c_tilde)/b_ref
  du_e/d(zeta_e) = (a_e/s_e) * (1-u_e^2)
L_k(a) = grad_zeta^T Sigma_k grad_zeta(Sigma 不写死为 I,矩阵乘实现)。
解析梯度与中心差分互检(tol 由 config utility.grad_check_tol)。
"""
from __future__ import annotations

import numpy as np

from src.evidence import safety_margin_from_evidence, sigmoid, softplus


# ----------------------------------------------------------------------
# utility 值
# ----------------------------------------------------------------------
def u_s_raw(g_tilde, c_tilde, cfg_e: dict, cfg_u: dict):
    m = safety_margin_from_evidence(float(g_tilde), float(c_tilde), cfg_e)
    return float(np.tanh(m / cfg_u["s_margin_m"]))


def u_e_raw(z_e, v_ref: float, horizon_s: float, cfg_u: dict):
    m_e = float(z_e) - v_ref * horizon_s
    return float(np.tanh(m_e / cfg_u["s_progress_m"]))


# ----------------------------------------------------------------------
# 对标准化 evidence zeta 的解析梯度(完整链式,a_* 显式)
# ----------------------------------------------------------------------
def grad_u_s_zeta(g_tilde: float, c_tilde: float, cfg_e: dict, cfg_u: dict,
                  cfg_p: dict) -> np.ndarray:
    """[du/d zeta_g, du/d zeta_c]。"""
    s_s = cfg_u["s_margin_m"]
    beta = cfg_e["softplus_beta"]
    b_ref = cfg_e["b_ref_mps2"]
    u = u_s_raw(g_tilde, c_tilde, cfg_e, cfg_u)
    c_plus = float(softplus(c_tilde, beta))
    du_dg = (1.0 - u ** 2) / s_s
    du_dc = -(1.0 - u ** 2) / s_s * c_plus * float(sigmoid(beta * c_tilde)) / b_ref
    return np.array([cfg_p["a_g_m"] * du_dg, cfg_p["a_c_mps"] * du_dc])


def grad_u_e_zeta(z_e: float, v_ref: float, horizon_s: float, cfg_u: dict,
                  cfg_p: dict) -> np.ndarray:
    """[du/d zeta_e];chain 因子 a_e/s_e 显式保留(勿用当前数值恰等删掉)。"""
    u = u_e_raw(z_e, v_ref, horizon_s, cfg_u)
    return np.array([(cfg_p["a_e_m"] / cfg_u["s_progress_m"]) * (1.0 - u ** 2)])


def L_quadratic(grad_zeta: np.ndarray, sigma: np.ndarray) -> float:
    """L = grad^T Sigma grad(矩阵形式,不写死 Sigma=I 的化简)。"""
    g = np.asarray(grad_zeta, dtype=float).reshape(-1)
    S = np.asarray(sigma, dtype=float)
    return float(g @ S @ g)


# ----------------------------------------------------------------------
# 数值互检
# ----------------------------------------------------------------------
def numeric_grad_s_zeta(g_tilde, c_tilde, cfg_e, cfg_u, cfg_p, h=1e-5):
    """对 zeta 的中心差分:zeta_g 扰动 h 即 g_tilde 扰动 a_g*h。"""
    a_g, a_c = cfg_p["a_g_m"], cfg_p["a_c_mps"]
    dg = (u_s_raw(g_tilde + a_g * h, c_tilde, cfg_e, cfg_u)
          - u_s_raw(g_tilde - a_g * h, c_tilde, cfg_e, cfg_u)) / (2 * h)
    dc = (u_s_raw(g_tilde, c_tilde + a_c * h, cfg_e, cfg_u)
          - u_s_raw(g_tilde, c_tilde - a_c * h, cfg_e, cfg_u)) / (2 * h)
    return np.array([dg, dc])


def numeric_grad_e_zeta(z_e, v_ref, horizon_s, cfg_u, cfg_p, h=1e-5):
    a_e = cfg_p["a_e_m"]
    de = (u_e_raw(z_e + a_e * h, v_ref, horizon_s, cfg_u)
          - u_e_raw(z_e - a_e * h, v_ref, horizon_s, cfg_u)) / (2 * h)
    return np.array([de])


def check_gradients(cfg_e: dict, cfg_u: dict, cfg_p: dict, v_ref: float,
                    horizon_s: float, g_grid, c_grid, ze_grid) -> float:
    """全网格解析 vs 数值(对 zeta)最大绝对误差。"""
    err = 0.0
    for g in g_grid:
        for c in c_grid:
            a = grad_u_s_zeta(g, c, cfg_e, cfg_u, cfg_p)
            n = numeric_grad_s_zeta(g, c, cfg_e, cfg_u, cfg_p)
            err = max(err, float(np.max(np.abs(a - n))))
    for z in ze_grid:
        a = grad_u_e_zeta(z, v_ref, horizon_s, cfg_u, cfg_p)
        n = numeric_grad_e_zeta(z, v_ref, horizon_s, cfg_u, cfg_p)
        err = max(err, float(np.max(np.abs(a - n))))
    return err