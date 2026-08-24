"""Step 2 任务 D/E — evidence 变量(二维 safety + 一维 efficiency,C1)。

Closing rate 符号(写死,勿改):
  c(t) = v_O(t) - v_E(t) = -d(gap)/dt
  c > 0  <=>  opponent 正在接近 ego(closing)。
  (canonical 纵轴下 gap = x_E - x_O 的 bumper 距,ego 在前、opponent 在后;
  rollout 表的 dv_Y/dv_H 列即为 c。)

Safety evidence(二维 trajectory functionals):
  g_tilde = smin_{rho_g, t}[ gap(t) - d_0 - h_ref * v_O(t) ]
            (扣除静止间距 d_0 与速度相关时距 h_ref*v_O 后的软最小,米)
  c_tilde = smax_{rho_c, t}[ c(t) ]
            (全程最不利 closing tendency 的软最大,m/s)
  m_s = g_tilde - c_+^2 / (2 b_ref),c_+ = softplus(c_tilde; beta)
  物理解释:actual gap - standstill clearance - speed-dependent headway
            - closing-distance requirement。
Efficiency evidence(一维):
  z_e = p_O = 分支 rollout progress = ∫ v_O dt(米,梯形积分);
  reference progress = v_ref * horizon,v_ref 为外生 site-level
  reference speed(逐 site pilot population P85;非 free-flow speed)。

符号约定:h_ref = 参考时距(s);rho_g = softmin 温度(m);
rho_c = softmax 温度(m/s)。参数全部来自 config evidence 段。
Raw TTC 仅诊断,不进 utility。
"""
from __future__ import annotations

import numpy as np


def softplus(x, beta: float):
    """(1/beta) log(1 + exp(beta x)),数值稳定;max(0,x) 的 C1 替代。"""
    return np.logaddexp(0.0, beta * np.asarray(x, dtype=float)) / beta


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def softmin_norm(m: np.ndarray, rho: float) -> float:
    """归一化软最小:-rho * log(mean exp(-m/rho));min <= smin <= mean。"""
    m = np.asarray(m, dtype=float)
    z = -m / rho
    zmax = z.max()
    return float(-rho * (zmax + np.log(np.mean(np.exp(z - zmax)))))


def softmax_norm(c: np.ndarray, rho: float) -> float:
    """归一化软最大:rho * log(mean exp(c/rho));mean <= smax <= max。"""
    return -softmin_norm(-np.asarray(c, dtype=float), rho)


def branch_evidence(gap: np.ndarray, v_opp: np.ndarray, c: np.ndarray,
                    cfg_e: dict, dt: float) -> dict:
    """单分支轨迹 -> evidence {g_tilde[m], c_tilde[m/s], z_e[m]}。
    c 必须按 c = v_O - v_E(正 = closing)传入。"""
    reserve = (np.asarray(gap, dtype=float)
               - cfg_e["d0_m"]
               - cfg_e["h_ref_s"] * np.asarray(v_opp, dtype=float))
    v = np.asarray(v_opp, dtype=float)
    return {
        "g_tilde": softmin_norm(reserve, cfg_e["rho_g_m"]),
        "c_tilde": softmax_norm(c, cfg_e["rho_c_mps"]),
        "z_e": float(np.sum(0.5 * (v[1:] + v[:-1]) * dt)),
    }


def safety_margin_from_evidence(g_tilde: float, c_tilde: float,
                                cfg_e: dict) -> float:
    """m_s = g_tilde - c_+^2/(2 b_ref)。"""
    c_plus = float(softplus(c_tilde, cfg_e["softplus_beta"]))
    return g_tilde - c_plus ** 2 / (2.0 * cfg_e["b_ref_mps2"])


def diagnostic_ttc(gap0: float, c0: float, cfg_e: dict, cap: float = 60.0) -> float:
    """诊断用 TTC(不进 utility):gap / softplus(c),封顶 cap。"""
    return float(min(gap0 / max(softplus(c0, cfg_e["softplus_beta"]), 1e-6), cap))