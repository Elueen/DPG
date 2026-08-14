"""Step 1 revision — 三个 stochastic specification 的 probit 随机响应(R-C1/C2)。

统一符号(revision 文档):theta_O = (w_s, kappa_s, kappa_e),
mu(c) = w_s*Δu_s(c) + w_e*Δu_e(c),w_e = 1 - w_s,
q_O(c; theta) = Phi(mu(c) / sigma_eff(c; theta))。

三个模型只在 sigma_eff 上不同:

M0a(legacy,第一轮 total-utility-scale additive noise):
    U(a) = sum_k w_k u_k(a) + sum_k eps_{k,a}/sqrt(kappa_k)
    sigma_eff_0a^2 = 2 * (1/kappa_s + 1/kappa_e)
    collapse 方向:1/kappa_s + 1/kappa_e = C(与 w、context 无关)

M0(component-evaluation additive noise):
    u~_k(a) = u_k(a) + eps_{k,a},  eps_{k,a} ~ N(0, 1/kappa_k)
    U(a) = sum_k w_k u~_k(a)
    sigma_eff_0^2 = 2 * (w_s^2/kappa_s + w_e^2/kappa_e)
    collapse 方向:固定 w_s 下 w_s^2/kappa_s + w_e^2/kappa_e = C
    (即使按正确的 component-evaluation 语义定义噪声,naive model
     仍无法分离 kappa_s, kappa_e —— structural no-go 依旧)

M1(perception-derived context-sensitive component uncertainty):
    z~_k = z_k + eta_k,  eta_k ~ N(0, Sigma_k/kappa_k),一阶 Taylor 下
    V_1(c) = w_s^2 G_s(c)/kappa_s + w_e^2 G_e(c)/kappa_e
    G_s(c)/G_e(c) 跨 context 变化时,kappa 分量以不同权重进入不同
    context 的方差 —— 这是打破 collapse 的候选机制(不预设成功)。

全部函数对 w_s / kappa_s / kappa_e 支持 numpy 广播(网格扫描直接用)。
q_m1_mc_full 为不做线性化的全非线性感知模拟(z 加噪后过真实 f_k),
用于校验一阶 Taylor 解析式的适用范围。
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from src.games.contexts import COMPONENTS, ContextQuantities, PerceptionModel

MODELS = ("m0a", "m0", "m1")


# ----------------------------------------------------------------------
# 基本量
# ----------------------------------------------------------------------
def mu_contrast(w_s, delta_u: np.ndarray):
    """mu(c) = w_s*Δu_s + (1-w_s)*Δu_e。"""
    w_s = np.asarray(w_s, dtype=float)
    return w_s * delta_u[0] + (1.0 - w_s) * delta_u[1]


def sigma_eff_m0a(kappa_s, kappa_e):
    return np.sqrt(2.0 * (1.0 / np.asarray(kappa_s, float) + 1.0 / np.asarray(kappa_e, float)))


def sigma_eff_m0(w_s, kappa_s, kappa_e):
    w_s = np.asarray(w_s, dtype=float)
    return np.sqrt(
        2.0 * (w_s**2 / np.asarray(kappa_s, float) + (1.0 - w_s) ** 2 / np.asarray(kappa_e, float))
    )


def sigma_eff_m1(w_s, kappa_s, kappa_e, G: np.ndarray):
    w_s = np.asarray(w_s, dtype=float)
    v1 = w_s**2 * G[0] / np.asarray(kappa_s, float) + (1.0 - w_s) ** 2 * G[1] / np.asarray(
        kappa_e, float
    )
    return np.sqrt(v1)


# ----------------------------------------------------------------------
# 选择概率 q_O(c; theta)
# ----------------------------------------------------------------------
def q_choice(model: str, w_s, kappa_s, kappa_e, ctx: ContextQuantities):
    """P[a_O = a1 | a_E, c, theta],按 model 选 sigma_eff。支持广播。"""
    mu = mu_contrast(w_s, ctx.delta_u)
    if model == "m0a":
        sig = sigma_eff_m0a(kappa_s, kappa_e)
    elif model == "m0":
        sig = sigma_eff_m0(w_s, kappa_s, kappa_e)
    elif model == "m1":
        sig = sigma_eff_m1(w_s, kappa_s, kappa_e, ctx.G)
    else:
        raise ValueError(f"model 须为 {MODELS},收到 {model!r}")
    return norm.cdf(mu / sig)


def q_vector(model: str, theta: np.ndarray, contexts: list[ContextQuantities]) -> np.ndarray:
    """q(theta) = [q_1, ..., q_C],theta = (w_s, kappa_s, kappa_e)。"""
    w_s, ks, ke = float(theta[0]), float(theta[1]), float(theta[2])
    if not (0.0 <= w_s <= 1.0) or ks <= 0 or ke <= 0:
        raise ValueError(f"非法 theta:{theta}")
    return np.array([float(q_choice(model, w_s, ks, ke, c)) for c in contexts])


# ----------------------------------------------------------------------
# M1 线性化 MC(数值实现一致性检查:与解析式同一模型,必须一致)
# ----------------------------------------------------------------------
def q_m1_mc_linearized(
    w_s: float,
    kappa_s: float,
    kappa_e: float,
    ctx: ContextQuantities,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """按一阶 Taylor 线性化模型精确模拟:utility 扰动 ~ N(0, L_k/kappa_k),
    跨 (分量, 动作) 独立。与 sigma_eff_m1 解析式是同一个模型,
    差异只能来自 MC 抽样误差。"""
    w = np.array([w_s, 1.0 - w_s])
    kappa = np.array([kappa_s, kappa_e])
    std = np.sqrt(ctx.L / kappa[:, None])               # (K, 2)
    pert = rng.standard_normal((n_samples, 2, 2)) * std[None, :, :]
    u = ctx.u[None, :, :] + pert
    total = np.einsum("nka,k->na", u, w)
    return float(np.mean(total[:, 0] > total[:, 1]))


# ----------------------------------------------------------------------
# M1 全非线性 MC 校验(不做 Taylor 线性化)
# ----------------------------------------------------------------------
def q_m1_mc_full(
    w_s: float,
    kappa_s: float,
    kappa_e: float,
    ctx: ContextQuantities,
    perception: PerceptionModel,
    n_samples: int,
    rng: np.random.Generator,
) -> float:
    """直接模拟感知过程:z + eta -> 真实 f_k -> 加权总效用 -> argmax 频率。

    与解析式的差 = 一阶 Taylor 线性化误差(O(Sigma/kappa)),
    用于校验解析式适用范围,不用于生产结果。
    """
    w = np.array([w_s, 1.0 - w_s])
    kappa = np.array([kappa_s, kappa_e])
    sigma = np.array([perception.sigma[k] for k in COMPONENTS])
    scale = np.sqrt(sigma / kappa)                      # (K,)
    eta = rng.standard_normal((n_samples, 2, 2)) * scale[None, :, None]
    z_noisy = ctx.z[None, :, :] + eta                   # (n, K, 2)
    u = np.empty_like(z_noisy)
    for i, k in enumerate(COMPONENTS):
        u[:, i, :] = perception.f[k](z_noisy[:, i, :])
    total = np.einsum("nka,k->na", u, w)                # (n, 2),列序 (a1, a0)
    return float(np.mean(total[:, 0] > total[:, 1]))