"""Component-probit quantal response.

模型(config 默认 separable 噪声形式):
    U(a) = sum_k w_k * u_k(a)  +  sum_k eps_k(a) / sqrt(kappa_k)
其中 eps_k(a) 为 iid 标准 Gaussian(按分量 x 动作独立),
u_k(a) 为对手混合策略下的期望分量 payoff。

二动作解析式的推导
------------------
记 ubar_k(a) = E_{对手混合策略}[u_k(a, ·)],DV = sum_k w_k [ubar_k(a1) - ubar_k(a2)]。
U(a1) - U(a2) 的噪声部分为 sum_k [eps_k(a1) - eps_k(a2)] / sqrt(kappa_k),
是独立 Gaussian 之和,方差 tau^2 = 2 * sum_k 1/kappa_k。于是

    P(a1) = Phi( DV / tau )

结构性事实(由上式直接可见,测试中显式验证):
separable 噪声下 kappa 仅通过标量 tau^2 = 2*sum_k 1/kappa_k 进入选择概率,
故 (kappa_s, kappa_e) 单独不可辨识,且该不可辨识性与情境参数 c 无关。
详见 docs/step1_findings.md。

Monte Carlo 路径作为独立交叉验证:直接采样 eps、构造 U、取 argmax、数频率。
使用 antithetic 采样(eps 与 -eps 成对)做方差缩减,不改变估计的无偏性。
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

PLAYER_NAMES = ("ego", "opponent")


# ----------------------------------------------------------------------
# 输入校验
# ----------------------------------------------------------------------
def _validate_w_kappa(w, kappa, atol: float = 1e-10):
    w = np.asarray(w, dtype=float)
    kappa = np.asarray(kappa, dtype=float)
    if w.shape != kappa.shape or w.ndim != 1:
        raise ValueError(f"w 与 kappa 须为同长度一维向量,收到 {w.shape} / {kappa.shape}")
    if np.any(w < -atol):
        raise ValueError(f"偏好权重 w 须非负,收到 {w}")
    if abs(w.sum() - 1.0) > 1e-8:
        raise ValueError(f"偏好权重须归一化 sum(w)=1,收到 sum={w.sum()}")
    if np.any(~np.isfinite(kappa)) or np.any(kappa <= 0):
        raise ValueError(f"精度 kappa 须为正有限数,收到 {kappa}")
    return w, kappa


def _validate_mixed(mixed, atol: float = 1e-8):
    mixed = np.asarray(mixed, dtype=float)
    if mixed.ndim != 1 or np.any(mixed < -atol) or abs(mixed.sum() - 1.0) > atol:
        raise ValueError(f"混合策略须为归一化非负向量,收到 {mixed}")
    return np.clip(mixed, 0.0, None)


# ----------------------------------------------------------------------
# 对手混合策略下的期望分量 payoff
# ----------------------------------------------------------------------
def marginal_component_payoffs(
    component_payoffs: np.ndarray, other_mixed, player: str
) -> np.ndarray:
    """对对手混合策略取期望。

    component_payoffs: 该玩家的分量 payoff,形状 (K, n_ego, n_opp),
                       索引约定 [k, ego_action, opp_action](与 ToyGame2x2 一致)
    other_mixed:       对手的混合策略
    player:            'ego'(对 opp 动作求和)或 'opponent'(对 ego 动作求和)
    返回:(K, n_own_actions)
    """
    u = np.asarray(component_payoffs, dtype=float)
    if u.ndim != 3:
        raise ValueError(f"component_payoffs 须为 3 维 (K, n_ego, n_opp),收到 {u.shape}")
    other_mixed = _validate_mixed(other_mixed)
    if player == "ego":
        if u.shape[2] != other_mixed.shape[0]:
            raise ValueError("对手混合策略长度与 opponent 动作数不符")
        return np.einsum("kij,j->ki", u, other_mixed)
    if player == "opponent":
        if u.shape[1] != other_mixed.shape[0]:
            raise ValueError("对手混合策略长度与 ego 动作数不符")
        return np.einsum("kij,i->kj", u, other_mixed)
    raise ValueError(f"player 须为 {PLAYER_NAMES},收到 {player!r}")


# ----------------------------------------------------------------------
# 解析路径(二动作 probit)
# ----------------------------------------------------------------------
def choice_probabilities_analytic(w, kappa, expected_payoffs: np.ndarray) -> np.ndarray:
    """P(a1) = Phi(DV / tau)。expected_payoffs 形状 (K, 2)。返回 (2,) 概率向量。"""
    w, kappa = _validate_w_kappa(w, kappa)
    ubar = np.asarray(expected_payoffs, dtype=float)
    if ubar.shape != (w.shape[0], 2):
        raise ValueError(
            f"解析路径仅支持二动作:expected_payoffs 须为 (K, 2),收到 {ubar.shape}"
        )
    dv = float(w @ (ubar[:, 0] - ubar[:, 1]))
    tau = float(np.sqrt(2.0 * np.sum(1.0 / kappa)))
    p1 = float(norm.cdf(dv / tau))
    return np.array([p1, 1.0 - p1])


# ----------------------------------------------------------------------
# Monte Carlo 路径(交叉验证用;支持任意动作数)
# ----------------------------------------------------------------------
def choice_probabilities_mc(
    w,
    kappa,
    expected_payoffs: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    chunk_size: int = 2_000_000,
) -> np.ndarray:
    """采样 eps -> U -> argmax -> 频率。antithetic:每对样本用 (eps, -eps)。

    n_samples 为总样本数(含 antithetic 配对),须为偶数。
    """
    w, kappa = _validate_w_kappa(w, kappa)
    ubar = np.asarray(expected_payoffs, dtype=float)
    n_k, n_a = ubar.shape
    if n_samples <= 0 or n_samples % 2 != 0:
        raise ValueError(f"n_samples 须为正偶数(antithetic 配对),收到 {n_samples}")

    mean_u = w @ ubar                      # (n_a,)
    noise_scale = 1.0 / np.sqrt(kappa)     # (n_k,)

    counts = np.zeros(n_a, dtype=np.int64)
    n_pairs = n_samples // 2
    done = 0
    while done < n_pairs:
        m = min(chunk_size, n_pairs - done)
        eps = rng.standard_normal((m, n_k, n_a))
        noise = np.einsum("mka,k->ma", eps, noise_scale)
        for signed in (noise, -noise):
            idx = np.argmax(mean_u[None, :] + signed, axis=1)
            counts += np.bincount(idx, minlength=n_a)
        done += m
    return counts / float(n_samples)


# ----------------------------------------------------------------------
# 统一入口:文档接口 (w, kappa, 对手混合策略, payoff 矩阵) -> 选择概率
# ----------------------------------------------------------------------
def choice_probabilities(
    w,
    kappa,
    other_mixed,
    component_payoffs: np.ndarray,
    player: str,
    method: str = "analytic",
    n_samples: int | None = None,
    rng: np.random.Generator | None = None,
    chunk_size: int = 2_000_000,
) -> np.ndarray:
    """按 method 选择解析或 MC 路径计算该玩家的选择概率向量。"""
    ubar = marginal_component_payoffs(component_payoffs, other_mixed, player)
    if method == "analytic":
        return choice_probabilities_analytic(w, kappa, ubar)
    if method == "mc":
        if n_samples is None or rng is None:
            raise ValueError("MC 路径需显式给出 n_samples 与 rng(seed 来自 config)")
        return choice_probabilities_mc(w, kappa, ubar, n_samples, rng, chunk_size)
    raise ValueError(f"method 须为 'analytic' 或 'mc',收到 {method!r}")