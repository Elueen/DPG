"""Damped fixed-point solver:双边 component-probit 响应的均衡(probit-QRE)。

均衡定义:策略组 (p, q) 满足
    p = R_ego(q; w_ego, kappa_ego),  q = R_opp(p; w_opp, kappa_opp)
其中 R 为 src/response/component_probit 的解析响应。

迭代格式(Jacobi 同时更新 + 阻尼):
    x_{t+1} = (1 - alpha) * x_t + alpha * T(x_t),  x = (p, q)
残差取 sup-norm ||T(x_t) - x_t||_inf,低于 tol 判收敛。
alpha、tol、max_iter、多起点数量与聚类容差全部来自 config 的 solver 段。

多起点:1 个均匀策略起点 + n_starts 个随机起点(rng 由 config seed 派生),
收敛结果按 sup-norm < match_tol 聚类,报告不同均衡个数(多重均衡检测)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.games.toy_2x2 import ToyGame2x2
from src.response.component_probit import (
    choice_probabilities_analytic,
    marginal_component_payoffs,
)

Profiles = dict  # {"ego": (w, kappa), "opponent": (w, kappa)}


# ----------------------------------------------------------------------
# 结果容器
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class FixedPointResult:
    ego_strategy: np.ndarray        # p,形状 (n_ego,)
    opponent_strategy: np.ndarray   # q,形状 (n_opp,)
    converged: bool
    n_iter: int
    residuals: np.ndarray           # 残差轨迹,长度 n_iter
    init: np.ndarray                # 起点 (p0, q0) 拼接,便于诊断

    @property
    def strategy_vector(self) -> np.ndarray:
        """(p, q) 拼接为一维向量,用于聚类与存储。"""
        return np.concatenate([self.ego_strategy, self.opponent_strategy])


@dataclass(frozen=True)
class MultiStartResult:
    runs: tuple                     # tuple[FixedPointResult, ...]
    equilibria: np.ndarray          # 去重后的均衡,形状 (n_unique, n_ego + n_opp)
    n_unique: int
    all_converged: bool

    @property
    def is_unique(self) -> bool:
        return self.n_unique == 1


# ----------------------------------------------------------------------
# 响应映射
# ----------------------------------------------------------------------
def _response_map(
    game: ToyGame2x2, profiles: Profiles, p: np.ndarray, q: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    w_e, k_e = profiles["ego"]
    w_o, k_o = profiles["opponent"]
    ubar_e = marginal_component_payoffs(game.component_payoffs("ego"), q, "ego")
    ubar_o = marginal_component_payoffs(game.component_payoffs("opponent"), p, "opponent")
    return (
        choice_probabilities_analytic(w_e, k_e, ubar_e),
        choice_probabilities_analytic(w_o, k_o, ubar_o),
    )


# ----------------------------------------------------------------------
# 单起点求解
# ----------------------------------------------------------------------
def solve_fixed_point(
    game: ToyGame2x2,
    profiles: Profiles,
    damping: float,
    tol: float,
    max_iter: int,
    init: tuple[np.ndarray, np.ndarray] | None = None,
) -> FixedPointResult:
    """从单一起点做 damped fixed-point iteration。init=None 时用均匀策略起点。"""
    if not (0.0 < damping <= 1.0):
        raise ValueError(f"阻尼系数须在 (0, 1],收到 {damping}")
    n_ego, n_opp = len(game.ego_actions), len(game.opponent_actions)
    if init is None:
        p = np.full(n_ego, 1.0 / n_ego)
        q = np.full(n_opp, 1.0 / n_opp)
    else:
        p = np.asarray(init[0], dtype=float).copy()
        q = np.asarray(init[1], dtype=float).copy()
    init_vec = np.concatenate([p, q])

    residuals = []
    converged = False
    for _ in range(max_iter):
        tp, tq = _response_map(game, profiles, p, q)
        residual = max(np.max(np.abs(tp - p)), np.max(np.abs(tq - q)))
        residuals.append(residual)
        p = (1.0 - damping) * p + damping * tp
        q = (1.0 - damping) * q + damping * tq
        if residual < tol:
            converged = True
            break

    return FixedPointResult(
        ego_strategy=p,
        opponent_strategy=q,
        converged=converged,
        n_iter=len(residuals),
        residuals=np.asarray(residuals),
        init=init_vec,
    )


# ----------------------------------------------------------------------
# 多起点求解 + 多重均衡检测
# ----------------------------------------------------------------------
def _cluster_equilibria(vectors: list[np.ndarray], match_tol: float) -> np.ndarray:
    unique: list[np.ndarray] = []
    for v in vectors:
        if not any(np.max(np.abs(v - u)) < match_tol for u in unique):
            unique.append(v)
    return np.asarray(unique) if unique else np.empty((0, 0))


def random_inits(
    n_starts: int, n_ego: int, n_opp: int, rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray]]:
    """n_starts 个随机起点:各玩家策略独立取自单纯形上的均匀分布(Dirichlet(1))。"""
    return [
        (rng.dirichlet(np.ones(n_ego)), rng.dirichlet(np.ones(n_opp)))
        for _ in range(n_starts)
    ]


def solve_multi_start(
    game: ToyGame2x2,
    profiles: Profiles,
    solver_cfg: dict,
    rng: np.random.Generator,
) -> MultiStartResult:
    """1 个均匀起点 + solver_cfg['n_starts'] 个随机起点,聚类检测多重均衡。

    solver_cfg 须含:damping, tol, max_iter, n_starts, match_tol(均来自 config)。
    """
    damping = solver_cfg["damping"]
    tol = solver_cfg["tol"]
    max_iter = solver_cfg["max_iter"]
    n_starts = solver_cfg["n_starts"]
    match_tol = solver_cfg["match_tol"]
    if n_starts < 10:
        raise ValueError(f"纪律要求随机起点 >= 10 个,config 给了 {n_starts}")

    n_ego, n_opp = len(game.ego_actions), len(game.opponent_actions)
    inits: list = [None] + random_inits(n_starts, n_ego, n_opp, rng)
    runs = tuple(
        solve_fixed_point(game, profiles, damping, tol, max_iter, init=i)
        for i in inits
    )
    converged_vecs = [r.strategy_vector for r in runs if r.converged]
    equilibria = _cluster_equilibria(converged_vecs, match_tol)
    return MultiStartResult(
        runs=runs,
        equilibria=equilibria,
        n_unique=len(equilibria),
        all_converged=all(r.converged for r in runs),
    )
