"""Step 1 revision — structural identifiability 分析(新 C3)。

参数化:theta_tilde = (w_s, log kappa_s, log kappa_e)。
log-precision 避开正性边界,数值微分可用对称中心差分;报告时转回 kappa。

对 C 个 contexts 的选择概率向量 q(theta) = [q_1, ..., q_C]^T 计算
数值 Jacobian J = dq/dtheta_tilde(中心差分,步长来自 config),做 SVD:

- singular values s_1 >= s_2 >= s_3
- numerical rank:#{ s_i / s_1 > rank_rtol }
- condition number:s_1 / s_r(r = numerical rank)与 s_1 / s_3

判读约定(不只报告 rank,同时报 sv 谱用于 weak identification 判断):
- M0a / M0:q 只依赖 (w_s, 各自的 aggregate),理论 rank <= 2,
  s_3/s_1 应落在数值微分噪声水平;
- M1:strong-diversity 下应 full rank;insufficient-diversity(G 恒定)
  下重新退化到 rank <= 2;moderate 介于其间,由 condition number 量化。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.games.contexts import ContextQuantities
from src.response.models import q_vector

PARAM_NAMES = ("w_s", "log_kappa_s", "log_kappa_e")


# ----------------------------------------------------------------------
# 参数变换
# ----------------------------------------------------------------------
def to_tilde(theta: np.ndarray) -> np.ndarray:
    """(w_s, ks, ke) -> (w_s, log ks, log ke)。"""
    t = np.asarray(theta, dtype=float)
    return np.array([t[0], np.log(t[1]), np.log(t[2])])


def from_tilde(tilde: np.ndarray) -> np.ndarray:
    t = np.asarray(tilde, dtype=float)
    return np.array([t[0], np.exp(t[1]), np.exp(t[2])])


def q_of_tilde(model: str, tilde: np.ndarray, contexts: list[ContextQuantities]) -> np.ndarray:
    return q_vector(model, from_tilde(tilde), contexts)


# ----------------------------------------------------------------------
# 数值 Jacobian
# ----------------------------------------------------------------------
def numerical_jacobian(
    model: str,
    theta: np.ndarray,
    contexts: list[ContextQuantities],
    step_w: float,
    step_logk: float,
) -> np.ndarray:
    """J[c, p] = dq_c / dtheta_tilde_p,中心差分。w_s 靠边界时收缩步长。"""
    tilde = to_tilde(theta)
    steps = np.array([step_w, step_logk, step_logk], dtype=float)
    # w_s 边界保护:步长不越出 [0, 1]
    steps[0] = min(steps[0], tilde[0] / 2 if tilde[0] > 0 else steps[0],
                   (1 - tilde[0]) / 2 if tilde[0] < 1 else steps[0])
    if steps[0] <= 0:
        raise ValueError(f"w_s={tilde[0]} 位于边界,无法做中心差分")
    cols = []
    for p in range(3):
        e = np.zeros(3)
        e[p] = steps[p]
        q_plus = q_of_tilde(model, tilde + e, contexts)
        q_minus = q_of_tilde(model, tilde - e, contexts)
        cols.append((q_plus - q_minus) / (2.0 * steps[p]))
    return np.stack(cols, axis=1)  # (C, 3)


# ----------------------------------------------------------------------
# SVD 判读
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class IdentifiabilityReport:
    model: str
    family: str
    theta: np.ndarray               # (w_s, ks, ke)
    singular_values: np.ndarray     # 降序,长度 min(C, 3)
    numerical_rank: int
    cond_full: float                # s_1 / s_min(全谱条件数;deficient 时巨大)
    cond_rank: float                # s_1 / s_rank(有效子空间条件数)
    null_direction: np.ndarray      # 最小奇异值对应的右奇异向量(theta_tilde 坐标)

    def summary_row(self) -> str:
        sv = " ".join(f"{s:.3e}" for s in self.singular_values)
        return (
            f"{self.model:4s} {self.family:22s} theta=({self.theta[0]:.2f},"
            f"{self.theta[1]:.2f},{self.theta[2]:.2f})  sv=[{sv}]  "
            f"rank={self.numerical_rank}  cond_rank={self.cond_rank:.2e}"
        )


def analyze(
    model: str,
    theta: np.ndarray,
    contexts: list[ContextQuantities],
    family: str,
    id_cfg: dict,
) -> IdentifiabilityReport:
    """完整局部可辨识性判读。id_cfg 须含 fd_step_w / fd_step_logk / rank_rtol。"""
    J = numerical_jacobian(
        model, theta, contexts, id_cfg["fd_step_w"], id_cfg["fd_step_logk"]
    )
    _, s, vt = np.linalg.svd(J, full_matrices=False)
    rank = int(np.sum(s / s[0] > id_cfg["rank_rtol"])) if s[0] > 0 else 0
    cond_full = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    cond_rank = float(s[0] / s[rank - 1]) if rank >= 1 else float("inf")
    return IdentifiabilityReport(
        model=model,
        family=family,
        theta=np.asarray(theta, dtype=float),
        singular_values=s,
        numerical_rank=rank,
        cond_full=cond_full,
        cond_rank=cond_rank,
        null_direction=vt[-1],
    )


def subset_ranks(
    model: str,
    theta: np.ndarray,
    contexts: list[ContextQuantities],
    id_cfg: dict,
) -> dict:
    """context 子集(大小 1..C)的 numerical rank:回答
    "M1 在什么 context 组合下达到 full local rank"。"""
    from itertools import combinations

    out = {}
    for r in range(1, len(contexts) + 1):
        for combo in combinations(range(len(contexts)), r):
            sub = [contexts[i] for i in combo]
            rep = analyze(model, theta, sub, "subset", id_cfg)
            key = "+".join(sub_c.name for sub_c in sub)
            out[key] = {
                "rank": rep.numerical_rank,
                "singular_values": rep.singular_values.tolist(),
            }
    return out