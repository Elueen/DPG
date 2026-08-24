"""Step 1 revision — 闭式选择概率的直接网格扫描(新 C5 数据层)。

与第一轮 grid_scan 的区别:revision 条件于 (a_E, c),q_O(c; theta) 有
闭式(src/response/models),无需不动点求解。本模块只负责在
(w_s, kappa_s, kappa_e) 网格上批量计算各 (model, context) 的 q 张量;
等价集合/指标/拓扑全部复用 src/analysis/equivalence 的第一轮机制。

理论 ridge 曲线(叠画用):collapse 方向为加权 aggregate 的 level set
    a/ks + b/ke = a/ks_ref + b/ke_ref
  M0a: (a, b) = (1, 1)
  M0 : (a, b) = (ws^2, we^2)          [w_s 固定于参考切片]
  M1(G 跨 context 恒定): (a, b) = (ws^2*Gs, we^2*Ge)
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.analysis.grid_scan import build_axis
from src.games.contexts import ContextQuantities
from src.response.models import q_choice


def build_grid(grid_cfg: dict, n_override: int | None = None) -> SimpleNamespace:
    """构造网格轴容器(与 equivalence.normalized_coords 兼容的属性名)。"""
    w = build_axis(grid_cfg["w_s"], n_override)
    ks = build_axis(grid_cfg["kappa_s"], n_override)
    ke = build_axis(grid_cfg["kappa_e"], n_override)
    if np.any(w < 0) or np.any(w > 1):
        raise ValueError("w_s 轴须在 [0, 1] 内")
    if np.any(ks <= 0) or np.any(ke <= 0):
        raise ValueError("kappa 轴须为正")
    return SimpleNamespace(w_s_axis=w, kappa_s_axis=ks, kappa_e_axis=ke)


def q_grid(model: str, ctx: ContextQuantities, grid: SimpleNamespace) -> np.ndarray:
    """单 (model, context) 的 q 张量,形状 (n_w, n_ks, n_ke)。闭式广播计算。"""
    W, KS, KE = np.meshgrid(
        grid.w_s_axis, grid.kappa_s_axis, grid.kappa_e_axis, indexing="ij"
    )
    return np.asarray(q_choice(model, W, KS, KE, ctx))


def q_grids(model: str, contexts: list[ContextQuantities], grid: SimpleNamespace) -> list[np.ndarray]:
    return [q_grid(model, c, grid) for c in contexts]


def snap(grid: SimpleNamespace, ref: dict) -> tuple[int, int, int]:
    """参考点吸附到最近网格点(与第一轮 snap_reference 同语义)。"""
    return (
        int(np.argmin(np.abs(grid.w_s_axis - ref["w_s"]))),
        int(np.argmin(np.abs(grid.kappa_s_axis - ref["kappa_s"]))),
        int(np.argmin(np.abs(grid.kappa_e_axis - ref["kappa_e"]))),
    )


def theory_ridge_ke(
    kappa_s_axis: np.ndarray, a: float, b: float, ks_ref: float, ke_ref: float
) -> np.ndarray:
    """加权 aggregate level set 在 kappa_s 轴上的 kappa_e 取值(非正区域 NaN)。"""
    const = a / ks_ref + b / ke_ref
    inv_ke = (const - a / kappa_s_axis) / b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(inv_ke > 0, 1.0 / inv_ke, np.nan)


def collapse_weights(model: str, w_s: float, G: np.ndarray | None) -> tuple[float, float] | None:
    """各模型 collapse 方向的 (a, b);M1 仅在 G 跨 context 恒定时有单一方向,
    否则返回 None(strong/moderate 家族无全局 ridge 曲线可画)。"""
    if model == "m0a":
        return 1.0, 1.0
    if model == "m0":
        return w_s**2, (1.0 - w_s) ** 2
    if model == "m1":
        if G is None:
            return None
        return w_s**2 * float(G[0]), (1.0 - w_s) ** 2 * float(G[1])
    raise ValueError(model)


def kappa_slice_diameter(mask: np.ndarray, ref_idx: tuple, grid: SimpleNamespace) -> float:
    """(kappa_s, kappa_e) 切片(w_s 固定于参考格点)内等价集合的
    Chebyshev 直径(两 kappa 轴 log 归一化坐标取大)。
    该量是 "collapse 方向是否被移除" 的判别量:
    M0 谱系应随 context 数保持;M1 在 strong diversity 下应收缩到格点级。"""

    def _norm(x):
        l = np.log(x)
        return (l - l.min()) / (l.max() - l.min())

    s = mask[ref_idx[0], :, :]
    if not s.any():
        return 0.0
    nks, nke = _norm(grid.kappa_s_axis), _norm(grid.kappa_e_axis)
    jj, kk = np.where(s)
    return float(
        max(nks[jj].max() - nks[jj].min(), nke[kk].max() - nke[kk].min())
    )