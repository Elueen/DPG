"""C4 网格扫描:opponent 画像 (w_s, kappa_s, kappa_e) 上的均衡选择概率。

角色约定(与两步法辨识路线对应,见 config scan 段注释):
  - 被扫描方 = opponent(被推断的周车画像),w_e = 1 - w_s
  - ego 画像固定,从 config 的 scan.ego_profiles 读取

向量化批量求解
--------------
数学与 src/solvers/fixed_point.py 完全相同(damped Jacobi + 解析 probit 响应),
仅利用 2x2 博弈的结构做批量化:二动作下策略由标量 (p, q) 表征
(p = P[ego=merge], q = P[opp=yield]),期望 payoff 差对 p、q 线性:

    DV_ego(q) = A_e * q + B_e * (1 - q)          (A_e, B_e 为标量,ego 画像固定)
    DV_opp(p) = A_o * p + B_o * (1 - p)          (A_o, B_o 随网格点 w_s 变化)
    p <- (1-a) p + a * Phi(DV_ego / tau_e),  q <- (1-a) q + a * Phi(DV_opp / tau_o)

整个 (起点 x 网格点) 张量同步迭代,已收敛元素冻结。
与参考求解器的逐位一致性由 tests/test_grid_scan.py 互检。

均衡多重性与规范选择(canonical selection)
------------------------------------------
每个网格点跑 1 个均匀起点 + n_starts 个随机起点,收敛结果按 match_tol
聚类得 n_unique。存储的"规范均衡"取均匀起点的收敛结果;若均匀起点
不收敛(极端 kappa 下的反协调振荡,见 findings C3),回退到第一个收敛的
随机起点并置 fallback 标志;全部不收敛则记 NaN。多重均衡区域由
n_unique > 1 的格点集合刻画。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from src.games.toy_2x2 import ToyGame2x2

MAX_EQUILIBRIA_SLOTS = 6  # 聚类槽位上限(2x2 QRE 实际上限为 3,留裕量)


# ----------------------------------------------------------------------
# 网格轴
# ----------------------------------------------------------------------
def build_axis(axis_cfg: dict, n_override: int | None = None) -> np.ndarray:
    """由 config 的 {min, max, n, spacing} 生成一维网格轴。"""
    n = int(n_override if n_override is not None else axis_cfg["n"])
    lo, hi = float(axis_cfg["min"]), float(axis_cfg["max"])
    spacing = axis_cfg.get("spacing", "linear")
    if spacing == "linear":
        return np.linspace(lo, hi, n)
    if spacing == "log":
        if lo <= 0:
            raise ValueError(f"log 轴要求 min > 0,收到 {lo}")
        return np.geomspace(lo, hi, n)
    raise ValueError(f"spacing 须为 linear/log,收到 {spacing!r}")


# ----------------------------------------------------------------------
# 结果容器
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ScanResult:
    """一个 (情境, ego 画像) 组合的扫描结果。网格展平顺序 = ij 索引
    np.meshgrid(w_s, kappa_s, kappa_e, indexing='ij') 后 reshape(-1)。"""

    w_s_axis: np.ndarray
    kappa_s_axis: np.ndarray
    kappa_e_axis: np.ndarray
    p_eq: np.ndarray          # 规范均衡 P[ego=merge],形状 (n_w, n_ks, n_ke)
    q_eq: np.ndarray          # 规范均衡 P[opp=yield],同形状
    n_unique: np.ndarray      # 每点不同均衡个数(int8),同形状
    n_converged: np.ndarray   # 每点收敛起点数(int8),同形状
    fallback: np.ndarray      # 规范均衡来自回退起点(bool),同形状
    situation_c: float
    ego_profile_name: str

    def to_npz_dict(self) -> dict:
        return {
            "w_s_axis": self.w_s_axis,
            "kappa_s_axis": self.kappa_s_axis,
            "kappa_e_axis": self.kappa_e_axis,
            "p_eq": self.p_eq,
            "q_eq": self.q_eq,
            "n_unique": self.n_unique,
            "n_converged": self.n_converged,
            "fallback": self.fallback,
            "situation_c": np.array(self.situation_c),
            "ego_profile_name": np.array(self.ego_profile_name),
        }


# ----------------------------------------------------------------------
# 响应映射的线性系数
# ----------------------------------------------------------------------
def _ego_coeffs(game: ToyGame2x2, w_ego: np.ndarray, kappa_ego: np.ndarray):
    """DV_ego(q) = A_e*q + B_e*(1-q);tau_e。ego 画像固定 -> 全为标量。"""
    u = game.component_payoffs("ego")               # (K, 2, 2)
    delta = u[:, 0, :] - u[:, 1, :]                 # (K, 2): merge - wait, 按 opp 动作
    a_e = float(w_ego @ delta[:, 0])                # opp = yield
    b_e = float(w_ego @ delta[:, 1])                # opp = hold
    tau_e = float(np.sqrt(2.0 * np.sum(1.0 / kappa_ego)))
    return a_e, b_e, tau_e


def _opp_coeffs(game: ToyGame2x2, w_s_flat: np.ndarray, tau_o_flat: np.ndarray):
    """DV_opp(p) = A_o*p + B_o*(1-p),A_o/B_o 随网格点 w_s 变化。"""
    u = game.component_payoffs("opponent")          # (K, 2, 2)
    delta = u[:, :, 0] - u[:, :, 1]                 # (K, 2): yield - hold, 按 ego 动作
    a_o = w_s_flat * delta[0, 0] + (1.0 - w_s_flat) * delta[1, 0]   # ego = merge
    b_o = w_s_flat * delta[0, 1] + (1.0 - w_s_flat) * delta[1, 1]   # ego = wait
    return a_o, b_o, tau_o_flat


# ----------------------------------------------------------------------
# 向量化批量不动点
# ----------------------------------------------------------------------
def _batch_fixed_point(
    a_e: float, b_e: float, tau_e: float,
    a_o: np.ndarray, b_o: np.ndarray, tau_o: np.ndarray,
    p0: np.ndarray, q0: np.ndarray,
    damping: float, tol: float, max_iter: int,
):
    """p0, q0 形状 (S, B)。返回 (p, q, converged) 同形状。已收敛元素冻结。"""
    S, B = p0.shape
    p, q = p0.copy(), q0.copy()
    ao = np.broadcast_to(a_o, (S, B))
    bo = np.broadcast_to(b_o, (S, B))
    to = np.broadcast_to(tau_o, (S, B))
    converged = np.zeros((S, B), dtype=bool)

    for _ in range(max_iter):
        act = ~converged
        if not act.any():
            break
        pa, qa = p[act], q[act]
        pn = norm.cdf((a_e * qa + b_e * (1.0 - qa)) / tau_e)
        qn = norm.cdf((ao[act] * pa + bo[act] * (1.0 - pa)) / to[act])
        res = np.maximum(np.abs(pn - pa), np.abs(qn - qa))
        p[act] = (1.0 - damping) * pa + damping * pn
        q[act] = (1.0 - damping) * qa + damping * qn
        conv_flat = converged[act]
        conv_flat |= res < tol
        converged[act] = conv_flat
    return p, q, converged


# ----------------------------------------------------------------------
# 聚类(逐起点、跨网格向量化)
# ----------------------------------------------------------------------
def _cluster_batch(p: np.ndarray, q: np.ndarray, converged: np.ndarray, match_tol: float):
    """按 match_tol 对每个网格点跨起点聚类。返回 n_unique (B,) int8。"""
    S, B = p.shape
    slot_p = np.zeros((MAX_EQUILIBRIA_SLOTS, B))
    slot_q = np.zeros((MAX_EQUILIBRIA_SLOTS, B))
    slot_used = np.zeros((MAX_EQUILIBRIA_SLOTS, B), dtype=bool)

    for s in range(S):
        remaining = converged[s].copy()
        ps, qs = p[s], q[s]
        for m in range(MAX_EQUILIBRIA_SLOTS):
            match = (
                remaining
                & slot_used[m]
                & (np.abs(ps - slot_p[m]) < match_tol)
                & (np.abs(qs - slot_q[m]) < match_tol)
            )
            remaining &= ~match
        for m in range(MAX_EQUILIBRIA_SLOTS):
            place = remaining & ~slot_used[m]
            slot_p[m][place] = ps[place]
            slot_q[m][place] = qs[place]
            slot_used[m][place] = True
            remaining &= ~place
        if remaining.any():
            raise RuntimeError("均衡槽位溢出:某网格点不同均衡数超过上限")
    return slot_used.sum(axis=0).astype(np.int8)


# ----------------------------------------------------------------------
# 扫描主入口
# ----------------------------------------------------------------------
def scan_situation(
    game_at_c: ToyGame2x2,
    ego_profile: dict,
    ego_profile_name: str,
    grid_cfg: dict,
    solver_cfg: dict,
    rng: np.random.Generator,
    n_override: int | None = None,
) -> ScanResult:
    """在固定情境的博弈上扫 opponent 画像网格。

    ego_profile: {'w': [...], 'kappa': [...]}(来自 config scan.ego_profiles)
    grid_cfg:    config scan.grid;n_override 用于鲁棒性粗网格
    solver_cfg:  config solver 段(damping/tol/max_iter/n_starts/match_tol)
    """
    if game_at_c.situation_c is None:
        raise ValueError("必须传入已做情境实例化的博弈(game.at_situation(c))")
    if solver_cfg["n_starts"] < 10:
        raise ValueError(f"纪律要求随机起点 >= 10 个,config 给了 {solver_cfg['n_starts']}")

    w_axis = build_axis(grid_cfg["w_s"], n_override)
    ks_axis = build_axis(grid_cfg["kappa_s"], n_override)
    ke_axis = build_axis(grid_cfg["kappa_e"], n_override)
    if np.any(w_axis < 0) or np.any(w_axis > 1):
        raise ValueError("w_s 轴须在 [0, 1] 内")
    if np.any(ks_axis <= 0) or np.any(ke_axis <= 0):
        raise ValueError("kappa 轴须为正")

    W, KS, KE = np.meshgrid(w_axis, ks_axis, ke_axis, indexing="ij")
    shape = W.shape
    w_flat = W.reshape(-1)
    tau_o_flat = np.sqrt(2.0 * (1.0 / KS.reshape(-1) + 1.0 / KE.reshape(-1)))
    B = w_flat.size

    w_ego = np.asarray(ego_profile["w"], dtype=float)
    kappa_ego = np.asarray(ego_profile["kappa"], dtype=float)
    a_e, b_e, tau_e = _ego_coeffs(game_at_c, w_ego, kappa_ego)
    a_o, b_o, tau_o = _opp_coeffs(game_at_c, w_flat, tau_o_flat)

    # 起点:index 0 = 均匀,其后 n_starts 个随机(每点独立)
    S = 1 + int(solver_cfg["n_starts"])
    p0 = np.empty((S, B))
    q0 = np.empty((S, B))
    p0[0], q0[0] = 0.5, 0.5
    p0[1:] = rng.random((S - 1, B))
    q0[1:] = rng.random((S - 1, B))

    p, q, converged = _batch_fixed_point(
        a_e, b_e, tau_e, a_o, b_o, tau_o, p0, q0,
        solver_cfg["damping"], solver_cfg["tol"], solver_cfg["max_iter"],
    )

    n_unique = _cluster_batch(p, q, converged, solver_cfg["match_tol"])
    n_converged = converged.sum(axis=0).astype(np.int8)

    # 规范均衡:均匀起点;不收敛则回退第一个收敛起点;全不收敛 -> NaN
    p_eq = np.where(converged[0], p[0], np.nan)
    q_eq = np.where(converged[0], q[0], np.nan)
    any_conv = converged.any(axis=0)
    fallback = ~converged[0] & any_conv
    if fallback.any():
        first = np.argmax(converged, axis=0)
        idx = np.where(fallback)[0]
        p_eq[idx] = p[first[idx], idx]
        q_eq[idx] = q[first[idx], idx]

    return ScanResult(
        w_s_axis=w_axis,
        kappa_s_axis=ks_axis,
        kappa_e_axis=ke_axis,
        p_eq=p_eq.reshape(shape),
        q_eq=q_eq.reshape(shape),
        n_unique=n_unique.reshape(shape),
        n_converged=n_converged.reshape(shape),
        fallback=fallback.reshape(shape),
        situation_c=float(game_at_c.situation_c),
        ego_profile_name=ego_profile_name,
    )