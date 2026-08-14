"""C5 等价类(等价脊线)分析。

观测等价的定义(角色定案后的口径,见 findings C4/C5):
可观测量为 opponent 的均衡选择概率 q* = P[opp=yield](ego 画像固定时
p* = Phi_e(q*) 是 q* 的确定函数,不含额外信息)。参数点 theta 与参考点
theta_ref 在情境集合 C 上观测等价 iff 对所有 c in C,
|q*_c(theta) - q*_c(theta_ref)| < tol(二动作下 sup-norm 即绝对差)。

指标:
- volume_fraction:等价格点数 / 总格点数
- diameter:归一化坐标(w_s 线性映到 [0,1],kappa 取 log 后映到 [0,1])
  下的 Chebyshev 直径 = max_axis (坐标最大值 - 最小值)
- n_components:2D 切片上等价集合的 4-连通分量数(scipy.ndimage.label),
  作为脊线拓扑(连通性)的定量口径,用于 ego 画像鲁棒性检查
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import ndimage


# ----------------------------------------------------------------------
# 扫描结果读取
# ----------------------------------------------------------------------
def load_scan(path: Path | str) -> SimpleNamespace:
    """读取 C4 的 scan npz,返回属性化对象。"""
    with np.load(Path(path), allow_pickle=False) as z:
        return SimpleNamespace(
            w_s_axis=z["w_s_axis"],
            kappa_s_axis=z["kappa_s_axis"],
            kappa_e_axis=z["kappa_e_axis"],
            p_eq=z["p_eq"],
            q_eq=z["q_eq"],
            n_unique=z["n_unique"],
            situation_c=float(z["situation_c"]),
            ego_profile_name=str(z["ego_profile_name"]),
        )


def check_same_grid(scans: list) -> None:
    """多情境分析要求各扫描共享同一网格。"""
    ref = scans[0]
    for s in scans[1:]:
        for name in ("w_s_axis", "kappa_s_axis", "kappa_e_axis"):
            if not np.array_equal(getattr(ref, name), getattr(s, name)):
                raise ValueError(f"扫描网格不一致:{name}")


# ----------------------------------------------------------------------
# 参考点吸附
# ----------------------------------------------------------------------
def nearest_index(axis: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(axis - float(value))))


def snap_reference(scan, ref: dict) -> tuple[int, int, int]:
    """参考点 {w_s, kappa_s, kappa_e} 吸附到最近网格点,返回 (i, j, k)。"""
    return (
        nearest_index(scan.w_s_axis, ref["w_s"]),
        nearest_index(scan.kappa_s_axis, ref["kappa_s"]),
        nearest_index(scan.kappa_e_axis, ref["kappa_e"]),
    )


# ----------------------------------------------------------------------
# 等价集合
# ----------------------------------------------------------------------
def equivalence_mask(q_eq: np.ndarray, ref_idx: tuple[int, int, int], tol: float) -> np.ndarray:
    """单情境等价集合(3D bool)。NaN 格点视为不等价。"""
    ref_q = q_eq[ref_idx]
    if np.isnan(ref_q):
        raise ValueError(f"参考格点 {ref_idx} 的均衡为 NaN,无法定义等价集合")
    with np.errstate(invalid="ignore"):
        return np.abs(q_eq - ref_q) < tol


def cumulative_masks(
    q_list: list[np.ndarray], ref_idx: tuple[int, int, int], tol: float
) -> list[np.ndarray]:
    """情境按给定顺序累积交集:返回 [1 情境, 前 2 情境交, 前 3 情境交, ...]。"""
    masks = []
    acc = None
    for q in q_list:
        m = equivalence_mask(q, ref_idx, tol)
        acc = m if acc is None else (acc & m)
        masks.append(acc.copy())
    return masks


# ----------------------------------------------------------------------
# 指标
# ----------------------------------------------------------------------
def normalized_coords(scan) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """各轴映到 [0, 1]:w_s 线性;kappa 先取 log 再线性。单点轴映为 0。"""

    def _norm(x: np.ndarray) -> np.ndarray:
        lo, hi = x.min(), x.max()
        return np.zeros_like(x) if hi == lo else (x - lo) / (hi - lo)

    return (
        _norm(scan.w_s_axis),
        _norm(np.log(scan.kappa_s_axis)),
        _norm(np.log(scan.kappa_e_axis)),
    )


def set_metrics(mask: np.ndarray, norm_axes: tuple) -> dict:
    """等价集合的体积占比与 Chebyshev 直径(归一化坐标)。"""
    count = int(mask.sum())
    volume_fraction = count / mask.size
    if count == 0:
        return {"count": 0, "volume_fraction": 0.0, "diameter": 0.0}
    idx = np.where(mask)
    diameter = max(
        float(ax[i].max() - ax[i].min()) for ax, i in zip(norm_axes, idx)
    )
    return {"count": count, "volume_fraction": volume_fraction, "diameter": diameter}


def count_components_2d(mask_2d: np.ndarray) -> int:
    """2D 切片上等价集合的 4-连通分量数(脊线拓扑的定量口径)。"""
    _, n = ndimage.label(mask_2d, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    return int(n)


# ----------------------------------------------------------------------
# 切片提取(画脊线图用)
# ----------------------------------------------------------------------
def slice_w_ks(mask: np.ndarray, ref_idx: tuple[int, int, int]) -> np.ndarray:
    """(w_s, kappa_s) 平面,kappa_e 固定在参考格点。"""
    return mask[:, :, ref_idx[2]]


def slice_ks_ke(mask: np.ndarray, ref_idx: tuple[int, int, int]) -> np.ndarray:
    """(kappa_s, kappa_e) 平面,w_s 固定在参考格点。"""
    return mask[ref_idx[0], :, :]


def kappa_aggregate_hyperbola(
    kappa_s_axis: np.ndarray, ref_kappa_s: float, ref_kappa_e: float
) -> np.ndarray:
    """理论预言曲线:1/ks + 1/ke = 1/ks_ref + 1/ke_ref 在 kappa_s 轴上的
    kappa_e 取值(非正区域记 NaN)。用于叠画在 (ks, ke) 脊线图上。"""
    agg = 1.0 / ref_kappa_s + 1.0 / ref_kappa_e
    inv_ke = agg - 1.0 / kappa_s_axis
    with np.errstate(divide="ignore", invalid="ignore"):
        ke = np.where(inv_ke > 0, 1.0 / inv_ke, np.nan)
    return ke