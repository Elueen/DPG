"""C5 测试:等价类分析。

关键项:
  1) 等价集合基本性质:含参考点自身;累积交集单调不增;NaN 视为不等价
  2) 指标正确性:手算小例上的 volume / Chebyshev 直径 / 连通分量数
  3) C2 结构性预言的数值确认:同一 w_s 下,kappa 聚合量 1/ks+1/ke 相同的
     网格点在**每个情境**都落入彼此的等价集合(双曲线脊线不随情境收缩)
  4) 参考点吸附正确性
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.analysis.equivalence import (
    count_components_2d,
    cumulative_masks,
    equivalence_mask,
    kappa_aggregate_hyperbola,
    nearest_index,
    normalized_coords,
    set_metrics,
    snap_reference,
)
from src.analysis.grid_scan import scan_situation
from src.games.toy_2x2 import ToyGame2x2

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "step1_toy_game.yaml"
N_COARSE = 9


@pytest.fixture(scope="module")
def config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def scans(config):
    """三个情境的粗网格扫描(内存中,不落盘)。"""
    game = ToyGame2x2.from_config(config["game"])
    prof = config["scan"]["ego_profiles"]["baseline"]
    out = []
    for name in config["equivalence"]["situation_order"]:
        rng = np.random.default_rng(config["meta"]["seed"])
        out.append(
            scan_situation(
                game.at_situation(config["situations"][name]), prof, "baseline",
                config["scan"]["grid"], config["solver"], rng, n_override=N_COARSE,
            )
        )
    return out


# ----------------------------------------------------------------------
# (4) 吸附
# ----------------------------------------------------------------------
def test_nearest_index_and_snap(scans):
    s = scans[0]
    assert nearest_index(s.w_s_axis, s.w_s_axis[3] + 1e-9) == 3
    i, j, k = snap_reference(s, {"w_s": 0.7, "kappa_s": 1.0, "kappa_e": 8.0})
    assert abs(s.w_s_axis[i] - 0.7) == np.min(np.abs(s.w_s_axis - 0.7))
    assert abs(s.kappa_s_axis[j] - 1.0) == np.min(np.abs(s.kappa_s_axis - 1.0))
    assert abs(s.kappa_e_axis[k] - 8.0) == np.min(np.abs(s.kappa_e_axis - 8.0))


# ----------------------------------------------------------------------
# (1) 集合基本性质
# ----------------------------------------------------------------------
def test_mask_contains_reference_and_monotone(scans, config):
    tol = config["equivalence"]["tol"]
    ref_idx = (4, 4, 4)
    q_list = [s.q_eq for s in scans]
    cums = cumulative_masks(q_list, ref_idx, tol)
    assert len(cums) == 3
    for m in cums:
        assert m[ref_idx]                      # 参考点自身恒在集合内
    assert cums[1].sum() <= cums[0].sum()      # 交集单调不增
    assert cums[2].sum() <= cums[1].sum()
    assert np.all(cums[2] <= cums[0])          # 集合包含关系


def test_nan_treated_as_nonequivalent():
    q = np.full((3, 3, 3), 0.5)
    q[0, 0, 0] = np.nan
    m = equivalence_mask(q, (1, 1, 1), 0.01)
    assert not m[0, 0, 0] and m[1, 1, 1]
    with pytest.raises(ValueError):
        equivalence_mask(q, (0, 0, 0), 0.01)   # NaN 参考点被拒绝


# ----------------------------------------------------------------------
# (2) 指标手算例
# ----------------------------------------------------------------------
def test_metrics_hand_example():
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1, 1, 1] = mask[1, 3, 1] = True       # 两点,第二轴索引差 2
    axes = (np.linspace(0, 1, 5),) * 3
    m = set_metrics(mask, axes)
    assert m["count"] == 2
    assert m["volume_fraction"] == pytest.approx(2 / 125)
    assert m["diameter"] == pytest.approx(0.5)  # (3-1)/4
    empty = set_metrics(np.zeros((5, 5, 5), bool), axes)
    assert empty["count"] == 0 and empty["diameter"] == 0.0


def test_components_hand_example():
    m = np.zeros((5, 5), dtype=bool)
    m[0, 0:2] = True                            # 分量 1
    m[3:5, 3] = True                            # 分量 2
    assert count_components_2d(m) == 2
    m[1:4, 1] = m[1, 0] = m[3, 2] = True        # 连成一片
    assert count_components_2d(m) == 1


# ----------------------------------------------------------------------
# (3) 双曲线预言:kappa 聚合量相同 -> 每个情境都等价
# ----------------------------------------------------------------------
def test_kappa_hyperbola_equivalence_in_every_situation(scans, config):
    """在每个 w_s 上取两组 (ks, ke),其聚合量 1/ks+1/ke 数值上接近,
    则两点的 q* 差应远小于等价 tol,且这在全部三个情境同时成立。"""
    s0 = scans[0]
    ks, ke = s0.kappa_s_axis, s0.kappa_e_axis
    inv = 1.0 / ks[:, None] + 1.0 / ke[None, :]
    # 找聚合量最接近的两个不同 (j,k) 对
    flat = inv.reshape(-1)
    order = np.argsort(flat)
    best_pair, best_gap = None, np.inf
    for a, b in zip(order[:-1], order[1:]):
        ja, ka = divmod(int(a), len(ke))
        jb, kb = divmod(int(b), len(ke))
        if (ja, ka) == (jb, kb):
            continue
        gap = abs(flat[a] - flat[b])
        if gap < best_gap:
            best_gap, best_pair = gap, ((ja, ka), (jb, kb))
    (ja, ka), (jb, kb) = best_pair
    tol = config["equivalence"]["tol"]
    for s in scans:                             # 每个情境
        for i in range(len(s.w_s_axis)):        # 每个 w_s
            dq = abs(s.q_eq[i, ja, ka] - s.q_eq[i, jb, kb])
            # 聚合量近似相同 -> q* 差远小于等价阈值
            assert dq < 0.1 * tol, (i, best_gap, dq)


def test_hyperbola_curve_passes_reference():
    ke = kappa_aggregate_hyperbola(np.array([2.0, 4.0]), 2.0, 8.0)
    assert ke[0] == pytest.approx(8.0)          # 曲线过参考点
    agg = 1 / 2 + 1 / 8
    assert 1 / 4 + 1 / ke[1] == pytest.approx(agg)