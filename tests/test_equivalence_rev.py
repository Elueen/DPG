"""R4(新 C5)测试:等价脊线,M0a/M0/M1 同框。

判别量:kappa 切片(w_s 固定于参考格点)内累积等价集合的
Chebyshev 直径 d_n(n = context 数)。粗网格(17^3)上验证:

  1) q_grid 闭式广播与逐点 q_vector 一致
  2) M0a / M0:d_3 / d_1 > 0.85(所有家族)—— collapse 方向存活,
     context diversity 无法移除 structural collapse
  3) M1 strong:d_3 < 0.05 —— 恰好理论要求的 diversity 将其移除
  4) M1 insufficient:d_3 == d_1(负对照,分毫不动)
  5) M1 moderate:居中(0.05 < d_3 < 0.8 * d_1)—— weak identification
  6) 累积交集单调、参考点恒在集合内(继承第一轮性质)
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.analysis.direct_scan import build_grid, kappa_slice_diameter, q_grid, q_grids, snap
from src.analysis.equivalence import cumulative_masks
from src.games.contexts import PerceptionModel
from src.response.models import q_vector

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "step1_revision.yaml"
N_COARSE = 17


@pytest.fixture(scope="module")
def config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def families(config) -> dict:
    pm = PerceptionModel(config["perception"])
    return {f: pm.build_context_set(c) for f, c in config["context_families"].items()}


@pytest.fixture(scope="module")
def grid(config):
    return build_grid(config["equivalence_rev"]["grid"], n_override=N_COARSE)


@pytest.fixture(scope="module")
def ridx(config, grid):
    ref = config["equivalence_rev"]["reference_points"]["mid"]
    return snap(grid, ref)


def _diams(config, families, grid, ridx, model, fam):
    qs = q_grids(model, families[fam], grid)
    cums = cumulative_masks(qs, ridx, config["equivalence_rev"]["tol"])
    return [kappa_slice_diameter(m, ridx, grid) for m in cums], cums


# ----------------------------------------------------------------------
# (1) 闭式广播一致性
# ----------------------------------------------------------------------
def test_q_grid_matches_pointwise(config, families, grid):
    ctx = families["strong_diversity"][0]
    qg = q_grid("m1", ctx, grid)
    rng = np.random.default_rng(config["meta"]["seed"])
    for _ in range(10):
        i, j, k = rng.integers(0, N_COARSE, 3)
        theta = np.array([grid.w_s_axis[i], grid.kappa_s_axis[j], grid.kappa_e_axis[k]])
        assert qg[i, j, k] == pytest.approx(float(q_vector("m1", theta, [ctx])[0]), rel=1e-12)


# ----------------------------------------------------------------------
# (2)-(5) collapse 存活 / 移除
# ----------------------------------------------------------------------
def test_m0_lineage_collapse_survives_all_families(config, families, grid, ridx):
    for model in ("m0a", "m0"):
        for fam in families:
            d, _ = _diams(config, families, grid, ridx, model, fam)
            assert d[0] > 0.3, (model, fam)               # 单 context 有实脊线
            assert d[2] / d[0] > 0.85, (model, fam, d)    # 3 contexts 后基本原样


def test_m1_strong_removes_collapse(config, families, grid, ridx):
    d, _ = _diams(config, families, grid, ridx, "m1", "strong_diversity")
    assert d[2] < 0.05, d


def test_m1_insufficient_unchanged(config, families, grid, ridx):
    d, _ = _diams(config, families, grid, ridx, "m1", "insufficient_diversity")
    assert d[0] > 0.3
    assert d[2] == pytest.approx(d[0], abs=1e-12)


def test_m1_moderate_partial(config, families, grid, ridx):
    d, _ = _diams(config, families, grid, ridx, "m1", "moderate_diversity")
    assert 0.05 < d[2] < 0.8 * d[0], d


# ----------------------------------------------------------------------
# (6) 继承性质
# ----------------------------------------------------------------------
def test_inclusion_and_reference_membership(config, families, grid, ridx):
    for model in ("m0a", "m0", "m1"):
        qs = q_grids(model, families["strong_diversity"], grid)
        cums = cumulative_masks(qs, ridx, config["equivalence_rev"]["tol"])
        for m in cums:
            assert m[ridx]
        assert np.all(cums[2] <= cums[1]) and np.all(cums[1] <= cums[0])