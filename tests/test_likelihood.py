"""C6 测试:观测长度模拟。

关键项:
  1) 预算拆分:总和守恒、余数分配规则
  2) 网格似然手算例:Binomial logL 逐位正确;N=0 情境零贡献
  3) 置信集合性质:含 MLE(网格 argmax);大 N 下含真参数;
     体积随 N 中位数意义上收缩;多情境集合不大于单情境(中位数)
  4) 同 seed 确定性
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.step1.run_likelihood import (
    confidence_mask,
    grid_log_likelihood,
    simulate_counts,
    split_budget,
)
from src.analysis.equivalence import normalized_coords, set_metrics, snap_reference
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
# (1) 预算拆分
# ----------------------------------------------------------------------
def test_split_budget():
    assert split_budget(10, 3) == [4, 3, 3]
    assert split_budget(9, 3) == [3, 3, 3]
    assert split_budget(1, 3) == [1, 0, 0]
    for n in [10, 30, 100, 300, 1000]:
        assert sum(split_budget(n, 3)) == n


# ----------------------------------------------------------------------
# (2) 似然手算例
# ----------------------------------------------------------------------
def test_grid_log_likelihood_hand_example():
    q1 = np.array([[0.2, 0.5]])
    q2 = np.array([[0.9, 0.5]])
    logl = grid_log_likelihood([q1, q2], counts=[3, 1], n_per_situation=[10, 2],
                               prob_clip=1e-12)
    expect_00 = 3 * np.log(0.2) + 7 * np.log(0.8) + 1 * np.log(0.9) + 1 * np.log(0.1)
    expect_01 = 4 * np.log(0.5) + 8 * np.log(0.5)
    np.testing.assert_allclose(logl[0, 0], expect_00)
    np.testing.assert_allclose(logl[0, 1], expect_01)
    # N=0 情境零贡献
    logl2 = grid_log_likelihood([q1, q2], counts=[3, 0], n_per_situation=[10, 0],
                                prob_clip=1e-12)
    np.testing.assert_allclose(
        logl2[0, 0], 3 * np.log(0.2) + 7 * np.log(0.8)
    )


def test_simulate_counts_bounds_and_determinism():
    q = [0.3, 0.7, 0.5]
    n = [10, 0, 5]
    c1 = simulate_counts(q, n, np.random.default_rng(7))
    c2 = simulate_counts(q, n, np.random.default_rng(7))
    assert c1 == c2
    assert all(0 <= k <= m for k, m in zip(c1, n))
    assert c1[1] == 0


# ----------------------------------------------------------------------
# (3) 置信集合性质
# ----------------------------------------------------------------------
def _metrics_for(scans, config, ref_idx, n_per, seed):
    q_grids = [s.q_eq for s in scans]
    q_true = [float(q[ref_idx]) for q in q_grids]
    rng = np.random.default_rng(seed)
    counts = simulate_counts(q_true, n_per, rng)
    logl = grid_log_likelihood(q_grids, counts, n_per,
                               config["observation"]["prob_clip"])
    mask = confidence_mask(logl, config["observation"]["confidence_level"])
    return logl, mask


def test_confidence_set_contains_mle_and_truth_at_large_n(scans, config):
    ref_idx = snap_reference(scans[0], config["equivalence"]["reference_points"]["cautious_high_prec"])
    logl, mask = _metrics_for(scans, config, ref_idx, [2000, 2000, 2000],
                              config["meta"]["seed"])
    mle_idx = np.unravel_index(np.argmax(logl), logl.shape)
    assert mask[mle_idx]                       # 集合恒含网格 MLE
    assert mask[ref_idx]                       # 大 N 下含真参数(似然比覆盖)


def test_volume_shrinks_with_n_median(scans, config):
    """中位数意义上体积随 N 收缩,且多情境不大于单情境。"""
    ref_idx = snap_reference(scans[0], config["equivalence"]["reference_points"]["cautious_high_prec"])
    norm_axes = normalized_coords(scans[0])
    n_rep = 11

    def median_volume(n_per, seed0):
        vols = []
        for r in range(n_rep):
            _, mask = _metrics_for(scans, config, ref_idx, n_per, seed0 + r)
            vols.append(set_metrics(mask, norm_axes)["volume_fraction"])
        return float(np.median(vols))

    v_small = median_volume([10, 0, 0], 100)
    v_large = median_volume([1000, 0, 0], 200)
    assert v_large < v_small                   # 单情境:N 增大 -> 收缩

    v_single = median_volume([300, 0, 0], 300)
    v_multi = median_volume([100, 100, 100], 400)
    assert v_multi <= v_single                 # 同预算:多情境不差于单情境