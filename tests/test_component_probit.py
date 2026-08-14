"""C2 测试:component-probit 响应。

覆盖(纪律要求 4 中与 C2 相关的全部项):
  1) 解析路径 vs Monte Carlo 路径互检,容差与样本数来自 config
  2) 退化情形:kappa -> 大 时趋近确定性最优反应;kappa -> 小 时趋近均匀随机
  3) 概率公理:归一化、动作交换反对称
  4) 期望分量 payoff 的边际化对 ego / opponent 两个方向都正确
  5) 结构性事实:separable 噪声下 kappa 仅通过 sum(1/kappa) 进入选择概率
     (kappa 分量级不可辨识,且与情境 c 无关)——解析路径上精确成立
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.games.toy_2x2 import ToyGame2x2
from src.response.component_probit import (
    choice_probabilities,
    choice_probabilities_analytic,
    choice_probabilities_mc,
    marginal_component_payoffs,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "step1_toy_game.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def game(config) -> ToyGame2x2:
    return ToyGame2x2.from_config(config["game"])


# ----------------------------------------------------------------------
# (4) 边际化正确性
# ----------------------------------------------------------------------
def test_marginalization_both_players(game):
    q = np.array([0.3, 0.7])   # opponent 混合策略
    p = np.array([0.6, 0.4])   # ego 混合策略
    u_ego = game.component_payoffs("ego")
    u_opp = game.component_payoffs("opponent")

    got_ego = marginal_component_payoffs(u_ego, q, "ego")
    expect_ego = u_ego[:, :, 0] * q[0] + u_ego[:, :, 1] * q[1]
    np.testing.assert_allclose(got_ego, expect_ego)

    got_opp = marginal_component_payoffs(u_opp, p, "opponent")
    expect_opp = u_opp[:, 0, :] * p[0] + u_opp[:, 1, :] * p[1]
    np.testing.assert_allclose(got_opp, expect_opp)


# ----------------------------------------------------------------------
# (3) 概率公理
# ----------------------------------------------------------------------
def test_normalization_and_swap_antisymmetry(game):
    w = np.array([0.6, 0.4])
    kappa = np.array([2.0, 5.0])
    q = np.array([0.5, 0.5])
    ubar = marginal_component_payoffs(game.component_payoffs("ego"), q, "ego")

    probs = choice_probabilities_analytic(w, kappa, ubar)
    assert probs.shape == (2,)
    assert np.all(probs >= 0) and probs.sum() == pytest.approx(1.0)

    swapped = choice_probabilities_analytic(w, kappa, ubar[:, ::-1])
    np.testing.assert_allclose(swapped, probs[::-1], rtol=1e-12)


# ----------------------------------------------------------------------
# (2) 退化情形
# ----------------------------------------------------------------------
def test_high_precision_approaches_best_response(game):
    w = np.array([0.5, 0.5])
    q = np.array([0.4, 0.6])
    ubar = marginal_component_payoffs(game.component_payoffs("ego"), q, "ego")
    best = int(np.argmax(w @ ubar))
    probs = choice_probabilities_analytic(w, np.array([1e8, 1e8]), ubar)
    assert probs[best] > 1.0 - 1e-6


def test_low_precision_approaches_uniform(game):
    w = np.array([0.5, 0.5])
    q = np.array([0.4, 0.6])
    ubar = marginal_component_payoffs(game.component_payoffs("ego"), q, "ego")
    probs = choice_probabilities_analytic(w, np.array([1e-8, 1e-8]), ubar)
    np.testing.assert_allclose(probs, [0.5, 0.5], atol=1e-5)


# ----------------------------------------------------------------------
# (5) 结构性事实:kappa 仅通过 sum(1/kappa) 进入
# ----------------------------------------------------------------------
def test_kappa_enters_only_through_aggregate(game, config):
    """1/k_s + 1/k_e 相同的任意 (k_s, k_e) 给出逐位相同的选择概率,
    且该性质在所有情境 c 下都成立(即情境变异无法分离 kappa 分量)。"""
    w = np.array([0.7, 0.3])
    q = np.array([0.35, 0.65])
    agg = 1.0 / 2.0 + 1.0 / 8.0  # 参考:kappa = (2, 8)
    kappa_variants = [
        np.array([2.0, 8.0]),
        np.array([8.0, 2.0]),
        np.array([1.0 / (agg / 2), 1.0 / (agg / 2)]),   # 等分
        np.array([1.0 / (agg * 0.9), 1.0 / (agg * 0.1)]),
    ]
    for c in config["situations"].values():
        g = game.at_situation(c)
        ubar = marginal_component_payoffs(g.component_payoffs("ego"), q, "ego")
        ref = choice_probabilities_analytic(w, kappa_variants[0], ubar)
        for kappa in kappa_variants[1:]:
            got = choice_probabilities_analytic(w, kappa, ubar)
            np.testing.assert_allclose(got, ref, rtol=1e-12)


# ----------------------------------------------------------------------
# (1) 解析 vs MC 互检(容差/样本数/seed 来自 config)
# ----------------------------------------------------------------------
CROSS_CHECK_POINTS = [
    # (w_s, kappa_s, kappa_e, q_yield, situation_key, player)
    (0.5, 1.0, 1.0, 0.5, "balanced", "ego"),
    (0.7, 2.0, 8.0, 0.35, "safety_dominant", "ego"),
    (0.3, 8.0, 2.0, 0.65, "efficiency_dominant", "ego"),
    (0.9, 0.5, 4.0, 0.5, "balanced", "opponent"),
    (0.2, 20.0, 20.0, 0.8, "safety_dominant", "opponent"),
]


@pytest.mark.parametrize("point", CROSS_CHECK_POINTS)
def test_analytic_vs_mc_cross_check(game, config, point):
    w_s, k_s, k_e, q0, situation_key, player = point
    cc = config["response"]["cross_check"]
    rng = np.random.default_rng(config["meta"]["seed"])

    g = game.at_situation(config["situations"][situation_key])
    w = np.array([w_s, 1.0 - w_s])
    kappa = np.array([k_s, k_e])
    other_mixed = np.array([q0, 1.0 - q0])
    u = g.component_payoffs(player)

    p_analytic = choice_probabilities(w, kappa, other_mixed, u, player, "analytic")
    p_mc = choice_probabilities(
        w, kappa, other_mixed, u, player, "mc",
        n_samples=cc["mc_samples"], rng=rng, chunk_size=cc["chunk_size"],
    )
    np.testing.assert_allclose(p_mc, p_analytic, atol=cc["tol"])


# ----------------------------------------------------------------------
# 接口防呆
# ----------------------------------------------------------------------
def test_invalid_inputs_rejected(game):
    ubar = np.zeros((2, 2))
    with pytest.raises(ValueError):
        choice_probabilities_analytic([0.6, 0.6], [1.0, 1.0], ubar)   # w 未归一化
    with pytest.raises(ValueError):
        choice_probabilities_analytic([0.5, 0.5], [1.0, -1.0], ubar)  # kappa 非正
    with pytest.raises(ValueError):
        choice_probabilities_mc(
            [0.5, 0.5], [1.0, 1.0], ubar, n_samples=101,              # 非偶数
            rng=np.random.default_rng(0),
        )