"""C1 测试:博弈构造正确性与情境缩放的三条不变量。

不变量(对每个玩家分别成立):
  1) at_situation(c) 后 disp(分量0)/disp(分量1) == c
  2) 两分量离散度几何均值保持不变(整体尺度不变)
  3) 各分量均值保持不变(mean-preserving)
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.games.toy_2x2 import PLAYERS, ToyGame2x2

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "step1_toy_game.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def game(config) -> ToyGame2x2:
    return ToyGame2x2.from_config(config["game"])


def test_shape_and_labels(game):
    assert game.payoffs.shape == (2, 2, 2, 2)
    assert game.ego_actions == ("merge", "wait")
    assert game.opponent_actions == ("yield", "hold")
    assert game.components == ("safety", "efficiency")
    assert game.situation_c is None


def test_component_payoffs_matches_config(game, config):
    for p in PLAYERS:
        for k_idx, k in enumerate(game.components):
            expected = np.array(config["game"]["payoffs"][p][k], dtype=float)
            np.testing.assert_allclose(game.component_payoffs(p)[k_idx], expected)


@pytest.mark.parametrize("c", [0.25, 0.3333333333, 1.0, 3.0, 10.0])
def test_situation_invariants(game, c):
    g = game.at_situation(c)
    s0, s = game.dispersions(), g.dispersions()
    # (1) 离散度之比 == c
    np.testing.assert_allclose(s[:, 0] / s[:, 1], c, rtol=1e-12)
    # (2) 几何均值不变
    np.testing.assert_allclose(s.prod(axis=1), s0.prod(axis=1), rtol=1e-12)
    # (3) 均值不变
    np.testing.assert_allclose(
        g.payoffs.mean(axis=(2, 3)), game.payoffs.mean(axis=(2, 3)), atol=1e-12
    )
    assert g.situation_c == pytest.approx(c)


def test_situation_identity_at_base_ratio(game):
    """c 取基础矩阵自身的离散度比时,几何均值约束下应还原基础矩阵。

    注意:该性质仅当两个玩家的基础离散度比相同时才对双方同时成立,
    因此这里对每个玩家分别取其自身比值检验对应玩家的还原。
    """
    ratios = game.dispersion_ratio()
    for p_idx in range(len(PLAYERS)):
        g = game.at_situation(ratios[p_idx])
        np.testing.assert_allclose(
            g.payoffs[p_idx], game.payoffs[p_idx], rtol=1e-10, atol=1e-12
        )


def test_situations_in_config_are_valid(game, config):
    for name, c in config["situations"].items():
        g = game.at_situation(c)
        assert np.all(np.isfinite(g.payoffs)), name


def test_invalid_c_rejected(game):
    for bad in [0.0, -1.0, float("inf"), float("nan")]:
        with pytest.raises(ValueError):
            game.at_situation(bad)