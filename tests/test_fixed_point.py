"""C3 测试:damped fixed-point 求解器。

覆盖(纪律要求 4 中与 C3 相关的全部项):
  1) 退化情形 kappa -> 大:趋近确定性最优反应均衡。当前 config payoff 在
     w=(0.5,0.5) 下的加权博弈为协调结构,有两个纯 Nash:
     (merge, yield) 与 (wait, hold)。多起点应恰好找到这两个;
     部分起点(含均匀起点)会落入 (merge,hold)<->(wait,yield) 反协调振荡
     不收敛——这是内部混合均衡在极端 kappa 下不稳定的已知现象,
     多起点机制应将其排除(见 docs/step1_findings.md)。
  2) 退化情形 kappa -> 小:唯一均衡趋近均匀随机。
  3) 自洽性:收敛点回代响应映射,残差与求解容差同量级。
  4) 收敛诊断:残差轨迹长度、终值、收敛标志一致。
  5) 多起点确定性:同 seed 结果逐位可复现。
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.games.toy_2x2 import ToyGame2x2
from src.solvers.fixed_point import (
    _response_map,
    solve_fixed_point,
    solve_multi_start,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "step1_toy_game.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def game(config) -> ToyGame2x2:
    return ToyGame2x2.from_config(config["game"])


def _symmetric_profiles(kappa_value: float, w_s: float = 0.5) -> dict:
    w = np.array([w_s, 1.0 - w_s])
    kappa = np.array([kappa_value, kappa_value])
    return {"ego": (w, kappa), "opponent": (w, kappa)}


# ----------------------------------------------------------------------
# (1) kappa -> 大:两个纯 Nash
# ----------------------------------------------------------------------
def test_high_kappa_finds_both_pure_nash(game, config):
    scfg = config["solver"]
    rng = np.random.default_rng(config["meta"]["seed"])
    ms = solve_multi_start(game, _symmetric_profiles(1e6), scfg, rng)

    # 稳定均衡恰好两个,分别贴近 (merge, yield) 与 (wait, hold)
    assert ms.n_unique == 2
    pure = {
        "merge_yield": np.array([1.0, 0.0, 1.0, 0.0]),
        "wait_hold": np.array([0.0, 1.0, 0.0, 1.0]),
    }
    matched = set()
    for eq in ms.equilibria:
        for name, target in pure.items():
            if np.max(np.abs(eq - target)) < 1e-3:
                matched.add(name)
    assert matched == set(pure)

    # 大多数起点应收敛(振荡起点少数),且未收敛属预期现象
    n_conv = sum(r.converged for r in ms.runs)
    assert n_conv >= 10
    assert not ms.all_converged  # 均匀起点落入反协调振荡(现象记录于 findings)


# ----------------------------------------------------------------------
# (2) kappa -> 小:唯一均匀均衡
# ----------------------------------------------------------------------
def test_low_kappa_unique_uniform(game, config):
    scfg = config["solver"]
    rng = np.random.default_rng(config["meta"]["seed"])
    ms = solve_multi_start(game, _symmetric_profiles(1e-6), scfg, rng)
    assert ms.all_converged
    assert ms.n_unique == 1
    np.testing.assert_allclose(ms.equilibria[0], 0.5, atol=1e-4)


# ----------------------------------------------------------------------
# (3) 自洽性(三个情境、中等 kappa)
# ----------------------------------------------------------------------
def test_self_consistency_across_situations(game, config):
    scfg = config["solver"]
    profiles = _symmetric_profiles(3.0, w_s=0.6)
    for c in config["situations"].values():
        g = game.at_situation(c)
        r = solve_fixed_point(g, profiles, scfg["damping"], scfg["tol"], scfg["max_iter"])
        assert r.converged
        tp, tq = _response_map(g, profiles, r.ego_strategy, r.opponent_strategy)
        residual = max(
            np.max(np.abs(tp - r.ego_strategy)),
            np.max(np.abs(tq - r.opponent_strategy)),
        )
        assert residual < 10 * scfg["tol"]


# ----------------------------------------------------------------------
# (4) 收敛诊断一致性
# ----------------------------------------------------------------------
def test_convergence_diagnostics(game, config):
    scfg = config["solver"]
    r = solve_fixed_point(
        game, _symmetric_profiles(3.0), scfg["damping"], scfg["tol"], scfg["max_iter"]
    )
    assert r.converged
    assert len(r.residuals) == r.n_iter <= scfg["max_iter"]
    assert r.residuals[-1] < scfg["tol"]
    assert r.residuals[0] > r.residuals[-1]
    # 策略是合法分布
    assert r.ego_strategy.sum() == pytest.approx(1.0)
    assert r.opponent_strategy.sum() == pytest.approx(1.0)
    assert np.all(r.ego_strategy >= 0) and np.all(r.opponent_strategy >= 0)


# ----------------------------------------------------------------------
# (5) 多起点确定性(同 seed 逐位复现)
# ----------------------------------------------------------------------
def test_multistart_deterministic(game, config):
    scfg = config["solver"]
    profiles = _symmetric_profiles(4.0)
    ms1 = solve_multi_start(
        game, profiles, scfg, np.random.default_rng(config["meta"]["seed"])
    )
    ms2 = solve_multi_start(
        game, profiles, scfg, np.random.default_rng(config["meta"]["seed"])
    )
    np.testing.assert_array_equal(ms1.equilibria, ms2.equilibria)
    assert [r.n_iter for r in ms1.runs] == [r.n_iter for r in ms2.runs]


# ----------------------------------------------------------------------
# 接口防呆
# ----------------------------------------------------------------------
def test_invalid_solver_params_rejected(game, config):
    with pytest.raises(ValueError):
        solve_fixed_point(game, _symmetric_profiles(1.0), damping=0.0, tol=1e-8, max_iter=10)
    bad_cfg = dict(config["solver"])
    bad_cfg["n_starts"] = 5  # 纪律要求 >= 10
    with pytest.raises(ValueError):
        solve_multi_start(
            game, _symmetric_profiles(1.0), bad_cfg, np.random.default_rng(0)
        )