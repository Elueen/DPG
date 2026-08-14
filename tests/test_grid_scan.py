"""C4 测试:向量化网格扫描。

关键项:向量化批量不动点与 C3 参考标量求解器在随机抽取的网格点上
逐位互检(同为均匀起点、同一 damping/tol,应一致到求解容差量级)。
另测网格轴构造、结果形状/取值合法性、同 seed 确定性、角色正确性
(ego 画像变动应改变均衡 —— 防止把固定方与扫描方写反)。
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.analysis.grid_scan import build_axis, scan_situation
from src.games.toy_2x2 import ToyGame2x2
from src.solvers.fixed_point import solve_fixed_point

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "step1_toy_game.yaml"

N_COARSE = 7          # 测试用粗网格(纯为速度;科学网格密度以 config 为准)
N_CROSS_CHECK = 15    # 与参考求解器互检的随机点数


@pytest.fixture(scope="module")
def config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def game(config) -> ToyGame2x2:
    return ToyGame2x2.from_config(config["game"])


@pytest.fixture(scope="module")
def scan_result(config, game):
    c = config["situations"]["balanced"]
    prof = config["scan"]["ego_profiles"][config["scan"]["main_ego_profile"]]
    rng = np.random.default_rng(config["meta"]["seed"])
    return scan_situation(
        game.at_situation(c), prof, "baseline",
        config["scan"]["grid"], config["solver"], rng, n_override=N_COARSE,
    )


def test_build_axis(config):
    grid = config["scan"]["grid"]
    w = build_axis(grid["w_s"], 11)
    assert w[0] == grid["w_s"]["min"] and w[-1] == grid["w_s"]["max"] and len(w) == 11
    ks = build_axis(grid["kappa_s"], 11)
    # log 轴:相邻比值恒定
    np.testing.assert_allclose(np.diff(np.log(ks)), np.diff(np.log(ks))[0])
    with pytest.raises(ValueError):
        build_axis({"min": -1.0, "max": 2.0, "n": 5, "spacing": "log"})


def test_scan_shapes_and_validity(scan_result):
    shape = (N_COARSE, N_COARSE, N_COARSE)
    for arr in (
        scan_result.p_eq, scan_result.q_eq, scan_result.n_unique,
        scan_result.n_converged, scan_result.fallback,
    ):
        assert arr.shape == shape
    ok = ~np.isnan(scan_result.p_eq)
    assert np.all((scan_result.p_eq[ok] >= 0) & (scan_result.p_eq[ok] <= 1))
    assert np.all((scan_result.q_eq[ok] >= 0) & (scan_result.q_eq[ok] <= 1))
    assert np.all(scan_result.n_unique >= 0)
    # 该参数范围应全部收敛且均衡唯一(若变化,先查 config 网格范围是否改动)
    assert ok.all()
    assert np.all(scan_result.n_unique == 1)


def test_vectorized_matches_reference_solver(config, game, scan_result):
    """随机网格点:向量化规范均衡(均匀起点) vs C3 参考求解器(均匀起点)。"""
    scfg = config["solver"]
    prof = config["scan"]["ego_profiles"]["baseline"]
    c = config["situations"]["balanced"]
    g = game.at_situation(c)
    w_ego = np.asarray(prof["w"], float)
    k_ego = np.asarray(prof["kappa"], float)

    rng = np.random.default_rng(config["meta"]["seed"] + 1)
    idx = rng.integers(0, N_COARSE, size=(N_CROSS_CHECK, 3))
    for i, j, k in idx:
        w_s = scan_result.w_s_axis[i]
        kappa = np.array([scan_result.kappa_s_axis[j], scan_result.kappa_e_axis[k]])
        profiles = {
            "ego": (w_ego, k_ego),
            "opponent": (np.array([w_s, 1.0 - w_s]), kappa),
        }
        ref = solve_fixed_point(g, profiles, scfg["damping"], scfg["tol"], scfg["max_iter"])
        assert ref.converged
        np.testing.assert_allclose(scan_result.p_eq[i, j, k], ref.ego_strategy[0], atol=1e-9)
        np.testing.assert_allclose(scan_result.q_eq[i, j, k], ref.opponent_strategy[0], atol=1e-9)


def test_scan_deterministic(config, game, scan_result):
    c = config["situations"]["balanced"]
    prof = config["scan"]["ego_profiles"]["baseline"]
    rng = np.random.default_rng(config["meta"]["seed"])
    res2 = scan_situation(
        game.at_situation(c), prof, "baseline",
        config["scan"]["grid"], config["solver"], rng, n_override=N_COARSE,
    )
    np.testing.assert_array_equal(scan_result.p_eq, res2.p_eq)
    np.testing.assert_array_equal(scan_result.q_eq, res2.q_eq)
    np.testing.assert_array_equal(scan_result.n_unique, res2.n_unique)


def test_ego_profile_matters(config, game, scan_result):
    """角色防错:换 ego 画像应改变均衡场(若不变,说明固定方接错)。"""
    c = config["situations"]["balanced"]
    prof = config["scan"]["ego_profiles"]["safety_biased"]
    rng = np.random.default_rng(config["meta"]["seed"])
    res2 = scan_situation(
        game.at_situation(c), prof, "safety_biased",
        config["scan"]["grid"], config["solver"], rng, n_override=N_COARSE,
    )
    assert np.max(np.abs(res2.q_eq - scan_result.q_eq)) > 1e-3


def test_unsituated_game_rejected(config, game):
    prof = config["scan"]["ego_profiles"]["baseline"]
    with pytest.raises(ValueError):
        scan_situation(
            game, prof, "baseline",
            config["scan"]["grid"], config["solver"],
            np.random.default_rng(0), n_override=3,
        )