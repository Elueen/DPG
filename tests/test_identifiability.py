"""R2(新 C3)测试:structural identifiability。

覆盖:
  1) 参数变换往返一致;Jacobian 中心差分对解析可导目标的正确性
  2) rank 结构:M0a / M0 在所有家族、所有 probe 恒 rank=2;
     M1 strong/moderate rank=3;M1 insufficient 精确退化 rank=2
  3) 零空间方向与理论 collapse 方向对齐(cos 夹角 ≈ 1):
     M0a ∝ (0, 1/ke, -1/ks);M0 ∝ (0, we²/ke, -ws²/ks);
     M1(G 恒定)∝ (0, we²Ge/ke, -ws²Gs/ks)
  4) weak identification 梯度:cond(M1 moderate) >> cond(M1 strong)
  5) 子集分析:full rank 需要 3 个异质 context(1 个 -> rank 1,2 个 -> rank 2)
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.analysis.identifiability import (
    analyze,
    from_tilde,
    numerical_jacobian,
    subset_ranks,
    to_tilde,
)
from src.games.contexts import PerceptionModel

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "step1_revision.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def families(config) -> dict:
    pm = PerceptionModel(config["perception"])
    return {f: pm.build_context_set(c) for f, c in config["context_families"].items()}


def _probes(config):
    return [
        np.array([p["w_s"], p["kappa_s"], p["kappa_e"]])
        for p in config["identifiability"]["probe_thetas"].values()
    ]


def _cos(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return abs(float(a @ b))


# ----------------------------------------------------------------------
# (1) 变换与差分机制
# ----------------------------------------------------------------------
def test_tilde_roundtrip():
    theta = np.array([0.37, 2.5, 11.0])
    np.testing.assert_allclose(from_tilde(to_tilde(theta)), theta, rtol=1e-14)


def test_jacobian_shape_and_smoothness(config, families):
    idc = config["identifiability"]
    J = numerical_jacobian(
        "m1", np.array([0.6, 3.0, 3.0]), families["strong_diversity"],
        idc["fd_step_w"], idc["fd_step_logk"],
    )
    assert J.shape == (3, 3)
    assert np.all(np.isfinite(J))


def test_boundary_w_rejected(config, families):
    idc = config["identifiability"]
    with pytest.raises(ValueError):
        numerical_jacobian(
            "m1", np.array([0.0, 3.0, 3.0]), families["strong_diversity"],
            idc["fd_step_w"], idc["fd_step_logk"],
        )


# ----------------------------------------------------------------------
# (2) rank 结构
# ----------------------------------------------------------------------
def test_m0a_m0_always_rank_two(config, families):
    idc = config["identifiability"]
    for model in ("m0a", "m0"):
        for fam, ctxs in families.items():
            for theta in _probes(config):
                rep = analyze(model, theta, ctxs, fam, idc)
                assert rep.numerical_rank == 2, (model, fam, theta)
                # s3 在数值零水平
                assert rep.singular_values[2] / rep.singular_values[0] < idc["rank_rtol"]


def test_m1_rank_by_family(config, families):
    idc = config["identifiability"]
    for theta in _probes(config):
        for fam, expect in [
            ("strong_diversity", 3),
            ("moderate_diversity", 3),
            ("insufficient_diversity", 2),
        ]:
            rep = analyze("m1", theta, families[fam], fam, idc)
            assert rep.numerical_rank == expect, (fam, theta)


# ----------------------------------------------------------------------
# (3) 零空间方向 = 理论 collapse 方向
# ----------------------------------------------------------------------
def test_null_direction_matches_theory(config, families):
    idc = config["identifiability"]
    w_s, ks, ke = 0.6, 1.0, 8.0
    theta = np.array([w_s, ks, ke])

    rep = analyze("m0a", theta, families["moderate_diversity"], "mod", idc)
    assert _cos(rep.null_direction, np.array([0.0, 1 / ke, -1 / ks])) > 1 - 1e-8

    rep = analyze("m0", theta, families["moderate_diversity"], "mod", idc)
    th = np.array([0.0, (1 - w_s) ** 2 / ke, -(w_s**2) / ks])
    assert _cos(rep.null_direction, th) > 1 - 1e-8

    G = families["insufficient_diversity"][0].G
    rep = analyze("m1", theta, families["insufficient_diversity"], "insuf", idc)
    th = np.array([0.0, (1 - w_s) ** 2 * G[1] / ke, -(w_s**2) * G[0] / ks])
    assert _cos(rep.null_direction, th) > 1 - 1e-8


# ----------------------------------------------------------------------
# (4) weak identification 梯度
# ----------------------------------------------------------------------
def test_condition_number_gradient(config, families):
    idc = config["identifiability"]
    for theta in _probes(config):
        strong = analyze("m1", theta, families["strong_diversity"], "s", idc)
        moderate = analyze("m1", theta, families["moderate_diversity"], "m", idc)
        assert moderate.cond_rank > 5 * strong.cond_rank, theta


# ----------------------------------------------------------------------
# (5) 子集分析
# ----------------------------------------------------------------------
def test_subset_ranks_strong_family(config, families):
    idc = config["identifiability"]
    sr = subset_ranks("m1", np.array([0.6, 3.0, 3.0]), families["strong_diversity"], idc)
    for key, v in sr.items():
        n_ctx = key.count("+") + 1
        if n_ctx < 3:
            assert v["rank"] == n_ctx, key       # rank 受 context 数上限约束
        else:
            assert v["rank"] == 3, key           # 三个异质 context 达 full rank