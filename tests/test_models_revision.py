"""Step 1 revision R1 测试:M0a / M0 / M1 三个 stochastic specification。

覆盖:
  1) G 计算机制:数值微分 vs 手写解析梯度;config 直写 G 被拒绝
  2) M0a 与第一轮 legacy 实现逐位一致(把 revision 与 round-1 代码钉在一起,
     不改动 round-1 代码)
  3) M0 structural no-go(解析层):固定 w_s 下,ws^2/ks + we^2/ke 相同的
     kappa 组合在所有家族、所有 context 上产生逐位相同的 q;偏离该
     level set 则 q 改变
  4) M0a no-go 同构验证(collapse 方向 1/ks + 1/ke,与 w 无关)
  5) M1:strong 家族下 M0 的 collapse 方向被打破(同 M0-aggregate 的
     kappa 对产生可分的 q 向量);insufficient 家族(G 恒定)下
     M1 重新退化到自身 aggregate 的精确 collapse
  6) 线性化 MC vs 解析一致(同一模型);全非线性 MC 的 Taylor 误差
     在工作区内有界且随 kappa 增大收敛
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.games.contexts import PerceptionModel, numerical_gradient, toy_function
from src.response.component_probit import choice_probabilities_analytic
from src.response.models import (
    q_choice,
    q_m1_mc_full,
    q_m1_mc_linearized,
    q_vector,
    sigma_eff_m0,
    sigma_eff_m0a,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "step1_revision.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def perception(config) -> PerceptionModel:
    return PerceptionModel(config["perception"])


@pytest.fixture(scope="module")
def families(config, perception) -> dict:
    return {
        fam: perception.build_context_set(ctxs)
        for fam, ctxs in config["context_families"].items()
    }


# ----------------------------------------------------------------------
# (1) G 计算机制
# ----------------------------------------------------------------------
def test_numerical_gradient_matches_analytic(config):
    fs_cfg = config["perception"]["f"]["safety"]
    fe_cfg = config["perception"]["f"]["efficiency"]
    f_s = toy_function("logistic", fs_cfg)
    f_e = toy_function("softplus", fe_cfg)
    step = config["perception"]["grad_step"]
    for z in [-1.3, -0.2, 0.0, 0.4, 1.7]:
        # logistic 解析梯度:A/s * sig * (1-sig)
        a, s, z0 = fs_cfg["amplitude"], fs_cfg["scale"], fs_cfg["shift"]
        sig = 1.0 / (1.0 + np.exp(-(z - z0) / s))
        assert numerical_gradient(f_s, z, step) == pytest.approx(a / s * sig * (1 - sig), rel=1e-6)
        # softplus 解析梯度:A * sigmoid
        a, s, z0 = fe_cfg["amplitude"], fe_cfg["scale"], fe_cfg["shift"]
        sig = 1.0 / (1.0 + np.exp(-(z - z0) / s))
        assert numerical_gradient(f_e, z, step) == pytest.approx(a * sig, rel=1e-6)


def test_config_written_g_rejected(perception):
    with pytest.raises(ValueError):
        perception.quantities(
            "bad", {"safety": {"a1": 0.1, "a0": 0.0}, "efficiency": {"a1": 0.1, "a0": 0.0},
                    "G_s": 1.0}
        )
    with pytest.raises(ValueError):
        perception.quantities(
            "bad2", {"safety": {"a1": 0.1, "a0": 0.0, "G": 1.0},
                     "efficiency": {"a1": 0.1, "a0": 0.0}}
        )


def test_insufficient_family_g_bitwise_constant(families):
    ctxs = families["insufficient_diversity"]
    for c in ctxs[1:]:
        # 相同至数值微分精度(符号翻转技巧;中心差分有 ~1e-12 浮点不对称)
        np.testing.assert_allclose(c.G, ctxs[0].G, rtol=1e-10)
    # 但 Δu_s 跨 context 变化(context 仍可分)
    du_s = [c.delta_u[0] for c in ctxs]
    assert len({round(x, 12) for x in du_s}) == 3


def test_strong_family_g_ratio_varies(families):
    ratios = [c.g_ratio for c in families["strong_diversity"]]
    assert max(ratios) / min(ratios) > 1e3


# ----------------------------------------------------------------------
# (2) M0a 与 legacy 第一轮实现逐位一致
# ----------------------------------------------------------------------
def test_m0a_matches_legacy_round1_implementation(families):
    w_s, ks, ke = 0.55, 2.0, 7.0
    for ctx in families["strong_diversity"]:
        q_new = float(q_choice("m0a", w_s, ks, ke, ctx))
        # legacy 接口:expected payoffs (K, 2) = ctx.u,同一公式 Phi(w·Δu / sqrt(2Σ1/κ))
        q_old = choice_probabilities_analytic(
            np.array([w_s, 1 - w_s]), np.array([ks, ke]), ctx.u
        )[0]
        assert q_new == pytest.approx(float(q_old), rel=1e-12)


# ----------------------------------------------------------------------
# (3)(4) M0 / M0a structural no-go
# ----------------------------------------------------------------------
def _kappa_on_m0_level_set(w_s, ks_ref, ke_ref, ks_new):
    """给定新 ks,解出使 ws^2/ks + we^2/ke 不变的 ke。"""
    agg = w_s**2 / ks_ref + (1 - w_s) ** 2 / ke_ref
    inv = (agg - w_s**2 / ks_new) / (1 - w_s) ** 2
    assert inv > 0
    return 1.0 / inv


def test_m0_no_go_all_families(families):
    w_s, ks_ref, ke_ref = 0.6, 3.0, 3.0
    # level set 可行域:ws^2/ks_new < agg  =>  ks_new > ws^2/agg ≈ 2.08
    for ks_new in [2.5, 5.0, 10.0]:
        ke_new = _kappa_on_m0_level_set(w_s, ks_ref, ke_ref, ks_new)
        assert sigma_eff_m0(w_s, ks_new, ke_new) == pytest.approx(
            float(sigma_eff_m0(w_s, ks_ref, ke_ref)), rel=1e-12
        )
        for ctxs in families.values():
            for ctx in ctxs:
                q_ref = float(q_choice("m0", w_s, ks_ref, ke_ref, ctx))
                q_new = float(q_choice("m0", w_s, ks_new, ke_new, ctx))
                assert q_new == pytest.approx(q_ref, rel=1e-12)
    # 偏离 level set:q 改变
    ctx = families["strong_diversity"][0]
    assert abs(
        float(q_choice("m0", w_s, ks_ref, ke_ref, ctx))
        - float(q_choice("m0", w_s, 30.0, 30.0, ctx))
    ) > 1e-3


def test_m0a_no_go_direction_is_w_independent(families):
    ks_ref, ke_ref = 2.0, 8.0
    ks_new = 8.0
    ke_new = 1.0 / (1 / ks_ref + 1 / ke_ref - 1 / ks_new)  # 1/ks+1/ke 不变
    for w_s in [0.2, 0.5, 0.8]:
        for ctx in families["moderate_diversity"]:
            q_ref = float(q_choice("m0a", w_s, ks_ref, ke_ref, ctx))
            q_new = float(q_choice("m0a", w_s, ks_new, ke_new, ctx))
            assert q_new == pytest.approx(q_ref, rel=1e-12)


# ----------------------------------------------------------------------
# (5) M1:collapse 的打破与重新退化
# ----------------------------------------------------------------------
def test_m1_breaks_m0_collapse_under_strong_diversity(families):
    """同一 M0-aggregate 的 kappa 对,在 strong 家族下 M1 的 q 向量可分。"""
    w_s, ks_ref, ke_ref = 0.6, 3.0, 3.0
    ks_new = 10.0
    ke_new = _kappa_on_m0_level_set(w_s, ks_ref, ke_ref, ks_new)
    q_ref = q_vector("m1", np.array([w_s, ks_ref, ke_ref]), families["strong_diversity"])
    q_new = q_vector("m1", np.array([w_s, ks_new, ke_new]), families["strong_diversity"])
    assert np.max(np.abs(q_ref - q_new)) > 0.01   # 超过等价 tol 量级


def test_m1_re_degenerates_under_insufficient_diversity(families):
    """G 恒定时,M1 的 collapse 方向 = ws^2*Gs/ks + we^2*Ge/ke = C。"""
    ctxs = families["insufficient_diversity"]
    G = ctxs[0].G
    w_s, ks_ref, ke_ref = 0.6, 3.0, 3.0
    agg = w_s**2 * G[0] / ks_ref + (1 - w_s) ** 2 * G[1] / ke_ref
    # 可行域:ws^2*Gs/ks_new < agg  =>  ks_new > ws^2*Gs/agg ≈ 2.24
    for ks_new in [2.5, 8.0]:
        inv = (agg - w_s**2 * G[0] / ks_new) / ((1 - w_s) ** 2 * G[1])
        assert inv > 0
        ke_new = 1.0 / inv
        q_ref = q_vector("m1", np.array([w_s, ks_ref, ke_ref]), ctxs)
        q_new = q_vector("m1", np.array([w_s, ks_new, ke_new]), ctxs)
        # 精确至 G 的数值微分精度
        np.testing.assert_allclose(q_new, q_ref, rtol=1e-9)


# ----------------------------------------------------------------------
# (6) 数值实现一致性 + Taylor 误差有界
# ----------------------------------------------------------------------
def test_linearized_mc_matches_analytic(config, families):
    cc = config["consistency_checks"]["lin_mc"]
    ctx = families["strong_diversity"][2]  # mixed
    w_s, ks, ke = 0.6, 3.0, 3.0
    qa = float(q_choice("m1", w_s, ks, ke, ctx))
    rng = np.random.default_rng(config["meta"]["seed"])
    qm = q_m1_mc_linearized(w_s, ks, ke, ctx, cc["n_samples"], rng)
    assert qm == pytest.approx(qa, abs=cc["tol"])


def test_full_nonlinear_mc_taylor_error_bounded_and_shrinking(config, perception, families):
    cc = config["consistency_checks"]["full_mc"]
    ctx = families["strong_diversity"][2]
    w_s = 0.6
    gaps = []
    for k in cc["kappa_probe"]:
        qa = float(q_choice("m1", w_s, k, k, ctx))
        rng = np.random.default_rng(config["meta"]["seed"])
        qf = q_m1_mc_full(w_s, k, k, ctx, perception, cc["n_samples"], rng)
        gap = abs(qf - qa)
        gaps.append(gap)
        assert gap < cc["tol_working_range"]
    assert gaps[-1] < gaps[0]   # kappa 增大,Taylor 误差收敛