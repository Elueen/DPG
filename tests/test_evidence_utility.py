"""任务 D/E 测试:二维 safety evidence 与解析 utility(物理真值 + D4 单调性
+ 完整链式梯度互检,专门钉死 a_g/a_c/a_e chain-rule factors)。

  1) margin 物理算例:g_tilde = gap - d0 - h_ref*v(常量轨迹时 smin 精确);
     m_s = g_tilde - c_+^2/(2 b_ref)
  2) smin/smax 性质与温度极限
  3) D4 单调性:gap 增 -> u_s 增;c 增 -> u_s 减;progress 增 -> u_e 增
  4) 链式因子钉死:du/d zeta_g == a_g*(1-u^2)/s_s 精确;
     du/d zeta_c 与数值一致且含 sigmoid 因子;du/d zeta_e == (a_e/s_e)(1-u^2)
  5) 全网格解析 vs 数值(对 zeta)误差 <= 1e-6
  6) Y/H trade-off:接近工况 Δu_s>0 且 Δu_e<0
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.evidence import (
    branch_evidence,
    safety_margin_from_evidence,
    sigmoid,
    softmax_norm,
    softmin_norm,
    softplus,
)
from src.utility import (
    check_gradients,
    grad_u_e_zeta,
    grad_u_s_zeta,
    numeric_grad_s_zeta,
    u_e_raw,
    u_s_raw,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "step2_pilot.yaml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_margin_physical_values(cfg):
    e = cfg["evidence"]
    # 常量轨迹:gap=20, v=10, c=0 -> g_tilde = 20-2-9 = 9(smin 对常量精确)
    ev = branch_evidence(np.full(41, 20.0), np.full(41, 10.0),
                         np.zeros(41), e, dt=0.1)
    assert ev["g_tilde"] == pytest.approx(20.0 - e["d0_m"] - e["h_ref_s"] * 10.0,
                                          abs=1e-9)
    assert ev["c_tilde"] == pytest.approx(0.0, abs=1e-9)
    m = safety_margin_from_evidence(ev["g_tilde"], ev["c_tilde"], e)
    cp = float(softplus(0.0, e["softplus_beta"]))
    assert m == pytest.approx(ev["g_tilde"] - cp ** 2 / (2 * e["b_ref_mps2"]),
                              abs=1e-12)
    # d0 的作用:v=0 时安全边界在 gap=d0 而非 0
    ev0 = branch_evidence(np.full(41, e["d0_m"]), np.zeros(41),
                          np.zeros(41), e, dt=0.1)
    assert ev0["g_tilde"] == pytest.approx(0.0, abs=1e-9)


def test_softmin_softmax_properties(cfg):
    e = cfg["evidence"]
    m = np.array([10.0, 4.0, 7.0, 12.0])
    sm = softmin_norm(m, e["rho_g_m"])
    assert m.min() <= sm <= m.mean()
    assert softmin_norm(m, 0.05) == pytest.approx(m.min(), abs=0.1)
    c = np.array([-1.0, 2.5, 0.3, 1.0])
    sx = softmax_norm(c, e["rho_c_mps"])
    assert c.mean() <= sx <= c.max()
    assert softmax_norm(c, 0.02) == pytest.approx(c.max(), abs=0.1)


def test_monotonicity_directions(cfg):
    e, u = cfg["evidence"], cfg["utility"]
    us = [u_s_raw(g, 1.0, e, u) for g in (0.0, 5.0, 15.0, 30.0)]
    assert np.all(np.diff(us) > 0)                     # g_tilde 增 -> u_s 增
    us2 = [u_s_raw(10.0, c, e, u) for c in (-2.0, 0.0, 2.0, 5.0)]
    assert np.all(np.diff(us2) < 0)                    # closing 增 -> u_s 减
    ue = [u_e_raw(z, 12.0, 4.0, u) for z in (20.0, 48.0, 70.0)]
    assert np.all(np.diff(ue) > 0)                     # progress 增 -> u_e 增


def test_chain_rule_factors_pinned(cfg):
    """专门钉死 a_g / a_c / a_e 链式因子。"""
    e, u, p = cfg["evidence"], cfg["utility"], cfg["perception_noise"]
    g_t, c_t = 8.0, 1.5
    us = u_s_raw(g_t, c_t, e, u)
    g = grad_u_s_zeta(g_t, c_t, e, u, p)
    # 解析:du/d zeta_g = a_g*(1-u^2)/s_s(精确闭式)
    assert g[0] == pytest.approx(p["a_g_m"] * (1 - us ** 2) / u["s_margin_m"],
                                 abs=1e-12)
    # du/d zeta_c 闭式含 c_+ * sigmoid(beta*c) / b_ref,且为负
    cp = float(softplus(c_t, e["softplus_beta"]))
    expect_c = (p["a_c_mps"] * (-(1 - us ** 2) / u["s_margin_m"])
                * cp * float(sigmoid(e["softplus_beta"] * c_t)) / e["b_ref_mps2"])
    assert g[1] == pytest.approx(expect_c, abs=1e-12)
    assert g[1] < 0
    # 数值互检逐分量
    n = numeric_grad_s_zeta(g_t, c_t, e, u, p)
    np.testing.assert_allclose(g, n, atol=1e-7)
    # efficiency:du/d zeta_e = (a_e/s_e)(1-u^2);a_e 因子显式(即使当前=1)
    ue = u_e_raw(50.0, 10.78, 4.0, u)
    ge = grad_u_e_zeta(50.0, 10.78, 4.0, u, p)
    assert ge[0] == pytest.approx(
        (p["a_e_m"] / u["s_progress_m"]) * (1 - ue ** 2), abs=1e-12)


def test_gradient_crosscheck_grid(cfg):
    e, u, p = cfg["evidence"], cfg["utility"], cfg["perception_noise"]
    err = check_gradients(e, u, p, v_ref=12.0, horizon_s=4.0,
                          g_grid=np.linspace(-10, 40, 11),
                          c_grid=np.linspace(-4, 6, 11),
                          ze_grid=np.linspace(0, 160, 17))
    assert err <= u["grad_check_tol"]


def test_branch_contrast_tradeoff(cfg):
    e, u = cfg["evidence"], cfg["utility"]
    t = np.arange(41) * 0.1
    v_ego, v0 = 8.0, 11.0
    v_h = np.full_like(t, v0)
    gap_h = 12.0 + (v_ego - v0) * t
    v_y = v0 - 2.0 * (t / 4.0)
    x_y = np.concatenate([[0.0], np.cumsum(0.5 * (v_y[1:] + v_y[:-1]) * 0.1)])
    gap_y = 12.0 + v_ego * t - x_y
    ev_h = branch_evidence(gap_h, v_h, v_h - v_ego, e, dt=0.1)
    ev_y = branch_evidence(gap_y, v_y, v_y - v_ego, e, dt=0.1)
    du_s = (u_s_raw(ev_y["g_tilde"], ev_y["c_tilde"], e, u)
            - u_s_raw(ev_h["g_tilde"], ev_h["c_tilde"], e, u))
    du_e = (u_e_raw(ev_y["z_e"], 12.0, 4.0, u)
            - u_e_raw(ev_h["z_e"], 12.0, 4.0, u))
    assert du_s > 0 and du_e < 0