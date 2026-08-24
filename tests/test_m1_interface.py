"""任务 G 测试:m1_quantities 统一接口。

  1) 有限性与结构:四量有限、G_k >= 0、Y/H 中间量齐全
  2) alpha-linearity(实现正确性测试):Sigma -> alpha*Sigma 时
     G_k^(alpha) / G_k^(1) == alpha 精确(0.5 / 2.0)
  3) trade-off 符号:接近工况 du_s > 0、du_e < 0
  4) L 不是 (1-u^2)^2 的写死化简:安全通道 L_s 显式含 closing 分量
     (构造 c_tilde 高的状态,L_s 应严格大于仅 gap 分量的平方)
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.m1_interface import CanonicalState, m1_quantities
from src.utility import grad_u_s_zeta

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "step2_pilot.yaml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_finite_structure_and_signs(cfg):
    st = CanonicalState(gap0=12.0, v_ego0=8.0, v_opp0=11.0, event_id="t1")
    q = m1_quantities(st, v_ref=10.78, cfg=cfg)
    for k in ("du_s", "du_e", "G_s", "G_e"):
        assert np.isfinite(q[k])
    assert q["G_s"] >= 0 and q["G_e"] >= 0
    assert q["G_s"] > 0 and q["G_e"] > 0
    assert q["du_s"] > 0 and q["du_e"] < 0            # 接近工况 trade-off
    for br in ("Y", "H"):
        for name in ("g_tilde", "c_tilde", "z_e", "u_s", "u_e", "L_s", "L_e"):
            assert np.isfinite(q[f"{name}_{br}"])


def test_alpha_linearity_exact(cfg):
    st = CanonicalState(gap0=20.0, v_ego0=15.0, v_opp0=16.0)
    base = m1_quantities(st, v_ref=16.04, cfg=cfg, sigma_scale=1.0)
    for alpha in (0.5, 2.0):
        q = m1_quantities(st, v_ref=16.04, cfg=cfg, sigma_scale=alpha)
        assert q["G_s"] / base["G_s"] == pytest.approx(alpha, rel=1e-12)
        assert q["G_e"] / base["G_e"] == pytest.approx(alpha, rel=1e-12)
        # Δu 不受 Sigma 影响
        assert q["du_s"] == pytest.approx(base["du_s"], abs=1e-15)


def test_L_s_includes_closing_channel(cfg):
    """强接近工况的 L_s 必须含 closing 分量(> 仅 gap 分量平方)。"""
    st = CanonicalState(gap0=8.0, v_ego0=5.0, v_opp0=10.0)   # c0 = +5
    q = m1_quantities(st, v_ref=10.78, cfg=cfg)
    g = grad_u_s_zeta(q["g_tilde_H"], q["c_tilde_H"], cfg["evidence"],
                      cfg["utility"], cfg["perception_noise"])
    assert abs(g[1]) > 1e-6                            # closing 通道非零
    assert q["L_s_H"] == pytest.approx(g[0] ** 2 + g[1] ** 2, rel=1e-12)
    assert q["L_s_H"] > g[0] ** 2                      # 不是单通道化简