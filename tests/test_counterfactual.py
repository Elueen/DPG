"""任务 C 测试:counterfactual 模板与 rollout(解析真值验证)。

  1) 升余弦解析式:无钳位时 dv(T)=a_peak*T/2、Δgap_end=a_peak*T^2/4
     (数值积分 vs 解析,0.1s 网格容差 0.02);max|a|=a_peak;
     max|jerk| 与 pi*a_peak/T 吻合
  2) hold 分支:gap_H(t)=gap0+(v_ego0-v_opp0)*t 精确;负 gap 允许并标记
  3) 速度地板:低速 opponent 触发钳位,v>=0,clamped_frames>0,
     Δgap_end < 解析值(钳位削弱让行)
  4) 敏感性网格:形状与单调性(a_peak 或 T 增大 -> Δgap_end 均值增大)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.counterfactual import rollout_pair, sensitivity_grid

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "step2_pilot.yaml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_yield_analytic_properties(cfg):
    c = cfg["counterfactual"]
    fr, s = rollout_pair(gap0=20.0, v_ego0=15.0, v_opp0=15.0,
                         cfg_c=c, dt=0.1)
    T, ap = c["horizon_s"], c["a_peak_mps2"]
    assert s["a_peak_realized"] == pytest.approx(ap, abs=1e-9)
    assert s["clamped_frames"] == 0
    # dv(T) = a_peak*T/2
    dv_end = fr["v_opp_Y"].iloc[-1] - 15.0
    assert dv_end == pytest.approx(-ap * T / 2, abs=0.02)
    # Δgap_end = a_peak*T^2/4
    assert s["delta_gap_end"] == pytest.approx(ap * T**2 / 4, abs=0.02)
    assert s["delta_gap_end"] == pytest.approx(
        s["delta_gap_end_analytic_unclamped"], abs=0.02)
    # jerk 上界解析吻合(离散估计略低于连续极值)
    assert s["max_jerk_realized"] == pytest.approx(
        s["max_jerk_analytic"], rel=0.05)


def test_hold_branch_exact_and_negative_gap(cfg):
    c = cfg["counterfactual"]
    # 接近中:opp 快 3 m/s,gap0=5 -> gap_H(4s) = 5 - 12 = -7
    fr, s = rollout_pair(gap0=5.0, v_ego0=10.0, v_opp0=13.0, cfg_c=c, dt=0.1)
    t = fr["t"].to_numpy()
    np.testing.assert_allclose(fr["gap_H"], 5.0 + (10.0 - 13.0) * t, atol=1e-9)
    assert s["gap_H_negative"] and s["min_gap_H"] == pytest.approx(-7.0, abs=1e-9)
    # yield 分支应缓解:gap_Y 终值 > gap_H 终值
    assert fr["gap_Y"].iloc[-1] > fr["gap_H"].iloc[-1]


def test_velocity_floor_clamp(cfg):
    c = cfg["counterfactual"]
    fr, s = rollout_pair(gap0=10.0, v_ego0=5.0, v_opp0=1.0, cfg_c=c, dt=0.1)
    assert s["clamped_frames"] > 0
    assert float(fr["v_opp_Y"].min()) >= 0.0
    # 钳位削弱让行:实际 Δgap_end < 无钳位解析值
    assert s["delta_gap_end"] < s["delta_gap_end_analytic_unclamped"] - 1e-6


def test_sensitivity_grid_shape_and_monotonicity(cfg):
    inter = pd.DataFrame([
        {"event_id": "e1", "segment": "s", "dataset": "d",
         "gap0": 20.0, "v_ego0": 15.0, "v_opp0": 15.0},
        {"event_id": "e2", "segment": "s", "dataset": "d",
         "gap0": 12.0, "v_ego0": 8.0, "v_opp0": 10.0},
    ])
    g = sensitivity_grid(inter, cfg)
    assert len(g) == 9
    base = g[(g["a_peak"] == 1.0) & (g["horizon_s"] == 4.0)].iloc[0]
    hi_a = g[(g["a_peak"] == 1.3) & (g["horizon_s"] == 4.0)].iloc[0]
    hi_T = g[(g["a_peak"] == 1.0) & (g["horizon_s"] == 5.0)].iloc[0]
    assert hi_a["delta_gap_end_mean"] > base["delta_gap_end_mean"]
    assert hi_T["delta_gap_end_mean"] > base["delta_gap_end_mean"]
    assert {"min_gap_Y_mean", "min_gap_H_mean"} <= set(g.columns)