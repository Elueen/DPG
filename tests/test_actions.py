"""任务 B 后半测试:D1/D2/D3 标签(合成轨迹,已知真值)。

三个手工场景(h_resp=4s,t0=0):
  yield 场景:opp 自 t0 起恒减速 a=0.5 m/s^2
      dv_red = 0.5*4 = 2.0;a_end = 0.5*0.5*16 = 4.0 => 三定义全 yield
  hold 场景:opp 匀速 => dv_red=0, a_end=0 => 全 hold
  ambiguous 场景:opp 短暂小幅减速后恢复(削减 0.6 m/s,让出 ~0.9 m)
      => D1/D2/D3 全 ambiguous
另验:accommodation 解析式精确;±30% 缩放的翻转机制(构造贴阈值样本)。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.actions import (
    build_label_table,
    labels_for,
    response_features,
    scaled_thresholds,
    sensitivity_flips,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "step2_pilot.yaml"
DT = 0.1


@pytest.fixture(scope="module")
def cfg() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _opp_track(v_profile: np.ndarray) -> pd.DataFrame:
    t = np.round(np.arange(len(v_profile)) * DT, 6)
    x = np.concatenate([[0.0], np.cumsum(0.5 * (v_profile[1:] + v_profile[:-1]) * DT)])
    return pd.DataFrame({"t": t, "x": x, "vx": v_profile,
                         "ax": np.gradient(v_profile, DT)})


def test_yield_scenario_exact(cfg):
    th = cfg["actions"]
    v = 20.0 - 0.5 * np.arange(41) * DT          # 4s 恒减速 0.5
    f = response_features(_opp_track(v), 0.0, 4.0)
    assert f["dv_red"] == pytest.approx(2.0, abs=1e-9)
    assert f["a_end"] == pytest.approx(0.5 * 0.5 * 16, abs=1e-6)   # 4.0 m
    assert labels_for(f, th) == {"d1_decel": "yield", "d2_gap": "yield",
                                 "d3_compound": "yield"}


def test_hold_scenario(cfg):
    th = cfg["actions"]
    f = response_features(_opp_track(np.full(41, 20.0)), 0.0, 4.0)
    assert f["dv_red"] == 0.0 and abs(f["a_end"]) < 1e-9
    assert labels_for(f, th) == {"d1_decel": "hold", "d2_gap": "hold",
                                 "d3_compound": "hold"}


def test_ambiguous_scenario(cfg):
    th = cfg["actions"]
    # 1s 减速(-0.6) -> 0.5s 保持 -> 1s 加速回 -> 匀速
    v = np.concatenate([
        20.0 - 0.6 * np.arange(11) * DT,
        np.full(5, 19.4),
        19.4 + 0.6 * np.arange(1, 11) * DT,
        np.full(15, 20.0),
    ])
    f = response_features(_opp_track(v), 0.0, 4.0)
    assert 0.4 < f["dv_red"] < 1.0                 # D1 中间带
    assert 0.5 < f["a_end"] < 2.0                  # D2 中间带
    lab = labels_for(f, th)
    assert lab == {"d1_decel": "ambiguous", "d2_gap": "ambiguous",
                   "d3_compound": "ambiguous"}


def test_scaled_thresholds_flip_near_boundary(cfg):
    th = cfg["actions"]
    # dv_red 恰在 1.1(基线 yield 阈 1.0 之上,x1.3 后 1.3 之下 => 翻转)
    v = 20.0 - (1.1 / 4.0) * np.arange(41) * DT
    f = response_features(_opp_track(v), 0.0, 4.0)
    assert labels_for(f, th)["d1_decel"] == "yield"
    assert labels_for(f, scaled_thresholds(th, 1.3))["d1_decel"] == "ambiguous"


def test_build_label_table_and_sensitivity(cfg):
    # 组一个最小 canonical df + interactions df,验证表构建与翻转统计管道
    v_yield = 20.0 - 0.5 * np.arange(61) * DT
    opp = _opp_track(v_yield)
    opp["segment"] = "seg"; opp["vid"] = 2
    for c in ("y", "vy", "ay"): opp[c] = 0.0
    opp["lane"] = 3; opp["length"] = 4.5; opp["width"] = 2.0
    for c, val in (("dataset", "synth"), ("recording", "r"), ("site", "s"),
                   ("regime", "test"), ("vclass", "Car"), ("preceding", 0),
                   ("following", 0)):
        opp[c] = val
    inter = pd.DataFrame([{
        "dataset": "synth", "segment": "seg", "regime": "test",
        "event_id": "seg|1|6.0", "onset_def": "fixed_lead", "ego_vid": 1,
        "opp_vid": 2, "t_cross": 6.0, "t0": 0.0, "lane_from": 2, "lane_to": 3,
        "gap0": 25.0, "dv0": 0.0, "v_ego0": 20.0, "v_opp0": 20.0,
        "valid": True, "invalid_reason": "",
    }])
    tab = build_label_table(opp, inter, cfg)
    assert len(tab) == 1
    assert tab.iloc[0]["d1_decel"] == "yield"
    flips = sensitivity_flips(tab)
    assert set(flips) == {"d1_decel", "d2_gap", "d3_compound"}
    # 强 yield 场景对 ±30% 稳健(不翻转)
    assert flips["d1_decel"]["flip_hi"] == 0.0