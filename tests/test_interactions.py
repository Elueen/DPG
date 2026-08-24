"""任务 B 前半测试:lane-change 事件、onset 三候选、opponent 选择(合成数据)。

场景构造(10Hz canonical 表,单 segment,两车道 y=4/y=8):
  ego(vid 1):lane 2 -> 3,t_cross=6.0s;5.0s 起 vy=+0.5 朝目标
             (更早时刻 vy=0)=> lat_vel/compound 的 t0=5.0,
             fixed_lead 的 t0=4.0
  opp(vid 2):lane 3,ego 后方 25m,匀速 => 应被选中且 valid
  far(vid 3):lane 3,ego 后方 60m(比 vid2 远)=> 不应被选中
  前车(vid 4):lane 3,ego 前方 => 不是 opponent
覆盖:事件检测与合并、三 onset 的 t0、toward 方向、opponent 唯一性、
      validity 全绿;再构造 opponent 中途换道的变体 => invalid_reason 正确。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.interactions import detect_lane_changes, extract_interactions

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "step2_pilot.yaml"
DT = 0.1


@pytest.fixture(scope="module")
def cfg() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _vehicle(vid, lane_arr, y_arr, x0, v, length=4.5, prec=0):
    n = len(lane_arr)
    t = np.round(np.arange(n) * DT, 6)
    return pd.DataFrame({
        "dataset": "synth", "segment": "seg", "recording": "r", "site": "s",
        "regime": "test", "vid": vid, "t": t,
        "x": x0 + v * t, "y": y_arr, "vx": v,
        "vy": np.gradient(y_arr, DT), "ax": 0.0, "ay": 0.0,
        "lane": lane_arr, "length": length, "width": 2.0, "vclass": "Car",
        "preceding": prec, "following": 0,
    })


@pytest.fixture(scope="module")
def synth(cfg) -> pd.DataFrame:
    n = 120                       # 12 s
    t = np.arange(n) * DT
    # ego:t<5 直行 lane2(y=4);5<=t<6 以 vy=+0.5 漂向 lane3;t>=6 在 lane3
    y_ego = np.where(t < 5.0, 4.0, np.where(t < 6.0, 4.0 + 0.5 * (t - 5.0), 4.5))
    y_ego = np.where(t >= 6.0, 4.5 + np.minimum(0.5 * (t - 6.0), 3.5), y_ego)
    lane_ego = np.where(t < 6.0, 2, 3)
    ego = _vehicle(1, lane_ego, y_ego, x0=100.0, v=20.0)
    opp = _vehicle(2, np.full(n, 3), np.full(n, 8.0), x0=70.0, v=20.0)
    far = _vehicle(3, np.full(n, 3), np.full(n, 8.0), x0=35.0, v=20.0)
    lead = _vehicle(4, np.full(n, 3), np.full(n, 8.0), x0=160.0, v=20.0)
    # 车道中心锚定车(保证 lane_centers 有充足观测)
    anchor2 = _vehicle(5, np.full(n, 2), np.full(n, 4.0), x0=300.0, v=20.0)
    return pd.concat([ego, opp, far, lead, anchor2], ignore_index=True)


def test_event_detection_and_merge(synth, cfg):
    events = detect_lane_changes(synth, cfg["interaction"]["merge_window_s"])
    ego_events = [e for e in events if e.ego_vid == 1]
    assert len(ego_events) == 1
    e = ego_events[0]
    assert e.lane_from == 2 and e.lane_to == 3
    assert e.t_cross == pytest.approx(6.0, abs=DT)


def test_onsets_and_opponent(synth, cfg):
    tab = extract_interactions(synth, cfg)
    tab = tab[tab["ego_vid"] == 1].set_index("onset_def")
    assert set(tab.index) == {"fixed_lead", "lat_vel", "compound"}
    # onset t0
    assert tab.loc["fixed_lead", "t0"] == pytest.approx(4.0, abs=DT)
    assert tab.loc["lat_vel", "t0"] == pytest.approx(5.0, abs=2 * DT)
    assert tab.loc["compound", "t0"] == pytest.approx(5.0, abs=2 * DT)
    # opponent:唯一且是最近后车 vid2(不是更远的 vid3,不是前车 vid4)
    for oname in ("fixed_lead", "lat_vel", "compound"):
        assert tab.loc[oname, "opp_vid"] == 2
        assert bool(tab.loc[oname, "valid"]), tab.loc[oname, "invalid_reason"]
    # gap0(fixed_lead,t0=4.0):x_ego=180, x_opp=150,gap=30-4.5=25.5
    assert tab.loc["fixed_lead", "gap0"] == pytest.approx(25.5, abs=0.2)


def test_opponent_lane_change_invalidates(synth, cfg):
    bad = synth.copy()
    # opp(vid2)在 7.5s 后自己换到 lane 2 => 响应窗内换道,invalid
    m = (bad["vid"] == 2) & (bad["t"] >= 7.5)
    bad.loc[m, "lane"] = 2
    tab = extract_interactions(bad, cfg)
    row = tab[(tab["ego_vid"] == 1) & (tab["onset_def"] == "fixed_lead")].iloc[0]
    assert not bool(row["valid"])
    assert row["invalid_reason"] == "opp_lane_change"


def test_no_opponent_flagged(synth, cfg):
    solo = synth[~synth["vid"].isin([2, 3])].copy()   # 目标车道后方无车
    tab = extract_interactions(solo, cfg)
    row = tab[(tab["ego_vid"] == 1) & (tab["onset_def"] == "fixed_lead")].iloc[0]
    assert not bool(row["valid"])
    assert row["invalid_reason"] == "no_opponent"
