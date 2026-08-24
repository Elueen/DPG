"""A3 canonical 层测试(合成 canonical schema 数据,零真实数据)。

覆盖:
  1) build_pairs:bumper-to-bumper gap / front_to_front / dv_closing
     公式对手工构造的已知场景精确成立;跨车道对 same_lane=False
  2) lane_centers / adjacent_lanes:经验中心与邻接排序正确;
     lateral_offset = y - 中心
  3) kinematic_sanity:对干净数据全绿,对注入超限数据正确报警
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.canonical import (
    adjacent_lanes,
    build_pairs,
    kinematic_sanity,
    lane_centers,
    lateral_offset,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "step2_pilot.yaml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def synth_canonical() -> pd.DataFrame:
    """两条车道、三辆车、60 帧的手工 canonical 表。
    lane 2(y≈4):vid 1(lead, 20 m/s)与 vid 2(follower, 22 m/s,接近);
    lane 3(y≈8):vid 3。vid2.preceding=1。"""
    t = np.round(np.arange(60) * 0.1, 6)
    rows = []
    for vid, lane, y0, x0, v, length, prec in [
        (1, 2, 4.0, 100.0, 20.0, 5.0, 0),
        (2, 2, 4.1, 60.0, 22.0, 4.0, 1),
        (3, 3, 8.0, 80.0, 25.0, 4.5, 0),
    ]:
        rows.append(pd.DataFrame({
            "dataset": "synth", "segment": "seg", "recording": "r", "site": "s",
            "regime": "test", "vid": vid, "t": t,
            "x": x0 + v * t, "y": y0, "vx": v, "vy": 0.0, "ax": 0.0, "ay": 0.0,
            "lane": lane, "length": length, "width": 2.0, "vclass": "Car",
            "preceding": prec, "following": 0,
        }))
    return pd.concat(rows, ignore_index=True)


def test_pair_formulas_exact(synth_canonical):
    pairs = build_pairs(synth_canonical)
    p0 = pairs[(pairs["vid"] == 2) & (pairs["t"] == 0.0)].iloc[0]
    # gap = 100 - 60 - (5+4)/2 = 35.5;front_to_front = 40 + (5-4)/2 = 40.5
    assert p0["gap_bumper"] == pytest.approx(35.5)
    assert p0["front_to_front"] == pytest.approx(40.5)
    assert p0["dv_closing"] == pytest.approx(2.0)     # 22-20,正=接近
    assert bool(p0["same_lane"])
    # gap 随时间线性缩小:t=5.9s 时 35.5 - 2*5.9 = 23.7
    p59 = pairs[(pairs["vid"] == 2) & (pairs["t"] == 5.9)].iloc[0]
    assert p59["gap_bumper"] == pytest.approx(23.7)


def test_lane_geometry(synth_canonical):
    lc = lane_centers(synth_canonical, min_obs=10)
    d = dict(zip(lc["lane"], lc["y_center"]))
    assert d[3] == pytest.approx(8.0)
    assert d[2] == pytest.approx(4.05, abs=0.06)      # 两车中位
    assert adjacent_lanes(lc, "seg", 2) == [3]
    assert adjacent_lanes(lc, "seg", 3) == [2]
    off = lateral_offset(synth_canonical, lc)
    assert np.nanmax(np.abs(off)) < 0.1


def test_kinematic_sanity_flags(synth_canonical, cfg):
    limits = cfg["canonical"]["checks"]
    clean = kinematic_sanity(synth_canonical, limits)
    assert clean["dt_uniform"] and clean["nan_ratio"] == 0.0
    assert clean["speed_over_limit_ratio"] == 0.0
    bad = synth_canonical.copy()
    bad.loc[bad.index[:30], "vx"] = 99.0
    bad.loc[bad.index[:30], "ax"] = 20.0
    flagged = kinematic_sanity(bad, limits)
    assert flagged["speed_over_limit_ratio"] > 0.1
    assert flagged["ax_over_limit_ratio"] > 0.1