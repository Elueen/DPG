"""Step 2 任务 B(前半)— interaction 抽取:lane-change 事件、onset、opponent。

场景锚定(任务书):ego lane-change / merge interaction。
ego = 发起换道的车辆;opponent = t0 时刻 ego 目标车道中、与 ego 形成
主要 gap interaction 的后方车辆。

B1 onset 候选(阈值全部来自 config,t_cross = ego lane 值跳变时刻):
  O1 fixed_lead : t0 = t_cross - lead_s
  O2 lat_vel    : 以 t_cross 结尾的"横向速度朝目标车道持续 >= vy_th
                  且时长 >= sustain_s"的运行段,t0 = 段起点
                  (受 max_lead_s 截断)
  O3 compound   : O2 且该段内朝目标累计横向位移 >= disp_th
"朝目标"方向 = sign(目标车道经验中心 y - 原车道经验中心 y)
(canonical lane_centers,零 dataset 分支)。

B2 opponent(确定性、可复现):t0 时刻目标车道上 x <= x_ego 的最近车。
validity flags:
  opp_exists / gap0_in_range / opp_coverage(响应窗全覆盖)/
  opp_lane_constant(响应窗内不自己换道)/ ego_coverage(前史+跨线后余量)
全部为真才 valid;任何 False 的事件保留在表中并标记,不进行为建模。

输出 interaction 表(每行 = 事件 x onset 定义):
  dataset, segment, regime, event_id, onset_def, ego_vid, opp_vid,
  t_cross, t0, lane_from, lane_to, gap0, dv0, v_ego0, v_opp0,
  valid, invalid_reason
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.canonical import lane_centers

INTERACTION_COLUMNS = [
    "dataset", "segment", "regime", "event_id", "onset_def", "ego_vid",
    "opp_vid", "t_cross", "t0", "lane_from", "lane_to", "gap0", "dv0",
    "v_ego0", "v_opp0", "valid", "invalid_reason",
]


# ----------------------------------------------------------------------
# 事件检测
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LaneChangeEvent:
    segment: str
    ego_vid: int
    t_cross: float
    lane_from: int
    lane_to: int


def detect_lane_changes(df: pd.DataFrame, merge_window_s: float) -> list[LaneChangeEvent]:
    """ego lane 值跳变 = 跨线;同一 ego 在 merge_window_s 内的连续跨线
    合并为一次事件(取首次;来回摆动/一次换两条都只记起始跨线)。"""
    events = []
    for (seg, vid), g in df.groupby(["segment", "vid"]):
        g = g.sort_values("t")
        lane = g["lane"].to_numpy()
        t = g["t"].to_numpy()
        idx = np.where(lane[1:] != lane[:-1])[0]
        last_t = -np.inf
        for i in idx:
            if t[i + 1] - last_t < merge_window_s:
                last_t = t[i + 1]
                continue
            events.append(LaneChangeEvent(seg, int(vid), float(t[i + 1]),
                                          int(lane[i]), int(lane[i + 1])))
            last_t = t[i + 1]
    return events


# ----------------------------------------------------------------------
# onset 候选
# ----------------------------------------------------------------------
def _toward_sign(centers: pd.DataFrame, segment: str, lane_from: int,
                 lane_to: int) -> float | None:
    seg = centers[centers["segment"] == segment].set_index("lane")["y_center"]
    if lane_from not in seg.index or lane_to not in seg.index:
        return None
    return float(np.sign(seg[lane_to] - seg[lane_from]))


def onset_fixed_lead(ev: LaneChangeEvent, ego: pd.DataFrame, cfg_b: dict,
                     toward: float) -> float | None:
    return ev.t_cross - float(cfg_b["onset"]["fixed_lead_s"])


def onset_lat_vel(ev: LaneChangeEvent, ego: pd.DataFrame, cfg_b: dict,
                  toward: float) -> float | None:
    """以 t_cross 结尾、vy*toward >= vy_th 持续 >= sustain_s 的运行段起点。"""
    o = cfg_b["onset"]
    dt = ego["t"].diff().median()
    pre = ego[(ego["t"] <= ev.t_cross)
              & (ego["t"] >= ev.t_cross - o["max_lead_s"])].sort_values("t")
    if len(pre) < 2:
        return None
    ok = (pre["vy"].to_numpy() * toward) >= o["lat_vel_th_mps"]
    if not ok[-1]:
        return None
    i = len(ok) - 1
    while i > 0 and ok[i - 1]:
        i -= 1
    t0 = float(pre["t"].to_numpy()[i])
    if ev.t_cross - t0 < o["sustain_s"]:
        return None
    return t0


def onset_compound(ev: LaneChangeEvent, ego: pd.DataFrame, cfg_b: dict,
                   toward: float) -> float | None:
    """O2 且运行段内朝目标累计横向位移 >= disp_th。"""
    t0 = onset_lat_vel(ev, ego, cfg_b, toward)
    if t0 is None:
        return None
    seg = ego[(ego["t"] >= t0) & (ego["t"] <= ev.t_cross)]
    disp = (seg["y"].iloc[-1] - seg["y"].iloc[0]) * toward
    if disp < cfg_b["onset"]["disp_th_m"]:
        return None
    return t0


ONSET_DEFS = {
    "fixed_lead": onset_fixed_lead,
    "lat_vel": onset_lat_vel,
    "compound": onset_compound,
}


# ----------------------------------------------------------------------
# opponent 选择 + validity
# ----------------------------------------------------------------------
def _coverage_ok(track: pd.DataFrame, t_a: float, t_b: float, dt: float) -> bool:
    """[t_a, t_b] 是否被该轨迹的 10Hz 帧完整覆盖(允许 1 帧容差)。"""
    tt = track["t"]
    if tt.empty:
        return False
    if tt.min() > t_a + dt / 2 or tt.max() < t_b - dt / 2:
        return False
    n_expect = int(round((t_b - t_a) / dt)) + 1
    n_have = int(((tt >= t_a - dt / 2) & (tt <= t_b + dt / 2)).sum())
    return n_have >= n_expect - 1


def extract_interactions(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """全 pilot 数据的 interaction 表(每行 = 事件 x onset 定义)。"""
    cfg_b = cfg["interaction"]
    dt = 1.0 / cfg["resample_hz"]
    h_pre = float(cfg_b["h_pre_s"])
    h_resp = float(cfg_b["h_response_s"])
    centers = lane_centers(df, cfg["canonical"]["lane_center_min_obs"])
    rows = []
    for seg, dseg in df.groupby("segment"):
        by_vid = {v: g.sort_values("t") for v, g in dseg.groupby("vid")}
        frame_index = dseg.set_index("t")
        events = detect_lane_changes(dseg, cfg_b["merge_window_s"])
        for k, ev in enumerate(events):
            toward = _toward_sign(centers, seg, ev.lane_from, ev.lane_to)
            ego = by_vid[ev.ego_vid]
            meta = dict(
                dataset=str(dseg["dataset"].iloc[0]), segment=seg,
                regime=str(dseg["regime"].iloc[0]),
                event_id=f"{seg}|{ev.ego_vid}|{ev.t_cross:.1f}",
                ego_vid=ev.ego_vid, t_cross=ev.t_cross,
                lane_from=ev.lane_from, lane_to=ev.lane_to,
            )
            for oname, ofun in ONSET_DEFS.items():
                row = dict(meta, onset_def=oname, opp_vid=-1, t0=np.nan,
                           gap0=np.nan, dv0=np.nan, v_ego0=np.nan,
                           v_opp0=np.nan, valid=False, invalid_reason="")
                if toward is None:
                    row["invalid_reason"] = "lane_center_missing"
                    rows.append(row)
                    continue
                t0 = ofun(ev, ego, cfg_b, toward)
                if t0 is None:
                    row["invalid_reason"] = "onset_not_found"
                    rows.append(row)
                    continue
                t0 = float(np.round(np.round(t0 / dt) * dt, 6))
                row["t0"] = t0
                if not _coverage_ok(ego, t0 - h_pre, ev.t_cross + 1.0, dt):
                    row["invalid_reason"] = "ego_coverage"
                    rows.append(row)
                    continue
                # t0 帧上目标车道、ego 后方最近的车
                try:
                    frame = frame_index.loc[[t0]]
                except KeyError:
                    row["invalid_reason"] = "t0_frame_missing"
                    rows.append(row)
                    continue
                ego_t0 = frame[frame["vid"] == ev.ego_vid]
                cand = frame[(frame["lane"] == ev.lane_to)
                             & (frame["vid"] != ev.ego_vid)]
                cand = cand[cand["x"] <= float(ego_t0["x"].iloc[0])]
                if cand.empty:
                    row["invalid_reason"] = "no_opponent"
                    rows.append(row)
                    continue
                opp = cand.iloc[int(np.argmax(cand["x"].to_numpy()))]
                opp_vid = int(opp["vid"])
                gap0 = (float(ego_t0["x"].iloc[0]) - float(opp["x"])
                        - 0.5 * (float(ego_t0["length"].iloc[0]) + float(opp["length"])))
                row.update(opp_vid=opp_vid, gap0=gap0,
                           dv0=float(opp["vx"]) - float(ego_t0["vx"].iloc[0]),
                           v_ego0=float(ego_t0["vx"].iloc[0]),
                           v_opp0=float(opp["vx"]))
                if not (0.0 <= gap0 <= cfg_b["gap0_max_m"]):
                    row["invalid_reason"] = "gap0_out_of_range"
                    rows.append(row)
                    continue
                opp_track = by_vid[opp_vid]
                if not _coverage_ok(opp_track, t0, t0 + h_resp, dt):
                    row["invalid_reason"] = "opp_coverage"
                    rows.append(row)
                    continue
                win = opp_track[(opp_track["t"] >= t0)
                                & (opp_track["t"] <= t0 + h_resp)]
                if not (win["lane"] == ev.lane_to).all():
                    row["invalid_reason"] = "opp_lane_change"
                    rows.append(row)
                    continue
                row["valid"] = True
                rows.append(row)
    out = pd.DataFrame(rows)
    return out[INTERACTION_COLUMNS] if len(out) else pd.DataFrame(
        columns=INTERACTION_COLUMNS)