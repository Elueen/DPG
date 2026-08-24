"""Step 2 A3 — canonical physical-state 层:派生量与语义冻结。

canonical schema(pilot_prep 输出,SI 单位,10 Hz 全局对齐网格):
    dataset, segment, recording, site, regime, vid, t,
    x, y, vx, vy, ax, ay, lane, length, width, vclass, preceding, following

冻结语义(Gate A):
- x:纵向,行驶方向为正,车辆几何中心。
  适配器负责中心转换:highD 原始 (x,y) 为包围盒左上角(+L/2, +W/2);
  NGSIM Local_Y 为车前端(-L/2)。两个假设由 run_canonical_checks 用
  数据集自带 dhw / Space_Headway 实证校验,不盲信文档口径。
- y:横向,车辆几何中心;不强加全局左右约定,横向关系一律经
  经验车道中心几何(lane_centers)判断。
- gap:bumper-to-bumper,gap = x_lead - x_follow - (L_lead + L_follow)/2。
- Δv(closing rate):dv_closing = vx_follow - vx_lead,正 = 接近。
- lane:数据集原生车道号;邻接关系由车道中心 y 排序决定,不做 id 算术。
- 从本层起,行为学代码不得再出现 dataset 分支;dataset/recording/site/
  regime 仅作元数据保留(不统一真实分布)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PAIR_COLUMNS = [
    "dataset", "segment", "t", "vid", "lead_vid", "lane", "lane_lead",
    "gap_bumper", "front_to_front", "dv_closing", "same_lane",
]


def lane_centers(df: pd.DataFrame, min_obs: int = 50) -> pd.DataFrame:
    """逐 (segment, lane) 的经验车道中心(中位 y)与观测数;lane<=0 剔除。
    车道邻接 = 中心 y 排序相邻。"""
    d = df[df["lane"] > 0]
    g = d.groupby(["segment", "lane"])["y"].agg(["median", "count"]).reset_index()
    g = g.rename(columns={"median": "y_center", "count": "n_obs"})
    return g[g["n_obs"] >= min_obs].reset_index(drop=True)


def adjacent_lanes(centers: pd.DataFrame, segment: str, lane: int) -> list[int]:
    """按中心 y 排序返回 lane 的相邻车道号(0/1/2 个)。"""
    seg = centers[centers["segment"] == segment].sort_values("y_center")
    order = seg["lane"].tolist()
    if lane not in order:
        return []
    i = order.index(lane)
    out = []
    if i > 0:
        out.append(int(order[i - 1]))
    if i < len(order) - 1:
        out.append(int(order[i + 1]))
    return out


def lateral_offset(df: pd.DataFrame, centers: pd.DataFrame) -> pd.Series:
    """y - 所在 lane 的经验中心(无中心记录的行为 NaN)。"""
    m = df.merge(centers[["segment", "lane", "y_center"]],
                 on=["segment", "lane"], how="left")
    return (m["y"] - m["y_center"]).rename("lateral_offset")


def build_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """follower-leader 同帧对(经 preceding id 连接)的纵向派生量。

    gap_bumper     = x_lead - x - (L_lead + L)/2      (bumper-to-bumper)
    front_to_front = x_lead - x + (L_lead - L)/2      (前端到前端,NGSIM
                     Space_Headway 口径,校验用)
    dv_closing     = vx - vx_lead                     (正 = 接近)
    """
    lead = df[["segment", "t", "vid", "x", "vx", "length", "lane"]].rename(
        columns={"vid": "lead_vid", "x": "x_lead", "vx": "vx_lead",
                 "length": "len_lead", "lane": "lane_lead"})
    foll = df[df["preceding"] > 0]
    pairs = foll.merge(
        lead, left_on=["segment", "t", "preceding"],
        right_on=["segment", "t", "lead_vid"], how="inner",
    )
    pairs["gap_bumper"] = (
        pairs["x_lead"] - pairs["x"] - 0.5 * (pairs["len_lead"] + pairs["length"])
    )
    pairs["front_to_front"] = (
        pairs["x_lead"] - pairs["x"] + 0.5 * (pairs["len_lead"] - pairs["length"])
    )
    pairs["dv_closing"] = pairs["vx"] - pairs["vx_lead"]
    pairs["same_lane"] = pairs["lane"] == pairs["lane_lead"]
    return pairs[PAIR_COLUMNS + ["x", "x_lead", "vx", "vx_lead"]]


def kinematic_sanity(df: pd.DataFrame, limits: dict) -> dict:
    """Gate A 基础运动学检查(阈值来自 config)。"""
    dt_ok = True
    for _, g in df.groupby("vid"):
        d = np.diff(np.sort(g["t"].to_numpy()))
        if len(d) and not np.allclose(d, d[0], atol=1e-6):
            dt_ok = False
            break
    return {
        "dt_uniform": bool(dt_ok),
        "speed_over_limit_ratio": float((df["vx"] > limits["max_speed_mps"]).mean()),
        "neg_speed_ratio": float((df["vx"] < 0).mean()),
        "ax_over_limit_ratio": float((df["ax"].abs() > limits["max_abs_ax"]).mean()),
        "ay_over_limit_ratio": float((df["ay"].abs() > limits["max_abs_ay"]).mean()),
        "nan_ratio": float(df[["x", "y", "vx", "vy", "ax", "ay"]].isna().mean().mean()),
    }