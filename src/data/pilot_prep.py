"""Step 2 任务 A — pilot 数据准备管线(canonical schema v2)。

canonical schema(全部 SI 单位,10 Hz 全局对齐网格,t = k*0.1 精确重合):
    dataset, segment, recording, site, regime, vid, t,
    x, y, vx, vy, ax, ay, lane, length, width, vclass, preceding, following

位置语义:x/y = 车辆几何中心;x 纵向、行驶方向为正(语义冻结见
src/data/canonical.py 与 inventory)。

处理规则(参数全部来自 configs/step2_pilot.yaml):
- highD:width/height 字段修正为 length/width;包围盒左上角 -> 几何中心
  (+L/2, +W/2);行驶方向归一(方向 1 翻转 x/vx/ax/y/vy/ay);
  官方轨迹不二次滤波;25 -> 10 Hz 全局对齐线性插值。
- NGSIM:官方 txt 完整版(CSV 版存在 Excel 1,048,575 行截断,加载即拒绝);
  英尺 -> 米;Local_Y(车前端)-> 几何中心(-L/2);逐车去重帧;
  位置 Savitzky-Golay 平滑,速度/加速度 deriv=1/2 重算;
  全局对齐 10 Hz 网格重建;取 Global_Time 偏移后 duration 秒片段。

纪律:pilot 数据不进 git(data/ 已忽略);本模块只提交代码。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

STANDARD_COLUMNS = [
    "dataset", "segment", "recording", "site", "regime", "vid", "t",
    "x", "y", "vx", "vy", "ax", "ay",
    "lane", "length", "width", "vclass", "preceding", "following",
]
FT_TO_M = 0.3048

NGSIM_COLUMNS = [
    "Vehicle_ID", "Frame_ID", "Total_Frames", "Global_Time", "Local_X",
    "Local_Y", "Global_X", "Global_Y", "v_Length", "v_Width", "v_Class",
    "v_Vel", "v_Acc", "Lane_ID", "Preceding", "Following",
    "Space_Headway", "Time_Headway",
]
# 官方 CSV 的拼写错误与缩写(txt 无表头,不受影响)
_NGSIM_ALIASES = {"Preceeding": "Preceding", "Space_Hdwy": "Space_Headway",
                  "Time_Hdwy": "Time_Headway"}


def _read_ngsim_raw(path: Path) -> pd.DataFrame:
    """读原始 NGSIM 轨迹:.txt(空白分隔无表头,官方完整版)或
    .csv(逗号带表头;官方 CSV 系 Excel 导出、有 1,048,575 行截断,
    读入即按行数拒绝)。列名归一到 NGSIM_COLUMNS。"""
    path = Path(path)
    if path.suffix.lower() == ".txt":
        df = pd.read_csv(path, sep=r"\s+", header=None, names=NGSIM_COLUMNS)
    else:
        df = pd.read_csv(path).rename(columns=_NGSIM_ALIASES)
        if len(df) == 1_048_575:
            raise ValueError(
                f"{path.name} 恰为 1,048,575 行(Excel 截断签名):"
                "该 CSV 不完整,请改用同目录 .txt 版本"
            )
    missing = set(NGSIM_COLUMNS) - set(df.columns)
    if missing:
        raise KeyError(f"{path.name} 缺列:{sorted(missing)}")
    return df


# ----------------------------------------------------------------------
# highD
# ----------------------------------------------------------------------
def load_highd_segment(
    tracks_csv: Path, tracks_meta_csv: Path, recording_meta_csv: Path,
    segment_name: str, regime: str, cfg: dict,
) -> tuple[pd.DataFrame, dict]:
    """返回 (canonical 轨迹表, 质量统计)。"""
    tracks = pd.read_csv(tracks_csv)
    tmeta = pd.read_csv(tracks_meta_csv)
    rmeta = pd.read_csv(recording_meta_csv)
    fps = float(rmeta["frameRate"].iloc[0])

    stats = {
        "raw_rows": int(len(tracks)),
        "raw_vehicles": int(tracks["id"].nunique()),
        "frame_rate": fps,
        "raw_columns": list(tracks.columns),
    }

    df = tracks.rename(columns={"id": "vid"}).copy()
    # 已知 schema 问题:tracks 的 width=纵向尺寸(车长),height=横向尺寸(车宽)
    if not cfg["highd"]["fix_width_length_swap"]:
        raise ValueError("fix_width_length_swap=false 无对应实现:该修正是既定处理规则")
    df["length"] = tracks["width"].astype(float)
    df["width"] = tracks["height"].astype(float)

    df["t"] = df["frame"].astype(float) / fps
    # canonical 位置 = 车辆几何中心(highD 原始 x,y 为包围盒左上角)
    if cfg["canonical"]["highd_bbox_topleft"]:
        df["x"] = df["x"].astype(float) + df["length"] / 2.0
        df["y"] = df["y"].astype(float) + df["width"] / 2.0
    else:
        df["x"] = df["x"].astype(float)
        df["y"] = df["y"].astype(float)
    df["vx"] = df["xVelocity"].astype(float)
    df["vy"] = df["yVelocity"].astype(float)
    df["ax"] = df["xAcceleration"].astype(float)
    df["ay"] = df["yAcceleration"].astype(float)
    df["lane"] = df["laneId"].astype(int)
    df["preceding"] = df["precedingId"].astype(int)
    df["following"] = df["followingId"].astype(int)

    # 行驶方向归一:direction==1(x 递减方向)翻转符号使纵向运动为正
    direction = tmeta.set_index("id")["drivingDirection"].to_dict()
    df["_dir"] = df["vid"].map(direction).fillna(2).astype(int)
    if cfg["highd"]["normalize_direction"]:
        flip = df["_dir"] == 1
        for col in ("x", "vx", "ax", "y", "vy", "ay"):
            df.loc[flip, col] = -df.loc[flip, col]
        stats["flipped_vehicles"] = int(df.loc[flip, "vid"].nunique())

    vclass = tmeta.set_index("id")["class"].to_dict()
    df["vclass"] = df["vid"].map(vclass).fillna("unknown")

    df["dataset"] = "highd"
    df["segment"] = segment_name
    df["recording"] = f"rec{int(rmeta['id'].iloc[0]):02d}"
    df["site"] = f"loc{int(rmeta['locationId'].iloc[0])}"
    df["regime"] = regime
    out = _resample_uniform(df, target_hz=cfg["resample_hz"])
    stats.update(_quality_stats(out))
    return out[STANDARD_COLUMNS], stats


# ----------------------------------------------------------------------
# NGSIM
# ----------------------------------------------------------------------
def load_ngsim_segment(
    trajectory_csv: Path, segment_name: str, site: str, cfg: dict,
) -> tuple[pd.DataFrame, dict]:
    trajectory_csv = Path(trajectory_csv)
    raw = _read_ngsim_raw(trajectory_csv)
    stats = {
        "raw_rows": int(len(raw)),
        "raw_vehicles": int(raw["Vehicle_ID"].nunique()),
        "raw_columns": list(raw.columns),
    }
    ng = cfg["ngsim"]

    # 片段裁剪:Global_Time(ms)起点偏移 + 时长
    t0 = raw["Global_Time"].min() + ng["segment_offset_s"] * 1000.0
    t1 = t0 + ng["segment_duration_s"] * 1000.0
    raw = raw[(raw["Global_Time"] >= t0) & (raw["Global_Time"] < t1)].copy()
    stats["segment_rows"] = int(len(raw))
    stats["segment_vehicles"] = int(raw["Vehicle_ID"].nunique())
    stats["segment_span_s"] = float(
        (raw["Global_Time"].max() - raw["Global_Time"].min()) / 1000.0
    )

    # 逐车去重帧(NGSIM 已知重复帧问题)
    before = len(raw)
    raw = raw.sort_values(["Vehicle_ID", "Global_Time"]).drop_duplicates(
        ["Vehicle_ID", "Global_Time"], keep="first"
    )
    stats["duplicate_frame_ratio"] = float((before - len(raw)) / max(before, 1))

    dt = 1.0 / cfg["resample_hz"]
    win = int(ng["savgol_window"])
    poly = int(ng["savgol_polyorder"])
    rows = []
    short_tracks = 0
    for vid, g in raw.groupby("Vehicle_ID"):
        g = g.sort_values("Global_Time")
        t = (g["Global_Time"].to_numpy() - raw["Global_Time"].min()) / 1000.0
        if len(g) < win:
            short_tracks += 1
            continue
        # 位置(米):NGSIM Local_Y 沿路(官方口径为车前端),Local_X 横向
        length_m = float(g["v_Length"].iloc[0]) * FT_TO_M
        x = g["Local_Y"].to_numpy() * FT_TO_M
        if cfg["canonical"]["ngsim_local_y_is_front"]:
            x = x - length_m / 2.0          # canonical 位置 = 几何中心
        y = g["Local_X"].to_numpy() * FT_TO_M
        # 全局对齐均匀网格重建(t = k*dt,跨车时间戳精确重合)
        k0 = int(np.ceil(round(t[0] / dt, 9)))
        k1 = int(np.floor(round(t[-1] / dt, 9)))
        if k1 < k0:
            short_tracks += 1
            continue
        tg = np.arange(k0, k1 + 1) * dt
        xi = np.interp(tg, t, x)
        yi = np.interp(tg, t, y)
        if len(tg) < win:
            short_tracks += 1
            continue
        xs = savgol_filter(xi, win, poly)
        ys = savgol_filter(yi, win, poly)
        vx = savgol_filter(xi, win, poly, deriv=1, delta=dt)
        vy = savgol_filter(yi, win, poly, deriv=1, delta=dt)
        ax = savgol_filter(xi, win, poly, deriv=2, delta=dt)
        ay = savgol_filter(yi, win, poly, deriv=2, delta=dt)
        sub = pd.DataFrame({
            "t": np.round(tg, 6), "x": xs, "y": ys, "vx": vx, "vy": vy,
            "ax": ax, "ay": ay,
        })
        sub["vid"] = int(vid)
        sub["lane"] = np.interp(tg, t, g["Lane_ID"].to_numpy()).round().astype(int)
        sub["length"] = length_m
        sub["width"] = float(g["v_Width"].iloc[0]) * FT_TO_M
        sub["vclass"] = int(g["v_Class"].iloc[0])
        sub["preceding"] = (
            g.set_index(t)["Preceding"].reindex(tg, method="nearest").to_numpy()
        )
        sub["following"] = (
            g.set_index(t)["Following"].reindex(tg, method="nearest").to_numpy()
        )
        rows.append(sub)
    stats["short_track_dropped"] = int(short_tracks)

    out = pd.concat(rows, ignore_index=True)
    out["dataset"] = "ngsim"
    out["segment"] = segment_name
    out["recording"] = trajectory_csv.stem
    out["site"] = site
    out["regime"] = "congested"
    stats.update(_quality_stats(out))
    return out[STANDARD_COLUMNS], stats


# ----------------------------------------------------------------------
# 通用
# ----------------------------------------------------------------------
def _resample_uniform(df: pd.DataFrame, target_hz: float) -> pd.DataFrame:
    """逐车重采样到与全局时间轴对齐的 target_hz 网格(t = k*dt,k 为整数),
    保证跨车时间戳精确重合(交互抽取依赖共同帧)。"""
    dt = 1.0 / target_hz
    rows = []
    for vid, g in df.groupby("vid"):
        g = g.sort_values("t")
        t = g["t"].to_numpy()
        if len(g) < 2:
            continue
        k0 = int(np.ceil(round(t[0] / dt, 9)))
        k1 = int(np.floor(round(t[-1] / dt, 9)))
        if k1 < k0:
            continue
        tg = np.arange(k0, k1 + 1) * dt
        sub = pd.DataFrame({"t": np.round(tg, 6)})
        for col in ("x", "y", "vx", "vy", "ax", "ay"):
            sub[col] = np.interp(tg, t, g[col].to_numpy(dtype=float))
        for col in ("lane", "preceding", "following"):
            sub[col] = (
                g.set_index(t)[col].reindex(tg, method="nearest").to_numpy()
            )
        for col in ("length", "width", "vclass", "dataset", "segment",
                    "recording", "site", "regime"):
            sub[col] = g[col].iloc[0]
        sub["vid"] = vid
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def _quality_stats(df: pd.DataFrame) -> dict:
    dur = df.groupby("vid")["t"].agg(lambda s: s.max() - s.min())
    return {
        "rows_10hz": int(len(df)),
        "vehicles_10hz": int(df["vid"].nunique()),
        "nan_ratio": float(
            df[["x", "y", "vx", "vy", "ax", "ay"]].isna().mean().mean()),
        "negative_vx_ratio": float((df["vx"] < 0).mean()),
        "abs_ax_gt_8_ratio": float((df["ax"].abs() > 8.0).mean()),
        "median_track_duration_s": float(dur.median()),
        "mean_speed_mps": float(df["vx"].mean()),
        "p95_speed_mps": float(df["vx"].quantile(0.95)),
        "vehicle_density_proxy": float(df.groupby("t")["vid"].count().mean()),
    }


# ----------------------------------------------------------------------
# 派生变量支持度(inventory 用)
# ----------------------------------------------------------------------
DERIVED_SUPPORT = {
    "highd": {
        "gap (dhw)": "直接可得(tracks.dhw,米;canonical 层亦可自算并已互校)",
        "THW": "直接可得(tracks.thw,秒)",
        "TTC": "直接可得(tracks.ttc,秒;无前车时为 0/负,需过滤)",
        "Δv (与前车)": "直接可得(precedingXVelocity - xVelocity);canonical 自算",
        "相对位置": "可推算(canonical build_pairs,同帧 preceding 连接)",
        "lateral offset": "可推算(canonical lane_centers,经验车道中心)",
        "predicted delay": "需推算(自定义,基于期望速度)",
        "期望速度差": "需推算(期望速度须外生定义)",
    },
    "ngsim": {
        "gap": "可推算(canonical build_pairs;Space_Headway 为前端-前端口径,仅校验用)",
        "THW": "直接可得(Time_Headway,秒;0 值为无效标记)",
        "TTC": "需推算(gap 与 dv_closing;平滑速度经 Preceding 连接)",
        "Δv (与前车)": "可推算(canonical build_pairs,平滑速度)",
        "相对位置": "可推算(canonical build_pairs)",
        "lateral offset": "可推算(canonical lane_centers)",
        "predicted delay": "需推算",
        "期望速度差": "需推算",
    },
}