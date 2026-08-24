"""Step 2 任务 A 测试:pilot 数据管线(合成 highD/NGSIM 格式数据)。

真实数据不进 git 也不进测试;本文件在临时目录构造与两数据集
schema 一致的合成数据,验证处理规则本身:
  1) highD:width/length 字段修正;行驶方向归一(方向 1 车辆的
     vx 翻正);重采样后严格 10 Hz 网格
  2) NGSIM:英尺->米;重复帧检出并剔除;SG 平滑速度对已知真值
     无偏且噪声远小于裸差分;短轨迹按窗口长度剔除
  3) 质量统计字段齐全且量纲合理
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.pilot_prep import (
    FT_TO_M,
    STANDARD_COLUMNS,
    load_highd_segment,
    load_ngsim_segment,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "step2_pilot.yaml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        c = yaml.safe_load(f)
    c["ngsim"]["segment_duration_s"] = 60.0
    return c


@pytest.fixture(scope="module")
def synth_dir() -> Path:
    """构造合成数据(schema 同真实数据),模块结束后清理。"""
    tmp = Path(tempfile.mkdtemp(prefix="pilot_synth_test_"))
    rng = np.random.default_rng(0)
    hd = tmp / "highd"
    hd.mkdir()
    fps, T = 25, 8.0
    frames = np.arange(int(T * fps))
    rows = []
    for vid, d in [(1, 2), (2, 1)]:
        t = frames / fps
        x = 100 + 20 * t if d == 2 else 300 - 22 * t
        vx = np.full_like(t, 20.0 if d == 2 else -22.0)
        rows.append(pd.DataFrame({
            "frame": frames, "id": vid, "x": x, "y": 8.0 + 0.1 * vid,
            "width": 4.5, "height": 1.9, "xVelocity": vx, "yVelocity": 0.0,
            "xAcceleration": 0.0, "yAcceleration": 0.0,
            "frontSightDistance": 0, "backSightDistance": 0,
            "dhw": 30.0, "thw": 1.5, "ttc": 5.0, "precedingXVelocity": vx,
            "precedingId": 0, "followingId": 0, "laneId": 3,
        }))
    pd.concat(rows).to_csv(hd / "02_tracks.csv", index=False)
    pd.DataFrame({"id": [1, 2], "drivingDirection": [2, 1],
                  "class": ["Car", "Truck"]}).to_csv(hd / "02_tracksMeta.csv", index=False)
    pd.DataFrame({"id": [2], "locationId": [1], "frameRate": [25],
                  "duration": [T], "numVehicles": [2], "speedLimit": [120]}
                 ).to_csv(hd / "02_recordingMeta.csv", index=False)

    rows = []
    for vid in [10, 11]:
        n = 700
        gt = 1_000_000 + np.arange(n) * 100
        v_fts = 50 + 5 * np.sin(np.arange(n) * 0.01)
        y_ft = np.cumsum(v_fts) * 0.1 + rng.normal(0, 1.0, n)
        df = pd.DataFrame({
            "Vehicle_ID": vid, "Frame_ID": np.arange(n), "Total_Frames": n,
            "Global_Time": gt, "Local_X": 12.0, "Local_Y": y_ft,
            "Global_X": 0.0, "Global_Y": 0.0, "v_Length": 15.0, "v_Width": 6.0,
            "v_Class": 2, "v_Vel": v_fts, "v_Acc": 0.0, "Lane_ID": 3,
            "Preceding": 0, "Following": 0,
            "Space_Headway": 100.0, "Time_Headway": 2.0,
        })
        rows.append(df)
        rows.append(df.iloc[:20])  # 重复帧
    # 一条短轨迹(应被窗口长度剔除)
    rows.append(rows[0].iloc[:5].assign(Vehicle_ID=99))
    pd.concat(rows).to_csv(tmp / "ngsim.csv", index=False)
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def highd_out(synth_dir, cfg):
    hd = synth_dir / "highd"
    return load_highd_segment(
        hd / "02_tracks.csv", hd / "02_tracksMeta.csv", hd / "02_recordingMeta.csv",
        "hd_test", "free_flow", cfg,
    )


@pytest.fixture(scope="module")
def ngsim_out(synth_dir, cfg):
    return load_ngsim_segment(synth_dir / "ngsim.csv", "ng_test", "i80", cfg)


# ----------------------------------------------------------------------
# (1) highD
# ----------------------------------------------------------------------
def test_highd_schema_and_swap_fix(highd_out):
    df, _ = highd_out
    assert list(df.columns) == STANDARD_COLUMNS
    assert df["length"].iloc[0] == pytest.approx(4.5)   # 修正后 length=原 width 列
    assert df["width"].iloc[0] == pytest.approx(1.9)


def test_highd_direction_normalized(highd_out):
    df, stats = highd_out
    means = df.groupby("vid")["vx"].mean()
    assert (means > 0).all()                            # 方向 1 车辆被翻正
    assert stats["flipped_vehicles"] == 1
    assert means[2] == pytest.approx(22.0, rel=1e-6)


def test_highd_uniform_10hz(highd_out, cfg):
    df, _ = highd_out
    dt = df[df["vid"] == 1]["t"].diff().dropna().to_numpy()
    np.testing.assert_allclose(dt, 1.0 / cfg["resample_hz"], rtol=1e-9)


# ----------------------------------------------------------------------
# (2) NGSIM
# ----------------------------------------------------------------------
def test_ngsim_units_and_dedup(ngsim_out):
    df, stats = ngsim_out
    assert df["length"].iloc[0] == pytest.approx(15.0 * FT_TO_M)
    assert stats["duplicate_frame_ratio"] > 0.02        # 植入的重复帧被检出
    assert stats["short_track_dropped"] >= 1            # 短轨迹被剔除
    assert 99 not in set(df["vid"])


def test_ngsim_savgol_velocity_unbiased_and_denoised(ngsim_out):
    df, _ = ngsim_out
    v = df[df["vid"] == 10]["vx"].to_numpy()
    v_mid = v[30:-30]
    true_mean = float(np.mean((50 + 5 * np.sin(np.arange(600) * 0.01)) * FT_TO_M))
    assert np.mean(v_mid) == pytest.approx(true_mean, rel=0.01)   # 无偏
    # 1 ft 位置噪声裸差分速度噪声 ~4.3 m/s;SG 后应远小
    assert np.std(np.diff(v_mid)) < 0.3
    assert np.all(np.isfinite(v))


# ----------------------------------------------------------------------
# (3) 质量统计
# ----------------------------------------------------------------------
def test_quality_stats_fields(highd_out, ngsim_out):
    for _, stats in (highd_out, ngsim_out):
        for key in ("rows_10hz", "vehicles_10hz", "nan_ratio",
                    "median_track_duration_s", "vehicle_density_proxy"):
            assert key in stats
        assert stats["nan_ratio"] == 0.0


def test_cross_vehicle_timestamp_alignment(highd_out):
    """跨车时间戳必须精确重合(交互抽取的前提):同帧车辆数均值应 > 1。"""
    df, stats = highd_out
    t1 = set(df[df["vid"] == 1]["t"])
    t2 = set(df[df["vid"] == 2]["t"])
    assert len(t1 & t2) > 50                      # 两车共同帧大量存在
    assert stats["vehicle_density_proxy"] > 1.5   # 合成数据两车全程共存


def test_ngsim_txt_format_equivalent(synth_dir, cfg):
    """txt(空白分隔无表头)与 csv 路线应产出相同结果;截断 csv 被拒绝。"""
    import pandas as pd
    from src.data.pilot_prep import NGSIM_COLUMNS, _read_ngsim_raw
    csv_path = synth_dir / "ngsim.csv"
    txt_path = synth_dir / "ngsim.txt"
    pd.read_csv(csv_path)[NGSIM_COLUMNS].to_csv(
        txt_path, sep=" ", header=False, index=False)
    df_txt, _ = load_ngsim_segment(txt_path, "ng_txt", "i80", cfg)
    df_csv, _ = load_ngsim_segment(csv_path, "ng_txt", "i80", cfg)
    pd.testing.assert_frame_equal(
        df_txt.reset_index(drop=True), df_csv.reset_index(drop=True))


def test_ngsim_truncated_csv_rejected(synth_dir):
    """1,048,575 行的 CSV(Excel 截断签名)应被拒绝。"""
    import pandas as pd
    import pytest as _pytest
    from src.data.pilot_prep import NGSIM_COLUMNS, _read_ngsim_raw
    bad = synth_dir / "truncated.csv"
    n = 1_048_575
    pd.DataFrame({c: np.zeros(n, dtype=np.int8) for c in NGSIM_COLUMNS}).to_csv(
        bad, index=False)
    with _pytest.raises(ValueError):
        _read_ngsim_raw(bad)