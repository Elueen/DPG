"""Gate A 检查入口:canonical 语义的实证校验 + 基础运动学 sanity。

运行(仓库根目录,需先跑过 run_pilot_prep.py 产出 processed 数据):
    python experiments/step2/run_canonical_checks.py

三组检查(阈值全部来自 config canonical.checks):
1. 位置假设实证校验:canonical 层计算的 gap/front_to_front 与数据集
   自带口径对比——
   - highD:computed gap_bumper vs tracks.dhw(bumper-to-bumper 口径);
     在 25Hz/10Hz 公共时间戳(0.2s 整数倍)上比对;
   - NGSIM:computed front_to_front vs Space_Headway*FT_TO_M(前端-前端
     口径);SG 平滑引入亚米级差异属预期。
   若出现 ~半车长(2m+)系统性偏差 => 中心转换假设错误,FAIL。
2. 运动学 sanity:速度/加速度范围、dt 均匀性、NaN。
3. 跨数据集语义一致性:同一 canonical 代码路径(零 dataset 分支)
   产出的 gap/dv 分布量纲一致(gap>0 为主、极端负值比例受控)。

产出 outputs/step2/canonical_checks_<hash>/canonical_checks_<hash>.json
并打印 Gate A 判定表。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.step1.common import REPO_ROOT, load_config, prepare_output_dir
from src.data.canonical import build_pairs, kinematic_sanity, lane_centers
from src.data.pilot_prep import FT_TO_M, _read_ngsim_raw

CFG_PATH = REPO_ROOT / "configs" / "step2_pilot.yaml"
PROCESSED_DIR = REPO_ROOT / "data" / "pilot" / "processed"


def _load_processed() -> pd.DataFrame:
    files = sorted(PROCESSED_DIR.glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"{PROCESSED_DIR} 为空:先跑 run_pilot_prep.py")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def _validate_highd(df: pd.DataFrame, cfg: dict, rng) -> dict:
    """computed gap_bumper vs highD tracks.dhw(公共时间戳 0.2s 倍数)。"""
    root = Path(cfg["paths"]["highd_root"])
    out = {}
    n_samp = int(cfg["canonical"]["checks"]["n_validation_sample"])
    for tag in ("free_flow", "dense"):
        rid = f"{int(cfg['highd']['recordings'][tag]):02d}"
        seg = f"highd_{tag}_rec{rid}"
        sub = df[df["segment"] == seg]
        pairs = build_pairs(sub)
        pairs = pairs[pairs["same_lane"]]
        raw = pd.read_csv(root / f"{rid}_tracks.csv",
                          usecols=["frame", "id", "dhw"])
        fps = 25.0
        raw = raw[(raw["dhw"] > 0) & (raw["frame"] % 5 == 0)].copy()
        raw["t"] = np.round(raw["frame"] / fps, 6)
        m = pairs.merge(raw.rename(columns={"id": "vid"}), on=["vid", "t"],
                        how="inner")
        if len(m) > n_samp:
            m = m.sample(n_samp, random_state=rng)
        err = m["gap_bumper"] - m["dhw"]
        out[seg] = {
            "n_compared": int(len(m)),
            "median_abs_err_m": float(err.abs().median()),
            "bias_m": float(err.median()),
        }
    return out


def _validate_ngsim(df: pd.DataFrame, cfg: dict, rng) -> dict:
    """computed front_to_front vs NGSIM Space_Headway(前端-前端口径)。"""
    out = {}
    n_samp = int(cfg["canonical"]["checks"]["n_validation_sample"])
    for tag, key in (("i80", "ngsim_i80"), ("us101", "ngsim_us101")):
        seg = f"ngsim_{tag}_15min"
        sub = df[df["segment"] == seg]
        pairs = build_pairs(sub)
        pairs = pairs[pairs["same_lane"]]
        raw = _read_ngsim_raw(Path(cfg["paths"][key]))
        t0 = raw["Global_Time"].min()
        raw = raw[raw["Space_Headway"] > 0].copy()
        raw["t"] = np.round((raw["Global_Time"] - t0) / 1000.0, 6)
        raw["sh_m"] = raw["Space_Headway"] * FT_TO_M
        m = pairs.merge(
            raw.rename(columns={"Vehicle_ID": "vid"})[["vid", "t", "sh_m"]],
            on=["vid", "t"], how="inner")
        if len(m) > n_samp:
            m = m.sample(n_samp, random_state=rng)
        err = m["front_to_front"] - m["sh_m"]
        out[seg] = {
            "n_compared": int(len(m)),
            "median_abs_err_m": float(err.abs().median()),
            "bias_m": float(err.median()),
        }
    return out


def main():
    cfg, cfg_hash = load_config(CFG_PATH)
    out_dir = prepare_output_dir("step2/canonical_checks", cfg_hash, CFG_PATH)
    checks_cfg = cfg["canonical"]["checks"]
    rng = int(cfg["meta"]["seed"])
    df = _load_processed()

    report = {"gap_convention_validation": {}, "kinematic": {}, "pairs": {},
              "lane_centers": {}}
    report["gap_convention_validation"].update(_validate_highd(df, cfg, rng))
    report["gap_convention_validation"].update(_validate_ngsim(df, cfg, rng))

    for seg, sub in df.groupby("segment"):
        report["kinematic"][seg] = kinematic_sanity(sub, checks_cfg)
        pairs = build_pairs(sub)
        sl = pairs[pairs["same_lane"]]
        report["pairs"][seg] = {
            "n_pairs_rows": int(len(sl)),
            "gap_median_m": float(sl["gap_bumper"].median()),
            "gap_negative_ratio": float((sl["gap_bumper"] < 0).mean()),
            "dv_closing_median": float(sl["dv_closing"].median()),
        }
        lc = lane_centers(sub, cfg["canonical"]["lane_center_min_obs"])
        report["lane_centers"][seg] = {
            "n_lanes": int(len(lc)),
            "centers_y": [round(float(v), 2) for v in lc["y_center"]],
        }

    # --- Gate A 判定 ---
    gate = {}
    for seg, v in report["gap_convention_validation"].items():
        gate[f"gap_convention[{seg}]"] = (
            v["median_abs_err_m"] <= checks_cfg["gap_median_abs_err_m"]
            and abs(v["bias_m"]) <= checks_cfg["gap_bias_abs_m"]
        )
    for seg, v in report["kinematic"].items():
        gate[f"kinematic[{seg}]"] = (
            v["dt_uniform"] and v["nan_ratio"] == 0.0
            and v["speed_over_limit_ratio"] < 0.001
            and v["ax_over_limit_ratio"] < 0.005
        )
    for seg, v in report["pairs"].items():
        gate[f"pairs[{seg}]"] = (
            v["n_pairs_rows"] > 1000 and v["gap_negative_ratio"] < 0.01
        )
    report["gate_a"] = gate

    print(f"config hash = {cfg_hash}")
    print("\n== 位置假设实证校验(computed vs 数据集自带口径)==")
    for seg, v in report["gap_convention_validation"].items():
        print(f"{seg:26s} n={v['n_compared']:>7d}  "
              f"median|err|={v['median_abs_err_m']:.3f}m  bias={v['bias_m']:+.3f}m")
    print("\n== Gate A ==")
    for k, ok in gate.items():
        print(f"{'PASS' if ok else 'FAIL':4s}  {k}")
    print("\n=>", "GATE A PASS" if all(gate.values()) else "GATE A FAIL(见上)")

    with open(out_dir / f"canonical_checks_{cfg_hash}.json", "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()