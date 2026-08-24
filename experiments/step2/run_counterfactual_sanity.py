"""任务 C 实验入口:counterfactual rollout 的 sanity 检查与预注册敏感性。

运行(仓库根目录,需已有 data/pilot/action_labels.csv):
    python experiments/step2/run_counterfactual_sanity.py

输入口径:onset=lat_vel 的 valid 且 opp_kin_physical 的 interactions。
输出:
  data/pilot/counterfactual_rollouts.csv.gz         (逐帧长表,git 忽略)
  outputs/step2/counterfactual_<hash>/counterfactual_report_<hash>.json
打印:
  C4 sanity:a_peak/jerk 实现值 vs 解析、地板钳位事件数、
             hold 分支负 gap 比例(逐 segment)、Δgap_end 分布
  预注册敏感性:a_peak x horizon 九组合的 Δgap_end 与
             min_gap_Y/min_gap_H(trajectory-level)均值/标准差
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.step1.common import REPO_ROOT, load_config, prepare_output_dir
from src.counterfactual import rollout_table, sensitivity_grid

CFG_PATH = REPO_ROOT / "configs" / "step2_pilot.yaml"
LABEL_TABLE = REPO_ROOT / "data" / "pilot" / "action_labels.csv"
OUT_FRAMES = REPO_ROOT / "data" / "pilot" / "counterfactual_rollouts.csv.gz"
PRIMARY_ONSET = "lat_vel"


def main():
    cfg, cfg_hash = load_config(CFG_PATH)
    out_dir = prepare_output_dir("step2/counterfactual", cfg_hash, CFG_PATH)
    tab = pd.read_csv(LABEL_TABLE)
    inter = tab[(tab["onset_def"] == PRIMARY_ONSET) & tab["valid"]
                & tab["opp_kin_physical"]].copy()
    print(f"config hash = {cfg_hash};rollout 输入 {len(inter)} 个 interactions "
          f"(onset={PRIMARY_ONSET}, valid & kin_physical)\n")

    frames, summ = rollout_table(inter, cfg)
    frames.to_csv(OUT_FRAMES, index=False, compression="gzip")

    report = {"config_hash": cfg_hash, "n_events": int(len(summ))}

    # --- C4 sanity ---
    c = cfg["counterfactual"]
    ap_ok = bool(np.allclose(summ["a_peak_realized"], c["a_peak_mps2"], atol=1e-9))
    jerk_ok = bool(np.allclose(summ["max_jerk_realized"],
                               summ["max_jerk_analytic"], rtol=0.05))
    report["sanity"] = {
        "a_peak_realized_ok": ap_ok,
        "jerk_bound_ok": jerk_ok,
        "events_clamped": int((summ["clamped_frames"] > 0).sum()),
        "delta_gap_end_analytic": float(c["a_peak_mps2"] * c["horizon_s"]**2 / 4),
        "delta_gap_end_median": float(summ["delta_gap_end"].median()),
        "delta_gap_end_min": float(summ["delta_gap_end"].min()),
    }
    print("== C4 sanity ==")
    print(f"a_peak 实现 = 配置值:{ap_ok};jerk 上界解析吻合:{jerk_ok}")
    print(f"速度地板钳位事件:{int((summ['clamped_frames'] > 0).sum())}/{len(summ)}")
    print(f"Δgap_end:解析(无钳位)= {report['sanity']['delta_gap_end_analytic']:.2f} m,"
          f"中位 = {report['sanity']['delta_gap_end_median']:.2f} m,"
          f"最小 = {report['sanity']['delta_gap_end_min']:.2f} m")
    print("\nhold 分支负 gap(碰撞风险信号,合法输入)逐 segment:")
    report["gap_H_negative_by_segment"] = {}
    for seg, g in summ.groupby("segment"):
        frac = float(g["gap_H_negative"].mean())
        report["gap_H_negative_by_segment"][seg] = frac
        print(f"  {seg:26s} {frac:.3f}  (min_gap_H 中位 {g['min_gap_H'].median():6.2f} m)")

    # --- 预注册敏感性 ---
    sens = sensitivity_grid(inter, cfg)
    report["sensitivity_grid"] = sens.to_dict(orient="records")
    print("\n== 预注册敏感性(a_peak x horizon)==")
    print(sens.round(3).to_string(index=False))

    with open(out_dir / f"counterfactual_report_{cfg_hash}.json", "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nrollouts -> {OUT_FRAMES}\n完成。")


if __name__ == "__main__":
    main()