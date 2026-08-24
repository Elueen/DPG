"""任务 B(B3/B4)实验入口:三候选动作定义的标签与诊断。

运行(仓库根目录,需已有 processed 数据与 data/pilot/interactions.csv):
    python experiments/step2/run_action_labels.py

输出:
  data/pilot/action_labels.csv                      (git 忽略)
  outputs/step2/action_labels_<hash>/action_labels_report_<hash>.json
打印(B4 诊断):
  1) 逐定义 x 逐 segment 的 yield/hold/ambiguous 比例(onset=lat_vel 主口径,
     其余 onset 进 json)
  2) 定义两两一致率(非 ambiguous 共同子集)+ 3x3 confusion
  3) ±30% 阈值敏感性:标签翻转比例
  4) dv_red 与 a_end 的相关性(两信号族的冗余度)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.step1.common import REPO_ROOT, load_config, prepare_output_dir
from src.data.actions import (
    DEFS,
    LABELS,
    build_label_table,
    label_proportions,
    pairwise_agreement,
    sensitivity_flips,
)

CFG_PATH = REPO_ROOT / "configs" / "step2_pilot.yaml"
PROCESSED_DIR = REPO_ROOT / "data" / "pilot" / "processed"
INTER_TABLE = REPO_ROOT / "data" / "pilot" / "interactions.csv"
OUT_TABLE = REPO_ROOT / "data" / "pilot" / "action_labels.csv"
PRIMARY_ONSET = "lat_vel"


def main():
    cfg, cfg_hash = load_config(CFG_PATH)
    out_dir = prepare_output_dir("step2/action_labels", cfg_hash, CFG_PATH)
    df = pd.concat([pd.read_csv(f) for f in sorted(PROCESSED_DIR.glob("*.csv.gz"))],
                   ignore_index=True)
    inter = pd.read_csv(INTER_TABLE)

    tab = build_label_table(df, inter, cfg)
    tab.to_csv(OUT_TABLE, index=False)
    report = {"config_hash": cfg_hash, "n_labeled": int(len(tab))}

    prim = tab[tab["onset_def"] == PRIMARY_ONSET]
    print(f"config hash = {cfg_hash};labeled 行 {len(tab)}"
          f"(主口径 onset={PRIMARY_ONSET}:{len(prim)})\n")

    # 1) 比例
    props = label_proportions(prim, ["segment"])
    report["proportions_primary_onset"] = props.to_dict(orient="records")
    report["proportions_all"] = label_proportions(
        tab, ["onset_def", "segment"]).to_dict(orient="records")
    print("== 标签比例(onset=lat_vel)==")
    print(f"{'definition':12s} {'segment':26s} {'n':>5s} {'yield':>7s} {'hold':>7s} {'ambig':>7s}")
    for r in props.to_dict(orient="records"):
        print(f"{r['definition']:12s} {r['segment']:26s} {r['n']:5d} "
              f"{r['yield']:7.3f} {r['hold']:7.3f} {r['ambiguous']:7.3f}")

    # 2) 一致率
    agree = pairwise_agreement(prim)
    report["pairwise_agreement_primary_onset"] = agree
    print("\n== 定义两两一致率(非 ambiguous 共同子集)==")
    for k, v in agree.items():
        print(f"{k:26s} agree={v['agreement_non_ambiguous']:.3f} "
              f"(n={v['n_non_ambiguous']})")
        conf = pd.DataFrame(v["confusion"], index=LABELS, columns=LABELS)
        print(conf.to_string(), "\n")

    # 3) 敏感性
    sens = sensitivity_flips(prim)
    report["sensitivity_primary_onset"] = sens
    print("== ±30% 阈值敏感性(标签翻转比例)==")
    for d, v in sens.items():
        print(f"{d:12s} x0.7 -> flip {v['flip_lo']:.3f};x1.3 -> flip {v['flip_hi']:.3f}")

    # 4) 两信号族相关
    corr = float(prim["dv_red"].corr(prim["a_end"]))
    report["corr_dvred_aend_primary_onset"] = corr
    print(f"\ncorr(dv_red, a_end) = {corr:.3f}")

    with open(out_dir / f"action_labels_report_{cfg_hash}.json", "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nlabels -> {OUT_TABLE}\n完成。")


if __name__ == "__main__":
    main()