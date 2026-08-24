"""任务 B(前半)实验入口:pilot 数据上抽取 interaction 表。

运行(仓库根目录,需已有 data/pilot/processed/):
    python experiments/step2/run_interactions.py

输出:
  data/pilot/interactions.csv                       (git 忽略,后续 B/C 共用)
  outputs/step2/interactions_<hash>/interactions_stats_<hash>.json
打印:每 segment x onset 的事件数 / valid 数 / invalid 原因分解,
以及三个 onset 定义间的 t0 一致性(共同 valid 事件上的 t0 差)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.step1.common import REPO_ROOT, load_config, prepare_output_dir
from src.data.interactions import extract_interactions

CFG_PATH = REPO_ROOT / "configs" / "step2_pilot.yaml"
PROCESSED_DIR = REPO_ROOT / "data" / "pilot" / "processed"
OUT_TABLE = REPO_ROOT / "data" / "pilot" / "interactions.csv"


def main():
    cfg, cfg_hash = load_config(CFG_PATH)
    out_dir = prepare_output_dir("step2/interactions", cfg_hash, CFG_PATH)
    files = sorted(PROCESSED_DIR.glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"{PROCESSED_DIR} 为空:先跑 run_pilot_prep.py")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    tab = extract_interactions(df, cfg)
    tab.to_csv(OUT_TABLE, index=False)

    stats = {"config_hash": cfg_hash, "n_rows": int(len(tab)), "per_segment": {}}
    print(f"config hash = {cfg_hash}\n")
    print(f"{'segment':26s} {'onset':10s} {'events':>7s} {'valid':>6s}  invalid 原因")
    for (seg, oname), g in tab.groupby(["segment", "onset_def"]):
        reasons = g[~g["valid"]]["invalid_reason"].value_counts().to_dict()
        stats["per_segment"].setdefault(seg, {})[oname] = {
            "events": int(len(g)),
            "valid": int(g["valid"].sum()),
            "invalid_reasons": reasons,
        }
        rtxt = ", ".join(f"{k}:{v}" for k, v in sorted(reasons.items()))
        print(f"{seg:26s} {oname:10s} {len(g):7d} {int(g['valid'].sum()):6d}  {rtxt}")

    # onset 间 t0 一致性(同一事件、三定义都 valid)
    v = tab[tab["valid"]]
    piv = v.pivot_table(index="event_id", columns="onset_def", values="t0")
    both = piv.dropna()
    if len(both) and {"fixed_lead", "lat_vel"} <= set(both.columns):
        d1 = (both["fixed_lead"] - both["lat_vel"]).abs()
        stats["t0_agreement"] = {
            "n_common_valid": int(len(both)),
            "median_abs_dt_fixed_vs_latvel_s": float(d1.median()),
        }
        print(f"\n共同 valid 事件 {len(both)} 个;"
              f"|t0(fixed_lead) - t0(lat_vel)| 中位 = {d1.median():.2f}s")

    with open(out_dir / f"interactions_stats_{cfg_hash}.json", "w",
              encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\ninteractions -> {OUT_TABLE}")
    print("完成。")


if __name__ == "__main__":
    main()