"""Step 2 任务 A 辅助:扫描 highD recordingMeta,打印密度统计供挑选。

运行(仓库根目录,先在 configs/step2_pilot.yaml 填 highd_root):
    python experiments/step2/list_highd_candidates.py

输出每个 recording 的:站点、时长、车辆数、车辆数/分钟(密度代理)。
挑 1 个偏自由流 + 1 个偏高密度(不同 locationId),把 ID 填进 config
的 highd.recordings。本脚本只读元数据,不动轨迹文件。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.step1.common import REPO_ROOT, load_config

CFG_PATH = REPO_ROOT / "configs" / "step2_pilot.yaml"


def main():
    cfg, _ = load_config(CFG_PATH)
    root = Path(cfg["paths"]["highd_root"])
    metas = sorted(root.glob("*_recordingMeta.csv"))
    if not metas:
        raise FileNotFoundError(f"{root} 下未找到 *_recordingMeta.csv,请检查 config 路径")
    rows = []
    for f in metas:
        m = pd.read_csv(f).iloc[0]
        dur_min = float(m["duration"]) / 60.0
        n_veh = float(m["numVehicles"])
        rows.append({
            "recording": int(m["id"]),
            "location": int(m["locationId"]),
            "duration_min": round(dur_min, 1),
            "vehicles": int(n_veh),
            "veh_per_min": round(n_veh / dur_min, 1),
            "speed_limit": float(m.get("speedLimit", float("nan"))),
        })
    df = pd.DataFrame(rows).sort_values("veh_per_min")
    print(df.to_string(index=False))
    print("\n提示:veh_per_min 低端选 free_flow,高端选 dense;"
          "两个 recording 须不同 locationId。选定后填入 config highd.recordings。")


if __name__ == "__main__":
    main()