"""任务 D 实验入口 1:计算外生 v_ref(逐 site pilot population P85)。

运行(仓库根目录,需已有 processed 数据):
    python experiments/step2/compute_vref.py

打印逐 site 的 vx P85(排除 vx<0.5 的静止帧,避免拥堵停车压低参照)。
把打印的值填入 configs/step2_pilot.yaml 的 utility.v_ref_mps 段并提交
(一次计算,永久冻结;外生参照,不用个体未来轨迹)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.step1.common import REPO_ROOT

PROCESSED_DIR = REPO_ROOT / "data" / "pilot" / "processed"
MIN_MOVING_MPS = 0.5


def main():
    df = pd.concat([pd.read_csv(f) for f in sorted(PROCESSED_DIR.glob("*.csv.gz"))],
                   ignore_index=True)
    moving = df[df["vx"] >= MIN_MOVING_MPS]
    print("逐 site vx P85(m/s,排除 vx<0.5 静止帧)——填入 config utility.v_ref_mps:\n")
    for site, g in moving.groupby("site"):
        print(f"  {site}: {g['vx'].quantile(0.85):.2f}")


if __name__ == "__main__":
    main()