"""任务 B(B5)实验入口:人工目检轨迹图。

运行(仓库根目录,需已有 processed / interactions.csv / action_labels.csv):
    python experiments/step2/run_action_validation.py

抽样(seed=config):主口径 onset=lat_vel,每定义 x 每标签(yield/hold/
ambiguous)最多 n_per_cell 张,跨 segment 分层。每张图五联:
  (1) 纵向位置:ego / opponent 实线 + opponent 匀速外推虚线
  (2) ego 横向位置 + 原/目标车道经验中心横线
  (3) 速度:ego / opponent
  (4) opponent 纵向加速度
  (5) ego-opponent bumper gap
标注:t0(实竖线)、t_cross(虚竖线)、标题含三定义标签与关键特征。
产出 outputs/step2/action_validation/<def>__<label>__<event>.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.step1.common import REPO_ROOT, load_config
from src.data.actions import DEFS, LABELS
from src.data.canonical import lane_centers

CFG_PATH = REPO_ROOT / "configs" / "step2_pilot.yaml"
PROCESSED_DIR = REPO_ROOT / "data" / "pilot" / "processed"
LABEL_TABLE = REPO_ROOT / "data" / "pilot" / "action_labels.csv"
OUT_DIR = REPO_ROOT / "outputs" / "step2" / "action_validation"
PRIMARY_ONSET = "lat_vel"
N_PER_CELL = 7          # 每定义 x 每标签,合计每定义 <= 21 张(任务书 10-20 量级)


def _plot_case(row, df, centers, h_pre, h_resp, path):
    seg = row["segment"]
    t0, tc = float(row["t0"]), float(row["t_cross"])
    ta, tb = t0 - h_pre, t0 + max(h_resp, tc - t0 + 1.0)
    d = df[df["segment"] == seg]
    ego = d[(d["vid"] == row["ego_vid"]) & d["t"].between(ta, tb)].sort_values("t")
    opp = d[(d["vid"] == row["opp_vid"]) & d["t"].between(ta, tb)].sort_values("t")
    o0 = opp[opp["t"] >= t0]
    x_cv = float(o0["x"].iloc[0]) + float(o0["vx"].iloc[0]) * (o0["t"] - t0)
    m = ego.merge(opp, on="t", suffixes=("_e", "_o"))
    gap = m["x_e"] - m["x_o"] - 0.5 * (m["length_e"] + m["length_o"])

    fig, axes = plt.subplots(5, 1, figsize=(7.5, 11), sharex=True)
    axes[0].plot(ego["t"], ego["x"], label="ego x")
    axes[0].plot(opp["t"], opp["x"], label="opp x")
    axes[0].plot(o0["t"], x_cv, "--", color="gray", label="opp const-v")
    axes[0].set_ylabel("x [m]")
    seg_c = centers[centers["segment"] == seg].set_index("lane")["y_center"]
    axes[1].plot(ego["t"], ego["y"], color="tab:blue")
    for ln in (row["lane_from"], row["lane_to"]):
        if ln in seg_c.index:
            axes[1].axhline(seg_c[ln], color="gray", lw=0.7, ls=":")
    axes[1].set_ylabel("ego y [m]")
    axes[2].plot(ego["t"], ego["vx"], label="ego v")
    axes[2].plot(opp["t"], opp["vx"], label="opp v")
    axes[2].set_ylabel("v [m/s]")
    axes[3].plot(opp["t"], opp["ax"], color="tab:red")
    axes[3].axhline(0, color="gray", lw=0.5)
    axes[3].set_ylabel("opp ax [m/s2]")
    axes[4].plot(m["t"], gap, color="tab:green")
    axes[4].set_ylabel("gap [m]")
    axes[4].set_xlabel("t [s]")
    for ax in axes:
        ax.axvline(t0, color="k", lw=1.0)
        ax.axvline(tc, color="k", lw=1.0, ls="--")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7)
    axes[2].legend(fontsize=7)
    fig.suptitle(
        f"{row['event_id']}  onset={PRIMARY_ONSET}\n"
        f"D1={row['d1_decel']}  D2={row['d2_gap']}  D3={row['d3_compound']}   "
        f"dv_red={row['dv_red']:.2f} m/s  a_end={row['a_end']:.2f} m  "
        f"gap0={row['gap0']:.1f} m",
        fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    cfg, cfg_hash = load_config(CFG_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(f) for f in sorted(PROCESSED_DIR.glob("*.csv.gz"))],
                   ignore_index=True)
    tab = pd.read_csv(LABEL_TABLE)
    tab = tab[tab["onset_def"] == PRIMARY_ONSET]
    centers = lane_centers(df, cfg["canonical"]["lane_center_min_obs"])
    rng = np.random.default_rng(cfg["meta"]["seed"])
    h_pre = float(cfg["interaction"]["h_pre_s"])
    h_resp = float(cfg["interaction"]["h_response_s"])

    n_done = 0
    for d in DEFS:
        for lab in LABELS:
            pool = tab[tab[d] == lab]
            if pool.empty:
                continue
            # 跨 segment 分层:每 segment 轮流抽,直到 N_PER_CELL
            picks = []
            for seg, g in pool.groupby("segment"):
                k = max(1, N_PER_CELL // pool["segment"].nunique())
                picks.append(g.sample(min(k, len(g)), random_state=rng))
            sample = pd.concat(picks).head(N_PER_CELL)
            for _, row in sample.iterrows():
                fname = (f"{d}__{lab}__" + row["event_id"].replace("|", "_")
                         + ".png")
                _plot_case(row, df, centers, h_pre, h_resp, OUT_DIR / fname)
                n_done += 1
    print(f"目检图 {n_done} 张 -> {OUT_DIR}")


if __name__ == "__main__":
    main()