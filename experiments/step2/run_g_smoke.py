"""任务 G 实验入口:最终 numerical smoke test(Step 2 收官)。

运行(仓库根目录,需已有 data/pilot/action_labels.csv):
    python experiments/step2/run_g_smoke.py

对全部 valid & kin_physical(onset=lat_vel)interactions 走统一接口
m1_quantities(CanonicalState, v_ref, cfg),输出四量与冒烟检查。
注意:1123 全跑仅因计算便宜;这仍是 numerical smoke test,
不是 Step 3 empirical analysis(G_k 总体 variation 是否充分不在此判断)。

输出:
  data/pilot/m1_quantities.csv                      (git 忽略)
  outputs/step2/g_smoke_<hash>/g_smoke_report_<hash>.json
检查(G4):finite / 无 NaN / 无 Inf / G_k>=0 / G_k 不全零 /
无数值爆炸 / Δu 不全退化 / 随机 10 行 sanity /
Sigma alpha∈{0.5,1,2} 线性缩放精确(实现正确性测试)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.step1.common import REPO_ROOT, load_config, prepare_output_dir
from src.m1_interface import CanonicalState, m1_quantities

CFG_PATH = REPO_ROOT / "configs" / "step2_pilot.yaml"
LABEL_TABLE = REPO_ROOT / "data" / "pilot" / "action_labels.csv"
OUT_TABLE = REPO_ROOT / "data" / "pilot" / "m1_quantities.csv"
PRIMARY_ONSET = "lat_vel"


def main():
    cfg, cfg_hash = load_config(CFG_PATH)
    out_dir = prepare_output_dir("step2/g_smoke", cfg_hash, CFG_PATH)
    vref_map = cfg["utility"]["v_ref_mps"]
    lab = pd.read_csv(LABEL_TABLE)
    lab = lab[(lab["onset_def"] == PRIMARY_ONSET) & lab["valid"]
              & lab["opp_kin_physical"]]

    rows = []
    for _, r in lab.iterrows():
        seg = str(r["segment"])
        site = ("loc2" if "rec03" in seg else "loc1" if "rec13" in seg
                else "i80" if "i80" in seg else "us101")
        st = CanonicalState(gap0=float(r["gap0"]), v_ego0=float(r["v_ego0"]),
                            v_opp0=float(r["v_opp0"]), site=site,
                            event_id=str(r["event_id"]))
        q = m1_quantities(st, float(vref_map[site]), cfg)
        q["segment"] = seg
        rows.append(q)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT_TABLE, index=False)

    report = {"config_hash": cfg_hash, "n_events": int(len(tab))}
    print(f"config hash = {cfg_hash};events = {len(tab)}"
          f"(numerical smoke test,非 Step 3 empirical analysis)\n")

    cols = ["du_s", "du_e", "G_s", "G_e"]
    checks = {
        "all_finite": bool(np.isfinite(tab[cols].to_numpy()).all()),
        "no_nan": bool(tab[cols].notna().all().all()),
        "G_nonneg": bool((tab["G_s"] >= 0).all() and (tab["G_e"] >= 0).all()),
        "G_not_all_zero": bool(tab["G_s"].max() > 0 and tab["G_e"].max() > 0),
        "no_explosion_G_max": float(max(tab["G_s"].max(), tab["G_e"].max())),
        "du_not_all_degenerate": bool((tab["du_s"].abs() > 1e-6).mean() > 0.5),
    }
    report["g4_checks"] = checks
    print("== G4 冒烟检查 ==")
    for k, v in checks.items():
        print(f"  {k}: {v}")

    print("\n== 四量分布(P5/P50/P95)==")
    for c in cols:
        print(f"  {c:5s} {tab[c].quantile([.05, .5, .95]).round(4).tolist()}")
    print("\n逐 segment G 中位:")
    for seg, g in tab.groupby("segment"):
        print(f"  {seg:26s} G_s={g['G_s'].median():.4f}  G_e={g['G_e'].median():.4f}")

    # Sigma alpha 线性(E3 预注册 + 实现正确性测试)
    print("\n== Sigma alpha-linearity(抽 50 事件,应精确等于 alpha)==")
    rng = np.random.default_rng(int(cfg["meta"]["seed"]))
    sub = lab.sample(min(50, len(lab)), random_state=rng)
    ratios = {a: [] for a in cfg["perception_noise"]["sens_sigma"] if a != 1.0}
    for _, r in sub.iterrows():
        seg = str(r["segment"])
        site = ("loc2" if "rec03" in seg else "loc1" if "rec13" in seg
                else "i80" if "i80" in seg else "us101")
        st = CanonicalState(float(r["gap0"]), float(r["v_ego0"]),
                            float(r["v_opp0"]))
        base = m1_quantities(st, float(vref_map[site]), cfg, sigma_scale=1.0)
        for a in ratios:
            q = m1_quantities(st, float(vref_map[site]), cfg, sigma_scale=a)
            ratios[a].append(q["G_s"] / base["G_s"])
            ratios[a].append(q["G_e"] / base["G_e"])
    alpha_ok = True
    for a, vals in ratios.items():
        dev = float(np.max(np.abs(np.array(vals) - a)))
        alpha_ok &= dev < 1e-9
        print(f"  alpha={a}: max|ratio - alpha| = {dev:.2e}")
    report["alpha_linearity_ok"] = bool(alpha_ok)

    print("\n== 随机 10 行 sanity ==")
    show = tab.sample(10, random_state=rng)[
        ["event_id", "du_s", "du_e", "G_s", "G_e"]].round(4)
    print(show.to_string(index=False))

    gate = all([checks["all_finite"], checks["no_nan"], checks["G_nonneg"],
                checks["G_not_all_zero"], checks["du_not_all_degenerate"],
                alpha_ok])
    report["g_smoke_pass"] = bool(gate)
    print(f"\n=> G SMOKE {'PASS' if gate else 'FAIL'}")

    with open(out_dir / f"g_smoke_report_{cfg_hash}.json", "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"m1_quantities -> {OUT_TABLE}\n完成。")


if __name__ == "__main__":
    main()