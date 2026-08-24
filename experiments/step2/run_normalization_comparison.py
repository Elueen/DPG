"""任务 F 实验入口:归一化三方案对比(选择留人工冻结)。

运行(仓库根目录,需已有重跑后的 data/pilot/utilities_raw.csv):
    python experiments/step2/run_normalization_comparison.py

三方案(T_k 作用在 u_k^raw 上;G_k^norm = G_k^raw / b_k^2):
  F-A intrinsic physics-based scaling:T_k(u) = u(b_k = 1)。
      归一化内生于 utility 设计(margin=0 <-> u=0,physical scale 控制
      饱和,值域 (-1,1));保留外生 site-level reference speed。
  F-B pooled pilot affine:T_k(u) = 2*(u - P5_k)/(P95_k - P5_k) - 1,
      P5/P95 为四 segment 合并(逐分支合并 Y+H 样本);**纯仿射,
      不 clipping**(硬截断在边界不可微,破坏 ∇f_k 与 G_k;
      越界值如实保留)。b_k = (P95_k - P5_k)/2。
  F-C dataset-specific affine:同 F-B 但逐 dataset 分别取分位——
      **diagnostic only**,展示相同物理状态因数据集不同而得到不同
      utility 的量级(domain shortcut 反例)。

输出对比表:
  - 各方案 b_k 与 G_k 缩放因子 1/b_k^2
  - 归一化后 u 分布(P5/P50/P95,逐 segment)
  - 物理锚点保持性:margin=0(u^raw=0)映射到的 T_k(0)
  - same-state cross-dataset consistency:u^raw 网格点在 F-C 下
    highD 版 vs NGSIM 版的最大差
  - Δu 尺度与越界(|T|>1)比例(F-B/F-C)
选择等待人工裁决后写入 docs/step2_normalization_spec.md 冻结。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.step1.common import REPO_ROOT, load_config, prepare_output_dir

CFG_PATH = REPO_ROOT / "configs" / "step2_pilot.yaml"
UTIL_TABLE = REPO_ROOT / "data" / "pilot" / "utilities_raw.csv"


def _affine(u, p5, p95):
    return 2.0 * (u - p5) / (p95 - p5) - 1.0


def main():
    cfg, cfg_hash = load_config(CFG_PATH)
    out_dir = prepare_output_dir("step2/normalization", cfg_hash, CFG_PATH)
    tab = pd.read_csv(UTIL_TABLE)
    report = {"config_hash": cfg_hash, "n_events": int(len(tab))}
    print(f"config hash = {cfg_hash};events = {len(tab)}\n")

    schemes = {}
    for k, cols in (("s", ["u_s_Y", "u_s_H"]), ("e", ["u_e_Y", "u_e_H"])):
        pooled = pd.concat([tab[c] for c in cols])
        p5, p95 = pooled.quantile(0.05), pooled.quantile(0.95)
        schemes[k] = {"FA": {"b": 1.0, "a": 0.0},
                      "FB": {"b": (p95 - p5) / 2.0, "p5": p5, "p95": p95}}
        per_ds = {}
        for ds, g in tab.groupby("dataset"):
            pd_ = pd.concat([g[c] for c in cols])
            per_ds[ds] = {"p5": pd_.quantile(0.05), "p95": pd_.quantile(0.95)}
        schemes[k]["FC"] = per_ds

    # --- b_k 与 G 缩放 ---
    print("== b_k 与 G_k 缩放因子(1/b^2)==")
    for k in ("s", "e"):
        bB = schemes[k]["FB"]["b"]
        print(f"  u_{k}: F-A b=1.000(G x1.000);"
              f"F-B b={bB:.3f}(G x{1/bB**2:.3f});F-C 逐数据集:")
        for ds, q in schemes[k]["FC"].items():
            b = (q["p95"] - q["p5"]) / 2.0
            print(f"        {ds:6s} b={b:.3f}(G x{1/b**2:.3f})")
    report["schemes"] = {k: {"FB_b": float(schemes[k]["FB"]["b"]),
                             "FC_b": {ds: float((q["p95"] - q["p5"]) / 2.0)
                                      for ds, q in schemes[k]["FC"].items()}}
                        for k in ("s", "e")}

    # --- 物理锚点保持性 ---
    print("\n== 物理锚点保持性:margin=0(u^raw=0)映射到 T(0) ==")
    for k in ("s", "e"):
        fb = schemes[k]["FB"]
        t0B = _affine(0.0, fb["p5"], fb["p95"])
        print(f"  u_{k}: F-A -> 0.000;F-B -> {t0B:+.3f};F-C ->", end=" ")
        for ds, q in schemes[k]["FC"].items():
            print(f"{ds}:{_affine(0.0, q['p5'], q['p95']):+.3f}", end="  ")
        print()

    # --- same-state cross-dataset consistency(F-C 反例量级)---
    grid = np.linspace(-0.9, 0.9, 19)
    print("\n== F-C same-state 不一致(u^raw 网格上 highd 版 vs ngsim 版最大差)==")
    for k in ("s", "e"):
        q_h = schemes[k]["FC"].get("highd")
        q_n = schemes[k]["FC"].get("ngsim")
        diff = np.abs(_affine(grid, q_h["p5"], q_h["p95"])
                      - _affine(grid, q_n["p5"], q_n["p95"]))
        report[f"FC_same_state_maxdiff_u{k}"] = float(diff.max())
        print(f"  u_{k}: max|Δ| = {diff.max():.3f}(相同物理状态,不同数据集)")

    # --- 归一化后分布 + 越界 + Δu 尺度 ---
    print("\n== 归一化后 H 分支分布(P5/P50/P95)与 |T|>1 越界比例 ==")
    for k, uc in (("s", "u_s_H"), ("e", "u_e_H")):
        fb = schemes[k]["FB"]
        tB = _affine(tab[uc], fb["p5"], fb["p95"])
        print(f"  u_{k} F-A {tab[uc].quantile([.05, .5, .95]).round(3).tolist()}"
              f"  F-B {tB.quantile([.05, .5, .95]).round(3).tolist()}"
              f"(越界 {float((tB.abs() > 1).mean()):.3f})")
    print("\n== Δu 尺度(sd)==")
    for k, dc in (("s", "du_s_raw"), ("e", "du_e_raw")):
        bB = schemes[k]["FB"]["b"]
        print(f"  Δu_{k}: F-A sd={tab[dc].std():.4f};"
              f"F-B sd={tab[dc].std() / bB:.4f}(=raw/b)")

    with open(out_dir / f"normalization_comparison_{cfg_hash}.json", "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    print("\n选择等待人工裁决(预期 F-A);冻结写 docs/step2_normalization_spec.md。")
    print("完成。")


if __name__ == "__main__":
    main()