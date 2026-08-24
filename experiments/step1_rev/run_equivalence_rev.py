"""R4(新 C5)实验入口:等价脊线,M0a / M0 / M1 同框。

运行(仓库根目录):
    python experiments/step1_rev/run_equivalence_rev.py

核心对比(revision 文档):
    M0: context diversity cannot remove structural collapse
    M1: only the theoretically required context diversity can remove it
判别量:(kappa_s, kappa_e) 切片(w_s 固定于参考格点)内 3-context
累积等价集合的 Chebyshev 直径 vs 单 context —— M0 谱系保持,
M1 在 strong 下收缩到格点级、insufficient 下分毫不动、moderate 居中。

产出(outputs/step1rev_equiv_<hash>/):
  - fig_key_contrast_<hash>.pdf/.png   3x3(模型 x 家族)kappa 切片:
      1-context(浅)与 3-context(深)集合叠加 + 理论 collapse 曲线
  - fig_shrink_<hash>.pdf/.png         切片直径与 3D 体积随 context 数的变化
  - fig_wks_<model>_<hash>.pdf/.png    (w_s, kappa_s) 平面累积叠加(每模型一张)
  - equivalence_rev_metrics_<hash>.json 全部参考点的数值指标
  - config_used_<hash>.yaml
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.step1.common import REPO_ROOT, load_config, prepare_output_dir, save_figure
from src.analysis.direct_scan import (
    build_grid,
    collapse_weights,
    kappa_slice_diameter,
    q_grids,
    snap,
    theory_ridge_ke,
)
from src.analysis.equivalence import (
    count_components_2d,
    cumulative_masks,
    normalized_coords,
    set_metrics,
    slice_ks_ke,
    slice_w_ks,
)
from src.games.contexts import PerceptionModel

REVISION_CONFIG = REPO_ROOT / "configs" / "step1_revision.yaml"
CUM_COLORS = ["#c6dbef", "#6baed6", "#08306b"]


def _metrics_for(model, ctxs, grid, ridx, tol, norm_axes):
    qs = q_grids(model, ctxs, grid)
    cums = cumulative_masks(qs, ridx, tol)
    out = {}
    for n, m in enumerate(cums, start=1):
        e = set_metrics(m, norm_axes)
        e["kk_slice_diameter"] = kappa_slice_diameter(m, ridx, grid)
        e["kk_slice_count"] = int(slice_ks_ke(m, ridx).sum())
        e["n_components_kk"] = count_components_2d(slice_ks_ke(m, ridx))
        out[f"contexts_{n}"] = e
    return out, cums


def main():
    cfg, cfg_hash = load_config(REVISION_CONFIG)
    out_dir = prepare_output_dir("step1rev_equiv", cfg_hash, REVISION_CONFIG)
    pm = PerceptionModel(cfg["perception"])
    families = {f: pm.build_context_set(c) for f, c in cfg["context_families"].items()}
    eq = cfg["equivalence_rev"]
    grid = build_grid(eq["grid"])
    norm_axes = normalized_coords(grid)

    print(f"config hash = {cfg_hash}, outputs -> {out_dir}")
    report = {}
    primary = eq["primary_reference"]
    masks_primary = {}  # (model, fam) -> cums,主参考点,画图用

    for ref_name, ref in eq["reference_points"].items():
        ridx = snap(grid, ref)
        report[ref_name] = {"ref_grid_point": {
            "w_s": float(grid.w_s_axis[ridx[0]]),
            "kappa_s": float(grid.kappa_s_axis[ridx[1]]),
            "kappa_e": float(grid.kappa_e_axis[ridx[2]]),
        }}
        for model in cfg["models"]:
            report[ref_name][model] = {}
            for fam, ctxs in families.items():
                m, cums = _metrics_for(model, ctxs, grid, ridx, eq["tol"], norm_axes)
                report[ref_name][model][fam] = m
                if ref_name == primary:
                    masks_primary[(model, fam)] = (cums, ridx)
                    d1 = m["contexts_1"]["kk_slice_diameter"]
                    d3 = m["contexts_3"]["kk_slice_diameter"]
                    print(f"{model:4s} {fam:22s} kk-slice diam 1->3: {d1:.2f} -> {d3:.2f}")

    # --- 关键对比图:3x3 kappa 切片 ---
    fam_names = list(families.keys())
    fig, axes = plt.subplots(3, 3, figsize=(12, 10.5), sharex=True, sharey=True)
    for r, model in enumerate(cfg["models"]):
        for c_i, fam in enumerate(fam_names):
            ax = axes[r, c_i]
            cums, ridx = masks_primary[(model, fam)]
            for m, color, alpha in [(cums[0], CUM_COLORS[0], 0.9), (cums[2], CUM_COLORS[2], 0.95)]:
                s = slice_ks_ke(m, ridx).astype(float)
                if s.any():
                    ax.contourf(grid.kappa_e_axis, grid.kappa_s_axis, s,
                                levels=[0.5, 1.5], colors=[color], alpha=alpha)
            # 理论 collapse 曲线(M1 仅 insufficient 家族有单一方向)
            G = families[fam][0].G if fam == "insufficient_diversity" else None
            w_ref = grid.w_s_axis[ridx[0]]
            ab = collapse_weights(model, float(w_ref), G)
            if ab is not None:
                ke_curve = theory_ridge_ke(
                    grid.kappa_s_axis, ab[0], ab[1],
                    grid.kappa_s_axis[ridx[1]], grid.kappa_e_axis[ridx[2]],
                )
                ax.plot(ke_curve, grid.kappa_s_axis, "r--", lw=1.0)
            ax.plot(grid.kappa_e_axis[ridx[2]], grid.kappa_s_axis[ridx[1]],
                    "r*", ms=10, mec="k", mew=0.4)
            ax.set_xscale("log")
            ax.set_yscale("log")
            d1 = report[primary][model][fam]["contexts_1"]["kk_slice_diameter"]
            d3 = report[primary][model][fam]["contexts_3"]["kk_slice_diameter"]
            ax.set_title(f"{model.upper()} | {fam.replace('_diversity','')}  "
                         f"d:{d1:.2f}\u2192{d3:.2f}", fontsize=8)
            if r == 2:
                ax.set_xlabel(r"$\kappa_e$", fontsize=8)
            if c_i == 0:
                ax.set_ylabel(r"$\kappa_s$", fontsize=8)
    fig.suptitle(
        "kappa-plane equivalence sets at fixed $w_s$ (light = 1 context, dark = 3 contexts;\n"
        "red dashed = theoretical collapse level set) — "
        "M0-lineage ridges persist; M1 ridge removed only under required diversity",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, out_dir, "fig_key_contrast", cfg_hash)
    plt.close(fig)

    # --- 收缩曲线图 ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    styles = {"strong_diversity": "-", "moderate_diversity": "--",
              "insufficient_diversity": ":"}
    colors = {"m0a": "#7f7f7f", "m0": "#1f77b4", "m1": "#d62728"}
    x = [1, 2, 3]
    for model in cfg["models"]:
        for fam in fam_names:
            d = [report[primary][model][fam][f"contexts_{n}"]["kk_slice_diameter"] for n in x]
            v = [report[primary][model][fam][f"contexts_{n}"]["volume_fraction"] for n in x]
            axes[0].plot(x, d, styles[fam], color=colors[model],
                         label=f"{model} {fam.split('_')[0]}")
            axes[1].plot(x, np.maximum(v, 1e-6), styles[fam], color=colors[model])
    axes[0].set_ylabel("kk-slice Chebyshev diameter")
    axes[1].set_ylabel("3D volume fraction")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("number of contexts")
        ax.set_xticks(x)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6, ncol=3)
    fig.suptitle("equivalence-set shrinkage: collapse-direction survival vs total volume",
                 fontsize=10)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_shrink", cfg_hash)
    plt.close(fig)

    # --- (w_s, kappa_s) 平面(每模型一张,strong 家族)---
    for model in cfg["models"]:
        cums, ridx = masks_primary[(model, "strong_diversity")]
        fig, ax = plt.subplots(figsize=(5.2, 4))
        for n_sit, (m, color) in enumerate(zip(cums, CUM_COLORS), start=1):
            s = slice_w_ks(m, ridx).astype(float)
            if s.any():
                ax.contourf(grid.kappa_s_axis, grid.w_s_axis, s,
                            levels=[0.5, 1.5], colors=[color], alpha=0.85)
        ax.plot(grid.kappa_s_axis[ridx[1]], grid.w_s_axis[ridx[0]],
                "r*", ms=11, mec="k", mew=0.5)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\kappa_s$")
        ax.set_ylabel(r"$w_s$")
        ax.set_title(f"{model.upper()} cumulative 1/2/3 contexts (strong family)", fontsize=9)
        fig.tight_layout()
        save_figure(fig, out_dir, f"fig_wks_{model}", cfg_hash)
        plt.close(fig)

    with open(out_dir / f"equivalence_rev_metrics_{cfg_hash}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("完成。")


if __name__ == "__main__":
    main()