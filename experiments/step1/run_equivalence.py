"""C5 实验入口:等价脊线分析(核心产出)。

运行(仓库根目录,需先跑 run_grid_scan.py):
    python experiments/step1/run_equivalence.py

输入:outputs/step1_scan_<当前config hash>/ 下的 scan npz
(强制哈希匹配:config 改动后须重跑扫描,保证分析与数据同源)

产出(outputs/step1_equiv_<hash>/,纪律 3):
  - fig_ridge_wks_{ref}_{hash}.pdf/.png   (w_s, kappa_s) 平面等价脊线:
      三个情境的单情境集合 + 1/2/3 情境累积交集
  - fig_ridge_kk_{ref}_{hash}.pdf/.png    (kappa_s, kappa_e) 平面,
      叠加理论双曲线 1/ks + 1/ke = const(C2 结构性预言)
  - fig_shrinkage_{hash}.pdf/.png         情境数 vs 等价集合体积/直径(关键对比图)
  - fig_robust_{profile}_{ref}_{hash}.pdf/.png  ego 画像鲁棒性:粗网格脊线拓扑对比
  - equivalence_metrics_{hash}.json       全部数值指标
  - config_used_{hash}.yaml
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

from experiments.step1.common import OUTPUTS_ROOT, load_config, prepare_output_dir, save_figure
from src.analysis.equivalence import (
    check_same_grid,
    count_components_2d,
    cumulative_masks,
    kappa_aggregate_hyperbola,
    load_scan,
    normalized_coords,
    set_metrics,
    slice_ks_ke,
    slice_w_ks,
    snap_reference,
)

SINGLE_COLORS = ["#1f77b4", "#2ca02c", "#9467bd"]
CUM_COLORS = ["#c6dbef", "#6baed6", "#08306b"]


# ----------------------------------------------------------------------
# 数据装载
# ----------------------------------------------------------------------
def _load_profile_scans(scan_dir: Path, cfg, cfg_hash: str, profile: str) -> list:
    order = cfg["equivalence"]["situation_order"]
    scans = []
    for situation in order:
        f = scan_dir / f"scan_{situation}_{profile}_{cfg_hash}.npz"
        if not f.exists():
            raise FileNotFoundError(
                f"缺少 {f}。config 改动会更换 hash:请先重跑 run_grid_scan.py"
            )
        scans.append(load_scan(f))
    check_same_grid(scans)
    return scans


# ----------------------------------------------------------------------
# 画图
# ----------------------------------------------------------------------
def _plot_mask_panels(
    axes_extent_x, axes_extent_y, singles, cums, order,
    xlab, ylab, xlog, ref_xy, title, hyperbola=None,
):
    """4 面板:三个单情境集合 + 累积交集叠加。返回 fig。"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, m, name, color in zip(axes[:3], singles, order, SINGLE_COLORS):
        ax.pcolormesh(
            axes_extent_x, axes_extent_y, m.astype(float), shading="nearest",
            cmap=matplotlib.colors.ListedColormap(["white", color]), vmin=0, vmax=1,
        )
        ax.set_title(f"single: {name}", fontsize=9)
    for n_sit, (m, color) in enumerate(zip(cums, CUM_COLORS), start=1):
        axes[3].contourf(
            axes_extent_x, axes_extent_y, m.astype(float), levels=[0.5, 1.5],
            colors=[color], alpha=0.85,
        )
    axes[3].set_title("cumulative: 1 / 2 / 3 situations", fontsize=9)
    for ax in axes:
        if xlog:
            ax.set_xscale("log")
        if hyperbola is not None:
            ax.plot(hyperbola[0], hyperbola[1], "r--", lw=1.2, label="1/ks+1/ke=const")
            ax.set_yscale("log")
        ax.plot(*ref_xy, marker="*", c="red", ms=11, mec="k", mew=0.5)
        ax.set_xlabel(xlab, fontsize=9)
    axes[0].set_ylabel(ylab, fontsize=9)
    if hyperbola is not None:
        axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def _ridge_figures(scans, ref_name, ref_idx, cfg, out_dir, cfg_hash, profile, suffix=""):
    order = cfg["equivalence"]["situation_order"]
    tol = cfg["equivalence"]["tol"]
    q_list = [s.q_eq for s in scans]
    cums = cumulative_masks(q_list, ref_idx, tol)
    # single mask 对每个情境用该情境自身的参考 q* 值
    singles = [np.abs(q - q[ref_idx]) < tol for q in q_list]
    s0 = scans[0]
    i, j, k = ref_idx

    # (w_s, kappa_s) 平面
    fig = _plot_mask_panels(
        s0.kappa_s_axis, s0.w_s_axis,
        [slice_w_ks(m, ref_idx) for m in singles],
        [slice_w_ks(m, ref_idx) for m in cums],
        order,
        r"$\kappa_s$", r"$w_s$", True,
        (s0.kappa_s_axis[j], s0.w_s_axis[i]),
        f"equivalence ridges in (w_s, kappa_s), kappa_e={s0.kappa_e_axis[k]:.3g}, "
        f"ref={ref_name}, ego={profile}, tol={tol}",
    )
    save_figure(fig, out_dir, f"fig_ridge_wks_{ref_name}{suffix}", cfg_hash)
    plt.close(fig)

    # (kappa_s, kappa_e) 平面 + 理论双曲线
    hyp_ke = kappa_aggregate_hyperbola(
        s0.kappa_s_axis, s0.kappa_s_axis[j], s0.kappa_e_axis[k]
    )
    fig = _plot_mask_panels(
        s0.kappa_e_axis, s0.kappa_s_axis,
        [slice_ks_ke(m, ref_idx) for m in singles],
        [slice_ks_ke(m, ref_idx) for m in cums],
        order,
        r"$\kappa_e$", r"$\kappa_s$", True,
        (s0.kappa_e_axis[k], s0.kappa_s_axis[j]),
        f"equivalence ridges in (kappa_s, kappa_e), w_s={s0.w_s_axis[i]:.3g}, "
        f"ref={ref_name}, ego={profile}, tol={tol}",
        hyperbola=(hyp_ke, s0.kappa_s_axis),
    )
    save_figure(fig, out_dir, f"fig_ridge_kk_{ref_name}{suffix}", cfg_hash)
    plt.close(fig)

    # 指标
    norm_axes = normalized_coords(s0)
    metrics = {}
    for n_sit, m in enumerate(cums, start=1):
        entry = set_metrics(m, norm_axes)
        entry["n_components_wks"] = count_components_2d(slice_w_ks(m, ref_idx))
        entry["n_components_kk"] = count_components_2d(slice_ks_ke(m, ref_idx))
        metrics[f"situations_{n_sit}"] = entry
    metrics["ref_grid_point"] = {
        "w_s": float(s0.w_s_axis[i]),
        "kappa_s": float(s0.kappa_s_axis[j]),
        "kappa_e": float(s0.kappa_e_axis[k]),
        "q_star_per_situation": [float(q[ref_idx]) for q in q_list],
    }
    return metrics


def _shrinkage_figure(all_metrics: dict, out_dir, cfg_hash):
    refs = list(all_metrics.keys())
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    x = [1, 2, 3]
    for ref in refs:
        vol = [all_metrics[ref][f"situations_{n}"]["volume_fraction"] for n in x]
        dia = [all_metrics[ref][f"situations_{n}"]["diameter"] for n in x]
        axes[0].plot(x, vol, "o-", label=ref)
        axes[1].plot(x, dia, "o-", label=ref)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("volume fraction of equivalence set")
    axes[1].set_ylabel("Chebyshev diameter (normalized coords)")
    for ax in axes:
        ax.set_xlabel("number of situations")
        ax.set_xticks(x)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("equivalence-set shrinkage with situation variation (ego=baseline)", fontsize=10)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_shrinkage", cfg_hash)
    plt.close(fig)


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    cfg, cfg_hash = load_config()
    scan_dir = OUTPUTS_ROOT / f"step1_scan_{cfg_hash}"
    if not scan_dir.exists():
        raise FileNotFoundError(
            f"未找到 {scan_dir}。请先运行 run_grid_scan.py(config 改动会更换 hash)"
        )
    out_dir = prepare_output_dir("step1_equiv", cfg_hash)
    eq_cfg = cfg["equivalence"]
    main_profile = cfg["scan"]["main_ego_profile"]

    print(f"config hash = {cfg_hash}, outputs -> {out_dir}")
    all_metrics = {}

    # 主实验:全网格 baseline
    scans = _load_profile_scans(scan_dir, cfg, cfg_hash, main_profile)
    for ref_name, ref in eq_cfg["reference_points"].items():
        ref_idx = snap_reference(scans[0], ref)
        m = _ridge_figures(scans, ref_name, ref_idx, cfg, out_dir, cfg_hash, main_profile)
        all_metrics[ref_name] = m
        v = [m[f"situations_{n}"]["volume_fraction"] for n in (1, 2, 3)]
        print(f"[{ref_name}] volume 1/2/3 situations: {v[0]:.4f} / {v[1]:.4f} / {v[2]:.4f}")
    _shrinkage_figure(all_metrics, out_dir, cfg_hash)

    # 鲁棒性:粗网格 ego 画像组,只看脊线拓扑
    robust_metrics = {}
    for profile in cfg["scan"]["robustness_ego_profiles"]:
        scans_r = _load_profile_scans(scan_dir, cfg, cfg_hash, profile)
        robust_metrics[profile] = {}
        for ref_name, ref in eq_cfg["reference_points"].items():
            ref_idx = snap_reference(scans_r[0], ref)
            m = _ridge_figures(
                scans_r, ref_name, ref_idx, cfg, out_dir, cfg_hash, profile,
                suffix=f"_robust_{profile}",
            )
            robust_metrics[profile][ref_name] = m
            print(
                f"[robust {profile} | {ref_name}] components (wks plane) 1/2/3: "
                + " / ".join(
                    str(m[f"situations_{n}"]["n_components_wks"]) for n in (1, 2, 3)
                )
            )

    with open(out_dir / f"equivalence_metrics_{cfg_hash}.json", "w", encoding="utf-8") as f:
        json.dump({"baseline": all_metrics, "robustness": robust_metrics}, f, indent=2)
    print("完成。")


if __name__ == "__main__":
    main()