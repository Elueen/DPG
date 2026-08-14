"""C4 实验入口:opponent 画像网格扫描。

运行(仓库根目录):
    python experiments/step1/run_grid_scan.py

产出(outputs/step1_scan_<cfg_hash>/,纪律 3):
  - scan_{situation}_{ego_profile}_{hash}.npz    结构化扫描结果
  - fig_scan_qeq_{situation}_{ego_profile}_{hash}.pdf/.png  诊断图:
    规范均衡 q* = P[opp=yield] 在 (w_s, kappa_s) 平面、kappa_e 取中位轴点的切片,
    并叠加多重均衡/回退/未收敛格点标记
  - config_used_{hash}.yaml                      所用 config 完整拷贝

主实验:main_ego_profile x 全部情境,网格密度按 config(默认 50^3)。
鲁棒性:robustness_ego_profiles x 全部情境,粗网格 robustness_grid_n^3。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.step1.common import load_config, prepare_output_dir, save_figure
from src.analysis.grid_scan import ScanResult, scan_situation
from src.games.toy_2x2 import ToyGame2x2


def _diagnostic_figure(res: ScanResult, situation_name: str):
    ke_idx = len(res.kappa_e_axis) // 2  # kappa_e 中位轴点切片
    q_slice = res.q_eq[:, :, ke_idx]
    multi = res.n_unique[:, :, ke_idx] > 1
    fallback = res.fallback[:, :, ke_idx]
    dead = np.isnan(q_slice)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    mesh = ax.pcolormesh(
        res.kappa_s_axis, res.w_s_axis, q_slice, shading="nearest",
        vmin=0.0, vmax=1.0, cmap="viridis",
    )
    fig.colorbar(mesh, ax=ax, label=r"$q^*=P[\mathrm{opp=yield}]$")
    for mask, marker, label, color in [
        (multi, "s", "multiple eq.", "red"),
        (fallback, "^", "fallback start", "orange"),
        (dead, "x", "no convergence", "white"),
    ]:
        if mask.any():
            ii, jj = np.where(mask)
            ax.scatter(
                res.kappa_s_axis[jj], res.w_s_axis[ii],
                s=12, marker=marker, c=color, label=label, linewidths=0.8,
            )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa_s$ (opponent)")
    ax.set_ylabel(r"$w_s$ (opponent)")
    ax.set_title(
        f"{situation_name} (c={res.situation_c:.3g}), ego={res.ego_profile_name}, "
        rf"$\kappa_e$={res.kappa_e_axis[ke_idx]:.3g}"
    )
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    return fig


def _run_one(game, cfg, situation_name, profile_name, n_override, out_dir, cfg_hash):
    scan_cfg = cfg["scan"]
    c = cfg["situations"][situation_name]
    rng = np.random.default_rng(cfg["meta"]["seed"])
    t0 = time.time()
    res = scan_situation(
        game.at_situation(c),
        scan_cfg["ego_profiles"][profile_name],
        profile_name,
        scan_cfg["grid"],
        cfg["solver"],
        rng,
        n_override=n_override,
    )
    n = len(res.w_s_axis)
    tag = f"{situation_name}_{profile_name}"
    np.savez_compressed(out_dir / f"scan_{tag}_{cfg_hash}.npz", **res.to_npz_dict())
    fig = _diagnostic_figure(res, situation_name)
    save_figure(fig, out_dir, f"fig_scan_qeq_{tag}", cfg_hash)
    plt.close(fig)
    n_multi = int((res.n_unique > 1).sum())
    n_fb = int(res.fallback.sum())
    n_dead = int(np.isnan(res.p_eq).sum())
    print(
        f"[{tag}] n={n}^3  {time.time()-t0:6.1f}s  "
        f"multi_eq={n_multi}  fallback={n_fb}  no_conv={n_dead}"
    )
    return res


def main():
    cfg, cfg_hash = load_config()
    out_dir = prepare_output_dir("step1_scan", cfg_hash)
    game = ToyGame2x2.from_config(cfg["game"])
    scan_cfg = cfg["scan"]
    situations = scan_cfg["situations_to_scan"]

    print(f"config hash = {cfg_hash}, outputs -> {out_dir}")
    print("== 主实验(全网格)==")
    for s in situations:
        _run_one(game, cfg, s, scan_cfg["main_ego_profile"], None, out_dir, cfg_hash)

    print("== 鲁棒性检查(粗网格)==")
    for prof in scan_cfg["robustness_ego_profiles"]:
        for s in situations:
            _run_one(
                game, cfg, s, prof, scan_cfg["robustness_grid_n"], out_dir, cfg_hash
            )
    print("完成。")


if __name__ == "__main__":
    main()
