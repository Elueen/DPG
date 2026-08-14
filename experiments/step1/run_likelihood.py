"""C6 实验入口:观测长度模拟(暴力网格似然的集中过程)。

运行(仓库根目录,需先跑 run_grid_scan.py):
    python experiments/step1/run_likelihood.py

观测模型:opponent 在均衡策略下的 N 次独立选择,n_yield ~ Binomial(N, q*_c)。
单情境版:全部 N 次来自 config observation.single_situation;
多情境版:同一总预算 N 按 situation_order 尽量均分(余数给靠前情境)。
似然:暴力网格,logL(theta) = sum_c [k_c log q*_c(theta) + (N_c-k_c) log(1-q*_c(theta))],
q*_c(theta) 直接取 C4 扫描的 q_eq(真参数吸附到网格点,似然良定义)。
置信集合:{theta: logL >= max logL - chi2.ppf(level, df=3)/2},
大小指标复用 C5 的 volume_fraction 与归一化 Chebyshev 直径,
n_replicates 次重复取中位数与四分位带。

产出(outputs/step1_lik_<hash>/,纪律 3):
  - fig_nconc_{hash}.pdf/.png            关键曲线:N -> 置信集合大小,
      单情境(虚线) vs 多情境(实线),三个真参数各一色
  - fig_surface_{true}_{hash}.pdf/.png   似然曲面集中过程快照:
      (w_s, kappa_s) 切片上的 Delta logL 与置信集合边界,单/多 x N
  - likelihood_metrics_{hash}.json       全部数值结果
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
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.step1.common import OUTPUTS_ROOT, load_config, prepare_output_dir, save_figure
from src.analysis.equivalence import (
    check_same_grid,
    load_scan,
    normalized_coords,
    set_metrics,
    snap_reference,
)

N_PARAMS = 3  # (w_s, kappa_s, kappa_e):似然比阈值的自由度


# ----------------------------------------------------------------------
# 核心函数(tests/test_likelihood.py 直接导入测试)
# ----------------------------------------------------------------------
def split_budget(n_total: int, n_situations: int) -> list[int]:
    """总预算尽量均分,余数给靠前情境。sum == n_total。"""
    base, rem = divmod(int(n_total), int(n_situations))
    return [base + 1 if i < rem else base for i in range(n_situations)]


def simulate_counts(
    q_true: list[float], n_per_situation: list[int], rng: np.random.Generator
) -> list[int]:
    """各情境独立 Binomial 抽样 yield 次数。"""
    return [int(rng.binomial(n, q)) for q, n in zip(q_true, n_per_situation)]


def grid_log_likelihood(
    q_grids: list[np.ndarray],
    counts: list[int],
    n_per_situation: list[int],
    prob_clip: float,
) -> np.ndarray:
    """暴力网格 log 似然(N=0 的情境自动零贡献)。"""
    logl = np.zeros_like(q_grids[0])
    for q, k, n in zip(q_grids, counts, n_per_situation):
        if n == 0:
            continue
        qc = np.clip(q, prob_clip, 1.0 - prob_clip)
        logl = logl + k * np.log(qc) + (n - k) * np.log(1.0 - qc)
    return logl


def confidence_mask(logl: np.ndarray, level: float) -> np.ndarray:
    """似然比置信集合:logL >= max logL - chi2.ppf(level, df=3)/2。"""
    threshold = float(np.nanmax(logl)) - 0.5 * chi2.ppf(level, df=N_PARAMS)
    with np.errstate(invalid="ignore"):
        return logl >= threshold


# ----------------------------------------------------------------------
# 实验流程
# ----------------------------------------------------------------------
def _load_scans(cfg, cfg_hash):
    scan_dir = OUTPUTS_ROOT / f"step1_scan_{cfg_hash}"
    if not scan_dir.exists():
        raise FileNotFoundError(
            f"未找到 {scan_dir}。请先运行 run_grid_scan.py(config 改动会更换 hash)"
        )
    order = cfg["equivalence"]["situation_order"]
    profile = cfg["scan"]["main_ego_profile"]
    scans = []
    for situation in order:
        scans.append(load_scan(scan_dir / f"scan_{situation}_{profile}_{cfg_hash}.npz"))
    check_same_grid(scans)
    return scans, order


def _run_condition(q_grids, q_true, n_per_situation, obs_cfg, rng, norm_axes):
    counts = simulate_counts(q_true, n_per_situation, rng)
    logl = grid_log_likelihood(q_grids, counts, n_per_situation, obs_cfg["prob_clip"])
    mask = confidence_mask(logl, obs_cfg["confidence_level"])
    m = set_metrics(mask, norm_axes)
    return m, logl, mask


def main():
    cfg, cfg_hash = load_config()
    scans, order = _load_scans(cfg, cfg_hash)
    out_dir = prepare_output_dir("step1_lik", cfg_hash)
    obs_cfg = cfg["observation"]
    n_values = obs_cfg["n_values"]
    n_rep = int(obs_cfg["n_replicates"])
    single_idx = order.index(obs_cfg["single_situation"])
    q_grids = [s.q_eq for s in scans]
    norm_axes = normalized_coords(scans[0])
    rng = np.random.default_rng(cfg["meta"]["seed"])

    print(f"config hash = {cfg_hash}, outputs -> {out_dir}")
    results = {}
    surfaces = {}  # 供曲面快照:每个 true point 存 replicate 0 的 (mode, N) -> (logl, mask)

    for true_name in obs_cfg["true_points"]:
        ref = cfg["equivalence"]["reference_points"][true_name]
        ref_idx = snap_reference(scans[0], ref)
        q_true = [float(q[ref_idx]) for q in q_grids]
        results[true_name] = {"q_true_per_situation": q_true, "modes": {}}
        surfaces[true_name] = {}

        for mode in ("single", "multi"):
            per_n = {}
            for n_total in n_values:
                if mode == "single":
                    n_per = [0] * len(order)
                    n_per[single_idx] = int(n_total)
                else:
                    n_per = split_budget(n_total, len(order))
                vols, dias, truth_in = [], [], []
                for rep in range(n_rep):
                    m, logl, mask = _run_condition(
                        q_grids, q_true, n_per, obs_cfg, rng, norm_axes
                    )
                    vols.append(m["volume_fraction"])
                    dias.append(m["diameter"])
                    truth_in.append(bool(mask[ref_idx]))
                    if rep == 0:
                        surfaces[true_name][(mode, n_total)] = (logl, mask, ref_idx)
                per_n[int(n_total)] = {
                    "volume_median": float(np.median(vols)),
                    "volume_q25": float(np.quantile(vols, 0.25)),
                    "volume_q75": float(np.quantile(vols, 0.75)),
                    "diameter_median": float(np.median(dias)),
                    "diameter_q25": float(np.quantile(dias, 0.25)),
                    "diameter_q75": float(np.quantile(dias, 0.75)),
                    "truth_coverage": float(np.mean(truth_in)),
                }
                results[true_name]["modes"].setdefault(mode, {})[int(n_total)] = per_n[int(n_total)]
            v = [per_n[int(n)]["volume_median"] for n in n_values]
            print(f"[{true_name} | {mode}] volume median vs N: "
                  + " ".join(f"{x:.4f}" for x in v))

    # 关键曲线图
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    colors = {n: c for n, c in zip(obs_cfg["true_points"], ["#d62728", "#2ca02c", "#1f77b4"])}
    for true_name in obs_cfg["true_points"]:
        for mode, ls in (("multi", "-"), ("single", "--")):
            d = results[true_name]["modes"][mode]
            vol = [d[int(n)]["volume_median"] for n in n_values]
            dia = [d[int(n)]["diameter_median"] for n in n_values]
            v25 = [d[int(n)]["volume_q25"] for n in n_values]
            v75 = [d[int(n)]["volume_q75"] for n in n_values]
            axes[0].plot(n_values, vol, ls, color=colors[true_name],
                         label=f"{true_name} ({mode})")
            axes[0].fill_between(n_values, v25, v75, color=colors[true_name], alpha=0.12)
            axes[1].plot(n_values, dia, ls, color=colors[true_name])
    for ax, ylab in zip(axes, ["confidence-set volume fraction (median)",
                               "Chebyshev diameter (median)"]):
        ax.set_xscale("log")
        ax.set_xlabel("N (total observation budget)")
        ax.set_ylabel(ylab, fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=6)
    fig.suptitle(
        "likelihood concentration vs observation length "
        "(solid = multi-situation, dashed = single-situation)", fontsize=9)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_nconc", cfg_hash)
    plt.close(fig)

    # 曲面集中过程快照(replicate 0,(w_s, kappa_s) 切片于真 kappa_e)
    s0 = scans[0]
    for true_name in obs_cfg["true_points"]:
        fig, axes = plt.subplots(2, len(n_values), figsize=(3.0 * len(n_values), 5.6),
                                 sharex=True, sharey=True)
        for r, mode in enumerate(("single", "multi")):
            for cidx, n_total in enumerate(n_values):
                logl, mask, ref_idx = surfaces[true_name][(mode, n_total)]
                k = ref_idx[2]
                dl = logl[:, :, k] - np.nanmax(logl)
                ax = axes[r, cidx]
                pm = ax.pcolormesh(
                    s0.kappa_s_axis, s0.w_s_axis, np.maximum(dl, -20.0),
                    shading="nearest", cmap="magma", vmin=-20.0, vmax=0.0,
                )
                ax.contour(s0.kappa_s_axis, s0.w_s_axis,
                           mask[:, :, k].astype(float), levels=[0.5],
                           colors="cyan", linewidths=1.0)
                ax.plot(s0.kappa_s_axis[ref_idx[1]], s0.w_s_axis[ref_idx[0]],
                        "w*", ms=9, mec="k", mew=0.5)
                ax.set_xscale("log")
                if r == 0:
                    ax.set_title(f"N={n_total}", fontsize=9)
                if cidx == 0:
                    ax.set_ylabel(f"{mode}\n$w_s$", fontsize=9)
                if r == 1:
                    ax.set_xlabel(r"$\kappa_s$", fontsize=9)
        fig.colorbar(pm, ax=axes, shrink=0.85, label=r"$\Delta \log L$ (capped)")
        fig.suptitle(
            f"likelihood surface concentration, true={true_name} "
            f"(slice at true kappa_e; cyan = confidence set)", fontsize=10)
        save_figure(fig, out_dir, f"fig_surface_{true_name}", cfg_hash)
        plt.close(fig)

    with open(out_dir / f"likelihood_metrics_{cfg_hash}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("完成。")


if __name__ == "__main__":
    main()