"""R5(新 C6)实验入口:finite-sample recovery(M0 与 M1 同跑)。

运行(仓库根目录):
    python experiments/step1_rev/run_recovery.py

设计:
- 数据生成与网格似然均在被测模型内(well-specified);
  k_c ~ Binomial(N_c, q_c(theta*)),theta* 引用 equivalence_rev 参考点。
- 观测预算:single = 全部 N 给 config 指定的单个 context;
  multi = 同一总预算按 allocation(even)均分;总样本严格守恒。
- 置信区域:95% likelihood-ratio confidence region
  {theta: logL >= max logL - chi2.ppf(level, 3)/2}(非 Bayesian posterior)。
- 复用第一轮机制:split_budget / simulate_counts / grid_log_likelihood /
  confidence_mask(experiments.step1.run_likelihood,零改动 import)。

输出(逐 模型 x 家族 x 模式 x N,replicates 次重复):
- 参数估计误差:MLE 格点 vs theta*,按参数报 |Δw_s|、|Δlog ks|、|Δlog ke| 中位数
- confidence-region volume / Chebyshev diameter(中位数与四分位)
- empirical coverage(theta* 格点落入区域的频率)
- crossover:multi 体积中位数首次低于 single 的 N(逐 configuration 记录,
  不预设普适阈值)

产出(outputs/step1rev_recov_<hash>/):
- recovery_metrics_<hash>.json
- fig_recov_volume_<hash>.pdf/.png      体积 vs N(single 虚线 / multi 实线)
- fig_recov_error_<hash>.pdf/.png       逐参数 MLE 误差 vs N(恢复 vs 正确失败)
- fig_recov_coverage_<hash>.pdf/.png    empirical coverage vs N
- config_used_<hash>.yaml
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.step1.common import REPO_ROOT, load_config, prepare_output_dir, save_figure
from experiments.step1.run_likelihood import (
    confidence_mask,
    grid_log_likelihood,
    simulate_counts,
    split_budget,
)
from src.analysis.direct_scan import build_grid, q_grids, snap
from src.analysis.equivalence import normalized_coords, set_metrics
from src.games.contexts import PerceptionModel

REVISION_CONFIG = REPO_ROOT / "configs" / "step1_revision.yaml"
PARAM_LABELS = ("abs_dw_s", "abs_dlog_ks", "abs_dlog_ke")


# ----------------------------------------------------------------------
# 单条件运行(tests 直接导入)
# ----------------------------------------------------------------------
def run_condition(
    q_list: list[np.ndarray],
    logq: list[np.ndarray],
    log1q: list[np.ndarray],
    grid,
    ridx: tuple,
    n_per_context: list[int],
    n_replicates: int,
    level: float,
    rng: np.random.Generator,
    norm_axes,
):
    """返回逐 replicate 的 (volume, diameter, covered, per-param abs error)。"""
    q_true = [float(q[ridx]) for q in q_list]
    tilde_true = np.array([
        grid.w_s_axis[ridx[0]],
        np.log(grid.kappa_s_axis[ridx[1]]),
        np.log(grid.kappa_e_axis[ridx[2]]),
    ])
    vols, dias, covered, errors = [], [], [], []
    for _ in range(n_replicates):
        counts = simulate_counts(q_true, n_per_context, rng)
        logl = np.zeros_like(q_list[0])
        for lq, l1q, k, n in zip(logq, log1q, counts, n_per_context):
            if n == 0:
                continue
            logl = logl + k * lq + (n - k) * l1q
        mask = confidence_mask(logl, level)
        m = set_metrics(mask, norm_axes)
        vols.append(m["volume_fraction"])
        dias.append(m["diameter"])
        covered.append(bool(mask[ridx]))
        i, j, k_idx = np.unravel_index(int(np.argmax(logl)), logl.shape)
        tilde_mle = np.array([
            grid.w_s_axis[i], np.log(grid.kappa_s_axis[j]), np.log(grid.kappa_e_axis[k_idx]),
        ])
        errors.append(np.abs(tilde_mle - tilde_true))
    return np.array(vols), np.array(dias), np.array(covered), np.array(errors)


def summarize(vols, dias, covered, errors) -> dict:
    med_err = np.median(errors, axis=0)
    return {
        "volume_median": float(np.median(vols)),
        "volume_q25": float(np.quantile(vols, 0.25)),
        "volume_q75": float(np.quantile(vols, 0.75)),
        "diameter_median": float(np.median(dias)),
        "coverage": float(np.mean(covered)),
        **{lab: float(v) for lab, v in zip(PARAM_LABELS, med_err)},
    }


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    cfg, cfg_hash = load_config(REVISION_CONFIG)
    out_dir = prepare_output_dir("step1rev_recov", cfg_hash, REVISION_CONFIG)
    pm = PerceptionModel(cfg["perception"])
    families = {f: pm.build_context_set(c) for f, c in cfg["context_families"].items()}
    rc = cfg["recovery"]
    eq = cfg["equivalence_rev"]
    grid = build_grid(eq["grid"], n_override=rc.get("grid_n"))
    norm_axes = normalized_coords(grid)
    ridx = snap(grid, eq["reference_points"][rc["true_theta_key"]])
    prob_clip = rc["prob_clip"]
    n_values = rc["n_values"]
    level = rc["confidence_level"]

    print(f"config hash = {cfg_hash}, outputs -> {out_dir}")
    print(f"theta* = grid({grid.w_s_axis[ridx[0]]:.3f}, "
          f"{grid.kappa_s_axis[ridx[1]]:.3f}, {grid.kappa_e_axis[ridx[2]]:.3f}), "
          f"replicates = {rc['n_replicates']}")

    report = {"true_theta_key": rc["true_theta_key"], "conditions": {}}
    rng = np.random.default_rng(cfg["meta"]["seed"])

    for model in rc["models"]:
        for fam, ctxs in families.items():
            t0 = time.time()
            q_list = q_grids(model, ctxs, grid)
            qc = [np.clip(q, prob_clip, 1 - prob_clip) for q in q_list]
            logq = [np.log(q) for q in qc]
            log1q = [np.log(1 - q) for q in qc]
            cond_key = f"{model}|{fam}"
            report["conditions"][cond_key] = {}
            for mode in ("single", "multi"):
                per_n = {}
                for n_total in n_values:
                    if mode == "single":
                        n_per = [0] * len(ctxs)
                        n_per[int(rc["single_context_index"])] = int(n_total)
                    else:
                        n_per = split_budget(int(n_total), len(ctxs))
                    vols, dias, cov, errs = run_condition(
                        q_list, logq, log1q, grid, ridx, n_per,
                        int(rc["n_replicates"]), level, rng, norm_axes,
                    )
                    per_n[int(n_total)] = summarize(vols, dias, cov, errs)
                report["conditions"][cond_key][mode] = per_n
            # crossover:multi 体积中位数首次 < single 的 N
            cross = None
            for n in n_values:
                if (report["conditions"][cond_key]["multi"][int(n)]["volume_median"]
                        < report["conditions"][cond_key]["single"][int(n)]["volume_median"]):
                    cross = int(n)
                    break
            report["conditions"][cond_key]["crossover_n"] = cross
            s = report["conditions"][cond_key]
            print(f"{cond_key:32s} {time.time()-t0:5.1f}s  "
                  f"cov(N=1000,multi)={s['multi'][1000]['coverage']:.3f}  "
                  f"vol(N=1000) single/multi={s['single'][1000]['volume_median']:.4f}/"
                  f"{s['multi'][1000]['volume_median']:.4f}  crossover={cross}")

    # --- 图 1:体积 vs N ---
    fam_names = list(families.keys())
    fig, axes = plt.subplots(1, len(rc["models"]), figsize=(5.2 * len(rc["models"]), 3.8),
                             sharey=True)
    colors = {"strong_diversity": "#d62728", "moderate_diversity": "#2ca02c",
              "insufficient_diversity": "#1f77b4"}
    for ax, model in zip(np.atleast_1d(axes), rc["models"]):
        for fam in fam_names:
            d = report["conditions"][f"{model}|{fam}"]
            for mode, ls in (("multi", "-"), ("single", "--")):
                v = [d[mode][int(n)]["volume_median"] for n in n_values]
                ax.plot(n_values, np.maximum(v, 1e-6), ls, color=colors[fam],
                        label=f"{fam.split('_')[0]} ({mode})" if model == rc["models"][0] else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(model.upper(), fontsize=10)
        ax.set_xlabel("N (total budget)")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("LR confidence-region volume (median)")
    np.atleast_1d(axes)[0].legend(fontsize=6)
    fig.suptitle("finite-sample concentration of 95% LR confidence regions", fontsize=10)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_recov_volume", cfg_hash)
    plt.close(fig)

    # --- 图 2:逐参数 MLE 误差 vs N(M1:恢复 vs 正确失败)---
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
    show = [("m1", "strong_diversity"), ("m1", "insufficient_diversity"),
            ("m0", "strong_diversity")]
    for ax, (model, fam) in zip(axes, show):
        d = report["conditions"][f"{model}|{fam}"]["multi"]
        for p_i, lab in enumerate(PARAM_LABELS):
            ax.plot(n_values, [d[int(n)][lab] for n in n_values], "o-", label=lab)
        ax.set_xscale("log")
        ax.set_title(f"{model.upper()} | {fam.replace('_diversity','')}", fontsize=9)
        ax.set_xlabel("N")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("median |MLE error| (tilde coords)")
    axes[0].legend(fontsize=7)
    fig.suptitle("parameter recovery vs correct failure (multi-context budget)", fontsize=10)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_recov_error", cfg_hash)
    plt.close(fig)

    # --- 图 3:coverage vs N ---
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for model in rc["models"]:
        for fam in fam_names:
            d = report["conditions"][f"{model}|{fam}"]["multi"]
            ax.plot(n_values, [d[int(n)]["coverage"] for n in n_values], "o-",
                    label=f"{model} {fam.split('_')[0]}", alpha=0.8)
    ax.axhline(level, color="gray", ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("95% LR confidence-region coverage")
    ax.set_ylim(0.8, 1.02)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_recov_coverage", cfg_hash)
    plt.close(fig)

    with open(out_dir / f"recovery_metrics_{cfg_hash}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("完成。")


if __name__ == "__main__":
    main()