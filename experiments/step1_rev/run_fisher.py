"""R3(新 C4)实验入口:Fisher information / context informativeness。

运行(仓库根目录):
    python experiments/step1_rev/run_fisher.py

两层分析(theta = fisher.probe_theta,theta_tilde 坐标):
1. 逐 context:六类探针(indifference / moderate / saturated /
   safety- / efficiency-sensitive / 同 G 负对照)的逐参数对角信息,
   检验"哪类 context 对哪个参数 informative"(不预设 q 偏离 0.5 越好)。
2. 集合层:各 context 家族 + 理论指导组合(designed_portfolio)在
   M0a/M0/M1 下的特征值谱、min_eig、log-det、条件数与 regime 分类。

产出(outputs/step1rev_fisher_<hash>/):
  - fisher_report_<hash>.json
  - fig_fisher_percontext_<hash>.pdf/.png   逐 context x 参数 信息热图(log10)
  - fig_fisher_sets_<hash>.pdf/.png         集合层 min_eig / log-det 对比
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
from src.analysis.fisher import classify_regime, fisher_information, resolve_probe_contexts
from src.analysis.identifiability import PARAM_NAMES
from src.games.contexts import PerceptionModel

REVISION_CONFIG = REPO_ROOT / "configs" / "step1_revision.yaml"


def main():
    cfg, cfg_hash = load_config(REVISION_CONFIG)
    out_dir = prepare_output_dir("step1rev_fisher", cfg_hash, REVISION_CONFIG)
    pm = PerceptionModel(cfg["perception"])
    fc = cfg["fisher"]
    idc = cfg["identifiability"]
    theta = np.array([fc["probe_theta"]["w_s"], fc["probe_theta"]["kappa_s"],
                      fc["probe_theta"]["kappa_e"]])
    families = {f: pm.build_context_set(c) for f, c in cfg["context_families"].items()}
    probes = resolve_probe_contexts(pm, fc, cfg["context_families"])
    probe_by_name = {c.name: c for c in probes}

    print(f"config hash = {cfg_hash}, outputs -> {out_dir}")
    report = {"theta": theta.tolist(), "n_per_context": fc["n_per_context"]}

    # --- 1. 逐 context(M1)---
    rep = fisher_information(
        "m1", theta, probes, fc["n_per_context"], idc["fd_step_w"], idc["fd_step_logk"]
    )
    diag = rep.per_context_diag()
    report["per_context_m1"] = {
        c.name: {"q": float(q), "diag_info": d.tolist()}
        for c, q, d in zip(probes, rep.q, diag)
    }
    print(f"{'context':22s} {'q':>7s}  " + "  ".join(f"{p:>10s}" for p in PARAM_NAMES))
    for c, q, d in zip(probes, rep.q, diag):
        print(f"{c.name:22s} {q:7.4f}  " + "  ".join(f"{v:10.3f}" for v in d))

    # --- 2. 集合层:三模型 x (家族 + designed portfolio) ---
    designed = [probe_by_name[n] for n in fc["designed_portfolio"]]
    sets = dict(families)
    sets["designed_portfolio"] = designed
    report["sets"] = {}
    print()
    for model in cfg["models"]:
        report["sets"][model] = {}
        for label, ctxs in sets.items():
            r = fisher_information(
                model, theta, ctxs, fc["n_per_context"],
                idc["fd_step_w"], idc["fd_step_logk"],
            )
            regime = classify_regime(r, fc["regime_thresholds"])
            report["sets"][model][label] = {
                "eigenvalues": r.eigenvalues.tolist(),
                "min_eigenvalue": r.min_eigenvalue,
                "log_det": r.log_det,
                "cond": r.cond,
                "regime": regime,
            }
            print(f"{model:4s} {label:22s} min_eig={r.min_eigenvalue:.3e}  "
                  f"logdet={r.log_det:8.2f}  cond={r.cond:.2e}  -> {regime}")

    # --- 图 1:逐 context x 参数 热图 ---
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    mat = np.log10(np.maximum(diag, 1e-6))
    im = ax.imshow(mat, cmap="viridis", aspect="auto")
    ax.set_xticks(range(3), PARAM_NAMES, fontsize=8)
    ax.set_yticks(range(len(probes)), [c.name for c in probes], fontsize=8)
    for i in range(len(probes)):
        for j in range(3):
            ax.text(j, i, f"{diag[i, j]:.2g}", ha="center", va="center",
                    fontsize=7, color="white")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}$ per-context diag info")
    ax.set_title("M1 per-context Fisher information by parameter", fontsize=10)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_fisher_percontext", cfg_hash)
    plt.close(fig)

    # --- 图 2:集合层 min_eig / logdet ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    labels = list(sets.keys())
    x = np.arange(len(labels))
    width = 0.25
    for i, model in enumerate(cfg["models"]):
        me = [max(report["sets"][model][l]["min_eigenvalue"], 1e-18) for l in labels]
        ld = [report["sets"][model][l]["log_det"] for l in labels]
        axes[0].bar(x + (i - 1) * width, np.log10(me), width, label=model.upper())
        axes[1].bar(x + (i - 1) * width, [v if np.isfinite(v) else -40 for v in ld], width)
    for thr_name, thr in cfg["fisher"]["regime_thresholds"].items():
        axes[0].axhline(np.log10(thr), color="gray", ls=":", lw=0.8)
    axes[0].set_ylabel(r"$\log_{10}$ min eigenvalue")
    axes[1].set_ylabel("log-det (singular -> -40)")
    for ax in axes:
        ax.set_xticks(x, labels, rotation=20, ha="right", fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle("set-level Fisher information across models and context sets", fontsize=10)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_fisher_sets", cfg_hash)
    plt.close(fig)

    with open(out_dir / f"fisher_report_{cfg_hash}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("完成。")


if __name__ == "__main__":
    main()