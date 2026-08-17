"""R2(新 C3)实验入口:structural identifiability 报告。

运行(仓库根目录):
    python experiments/step1_rev/run_identifiability.py

对 模型(m0a/m0/m1) x context 家族 x probe theta 的全组合:
数值 Jacobian(theta_tilde = (w_s, log ks, log ke))-> SVD ->
singular values / numerical rank / condition number / null direction。
另做 M1 的 context 子集分析(什么组合达到 full local rank)。

产出(outputs/step1rev_ident_<hash>/):
  - identifiability_report_<hash>.json   全部数值(sv、rank、cond、null 方向)
  - fig_sv_spectrum_<hash>.pdf/.png      奇异值谱:模型 x 家族(mid probe),
    log10 尺度,直观展示 rank-deficiency 与 weak identification 梯度
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
from src.analysis.identifiability import analyze, subset_ranks
from src.games.contexts import PerceptionModel

REVISION_CONFIG = REPO_ROOT / "configs" / "step1_revision.yaml"


def main():
    cfg, cfg_hash = load_config(REVISION_CONFIG)
    out_dir = prepare_output_dir("step1rev_ident", cfg_hash, REVISION_CONFIG)
    pm = PerceptionModel(cfg["perception"])
    families = {f: pm.build_context_set(c) for f, c in cfg["context_families"].items()}
    idc = cfg["identifiability"]

    print(f"config hash = {cfg_hash}, outputs -> {out_dir}")
    report = {}
    for model in cfg["models"]:
        report[model] = {}
        for fam, ctxs in families.items():
            report[model][fam] = {}
            for pname, p in idc["probe_thetas"].items():
                theta = np.array([p["w_s"], p["kappa_s"], p["kappa_e"]])
                rep = analyze(model, theta, ctxs, fam, idc)
                report[model][fam][pname] = {
                    "theta": rep.theta.tolist(),
                    "singular_values": rep.singular_values.tolist(),
                    "numerical_rank": rep.numerical_rank,
                    "cond_rank": rep.cond_rank,
                    "cond_full": rep.cond_full,
                    "null_direction_tilde": rep.null_direction.tolist(),
                }
                print(f"{pname:10s} " + rep.summary_row())
        print()

    # M1 子集分析:哪些 context 组合达到 full rank(mid probe,各家族)
    p = idc["probe_thetas"]["mid"]
    theta = np.array([p["w_s"], p["kappa_s"], p["kappa_e"]])
    report["m1_subset_analysis"] = {
        fam: subset_ranks("m1", theta, ctxs, idc) for fam, ctxs in families.items()
    }
    for fam, subs in report["m1_subset_analysis"].items():
        full = [k for k, v in subs.items() if v["rank"] == 3]
        print(f"M1 {fam}: full-rank 子集 = {full if full else '无'}")

    # 奇异值谱图(mid probe)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    fam_names = list(families.keys())
    x = np.arange(3)
    width = 0.25
    for ax, model in zip(axes, cfg["models"]):
        for i, fam in enumerate(fam_names):
            sv = np.array(report[model][fam]["mid"]["singular_values"])
            ax.bar(x + (i - 1) * width, np.log10(np.maximum(sv, 1e-18)), width,
                   label=fam if model == "m0a" else None)
        ax.axhline(np.log10(idc["rank_rtol"]) + np.log10(
            max(report[model][f]["mid"]["singular_values"][0] for f in fam_names)),
            color="gray", ls=":", lw=0.8)
        ax.set_title(model.upper(), fontsize=10)
        ax.set_xticks(x, [r"$s_1$", r"$s_2$", r"$s_3$"])
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel(r"$\log_{10}$ singular value")
    fig.legend(loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(
        "Jacobian singular-value spectra at mid probe "
        "(s3 at numerical zero = structural rank deficiency)", fontsize=10)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    save_figure(fig, out_dir, "fig_sv_spectrum", cfg_hash)
    plt.close(fig)

    with open(out_dir / f"identifiability_report_{cfg_hash}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("完成。")


if __name__ == "__main__":
    main()