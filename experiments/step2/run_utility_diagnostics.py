"""任务 D 实验入口 2:raw utility 诊断(D5 全项;二维 safety evidence 版)。

运行(仓库根目录,需已有 counterfactual_rollouts 与冻结的 v_ref_mps):
    python experiments/step2/run_utility_diagnostics.py

对每个 valid interaction 的 Y/H 两分支计算 (g_tilde, c_tilde, z_e),
u_s^raw, u_e^raw 及 Δu。输出 data/pilot/utilities_raw.csv 与诊断报告。
F-A 复核条款:F-A remains the frozen primary specification unless this
rerun reveals severe saturation, degeneracy, or a semantic inconsistency。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.step1.common import REPO_ROOT, load_config, prepare_output_dir
from src.evidence import branch_evidence, diagnostic_ttc, safety_margin_from_evidence
from src.utility import check_gradients, u_e_raw, u_s_raw

CFG_PATH = REPO_ROOT / "configs" / "step2_pilot.yaml"
LABEL_TABLE = REPO_ROOT / "data" / "pilot" / "action_labels.csv"
ROLLOUTS = REPO_ROOT / "data" / "pilot" / "counterfactual_rollouts.csv.gz"
OUT_TABLE = REPO_ROOT / "data" / "pilot" / "utilities_raw.csv"
PRIMARY_ONSET = "lat_vel"


def main():
    cfg, cfg_hash = load_config(CFG_PATH)
    out_dir = prepare_output_dir("step2/utility_diag", cfg_hash, CFG_PATH)
    cfg_e, cfg_u, cfg_p = cfg["evidence"], cfg["utility"], cfg["perception_noise"]
    vref_map = cfg_u["v_ref_mps"]
    if any(v is None for v in vref_map.values()):
        raise ValueError("config utility.v_ref_mps 未冻结:先跑 compute_vref.py 并填入")

    lab = pd.read_csv(LABEL_TABLE)
    lab = lab[(lab["onset_def"] == PRIMARY_ONSET) & lab["valid"]
              & lab["opp_kin_physical"]]
    meta = lab.set_index("event_id")
    roll = pd.read_csv(ROLLOUTS)
    dt = 1.0 / cfg["resample_hz"]
    horizon = float(cfg["counterfactual"]["horizon_s"])

    rows = []
    for ev, g in roll.groupby("event_id"):
        if ev not in meta.index:
            continue
        r = meta.loc[ev]
        seg = str(r["segment"])
        site = ("loc2" if "rec03" in seg else "loc1" if "rec13" in seg
                else "i80" if "i80" in seg else "us101")
        v_ref = float(vref_map[site])
        g = g.sort_values("t")
        z = {}
        for br, gcol, vcol, ccol in (("Y", "gap_Y", "v_opp_Y", "dv_Y"),
                                     ("H", "gap_H", "v_opp_H", "dv_H")):
            e = branch_evidence(g[gcol].to_numpy(), g[vcol].to_numpy(),
                                g[ccol].to_numpy(), cfg_e, dt)
            z[f"g_tilde_{br}"] = e["g_tilde"]
            z[f"c_tilde_{br}"] = e["c_tilde"]
            z[f"m_s_{br}"] = safety_margin_from_evidence(
                e["g_tilde"], e["c_tilde"], cfg_e)
            z[f"z_e_{br}"] = e["z_e"]
            z[f"u_s_{br}"] = u_s_raw(e["g_tilde"], e["c_tilde"], cfg_e, cfg_u)
            z[f"u_e_{br}"] = u_e_raw(e["z_e"], v_ref, horizon, cfg_u)
        rows.append({
            "event_id": ev, "segment": seg, "dataset": r["dataset"],
            "site": site, "v_ref": v_ref, "d2_gap": r["d2_gap"], **z,
            "du_s_raw": z["u_s_Y"] - z["u_s_H"],
            "du_e_raw": z["u_e_Y"] - z["u_e_H"],
            "ttc_diag": diagnostic_ttc(float(r["gap0"]),
                                       float(r["v_opp0"] - r["v_ego0"]), cfg_e),
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT_TABLE, index=False)
    report = {"config_hash": cfg_hash, "n_events": int(len(tab)),
              "v_ref_mps": vref_map}
    print(f"config hash = {cfg_hash};events = {len(tab)}\n")

    print("== evidence 分布(H 分支;P5/P50/P95)==")
    for seg, g in tab.groupby("segment"):
        qg = g["g_tilde_H"].quantile([.05, .5, .95]).round(2).tolist()
        qc = g["c_tilde_H"].quantile([.05, .5, .95]).round(2).tolist()
        print(f"  {seg:26s} g_tilde {qg}  c_tilde {qc}")

    print("\n== u^raw 分布(逐 segment,H 分支;P5/P50/P95)==")
    for seg, g in tab.groupby("segment"):
        qs = g["u_s_H"].quantile([.05, .5, .95]).round(3).tolist()
        qe = g["u_e_H"].quantile([.05, .5, .95]).round(3).tolist()
        print(f"  {seg:26s} u_s {qs}  u_e {qe}")

    sat = {}
    for name, ucol, arg in (
            ("u_s", "u_s_H", tab["m_s_H"] / cfg_u["s_margin_m"]),
            ("u_e", "u_e_H",
             (tab["z_e_H"] - tab["v_ref"] * horizon) / cfg_u["s_progress_m"])):
        sat[name] = {
            "saturation_frac_|arg|>2": float((arg.abs() > 2).mean()),
            "sd": float(tab[ucol].std()),
            "near_constant": bool(tab[ucol].std() < 0.05),
        }
    report["saturation_near_constant"] = sat
    print("\n== 饱和 / 近常数 ==")
    for k, v in sat.items():
        print(f"  {k}: 饱和比例={v['saturation_frac_|arg|>2']:.3f}  "
              f"sd={v['sd']:.3f}  near_constant={v['near_constant']}")

    print("\n== Δu^raw 分布(逐 segment;P5/P50/P95)==")
    for seg, g in tab.groupby("segment"):
        ds = g["du_s_raw"].quantile([.05, .5, .95]).round(4).tolist()
        de = g["du_e_raw"].quantile([.05, .5, .95]).round(4).tolist()
        print(f"  {seg:26s} Δu_s {ds}  Δu_e {de}")
    sign = {
        "du_s>=0_frac": float((tab["du_s_raw"] >= -1e-12).mean()),
        "du_e<=0_frac": float((tab["du_e_raw"] <= 1e-12).mean()),
        "du_s_degenerate_zero_frac": float((tab["du_s_raw"].abs() < 1e-6).mean()),
    }
    report["delta_sign_structure"] = sign
    print(f"\nΔu_s>=0 比例 {sign['du_s>=0_frac']:.3f};"
          f"Δu_e<=0 比例 {sign['du_e<=0_frac']:.3f};"
          f"Δu_s≈0(退化)比例 {sign['du_s_degenerate_zero_frac']:.3f}")

    red = {
        "corr_us_ue_H": float(tab["u_s_H"].corr(tab["u_e_H"])),
        "corr_us_ue_Y": float(tab["u_s_Y"].corr(tab["u_e_Y"])),
        "corr_dus_due": float(tab["du_s_raw"].corr(tab["du_e_raw"])),
        "corr_gtilde_ctilde_H": float(tab["g_tilde_H"].corr(tab["c_tilde_H"])),
    }
    report["redundancy"] = red
    print("\n== 冗余 / 近共线 ==")
    for k, v in red.items():
        print(f"  {k} = {v:.3f}")
    if max(abs(red["corr_us_ue_H"]), abs(red["corr_us_ue_Y"]),
           abs(red["corr_dus_due"])) > 0.95:
        print("  !! 近共线警告:记 specification problem,等人工裁决")

    print(f"\n诊断 TTC(t0,封顶60s):P5/P50/P95 = "
          f"{tab['ttc_diag'].quantile([.05, .5, .95]).round(1).tolist()}")
    err = check_gradients(cfg_e, cfg_u, cfg_p,
                          v_ref=float(np.mean(list(vref_map.values()))),
                          horizon_s=horizon,
                          g_grid=np.linspace(-10, 40, 11),
                          c_grid=np.linspace(-4, 6, 11),
                          ze_grid=np.linspace(0, 160, 17))
    report["grad_check_max_err"] = err
    ok = err <= cfg_u["grad_check_tol"]
    print(f"梯度互检(对 zeta,含 a_g/a_c/a_e 链式因子):max_err = {err:.2e}"
          f"(tol {cfg_u['grad_check_tol']:.0e})-> {'PASS' if ok else 'FAIL'}")

    with open(out_dir / f"utility_diag_report_{cfg_hash}.json", "w",
              encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nutilities -> {OUT_TABLE}\n完成。")


if __name__ == "__main__":
    main()