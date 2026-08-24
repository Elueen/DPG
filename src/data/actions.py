"""Step 2 任务 B(后半)— 观测动作 a_O 的三个候选定义(B3)。

响应窗 W = [t0, t0 + h_response_s],特征全部来自 opponent 轨迹:
  dv_red = v_opp(t0) - min_W v_opp           (速度削减,>=0)
  dv_inc = max_W v_opp - v_opp(t0)           (速度提升,>=0)
  a_end  = accommodation:x_cv(t_end) - x_opp(t_end),
           x_cv(t) = x_opp(t0) + v_opp(t0)*(t - t0)
           (相对"无响应=匀速外推"的让出距离;正 = 落后于匀速预测 = 让行)
  mean_ax = 响应窗内平均纵向加速度

三个候选定义(阈值全部来自 config actions 段;输出 yield/hold/ambiguous):
  D1 decel : dv_red >= d1_yield -> yield;dv_red <= d1_hold -> hold;否则 ambiguous
  D2 gap   : a_end >= d2_yield -> yield;a_end <= d2_hold -> hold;否则 ambiguous
  D3 compound:
       yield 若 dv_red >= d1_yield 或 a_end >= d2_yield
             或 (dv_red >= c*d1_yield 且 a_end >= c*d2_yield),c = d3_joint_frac
       hold  若 dv_red <= d1_hold 且 a_end <= d2_hold
       否则 ambiguous

敏感性:对 (d1_yield, d1_hold) 与 (d2_yield, d2_hold) 整体乘 0.7 / 1.3
(比例缩放保持 yield/hold 阈值的相对结构),报告标签翻转比例。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABELS = ("yield", "hold", "ambiguous")
DEFS = ("d1_decel", "d2_gap", "d3_compound")


# ----------------------------------------------------------------------
# 响应特征
# ----------------------------------------------------------------------
def response_features(opp_track: pd.DataFrame, t0: float, h_resp: float) -> dict | None:
    """opp_track: 单车 canonical 轨迹(已按 t 排序)。窗内帧不足返回 None。"""
    w = opp_track[(opp_track["t"] >= t0 - 1e-9)
                  & (opp_track["t"] <= t0 + h_resp + 1e-9)].sort_values("t")
    if len(w) < 2:
        return None
    t = w["t"].to_numpy()
    v = w["vx"].to_numpy()
    x = w["x"].to_numpy()
    v0, x0 = float(v[0]), float(x[0])
    x_cv_end = x0 + v0 * (t[-1] - t[0])
    return {
        "v_opp_t0": v0,
        "dv_red": float(v0 - v.min()),
        "dv_inc": float(v.max() - v0),
        "a_end": float(x_cv_end - x[-1]),
        "mean_ax": float(w["ax"].mean()),
        "max_abs_ax": float(np.abs(w["ax"]).max()),
        "n_frames": int(len(w)),
    }


# ----------------------------------------------------------------------
# 标签
# ----------------------------------------------------------------------
def _label_d1(f: dict, th: dict) -> str:
    if f["dv_red"] >= th["d1_yield_mps"]:
        return "yield"
    if f["dv_red"] <= th["d1_hold_mps"]:
        return "hold"
    return "ambiguous"


def _label_d2(f: dict, th: dict) -> str:
    if f["a_end"] >= th["d2_yield_m"]:
        return "yield"
    if f["a_end"] <= th["d2_hold_m"]:
        return "hold"
    return "ambiguous"


def _label_d3(f: dict, th: dict) -> str:
    c = th["d3_joint_frac"]
    if (f["dv_red"] >= th["d1_yield_mps"] or f["a_end"] >= th["d2_yield_m"]
            or (f["dv_red"] >= c * th["d1_yield_mps"]
                and f["a_end"] >= c * th["d2_yield_m"])):
        return "yield"
    if f["dv_red"] <= th["d1_hold_mps"] and f["a_end"] <= th["d2_hold_m"]:
        return "hold"
    return "ambiguous"


_LABELERS = {"d1_decel": _label_d1, "d2_gap": _label_d2, "d3_compound": _label_d3}


def labels_for(f: dict, th: dict) -> dict:
    return {d: fn(f, th) for d, fn in _LABELERS.items()}


def scaled_thresholds(th: dict, factor: float) -> dict:
    out = dict(th)
    for k in ("d1_yield_mps", "d1_hold_mps", "d2_yield_m", "d2_hold_m"):
        out[k] = th[k] * factor
    return out


# ----------------------------------------------------------------------
# 全表
# ----------------------------------------------------------------------
def build_label_table(df: pd.DataFrame, interactions: pd.DataFrame,
                      cfg: dict) -> pd.DataFrame:
    """valid interaction 行 -> 响应特征 + 三定义标签 + ±30% 敏感性标签。
    输出每行 = (event x onset),含列:
      ...interaction 元数据..., dv_red, dv_inc, a_end, mean_ax,
      d1_decel, d2_gap, d3_compound,
      <def>__lo / <def>__hi(阈值 x sens_factors 后的标签)
    """
    th = cfg["actions"]
    h_resp = float(cfg["interaction"]["h_response_s"])
    lo_f, hi_f = cfg["actions"]["sens_factors"]
    by = {(seg, v): g.sort_values("t")
          for (seg, v), g in df.groupby(["segment", "vid"])}
    rows = []
    for _, it in interactions[interactions["valid"]].iterrows():
        opp = by.get((it["segment"], it["opp_vid"]))
        if opp is None:
            continue
        f = response_features(opp, float(it["t0"]), h_resp)
        if f is None:
            continue
        row = dict(it)
        row.update(f)
        row["opp_kin_physical"] = bool(f["max_abs_ax"] <= th["max_abs_ax_physical"])
        row.update(labels_for(f, th))
        for tag, fac in (("lo", lo_f), ("hi", hi_f)):
            lab = labels_for(f, scaled_thresholds(th, fac))
            row.update({f"{d}__{tag}": lab[d] for d in DEFS})
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 诊断
# ----------------------------------------------------------------------
def label_proportions(tab: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """逐定义、逐分组的 yield/hold/ambiguous 比例。"""
    recs = []
    for d in DEFS:
        for keys, g in tab.groupby(group_cols):
            keys = keys if isinstance(keys, tuple) else (keys,)
            n = len(g)
            rec = dict(zip(group_cols, keys), definition=d, n=n)
            for lab in LABELS:
                rec[lab] = float((g[d] == lab).mean()) if n else np.nan
            recs.append(rec)
    return pd.DataFrame(recs)


def pairwise_agreement(tab: pd.DataFrame) -> dict:
    """定义两两:非 ambiguous 共同子集上的一致率 + 3x3 confusion。"""
    out = {}
    for i, a in enumerate(DEFS):
        for b in DEFS[i + 1:]:
            conf = pd.crosstab(tab[a], tab[b]).reindex(
                index=LABELS, columns=LABELS, fill_value=0)
            both = tab[(tab[a] != "ambiguous") & (tab[b] != "ambiguous")]
            agree = float((both[a] == both[b]).mean()) if len(both) else np.nan
            out[f"{a}|{b}"] = {
                "agreement_non_ambiguous": agree,
                "n_non_ambiguous": int(len(both)),
                "confusion": conf.values.tolist(),
            }
    return out


def sensitivity_flips(tab: pd.DataFrame) -> dict:
    """±30% 阈值缩放下的标签翻转比例(逐定义)。"""
    out = {}
    for d in DEFS:
        out[d] = {
            "flip_lo": float((tab[d] != tab[f"{d}__lo"]).mean()),
            "flip_hi": float((tab[d] != tab[f"{d}__hi"]).mean()),
        }
    return out