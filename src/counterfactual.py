"""Step 2 任务 C — counterfactual tactical action 模板与运动学 rollout。

从同一 interaction 状态 x_t0(gap0, v_ego0, v_opp0)构造两个标准
counterfactual 分支(参数全部来自 config counterfactual 段):

Hold 模板:a_O^H(t) = 0,opponent 匀速 v_opp(t0)。
  与 D2 观测动作定义的"无响应=匀速外推"参照是同一数学对象(语义自洽)。

Yield 模板:升余弦减速脉冲
  a_O^Y(t) = -a_peak * 0.5 * (1 - cos(2*pi*t/T)),t in [0, T]
  光滑起止:瞬时进入/退出恒减速会产生 acceleration discontinuity /
  impulsive jerk,升余弦的 jerk 有界(max|jerk| = pi*a_peak/T)。
  解析性质(无地板钳位时):dv(T) = a_peak*T/2;让出距离 A(T) = a_peak*T^2/4。

Ego:两个分支中同样以 t0 状态匀速外推(constant-velocity ego 是当前
Step-2 的 local longitudinal approximation;若后续 utility 依赖 lateral
merge geometry,再升级为固定 canonical ego trajectory,本阶段不扩展)。
不使用任何真实未来数据——两分支唯一差异 = opponent 模板,Δu 的全部
变异严格来自 tactical action。

速度地板:v_opp >= v_floor(默认 0);触发则钳位并计数(clamped 帧数)。
gap 允许为负(hold 分支的碰撞风险信号,是 safety utility 的合法输入)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROLLOUT_COLUMNS = [
    "t", "a_opp_Y", "v_opp_Y", "v_opp_H", "gap_Y", "gap_H", "dv_Y", "dv_H",
]


def yield_accel_profile(t: np.ndarray, a_peak: float, horizon: float) -> np.ndarray:
    """升余弦减速脉冲(t 超出 [0, horizon] 处为 0)。"""
    a = -a_peak * 0.5 * (1.0 - np.cos(2.0 * np.pi * t / horizon))
    a[(t < 0) | (t > horizon)] = 0.0
    return a


def rollout_pair(gap0: float, v_ego0: float, v_opp0: float, cfg_c: dict,
                 dt: float, a_peak: float | None = None,
                 horizon: float | None = None) -> tuple[pd.DataFrame, dict]:
    """双分支确定性运动学 rollout。返回 (逐帧表, 摘要)。

    相对量演化(ego 匀速,故 bumper 偏置常数):
      gap_H(t) = gap0 + (v_ego0 - v_opp0) * t
      gap_Y(t) = gap_H(t) + A(t),A = opponent 相对匀速的让出距离
    """
    a_peak = float(cfg_c["a_peak_mps2"] if a_peak is None else a_peak)
    horizon = float(cfg_c["horizon_s"] if horizon is None else horizon)
    v_floor = float(cfg_c["v_floor_mps"])
    n = int(round(horizon / dt))
    t = np.arange(n + 1) * dt

    a_y = yield_accel_profile(t, a_peak, horizon)
    v_y = v_opp0 + np.concatenate(
        [[0.0], np.cumsum(0.5 * (a_y[1:] + a_y[:-1]) * dt)])
    clamped = int(np.sum(v_y < v_floor - 1e-12))
    v_y = np.maximum(v_y, v_floor)
    x_rel_y = np.concatenate(
        [[0.0], np.cumsum(0.5 * (v_y[1:] + v_y[:-1]) * dt)])   # opp 位移
    x_rel_h = v_opp0 * t
    gap_h = gap0 + (v_ego0 - v_opp0) * t
    gap_y = gap0 + v_ego0 * t - x_rel_y

    out = pd.DataFrame({
        "t": np.round(t, 6), "a_opp_Y": a_y, "v_opp_Y": v_y,
        "v_opp_H": np.full_like(t, v_opp0),
        "gap_Y": gap_y, "gap_H": gap_h,
        "dv_Y": v_y - v_ego0, "dv_H": np.full_like(t, v_opp0 - v_ego0),
    })
    jerk = np.abs(np.diff(a_y)) / dt
    summary = {
        "a_peak_realized": float(np.abs(a_y).max()),
        "max_jerk_realized": float(jerk.max()) if len(jerk) else 0.0,
        "max_jerk_analytic": float(np.pi * a_peak / horizon),
        "clamped_frames": clamped,
        "delta_gap_end": float(gap_y[-1] - gap_h[-1]),
        "delta_gap_end_analytic_unclamped": float(a_peak * horizon**2 / 4.0),
        "min_gap_Y": float(gap_y.min()),
        "min_gap_H": float(gap_h.min()),
        "gap_H_negative": bool(gap_h.min() < 0),
    }
    return out[ROLLOUT_COLUMNS], summary


def rollout_table(interactions: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """valid interactions 全量 rollout。返回 (逐帧长表, 逐事件摘要表)。"""
    cfg_c = cfg["counterfactual"]
    dt = 1.0 / cfg["resample_hz"]
    frames, summaries = [], []
    for _, r in interactions.iterrows():
        fr, s = rollout_pair(float(r["gap0"]), float(r["v_ego0"]),
                             float(r["v_opp0"]), cfg_c, dt)
        fr.insert(0, "event_id", r["event_id"])
        frames.append(fr)
        summaries.append(dict(event_id=r["event_id"], segment=r["segment"],
                              dataset=r["dataset"], **s))
    return (pd.concat(frames, ignore_index=True),
            pd.DataFrame(summaries))


def sensitivity_grid(interactions: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """预注册敏感性:a_peak x horizon 网格(config sens_a_factors /
    sens_horizons),逐组合报告 Δgap_end 与 trajectory-level 量
    (min_gap_Y / min_gap_H)的均值与标准差。仅此网格,不做参数搜索。"""
    cfg_c = cfg["counterfactual"]
    dt = 1.0 / cfg["resample_hz"]
    base_a = float(cfg_c["a_peak_mps2"])
    rows = []
    for fac in cfg_c["sens_a_factors"]:
        for hor in cfg_c["sens_horizons"]:
            dge, mgy, mgh, clamp = [], [], [], 0
            for _, r in interactions.iterrows():
                _, s = rollout_pair(float(r["gap0"]), float(r["v_ego0"]),
                                    float(r["v_opp0"]), cfg_c, dt,
                                    a_peak=base_a * fac, horizon=float(hor))
                dge.append(s["delta_gap_end"])
                mgy.append(s["min_gap_Y"])
                mgh.append(s["min_gap_H"])
                clamp += int(s["clamped_frames"] > 0)
            rows.append({
                "a_peak": base_a * fac, "horizon_s": float(hor),
                "delta_gap_end_mean": float(np.mean(dge)),
                "delta_gap_end_sd": float(np.std(dge)),
                "min_gap_Y_mean": float(np.mean(mgy)),
                "min_gap_H_mean": float(np.mean(mgh)),
                "events_clamped": int(clamp),
            })
    return pd.DataFrame(rows)