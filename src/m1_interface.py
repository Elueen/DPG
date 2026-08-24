"""Step 2 任务 G — 统一接口:canonical real interaction -> [Δu_s, Δu_e, G_s, G_e]。

严格串联已冻结组件,零新科学参数:
  C rollout(hold=匀速 / yield=升余弦) -> D evidence(g_tilde, c_tilde, z_e)
  -> E evidence scaling(zeta;a_g, a_c, a_e)-> D/F 最终 utility(F-A: T=id)
  -> L_k(a) = grad_zeta u_k(a)^T Sigma_k grad_zeta u_k(a)
  -> G_k = L_k(Y) + L_k(H);Δu_k = u_k(Y) - u_k(H)。

接口对 theta_O = (w_s, kappa_s, kappa_e) 无知;不依赖 NN;不用 test data;
不依数据集改变语义。Sigma_k 以矩阵形式进入(不写死 I 化简);
sigma_scale 参数供 E3 预注册敏感性(Sigma -> alpha*Sigma,
理论上 G_k^(alpha) = alpha * G_k^(1),作为实现正确性测试)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.counterfactual import rollout_pair
from src.evidence import branch_evidence
from src.utility import (
    L_quadratic,
    grad_u_e_zeta,
    grad_u_s_zeta,
    u_e_raw,
    u_s_raw,
)


@dataclass(frozen=True)
class CanonicalState:
    """t0 时刻的 canonical interaction 状态(可扩展,勿绑死裸标量)。
    gap0:bumper-to-bumper gap(米,ego 在前);
    v_ego0 / v_opp0:纵向速度(m/s);closing c = v_opp - v_ego。"""
    gap0: float
    v_ego0: float
    v_opp0: float
    site: str = ""
    event_id: str = ""


def m1_quantities(state: CanonicalState, v_ref: float, cfg: dict,
                  sigma_scale: float = 1.0) -> dict:
    """canonical state -> M1 所需四量(附各分支中间量供诊断)。"""
    cfg_e, cfg_u = cfg["evidence"], cfg["utility"]
    cfg_p = cfg["perception_noise"]
    dt = 1.0 / cfg["resample_hz"]
    horizon = float(cfg["counterfactual"]["horizon_s"])
    sigma_s = sigma_scale * cfg_p["sigma_s_ref"] * np.eye(2)
    sigma_e = sigma_scale * cfg_p["sigma_e_ref"] * np.eye(1)

    frames, _ = rollout_pair(state.gap0, state.v_ego0, state.v_opp0,
                             cfg["counterfactual"], dt)
    out = {"event_id": state.event_id, "site": state.site}
    L = {}
    for br, gcol, vcol, ccol in (("Y", "gap_Y", "v_opp_Y", "dv_Y"),
                                 ("H", "gap_H", "v_opp_H", "dv_H")):
        # rollout 的 dv 列即 c = v_O - v_E(正 = closing)
        ev = branch_evidence(frames[gcol].to_numpy(), frames[vcol].to_numpy(),
                             frames[ccol].to_numpy(), cfg_e, dt)
        u_s = u_s_raw(ev["g_tilde"], ev["c_tilde"], cfg_e, cfg_u)
        u_e = u_e_raw(ev["z_e"], v_ref, horizon, cfg_u)
        gs = grad_u_s_zeta(ev["g_tilde"], ev["c_tilde"], cfg_e, cfg_u, cfg_p)
        ge = grad_u_e_zeta(ev["z_e"], v_ref, horizon, cfg_u, cfg_p)
        L[f"s_{br}"] = L_quadratic(gs, sigma_s)
        L[f"e_{br}"] = L_quadratic(ge, sigma_e)
        out.update({
            f"g_tilde_{br}": ev["g_tilde"], f"c_tilde_{br}": ev["c_tilde"],
            f"z_e_{br}": ev["z_e"], f"u_s_{br}": u_s, f"u_e_{br}": u_e,
            f"L_s_{br}": L[f"s_{br}"], f"L_e_{br}": L[f"e_{br}"],
        })
    out["du_s"] = out["u_s_Y"] - out["u_s_H"]
    out["du_e"] = out["u_e_Y"] - out["u_e_H"]
    out["G_s"] = L["s_Y"] + L["s_H"]
    out["G_e"] = L["e_Y"] + L["e_H"]
    return out