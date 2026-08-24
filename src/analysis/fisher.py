"""Step 1 revision — Fisher information / context informativeness(新 C4)。

Bernoulli 选择的 Fisher 信息(theta_tilde = (w_s, log ks, log ke) 坐标):

    I(theta) = sum_c N_c / (q_c (1 - q_c)) * grad q_c grad q_c^T
             = J^T W J,   W = diag(N_c / (q_c(1-q_c)))

梯度 grad q_c 复用 identifiability 的数值 Jacobian(中心差分)。

输出:特征值谱(最小特征值)、log-determinant(奇异时报 -inf 并以
min_eig 判读)、条件数、逐 context 的信息贡献 I_c 及其对角
(每参数的 per-observation 信息)。

probit 结构提示(供判读,不作为断言前提):记 x = mu/sigma_eff,
kappa 类参数的梯度 ∝ x * phi(x) —— indifference(x≈0)对 kappa 零信息;
saturated(|x| 大)对所有参数信息急剧衰减;w_s 的梯度含 Δu 差异项,
indifference 处未必为零。"哪类 context informative"由数值给出,不预设。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.analysis.identifiability import numerical_jacobian
from src.games.contexts import ContextQuantities
from src.response.models import q_vector


@dataclass(frozen=True)
class FisherReport:
    model: str
    theta: np.ndarray                 # (w_s, ks, ke)
    n_per_context: np.ndarray         # (C,)
    q: np.ndarray                     # (C,)
    jacobian: np.ndarray              # (C, 3),theta_tilde 坐标
    information: np.ndarray           # (3, 3) 总 Fisher 矩阵
    per_context: np.ndarray           # (C, 3, 3) 逐 context 贡献
    eigenvalues: np.ndarray           # 升序
    min_eigenvalue: float
    log_det: float                    # 奇异时 -inf
    cond: float                       # max_eig / min_eig(奇异时 inf)

    def per_context_diag(self) -> np.ndarray:
        """(C, 3):每个 context 对每个参数的对角信息贡献。"""
        return np.stack([np.diag(m) for m in self.per_context])


def fisher_information(
    model: str,
    theta: np.ndarray,
    contexts: list[ContextQuantities],
    n_per_context,
    fd_step_w: float,
    fd_step_logk: float,
    prob_clip: float = 1e-12,
) -> FisherReport:
    """n_per_context 可为标量(各 context 等量)或长度 C 的数组。"""
    C = len(contexts)
    n = np.broadcast_to(np.asarray(n_per_context, dtype=float), (C,)).copy()
    if np.any(n < 0):
        raise ValueError("N_c 须非负")
    q = q_vector(model, theta, contexts)
    qc = np.clip(q, prob_clip, 1.0 - prob_clip)
    J = numerical_jacobian(model, theta, contexts, fd_step_w, fd_step_logk)
    weights = n / (qc * (1.0 - qc))                       # (C,)
    per_context = weights[:, None, None] * J[:, :, None] * J[:, None, :]
    info = per_context.sum(axis=0)
    info = 0.5 * (info + info.T)                          # 数值对称化
    eig = np.linalg.eigvalsh(info)
    min_eig = float(eig[0])
    sign, logabsdet = np.linalg.slogdet(info)
    log_det = float(logabsdet) if sign > 0 else float("-inf")
    cond = float(eig[-1] / eig[0]) if eig[0] > 0 else float("inf")
    return FisherReport(
        model=model,
        theta=np.asarray(theta, dtype=float),
        n_per_context=n,
        q=q,
        jacobian=J,
        information=info,
        per_context=per_context,
        eigenvalues=eig,
        min_eigenvalue=min_eig,
        log_det=log_det,
        cond=cond,
    )


def classify_regime(report: FisherReport, thresholds: dict) -> str:
    """identifiable / weakly_identifiable / non_identifiable。
    thresholds: {min_eig_identifiable, min_eig_weak}(来自 config,
    仅作报告口径,不参与任何计算)。"""
    if report.min_eigenvalue >= thresholds["min_eig_identifiable"]:
        return "identifiable"
    if report.min_eigenvalue >= thresholds["min_eig_weak"]:
        return "weakly_identifiable"
    return "non_identifiable"


def resolve_probe_contexts(perception, fisher_cfg: dict, families_cfg: dict) -> list:
    """解析 fisher.probe_contexts:条目为 inline evidence,或
    {family_ref, context} 引用 context_families 中的既有 context(避免重复定义)。"""
    out = []
    for name, spec in fisher_cfg["probe_contexts"].items():
        if "family_ref" in spec:
            src = families_cfg[spec["family_ref"]][spec["context"]]
            out.append(perception.quantities(name, src))
        else:
            out.append(perception.quantities(name, spec))
    return out
