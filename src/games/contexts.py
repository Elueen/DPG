"""Step 1 revision — evidence-based contexts 与 perception 派生量(R-C1/C2)。

决策过程:traffic evidence z_k -> component utility u_k = f_k(z_k) -> total utility。
opponent 对 evidence 的感知有噪声:z~_k = z_k + eta_k,eta_k ~ N(0, Sigma_k / kappa_k)。
一阶 Taylor 展开下,utility 层的感知扰动方差(单动作、单分量)为 L_k/kappa_k:

    L_k(a, c) = grad f_k(z_k)^T Sigma_k grad f_k(z_k)
    G_k(c)    = L_k(a1, c) + L_k(a0, c)        (二元动作,噪声跨动作独立)

纪律:
- f_k 为 identification toy functions(非最终真实驾驶 utility),
  函数族与参数、evidence 取值、Sigma_k 全部来自 config;
- G_k 永远不允许直接写在 config 中,必须由 f_k / z_k / Sigma_k 经
  数值微分计算得到(build 时对含 G/g_s/g_e 键的 context 配置直接报错);
- Step 1 revision 条件于 (a_E, c),不解 ego 侧:a_E 已吸收进 context 定义
  (config 中每个 context 的 evidence 即"给定该 a_E 与场景"的取值)。

evidence 本版为标量(每 分量 x 动作 一个 z),Sigma_k 为标量方差尺度。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COMPONENTS = ("safety", "efficiency")
ACTIONS = ("a1", "a0")  # a1 = yield, a0 = hold(约定见 config)
_FORBIDDEN_KEYS = {"G", "g", "g_s", "g_e", "G_s", "G_e", "gs", "ge"}


# ----------------------------------------------------------------------
# identification toy functions
# ----------------------------------------------------------------------
def toy_function(kind: str, params: dict):
    """返回向量化的 f(z)。函数族:logistic / softplus / linear。

    logistic: f(z) = A * (sigmoid((z - z0)/s) - 1/2)     奇函数(关于 z0)
    softplus: f(z) = A * s * log(1 + exp((z - z0)/s))    梯度 = A*sigmoid
    linear:   f(z) = A * (z - z0)                        梯度恒定(负对照可用)
    """
    a = float(params["amplitude"])
    s = float(params.get("scale", 1.0))
    z0 = float(params.get("shift", 0.0))
    if kind == "logistic":
        return lambda z: a * (1.0 / (1.0 + np.exp(-(np.asarray(z, float) - z0) / s)) - 0.5)
    if kind == "softplus":
        return lambda z: a * s * np.log1p(np.exp((np.asarray(z, float) - z0) / s))
    if kind == "linear":
        return lambda z: a * (np.asarray(z, float) - z0)
    raise ValueError(f"未知 toy 函数族 {kind!r}(可用:logistic/softplus/linear)")


def numerical_gradient(f, z: float, step: float) -> float:
    """中心差分数值微分(纪律:G 必须由 derivative 计算,不允许 config 直写)。"""
    return float((f(z + step) - f(z - step)) / (2.0 * step))


# ----------------------------------------------------------------------
# context 派生量
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class ContextQuantities:
    """单个 context 在给定感知模型下的全部派生量(K=2,分量顺序 = COMPONENTS)。"""

    name: str
    z: np.ndarray        # evidence,形状 (K, 2),列序 (a1, a0)
    u: np.ndarray        # component utility f_k(z),形状 (K, 2)
    delta_u: np.ndarray  # Δu_k = u_k(a1) - u_k(a0),形状 (K,)
    grad: np.ndarray     # f_k'(z),形状 (K, 2)
    L: np.ndarray        # utility sensitivity grad^2 * Sigma_k,形状 (K, 2)
    G: np.ndarray        # G_k = L_k(a1) + L_k(a0),形状 (K,)

    @property
    def g_ratio(self) -> float:
        """G_s / G_e:跨 context 是否变化 = M1 能否打破 precision collapse。"""
        return float(self.G[0] / self.G[1])


class PerceptionModel:
    """f_k、Sigma_k 与数值微分步长的容器;从 context 配置计算派生量。"""

    def __init__(self, perception_cfg: dict):
        self.f = {
            k: toy_function(perception_cfg["f"][k]["type"], perception_cfg["f"][k])
            for k in COMPONENTS
        }
        self.sigma = {k: float(perception_cfg["sigma"][k]) for k in COMPONENTS}
        if any(v <= 0 for v in self.sigma.values()):
            raise ValueError("Sigma_k 须为正")
        self.grad_step = float(perception_cfg["grad_step"])

    def quantities(self, name: str, context_cfg: dict) -> ContextQuantities:
        bad = _FORBIDDEN_KEYS & {
            key for comp in context_cfg.values() if isinstance(comp, dict)
            for key in comp
        } | (_FORBIDDEN_KEYS & set(context_cfg.keys()))
        if bad:
            raise ValueError(
                f"context {name!r} 含被禁键 {sorted(bad)}:G 必须由 f/z/Sigma 的 "
                "derivative 计算,不允许在 config 中直写"
            )
        z = np.array(
            [[float(context_cfg[k][a]) for a in ACTIONS] for k in COMPONENTS]
        )
        u = np.array([self.f[k](z[i]) for i, k in enumerate(COMPONENTS)])
        grad = np.array(
            [
                [numerical_gradient(self.f[k], z[i, j], self.grad_step) for j in range(2)]
                for i, k in enumerate(COMPONENTS)
            ]
        )
        L = grad**2 * np.array([[self.sigma[k]] for k in COMPONENTS])
        G = L.sum(axis=1)
        if np.any(G <= 0):
            raise ValueError(
                f"context {name!r} 某分量 G_k <= 0(f 梯度为零):无法定义 M1 方差"
            )
        return ContextQuantities(
            name=name, z=z, u=u, delta_u=u[:, 0] - u[:, 1], grad=grad, L=L, G=G
        )

    def build_context_set(self, contexts_cfg: dict) -> list[ContextQuantities]:
        return [self.quantities(name, c) for name, c in contexts_cfg.items()]