"""2x2 玩具博弈:ego ∈ {merge, wait} × opponent ∈ {yield, hold}.

payoff 分量 (u_safety, u_efficiency) 全部来自 config,本模块不内嵌任何数值。

情境参数 c 的定义与实现
-----------------------
c 控制两个分量在动作组合间的"离散度之比"(dispersion ratio):

    c = std(u_{分量0}) / std(u_{分量1})      对每个玩家分别成立

其中 std 取该玩家该分量在全部 2x2=4 个动作组合上的总体标准差 (ddof=0)。

缩放方式为 mean-preserving:只缩放各分量对其均值的偏差,
并保持两分量离散度的几何均值不变(sigma_0 * sigma_1 = 常数),
因此 c 只改变"哪个分量的动作间差异主导",不改变 payoff 的整体尺度,
也不改变各分量的均值水平。

    c > 1  -> 安全差异主导情境
    c = 1  -> 两分量离散度相等(均衡情境)
    c < 1  -> 效率差异主导情境
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

PLAYERS: tuple[str, str] = ("ego", "opponent")


@dataclass(frozen=True)
class ToyGame2x2:
    """不可变的博弈实例。

    payoffs 形状:(n_players, n_components, n_ego_actions, n_opp_actions)
    索引约定:payoffs[p, k, i, j] = 玩家 p 的分量 k 在
    (ego 选动作 i, opponent 选动作 j) 时的取值。
    """

    ego_actions: tuple[str, ...]
    opponent_actions: tuple[str, ...]
    components: tuple[str, ...]
    payoffs: np.ndarray
    situation_c: Optional[float] = None  # None = config 原始矩阵,未做情境缩放

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, game_cfg: dict) -> "ToyGame2x2":
        """从 config 的 game 段构造博弈。"""
        ego_actions = tuple(game_cfg["ego_actions"])
        opp_actions = tuple(game_cfg["opponent_actions"])
        components = tuple(game_cfg["components"])
        if len(components) != 2:
            raise ValueError("情境参数 c 是两分量离散度之比,components 必须恰为 2 个")

        payoffs = np.array(
            [
                [game_cfg["payoffs"][p][k] for k in components]
                for p in PLAYERS
            ],
            dtype=float,
        )
        expected = (len(PLAYERS), len(components), len(ego_actions), len(opp_actions))
        if payoffs.shape != expected:
            raise ValueError(
                f"payoff 形状 {payoffs.shape} 与动作/分量定义 {expected} 不一致"
            )
        return cls(
            ego_actions=ego_actions,
            opponent_actions=opp_actions,
            components=components,
            payoffs=payoffs,
        )

    # ------------------------------------------------------------------
    # 访问器
    # ------------------------------------------------------------------
    def component_payoffs(self, player: str) -> np.ndarray:
        """返回某玩家的分量 payoff,形状 (n_components, n_ego, n_opp)。"""
        return self.payoffs[PLAYERS.index(player)]

    def dispersions(self) -> np.ndarray:
        """各玩家各分量在动作组合上的 std,形状 (n_players, n_components)。"""
        return self.payoffs.std(axis=(2, 3), ddof=0)

    def dispersion_ratio(self) -> np.ndarray:
        """各玩家的 disp(分量0)/disp(分量1),形状 (n_players,)。"""
        s = self.dispersions()
        return s[:, 0] / s[:, 1]

    # ------------------------------------------------------------------
    # 情境实例化
    # ------------------------------------------------------------------
    def at_situation(self, c: float) -> "ToyGame2x2":
        """按情境参数 c 重缩放分量离散度,返回新博弈实例。

        对每个玩家:目标 sigma_0/sigma_1 = c 且 sigma_0*sigma_1 保持不变。
        """
        c = float(c)
        if not np.isfinite(c) or c <= 0:
            raise ValueError(f"情境参数 c 必须为正有限数,收到 {c}")

        means = self.payoffs.mean(axis=(2, 3), keepdims=True)
        deviations = self.payoffs - means
        sigma = self.payoffs.std(axis=(2, 3), ddof=0)  # (P, K)
        if np.any(sigma <= 0):
            raise ValueError(
                "config 基础 payoff 存在离散度为 0 的分量,无法定义离散度之比"
            )

        geo_mean = np.sqrt(sigma[:, 0] * sigma[:, 1])          # (P,)
        target = np.stack(
            [geo_mean * np.sqrt(c), geo_mean / np.sqrt(c)], axis=1
        )                                                       # (P, K)
        scale = (target / sigma)[:, :, None, None]
        new_payoffs = means + scale * deviations
        return replace(self, payoffs=new_payoffs, situation_c=c)
    