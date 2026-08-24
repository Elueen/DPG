"""R5(新 C6)测试:finite-sample recovery。

粗网格(15^3)、60 replicates 的快速版本,验证结构性行为:
  1) coverage:m1 strong(identifiable)在 N=300 多 context 下
     95% LR confidence region 的 empirical coverage >= 0.85
  2) 恢复:m1 strong 的 kappa 类 MLE 误差随 N 收敛
     (N=1000 误差 < 0.4 x N=10 误差)
  3) 正确失败:m1 insufficient 与 m0 strong 在 N=1000 时
     w_s 误差很小(< 0.1)而 kappa 类最大误差保持大(> 0.5)
     ——collapse 方向上似然平坦,数据量救不了
  4) 体积:随 N 收缩;m1 strong 在 N=1000 时 multi <= single
  5) 预算守恒与同 seed 确定性
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.step1.run_likelihood import split_budget
from experiments.step1_rev.run_recovery import run_condition, summarize
from src.analysis.direct_scan import build_grid, q_grids, snap
from src.analysis.equivalence import normalized_coords
from src.games.contexts import PerceptionModel

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "step1_revision.yaml"
N_GRID = 15
N_REP = 60


@pytest.fixture(scope="module")
def config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def env(config):
    pm = PerceptionModel(config["perception"])
    families = {f: pm.build_context_set(c) for f, c in config["context_families"].items()}
    grid = build_grid(config["equivalence_rev"]["grid"], n_override=N_GRID)
    ridx = snap(grid, config["equivalence_rev"]["reference_points"]["mid"])
    return families, grid, ridx, normalized_coords(grid)


def _run(config, env, model, fam, n_total, mode, seed_offset=0):
    families, grid, ridx, norm_axes = env
    ctxs = families[fam]
    rc = config["recovery"]
    q_list = q_grids(model, ctxs, grid)
    qc = [np.clip(q, rc["prob_clip"], 1 - rc["prob_clip"]) for q in q_list]
    logq = [np.log(q) for q in qc]
    log1q = [np.log(1 - q) for q in qc]
    if mode == "single":
        n_per = [0] * len(ctxs)
        n_per[int(rc["single_context_index"])] = int(n_total)
    else:
        n_per = split_budget(int(n_total), len(ctxs))
    assert sum(n_per) == n_total                      # 预算守恒
    rng = np.random.default_rng(config["meta"]["seed"] + seed_offset)
    out = run_condition(
        q_list, logq, log1q, grid, ridx, n_per, N_REP,
        rc["confidence_level"], rng, norm_axes,
    )
    return summarize(*out)


# ----------------------------------------------------------------------
# (1) coverage
# ----------------------------------------------------------------------
def test_coverage_identifiable_regime(config, env):
    s = _run(config, env, "m1", "strong_diversity", 300, "multi")
    assert s["coverage"] >= 0.85


# ----------------------------------------------------------------------
# (2) 恢复
# ----------------------------------------------------------------------
def test_m1_strong_kappa_errors_converge(config, env):
    s10 = _run(config, env, "m1", "strong_diversity", 10, "multi")
    s1000 = _run(config, env, "m1", "strong_diversity", 1000, "multi")
    for lab in ("abs_dlog_ks", "abs_dlog_ke"):
        assert s1000[lab] < 0.4 * max(s10[lab], 1e-9), lab
    assert s1000["abs_dw_s"] <= s10["abs_dw_s"]


# ----------------------------------------------------------------------
# (3) 正确失败
# ----------------------------------------------------------------------
def test_non_identifiable_regimes_fail_correctly(config, env):
    for model, fam in [("m1", "insufficient_diversity"), ("m0", "strong_diversity")]:
        s = _run(config, env, model, fam, 1000, "multi")
        assert s["abs_dw_s"] < 0.1, (model, fam, s["abs_dw_s"])            # w 恢复
        kappa_err = max(s["abs_dlog_ks"], s["abs_dlog_ke"])
        assert kappa_err > 0.5, (model, fam, kappa_err)                     # kappa 卡死
        assert s["coverage"] >= 0.85                                        # 覆盖仍近名义


# ----------------------------------------------------------------------
# (4) 体积
# ----------------------------------------------------------------------
def test_volume_shrinks_and_multi_wins_when_identifiable(config, env):
    s10 = _run(config, env, "m1", "strong_diversity", 10, "multi")
    s1000 = _run(config, env, "m1", "strong_diversity", 1000, "multi")
    assert s1000["volume_median"] < s10["volume_median"]
    single = _run(config, env, "m1", "strong_diversity", 1000, "single")
    assert s1000["volume_median"] <= single["volume_median"]


# ----------------------------------------------------------------------
# (5) 确定性
# ----------------------------------------------------------------------
def test_deterministic_with_seed(config, env):
    a = _run(config, env, "m1", "strong_diversity", 100, "multi", seed_offset=7)
    b = _run(config, env, "m1", "strong_diversity", 100, "multi", seed_offset=7)
    assert a == b