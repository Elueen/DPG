"""R3(新 C4)测试:Fisher information。

覆盖:
  1) 矩阵性质:对称、PSD、逐 context 可加性、I = J^T W J 一致
  2) 逐参数信息结构(不预设"q 偏离 0.5 越好",数值判定):
     - indifference:kappa 类信息 ~ 0,w_s 信息为全场最大
     - saturated:所有参数信息趋零
     - safety_/efficiency_sensitive:选择性钉住对应 kappa
  3) 集合层 regime:M1 strong 为 identifiable、moderate 为 weakly、
     insufficient 与 M0/M0a(任何家族)为 non_identifiable
  4) 理论指导组合(designed_portfolio)min_eig 优于 strong 家族
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.analysis.fisher import (
    classify_regime,
    fisher_information,
    resolve_probe_contexts,
)
from src.games.contexts import PerceptionModel

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "step1_revision.yaml"


@pytest.fixture(scope="module")
def config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def pm(config) -> PerceptionModel:
    return PerceptionModel(config["perception"])


@pytest.fixture(scope="module")
def families(config, pm) -> dict:
    return {f: pm.build_context_set(c) for f, c in config["context_families"].items()}


@pytest.fixture(scope="module")
def probes(config, pm) -> dict:
    ctxs = resolve_probe_contexts(pm, config["fisher"], config["context_families"])
    return {c.name: c for c in ctxs}


def _theta(config):
    p = config["fisher"]["probe_theta"]
    return np.array([p["w_s"], p["kappa_s"], p["kappa_e"]])


def _fi(config, model, ctxs, n=None):
    idc = config["identifiability"]
    return fisher_information(
        model, _theta(config), ctxs,
        n if n is not None else config["fisher"]["n_per_context"],
        idc["fd_step_w"], idc["fd_step_logk"],
    )


# ----------------------------------------------------------------------
# (1) 矩阵性质
# ----------------------------------------------------------------------
def test_matrix_properties(config, families):
    rep = _fi(config, "m1", families["strong_diversity"])
    info = rep.information
    np.testing.assert_allclose(info, info.T, rtol=1e-12)
    assert np.all(rep.eigenvalues >= -1e-12)                   # PSD
    np.testing.assert_allclose(rep.per_context.sum(axis=0), info, rtol=1e-10)
    # I = J^T W J
    W = np.diag(rep.n_per_context / (rep.q * (1 - rep.q)))
    np.testing.assert_allclose(rep.jacobian.T @ W @ rep.jacobian, info, rtol=1e-10)


def test_scales_linearly_with_n(config, families):
    r1 = _fi(config, "m1", families["strong_diversity"], n=100)
    r2 = _fi(config, "m1", families["strong_diversity"], n=300)
    np.testing.assert_allclose(r2.information, 3.0 * r1.information, rtol=1e-10)


# ----------------------------------------------------------------------
# (2) 逐参数信息结构
# ----------------------------------------------------------------------
def test_indifference_pins_w_not_kappa(config, probes):
    rep = _fi(config, "m1", list(probes.values()))
    diag = rep.per_context_diag()
    names = list(probes.keys())
    ind = diag[names.index("indifference")]
    assert ind[1] < 0.05 and ind[2] < 0.05          # kappa 类信息 ~ 0
    assert ind[0] > 50                               # w_s 信息巨大
    assert ind[0] == max(diag[:, 0])                 # 且为全场 w_s 之最


def test_saturated_uninformative_for_all(config, probes):
    rep = _fi(config, "m1", list(probes.values()))
    diag = rep.per_context_diag()
    sat = diag[list(probes.keys()).index("saturated")]
    assert float(sat.sum()) < 0.05


def test_sensitivity_contexts_pin_matching_kappa(config, probes):
    rep = _fi(config, "m1", list(probes.values()))
    diag = rep.per_context_diag()
    names = list(probes.keys())
    ss = diag[names.index("safety_sensitive")]
    es = diag[names.index("efficiency_sensitive")]
    assert ss[1] > 100 * max(ss[2], 1e-9)            # 钉 kappa_s,不碰 kappa_e
    assert es[2] > 100 * max(es[1], 1e-9)            # 反之


# ----------------------------------------------------------------------
# (3) 集合层 regime
# ----------------------------------------------------------------------
def test_regime_classification(config, families):
    thr = config["fisher"]["regime_thresholds"]
    assert classify_regime(_fi(config, "m1", families["strong_diversity"]), thr) == "identifiable"
    assert classify_regime(_fi(config, "m1", families["moderate_diversity"]), thr) == "weakly_identifiable"
    assert classify_regime(_fi(config, "m1", families["insufficient_diversity"]), thr) == "non_identifiable"
    for model in ("m0", "m0a"):
        for fam in families.values():
            rep = _fi(config, model, fam)
            assert classify_regime(rep, thr) == "non_identifiable"
            assert rep.min_eigenvalue < 1e-10


# ----------------------------------------------------------------------
# (4) 理论指导组合
# ----------------------------------------------------------------------
def test_designed_portfolio_beats_strong_family(config, families, probes):
    designed = [probes[n] for n in config["fisher"]["designed_portfolio"]]
    r_designed = _fi(config, "m1", designed)
    r_strong = _fi(config, "m1", families["strong_diversity"])
    assert r_designed.min_eigenvalue > r_strong.min_eigenvalue
    assert r_designed.log_det > r_strong.log_det