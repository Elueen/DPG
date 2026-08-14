"""experiments/step1 共享工具(非实验入口):

- config 读取与 sha256 哈希(纪律 2:产物文件名带 config hash 前八位)
- 输出目录准备与 config 完整拷贝(纪律 3)
- 图的 pdf + png 双格式保存(纪律 3)
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "step1_toy_game.yaml"
OUTPUTS_ROOT = REPO_ROOT / "outputs"


def load_config(path: Path | str = DEFAULT_CONFIG) -> tuple[dict, str]:
    """读取 config,返回 (dict, 文件字节 sha256 的前 8 位)。"""
    path = Path(path)
    raw = path.read_bytes()
    cfg = yaml.safe_load(raw.decode("utf-8"))
    return cfg, hashlib.sha256(raw).hexdigest()[:8]


def prepare_output_dir(experiment_name: str, cfg_hash: str, config_path: Path | str = DEFAULT_CONFIG) -> Path:
    """建 outputs/<experiment>_<hash>/ 并放入 config 完整拷贝。"""
    out_dir = OUTPUTS_ROOT / f"{experiment_name}_{cfg_hash}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, out_dir / f"config_used_{cfg_hash}.yaml")
    return out_dir


def save_figure(fig, out_dir: Path, stem: str, cfg_hash: str) -> None:
    """pdf + png 双格式保存,文件名带 config hash。"""
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}_{cfg_hash}.{ext}", dpi=200, bbox_inches="tight")