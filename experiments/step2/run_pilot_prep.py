"""Step 2 任务 A 主入口:pilot 数据处理 + inventory 文档生成。

运行(仓库根目录,先填 config 路径与 highd.recordings):
    python experiments/step2/run_pilot_prep.py

流程:
  highD x2(free_flow / dense)+ NGSIM x2(I-80 / US-101 各 15 min)
  -> 标准化 10 Hz 轨迹表 data/pilot/processed/<name>.csv.gz(git 忽略)
  -> 质量统计 outputs/step2_prep_<hash>/pilot_prep_stats_<hash>.json
  -> 自动生成 docs/step2_data_inventory.md(字段清单、派生变量支持度、
     缺失/无效比例、pilot ID 永久登记)

纪律:pilot 数据不进 git;inventory 文档与代码进 git。
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.step1.common import REPO_ROOT, load_config, prepare_output_dir
from src.data.pilot_prep import (
    DERIVED_SUPPORT,
    load_highd_segment,
    load_ngsim_segment,
)

CFG_PATH = REPO_ROOT / "configs" / "step2_pilot.yaml"
PROCESSED_DIR = REPO_ROOT / "data" / "pilot" / "processed"
INVENTORY = REPO_ROOT / "docs" / "step2_data_inventory.md"


def _process_all(cfg) -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    root = Path(cfg["paths"]["highd_root"])
    stats = {}
    # --- highD ---
    for tag in ("free_flow", "dense"):
        rec = cfg["highd"]["recordings"][tag]
        if rec is None:
            raise ValueError(
                f"config highd.recordings.{tag} 未填:先跑 list_highd_candidates.py 挑选"
            )
        rid = f"{int(rec):02d}"
        name = f"highd_{tag}_rec{rid}"
        df, s = load_highd_segment(
            root / f"{rid}_tracks.csv",
            root / f"{rid}_tracksMeta.csv",
            root / f"{rid}_recordingMeta.csv",
            name, tag, cfg,
        )
        df.to_csv(PROCESSED_DIR / f"{name}.csv.gz", index=False)
        s["pilot_id"] = f"highD recording {rid}"
        stats[name] = s
        print(f"[{name}] rows={s['rows_10hz']} vehicles={s['vehicles_10hz']} "
              f"density_proxy={s['vehicle_density_proxy']:.1f}")
    # --- NGSIM ---
    for tag, key in (("i80", "ngsim_i80"), ("us101", "ngsim_us101")):
        name = f"ngsim_{tag}_15min"
        df, s = load_ngsim_segment(Path(cfg["paths"][key]), name, tag, cfg)
        df.to_csv(PROCESSED_DIR / f"{name}.csv.gz", index=False)
        s["pilot_id"] = (
            f"NGSIM {tag.upper()} 文件 {Path(cfg['paths'][key]).name},"
            f"offset={cfg['ngsim']['segment_offset_s']}s,"
            f"duration={cfg['ngsim']['segment_duration_s']}s"
        )
        stats[name] = s
        print(f"[{name}] rows={s['rows_10hz']} vehicles={s['vehicles_10hz']} "
              f"dup_frames={s['duplicate_frame_ratio']:.4f}")
    return stats


def _write_inventory(cfg, cfg_hash: str, stats: dict) -> None:
    lines = []
    lines.append("# Step 2 Pilot Data Inventory\n")
    lines.append(
        "> **⚠️ SPLIT 登记(显著位置,永久有效)**:本文件登记的全部 pilot\n"
        "> recording / 片段 **强制划入 train/dev split,永不进 test**。\n"
        f"> 登记日期 {date.today().isoformat()},config hash `{cfg_hash}`,\n"
        "> 处理代码 src/data/pilot_prep.py(参数见 configs/step2_pilot.yaml)。\n"
    )
    lines.append("## Pilot ID 登记\n")
    for name, s in stats.items():
        lines.append(f"- **{name}**:{s['pilot_id']}")
    lines.append("")

    lines.append("## 处理规则摘要\n")
    lines.append(
        "- 统一重采样 10 Hz,SI 单位;标准化 schema:dataset, segment, vid, "
        "t, x, y, vx, vy, ax, lane, length, width, vclass, preceding, following\n"
        "- highD:width/length 字段互换修正;行驶方向归一(方向 1 翻转符号);"
        "官方轨迹不二次滤波;25→10 Hz 线性插值\n"
        "- NGSIM:英尺→米;逐车去重帧;位置 Savitzky–Golay 平滑"
        f"(window={cfg['ngsim']['savgol_window']}, "
        f"polyorder={cfg['ngsim']['savgol_polyorder']}),"
        "速度/加速度由平滑轨迹 deriv=1/2 重算;取 15 分钟片段\n")

    lines.append(
        "- **数据源决定(含数据事故记录)**:NGSIM 采用 USDOT 官方发布包"
        "(I-80-Emeryville-CA.zip / US-101-LosAngeles-CA.zip)中的 "
        "**trajectories-*.txt 完整版**(空白分隔无表头,原始 schema,英尺)。"
        "依据:官方包内及 DG 目录下的全部 CSV 版本均恰为 1,048,575 数据行"
        "(=2^20-1,Excel 最大行数签名)且时间跨度 718-890s < 900s——"
        "系发布/流转链条中经 Excel 导出而被**行截断**,不完整;"
        "txt 版行数 111-118 万、span≈900s,完整性自证。"
        "DG 转换版 track_trajectories-*[_HD].csv 行数与截断 CSV 逐位相同,"
        "证明其转换自截断源,连同 provenance 不可验证一并排除;"
        "Neo/ 无轨迹数据,排除。加载器对 1,048,575 行的 CSV 直接报错拒绝。\n")
    lines.append("## Canonical 语义冻结(A3,Gate A)\n")
    lines.append(
        "- canonical schema:dataset, segment, recording, site, regime, vid, t, "
        "x, y, vx, vy, ax, ay, lane, length, width, vclass, preceding, following"
        "(SI 单位,10 Hz 全局对齐网格)\n"
        "- **位置语义**:x/y = 车辆**几何中心**;x 纵向、行驶方向为正。"
        "适配器转换:highD 原始 (x,y) 为包围盒左上角(+L/2, +W/2);"
        "NGSIM Local_Y 为车前端(−L/2)。两个假设由 canonical checks 用"
        "数据集自带 dhw / Space_Headway 实证校验\n"
        "- **gap 口径**:bumper-to-bumper,gap = x_lead − x_follow − (L_lead+L_follow)/2\n"
        "- **相对速度**:dv_closing = vx_follow − vx_lead,正 = 接近\n"
        "- **横向/车道语义**:不强加全局左右约定;车道邻接与横向关系一律经"
        "逐 segment 经验车道中心(中位 y)几何判断\n"
        "- **元数据**:dataset/recording/site/regime 全程保留;只统一表示语义,"
        "不统一真实分布\n"
        "- 从 canonical 层起,行为学代码不得出现 dataset 分支\n")

    lines.append("## 原始字段清单\n")
    for name, s in stats.items():
        lines.append(f"### {name}\n")
        lines.append("`" + "`, `".join(s["raw_columns"]) + "`\n")

    lines.append("## 派生变量支持度(直接可得 vs 需推算)\n")
    for ds, table in DERIVED_SUPPORT.items():
        lines.append(f"### {ds}\n")
        lines.append("| 变量 | 支持度 |")
        lines.append("|---|---|")
        for k, v in table.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    lines.append("## 质量统计(处理后 10 Hz 数据)\n")
    keys = ["rows_10hz", "vehicles_10hz", "median_track_duration_s",
            "mean_speed_mps", "p95_speed_mps", "vehicle_density_proxy",
            "nan_ratio", "negative_vx_ratio", "abs_ax_gt_8_ratio"]
    header = "| segment | " + " | ".join(keys) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(keys) + 1))
    for name, s in stats.items():
        vals = [f"{s.get(k, float('nan')):.4g}" if isinstance(s.get(k), float)
                else str(s.get(k, "-")) for k in keys]
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    lines.append("")
    lines.append("NGSIM 附加:")
    for name, s in stats.items():
        if name.startswith("ngsim"):
            lines.append(
                f"- {name}: duplicate_frame_ratio={s['duplicate_frame_ratio']:.4f}, "
                f"short_track_dropped={s['short_track_dropped']}"
            )
    INVENTORY.write_text("\n".join(lines), encoding="utf-8")
    print(f"inventory -> {INVENTORY}")


def main():
    cfg, cfg_hash = load_config(CFG_PATH)
    out_dir = prepare_output_dir("step2_prep", cfg_hash, CFG_PATH)
    stats = _process_all(cfg)
    with open(out_dir / f"pilot_prep_stats_{cfg_hash}.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    _write_inventory(cfg, cfg_hash, stats)
    print("完成。")


if __name__ == "__main__":
    main()