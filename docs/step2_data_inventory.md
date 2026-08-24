# Step 2 Pilot Data Inventory

> **⚠️ SPLIT 登记(显著位置,永久有效)**:本文件登记的全部 pilot
> recording / 片段 **强制划入 train/dev split,永不进 test**。
> 登记日期 2026-08-18,config hash `e78fbbed`,
> 处理代码 src/data/pilot_prep.py(参数见 configs/step2_pilot.yaml)。

## Pilot ID 登记

- **highd_free_flow_rec03**:highD recording 03
- **highd_dense_rec13**:highD recording 13
- **ngsim_i80_15min**:NGSIM I80 文件 trajectories-0400-0415.txt,offset=0.0s,duration=900.0s
- **ngsim_us101_15min**:NGSIM US101 文件 trajectories-0750am-0805am.txt,offset=0.0s,duration=900.0s

## 处理规则摘要

- 统一重采样 10 Hz,SI 单位;标准化 schema:dataset, segment, vid, t, x, y, vx, vy, ax, lane, length, width, vclass, preceding, following
- highD:width/length 字段互换修正;行驶方向归一(方向 1 翻转符号);官方轨迹不二次滤波;25→10 Hz 线性插值
- NGSIM:英尺→米;逐车去重帧;位置 Savitzky–Golay 平滑(window=21, polyorder=3),速度/加速度由平滑轨迹 deriv=1/2 重算;取 15 分钟片段

- **数据源决定(含数据事故记录)**:NGSIM 采用 USDOT 官方发布包(I-80-Emeryville-CA.zip / US-101-LosAngeles-CA.zip)中的 **trajectories-*.txt 完整版**(空白分隔无表头,原始 schema,英尺)。依据:官方包内及 DG 目录下的全部 CSV 版本均恰为 1,048,575 数据行(=2^20-1,Excel 最大行数签名)且时间跨度 718-890s < 900s——系发布/流转链条中经 Excel 导出而被**行截断**,不完整;txt 版行数 111-118 万、span≈900s,完整性自证。DG 转换版 track_trajectories-*[_HD].csv 行数与截断 CSV 逐位相同,证明其转换自截断源,连同 provenance 不可验证一并排除;Neo/ 无轨迹数据,排除。加载器对 1,048,575 行的 CSV 直接报错拒绝。

## Canonical 语义冻结(A3,Gate A)

- canonical schema:dataset, segment, recording, site, regime, vid, t, x, y, vx, vy, ax, ay, lane, length, width, vclass, preceding, following(SI 单位,10 Hz 全局对齐网格)
- **位置语义**:x/y = 车辆**几何中心**;x 纵向、行驶方向为正。适配器转换:highD 原始 (x,y) 为包围盒左上角(+L/2, +W/2);NGSIM Local_Y 为车前端(−L/2)。两个假设由 canonical checks 用数据集自带 dhw / Space_Headway 实证校验
- **gap 口径**:bumper-to-bumper,gap = x_lead − x_follow − (L_lead+L_follow)/2
- **相对速度**:dv_closing = vx_follow − vx_lead,正 = 接近
- **横向/车道语义**:不强加全局左右约定;车道邻接与横向关系一律经逐 segment 经验车道中心(中位 y)几何判断
- **元数据**:dataset/recording/site/regime 全程保留;只统一表示语义,不统一真实分布
- 从 canonical 层起,行为学代码不得出现 dataset 分支

## 原始字段清单

### highd_free_flow_rec03

`frame`, `id`, `x`, `y`, `width`, `height`, `xVelocity`, `yVelocity`, `xAcceleration`, `yAcceleration`, `frontSightDistance`, `backSightDistance`, `dhw`, `thw`, `ttc`, `precedingXVelocity`, `precedingId`, `followingId`, `leftPrecedingId`, `leftAlongsideId`, `leftFollowingId`, `rightPrecedingId`, `rightAlongsideId`, `rightFollowingId`, `laneId`, `leftPreceding_V`, `leftPreceding_TTC`, `leftPreceding_indicator`, `leftPreceding_iTTCp`, `leftPreceding_DHW`, `rightPreceding_V`, `rightPreceding_TTC`, `rightPreceding_indicator`, `rightPreceding_iTTCp`, `rightPreceding_DHW`, `leftAlongside_V`, `leftAlongside_TTC`, `leftAlongside_indicator`, `leftAlongside_iTTCp`, `leftAlongside_DHW`, `rightAlongside_V`, `rightAlongside_TTC`, `rightAlongside_indicator`, `rightAlongside_iTTCp`, `rightAlongside_DHW`, `leftFollowing_V`, `leftFollowing_TTC`, `leftFollowing_indicator`, `leftFollowing_iTTCp`, `leftFollowing_DHW`, `rightFollowing_V`, `rightFollowing_TTC`, `rightFollowing_indicator`, `rightFollowing_iTTCp`, `rightFollowing_DHW`, `ttcIndicator`, `iTTCp`

### highd_dense_rec13

`frame`, `id`, `x`, `y`, `width`, `height`, `xVelocity`, `yVelocity`, `xAcceleration`, `yAcceleration`, `frontSightDistance`, `backSightDistance`, `dhw`, `thw`, `ttc`, `precedingXVelocity`, `precedingId`, `followingId`, `leftPrecedingId`, `leftAlongsideId`, `leftFollowingId`, `rightPrecedingId`, `rightAlongsideId`, `rightFollowingId`, `laneId`, `leftPreceding_V`, `leftPreceding_TTC`, `leftPreceding_indicator`, `leftPreceding_iTTCp`, `leftPreceding_DHW`, `rightPreceding_V`, `rightPreceding_TTC`, `rightPreceding_indicator`, `rightPreceding_iTTCp`, `rightPreceding_DHW`, `leftAlongside_V`, `leftAlongside_TTC`, `leftAlongside_indicator`, `leftAlongside_iTTCp`, `leftAlongside_DHW`, `rightAlongside_V`, `rightAlongside_TTC`, `rightAlongside_indicator`, `rightAlongside_iTTCp`, `rightAlongside_DHW`, `leftFollowing_V`, `leftFollowing_TTC`, `leftFollowing_indicator`, `leftFollowing_iTTCp`, `leftFollowing_DHW`, `rightFollowing_V`, `rightFollowing_TTC`, `rightFollowing_indicator`, `rightFollowing_iTTCp`, `rightFollowing_DHW`, `ttcIndicator`, `iTTCp`

### ngsim_i80_15min

`Vehicle_ID`, `Frame_ID`, `Total_Frames`, `Global_Time`, `Local_X`, `Local_Y`, `Global_X`, `Global_Y`, `v_Length`, `v_Width`, `v_Class`, `v_Vel`, `v_Acc`, `Lane_ID`, `Preceding`, `Following`, `Space_Headway`, `Time_Headway`

### ngsim_us101_15min

`Vehicle_ID`, `Frame_ID`, `Total_Frames`, `Global_Time`, `Local_X`, `Local_Y`, `Global_X`, `Global_Y`, `v_Length`, `v_Width`, `v_Class`, `v_Vel`, `v_Acc`, `Lane_ID`, `Preceding`, `Following`, `Space_Headway`, `Time_Headway`

## 派生变量支持度(直接可得 vs 需推算)

### highd

| 变量 | 支持度 |
|---|---|
| gap (dhw) | 直接可得(tracks.dhw,米;canonical 层亦可自算并已互校) |
| THW | 直接可得(tracks.thw,秒) |
| TTC | 直接可得(tracks.ttc,秒;无前车时为 0/负,需过滤) |
| Δv (与前车) | 直接可得(precedingXVelocity - xVelocity);canonical 自算 |
| 相对位置 | 可推算(canonical build_pairs,同帧 preceding 连接) |
| lateral offset | 可推算(canonical lane_centers,经验车道中心) |
| predicted delay | 需推算(自定义,基于期望速度) |
| 期望速度差 | 需推算(期望速度须外生定义) |

### ngsim

| 变量 | 支持度 |
|---|---|
| gap | 可推算(canonical build_pairs;Space_Headway 为前端-前端口径,仅校验用) |
| THW | 直接可得(Time_Headway,秒;0 值为无效标记) |
| TTC | 需推算(gap 与 dv_closing;平滑速度经 Preceding 连接) |
| Δv (与前车) | 可推算(canonical build_pairs,平滑速度) |
| 相对位置 | 可推算(canonical build_pairs) |
| lateral offset | 可推算(canonical lane_centers) |
| predicted delay | 需推算 |
| 期望速度差 | 需推算 |

## 质量统计(处理后 10 Hz 数据)

| segment | rows_10hz | vehicles_10hz | median_track_duration_s | mean_speed_mps | p95_speed_mps | vehicle_density_proxy | nan_ratio | negative_vx_ratio | abs_ax_gt_8_ratio |
|---|---|---|---|---|---|---|---|---|---|
| highd_free_flow_rec03 | 121218 | 914 | 12.5 | 30.43 | 41.15 | 11.98 | 0 | 0 | 0 |
| highd_dense_rec13 | 417404 | 2949 | 13.8 | 28.56 | 36.02 | 39 | 0 | 0 | 0 |
| ngsim_i80_15min | 1172935 | 1967 | 62.8 | 7.692 | 16.09 | 130.3 | 0 | 0.007601 | 0.001108 |
| ngsim_us101_15min | 1151501 | 2168 | 50.4 | 11.38 | 18.38 | 127.9 | 0 | 0.01473 | 0.0004516 |

NGSIM 附加:
- ngsim_i80_15min: duplicate_frame_ratio=0.0000, short_track_dropped=5
- ngsim_us101_15min: duplicate_frame_ratio=0.0000, short_track_dropped=0