# Step 2 Interaction & Action Specification(Gate B 冻结)

> 冻结日期 2026-08-19。本文档冻结 interaction onset、opponent 定义、
> H_pre/H_response、观测动作 operational definition 与 validity mask。
> 参数值以 configs/step2_pilot.yaml(interaction / actions 段)为准;
> 代码 src/data/interactions.py、src/data/actions.py。
> 冻结后不得根据下游结果(utility/G_k/可辨识性)回改本定义。

## 1. 场景与角色
- 场景:ego lane-change / merge interaction(pilot 为高速直路段换道)。
- ego = 发起换道的车辆;opponent = t0 时刻 ego 目标车道中 x <= x_ego
  的最近车辆(canonical 几何中心坐标)。

## 2. 事件与 onset(冻结:lat_vel)
- 事件:ego lane 值跳变 = t_cross;同一 ego 5s 内连续跨线合并(取首次)。
- **onset 冻结为 lat_vel**:以 t_cross 结尾、vy·toward >= 0.2 m/s 持续
  >= 0.5 s 的运行段起点(上限 t_cross - 4 s);toward 由经验车道中心
  几何给出(canonical lane_centers,零 dataset 分支)。
- 弃选理由:fixed_lead(t_cross - 2 s)与真实机动动力学无关,仅作
  锚点诊断;compound 在拥堵工况额外损失 4-9% 事件而 t0 与 lat_vel
  几乎相同。lat_vel 与 fixed_lead 的 t0 中位差 0.70 s。
- H_pre = 3.0 s,H_response = 4.0 s(自 t0)。

## 3. Validity mask(全部满足才 valid;否则保留登记、不进建模)
1. onset 可定位(onset_not_found 剔除)
2. ego 覆盖 [t0 - H_pre, t_cross + 1 s]
3. opponent 存在(目标车道后方)
4. gap0 ∈ [0, 80 m](bumper-to-bumper)
5. opponent 覆盖 [t0, t0 + H_response] 且期间车道恒为目标车道
6. **物理合理性**:响应窗内 opp max|ax| <= 8 m/s²(NGSIM 跟踪毛刺剔除;
   pilot 中 dv_inc > 6 m/s 的疑似毛刺 3/1125,全部 D2=hold,不影响结论)

## 4. 观测动作定义(冻结:D2 gap-evolution 为主定义)
响应特征(opponent 轨迹,W = [t0, t0+4s]):
- dv_red = v_opp(t0) - min_W v_opp
- **a_end(accommodation)= [x_opp(t0) + v_opp(t0)·H_resp] - x_opp(t0+H_resp)**
  (相对"无响应=匀速外推"的让出距离;正 = 让行)

**主定义 D2**:a_end >= 2.0 m -> yield;a_end <= 0.5 m -> hold;否则 ambiguous。

弃选 D1(dv_red 阈值)与 D3(D1 OR D2 复合)的实证理由(onset=lat_vel,n=1125):
- D1 在拥堵工况被 stop-and-go 交通波污染:29 例 D1=yield & D2=hold 全部
  来自 NGSIM(I-80 D1-yield 的 10%、US-101 5%,highD 0%),逐帧核查
  确认其速度谷在响应窗末端、净位移反超匀速外推(a_end 中位 -0.68 m),
  系交通波振荡而非让行;D3 的 OR 结构继承该污染。
- D2 为积分量,免疫逐点振荡;正确捕捉 D1 漏掉的持续缓让
  (8 例 D1=ambiguous & D2=yield,逐帧确认为"缓降后恒速、gap 线性扩大")。
- 稳健性:±30% 阈值翻转率 D2 5-8% < D3 9-10% < D1 10-13%。
- 12 例分层人工复核(数值+逐帧轨迹)D2 全部正确。
- D1/D3 保留在代码与报告中作对比定义,不用于建模。

## 5. Ambiguous 策略
ambiguous(含 invalid)不进入后续行为建模的似然;计数与比例在所有
报告中保留(pilot 主口径 12-20%,见 findings)。

## 6. 标签分布快照(D2,onset=lat_vel)
| segment | n | yield | hold | ambiguous |
|---|---|---|---|---|
| highd_free_flow_rec03 | 27 | 0.111 | 0.704 | 0.185 |
| highd_dense_rec13 | 106 | 0.170 | 0.632 | 0.198 |
| ngsim_us101_15min | 440 | 0.200 | 0.675 | 0.125 |
| ngsim_i80_15min | 552 | 0.304 | 0.556 | 0.139 |

yield 率随拥堵/强制汇入程度单调上升(自由流 < US-101 < I-80),
行为学表面效度成立。

## 7. 已知局限(登记,不在 pilot 修复)
- 近停车 opponent(v_opp(t0) < 1 m/s)下 dv_red/a_end 均受物理约束,
  "静止让行"不可辨——pilot 仅 5/1125 例,记为边界情形。
- highD 观测窗短(中位轨迹 12-14 s),约半数事件因 ego_coverage 失效,
  属相机段长物理约束;NGSIM 不受此限。
- t_cross 依赖数据集自带 lane 标注的跳变时刻(10 Hz 量化)。