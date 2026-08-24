# Step 2 Counterfactual Specification(Gate C 冻结)

> 冻结日期 2026-08-19,config hash 5ff6b782(counterfactual 段)。
> 代码 src/counterfactual.py;冻结后不得按下游 utility/G_k 结果回调模板。

## 1. 分支定义(同一 x_t0 = (gap0, v_ego0, v_opp0))
- **Hold**:a_O^H(t) ≡ 0,opponent 匀速 v_opp(t0)。与 Gate B 冻结的
  D2 观测动作定义的"无响应=匀速外推"参照是同一数学对象(语义自洽)。
- **Yield**:升余弦减速脉冲 a_O^Y(t) = −a_peak·½(1−cos 2πt/T),
  t∈[0,T],a_peak=1.0 m/s²,T=4.0 s。
  - 光滑性:瞬时进入/退出恒减速会产生 acceleration discontinuity /
    impulsive jerk;升余弦 jerk 有界,max|jerk| = π·a_peak/T ≈ 0.79 m/s³。
  - 解析性质(无钳位):Δv(T)=a_peak·T/2=2 m/s;让出距离
    A(T)=a_peak·T²/4=4 m——与 pilot 观测 D2-yield 的 a_end 分布中心
    量级对齐(典型 2.7 m,中强让行 4–12 m)。

## 2. Ego 近似(适用范围声明)
两分支中 ego 均以 t0 状态匀速外推。constant-velocity ego 是当前
Step-2 的 **local longitudinal approximation**;若后续 utility 依赖
lateral merge geometry,再升级为固定 canonical ego trajectory,
本阶段不扩展。不使用任何真实未来轨迹(真实未来含 opponent 实际反应
的因果回路);两分支唯一差异 = opponent 模板,Δu_k 的全部变异严格
来自 tactical action。

## 3. 运动学与边界
- 确定性运动学积分,10 Hz;速度地板 v_opp ≥ 0,触发即钳位并计数。
- gap 允许为负:hold 分支的负 gap 是碰撞风险信号,为 safety utility
  的合法输入,不做截断。

## 4. Sanity 结果(n=1123,onset=lat_vel,valid & kin_physical)
- a_peak 实现值 = 配置值;jerk 上界与解析吻合(均 True)。
- 钳位事件 11/1123(≈1%,拥堵蠕行;钳位削弱让出距离,极端 v0≈0 时
  Δgap_end→0,系物理事实而非缺陷)。
- Δgap_end 中位 4.00 m = 解析值;hold 分支负 gap 比例:
  I-80 0.105 > US-101 0.030 > highD 0.000(与拥堵程度同构)。

## 5. 预注册敏感性(a_peak×{0.7,1,1.3} × T×{3,4,5}s,不做参数搜索)
- Δgap_end 均值 1.57–8.06 m,随参数平滑单调,sd/mean ≈ 5–7%;
- trajectory-level 量:min_gap_Y 恒 > min_gap_H(让行缓解风险,
  方向一致);events_clamped 随模板强度单调(11→35)。
- 结论:Δu 对模板参数的依赖平滑、无悬崖,不被单一任意选择主宰。