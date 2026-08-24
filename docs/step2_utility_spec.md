# Step 2 Utility Specification(任务 D 冻结,progress-margin 修订版)

> 冻结日期 2026-08-19,config hash b622624f(evidence / utility 段)。
> 代码 src/evidence.py、src/utility.py;归一化裁决见
> docs/step2_normalization_spec.md(F-A:T_k = identity)。

## 1. Evidence(分支轨迹 -> 标量,C1)
- **z_s(safety)**:逐帧运动学 safety margin
  m(t) = gap(t) − τ_h·v_opp(t) − softplus(dv(t);β)²/(2b)
  (τ_h=0.9 s, b=2.5 m/s², β=2;零内部权重),
  z_s = 归一化 soft-min(τ=2 m;min ≤ softmin ≤ mean,处处可微)。
- **z_e(efficiency)**:z_e = p_O = 分支 rollout progress = ∫v_opp dt
  (米,梯形积分);reference progress p_ref = v_ref·T(T=4 s)。
- **v_ref**:外生 **site-level reference speed**(逐 site pilot
  population P85,排除 vx<0.5 静止帧;非 free-flow speed——未筛
  自由流状态):loc2=37.13, loc1=33.66, i80=10.78, us101=16.04 m/s。
- Raw TTC 仅诊断(closing->0 奇异),不进 utility。

## 2. 解析 utility(identification toy functions,同构形式)
  u_s^raw = tanh(m_s / 10 m),m_s = z_s
  u_e^raw = tanh(m_e / 20 m),m_e = z_e − v_ref·T
两者均为 u_k = tanh(physical margin_k / physical scale_k):
margin=0 <-> u=0(reference boundary);正/负两侧对称;值域 (−1,1);
饱和由物理尺度控制;两数据集同一含义。处处 C1、零硬 min/max/阶跃。
与速度形式数学恒等:tanh((p_O−v_ref·T)/20) ≡ tanh((v̄−v_ref)/5)
(s_progress = 原 s_speed × horizon;恒等性有测试)。
D4 单调性与解析梯度 (1−tanh²)/s 互检(max_err ~1e-11,tol 1e-6)
均内建为测试。尺度常数由 pilot 分布知情,不拟合行为参数。

## 3. D5 诊断结果(n=1123,progress 形式重跑)
- 饱和 u_s 11.3% / u_e 5.3%;sd 0.61/0.42,无近常数退化。
- Δu_s≥0 与 Δu_e≤0 比例均 1.000;退化(≈0)0.7%;
  Δu_s 中位随拥堵单调(free-flow ~0.0003 → I-80 ~0.095)。
- corr(u_s,u_e)≈−0.15/−0.11,corr(Δu_s,Δu_e)≈−0.27:语义可分,无
  specification problem。
- u_e 中位为负系 v_ref=P85 参照的构造性含义("多数车慢于参照"),
  F-A 下如实保留,见 normalization spec。