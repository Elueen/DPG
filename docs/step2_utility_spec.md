# Step 2 Utility Specification(任务 D 冻结,二维 evidence 定稿版)

> 冻结日期 2026-08-19,config hash 14ccab6d(evidence / utility 段)。
> 代码 src/evidence.py、src/utility.py;归一化裁决 F-A(见
> docs/step2_normalization_spec.md);感知噪声空间见
> docs/step2_perception_noise_spec.md。

## 1. 符号与约定
- closing rate:c(t) = v_O(t) − v_E(t) = −ġ(t),c > 0 <=> closing。
- h_ref = 参考时距(0.9 s);d0 = standstill clearance(2 m);
  b_ref = 参考制动减速度(2.5 m/s²);ρ_g = softmin 温度(2 m);
  ρ_c = softmax 温度(0.5 m/s);β = softplus 陡度(2)。

## 2. Evidence(分支轨迹 -> C1 泛函)
- **safety(二维)**:
  g̃ = smin_{ρ_g, t}[ gap(t) − d0 − h_ref·v_O(t) ](米)
  c̃ = smax_{ρ_c, t}[ c(t) ](m/s)
  归一化 log-mean-exp 软极值:min ≤ smin ≤ mean、mean ≤ smax ≤ max,
  处处可微(hard min/max 被禁)。
- **efficiency(一维)**:z_e = p_O = ∫ v_O dt(米,梯形积分)。
- **v_ref**:外生 site-level reference speed(逐 site pilot population
  P85,排除 vx<0.5 静止帧;非 free-flow speed):loc2=37.13,
  loc1=33.66, i80=10.78, us101=16.04 m/s。
- Raw TTC 仅诊断(closing→0 奇异),不进 utility。

## 3. 解析 utility(identification toy functions,同构形式)
  m_s = g̃ − c₊²/(2 b_ref),c₊ = softplus(c̃; β)
  u_s^raw = tanh(m_s / 10 m)
  m_e = z_e − v_ref·T(T = 4 s)
  u_e^raw = tanh(m_e / 20 m)
物理解释:m_s = actual gap − standstill clearance − speed-dependent
headway − closing-distance requirement;margin = 0 <-> u = 0(reference
boundary);值域 (−1,1);零内部权重、零硬极值、处处 C1。
对标准化 evidence ζ 的完整链式梯度(a_g/a_c/a_e 显式,测试钉死;
L 以 ∇ᵀΣ∇ 矩阵形式实现)见 perception noise spec。
尺度常数由 pilot 分布知情,不拟合行为参数。

## 4. D5 诊断结果(n=1123,hash 14ccab6d)
- evidence:corr(g̃, c̃) = −0.145——两 safety 通道各携独立信息
  (二维结构的目的);g̃/c̃ 分布跨 regime 合理分层。
- 分布健康:饱和 u_s 11.0% / u_e 5.3%;sd 0.63/0.42,无近常数退化。
- Δu 结构:Δu_s≥0 与 Δu_e≤0 比例均 1.000;退化 0.6%;Δu_s 中位随
  拥堵单调(free-flow 0.0002 → i80 0.104)。
- 冗余:corr(u_s,u_e) = −0.13(H)/−0.10(Y),corr(Δu_s,Δu_e) = −0.26:
  语义可分,无 specification problem。
- 梯度互检(对 ζ,含链式因子):max_err 8.1e-11(tol 1e-6)PASS。
- u_e 中位为负系 v_ref=P85 参照的构造性含义,F-A 下如实保留。