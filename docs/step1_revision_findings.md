# Step 1 Revision 发现记录

> 第一轮结果封存于 git tag `step1-round1` 与 docs/step1_findings.md(不改动)。
> 本文件记录 revision(M0a/M0/M1 谱系)的发现。符号:theta_O=(w_s,κ_s,κ_e),
> sigma_eff,q_O(c;theta),probit quantal response(不再用 τ / q* / QRE)。

## R1:三模型定义与两个 structural no-go(解析 + 数值)

**模型谱系(stochastic assumptions 与公式)**
- **M0a(legacy 第一轮)**:噪声加在 total-utility 尺度,
  U(a)=Σ w_k u_k(a)+Σ ε_{k,a}/√κ_k;σ²_eff = 2(1/κ_s+1/κ_e)。
  collapse:1/κ_s+1/κ_e=C,与 w、context 无关。
- **M0(component-evaluation 语义)**:ũ_k=u_k+ε_{k,a},ε~N(0,1/κ_k),
  U=Σ w_k ũ_k;σ²_eff = 2(w_s²/κ_s+w_e²/κ_e)。
  **no-go 依旧**:固定 w_s 下仅 w_s²/κ_s+w_e²/κ_e 可观测
  (`test_m0_no_go_all_families`,三家族全部 context,rtol 1e-12)。
  结论:即使按正确的 component-evaluation 语义定义噪声,naive model
  仍无法分离 κ_s、κ_e——问题不在噪声挂载点,而在方差聚合与 context 无关。
- **M1(perception-derived)**:z̃_k=z_k+η_k,η~N(0,Σ_k/κ_k),一阶 Taylor 下
  V₁(c)=w_s²G_s(c)/κ_s+w_e²G_e(c)/κ_e,G_k(c)=Σ_a f_k'(z)²Σ_k(数值微分计算,
  config 禁写 G)。candidate model,不预设成功。

**R1 已验证的结构事实**
1. M0a 与第一轮 legacy 实现逐位一致(rel 1e-12)——revision 与 round-1
   代码钉死,未改旧代码。
2. M1 在 strong_diversity 家族(G_s/G_e 跨 context 变化 4 个数量级)下
   打破 M0 的 collapse:同一 M0-aggregate 的 κ 对产生 max|Δq|>0.01 的
   可分 q 向量;交换 κ=(1,8)↔(8,1) 使 safety_sensitive / efficiency_sensitive
   两 context 的 q 反向移动——分量精度留下不同 observable signature。
3. **M1 的负对照按理论重新退化**:insufficient_diversity 家族
   (G 跨 context 恒定至数值微分精度;构造:效率侧 evidence 固定 +
   安全侧符号翻转,依赖 logistic 梯度偶对称)下,
   w_s²G_s/κ_s+w_e²G_e/κ_e=C 的 κ 对产生 rtol 1e-9 相同的 q 向量。
   即:M1 的可辨识性恰好由 G_s/G_e 的 context 变异开关,机制清晰。

**数值一致性校准**
- 线性化 MC vs 解析:|Δq| ≤ 2.5e-4(4×10⁶ 样本,纯抽样误差)。
- 全非线性感知 MC vs 一阶 Taylor 解析式:κ∈[0.5,200] 内最大 |Δq|=3.9e-3
  (κ=0.5),随 κ 增大衰减(κ=3:6.6e-5;κ=200:~0)。
  远小于等价 tol=0.01 → M1 解析式在工作区适用;此为 Taylor 近似的
  适用范围声明,进 Gate 的"analytic/numerical/MC 一致"证据。

**实现层决定(报备)**:旧代码与旧 config 零改动;revision 用独立
configs/step1_revision.yaml;新模块 src/games/contexts.py、
src/response/models.py;后续实验入口放 experiments/step1_rev/。