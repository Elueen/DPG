# Step 1 Revision:发现记录与 Gate 判定

> **一句话结论:Gate 判定 PASS。** 在行为学上可辩护的随机模型(M1,
> perception-derived noise)下,preference(w)与 component-specific
> precision(κ_s, κ_e)**能够**产生可区分的 observable effects——
> 但当且仅当驾驶 context 使不同 payoff 分量对感知误差的敏感度之比
> G_s(c)/G_e(c) 跨 context 变化。naive 噪声谱系(M0a/M0)无论怎样
> 选择 context 都不可能做到,这是结构性定理级的 negative result。
>
> 全部数值在开发端与本地(macOS, conda env `dpg`)独立运行中逐位一致;
> 82 项测试全过。第一轮结果封存于 git tag `step1-round1` 与
> docs/step1_findings.md(未改动)。符号约定:θ_O=(w_s, κ_s, κ_e),
> σ_eff,q_O(c;θ),probit quantal response(不用 τ / q\* / QRE)。

---

## 0. Gate 判定书(逐条对照 revision 文档的 PASS 条件)

| # | PASS 条件 | 判定 | 证据 |
|---|---|---|---|
| 1 | M0 的 component-precision collapse 被解析和数值同时验证 | ✅ | §2 公式推导;`test_m0_no_go_all_families`(rtol 1e-12);Jacobian 零方向与理论 cos=1.0(§3.1) |
| 2 | M1 不再存在同一个 structural aggregate collapse | ✅ | strong 家族下无任何全局 collapse 方向:rank=3(§3.2)、κ 切片脊线 d₃=0.00(§3.3);M1 的退化只在 G 比恒定的负对照下按理论出现 |
| 3 | 存在明确可描述的 context 条件使 J(θ) full local rank 且非严重病态 | ✅ | 条件 = "G_s/G_e 跨 ≥3 个 context 变化";strong 家族 cond_rank 4.3–7.1(§3.2) |
| 4 | negative-control contexts 正确产生 weak/non-identification | ✅ | insufficient 家族:rank=2、Fisher min_eig~1e-16、脊线分毫不动、κ 恢复卡死(§3.2–3.5);moderate 家族给出 weak 的连续量化(cond 95–837) |
| 5 | finite-sample 在 identifiable regime 恢复 (w_s,κ_s,κ_e),在 non-identifiable regime 正确失败 | ✅ | §3.5:strong 下三参数全收敛;insufficient/M0 下 w 收敛而 κ 卡死,coverage 仍近名义 |
| 6 | analytic / numerical implementation / MC 一致 | ✅ | §4:线性化 MC 差 ≤2.5e-4;向量化-参考互检;M0a 与第一轮实现逐位一致 |

**判定:PASS,进入 Step 2 无结构性障碍。** 附带条件见 §6(Taylor 适用范围、
toy functions 的定位)。

---

## 1. 研究问题与模型谱系

**核心问题**:Can preference and component-specific precision generate
distinct observable effects under a behaviorally defensible stochastic model?

对象:opponent 的二元选择 a_O ∈ {a₁=yield, a₀=hold},条件于 (a_E, c)。
q_O(c;θ) = Φ(μ(c)/σ_eff),μ(c) = w_sΔu_s(c) + w_eΔu_e(c)。
三个模型只在 σ_eff 上不同:

| 模型 | stochastic assumption | σ²_eff | collapse 方向(固定 w_s) |
|---|---|---|---|
| **M0a**(legacy 第一轮) | 噪声加在 total-utility 尺度:U=Σw_ku_k+Σε_{k,a}/√κ_k | 2(1/κ_s+1/κ_e) | 1/κ_s+1/κ_e=C(与 w、c 无关) |
| **M0**(component-evaluation) | ũ_k=u_k+ε_{k,a},ε~N(0,1/κ_k),U=Σw_kũ_k | 2(w_s²/κ_s+w_e²/κ_e) | w_s²/κ_s+w_e²/κ_e=C(与 c 无关) |
| **M1**(perception-derived) | z̃_k=z_k+η_k,η~N(0,Σ_k/κ_k),一阶 Taylor;G_k(c)=Σ_a f_k'(z)²Σ_k | w_s²G_s(c)/κ_s+w_e²G_e(c)/κ_e | 仅当 G_s/G_e 跨 c 恒定时存在 |

M0 的意义:证明**问题不在噪声挂载点**(即使按正确的 component-evaluation
语义定义噪声,方差聚合仍与 context 无关)。M1 的机制:κ 分量以
context 相关的权重 G_k(c) 进入方差——G 比变化 ⇔ 分量精度留下不同
observable signature。G 一律由 f_k/z_k/Σ_k 的数值微分计算,config 禁写。

**context 三家族(设计与实测)**:strong_diversity(G_s/G_e 跨 4 个数量级:
166 / 0.01 / 1.8)、moderate_diversity(1.53 / 1.38 / 0.93)、
insufficient_diversity(负对照,G 恒定至数值微分精度;构造依赖 logistic
梯度偶对称的符号翻转技巧,μ 仍跨 context 变化)。

---

## 2. Structural no-go:naive 谱系为何必然失败(R1)

二动作 probit 下 q 只通过 (μ, σ_eff) 依赖参数。M0a/M0 的 σ_eff 与
context 无关 ⇒ 每个 context 只提供关于同一个二元组 (w_s, aggregate)
的约束 ⇒ κ_s、κ_e 永远只以聚合量可观测。数值验证:

- 同 aggregate 的 κ 对在三家族全部 context 上产生逐位相同的 q(rtol 1e-12);
- M0a 与第一轮 legacy 实现逐位一致(rel 1e-12)——revision 与 round-1
  代码互相钉死,旧代码零改动;
- M1 在 strong 家族打破 M0 的 collapse(同 M0-aggregate 的 κ 对给出
  max|Δq|>0.01 的可分 q 向量;交换 κ=(1,8)↔(8,1) 使 safety_/efficiency_
  sensitive 两 context 的 q 反向移动);
- M1 负对照按理论重新退化:G 恒定时 w_s²G_s/κ_s+w_e²G_e/κ_e=C 的
  κ 对给出 rtol 1e-9 相同的 q 向量。**M1 的可辨识性恰好由 G 比的
  context 变异开关,机制清晰、可正可反。**

---

## 3. 证据链:四条独立分析路线

### 3.1 局部可辨识性:Jacobian + SVD(R2)

参数化 θ̃=(w_s, log κ_s, log κ_e),中心差分,rank 阈值 s_i/s₁>1e-7。
3 模型 × 3 家族 × 3 probe 全组合:

- M0a/M0 恒 **rank=2**(s₃/s₁ ∈ 1e-11~1e-18,数值零);
- **零空间方向与理论 collapse 方向 cos=1.0**(容差 1e-8):三个模型的
  数值零方向分别对齐 (0,1/κ_e,−1/κ_s)、(0,w_e²/κ_e,−w_s²/κ_s)、
  (0,w_e²G_e/κ_e,−w_s²G_s/κ_s)——解析与数值完全互证;
- M1:strong → rank=3 且良态(cond 4.3–7.1);moderate → rank=3 但
  cond 95–837(weak 的连续量化);insufficient → 精确 rank=2;
- 子集分析:full rank 需要 ≥3 个异质 context(每 context 一条
  (μ,σ_eff) 约束的计数直觉);insufficient 任何子集到不了 rank 3。

### 3.2 Fisher information:什么 context 携带什么信息(R3)

I(θ)=Σ_c N_c/(q_c(1−q_c))∇q_c∇q_cᵀ,N_c=100。逐 context 对角信息(M1):

| context(x=μ/σ_eff) | I_ws | I_logκs | I_logκe |
|---|---|---|---|
| indifference(0.04) | **179.2** | 0.015 | 0.001 |
| moderate_contrast(0.93) | 28.7 | 6.43 | 0.39 |
| saturated(4.88) | 0.008 | 0.002 | 0.002 |
| safety_sensitive(0.53) | 0.83 | **3.94** | 0.000 |
| efficiency_sensitive(0.81) | 4.49 | 0.006 | **7.78** |

- **"q 离 0.5 越远越好"被证伪,正确表述是逐参数的**:indifference 对 κ
  零信息(κ 梯度 ∝ x·φ(x))却是 w_s 的全场最大信息源;saturation 对
  一切归零;sensitivity 类各自选择性钉住对应 κ(选择比 >10³)。
  这把第一轮"indifference 处 τ 弱可辨识"精细化为逐参数版本。
- **理论指导组合胜过粗放多样性**:{indifference + safety_sensitive +
  efficiency_sensitive} 的 min_eig=3.94 / logdet=8.61,优于 strong 家族
  整体(1.73 / 6.51)——context portfolio 可按"参数-信息映射"设计,
  是 Step 2+ 数据采集与情境选择的直接可操作结论。
- 集合层 regime(min_eig ≥0.1 identifiable,≥1e-4 weak):M1
  strong/moderate/insufficient = identifiable / weak / non;
  **M0a/M0 在全部集合(含 designed portfolio)下 non**。

### 3.3 等价脊线:collapse 方向的存活与移除(R4)

方法论要点:3D 总体积对所有模型都随 context 数收缩(w 方向总被钉住),
**正确判别量是 κ 切片(w_s 固定于参考格点)内等价集合的 Chebyshev
直径 d_n**。50³ 网格、tol=0.01、参考点 (0.6,3,3):

| 模型 | strong d₁→d₃ | moderate d₁→d₃ | insufficient d₁→d₃ |
|---|---|---|---|
| M0a | 0.59→0.55 | 0.55→0.53 | 0.53→0.53 |
| M0 | 0.63→0.61 | 0.61→0.59 | 0.59→0.59 |
| M1 | **1.00→0.00** | 0.65→0.35 | 0.63→0.63 |

M0 谱系:脊线随 context 变细但**从不断裂**(精确 level set 上的参数对
永远互在等价集合内,d₃/d₁≥0.93)——*context diversity cannot remove
structural collapse*。M1:strong 下 3 contexts 后 κ 切片收缩到单个格点;
insufficient 下 d₃ 与 d₁ 逐位相同并骑在理论加权双曲线上——*only the
theoretically required diversity can remove it*。九宫格主图:
fig_key_contrast。

### 3.4 三线互证

| 家族(M1) | Jacobian(§3.1) | Fisher(§3.2) | 脊线 d₃(§3.3) | recovery(§3.5) |
|---|---|---|---|---|
| strong | rank 3,cond<10 | identifiable(1.73) | 0.00 | 三参数全恢复 |
| moderate | rank 3,cond 10²–10³ | weak(6.3e-3) | 0.35 | κ 不收敛 |
| insufficient | rank 2 | non(~1e-16) | 0.63(不动) | κ 卡死 |

四条独立路线给出同一张可辨识性地图。

### 3.5 Finite-sample recovery(R5)

30³ 网格,500 replicates,θ\*=(0.586, 2.832, 2.832)(网格吸附),
预算严格守恒(multi 均分)。**95% likelihood-ratio confidence-region
coverage 全条件 0.966–0.984**(近名义、略保守,真参数网格化所致)。

逐参数 MLE 误差中位数(multi,N=10→1000):
- **m1|strong:成功恢复**——|Δlog κ| 2.36→0.39/0.20(≈1–2 格胞),w 亦收敛;
- m1|moderate:|Δlog κ_s|~0.98、|Δlog κ_e|~2.0 不随 N 收敛(weak);
- **m1|insufficient 与 m0|strong:正确失败**——w_s 收敛(0.034/0.069)
  而 κ 类最大误差保持 0.79–0.98(MLE 沿 collapse ridge 滑动,数据量
  救不了),coverage 仍近名义(LR 区域诚实包含整条 ridge)。

---

## 4. 数值可信度(Gate 条件 6)

- **解析 vs 线性化 MC**(同一模型):|Δq| ≤ 2.5e-4(4×10⁶ 样本,纯抽样误差)。
- **一阶 Taylor 适用范围**(解析 vs 全非线性感知 MC):κ∈[0.5,200] 内
  最大 |Δq|=3.9e-3(κ=0.5),随 κ 衰减(κ=3:6.6e-5)——远小于等价
  tol=0.01,M1 解析式在全部工作区适用。
- **实现互证**:M0a ≡ 第一轮 legacy 实现(rel 1e-12);闭式广播网格
  ≡ 逐点计算(rel 1e-12);数值微分 ≡ 手写解析梯度(rel 1e-6)。
- **可复现性**:seed=42,config hash 强制数据-参数同源;R1–R5 全部
  关键数值在开发端与本地独立运行中逐位一致;82 项测试全过。

---

## 5. 诚实记录(不利/易误读之处)

1. **塌缩模型里 w_s 反而收敛更快**(insufficient 0.034 vs strong 0.207):
   有效参数少一维。解读恢复曲线时,单参数误差小 ≠ 模型更可辨识。
2. **crossover 无普适 N 阈值**:本设定下 multi 在 N=10–30 即优于 single
   (12 个条件 crossover ∈ {10,30}),与第一轮均衡设定的 ~100–300 不同。
   只记录,不外推。
3. **f_k 为 identification toy functions**,非最终真实驾驶 utility;
   全部结论的因果链是"G 比是否跨 context 变化",不依赖 f 的具体形状,
   但 Step 2 换真实 utility 结构时 G 的量级需重新测。
4. Taylor 线性化在极低 κ(<0.5)下误差会继续增长;若 Step 2 需要
   极低精度 regime,需改用全非线性 MC 或高阶展开(待裁决)。

---

## 6. 对 Step 2 的直接含义

1. **模型层**:采用 M1 形式(感知噪声经效用敏感度进入决策方差);
   M0 谱系只作为 baseline/消融,不作为推断模型。
2. **数据/情境层**:辨识 (w_s, κ_s, κ_e) 的充分条件是 context 集合中
   G_s/G_e 有变化且 ≥3 个异质 context;最优采集组合按"参数-信息映射"
   设计(indifference 钉 w,sensitivity 类钉各 κ),避开 saturated 场景。
3. **诊断层**:Jacobian-SVD + Fisher min_eig + κ 切片脊线三件套可直接
   移植为真实数据集的 context 组合"可辨识性体检"。

---

## 7. 复现

```bash
conda activate dpg
pytest tests/ -q                                      # 82 passed
python experiments/step1_rev/run_identifiability.py   # R2
python experiments/step1_rev/run_fisher.py            # R3
python experiments/step1_rev/run_equivalence_rev.py   # R4
python experiments/step1_rev/run_recovery.py          # R5
```

产物在 `outputs/step1rev_{ident,fisher,equiv,recov}_<hash>/`。
第一轮(M0a,均衡设定)的复现见 docs/step1_findings.md 与 tag
`step1-round1`,与本轮完全隔离。