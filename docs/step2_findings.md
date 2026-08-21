# Step 2 Findings(data-aware model specification,pilot 阶段)

> 随做随写;各任务的冻结定义见对应 spec 文档,本文件记录过程发现、
> 修订理由与最终判定。config hash 演进以各段落标注为准,收官 hash
> 14ccab6d。pilot 数据不进 git,仅 findings 与 spec 入库。

## A. Pilot 数据与 canonical 层(Gate A,2026-08-18)

- pilot 四段登记(强制 train/dev,永不进 test):highD rec03(loc2,
  free_flow,54.2 veh/min)/ rec13(loc1,dense,165.3 veh/min)/
  NGSIM I-80 0400-0415 / US-101 0750-0805(各 15 min)。
- **数据事故(重要留档)**:NGSIM 官方发布包及 DG 目录下全部 CSV 均恰为
  1,048,575 数据行(=2^20-1,Excel 最大行数签名)且 span 718-890s<900s
  ——发布/流转链条经 Excel 导出被行截断;改用官方 zip 内 trajectories-*.txt
  完整版(111-118 万行,span≈900s);DG 转换版行数与截断 CSV 逐位相同,
  证明转换自截断源,一并排除;加载器对 1,048,575 行 CSV 直接报错拒绝。
- 修复:重采样网格全局对齐(t=k·0.1,跨车时间戳精确重合)——修复前
  density_proxy≈1 的 bug 即由网格不对齐造成。
- canonical schema v2 语义冻结(几何中心、bumper gap、dv_closing 符号、
  经验车道中心几何、dataset/recording/site/regime 元数据、canonical 层后
  零 dataset 分支)。**位置假设实证钉死**:highD 包围盒假设 computed gap
  vs dhw median|err|=0.000m;NGSIM 前端假设 vs Space_Headway 0.04m 无偏
  ——Gate A 全 PASS。
- 质量:span=899.9s;mean_v 与拥堵状态自洽(i80 7.7 / us101 11.4 /
  rec13 28.6 / rec03 30.4 m/s);nan=0。

## B. Interaction 抽取与观测动作定义(Gate B,2026-08-19)

- 事件规模:highD rec03/rec13 = 119/316,NGSIM i80/us101 = 863/902;
  valid(onset=lat_vel)= 27/106/552/440。highD 约半数事件因 ego_coverage
  失效(观测窗短的物理约束);NGSIM onset_not_found 135-239(蠕行工况
  横向速度判据保守,属定义对比信息而非缺陷)。
- **交通波污染发现**:D1(减速判据)在 NGSIM 把 stop-and-go 振荡误判为
  yield(I-80 10%、US-101 5%、highD 0%);29 例不一致 case 逐帧核查全部
  证实(速度谷在响应窗末端、净位移反超匀速外推)。D3 继承污染。据此
  主定义选 D2(空间 accommodation),12 例分层人工复核(数值+逐帧轨迹)
  零误判。
- 数据质量:发现 NGSIM 跟踪毛刺(2s 内 12→0.3→12 m/s,|ax|~11、
  jerk~29 m/s³,物理不可能);物理合理性 flag(响应窗 max|ax|≤8)
  抓出 **2/1125**——其中 1 例(us101|2748|860.7,dv_red=12.2、
  a_end=16.1)曾被 D2 判 yield,flag 正确剔除;另 1 例 D2=hold 无害。
- corr(dv_red, a_end)=0.796:两信号族相关不冗余,D1 保留为诊断定义。
- 冻结:onset=lat_vel、H_pre=3s、H_resp=4s、opponent 规则、D2 主定义、
  ambiguous 排除似然但保留计数。详见 docs/step2_interaction_action_spec.md。

## C. Counterfactual 模板与 rollout(Gate C,2026-08-19)

- 模板:hold=匀速外推(与 D2 参照同一数学对象);yield=升余弦脉冲
  (a_peak=1.0, T=4s;Δv=2 m/s、A=4 m,与观测 a_end 分布中心对齐)。
- Sanity 全绿:解析式逐项吻合;钳位 11/1123(拥堵蠕行,物理事实);
  hold 分支负 gap 比例随拥堵单调(I-80 10.5%),safety 信号如实出现。
- 预注册敏感性:Δgap_end 1.57–8.06 m 平滑单调,sd/mean 5–7%,
  min_gap_Y 恒优于 min_gap_H。counterfactual 选择不主宰 Δu。
- 冻结见 docs/step2_counterfactual_spec.md(constant-velocity ego 为
  Step-2 local longitudinal approximation,适用范围已声明)。

## D. Evidence 与解析 utility(2026-08-19)

- 用户裁决修订:efficiency 弃 delay(与 v̄ 按构造近共线),单进度量;
  safety 改单一物理 margin(gap−反应距离−制动距离),零内部权重。
  随后 efficiency 显式改写为 progress-margin 形式(数学恒等,
  s_progress=20m=原 s_speed×horizon,恒等性有测试)。
- v_ref 冻结(外生 site-level reference speed,逐 site P85,非
  free-flow):loc2=37.13, loc1=33.66, i80=10.78, us101=16.04 m/s;
  单位全链 SI(canonical adapter 后无二次换算)。
- 诊断全过(标量版初跑):饱和 11.3%/5.3%;Δu_s≥0、Δu_e≤0 均 1.000;
  退化 0.7%;Δu_s 中位随拥堵单调;corr(Δu_s,Δu_e)=−0.27 无冗余;
  梯度互检 8.2e-12。u_e 因 P85 参照整体左移(构造性含义),F-A 下
  如实保留。二维修订后的定稿数字见 D/E 修订记录与 utility spec §4。

## E. Perception noise 参照(Gate E,2026-08-19)

- 标准化空间物理锚定(与 utility 物理尺度同源);Σ 冻结为 reference
  specification,κ_s/κ_e 承担 precision 的全部个体差异;预注册
  Σ→αΣ,α∈{0.5,1,2} 敏感性(G 冒烟内建);禁止按 identifiability 调 Σ。
  二维定稿(ζ_s=(g̃/a_g, c̃/a_c),Σ_s=I₂)见 D/E 修订记录与
  docs/step2_perception_noise_spec.md。

## F. Normalization(Gate F,2026-08-19,裁决 F-A;二维 evidence 复核维持)

- 三案对比(初版)与二维 evidence 复核(hash 14ccab6d)结论一致:
  F-B 把物理锚点 u_e=0 拖到 +0.561 且 G_e×2.59;F-C(diagnostic only)
  same-state max|Δ| u_s 0.199 / u_e 0.518,同一物理状态 G_e 缩放跨数据集
  差 29%——domain shortcut 定量反例。
- 冻结 F-A(intrinsic physics-based scaling):T_k=identity;
  复核条款(severe saturation / degeneracy / semantic inconsistency)
  均未触发,F-A 维持冻结。

## D/E 修订记录(2026-08-19,Gate G 前;结构正确性驱动,非结果驱动)

- 用户审查发现:标量 z_s(margin softmin)使感知噪声作用在内部推导量
  而非原始感知量,G_s 退化为纯饱和函数、closing 通道贡献被压平,
  伤及 Step 3 对 κ_s 的可辨识性。
- 修订(用户公式):safety evidence 升二维 (g̃, c̃)——
  g̃ = smin_ρg[gap − d0 − h_ref·v_O](d0=2m 静止间距,新增),
  c̃ = smax_ρc[c],c = v_O − v_E(符号写死,正=closing);
  m_s = g̃ − c₊²/(2 b_ref);u_s = tanh(m_s/s_s)。
  E:ζ_s=(g̃/a_g, c̃/a_c),Σ_s=I₂;∂u/∂ζ 完整链式并被测试钉死
  (含 a_g/a_c/a_e 因子,L 以矩阵形式实现)。
- 重跑影响(hash 14ccab6d):u_s 中位 −0.035(d0 扣除的预期收紧);
  corr(g̃,c̃)=−0.145,两 safety 通道各携独立信息(二维结构的目的);
  饱和 11.0%/5.3%;Δu 符号结构 1.000/1.000、退化 0.6%;梯度互检
  8.1e-11;其余诊断结论不变;F-A 复核维持。
  符号统一:h_ref/ρ_g/ρ_c/b_ref。

## G. 统一接口与最终冒烟(2026-08-19,G SMOKE PASS)

- src/m1_interface.py:CanonicalState -> m1_quantities ->
  [Δu_s, Δu_e, G_s, G_e],严格串联 C->D->E->F 冻结件,零新科学参数,
  对 θ_O 无知;G_k = L_k(Y)+L_k(H),L_k = ∇_ζu_kᵀΣ_k∇_ζu_k。
- G4 全绿(n=1123):有限、无 NaN、G_k≥0 非全零、G_max=4.53 无爆炸、
  Δu 非退化;Σ α-linearity 实测偏差 0.0e+00(实现正确性)。
- 观察(仅记录,判断留 Step 3):G_s/G_e 跨 regime 变异丰富
  (G_s 中位:free-flow 0.05 / dense 0.21 / us101 0.99 / i80 1.49),
  与 Step 1 context-portfolio 结构方向一致。
- 1123 全量仅因计算便宜,性质仍为 numerical smoke test。

---

# Step 2 最终 Gate 判定书(2026-08-19,全部 PASS,Step 2 关闭)

逐条对照任务书:

| Gate | 判定 | 依据 |
|---|---|---|
| A 数据/canonical | **PASS** | pilot 四段登记(强制 train/dev);canonical schema v2 语义冻结;位置假设实证 0 偏差(dhw/Space_Headway 互校);NGSIM Excel 截断事故确诊并换官方 txt 完整版留档 |
| B interaction/动作 | **PASS** | onset=lat_vel 冻结(三候选对比);opponent 规则+validity(含 max\|ax\|≤8 物理 flag,2/1125 毛刺剔除);主定义 D2(交通波污染证据链:29 例逐帧核查);12 例数值+轨迹人工复核零误判 |
| C counterfactual | **PASS** | hold=匀速(与 D2 参照同一数学对象)/yield=升余弦(jerk 有界);sanity 解析吻合;预注册敏感性 3×3 平滑单调;constant-velocity ego 适用范围声明 |
| D evidence/utility | **PASS** | 二维 safety (g̃,c̃)(d0+h_ref;c 符号写死)+一维 progress;同构 tanh(margin/scale);D4 单调性与 C1 全测试;corr(g̃,c̃)=−0.145 两通道独立;饱和 11%/5%;Δu 符号结构 1.000/1.000 |
| E perception noise | **PASS** | ζ 物理锚定(a_g/a_c/a_e);Σ_s=I₂、Σ_e=1 reference specification;完整链式 ∂u/∂ζ 测试钉死;α∈{0.5,2} 敏感性内建 |
| F normalization | **PASS** | 裁决 F-A(intrinsic physics-based scaling);三案对比+二维复核维持;F-B/F-C 仅 diagnostic;test set 重算永久禁止 |
| G 统一接口/冒烟 | **PASS** | m1_quantities 对 θ_O 无知、零新参数;G4 全绿(n=1123,G_max 4.53);Σ α-linearity 偏差 0.0e+00 |

**纪律自查**:全程零 NN、零 (w,κ) 拟合、零 benchmark、M1 数学结构未改、
科学参数全进 config(最终 hash 14ccab6d)、pilot 数据未进 git、
seed=42、findings 随做随写、三次用户裁决(D 修正/F 选择/D-E 二维修订)
均留档且修订理由为结构正确性而非结果驱动。

**移交 Step 3 的资产**:m1_quantities.csv(1123 事件四量)、
action_labels(D2 标签)、rollouts、全部冻结 spec(6 份)与统一接口。
**留给 Step 3 的问题(本步不判断)**:G_s/G_e 总体 variation 是否支撑
(w_s,κ_s,κ_e) 可辨识;真实数据 Jacobian/Fisher/recovery;
G 跨 regime 分层(0.05→1.49)与 Step 1 context-portfolio 结构的对应。