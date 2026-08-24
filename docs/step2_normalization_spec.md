# Step 2 Normalization Specification(Gate F 冻结)

> 冻结日期 2026-08-19,config hash b622624f。裁决:F-A。

## 1. 冻结方案:F-A — intrinsic physics-based scaling
  T_s(u) = u,T_e(u) = u(a_k = 0, b_k = 1;G_k^norm = G_k^raw)。
这不是"未做 normalization",而是 **normalization 已内生于 utility
设计**:两个 utility 同构为 u_k = tanh(physical margin_k / physical
scale_k)(safety:m_s = g̃ − τ_h v − c₊²/2b,scale 10 m;efficiency:
m_e = p_O − v_ref·T,scale 20 m),margin = 0 <-> u = 0 为 reference
boundary,值域均 (−1,1),饱和由物理尺度控制,两数据集同一含义。
精确表述:F-A avoids **empirical rescaling of utility outputs**,
while retaining the pre-specified site-level reference speed used in
the efficiency definition(v_ref 为 evidence 定义内的外生参照,
非对 utility 输出的经验缩放)。

## 2. 对比证据(n=1123;F-B/F-C 完整数字见
outputs/step# Step 2 Normalization Specification(Gate F 冻结)

> 冻结日期 2026-08-19,config hash b622624f。裁决:F-A。

## 1. 冻结方案:F-A — intrinsic physics-based scaling
  T_s(u) = u,T_e(u) = u(a_k = 0, b_k = 1;G_k^norm = G_k^raw)。
这不是"未做 normalization",而是 **normalization 已内生于 utility
设计**:两个 utility 同构为 u_k = tanh(physical margin_k / physical
scale_k)(safety:m_s = g̃ − τ_h v − c₊²/2b,scale 10 m;efficiency:
m_e = p_O − v_ref·T,scale 20 m),margin = 0 <-> u = 0 为 reference
boundary,值域均 (−1,1),饱和由物理尺度控制,两数据集同一含义。
精确表述:F-A avoids **empirical rescaling of utility outputs**,
while retaining the pre-specified site-level reference speed used in
the efficiency definition(v_ref 为 evidence 定义内的外生参照,
非对 utility 输出的经验缩放)。

## 2. 对比证据(n=1123;F-B/F-C 完整数字见
outputs/step2/normalization_.../normalization_comparison_b622624f.json)
- 物理锚点:margin=0 的映射——F-A 0.000/0.000;F-B −0.086/+0.561;
  F-C(diagnostic only)highd −0.019/+0.848、ngsim −0.174/+0.531。
  F-B/F-C 把"达到参照进度"错标为显著正效用。
- G_k 经验缩放:F-B G_s×1.19、G_e×2.59;F-C 下同一物理状态的 G_e
  在 highd(×3.46)与 ngsim(×2.68)间相差 29%——domain shortcut
  定量反例。same-state max|Δ|:u_s 0.300、u_e 0.518。
- F-B 纯仿射(不 clipping,越界 ~10.6% 如实保留)仍无法保物理锚点。

## 3. 纪律
- 常数记录:a_s=10 m、a_e(progress scale)=20 m、v_ref(site-level
  reference speed,P85)= {loc2:37.13, loc1:33.66, i80:10.78,
  us101:16.04} m/s;pilot 版本与处理链见 inventory;hash 见上。
- 永久禁止用 test set 重算任何上述常数。
- F-B/F-C 保留为 diagnostic(代码 run_normalization_comparison.py),
  不得转正。2/normalization_.../normalization_comparison_b622624f.json)
- 物理锚点:margin=0 的映射——F-A 0.000/0.000;F-B −0.086/+0.561;
  F-C(diagnostic only)highd −0.019/+0.848、ngsim −0.174/+0.531。
  F-B/F-C 把"达到参照进度"错标为显著正效用。
- G_k 经验缩放:F-B G_s×1.19、G_e×2.59;F-C 下同一物理状态的 G_e
  在 highd(×3.46)与 ngsim(×2.68)间相差 29%——domain shortcut
  定量反例。same-state max|Δ|:u_s 0.300、u_e 0.518。
- F-B 纯仿射(不 clipping,越界 ~10.6% 如实保留)仍无法保物理锚点。

## 3. 纪律
- 常数记录:a_s=10 m、a_e(progress scale)=20 m、v_ref(site-level
  reference speed,P85)= {loc2:37.13, loc1:33.66, i80:10.78,
  us101:16.04} m/s;pilot 版本与处理链见 inventory;hash 见上。
- 永久禁止用 test set 重算任何上述常数。
- F-B/F-C 保留为 diagnostic(代码 run_normalization_comparison.py),
  不得转正。v