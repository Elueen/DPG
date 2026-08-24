# Step 2 Perception Noise Specification(Gate E 冻结,progress 修订版)

> 冻结日期 2026-08-19。M1 感知噪声:z̃_k = z_k + η_k,
> η_k ~ N(0, Σ_k/κ_k)。

## 1. Evidence scaling(E1:固定、跨数据集一致)
标准化 evidence 空间取 **物理锚定尺度**(与 utility 物理尺度同源,
使 f_k(ζ_k) = tanh(ζ_k) 为规范形式):
  ζ_s = z_s / a_s,a_s = 10 m(margin 尺度);
  ζ_e = (z_e − v_ref·T) / a_e,a_e = 20 m(progress 尺度)。
同一物理状态在两数据集得到同一 ζ(v_ref 为外生 site-level reference
speed,属状态语义而非数据集缩放);不用 dataset-specific z-score。

## 2. Reference covariance(E2)
z_s、z_e 均为标量,标准化空间中冻结:**Σ_s = 1,Σ_e = 1**。
这不是声称人类感知噪声协方差为单位量,而是 reference specification:
让 κ_s、κ_e 承担 component-specific precision 的全部个体差异。

## 3. 预注册敏感性(E3)
diagonal alternatives:Σ_k ∈ {0.5, 1, 2}(标量情形下 G_k 严格按 Σ_k
倍缩放,任务 G 冒烟一并输出)。禁止依据 G_s/G_e 是否有利于
identifiability 调整 Σ。