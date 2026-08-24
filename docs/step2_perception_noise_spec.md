# Step 2 Perception Noise Specification(Gate E 冻结,二维 evidence 版)

> 冻结日期 2026-08-19。M1 感知噪声:ζ̃_k = ζ_k + η_k,
> η_k ~ N(0, Σ_k/κ_k),作用在标准化 evidence 空间。

## 1. Evidence scaling(E1:固定、跨数据集一致)
物理锚定标准化(常数与 utility 物理尺度同源):
  ζ_s = (g̃/a_g, c̃/a_c),a_g = 10 m,a_c = 5 m/s;
  ζ_e = (z_e − v_ref·T)/a_e,a_e = 20 m。
同一物理状态在两数据集得到同一 ζ(v_ref 为外生 site-level reference
speed,属状态语义而非数据集缩放);不用 dataset-specific z-score。
utility 对 ζ 的梯度按完整链式实现并被测试钉死:
  ∂u_s/∂ζ_g = a_g(1−u_s²)/s_s;
  ∂u_s/∂ζ_c = −a_c(1−u_s²)/s_s · c₊·σ(βc̃)/b_ref;
  ∂u_e/∂ζ_e = (a_e/s_e)(1−u_e²)(a_e/s_e 显式保留,不因当前=1 省略)。

## 2. Reference covariance(E2)
  **Σ_s = I₂(g̃, c̃ 两通道),Σ_e = 1**。
这不是声称人类感知噪声协方差为单位阵,而是 reference specification:
让 κ_s、κ_e 承担 component-specific precision 的全部个体差异;
L_k = ∇_ζ u_kᵀ Σ_k ∇_ζ u_k 以矩阵形式实现,不写死 Σ=I 的化简。

## 3. 预注册敏感性(E3)
Σ_k -> αΣ_k,α ∈ {0.5, 1, 2}:理论上 G_k^(α) = α·G_k^(1),
作为实现正确性测试内建于 G 冒烟(实测偏差 0.0e+00)。
禁止依据 G_s/G_e 是否有利于 identifiability 调整 Σ。