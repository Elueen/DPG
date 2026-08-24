# Step 3 Split Proposal(已批准,2026-08-21;Task A 按此生成 manifest 后冻结)

> 原则:recording / temporal-block 级分割(非 interaction 级);
> step3_analysis 覆盖全部密度层与 ≥4 个 highD 站点;final_test 每站点
> 留 ≥1 recording,NGSIM 各站点留最后一个时段;pilot 四段维持
> train/dev 永久登记。批准前未查看任何 G/Jacobian/Fisher 量。

## step2_pilot(仅 replication / descriptive comparison,非 primary)
- highD: rec03(loc2), rec13(loc1)
- NGSIM: i80 0400-0415, us101 0750-0805

## step3_analysis(primary)
- highD loc3: rec04, rec05, rec06
- highD loc4: rec08, rec09
- highD loc5: rec16, rec17, rec19, rec21
- highD loc6: rec60
- highD loc1: rec28, rec35(中密度补层)
- NGSIM: i80 0500-0515, us101 0805-0820

## final_test(冻结,Step 3 全程禁触)
- highD loc2: rec01, rec02
- highD loc4: rec07, rec10
- highD loc5: rec20, rec23
- highD loc6: rec58, rec59
- highD loc1: rec44, rec47
- NGSIM: i80 0515-0530, us101 0820-0835

## 已声明局限
NGSIM 仅两个高速站点且均在 pilot 出现过——NGSIM 侧分割为
temporal-block 级而非 site 级;site-level unseen 证据由 highD
(loc3/loc4/loc5/loc6 四个 Step 2 未触站点)承担。