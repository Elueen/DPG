# DPG 项目约定(Claude Code 每 session 必读)

## 项目状态
- Step 1/2 已关闭(tags: step1-revision-complete, step2-complete)。
- 当前:Step 3(real-context identifiability)。任务书见
  docs/step3 任务书(用户提供);split 提案已批准:docs/step3_split_proposal.md。
- Step 2 冻结件(禁改):src/m1_interface.py 及其上游
  (canonical/interactions/actions/counterfactual/evidence/utility)、
  configs/step2_pilot.yaml(hash 14ccab6d)、六份 docs/step2_*.md spec。

## 纪律(违反即返工)
- 科学参数只进 config(step3 用 configs/step3.yaml);seed=42;
  输出目录带 config hash;findings 随做随写(docs/step3_findings.md)。
- 禁:NN、拟合 (w,κ)、MLE/MAP、synthetic choice、动 Step2 冻结件、
  按 identifiability 结果调 utility/Σ、触碰 final_test 分组。
- pilot/step3 数据不进 git(data/ 已忽略);只提交代码与文档。
- 修改前先跑 pytest tests/ -q 全绿;交付新代码必须带测试。

## 环境与执行
- conda: module load apps/binapps/anaconda3/2023.09 && source activate dpg
- 重计算一律 sbatch(模板见 slurm/*.sh;partition=multicore,
  --cpus-per-task>=2);登录节点只跑测试与秒级脚本。
- 数据路径见 configs/step2_pilot.yaml paths 段;NGSIM 用官方 txt
  (CSV 有 Excel 截断,加载器会拒绝)。

## 工作流
- 每个 Gate 一个 session;开工先做对接自检:
  1) 读 docs/step2_findings.md 末尾 Gate 判定书 + 相关 spec;
  2) pytest tests/ -q 全绿;
  3) 抽 3 个 pilot event 用 m1_quantities 重算,与
     data/pilot/m1_quantities.csv 逐位比对。
- Gate 收口 = 数字报给用户确认 -> spec/findings 落档 -> git commit。
- 分支:step3-dev;commit 信息前缀 [step3]。