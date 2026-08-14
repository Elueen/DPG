# driving-profile-game

博弈论驾驶行为建模:研究驾驶员行为画像中 preference profile **w**(偏好权重)与 precision profile **κ**(分量级决策精度)的可观测含义与可辨识性结构。

## Step 1:玩具博弈上的参数可辨识性数值探察

在一个 2×2 玩具博弈(ego ∈ {merge, wait} × opponent ∈ {yield, hold})上,用数值方法回答:

> **w 与 κ 在什么条件下产生不同的可观测含义(均衡选择概率)?什么情境变异能把它们分离开?**

技术路线:

- **响应模型**:component-probit —— 效用分量级 iid Gaussian 扰动,U(a) = w_s·u_s(a) + w_e·u_e(a) + ε_s(a)/√κ_s + ε_e(a)/√κ_e;二动作情形有 Φ(·) 解析式,并以 Monte Carlo 交叉验证
- **均衡**:damped fixed-point iteration,多起点收敛检查
- **分析**:(w_s, κ_s, κ_e) 网格扫描 → 观测等价类(等价脊线)分析,单情境 vs 多情境(情境参数 c 控制 safety/efficiency 分量离散度之比)
- **观测长度**:模拟 N 次独立选择,暴力网格似然观察置信集合随 N 的集中

## 目录结构

```
configs/            所有科学参数与实验参数(YAML),脚本不得内嵌魔法数字
src/games/          博弈定义(toy_2x2)
src/response/       quantal response 计算(component_probit)
src/solvers/        均衡求解(fixed_point)
src/analysis/       网格扫描与等价类分析
experiments/step1/  每个实验一个入口脚本
tests/              解析 vs MC 互检、退化情形正确性
outputs/            图 + 数值结果 + config 拷贝(git 忽略)
data/               占位,本阶段禁止放任何真实数据
docs/               step1_findings.md 实验发现记录
```

## 环境

- Python 3.11 + venv(PyCharm virtualenv)
- 依赖仅限:numpy, scipy, matplotlib, pandas, pytest, pyyaml(见 `requirements.txt`,锁定版本;本阶段无神经网络)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 纪律约束(摘要)

1. 科学参数只从 `configs/` 读取
2. 随机性走显式 seed(config 指定,默认 42);产物文件名带 config hash 前八位
3. 每个实验自动写 outputs/:图(pdf+png)+ 数值结果 + config 完整拷贝
4. main 分支只放能跑通的代码;日常工作在 `step1-dev`,按任务节点 merge
5. 发现问题记录在 `docs/step1_findings.md`,不自行扩展范围
