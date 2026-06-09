# DLT 大乐透多策略融合预测系统

> 中国体育彩票超级大乐透（DLT）智能预测技能 V3.0.0

---

## 概述

基于**六池加权采样 + 跨期模式识别 + 博弈论遗传算法 + 神经网络集成**的多策略融合预测系统。数据自动同步（体彩API + 500彩票网双源fallback），支持单式、复式（12种）、胆拖投注，以及凯利公式资金管理。

**代码规模：** ~7,600 行 Python（28个模块）

---

## 核心架构

### 1. 六池加权采样 (MultiPoolSampler)

前/后区各6个独立采样池：

| 池 | 权重 | 策略 |
|----|------|------|
| 🔥 热号池 | 30% | 近30期高频号码 |
| ❄️ 冷号池 | 15% | 近30期低频号码 |
| ⚖️ 均衡池 | 20% | 热冷折中 |
| 📈 趋势池 | 20% | 和值趋势方向加权（上行→大号区，下行→小号区） |
| 🎯 博弈池 | 10% | 博弈论期望值排序 |
| 🧬 遗传池 | 5% | 遗传算法进化组合 |

### 2. 跨期模式识别 (DLTPatternRecognizer)

9种特征频率分布 + 第7池（模式池）：

| 模式 | 权重 | 描述 |
|------|------|------|
| 跨度 | 12% | 最大号与最小号的差值 |
| 连号 | 10% | 相邻号码间隔分布 |
| 和值 | 12% | 5号之和，按5分组距 |
| 奇偶比 | 10% | 3:2 或 2:3 为高频 |
| 重号 | 15% | 与上期重复的号码数 |
| 尾号 | 10% | 个位数字分布多样性 |
| 三区分布 | 12% | 1-12 / 13-24 / 25-35 |
| 质数 | 7% | 1-35中11个质数的命中数 |
| AC值 | 12% | 算术复杂度（6为最高频） |

### 3. 神经网络集成 (NeuralEnsemble)

三模型集成评分，占 final_score 的 25%：

| 模型 | 权重 | 类型 |
|------|------|------|
| TabNet | 25% | 注意力机制表格网络 |
| LSTM | 35% | 长短期记忆序列网络（2层，seq_len=20） |
| Transformer | 40% | Transformer编码器（4头注意力，2层） |

### 4. 策略融合引擎 (StrategyFusionEngine)

5组独立策略线，每组5个评分维度。融合算法：加权投票 + 贝叶斯融合 + 博弈论均衡 + ML概率融合。

### 5. 后处理管线

```
候选池 → 博弈论评分 → 遗传算法 → 综合评分
  → 跨期模式增强 → 区间漂移补偿 → 隔期重号增强
  → 重号惩罚 → 6项特征评分(区间/散度/尾号/AC值/Neural)
  → 偏差仪表盘校准 → 策略约束验证 → 去重+后区分配
  → 最终排名 → 评分校准(P1/P4/P5) → 存档
```

---

## 功能

### 预测类型

| 类型 | 方法 | 说明 |
|------|------|------|
| **单式** | `predict()` | 5注标准号码（前区5个+后区2个） |
| **复式** (12种) | `generate_compound_bets()` | 6+3~9+6，全量枚举C(N,5)×C(M,2) |
| **胆拖** | `generate_dantuo_bets()` | 支持自定义胆码+自动高频选胆 |
| **凯利公式** | `recommend_stake()` | half-Kelly基于命中概率计算投注额 |

### 复式类型

| 类型 | 注数 | 金额 |
|------|------|------|
| 6+3 | 18注 | 36元 |
| 6+4 | 36注 | 72元 |
| 7+2 | 21注 | 42元 |
| 7+3 | 63注 | 126元 |
| 7+4 | 126注 | 252元 |
| 8+2 | 56注 | 112元 |
| 8+3 | 168注 | 336元 |
| 8+4 | 336注 | 672元 |
| 8+5 | 560注 | 1,120元 |
| 9+3 | 378注 | 756元 |
| 9+4 | 756注 | 1,512元 |
| 9+6 | 1,890注 | 3,780元 |

### 预测质量优化（V3.0.0）

| 编号 | 优化 | 方法 |
|------|------|------|
| P1 | 号码过度集中抑制 | 号码出现>40%的候选×0.93，>60%×0.85 |
| P3 | 后区覆盖扩宽 | 前6注覆盖不同后区对，同后区出现≥2次自动切换 |
| P4 | 和值中间带补充 | 中间带(100-120)覆盖<20%时从平衡池补充 |
| P5 | 三区分布动态校准 | Z1/Z3区偏差>30%时对补偿方向×1.02 |

---

## 快速开始

```python
from pathlib import Path
import sys

skill_dir = Path(__file__).resolve().parent / 'skills' / 'dlt-lottery-prediction'
scripts_dir = skill_dir / 'scripts'
sys.path.insert(0, str(scripts_dir))

from dlt_fusion_complete import DLTFusionComplete, data_dir

# 初始化
fusion = DLTFusionComplete(data_dir())

# 预测（单式+复式+胆拖）
result = fusion.predict(top_n=5, include_compound=True)
print(result['single_bets'])      # 5注单式
print(result['compound_bets'])    # 12种复式
print(result['dan_tuo_bets'])     # 胆拖方案

# 复式单独生成
compound = fusion.generate_compound_bets('all', n_per_type=2)

# 胆拖投注（自动选胆）
dantuo = fusion.generate_dantuo_bets()

# 胆拖投注（指定胆码）
dantuo = fusion.generate_dantuo_bets(
    dan_front=[8, 22],
    tuo_front_size=8,
    n_sets=3
)

# 凯利公式投注建议（基于命中概率）
stakes = fusion.recommend_stake(budget=200)

# 回测
bt = fusion.backtest(n_recent=100)
print(bt['pool_performance'])
```

### CLI 使用

```bash
# 预测
python dlt_lottery_skill.py predict
python dlt_lottery_skill.py predict --top-k 3 --no-compound

# 复式
python dlt_lottery_skill.py compound
python dlt_lottery_skill.py compound --type 7+3

# 胆拖
python dlt_lottery_skill.py dantuo
python dlt_lottery_skill.py dantuo --dan-front 8 22

# 回测
python dlt_lottery_skill.py backtest --periods 50

# 凯利公式
python dlt_lottery_skill.py stake --budget 200

# 技能信息
python dlt_lottery_skill.py info
```

---

## 数据

- **来源：** 体彩API（webapi.sporttery.cn）/ 500彩票网（双源自动fallback）
- **范围：** 7001期 ~ 最新
- **格式：** Excel（`assets/data/DLT历史数据_适配模型版.xlsx`）
- **同步：** `predict()` 首次调用时自动同步

---

## 文件结构

```
dlt-lottery-prediction/
├── README.md                              # 本文档
├── SKILL.md                               # 技能详细说明 + 版本历史
├── _meta.json                             # ClawHub元数据
├── assets/data/
│   └── DLT历史数据_适配模型版.xlsx         # 历史开奖数据（~2,880期）
├── references/                            # 参考文档
│   ├── dlt_skill_config.json              # 运行时配置定义
│   ├── skill.yaml                         # 技能描述文件
│   ├── STRATEGY_FUSION_DESIGN.md          # 策略融合架构设计文档
│   └── sync_check.py                      # 版本同步检查工具
└── scripts/                               # Python代码
    ├── __init__.py                        # 包入口（32个导出）
    ├── dlt_fusion_complete.py             # 主入口 (~2,600行)
    ├── dlt_lottery_skill.py               # CLI适配层（6个子命令）
    ├── dlt_predictor_upgraded.py          # 升级版预测器
    ├── strategy_fusion_engine.py          # 策略融合引擎
    ├── five_pool_sampler_complete_final.py # 多池采样器
    ├── dlt_back_fusion.py                 # 后区融合
    ├── dlt_constraint_engine.py           # 约束引擎
    ├── dlt_data_updater.py                # 自动数据更新器
    ├── prediction_store.py                # 预测存储（保留5期）
    ├── lottery_*.py                       # 贝叶斯/校准/指标/交叉验证
    ├── modules/
    │   ├── dlt_game_theory.py             # 博弈论分析
    │   ├── dlt_genetic_optimizer.py       # 遗传算法
    │   ├── dlt_math_filter.py             # 数学过滤
    │   ├── dlt_statistics_analyzer.py     # 统计分析
    │   ├── dlt_pattern_recognizer.py      # 跨期模式识别
    │   ├── neural_models.py               # TabNet+LSTM+Transformer
    │   ├── dlt_compound_betting.py        # 复式投注生成
    │   ├── dlt_kill_number.py             # 杀号
    │   ├── dlt_number_gravity.py          # 号码引力
    │   ├── dlt_difference_sequence.py     # 差数序列
    │   ├── dlt_matrix_displacement.py     # 矩阵位移
    │   └── dlt_strategy_recommender.py    # 策略推荐
    └── memory/                            # 运行时可写
        └── lottery_predictions.json       # 预测存档（保留5期）
```

---

## 版本历史

| 版本 | 日期 | 内容 |
|------|------|------|
| V1.0 | 2026-05-18 | 五池采样 + 博弈论遗传算法 + 约束引擎 |
| V1.1 | 2026-05-21 | 叠加跨期模式识别（9种特征） |
| V2.0 | 2026-05-28 | 四大隔期重号优化 |
| V2.1.0 | 2026-06-02 | 六池+趋势池+尾号聚合+AC值跟踪+偏差校准+NeuralEnsemble |
| **V3.0.0** | **2026-06-07** | **代码清理+复式重写+胆拖投注+凯利公式+四大预测质量优化** |

---

## 技术要求

- Python ≥ 3.8
- 依赖：PyTorch ≥ 2.0, pandas ≥ 2.0, numpy ≥ 1.24, scikit-learn ≥ 1.3, openpyxl

---

## 免责声明

彩票号码不可预测，本系统仅基于历史数据的模式分析提供概率参考。所有预测结果仅供娱乐，不构成投注建议。请理性购彩，量力而行。

⚠️ 数据说话，不搞玄学。回测结果 > 直觉判断。没有"必中"——这是彩票，不是印钞机。
