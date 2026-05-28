# DLT大乐透预测技能 V1.1

## 简介

DLT大乐透智能预测系统 V1.1。五池采样（热号/冷号/均衡/博弈/遗传）+ 博弈论遗传算法融合 + 约束满足引擎 + **🧩 跨期模式识别（V1.1新增）**，支持前后区复式投注生成，池级别回测全部跑赢随机基准。

**数据源**: 自动从体彩数据API (webapi.sporttery.cn) 同步最新开奖数据

---

## 核心功能

### 0. 📡 触发预测时自动同步（双数据源）
仅在调用 `predict()` 时自动同步最新开奖数据：
1. 读取本地 Excel 数据文件的最后一期期号
2. 优先从 **体彩数据API (webapi.sporttery.cn)** 检查新开奖数据
3. 若体彩API失败或无数据 → 自动 **fallback 到 500彩票网 (datachart.500.com)** 解析HTML表格
4. 发现新期号 → 自动解析号码 → 追加到 Excel → 更新完成
5. 无新数据 → 跳过 → 直接使用现有数据
6. 若有新数据，自动重新初始化子模块（池采样器/遗传算法/模式识别器等）

> ⚠️ 体彩数据API受EdgeOne防护，部分环境可能被拦截。500彩票网作为可靠备选，
> 通过HTML表格解析获取数据，数据范围覆盖2007年至今所有期号。
> 两个数据源均失败时，可手动补充数据到Excel文件，或离线使用已有数据。

初始化时不主动同步，仅在预测时触发数据检查。

### 1. 五池加权融合采样
- 🔥 热号池（高频出现的号码）
- ❄️ 冷号池（长期未出现的号码）
- ⚖️ 均衡池（出现频率接近平均值的号码）
- 📈 趋势池（近期出现频率上升的号码）
- 🧬 质数池（数学上具有特殊性质的号码）

前后区各有独立的五个池，共计10个池生成器。

### 2. 博弈论遗传算法融合
- 博弈论输出层：纳什均衡优化多策略输出
- 遗传算法：全局优化号码组合适应度
- 适应度函数综合考虑号码频率、遗漏值、奇偶比、和值

### 3. 约束满足引擎
- ✅ 唯一性约束：每注内号码不重复（修复：池间重复采样Bug）
- 范围约束：前区1-35，后区1-12
- 格式约束：5+2标准注
- 数学关系约束：和值、AC值、跨度限制

### 4. 复式投注生成
支持12种复式投注方案：
- 前区：6+3 / 6+4 / 7+2 / 7+3 / 7+4 / 8+2 / 8+3 / 8+4 / 8+5 / 9+3 / 9+4 / 9+6
- 后区：2~6个号码

### 5. 🧩 跨期模式识别（V1.1新增）
在五池体系上叠加跨期模式识别层，从历史开奖数据中提取9种模式特征的频率分布，对候选组合计算模式匹配度评分。

| 模式 | 权重 | 描述 |
|------|------|------|
| 跨度 | 12% | 最大号与最小号的差值 |
| 连号 | 10% | 相邻号码的间隔（连号/隔号计数） |
| 和值 | 12% | 5个号码之和（按5分组距统计） |
| 奇偶比 | 10% | 奇数和偶数的比例（3:2或2:3为高频） |
| 重号 | 15% | 与上期重复的号码个数 |
| 尾号 | 10% | 号码个位数字的分布多样性 |
| 三区分布 | 12% | 1-12/13-24/25-35三区分布模式 |
| 质数 | 7% | 质数个数（1-35中质数共11个） |
| AC值 | 12% | 号码组合的算术复杂度（6为最高频） |

**集成方式**:
- **模式池**：作为第6个池，从高频模式反推号码集合，经分区均衡后输出
- **评分增强**：在 `base_score + gt_score + genetic_score` 基础上叠加35%权重的模式匹配度
- **遗传算法**：适应度函数中30%权重来自模式匹配度
- **多样性池**：覆盖不同模式值的代表性号码（与模式池互补）

### 6. 池级别回测
对每个池独立进行历史回测，验证策略有效性。V1.0回测结果：5个池在428期验证集上**全部跑赢随机基准**。

---

## 使用方法

```python
from pathlib import Path
import sys

# 自动定位技能包目录（无需硬编码路径，基于当前脚本所在目录推算）
skill_dir = Path(__file__).resolve().parent / 'skills' / 'dlt-lottery-prediction'
sys.path.insert(0, str(skill_dir))

from dlt_fusion_complete import DLTFusionComplete

# 初始化（不检查网络）
fusion = DLTFusionComplete()

# 预测（返回单式+复式，已保证每注号码唯一）
result = fusion.predict(include_compound=True)
print(result['single_bets'])    # 5注单式
print(result['compound_bets'])  # 12种复式

# 回测验证
bt = fusion.backtest(n_recent=100)
print(bt['pool_performance'])

# 独立检查数据更新
from dlt_data_updater import check_and_update
result = check_and_update()
print(f"新增{result['new_count']}期，最新{result['last_period']}")
```

---

## 技术指标

| 指标 | 数值 |
|------|------|
| 历史数据 | 自动同步（当前 2870期） |
| 数据范围 | 7001期 ~ 最新 |
| 前区范围 | 1-35 |
| 后区范围 | 1-12 |
| 复式方案 | 12种 |
| 回测基准 | 全部跑赢随机 |
| 数据源 | 体彩数据API (webapi.sporttery.cn) |
| 🧩 模式池 | 9种模式特征识别（V1.1） |
| 模式评分权重 | 最终评分占35%，遗传适应度占30% |

---

## 文件结构

```
dlt_lottery_prediction/
├── dlt_fusion_complete.py              # 主入口（DLTFusionComplete类）
├── dlt_data_updater.py                 # 📡 自动数据更新器（predict时触发，双源fallback）（体彩数据API）
├── dlt_five_pool_fusion.py             # 五池融合
├── dlt_five_pool_sampler.py            # 五池采样器
├── five_pool_sampler_complete_final.py # 最终采样器（修复池间重复Bug）
├── dlt_constraint_engine.py            # 约束引擎
├── strategy_fusion_engine.py           # 策略融合引擎
├── dlt_back_fusion.py                  # 后区融合
├── dlt_predictor_upgraded.py           # 升级版预测器
├── modules/                            # 子模块
│   ├── dlt_game_theory.py              # 博弈论分析
│   ├── dlt_genetic_optimizer.py        # 遗传算法
│   ├── dlt_math_filter.py              # 数学过滤
│   ├── dlt_statistics_analyzer.py      # 统计分析
│   ├── dlt_number_gravity.py           # 号码引力
│   ├── dlt_kill_number.py              # 杀号
│   ├── dlt_difference_sequence.py      # 差数序列
│   ├── dlt_matrix_displacement.py      # 矩阵位移
│   ├── dlt_compound_betting.py         # 复式投注
│   ├── dlt_betting.py                  # 基础投注
│   ├── dlt_strategy_recommender.py     # 策略推荐
│   └── dlt_pattern_recognizer.py       # 🧩 跨期模式识别器（V1.1新增）
├── data/
│   └── DLT历史数据_适配模型版.xlsx      # 历史开奖数据（体彩数据API自动更新）
└── SKILL.md                           # 本文档
```

---

## 版本历史

- **V1.1** (2026-05-21): 新增跨期模式识别模块（方案二），9种模式特征评分+模式池+遗传算法适应度集成
- **V1.0** (2026-05-18): 统一版本号，整合并标准化所有描述
- **V2.0** (2026-05-28): 四大优化方案
  - 方案A: 隔期重号评分增强 — 匹配上上期号码时加分(×1.08~×1.20)
  - 方案B: 双期重号参考候选池 — 基于双期号码生成隔期回归候选
  - 方案C: 后区隔期重号检测 — 上上期后区号码加权(0.85)
  - 方案D: 智能重号惩罚 — 区分热号堆叠(×0.95)与冷号趋势延续(×1.03)
  - 新增预测存储模块 prediction_store.py — 自动存档，保留最近2期
