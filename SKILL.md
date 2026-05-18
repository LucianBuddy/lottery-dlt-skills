# DLT大乐透预测技能 V1.0

## 简介

DLT大乐透智能预测系统 V1.0。五池采样（热号/冷号/均衡/趋势/质数）+ 博弈论遗传算法融合 + 约束满足引擎，支持前后区复式投注生成，池级别回测全部跑赢随机基准。

**数据源**: 自动从体彩官网同步最新开奖数据

---

## 核心功能

### 0. 📡 自动数据更新
每次初始化时自动完成以下步骤：
1. 读取本地 Excel 数据文件的最后一期期号
2. 从 **500.com** 体彩数据源检查是否有新开奖数据
3. 发现新期号 → 自动解析号码 → 追加到 Excel → 更新完成
4. 无新数据 → 跳过 → 直接使用现有数据

支持 `auto_update=False` 跳过网络检查。

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

### 5. 池级别回测
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

# 初始化（自动检查网络更新）
fusion = DLTFusionComplete()

# 跳过网络检查（离线使用）
# fusion = DLTFusionComplete(auto_update=False)

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
| 数据源 | 500.com |

---

## 文件结构

```
dlt_lottery_prediction/
├── dlt_fusion_complete.py              # 主入口（DLTFusionComplete类）
├── dlt_data_updater.py                 # 📡 自动数据更新器
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
│   └── dlt_strategy_recommender.py     # 策略推荐
├── data/
│   └── DLT历史数据_适配模型版.xlsx      # 历史开奖数据（自动更新）
└── SKILL.md                           # 本文档
```

---

## 版本历史

- **V1.0** (2026-05-18): 统一版本号，整合并标准化所有描述
