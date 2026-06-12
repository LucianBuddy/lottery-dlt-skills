---
name: dlt-lottery-prediction
description: 中国体育彩票超级大乐透（DLT）预测技能，基于深度学习的彩票号码预测系统。六池采样（热号/冷号/均衡/博弈/遗传/趋势）+ 博弈论遗传算法融合 + 约束满足引擎 + 跨期模式识别 + 尾号聚合检测 + AC值跟踪 + 偏差仪表盘校准。使用场景：(1) 预测下一期DLT大乐透号码 (2) 生成复式投注方案 (3) 回测模型性能 (4) 自动同步最新开奖数据
---

# DLT大乐透预测技能

六池采样 + 博弈论遗传算法融合 + 跨期模式识别 + 约束引擎，数据自动同步，支持复式+胆拖投注。

---

## 📡 数据同步

仅在调用 `predict()` 时触发。双源自动切换：
1. 优先查体彩API（webapi.sporttery.cn），被拦截则回退到 500彩票网
2. 发现新期号 → 追加到 Excel → 重新初始化子模块
3. 无新数据 → 跳过，直接使用现有数据

> 两个数据源均失败时可手动补充 Excel 文件。

---

## 核心架构

### 1. 多池加权采样（MultiPoolSampler，6池）

前后区各 6 个独立池，共 12 个生成器：

| 池 | 权重 | 策略 |
|----|------|------|
| 🔥 热号池 | 30% | 最近30期高频号码 |
| ❄️ 冷号池 | 15% | 最近30期低频号码 |
| ⚖️ 均衡池 | 20% | 热冷折中 |
| 📈 趋势池 | 20% | 和值趋势方向加权（上行→大号区，下行→小号区） |
| 🎯 博弈池 | 10% | 博弈论期望值排序 |
| 🧬 遗传池 | 5% | 遗传算法进化组合 |

### 2. 博弈论 + 遗传算法

- **博弈论输出层**：纳什均衡优化多策略输出
- **遗传算法**：全局组合适应度优化（频率 + 遗漏值 + 奇偶比 + 和值 + 30%模式匹配度）

### 3. 🧩 跨期模式识别（9种特征 + 第7池）

在6池基础上叠加模式识别层，从历史数据提取9种特征频率分布：

| 模式 | 权重 | 描述 |
|------|------|------|
| 跨度 | 12% | 最大号与最小号的差值 |
| 连号 | 10% | 相邻号码间隔分布 |
| 和值 | 12% | 5号之和，按5分组距 |
| 奇偶比 | 10% | 3:2 或 2:3 为高频 |
| 重号 | 15% | 与上期重复的号码数 |
| 尾号 | 10% | 个位数字分布多样性 |
| 三区分布 | 12% | 1-12/13-24/25-35 分布 |
| 质数 | 7% | 1-35中11个质数的命中数 |
| AC值 | 12% | 算术复杂度（6为最高频） |

**集成方式**：
- **模式池**（第7池）：从高频模式反推号码集合
- **评分增强**：base + 模式匹配度×35%
- **多样性池**：覆盖不同模式值的代表性号码

### 4. 尾号聚合 + AC值跟踪 + 偏差校准（V2.1.0）

- **尾号聚合检测**：候选组合尾号≤3种时降3%，4种(1重复)加1%
- **AC值跟踪评分**：基于最近100期AC值概率分布调整候选
- **偏差仪表盘**：自动跟踪和值/跨度/大小号偏差，偏差过大时补偿覆盖被低估方向的候选
- **置信度重校准**：根据偏差修正最终评分

### 5. 约束满足引擎

- 唯一性：每注内号码不重复
- 范围：前区1-35，后区1-12
- 格式：5+2 标准注
- 数学：和值、AC值、跨度限制

### 6. 复式投注生成

12种方案：6+3(18注) / 6+4(36注) / 7+2(21注) / 7+3(63注) / 7+4(126注) / 8+2(56注) / 8+3(168注) / 8+4(336注) / 8+5(560注) / 9+3(378注) / 9+4(756注) / 9+6(1890注)

---

## 使用方法

```python
from pathlib import Path
import sys

# 自动定位技能包的 scripts/ 目录
skill_dir = Path(__file__).resolve().parent / 'skills' / 'dlt-lottery-prediction'
scripts_dir = skill_dir / 'scripts'
sys.path.insert(0, str(scripts_dir))

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

## 输出格式

所有群聊和单聊的预测回复，严格遵循 `scripts/dlt_lottery_skill.py` 中 `_print_predict()` 的终端输出格式。

### 单式方案
```
🎯 单式方案 (Top 5)
 1. [xx xx xx xx xx] + [xx xx]  score=0.xxxx  p=xx.x%
 2. [xx xx xx xx xx] + [xx xx]  score=0.xxxx  p=xx.x%
```

### 复式方案
```
📋 复式方案
  6+3 (18注, 36元) × 2组:
    1: [xx xx xx xx xx] + [xx xx]
    2: [xx xx xx xx xx] + [xx xx]
```

### 胆拖方案
```
📊 胆拖方案
  🎯 多池融合胆拖:
    2胆3拖: 胆[xx xx] 拖[xx xx xx] 后区[xx xx]  336注672元  p=xx.x%
```

末尾加：`⚠️  仅供参考娱乐，请理性投注！`

---

## 技术指标

| 指标 | 数值 |
|------|------|
| 历史数据 | 自动同步（体彩API + 500彩票网双源） |
| 数据范围 | 7001期 ~ 最新 |
| 前区/后区 | 1-35 / 1-12 |
| 基础池数 | 6池 × 前后区 = 12个生成器 |
| 模式池 | 9种特征（第7池） |
| 复式方案 | 12种 |

---

## 文件结构

```
dlt-lottery-prediction/
├── SKILL.md                           # 技能说明文档（本文档）
├── scripts/                            # 可执行代码
│   ├── __init__.py                     # Python包入口
│   ├── dlt_fusion_complete.py          # 主入口（DLTFusionComplete类）
│   ├── dlt_data_updater.py             # 📡 自动数据更新器
│   ├── dlt_five_pool_fusion.py         # 多池融合（类名历史遗留）
│   ├── dlt_five_pool_sampler.py        # 旧版采样器（历史遗留）
│   ├── five_pool_sampler_complete_final.py # 多池采样器 (MultiPoolSampler)
│   ├── dlt_constraint_engine.py        # 约束引擎
│   ├── strategy_fusion_engine.py       # 策略融合引擎
│   ├── dlt_back_fusion.py              # 后区融合
│   ├── dlt_predictor_upgraded.py       # 升级版预测器
│   ├── dlt_optimized_predictor.py      # 优化预测器
│   ├── dlt_lottery_skill.py            # OpenClaw技能封装入口
│   ├── lottery_bayesian.py             # 贝叶斯分析
│   ├── lottery_calibration.py          # 校准模块
│   ├── lottery_metrics.py              # 指标计算
│   ├── lottery_time_series_cv.py       # 时间序列交叉验证
│   ├── prediction_store.py             # 预测存储模块
│   └── modules/                        # 子模块
│       ├── dlt_game_theory.py           # 博弈论分析
│       ├── dlt_genetic_optimizer.py     # 遗传算法
│       ├── dlt_math_filter.py           # 数学过滤
│       ├── dlt_statistics_analyzer.py   # 统计分析
│       ├── dlt_number_gravity.py        # 号码引力
│       ├── dlt_kill_number.py           # 杀号
│       ├── dlt_difference_sequence.py   # 差数序列
│       ├── dlt_matrix_displacement.py   # 矩阵位移
│       ├── dlt_compound_betting.py      # 复式投注
│       ├── dlt_betting.py               # 基础投注
│       ├── dlt_strategy_recommender.py  # 策略推荐
│       ├── dlt_pattern_recognizer.py    # 🧩 跨期模式识别器
│       └── neural_models.py             # 神经网络模型
├── assets/                             # 运行时文件
│   └── data/
│       └── DLT历史数据_适配模型版.xlsx   # 历史开奖数据
├── references/                         # 参考文档（需与代码版本同步）
│   ├── STRATEGY_FUSION_DESIGN.md       # 策略融合架构设计文档
│   ├── dlt_skill_config.json           # 运行时配置定义
│   ├── skill.yaml                      # 技能描述文件
│   └── sync_check.py                   # 🔁 版本同步检查工具
├── _meta.json                          # ClawHub元数据
└── .clawhub/                           # ClawHub元数据
```

---

## 版本历史

**V3.0.3** (2026-06-11) — 基于26064期对比分析的六项优化。

### 🎯 评分权重滑动窗口校准（优化②）
- 新增 `_recalibrate_score_weights()`：基于最近20期回测动态搜索最优 base/gt/genetic 权重组合
- 步长0.05，遍历权重空间，选择使Top5平均前区命中数最大的组合
- 回退机制：校准失败时使用默认 0.4/0.3/0.3 固定权重

### 📊 连续大和值偏斜补偿（优化③）
- `_compute_deviation_dashboard()` 新增 `consecutive_large_bias` 指标
- 检测最近5期和值>110的连续期数≥3时触发
- `_recalibrate_confidence()` 中对低和值(<95)候选×1.03补偿，纠正大号偏好

### 🔍 候选集盲区最小覆盖（优化④）
- 新增 `_ensure_min_coverage()`：检查并填补14/21/28边界号、遗漏质数、高遗漏号码
- 从历史开奖中取样含缺失号码的实际组合，确保覆盖面

### 🔗 强制配对重号组合（优化⑤）
- 新增 `_force_paired_repeat_combo()`：上期号码在候选集中频次≥30%时强制配对
- 与平衡池补充号码组合成一注，避免优质重号被分散到不同注

### 📐 复式方案三区配额修复（优化⑥）
- `_get_compound_front_pool()`：每区最低配额 target_size×20%，按近期频率补充
- `_get_compound_back_pool()`：强制1-4/5-8/9-12三段各至少1号码

---

**V3.0.2** (2026-06-09) — 基于26063期对比分析的五项优化。

### 🎯 奇偶比分布补充（优化1）
- 新增 `_enrich_parity_distribution()`：补充候选集缺失的奇偶比模式（4:1、1:4等）
- 26063期实际4:1（4奇1偶），预测全部2:3/3:2，完全遗漏，本项修复此盲区
- 每种奇偶模式保证最低覆盖率，从历史匹配号码池中补充

### 📐 前区三区覆盖强制约束（优化2）
- `stratified_sample()` 新增三区覆盖检查：每组候选必须覆盖至少2个区间
- 缺失区自动补充一个号码，避免候选过度集中在10-12/26-35区间

### 📊 区间漂移检测增强（优化3）
- 启动阈值从 0.15 降至 0.10，更早触发补偿
- 调整幅度从 max 8% 提高到 max 15%
- 权重系数从 ±0.3 提高到 ±0.5/±0.6，漂移信号影响更大

### ⚖️ 隔期重号奖惩平衡（优化4）
- 热号堆叠惩罚放松：3热号×0.93→×0.97，取消对2个热号的常规惩罚
- 冷号趋势延续加分提高：×1.03→×1.04
- 新增混合重号(1热+1冷)的中性加分类别：×1.02
- 隔期重号加分大幅提高：前区匹配×1.12→×1.20，双匹配×1.20→×1.28

### 🔄 复式方案独立采样（优化5）
- `_get_compound_front_pool()` 采样量扩展至 target_size×2
- 不再排除上期号码（保留合理重号）
- 强制三区均有代表号码
- 后区池主动补全所有1-12号码
- `_get_compound_back_pool()` 新增 trend_pool 来源

**V3.0.1** (2026-06-08) — 胆拖重构升级 + Python 3.11 迁移 + 输出格式标准化。

### 🎯 胆拖投注重构
- predict() 主流程胆拖生成从单池采样（`_generate_dan_tuo`）改为多池融合（`generate_dantuo_bets`）
- 新增 `n_dan_front` 参数，支持自动选胆时指定1胆/2胆/3胆
- 删除废弃的 `_generate_dan_tuo()` 方法，移除不存在的 'prime' 池

### 🔧 兼容性修复
- 修复 `dlt_compound_betting.py` 中 `math.comb` 在 Python 3.6 下的 ImportError
- Python 环境从 3.6.8 升级到 3.11.13（alinux3 源 + 阿里云 PyPI 镜像）

### 📋 输出格式标准化
- SKILL.md 新增「输出格式」章节，统一单式/复式/胆拖展示模板

---

**V3.0.0** (2026-06-07) — 全面代码清理 + 复式重写 + 胆拖投注 + 四大预测优化：

### 🧹 代码清理
- 重写 `dlt_lottery_skill.py`：从不可用组件（10个模块全部None）改为纯CLI适配层，6个子命令全部委托给DLTFusionComplete
- 重写 `__init__.py`：移除3个断裂import（`dlt_multi_fusion_pipeline`, `dlt_ranking_output`, `dlt_strategy_fusion_v2`），新增32个可用导出
- 识别5个死文件（`dlt_five_pool_fusion.py`, `dlt_five_pool_sampler.py`, `dlt_optimized_predictor.py`, `modules/dlt_betting.py`, `prediction_store.py`）

### 🔄 复式投注重写
- 集成 `modules/dlt_compound_betting.py`：全量枚举 C(N,5)×C(M,2) 组合代替单池前N个
- 6池合成候选池（热/冷/均衡/趋势/博弈/遗传）代替单策略硬编码
- 数学过滤（和值/奇偶/AC值）+ 多样性贪心选择
- 注入 final_score（5维评分体系：博弈论25%+数学特征20%+历史频率20%+后区命中率15%+模式匹配度20%）

### 🆕 胆拖投注（`generate_dantuo_bets`）
- 支持用户自定义胆码 + 自动高频选胆
- 多池合成拖码 + 全排列 + 多样性选择

### 🆕 凯利公式资金管理（`recommend_stake`）
- half-Kelly 计算基于命中概率的投注比例
- 归一化到预算

### 🎯 四大预测质量优化

| 编号 | 优化 | 方法 | 效果 |
|------|------|------|------|
| P1 | 🎯 号码过度集中抑制 | `_apply_diversity_penalty()` → 号码出现>40%的候选×0.93, >60%×0.85 | 26062期31(80%)垄断被消除 |
| P3 | 🔄 后区覆盖扩宽 | `_deduplicate_and_assign_back()` → 后区扩宽模式，前6注覆盖不同后区对 | 26062期01(80%)被分散 |
| P4 | 📊 和值中间带补充 | `_compensate_mid_sum()` → 中间带<20%时从平衡池补充到≥20% | 26062期110-120空白被填补 |
| P5 | 📐 三区分布动态校准 | `_recalibrate_confidence()` → Z1/Z3区偏差>30%时对补偿方向×1.02 | 26062期大号区60%(实际34%)被校准 |

### 📋 其他修复
- 修复复式注数计算：`front_count` 改为复式池大小（6/7/8/9）而非选中号码数（5）
- 修复 `select_diverse` 3元组不兼容问题

**V2.1.0** (2026-06-02, Neural集成 2026-06-06) — 六大优化 + 🧠 NeuralEnsemble：
1. 趋势池修正 → 5池变6池（热30%+冷15%+均衡20%+趋势20%+博弈10%+遗传5%）
2. 重号策略动态化：和值>120无重号预期，<70少重号
3. 尾号聚合检测 + AC值跟踪评分
4. 偏差仪表盘 + 置信度重校准
5. 奇偶约束放宽，允许全偶/全奇路径
6. 🧠 NeuralEnsemble集成：TabNet(25%)+LSTM(35%)+Transformer(40%)三模型集成，评分权重25%混入最终评分

**V2.0** (2026-05-28, 在 V1.1 基础上叠加) — 四大隔期重号优化：
- 隔期重号评分增强（×1.08~×1.20）
- 双期重号参考候选池
- 后区隔期重号检测（上上期加权0.85）
- 智能重号惩罚：热号堆叠×0.95 vs 冷号趋势延续×1.03
- 新增 `prediction_store.py` 预测存储模块

**V1.1** (2026-05-21) — 叠加跨期模式识别：9种模式特征评分 + 模式池 + 遗传适应度集成。

**V1.0** (2026-05-18) — 五池采样 + 博弈论遗传算法 + 约束引擎。

---

## 🔁 references/ 版本同步机制

版本升级时，必须同步更新 `references/` 下的文件，确保文档、配置与代码一致。

### 同步检查流程

```bash
# 运行同步检查脚本
python skills/dlt-lottery-prediction/references/sync_check.py
```

脚本自动检查以下内容：
- `dlt_fusion_complete.py` 中的 `VERSION` 常量
- `references/dlt_skill_config.json` 中的 `reference_sync_version`
- `references/skill.yaml` 中的 `reference_sync_version`
- `references/STRATEGY_FUSION_DESIGN.md` 中的版本标记

### 版本升级 Checklist

每次新版发布时：

1. **更新 `dlt_fusion_complete.py`**：修改 `VERSION` 和 `RELEASE_DATE` 常量
2. **更新 `references/dlt_skill_config.json`**：
   - `version` → 新版本号
   - `release_date` → 新日期
   - `reference_sync_version` → 新版本号
   - `reference_sync_date` → 新日期
3. **更新 `references/skill.yaml`**：
   - `version` → 新版本号
   - `metadata.updated` → 新日期
   - `metadata.reference_sync_version` → 新版本号
   - `metadata.reference_sync_date` → 新日期
4. **更新 `references/STRATEGY_FUSION_DESIGN.md`**：
   - 顶部的版本标记和同步日期
   - 如有架构变更，同步更新文档正文
5. **运行 `python sync_check.py` 验证**

### 运行时自动检查

`DLTFusionComplete.predict()` 调用时自动执行版本同步检查，发现不匹配会输出 warning（不影响预测流程）：
```
⚠️ 版本不匹配: 代码 VERSION=2.1.0, 配置 reference_sync_version=2.0.0
```
