---
name: dlt-lottery-prediction
description: 中国体育彩票超级大乐透（DLT）预测技能，基于深度学习的彩票号码预测系统。六池采样（热号/冷号/均衡/博弈/遗传/趋势）+ 博弈论遗传算法融合 + 约束满足引擎 + 跨期模式识别 + 尾号聚合检测 + AC值跟踪 + 偏差仪表盘校准 + 可学习排序模型(P1) + 决策树评分(P3) + 条件计算图(P4) + 内存分级推理(方案4)。使用场景：(1) 预测下一期DLT大乐透号码 (2) 生成复式投注方案 (3) 回测模型性能 (4) 自动同步最新开奖数据
---

# DLT大乐透预测技能 V3.8.0

六池采样 + pairwise号码共现惩罚 + 跨期MLP模式迁移(2层网络) + 成本感知加权标签(按奖项等级) + 参数配置外化 + 集成投票 + 发散度控制 + 异常响应 + 不确定性量化。数据自动同步，支持复式+胆拖投注。

---

## 数据同步

调用 `predict()` 时自动触发。体彩API优先 → 500彩票网回退，新期号追加到 Excel。两个数据源均失败时可手动补充 Excel 文件。

## 核心调用

```python
from dlt_fusion_complete import DLTFusionComplete

fusion = DLTFusionComplete()
result = fusion.predict(include_compound=True)
print(result['single_bets'])    # 5注单式
print(result['compound_bets'])  # 12种复式

bt = fusion.backtest(n_recent=100)  # 回测
```

## 输出格式

所有回复严格遵循 `dlt_lottery_skill.py` 中 `_print_predict()` 的终端格式。

```
单式: [xx xx xx xx xx] + [xx xx]  score=0.xxxx  p=xx.x%
复式: 6+3 (18注, 36元) / 7+2 (21注) ... 12种方案
胆拖: 胆[xx xx] 拖[xx xx xx] 后区[xx xx]
末尾: ⚠️  仅供参考娱乐，请理性投注！
```

## 架构快速参考

详细设计文档见 `references/STRATEGY_FUSION_DESIGN.md`，完整参数配置见 `references/dlt_skill_config.json`。

- **多池采样**：6池（热30%/冷15%/均衡20%/趋势20%/博弈10%/遗传5%）× 前后区
- **模式识别**（第7池）：10种跨期特征，评分增强 base + 匹配度×20%
- **融合引擎**：5组独立策略（加权投票/贝叶斯/博弈论/ML概率）→ 约束过滤 → 全局排序
- **后处理**：尾号聚合、AC值跟踪、偏差仪表盘校准、智能重号惩罚、置信度重校准
- **约束引擎**：唯一性、范围(1-35/1-12)、格式(5+2)、和值/AC值/跨度限制
- **复式**：12种方案，从 `dlt_skill_config.json` 读取

## 性能注意事项

- 纯 CPU 推理，首次 predict 触发神经网络训练（~120s），后续复用缓存
- 内存 <800MB 时 GA 参数自动降档（pop/gen/elite 缩放），compound 枚举 >10000 组合时剪枝保留 Top 60%，权重校准始终执行（OOM防护）
