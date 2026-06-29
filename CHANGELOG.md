# CHANGELOG

## 3.10.0 (2026-06-26)

### 新增功能
- **β - 号码排除模型**：`predict_exclusion()` 反转GBR输出为排除概率(1 - score/max_score)，`_apply_exclusion_filter()` 对排除概率>0.7的候选降分15%

## 3.9.0 (2026-06-26)

### 新增功能
- **Y - 选择性预测**：`_should_skip_prediction()` 当Bootstrap不确定性均值>0.5时替换输出为热号参考(近30期Top5)
- **X - 祖先采样**：`_generate_ancestral_candidates()` 从跨期MLP的35维概率分布中重复伯努利试验生成候选，注入预测池
- **Z - 决策可解释性**：`_explain_top1()` 为Top1候选生成自然语言解释(重号/冷号/区间/来源标签)

## 3.8.0 (2026-06-26)

### 新增功能
- **限制1 - 号码间依赖建模**：`_apply_pairwise_penalty()` 基于35×35共现矩阵，共现率<15%的号码对降分最高15%
- **限制2 - 跨期MLP模式迁移**：numpy实现2层网络(35→64→35)，SGD训练50epoch，学习号码模式整体迁移，15%权重融入final_score
- **限制3 - 成本感知加权标签**：`compute_weighted_label()` 按奖项等级加权（一等5.0→无奖0.3），ranking model训练时使用

## 3.7.0 (2026-06-26)

### 新增功能
- **缺口A - 参数配置外化**：`ConfigLoader` 类 + `references/predictor_config.json`，所有魔法数字提取到JSON，支持热加载(hot_reload)
- **缺口D - 异常检测响应回路**：`_respond_to_anomalies()` 根据异常率自动调整(5-15%降延续性/≥15%冻结跨期特征)
- **缺口B - 预测不确定性量化**：`_estimate_uncertainty()` Bootstrap扰动估计候选评分置信区间，输出std/volatility/ci_95

## 3.6.0 (2026-06-26)

### 新增功能
- **缺口1 - 多模型集成投票**：`_ensemble_vote()` 四模型加权投票(GBR+决策树+时序LR+频率基线)，权重按最近MRR动态分配，25%权重融合final_score
- **缺口3 - 预测发散度控制**：`_diverse_topk_selection()` 确定性退火选择，确保Top5两两Jaccard距离≤0.5，逐步降低阈值补满

### 优化改进
- **缺口2 - 动态特征衰减**：`FeatureDecayer` 类追踪66维特征的Top10/Bottom10区分度，低效特征自动休眠(置零)，滑动窗口20期

## 3.5.0 (2026-06-26)

### 新增功能
- **方向3 - 逆强化反馈学习**：`post_draw_analysis()` 开奖后反事实分析(遗憾排名/偏差归因/Top5命中详情)，`_capture_candidate_snapshot()` 保存候选池快照
- **方向1 - 号码时序预测**：`_generate_ts_candidates()` 35维时间序列+LogisticRegression 35-output，Top16号码枚举C(16,5)=4368候选，选Top20注入候选池
- **方向5 - 训练-推理分离**：`train_model()` 独立训练接口，自动加载缓存/快照/轻量训练三路fallback，模型版本管理保留最近3个快照

## 3.4.0 (2026-06-26)

### 新增功能
- **方向C - 在线学习闭环**：`_online_update_if_needed()` 检测新开奖数据并增量训练ranking model，滑动窗口2000期，新模型验证失败时自动回滚旧版本
- **方向D - 复式预算优化器**：`optimize_for_budget(max_budget)` 根据预算(元)自动选最优复式方案，按性价比(覆盖率/元)排序

### 优化改进
- **方向A - 候选质量门禁**：`_filter_low_quality_candidates()` 评分前过滤低质量候选，规则：延续性≤1/和值[60,150]/三区≥2/后区多样性，候选数从60降至25-30
- **方向E - 数据异常检测**：`_detect_anomalies()` 检测和值Z-score>3σ/号码分布卡方偏斜/重复期
- **方向B - 特征工程增强**：特征维度55→66，新增12个交叉特征(频率×遗漏/和值×跨度/奇偶×012路等)

## 3.3.0 (2026-06-26)

### 新增功能
- **P1 - 可学习排序模型**：新增 `modules/ranking_feature_extractor.py`(55维特征向量) + `modules/ranking_model.py`(sklearn GBR)，首次predict时用最近30期回放自动训练，替代手动调参pipeline
- **P3 - 决策树排序评分**：`_apply_decision_tree_scoring()` 基于7维跨期特征（和值/跨度/奇偶/AC/三区分布）训练DecisionTreeRegressor预测号码概率，15%权重融入final_score
- **P4 - 条件计算图**：`_build_scoring_graph()` + `_execute_scoring_graph()` 构建11节点DAG引擎，按拓扑序独立try/except执行
- **P7 - 后区条件概率分配**：`_deduplicate_and_assign_back()` 重构为基于 `P(back|front_max_bucket)` 条件概率矩阵分配后区
- **P10 - 增强回测指标**：`backtest()` 输出新增 MRR/NDCG@5/ECE/热号基线
- **方案5 - 三基线对比**：backtest增加随机/热号/模式三基线对比 + `model_beats_all_%` 指标

### 优化改进
- **P2 - 评分标准化**：`_compute_final_scores()` 改为百分位排名加权平均后归一化到[0.5,1.0]
- **P5 - 神经降级路径**：Step 0.3/7c2.7 三档内存调度（full/freq/skip）
- **P6 - 约束软化**：策略验证从 `pass_cnt >= 2` 硬过滤改为评分因子（0-2→降权，≥2→加分）
- **P8 - 线性趋势和值预测**：LinearRegression 预测下一期和值，残差>8时调整候选
- **方案4 - 分级推理策略表**：四级统一调度(≥800/≥500/≥300/<300MB)，中央分配GA/神经/复式/候选资源

## 3.2.0 (2026-06-25)

### 优化改进
- [方向A] 冷热号动态分位数阈值(均值±0.5std替代固定Top-N)
- [方向A] 极冷号强制注入(遗漏>30期的号码替换冷号池末位)
- [方向C] 后区全枚举(C12,2=66组合)+K-Medoids覆盖优化
- [方向C] get_back_recommendations()重写为全枚举路径
