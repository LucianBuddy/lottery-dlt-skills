"""DLT大乐透预测技能 — Python包入口 V3.0.0
所有核心模块统一从这里导出，import 后可直接使用。
"""

# ── 顶层入口 ──
from dlt_fusion_complete import DLTFusionComplete, data_dir, check_reference_sync

# ── 策略融合引擎 ──
from strategy_fusion_engine import StrategyFusionEngine
from five_pool_sampler_complete_final import MultiPoolSampler
from dlt_back_fusion import BackZoneFusion

# ── 预测器 ──
from dlt_predictor_upgraded import DLTPredictorUpgraded, load_dlt_data

# ── 约束引擎 ──
from dlt_constraint_engine import DLTConstraintEngine

# ── 数据更新 ──
from dlt_data_updater import check_and_update

# ── 数学与博弈论 ──
from modules.dlt_game_theory import DLTGameTheoryAnalyzer
from modules.dlt_genetic_optimizer import DLTGeneticOptimizer
from modules.dlt_math_filter import DLTMathFilter
from modules.dlt_statistics_analyzer import DLTStatisticsAnalyzer
from modules.dlt_kill_number import DLTKillNumber as DLTKillNumberAnalyzer
from modules.dlt_number_gravity import DLTNumberGravity
from modules.dlt_difference_sequence import DLTDifferenceSequence
from modules.dlt_matrix_displacement import DLTMatrixDisplacement
from modules.dlt_strategy_recommender import DLTStrategyRecommender

# ── 模式识别 ──
from modules.dlt_pattern_recognizer import DLTPatternRecognizer

# ── 神经网络 ──
from modules.neural_models import NeuralEnsemble, TORCH_AVAILABLE as TORCH_OK

# ── 复式投注（新集成） ──
from modules.dlt_compound_betting import (
    generate_all_compound, filter_and_score, select_diverse,
    compound_info, generate_compound_predictions, generate_all_compound_types,
)

# ── 回测与指标 ──
from lottery_metrics import LotteryMetrics
from lottery_bayesian import BayesianNumberModel
from lottery_calibration import PlattScaling
from lottery_time_series_cv import BacktestRunner

__all__ = [
    # 顶层入口
    'DLTFusionComplete', 'data_dir', 'check_reference_sync',
    # 策略融合
    'StrategyFusionEngine', 'MultiPoolSampler', 'BackZoneFusion',
    # 预测器
    'DLTPredictorUpgraded', 'load_dlt_data',
    # 约束引擎
    'DLTConstraintEngine',
    # 数据更新
    'check_and_update',
    # 数学与博弈论
    'DLTGameTheoryAnalyzer', 'DLTGeneticOptimizer', 'DLTMathFilter',
    'DLTStatisticsAnalyzer', 'DLTKillNumberAnalyzer', 'DLTNumberGravity',
    'DLTDifferenceSequence', 'DLTMatrixDisplacement', 'DLTStrategyRecommender',
    # 模式识别
    'DLTPatternRecognizer',
    # 神经网络
    'NeuralEnsemble', 'TORCH_OK',
    # 复式投注
    'generate_all_compound', 'filter_and_score', 'select_diverse',
    'compound_info', 'generate_compound_predictions', 'generate_all_compound_types',
    # 回测与指标
    'LotteryMetrics', 'BayesianNumberModel', 'PlattScaling', 'BacktestRunner',
]
