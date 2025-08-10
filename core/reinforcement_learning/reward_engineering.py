"""
强化学习奖励工程系统
设计和优化各种奖励函数，提升交易策略的学习效果
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings('ignore')

class RewardType(Enum):
    """奖励函数类型"""
    PROFIT_BASED = "profit_based"
    RISK_ADJUSTED = "risk_adjusted"
    SHARPE_OPTIMIZED = "sharpe_optimized"
    DRAWDOWN_PENALIZED = "drawdown_penalized"
    MARKET_ADAPTIVE = "market_adaptive"
    MULTI_OBJECTIVE = "multi_objective"
    BEHAVIORAL_SHAPED = "behavioral_shaped"

@dataclass
class RewardConfig:
    """奖励配置"""
    reward_type: RewardType
    weights: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # 设置默认权重
        if not self.weights:
            self.weights = self._get_default_weights()
    
    def _get_default_weights(self) -> Dict[str, float]:
        """获取默认权重"""
        default_weights = {
            RewardType.PROFIT_BASED: {
                "profit": 1.0,
                "transaction_cost": -0.001
            },
            RewardType.RISK_ADJUSTED: {
                "profit": 1.0,
                "risk_penalty": -0.1,
                "transaction_cost": -0.001
            },
            RewardType.SHARPE_OPTIMIZED: {
                "return": 1.0,
                "volatility_penalty": -0.5,
                "transaction_cost": -0.001
            },
            RewardType.DRAWDOWN_PENALIZED: {
                "profit": 1.0,
                "drawdown_penalty": -2.0,
                "transaction_cost": -0.001
            },
            RewardType.MARKET_ADAPTIVE: {
                "base_profit": 1.0,
                "regime_bonus": 0.2,
                "momentum_bonus": 0.1,
                "transaction_cost": -0.001
            },
            RewardType.MULTI_OBJECTIVE: {
                "profit": 0.4,
                "sharpe": 0.3,
                "drawdown": -0.2,
                "consistency": 0.1
            },
            RewardType.BEHAVIORAL_SHAPED: {
                "profit": 0.6,
                "exploration_bonus": 0.1,
                "stability_bonus": 0.1,
                "overtrading_penalty": -0.2
            }
        }
        
        return default_weights.get(self.reward_type, {"profit": 1.0})

class BaseRewardFunction(ABC):
    """基础奖励函数"""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 历史数据存储
        self.reward_history: List[float] = []
        self.portfolio_history: List[float] = []
        self.action_history: List[int] = []
        self.state_history: List[np.ndarray] = []
    
    @abstractmethod
    def calculate_reward(self, current_state: Dict[str, Any], 
                        previous_state: Dict[str, Any],
                        action: int, trade_executed: bool) -> float:
        """计算奖励"""
        pass
    
    def reset(self):
        """重置历史记录"""
        self.reward_history.clear()
        self.portfolio_history.clear()
        self.action_history.clear()
        self.state_history.clear()
    
    def get_reward_statistics(self) -> Dict[str, float]:
        """获取奖励统计"""
        if not self.reward_history:
            return {}
        
        return {
            "mean_reward": np.mean(self.reward_history),
            "std_reward": np.std(self.reward_history),
            "total_reward": sum(self.reward_history),
            "max_reward": max(self.reward_history),
            "min_reward": min(self.reward_history),
            "reward_volatility": np.std(self.reward_history) / abs(np.mean(self.reward_history)) if np.mean(self.reward_history) != 0 else 0
        }

class ProfitBasedReward(BaseRewardFunction):
    """基于利润的奖励函数"""
    
    def calculate_reward(self, current_state: Dict[str, Any], 
                        previous_state: Dict[str, Any],
                        action: int, trade_executed: bool) -> float:
        
        # 基础利润奖励
        profit_change = current_state["portfolio_value"] - previous_state["portfolio_value"]
        profit_reward = profit_change * self.config.weights.get("profit", 1.0)
        
        # 交易成本惩罚
        transaction_cost = 0
        if trade_executed:
            position_change = abs(current_state["position"] - previous_state["position"])
            transaction_cost = position_change * self.config.weights.get("transaction_cost", -0.001)
        
        total_reward = profit_reward + transaction_cost
        
        # 记录历史
        self.reward_history.append(total_reward)
        self.portfolio_history.append(current_state["portfolio_value"])
        self.action_history.append(action)
        
        return total_reward

class RiskAdjustedReward(BaseRewardFunction):
    """风险调整奖励函数"""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.risk_free_rate = config.parameters.get("risk_free_rate", 0.02)
        self.lookback_period = config.parameters.get("lookback_period", 50)
    
    def calculate_reward(self, current_state: Dict[str, Any], 
                        previous_state: Dict[str, Any],
                        action: int, trade_executed: bool) -> float:
        
        # 基础利润奖励
        profit_change = current_state["portfolio_value"] - previous_state["portfolio_value"]
        profit_reward = profit_change * self.config.weights.get("profit", 1.0)
        
        # 风险惩罚
        risk_penalty = self._calculate_risk_penalty(current_state)
        
        # 交易成本
        transaction_cost = 0
        if trade_executed:
            position_change = abs(current_state["position"] - previous_state["position"])
            transaction_cost = position_change * self.config.weights.get("transaction_cost", -0.001)
        
        total_reward = profit_reward + risk_penalty + transaction_cost
        
        # 记录历史
        self.reward_history.append(total_reward)
        self.portfolio_history.append(current_state["portfolio_value"])
        self.action_history.append(action)
        
        return total_reward
    
    def _calculate_risk_penalty(self, current_state: Dict[str, Any]) -> float:
        """计算风险惩罚"""
        if len(self.portfolio_history) < self.lookback_period:
            return 0
        
        # 计算收益率
        recent_values = self.portfolio_history[-self.lookback_period:]
        returns = np.diff(recent_values) / recent_values[:-1]
        
        if len(returns) == 0:
            return 0
        
        # 计算风险指标
        volatility = np.std(returns)
        var_95 = np.percentile(returns, 5)  # VaR at 95% confidence
        
        # 风险惩罚
        risk_penalty = (volatility * self.config.weights.get("volatility_penalty", -0.1) +
                       var_95 * self.config.weights.get("var_penalty", -0.05))
        
        return risk_penalty

class SharpeOptimizedReward(BaseRewardFunction):
    """夏普比率优化奖励函数"""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.risk_free_rate = config.parameters.get("risk_free_rate", 0.02)
        self.target_sharpe = config.parameters.get("target_sharpe", 1.0)
        self.rebalance_frequency = config.parameters.get("rebalance_frequency", 20)
    
    def calculate_reward(self, current_state: Dict[str, Any], 
                        previous_state: Dict[str, Any],
                        action: int, trade_executed: bool) -> float:
        
        # 基础收益
        profit_change = current_state["portfolio_value"] - previous_state["portfolio_value"]
        return_reward = profit_change * self.config.weights.get("return", 1.0)
        
        # 夏普比率调整
        sharpe_adjustment = self._calculate_sharpe_adjustment()
        
        # 波动率惩罚
        volatility_penalty = self._calculate_volatility_penalty()
        
        # 交易成本
        transaction_cost = 0
        if trade_executed:
            position_change = abs(current_state["position"] - previous_state["position"])
            transaction_cost = position_change * self.config.weights.get("transaction_cost", -0.001)
        
        total_reward = return_reward + sharpe_adjustment + volatility_penalty + transaction_cost
        
        # 记录历史
        self.reward_history.append(total_reward)
        self.portfolio_history.append(current_state["portfolio_value"])
        self.action_history.append(action)
        
        return total_reward
    
    def _calculate_sharpe_adjustment(self) -> float:
        """计算夏普比率调整"""
        if len(self.portfolio_history) < 20:
            return 0
        
        # 计算夏普比率
        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        
        excess_returns = returns - (self.risk_free_rate / 252)  # 日化
        current_sharpe = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
        
        # 夏普比率奖励/惩罚
        sharpe_bonus = (current_sharpe - self.target_sharpe) * self.config.weights.get("sharpe_bonus", 0.1)
        
        return sharpe_bonus
    
    def _calculate_volatility_penalty(self) -> float:
        """计算波动率惩罚"""
        if len(self.portfolio_history) < 10:
            return 0
        
        recent_returns = np.diff(self.portfolio_history[-20:]) / self.portfolio_history[-20:-1]
        if len(recent_returns) == 0:
            return 0
        
        volatility = np.std(recent_returns)
        penalty = volatility * self.config.weights.get("volatility_penalty", -0.5)
        
        return penalty

class DrawdownPenalizedReward(BaseRewardFunction):
    """回撤惩罚奖励函数"""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.max_drawdown_threshold = config.parameters.get("max_drawdown_threshold", 0.1)
        self.drawdown_lookback = config.parameters.get("drawdown_lookback", 100)
    
    def calculate_reward(self, current_state: Dict[str, Any], 
                        previous_state: Dict[str, Any],
                        action: int, trade_executed: bool) -> float:
        
        # 基础利润奖励
        profit_change = current_state["portfolio_value"] - previous_state["portfolio_value"]
        profit_reward = profit_change * self.config.weights.get("profit", 1.0)
        
        # 回撤惩罚
        drawdown_penalty = self._calculate_drawdown_penalty(current_state["portfolio_value"])
        
        # 交易成本
        transaction_cost = 0
        if trade_executed:
            position_change = abs(current_state["position"] - previous_state["position"])
            transaction_cost = position_change * self.config.weights.get("transaction_cost", -0.001)
        
        total_reward = profit_reward + drawdown_penalty + transaction_cost
        
        # 记录历史
        self.reward_history.append(total_reward)
        self.portfolio_history.append(current_state["portfolio_value"])
        self.action_history.append(action)
        
        return total_reward
    
    def _calculate_drawdown_penalty(self, current_value: float) -> float:
        """计算回撤惩罚"""
        if len(self.portfolio_history) < 2:
            return 0
        
        # 计算历史最高点
        lookback_values = self.portfolio_history[-self.drawdown_lookback:]
        peak_value = max(lookback_values)
        
        # 计算当前回撤
        current_drawdown = (peak_value - current_value) / peak_value
        
        # 回撤惩罚
        if current_drawdown > self.max_drawdown_threshold:
            penalty = (current_drawdown - self.max_drawdown_threshold) ** 2
            penalty *= self.config.weights.get("drawdown_penalty", -2.0)
        else:
            penalty = 0
        
        return penalty

class MarketAdaptiveReward(BaseRewardFunction):
    """市场自适应奖励函数"""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.trend_window = config.parameters.get("trend_window", 20)
        self.volatility_window = config.parameters.get("volatility_window", 10)
    
    def calculate_reward(self, current_state: Dict[str, Any], 
                        previous_state: Dict[str, Any],
                        action: int, trade_executed: bool) -> float:
        
        # 基础利润奖励
        profit_change = current_state["portfolio_value"] - previous_state["portfolio_value"]
        base_reward = profit_change * self.config.weights.get("base_profit", 1.0)
        
        # 市场状态适应性奖励
        regime_bonus = self._calculate_regime_bonus(current_state, action)
        momentum_bonus = self._calculate_momentum_bonus(current_state, action)
        
        # 交易成本
        transaction_cost = 0
        if trade_executed:
            position_change = abs(current_state["position"] - previous_state["position"])
            transaction_cost = position_change * self.config.weights.get("transaction_cost", -0.001)
        
        total_reward = base_reward + regime_bonus + momentum_bonus + transaction_cost
        
        # 记录历史
        self.reward_history.append(total_reward)
        self.portfolio_history.append(current_state["portfolio_value"])
        self.action_history.append(action)
        
        return total_reward
    
    def _calculate_regime_bonus(self, current_state: Dict[str, Any], action: int) -> float:
        """计算市场状态适应奖励"""
        market_regime = current_state.get("market_regime", 0)  # 0: 上涨, 1: 下跌, 2: 横盘, 3: 高波动
        
        regime_bonus = 0
        bonus_weight = self.config.weights.get("regime_bonus", 0.2)
        
        # 根据市场状态给予奖励
        if market_regime == 0 and action == 0:  # 上涨趋势中买入
            regime_bonus = bonus_weight
        elif market_regime == 1 and action == 1:  # 下跌趋势中卖出
            regime_bonus = bonus_weight
        elif market_regime == 2 and action == 2:  # 横盘市场中持有
            regime_bonus = bonus_weight * 0.5
        elif market_regime == 3 and action == 2:  # 高波动市场中谨慎持有
            regime_bonus = bonus_weight * 0.3
        
        return regime_bonus
    
    def _calculate_momentum_bonus(self, current_state: Dict[str, Any], action: int) -> float:
        """计算动量适应奖励"""
        if len(self.portfolio_history) < 5:
            return 0
        
        # 计算短期动量
        recent_changes = np.diff(self.portfolio_history[-5:])
        momentum = np.mean(recent_changes)
        
        momentum_bonus = 0
        bonus_weight = self.config.weights.get("momentum_bonus", 0.1)
        
        # 顺势交易奖励
        if momentum > 0 and action == 0:  # 上升动量中买入
            momentum_bonus = bonus_weight
        elif momentum < 0 and action == 1:  # 下降动量中卖出
            momentum_bonus = bonus_weight
        
        return momentum_bonus

class MultiObjectiveReward(BaseRewardFunction):
    """多目标奖励函数"""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.objectives = {
            "profit": self._calculate_profit_score,
            "sharpe": self._calculate_sharpe_score,
            "drawdown": self._calculate_drawdown_score,
            "consistency": self._calculate_consistency_score
        }
    
    def calculate_reward(self, current_state: Dict[str, Any], 
                        previous_state: Dict[str, Any],
                        action: int, trade_executed: bool) -> float:
        
        # 记录状态
        self.portfolio_history.append(current_state["portfolio_value"])
        self.action_history.append(action)
        
        # 计算各个目标的得分
        total_reward = 0
        for objective, weight in self.config.weights.items():
            if objective in self.objectives:
                score = self.objectives[objective](current_state, previous_state)
                total_reward += weight * score
        
        # 交易成本
        if trade_executed:
            position_change = abs(current_state["position"] - previous_state["position"])
            transaction_cost = position_change * 0.001
            total_reward -= transaction_cost
        
        self.reward_history.append(total_reward)
        return total_reward
    
    def _calculate_profit_score(self, current_state: Dict[str, Any], 
                               previous_state: Dict[str, Any]) -> float:
        """利润得分"""
        return current_state["portfolio_value"] - previous_state["portfolio_value"]
    
    def _calculate_sharpe_score(self, current_state: Dict[str, Any], 
                               previous_state: Dict[str, Any]) -> float:
        """夏普比率得分"""
        if len(self.portfolio_history) < 20:
            return 0
        
        returns = np.diff(self.portfolio_history[-20:]) / self.portfolio_history[-20:-1]
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        return sharpe
    
    def _calculate_drawdown_score(self, current_state: Dict[str, Any], 
                                 previous_state: Dict[str, Any]) -> float:
        """回撤得分 (负值，因为要最小化回撤)"""
        if len(self.portfolio_history) < 2:
            return 0
        
        peak_value = max(self.portfolio_history)
        current_drawdown = (peak_value - current_state["portfolio_value"]) / peak_value
        return -current_drawdown
    
    def _calculate_consistency_score(self, current_state: Dict[str, Any], 
                                    previous_state: Dict[str, Any]) -> float:
        """一致性得分"""
        if len(self.reward_history) < 10:
            return 0
        
        recent_rewards = self.reward_history[-10:]
        consistency = -np.std(recent_rewards)  # 负标准差，鼓励稳定性
        return consistency

class BehavioralShapedReward(BaseRewardFunction):
    """行为塑造奖励函数"""
    
    def __init__(self, config: RewardConfig):
        super().__init__(config)
        self.exploration_bonus_decay = config.parameters.get("exploration_bonus_decay", 0.995)
        self.current_exploration_bonus = config.parameters.get("initial_exploration_bonus", 0.1)
        self.overtrading_threshold = config.parameters.get("overtrading_threshold", 10)
        
    def calculate_reward(self, current_state: Dict[str, Any], 
                        previous_state: Dict[str, Any],
                        action: int, trade_executed: bool) -> float:
        
        # 基础利润奖励
        profit_change = current_state["portfolio_value"] - previous_state["portfolio_value"]
        profit_reward = profit_change * self.config.weights.get("profit", 0.6)
        
        # 探索奖励 (逐渐衰减)
        exploration_bonus = self._calculate_exploration_bonus(action)
        
        # 稳定性奖励
        stability_bonus = self._calculate_stability_bonus()
        
        # 过度交易惩罚
        overtrading_penalty = self._calculate_overtrading_penalty()
        
        total_reward = (profit_reward + exploration_bonus + 
                       stability_bonus + overtrading_penalty)
        
        # 记录历史
        self.reward_history.append(total_reward)
        self.portfolio_history.append(current_state["portfolio_value"])
        self.action_history.append(action)
        
        # 衰减探索奖励
        self.current_exploration_bonus *= self.exploration_bonus_decay
        
        return total_reward
    
    def _calculate_exploration_bonus(self, action: int) -> float:
        """探索奖励"""
        # 鼓励在早期阶段尝试不同动作
        if len(self.action_history) < 100:
            # 计算动作多样性
            if len(set(self.action_history[-10:])) > 1:
                return self.current_exploration_bonus * self.config.weights.get("exploration_bonus", 0.1)
        return 0
    
    def _calculate_stability_bonus(self) -> float:
        """稳定性奖励"""
        if len(self.reward_history) < 10:
            return 0
        
        recent_rewards = self.reward_history[-10:]
        stability = 1 / (1 + np.std(recent_rewards))  # 稳定性得分
        return stability * self.config.weights.get("stability_bonus", 0.1)
    
    def _calculate_overtrading_penalty(self) -> float:
        """过度交易惩罚"""
        if len(self.action_history) < self.overtrading_threshold:
            return 0
        
        recent_actions = self.action_history[-self.overtrading_threshold:]
        
        # 计算交易频率 (非持有动作的比例)
        trading_actions = [a for a in recent_actions if a != 2]  # 2是持有动作
        trading_frequency = len(trading_actions) / len(recent_actions)
        
        # 过度交易惩罚
        if trading_frequency > 0.7:  # 如果70%以上时间在交易
            penalty = (trading_frequency - 0.7) ** 2
            return penalty * self.config.weights.get("overtrading_penalty", -0.2)
        
        return 0

class RewardFunctionFactory:
    """奖励函数工厂"""
    
    _reward_functions = {
        RewardType.PROFIT_BASED: ProfitBasedReward,
        RewardType.RISK_ADJUSTED: RiskAdjustedReward,
        RewardType.SHARPE_OPTIMIZED: SharpeOptimizedReward,
        RewardType.DRAWDOWN_PENALIZED: DrawdownPenalizedReward,
        RewardType.MARKET_ADAPTIVE: MarketAdaptiveReward,
        RewardType.MULTI_OBJECTIVE: MultiObjectiveReward,
        RewardType.BEHAVIORAL_SHAPED: BehavioralShapedReward
    }
    
    @classmethod
    def create_reward_function(cls, config: RewardConfig) -> BaseRewardFunction:
        """创建奖励函数"""
        reward_class = cls._reward_functions.get(config.reward_type)
        if not reward_class:
            raise ValueError(f"不支持的奖励类型: {config.reward_type}")
        
        return reward_class(config)
    
    @classmethod
    def list_available_types(cls) -> List[RewardType]:
        """列出可用的奖励类型"""
        return list(cls._reward_functions.keys())

class RewardOptimizer:
    """奖励函数优化器"""
    
    def __init__(self):
        self.logger = logging.getLogger("RewardOptimizer")
        self.optimization_history = []
    
    def optimize_weights(self, reward_type: RewardType, 
                        historical_data: Dict[str, np.ndarray],
                        optimization_target: str = "sharpe_ratio",
                        n_iterations: int = 100) -> Dict[str, float]:
        """优化奖励函数权重"""
        
        from scipy.optimize import minimize
        
        # 初始权重
        config = RewardConfig(reward_type)
        initial_weights = list(config.weights.values())
        weight_keys = list(config.weights.keys())
        
        def objective_function(weights):
            # 创建新的配置
            new_weights = dict(zip(weight_keys, weights))
            new_config = RewardConfig(reward_type, weights=new_weights)
            
            # 模拟奖励函数性能
            performance = self._simulate_performance(new_config, historical_data)
            
            # 返回优化目标的负值 (因为minimize要最小化)
            return -performance.get(optimization_target, 0)
        
        # 权重约束 (所有权重之和为1，且非负)
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1},  # 权重和为1
        ]
        bounds = [(-2, 2) for _ in initial_weights]  # 权重范围
        
        # 优化
        result = minimize(
            objective_function,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': n_iterations}
        )
        
        # 返回优化后的权重
        optimized_weights = dict(zip(weight_keys, result.x))
        
        self.logger.info(f"权重优化完成: {optimized_weights}")
        return optimized_weights
    
    def _simulate_performance(self, config: RewardConfig, 
                             historical_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """模拟奖励函数性能"""
        
        reward_function = RewardFunctionFactory.create_reward_function(config)
        
        # 模拟交易过程
        prices = historical_data.get("prices", np.random.randn(1000))
        portfolio_values = [100000]  # 初始资金
        rewards = []
        
        for i in range(1, len(prices)):
            # 模拟状态
            current_state = {
                "portfolio_value": portfolio_values[-1] * (1 + (prices[i] - prices[i-1]) / prices[i-1] * 0.1),
                "position": 0.5,
                "market_regime": i % 4
            }
            
            previous_state = {
                "portfolio_value": portfolio_values[-1],
                "position": 0.5
            }
            
            # 随机动作
            action = np.random.choice([0, 1, 2])
            trade_executed = action != 2
            
            # 计算奖励
            reward = reward_function.calculate_reward(
                current_state, previous_state, action, trade_executed
            )
            
            rewards.append(reward)
            portfolio_values.append(current_state["portfolio_value"])
        
        # 计算性能指标
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        performance = {
            "total_return": (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0],
            "sharpe_ratio": np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
            "max_drawdown": self._calculate_max_drawdown(portfolio_values),
            "reward_consistency": -np.std(rewards) if rewards else 0
        }
        
        return performance
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """计算最大回撤"""
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd

# 全局奖励工程管理器
class GlobalRewardManager:
    """全局奖励管理器"""
    
    def __init__(self):
        self.registered_configs: Dict[str, RewardConfig] = {}
        self.performance_history: Dict[str, List[Dict[str, float]]] = {}
        self.logger = logging.getLogger("GlobalRewardManager")
    
    def register_reward_config(self, name: str, config: RewardConfig):
        """注册奖励配置"""
        self.registered_configs[name] = config
        self.performance_history[name] = []
        self.logger.info(f"注册奖励配置: {name}")
    
    def get_reward_function(self, name: str) -> Optional[BaseRewardFunction]:
        """获取奖励函数"""
        config = self.registered_configs.get(name)
        if not config:
            return None
        
        return RewardFunctionFactory.create_reward_function(config)
    
    def compare_reward_functions(self, names: List[str], 
                                historical_data: Dict[str, np.ndarray]) -> pd.DataFrame:
        """比较奖励函数"""
        results = []
        
        optimizer = RewardOptimizer()
        
        for name in names:
            if name not in self.registered_configs:
                continue
            
            config = self.registered_configs[name]
            performance = optimizer._simulate_performance(config, historical_data)
            
            result = {
                "name": name,
                "reward_type": config.reward_type.value,
                **performance
            }
            results.append(result)
        
        return pd.DataFrame(results)

# 全局实例
reward_manager = GlobalRewardManager()