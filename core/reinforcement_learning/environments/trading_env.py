"""
强化学习交易环境
提供标准化的交易环境接口，支持多种交易场景和奖励机制
"""

import gym
import numpy as np
import pandas as pd
from gym import spaces
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ActionType(Enum):
    """动作类型"""
    BUY = 0
    SELL = 1
    HOLD = 2

class MarketRegime(Enum):
    """市场状态"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"

@dataclass
class TradingState:
    """交易状态"""
    # 价格信息
    current_price: float = 0.0
    price_change: float = 0.0
    price_volatility: float = 0.0
    
    # 技术指标
    rsi: float = 50.0
    macd: float = 0.0
    bb_position: float = 0.5  # 布林带位置 (0-1)
    
    # 持仓信息
    position: float = 0.0  # -1 (空仓) 到 +1 (满仓)
    cash: float = 1.0
    portfolio_value: float = 1.0
    unrealized_pnl: float = 0.0
    
    # 市场信息
    volume_ratio: float = 1.0
    market_regime: int = 0  # MarketRegime的索引
    
    # 风险指标
    drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """转换为numpy数组"""
        return np.array([
            self.current_price,
            self.price_change,
            self.price_volatility,
            self.rsi / 100.0,  # 归一化到0-1
            self.macd,
            self.bb_position,
            self.position,
            self.cash,
            self.portfolio_value,
            self.unrealized_pnl,
            self.volume_ratio,
            self.market_regime / 3.0,  # 归一化到0-1
            self.drawdown,
            self.sharpe_ratio
        ], dtype=np.float32)

class TradingReward:
    """奖励函数计算器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 奖励权重
        self.profit_weight = self.config.get("profit_weight", 1.0)
        self.risk_penalty = self.config.get("risk_penalty", 0.1)
        self.transaction_cost = self.config.get("transaction_cost", 0.001)
        self.holding_penalty = self.config.get("holding_penalty", 0.0001)
        self.sharpe_bonus = self.config.get("sharpe_bonus", 0.1)
        
    def calculate_reward(
        self, 
        action: int, 
        prev_state: TradingState, 
        current_state: TradingState,
        trade_executed: bool = False
    ) -> float:
        """计算奖励"""
        reward = 0.0
        
        # 1. 利润奖励
        profit_reward = (current_state.portfolio_value - prev_state.portfolio_value) * self.profit_weight
        reward += profit_reward
        
        # 2. 交易成本惩罚
        if trade_executed:
            reward -= self.transaction_cost * abs(current_state.position - prev_state.position)
        
        # 3. 风险惩罚
        risk_penalty = 0.0
        
        # 回撤惩罚
        if current_state.drawdown > prev_state.drawdown:
            risk_penalty += (current_state.drawdown - prev_state.drawdown) * self.risk_penalty
        
        # 持仓时间惩罚（鼓励活跃交易）
        if abs(current_state.position) > 0.1:
            risk_penalty += self.holding_penalty
        
        reward -= risk_penalty
        
        # 4. 夏普比率奖励
        if current_state.sharpe_ratio > prev_state.sharpe_ratio:
            reward += (current_state.sharpe_ratio - prev_state.sharpe_ratio) * self.sharpe_bonus
        
        # 5. 市场适应性奖励
        market_reward = self._calculate_market_adaptation_reward(action, current_state)
        reward += market_reward
        
        return reward
    
    def _calculate_market_adaptation_reward(self, action: int, state: TradingState) -> float:
        """市场适应性奖励"""
        regime_reward = 0.0
        
        # 根据市场状态给出不同的奖励
        if state.market_regime == MarketRegime.TRENDING_UP.value:
            if action == ActionType.BUY.value and state.position > 0:
                regime_reward = 0.01  # 上涨趋势中做多
        elif state.market_regime == MarketRegime.TRENDING_DOWN.value:
            if action == ActionType.SELL.value and state.position < 0:
                regime_reward = 0.01  # 下跌趋势中做空
        elif state.market_regime == MarketRegime.SIDEWAYS.value:
            if action == ActionType.HOLD.value:
                regime_reward = 0.005  # 震荡市中持有
        
        return regime_reward

class TechnicalIndicators:
    """技术指标计算"""
    
    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> float:
        """计算RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.mean(gains[-period:])
        avg_losses = np.mean(losses[-period:])
        
        if avg_losses == 0:
            return 100.0
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26) -> float:
        """计算MACD"""
        if len(prices) < slow:
            return 0.0
        
        ema_fast = TechnicalIndicators._calculate_ema(prices, fast)
        ema_slow = TechnicalIndicators._calculate_ema(prices, slow)
        
        macd = ema_fast - ema_slow
        return macd / prices[-1] if prices[-1] != 0 else 0  # 归一化
    
    @staticmethod
    def calculate_bollinger_position(prices: np.ndarray, period: int = 20, std_dev: int = 2) -> float:
        """计算布林带位置 (0-1)"""
        if len(prices) < period:
            return 0.5
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        current_price = prices[-1]
        
        if upper_band == lower_band:
            return 0.5
        
        position = (current_price - lower_band) / (upper_band - lower_band)
        return np.clip(position, 0, 1)
    
    @staticmethod
    def _calculate_ema(prices: np.ndarray, period: int) -> float:
        """计算EMA"""
        if len(prices) < period:
            return np.mean(prices)
        
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    @staticmethod
    def detect_market_regime(prices: np.ndarray, volumes: np.ndarray = None) -> MarketRegime:
        """检测市场状态"""
        if len(prices) < 20:
            return MarketRegime.SIDEWAYS
        
        # 计算价格趋势
        short_sma = np.mean(prices[-10:])
        long_sma = np.mean(prices[-20:])
        trend_strength = abs(short_sma - long_sma) / long_sma
        
        # 计算波动率
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns[-20:])
        
        # 判断市场状态
        if volatility > 0.03:  # 高波动率
            return MarketRegime.VOLATILE
        elif trend_strength > 0.02:  # 有明显趋势
            if short_sma > long_sma:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.SIDEWAYS

class TradingEnvironment(gym.Env):
    """强化学习交易环境"""
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        
        self.config = config or {}
        self.logger = logging.getLogger("TradingEnvironment")
        
        # 环境参数
        self.max_steps = self.config.get("max_steps", 1000)
        self.initial_balance = self.config.get("initial_balance", 100000.0)
        self.max_position = self.config.get("max_position", 1.0)
        self.price_history_length = self.config.get("price_history_length", 100)
        
        # 数据相关
        self.price_data: Optional[np.ndarray] = None
        self.volume_data: Optional[np.ndarray] = None
        self.current_step = 0
        
        # 交易状态
        self.balance = self.initial_balance
        self.position = 0.0  # -1 (空仓) 到 +1 (满仓)
        self.portfolio_value = self.initial_balance
        self.entry_price = 0.0
        self.max_portfolio_value = self.initial_balance
        
        # 历史记录
        self.portfolio_history: List[float] = []
        self.action_history: List[int] = []
        self.trade_history: List[Dict] = []
        
        # 奖励计算器
        self.reward_calculator = TradingReward(config.get("reward_config", {}))
        
        # 定义动作空间和状态空间
        # 动作空间：0=买入, 1=卖出, 2=持有
        self.action_space = spaces.Discrete(3)
        
        # 状态空间：14维特征
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14,),
            dtype=np.float32
        )
        
    def load_data(self, price_data: np.ndarray, volume_data: Optional[np.ndarray] = None):
        """加载交易数据"""
        self.price_data = price_data
        self.volume_data = volume_data if volume_data is not None else np.ones_like(price_data)
        
        if len(self.price_data) < self.price_history_length:
            raise ValueError(f"价格数据长度 ({len(self.price_data)}) 小于所需历史长度 ({self.price_history_length})")
        
        self.logger.info(f"加载数据: {len(self.price_data)} 个价格点")
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        # 重置交易状态
        self.current_step = self.price_history_length
        self.balance = self.initial_balance
        self.position = 0.0
        self.portfolio_value = self.initial_balance
        self.entry_price = 0.0
        self.max_portfolio_value = self.initial_balance
        
        # 清空历史记录
        self.portfolio_history = [self.initial_balance]
        self.action_history = []
        self.trade_history = []
        
        # 返回初始状态
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步动作"""
        if self.price_data is None:
            raise ValueError("请先加载数据")
        
        # 保存前一个状态
        prev_state = self._get_current_state()
        
        # 执行动作
        trade_executed = self._execute_action(action)
        
        # 更新状态
        self.current_step += 1
        current_state = self._get_current_state()
        
        # 计算奖励
        reward = self.reward_calculator.calculate_reward(
            action, prev_state, current_state, trade_executed
        )
        
        # 检查是否完成
        done = self._is_done()
        
        # 记录历史
        self.portfolio_history.append(self.portfolio_value)
        self.action_history.append(action)
        
        # 准备info字典
        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "balance": self.balance,
            "current_price": self._get_current_price(),
            "trade_executed": trade_executed,
            "step": self.current_step
        }
        
        return self._get_observation(), reward, done, info
    
    def _execute_action(self, action: int) -> bool:
        """执行交易动作"""
        current_price = self._get_current_price()
        trade_executed = False
        
        if action == ActionType.BUY.value:
            # 买入逻辑
            if self.position < self.max_position:
                # 计算买入数量
                available_cash = self.balance * 0.95  # 保留5%现金
                max_buy_amount = (self.max_position - self.position) * self.portfolio_value
                buy_amount = min(available_cash, max_buy_amount)
                
                if buy_amount > 0:
                    buy_quantity = buy_amount / current_price
                    position_change = buy_quantity * current_price / self.portfolio_value
                    
                    self.position = min(self.max_position, self.position + position_change)
                    self.balance -= buy_amount
                    self.entry_price = current_price
                    
                    # 记录交易
                    self.trade_history.append({
                        "step": self.current_step,
                        "action": "BUY",
                        "price": current_price,
                        "quantity": buy_quantity,
                        "amount": buy_amount
                    })
                    
                    trade_executed = True
        
        elif action == ActionType.SELL.value:
            # 卖出逻辑
            if self.position > -self.max_position:
                # 计算卖出数量
                current_position_value = abs(self.position) * self.portfolio_value
                max_sell_amount = current_position_value
                
                if max_sell_amount > 0:
                    sell_quantity = max_sell_amount / current_price
                    position_change = sell_quantity * current_price / self.portfolio_value
                    
                    if self.position > 0:
                        # 平多仓
                        position_change = min(position_change, self.position)
                        self.position -= position_change
                        proceeds = position_change * self.portfolio_value
                    else:
                        # 开空仓
                        self.position = max(-self.max_position, self.position - position_change)
                        proceeds = position_change * self.portfolio_value
                    
                    self.balance += proceeds
                    self.entry_price = current_price
                    
                    # 记录交易
                    self.trade_history.append({
                        "step": self.current_step,
                        "action": "SELL",
                        "price": current_price,
                        "quantity": sell_quantity,
                        "amount": proceeds
                    })
                    
                    trade_executed = True
        
        # 更新投资组合价值
        self._update_portfolio_value()
        
        return trade_executed
    
    def _update_portfolio_value(self):
        """更新投资组合价值"""
        current_price = self._get_current_price()
        
        # 计算持仓价值
        position_value = self.position * self.portfolio_value
        
        if self.position > 0:
            # 做多仓位
            if hasattr(self, 'entry_price') and self.entry_price > 0:
                pnl = position_value * (current_price / self.entry_price - 1)
            else:
                pnl = 0
        elif self.position < 0:
            # 做空仓位
            if hasattr(self, 'entry_price') and self.entry_price > 0:
                pnl = abs(position_value) * (self.entry_price / current_price - 1)
            else:
                pnl = 0
        else:
            pnl = 0
        
        self.portfolio_value = self.balance + abs(position_value) + pnl
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
    
    def _get_current_price(self) -> float:
        """获取当前价格"""
        if self.price_data is None or self.current_step >= len(self.price_data):
            return 0.0
        return self.price_data[self.current_step]
    
    def _get_price_history(self, length: int = None) -> np.ndarray:
        """获取价格历史"""
        if length is None:
            length = self.price_history_length
        
        start_idx = max(0, self.current_step - length)
        end_idx = self.current_step + 1
        
        return self.price_data[start_idx:end_idx]
    
    def _get_current_state(self) -> TradingState:
        """获取当前交易状态"""
        current_price = self._get_current_price()
        price_history = self._get_price_history()
        
        state = TradingState()
        
        # 价格信息
        state.current_price = current_price / price_history[0] if price_history[0] != 0 else 1.0  # 归一化
        
        if len(price_history) > 1:
            state.price_change = (price_history[-1] - price_history[-2]) / price_history[-2]
            returns = np.diff(price_history) / price_history[:-1]
            state.price_volatility = np.std(returns) if len(returns) > 1 else 0.0
        
        # 技术指标
        state.rsi = TechnicalIndicators.calculate_rsi(price_history)
        state.macd = TechnicalIndicators.calculate_macd(price_history)
        state.bb_position = TechnicalIndicators.calculate_bollinger_position(price_history)
        
        # 持仓信息
        state.position = self.position
        state.cash = self.balance / self.initial_balance  # 归一化
        state.portfolio_value = self.portfolio_value / self.initial_balance  # 归一化
        
        # 计算未实现盈亏
        if abs(self.position) > 0 and hasattr(self, 'entry_price') and self.entry_price > 0:
            if self.position > 0:
                state.unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            else:
                state.unrealized_pnl = (self.entry_price - current_price) / self.entry_price
        
        # 市场信息
        volume_history = self.volume_data[max(0, self.current_step-10):self.current_step+1]
        if len(volume_history) > 1:
            state.volume_ratio = volume_history[-1] / np.mean(volume_history[:-1])
        
        market_regime = TechnicalIndicators.detect_market_regime(price_history)
        state.market_regime = list(MarketRegime).index(market_regime)
        
        # 风险指标
        state.drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        
        # 夏普比率（简化版本）
        if len(self.portfolio_history) > 10:
            returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
            if np.std(returns) > 0:
                state.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        return state
    
    def _get_observation(self) -> np.ndarray:
        """获取观察状态"""
        state = self._get_current_state()
        return state.to_array()
    
    def _is_done(self) -> bool:
        """判断是否结束"""
        # 达到最大步数
        if self.current_step >= min(len(self.price_data) - 1, self.max_steps):
            return True
        
        # 资产损失过大
        if self.portfolio_value <= self.initial_balance * 0.1:  # 损失90%
            return True
        
        return False
    
    def render(self, mode='human'):
        """渲染环境"""
        if mode == 'human':
            current_price = self._get_current_price()
            print(f"Step: {self.current_step}")
            print(f"Price: {current_price:.2f}")
            print(f"Position: {self.position:.2f}")
            print(f"Portfolio Value: {self.portfolio_value:.2f}")
            print(f"Balance: {self.balance:.2f}")
            print("-" * 30)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        if len(self.portfolio_history) < 2:
            return {}
        
        portfolio_returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        
        total_return = (self.portfolio_value - self.initial_balance) / self.initial_balance
        max_drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        
        # 夏普比率
        sharpe_ratio = 0.0
        if np.std(portfolio_returns) > 0:
            sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)
        
        # 最大回撤
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_dd_historical = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # 胜率
        profitable_trades = sum(1 for trade in self.trade_history if trade.get("pnl", 0) > 0)
        total_trades = len(self.trade_history)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_dd_historical,
            "current_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "final_portfolio_value": self.portfolio_value
        }

# 创建环境工厂函数
def create_trading_environment(config: Dict[str, Any] = None) -> TradingEnvironment:
    """创建交易环境"""
    return TradingEnvironment(config)