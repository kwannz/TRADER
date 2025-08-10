"""
策略Agent - 智能交易策略生成和优化
负责策略创建、回测、优化和实时策略调整
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

from .base_agent import BaseAgent, AgentType, AgentCapability, AgentMessage, MessageType

# 策略类型枚举
class StrategyType(Enum):
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    PAIRS_TRADING = "pairs_trading"
    ML_PREDICTION = "ml_prediction"
    MULTI_FACTOR = "multi_factor"

# 信号强度枚举
class SignalStrength(Enum):
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5

@dataclass
class TradingSignal:
    """交易信号"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    strength: SignalStrength
    confidence: float  # 0-1
    price_target: float
    stop_loss: float
    take_profit: float
    position_size: float
    reasoning: str
    strategy_name: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "price_target": self.price_target,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_size": self.position_size,
            "reasoning": self.reasoning,
            "strategy_name": self.strategy_name,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

@dataclass
class StrategyPerformance:
    """策略性能指标"""
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def update_metrics(self, returns: List[float]):
        """更新性能指标"""
        if not returns:
            return
        
        returns_array = np.array(returns)
        self.total_return = np.sum(returns_array)
        
        # 夏普比率
        if len(returns) > 1:
            self.sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
        
        # 最大回撤
        cumulative_returns = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        self.max_drawdown = np.min(drawdowns)
        
        # 胜率相关指标
        wins = returns_array[returns_array > 0]
        losses = returns_array[returns_array < 0]
        
        self.winning_trades = len(wins)
        self.losing_trades = len(losses)
        self.total_trades = len(returns_array)
        
        if self.total_trades > 0:
            self.win_rate = self.winning_trades / self.total_trades
        
        if len(wins) > 0:
            self.avg_win = np.mean(wins)
        
        if len(losses) > 0:
            self.avg_loss = np.mean(losses)
        
        # 盈亏比
        if self.avg_loss != 0:
            self.profit_factor = abs(self.avg_win / self.avg_loss)
        
        # Calmar比率
        if self.max_drawdown != 0:
            annual_return = self.total_return * (252 / len(returns))
            self.calmar_ratio = annual_return / abs(self.max_drawdown)
        
        self.last_updated = datetime.utcnow()

class BaseStrategy:
    """策略基类"""
    
    def __init__(self, name: str, strategy_type: StrategyType, config: Dict[str, Any] = None):
        self.name = name
        self.strategy_type = strategy_type
        self.config = config or {}
        self.performance = StrategyPerformance(name)
        self.is_active = True
        self.logger = logging.getLogger(f"Strategy-{name}")
        
    async def generate_signal(self, market_data: Dict[str, Any]) -> Optional[TradingSignal]:
        """生成交易信号（子类实现）"""
        raise NotImplementedError
    
    async def update_parameters(self, new_params: Dict[str, Any]):
        """更新策略参数"""
        self.config.update(new_params)
    
    def get_performance(self) -> Dict[str, Any]:
        """获取策略性能"""
        return {
            "name": self.name,
            "type": self.strategy_type.value,
            "performance": {
                "total_trades": self.performance.total_trades,
                "win_rate": self.performance.win_rate,
                "total_return": self.performance.total_return,
                "sharpe_ratio": self.performance.sharpe_ratio,
                "max_drawdown": self.performance.max_drawdown,
                "profit_factor": self.performance.profit_factor,
                "calmar_ratio": self.performance.calmar_ratio
            },
            "is_active": self.is_active,
            "last_updated": self.performance.last_updated.isoformat()
        }

class TrendFollowingStrategy(BaseStrategy):
    """趋势跟随策略"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            "TrendFollowing", 
            StrategyType.TREND_FOLLOWING, 
            config
        )
        self.sma_short = config.get("sma_short", 10)
        self.sma_long = config.get("sma_long", 30)
        self.rsi_period = config.get("rsi_period", 14)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)
    
    async def generate_signal(self, market_data: Dict[str, Any]) -> Optional[TradingSignal]:
        try:
            symbol = market_data.get("symbol")
            prices = market_data.get("prices", [])
            volumes = market_data.get("volumes", [])
            
            if len(prices) < max(self.sma_long, self.rsi_period):
                return None
            
            # 计算技术指标
            prices_array = np.array(prices)
            sma_short = np.mean(prices_array[-self.sma_short:])
            sma_long = np.mean(prices_array[-self.sma_long:])
            
            # 计算RSI
            rsi = self._calculate_rsi(prices_array, self.rsi_period)
            
            current_price = prices[-1]
            
            # 生成信号
            if sma_short > sma_long and rsi < self.rsi_overbought:
                # 上升趋势，买入信号
                action = "buy"
                strength = SignalStrength.STRONG if rsi < 50 else SignalStrength.MODERATE
                confidence = min(0.9, (sma_short - sma_long) / sma_long * 10)
                price_target = current_price * 1.02
                stop_loss = current_price * 0.98
                take_profit = current_price * 1.05
                reasoning = f"趋势向上，SMA{self.sma_short}({sma_short:.2f}) > SMA{self.sma_long}({sma_long:.2f}), RSI={rsi:.1f}"
                
            elif sma_short < sma_long and rsi > self.rsi_oversold:
                # 下降趋势，卖出信号
                action = "sell"
                strength = SignalStrength.STRONG if rsi > 50 else SignalStrength.MODERATE
                confidence = min(0.9, (sma_long - sma_short) / sma_long * 10)
                price_target = current_price * 0.98
                stop_loss = current_price * 1.02
                take_profit = current_price * 0.95
                reasoning = f"趋势向下，SMA{self.sma_short}({sma_short:.2f}) < SMA{self.sma_long}({sma_long:.2f}), RSI={rsi:.1f}"
                
            else:
                # 无明确信号
                return None
            
            return TradingSignal(
                symbol=symbol,
                action=action,
                strength=strength,
                confidence=confidence,
                price_target=price_target,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=0.1,  # 10%仓位
                reasoning=reasoning,
                strategy_name=self.name,
                metadata={
                    "sma_short": sma_short,
                    "sma_long": sma_long,
                    "rsi": rsi
                }
            )
            
        except Exception as e:
            self.logger.error(f"生成趋势跟随信号失败: {e}")
            return None
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """计算RSI指标"""
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

class MeanReversionStrategy(BaseStrategy):
    """均值回归策略"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            "MeanReversion", 
            StrategyType.MEAN_REVERSION, 
            config
        )
        self.lookback_period = config.get("lookback_period", 20)
        self.std_threshold = config.get("std_threshold", 2.0)
        self.rsi_period = config.get("rsi_period", 14)
    
    async def generate_signal(self, market_data: Dict[str, Any]) -> Optional[TradingSignal]:
        try:
            symbol = market_data.get("symbol")
            prices = market_data.get("prices", [])
            
            if len(prices) < max(self.lookback_period, self.rsi_period):
                return None
            
            prices_array = np.array(prices)
            current_price = prices[-1]
            
            # 计算布林带
            sma = np.mean(prices_array[-self.lookback_period:])
            std = np.std(prices_array[-self.lookback_period:])
            upper_band = sma + (self.std_threshold * std)
            lower_band = sma - (self.std_threshold * std)
            
            # 计算RSI
            rsi = self._calculate_rsi(prices_array, self.rsi_period)
            
            # 生成信号
            if current_price <= lower_band and rsi < 30:
                # 超卖，买入信号
                action = "buy"
                strength = SignalStrength.STRONG if rsi < 20 else SignalStrength.MODERATE
                confidence = min(0.9, (lower_band - current_price) / lower_band)
                price_target = sma
                stop_loss = current_price * 0.97
                take_profit = upper_band
                reasoning = f"超卖回归，价格({current_price:.2f})低于下轨({lower_band:.2f}), RSI={rsi:.1f}"
                
            elif current_price >= upper_band and rsi > 70:
                # 超买，卖出信号
                action = "sell"
                strength = SignalStrength.STRONG if rsi > 80 else SignalStrength.MODERATE
                confidence = min(0.9, (current_price - upper_band) / upper_band)
                price_target = sma
                stop_loss = current_price * 1.03
                take_profit = lower_band
                reasoning = f"超买回归，价格({current_price:.2f})高于上轨({upper_band:.2f}), RSI={rsi:.1f}"
                
            else:
                return None
            
            return TradingSignal(
                symbol=symbol,
                action=action,
                strength=strength,
                confidence=confidence,
                price_target=price_target,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=0.08,  # 8%仓位
                reasoning=reasoning,
                strategy_name=self.name,
                metadata={
                    "sma": sma,
                    "upper_band": upper_band,
                    "lower_band": lower_band,
                    "rsi": rsi
                }
            )
            
        except Exception as e:
            self.logger.error(f"生成均值回归信号失败: {e}")
            return None
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """计算RSI指标"""
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

class MLPredictionStrategy(BaseStrategy):
    """机器学习预测策略"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            "MLPrediction", 
            StrategyType.ML_PREDICTION, 
            config
        )
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = config.get("feature_columns", [
            "returns_1", "returns_5", "returns_10", 
            "rsi", "macd", "bb_position", "volume_ratio"
        ])
        self.prediction_threshold = config.get("prediction_threshold", 0.6)
        self.model_path = config.get("model_path", "models/ml_strategy_model.joblib")
        
        # 尝试加载预训练模型
        self._load_model()
    
    def _load_model(self):
        """加载预训练模型"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.logger.info("成功加载ML预测模型")
            else:
                self.logger.info("未找到预训练模型，将使用默认模型")
                self._create_default_model()
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            self._create_default_model()
    
    def _create_default_model(self):
        """创建默认模型"""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    async def generate_signal(self, market_data: Dict[str, Any]) -> Optional[TradingSignal]:
        try:
            symbol = market_data.get("symbol")
            
            # 提取特征
            features = await self._extract_features(market_data)
            if features is None:
                return None
            
            # 预测
            prediction_proba = self.model.predict_proba([features])[0]
            prediction_class = self.model.predict([features])[0]
            
            # 类别: 0=卖出, 1=持有, 2=买入
            max_proba = max(prediction_proba)
            
            if max_proba < self.prediction_threshold:
                return None  # 预测不够确定
            
            current_price = market_data.get("current_price", 0)
            
            if prediction_class == 2:  # 买入
                action = "buy"
                confidence = prediction_proba[2]
                strength = self._get_signal_strength(confidence)
                price_target = current_price * (1 + 0.02 * confidence)
                stop_loss = current_price * 0.98
                take_profit = current_price * 1.05
                reasoning = f"ML预测买入信号，置信度={confidence:.2f}"
                
            elif prediction_class == 0:  # 卖出
                action = "sell"
                confidence = prediction_proba[0]
                strength = self._get_signal_strength(confidence)
                price_target = current_price * (1 - 0.02 * confidence)
                stop_loss = current_price * 1.02
                take_profit = current_price * 0.95
                reasoning = f"ML预测卖出信号，置信度={confidence:.2f}"
                
            else:  # 持有
                return None
            
            return TradingSignal(
                symbol=symbol,
                action=action,
                strength=strength,
                confidence=confidence,
                price_target=price_target,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=0.12 * confidence,  # 根据置信度调整仓位
                reasoning=reasoning,
                strategy_name=self.name,
                metadata={
                    "prediction_proba": prediction_proba.tolist(),
                    "features": features
                }
            )
            
        except Exception as e:
            self.logger.error(f"生成ML预测信号失败: {e}")
            return None
    
    async def _extract_features(self, market_data: Dict[str, Any]) -> Optional[List[float]]:
        """提取特征"""
        try:
            prices = market_data.get("prices", [])
            volumes = market_data.get("volumes", [])
            
            if len(prices) < 20:
                return None
            
            prices_array = np.array(prices)
            volumes_array = np.array(volumes) if volumes else np.ones_like(prices_array)
            
            features = []
            
            # 收益率特征
            returns = np.diff(prices_array) / prices_array[:-1]
            features.extend([
                returns[-1] if len(returns) >= 1 else 0,  # 1日收益率
                np.mean(returns[-5:]) if len(returns) >= 5 else 0,  # 5日平均收益率
                np.mean(returns[-10:]) if len(returns) >= 10 else 0,  # 10日平均收益率
            ])
            
            # RSI特征
            rsi = self._calculate_rsi(prices_array, 14)
            features.append(rsi / 100.0)  # 归一化到0-1
            
            # MACD特征
            macd, signal = self._calculate_macd(prices_array)
            features.append(macd)
            
            # 布林带位置特征
            sma = np.mean(prices_array[-20:])
            std = np.std(prices_array[-20:])
            bb_position = (prices_array[-1] - sma) / (2 * std) + 0.5  # 归一化到0-1
            features.append(bb_position)
            
            # 成交量特征
            vol_ratio = volumes_array[-1] / np.mean(volumes_array[-10:]) if len(volumes_array) >= 10 else 1.0
            features.append(min(vol_ratio, 5.0))  # 限制最大值
            
            return features
            
        except Exception as e:
            self.logger.error(f"特征提取失败: {e}")
            return None
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
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
    
    def _calculate_macd(self, prices: np.ndarray) -> Tuple[float, float]:
        """计算MACD"""
        if len(prices) < 26:
            return 0.0, 0.0
        
        ema12 = self._calculate_ema(prices, 12)
        ema26 = self._calculate_ema(prices, 26)
        macd = ema12 - ema26
        
        # 简化的信号线
        signal = macd * 0.1  # 简化处理
        
        return macd / prices[-1], signal / prices[-1]  # 归一化
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """计算EMA"""
        if len(prices) < period:
            return np.mean(prices)
        
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def _get_signal_strength(self, confidence: float) -> SignalStrength:
        """根据置信度确定信号强度"""
        if confidence >= 0.9:
            return SignalStrength.VERY_STRONG
        elif confidence >= 0.8:
            return SignalStrength.STRONG
        elif confidence >= 0.7:
            return SignalStrength.MODERATE
        elif confidence >= 0.6:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK

class StrategyAgent(BaseAgent):
    """策略Agent - 智能交易策略生成和管理"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id="strategy_agent",
            agent_type=AgentType.STRATEGY,
            capabilities=[
                AgentCapability.STRATEGY_GENERATION,
                AgentCapability.DATA_ANALYSIS,
                AgentCapability.PATTERN_RECOGNITION,
                AgentCapability.PREDICTION
            ],
            config=config or {}
        )
        
        # 策略管理
        self.strategies: Dict[str, BaseStrategy] = {}
        self.active_strategies: Set[str] = set()
        self.strategy_signals: Dict[str, List[TradingSignal]] = {}
        
        # 市场数据缓存
        self.market_data_cache: Dict[str, Dict[str, Any]] = {}
        self.data_update_interval = config.get("data_update_interval", 60)  # 秒
        
        # 策略配置
        self.max_strategies = config.get("max_strategies", 10)
        self.signal_history_limit = config.get("signal_history_limit", 1000)
        
        # 性能监控
        self.strategy_performance: Dict[str, StrategyPerformance] = {}
        
    async def _initialize(self):
        """初始化策略Agent"""
        # 创建默认策略
        await self._create_default_strategies()
        
        # 启动策略信号生成循环
        asyncio.create_task(self._signal_generation_loop())
        
        # 启动性能监控循环
        asyncio.create_task(self._performance_monitoring_loop())
        
        self.logger.info("🎯 策略Agent初始化完成")
    
    async def _create_default_strategies(self):
        """创建默认策略集"""
        try:
            # 趋势跟随策略
            trend_strategy = TrendFollowingStrategy({
                "sma_short": 10,
                "sma_long": 30,
                "rsi_period": 14
            })
            await self.add_strategy(trend_strategy)
            
            # 均值回归策略
            mean_reversion_strategy = MeanReversionStrategy({
                "lookback_period": 20,
                "std_threshold": 2.0
            })
            await self.add_strategy(mean_reversion_strategy)
            
            # ML预测策略
            ml_strategy = MLPredictionStrategy({
                "prediction_threshold": 0.65
            })
            await self.add_strategy(ml_strategy)
            
            # 默认激活所有策略
            for strategy_name in self.strategies.keys():
                self.active_strategies.add(strategy_name)
            
            self.logger.info(f"创建了{len(self.strategies)}个默认策略")
            
        except Exception as e:
            self.logger.error(f"创建默认策略失败: {e}")
    
    async def add_strategy(self, strategy: BaseStrategy) -> bool:
        """添加策略"""
        try:
            if len(self.strategies) >= self.max_strategies:
                self.logger.warning("策略数量已达上限")
                return False
            
            self.strategies[strategy.name] = strategy
            self.strategy_signals[strategy.name] = []
            self.strategy_performance[strategy.name] = strategy.performance
            
            self.logger.info(f"添加策略: {strategy.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"添加策略失败: {e}")
            return False
    
    async def _handle_command(self, message: AgentMessage):
        """处理命令消息"""
        command = message.content.get("command")
        params = message.content.get("params", {})
        
        if command == "generate_signals":
            await self._handle_generate_signals(message, params)
        elif command == "add_strategy":
            await self._handle_add_strategy(message, params)
        elif command == "activate_strategy":
            await self._handle_activate_strategy(message, params)
        elif command == "deactivate_strategy":
            await self._handle_deactivate_strategy(message, params)
        elif command == "update_market_data":
            await self._handle_update_market_data(message, params)
        elif command == "optimize_strategy":
            await self._handle_optimize_strategy(message, params)
        else:
            await self.send_response(message, {"error": f"未知命令: {command}"})
    
    async def _handle_query(self, message: AgentMessage):
        """处理查询消息"""
        query = message.content.get("query")
        params = message.content.get("params", {})
        
        if query == "get_strategies":
            response = await self._get_strategies_info()
        elif query == "get_signals":
            response = await self._get_recent_signals(params)
        elif query == "get_performance":
            response = await self._get_strategy_performance(params)
        elif query == "get_market_overview":
            response = await self._get_market_overview()
        else:
            response = {"error": f"未知查询: {query}"}
        
        await self.send_response(message, response)
    
    async def _handle_generate_signals(self, message: AgentMessage, params: Dict[str, Any]):
        """处理信号生成请求"""
        try:
            symbols = params.get("symbols", [])
            strategy_names = params.get("strategies", list(self.active_strategies))
            
            signals = await self._generate_signals_for_symbols(symbols, strategy_names)
            
            await self.send_response(message, {
                "signals": [signal.to_dict() for signal in signals],
                "count": len(signals),
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            await self.send_response(message, {"error": str(e)})
    
    async def _handle_update_market_data(self, message: AgentMessage, params: Dict[str, Any]):
        """处理市场数据更新"""
        try:
            symbol = params.get("symbol")
            market_data = params.get("data")
            
            if symbol and market_data:
                self.market_data_cache[symbol] = market_data
                self.market_data_cache[symbol]["updated_at"] = datetime.utcnow()
            
            await self.send_response(message, {"status": "updated"})
            
        except Exception as e:
            await self.send_response(message, {"error": str(e)})
    
    async def _signal_generation_loop(self):
        """信号生成循环"""
        self.logger.info("🔄 启动信号生成循环")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(30)  # 每30秒生成一次信号
                
                # 为所有缓存的市场数据生成信号
                for symbol, data in self.market_data_cache.items():
                    if self._is_data_fresh(data):
                        await self._generate_signals_for_symbol(symbol, data)
                
            except Exception as e:
                self.logger.error(f"信号生成循环错误: {e}")
    
    async def _generate_signals_for_symbol(self, symbol: str, market_data: Dict[str, Any]):
        """为单个标的生成信号"""
        market_data["symbol"] = symbol
        
        for strategy_name in self.active_strategies:
            if strategy_name in self.strategies:
                try:
                    strategy = self.strategies[strategy_name]
                    signal = await strategy.generate_signal(market_data)
                    
                    if signal:
                        # 存储信号
                        self.strategy_signals[strategy_name].append(signal)
                        
                        # 限制信号历史长度
                        if len(self.strategy_signals[strategy_name]) > self.signal_history_limit:
                            self.strategy_signals[strategy_name].pop(0)
                        
                        # 广播信号给其他Agent
                        await self._broadcast_signal(signal)
                        
                        self.logger.info(f"📊 {strategy_name}生成信号: {symbol} {signal.action} (置信度: {signal.confidence:.2f})")
                        
                except Exception as e:
                    self.logger.error(f"策略{strategy_name}生成信号失败: {e}")
    
    async def _generate_signals_for_symbols(self, symbols: List[str], strategy_names: List[str]) -> List[TradingSignal]:
        """为多个标的生成信号"""
        signals = []
        
        for symbol in symbols:
            if symbol in self.market_data_cache:
                market_data = self.market_data_cache[symbol]
                if self._is_data_fresh(market_data):
                    market_data["symbol"] = symbol
                    
                    for strategy_name in strategy_names:
                        if strategy_name in self.strategies and strategy_name in self.active_strategies:
                            try:
                                strategy = self.strategies[strategy_name]
                                signal = await strategy.generate_signal(market_data)
                                if signal:
                                    signals.append(signal)
                            except Exception as e:
                                self.logger.error(f"策略{strategy_name}为{symbol}生成信号失败: {e}")
        
        return signals
    
    def _is_data_fresh(self, data: Dict[str, Any]) -> bool:
        """检查数据是否新鲜"""
        updated_at = data.get("updated_at")
        if not updated_at:
            return False
        
        age = (datetime.utcnow() - updated_at).total_seconds()
        return age < self.data_update_interval * 2  # 数据年龄小于2个更新间隔
    
    async def _broadcast_signal(self, signal: TradingSignal):
        """广播交易信号"""
        try:
            signal_message = AgentMessage(
                receiver_id="*",  # 广播
                message_type=MessageType.EVENT,
                priority=5,
                content={
                    "event_type": "trading_signal",
                    "signal": signal.to_dict()
                }
            )
            
            await self.send_message(signal_message)
            
        except Exception as e:
            self.logger.error(f"广播信号失败: {e}")
    
    async def _performance_monitoring_loop(self):
        """策略性能监控循环"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # 每5分钟更新一次性能指标
                
                for strategy_name, strategy in self.strategies.items():
                    # 从历史信号计算性能（简化版本）
                    signals = self.strategy_signals.get(strategy_name, [])
                    if len(signals) > 10:  # 至少有10个信号才计算性能
                        # 这里应该结合实际交易结果来计算，暂时用模拟数据
                        mock_returns = np.random.normal(0.001, 0.02, len(signals))
                        strategy.performance.update_metrics(mock_returns.tolist())
                
            except Exception as e:
                self.logger.error(f"性能监控错误: {e}")
    
    async def _get_strategies_info(self) -> Dict[str, Any]:
        """获取策略信息"""
        strategies_info = []
        
        for name, strategy in self.strategies.items():
            info = strategy.get_performance()
            info["is_active"] = name in self.active_strategies
            info["signal_count"] = len(self.strategy_signals.get(name, []))
            strategies_info.append(info)
        
        return {
            "strategies": strategies_info,
            "active_count": len(self.active_strategies),
            "total_count": len(self.strategies)
        }
    
    async def _get_recent_signals(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """获取最近的信号"""
        strategy_name = params.get("strategy")
        limit = params.get("limit", 50)
        
        if strategy_name and strategy_name in self.strategy_signals:
            signals = self.strategy_signals[strategy_name][-limit:]
            return {
                "strategy": strategy_name,
                "signals": [signal.to_dict() for signal in signals],
                "count": len(signals)
            }
        else:
            # 返回所有策略的信号
            all_signals = []
            for name, signals in self.strategy_signals.items():
                for signal in signals[-limit//len(self.strategy_signals):]:
                    all_signals.append(signal.to_dict())
            
            # 按时间排序
            all_signals.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return {
                "signals": all_signals[:limit],
                "count": len(all_signals)
            }
    
    async def _get_strategy_performance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """获取策略性能"""
        strategy_name = params.get("strategy")
        
        if strategy_name and strategy_name in self.strategies:
            return self.strategies[strategy_name].get_performance()
        else:
            # 返回所有策略的性能
            performance_data = {}
            for name, strategy in self.strategies.items():
                performance_data[name] = strategy.get_performance()
            
            return {"strategies_performance": performance_data}
    
    async def _get_market_overview(self) -> Dict[str, Any]:
        """获取市场概览"""
        return {
            "cached_symbols": len(self.market_data_cache),
            "fresh_data_count": sum(1 for data in self.market_data_cache.values() if self._is_data_fresh(data)),
            "last_update": max(
                (data.get("updated_at", datetime.min) for data in self.market_data_cache.values()),
                default=datetime.min
            ).isoformat() if self.market_data_cache else None
        }