"""
风险Agent - 智能风险评估和管理
负责实时风险监控、风险评估、风险预警和风险控制决策
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .base_agent import BaseAgent, AgentType, AgentCapability, AgentMessage, MessageType

# 风险等级枚举
class RiskLevel(Enum):
    VERY_LOW = 1
    LOW = 2
    MODERATE = 3
    HIGH = 4
    VERY_HIGH = 5
    EXTREME = 6

# 风险类型枚举
class RiskType(Enum):
    MARKET_RISK = "market_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    CONCENTRATION_RISK = "concentration_risk"
    VOLATILITY_RISK = "volatility_risk"
    CREDIT_RISK = "credit_risk"
    OPERATIONAL_RISK = "operational_risk"
    CORRELATION_RISK = "correlation_risk"
    TAIL_RISK = "tail_risk"

# 风险警报类型
class AlertType(Enum):
    POSITION_SIZE = "position_size"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    VAR_BREACH = "var_breach"
    STOP_LOSS = "stop_loss"
    MARGIN_CALL = "margin_call"
    BLACK_SWAN = "black_swan"

@dataclass
class RiskMetrics:
    """风险指标"""
    symbol: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # 基础风险指标
    volatility_1d: float = 0.0
    volatility_7d: float = 0.0
    volatility_30d: float = 0.0
    
    # VaR指标
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    
    # 最大回撤
    max_drawdown_1d: float = 0.0
    max_drawdown_7d: float = 0.0
    max_drawdown_30d: float = 0.0
    
    # 相关性风险
    correlation_risk: float = 0.0
    beta: float = 0.0
    
    # 流动性风险
    liquidity_score: float = 1.0
    bid_ask_spread: float = 0.0
    
    # 集中度风险
    concentration_score: float = 0.0
    
    # 综合风险评分
    overall_risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.MODERATE

@dataclass
class RiskAlert:
    """风险警报"""
    alert_id: str
    alert_type: AlertType
    risk_level: RiskLevel
    symbol: str
    message: str
    threshold_value: float
    current_value: float
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PortfolioRisk:
    """投资组合风险"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # 投资组合总风险
    portfolio_var_95: float = 0.0
    portfolio_var_99: float = 0.0
    portfolio_volatility: float = 0.0
    portfolio_beta: float = 0.0
    
    # 分散化指标
    diversification_ratio: float = 0.0
    effective_positions: int = 0
    
    # 集中度风险
    max_position_weight: float = 0.0
    top5_concentration: float = 0.0
    
    # 流动性风险
    portfolio_liquidity_score: float = 1.0
    days_to_liquidate: float = 1.0
    
    # 尾部风险
    tail_ratio: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    # 整体风险等级
    overall_risk_level: RiskLevel = RiskLevel.MODERATE

class RiskCalculator:
    """风险计算器"""
    
    @staticmethod
    def calculate_volatility(prices: np.ndarray, window: int = 30) -> float:
        """计算波动率（年化）"""
        if len(prices) < 2:
            return 0.0
        
        returns = np.diff(prices) / prices[:-1]
        if len(returns) < window:
            window = len(returns)
        
        vol = np.std(returns[-window:]) * np.sqrt(252)  # 年化
        return vol
    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """计算在险价值(VaR)"""
        if len(returns) < 10:
            return 0.0
        
        return np.percentile(returns, (1 - confidence) * 100)
    
    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
        """计算条件在险价值(CVaR)"""
        var = RiskCalculator.calculate_var(returns, confidence)
        cvar = np.mean(returns[returns <= var])
        return cvar if not np.isnan(cvar) else 0.0
    
    @staticmethod
    def calculate_max_drawdown(prices: np.ndarray) -> float:
        """计算最大回撤"""
        if len(prices) < 2:
            return 0.0
        
        cumulative = np.cumprod(1 + np.diff(prices) / prices[:-1])
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative / running_max - 1
        return np.min(drawdowns)
    
    @staticmethod
    def calculate_beta(asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """计算Beta值"""
        if len(asset_returns) != len(market_returns) or len(asset_returns) < 10:
            return 1.0
        
        covariance = np.cov(asset_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        if market_variance == 0:
            return 1.0
        
        return covariance / market_variance
    
    @staticmethod
    def calculate_correlation_matrix(returns_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """计算相关性矩阵"""
        if not returns_dict:
            return np.array([[1.0]])
        
        symbols = list(returns_dict.keys())
        min_length = min(len(returns) for returns in returns_dict.values())
        
        returns_matrix = np.array([
            returns_dict[symbol][-min_length:] for symbol in symbols
        ])
        
        return np.corrcoef(returns_matrix)
    
    @staticmethod
    def calculate_portfolio_var(
        weights: np.ndarray, 
        returns: np.ndarray, 
        correlation_matrix: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """计算投资组合VaR"""
        if len(weights) != returns.shape[0] or len(weights) != correlation_matrix.shape[0]:
            return 0.0
        
        # 投资组合方差
        portfolio_variance = np.dot(weights.T, np.dot(correlation_matrix * np.outer(np.std(returns, axis=1), np.std(returns, axis=1)), weights))
        portfolio_std = np.sqrt(portfolio_variance)
        
        # 假设正态分布
        z_score = stats.norm.ppf(1 - confidence)
        portfolio_var = -z_score * portfolio_std
        
        return portfolio_var

class RiskAgent(BaseAgent):
    """风险Agent - 智能风险评估和管理"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id="risk_agent",
            agent_type=AgentType.RISK,
            capabilities=[
                AgentCapability.RISK_ASSESSMENT,
                AgentCapability.DATA_ANALYSIS,
                AgentCapability.PATTERN_RECOGNITION,
                AgentCapability.PREDICTION
            ],
            config=config or {}
        )
        
        # 风险配置
        self.risk_thresholds = {
            "max_position_size": config.get("max_position_size", 0.10),  # 最大单仓位10%
            "max_portfolio_var": config.get("max_portfolio_var", 0.05),  # 最大投资组合VaR 5%
            "max_drawdown": config.get("max_drawdown", 0.10),  # 最大回撤10%
            "min_liquidity_score": config.get("min_liquidity_score", 0.5),  # 最小流动性评分
            "max_correlation": config.get("max_correlation", 0.8),  # 最大相关性
            "volatility_threshold": config.get("volatility_threshold", 0.5),  # 波动率阈值50%
        }
        
        # 数据存储
        self.price_history: Dict[str, List[float]] = {}
        self.position_data: Dict[str, Dict[str, Any]] = {}
        self.risk_metrics_cache: Dict[str, RiskMetrics] = {}
        self.portfolio_risk_cache: Optional[PortfolioRisk] = None
        
        # 警报管理
        self.active_alerts: Dict[str, RiskAlert] = {}
        self.alert_history: List[RiskAlert] = []
        self.max_alert_history = config.get("max_alert_history", 1000)
        
        # 市场数据
        self.market_data_cache: Dict[str, Dict[str, Any]] = {}
        
        # 风险计算器
        self.risk_calculator = RiskCalculator()
        
        # 监控配置
        self.risk_update_interval = config.get("risk_update_interval", 30)  # 秒
        self.alert_check_interval = config.get("alert_check_interval", 10)  # 秒
        
    async def _initialize(self):
        """初始化风险Agent"""
        # 启动风险监控循环
        asyncio.create_task(self._risk_monitoring_loop())
        
        # 启动警报检查循环
        asyncio.create_task(self._alert_monitoring_loop())
        
        # 启动投资组合风险计算循环
        asyncio.create_task(self._portfolio_risk_loop())
        
        self.logger.info("🛡️ 风险Agent初始化完成")
    
    async def _handle_command(self, message: AgentMessage):
        """处理命令消息"""
        command = message.content.get("command")
        params = message.content.get("params", {})
        
        if command == "assess_risk":
            await self._handle_assess_risk(message, params)
        elif command == "update_position":
            await self._handle_update_position(message, params)
        elif command == "update_market_data":
            await self._handle_update_market_data(message, params)
        elif command == "set_risk_limit":
            await self._handle_set_risk_limit(message, params)
        elif command == "acknowledge_alert":
            await self._handle_acknowledge_alert(message, params)
        elif command == "validate_trade":
            await self._handle_validate_trade(message, params)
        else:
            await self.send_response(message, {"error": f"未知命令: {command}"})
    
    async def _handle_query(self, message: AgentMessage):
        """处理查询消息"""
        query = message.content.get("query")
        params = message.content.get("params", {})
        
        if query == "get_risk_metrics":
            response = await self._get_risk_metrics(params)
        elif query == "get_portfolio_risk":
            response = await self._get_portfolio_risk()
        elif query == "get_active_alerts":
            response = await self._get_active_alerts()
        elif query == "get_risk_report":
            response = await self._get_risk_report(params)
        elif query == "check_risk_limits":
            response = await self._check_risk_limits(params)
        else:
            response = {"error": f"未知查询: {query}"}
        
        await self.send_response(message, response)
    
    async def _handle_assess_risk(self, message: AgentMessage, params: Dict[str, Any]):
        """处理风险评估请求"""
        try:
            symbol = params.get("symbol")
            
            if symbol:
                # 单个资产风险评估
                risk_metrics = await self._calculate_asset_risk(symbol)
                response = {
                    "symbol": symbol,
                    "risk_metrics": risk_metrics.to_dict() if risk_metrics else None,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                # 投资组合风险评估
                portfolio_risk = await self._calculate_portfolio_risk()
                response = {
                    "portfolio_risk": portfolio_risk.to_dict() if portfolio_risk else None,
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            await self.send_response(message, response)
            
        except Exception as e:
            await self.send_response(message, {"error": str(e)})
    
    async def _handle_update_position(self, message: AgentMessage, params: Dict[str, Any]):
        """处理持仓更新"""
        try:
            symbol = params.get("symbol")
            position_data = params.get("position_data")
            
            if symbol and position_data:
                self.position_data[symbol] = {
                    "size": position_data.get("size", 0),
                    "value": position_data.get("value", 0),
                    "entry_price": position_data.get("entry_price", 0),
                    "current_price": position_data.get("current_price", 0),
                    "unrealized_pnl": position_data.get("unrealized_pnl", 0),
                    "updated_at": datetime.utcnow()
                }
                
                # 立即进行风险检查
                await self._check_position_risk(symbol)
            
            await self.send_response(message, {"status": "updated"})
            
        except Exception as e:
            await self.send_response(message, {"error": str(e)})
    
    async def _handle_update_market_data(self, message: AgentMessage, params: Dict[str, Any]):
        """处理市场数据更新"""
        try:
            symbol = params.get("symbol")
            market_data = params.get("data")
            
            if symbol and market_data:
                self.market_data_cache[symbol] = market_data
                
                # 更新价格历史
                if "price" in market_data:
                    if symbol not in self.price_history:
                        self.price_history[symbol] = []
                    
                    self.price_history[symbol].append(market_data["price"])
                    
                    # 限制历史数据长度
                    if len(self.price_history[symbol]) > 1000:
                        self.price_history[symbol] = self.price_history[symbol][-500:]
            
            await self.send_response(message, {"status": "updated"})
            
        except Exception as e:
            await self.send_response(message, {"error": str(e)})
    
    async def _handle_validate_trade(self, message: AgentMessage, params: Dict[str, Any]):
        """处理交易验证请求"""
        try:
            trade_data = params.get("trade")
            symbol = trade_data.get("symbol")
            size = trade_data.get("size", 0)
            action = trade_data.get("action", "buy")
            
            # 风险验证
            validation_result = await self._validate_trade_risk(symbol, size, action)
            
            await self.send_response(message, validation_result)
            
        except Exception as e:
            await self.send_response(message, {"error": str(e)})
    
    async def _risk_monitoring_loop(self):
        """风险监控循环"""
        self.logger.info("🔄 启动风险监控循环")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.risk_update_interval)
                
                # 计算所有持仓的风险指标
                for symbol in self.position_data.keys():
                    try:
                        risk_metrics = await self._calculate_asset_risk(symbol)
                        if risk_metrics:
                            self.risk_metrics_cache[symbol] = risk_metrics
                    except Exception as e:
                        self.logger.error(f"计算{symbol}风险指标失败: {e}")
                
                # 计算投资组合风险
                try:
                    portfolio_risk = await self._calculate_portfolio_risk()
                    if portfolio_risk:
                        self.portfolio_risk_cache = portfolio_risk
                except Exception as e:
                    self.logger.error(f"计算投资组合风险失败: {e}")
                
            except Exception as e:
                self.logger.error(f"风险监控循环错误: {e}")
    
    async def _alert_monitoring_loop(self):
        """警报监控循环"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.alert_check_interval)
                
                # 检查各种风险警报
                await self._check_all_risk_alerts()
                
            except Exception as e:
                self.logger.error(f"警报监控循环错误: {e}")
    
    async def _portfolio_risk_loop(self):
        """投资组合风险计算循环"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # 每分钟计算一次投资组合风险
                
                portfolio_risk = await self._calculate_portfolio_risk()
                if portfolio_risk:
                    self.portfolio_risk_cache = portfolio_risk
                    
                    # 检查投资组合风险警报
                    await self._check_portfolio_risk_alerts(portfolio_risk)
                
            except Exception as e:
                self.logger.error(f"投资组合风险循环错误: {e}")
    
    async def _calculate_asset_risk(self, symbol: str) -> Optional[RiskMetrics]:
        """计算单个资产的风险指标"""
        try:
            if symbol not in self.price_history or len(self.price_history[symbol]) < 10:
                return None
            
            prices = np.array(self.price_history[symbol])
            returns = np.diff(prices) / prices[:-1]
            
            # 计算风险指标
            risk_metrics = RiskMetrics(symbol=symbol)
            
            # 波动率（不同时间窗口）
            if len(returns) >= 1:
                risk_metrics.volatility_1d = self.risk_calculator.calculate_volatility(prices, 1)
            if len(returns) >= 7:
                risk_metrics.volatility_7d = self.risk_calculator.calculate_volatility(prices, 7)
            if len(returns) >= 30:
                risk_metrics.volatility_30d = self.risk_calculator.calculate_volatility(prices, 30)
            
            # VaR和CVaR
            risk_metrics.var_95 = self.risk_calculator.calculate_var(returns, 0.95)
            risk_metrics.var_99 = self.risk_calculator.calculate_var(returns, 0.99)
            risk_metrics.cvar_95 = self.risk_calculator.calculate_cvar(returns, 0.95)
            risk_metrics.cvar_99 = self.risk_calculator.calculate_cvar(returns, 0.99)
            
            # 最大回撤
            if len(prices) >= 2:
                risk_metrics.max_drawdown_1d = self.risk_calculator.calculate_max_drawdown(prices[-2:])
            if len(prices) >= 7:
                risk_metrics.max_drawdown_7d = self.risk_calculator.calculate_max_drawdown(prices[-7:])
            if len(prices) >= 30:
                risk_metrics.max_drawdown_30d = self.risk_calculator.calculate_max_drawdown(prices[-30:])
            
            # 流动性评分（基于买卖价差和成交量）
            market_data = self.market_data_cache.get(symbol, {})
            bid_ask_spread = market_data.get("bid_ask_spread", 0.001)
            volume = market_data.get("volume", 1000000)
            
            risk_metrics.bid_ask_spread = bid_ask_spread
            risk_metrics.liquidity_score = min(1.0, volume / 1000000 / max(bid_ask_spread * 100, 0.1))
            
            # 计算综合风险评分
            risk_score = self._calculate_overall_risk_score(risk_metrics)
            risk_metrics.overall_risk_score = risk_score
            risk_metrics.risk_level = self._get_risk_level(risk_score)
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"计算{symbol}风险指标失败: {e}")
            return None
    
    async def _calculate_portfolio_risk(self) -> Optional[PortfolioRisk]:
        """计算投资组合风险"""
        try:
            if not self.position_data:
                return None
            
            portfolio_risk = PortfolioRisk()
            
            # 收集投资组合数据
            symbols = list(self.position_data.keys())
            weights = []
            returns_dict = {}
            total_value = sum(pos.get("value", 0) for pos in self.position_data.values())
            
            if total_value == 0:
                return portfolio_risk
            
            for symbol in symbols:
                position = self.position_data[symbol]
                weight = position.get("value", 0) / total_value
                weights.append(weight)
                
                if symbol in self.price_history and len(self.price_history[symbol]) > 10:
                    prices = np.array(self.price_history[symbol])
                    returns = np.diff(prices) / prices[:-1]
                    returns_dict[symbol] = returns
            
            if not returns_dict:
                return portfolio_risk
            
            weights = np.array(weights[:len(returns_dict)])
            
            # 计算相关性矩阵
            correlation_matrix = self.risk_calculator.calculate_correlation_matrix(returns_dict)
            
            # 计算投资组合波动率
            returns_matrix = np.array([returns_dict[symbol] for symbol in returns_dict.keys()])
            if len(returns_matrix) > 0:
                portfolio_returns = np.dot(weights, returns_matrix)
                portfolio_risk.portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252)
            
            # 计算投资组合VaR
            portfolio_risk.portfolio_var_95 = self.risk_calculator.calculate_portfolio_var(
                weights, returns_matrix, correlation_matrix, 0.95
            )
            portfolio_risk.portfolio_var_99 = self.risk_calculator.calculate_portfolio_var(
                weights, returns_matrix, correlation_matrix, 0.99
            )
            
            # 集中度风险
            portfolio_risk.max_position_weight = max(weights) if len(weights) > 0 else 0
            sorted_weights = sorted(weights, reverse=True)
            portfolio_risk.top5_concentration = sum(sorted_weights[:5])
            
            # 有效持仓数量
            portfolio_risk.effective_positions = 1 / sum(w**2 for w in weights) if weights.any() else 0
            
            # 分散化比率
            individual_vol = np.sqrt(sum(w**2 * np.var(returns_dict[symbol]) for w, symbol in zip(weights, returns_dict.keys())))
            portfolio_vol = np.std(portfolio_returns) if len(returns_matrix) > 0 else 0
            portfolio_risk.diversification_ratio = individual_vol / portfolio_vol if portfolio_vol > 0 else 1
            
            # 流动性评分
            liquidity_scores = []
            for symbol in symbols:
                risk_metrics = self.risk_metrics_cache.get(symbol)
                if risk_metrics:
                    liquidity_scores.append(risk_metrics.liquidity_score)
            
            if liquidity_scores:
                portfolio_risk.portfolio_liquidity_score = np.mean(liquidity_scores)
            
            # 尾部风险指标
            if len(returns_matrix) > 0 and len(portfolio_returns) > 10:
                portfolio_risk.skewness = stats.skew(portfolio_returns)
                portfolio_risk.kurtosis = stats.kurtosis(portfolio_returns)
                
                # 尾部比率（极端损失概率）
                extreme_losses = portfolio_returns[portfolio_returns < np.percentile(portfolio_returns, 5)]
                portfolio_risk.tail_ratio = len(extreme_losses) / len(portfolio_returns)
            
            # 整体风险等级
            risk_score = self._calculate_portfolio_risk_score(portfolio_risk)
            portfolio_risk.overall_risk_level = self._get_risk_level(risk_score)
            
            return portfolio_risk
            
        except Exception as e:
            self.logger.error(f"计算投资组合风险失败: {e}")
            return None
    
    def _calculate_overall_risk_score(self, metrics: RiskMetrics) -> float:
        """计算综合风险评分 (0-100)"""
        try:
            score = 0
            
            # 波动率分数 (0-30)
            vol_score = min(30, metrics.volatility_30d * 60)
            score += vol_score
            
            # VaR分数 (0-25)
            var_score = min(25, abs(metrics.var_95) * 500)
            score += var_score
            
            # 最大回撤分数 (0-25)
            dd_score = min(25, abs(metrics.max_drawdown_30d) * 250)
            score += dd_score
            
            # 流动性分数 (0-20, 流动性越低分数越高)
            liquidity_score = (1 - metrics.liquidity_score) * 20
            score += liquidity_score
            
            return min(100, score)
            
        except Exception:
            return 50  # 默认中等风险
    
    def _calculate_portfolio_risk_score(self, portfolio_risk: PortfolioRisk) -> float:
        """计算投资组合风险评分 (0-100)"""
        try:
            score = 0
            
            # 投资组合波动率 (0-25)
            vol_score = min(25, portfolio_risk.portfolio_volatility * 50)
            score += vol_score
            
            # VaR分数 (0-25)
            var_score = min(25, portfolio_risk.portfolio_var_95 * 500)
            score += var_score
            
            # 集中度风险 (0-25)
            concentration_score = portfolio_risk.max_position_weight * 25
            score += concentration_score
            
            # 流动性风险 (0-15)
            liquidity_score = (1 - portfolio_risk.portfolio_liquidity_score) * 15
            score += liquidity_score
            
            # 尾部风险 (0-10)
            tail_score = portfolio_risk.tail_ratio * 100
            score += min(10, tail_score)
            
            return min(100, score)
            
        except Exception:
            return 50  # 默认中等风险
    
    def _get_risk_level(self, risk_score: float) -> RiskLevel:
        """根据风险评分确定风险等级"""
        if risk_score <= 20:
            return RiskLevel.VERY_LOW
        elif risk_score <= 35:
            return RiskLevel.LOW
        elif risk_score <= 50:
            return RiskLevel.MODERATE
        elif risk_score <= 70:
            return RiskLevel.HIGH
        elif risk_score <= 85:
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.EXTREME
    
    async def _check_all_risk_alerts(self):
        """检查所有风险警报"""
        # 检查持仓风险
        for symbol in self.position_data.keys():
            await self._check_position_risk(symbol)
        
        # 检查投资组合风险
        if self.portfolio_risk_cache:
            await self._check_portfolio_risk_alerts(self.portfolio_risk_cache)
    
    async def _check_position_risk(self, symbol: str):
        """检查单个持仓风险"""
        try:
            position = self.position_data.get(symbol)
            risk_metrics = self.risk_metrics_cache.get(symbol)
            
            if not position or not risk_metrics:
                return
            
            # 检查仓位大小
            position_size = abs(position.get("size", 0))
            max_position = self.risk_thresholds["max_position_size"]
            
            if position_size > max_position:
                await self._create_alert(
                    AlertType.POSITION_SIZE,
                    RiskLevel.HIGH,
                    symbol,
                    f"持仓超过限制: {position_size:.1%} > {max_position:.1%}",
                    max_position,
                    position_size
                )
            
            # 检查波动率
            if risk_metrics.volatility_30d > self.risk_thresholds["volatility_threshold"]:
                await self._create_alert(
                    AlertType.VOLATILITY,
                    RiskLevel.HIGH,
                    symbol,
                    f"波动率过高: {risk_metrics.volatility_30d:.1%}",
                    self.risk_thresholds["volatility_threshold"],
                    risk_metrics.volatility_30d
                )
            
            # 检查最大回撤
            if abs(risk_metrics.max_drawdown_30d) > self.risk_thresholds["max_drawdown"]:
                await self._create_alert(
                    AlertType.DRAWDOWN,
                    RiskLevel.HIGH,
                    symbol,
                    f"最大回撤过大: {risk_metrics.max_drawdown_30d:.1%}",
                    -self.risk_thresholds["max_drawdown"],
                    risk_metrics.max_drawdown_30d
                )
            
            # 检查流动性
            if risk_metrics.liquidity_score < self.risk_thresholds["min_liquidity_score"]:
                await self._create_alert(
                    AlertType.LIQUIDITY_RISK,
                    RiskLevel.MODERATE,
                    symbol,
                    f"流动性不足: {risk_metrics.liquidity_score:.2f}",
                    self.risk_thresholds["min_liquidity_score"],
                    risk_metrics.liquidity_score
                )
            
        except Exception as e:
            self.logger.error(f"检查{symbol}持仓风险失败: {e}")
    
    async def _check_portfolio_risk_alerts(self, portfolio_risk: PortfolioRisk):
        """检查投资组合风险警报"""
        try:
            # 检查投资组合VaR
            if portfolio_risk.portfolio_var_95 > self.risk_thresholds["max_portfolio_var"]:
                await self._create_alert(
                    AlertType.VAR_BREACH,
                    RiskLevel.HIGH,
                    "PORTFOLIO",
                    f"投资组合VaR超限: {portfolio_risk.portfolio_var_95:.1%}",
                    self.risk_thresholds["max_portfolio_var"],
                    portfolio_risk.portfolio_var_95
                )
            
            # 检查集中度风险
            if portfolio_risk.max_position_weight > self.risk_thresholds["max_position_size"]:
                await self._create_alert(
                    AlertType.CONCENTRATION_RISK,
                    RiskLevel.MODERATE,
                    "PORTFOLIO",
                    f"持仓过于集中: {portfolio_risk.max_position_weight:.1%}",
                    self.risk_thresholds["max_position_size"],
                    portfolio_risk.max_position_weight
                )
            
        except Exception as e:
            self.logger.error(f"检查投资组合风险警报失败: {e}")
    
    async def _create_alert(
        self,
        alert_type: AlertType,
        risk_level: RiskLevel,
        symbol: str,
        message: str,
        threshold_value: float,
        current_value: float
    ):
        """创建风险警报"""
        try:
            alert_id = f"{alert_type.value}_{symbol}_{int(datetime.utcnow().timestamp())}"
            
            # 检查是否已有相同类型的活跃警报
            existing_key = f"{alert_type.value}_{symbol}"
            if any(alert.alert_id.startswith(existing_key) for alert in self.active_alerts.values()):
                return  # 避免重复警报
            
            alert = RiskAlert(
                alert_id=alert_id,
                alert_type=alert_type,
                risk_level=risk_level,
                symbol=symbol,
                message=message,
                threshold_value=threshold_value,
                current_value=current_value
            )
            
            self.active_alerts[alert_id] = alert
            
            # 广播警报
            await self._broadcast_alert(alert)
            
            # 记录日志
            self.logger.warning(f"🚨 {risk_level.name}风险警报: {message}")
            
        except Exception as e:
            self.logger.error(f"创建风险警报失败: {e}")
    
    async def _broadcast_alert(self, alert: RiskAlert):
        """广播风险警报"""
        try:
            alert_message = AgentMessage(
                receiver_id="*",  # 广播给所有Agent
                message_type=MessageType.EVENT,
                priority=8,  # 高优先级
                content={
                    "event_type": "risk_alert",
                    "alert": {
                        "alert_id": alert.alert_id,
                        "alert_type": alert.alert_type.value,
                        "risk_level": alert.risk_level.name,
                        "symbol": alert.symbol,
                        "message": alert.message,
                        "threshold_value": alert.threshold_value,
                        "current_value": alert.current_value,
                        "created_at": alert.created_at.isoformat()
                    }
                }
            )
            
            await self.send_message(alert_message)
            
        except Exception as e:
            self.logger.error(f"广播警报失败: {e}")
    
    async def _validate_trade_risk(self, symbol: str, size: float, action: str) -> Dict[str, Any]:
        """验证交易风险"""
        try:
            validation_result = {
                "approved": True,
                "risk_level": "LOW",
                "warnings": [],
                "blocks": []
            }
            
            # 检查仓位大小限制
            current_position = self.position_data.get(symbol, {}).get("size", 0)
            new_position_size = abs(current_position + (size if action == "buy" else -size))
            max_position = self.risk_thresholds["max_position_size"]
            
            if new_position_size > max_position:
                validation_result["blocks"].append(f"交易后仓位将超过限制: {new_position_size:.1%} > {max_position:.1%}")
                validation_result["approved"] = False
            
            # 检查风险指标
            risk_metrics = self.risk_metrics_cache.get(symbol)
            if risk_metrics:
                if risk_metrics.risk_level.value >= 4:  # HIGH或更高
                    validation_result["warnings"].append(f"标的风险等级较高: {risk_metrics.risk_level.name}")
                    validation_result["risk_level"] = "HIGH"
                
                if risk_metrics.liquidity_score < self.risk_thresholds["min_liquidity_score"]:
                    validation_result["warnings"].append(f"流动性不足: {risk_metrics.liquidity_score:.2f}")
            
            # 检查投资组合风险
            if self.portfolio_risk_cache:
                if self.portfolio_risk_cache.portfolio_var_95 > self.risk_thresholds["max_portfolio_var"]:
                    validation_result["warnings"].append("投资组合VaR已接近上限")
            
            return validation_result
            
        except Exception as e:
            return {
                "approved": False,
                "risk_level": "UNKNOWN",
                "warnings": [],
                "blocks": [f"风险验证失败: {str(e)}"]
            }
    
    async def _get_risk_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """获取风险指标"""
        symbol = params.get("symbol")
        
        if symbol and symbol in self.risk_metrics_cache:
            metrics = self.risk_metrics_cache[symbol]
            return {
                "symbol": symbol,
                "risk_metrics": {
                    "volatility_30d": metrics.volatility_30d,
                    "var_95": metrics.var_95,
                    "var_99": metrics.var_99,
                    "max_drawdown_30d": metrics.max_drawdown_30d,
                    "liquidity_score": metrics.liquidity_score,
                    "overall_risk_score": metrics.overall_risk_score,
                    "risk_level": metrics.risk_level.name
                },
                "timestamp": metrics.timestamp.isoformat()
            }
        else:
            # 返回所有风险指标
            all_metrics = {}
            for sym, metrics in self.risk_metrics_cache.items():
                all_metrics[sym] = {
                    "volatility_30d": metrics.volatility_30d,
                    "var_95": metrics.var_95,
                    "risk_level": metrics.risk_level.name,
                    "overall_risk_score": metrics.overall_risk_score
                }
            return {"all_metrics": all_metrics}
    
    async def _get_portfolio_risk(self) -> Dict[str, Any]:
        """获取投资组合风险"""
        if self.portfolio_risk_cache:
            return {
                "portfolio_risk": {
                    "portfolio_var_95": self.portfolio_risk_cache.portfolio_var_95,
                    "portfolio_volatility": self.portfolio_risk_cache.portfolio_volatility,
                    "max_position_weight": self.portfolio_risk_cache.max_position_weight,
                    "diversification_ratio": self.portfolio_risk_cache.diversification_ratio,
                    "portfolio_liquidity_score": self.portfolio_risk_cache.portfolio_liquidity_score,
                    "overall_risk_level": self.portfolio_risk_cache.overall_risk_level.name
                },
                "timestamp": self.portfolio_risk_cache.timestamp.isoformat()
            }
        else:
            return {"error": "投资组合风险数据不可用"}
    
    async def _get_active_alerts(self) -> Dict[str, Any]:
        """获取活跃警报"""
        alerts = []
        for alert in self.active_alerts.values():
            alerts.append({
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type.value,
                "risk_level": alert.risk_level.name,
                "symbol": alert.symbol,
                "message": alert.message,
                "created_at": alert.created_at.isoformat(),
                "acknowledged": alert.acknowledged
            })
        
        return {
            "active_alerts": alerts,
            "count": len(alerts)
        }