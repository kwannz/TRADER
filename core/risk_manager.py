"""
高级风险管理系统

实现实时风险监控和管理：
- 实时仓位监控
- 动态风险评估
- 自动风控触发
- 极端市场保护
- 流动性风险管控
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np

from .ai_engine import ai_engine
from .trading_simulator import trading_simulator
from .data_manager import data_manager
from .unified_logger import get_logger, LogCategory

logger = get_logger()

@dataclass
class RiskMetrics:
    """风险指标"""
    var_95: float  # 95% Value at Risk
    var_99: float  # 99% Value at Risk
    max_drawdown: float
    sharpe_ratio: float
    volatility: float
    beta: float
    tracking_error: float
    information_ratio: float

@dataclass
class PositionRisk:
    """仓位风险"""
    symbol: str
    position_size: float
    market_value: float
    unrealized_pnl: float
    position_ratio: float  # 占总资产比例
    concentration_risk: float
    liquidity_score: float
    correlation_risk: float

@dataclass
class RiskAlert:
    """风险警报"""
    alert_id: str
    risk_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    symbol: Optional[str]
    message: str
    threshold_value: float
    current_value: float
    timestamp: datetime
    action_required: bool
    suggested_action: str

class RiskCalculator:
    """风险计算器"""
    
    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        
    def calculate_var(self, returns: np.ndarray, confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, float]:
        """计算Value at Risk"""
        try:
            if len(returns) < 30:
                return {f'var_{int(cl*100)}': 0.0 for cl in confidence_levels}
            
            # 历史模拟法计算VaR
            sorted_returns = np.sort(returns)
            var_results = {}
            
            for cl in confidence_levels:
                index = int((1 - cl) * len(sorted_returns))
                var_value = abs(sorted_returns[index]) if index < len(sorted_returns) else 0.0
                var_results[f'var_{int(cl*100)}'] = var_value
            
            return var_results
            
        except Exception as e:
            logger.error(f"计算VaR失败: {e}")
            return {f'var_{int(cl*100)}': 0.0 for cl in confidence_levels}
    
    def calculate_drawdown(self, equity_curve: np.ndarray) -> Tuple[float, float]:
        """计算最大回撤"""
        try:
            if len(equity_curve) < 2:
                return 0.0, 0.0
            
            # 计算累积最高点
            peak = np.maximum.accumulate(equity_curve)
            
            # 计算回撤
            drawdown = (peak - equity_curve) / peak
            
            max_drawdown = np.max(drawdown)
            current_drawdown = drawdown[-1] if len(drawdown) > 0 else 0.0
            
            return max_drawdown, current_drawdown
            
        except Exception as e:
            logger.error(f"计算回撤失败: {e}")
            return 0.0, 0.0
    
    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        try:
            if len(returns) < 2:
                return 0.0
            
            excess_returns = returns - risk_free_rate / 252  # 日化无风险利率
            
            if np.std(excess_returns) == 0:
                return 0.0
            
            sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            return sharpe
            
        except Exception as e:
            logger.error(f"计算夏普比率失败: {e}")
            return 0.0
    
    def calculate_beta(self, asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """计算Beta系数"""
        try:
            if len(asset_returns) < 30 or len(market_returns) < 30:
                return 1.0
            
            # 确保长度一致
            min_length = min(len(asset_returns), len(market_returns))
            asset_returns = asset_returns[-min_length:]
            market_returns = market_returns[-min_length:]
            
            covariance = np.cov(asset_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            
            if market_variance == 0:
                return 1.0
            
            beta = covariance / market_variance
            return beta
            
        except Exception as e:
            logger.error(f"计算Beta失败: {e}")
            return 1.0
    
    def calculate_correlation_matrix(self, returns_data: Dict[str, np.ndarray]) -> np.ndarray:
        """计算相关性矩阵"""
        try:
            symbols = list(returns_data.keys())
            if len(symbols) < 2:
                return np.eye(len(symbols))
            
            # 构建收益率矩阵
            min_length = min(len(returns) for returns in returns_data.values())
            returns_matrix = np.array([
                returns_data[symbol][-min_length:] for symbol in symbols
            ])
            
            # 计算相关性矩阵
            correlation_matrix = np.corrcoef(returns_matrix)
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"计算相关性矩阵失败: {e}")
            return np.eye(len(returns_data))

class AdvancedRiskManager:
    """高级风险管理器"""
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # 风险参数
        self.max_portfolio_var = 0.05  # 最大组合VaR 5%
        self.max_single_position = 0.2  # 单个仓位最大20%
        self.max_sector_concentration = 0.4  # 单个板块最大40%
        self.max_correlation = 0.7  # 最大相关性
        self.max_drawdown_limit = 0.15  # 最大回撤限制15%
        self.liquidity_threshold = 0.3  # 流动性阈值
        
        # 风险计算器
        self.risk_calculator = RiskCalculator()
        
        # 数据存储
        self.portfolio_history = deque(maxlen=1000)
        self.returns_history = {}  # symbol -> deque of returns
        self.position_data = {}  # symbol -> PositionRisk
        self.active_alerts = {}  # alert_id -> RiskAlert
        
        # 运行状态
        self.is_monitoring = False
        self.monitoring_tasks = []
        
        # 统计信息
        self.risk_stats = {
            'total_alerts': 0,
            'critical_alerts': 0,
            'positions_closed': 0,
            'emergency_stops': 0
        }
        
        logger.info("高级风险管理器初始化完成")
    
    async def start_monitoring(self) -> None:
        """启动风险监控"""
        if self.is_monitoring:
            return
        
        try:
            self.is_monitoring = True
            
            # 订阅市场数据
            market_simulator.subscribe_tick_data(self._on_price_update)
            
            # 启动监控任务
            self.monitoring_tasks = [
                asyncio.create_task(self._portfolio_monitoring_loop()),
                asyncio.create_task(self._position_risk_loop()),
                asyncio.create_task(self._liquidity_monitoring_loop()),
                asyncio.create_task(self._correlation_monitoring_loop()),
                asyncio.create_task(self._emergency_monitoring_loop())
            ]
            
            logger.info("风险监控系统已启动")
            
        except Exception as e:
            logger.error(f"启动风险监控失败: {e}")
            await self.stop_monitoring()
    
    async def stop_monitoring(self) -> None:
        """停止风险监控"""
        try:
            self.is_monitoring = False
            
            # 取消所有监控任务
            for task in self.monitoring_tasks:
                if not task.done():
                    task.cancel()
            
            if self.monitoring_tasks:
                await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
            
            self.monitoring_tasks = []
            logger.info("风险监控系统已停止")
            
        except Exception as e:
            logger.error(f"停止风险监控失败: {e}")
    
    async def _on_price_update(self, tick) -> None:
        """处理价格更新"""
        try:
            symbol = tick.symbol
            price = tick.price
            
            # 更新仓位市值
            if symbol in self.position_data:
                position = self.position_data[symbol]
                old_value = position.market_value
                position.market_value = position.position_size * price
                
                # 计算未实现盈亏变化
                value_change = position.market_value - old_value
                position.unrealized_pnl += value_change
                
                # 更新仓位比例
                total_portfolio_value = self._calculate_total_portfolio_value()
                if total_portfolio_value > 0:
                    position.position_ratio = abs(position.market_value) / total_portfolio_value
                
                # 检查单仓位风险
                await self._check_position_risk(position)
            
        except Exception as e:
            logger.error(f"处理价格更新失败: {e}")
    
    async def _portfolio_monitoring_loop(self) -> None:
        """投资组合监控循环"""
        while self.is_monitoring:
            try:
                # 计算投资组合风险指标
                portfolio_value = self._calculate_total_portfolio_value()
                self.portfolio_history.append({
                    'timestamp': datetime.utcnow(),
                    'value': portfolio_value,
                    'cash': self.current_balance
                })
                
                # 计算投资组合收益率
                if len(self.portfolio_history) >= 2:
                    returns = self._calculate_portfolio_returns()
                    
                    # 计算风险指标
                    risk_metrics = await self._calculate_portfolio_risk_metrics(returns)
                    
                    # 检查投资组合层面风险
                    await self._check_portfolio_risk(risk_metrics)
                
                await asyncio.sleep(60)  # 每分钟检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"投资组合监控循环错误: {e}")
                await asyncio.sleep(30)
    
    async def _position_risk_loop(self) -> None:
        """仓位风险监控循环"""
        while self.is_monitoring:
            try:
                # 检查所有仓位风险
                for symbol, position in self.position_data.items():
                    await self._comprehensive_position_analysis(position)
                
                await asyncio.sleep(30)  # 每30秒检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"仓位风险监控循环错误: {e}")
                await asyncio.sleep(30)
    
    async def _liquidity_monitoring_loop(self) -> None:
        """流动性监控循环"""
        while self.is_monitoring:
            try:
                # 评估各仓位流动性
                for symbol, position in self.position_data.items():
                    liquidity_score = await self._assess_liquidity(symbol)
                    position.liquidity_score = liquidity_score
                    
                    # 检查流动性风险
                    if liquidity_score < self.liquidity_threshold:
                        await self._handle_liquidity_risk(position)
                
                await asyncio.sleep(300)  # 每5分钟检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"流动性监控循环错误: {e}")
                await asyncio.sleep(60)
    
    async def _correlation_monitoring_loop(self) -> None:
        """相关性监控循环"""
        while self.is_monitoring:
            try:
                if len(self.position_data) >= 2:
                    # 计算仓位间相关性
                    correlation_matrix = await self._calculate_position_correlations()
                    
                    # 检查过高相关性
                    await self._check_correlation_risk(correlation_matrix)
                
                await asyncio.sleep(600)  # 每10分钟检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"相关性监控循环错误: {e}")
                await asyncio.sleep(120)
    
    async def _emergency_monitoring_loop(self) -> None:
        """紧急风控监控循环"""
        while self.is_monitoring:
            try:
                # 检查极端市场条件
                await self._check_extreme_market_conditions()
                
                # 检查系统性风险
                await self._check_systemic_risk()
                
                await asyncio.sleep(10)  # 每10秒检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"紧急监控循环错误: {e}")
                await asyncio.sleep(30)
    
    def update_position(self, symbol: str, position_size: float, entry_price: float) -> None:
        """更新仓位信息"""
        try:
            current_price = market_simulator.get_current_prices().get(symbol, entry_price)
            market_value = position_size * current_price
            unrealized_pnl = (current_price - entry_price) * position_size
            
            total_portfolio_value = self._calculate_total_portfolio_value()
            position_ratio = abs(market_value) / max(total_portfolio_value, self.initial_balance)
            
            self.position_data[symbol] = PositionRisk(
                symbol=symbol,
                position_size=position_size,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                position_ratio=position_ratio,
                concentration_risk=position_ratio,
                liquidity_score=1.0,  # 默认高流动性
                correlation_risk=0.0   # 待计算
            )
            
        except Exception as e:
            logger.error(f"更新仓位失败: {e}")
    
    def close_position(self, symbol: str) -> None:
        """关闭仓位"""
        try:
            if symbol in self.position_data:
                position = self.position_data[symbol]
                
                # 更新现金余额
                self.current_balance += position.unrealized_pnl
                
                # 移除仓位记录
                del self.position_data[symbol]
                
                logger.info(f"仓位已关闭: {symbol}")
                
        except Exception as e:
            logger.error(f"关闭仓位失败: {e}")
    
    def _calculate_total_portfolio_value(self) -> float:
        """计算总投资组合价值"""
        try:
            total_market_value = sum(position.market_value for position in self.position_data.values())
            return self.current_balance + total_market_value
        except Exception as e:
            logger.error(f"计算投资组合价值失败: {e}")
            return self.current_balance
    
    def _calculate_portfolio_returns(self) -> np.ndarray:
        """计算投资组合收益率"""
        try:
            if len(self.portfolio_history) < 2:
                return np.array([])
            
            values = [item['value'] for item in self.portfolio_history]
            returns = np.diff(values) / values[:-1]
            
            return returns[-252:]  # 最近一年的收益率
            
        except Exception as e:
            logger.error(f"计算投资组合收益率失败: {e}")
            return np.array([])
    
    async def _calculate_portfolio_risk_metrics(self, returns: np.ndarray) -> RiskMetrics:
        """计算投资组合风险指标"""
        try:
            if len(returns) < 30:
                return RiskMetrics(0, 0, 0, 0, 0, 1, 0, 0)
            
            # VaR计算
            var_results = self.risk_calculator.calculate_var(returns)
            
            # 其他风险指标
            volatility = np.std(returns) * np.sqrt(252)
            
            portfolio_values = np.array([item['value'] for item in self.portfolio_history])
            max_drawdown, current_drawdown = self.risk_calculator.calculate_drawdown(portfolio_values)
            
            sharpe_ratio = self.risk_calculator.calculate_sharpe_ratio(returns)
            
            # 基准收益率（简化为市场平均）
            market_returns = np.random.normal(0.0001, 0.02, len(returns))  # 模拟市场收益
            beta = self.risk_calculator.calculate_beta(returns, market_returns)
            
            tracking_error = np.std(returns - market_returns) * np.sqrt(252)
            information_ratio = np.mean(returns - market_returns) / max(tracking_error, 0.001) * np.sqrt(252)
            
            return RiskMetrics(
                var_95=var_results.get('var_95', 0),
                var_99=var_results.get('var_99', 0),
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                volatility=volatility,
                beta=beta,
                tracking_error=tracking_error,
                information_ratio=information_ratio
            )
            
        except Exception as e:
            logger.error(f"计算投资组合风险指标失败: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 1, 0, 0)
    
    async def _check_portfolio_risk(self, risk_metrics: RiskMetrics) -> None:
        """检查投资组合风险"""
        try:
            alerts_to_create = []
            
            # VaR风险检查
            if risk_metrics.var_95 > self.max_portfolio_var:
                alerts_to_create.append(self._create_alert(
                    risk_type="portfolio_var",
                    severity="high",
                    message=f"投资组合VaR(95%)过高: {risk_metrics.var_95:.2%}",
                    threshold_value=self.max_portfolio_var,
                    current_value=risk_metrics.var_95,
                    action_required=True,
                    suggested_action="减少仓位或对冲"
                ))
            
            # 最大回撤检查
            if risk_metrics.max_drawdown > self.max_drawdown_limit:
                alerts_to_create.append(self._create_alert(
                    risk_type="max_drawdown",
                    severity="critical",
                    message=f"最大回撤超限: {risk_metrics.max_drawdown:.2%}",
                    threshold_value=self.max_drawdown_limit,
                    current_value=risk_metrics.max_drawdown,
                    action_required=True,
                    suggested_action="立即减仓或停止交易"
                ))
            
            # 波动率检查
            if risk_metrics.volatility > 0.3:  # 年化波动率30%
                alerts_to_create.append(self._create_alert(
                    risk_type="high_volatility",
                    severity="medium",
                    message=f"投资组合波动率过高: {risk_metrics.volatility:.2%}",
                    threshold_value=0.3,
                    current_value=risk_metrics.volatility,
                    action_required=False,
                    suggested_action="考虑分散投资"
                ))
            
            # 创建警报
            for alert in alerts_to_create:
                await self._trigger_alert(alert)
                
        except Exception as e:
            logger.error(f"检查投资组合风险失败: {e}")
    
    async def _check_position_risk(self, position: PositionRisk) -> None:
        """检查单个仓位风险"""
        try:
            alerts_to_create = []
            
            # 仓位比例检查
            if position.position_ratio > self.max_single_position:
                alerts_to_create.append(self._create_alert(
                    risk_type="position_concentration",
                    severity="high",
                    symbol=position.symbol,
                    message=f"{position.symbol}仓位比例过高: {position.position_ratio:.2%}",
                    threshold_value=self.max_single_position,
                    current_value=position.position_ratio,
                    action_required=True,
                    suggested_action="减少仓位"
                ))
            
            # 未实现亏损检查
            loss_ratio = position.unrealized_pnl / max(abs(position.market_value), 1)
            if loss_ratio < -0.1:  # 亏损超过10%
                severity = "critical" if loss_ratio < -0.2 else "high"
                alerts_to_create.append(self._create_alert(
                    risk_type="position_loss",
                    severity=severity,
                    symbol=position.symbol,
                    message=f"{position.symbol}亏损过大: {loss_ratio:.2%}",
                    threshold_value=-0.1,
                    current_value=loss_ratio,
                    action_required=True,
                    suggested_action="考虑止损"
                ))
            
            # 创建警报
            for alert in alerts_to_create:
                await self._trigger_alert(alert)
                
        except Exception as e:
            logger.error(f"检查仓位风险失败: {e}")
    
    async def _comprehensive_position_analysis(self, position: PositionRisk) -> None:
        """综合仓位分析"""
        try:
            # 基础风险检查
            await self._check_position_risk(position)
            
            # 技术面风险检查
            await self._check_technical_risk(position)
            
            # 基本面风险检查（如果有新闻）
            await self._check_fundamental_risk(position)
            
        except Exception as e:
            logger.error(f"综合仓位分析失败: {e}")
    
    async def _check_technical_risk(self, position: PositionRisk) -> None:
        """检查技术面风险"""
        try:
            # 获取价格历史
            symbol = position.symbol
            current_prices = market_simulator.get_current_prices()
            current_price = current_prices.get(symbol, 0)
            
            if current_price == 0:
                return
            
            # 检查价格变动幅度
            if symbol in self.returns_history and len(self.returns_history[symbol]) > 0:
                recent_return = self.returns_history[symbol][-1]
                
                # 异常波动检查
                if abs(recent_return) > 0.05:  # 5%的异常波动
                    await self._trigger_alert(self._create_alert(
                        risk_type="abnormal_volatility",
                        severity="medium",
                        symbol=symbol,
                        message=f"{symbol}价格异常波动: {recent_return:.2%}",
                        threshold_value=0.05,
                        current_value=abs(recent_return),
                        action_required=False,
                        suggested_action="密切关注"
                    ))
                    
        except Exception as e:
            logger.error(f"检查技术面风险失败: {e}")
    
    async def _check_fundamental_risk(self, position: PositionRisk) -> None:
        """检查基本面风险"""
        try:
            # 获取最近新闻
            recent_news = await data_manager.get_recent_news(hours=6)
            
            symbol_news = [
                news for news in recent_news 
                if position.symbol.split('/')[0] in news.get('title', '').upper()
            ]
            
            # 检查负面新闻
            negative_news = [
                news for news in symbol_news 
                if news.get('sentiment') == 'negative'
            ]
            
            if len(negative_news) > 2:  # 多条负面新闻
                await self._trigger_alert(self._create_alert(
                    risk_type="negative_news",
                    severity="medium",
                    symbol=position.symbol,
                    message=f"{position.symbol}存在多条负面新闻",
                    threshold_value=2,
                    current_value=len(negative_news),
                    action_required=False,
                    suggested_action="评估基本面变化"
                ))
                
        except Exception as e:
            logger.error(f"检查基本面风险失败: {e}")
    
    async def _assess_liquidity(self, symbol: str) -> float:
        """评估流动性"""
        try:
            # 简化的流动性评估
            # 实际应用中应考虑交易量、买卖价差、市场深度等
            
            market_summary = market_simulator.get_market_summary()
            if symbol in market_summary:
                volume_24h = market_summary[symbol]['volume_24h']
                
                # 基于交易量评估流动性
                if volume_24h > 1000000:  # 高交易量
                    return 1.0
                elif volume_24h > 100000:  # 中等交易量
                    return 0.7
                else:  # 低交易量
                    return 0.3
            
            return 0.5  # 默认中等流动性
            
        except Exception as e:
            logger.error(f"评估流动性失败: {e}")
            return 0.5
    
    async def _handle_liquidity_risk(self, position: PositionRisk) -> None:
        """处理流动性风险"""
        try:
            await self._trigger_alert(self._create_alert(
                risk_type="liquidity_risk",
                severity="high",
                symbol=position.symbol,
                message=f"{position.symbol}流动性不足: {position.liquidity_score:.2f}",
                threshold_value=self.liquidity_threshold,
                current_value=position.liquidity_score,
                action_required=True,
                suggested_action="考虑分批减仓"
            ))
            
        except Exception as e:
            logger.error(f"处理流动性风险失败: {e}")
    
    async def _calculate_position_correlations(self) -> np.ndarray:
        """计算仓位间相关性"""
        try:
            if len(self.position_data) < 2:
                return np.eye(1)
            
            symbols = list(self.position_data.keys())
            returns_data = {}
            
            # 收集收益率数据
            for symbol in symbols:
                if symbol in self.returns_history:
                    returns_data[symbol] = np.array(list(self.returns_history[symbol]))
            
            if len(returns_data) < 2:
                return np.eye(len(symbols))
            
            # 计算相关性矩阵
            correlation_matrix = self.risk_calculator.calculate_correlation_matrix(returns_data)
            
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"计算仓位相关性失败: {e}")
            return np.eye(len(self.position_data))
    
    async def _check_correlation_risk(self, correlation_matrix: np.ndarray) -> None:
        """检查相关性风险"""
        try:
            if correlation_matrix.shape[0] < 2:
                return
            
            # 找到高相关性对
            symbols = list(self.position_data.keys())
            high_correlations = []
            
            for i in range(len(symbols)):
                for j in range(i+1, len(symbols)):
                    correlation = correlation_matrix[i, j]
                    
                    if abs(correlation) > self.max_correlation:
                        high_correlations.append((symbols[i], symbols[j], correlation))
            
            # 创建相关性风险警报
            for symbol1, symbol2, corr in high_correlations:
                await self._trigger_alert(self._create_alert(
                    risk_type="high_correlation",
                    severity="medium",
                    message=f"{symbol1}与{symbol2}相关性过高: {corr:.2f}",
                    threshold_value=self.max_correlation,
                    current_value=abs(corr),
                    action_required=False,
                    suggested_action="考虑分散投资"
                ))
                
        except Exception as e:
            logger.error(f"检查相关性风险失败: {e}")
    
    async def _check_extreme_market_conditions(self) -> None:
        """检查极端市场条件"""
        try:
            market_summary = market_simulator.get_market_summary()
            
            # 检查整体市场波动
            price_changes = [
                abs(data['change_24h']) for data in market_summary.values()
            ]
            
            if price_changes:
                avg_volatility = np.mean(price_changes)
                max_volatility = max(price_changes)
                
                # 极端波动检查
                if max_volatility > 0.2:  # 单日波动超过20%
                    await self._trigger_alert(self._create_alert(
                        risk_type="extreme_volatility",
                        severity="critical",
                        message=f"市场出现极端波动: {max_volatility:.2%}",
                        threshold_value=0.2,
                        current_value=max_volatility,
                        action_required=True,
                        suggested_action="考虑紧急避险"
                    ))
                
                # 市场恐慌检查
                volatile_assets = sum(1 for change in price_changes if change > 0.1)
                if volatile_assets > len(price_changes) * 0.5:  # 超过一半资产大幅波动
                    await self._trigger_alert(self._create_alert(
                        risk_type="market_panic",
                        severity="high",
                        message=f"市场恐慌状态: {volatile_assets}个资产大幅波动",
                        threshold_value=len(price_changes) * 0.5,
                        current_value=volatile_assets,
                        action_required=True,
                        suggested_action="启动防御性策略"
                    ))
                    
        except Exception as e:
            logger.error(f"检查极端市场条件失败: {e}")
    
    async def _check_systemic_risk(self) -> None:
        """检查系统性风险"""
        try:
            # 检查总体投资组合价值变化
            if len(self.portfolio_history) >= 2:
                current_value = self.portfolio_history[-1]['value']
                prev_value = self.portfolio_history[-2]['value']
                
                value_change = (current_value - prev_value) / prev_value
                
                # 系统性损失检查
                if value_change < -0.05:  # 总价值下跌超过5%
                    severity = "critical" if value_change < -0.1 else "high"
                    
                    await self._trigger_alert(self._create_alert(
                        risk_type="systemic_loss",
                        severity=severity,
                        message=f"投资组合系统性损失: {value_change:.2%}",
                        threshold_value=-0.05,
                        current_value=value_change,
                        action_required=True,
                        suggested_action="评估整体策略并考虑减仓"
                    ))
                    
        except Exception as e:
            logger.error(f"检查系统性风险失败: {e}")
    
    def _create_alert(self, risk_type: str, severity: str, message: str,
                     threshold_value: float, current_value: float,
                     action_required: bool, suggested_action: str,
                     symbol: Optional[str] = None) -> RiskAlert:
        """创建风险警报"""
        import uuid
        
        return RiskAlert(
            alert_id=str(uuid.uuid4()),
            risk_type=risk_type,
            severity=severity,
            symbol=symbol,
            message=message,
            threshold_value=threshold_value,
            current_value=current_value,
            timestamp=datetime.utcnow(),
            action_required=action_required,
            suggested_action=suggested_action
        )
    
    async def _trigger_alert(self, alert: RiskAlert) -> None:
        """触发风险警报"""
        try:
            # 添加到活跃警报
            self.active_alerts[alert.alert_id] = alert
            
            # 更新统计
            self.risk_stats['total_alerts'] += 1
            if alert.severity == 'critical':
                self.risk_stats['critical_alerts'] += 1
            
            # 记录警报
            logger.warning(f"风险警报 [{alert.severity.upper()}]: {alert.message}")
            
            # 自动处理关键警报
            if alert.severity == 'critical' and alert.action_required:
                await self._auto_handle_critical_alert(alert)
            
            # 保存到数据库
            if hasattr(data_manager, '_initialized') and data_manager._initialized:
                alert_data = {
                    'alert_id': alert.alert_id,
                    'risk_type': alert.risk_type,
                    'severity': alert.severity,
                    'symbol': alert.symbol,
                    'message': alert.message,
                    'threshold_value': alert.threshold_value,
                    'current_value': alert.current_value,
                    'action_required': alert.action_required,
                    'suggested_action': alert.suggested_action,
                    'timestamp': alert.timestamp
                }
                # 这里可以添加保存警报到数据库的逻辑
                
        except Exception as e:
            logger.error(f"触发风险警报失败: {e}")
    
    async def _auto_handle_critical_alert(self, alert: RiskAlert) -> None:
        """自动处理关键警报"""
        try:
            if alert.risk_type == "max_drawdown":
                # 最大回撤触发紧急停止
                await self._emergency_stop_all_positions()
                
            elif alert.risk_type == "systemic_loss":
                # 系统性损失触发部分减仓
                await self._emergency_position_reduction(0.5)
                
            elif alert.risk_type == "position_loss" and alert.symbol:
                # 单仓位严重亏损触发止损
                await self._emergency_close_position(alert.symbol)
            
            self.risk_stats['emergency_stops'] += 1
            
        except Exception as e:
            logger.error(f"自动处理关键警报失败: {e}")
    
    async def _emergency_stop_all_positions(self) -> None:
        """紧急停止所有仓位"""
        try:
            logger.critical("执行紧急停止 - 关闭所有仓位")
            
            symbols_to_close = list(self.position_data.keys())
            for symbol in symbols_to_close:
                self.close_position(symbol)
            
            self.risk_stats['positions_closed'] += len(symbols_to_close)
            
        except Exception as e:
            logger.error(f"紧急停止失败: {e}")
    
    async def _emergency_position_reduction(self, reduction_ratio: float) -> None:
        """紧急减仓"""
        try:
            logger.critical(f"执行紧急减仓 - 减仓比例: {reduction_ratio:.0%}")
            
            for symbol, position in self.position_data.items():
                # 减少仓位大小
                position.position_size *= (1 - reduction_ratio)
                position.market_value *= (1 - reduction_ratio)
                
                # 更新现金余额
                self.current_balance += position.market_value * reduction_ratio
            
        except Exception as e:
            logger.error(f"紧急减仓失败: {e}")
    
    async def _emergency_close_position(self, symbol: str) -> None:
        """紧急关闭特定仓位"""
        try:
            logger.critical(f"紧急关闭仓位: {symbol}")
            self.close_position(symbol)
            self.risk_stats['positions_closed'] += 1
            
        except Exception as e:
            logger.error(f"紧急关闭仓位失败: {e}")
    
    def get_risk_report(self) -> Dict[str, Any]:
        """获取风险报告"""
        try:
            total_value = self._calculate_total_portfolio_value()
            
            # 仓位风险摘要
            position_summary = {}
            for symbol, position in self.position_data.items():
                position_summary[symbol] = {
                    'market_value': position.market_value,
                    'position_ratio': position.position_ratio,
                    'unrealized_pnl': position.unrealized_pnl,
                    'liquidity_score': position.liquidity_score
                }
            
            # 活跃警报摘要
            alert_summary = {
                'total_alerts': len(self.active_alerts),
                'critical_alerts': len([a for a in self.active_alerts.values() if a.severity == 'critical']),
                'high_alerts': len([a for a in self.active_alerts.values() if a.severity == 'high']),
                'medium_alerts': len([a for a in self.active_alerts.values() if a.severity == 'medium'])
            }
            
            return {
                'portfolio_value': total_value,
                'cash_balance': self.current_balance,
                'total_positions': len(self.position_data),
                'position_summary': position_summary,
                'alert_summary': alert_summary,
                'risk_stats': self.risk_stats,
                'monitoring_status': self.is_monitoring,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取风险报告失败: {e}")
            return {}
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃警报"""
        try:
            return [
                {
                    'alert_id': alert.alert_id,
                    'risk_type': alert.risk_type,
                    'severity': alert.severity,
                    'symbol': alert.symbol,
                    'message': alert.message,
                    'current_value': alert.current_value,
                    'threshold_value': alert.threshold_value,
                    'action_required': alert.action_required,
                    'suggested_action': alert.suggested_action,
                    'timestamp': alert.timestamp.isoformat(),
                    'age_minutes': (datetime.utcnow() - alert.timestamp).total_seconds() / 60
                }
                for alert in self.active_alerts.values()
            ]
            
        except Exception as e:
            logger.error(f"获取活跃警报失败: {e}")
            return []

# 全局高级风险管理器实例
advanced_risk_manager = AdvancedRiskManager()