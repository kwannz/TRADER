"""
AI驱动的交易决策系统

基于深度学习和大语言模型的智能交易决策：
- 多模态市场数据分析
- 实时交易信号生成
- 智能风险管理
- 自适应策略优化
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import numpy as np
from dataclasses import dataclass
from collections import deque

from .ai_engine import ai_engine
from .market_simulator import market_simulator, MarketTick, KLineData
from .data_manager import data_manager
from .strategy_engine import strategy_engine, BaseStrategy
from core.unified_logger import get_logger, LogCategory

logger = get_logger()

@dataclass
class TradingSignal:
    """交易信号"""
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0-1
    price_target: float
    stop_loss: float
    take_profit: float
    position_size: float
    reasoning: str
    timestamp: datetime
    strategy_type: str

@dataclass
class MarketAnalysis:
    """市场分析结果"""
    trend_direction: str  # 'up', 'down', 'sideways'
    trend_strength: float  # 0-1
    support_level: float
    resistance_level: float
    volatility_level: str  # 'low', 'medium', 'high'
    market_sentiment: str  # 'bullish', 'bearish', 'neutral'
    key_factors: List[str]
    timestamp: datetime

class TechnicalAnalyzer:
    """技术分析器"""
    
    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        
    def calculate_indicators(self, prices: List[float]) -> Dict[str, float]:
        """计算技术指标"""
        try:
            if len(prices) < 20:
                return {}
            
            prices_array = np.array(prices)
            
            # 简单移动平均线
            sma_20 = np.mean(prices_array[-20:])
            sma_50 = np.mean(prices_array[-50:]) if len(prices) >= 50 else sma_20
            
            # RSI
            rsi = self._calculate_rsi(prices_array)
            
            # MACD
            macd, macd_signal = self._calculate_macd(prices_array)
            
            # 布林带
            bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(prices_array)
            
            # 支撑阻力位
            support, resistance = self._calculate_support_resistance(prices_array)
            
            return {
                "sma_20": sma_20,
                "sma_50": sma_50,
                "rsi": rsi,
                "macd": macd,
                "macd_signal": macd_signal,
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
                "bb_middle": bb_middle,
                "support": support,
                "resistance": resistance,
                "current_price": prices[-1]
            }
            
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            return {}
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """计算RSI指标"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            deltas = np.diff(prices)
            gains = deltas.copy()
            losses = deltas.copy()
            
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            losses = np.abs(losses)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"计算RSI失败: {e}")
            return 50.0
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """计算MACD指标"""
        try:
            if len(prices) < slow:
                return 0.0, 0.0
            
            # 计算EMA
            def ema(data, span):
                return data.ewm(span=span).mean()
            
            prices_series = prices
            ema_fast = np.mean(prices_series[-fast:])
            ema_slow = np.mean(prices_series[-slow:])
            
            macd_line = ema_fast - ema_slow
            
            # 简化信号线计算
            signal_line = macd_line * 0.9  # 简化
            
            return macd_line, signal_line
            
        except Exception as e:
            logger.error(f"计算MACD失败: {e}")
            return 0.0, 0.0
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[float, float, float]:
        """计算布林带"""
        try:
            if len(prices) < period:
                current_price = prices[-1]
                return current_price * 1.02, current_price * 0.98, current_price
            
            recent_prices = prices[-period:]
            middle = np.mean(recent_prices)
            std = np.std(recent_prices)
            
            upper = middle + (std * std_dev)
            lower = middle - (std * std_dev)
            
            return upper, lower, middle
            
        except Exception as e:
            logger.error(f"计算布林带失败: {e}")
            current_price = prices[-1] if len(prices) > 0 else 0
            return current_price * 1.02, current_price * 0.98, current_price
    
    def _calculate_support_resistance(self, prices: np.ndarray, window: int = 20) -> Tuple[float, float]:
        """计算支撑阻力位"""
        try:
            if len(prices) < window:
                current_price = prices[-1]
                return current_price * 0.98, current_price * 1.02
            
            recent_prices = prices[-window:]
            
            # 简化的支撑阻力计算
            highs = []
            lows = []
            
            for i in range(2, len(recent_prices) - 2):
                # 寻找局部高点
                if (recent_prices[i] > recent_prices[i-1] and 
                    recent_prices[i] > recent_prices[i+1] and
                    recent_prices[i] > recent_prices[i-2] and 
                    recent_prices[i] > recent_prices[i+2]):
                    highs.append(recent_prices[i])
                
                # 寻找局部低点
                if (recent_prices[i] < recent_prices[i-1] and 
                    recent_prices[i] < recent_prices[i+1] and
                    recent_prices[i] < recent_prices[i-2] and 
                    recent_prices[i] < recent_prices[i+2]):
                    lows.append(recent_prices[i])
            
            resistance = np.mean(highs) if highs else np.max(recent_prices)
            support = np.mean(lows) if lows else np.min(recent_prices)
            
            return support, resistance
            
        except Exception as e:
            logger.error(f"计算支撑阻力失败: {e}")
            current_price = prices[-1] if len(prices) > 0 else 0
            return current_price * 0.98, current_price * 1.02

class AITradingEngine:
    """AI交易引擎主类"""
    
    def __init__(self):
        self.technical_analyzer = TechnicalAnalyzer()
        self.price_history = {}  # symbol -> deque of prices
        self.active_signals = {}  # symbol -> TradingSignal
        self.market_analysis_cache = {}  # symbol -> MarketAnalysis
        
        # AI模型配置
        self.ai_models = {
            'trend_predictor': 'deepseek',
            'sentiment_analyzer': 'gemini',
            'risk_assessor': 'deepseek'
        }
        
        # 交易参数
        self.confidence_threshold = 0.7
        self.max_position_size = 0.1
        self.stop_loss_ratio = 0.02
        self.take_profit_ratio = 0.04
        
        # 运行状态
        self.is_running = False
        self.analysis_tasks = []
        
        # 性能统计
        self.signal_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'failed_signals': 0,
            'avg_confidence': 0.0
        }
        
        logger.info("AI交易引擎初始化完成")
    
    async def start_engine(self) -> None:
        """启动AI交易引擎"""
        if self.is_running:
            return
        
        try:
            self.is_running = True
            
            # 订阅市场数据
            market_simulator.subscribe_tick_data(self._on_tick_data)
            market_simulator.subscribe_kline_data(self._on_kline_data)
            
            # 启动分析任务
            self.analysis_tasks = [
                asyncio.create_task(self._market_analysis_loop()),
                asyncio.create_task(self._signal_generation_loop()),
                asyncio.create_task(self._signal_monitoring_loop())
            ]
            
            logger.info("AI交易引擎已启动")
            
        except Exception as e:
            logger.error(f"启动AI交易引擎失败: {e}")
            await self.stop_engine()
    
    async def stop_engine(self) -> None:
        """停止AI交易引擎"""
        try:
            self.is_running = False
            
            # 取消所有任务
            for task in self.analysis_tasks:
                if not task.done():
                    task.cancel()
            
            if self.analysis_tasks:
                await asyncio.gather(*self.analysis_tasks, return_exceptions=True)
            
            self.analysis_tasks = []
            logger.info("AI交易引擎已停止")
            
        except Exception as e:
            logger.error(f"停止AI交易引擎失败: {e}")
    
    async def _on_tick_data(self, tick: MarketTick) -> None:
        """处理Tick数据"""
        try:
            symbol = tick.symbol
            
            # 更新价格历史
            if symbol not in self.price_history:
                self.price_history[symbol] = deque(maxlen=1000)
            
            self.price_history[symbol].append(tick.price)
            
            # 实时分析触发
            if len(self.price_history[symbol]) >= 20:
                await self._analyze_price_movement(symbol, tick.price)
                
        except Exception as e:
            logger.error(f"处理Tick数据失败: {e}")
    
    async def _on_kline_data(self, kline: KLineData) -> None:
        """处理K线数据"""
        try:
            symbol = kline.symbol
            
            # 触发深度分析
            await self._deep_market_analysis(symbol, kline)
            
        except Exception as e:
            logger.error(f"处理K线数据失败: {e}")
    
    async def _market_analysis_loop(self) -> None:
        """市场分析循环"""
        while self.is_running:
            try:
                # 对所有跟踪的币种进行定期分析
                for symbol in self.price_history:
                    if len(self.price_history[symbol]) >= 50:
                        await self._comprehensive_market_analysis(symbol)
                
                # 每5分钟执行一次
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"市场分析循环错误: {e}")
                await asyncio.sleep(60)
    
    async def _signal_generation_loop(self) -> None:
        """信号生成循环"""
        while self.is_running:
            try:
                # 基于分析结果生成交易信号
                for symbol in self.market_analysis_cache:
                    analysis = self.market_analysis_cache[symbol]
                    
                    # 检查是否需要生成新信号
                    if self._should_generate_signal(symbol, analysis):
                        signal = await self._generate_trading_signal(symbol, analysis)
                        
                        if signal and signal.confidence >= self.confidence_threshold:
                            await self._execute_ai_signal(signal)
                
                # 每30秒检查一次
                await asyncio.sleep(30)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"信号生成循环错误: {e}")
                await asyncio.sleep(60)
    
    async def _signal_monitoring_loop(self) -> None:
        """信号监控循环"""
        while self.is_running:
            try:
                # 监控活跃信号的执行情况
                for symbol, signal in list(self.active_signals.items()):
                    await self._monitor_signal_performance(symbol, signal)
                
                # 每10秒检查一次
                await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"信号监控循环错误: {e}")
                await asyncio.sleep(30)
    
    async def _analyze_price_movement(self, symbol: str, current_price: float) -> None:
        """分析价格变动"""
        try:
            prices = list(self.price_history[symbol])
            if len(prices) < 20:
                return
            
            # 计算价格变化
            price_change_5m = (current_price - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
            price_change_1m = (current_price - prices[-1]) / prices[-1] if len(prices) >= 1 else 0
            
            # 检测异常价格移动
            if abs(price_change_1m) > 0.01:  # 1%的快速变动
                await self._handle_price_anomaly(symbol, current_price, price_change_1m)
                
            # 检测突破信号
            indicators = self.technical_analyzer.calculate_indicators(prices)
            if indicators:
                await self._check_breakout_signals(symbol, current_price, indicators)
                
        except Exception as e:
            logger.error(f"分析价格变动失败: {e}")
    
    async def _deep_market_analysis(self, symbol: str, kline: KLineData) -> None:
        """深度市场分析"""
        try:
            prices = list(self.price_history.get(symbol, []))
            if len(prices) < 50:
                return
            
            # 技术指标分析
            indicators = self.technical_analyzer.calculate_indicators(prices)
            
            # AI市场趋势分析
            market_context = {
                "symbol": symbol,
                "current_price": kline.close,
                "volume": kline.volume,
                "indicators": indicators,
                "price_history": prices[-20:]  # 最近20个价格点
            }
            
            ai_analysis = await ai_engine.analyze_market_trends([symbol])
            
            # 合成分析结果
            analysis = await self._synthesize_analysis(symbol, market_context, ai_analysis, indicators)
            
            if analysis:
                self.market_analysis_cache[symbol] = analysis
                
        except Exception as e:
            logger.error(f"深度市场分析失败: {e}")
    
    async def _comprehensive_market_analysis(self, symbol: str) -> None:
        """全面市场分析"""
        try:
            prices = list(self.price_history[symbol])
            
            # 多时间框架分析
            short_term_trend = self._calculate_trend(prices[-20:])  # 短期
            medium_term_trend = self._calculate_trend(prices[-50:])  # 中期
            long_term_trend = self._calculate_trend(prices[-100:]) if len(prices) >= 100 else medium_term_trend  # 长期
            
            # 技术指标
            indicators = self.technical_analyzer.calculate_indicators(prices)
            
            # AI情绪分析
            sentiment = await ai_engine.analyze_market_sentiment()
            
            # 新闻影响分析
            recent_news = await data_manager.get_recent_news(hours=24)
            news_impact = self._analyze_news_impact(recent_news, symbol)
            
            # 综合分析
            analysis = MarketAnalysis(
                trend_direction=self._determine_overall_trend(short_term_trend, medium_term_trend, long_term_trend),
                trend_strength=self._calculate_trend_strength(indicators),
                support_level=indicators.get('support', prices[-1] * 0.98),
                resistance_level=indicators.get('resistance', prices[-1] * 1.02),
                volatility_level=self._assess_volatility(prices),
                market_sentiment=sentiment.get('overall_sentiment', 'neutral'),
                key_factors=self._identify_key_factors(indicators, news_impact),
                timestamp=datetime.utcnow()
            )
            
            self.market_analysis_cache[symbol] = analysis
            
        except Exception as e:
            logger.error(f"全面市场分析失败: {e}")
    
    def _calculate_trend(self, prices: List[float]) -> str:
        """计算趋势方向"""
        if len(prices) < 10:
            return "sideways"
        
        # 线性回归斜率
        x = np.arange(len(prices))
        y = np.array(prices)
        
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > prices[0] * 0.001:  # 上涨超过0.1%
            return "up"
        elif slope < -prices[0] * 0.001:  # 下跌超过0.1%
            return "down"
        else:
            return "sideways"
    
    def _calculate_trend_strength(self, indicators: Dict[str, float]) -> float:
        """计算趋势强度"""
        try:
            strength_factors = []
            
            # RSI强度
            rsi = indicators.get('rsi', 50)
            if rsi > 70 or rsi < 30:
                strength_factors.append(0.8)
            else:
                strength_factors.append(0.3)
            
            # MACD强度
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            
            if abs(macd - macd_signal) > 0:
                strength_factors.append(0.7)
            else:
                strength_factors.append(0.2)
            
            # 价格位置强度
            current_price = indicators.get('current_price', 0)
            bb_upper = indicators.get('bb_upper', current_price)
            bb_lower = indicators.get('bb_lower', current_price)
            
            if current_price > bb_upper or current_price < bb_lower:
                strength_factors.append(0.9)
            else:
                strength_factors.append(0.4)
            
            return np.mean(strength_factors)
            
        except Exception as e:
            logger.error(f"计算趋势强度失败: {e}")
            return 0.5
    
    def _assess_volatility(self, prices: List[float]) -> str:
        """评估波动率"""
        if len(prices) < 20:
            return "medium"
        
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        if volatility > 0.02:
            return "high"
        elif volatility < 0.01:
            return "low"
        else:
            return "medium"
    
    def _determine_overall_trend(self, short: str, medium: str, long: str) -> str:
        """确定总体趋势"""
        trends = [short, medium, long]
        
        if trends.count("up") >= 2:
            return "up"
        elif trends.count("down") >= 2:
            return "down"
        else:
            return "sideways"
    
    def _identify_key_factors(self, indicators: Dict[str, float], news_impact: str) -> List[str]:
        """识别关键因素"""
        factors = []
        
        # 技术因素
        rsi = indicators.get('rsi', 50)
        if rsi > 70:
            factors.append("RSI超买")
        elif rsi < 30:
            factors.append("RSI超卖")
        
        current_price = indicators.get('current_price', 0)
        resistance = indicators.get('resistance', 0)
        support = indicators.get('support', 0)
        
        if abs(current_price - resistance) / current_price < 0.01:
            factors.append("接近阻力位")
        elif abs(current_price - support) / current_price < 0.01:
            factors.append("接近支撑位")
        
        # 新闻影响
        if news_impact != "neutral":
            factors.append(f"新闻影响: {news_impact}")
        
        return factors
    
    def _analyze_news_impact(self, news_list: List[Dict], symbol: str) -> str:
        """分析新闻影响"""
        if not news_list:
            return "neutral"
        
        # 简化新闻影响分析
        positive_count = 0
        negative_count = 0
        
        for news in news_list:
            sentiment = news.get('sentiment', 'neutral')
            if sentiment == 'positive':
                positive_count += 1
            elif sentiment == 'negative':
                negative_count += 1
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    async def _synthesize_analysis(self, symbol: str, market_context: Dict, ai_analysis: Dict, indicators: Dict) -> Optional[MarketAnalysis]:
        """合成分析结果"""
        try:
            # 综合AI分析和技术分析
            ai_trend = ai_analysis.get('overall_trend', 'sideways')
            
            # 技术趋势
            prices = market_context['price_history']
            tech_trend = self._calculate_trend(prices)
            
            # 最终趋势判断
            if ai_trend == tech_trend:
                final_trend = ai_trend
                trend_strength = 0.8
            else:
                final_trend = "sideways"
                trend_strength = 0.3
            
            analysis = MarketAnalysis(
                trend_direction=final_trend,
                trend_strength=trend_strength,
                support_level=indicators.get('support', market_context['current_price'] * 0.98),
                resistance_level=indicators.get('resistance', market_context['current_price'] * 1.02),
                volatility_level=self._assess_volatility(prices),
                market_sentiment=ai_analysis.get('market_sentiment', 'neutral'),
                key_factors=ai_analysis.get('key_factors', []),
                timestamp=datetime.utcnow()
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"合成分析结果失败: {e}")
            return None
    
    def _should_generate_signal(self, symbol: str, analysis: MarketAnalysis) -> bool:
        """判断是否应该生成信号"""
        try:
            # 检查是否已有活跃信号
            if symbol in self.active_signals:
                active_signal = self.active_signals[symbol]
                time_since_signal = (datetime.utcnow() - active_signal.timestamp).total_seconds()
                
                # 如果信号还很新（小于30分钟），不生成新信号
                if time_since_signal < 1800:
                    return False
            
            # 检查趋势强度
            if analysis.trend_strength < 0.6:
                return False
            
            # 检查趋势方向
            if analysis.trend_direction == "sideways":
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"判断信号生成失败: {e}")
            return False
    
    async def _generate_trading_signal(self, symbol: str, analysis: MarketAnalysis) -> Optional[TradingSignal]:
        """生成交易信号"""
        try:
            current_prices = market_simulator.get_current_prices()
            current_price = current_prices.get(symbol, 0)
            
            if current_price == 0:
                return None
            
            # 基于分析生成信号
            if analysis.trend_direction == "up":
                action = "buy"
                price_target = min(analysis.resistance_level, current_price * 1.03)
                stop_loss = max(analysis.support_level, current_price * (1 - self.stop_loss_ratio))
                take_profit = current_price * (1 + self.take_profit_ratio)
            elif analysis.trend_direction == "down":
                action = "sell"
                price_target = max(analysis.support_level, current_price * 0.97)
                stop_loss = min(analysis.resistance_level, current_price * (1 + self.stop_loss_ratio))
                take_profit = current_price * (1 - self.take_profit_ratio)
            else:
                return None
            
            # AI信心度评估
            ai_confidence = await ai_engine.evaluate_trading_opportunity(
                symbol, action, current_price, analysis.__dict__
            )
            
            confidence = min(analysis.trend_strength * ai_confidence.get('confidence', 0.5), 1.0)
            
            # 计算仓位大小
            position_size = min(self.max_position_size * confidence, self.max_position_size)
            
            # 生成推理说明
            reasoning = f"基于{analysis.trend_direction}趋势(强度:{analysis.trend_strength:.2f}), "
            reasoning += f"技术指标显示{analysis.volatility_level}波动性, "
            reasoning += f"市场情绪{analysis.market_sentiment}"
            
            if analysis.key_factors:
                reasoning += f", 关键因素: {', '.join(analysis.key_factors[:3])}"
            
            signal = TradingSignal(
                symbol=symbol,
                action=action,
                confidence=confidence,
                price_target=price_target,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                reasoning=reasoning,
                timestamp=datetime.utcnow(),
                strategy_type="ai_generated"
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"生成交易信号失败: {e}")
            return None
    
    async def _execute_ai_signal(self, signal: TradingSignal) -> None:
        """执行AI信号"""
        try:
            # 记录信号统计
            self.signal_stats['total_signals'] += 1
            self.signal_stats['avg_confidence'] = (
                (self.signal_stats['avg_confidence'] * (self.signal_stats['total_signals'] - 1) + signal.confidence) / 
                self.signal_stats['total_signals']
            )
            
            # 提交信号到策略引擎
            signal_data = {
                "type": signal.strategy_type,
                "side": signal.action,
                "price": signal.price_target,
                "quantity": signal.position_size,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "confidence": signal.confidence,
                "reasoning": signal.reasoning,
                "ai_generated": True
            }
            
            # 创建临时AI策略或直接提交信号
            await self._submit_ai_signal_to_engine(signal.symbol, signal_data)
            
            # 记录活跃信号
            self.active_signals[signal.symbol] = signal
            
            logger.info(f"AI信号已执行: {signal.symbol} {signal.action} @ {signal.price_target:.2f} (信心度: {signal.confidence:.2f})")
            
        except Exception as e:
            logger.error(f"执行AI信号失败: {e}")
    
    async def _submit_ai_signal_to_engine(self, symbol: str, signal_data: Dict) -> None:
        """向策略引擎提交AI信号"""
        try:
            # 查找或创建AI策略
            ai_strategy_id = None
            
            for strategy_id, strategy in strategy_engine.strategies.items():
                if (hasattr(strategy, 'config') and 
                    strategy.config.get('symbol') == symbol and
                    isinstance(strategy, AIStrategy)):
                    ai_strategy_id = strategy_id
                    break
            
            if ai_strategy_id:
                # 直接提交信号
                await strategy_engine.submit_signal(ai_strategy_id, signal_data)
            else:
                logger.warning(f"未找到{symbol}的AI策略，无法提交信号")
                
        except Exception as e:
            logger.error(f"提交AI信号到引擎失败: {e}")
    
    async def _monitor_signal_performance(self, symbol: str, signal: TradingSignal) -> None:
        """监控信号性能"""
        try:
            current_prices = market_simulator.get_current_prices()
            current_price = current_prices.get(symbol, 0)
            
            if current_price == 0:
                return
            
            # 检查信号是否达到目标或止损
            if signal.action == "buy":
                if current_price >= signal.take_profit:
                    await self._close_signal(symbol, signal, "take_profit", current_price)
                elif current_price <= signal.stop_loss:
                    await self._close_signal(symbol, signal, "stop_loss", current_price)
            elif signal.action == "sell":
                if current_price <= signal.take_profit:
                    await self._close_signal(symbol, signal, "take_profit", current_price)
                elif current_price >= signal.stop_loss:
                    await self._close_signal(symbol, signal, "stop_loss", current_price)
            
            # 检查信号时效性（4小时后自动关闭）
            signal_age = (datetime.utcnow() - signal.timestamp).total_seconds()
            if signal_age > 14400:  # 4小时
                await self._close_signal(symbol, signal, "timeout", current_price)
                
        except Exception as e:
            logger.error(f"监控信号性能失败: {e}")
    
    async def _close_signal(self, symbol: str, signal: TradingSignal, reason: str, close_price: float) -> None:
        """关闭信号"""
        try:
            # 计算盈亏
            if signal.action == "buy":
                pnl_pct = (close_price - signal.price_target) / signal.price_target
            else:  # sell
                pnl_pct = (signal.price_target - close_price) / signal.price_target
            
            # 更新统计
            if pnl_pct > 0:
                self.signal_stats['successful_signals'] += 1
            else:
                self.signal_stats['failed_signals'] += 1
            
            # 移除活跃信号
            if symbol in self.active_signals:
                del self.active_signals[symbol]
            
            logger.info(f"AI信号已关闭: {symbol} {reason} 盈亏:{pnl_pct:.2%}")
            
        except Exception as e:
            logger.error(f"关闭信号失败: {e}")
    
    async def _handle_price_anomaly(self, symbol: str, price: float, change_pct: float) -> None:
        """处理价格异常"""
        try:
            # 快速反应机制
            if abs(change_pct) > 0.05:  # 5%的急剧变化
                logger.warning(f"检测到{symbol}价格异常变动: {change_pct:.2%}")
                
                # 暂停相关信号
                if symbol in self.active_signals:
                    signal = self.active_signals[symbol]
                    await self._close_signal(symbol, signal, "price_anomaly", price)
                
                # 触发风险评估
                await self._emergency_risk_assessment(symbol, price, change_pct)
                
        except Exception as e:
            logger.error(f"处理价格异常失败: {e}")
    
    async def _emergency_risk_assessment(self, symbol: str, price: float, change_pct: float) -> None:
        """紧急风险评估"""
        try:
            # 评估市场影响
            risk_level = "high" if abs(change_pct) > 0.1 else "medium"
            
            # 通知风险管理系统
            risk_data = {
                "symbol": symbol,
                "price": price,
                "change_pct": change_pct,
                "risk_level": risk_level,
                "timestamp": datetime.utcnow(),
                "trigger": "price_anomaly"
            }
            
            # 这里可以集成到风险管理系统
            logger.warning(f"紧急风险评估: {symbol} 风险等级{risk_level}")
            
        except Exception as e:
            logger.error(f"紧急风险评估失败: {e}")
    
    async def _check_breakout_signals(self, symbol: str, price: float, indicators: Dict[str, float]) -> None:
        """检查突破信号"""
        try:
            resistance = indicators.get('resistance', 0)
            support = indicators.get('support', 0)
            bb_upper = indicators.get('bb_upper', 0)
            bb_lower = indicators.get('bb_lower', 0)
            
            # 检查阻力突破
            if price > resistance and resistance > 0:
                await self._handle_breakout(symbol, "resistance_breakout", price, resistance)
            
            # 检查支撑跌破
            elif price < support and support > 0:
                await self._handle_breakout(symbol, "support_breakdown", price, support)
            
            # 检查布林带突破
            elif price > bb_upper and bb_upper > 0:
                await self._handle_breakout(symbol, "bb_upper_breakout", price, bb_upper)
            
            elif price < bb_lower and bb_lower > 0:
                await self._handle_breakout(symbol, "bb_lower_breakdown", price, bb_lower)
                
        except Exception as e:
            logger.error(f"检查突破信号失败: {e}")
    
    async def _handle_breakout(self, symbol: str, breakout_type: str, price: float, level: float) -> None:
        """处理突破信号"""
        try:
            logger.info(f"检测到突破信号: {symbol} {breakout_type} @ {price:.2f} (突破位: {level:.2f})")
            
            # 生成快速响应信号
            if "upper" in breakout_type or "resistance" in breakout_type:
                action = "buy"
                target = price * 1.02
                stop = level * 0.998
            else:
                action = "sell"
                target = price * 0.98
                stop = level * 1.002
            
            quick_signal = TradingSignal(
                symbol=symbol,
                action=action,
                confidence=0.8,  # 突破信号通常信心度较高
                price_target=target,
                stop_loss=stop,
                take_profit=target,
                position_size=self.max_position_size * 0.5,  # 较小仓位快速反应
                reasoning=f"技术突破信号: {breakout_type}",
                timestamp=datetime.utcnow(),
                strategy_type="breakout"
            )
            
            # 如果满足条件，执行信号
            if quick_signal.confidence >= self.confidence_threshold:
                await self._execute_ai_signal(quick_signal)
                
        except Exception as e:
            logger.error(f"处理突破信号失败: {e}")
    
    def get_engine_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        return {
            "is_running": self.is_running,
            "tracked_symbols": list(self.price_history.keys()),
            "active_signals_count": len(self.active_signals),
            "total_signals": self.signal_stats['total_signals'],
            "successful_signals": self.signal_stats['successful_signals'],
            "success_rate": (self.signal_stats['successful_signals'] / max(self.signal_stats['total_signals'], 1)) * 100,
            "avg_confidence": self.signal_stats['avg_confidence'],
            "last_analysis_time": max([analysis.timestamp for analysis in self.market_analysis_cache.values()]) if self.market_analysis_cache else None
        }
    
    def get_active_signals(self) -> Dict[str, Dict[str, Any]]:
        """获取活跃信号"""
        return {
            symbol: {
                "action": signal.action,
                "confidence": signal.confidence,
                "price_target": signal.price_target,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "position_size": signal.position_size,
                "reasoning": signal.reasoning,
                "timestamp": signal.timestamp.isoformat(),
                "age_seconds": (datetime.utcnow() - signal.timestamp).total_seconds()
            }
            for symbol, signal in self.active_signals.items()
        }

# 全局AI交易引擎实例
ai_trading_engine = AITradingEngine()