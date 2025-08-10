"""
市场数据模拟器

实现高频实时数据仿真：
- WebSocket数据流模拟
- K线数据生成
- Tick级别数据仿真
- 新闻事件影响模拟
- 多交易所数据源模拟
"""

import asyncio
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from collections import deque
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_logger import get_logger, LogCategory
from .data_manager import data_manager

logger = get_logger()

@dataclass
class MarketTick:
    """市场Tick数据"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: float
    ask: float
    side: str  # 'buy' or 'sell'

@dataclass
class KLineData:
    """K线数据"""
    symbol: str
    timestamp: datetime
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class NewsEvent:
    """新闻事件"""
    timestamp: datetime
    title: str
    content: str
    impact_level: str  # 'low', 'medium', 'high'
    sentiment: str     # 'positive', 'negative', 'neutral'
    affected_symbols: List[str]

class PriceGenerator:
    """价格生成器 - 基于几何布朗运动"""
    
    def __init__(self, 
                 initial_price: float = 45000.0,
                 volatility: float = 0.02,
                 drift: float = 0.0001,
                 mean_reversion: float = 0.001):
        self.current_price = initial_price
        self.initial_price = initial_price
        self.volatility = volatility
        self.drift = drift
        self.mean_reversion = mean_reversion
        self.price_history = deque(maxlen=1000)
        
    def next_price(self, dt: float = 1/3600) -> float:
        """生成下一个价格点"""
        try:
            # 几何布朗运动 + 均值回归
            random_shock = np.random.normal(0, 1)
            
            # 均值回归项
            mean_reversion_term = -self.mean_reversion * (self.current_price - self.initial_price) * dt
            
            # 价格变化
            price_change = (
                self.drift * dt +  # 趋势项
                mean_reversion_term +  # 均值回归项
                self.volatility * math.sqrt(dt) * random_shock  # 随机项
            )
            
            # 更新价格（确保正数）
            self.current_price = max(0.01, self.current_price * (1 + price_change))
            
            # 记录历史
            self.price_history.append(self.current_price)
            
            return self.current_price
            
        except Exception as e:
            logger.error(f"生成价格失败: {e}")
            return self.current_price

class VolumeGenerator:
    """交易量生成器"""
    
    def __init__(self, base_volume: float = 1000000):
        self.base_volume = base_volume
        self.volume_factor = 1.0
        
    def next_volume(self, price_change_pct: float = 0) -> float:
        """生成交易量"""
        try:
            # 价格变化越大，交易量越大
            volatility_factor = 1 + abs(price_change_pct) * 10
            
            # 随机因子
            random_factor = random.uniform(0.3, 2.0)
            
            # 生成交易量
            volume = self.base_volume * volatility_factor * random_factor * self.volume_factor
            
            return max(100, volume)
            
        except Exception as e:
            logger.error(f"生成交易量失败: {e}")
            return self.base_volume

class NewsEventGenerator:
    """新闻事件生成器"""
    
    def __init__(self):
        self.news_templates = [
            {
                "title": "{symbol}突破关键技术位，市场情绪转好",
                "content": "{symbol}价格突破{price}美元关键阻力位，技术面呈现强势信号。",
                "impact": "medium",
                "sentiment": "positive"
            },
            {
                "title": "大型机构增持{symbol}，长期前景看好",
                "content": "据报告显示，多家机构投资者大幅增持{symbol}，市场信心增强。",
                "impact": "high",
                "sentiment": "positive"
            },
            {
                "title": "监管政策影响，{symbol}短期承压",
                "content": "最新监管政策可能对{symbol}产生短期负面影响。",
                "impact": "medium",
                "sentiment": "negative"
            },
            {
                "title": "{symbol}网络升级完成，功能显著增强",
                "content": "{symbol}网络升级顺利完成，新功能提升用户体验。",
                "impact": "low",
                "sentiment": "positive"
            }
        ]
        
    def generate_news(self, symbol: str, price: float) -> NewsEvent:
        """生成新闻事件"""
        try:
            template = random.choice(self.news_templates)
            
            title = template["title"].format(symbol=symbol, price=int(price))
            content = template["content"].format(symbol=symbol, price=int(price))
            
            return NewsEvent(
                timestamp=datetime.utcnow(),
                title=title,
                content=content,
                impact_level=template["impact"],
                sentiment=template["sentiment"],
                affected_symbols=[symbol]
            )
            
        except Exception as e:
            logger.error(f"生成新闻事件失败: {e}")
            return NewsEvent(
                timestamp=datetime.utcnow(),
                title="市场动态",
                content="市场保持活跃交易",
                impact_level="low",
                sentiment="neutral",
                affected_symbols=[symbol]
            )

class TradingSimulator:
    """仿真交易机主类 - 集成市场仿真和回测功能"""
    
    def __init__(self, mode: str = "simulation"):
        # 运行模式: "simulation"(实时仿真) 或 "backtest"(历史回测)
        self.mode = mode
        
        self.symbols = {
            "BTC/USDT": {"base_price": 45000, "volatility": 0.025},
            "ETH/USDT": {"base_price": 2800, "volatility": 0.030},
            "ADA/USDT": {"base_price": 1.20, "volatility": 0.040},
            "DOT/USDT": {"base_price": 35, "volatility": 0.035},
            "LINK/USDT": {"base_price": 28, "volatility": 0.038}
        }
        
        # 价格生成器
        self.price_generators = {}
        self.volume_generators = {}
        
        # 新闻生成器
        self.news_generator = NewsEventGenerator()
        
        # 数据订阅者
        self.tick_subscribers: List[Callable] = []
        self.kline_subscribers: List[Callable] = []
        self.news_subscribers: List[Callable] = []
        
        # 运行状态
        self.is_running = False
        self.simulation_tasks: List[asyncio.Task] = []
        
        # K线数据缓存
        self.kline_cache = {}
        
        # 回测相关组件
        self.backtest_engine: Optional[BacktestEngine] = None
        self.trading_env: Optional[TradingEnvironment] = None
        self.data_replay: Optional[DataReplaySystem] = None
        self.performance_analyzer: Optional[PerformanceAnalyzer] = None
        
        # 策略相关
        self.strategies: Dict[str, Callable] = {}
        self.current_backtest_config: Optional[BacktestConfig] = None
        
        self._initialize_generators()
        
    def _initialize_generators(self) -> None:
        """初始化生成器"""
        try:
            for symbol, config in self.symbols.items():
                self.price_generators[symbol] = PriceGenerator(
                    initial_price=config["base_price"],
                    volatility=config["volatility"]
                )
                
                self.volume_generators[symbol] = VolumeGenerator(
                    base_volume=1000000 if symbol == "BTC/USDT" else 500000
                )
                
                # 初始化K线缓存
                self.kline_cache[symbol] = {}
                
            logger.info("市场模拟器生成器初始化完成")
            
        except Exception as e:
            logger.error(f"初始化生成器失败: {e}")
    
    def subscribe_tick_data(self, callback: Callable[[MarketTick], None]) -> None:
        """订阅Tick数据"""
        self.tick_subscribers.append(callback)
    
    def subscribe_kline_data(self, callback: Callable[[KLineData], None]) -> None:
        """订阅K线数据"""
        self.kline_subscribers.append(callback)
    
    def subscribe_news_data(self, callback: Callable[[NewsEvent], None]) -> None:
        """订阅新闻数据"""
        self.news_subscribers.append(callback)
    
    async def start_simulation(self) -> None:
        """启动市场模拟"""
        if self.is_running:
            logger.warning("市场模拟器已在运行")
            return
            
        try:
            self.is_running = True
            
            # 启动各种数据流模拟任务
            self.simulation_tasks = [
                asyncio.create_task(self._tick_data_loop()),
                asyncio.create_task(self._kline_data_loop()),
                asyncio.create_task(self._news_event_loop()),
                asyncio.create_task(self._market_event_loop())
            ]
            
            # 等待所有任务完成
            await asyncio.gather(*self.simulation_tasks, return_exceptions=True)
            
            logger.info("市场模拟器已启动")
            
        except Exception as e:
            logger.error(f"启动市场模拟失败: {e}")
            await self.stop_simulation()
    
    async def stop_simulation(self) -> None:
        """停止市场模拟"""
        try:
            self.is_running = False
            
            # 取消所有任务
            for task in self.simulation_tasks:
                if not task.done():
                    task.cancel()
            
            # 等待任务完成
            if self.simulation_tasks:
                await asyncio.gather(*self.simulation_tasks, return_exceptions=True)
            
            self.simulation_tasks = []
            logger.info("市场模拟器已停止")
            
        except Exception as e:
            logger.error(f"停止市场模拟失败: {e}")
    
    async def _tick_data_loop(self) -> None:
        """Tick数据生成循环"""
        while self.is_running:
            try:
                for symbol in self.symbols:
                    # 生成价格和交易量
                    price_gen = self.price_generators[symbol]
                    volume_gen = self.volume_generators[symbol]
                    
                    new_price = price_gen.next_price()
                    
                    # 计算价格变化
                    price_change_pct = 0
                    if len(price_gen.price_history) > 1:
                        old_price = price_gen.price_history[-2]
                        price_change_pct = (new_price - old_price) / old_price
                    
                    volume = volume_gen.next_volume(price_change_pct)
                    
                    # 生成买卖价差
                    spread = new_price * 0.0001  # 0.01% 价差
                    bid = new_price - spread / 2
                    ask = new_price + spread / 2
                    
                    # 随机决定买卖方向
                    side = "buy" if random.random() > 0.5 else "sell"
                    
                    # 创建Tick数据
                    tick = MarketTick(
                        symbol=symbol,
                        timestamp=datetime.utcnow(),
                        price=new_price,
                        volume=volume / 100,  # 单笔交易量较小
                        bid=bid,
                        ask=ask,
                        side=side
                    )
                    
                    # 通知订阅者
                    await self._notify_tick_subscribers(tick)
                    
                    # 缓存到数据管理器
                    if hasattr(data_manager, '_initialized') and data_manager._initialized:
                        tick_data = {
                            "timestamp": tick.timestamp.timestamp() * 1000,
                            "price": tick.price,
                            "volume": tick.volume,
                            "side": tick.side
                        }
                        await data_manager.time_series_manager.insert_tick_data(symbol, tick_data)
                
                # 控制频率 - 模拟高频数据
                await asyncio.sleep(0.1)  # 10Hz
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Tick数据生成错误: {e}")
                await asyncio.sleep(1)
    
    async def _kline_data_loop(self) -> None:
        """K线数据生成循环"""
        kline_intervals = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600
        }
        
        last_update = {}
        
        while self.is_running:
            try:
                current_time = datetime.utcnow()
                
                for symbol in self.symbols:
                    price_gen = self.price_generators[symbol]
                    
                    if not price_gen.price_history:
                        continue
                    
                    for timeframe, interval_seconds in kline_intervals.items():
                        cache_key = f"{symbol}_{timeframe}"
                        
                        # 检查是否需要更新
                        if cache_key not in last_update:
                            last_update[cache_key] = current_time
                            continue
                        
                        time_diff = (current_time - last_update[cache_key]).total_seconds()
                        
                        if time_diff >= interval_seconds:
                            # 生成K线数据
                            kline = await self._generate_kline(symbol, timeframe, current_time, interval_seconds)
                            
                            if kline:
                                # 通知订阅者
                                await self._notify_kline_subscribers(kline)
                                
                                # 更新时间戳
                                last_update[cache_key] = current_time
                
                await asyncio.sleep(10)  # 每10秒检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"K线数据生成错误: {e}")
                await asyncio.sleep(5)
    
    async def _generate_kline(self, symbol: str, timeframe: str, end_time: datetime, interval_seconds: int) -> Optional[KLineData]:
        """生成K线数据"""
        try:
            price_gen = self.price_generators[symbol]
            volume_gen = self.volume_generators[symbol]
            
            if len(price_gen.price_history) < 10:
                return None
            
            # 获取价格历史
            prices = list(price_gen.price_history)[-10:]  # 最近10个点
            
            # 计算OHLC
            open_price = prices[0]
            close_price = prices[-1]
            high_price = max(prices)
            low_price = min(prices)
            
            # 生成交易量
            avg_price_change = abs(close_price - open_price) / open_price
            volume = volume_gen.next_volume(avg_price_change) * (interval_seconds / 60)  # 按时间调整
            
            kline = KLineData(
                symbol=symbol,
                timestamp=end_time,
                timeframe=timeframe,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            )
            
            # 保存到数据管理器
            if hasattr(data_manager, '_initialized') and data_manager._initialized:
                kline_data = [{
                    "timestamp": int(kline.timestamp.timestamp() * 1000),
                    "open": kline.open,
                    "high": kline.high,
                    "low": kline.low,
                    "close": kline.close,
                    "volume": kline.volume
                }]
                await data_manager.time_series_manager.insert_kline_data(symbol, timeframe, kline_data)
            
            return kline
            
        except Exception as e:
            logger.error(f"生成K线数据失败: {e}")
            return None
    
    async def _news_event_loop(self) -> None:
        """新闻事件生成循环"""
        while self.is_running:
            try:
                # 随机生成新闻事件
                if random.random() < 0.1:  # 10%概率
                    symbol = random.choice(list(self.symbols.keys()))
                    price_gen = self.price_generators[symbol]
                    
                    news = self.news_generator.generate_news(symbol, price_gen.current_price)
                    
                    # 通知订阅者
                    await self._notify_news_subscribers(news)
                    
                    # 保存到数据管理器
                    if hasattr(data_manager, '_initialized') and data_manager._initialized:
                        news_data = {
                            "title": news.title,
                            "content": news.content,
                            "impact": news.impact_level,
                            "sentiment": news.sentiment,
                            "symbols": news.affected_symbols,
                            "source": "Market Simulator"
                        }
                        await data_manager.save_news(news_data)
                
                # 随机间隔 (5-30分钟)
                await asyncio.sleep(random.randint(300, 1800))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"新闻事件生成错误: {e}")
                await asyncio.sleep(60)
    
    async def _market_event_loop(self) -> None:
        """市场事件循环 - 模拟突发事件"""
        while self.is_running:
            try:
                # 随机市场事件
                if random.random() < 0.05:  # 5%概率
                    event_type = random.choice(['flash_crash', 'pump', 'high_volatility'])
                    await self._trigger_market_event(event_type)
                
                await asyncio.sleep(600)  # 每10分钟检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"市场事件循环错误: {e}")
                await asyncio.sleep(60)
    
    async def _trigger_market_event(self, event_type: str) -> None:
        """触发市场事件"""
        try:
            logger.info(f"触发市场事件: {event_type}")
            
            if event_type == 'flash_crash':
                # 闪崩 - 短期大幅下跌
                for symbol, price_gen in self.price_generators.items():
                    price_gen.volatility *= 3  # 增加波动率
                    price_gen.drift = -0.01   # 负趋势
                
                # 5分钟后恢复
                await asyncio.sleep(300)
                for symbol, config in self.symbols.items():
                    self.price_generators[symbol].volatility = config["volatility"]
                    self.price_generators[symbol].drift = 0.0001
                    
            elif event_type == 'pump':
                # 拉升 - 短期大幅上涨
                for symbol, price_gen in self.price_generators.items():
                    price_gen.volatility *= 2
                    price_gen.drift = 0.005  # 正趋势
                
                await asyncio.sleep(600)  # 10分钟后恢复
                for symbol, config in self.symbols.items():
                    self.price_generators[symbol].volatility = config["volatility"]
                    self.price_generators[symbol].drift = 0.0001
                    
            elif event_type == 'high_volatility':
                # 高波动 - 增加市场波动性
                for symbol, price_gen in self.price_generators.items():
                    price_gen.volatility *= 2
                
                await asyncio.sleep(1800)  # 30分钟后恢复
                for symbol, config in self.symbols.items():
                    self.price_generators[symbol].volatility = config["volatility"]
            
        except Exception as e:
            logger.error(f"触发市场事件失败: {e}")
    
    async def _notify_tick_subscribers(self, tick: MarketTick) -> None:
        """通知Tick数据订阅者"""
        try:
            for callback in self.tick_subscribers:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(tick)
                    else:
                        callback(tick)
                except Exception as e:
                    logger.error(f"通知Tick订阅者失败: {e}")
        except Exception as e:
            logger.error(f"通知Tick订阅者错误: {e}")
    
    async def _notify_kline_subscribers(self, kline: KLineData) -> None:
        """通知K线数据订阅者"""
        try:
            for callback in self.kline_subscribers:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(kline)
                    else:
                        callback(kline)
                except Exception as e:
                    logger.error(f"通知K线订阅者失败: {e}")
        except Exception as e:
            logger.error(f"通知K线订阅者错误: {e}")
    
    async def _notify_news_subscribers(self, news: NewsEvent) -> None:
        """通知新闻数据订阅者"""
        try:
            for callback in self.news_subscribers:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(news)
                    else:
                        callback(news)
                except Exception as e:
                    logger.error(f"通知新闻订阅者失败: {e}")
        except Exception as e:
            logger.error(f"通知新闻订阅者错误: {e}")
    
    def get_current_prices(self) -> Dict[str, float]:
        """获取当前价格"""
        return {
            symbol: gen.current_price 
            for symbol, gen in self.price_generators.items()
        }
    
    def get_market_summary(self) -> Dict[str, Any]:
        """获取市场摘要"""
        try:
            summary = {}
            
            for symbol, price_gen in self.price_generators.items():
                if len(price_gen.price_history) >= 2:
                    current_price = price_gen.current_price
                    prev_price = price_gen.price_history[-2]
                    change_24h = (current_price - prev_price) / prev_price
                    
                    volume_gen = self.volume_generators[symbol]
                    volume_24h = volume_gen.base_volume * 24  # 模拟24小时交易量
                    
                    summary[symbol] = {
                        "price": current_price,
                        "change_24h": change_24h,
                        "volume_24h": volume_24h,
                        "high_24h": max(list(price_gen.price_history)[-24:] if len(price_gen.price_history) >= 24 else price_gen.price_history),
                        "low_24h": min(list(price_gen.price_history)[-24:] if len(price_gen.price_history) >= 24 else price_gen.price_history),
                        "timestamp": datetime.utcnow()
                    }
            
            return summary
            
        except Exception as e:
            logger.error(f"获取市场摘要失败: {e}")
            return {}
    
    # 回测功能
    async def setup_backtest(self, config: Union["BacktestConfig", Dict[str, Any]]) -> None:
        """设置回测配置"""
        try:
            if isinstance(config, dict):
                # 转换字典为BacktestConfig对象
                self.current_backtest_config = BacktestConfig(**config)
            else:
                self.current_backtest_config = config
            
            # 切换到回测模式
            self.mode = "backtest"
            
            # 初始化回测引擎
            self.backtest_engine = BacktestEngine(self.current_backtest_config)
            await self.backtest_engine.initialize()
            
            # 添加已注册的策略
            for strategy_id, strategy_func in self.strategies.items():
                self.backtest_engine.add_strategy(strategy_id, strategy_func)
            
            logger.info(f"回测配置完成: {self.current_backtest_config.start_date} - {self.current_backtest_config.end_date}")
            
        except Exception as e:
            logger.error(f"设置回测配置失败: {e}")
            raise
    
    def add_strategy(self, strategy_id: str, strategy_func: Callable, 
                    initial_state: Optional[Dict] = None) -> None:
        """添加交易策略"""
        try:
            self.strategies[strategy_id] = strategy_func
            
            # 如果回测引擎已初始化，直接添加策略
            if self.backtest_engine:
                self.backtest_engine.add_strategy(strategy_id, strategy_func, initial_state)
            
            logger.info(f"策略已添加: {strategy_id}")
            
        except Exception as e:
            logger.error(f"添加策略失败: {e}")
    
    async def run_backtest(self) -> Optional[Any]:
        """运行回测"""
        try:
            if self.mode != "backtest" or not self.backtest_engine:
                raise RuntimeError("请先调用setup_backtest()配置回测")
            
            result = await self.backtest_engine.run_backtest()
            logger.success(f"回测完成: 总收益率 {result.performance_metrics.total_return:.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"运行回测失败: {e}")
            raise
    
    def get_backtest_status(self) -> Dict[str, Any]:
        """获取回测状态"""
        if self.backtest_engine:
            return self.backtest_engine.get_status()
        return {'status': 'not_initialized'}
    
    def pause_backtest(self) -> None:
        """暂停回测"""
        if self.backtest_engine:
            self.backtest_engine.pause_backtest()
    
    def resume_backtest(self) -> None:
        """恢复回测"""
        if self.backtest_engine:
            self.backtest_engine.resume_backtest()
    
    def set_time_acceleration(self, factor: float) -> None:
        """设置时间加速倍数"""
        if self.backtest_engine:
            self.backtest_engine.set_time_acceleration(factor)
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """获取投资组合状态"""
        if self.backtest_engine:
            return self.backtest_engine.get_current_portfolio_status()
        return {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if self.backtest_engine:
            return self.backtest_engine.get_performance_summary()
        return {}
    
    def switch_mode(self, mode: str) -> None:
        """切换运行模式"""
        if mode not in ["simulation", "backtest"]:
            raise ValueError("模式必须是 'simulation' 或 'backtest'")
        
        if self.is_running:
            logger.warning("请先停止当前运行再切换模式")
            return
        
        self.mode = mode
        logger.info(f"已切换到 {mode} 模式")

# 全局仿真交易机实例 (保持向后兼容)
market_simulator = TradingSimulator(mode="simulation")
trading_simulator = market_simulator  # 新名称