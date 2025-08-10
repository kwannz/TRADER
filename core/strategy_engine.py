"""
策略执行引擎
管理和执行量化交易策略，支持多策略并行运行
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import pandas as pd
from loguru import logger

from config.settings import settings
from .data_manager import data_manager
from .ai_engine import ai_engine

class StrategyStatus(Enum):
    """策略状态枚举"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

class OrderSide(Enum):
    """订单方向枚举"""
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    """订单类型枚举"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

class BaseStrategy:
    """策略基类"""
    
    def __init__(self, strategy_id: str, name: str, config: Dict[str, Any]):
        self.strategy_id = strategy_id
        self.name = name
        self.config = config
        self.status = StrategyStatus.DRAFT
        self.position = 0.0  # 当前仓位
        self.equity = 0.0    # 策略权益
        self.pnl = 0.0       # 未实现盈亏
        self.trades_count = 0
        self.last_signal_time = None
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
    async def initialize(self):
        """策略初始化 - 子类实现"""
        pass
    
    async def on_tick(self, tick_data: Dict):
        """处理Tick数据 - 子类实现"""
        pass
    
    async def on_bar(self, bar_data: Dict):
        """处理K线数据 - 子类实现"""
        pass
    
    async def generate_signal(self) -> Optional[Dict]:
        """生成交易信号 - 子类实现"""
        return None
    
    async def calculate_position_size(self, signal: Dict) -> float:
        """计算仓位大小 - 子类实现"""
        return 0.0
    
    async def should_exit(self) -> bool:
        """检查是否应该退出 - 子类实现"""
        return False
    
    def update_status(self, status: StrategyStatus):
        """更新策略状态"""
        self.status = status
        self.updated_at = datetime.utcnow()

class GridStrategy(BaseStrategy):
    """网格策略"""
    
    def __init__(self, strategy_id: str, name: str, config: Dict[str, Any]):
        super().__init__(strategy_id, name, config)
        self.grid_levels = []
        self.orders = {}
        
    async def initialize(self):
        """初始化网格"""
        try:
            symbol = self.config.get("symbol", "BTC-USDT")
            grid_count = self.config.get("grid_count", 10)
            price_range = self.config.get("price_range", 0.1)  # 10%
            
            # 获取当前价格
            market_data = await data_manager.cache_manager.get_market_data(f"OKX:{symbol}")
            if not market_data:
                raise ValueError("无法获取市场数据")
            
            current_price = float(market_data["price"])
            
            # 计算网格价格
            upper_price = current_price * (1 + price_range / 2)
            lower_price = current_price * (1 - price_range / 2)
            price_step = (upper_price - lower_price) / grid_count
            
            self.grid_levels = []
            for i in range(grid_count + 1):
                price = lower_price + i * price_step
                self.grid_levels.append({
                    "price": price,
                    "side": OrderSide.BUY if price < current_price else OrderSide.SELL,
                    "filled": False
                })
            
            logger.info(f"网格策略初始化完成: {len(self.grid_levels)}个网格点")
            
        except Exception as e:
            logger.error(f"网格策略初始化失败: {e}")
            raise
    
    async def on_tick(self, tick_data: Dict):
        """处理价格变化"""
        try:
            current_price = float(tick_data["price"])
            
            # 检查网格触发
            for level in self.grid_levels:
                if not level["filled"]:
                    if (level["side"] == OrderSide.BUY and current_price <= level["price"]) or \
                       (level["side"] == OrderSide.SELL and current_price >= level["price"]):
                        
                        # 生成交易信号
                        signal = {
                            "type": "grid_trigger",
                            "side": level["side"].value,
                            "price": level["price"],
                            "quantity": self.config.get("quantity_per_grid", 0.001)
                        }
                        
                        # 提交给策略引擎执行
                        await self._submit_signal(signal)
                        level["filled"] = True
                        
        except Exception as e:
            logger.error(f"网格策略处理Tick失败: {e}")
    
    async def _submit_signal(self, signal: Dict):
        """提交交易信号"""
        # 这里会被策略引擎捕获和执行
        self.last_signal_time = datetime.utcnow()
        logger.info(f"网格策略生成信号: {signal}")

class DCAStrategy(BaseStrategy):
    """定投策略 (Dollar Cost Averaging)"""
    
    def __init__(self, strategy_id: str, name: str, config: Dict[str, Any]):
        super().__init__(strategy_id, name, config)
        self.last_buy_time = None
        
    async def initialize(self):
        """初始化DCA策略"""
        try:
            self.last_buy_time = datetime.utcnow()
            logger.info("DCA策略初始化完成")
        except Exception as e:
            logger.error(f"DCA策略初始化失败: {e}")
            raise
    
    async def on_bar(self, bar_data: Dict):
        """处理K线数据"""
        try:
            # 检查是否到了定投时间
            interval_minutes = self.config.get("interval_minutes", 60)  # 1小时
            
            if self.last_buy_time is None:
                self.last_buy_time = datetime.utcnow()
                return
            
            time_since_last = datetime.utcnow() - self.last_buy_time
            if time_since_last.total_seconds() >= interval_minutes * 60:
                
                # 生成买入信号
                signal = {
                    "type": "dca_buy",
                    "side": "buy",
                    "price": float(bar_data["close"]),
                    "quantity": self.config.get("buy_amount", 0.001)
                }
                
                await self._submit_signal(signal)
                self.last_buy_time = datetime.utcnow()
                
        except Exception as e:
            logger.error(f"DCA策略处理K线失败: {e}")
    
    async def _submit_signal(self, signal: Dict):
        """提交交易信号"""
        self.last_signal_time = datetime.utcnow()
        logger.info(f"DCA策略生成信号: {signal}")

class AIStrategy(BaseStrategy):
    """AI策略"""
    
    def __init__(self, strategy_id: str, name: str, config: Dict[str, Any]):
        super().__init__(strategy_id, name, config)
        self.ai_model = config.get("ai_model", "gemini")
        self.last_analysis_time = None
        self.analysis_interval = config.get("analysis_interval", 1800)  # 30分钟
        
    async def initialize(self):
        """初始化AI策略"""
        try:
            logger.info(f"AI策略初始化完成: {self.ai_model}")
        except Exception as e:
            logger.error(f"AI策略初始化失败: {e}")
            raise
    
    async def on_bar(self, bar_data: Dict):
        """处理K线数据"""
        try:
            # 检查是否需要AI分析
            now = datetime.utcnow()
            if (self.last_analysis_time is None or 
                (now - self.last_analysis_time).total_seconds() >= self.analysis_interval):
                
                await self._perform_ai_analysis(bar_data)
                self.last_analysis_time = now
                
        except Exception as e:
            logger.error(f"AI策略处理K线失败: {e}")
    
    async def _perform_ai_analysis(self, bar_data: Dict):
        """执行AI分析"""
        try:
            symbol = self.config.get("symbol", "BTC-USDT")
            
            # 获取市场预测
            prediction = await ai_engine.predict_market_movement([symbol])
            
            # 根据预测生成信号
            if prediction.get("trend_direction") == "up" and prediction.get("confidence", 0) > 0.7:
                signal = {
                    "type": "ai_prediction",
                    "side": "buy",
                    "price": float(bar_data["close"]),
                    "quantity": self.config.get("position_size", 0.001),
                    "confidence": prediction.get("confidence"),
                    "reasoning": prediction.get("reasoning", "AI预测上涨")
                }
                await self._submit_signal(signal)
                
            elif prediction.get("trend_direction") == "down" and prediction.get("confidence", 0) > 0.7:
                signal = {
                    "type": "ai_prediction",
                    "side": "sell",
                    "price": float(bar_data["close"]),
                    "quantity": self.config.get("position_size", 0.001),
                    "confidence": prediction.get("confidence"),
                    "reasoning": prediction.get("reasoning", "AI预测下跌")
                }
                await self._submit_signal(signal)
                
        except Exception as e:
            logger.error(f"AI分析失败: {e}")
    
    async def _submit_signal(self, signal: Dict):
        """提交交易信号"""
        self.last_signal_time = datetime.utcnow()
        logger.info(f"AI策略生成信号: {signal}")

class RiskManager:
    """风控管理器"""
    
    def __init__(self):
        self.max_position_size = settings.max_position_size
        self.hard_stop_loss = settings.hard_stop_loss
        self.initial_balance = settings.initial_balance
        self.daily_loss_limit = self.initial_balance * 0.1  # 日亏损限制10%
        
    async def check_signal(self, strategy: BaseStrategy, signal: Dict) -> Dict[str, Any]:
        """检查交易信号风控"""
        try:
            risk_result = {
                "approved": True,
                "rejected_reason": None,
                "adjusted_quantity": signal.get("quantity", 0),
                "warnings": []
            }
            
            # 1. 检查硬止损
            current_equity = await self._get_current_equity()
            if current_equity <= self.hard_stop_loss:
                risk_result["approved"] = False
                risk_result["rejected_reason"] = f"触发硬止损线 {self.hard_stop_loss} USDT"
                return risk_result
            
            # 2. 检查单策略仓位限制
            strategy_position_value = abs(strategy.position * signal.get("price", 0))
            max_strategy_position = self.initial_balance * 0.2  # 单策略最大20%
            
            if strategy_position_value > max_strategy_position:
                risk_result["approved"] = False
                risk_result["rejected_reason"] = "超出单策略仓位限制"
                return risk_result
            
            # 3. 检查总仓位限制
            total_position_ratio = await self._get_total_position_ratio()
            if total_position_ratio > self.max_position_size:
                risk_result["approved"] = False
                risk_result["rejected_reason"] = "超出总仓位限制"
                return risk_result
            
            # 4. 调整仓位大小
            proposed_quantity = signal.get("quantity", 0)
            max_safe_quantity = await self._calculate_max_safe_quantity(strategy, signal)
            
            if proposed_quantity > max_safe_quantity:
                risk_result["adjusted_quantity"] = max_safe_quantity
                risk_result["warnings"].append(f"仓位调整: {proposed_quantity} -> {max_safe_quantity}")
            
            # 5. AI风控检查
            if strategy.status == StrategyStatus.ACTIVE:
                portfolio_data = await self._get_portfolio_data()
                ai_risk = await ai_engine.assess_portfolio_risk(portfolio_data)
                
                if ai_risk.get("urgent_action_needed", False):
                    risk_result["warnings"].append("AI风控警告: " + ai_risk.get("risk_level", "未知风险"))
            
            return risk_result
            
        except Exception as e:
            logger.error(f"风控检查失败: {e}")
            return {
                "approved": False,
                "rejected_reason": f"风控系统错误: {e}",
                "adjusted_quantity": 0,
                "warnings": []
            }
    
    async def _get_current_equity(self) -> float:
        """获取当前权益"""
        try:
            # 简化实现 - 从数据库获取最新权益
            return self.initial_balance  # 临时返回初始资金
        except:
            return self.initial_balance
    
    async def _get_total_position_ratio(self) -> float:
        """获取总仓位占比"""
        try:
            # 简化实现
            return 0.5  # 假设50%仓位
        except:
            return 0.0
    
    async def _calculate_max_safe_quantity(self, strategy: BaseStrategy, signal: Dict) -> float:
        """计算最大安全仓位"""
        try:
            # 简化实现 - 基于策略类型和风险偏好
            base_quantity = signal.get("quantity", 0)
            risk_multiplier = 1.0
            
            # 根据策略类型调整
            if isinstance(strategy, GridStrategy):
                risk_multiplier = 0.8  # 网格策略更保守
            elif isinstance(strategy, AIStrategy):
                confidence = signal.get("confidence", 0.5)
                risk_multiplier = min(confidence * 2, 1.0)  # 基于AI信心度
            
            return base_quantity * risk_multiplier
            
        except:
            return 0.001  # 最小仓位
    
    async def _get_portfolio_data(self) -> Dict[str, Any]:
        """获取投资组合数据"""
        try:
            return {
                "total_value": await self._get_current_equity(),
                "position_ratio": await self._get_total_position_ratio(),
                "unrealized_pnl": 0.0,  # 简化
                "active_strategies": 3   # 简化
            }
        except:
            return {}

class StrategyEngine:
    """策略执行引擎"""
    
    def __init__(self):
        self.strategies: Dict[str, BaseStrategy] = {}
        self.risk_manager = RiskManager()
        self.is_running = False
        self._execution_tasks = {}
        self._signal_queue = asyncio.Queue()
        
    async def initialize(self):
        """初始化策略引擎"""
        try:
            logger.info("策略引擎初始化中...")
            
            # 加载已保存的策略
            await self._load_saved_strategies()
            
            # 启动信号处理任务
            asyncio.create_task(self._process_signals())
            
            logger.info("策略引擎初始化完成")
            
        except Exception as e:
            logger.error(f"策略引擎初始化失败: {e}")
            raise
    
    async def _load_saved_strategies(self):
        """加载已保存的策略"""
        try:
            saved_strategies = await data_manager.get_strategies(status="active")
            
            for strategy_data in saved_strategies:
                try:
                    strategy = await self._create_strategy_instance(strategy_data)
                    if strategy:
                        await self.add_strategy(strategy)
                        logger.info(f"加载策略: {strategy.name}")
                except Exception as e:
                    logger.error(f"加载策略失败 {strategy_data.get('name', 'Unknown')}: {e}")
                    
        except Exception as e:
            logger.error(f"加载已保存策略失败: {e}")
    
    async def _create_strategy_instance(self, strategy_data: Dict) -> Optional[BaseStrategy]:
        """根据数据创建策略实例"""
        try:
            strategy_id = strategy_data["_id"]
            name = strategy_data["name"]
            strategy_type = strategy_data["type"]
            config = strategy_data.get("config", {})
            
            if strategy_type == "grid":
                return GridStrategy(strategy_id, name, config)
            elif strategy_type == "dca":
                return DCAStrategy(strategy_id, name, config)
            elif strategy_type == "ai_generated":
                return AIStrategy(strategy_id, name, config)
            else:
                logger.warning(f"未知策略类型: {strategy_type}")
                return None
                
        except Exception as e:
            logger.error(f"创建策略实例失败: {e}")
            return None
    
    async def add_strategy(self, strategy: BaseStrategy):
        """添加策略"""
        try:
            await strategy.initialize()
            self.strategies[strategy.strategy_id] = strategy
            
            # 启动策略执行任务
            task = asyncio.create_task(self._run_strategy(strategy))
            self._execution_tasks[strategy.strategy_id] = task
            
            logger.info(f"策略已添加: {strategy.name}")
            
        except Exception as e:
            logger.error(f"添加策略失败 {strategy.name}: {e}")
            raise
    
    async def remove_strategy(self, strategy_id: str):
        """移除策略"""
        try:
            if strategy_id in self.strategies:
                strategy = self.strategies[strategy_id]
                strategy.update_status(StrategyStatus.STOPPED)
                
                # 取消执行任务
                if strategy_id in self._execution_tasks:
                    self._execution_tasks[strategy_id].cancel()
                    del self._execution_tasks[strategy_id]
                
                del self.strategies[strategy_id]
                
                # 更新数据库状态
                await data_manager.update_strategy(strategy_id, {"status": "stopped"})
                
                logger.info(f"策略已移除: {strategy.name}")
                
        except Exception as e:
            logger.error(f"移除策略失败: {e}")
    
    async def start_strategy(self, strategy_id: str):
        """启动策略"""
        try:
            if strategy_id in self.strategies:
                strategy = self.strategies[strategy_id]
                strategy.update_status(StrategyStatus.ACTIVE)
                
                # 更新数据库状态
                await data_manager.update_strategy(strategy_id, {"status": "active"})
                
                logger.info(f"策略已启动: {strategy.name}")
            else:
                raise ValueError(f"策略不存在: {strategy_id}")
                
        except Exception as e:
            logger.error(f"启动策略失败: {e}")
    
    async def pause_strategy(self, strategy_id: str):
        """暂停策略"""
        try:
            if strategy_id in self.strategies:
                strategy = self.strategies[strategy_id]
                strategy.update_status(StrategyStatus.PAUSED)
                
                # 更新数据库状态
                await data_manager.update_strategy(strategy_id, {"status": "paused"})
                
                logger.info(f"策略已暂停: {strategy.name}")
            else:
                raise ValueError(f"策略不存在: {strategy_id}")
                
        except Exception as e:
            logger.error(f"暂停策略失败: {e}")
    
    async def _run_strategy(self, strategy: BaseStrategy):
        """运行单个策略"""
        try:
            logger.info(f"策略开始运行: {strategy.name}")
            
            while strategy.status != StrategyStatus.STOPPED:
                try:
                    if strategy.status == StrategyStatus.ACTIVE:
                        # 模拟接收市场数据
                        await self._feed_market_data_to_strategy(strategy)
                    
                    # 策略检查间隔
                    await asyncio.sleep(settings.strategy_check_interval)
                    
                except Exception as e:
                    logger.error(f"策略运行错误 {strategy.name}: {e}")
                    strategy.update_status(StrategyStatus.ERROR)
                    
        except asyncio.CancelledError:
            logger.info(f"策略运行已取消: {strategy.name}")
        except Exception as e:
            logger.error(f"策略运行失败 {strategy.name}: {e}")
    
    async def _feed_market_data_to_strategy(self, strategy: BaseStrategy):
        """向策略提供市场数据"""
        try:
            symbol = strategy.config.get("symbol", "BTC/USDT")
            
            # 首先尝试从市场模拟器获取实时数据
            from .market_simulator import market_simulator
            current_prices = market_simulator.get_current_prices()
            
            if symbol in current_prices:
                # 构造Tick数据
                tick_data = {
                    "symbol": symbol,
                    "price": current_prices[symbol],
                    "timestamp": datetime.utcnow().timestamp() * 1000,
                    "volume": 1000,  # 模拟交易量
                }
                await strategy.on_tick(tick_data)
                
                # 同时缓存到Redis
                if hasattr(data_manager, 'cache_manager') and data_manager.cache_manager:
                    await data_manager.cache_manager.set_market_data(
                        symbol,
                        {
                            "price": str(current_prices[symbol]),
                            "timestamp": str(int(datetime.utcnow().timestamp())),
                            "volume": "1000"
                        }
                    )
            else:
                # 备用：从缓存获取数据
                market_data = await data_manager.cache_manager.get_market_data(symbol)
                if market_data:
                    await strategy.on_tick(market_data)
            
            # 获取最新K线数据
            try:
                recent_klines = await data_manager.time_series_manager.get_kline_data(
                    symbol, "1m", limit=1
                )
                
                if not recent_klines.empty:
                    bar_data = recent_klines.iloc[-1].to_dict()
                    await strategy.on_bar(bar_data)
            except Exception as kline_error:
                # 如果无法获取K线数据，使用模拟数据
                if symbol in current_prices:
                    bar_data = {
                        "symbol": symbol,
                        "timestamp": datetime.utcnow(),
                        "open": current_prices[symbol],
                        "high": current_prices[symbol] * 1.001,
                        "low": current_prices[symbol] * 0.999,
                        "close": current_prices[symbol],
                        "volume": 1000
                    }
                    await strategy.on_bar(bar_data)
                
        except Exception as e:
            logger.error(f"市场数据推送失败 {strategy.name}: {e}")
    
    async def _process_signals(self):
        """处理交易信号队列"""
        try:
            while True:
                try:
                    # 等待信号
                    signal_data = await self._signal_queue.get()
                    strategy = signal_data["strategy"]
                    signal = signal_data["signal"]
                    
                    # 风控检查
                    risk_check = await self.risk_manager.check_signal(strategy, signal)
                    
                    if risk_check["approved"]:
                        # 执行交易
                        await self._execute_trade(strategy, signal, risk_check)
                    else:
                        logger.warning(f"交易信号被拒绝: {risk_check['rejected_reason']}")
                    
                    self._signal_queue.task_done()
                    
                except Exception as e:
                    logger.error(f"信号处理失败: {e}")
                    
        except asyncio.CancelledError:
            logger.info("信号处理任务已取消")
    
    async def _execute_trade(self, strategy: BaseStrategy, signal: Dict, risk_check: Dict):
        """执行交易"""
        try:
            # 创建交易记录
            trade_data = {
                "strategy_id": strategy.strategy_id,
                "symbol": strategy.config.get("symbol", "BTC-USDT"),
                "side": signal["side"],
                "quantity": risk_check["adjusted_quantity"],
                "price": signal["price"],
                "type": signal.get("type", "market"),
                "status": "filled",  # 简化 - 假设立即成交
                "pnl": 0.0,  # 计算实际盈亏
                "commission": 0.0,
                "timestamp": datetime.utcnow(),
                "signal_data": signal
            }
            
            # 保存交易记录
            await data_manager.save_trade(trade_data)
            
            # 更新策略状态
            if signal["side"] == "buy":
                strategy.position += risk_check["adjusted_quantity"]
            else:
                strategy.position -= risk_check["adjusted_quantity"]
            
            strategy.trades_count += 1
            strategy.updated_at = datetime.utcnow()
            
            logger.info(f"交易已执行: {strategy.name} {signal['side']} {risk_check['adjusted_quantity']} @ {signal['price']}")
            
        except Exception as e:
            logger.error(f"交易执行失败: {e}")
    
    async def submit_signal(self, strategy_id: str, signal: Dict):
        """提交交易信号"""
        try:
            if strategy_id in self.strategies:
                strategy = self.strategies[strategy_id]
                await self._signal_queue.put({
                    "strategy": strategy,
                    "signal": signal
                })
            else:
                logger.warning(f"未找到策略: {strategy_id}")
                
        except Exception as e:
            logger.error(f"提交信号失败: {e}")
    
    def get_strategy_status(self) -> Dict[str, Dict]:
        """获取所有策略状态"""
        return {
            strategy_id: {
                "name": strategy.name,
                "status": strategy.status.value,
                "position": strategy.position,
                "pnl": strategy.pnl,
                "trades_count": strategy.trades_count,
                "last_signal_time": strategy.last_signal_time.isoformat() if strategy.last_signal_time else None,
                "created_at": strategy.created_at.isoformat(),
                "updated_at": strategy.updated_at.isoformat()
            }
            for strategy_id, strategy in self.strategies.items()
        }
    
    async def shutdown(self):
        """关闭策略引擎"""
        try:
            logger.info("正在关闭策略引擎...")
            
            # 停止所有策略
            for strategy in self.strategies.values():
                strategy.update_status(StrategyStatus.STOPPED)
            
            # 取消所有执行任务
            for task in self._execution_tasks.values():
                task.cancel()
            
            # 等待任务完成
            if self._execution_tasks:
                await asyncio.gather(*self._execution_tasks.values(), return_exceptions=True)
            
            self.is_running = False
            logger.info("策略引擎已关闭")
            
        except Exception as e:
            logger.error(f"关闭策略引擎失败: {e}")

# 全局策略引擎实例
strategy_engine = StrategyEngine()