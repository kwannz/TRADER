"""
回测引擎

提供完整的策略回测功能：
- 历史数据回放
- 策略执行模拟
- 性能分析和报告生成
- 多策略并行回测
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from pathlib import Path
import json

from ..performance_analyzer import PerformanceAnalyzer, PerformanceMetrics
from .trading_environment import TradingEnvironment
from .data_replay_system import DataReplaySystem
from .portfolio_manager import PortfolioManager
from core.unified_logger import get_logger, LogCategory

logger = get_logger()

class BacktestStatus(Enum):
    """回测状态"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class BacktestConfig:
    """回测配置"""
    start_date: str
    end_date: str
    initial_balance: float
    symbols: List[str]
    timeframes: List[str]
    data_source: str = "historical"
    commission_rate: float = 0.001  # 0.1%手续费
    slippage_bps: float = 2.0       # 2个基点滑点
    max_slippage_pct: float = 0.05  # 最大5%滑点
    enable_market_impact: bool = True
    benchmark_symbol: str = "BTC/USDT"
    save_results: bool = True
    results_path: str = "backtest_results"

@dataclass
class BacktestResult:
    """回测结果"""
    config: BacktestConfig
    start_time: datetime
    end_time: datetime
    duration: float
    performance_metrics: PerformanceMetrics
    trades: List[Dict[str, Any]]
    equity_curve: List[Dict[str, Any]]
    drawdowns: List[Dict[str, Any]]
    monthly_returns: Dict[str, float]
    strategy_attribution: Dict[str, Any]
    benchmark_comparison: Dict[str, Any]

class BacktestEngine:
    """回测引擎主类"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.status = BacktestStatus.IDLE
        
        # 核心组件
        self.trading_env: Optional[TradingEnvironment] = None
        self.data_replay: Optional[DataReplaySystem] = None
        self.portfolio_mgr: Optional[PortfolioManager] = None
        self.performance_analyzer: Optional[PerformanceAnalyzer] = None
        
        # 回测状态
        self.current_time: Optional[datetime] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # 策略和结果
        self.strategies: Dict[str, Callable] = {}
        self.strategy_states: Dict[str, Dict] = {}
        self.results: Optional[BacktestResult] = None
        
        # 事件订阅
        self.event_handlers: Dict[str, List[Callable]] = {
            'on_data': [],
            'on_trade': [],
            'on_order': [],
            'on_bar': [],
            'on_tick': []
        }
        
        # 控制参数
        self.time_acceleration = 1.0  # 时间加速倍数
        self.is_paused = False
        self.abort_flag = False
        
        logger.info(f"回测引擎初始化完成: {config.start_date} - {config.end_date}")
    
    async def initialize(self) -> None:
        """初始化回测系统"""
        try:
            self.status = BacktestStatus.INITIALIZING
            
            # 解析时间
            self.start_time = pd.to_datetime(self.config.start_date)
            self.end_time = pd.to_datetime(self.config.end_date)
            self.current_time = self.start_time
            
            # 初始化交易环境
            self.trading_env = TradingEnvironment(
                initial_balance=self.config.initial_balance,
                commission_rate=self.config.commission_rate,
                slippage_bps=self.config.slippage_bps
            )
            await self.trading_env.initialize()
            
            # 初始化数据回放系统
            self.data_replay = DataReplaySystem(
                symbols=self.config.symbols,
                timeframes=self.config.timeframes,
                start_date=self.start_time,
                end_date=self.end_time
            )
            await self.data_replay.initialize()
            
            # 初始化投资组合管理器
            self.portfolio_mgr = PortfolioManager(
                initial_balance=self.config.initial_balance
            )
            
            # 初始化性能分析器
            self.performance_analyzer = PerformanceAnalyzer(
                initial_balance=self.config.initial_balance
            )
            
            # 连接事件处理器
            self._setup_event_handlers()
            
            self.status = BacktestStatus.IDLE
            logger.info("回测系统初始化完成")
            
        except Exception as e:
            self.status = BacktestStatus.ERROR
            logger.error(f"回测系统初始化失败: {e}")
            raise
    
    def _setup_event_handlers(self) -> None:
        """设置事件处理器"""
        try:
            # 数据事件处理
            self.data_replay.subscribe_data_event(self._on_market_data)
            self.data_replay.subscribe_bar_event(self._on_bar_data)
            self.data_replay.subscribe_tick_event(self._on_tick_data)
            
            # 交易事件处理
            self.trading_env.subscribe_trade_event(self._on_trade_executed)
            self.trading_env.subscribe_order_event(self._on_order_update)
            
        except Exception as e:
            logger.error(f"设置事件处理器失败: {e}")
    
    def add_strategy(self, strategy_id: str, strategy_func: Callable, 
                    initial_state: Optional[Dict] = None) -> None:
        """添加策略"""
        try:
            self.strategies[strategy_id] = strategy_func
            self.strategy_states[strategy_id] = initial_state or {}
            
            logger.info(f"策略已添加: {strategy_id}")
            
        except Exception as e:
            logger.error(f"添加策略失败: {e}")
    
    def subscribe_event(self, event_type: str, handler: Callable) -> None:
        """订阅事件"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
    
    async def run_backtest(self) -> BacktestResult:
        """运行回测"""
        try:
            if self.status != BacktestStatus.IDLE:
                raise RuntimeError(f"回测状态错误: {self.status}")
            
            self.status = BacktestStatus.RUNNING
            backtest_start = datetime.utcnow()
            
            logger.info(f"开始回测: {self.start_time} - {self.end_time}")
            
            # 启动性能分析器
            await self.performance_analyzer.start_analysis()
            
            # 主要回测循环
            async for market_data in self.data_replay.replay_data():
                if self.abort_flag:
                    break
                    
                # 暂停处理
                while self.is_paused:
                    await asyncio.sleep(0.1)
                
                # 更新当前时间
                self.current_time = market_data['timestamp']
                
                # 执行策略
                await self._execute_strategies(market_data)
                
                # 更新投资组合
                await self._update_portfolio(market_data)
                
                # 时间加速控制
                if self.time_acceleration < 1000:
                    await asyncio.sleep(0.001 / self.time_acceleration)
            
            # 完成回测
            await self._finalize_backtest()
            
            backtest_end = datetime.utcnow()
            duration = (backtest_end - backtest_start).total_seconds()
            
            # 生成结果
            self.results = await self._generate_results(
                backtest_start, backtest_end, duration
            )
            
            self.status = BacktestStatus.COMPLETED
            logger.success(f"回测完成，耗时: {duration:.2f}秒")
            
            return self.results
            
        except Exception as e:
            self.status = BacktestStatus.ERROR
            logger.error(f"回测执行失败: {e}")
            raise
        finally:
            await self.performance_analyzer.stop_analysis()
    
    async def _execute_strategies(self, market_data: Dict[str, Any]) -> None:
        """执行策略"""
        try:
            for strategy_id, strategy_func in self.strategies.items():
                try:
                    # 准备策略上下文
                    context = {
                        'current_time': self.current_time,
                        'market_data': market_data,
                        'portfolio': self.portfolio_mgr.get_portfolio_status(),
                        'trading_env': self.trading_env,
                        'state': self.strategy_states[strategy_id]
                    }
                    
                    # 执行策略
                    if asyncio.iscoroutinefunction(strategy_func):
                        await strategy_func(context)
                    else:
                        strategy_func(context)
                        
                except Exception as e:
                    logger.error(f"策略执行失败 {strategy_id}: {e}")
                    
        except Exception as e:
            logger.error(f"执行策略失败: {e}")
    
    async def _update_portfolio(self, market_data: Dict[str, Any]) -> None:
        """更新投资组合"""
        try:
            # 更新持仓价值
            await self.portfolio_mgr.update_portfolio_value(market_data)
            
            # 更新性能分析数据
            portfolio_value = self.portfolio_mgr.get_total_value()
            
            # 记录到性能分析器
            equity_data = {
                'timestamp': self.current_time,
                'equity': portfolio_value,
                'balance': self.trading_env.get_balance(),
                'positions': self.portfolio_mgr.get_positions_summary()
            }
            
            await self._record_equity_point(equity_data)
            
        except Exception as e:
            logger.error(f"更新投资组合失败: {e}")
    
    async def _record_equity_point(self, equity_data: Dict[str, Any]) -> None:
        """记录权益曲线点"""
        try:
            # 添加到性能分析器
            if hasattr(self.performance_analyzer, 'equity_curve'):
                self.performance_analyzer.equity_curve.append(equity_data)
                
        except Exception as e:
            logger.error(f"记录权益点失败: {e}")
    
    async def _finalize_backtest(self) -> None:
        """完成回测"""
        try:
            # 关闭所有仓位
            await self.trading_env.close_all_positions()
            
            # 最终投资组合更新
            await self.portfolio_mgr.finalize_portfolio()
            
            logger.info("回测清理完成")
            
        except Exception as e:
            logger.error(f"回测清理失败: {e}")
    
    async def _generate_results(self, start_time: datetime, 
                              end_time: datetime, duration: float) -> BacktestResult:
        """生成回测结果"""
        try:
            # 计算性能指标
            performance_metrics = await self.performance_analyzer._calculate_performance_metrics()
            
            # 获取交易记录
            trades = await self.trading_env.get_trade_history()
            
            # 获取权益曲线
            equity_curve = list(self.performance_analyzer.equity_curve)
            
            # 计算回撤
            drawdowns = self._calculate_drawdowns(equity_curve)
            
            # 计算月度收益
            monthly_returns = self._calculate_monthly_returns(equity_curve)
            
            # 策略归因分析
            strategy_attribution = self._calculate_strategy_attribution(trades)
            
            # 基准比较
            benchmark_comparison = await self._compare_with_benchmark()
            
            result = BacktestResult(
                config=self.config,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                performance_metrics=performance_metrics,
                trades=trades,
                equity_curve=equity_curve,
                drawdowns=drawdowns,
                monthly_returns=monthly_returns,
                strategy_attribution=strategy_attribution,
                benchmark_comparison=benchmark_comparison
            )
            
            # 保存结果
            if self.config.save_results:
                await self._save_results(result)
            
            return result
            
        except Exception as e:
            logger.error(f"生成回测结果失败: {e}")
            raise
    
    def _calculate_drawdowns(self, equity_curve: List[Dict]) -> List[Dict]:
        """计算回撤"""
        try:
            if len(equity_curve) < 2:
                return []
            
            drawdowns = []
            peak = equity_curve[0]['equity']
            peak_time = equity_curve[0]['timestamp']
            
            for point in equity_curve:
                equity = point['equity']
                timestamp = point['timestamp']
                
                if equity > peak:
                    peak = equity
                    peak_time = timestamp
                else:
                    drawdown_pct = (peak - equity) / peak
                    if drawdown_pct > 0.01:  # 只记录1%以上的回撤
                        drawdowns.append({
                            'start_time': peak_time,
                            'current_time': timestamp,
                            'peak_equity': peak,
                            'current_equity': equity,
                            'drawdown_pct': drawdown_pct,
                            'duration_days': (timestamp - peak_time).days
                        })
            
            return drawdowns
            
        except Exception as e:
            logger.error(f"计算回撤失败: {e}")
            return []
    
    def _calculate_monthly_returns(self, equity_curve: List[Dict]) -> Dict[str, float]:
        """计算月度收益"""
        try:
            if len(equity_curve) < 2:
                return {}
            
            df = pd.DataFrame(equity_curve)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # 月末重采样
            monthly = df['equity'].resample('M').last()
            monthly_returns = monthly.pct_change().dropna()
            
            return {
                date.strftime('%Y-%m'): ret 
                for date, ret in monthly_returns.items()
            }
            
        except Exception as e:
            logger.error(f"计算月度收益失败: {e}")
            return {}
    
    def _calculate_strategy_attribution(self, trades: List[Dict]) -> Dict[str, Any]:
        """计算策略归因"""
        try:
            attribution = {}
            
            # 按策略分组交易
            strategy_trades = {}
            for trade in trades:
                strategy_id = trade.get('strategy_id', 'unknown')
                if strategy_id not in strategy_trades:
                    strategy_trades[strategy_id] = []
                strategy_trades[strategy_id].append(trade)
            
            # 计算各策略表现
            total_pnl = sum(trade.get('pnl', 0) for trade in trades)
            
            for strategy_id, s_trades in strategy_trades.items():
                strategy_pnl = sum(trade.get('pnl', 0) for trade in s_trades)
                win_trades = [t for t in s_trades if t.get('pnl', 0) > 0]
                
                attribution[strategy_id] = {
                    'total_trades': len(s_trades),
                    'total_pnl': strategy_pnl,
                    'win_rate': len(win_trades) / len(s_trades) if s_trades else 0,
                    'avg_pnl_per_trade': strategy_pnl / len(s_trades) if s_trades else 0,
                    'contribution_pct': strategy_pnl / total_pnl if total_pnl != 0 else 0
                }
            
            return attribution
            
        except Exception as e:
            logger.error(f"计算策略归因失败: {e}")
            return {}
    
    async def _compare_with_benchmark(self) -> Dict[str, Any]:
        """基准比较"""
        try:
            # 获取基准数据
            benchmark_data = await self.data_replay.get_benchmark_data(
                self.config.benchmark_symbol
            )
            
            if not benchmark_data:
                return {}
            
            # 计算基准收益
            benchmark_returns = []
            for i in range(1, len(benchmark_data)):
                prev_price = benchmark_data[i-1]['close']
                curr_price = benchmark_data[i]['close']
                ret = (curr_price - prev_price) / prev_price
                benchmark_returns.append(ret)
            
            # 计算组合收益
            equity_curve = list(self.performance_analyzer.equity_curve)
            portfolio_returns = []
            for i in range(1, len(equity_curve)):
                prev_equity = equity_curve[i-1]['equity']
                curr_equity = equity_curve[i]['equity']
                ret = (curr_equity - prev_equity) / prev_equity
                portfolio_returns.append(ret)
            
            # 计算比较指标
            if len(portfolio_returns) > 0 and len(benchmark_returns) > 0:
                min_len = min(len(portfolio_returns), len(benchmark_returns))
                port_rets = np.array(portfolio_returns[:min_len])
                bench_rets = np.array(benchmark_returns[:min_len])
                
                # 超额收益
                excess_returns = port_rets - bench_rets
                
                # 跟踪误差
                tracking_error = np.std(excess_returns) * np.sqrt(252)
                
                # 信息比率
                info_ratio = (np.mean(excess_returns) / np.std(excess_returns) * 
                            np.sqrt(252)) if np.std(excess_returns) > 0 else 0
                
                # Beta和Alpha
                if np.var(bench_rets) > 0:
                    beta = np.cov(port_rets, bench_rets)[0,1] / np.var(bench_rets)
                else:
                    beta = 1.0
                
                alpha = np.mean(port_rets) * 252 - beta * np.mean(bench_rets) * 252
                
                return {
                    'benchmark_symbol': self.config.benchmark_symbol,
                    'tracking_error': tracking_error,
                    'information_ratio': info_ratio,
                    'beta': beta,
                    'alpha': alpha,
                    'correlation': np.corrcoef(port_rets, bench_rets)[0,1],
                    'outperformance_days': len(excess_returns[excess_returns > 0]) / len(excess_returns)
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"基准比较失败: {e}")
            return {}
    
    async def _save_results(self, result: BacktestResult) -> None:
        """保存回测结果"""
        try:
            # 创建结果目录
            results_dir = Path(self.config.results_path)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"backtest_result_{timestamp}.json"
            filepath = results_dir / filename
            
            # 序列化结果
            result_dict = asdict(result)
            
            # 处理不能序列化的对象
            result_dict['start_time'] = result_dict['start_time'].isoformat()
            result_dict['end_time'] = result_dict['end_time'].isoformat()
            
            # 保存到文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"回测结果已保存: {filepath}")
            
        except Exception as e:
            logger.error(f"保存回测结果失败: {e}")
    
    # 事件处理方法
    async def _on_market_data(self, data: Dict[str, Any]) -> None:
        """市场数据事件处理"""
        for handler in self.event_handlers['on_data']:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"数据事件处理失败: {e}")
    
    async def _on_bar_data(self, bar_data: Dict[str, Any]) -> None:
        """K线数据事件处理"""
        for handler in self.event_handlers['on_bar']:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(bar_data)
                else:
                    handler(bar_data)
            except Exception as e:
                logger.error(f"K线事件处理失败: {e}")
    
    async def _on_tick_data(self, tick_data: Dict[str, Any]) -> None:
        """Tick数据事件处理"""
        for handler in self.event_handlers['on_tick']:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(tick_data)
                else:
                    handler(tick_data)
            except Exception as e:
                logger.error(f"Tick事件处理失败: {e}")
    
    async def _on_trade_executed(self, trade_data: Dict[str, Any]) -> None:
        """交易执行事件处理"""
        # 记录到性能分析器
        self.performance_analyzer.add_trade(trade_data)
        
        # 通知订阅者
        for handler in self.event_handlers['on_trade']:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(trade_data)
                else:
                    handler(trade_data)
            except Exception as e:
                logger.error(f"交易事件处理失败: {e}")
    
    async def _on_order_update(self, order_data: Dict[str, Any]) -> None:
        """订单更新事件处理"""
        for handler in self.event_handlers['on_order']:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(order_data)
                else:
                    handler(order_data)
            except Exception as e:
                logger.error(f"订单事件处理失败: {e}")
    
    # 控制方法
    def pause_backtest(self) -> None:
        """暂停回测"""
        self.is_paused = True
        self.status = BacktestStatus.PAUSED
        logger.info("回测已暂停")
    
    def resume_backtest(self) -> None:
        """恢复回测"""
        self.is_paused = False
        self.status = BacktestStatus.RUNNING
        logger.info("回测已恢复")
    
    def abort_backtest(self) -> None:
        """中止回测"""
        self.abort_flag = True
        logger.warning("回测已中止")
    
    def set_time_acceleration(self, factor: float) -> None:
        """设置时间加速倍数"""
        self.time_acceleration = max(0.1, min(1000.0, factor))
        logger.info(f"时间加速设置为: {self.time_acceleration}x")
    
    # 查询方法
    def get_status(self) -> Dict[str, Any]:
        """获取回测状态"""
        return {
            'status': self.status.value,
            'current_time': self.current_time.isoformat() if self.current_time else None,
            'progress': self._calculate_progress(),
            'time_acceleration': self.time_acceleration,
            'is_paused': self.is_paused
        }
    
    def _calculate_progress(self) -> float:
        """计算回测进度"""
        if not self.start_time or not self.end_time or not self.current_time:
            return 0.0
        
        total_duration = (self.end_time - self.start_time).total_seconds()
        current_duration = (self.current_time - self.start_time).total_seconds()
        
        return min(1.0, max(0.0, current_duration / total_duration))
    
    def get_current_portfolio_status(self) -> Dict[str, Any]:
        """获取当前投资组合状态"""
        if self.portfolio_mgr:
            return self.portfolio_mgr.get_portfolio_status()
        return {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if self.performance_analyzer and len(self.performance_analyzer.equity_curve) > 0:
            return self.performance_analyzer.get_real_time_metrics()
        return {}