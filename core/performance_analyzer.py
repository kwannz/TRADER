"""
性能分析系统

提供全面的交易性能分析和报告：
- 实时收益分析
- 风险调整收益指标
- 策略归因分析
- 基准比较
- 详细的性能报告
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np
import pandas as pd

from .data_manager import data_manager
from .strategy_engine import strategy_engine
from .trading_simulator import trading_simulator
from core.unified_logger import get_logger, LogCategory

logger = get_logger()

@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    
@dataclass
class StrategyPerformance:
    """策略性能"""
    strategy_id: str
    strategy_name: str
    total_pnl: float
    win_trades: int
    loss_trades: int
    win_rate: float
    avg_win_amount: float
    avg_loss_amount: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    sharpe_ratio: float
    max_drawdown: float

class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        
        # 数据存储
        self.equity_curve = deque(maxlen=10000)
        self.trade_history = []
        self.daily_returns = deque(maxlen=365)
        self.strategy_performance = {}
        
        # 基准数据
        self.benchmark_returns = deque(maxlen=365)
        
        # 分析参数
        self.risk_free_rate = 0.02  # 2%年化无风险利率
        self.trading_days_per_year = 252
        
        # 运行状态
        self.is_analyzing = False
        self.analysis_tasks = []
        
        logger.info("性能分析器初始化完成")
    
    async def start_analysis(self) -> None:
        """启动性能分析"""
        if self.is_analyzing:
            return
        
        try:
            self.is_analyzing = True
            
            # 启动分析任务
            self.analysis_tasks = [
                asyncio.create_task(self._equity_curve_update_loop()),
                asyncio.create_task(self._performance_calculation_loop()),
                asyncio.create_task(self._strategy_analysis_loop())
            ]
            
            logger.info("性能分析系统已启动")
            
        except Exception as e:
            logger.error(f"启动性能分析失败: {e}")
            await self.stop_analysis()
    
    async def stop_analysis(self) -> None:
        """停止性能分析"""
        try:
            self.is_analyzing = False
            
            # 取消所有分析任务
            for task in self.analysis_tasks:
                if not task.done():
                    task.cancel()
            
            if self.analysis_tasks:
                await asyncio.gather(*self.analysis_tasks, return_exceptions=True)
            
            self.analysis_tasks = []
            logger.info("性能分析系统已停止")
            
        except Exception as e:
            logger.error(f"停止性能分析失败: {e}")
    
    async def _equity_curve_update_loop(self) -> None:
        """权益曲线更新循环"""
        while self.is_analyzing:
            try:
                # 计算当前总权益
                total_equity = await self._calculate_total_equity()
                
                # 更新权益曲线
                self.equity_curve.append({
                    'timestamp': datetime.utcnow(),
                    'equity': total_equity
                })
                
                # 计算日收益率
                if len(self.equity_curve) >= 2:
                    prev_equity = self.equity_curve[-2]['equity']
                    current_equity = self.equity_curve[-1]['equity']
                    
                    daily_return = (current_equity - prev_equity) / prev_equity
                    self.daily_returns.append(daily_return)
                
                await asyncio.sleep(60)  # 每分钟更新一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"权益曲线更新错误: {e}")
                await asyncio.sleep(30)
    
    async def _performance_calculation_loop(self) -> None:
        """性能计算循环"""
        while self.is_analyzing:
            try:
                # 每5分钟重新计算性能指标
                await asyncio.sleep(300)
                
                if len(self.daily_returns) >= 10:
                    metrics = await self._calculate_performance_metrics()
                    
                    # 这里可以将指标保存到数据库或缓存
                    logger.debug(f"性能指标更新 - 夏普比率: {metrics.sharpe_ratio:.2f}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"性能计算循环错误: {e}")
                await asyncio.sleep(60)
    
    async def _strategy_analysis_loop(self) -> None:
        """策略分析循环"""
        while self.is_analyzing:
            try:
                # 分析各策略表现
                strategy_status = strategy_engine.get_strategy_status()
                
                for strategy_id, strategy_info in strategy_status.items():
                    await self._analyze_strategy_performance(strategy_id, strategy_info)
                
                await asyncio.sleep(600)  # 每10分钟分析一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"策略分析循环错误: {e}")
                await asyncio.sleep(120)
    
    async def _calculate_total_equity(self) -> float:
        """计算总权益"""
        try:
            # 获取当前价格
            current_prices = market_simulator.get_current_prices()
            
            # 计算总权益（简化实现）
            total_equity = self.initial_balance
            
            # 这里应该加上所有仓位的当前市值
            # 简化为使用模拟数据
            import random
            total_equity += random.uniform(-500, 1000)  # 模拟盈亏
            
            return max(total_equity, 100)  # 确保不为负数
            
        except Exception as e:
            logger.error(f"计算总权益失败: {e}")
            return self.initial_balance
    
    async def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """计算性能指标"""
        try:
            if len(self.daily_returns) < 10:
                return self._default_metrics()
            
            returns = np.array(list(self.daily_returns))
            
            # 基础收益指标
            total_return = self._calculate_total_return()
            annualized_return = self._calculate_annualized_return(returns)
            volatility = np.std(returns) * np.sqrt(self.trading_days_per_year)
            
            # 风险调整收益指标
            sharpe_ratio = self._calculate_sharpe_ratio(returns, volatility)
            sortino_ratio = self._calculate_sortino_ratio(returns)
            
            # 回撤指标
            max_drawdown = self._calculate_max_drawdown()
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
            
            # 交易统计
            trade_stats = self._calculate_trade_statistics()
            
            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                win_rate=trade_stats['win_rate'],
                profit_factor=trade_stats['profit_factor'],
                avg_win=trade_stats['avg_win'],
                avg_loss=trade_stats['avg_loss'],
                total_trades=trade_stats['total_trades']
            )
            
        except Exception as e:
            logger.error(f"计算性能指标失败: {e}")
            return self._default_metrics()
    
    def _default_metrics(self) -> PerformanceMetrics:
        """默认指标"""
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            total_trades=0
        )
    
    def _calculate_total_return(self) -> float:
        """计算总收益率"""
        try:
            if len(self.equity_curve) < 2:
                return 0.0
            
            initial_equity = self.equity_curve[0]['equity']
            current_equity = self.equity_curve[-1]['equity']
            
            return (current_equity - initial_equity) / initial_equity
            
        except Exception as e:
            logger.error(f"计算总收益率失败: {e}")
            return 0.0
    
    def _calculate_annualized_return(self, returns: np.ndarray) -> float:
        """计算年化收益率"""
        try:
            if len(returns) == 0:
                return 0.0
            
            # 几何平均收益率
            cumulative_return = np.prod(1 + returns)
            days = len(returns)
            
            if days == 0:
                return 0.0
            
            annualized = (cumulative_return ** (self.trading_days_per_year / days)) - 1
            return annualized
            
        except Exception as e:
            logger.error(f"计算年化收益率失败: {e}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, volatility: float) -> float:
        """计算夏普比率"""
        try:
            if volatility == 0:
                return 0.0
            
            excess_returns = returns - (self.risk_free_rate / self.trading_days_per_year)
            return np.mean(excess_returns) / np.std(returns) * np.sqrt(self.trading_days_per_year)
            
        except Exception as e:
            logger.error(f"计算夏普比率失败: {e}")
            return 0.0
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """计算索提诺比率"""
        try:
            excess_returns = returns - (self.risk_free_rate / self.trading_days_per_year)
            negative_returns = returns[returns < 0]
            
            if len(negative_returns) == 0:
                return float('inf') if np.mean(excess_returns) > 0 else 0.0
            
            downside_deviation = np.std(negative_returns) * np.sqrt(self.trading_days_per_year)
            
            if downside_deviation == 0:
                return 0.0
            
            return np.mean(excess_returns) * np.sqrt(self.trading_days_per_year) / downside_deviation
            
        except Exception as e:
            logger.error(f"计算索提诺比率失败: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        try:
            if len(self.equity_curve) < 2:
                return 0.0
            
            equity_values = [item['equity'] for item in self.equity_curve]
            peak = equity_values[0]
            max_drawdown = 0.0
            
            for value in equity_values:
                if value > peak:
                    peak = value
                else:
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
            
            return max_drawdown
            
        except Exception as e:
            logger.error(f"计算最大回撤失败: {e}")
            return 0.0
    
    def _calculate_trade_statistics(self) -> Dict[str, float]:
        """计算交易统计"""
        try:
            if not self.trade_history:
                return {
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'total_trades': 0
                }
            
            winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
            losing_trades = [t for t in self.trade_history if t['pnl'] <= 0]
            
            total_trades = len(self.trade_history)
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
            
            total_wins = sum(t['pnl'] for t in winning_trades)
            total_losses = abs(sum(t['pnl'] for t in losing_trades))
            
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            avg_win = total_wins / len(winning_trades) if winning_trades else 0.0
            avg_loss = total_losses / len(losing_trades) if losing_trades else 0.0
            
            return {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'total_trades': total_trades
            }
            
        except Exception as e:
            logger.error(f"计算交易统计失败: {e}")
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'total_trades': 0
            }
    
    async def _analyze_strategy_performance(self, strategy_id: str, strategy_info: Dict) -> None:
        """分析策略性能"""
        try:
            # 获取策略交易记录
            strategy_trades = await data_manager.get_trades(strategy_id=strategy_id)
            
            if not strategy_trades:
                return
            
            # 计算策略特定指标
            winning_trades = [t for t in strategy_trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in strategy_trades if t.get('pnl', 0) <= 0]
            
            total_pnl = sum(t.get('pnl', 0) for t in strategy_trades)
            win_rate = len(winning_trades) / len(strategy_trades) if strategy_trades else 0.0
            
            avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0.0
            avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0.0
            
            # 计算连续盈亏
            max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive_trades(strategy_trades)
            
            # 计算策略夏普比率（简化）
            strategy_returns = [t.get('pnl', 0) / 1000 for t in strategy_trades]  # 简化收益率
            sharpe = self._calculate_sharpe_ratio(np.array(strategy_returns), np.std(strategy_returns)) if len(strategy_returns) > 1 else 0.0
            
            # 存储策略性能
            self.strategy_performance[strategy_id] = StrategyPerformance(
                strategy_id=strategy_id,
                strategy_name=strategy_info.get('name', 'Unknown'),
                total_pnl=total_pnl,
                win_trades=len(winning_trades),
                loss_trades=len(losing_trades),
                win_rate=win_rate,
                avg_win_amount=avg_win,
                avg_loss_amount=avg_loss,
                max_consecutive_wins=max_consecutive_wins,
                max_consecutive_losses=max_consecutive_losses,
                sharpe_ratio=sharpe,
                max_drawdown=0.0  # 需要额外计算
            )
            
        except Exception as e:
            logger.error(f"分析策略性能失败: {e}")
    
    def _calculate_consecutive_trades(self, trades: List[Dict]) -> Tuple[int, int]:
        """计算最大连续盈利/亏损次数"""
        try:
            if not trades:
                return 0, 0
            
            current_wins = 0
            current_losses = 0
            max_wins = 0
            max_losses = 0
            
            for trade in sorted(trades, key=lambda x: x.get('timestamp', datetime.min)):
                pnl = trade.get('pnl', 0)
                
                if pnl > 0:
                    current_wins += 1
                    current_losses = 0
                    max_wins = max(max_wins, current_wins)
                elif pnl < 0:
                    current_losses += 1
                    current_wins = 0
                    max_losses = max(max_losses, current_losses)
            
            return max_wins, max_losses
            
        except Exception as e:
            logger.error(f"计算连续交易失败: {e}")
            return 0, 0
    
    def add_trade(self, trade_data: Dict) -> None:
        """添加交易记录"""
        try:
            self.trade_history.append({
                'timestamp': trade_data.get('timestamp', datetime.utcnow()),
                'symbol': trade_data.get('symbol', ''),
                'side': trade_data.get('side', ''),
                'quantity': trade_data.get('quantity', 0),
                'price': trade_data.get('price', 0),
                'pnl': trade_data.get('pnl', 0),
                'strategy_id': trade_data.get('strategy_id', '')
            })
            
        except Exception as e:
            logger.error(f"添加交易记录失败: {e}")
    
    async def generate_performance_report(self, period_days: int = 30) -> Dict[str, Any]:
        """生成性能报告"""
        try:
            # 计算整体性能
            overall_metrics = await self._calculate_performance_metrics()
            
            # 时期收益分析
            period_analysis = self._analyze_period_performance(period_days)
            
            # 策略归因分析
            strategy_attribution = self._calculate_strategy_attribution()
            
            # 风险分析
            risk_analysis = await self._generate_risk_analysis()
            
            # 基准比较
            benchmark_comparison = self._compare_with_benchmark()
            
            # 月度/周度表现
            periodic_performance = self._calculate_periodic_performance()
            
            report = {
                'report_timestamp': datetime.utcnow().isoformat(),
                'analysis_period_days': period_days,
                
                'overall_performance': {
                    'total_return': overall_metrics.total_return,
                    'annualized_return': overall_metrics.annualized_return,
                    'volatility': overall_metrics.volatility,
                    'sharpe_ratio': overall_metrics.sharpe_ratio,
                    'sortino_ratio': overall_metrics.sortino_ratio,
                    'max_drawdown': overall_metrics.max_drawdown,
                    'calmar_ratio': overall_metrics.calmar_ratio
                },
                
                'trading_statistics': {
                    'total_trades': overall_metrics.total_trades,
                    'win_rate': overall_metrics.win_rate,
                    'profit_factor': overall_metrics.profit_factor,
                    'avg_win': overall_metrics.avg_win,
                    'avg_loss': overall_metrics.avg_loss
                },
                
                'period_analysis': period_analysis,
                'strategy_attribution': strategy_attribution,
                'risk_analysis': risk_analysis,
                'benchmark_comparison': benchmark_comparison,
                'periodic_performance': periodic_performance,
                
                'equity_curve_summary': {
                    'total_points': len(self.equity_curve),
                    'start_equity': self.equity_curve[0]['equity'] if self.equity_curve else self.initial_balance,
                    'current_equity': self.equity_curve[-1]['equity'] if self.equity_curve else self.initial_balance,
                    'peak_equity': max([p['equity'] for p in self.equity_curve]) if self.equity_curve else self.initial_balance
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"生成性能报告失败: {e}")
            return {}
    
    def _analyze_period_performance(self, period_days: int) -> Dict[str, Any]:
        """分析时期表现"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=period_days)
            
            # 筛选时期内的权益数据
            period_equity = [
                item for item in self.equity_curve 
                if item['timestamp'] >= cutoff_date
            ]
            
            if len(period_equity) < 2:
                return {'period_return': 0.0, 'period_volatility': 0.0, 'period_sharpe': 0.0}
            
            start_equity = period_equity[0]['equity']
            end_equity = period_equity[-1]['equity']
            
            period_return = (end_equity - start_equity) / start_equity
            
            # 计算时期内日收益率
            period_returns = []
            for i in range(1, len(period_equity)):
                prev_equity = period_equity[i-1]['equity']
                curr_equity = period_equity[i]['equity']
                daily_return = (curr_equity - prev_equity) / prev_equity
                period_returns.append(daily_return)
            
            period_returns = np.array(period_returns)
            period_volatility = np.std(period_returns) * np.sqrt(self.trading_days_per_year)
            period_sharpe = self._calculate_sharpe_ratio(period_returns, period_volatility)
            
            return {
                'period_return': period_return,
                'period_volatility': period_volatility,
                'period_sharpe': period_sharpe,
                'best_day': np.max(period_returns) if len(period_returns) > 0 else 0.0,
                'worst_day': np.min(period_returns) if len(period_returns) > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"分析时期表现失败: {e}")
            return {}
    
    def _calculate_strategy_attribution(self) -> Dict[str, Any]:
        """计算策略归因"""
        try:
            attribution = {}
            
            for strategy_id, perf in self.strategy_performance.items():
                attribution[strategy_id] = {
                    'strategy_name': perf.strategy_name,
                    'total_pnl': perf.total_pnl,
                    'win_rate': perf.win_rate,
                    'trades_count': perf.win_trades + perf.loss_trades,
                    'sharpe_ratio': perf.sharpe_ratio,
                    'contribution_to_total': 0.0  # 需要计算相对贡献
                }
            
            # 计算各策略对总收益的贡献
            total_pnl = sum(perf.total_pnl for perf in self.strategy_performance.values())
            
            if total_pnl != 0:
                for strategy_id in attribution:
                    strategy_pnl = attribution[strategy_id]['total_pnl']
                    attribution[strategy_id]['contribution_to_total'] = strategy_pnl / total_pnl
            
            return attribution
            
        except Exception as e:
            logger.error(f"计算策略归因失败: {e}")
            return {}
    
    async def _generate_risk_analysis(self) -> Dict[str, Any]:
        """生成风险分析"""
        try:
            if len(self.daily_returns) < 30:
                return {}
            
            returns = np.array(list(self.daily_returns))
            
            # VaR计算
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # 条件VaR (Expected Shortfall)
            cvar_95 = np.mean(returns[returns <= var_95])
            cvar_99 = np.mean(returns[returns <= var_99])
            
            # 下行风险
            negative_returns = returns[returns < 0]
            downside_risk = np.std(negative_returns) if len(negative_returns) > 0 else 0.0
            
            # 最大连续亏损天数
            max_loss_streak = self._calculate_max_loss_streak(returns)
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'downside_risk': downside_risk,
                'max_loss_streak_days': max_loss_streak,
                'positive_days_ratio': len(returns[returns > 0]) / len(returns)
            }
            
        except Exception as e:
            logger.error(f"生成风险分析失败: {e}")
            return {}
    
    def _calculate_max_loss_streak(self, returns: np.ndarray) -> int:
        """计算最大连续亏损天数"""
        try:
            current_streak = 0
            max_streak = 0
            
            for ret in returns:
                if ret < 0:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0
            
            return max_streak
            
        except Exception as e:
            logger.error(f"计算最大连续亏损天数失败: {e}")
            return 0
    
    def _compare_with_benchmark(self) -> Dict[str, Any]:
        """与基准比较"""
        try:
            # 简化的基准比较（实际应用中应使用真实基准数据）
            if len(self.daily_returns) < 10:
                return {}
            
            portfolio_returns = np.array(list(self.daily_returns))
            
            # 模拟基准收益（如BTC指数）
            benchmark_returns = np.random.normal(0.0005, 0.02, len(portfolio_returns))
            
            # 超额收益
            excess_returns = portfolio_returns - benchmark_returns
            
            # 跟踪误差
            tracking_error = np.std(excess_returns) * np.sqrt(self.trading_days_per_year)
            
            # 信息比率
            information_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.trading_days_per_year) if np.std(excess_returns) > 0 else 0
            
            # Beta系数
            if np.var(benchmark_returns) > 0:
                beta = np.cov(portfolio_returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
            else:
                beta = 1.0
            
            # Alpha
            portfolio_return = np.mean(portfolio_returns) * self.trading_days_per_year
            benchmark_return = np.mean(benchmark_returns) * self.trading_days_per_year
            alpha = portfolio_return - beta * benchmark_return
            
            return {
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'beta': beta,
                'alpha': alpha,
                'correlation': np.corrcoef(portfolio_returns, benchmark_returns)[0, 1],
                'outperformance_days': len(excess_returns[excess_returns > 0]) / len(excess_returns)
            }
            
        except Exception as e:
            logger.error(f"基准比较失败: {e}")
            return {}
    
    def _calculate_periodic_performance(self) -> Dict[str, Any]:
        """计算周期性表现"""
        try:
            if len(self.equity_curve) < 30:
                return {}
            
            # 按天分组计算表现
            daily_performance = {}
            monthly_performance = {}
            
            for item in self.equity_curve:
                timestamp = item['timestamp']
                equity = item['equity']
                
                # 按月分组
                month_key = timestamp.strftime('%Y-%m')
                if month_key not in monthly_performance:
                    monthly_performance[month_key] = {'start': equity, 'end': equity, 'high': equity, 'low': equity}
                else:
                    monthly_performance[month_key]['end'] = equity
                    monthly_performance[month_key]['high'] = max(monthly_performance[month_key]['high'], equity)
                    monthly_performance[month_key]['low'] = min(monthly_performance[month_key]['low'], equity)
            
            # 计算月度收益
            monthly_returns = {}
            for month, data in monthly_performance.items():
                if data['start'] > 0:
                    monthly_returns[month] = (data['end'] - data['start']) / data['start']
            
            return {
                'monthly_returns': monthly_returns,
                'best_month': max(monthly_returns.values()) if monthly_returns else 0.0,
                'worst_month': min(monthly_returns.values()) if monthly_returns else 0.0,
                'positive_months': len([r for r in monthly_returns.values() if r > 0]),
                'total_months': len(monthly_returns)
            }
            
        except Exception as e:
            logger.error(f"计算周期性表现失败: {e}")
            return {}
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """获取实时指标"""
        try:
            if len(self.equity_curve) < 2:
                return {}
            
            current_equity = self.equity_curve[-1]['equity']
            initial_equity = self.initial_balance
            
            # 当前收益率
            current_return = (current_equity - initial_equity) / initial_equity
            
            # 今日表现
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            today_equity = [item for item in self.equity_curve if item['timestamp'] >= today_start]
            
            daily_return = 0.0
            if len(today_equity) >= 2:
                daily_return = (today_equity[-1]['equity'] - today_equity[0]['equity']) / today_equity[0]['equity']
            
            # 当前回撤
            equity_values = [item['equity'] for item in self.equity_curve]
            peak_equity = max(equity_values)
            current_drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0
            
            return {
                'current_equity': current_equity,
                'current_return': current_return,
                'daily_return': daily_return,
                'current_drawdown': current_drawdown,
                'peak_equity': peak_equity,
                'total_trades_today': len([t for t in self.trade_history 
                                         if t['timestamp'].date() == datetime.utcnow().date()]),
                'last_update': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取实时指标失败: {e}")
            return {}

# 全局性能分析器实例
performance_analyzer = PerformanceAnalyzer()