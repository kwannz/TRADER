#!/usr/bin/env python3
"""
Risk Indicators Checker
加密货币风险指标专业检查工具 - 实时监控四大核心风险因子
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Rich美化输出
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.columns import Columns
    from rich.text import Text
    from rich.layout import Layout
except ImportError:
    Console = object

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.factor_engine.crypto_specialized.crypto_factor_utils import CryptoFactorUtils, CryptoDataProcessor

class RiskIndicatorsChecker:
    """加密货币风险指标专业检查工具"""
    
    def __init__(self):
        self.console = Console() if Console != object else None
        self.crypto_utils = CryptoFactorUtils()
        self.data_processor = CryptoDataProcessor()
        
        # 风险阈值配置
        self.thresholds = {
            'funding_rate': {
                'extreme': 1.5,      # 极端动量阈值
                'high': 0.8,         # 高动量阈值
                'normal_max': 0.5,   # 正常范围上限
                'normal_min': -0.5   # 正常范围下限
            },
            'whale_alert': {
                'high_impact': 2.0,      # 高影响阈值
                'medium_impact': 1.0,    # 中等影响阈值
                'frequency_high': 10,    # 高频率阈值（次/周）
                'frequency_medium': 5    # 中等频率阈值
            },
            'fear_greed': {
                'extreme_greed': 75,     # 极度贪婪
                'greed': 60,            # 贪婪
                'neutral_high': 55,      # 中性上限
                'neutral_low': 45,       # 中性下限
                'fear': 25,             # 恐惧
                'extreme_fear': 25       # 极度恐惧
            },
            'liquidity_risk': {
                'high_volatility': 150,     # 高波动率 (年化%)
                'medium_volatility': 100,   # 中等波动率
                'low_volatility': 50        # 低波动率
            }
        }
    
    def print_header(self, title, emoji="🔍"):
        """打印美化的标题"""
        if self.console:
            panel = Panel(
                f"[bold cyan]{emoji} {title}[/bold cyan]",
                style="bright_blue"
            )
            self.console.print(panel)
        else:
            print(f"\n{emoji} {title}")
            print("=" * 60)
    
    def print_section(self, title, emoji="📊"):
        """打印章节标题"""
        if self.console:
            text = Text(f"{emoji} {title}", style="bold yellow")
            self.console.print(f"\n{text}")
            self.console.print("-" * 50)
        else:
            print(f"\n{emoji} {title}")
            print("-" * 40)
    
    def generate_test_data(self, symbols=['BTC/USDT', 'ETH/USDT'], days=30):
        """生成测试数据"""
        self.print_section("生成测试市场数据", "🔄")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='1h')[:500]
        
        market_data = {}
        
        for symbol in symbols:
            np.random.seed(hash(symbol) % 1000)  # 确保可重现
            
            # 基础价格
            base_prices = {'BTC/USDT': 45000, 'ETH/USDT': 3000, 'BNB/USDT': 400}
            base_price = base_prices.get(symbol, 1000)
            
            # 生成价格路径（几何布朗运动 + 跳跃）
            mu = 0.0001
            sigma = 0.02 if symbol.startswith('BTC') else 0.025
            dt = 1/24
            
            returns = np.random.normal(mu*dt, sigma*np.sqrt(dt), len(dates))
            
            # 添加跳跃过程（加密市场特色）
            jump_intensity = 0.02
            jump_size = np.random.normal(0, 0.05, len(dates))
            jumps = np.random.binomial(1, jump_intensity, len(dates)) * jump_size
            
            total_returns = returns + jumps
            prices = base_price * np.exp(np.cumsum(total_returns))
            
            # 成交量（对数正态分布）
            volume = np.random.lognormal(13 if symbol.startswith('BTC') else 12, 0.5, len(dates))
            
            # 成交额
            amount = prices * volume
            
            market_data[symbol] = {
                'price': pd.Series(prices, index=dates),
                'volume': pd.Series(volume, index=dates),
                'amount': pd.Series(amount, index=dates)
            }
        
        # 生成资金费率数据（8小时间隔）
        funding_dates = pd.date_range(start=start_date, end=end_date, freq='8h')[:90]
        funding_rates = np.random.normal(0.0001, 0.0005, len(funding_dates))
        
        # 添加一些极端费率
        extreme_indices = np.random.choice(len(funding_rates), size=5, replace=False)
        funding_rates[extreme_indices] *= np.random.choice([-1, 1], size=5) * np.random.uniform(10, 20, size=5)
        
        market_data['funding_rates'] = pd.Series(funding_rates, index=funding_dates)
        
        print(f"✅ 生成完成: {len(symbols)} 币种 × {len(dates)} 数据点")
        print(f"📅 时间范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
        print(f"💰 资金费率: {len(funding_dates)} 个数据点")
        
        return market_data
    
    def check_funding_rate_momentum(self, funding_rates):
        """检查资金费率动量指标"""
        self.print_section("资金费率动量检查", "💰")
        
        # 计算动量
        momentum = self.crypto_utils.FUNDING_RATE_MOMENTUM(funding_rates, window=24)
        current_momentum = momentum.dropna().iloc[-1] if not momentum.dropna().empty else 0
        
        # 数据质量检查
        data_quality = {
            'total_points': len(funding_rates),
            'valid_points': len(momentum.dropna()),
            'missing_rate': (len(momentum) - len(momentum.dropna())) / len(momentum) * 100,
            'extreme_rates': (funding_rates.abs() > 0.01).sum(),
            'data_range': f"{funding_rates.min():.6f} ~ {funding_rates.max():.6f}"
        }
        
        # 风险评估
        if abs(current_momentum) > self.thresholds['funding_rate']['extreme']:
            risk_level = "🔴 极端"
            risk_desc = "费率动量极端，强烈反转信号"
            if current_momentum > 0:
                action = "考虑做空，费率过于偏多"
            else:
                action = "考虑做多，费率过于偏空"
        elif abs(current_momentum) > self.thresholds['funding_rate']['high']:
            risk_level = "🟠 偏高"
            risk_desc = "费率动量偏高，关注反转机会"
            action = "谨慎操作，观察费率变化趋势"
        elif (self.thresholds['funding_rate']['normal_min'] <= 
              current_momentum <= self.thresholds['funding_rate']['normal_max']):
            risk_level = "🟢 正常"
            risk_desc = "费率动量在正常范围"
            action = "正常交易，继续监控"
        else:
            risk_level = "🟡 中等"
            risk_desc = "费率动量中等水平"
            action = "保持观察，注意趋势变化"
        
        # 历史统计
        momentum_stats = {
            'mean': momentum.dropna().mean(),
            'std': momentum.dropna().std(),
            'max': momentum.dropna().max(),
            'min': momentum.dropna().min(),
            'extreme_count': (abs(momentum.dropna()) > 1.5).sum()
        }
        
        # 输出结果
        if self.console:
            # 创建检查结果表格
            table = Table(title="资金费率动量检查结果")
            table.add_column("检查项目", style="cyan", no_wrap=True)
            table.add_column("数值", style="white")
            table.add_column("状态", style="green")
            
            table.add_row("当前动量值", f"{current_momentum:.3f}", risk_level)
            table.add_row("风险描述", risk_desc, "")
            table.add_row("操作建议", action, "")
            table.add_row("数据完整性", f"{data_quality['valid_points']}/{data_quality['total_points']}", 
                         "✅ 良好" if data_quality['missing_rate'] < 10 else "⚠️ 注意")
            table.add_row("极端费率次数", str(data_quality['extreme_rates']), 
                         "⚠️ 频繁" if data_quality['extreme_rates'] > 5 else "✅ 正常")
            table.add_row("历史极端次数", str(momentum_stats['extreme_count']), "")
            
            self.console.print(table)
        else:
            print(f"当前资金费率动量: {current_momentum:.3f}")
            print(f"风险等级: {risk_level}")
            print(f"操作建议: {action}")
            print(f"数据完整性: {data_quality['valid_points']}/{data_quality['total_points']}")
        
        return {
            'current_value': current_momentum,
            'risk_level': risk_level,
            'risk_description': risk_desc,
            'action_advice': action,
            'data_quality': data_quality,
            'statistics': momentum_stats
        }
    
    def check_whale_alert(self, volume, amount):
        """检查巨鲸交易检测"""
        self.print_section("巨鲸交易检测检查", "🐋")
        
        # 计算巨鲸预警
        whale_alerts = self.crypto_utils.WHALE_ALERT(volume, amount, threshold_std=2.5)
        
        # 统计分析
        significant_whales = whale_alerts[abs(whale_alerts) > 1.0]
        high_impact_whales = whale_alerts[abs(whale_alerts) > 2.0]
        
        # 数据质量检查
        data_quality = {
            'total_points': len(whale_alerts),
            'valid_points': len(whale_alerts.dropna()),
            'missing_rate': whale_alerts.isna().mean() * 100,
            'zero_volume_count': (volume == 0).sum(),
            'volume_range': f"{volume.min():.0f} ~ {volume.max():.0f}"
        }
        
        # 巨鲸活动分析
        whale_stats = {
            'total_alerts': len(significant_whales),
            'high_impact_count': len(high_impact_whales),
            'alert_frequency': len(significant_whales) / (len(whale_alerts) / (24*7)),  # 每周频率
            'max_alert_strength': whale_alerts.abs().max() if not whale_alerts.empty else 0,
            'avg_alert_strength': significant_whales.abs().mean() if not significant_whales.empty else 0
        }
        
        # 风险评估
        weekly_frequency = whale_stats['alert_frequency']
        if weekly_frequency > self.thresholds['whale_alert']['frequency_high']:
            liquidity_risk = "🔴 高风险"
            risk_desc = "巨鲸交易频繁，流动性受到显著影响"
            action = "谨慎大额交易，关注市场深度"
        elif weekly_frequency > self.thresholds['whale_alert']['frequency_medium']:
            liquidity_risk = "🟠 中等风险"
            risk_desc = "巨鲸活动较为活跃，需要关注"
            action = "适度降低交易规模，监控大户动向"
        else:
            liquidity_risk = "🟢 低风险"
            risk_desc = "巨鲸交易活动正常，流动性良好"
            action = "正常交易，保持监控"
        
        # 最近巨鲸活动
        recent_whales = significant_whales.tail(5)
        
        # 输出结果
        if self.console:
            table = Table(title="巨鲸交易检测结果")
            table.add_column("检查项目", style="cyan")
            table.add_column("数值", style="white")
            table.add_column("状态", style="green")
            
            table.add_row("检测总次数", str(whale_stats['total_alerts']), liquidity_risk)
            table.add_row("高影响交易", str(whale_stats['high_impact_count']), 
                         "⚠️ 关注" if whale_stats['high_impact_count'] > 3 else "✅ 正常")
            table.add_row("每周频率", f"{weekly_frequency:.1f} 次", "")
            table.add_row("最大警报强度", f"{whale_stats['max_alert_strength']:.2f}", "")
            table.add_row("数据质量", f"{data_quality['valid_points']}/{data_quality['total_points']}", 
                         "✅ 良好" if data_quality['missing_rate'] < 5 else "⚠️ 注意")
            table.add_row("零成交量", str(data_quality['zero_volume_count']), 
                         "⚠️ 异常" if data_quality['zero_volume_count'] > 0 else "✅ 正常")
            
            self.console.print(table)
            
            # 显示最近巨鲸活动
            if len(recent_whales) > 0:
                whale_activity_table = Table(title="最近巨鲸交易活动")
                whale_activity_table.add_column("时间", style="cyan")
                whale_activity_table.add_column("警报强度", style="yellow")
                whale_activity_table.add_column("影响评估", style="red")
                
                for timestamp, alert_value in recent_whales.items():
                    impact = "🔴 高影响" if abs(alert_value) > 2 else "🟡 中等影响"
                    whale_activity_table.add_row(
                        timestamp.strftime("%m-%d %H:%M"),
                        f"{alert_value:.2f}",
                        impact
                    )
                
                self.console.print(whale_activity_table)
        
        else:
            print(f"巨鲸检测总数: {whale_stats['total_alerts']} 次")
            print(f"流动性风险: {liquidity_risk}")
            print(f"每周频率: {weekly_frequency:.1f} 次")
            print(f"操作建议: {action}")
        
        return {
            'total_alerts': whale_stats['total_alerts'],
            'weekly_frequency': weekly_frequency,
            'risk_level': liquidity_risk,
            'risk_description': risk_desc,
            'action_advice': action,
            'data_quality': data_quality,
            'statistics': whale_stats,
            'recent_activity': recent_whales
        }
    
    def check_fear_greed_index(self, price, volume):
        """检查恐惧贪婪指数"""
        self.print_section("恐惧贪婪指数检查", "😰")
        
        # 计算恐惧贪婪指数
        fg_index = self.crypto_utils.FEAR_GREED_INDEX(price, volume)
        current_fg = fg_index.dropna().iloc[-1] if not fg_index.dropna().empty else 50
        
        # 数据质量检查
        data_quality = {
            'total_points': len(fg_index),
            'valid_points': len(fg_index.dropna()),
            'missing_rate': fg_index.isna().mean() * 100,
            'price_range': f"{price.min():.2f} ~ {price.max():.2f}",
            'volume_zeros': (volume == 0).sum()
        }
        
        # 历史统计
        fg_stats = {
            'mean': fg_index.dropna().mean(),
            'std': fg_index.dropna().std(),
            'max': fg_index.dropna().max(),
            'min': fg_index.dropna().min(),
            'extreme_greed_periods': (fg_index.dropna() > 75).sum(),
            'extreme_fear_periods': (fg_index.dropna() < 25).sum()
        }
        
        # 风险评估和建议
        if current_fg > self.thresholds['fear_greed']['extreme_greed']:
            emotion_state = "🔴 极度贪婪"
            risk_desc = "市场情绪过热，存在回调风险"
            action = "考虑减仓，准备应对反转"
            signal_strength = "强烈卖出信号"
        elif current_fg > self.thresholds['fear_greed']['greed']:
            emotion_state = "🟠 贪婪"
            risk_desc = "市场情绪偏热，需要谨慎"
            action = "谨慎操作，观察反转信号"
            signal_strength = "弱卖出信号"
        elif current_fg > self.thresholds['fear_greed']['neutral_high']:
            emotion_state = "🟡 偏向贪婪"
            risk_desc = "市场情绪略偏乐观"
            action = "保持观察，适度操作"
            signal_strength = "中性偏空"
        elif current_fg > self.thresholds['fear_greed']['neutral_low']:
            emotion_state = "⚪ 中性"
            risk_desc = "市场情绪平衡"
            action = "观察等待，寻找方向"
            signal_strength = "中性"
        elif current_fg > self.thresholds['fear_greed']['fear']:
            emotion_state = "🔵 恐惧"
            risk_desc = "市场情绪偏悲观，潜在机会"
            action = "关注抄底机会"
            signal_strength = "弱买入信号"
        else:
            emotion_state = "🟢 极度恐惧"
            risk_desc = "市场情绪极度悲观，通常是好的买入时机"
            action = "积极寻找抄底机会"
            signal_strength = "强烈买入信号"
        
        # 趋势分析
        if len(fg_index.dropna()) >= 24:
            recent_trend = fg_index.dropna().tail(24).pct_change().mean()
            if recent_trend > 0.01:
                trend_desc = "📈 快速上升"
            elif recent_trend > 0.005:
                trend_desc = "📈 缓慢上升"
            elif recent_trend < -0.01:
                trend_desc = "📉 快速下降"
            elif recent_trend < -0.005:
                trend_desc = "📉 缓慢下降"
            else:
                trend_desc = "➡️ 相对稳定"
        else:
            trend_desc = "📊 数据不足"
        
        # 输出结果
        if self.console:
            table = Table(title="恐惧贪婪指数检查结果")
            table.add_column("检查项目", style="cyan")
            table.add_column("数值", style="white")
            table.add_column("状态", style="green")
            
            table.add_row("当前指数", f"{current_fg:.1f}/100", emotion_state)
            table.add_row("市场情绪", emotion_state.split(' ', 1)[1], "")
            table.add_row("信号强度", signal_strength, "")
            table.add_row("近期趋势", trend_desc, "")
            table.add_row("操作建议", action, "")
            table.add_row("数据完整性", f"{data_quality['valid_points']}/{data_quality['total_points']}", 
                         "✅ 良好" if data_quality['missing_rate'] < 10 else "⚠️ 注意")
            table.add_row("历史均值", f"{fg_stats['mean']:.1f}", "")
            table.add_row("极端贪婪次数", str(fg_stats['extreme_greed_periods']), "")
            table.add_row("极端恐惧次数", str(fg_stats['extreme_fear_periods']), "")
            
            self.console.print(table)
        else:
            print(f"当前恐惧贪婪指数: {current_fg:.1f}/100")
            print(f"市场情绪: {emotion_state}")
            print(f"操作建议: {action}")
            print(f"信号强度: {signal_strength}")
        
        return {
            'current_value': current_fg,
            'emotion_state': emotion_state,
            'risk_description': risk_desc,
            'action_advice': action,
            'signal_strength': signal_strength,
            'trend_description': trend_desc,
            'data_quality': data_quality,
            'statistics': fg_stats
        }
    
    def check_liquidity_risk(self, price, volume):
        """检查流动性风险（基于波动率）"""
        self.print_section("流动性风险检查", "🌊")
        
        # 计算波动率指标
        returns = price.pct_change().dropna()
        volatility_daily = returns.std()
        volatility_annualized = volatility_daily * np.sqrt(365) * 100  # 年化百分比
        
        # 滚动波动率
        rolling_vol = returns.rolling(window=24).std() * np.sqrt(24*365) * 100
        
        # 数据质量检查
        data_quality = {
            'total_points': len(price),
            'valid_returns': len(returns),
            'missing_rate': (len(price) - len(returns)) / len(price) * 100,
            'price_jumps': (returns.abs() > 0.1).sum(),  # >10%价格跳跃
            'volume_consistency': (volume > 0).mean() * 100
        }
        
        # 波动率统计
        vol_stats = {
            'current_vol': volatility_annualized,
            'mean_vol': rolling_vol.mean(),
            'max_vol': rolling_vol.max(),
            'min_vol': rolling_vol.min(),
            'vol_percentile_90': rolling_vol.quantile(0.9),
            'high_vol_periods': (rolling_vol > 100).sum()
        }
        
        # 流动性风险评估
        if volatility_annualized > self.thresholds['liquidity_risk']['high_volatility']:
            liquidity_risk = "🔴 高风险"
            risk_desc = "市场波动率极高，流动性紧张"
            action = "减少交易规模，避免大额操作"
        elif volatility_annualized > self.thresholds['liquidity_risk']['medium_volatility']:
            liquidity_risk = "🟠 中等风险"
            risk_desc = "市场波动率偏高，需要谨慎"
            action = "适度降低仓位，关注市场深度"
        elif volatility_annualized < self.thresholds['liquidity_risk']['low_volatility']:
            liquidity_risk = "🟢 低风险"
            risk_desc = "市场波动率低，流动性良好"
            action = "正常交易，可适度增加仓位"
        else:
            liquidity_risk = "🟡 中等"
            risk_desc = "市场波动率正常"
            action = "正常操作，保持监控"
        
        # 市场深度预估（基于成交量稳定性）
        volume_cv = volume.std() / volume.mean()  # 变异系数
        if volume_cv < 0.5:
            market_depth = "🟢 深度良好"
        elif volume_cv < 1.0:
            market_depth = "🟡 深度一般"
        else:
            market_depth = "🔴 深度不足"
        
        # 输出结果
        if self.console:
            table = Table(title="流动性风险检查结果")
            table.add_column("检查项目", style="cyan")
            table.add_column("数值", style="white")
            table.add_column("状态", style="green")
            
            table.add_row("年化波动率", f"{volatility_annualized:.1f}%", liquidity_risk)
            table.add_row("风险描述", risk_desc, "")
            table.add_row("市场深度", market_depth, "")
            table.add_row("操作建议", action, "")
            table.add_row("价格跳跃次数", str(data_quality['price_jumps']), 
                         "⚠️ 频繁" if data_quality['price_jumps'] > 5 else "✅ 正常")
            table.add_row("成交量一致性", f"{data_quality['volume_consistency']:.1f}%", 
                         "✅ 良好" if data_quality['volume_consistency'] > 95 else "⚠️ 注意")
            table.add_row("历史最大波动", f"{vol_stats['max_vol']:.1f}%", "")
            table.add_row("高波动期数", str(vol_stats['high_vol_periods']), "")
            
            self.console.print(table)
        else:
            print(f"年化波动率: {volatility_annualized:.1f}%")
            print(f"流动性风险: {liquidity_risk}")
            print(f"操作建议: {action}")
        
        return {
            'volatility_annualized': volatility_annualized,
            'risk_level': liquidity_risk,
            'risk_description': risk_desc,
            'action_advice': action,
            'market_depth': market_depth,
            'data_quality': data_quality,
            'statistics': vol_stats
        }
    
    def generate_comprehensive_report(self, funding_result, whale_result, fg_result, liquidity_result):
        """生成综合风险报告"""
        self.print_section("综合风险评估报告", "📋")
        
        # 计算综合风险评分
        risk_scores = {
            '🔴 极端': 5, '🔴 高风险': 4, '🔴 高影响': 4, '🔴 极度贪婪': 4,
            '🟠 偏高': 3, '🟠 中等风险': 3, '🟠 贪婪': 3,
            '🟡 中等': 2, '🟡 偏向贪婪': 2, '🟡 中等影响': 2,
            '🟢 正常': 1, '🟢 低风险': 1, '🟢 极度恐惧': 1,
            '⚪ 中性': 1, '🔵 恐惧': 1
        }
        
        # 提取风险等级
        funding_score = risk_scores.get(funding_result['risk_level'], 2)
        whale_score = risk_scores.get(whale_result['risk_level'], 2)
        fg_score = risk_scores.get(fg_result['emotion_state'], 2)
        liquidity_score = risk_scores.get(liquidity_result['risk_level'], 2)
        
        # 计算加权综合评分
        weights = {'funding': 0.3, 'whale': 0.25, 'fear_greed': 0.25, 'liquidity': 0.2}
        total_score = (
            funding_score * weights['funding'] +
            whale_score * weights['whale'] + 
            fg_score * weights['fear_greed'] +
            liquidity_score * weights['liquidity']
        )
        
        risk_percentage = (total_score / 5) * 100
        
        # 综合风险等级判断
        if risk_percentage > 70:
            overall_risk = "🔴 高风险"
            overall_advice = "建议降低仓位，避免高风险交易"
        elif risk_percentage > 50:
            overall_risk = "🟠 中高风险"  
            overall_advice = "谨慎操作，密切监控市场变化"
        elif risk_percentage > 30:
            overall_risk = "🟡 中等风险"
            overall_advice = "正常操作，保持风险意识"
        else:
            overall_risk = "🟢 低风险"
            overall_advice = "相对安全，可考虑适度增仓"
        
        # 生成操作建议
        action_recommendations = []
        
        # 基于资金费率的建议
        if abs(funding_result['current_value']) > 1.5:
            if funding_result['current_value'] > 0:
                action_recommendations.append("资金费率极度偏多，考虑做空机会")
            else:
                action_recommendations.append("资金费率极度偏空，考虑做多机会")
        
        # 基于巨鲸活动的建议
        if whale_result['weekly_frequency'] > 10:
            action_recommendations.append("巨鲸交易频繁，减小交易规模")
        
        # 基于恐惧贪婪的建议
        if fg_result['current_value'] > 75:
            action_recommendations.append("市场极度贪婪，准备获利了结")
        elif fg_result['current_value'] < 25:
            action_recommendations.append("市场极度恐惧，寻找抄底机会")
        
        # 基于流动性的建议  
        if liquidity_result['volatility_annualized'] > 150:
            action_recommendations.append("波动率极高，避免大额交易")
        
        # 输出综合报告
        if self.console:
            # 风险概览表
            overview_table = Table(title="风险指标概览")
            overview_table.add_column("风险类别", style="cyan")
            overview_table.add_column("当前状态", style="white")
            overview_table.add_column("风险等级", style="yellow")
            overview_table.add_column("权重", style="dim")
            
            overview_table.add_row(
                "资金费率动量", 
                f"{funding_result['current_value']:.3f}",
                funding_result['risk_level'],
                f"{weights['funding']*100:.0f}%"
            )
            overview_table.add_row(
                "巨鲸交易活动",
                f"{whale_result['weekly_frequency']:.1f} 次/周",
                whale_result['risk_level'],
                f"{weights['whale']*100:.0f}%"
            )
            overview_table.add_row(
                "恐惧贪婪指数",
                f"{fg_result['current_value']:.1f}/100",
                fg_result['emotion_state'],
                f"{weights['fear_greed']*100:.0f}%"
            )
            overview_table.add_row(
                "流动性风险",
                f"{liquidity_result['volatility_annualized']:.1f}%",
                liquidity_result['risk_level'],
                f"{weights['liquidity']*100:.0f}%"
            )
            
            self.console.print(overview_table)
            
            # 综合评估结果
            summary_panel = Panel(
                f"""[bold]🎯 综合风险评估结果[/bold]

[yellow]综合风险等级:[/yellow] {overall_risk}
[yellow]风险评分:[/yellow] {risk_percentage:.1f}/100
[yellow]总体建议:[/yellow] {overall_advice}

[cyan]具体操作建议:[/cyan]
{chr(10).join(['• ' + rec for rec in action_recommendations]) if action_recommendations else '• 当前风险状况正常，继续监控'}

[dim]评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
风险模型: 四维度加权评估 (资金费率30% + 巨鲸活动25% + 情绪指标25% + 流动性20%)[/dim]
                """,
                title="📊 综合风险报告",
                border_style="bright_blue"
            )
            self.console.print(summary_panel)
        
        else:
            print(f"\n📊 综合风险评估:")
            print(f"综合风险等级: {overall_risk}")
            print(f"风险评分: {risk_percentage:.1f}/100") 
            print(f"总体建议: {overall_advice}")
            if action_recommendations:
                print("\n具体建议:")
                for rec in action_recommendations:
                    print(f"• {rec}")
        
        return {
            'overall_risk_level': overall_risk,
            'risk_percentage': risk_percentage,
            'overall_advice': overall_advice,
            'action_recommendations': action_recommendations,
            'component_scores': {
                'funding': funding_score,
                'whale': whale_score,
                'fear_greed': fg_score,
                'liquidity': liquidity_score
            },
            'weights': weights
        }
    
    def run_full_check(self, symbols=['BTC/USDT', 'ETH/USDT']):
        """运行完整的风险指标检查"""
        self.print_header("加密货币风险指标全面检查", "🔍")
        
        print("🚀 开始全面风险指标检查...")
        print(f"📊 检查币种: {', '.join(symbols)}")
        print(f"⏰ 检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 生成测试数据
        market_data = self.generate_test_data(symbols)
        
        # 使用第一个币种的数据进行分析（可扩展为多币种）
        symbol = symbols[0]
        price = market_data[symbol]['price']
        volume = market_data[symbol]['volume']
        amount = market_data[symbol]['amount']
        funding_rates = market_data['funding_rates']
        
        print(f"\n🎯 正在分析 {symbol} 的风险指标...")
        
        # 执行四大风险检查
        funding_result = self.check_funding_rate_momentum(funding_rates)
        whale_result = self.check_whale_alert(volume, amount)
        fg_result = self.check_fear_greed_index(price, volume)
        liquidity_result = self.check_liquidity_risk(price, volume)
        
        # 生成综合报告
        comprehensive_result = self.generate_comprehensive_report(
            funding_result, whale_result, fg_result, liquidity_result
        )
        
        # 返回完整结果
        return {
            'symbol': symbol,
            'check_time': datetime.now(),
            'funding_rate': funding_result,
            'whale_alert': whale_result,
            'fear_greed': fg_result,
            'liquidity_risk': liquidity_result,
            'comprehensive': comprehensive_result,
            'market_data_summary': {
                'data_points': len(price),
                'time_range': f"{price.index[0].strftime('%Y-%m-%d')} ~ {price.index[-1].strftime('%Y-%m-%d')}",
                'price_change': (price.iloc[-1] / price.iloc[0] - 1) * 100
            }
        }


def main():
    """主函数"""
    checker = RiskIndicatorsChecker()
    
    try:
        # 运行完整检查
        result = checker.run_full_check(['BTC/USDT'])
        
        print("\n🎉 风险指标检查完成！")
        print("\n💡 使用说明:")
        print("• 本工具提供4个核心风险指标的专业检查")
        print("• 建议每日运行一次，及时了解市场风险变化")
        print("• 可通过修改 symbols 参数检查不同币种")
        print("• 风险评估结果仅供参考，请结合实际市场情况判断")
        
        return result
        
    except Exception as e:
        print(f"❌ 检查过程出现错误: {str(e)}")
        print("🔧 请检查数据源和网络连接")
        return None


if __name__ == "__main__":
    main()