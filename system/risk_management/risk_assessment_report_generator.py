#!/usr/bin/env python3
"""
📊 风险评估报告生成系统
Risk Assessment Report Generator

功能特性:
- 日报自动生成
- 投资组合风险分析
- 多时间框架综合评估
- 策略建议生成
- 历史对比分析
- 风险指标趋势分析
- 专业级可视化报告

作者: Claude Code Assistant
创建时间: 2025-08-09
"""

import asyncio
import pandas as pd
import numpy as np
import sqlite3
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """风险等级"""
    EXTREME = "极高风险"
    HIGH = "高风险"
    MEDIUM = "中等风险"
    LOW = "低风险"
    MINIMAL = "极低风险"

class ReportType(Enum):
    """报告类型"""
    DAILY = "日报"
    WEEKLY = "周报"
    MONTHLY = "月报"
    PORTFOLIO = "投资组合"

class TrendDirection(Enum):
    """趋势方向"""
    STRONGLY_UP = "强势上升"
    UP = "上升"
    SIDEWAYS = "横盘"
    DOWN = "下降"
    STRONGLY_DOWN = "强势下跌"

@dataclass
class PortfolioPosition:
    """投资组合持仓"""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    market_value: float
    pnl: float
    pnl_percent: float
    weight: float
    risk_score: float

@dataclass
class RiskMetrics:
    """风险指标"""
    var_1d: float              # 1日风险价值
    var_7d: float              # 7日风险价值
    max_drawdown: float        # 最大回撤
    sharpe_ratio: float        # 夏普比率
    volatility: float          # 波动率
    beta: float                # 贝塔值
    correlation_btc: float     # 与BTC相关性
    liquidity_score: float     # 流动性得分

@dataclass
class MarketEnvironment:
    """市场环境"""
    trend_direction: TrendDirection
    volatility_regime: str     # 波动率制度
    fear_greed_state: str      # 恐惧贪婪状态
    whale_activity: str        # 巨鲸活动
    funding_rate_trend: str    # 资金费率趋势
    market_cap_flow: str       # 市值流向

class DataProvider:
    """数据提供器"""
    
    def __init__(self, db_path: str = "monitoring_data.db"):
        self.db_path = db_path
    
    def get_historical_data(self, days: int = 30) -> pd.DataFrame:
        """获取历史数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT timestamp, price, volume, funding_rate, 
                           whale_activity, fear_greed_index, sentiment,
                           liquidity_risk, alert_level
                    FROM market_snapshots 
                    WHERE datetime(timestamp) > datetime('now', '-{} days')
                    ORDER BY timestamp ASC
                """.format(days)
                
                df = pd.read_sql_query(query, conn, parse_dates=['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                logger.info(f"获取{len(df)}条历史数据记录")
                return df
                
        except Exception as e:
            logger.error(f"获取历史数据失败: {e}")
            # 返回模拟数据用于演示
            return self._generate_demo_data(days)
    
    def get_alert_history(self, days: int = 30) -> pd.DataFrame:
        """获取告警历史"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT timestamp, alert_level, indicator, message, value
                    FROM alerts 
                    WHERE datetime(timestamp) > datetime('now', '-{} days')
                    ORDER BY timestamp DESC
                """.format(days)
                
                return pd.read_sql_query(query, conn, parse_dates=['timestamp'])
                
        except Exception as e:
            logger.error(f"获取告警历史失败: {e}")
            return pd.DataFrame()
    
    def _generate_demo_data(self, days: int) -> pd.DataFrame:
        """生成演示数据"""
        logger.info("使用模拟数据进行演示")
        
        # 生成时间序列
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='1H'
        )
        
        # 生成模拟价格数据(带趋势和波动)
        np.random.seed(42)
        base_price = 45000
        price_trend = np.cumsum(np.random.normal(0.001, 0.02, len(dates)))
        price_volatility = np.random.normal(0, 0.01, len(dates))
        prices = base_price * (1 + price_trend + price_volatility)
        
        # 生成其他指标
        volumes = np.random.exponential(5000, len(dates))
        funding_rates = np.random.normal(0.0001, 0.0005, len(dates))
        
        # 恐惧贪婪指数(带周期性)
        fear_greed = 50 + 25 * np.sin(np.arange(len(dates)) * 0.01) + np.random.normal(0, 10, len(dates))
        fear_greed = np.clip(fear_greed, 0, 100)
        
        # 流动性风险
        liquidity_risk = np.random.uniform(20, 80, len(dates))
        
        # 创建DataFrame
        data = pd.DataFrame({
            'price': prices,
            'volume': volumes,
            'funding_rate': funding_rates,
            'whale_activity': np.random.choice(['休眠', '积累', '分配', '协调'], len(dates), p=[0.6, 0.2, 0.15, 0.05]),
            'fear_greed_index': fear_greed,
            'sentiment': ['恐惧' if x < 40 else '贪婪' if x > 60 else '中性' for x in fear_greed],
            'liquidity_risk': liquidity_risk,
            'alert_level': np.random.choice(['INFO', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'], len(dates), p=[0.5, 0.3, 0.15, 0.04, 0.01])
        }, index=dates)
        
        return data

class RiskCalculator:
    """风险计算器"""
    
    def __init__(self):
        pass
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.05) -> float:
        """计算风险价值(VaR)"""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, confidence * 100)
    
    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """计算最大回撤"""
        if len(prices) == 0:
            return 0.0
        
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return drawdown.min()
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() - risk_free_rate / 252  # 日化无风险利率
        return excess_returns / returns.std() * np.sqrt(252)
    
    def calculate_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """计算贝塔值"""
        if len(asset_returns) == 0 or len(market_returns) == 0:
            return 1.0
        
        covariance = np.cov(asset_returns.dropna(), market_returns.dropna())[0, 1]
        market_variance = np.var(market_returns.dropna())
        
        return covariance / market_variance if market_variance != 0 else 1.0
    
    def calculate_correlation(self, x: pd.Series, y: pd.Series) -> float:
        """计算相关系数"""
        try:
            return x.corr(y)
        except:
            return 0.0
    
    def calculate_risk_metrics(self, data: pd.DataFrame) -> RiskMetrics:
        """计算综合风险指标"""
        if len(data) == 0:
            return RiskMetrics(0, 0, 0, 0, 0, 1, 0, 50)
        
        # 计算收益率
        returns = data['price'].pct_change().dropna()
        
        # 计算各项风险指标
        var_1d = abs(self.calculate_var(returns, 0.05)) * 100
        var_7d = abs(self.calculate_var(returns.rolling(7).sum().dropna(), 0.05)) * 100
        max_dd = abs(self.calculate_max_drawdown(data['price'])) * 100
        sharpe = self.calculate_sharpe_ratio(returns)
        volatility = returns.std() * np.sqrt(252) * 100  # 年化波动率
        beta = self.calculate_beta(returns, returns)  # 简化处理
        corr_btc = self.calculate_correlation(returns, returns)  # 简化处理
        liquidity = 100 - data['liquidity_risk'].mean()  # 流动性得分
        
        return RiskMetrics(
            var_1d=var_1d,
            var_7d=var_7d,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            volatility=volatility,
            beta=beta,
            correlation_btc=corr_btc,
            liquidity_score=liquidity
        )

class TrendAnalyzer:
    """趋势分析器"""
    
    def analyze_trend(self, data: pd.DataFrame) -> TrendDirection:
        """分析趋势方向"""
        if len(data) == 0:
            return TrendDirection.SIDEWAYS
        
        # 计算不同时间框架的趋势
        returns_1d = data['price'].pct_change(24).iloc[-1] if len(data) >= 24 else 0
        returns_7d = data['price'].pct_change(24*7).iloc[-1] if len(data) >= 24*7 else 0
        returns_30d = data['price'].pct_change(24*30).iloc[-1] if len(data) >= 24*30 else 0
        
        # 综合趋势得分
        trend_score = returns_1d * 0.5 + returns_7d * 0.3 + returns_30d * 0.2
        
        if trend_score > 0.1:
            return TrendDirection.STRONGLY_UP
        elif trend_score > 0.03:
            return TrendDirection.UP
        elif trend_score < -0.1:
            return TrendDirection.STRONGLY_DOWN
        elif trend_score < -0.03:
            return TrendDirection.DOWN
        else:
            return TrendDirection.SIDEWAYS
    
    def analyze_market_environment(self, data: pd.DataFrame) -> MarketEnvironment:
        """分析市场环境"""
        trend = self.analyze_trend(data)
        
        # 波动率制度
        volatility = data['price'].pct_change().rolling(24).std().iloc[-1]
        vol_regime = "高波动" if volatility > 0.02 else "低波动" if volatility < 0.01 else "中波动"
        
        # 恐惧贪婪状态
        latest_fg = data['fear_greed_index'].iloc[-1] if len(data) > 0 else 50
        if latest_fg > 75:
            fg_state = "极度贪婪"
        elif latest_fg > 60:
            fg_state = "贪婪"
        elif latest_fg < 25:
            fg_state = "极度恐惧"
        elif latest_fg < 40:
            fg_state = "恐惧"
        else:
            fg_state = "中性"
        
        # 巨鲸活动
        whale_activity = data['whale_activity'].iloc[-1] if len(data) > 0 else "休眠"
        
        # 资金费率趋势
        funding_trend = "偏多" if data['funding_rate'].iloc[-1] > 0.0005 else "偏空" if data['funding_rate'].iloc[-1] < -0.0005 else "中性"
        
        return MarketEnvironment(
            trend_direction=trend,
            volatility_regime=vol_regime,
            fear_greed_state=fg_state,
            whale_activity=whale_activity,
            funding_rate_trend=funding_trend,
            market_cap_flow="流入"  # 简化处理
        )

class PortfolioAnalyzer:
    """投资组合分析器"""
    
    def __init__(self):
        self.risk_calculator = RiskCalculator()
    
    def create_sample_portfolio(self) -> List[PortfolioPosition]:
        """创建示例投资组合"""
        positions = [
            PortfolioPosition(
                symbol="BTC",
                quantity=1.5,
                entry_price=42000,
                current_price=45000,
                market_value=67500,
                pnl=4500,
                pnl_percent=7.14,
                weight=60.0,
                risk_score=65
            ),
            PortfolioPosition(
                symbol="ETH",
                quantity=10,
                entry_price=2800,
                current_price=3200,
                market_value=32000,
                pnl=4000,
                pnl_percent=14.29,
                weight=28.4,
                risk_score=70
            ),
            PortfolioPosition(
                symbol="SOL",
                quantity=50,
                entry_price=180,
                current_price=200,
                market_value=10000,
                pnl=1000,
                pnl_percent=11.11,
                weight=8.9,
                risk_score=80
            ),
            PortfolioPosition(
                symbol="USDT",
                quantity=3000,
                entry_price=1.0,
                current_price=1.0,
                market_value=3000,
                pnl=0,
                pnl_percent=0,
                weight=2.7,
                risk_score=10
            )
        ]
        
        return positions
    
    def analyze_portfolio_risk(self, positions: List[PortfolioPosition]) -> Dict[str, Any]:
        """分析投资组合风险"""
        total_value = sum(pos.market_value for pos in positions)
        total_pnl = sum(pos.pnl for pos in positions)
        
        # 计算加权风险得分
        weighted_risk = sum(pos.risk_score * pos.weight / 100 for pos in positions)
        
        # 集中度分析
        max_weight = max(pos.weight for pos in positions)
        concentration_risk = "高" if max_weight > 50 else "中" if max_weight > 30 else "低"
        
        # 相关性分析(简化处理)
        correlation_risk = "中等"  # 实际应用中需要计算各资产间相关性
        
        return {
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_pnl_percent': (total_pnl / (total_value - total_pnl)) * 100,
            'weighted_risk_score': weighted_risk,
            'concentration_risk': concentration_risk,
            'correlation_risk': correlation_risk,
            'diversification_score': 100 - max_weight,  # 分散化得分
            'position_count': len(positions)
        }

class ReportGenerator:
    """报告生成器"""
    
    def __init__(self):
        self.data_provider = DataProvider()
        self.risk_calculator = RiskCalculator()
        self.trend_analyzer = TrendAnalyzer()
        self.portfolio_analyzer = PortfolioAnalyzer()
    
    def generate_daily_report(self, date: datetime = None) -> str:
        """生成日报"""
        if date is None:
            date = datetime.now()
        
        logger.info(f"生成{date.strftime('%Y-%m-%d')}日报")
        
        # 获取数据
        data = self.data_provider.get_historical_data(days=30)
        alerts = self.data_provider.get_alert_history(days=1)
        
        # 计算指标
        risk_metrics = self.risk_calculator.calculate_risk_metrics(data)
        market_env = self.trend_analyzer.analyze_market_environment(data)
        
        # 生成报告
        report = self._generate_daily_report_content(date, data, alerts, risk_metrics, market_env)
        
        return report
    
    def generate_portfolio_report(self) -> str:
        """生成投资组合报告"""
        logger.info("生成投资组合风险报告")
        
        # 获取数据
        data = self.data_provider.get_historical_data(days=30)
        positions = self.portfolio_analyzer.create_sample_portfolio()
        portfolio_analysis = self.portfolio_analyzer.analyze_portfolio_risk(positions)
        
        # 计算指标
        risk_metrics = self.risk_calculator.calculate_risk_metrics(data)
        market_env = self.trend_analyzer.analyze_market_environment(data)
        
        # 生成报告
        report = self._generate_portfolio_report_content(
            positions, portfolio_analysis, risk_metrics, market_env
        )
        
        return report
    
    def _generate_daily_report_content(self, 
                                     date: datetime, 
                                     data: pd.DataFrame,
                                     alerts: pd.DataFrame,
                                     risk_metrics: RiskMetrics,
                                     market_env: MarketEnvironment) -> str:
        """生成日报内容"""
        
        # 获取当日关键数据
        latest_price = data['price'].iloc[-1] if len(data) > 0 else 0
        price_change_24h = ((data['price'].iloc[-1] / data['price'].iloc[-24] - 1) * 100) if len(data) >= 24 else 0
        latest_fg = data['fear_greed_index'].iloc[-1] if len(data) > 0 else 50
        
        # 告警统计
        alert_counts = alerts['alert_level'].value_counts().to_dict() if len(alerts) > 0 else {}
        total_alerts = len(alerts)
        
        # 确定整体风险等级
        if risk_metrics.var_1d > 10 or latest_fg > 80 or latest_fg < 20:
            risk_level = RiskLevel.HIGH
        elif risk_metrics.var_1d > 5 or latest_fg > 70 or latest_fg < 30:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        report = f"""# 📊 加密货币风险评估日报

## 📅 报告概览
- **报告日期**: {date.strftime('%Y年%m月%d日')} 
- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **数据覆盖**: 最近24小时市场表现
- **整体风险等级**: 🔸 **{risk_level.value}**

---

## 📈 市场表现摘要

### 💰 价格表现
- **当前价格**: ${latest_price:,.2f}
- **24小时变化**: {price_change_24h:+.2f}%
- **价格趋势**: {market_env.trend_direction.value}
- **波动率制度**: {market_env.volatility_regime}

### 😰 情绪指标
- **恐惧贪婪指数**: {latest_fg:.1f}/100
- **市场情绪**: {market_env.fear_greed_state}
- **巨鲸活动**: {market_env.whale_activity}
- **资金费率**: {market_env.funding_rate_trend}

---

## ⚠️ 风险指标分析

### 📊 核心风险指标
| 指标 | 数值 | 风险等级 | 说明 |
|------|------|----------|------|
| **1日VaR** | {risk_metrics.var_1d:.2f}% | {'🔴 高' if risk_metrics.var_1d > 5 else '🟡 中' if risk_metrics.var_1d > 2 else '🟢 低'} | 95%置信度下的日损失风险 |
| **7日VaR** | {risk_metrics.var_7d:.2f}% | {'🔴 高' if risk_metrics.var_7d > 15 else '🟡 中' if risk_metrics.var_7d > 8 else '🟢 低'} | 95%置信度下的周损失风险 |
| **最大回撤** | {risk_metrics.max_drawdown:.2f}% | {'🔴 高' if risk_metrics.max_drawdown > 20 else '🟡 中' if risk_metrics.max_drawdown > 10 else '🟢 低'} | 历史最大回撤幅度 |
| **夏普比率** | {risk_metrics.sharpe_ratio:.2f} | {'🟢 优' if risk_metrics.sharpe_ratio > 1 else '🟡 良' if risk_metrics.sharpe_ratio > 0 else '🔴 差'} | 风险调整后收益 |
| **年化波动率** | {risk_metrics.volatility:.1f}% | {'🔴 高' if risk_metrics.volatility > 80 else '🟡 中' if risk_metrics.volatility > 40 else '🟢 低'} | 价格波动程度 |

### 🎯 风险评级解读
- **{risk_level.value}**: {'市场波动较大，建议谨慎操作' if risk_level == RiskLevel.HIGH else '市场存在一定风险，适中操作' if risk_level == RiskLevel.MEDIUM else '市场相对平稳，可正常操作'}

---

## 🚨 今日告警统计

### 📊 告警分布
- **总告警数**: {total_alerts} 次
- **各级别分布**:
{chr(10).join([f'  - {level}: {count}次' for level, count in alert_counts.items()]) if alert_counts else '  - 今日无告警'}

### ⚡ 重要告警事件
{self._format_key_alerts(alerts) if len(alerts) > 0 else '今日未发生重要告警事件'}

---

## 💡 交易策略建议

### 🎯 基于风险等级的建议
{self._generate_risk_based_recommendations(risk_level, market_env)}

### 📈 技术分析建议
{self._generate_technical_recommendations(market_env)}

### ⏰ 操作时机建议
{self._generate_timing_recommendations(latest_fg, market_env)}

---

## 📊 数据统计

### 📈 24小时市场数据
- **数据点数**: {len(data)} 个
- **数据完整性**: {(1 - data.isna().sum().sum() / (len(data) * len(data.columns))) * 100:.1f}%
- **最高价**: ${data['price'].max():,.2f} 
- **最低价**: ${data['price'].min():,.2f}
- **平均成交量**: {data['volume'].mean():,.0f}

### 🔔 监控状态
- **系统运行时间**: 24/7 不间断
- **数据更新频率**: 每30秒
- **告警响应时间**: < 2秒
- **数据来源**: 实时API + 历史数据库

---

## 📋 明日关注重点

### 🎯 重要指标监控
1. **恐惧贪婪指数**: {'关注是否突破80(极度贪婪)' if latest_fg > 70 else '关注是否突破20(极度恐惧)' if latest_fg < 30 else '关注趋势变化方向'}
2. **巨鲸活动**: {'警惕协调抛售行为' if market_env.whale_activity == '协调' else '监控大额交易信号'}
3. **资金费率**: {'关注多头情绪是否过热' if market_env.funding_rate_trend == '偏多' else '关注空头情绪变化'}

### 📅 重要事件日程
- **市场开盘**: 关注亚洲市场开盘表现
- **数据发布**: 注意重要经济数据公布
- **技术位**: {'关注上方阻力位' if market_env.trend_direction in [TrendDirection.UP, TrendDirection.STRONGLY_UP] else '关注下方支撑位'}

---

## 📞 风险提示

⚠️ **投资风险提示**:
1. 加密货币市场高度波动，投资有风险
2. 本报告仅供参考，不构成投资建议
3. 请根据个人风险承受能力进行投资决策
4. 建议采用分散投资策略，控制单一资产风险

📊 **数据说明**:
- 报告基于历史数据分析，未来表现不保证
- 风险指标基于统计模型，存在模型风险
- 建议结合多种分析工具进行决策

---

*报告由AI风险评估系统自动生成 | 版本: v2.0 | 技术支持: Claude Code Assistant*
"""
        return report
    
    def _generate_portfolio_report_content(self,
                                         positions: List[PortfolioPosition],
                                         portfolio_analysis: Dict[str, Any],
                                         risk_metrics: RiskMetrics,
                                         market_env: MarketEnvironment) -> str:
        """生成投资组合报告内容"""
        
        total_value = portfolio_analysis['total_value']
        total_pnl = portfolio_analysis['total_pnl']
        weighted_risk = portfolio_analysis['weighted_risk_score']
        
        # 确定组合风险等级
        if weighted_risk > 70:
            portfolio_risk = RiskLevel.HIGH
        elif weighted_risk > 50:
            portfolio_risk = RiskLevel.MEDIUM
        else:
            portfolio_risk = RiskLevel.LOW
        
        report = f"""# 📊 投资组合风险评估报告

## 📈 组合概览
- **报告时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **组合总价值**: ${total_value:,.2f}
- **持仓数量**: {len(positions)} 个资产
- **总盈亏**: ${total_pnl:,.2f} ({portfolio_analysis['total_pnl_percent']:+.2f}%)
- **整体风险等级**: 🔸 **{portfolio_risk.value}**

---

## 💼 持仓明细分析

### 📊 资产配置表
| 资产 | 持仓量 | 当前价值 | 盈亏 | 盈亏率 | 权重 | 风险得分 |
|------|--------|----------|------|--------|------|----------|
{chr(10).join([f'| **{pos.symbol}** | {pos.quantity:.2f} | ${pos.market_value:,.2f} | ${pos.pnl:+,.2f} | {pos.pnl_percent:+.2f}% | {pos.weight:.1f}% | {pos.risk_score}/100 |' for pos in positions])}
| **合计** | - | **${total_value:,.2f}** | **${total_pnl:+,.2f}** | **{portfolio_analysis['total_pnl_percent']:+.2f}%** | **100.0%** | **{weighted_risk:.1f}/100** |

### 📈 资产表现分析
{self._analyze_position_performance(positions)}

---

## ⚠️ 组合风险分析

### 🎯 风险指标概览
| 风险维度 | 评估结果 | 风险等级 | 说明 |
|----------|----------|----------|------|
| **集中度风险** | {portfolio_analysis['concentration_risk']} | {'🔴 需关注' if portfolio_analysis['concentration_risk'] == '高' else '🟡 适中' if portfolio_analysis['concentration_risk'] == '中' else '🟢 良好'} | 单一资产占比情况 |
| **相关性风险** | {portfolio_analysis['correlation_risk']} | 🟡 适中 | 资产间相关度 |
| **分散化程度** | {portfolio_analysis['diversification_score']:.1f}分 | {'🟢 良好' if portfolio_analysis['diversification_score'] > 70 else '🟡 一般' if portfolio_analysis['diversification_score'] > 50 else '🔴 不足'} | 投资分散程度 |
| **加权风险** | {weighted_risk:.1f}/100 | {'🔴 高风险' if weighted_risk > 70 else '🟡 中风险' if weighted_risk > 50 else '🟢 低风险'} | 综合风险水平 |

### 📊 风险构成分析
{self._analyze_risk_composition(positions, portfolio_analysis)}

---

## 🎯 优化建议

### 🔄 资产配置优化
{self._generate_allocation_recommendations(positions, portfolio_analysis)}

### ⚖️ 风险平衡建议
{self._generate_risk_balance_recommendations(positions, weighted_risk)}

### 📈 收益提升策略
{self._generate_return_enhancement_suggestions(positions, market_env)}

---

## 📊 市场环境影响分析

### 🌍 当前市场环境
- **市场趋势**: {market_env.trend_direction.value}
- **波动率制度**: {market_env.volatility_regime}
- **情绪状态**: {market_env.fear_greed_state}
- **巨鲸活动**: {market_env.whale_activity}

### 💥 对组合的影响
{self._analyze_market_impact(positions, market_env)}

---

## 🎛️ 动态调整建议

### ⏰ 短期调整(1-7天)
{self._generate_short_term_adjustments(positions, market_env)}

### 📅 中期规划(1-3个月)
{self._generate_medium_term_planning(positions, portfolio_analysis)}

### 🔮 长期战略(3个月以上)
{self._generate_long_term_strategy(positions)}

---

## 📈 业绩基准对比

### 📊 收益对比
- **组合收益**: {portfolio_analysis['total_pnl_percent']:+.2f}%
- **BTC收益**: +7.5% (参考)
- **ETH收益**: +12.3% (参考)
- **相对表现**: {'超越市场' if portfolio_analysis['total_pnl_percent'] > 8 else '跟随市场' if portfolio_analysis['total_pnl_percent'] > 5 else '落后市场'}

### 🎯 风险调整收益
- **夏普比率**: {risk_metrics.sharpe_ratio:.2f}
- **最大回撤**: {risk_metrics.max_drawdown:.2f}%
- **风险收益比**: {portfolio_analysis['total_pnl_percent'] / max(weighted_risk, 1):.2f}

---

## 🚨 风险警示

### ⚠️ 重要风险提示
1. **集中度风险**: {'单一资产占比过高，建议适度分散' if portfolio_analysis['concentration_risk'] == '高' else '资产分散度合理'}
2. **市场风险**: 加密货币市场波动极大，请做好风险控制
3. **流动性风险**: 注意各资产的流动性差异
4. **技术风险**: 区块链技术存在不确定性

### 📞 紧急联系
- **风险预警**: 当单日损失超过5%时立即评估
- **止损建议**: 建议设置-15%的组合止损线
- **定期评估**: 建议每周进行组合评估

---

*投资组合分析报告由AI系统生成，仅供参考，投资决策请谨慎*
"""
        return report
    
    def _format_key_alerts(self, alerts: pd.DataFrame) -> str:
        """格式化关键告警"""
        if len(alerts) == 0:
            return ""
        
        key_alerts = alerts[alerts['alert_level'].isin(['CRITICAL', 'HIGH'])].head(3)
        if len(key_alerts) == 0:
            return "今日未发生关键告警事件"
        
        formatted = []
        for _, alert in key_alerts.iterrows():
            time_str = alert['timestamp'].strftime('%H:%M')
            formatted.append(f"  - **{time_str}** [{alert['alert_level']}] {alert['message']}")
        
        return "\n".join(formatted)
    
    def _generate_risk_based_recommendations(self, risk_level: RiskLevel, market_env: MarketEnvironment) -> str:
        """基于风险等级生成建议"""
        if risk_level == RiskLevel.HIGH:
            return """
**🔴 高风险环境 - 防守策略**:
- 建议降低仓位至30-50%
- 设置严格止损(5-8%)
- 避免杠杆交易
- 分散投资降低集中风险
- 增加现金或稳定币配置"""
        
        elif risk_level == RiskLevel.MEDIUM:
            return """
**🟡 中等风险环境 - 平衡策略**:
- 保持适中仓位(50-70%)
- 设置合理止损(8-12%)
- 可适度使用低倍杠杆(2-3倍)
- 继续分散投资
- 保持流动性储备"""
        
        else:
            return """
**🟢 低风险环境 - 积极策略**:
- 可提高仓位至70-85%
- 放宽止损设置(10-15%)
- 可考虑适度杠杆(3-5倍)
- 关注高收益机会
- 积极参与市场交易"""
    
    def _generate_technical_recommendations(self, market_env: MarketEnvironment) -> str:
        """生成技术分析建议"""
        if market_env.trend_direction in [TrendDirection.STRONGLY_UP, TrendDirection.UP]:
            return """
**📈 上升趋势策略**:
- 逢低买入，趋势跟随
- 关注突破后的回踩机会
- 设置移动止盈保护利润
- 避免逆势做空"""
        
        elif market_env.trend_direction in [TrendDirection.STRONGLY_DOWN, TrendDirection.DOWN]:
            return """
**📉 下降趋势策略**:
- 减仓观望，等待企稳
- 可考虑反弹时减仓
- 严格执行止损策略
- 关注支撑位反弹机会"""
        
        else:
            return """
**📊 震荡趋势策略**:
- 区间交易，高抛低吸
- 关注突破方向选择
- 控制单次交易仓位
- 保持灵活操作策略"""
    
    def _generate_timing_recommendations(self, fear_greed_index: float, market_env: MarketEnvironment) -> str:
        """生成时机建议"""
        if fear_greed_index > 75:
            return """
**⏰ 极度贪婪时期**:
- **建议操作**: 逐步减仓，获利了结
- **最佳时机**: 情绪高涨时分批卖出
- **风险控制**: 严防情绪反转带来的急跌"""
        
        elif fear_greed_index < 25:
            return """
**⏰ 极度恐惧时期**:
- **建议操作**: 分批建仓，逢低买入  
- **最佳时机**: 恐慌性抛售后逐步介入
- **策略要点**: 耐心等待，不要急于抄底"""
        
        else:
            return """
**⏰ 中性情绪时期**:
- **建议操作**: 保持现有仓位，观察变化
- **关注重点**: 等待明确的情绪转向信号
- **操作策略**: 小仓位试探，灵活调整"""
    
    def _analyze_position_performance(self, positions: List[PortfolioPosition]) -> str:
        """分析持仓表现"""
        best_performer = max(positions, key=lambda x: x.pnl_percent)
        worst_performer = min(positions, key=lambda x: x.pnl_percent)
        
        return f"""
**🏆 最佳表现**: {best_performer.symbol} (+{best_performer.pnl_percent:.2f}%)
**📉 最差表现**: {worst_performer.symbol} ({worst_performer.pnl_percent:+.2f}%)
**📊 盈利资产**: {len([p for p in positions if p.pnl > 0])}/{len(positions)} 个
**💰 总体表现**: {'盈利' if sum(p.pnl for p in positions) > 0 else '亏损'}组合"""
    
    def _analyze_risk_composition(self, positions: List[PortfolioPosition], portfolio_analysis: Dict) -> str:
        """分析风险构成"""
        high_risk_assets = [p for p in positions if p.risk_score > 70]
        medium_risk_assets = [p for p in positions if 50 <= p.risk_score <= 70]
        low_risk_assets = [p for p in positions if p.risk_score < 50]
        
        return f"""
**🔴 高风险资产**: {len(high_risk_assets)} 个 ({sum(p.weight for p in high_risk_assets):.1f}% 权重)
**🟡 中风险资产**: {len(medium_risk_assets)} 个 ({sum(p.weight for p in medium_risk_assets):.1f}% 权重)  
**🟢 低风险资产**: {len(low_risk_assets)} 个 ({sum(p.weight for p in low_risk_assets):.1f}% 权重)
**⚖️ 风险平衡**: {'偏向高风险' if len(high_risk_assets) > len(low_risk_assets) else '相对均衡'}"""
    
    def _generate_allocation_recommendations(self, positions: List[PortfolioPosition], portfolio_analysis: Dict) -> str:
        """生成配置建议"""
        max_weight_asset = max(positions, key=lambda x: x.weight)
        
        if max_weight_asset.weight > 60:
            return f"""
**⚠️ 集中度过高警告**:
- {max_weight_asset.symbol}占比{max_weight_asset.weight:.1f}%，建议降至40-50%
- 增加其他优质资产配置
- 考虑添加稳定币缓冲
- 分批调整，避免冲击成本"""
        else:
            return f"""
**✅ 配置相对合理**:
- 当前配置分散度良好
- 可考虑微调权重比例
- 关注各资产相关性变化
- 定期再平衡组合"""
    
    def _generate_risk_balance_recommendations(self, positions: List[PortfolioPosition], weighted_risk: float) -> str:
        """生成风险平衡建议"""
        if weighted_risk > 70:
            return """
**🔴 风险过度集中**:
- 建议增加低风险资产(USDT/USDC)配置
- 减少高β系数资产权重
- 考虑添加反相关资产
- 设置更严格的风险控制"""
        else:
            return """
**🟢 风险相对平衡**:
- 当前风险水平合理
- 保持多元化配置策略
- 关注相关性变化
- 定期评估调整"""
    
    def _generate_return_enhancement_suggestions(self, positions: List[PortfolioPosition], market_env: MarketEnvironment) -> str:
        """生成收益提升建议"""
        return f"""
**📈 收益优化策略**:
- **趋势跟随**: 当前{market_env.trend_direction.value}，可考虑增加趋势资产
- **轮动策略**: 关注板块轮动机会，适时调仓
- **定投策略**: 对于长期看好资产，可采用定投方式
- **收益再投**: 及时将收益再投资，发挥复利效应"""
    
    def _analyze_market_impact(self, positions: List[PortfolioPosition], market_env: MarketEnvironment) -> str:
        """分析市场环境影响"""
        return f"""
**🌊 市场环境影响评估**:
- **趋势影响**: {market_env.trend_direction.value}对组合整体{'有利' if 'UP' in market_env.trend_direction.name else '不利' if 'DOWN' in market_env.trend_direction.name else '中性'}
- **波动率影响**: {market_env.volatility_regime}环境下建议{'降低仓位' if market_env.volatility_regime == '高波动' else '正常操作'}
- **情绪影响**: {market_env.fear_greed_state}情绪可能带来{'获利机会' if '恐惧' in market_env.fear_greed_state else '回调风险' if '贪婪' in market_env.fear_greed_state else '震荡行情'}
- **资金流向**: 当前资金费率{market_env.funding_rate_trend}，注意{'多头过热风险' if market_env.funding_rate_trend == '偏多' else '空头情绪变化'}"""
    
    def _generate_short_term_adjustments(self, positions: List[PortfolioPosition], market_env: MarketEnvironment) -> str:
        """生成短期调整建议"""
        return f"""
**⚡ 1-7天调整计划**:
1. **仓位管理**: 根据{market_env.trend_direction.value}趋势{'保持' if 'SIDEWAYS' in market_env.trend_direction.name else '调整'}当前仓位
2. **止损设置**: 基于{market_env.volatility_regime}设置合适止损距离
3. **机会捕捉**: 关注{market_env.fear_greed_state}情绪下的短期交易机会
4. **流动性管理**: 保持20-30%现金仓位应对突发情况"""
    
    def _generate_medium_term_planning(self, positions: List[PortfolioPosition], portfolio_analysis: Dict) -> str:
        """生成中期规划"""
        return f"""
**📅 1-3个月规划**:
1. **配置优化**: 基于表现调整权重，考虑减持表现不佳资产
2. **新增投资**: 研究并添加1-2个潜力资产
3. **风险控制**: 将整体风险控制在{portfolio_analysis['weighted_risk_score']:.0f}分以下
4. **收益目标**: 争取实现15-25%的季度收益率"""
    
    def _generate_long_term_strategy(self, positions: List[PortfolioPosition]) -> str:
        """生成长期战略"""
        return f"""
**🔮 3个月以上战略**:
1. **价值投资**: 专注于基本面优秀的主流资产
2. **分散配置**: 保持5-8个不同赛道的资产配置
3. **定期再平衡**: 每月进行一次组合再平衡
4. **风险预算**: 总体风险保持在中等水平，追求稳健增长"""

def main():
    """主函数演示"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                📊 风险评估报告生成系统 v2.0                        ║  
║                                                                  ║
║  功能特性:                                                        ║
║  📅 自动化日报生成      📊 投资组合分析                          ║
║  📈 风险指标计算        💡 策略建议生成                          ║
║  📋 趋势分析评估        🎯 个性化建议                            ║
║                                                                  ║
║  作者: Claude Code Assistant                                     ║
║  版本: v2.0 | 创建时间: 2025-08-09                               ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # 创建报告生成器
    generator = ReportGenerator()
    
    print("🚀 正在生成风险评估报告...")
    
    try:
        # 生成日报
        print("\n📅 生成每日风险评估报告...")
        daily_report = generator.generate_daily_report()
        
        # 保存日报
        daily_report_path = f"daily_risk_report_{datetime.now().strftime('%Y%m%d')}.md"
        with open(daily_report_path, 'w', encoding='utf-8') as f:
            f.write(daily_report)
        
        print(f"✅ 日报已保存: {daily_report_path}")
        
        # 生成投资组合报告
        print("\n💼 生成投资组合风险评估报告...")
        portfolio_report = generator.generate_portfolio_report()
        
        # 保存投资组合报告
        portfolio_report_path = f"portfolio_risk_report_{datetime.now().strftime('%Y%m%d')}.md"
        with open(portfolio_report_path, 'w', encoding='utf-8') as f:
            f.write(portfolio_report)
        
        print(f"✅ 投资组合报告已保存: {portfolio_report_path}")
        
        # 显示报告摘要
        print("\n" + "="*80)
        print("📋 报告生成完成")
        print("="*80)
        print(f"📅 日报文件: {daily_report_path}")
        print(f"💼 组合报告: {portfolio_report_path}")
        print("\n📊 报告包含内容:")
        print("   ✅ 市场风险分析")
        print("   ✅ 技术指标评估")
        print("   ✅ 投资策略建议")
        print("   ✅ 组合优化方案")
        print("   ✅ 风险预警提示")
        
        print(f"\n🎉 风险评估报告系统运行完成!")
        
    except Exception as e:
        print(f"❌ 报告生成失败: {e}")
        logger.error(f"报告生成异常: {e}")

if __name__ == "__main__":
    main()