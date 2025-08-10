#!/usr/bin/env python3
"""
Advanced Alert System
高级预警系统 - 多级预警、动态阈值、智能告警
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
from typing import Dict, List, Optional, Any, Tuple
warnings.filterwarnings('ignore')

# Rich美化输出
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.text import Text
    from rich.rule import Rule
    from rich.columns import Columns
    from rich.tree import Tree
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.factor_engine.crypto_specialized.crypto_factor_utils import CryptoFactorUtils, CryptoDataProcessor

class AdvancedAlertSystem:
    """高级预警系统"""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
        self.crypto_utils = CryptoFactorUtils()
        self.data_processor = CryptoDataProcessor()
        
        # 预警配置
        self.alert_config = {
            'severity_levels': {
                'CRITICAL': {'emoji': '🚨', 'priority': 5, 'color': 'red'},
                'HIGH': {'emoji': '🔴', 'priority': 4, 'color': 'bright_red'},
                'MEDIUM': {'emoji': '🟠', 'priority': 3, 'color': 'yellow'},
                'LOW': {'emoji': '🟡', 'priority': 2, 'color': 'blue'},
                'INFO': {'emoji': '🔵', 'priority': 1, 'color': 'cyan'}
            },
            'notification_channels': {
                'console': True,
                'file': True,
                'email': False,  # 预留
                'webhook': False  # 预留
            }
        }
        
        # 动态阈值管理
        self.dynamic_thresholds = {
            'funding_rate': {
                'current': {'extreme': 1.5, 'high': 0.8, 'medium': 0.4},
                'history': [],
                'adaptation_rate': 0.1,
                'volatility_multiplier': 1.2
            },
            'whale_alert': {
                'current': {'high_frequency': 15, 'medium_frequency': 8, 'high_impact': 2.0},
                'history': [],
                'adaptation_rate': 0.05,
                'volatility_multiplier': 1.5
            },
            'fear_greed': {
                'current': {'extreme_greed': 80, 'greed': 65, 'fear': 35, 'extreme_fear': 20},
                'history': [],
                'adaptation_rate': 0.2,
                'volatility_multiplier': 1.1
            },
            'volatility': {
                'current': {'extreme': 200, 'high': 150, 'medium': 100},
                'history': [],
                'adaptation_rate': 0.15,
                'volatility_multiplier': 1.3
            }
        }
        
        # 预警历史和状态
        self.alert_history = []
        self.active_alerts = {}
        self.alert_statistics = {
            'total_alerts': 0,
            'alerts_by_severity': {level: 0 for level in self.alert_config['severity_levels']},
            'alerts_by_type': {},
            'false_positive_rate': 0.0,
            'response_times': []
        }
        
        # 智能预警规则
        self.smart_rules = {
            'correlation_alerts': True,      # 多指标相关性警报
            'trend_reversal_alerts': True,   # 趋势反转警报
            'volatility_breakout': True,     # 波动率突破警报
            'liquidity_crisis': True,       # 流动性危机警报
            'cascade_risk': True            # 级联风险警报
        }
    
    def print_alert_header(self, title):
        """打印预警系统标题"""
        if self.console:
            panel = Panel(
                f"[bold red]🚨 {title}[/bold red]",
                style="bright_red"
            )
            self.console.print(panel)
        else:
            print(f"\n🚨 {title}")
            print("=" * 60)
    
    def create_alert(self, 
                    alert_type: str, 
                    severity: str, 
                    message: str, 
                    details: Dict = None,
                    factor_values: Dict = None) -> Dict:
        """创建预警"""
        alert = {
            'id': f"{alert_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now(),
            'type': alert_type,
            'severity': severity,
            'message': message,
            'details': details or {},
            'factor_values': factor_values or {},
            'status': 'ACTIVE',
            'resolved_at': None,
            'false_positive': False
        }
        
        # 添加到历史和活跃警报
        self.alert_history.append(alert)
        self.active_alerts[alert['id']] = alert
        
        # 更新统计
        self.alert_statistics['total_alerts'] += 1
        self.alert_statistics['alerts_by_severity'][severity] += 1
        
        if alert_type not in self.alert_statistics['alerts_by_type']:
            self.alert_statistics['alerts_by_type'][alert_type] = 0
        self.alert_statistics['alerts_by_type'][alert_type] += 1
        
        return alert
    
    def update_dynamic_thresholds(self, factor_type: str, current_values: List[float]):
        """更新动态阈值"""
        if factor_type not in self.dynamic_thresholds:
            return
        
        threshold_config = self.dynamic_thresholds[factor_type]
        
        # 添加到历史
        threshold_config['history'].extend(current_values)
        
        # 保持历史长度在合理范围内
        if len(threshold_config['history']) > 1000:
            threshold_config['history'] = threshold_config['history'][-500:]
        
        if len(threshold_config['history']) < 20:
            return  # 数据不足，不更新阈值
        
        # 计算统计指标
        history = np.array(threshold_config['history'])
        mean_val = np.mean(history)
        std_val = np.std(history)
        
        # 根据因子类型调整阈值
        adaptation_rate = threshold_config['adaptation_rate']
        vol_multiplier = threshold_config['volatility_multiplier']
        
        if factor_type == 'funding_rate':
            # 资金费率阈值更新
            new_extreme = mean_val + (std_val * vol_multiplier * 2)
            new_high = mean_val + (std_val * vol_multiplier * 1.5)
            new_medium = mean_val + (std_val * vol_multiplier)
            
            # 平滑更新
            threshold_config['current']['extreme'] = (
                threshold_config['current']['extreme'] * (1 - adaptation_rate) + 
                new_extreme * adaptation_rate
            )
            threshold_config['current']['high'] = (
                threshold_config['current']['high'] * (1 - adaptation_rate) + 
                new_high * adaptation_rate
            )
            threshold_config['current']['medium'] = (
                threshold_config['current']['medium'] * (1 - adaptation_rate) + 
                new_medium * adaptation_rate
            )
        
        elif factor_type == 'whale_alert':
            # 巨鲸检测阈值更新
            mean_frequency = np.mean([abs(x) for x in history if abs(x) > 0.5])
            if not np.isnan(mean_frequency):
                new_high_freq = mean_frequency * vol_multiplier * 1.5
                new_med_freq = mean_frequency * vol_multiplier
                
                threshold_config['current']['high_frequency'] = (
                    threshold_config['current']['high_frequency'] * (1 - adaptation_rate) + 
                    new_high_freq * adaptation_rate
                )
                threshold_config['current']['medium_frequency'] = (
                    threshold_config['current']['medium_frequency'] * (1 - adaptation_rate) + 
                    new_med_freq * adaptation_rate
                )
        
        elif factor_type == 'fear_greed':
            # 恐惧贪婪指数阈值更新
            percentile_95 = np.percentile(history, 95)
            percentile_75 = np.percentile(history, 75)
            percentile_25 = np.percentile(history, 25)
            percentile_5 = np.percentile(history, 5)
            
            threshold_config['current']['extreme_greed'] = (
                threshold_config['current']['extreme_greed'] * (1 - adaptation_rate) + 
                percentile_95 * adaptation_rate
            )
            threshold_config['current']['greed'] = (
                threshold_config['current']['greed'] * (1 - adaptation_rate) + 
                percentile_75 * adaptation_rate
            )
            threshold_config['current']['fear'] = (
                threshold_config['current']['fear'] * (1 - adaptation_rate) + 
                percentile_25 * adaptation_rate
            )
            threshold_config['current']['extreme_fear'] = (
                threshold_config['current']['extreme_fear'] * (1 - adaptation_rate) + 
                percentile_5 * adaptation_rate
            )
        
        elif factor_type == 'volatility':
            # 波动率阈值更新
            percentile_90 = np.percentile(history, 90)
            percentile_75 = np.percentile(history, 75)
            percentile_60 = np.percentile(history, 60)
            
            threshold_config['current']['extreme'] = (
                threshold_config['current']['extreme'] * (1 - adaptation_rate) + 
                percentile_90 * adaptation_rate
            )
            threshold_config['current']['high'] = (
                threshold_config['current']['high'] * (1 - adaptation_rate) + 
                percentile_75 * adaptation_rate
            )
            threshold_config['current']['medium'] = (
                threshold_config['current']['medium'] * (1 - adaptation_rate) + 
                percentile_60 * adaptation_rate
            )
    
    def analyze_factor_for_alerts(self, 
                                 factor_type: str, 
                                 factor_values: pd.Series, 
                                 metadata: Dict = None) -> List[Dict]:
        """分析因子并生成预警"""
        alerts = []
        
        if factor_values.empty:
            return alerts
        
        current_value = factor_values.iloc[-1]
        factor_name = metadata.get('name', factor_type) if metadata else factor_type
        
        # 更新动态阈值
        recent_values = factor_values.dropna().tail(20).tolist()
        self.update_dynamic_thresholds(factor_type, recent_values)
        
        thresholds = self.dynamic_thresholds.get(factor_type, {}).get('current', {})
        
        if factor_type == 'funding_rate':
            alerts.extend(self._analyze_funding_rate_alerts(
                current_value, factor_values, thresholds, factor_name
            ))
        elif factor_type == 'whale_alert':
            alerts.extend(self._analyze_whale_alerts(
                factor_values, thresholds, factor_name
            ))
        elif factor_type == 'fear_greed':
            alerts.extend(self._analyze_fear_greed_alerts(
                current_value, factor_values, thresholds, factor_name
            ))
        elif factor_type == 'volatility':
            alerts.extend(self._analyze_volatility_alerts(
                current_value, factor_values, thresholds, factor_name
            ))
        
        # 智能预警规则检查
        if self.smart_rules.get('trend_reversal_alerts', False):
            reversal_alerts = self._check_trend_reversal(factor_type, factor_values)
            alerts.extend(reversal_alerts)
        
        return alerts
    
    def _analyze_funding_rate_alerts(self, 
                                   current_value: float, 
                                   series: pd.Series, 
                                   thresholds: Dict,
                                   factor_name: str) -> List[Dict]:
        """分析资金费率预警"""
        alerts = []
        
        extreme_threshold = thresholds.get('extreme', 1.5)
        high_threshold = thresholds.get('high', 0.8)
        
        if abs(current_value) > extreme_threshold:
            severity = 'CRITICAL'
            direction = '极端偏多' if current_value > 0 else '极端偏空'
            message = f"{factor_name}触发极端预警: {direction} ({current_value:.3f})"
            
            alert = self.create_alert(
                'funding_rate_extreme',
                severity,
                message,
                {
                    'current_value': current_value,
                    'threshold': extreme_threshold,
                    'direction': 'long_biased' if current_value > 0 else 'short_biased',
                    'suggestion': '考虑反向操作' if abs(current_value) > 2.0 else '密切关注'
                },
                {'funding_rate_momentum': current_value}
            )
            alerts.append(alert)
        
        elif abs(current_value) > high_threshold:
            severity = 'HIGH'
            direction = '偏多' if current_value > 0 else '偏空'
            message = f"{factor_name}高风险预警: {direction} ({current_value:.3f})"
            
            alert = self.create_alert(
                'funding_rate_high',
                severity,
                message,
                {
                    'current_value': current_value,
                    'threshold': high_threshold,
                    'suggestion': '谨慎操作，观察趋势变化'
                },
                {'funding_rate_momentum': current_value}
            )
            alerts.append(alert)
        
        # 检查资金费率快速变化
        if len(series) >= 5:
            recent_change = abs(series.iloc[-1] - series.iloc[-5])
            if recent_change > 0.5:
                alert = self.create_alert(
                    'funding_rate_rapid_change',
                    'MEDIUM',
                    f"{factor_name}快速变化预警: 变化幅度 {recent_change:.3f}",
                    {
                        'change_magnitude': recent_change,
                        'time_window': '5个数据点',
                        'suggestion': '关注市场情绪突变'
                    }
                )
                alerts.append(alert)
        
        return alerts
    
    def _analyze_whale_alerts(self, 
                            series: pd.Series, 
                            thresholds: Dict,
                            factor_name: str) -> List[Dict]:
        """分析巨鲸交易预警"""
        alerts = []
        
        if series.empty:
            return alerts
        
        # 计算巨鲸活动频率
        significant_whales = series[abs(series) > 1.0]
        high_impact_whales = series[abs(series) > 2.0]
        
        # 计算每周频率（假设数据是小时级别）
        hours_per_week = 24 * 7
        data_hours = len(series)
        
        weekly_frequency = len(significant_whales) * (hours_per_week / data_hours) if data_hours > 0 else 0
        high_impact_frequency = len(high_impact_whales) * (hours_per_week / data_hours) if data_hours > 0 else 0
        
        high_freq_threshold = thresholds.get('high_frequency', 15)
        med_freq_threshold = thresholds.get('medium_frequency', 8)
        
        if weekly_frequency > high_freq_threshold:
            alert = self.create_alert(
                'whale_high_frequency',
                'HIGH',
                f"{factor_name}高频预警: {weekly_frequency:.1f} 次/周",
                {
                    'weekly_frequency': weekly_frequency,
                    'high_impact_count': len(high_impact_whales),
                    'threshold': high_freq_threshold,
                    'suggestion': '减少大额交易，关注流动性'
                },
                {'whale_weekly_frequency': weekly_frequency}
            )
            alerts.append(alert)
        
        elif weekly_frequency > med_freq_threshold:
            alert = self.create_alert(
                'whale_medium_frequency',
                'MEDIUM',
                f"{factor_name}中频预警: {weekly_frequency:.1f} 次/周",
                {
                    'weekly_frequency': weekly_frequency,
                    'suggestion': '适度关注，监控大户动向'
                }
            )
            alerts.append(alert)
        
        # 检查巨鲸交易集中性
        if len(series) >= 24:  # 至少24小时数据
            recent_24h = series.tail(24)
            recent_whales = len(recent_24h[abs(recent_24h) > 1.0])
            
            if recent_whales > 5:  # 24小时内超过5次巨鲸交易
                alert = self.create_alert(
                    'whale_cluster',
                    'MEDIUM',
                    f"{factor_name}集中交易预警: 24h内 {recent_whales} 次",
                    {
                        'recent_count': recent_whales,
                        'time_window': '24小时',
                        'suggestion': '市场可能面临大幅波动'
                    }
                )
                alerts.append(alert)
        
        return alerts
    
    def _analyze_fear_greed_alerts(self, 
                                  current_value: float, 
                                  series: pd.Series, 
                                  thresholds: Dict,
                                  factor_name: str) -> List[Dict]:
        """分析恐惧贪婪指数预警"""
        alerts = []
        
        extreme_greed = thresholds.get('extreme_greed', 80)
        greed = thresholds.get('greed', 65)
        fear = thresholds.get('fear', 35)
        extreme_fear = thresholds.get('extreme_fear', 20)
        
        if current_value > extreme_greed:
            alert = self.create_alert(
                'fear_greed_extreme_greed',
                'HIGH',
                f"{factor_name}极度贪婪预警: {current_value:.1f}/100",
                {
                    'current_value': current_value,
                    'threshold': extreme_greed,
                    'emotion': '极度贪婪',
                    'suggestion': '考虑获利了结，准备应对回调'
                },
                {'fear_greed_index': current_value}
            )
            alerts.append(alert)
        
        elif current_value < extreme_fear:
            alert = self.create_alert(
                'fear_greed_extreme_fear',
                'MEDIUM',
                f"{factor_name}极度恐惧预警: {current_value:.1f}/100",
                {
                    'current_value': current_value,
                    'threshold': extreme_fear,
                    'emotion': '极度恐惧',
                    'suggestion': '寻找抄底机会，但要谨慎'
                },
                {'fear_greed_index': current_value}
            )
            alerts.append(alert)
        
        # 检查情绪急剧变化
        if len(series) >= 10:
            recent_change = abs(series.iloc[-1] - series.iloc[-10])
            if recent_change > 20:  # 短期内变化超过20点
                alert = self.create_alert(
                    'fear_greed_rapid_change',
                    'MEDIUM',
                    f"{factor_name}情绪急变预警: 变化 {recent_change:.1f} 点",
                    {
                        'change_magnitude': recent_change,
                        'time_window': '10个数据点',
                        'suggestion': '市场情绪剧烈波动，注意趋势反转'
                    }
                )
                alerts.append(alert)
        
        return alerts
    
    def _analyze_volatility_alerts(self, 
                                 current_value: float, 
                                 series: pd.Series, 
                                 thresholds: Dict,
                                 factor_name: str) -> List[Dict]:
        """分析波动率预警"""
        alerts = []
        
        extreme_threshold = thresholds.get('extreme', 200)
        high_threshold = thresholds.get('high', 150)
        
        if current_value > extreme_threshold:
            alert = self.create_alert(
                'volatility_extreme',
                'CRITICAL',
                f"{factor_name}极端波动预警: {current_value:.1f}%",
                {
                    'current_value': current_value,
                    'threshold': extreme_threshold,
                    'suggestion': '避免大额交易，等待波动率回落'
                },
                {'volatility_annualized': current_value}
            )
            alerts.append(alert)
        
        elif current_value > high_threshold:
            alert = self.create_alert(
                'volatility_high',
                'HIGH',
                f"{factor_name}高波动预警: {current_value:.1f}%",
                {
                    'current_value': current_value,
                    'threshold': high_threshold,
                    'suggestion': '减少仓位，控制风险'
                }
            )
            alerts.append(alert)
        
        return alerts
    
    def _check_trend_reversal(self, factor_type: str, series: pd.Series) -> List[Dict]:
        """检查趋势反转预警"""
        alerts = []
        
        if len(series) < 20:
            return alerts
        
        # 计算短期和长期趋势
        short_trend = series.tail(5).mean()
        long_trend = series.tail(20).mean()
        recent_value = series.iloc[-1]
        
        # 检查趋势反转信号
        if factor_type in ['funding_rate', 'fear_greed']:
            # 对于有方向性的指标
            if abs(short_trend - long_trend) > abs(long_trend) * 0.3:
                trend_direction = '上升' if short_trend > long_trend else '下降'
                
                alert = self.create_alert(
                    f'{factor_type}_trend_reversal',
                    'LOW',
                    f"{factor_type}趋势反转信号: {trend_direction}趋势",
                    {
                        'short_trend': short_trend,
                        'long_trend': long_trend,
                        'trend_strength': abs(short_trend - long_trend) / abs(long_trend),
                        'suggestion': '关注趋势变化，准备调整策略'
                    }
                )
                alerts.append(alert)
        
        return alerts
    
    def check_correlation_alerts(self, factor_data: Dict[str, pd.Series]) -> List[Dict]:
        """检查多因子相关性预警"""
        alerts = []
        
        if not self.smart_rules.get('correlation_alerts', False):
            return alerts
        
        if len(factor_data) < 2:
            return alerts
        
        # 计算因子间相关性
        correlations = {}
        factor_names = list(factor_data.keys())
        
        for i, factor1 in enumerate(factor_names):
            for factor2 in factor_names[i+1:]:
                if not factor_data[factor1].empty and not factor_data[factor2].empty:
                    # 找到共同的时间索引
                    common_index = factor_data[factor1].index.intersection(factor_data[factor2].index)
                    if len(common_index) >= 10:
                        series1 = factor_data[factor1].loc[common_index]
                        series2 = factor_data[factor2].loc[common_index]
                        
                        correlation = series1.corr(series2)
                        if not np.isnan(correlation):
                            correlations[f"{factor1}_{factor2}"] = correlation
        
        # 检查异常相关性
        for pair, corr in correlations.items():
            if abs(corr) > 0.8:  # 强相关
                factor1, factor2 = pair.split('_', 1)
                
                alert = self.create_alert(
                    'factor_correlation',
                    'INFO',
                    f"因子强相关预警: {factor1} 与 {factor2} 相关性 {corr:.3f}",
                    {
                        'correlation': corr,
                        'factor_pair': [factor1, factor2],
                        'suggestion': '注意因子冗余，可能影响模型效果'
                    }
                )
                alerts.append(alert)
        
        return alerts
    
    def check_cascade_risk(self, factor_data: Dict[str, pd.Series]) -> List[Dict]:
        """检查级联风险预警"""
        alerts = []
        
        if not self.smart_rules.get('cascade_risk', False):
            return alerts
        
        # 检查多个风险指标同时恶化的情况
        risk_factors = ['funding_rate', 'volatility', 'whale_alert']
        high_risk_count = 0
        risk_details = {}
        
        for factor_type, series in factor_data.items():
            if factor_type in risk_factors and not series.empty:
                current_value = series.iloc[-1]
                thresholds = self.dynamic_thresholds.get(factor_type, {}).get('current', {})
                
                is_high_risk = False
                if factor_type == 'funding_rate':
                    is_high_risk = abs(current_value) > thresholds.get('high', 0.8)
                elif factor_type == 'volatility':
                    is_high_risk = current_value > thresholds.get('high', 150)
                elif factor_type == 'whale_alert':
                    # 计算巨鲸活动频率
                    significant_whales = series[abs(series) > 1.0]
                    weekly_freq = len(significant_whales) * 7 / (len(series) / 24) if len(series) > 0 else 0
                    is_high_risk = weekly_freq > thresholds.get('medium_frequency', 8)
                
                if is_high_risk:
                    high_risk_count += 1
                    risk_details[factor_type] = current_value
        
        # 如果多个风险因子同时报警
        if high_risk_count >= 2:
            alert = self.create_alert(
                'cascade_risk',
                'CRITICAL',
                f"级联风险预警: {high_risk_count} 个风险指标同时异常",
                {
                    'risk_factor_count': high_risk_count,
                    'risk_details': risk_details,
                    'suggestion': '系统性风险上升，建议大幅降低仓位'
                }
            )
            alerts.append(alert)
        
        return alerts
    
    def process_comprehensive_alerts(self, 
                                   market_data: Dict, 
                                   factor_results: Dict) -> List[Dict]:
        """处理综合预警"""
        all_alerts = []
        
        # 单因子预警分析
        for factor_type, factor_series in factor_results.items():
            if isinstance(factor_series, pd.Series) and not factor_series.empty:
                factor_alerts = self.analyze_factor_for_alerts(factor_type, factor_series)
                all_alerts.extend(factor_alerts)
        
        # 多因子关联预警
        correlation_alerts = self.check_correlation_alerts(factor_results)
        all_alerts.extend(correlation_alerts)
        
        # 级联风险预警
        cascade_alerts = self.check_cascade_risk(factor_results)
        all_alerts.extend(cascade_alerts)
        
        # 按严重程度排序
        all_alerts.sort(key=lambda x: self.alert_config['severity_levels'][x['severity']]['priority'], 
                       reverse=True)
        
        return all_alerts
    
    def display_alerts(self, alerts: List[Dict]):
        """显示预警信息"""
        if not alerts:
            if self.console:
                no_alerts_panel = Panel(
                    "[green]✅ 当前无预警信息，系统运行正常[/green]",
                    title="预警状态",
                    border_style="green"
                )
                self.console.print(no_alerts_panel)
            else:
                print("✅ 当前无预警信息，系统运行正常")
            return
        
        if self.console:
            # Rich版本显示
            alert_table = Table(title="🚨 实时预警信息")
            alert_table.add_column("时间", style="cyan", width=8)
            alert_table.add_column("级别", style="white", width=8)
            alert_table.add_column("类型", style="yellow", width=15)
            alert_table.add_column("预警信息", style="white")
            alert_table.add_column("建议", style="green")
            
            for alert in alerts[-10:]:  # 显示最新10个预警
                severity_config = self.alert_config['severity_levels'][alert['severity']]
                emoji = severity_config['emoji']
                
                alert_table.add_row(
                    alert['timestamp'].strftime("%H:%M:%S"),
                    f"{emoji} {alert['severity']}",
                    alert['type'],
                    alert['message'],
                    alert['details'].get('suggestion', '持续观察')
                )
            
            self.console.print(alert_table)
            
            # 显示预警统计
            stats_content = f"""[bold]📊 预警统计信息[/bold]

[cyan]总预警数:[/cyan] {self.alert_statistics['total_alerts']}
[cyan]活跃预警:[/cyan] {len(self.active_alerts)}
[cyan]本次检查:[/cyan] {len(alerts)} 个新预警

[yellow]按严重程度分布:[/yellow]
{chr(10).join([f'• {level}: {count}次' for level, count in self.alert_statistics['alerts_by_severity'].items() if count > 0])}
            """
            
            stats_panel = Panel(stats_content, title="统计信息", border_style="blue")
            self.console.print(stats_panel)
        
        else:
            # 简单文本版本显示
            print("\n🚨 实时预警信息:")
            print("=" * 60)
            
            for i, alert in enumerate(alerts[-5:], 1):  # 显示最新5个预警
                severity_config = self.alert_config['severity_levels'][alert['severity']]
                emoji = severity_config['emoji']
                
                print(f"{i}. {emoji} [{alert['severity']}] {alert['message']}")
                print(f"   时间: {alert['timestamp'].strftime('%H:%M:%S')}")
                print(f"   建议: {alert['details'].get('suggestion', '持续观察')}")
                print()
    
    def generate_alert_report(self) -> str:
        """生成预警报告"""
        report_time = datetime.now()
        
        # 统计最近24小时的预警
        recent_alerts = [
            alert for alert in self.alert_history 
            if alert['timestamp'] > report_time - timedelta(hours=24)
        ]
        
        critical_alerts = [a for a in recent_alerts if a['severity'] == 'CRITICAL']
        high_alerts = [a for a in recent_alerts if a['severity'] == 'HIGH']
        
        report = f"""# 加密货币风险预警报告

**生成时间**: {report_time.strftime('%Y-%m-%d %H:%M:%S')}
**报告周期**: 最近24小时

## 预警概览

- **总预警数**: {len(recent_alerts)}
- **严重预警**: {len(critical_alerts)} 个
- **高级预警**: {len(high_alerts)} 个
- **活跃预警**: {len(self.active_alerts)} 个

## 关键预警详情

### 🚨 严重预警 (CRITICAL)
"""
        
        for alert in critical_alerts:
            report += f"""
**{alert['type']}** - {alert['timestamp'].strftime('%H:%M:%S')}
- 信息: {alert['message']}
- 建议: {alert['details'].get('suggestion', '立即处理')}
"""
        
        report += f"""
### 🔴 高级预警 (HIGH)
"""
        
        for alert in high_alerts[-5:]:  # 最新5个高级预警
            report += f"""
**{alert['type']}** - {alert['timestamp'].strftime('%H:%M:%S')}  
- 信息: {alert['message']}
- 建议: {alert['details'].get('suggestion', '密切关注')}
"""
        
        report += f"""
## 系统统计

- **总预警数**: {self.alert_statistics['total_alerts']}
- **预警类型分布**: {dict(self.alert_statistics['alerts_by_type'])}
- **误报率**: {self.alert_statistics['false_positive_rate']:.2%}

## 动态阈值状态

### 资金费率
- 极端: {self.dynamic_thresholds['funding_rate']['current']['extreme']:.3f}
- 高风险: {self.dynamic_thresholds['funding_rate']['current']['high']:.3f}

### 恐惧贪婪指数  
- 极度贪婪: {self.dynamic_thresholds['fear_greed']['current']['extreme_greed']:.1f}
- 极度恐惧: {self.dynamic_thresholds['fear_greed']['current']['extreme_fear']:.1f}

---
*报告由高级预警系统自动生成*
"""
        
        return report
    
    def save_alert_report(self, filename: str = None) -> str:
        """保存预警报告"""
        if filename is None:
            filename = f"alert_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        report_content = self.generate_alert_report()
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
            return filename
        except Exception as e:
            print(f"保存报告失败: {e}")
            return None


def main():
    """主函数 - 演示高级预警系统"""
    print("🚀 启动高级预警系统演示...")
    
    # 初始化预警系统
    alert_system = AdvancedAlertSystem()
    alert_system.print_alert_header("高级预警系统演示")
    
    # 生成测试数据
    from risk_indicators_checker import RiskIndicatorsChecker
    
    checker = RiskIndicatorsChecker()
    market_data = checker.generate_test_data(['BTC/USDT'])
    
    # 计算因子
    symbol = 'BTC/USDT'
    price = market_data[symbol]['price']
    volume = market_data[symbol]['volume'] 
    amount = market_data[symbol]['amount']
    funding_rates = market_data.get('funding_rates', pd.Series())
    
    factor_results = {}
    
    # 计算各类因子
    try:
        crypto_utils = CryptoFactorUtils()
        
        # 恐惧贪婪指数
        fg_index = crypto_utils.FEAR_GREED_INDEX(price, volume)
        factor_results['fear_greed'] = fg_index
        
        # 巨鲸交易检测
        whale_alerts = crypto_utils.WHALE_ALERT(volume, amount, 2.5)
        factor_results['whale_alert'] = whale_alerts
        
        # 资金费率动量
        if not funding_rates.empty:
            funding_momentum = crypto_utils.FUNDING_RATE_MOMENTUM(funding_rates, 24)
            factor_results['funding_rate'] = funding_momentum
        
        # 计算波动率
        returns = price.pct_change().dropna()
        volatility = returns.rolling(window=24).std() * np.sqrt(365) * 100
        factor_results['volatility'] = volatility
        
        print("✅ 因子计算完成")
        
        # 运行综合预警分析
        print("\n🔄 运行综合预警分析...")
        alerts = alert_system.process_comprehensive_alerts(market_data, factor_results)
        
        # 显示预警结果
        alert_system.display_alerts(alerts)
        
        # 生成并保存报告
        print("\n📄 生成预警报告...")
        report_filename = alert_system.save_alert_report()
        if report_filename:
            print(f"✅ 预警报告已保存: {report_filename}")
        
        # 显示动态阈值状态
        if RICH_AVAILABLE and alert_system.console:
            threshold_content = f"""[bold]🎯 动态阈值当前状态[/bold]

[cyan]资金费率阈值:[/cyan]
• 极端: {alert_system.dynamic_thresholds['funding_rate']['current']['extreme']:.3f}
• 高风险: {alert_system.dynamic_thresholds['funding_rate']['current']['high']:.3f}
• 中等: {alert_system.dynamic_thresholds['funding_rate']['current']['medium']:.3f}

[cyan]恐惧贪婪阈值:[/cyan]
• 极度贪婪: {alert_system.dynamic_thresholds['fear_greed']['current']['extreme_greed']:.1f}
• 贪婪: {alert_system.dynamic_thresholds['fear_greed']['current']['greed']:.1f}
• 恐惧: {alert_system.dynamic_thresholds['fear_greed']['current']['fear']:.1f}
• 极度恐惧: {alert_system.dynamic_thresholds['fear_greed']['current']['extreme_fear']:.1f}

[dim]阈值基于历史数据动态调整，适应市场变化[/dim]
            """
            
            threshold_panel = Panel(threshold_content, title="📊 动态阈值", border_style="cyan")
            alert_system.console.print(threshold_panel)
        
        print("\n🎉 高级预警系统演示完成！")
        print("\n💡 系统特性:")
        print("• 多级预警机制 (CRITICAL/HIGH/MEDIUM/LOW/INFO)")
        print("• 动态阈值自适应调整")
        print("• 智能关联性分析")
        print("• 级联风险检测")
        print("• 趋势反转预警")
        print("• 详细预警报告生成")
        
    except Exception as e:
        print(f"❌ 演示过程出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()