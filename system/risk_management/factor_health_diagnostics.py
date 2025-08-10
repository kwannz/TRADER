#!/usr/bin/env python3
"""
Factor Health Diagnostics
因子健康状态诊断系统 - 深度分析因子计算质量和有效性
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
    from rich.tree import Tree
except ImportError:
    Console = object

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.factor_engine.crypto_specialized.crypto_factor_utils import CryptoFactorUtils, CryptoDataProcessor

class FactorHealthDiagnostics:
    """因子健康状态诊断系统"""
    
    def __init__(self):
        self.console = Console() if Console != object else None
        self.crypto_utils = CryptoFactorUtils()
        self.data_processor = CryptoDataProcessor()
        
        # 健康度评估标准
        self.health_thresholds = {
            'data_quality': {
                'excellent': 95,    # 95%以上数据完整性
                'good': 85,         # 85-95%
                'acceptable': 70,   # 70-85%
                'poor': 50         # 50-70%，低于50%为严重问题
            },
            'stability': {
                'very_stable': 0.1,     # 变异系数<0.1
                'stable': 0.3,          # 0.1-0.3
                'moderate': 0.7,        # 0.3-0.7
                'unstable': 1.0         # >1.0
            },
            'predictability': {
                'high': 0.3,           # 自相关系数>0.3
                'medium': 0.1,         # 0.1-0.3
                'low': -0.1,          # -0.1-0.1
                'very_low': -0.3       # <-0.3
            }
        }
    
    def print_diagnostic_header(self, title):
        """打印诊断标题"""
        if self.console:
            panel = Panel(
                f"[bold green]🔬 {title}[/bold green]",
                style="bright_green"
            )
            self.console.print(panel)
        else:
            print(f"\n🔬 {title}")
            print("=" * 60)
    
    def print_diagnostic_section(self, title):
        """打印诊断章节"""
        if self.console:
            text = Text(f"📋 {title}", style="bold cyan")
            self.console.print(f"\n{text}")
            self.console.print("─" * 50)
        else:
            print(f"\n📋 {title}")
            print("─" * 40)
    
    def assess_data_quality(self, data_series, factor_name):
        """评估数据质量"""
        quality_metrics = {}
        
        # 1. 完整性检查
        total_points = len(data_series)
        valid_points = data_series.count()
        missing_points = total_points - valid_points
        completeness = (valid_points / total_points) * 100
        
        quality_metrics['completeness'] = {
            'total_points': total_points,
            'valid_points': valid_points,
            'missing_points': missing_points,
            'completeness_rate': completeness
        }
        
        # 2. 异常值检测
        if valid_points > 0:
            q1 = data_series.quantile(0.25)
            q3 = data_series.quantile(0.75)
            iqr = q3 - q1
            
            # IQR方法检测异常值
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = ((data_series < lower_bound) | (data_series > upper_bound)).sum()
            
            # Z-score方法检测极端异常值
            z_scores = np.abs((data_series - data_series.mean()) / data_series.std())
            extreme_outliers = (z_scores > 3).sum()
            
            quality_metrics['outliers'] = {
                'mild_outliers': outliers,
                'extreme_outliers': extreme_outliers,
                'outlier_rate': (outliers / valid_points) * 100,
                'extreme_rate': (extreme_outliers / valid_points) * 100
            }
        else:
            quality_metrics['outliers'] = {
                'mild_outliers': 0,
                'extreme_outliers': 0,
                'outlier_rate': 0,
                'extreme_rate': 0
            }
        
        # 3. 连续性检查
        if len(data_series.index) > 1:
            # 检查时间间隔的一致性
            time_diffs = pd.Series(data_series.index).diff().dropna()
            if len(time_diffs) > 0:
                mode_interval = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else time_diffs.iloc[0]
                irregular_intervals = (time_diffs != mode_interval).sum()
                continuity_rate = (1 - irregular_intervals / len(time_diffs)) * 100
            else:
                continuity_rate = 100
        else:
            continuity_rate = 100
        
        quality_metrics['continuity'] = {
            'irregular_intervals': irregular_intervals if len(data_series.index) > 1 else 0,
            'continuity_rate': continuity_rate
        }
        
        # 4. 数值范围合理性
        if valid_points > 0:
            data_range = data_series.max() - data_series.min()
            mean_val = data_series.mean()
            std_val = data_series.std()
            
            # 检查是否有无穷大或NaN值
            inf_count = np.isinf(data_series).sum()
            zero_variance = std_val == 0
            
            quality_metrics['value_range'] = {
                'min_value': data_series.min(),
                'max_value': data_series.max(),
                'range': data_range,
                'mean': mean_val,
                'std': std_val,
                'infinite_values': inf_count,
                'zero_variance': zero_variance
            }
        else:
            quality_metrics['value_range'] = {
                'min_value': np.nan,
                'max_value': np.nan,
                'range': 0,
                'mean': np.nan,
                'std': np.nan,
                'infinite_values': 0,
                'zero_variance': True
            }
        
        # 5. 综合质量评分
        completeness_score = min(completeness / 95 * 100, 100)  # 以95%为满分
        outlier_penalty = min(quality_metrics['outliers']['outlier_rate'] * 2, 30)  # 异常值扣分，最多扣30分
        continuity_score = min(continuity_rate / 95 * 100, 100)
        
        overall_score = max(0, (completeness_score + continuity_score) / 2 - outlier_penalty)
        
        # 质量等级评定
        if overall_score >= self.health_thresholds['data_quality']['excellent']:
            quality_grade = "🟢 优秀"
            quality_desc = "数据质量优秀，可靠性高"
        elif overall_score >= self.health_thresholds['data_quality']['good']:
            quality_grade = "🔵 良好"
            quality_desc = "数据质量良好，基本可靠"
        elif overall_score >= self.health_thresholds['data_quality']['acceptable']:
            quality_grade = "🟡 可接受"
            quality_desc = "数据质量可接受，需要注意"
        elif overall_score >= self.health_thresholds['data_quality']['poor']:
            quality_grade = "🟠 较差"
            quality_desc = "数据质量较差，需要改进"
        else:
            quality_grade = "🔴 严重问题"
            quality_desc = "数据质量存在严重问题"
        
        quality_metrics['overall'] = {
            'score': overall_score,
            'grade': quality_grade,
            'description': quality_desc
        }
        
        return quality_metrics
    
    def assess_factor_stability(self, factor_series, factor_name):
        """评估因子稳定性"""
        stability_metrics = {}
        
        if len(factor_series.dropna()) < 2:
            return {
                'overall': {
                    'grade': "🔴 无法评估",
                    'description': "数据不足，无法评估稳定性"
                }
            }
        
        clean_data = factor_series.dropna()
        
        # 1. 变异系数（相对标准差）
        mean_val = clean_data.mean()
        std_val = clean_data.std()
        cv = std_val / abs(mean_val) if mean_val != 0 else float('inf')
        
        stability_metrics['variation'] = {
            'coefficient_of_variation': cv,
            'mean': mean_val,
            'std': std_val
        }
        
        # 2. 滚动标准差稳定性
        if len(clean_data) >= 20:
            rolling_std = clean_data.rolling(window=20).std()
            rolling_std_clean = rolling_std.dropna()
            
            if len(rolling_std_clean) > 0:
                std_of_std = rolling_std_clean.std()
                mean_of_std = rolling_std_clean.mean()
                std_stability = std_of_std / mean_of_std if mean_of_std != 0 else float('inf')
            else:
                std_stability = float('inf')
        else:
            std_stability = float('inf')
        
        stability_metrics['rolling_stability'] = {
            'std_of_rolling_std': std_stability
        }
        
        # 3. 趋势稳定性
        if len(clean_data) >= 10:
            # 计算局部趋势的方差
            window_size = min(20, len(clean_data) // 5)
            trends = []
            
            for i in range(window_size, len(clean_data)):
                window_data = clean_data.iloc[i-window_size:i]
                if len(window_data) > 1:
                    x = np.arange(len(window_data))
                    trend = np.polyfit(x, window_data.values, 1)[0]
                    trends.append(trend)
            
            if trends:
                trend_stability = np.std(trends) / (np.abs(np.mean(trends)) + 1e-8)
            else:
                trend_stability = float('inf')
        else:
            trend_stability = float('inf')
        
        stability_metrics['trend_stability'] = {
            'trend_variation': trend_stability
        }
        
        # 4. 综合稳定性评分
        cv_score = 100 if cv <= self.health_thresholds['stability']['very_stable'] else \
                  80 if cv <= self.health_thresholds['stability']['stable'] else \
                  60 if cv <= self.health_thresholds['stability']['moderate'] else \
                  40 if cv <= self.health_thresholds['stability']['unstable'] else 20
        
        stability_score = cv_score  # 主要基于变异系数
        
        # 稳定性等级评定
        if stability_score >= 90:
            stability_grade = "🟢 非常稳定"
            stability_desc = "因子表现非常稳定，可靠性极高"
        elif stability_score >= 70:
            stability_grade = "🔵 稳定"
            stability_desc = "因子表现稳定，可靠性良好"
        elif stability_score >= 50:
            stability_grade = "🟡 中等稳定"
            stability_desc = "因子稳定性中等，需要监控"
        elif stability_score >= 30:
            stability_grade = "🟠 不稳定"
            stability_desc = "因子不够稳定，风险较高"
        else:
            stability_grade = "🔴 高度不稳定"
            stability_desc = "因子高度不稳定，不建议使用"
        
        stability_metrics['overall'] = {
            'score': stability_score,
            'grade': stability_grade,
            'description': stability_desc
        }
        
        return stability_metrics
    
    def assess_factor_predictability(self, factor_series, factor_name):
        """评估因子可预测性"""
        predictability_metrics = {}
        
        if len(factor_series.dropna()) < 5:
            return {
                'overall': {
                    'grade': "🔴 无法评估",
                    'description': "数据不足，无法评估可预测性"
                }
            }
        
        clean_data = factor_series.dropna()
        
        # 1. 自相关分析
        autocorr_lags = [1, 5, 10, 20]
        autocorr_results = {}
        
        for lag in autocorr_lags:
            if len(clean_data) > lag:
                autocorr = clean_data.autocorr(lag=lag)
                autocorr_results[f'lag_{lag}'] = autocorr if not np.isnan(autocorr) else 0
            else:
                autocorr_results[f'lag_{lag}'] = 0
        
        predictability_metrics['autocorrelation'] = autocorr_results
        
        # 2. 趋势持续性
        if len(clean_data) >= 20:
            # 计算局部趋势的一致性
            window_size = 10
            trend_signs = []
            
            for i in range(window_size, len(clean_data)):
                window_data = clean_data.iloc[i-window_size:i]
                if len(window_data) > 1:
                    x = np.arange(len(window_data))
                    trend = np.polyfit(x, window_data.values, 1)[0]
                    trend_signs.append(1 if trend > 0 else -1 if trend < 0 else 0)
            
            if trend_signs:
                # 计算趋势一致性
                trend_consistency = np.abs(np.mean(trend_signs))
                # 计算趋势转换频率
                trend_changes = sum(1 for i in range(1, len(trend_signs)) 
                                  if trend_signs[i] != trend_signs[i-1])
                change_rate = trend_changes / len(trend_signs) if trend_signs else 0
            else:
                trend_consistency = 0
                change_rate = 1
        else:
            trend_consistency = 0
            change_rate = 1
        
        predictability_metrics['trend_analysis'] = {
            'trend_consistency': trend_consistency,
            'trend_change_rate': change_rate
        }
        
        # 3. 周期性检测
        if len(clean_data) >= 50:
            # 简单的周期性检测（基于自相关峰值）
            max_lag = min(50, len(clean_data) // 3)
            autocorr_series = [clean_data.autocorr(lag=i) for i in range(1, max_lag)]
            autocorr_series = [x for x in autocorr_series if not np.isnan(x)]
            
            if autocorr_series:
                # 寻找显著的正自相关
                significant_correlations = [abs(x) for x in autocorr_series if abs(x) > 0.2]
                periodicity_strength = max(significant_correlations) if significant_correlations else 0
            else:
                periodicity_strength = 0
        else:
            periodicity_strength = 0
        
        predictability_metrics['periodicity'] = {
            'strength': periodicity_strength
        }
        
        # 4. 综合可预测性评分
        # 主要基于1期滞后自相关
        main_autocorr = autocorr_results.get('lag_1', 0)
        
        if main_autocorr > self.health_thresholds['predictability']['high']:
            pred_score = 90
            pred_grade = "🟢 高可预测性"
            pred_desc = "因子具有良好的可预测性"
        elif main_autocorr > self.health_thresholds['predictability']['medium']:
            pred_score = 70
            pred_grade = "🔵 中等可预测性"
            pred_desc = "因子具有一定的可预测性"
        elif main_autocorr > self.health_thresholds['predictability']['low']:
            pred_score = 50
            pred_grade = "🟡 低可预测性"
            pred_desc = "因子可预测性较低"
        elif main_autocorr > self.health_thresholds['predictability']['very_low']:
            pred_score = 30
            pred_grade = "🟠 很低可预测性"
            pred_desc = "因子几乎不可预测"
        else:
            pred_score = 10
            pred_grade = "🔴 不可预测"
            pred_desc = "因子不可预测或反向可预测"
        
        predictability_metrics['overall'] = {
            'score': pred_score,
            'grade': pred_grade,
            'description': pred_desc
        }
        
        return predictability_metrics
    
    def diagnose_factor_health(self, factor_series, factor_name, market_data=None):
        """综合诊断因子健康状态"""
        self.print_diagnostic_section(f"{factor_name} 因子健康诊断")
        
        # 1. 数据质量诊断
        quality_metrics = self.assess_data_quality(factor_series, factor_name)
        
        # 2. 稳定性诊断
        stability_metrics = self.assess_factor_stability(factor_series, factor_name)
        
        # 3. 可预测性诊断
        predictability_metrics = self.assess_factor_predictability(factor_series, factor_name)
        
        # 4. 因子特定检查
        specific_checks = self.perform_factor_specific_checks(factor_series, factor_name, market_data)
        
        # 5. 综合健康评分
        quality_score = quality_metrics['overall']['score']
        stability_score = stability_metrics['overall']['score']
        predictability_score = predictability_metrics['overall']['score']
        
        # 加权综合评分
        weights = {'quality': 0.4, 'stability': 0.35, 'predictability': 0.25}
        overall_health_score = (
            quality_score * weights['quality'] +
            stability_score * weights['stability'] + 
            predictability_score * weights['predictability']
        )
        
        # 健康等级评定
        if overall_health_score >= 85:
            health_grade = "🟢 健康"
            health_desc = "因子整体健康状况优秀"
            health_advice = "可以正常使用，建议持续监控"
        elif overall_health_score >= 70:
            health_grade = "🔵 良好"
            health_desc = "因子健康状况良好"
            health_advice = "可以使用，注意定期检查"
        elif overall_health_score >= 55:
            health_grade = "🟡 一般"
            health_desc = "因子健康状况一般"
            health_advice = "可谨慎使用，需要密切监控"
        elif overall_health_score >= 40:
            health_grade = "🟠 较差"
            health_desc = "因子健康状况较差"
            health_advice = "不建议使用，需要优化改进"
        else:
            health_grade = "🔴 严重问题"
            health_desc = "因子存在严重健康问题"
            health_advice = "不建议使用，需要重新设计"
        
        # 输出诊断结果
        if self.console:
            health_table = Table(title=f"{factor_name} 因子健康诊断报告")
            health_table.add_column("诊断维度", style="cyan")
            health_table.add_column("评分", style="white")
            health_table.add_column("等级", style="yellow")
            health_table.add_column("权重", style="dim")
            
            health_table.add_row(
                "数据质量", 
                f"{quality_score:.1f}",
                quality_metrics['overall']['grade'],
                f"{weights['quality']*100:.0f}%"
            )
            health_table.add_row(
                "稳定性",
                f"{stability_score:.1f}",
                stability_metrics['overall']['grade'], 
                f"{weights['stability']*100:.0f}%"
            )
            health_table.add_row(
                "可预测性",
                f"{predictability_score:.1f}",
                predictability_metrics['overall']['grade'],
                f"{weights['predictability']*100:.0f}%"
            )
            
            self.console.print(health_table)
            
            # 综合健康报告
            health_panel = Panel(
                f"""[bold]🏥 综合健康评估[/bold]

[yellow]整体健康评分:[/yellow] {overall_health_score:.1f}/100
[yellow]健康等级:[/yellow] {health_grade}
[yellow]健康描述:[/yellow] {health_desc}

[cyan]使用建议:[/cyan] {health_advice}

[dim]关键指标:[/dim]
• 数据完整性: {quality_metrics['completeness']['completeness_rate']:.1f}%
• 异常值比例: {quality_metrics['outliers']['outlier_rate']:.1f}%  
• 变异系数: {stability_metrics['variation']['coefficient_of_variation']:.3f}
• 自相关(lag1): {predictability_metrics['autocorrelation']['lag_1']:.3f}
                """,
                title=f"📊 {factor_name} 健康报告",
                border_style="bright_blue"
            )
            self.console.print(health_panel)
        else:
            print(f"\n📊 {factor_name} 因子健康诊断:")
            print(f"整体健康评分: {overall_health_score:.1f}/100")
            print(f"健康等级: {health_grade}")
            print(f"使用建议: {health_advice}")
        
        return {
            'factor_name': factor_name,
            'overall_health': {
                'score': overall_health_score,
                'grade': health_grade,
                'description': health_desc,
                'advice': health_advice
            },
            'quality_metrics': quality_metrics,
            'stability_metrics': stability_metrics,
            'predictability_metrics': predictability_metrics,
            'specific_checks': specific_checks
        }
    
    def perform_factor_specific_checks(self, factor_series, factor_name, market_data):
        """执行因子特定的检查"""
        specific_checks = {}
        
        if 'funding' in factor_name.lower():
            # 资金费率相关检查
            specific_checks['extreme_values'] = (abs(factor_series) > 2.0).sum()
            specific_checks['zero_values'] = (factor_series == 0).sum()
            specific_checks['sign_changes'] = sum(
                1 for i in range(1, len(factor_series)) 
                if np.sign(factor_series.iloc[i]) != np.sign(factor_series.iloc[i-1])
            ) if len(factor_series) > 1 else 0
            
        elif 'whale' in factor_name.lower():
            # 巨鲸检测相关检查
            specific_checks['alert_frequency'] = (abs(factor_series) > 1.0).sum()
            specific_checks['high_impact_alerts'] = (abs(factor_series) > 2.0).sum()
            specific_checks['max_alert_strength'] = abs(factor_series).max()
            
        elif 'fear' in factor_name.lower() or 'greed' in factor_name.lower():
            # 恐惧贪婪指数相关检查
            specific_checks['extreme_fear'] = (factor_series < 25).sum()
            specific_checks['extreme_greed'] = (factor_series > 75).sum()
            specific_checks['neutral_periods'] = ((factor_series >= 45) & (factor_series <= 55)).sum()
            specific_checks['value_range_check'] = {
                'min_valid': factor_series.min() >= 0,
                'max_valid': factor_series.max() <= 100
            }
            
        elif 'liquidity' in factor_name.lower() or 'volatility' in factor_name.lower():
            # 流动性/波动率相关检查
            specific_checks['high_volatility_periods'] = (factor_series > 100).sum() if factor_series.max() > 1 else 0
            specific_checks['zero_volatility'] = (factor_series == 0).sum()
            specific_checks['negative_values'] = (factor_series < 0).sum()
        
        return specific_checks
    
    def run_comprehensive_diagnostics(self, market_data):
        """运行综合诊断"""
        self.print_diagnostic_header("因子健康状态综合诊断")
        
        print("🚀 开始因子健康诊断...")
        print(f"⏰ 诊断时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 准备因子数据
        symbol = list(market_data.keys())[0] if market_data else 'BTC/USDT'
        if symbol != 'funding_rates':
            price = market_data[symbol]['price']
            volume = market_data[symbol]['volume']
            amount = market_data[symbol]['amount']
            funding_rates = market_data.get('funding_rates', pd.Series())
            
            # 计算各个因子
            factors_to_diagnose = {}
            
            # 1. 恐惧贪婪指数
            try:
                fg_index = self.crypto_utils.FEAR_GREED_INDEX(price, volume)
                factors_to_diagnose['恐惧贪婪指数'] = fg_index
            except Exception as e:
                print(f"⚠️ 恐惧贪婪指数计算失败: {e}")
            
            # 2. 巨鲸交易检测
            try:
                whale_alerts = self.crypto_utils.WHALE_ALERT(volume, amount, 2.5)
                factors_to_diagnose['巨鲸交易检测'] = whale_alerts
            except Exception as e:
                print(f"⚠️ 巨鲸交易检测计算失败: {e}")
            
            # 3. 资金费率动量
            if not funding_rates.empty:
                try:
                    funding_momentum = self.crypto_utils.FUNDING_RATE_MOMENTUM(funding_rates, 24)
                    factors_to_diagnose['资金费率动量'] = funding_momentum
                except Exception as e:
                    print(f"⚠️ 资金费率动量计算失败: {e}")
            
            # 4. 市场制度识别
            try:
                market_regime = self.data_processor.detect_market_regime(price, volume)
                # 将制度转换为数值以便分析
                regime_mapping = {'bull_quiet': 4, 'bull_volatile': 3, 'sideways': 2, 
                                'bear_quiet': 1, 'bear_volatile': 0}
                numeric_regime = market_regime.map(regime_mapping).fillna(2)
                factors_to_diagnose['市场制度识别'] = numeric_regime
            except Exception as e:
                print(f"⚠️ 市场制度识别计算失败: {e}")
            
            # 执行诊断
            diagnostic_results = {}
            for factor_name, factor_data in factors_to_diagnose.items():
                if not factor_data.empty:
                    diagnostic_results[factor_name] = self.diagnose_factor_health(
                        factor_data, factor_name, market_data
                    )
            
            # 生成诊断总结
            self.generate_diagnostic_summary(diagnostic_results)
            
            return diagnostic_results
        
        return {}
    
    def generate_diagnostic_summary(self, diagnostic_results):
        """生成诊断总结"""
        self.print_diagnostic_section("诊断结果总结")
        
        if not diagnostic_results:
            print("❌ 无诊断结果")
            return
        
        # 计算平均健康评分
        health_scores = [result['overall_health']['score'] for result in diagnostic_results.values()]
        avg_health_score = np.mean(health_scores)
        
        # 统计各等级因子数量
        grade_counts = {}
        recommendations = []
        critical_issues = []
        
        for factor_name, result in diagnostic_results.items():
            grade = result['overall_health']['grade']
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
            
            # 收集关键问题
            quality_score = result['quality_metrics']['overall']['score']
            stability_score = result['stability_metrics']['overall']['score']
            
            if quality_score < 50:
                critical_issues.append(f"{factor_name}: 数据质量严重问题")
            if stability_score < 30:
                critical_issues.append(f"{factor_name}: 稳定性极差")
            
            # 生成建议
            if result['overall_health']['score'] < 40:
                recommendations.append(f"🔴 {factor_name}: 需要重新设计或停用")
            elif result['overall_health']['score'] < 70:
                recommendations.append(f"🟡 {factor_name}: 需要优化改进")
        
        # 输出总结
        if self.console:
            summary_table = Table(title="因子健康诊断总结")
            summary_table.add_column("因子名称", style="cyan")
            summary_table.add_column("健康评分", style="white")
            summary_table.add_column("健康等级", style="yellow")
            summary_table.add_column("主要问题", style="red")
            
            for factor_name, result in diagnostic_results.items():
                main_issue = ""
                if result['quality_metrics']['overall']['score'] < 70:
                    main_issue = "数据质量"
                elif result['stability_metrics']['overall']['score'] < 70:
                    main_issue = "稳定性"
                elif result['predictability_metrics']['overall']['score'] < 50:
                    main_issue = "可预测性"
                else:
                    main_issue = "无"
                
                summary_table.add_row(
                    factor_name,
                    f"{result['overall_health']['score']:.1f}",
                    result['overall_health']['grade'],
                    main_issue
                )
            
            self.console.print(summary_table)
            
            # 总体评估
            overall_panel = Panel(
                f"""[bold]🎯 整体诊断结果[/bold]

[yellow]平均健康评分:[/yellow] {avg_health_score:.1f}/100
[yellow]诊断因子数量:[/yellow] {len(diagnostic_results)} 个

[cyan]健康等级分布:[/cyan]
{chr(10).join([f'• {grade}: {count}个' for grade, count in grade_counts.items()])}

[red]关键问题:[/red]
{chr(10).join(['• ' + issue for issue in critical_issues]) if critical_issues else '• 无关键问题'}

[green]优化建议:[/green]
{chr(10).join(['• ' + rec for rec in recommendations]) if recommendations else '• 所有因子状态良好'}

[dim]诊断完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]
                """,
                title="📋 综合诊断报告",
                border_style="bright_green"
            )
            self.console.print(overall_panel)
        else:
            print(f"\n📊 整体诊断结果:")
            print(f"平均健康评分: {avg_health_score:.1f}/100")
            print(f"诊断因子数量: {len(diagnostic_results)} 个")
            
            if critical_issues:
                print("\n关键问题:")
                for issue in critical_issues:
                    print(f"• {issue}")
            
            if recommendations:
                print("\n优化建议:")
                for rec in recommendations:
                    print(f"• {rec}")


def main():
    """主函数"""
    diagnostics = FactorHealthDiagnostics()
    
    print("🔬 启动因子健康诊断系统...")
    
    # 生成测试数据（这里应该替换为实际的市场数据）
    from risk_indicators_checker import RiskIndicatorsChecker
    checker = RiskIndicatorsChecker()
    market_data = checker.generate_test_data(['BTC/USDT'])
    
    try:
        # 运行综合诊断
        results = diagnostics.run_comprehensive_diagnostics(market_data)
        
        print("\n🎉 因子健康诊断完成！")
        print("\n💡 诊断说明:")
        print("• 数据质量: 评估数据完整性、异常值、连续性等")
        print("• 稳定性: 评估因子数值的稳定程度和可靠性")  
        print("• 可预测性: 评估因子的自相关性和趋势一致性")
        print("• 建议定期运行诊断以监控因子健康状况")
        
        return results
        
    except Exception as e:
        print(f"❌ 诊断过程出现错误: {str(e)}")
        return None


if __name__ == "__main__":
    main()