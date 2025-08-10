"""
AI Factor Discovery & CTBench Integration Bridge
AI因子发现与CTBench系统集成桥接器
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass

from .model_service import CTBenchModelService, get_ctbench_service
from .risk_control.enhanced_risk_manager import EnhancedRiskManager

@dataclass
class FactorEnhancementRequest:
    """因子增强请求"""
    factor_name: str
    factor_formula: str
    factor_type: str
    base_data: np.ndarray
    enhancement_scenarios: List[str]
    validation_period: int = 252

@dataclass
class FactorValidationResult:
    """因子验证结果"""
    factor_name: str
    ctbench_ic: float
    synthetic_data_ic: float
    stress_test_results: Dict[str, float]
    robustness_score: float
    enhancement_recommendations: List[str]

class AIFactorCTBenchBridge:
    """AI因子发现与CTBench集成桥接器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ctbench_service: Optional[CTBenchModelService] = None
        self.risk_manager: Optional[EnhancedRiskManager] = None
        
        # 集成配置
        self.config = {
            'data_augmentation_factor': 10,
            'stress_test_scenarios': ['black_swan', 'high_volatility', 'bear_market'],
            'validation_thresholds': {
                'min_ic': 0.02,
                'min_robustness': 0.6,
                'max_drawdown': 0.15
            }
        }
        
    async def initialize(self):
        """初始化集成服务"""
        try:
            self.ctbench_service = await get_ctbench_service()
            self.risk_manager = EnhancedRiskManager()
            await self.risk_manager.initialize()
            
            self.logger.info("AI因子发现与CTBench集成桥接器初始化成功")
        except Exception as e:
            self.logger.error(f"集成桥接器初始化失败: {e}")
            raise
            
    async def enhance_factor_with_synthetic_data(self, 
                                               factor_request: FactorEnhancementRequest) -> Dict[str, Any]:
        """使用合成数据增强因子验证"""
        try:
            self.logger.info(f"开始增强因子: {factor_request.factor_name}")
            
            # 1. 生成基础合成数据进行因子计算
            synthetic_data_result = await self._generate_synthetic_data_for_factor(
                factor_request.base_data,
                factor_request.enhancement_scenarios
            )
            
            # 2. 在合成数据上计算因子值
            synthetic_factor_values = self._calculate_factor_on_synthetic_data(
                factor_request.factor_formula,
                synthetic_data_result['augmented_data']
            )
            
            # 3. 进行压力测试
            stress_test_results = await self._stress_test_factor(
                factor_request,
                synthetic_factor_values
            )
            
            # 4. 计算因子稳健性评分
            robustness_score = self._calculate_factor_robustness(
                synthetic_factor_values,
                stress_test_results
            )
            
            # 5. 生成增强建议
            enhancement_recommendations = await self._generate_enhancement_recommendations(
                factor_request,
                synthetic_factor_values,
                stress_test_results
            )
            
            return {
                'success': True,
                'factor_name': factor_request.factor_name,
                'synthetic_data_stats': {
                    'original_samples': factor_request.base_data.shape[0],
                    'augmented_samples': synthetic_data_result['augmentation_factor'],
                    'scenarios_tested': len(factor_request.enhancement_scenarios)
                },
                'factor_performance': {
                    'synthetic_ic_mean': np.mean([v['ic'] for v in synthetic_factor_values.values()]),
                    'robustness_score': robustness_score,
                    'stress_test_survival_rate': self._calculate_survival_rate(stress_test_results)
                },
                'stress_test_results': stress_test_results,
                'enhancement_recommendations': enhancement_recommendations,
                'ctbench_integration_metrics': {
                    'data_augmentation_quality': synthetic_data_result.get('quality_score', 0.0),
                    'scenario_coverage': len(factor_request.enhancement_scenarios) / 5.0,
                    'validation_confidence': robustness_score
                }
            }
            
        except Exception as e:
            self.logger.error(f"因子增强失败: {e}")
            return {'success': False, 'error': str(e)}
            
    async def _generate_synthetic_data_for_factor(self, 
                                                base_data: np.ndarray,
                                                scenarios: List[str]) -> Dict[str, Any]:
        """为因子验证生成合成数据"""
        # 确保数据格式正确
        if len(base_data.shape) == 2:
            base_data = base_data.reshape(1, base_data.shape[0], base_data.shape[1])
            
        # 生成多场景增强数据
        augmented_results = {}
        total_augmented_samples = 0
        
        for scenario in scenarios:
            try:
                scenario_result = self.ctbench_service.synthetic_manager.generate_market_scenarios(
                    scenario, base_data, 50  # 每个场景生成50个样本
                )
                
                if scenario_result['success']:
                    augmented_results[scenario] = scenario_result['data']
                    total_augmented_samples += scenario_result['num_scenarios']
                    
            except Exception as e:
                self.logger.warning(f"场景 {scenario} 数据生成失败: {e}")
                
        # 合并所有增强数据
        all_augmented_data = []
        for scenario_data in augmented_results.values():
            all_augmented_data.extend(scenario_data)
            
        return {
            'success': True,
            'augmented_data': np.array(all_augmented_data) if all_augmented_data else base_data,
            'augmentation_factor': total_augmented_samples,
            'scenarios': augmented_results,
            'quality_score': self._assess_synthetic_data_quality(all_augmented_data, base_data)
        }
        
    def _calculate_factor_on_synthetic_data(self, 
                                          factor_formula: str,
                                          synthetic_data: np.ndarray) -> Dict[str, Any]:
        """在合成数据上计算因子值"""
        factor_results = {}
        
        try:
            # 为每个合成样本计算因子值
            for i, sample in enumerate(synthetic_data):
                # 转换为DataFrame格式便于计算
                sample_df = pd.DataFrame(sample, columns=['open', 'high', 'low', 'close', 'volume', 'adj_close'])
                
                # 计算因子值（这里简化处理，实际需要解析factor_formula）
                factor_value = self._evaluate_factor_formula(factor_formula, sample_df)
                
                # 计算前瞻收益率
                forward_returns = self._calculate_forward_returns(sample_df['close'], periods=5)
                
                # 计算IC
                if len(factor_value) > 0 and len(forward_returns) > 0:
                    ic = np.corrcoef(factor_value, forward_returns)[0, 1] if not np.isnan(np.corrcoef(factor_value, forward_returns)[0, 1]) else 0
                    factor_results[f'sample_{i}'] = {
                        'factor_values': factor_value,
                        'forward_returns': forward_returns,
                        'ic': ic
                    }
                    
        except Exception as e:
            self.logger.error(f"合成数据因子计算失败: {e}")
            
        return factor_results
        
    def _evaluate_factor_formula(self, formula: str, data: pd.DataFrame) -> np.ndarray:
        """评估因子公式（简化实现）"""
        try:
            # 添加常用的技术指标计算
            data['returns'] = data['close'].pct_change()
            data['sma_20'] = data['close'].rolling(20).mean()
            data['std_20'] = data['close'].rolling(20).std()
            data['rsi'] = self._calculate_rsi(data['close'])
            
            # 简化的公式评估（实际应该使用更安全的表达式解析）
            # 这里只是示例，生产环境需要更严格的公式解析
            if 'sma' in formula.lower():
                return data['sma_20'].values
            elif 'rsi' in formula.lower():
                return data['rsi'].values
            elif 'returns' in formula.lower():
                return data['returns'].values
            else:
                # 默认返回价格动量
                return data['close'].pct_change(5).values
                
        except Exception as e:
            self.logger.warning(f"因子公式评估失败: {e}")
            return np.zeros(len(data))
            
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_forward_returns(self, prices: pd.Series, periods: int = 5) -> np.ndarray:
        """计算前瞻收益率"""
        return prices.pct_change(periods).shift(-periods).fillna(0).values
        
    async def _stress_test_factor(self, 
                                factor_request: FactorEnhancementRequest,
                                synthetic_factor_values: Dict[str, Any]) -> Dict[str, float]:
        """对因子进行压力测试"""
        stress_results = {}
        
        try:
            # 使用CTBench的压力测试功能
            base_data_reshaped = factor_request.base_data.reshape(1, factor_request.base_data.shape[0], -1)
            
            stress_test_result = await self.ctbench_service.generate_stress_test_scenarios(
                base_data_reshaped,
                ['black_swan', 'high_volatility', 'bear_market']
            )
            
            if stress_test_result['success']:
                stress_scenarios = stress_test_result['stress_scenarios']
                
                for scenario_type, scenarios in stress_scenarios.items():
                    # 在每个压力场景下测试因子表现
                    scenario_ics = []
                    
                    for scenario in scenarios[:10]:  # 测试前10个场景
                        scenario_df = pd.DataFrame(scenario, columns=['open', 'high', 'low', 'close', 'volume', 'adj_close'])
                        
                        # 计算因子值
                        factor_values = self._evaluate_factor_formula(factor_request.factor_formula, scenario_df)
                        forward_returns = self._calculate_forward_returns(scenario_df['close'])
                        
                        # 计算IC
                        if len(factor_values) > 5 and len(forward_returns) > 5:
                            ic = np.corrcoef(factor_values[5:], forward_returns[5:])[0, 1]
                            if not np.isnan(ic):
                                scenario_ics.append(ic)
                                
                    if scenario_ics:
                        stress_results[f'{scenario_type}_mean_ic'] = np.mean(scenario_ics)
                        stress_results[f'{scenario_type}_std_ic'] = np.std(scenario_ics)
                        stress_results[f'{scenario_type}_min_ic'] = np.min(scenario_ics)
                        
        except Exception as e:
            self.logger.error(f"因子压力测试失败: {e}")
            
        return stress_results
        
    def _calculate_factor_robustness(self, 
                                   synthetic_factor_values: Dict[str, Any],
                                   stress_test_results: Dict[str, float]) -> float:
        """计算因子稳健性评分"""
        try:
            # 1. 合成数据IC稳定性评分
            ics = [result['ic'] for result in synthetic_factor_values.values() if not np.isnan(result['ic'])]
            
            if not ics:
                return 0.0
                
            ic_stability = 1.0 - (np.std(ics) / (np.abs(np.mean(ics)) + 1e-8))
            ic_strength = min(abs(np.mean(ics)) / 0.05, 1.0)  # 标准化到0.05为满分
            
            # 2. 压力测试稳定性评分
            stress_stability = 0.0
            if stress_test_results:
                min_ics = [v for k, v in stress_test_results.items() if 'min_ic' in k]
                if min_ics:
                    stress_stability = min(1.0, (np.mean(min_ics) + 0.02) / 0.02)  # -0.02以上为及格
                    
            # 3. 综合稳健性评分
            robustness_score = 0.4 * ic_stability + 0.4 * ic_strength + 0.2 * stress_stability
            
            return max(0.0, min(1.0, robustness_score))
            
        except Exception as e:
            self.logger.error(f"稳健性评分计算失败: {e}")
            return 0.0
            
    def _calculate_survival_rate(self, stress_test_results: Dict[str, float]) -> float:
        """计算压力测试生存率"""
        if not stress_test_results:
            return 0.0
            
        min_ic_threshold = -0.05  # IC低于-0.05视为失效
        survival_count = 0
        total_tests = 0
        
        for key, value in stress_test_results.items():
            if 'min_ic' in key:
                total_tests += 1
                if value > min_ic_threshold:
                    survival_count += 1
                    
        return survival_count / total_tests if total_tests > 0 else 0.0
        
    async def _generate_enhancement_recommendations(self,
                                                  factor_request: FactorEnhancementRequest,
                                                  synthetic_factor_values: Dict[str, Any],
                                                  stress_test_results: Dict[str, float]) -> List[str]:
        """生成因子增强建议"""
        recommendations = []
        
        try:
            # 分析因子表现
            ics = [result['ic'] for result in synthetic_factor_values.values() if not np.isnan(result['ic'])]
            avg_ic = np.mean(ics) if ics else 0
            
            # 基于表现生成建议
            if abs(avg_ic) < 0.02:
                recommendations.append("🔍 因子信号较弱，建议结合其他指标或调整参数")
                
            if len(ics) > 0 and np.std(ics) > 0.05:
                recommendations.append("⚡ 因子稳定性较差，建议增加平滑处理或风控机制")
                
            # 基于压力测试结果的建议
            survival_rate = self._calculate_survival_rate(stress_test_results)
            if survival_rate < 0.7:
                recommendations.append("🛡️ 极端市场下表现不佳，建议增加防御性调整")
                
            # CTBench特定建议
            recommendations.extend([
                "📊 建议使用CTBench生成更多样化的测试场景",
                "🔄 可通过CTBench的数据增强功能扩大因子验证样本",
                "⚠️ 建议集成实时风险监控系统监控因子表现"
            ])
            
        except Exception as e:
            self.logger.error(f"生成增强建议失败: {e}")
            recommendations.append("❌ 建议生成过程出错，需要人工检查")
            
        return recommendations
        
    def _assess_synthetic_data_quality(self, 
                                     synthetic_data: List[np.ndarray],
                                     original_data: np.ndarray) -> float:
        """评估合成数据质量"""
        try:
            if not synthetic_data:
                return 0.0
                
            # 计算统计相似性
            orig_stats = {
                'mean': np.mean(original_data[:, :, 0]),  # 价格均值
                'std': np.std(original_data[:, :, 0]),    # 价格标准差
                'vol': np.std(np.diff(original_data[0, :, 0]) / original_data[0, :-1, 0])  # 收益率波动率
            }
            
            synthetic_stats = []
            for sample in synthetic_data:
                sample_stats = {
                    'mean': np.mean(sample[:, 0]),
                    'std': np.std(sample[:, 0]),
                    'vol': np.std(np.diff(sample[:, 0]) / sample[:-1, 0])
                }
                synthetic_stats.append(sample_stats)
                
            # 计算相似性评分
            similarity_scores = []
            for stat_key in orig_stats.keys():
                orig_val = orig_stats[stat_key]
                synth_vals = [s[stat_key] for s in synthetic_stats]
                
                if orig_val != 0:
                    relative_errors = [abs(s - orig_val) / abs(orig_val) for s in synth_vals]
                    similarity_score = 1.0 - min(1.0, np.mean(relative_errors))
                    similarity_scores.append(similarity_score)
                    
            return np.mean(similarity_scores) if similarity_scores else 0.0
            
        except Exception as e:
            self.logger.error(f"合成数据质量评估失败: {e}")
            return 0.0
            
    async def batch_factor_validation(self, 
                                    factor_requests: List[FactorEnhancementRequest]) -> List[Dict[str, Any]]:
        """批量因子验证"""
        validation_results = []
        
        self.logger.info(f"开始批量验证 {len(factor_requests)} 个因子")
        
        # 并行处理多个因子
        tasks = []
        for factor_request in factor_requests:
            task = asyncio.create_task(
                self.enhance_factor_with_synthetic_data(factor_request)
            )
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"因子 {factor_requests[i].factor_name} 验证失败: {result}")
                validation_results.append({
                    'success': False,
                    'factor_name': factor_requests[i].factor_name,
                    'error': str(result)
                })
            else:
                validation_results.append(result)
                
        return validation_results
        
    async def real_time_factor_monitoring(self, factor_names: List[str]):
        """实时因子监控"""
        self.logger.info(f"开始实时监控 {len(factor_names)} 个因子")
        
        while True:
            try:
                for factor_name in factor_names:
                    # 获取最新市场数据
                    # latest_data = await self.get_latest_market_data()
                    
                    # 使用CTBench生成测试场景
                    # test_scenarios = await self.generate_test_scenarios(latest_data)
                    
                    # 监控因子在新场景下的表现
                    # performance = await self.monitor_factor_performance(factor_name, test_scenarios)
                    
                    # 发送监控报告
                    # await self.send_monitoring_report(factor_name, performance)
                    pass
                    
                await asyncio.sleep(300)  # 5分钟检查一次
                
            except Exception as e:
                self.logger.error(f"实时因子监控出错: {e}")
                await asyncio.sleep(60)
                
    def generate_integration_report(self, validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成集成报告"""
        successful_validations = [r for r in validation_results if r.get('success', False)]
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_factors': len(validation_results),
            'successful_validations': len(successful_validations),
            'success_rate': len(successful_validations) / len(validation_results) if validation_results else 0,
            'ctbench_integration_stats': {
                'total_synthetic_samples_generated': sum(
                    r.get('synthetic_data_stats', {}).get('augmented_samples', 0) 
                    for r in successful_validations
                ),
                'scenarios_tested': sum(
                    r.get('synthetic_data_stats', {}).get('scenarios_tested', 0)
                    for r in successful_validations
                ),
                'average_robustness_score': np.mean([
                    r.get('factor_performance', {}).get('robustness_score', 0)
                    for r in successful_validations
                ]) if successful_validations else 0
            },
            'top_performing_factors': sorted(
                successful_validations,
                key=lambda x: x.get('factor_performance', {}).get('robustness_score', 0),
                reverse=True
            )[:5],
            'common_recommendations': self._extract_common_recommendations(successful_validations)
        }
        
        return report
        
    def _extract_common_recommendations(self, validation_results: List[Dict[str, Any]]) -> List[str]:
        """提取通用建议"""
        all_recommendations = []
        for result in validation_results:
            all_recommendations.extend(result.get('enhancement_recommendations', []))
            
        # 统计最常见的建议
        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
            
        # 返回出现频率最高的建议
        common_recommendations = sorted(
            recommendation_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return [rec[0] for rec in common_recommendations]

# 单例模式的集成桥接器
ai_factor_ctbench_bridge = None

async def get_ai_factor_bridge() -> AIFactorCTBenchBridge:
    """获取AI因子CTBench集成桥接器单例"""
    global ai_factor_ctbench_bridge
    if ai_factor_ctbench_bridge is None:
        ai_factor_ctbench_bridge = AIFactorCTBenchBridge()
        await ai_factor_ctbench_bridge.initialize()
    return ai_factor_ctbench_bridge