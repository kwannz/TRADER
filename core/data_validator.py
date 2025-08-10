"""
数据验证和异常处理框架
实现多层数据验证、异常检测和数据质量监控
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import statistics
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_logger import get_logger, LogCategory

logger = get_logger()

class ValidationLevel(Enum):
    """验证级别"""
    BASIC = "basic"           # 基础验证
    STANDARD = "standard"     # 标准验证
    STRICT = "strict"         # 严格验证
    ENTERPRISE = "enterprise" # 企业级验证

class DataQuality(Enum):
    """数据质量等级"""
    EXCELLENT = "excellent"   # 优秀 (95-100%)
    GOOD = "good"             # 良好 (80-94%)
    ACCEPTABLE = "acceptable" # 可接受 (60-79%)
    POOR = "poor"             # 较差 (40-59%)
    INVALID = "invalid"       # 无效 (0-39%)

@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    quality_score: float  # 0-100
    quality_level: DataQuality
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def add_issue(self, issue: str):
        """添加问题"""
        self.issues.append(issue)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """添加警告"""
        self.warnings.append(warning)
    
    def update_quality(self):
        """更新质量等级"""
        if self.quality_score >= 95:
            self.quality_level = DataQuality.EXCELLENT
        elif self.quality_score >= 80:
            self.quality_level = DataQuality.GOOD
        elif self.quality_score >= 60:
            self.quality_level = DataQuality.ACCEPTABLE
        elif self.quality_score >= 40:
            self.quality_level = DataQuality.POOR
        else:
            self.quality_level = DataQuality.INVALID

@dataclass
class DataQualityMetrics:
    """数据质量指标"""
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    warning_records: int = 0
    
    avg_quality_score: float = 0.0
    min_quality_score: float = 100.0
    max_quality_score: float = 0.0
    
    validation_time_ms: float = 0.0
    throughput_per_second: float = 0.0
    
    last_update: datetime = field(default_factory=datetime.utcnow)
    
    def update_stats(self, results: List[ValidationResult]):
        """更新统计信息"""
        start_time = datetime.utcnow()
        
        self.total_records = len(results)
        self.valid_records = sum(1 for r in results if r.is_valid)
        self.invalid_records = self.total_records - self.valid_records
        self.warning_records = sum(1 for r in results if r.warnings)
        
        if results:
            scores = [r.quality_score for r in results]
            self.avg_quality_score = statistics.mean(scores)
            self.min_quality_score = min(scores)
            self.max_quality_score = max(scores)
        
        # 计算性能指标
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        self.validation_time_ms = duration
        self.throughput_per_second = self.total_records / max(duration / 1000, 0.001)
        
        self.last_update = datetime.utcnow()
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        return (self.valid_records / max(self.total_records, 1)) * 100

class PriceValidator:
    """价格数据验证器"""
    
    def __init__(self):
        self.price_history = {}  # 存储历史价格用于异常检测
        self.max_history_length = 100
        
        # 配置参数
        self.min_price = 0.00001  # 最小价格
        self.max_price = 10000000  # 最大价格
        self.max_price_change_pct = 50  # 最大价格变化百分比
        self.max_volume_spike_ratio = 100  # 最大成交量异常倍数
        self.data_staleness_threshold = 300  # 数据过期阈值(秒)
    
    def validate_price_data(self, data: Dict, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """验证价格数据"""
        result = ValidationResult(is_valid=True, quality_score=100.0, quality_level=DataQuality.EXCELLENT)
        
        try:
            symbol = data.get("symbol", "UNKNOWN")
            exchange = data.get("exchange", "UNKNOWN")
            
            # 基础字段检查
            required_fields = ["symbol", "exchange", "price", "timestamp"]
            for field in required_fields:
                if field not in data:
                    result.add_issue(f"缺少必需字段: {field}")
                    result.quality_score -= 20
            
            if not result.is_valid:
                result.update_quality()
                return result
            
            # 提取价格数据
            price = float(data.get("price", 0))
            volume = float(data.get("volume_24h", 0))
            change_pct = float(data.get("change_24h_pct", 0))
            timestamp = data.get("timestamp", 0)
            
            # 价格范围检查
            if price <= self.min_price or price >= self.max_price:
                result.add_issue(f"价格超出合理范围: {price}")
                result.quality_score -= 30
            
            # 成交量检查
            if volume < 0:
                result.add_issue(f"成交量异常: {volume}")
                result.quality_score -= 25
            
            # 时间戳检查
            if timestamp > 0:
                data_time = datetime.fromtimestamp(timestamp / 1000)
                age = (datetime.utcnow() - data_time).total_seconds()
                if age > self.data_staleness_threshold:
                    result.add_warning(f"数据过期: {age:.1f}秒")
                    result.quality_score -= 10
                elif age < 0:
                    result.add_issue(f"未来时间戳异常: {age:.1f}秒")
                    result.quality_score -= 20
            
            # 历史价格对比 (标准级别以上)
            if level.value in ["standard", "strict", "enterprise"]:
                price_key = f"{exchange}:{symbol}"
                
                if price_key in self.price_history:
                    last_price = self.price_history[price_key][-1] if self.price_history[price_key] else price
                    
                    if last_price > 0:
                        actual_change = abs(price - last_price) / last_price * 100
                        
                        # 价格变化异常检测
                        if actual_change > self.max_price_change_pct:
                            result.add_warning(f"价格变化异常: {actual_change:.2f}% (上次: ${last_price}, 当前: ${price})")
                            result.quality_score -= 15
                        
                        # 异常值检测 (严格级别以上)
                        if level.value in ["strict", "enterprise"] and len(self.price_history[price_key]) >= 10:
                            price_array = np.array(self.price_history[price_key])
                            mean_price = np.mean(price_array)
                            std_price = np.std(price_array)
                            
                            # 3-sigma异常检测
                            if abs(price - mean_price) > 3 * std_price:
                                result.add_warning(f"价格3-sigma异常: 当前${price:.2f}, 均值${mean_price:.2f}±{std_price:.2f}")
                                result.quality_score -= 10
                
                # 更新价格历史
                if price_key not in self.price_history:
                    self.price_history[price_key] = []
                
                self.price_history[price_key].append(price)
                if len(self.price_history[price_key]) > self.max_history_length:
                    self.price_history[price_key] = self.price_history[price_key][-self.max_history_length:]
            
            # 成交量异常检测 (企业级)
            if level == ValidationLevel.ENTERPRISE and volume > 0:
                volume_key = f"{exchange}:{symbol}:volume"
                
                if volume_key in self.price_history:
                    recent_volumes = self.price_history[volume_key]
                    if recent_volumes:
                        avg_volume = statistics.mean(recent_volumes)
                        if avg_volume > 0 and volume / avg_volume > self.max_volume_spike_ratio:
                            result.add_warning(f"成交量异常飙升: {volume/avg_volume:.1f}倍")
                            result.quality_score -= 5
                
                if volume_key not in self.price_history:
                    self.price_history[volume_key] = []
                
                self.price_history[volume_key].append(volume)
                if len(self.price_history[volume_key]) > self.max_history_length:
                    self.price_history[volume_key] = self.price_history[volume_key][-self.max_history_length:]
            
            # 数据一致性检查
            if abs(change_pct) > self.max_price_change_pct:
                result.add_warning(f"24h变化百分比异常: {change_pct:.2f}%")
                result.quality_score -= 5
            
            # 字段完整性评分
            optional_fields = ["high_24h", "low_24h", "volume_24h", "change_24h", "change_24h_pct"]
            present_optional = sum(1 for field in optional_fields if field in data and data[field] is not None)
            completeness_score = present_optional / len(optional_fields) * 10
            result.quality_score = min(100, result.quality_score + completeness_score - 10)
            
            # 更新质量等级
            result.quality_score = max(0, result.quality_score)
            result.update_quality()
            
            # 添加元数据
            result.metadata = {
                "symbol": symbol,
                "exchange": exchange,
                "price": price,
                "validation_level": level.value,
                "history_length": len(self.price_history.get(price_key, [])),
                "data_age_seconds": age if timestamp > 0 else None
            }
            
        except (ValueError, TypeError, KeyError) as e:
            result.add_issue(f"数据格式错误: {str(e)}")
            result.quality_score = 0
            result.update_quality()
        
        return result

class CandleValidator:
    """K线数据验证器"""
    
    def __init__(self):
        self.candle_history = {}
        self.max_history_length = 50
    
    def validate_candle_data(self, data: List[Dict], level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """验证K线数据"""
        result = ValidationResult(is_valid=True, quality_score=100.0, quality_level=DataQuality.EXCELLENT)
        
        if not data:
            result.add_issue("K线数据为空")
            result.quality_score = 0
            result.update_quality()
            return result
        
        try:
            valid_candles = 0
            
            for i, candle in enumerate(data):
                candle_result = self._validate_single_candle(candle, i, level)
                
                if candle_result.is_valid:
                    valid_candles += 1
                else:
                    result.issues.extend([f"K线{i}: {issue}" for issue in candle_result.issues])
                
                result.warnings.extend([f"K线{i}: {warning}" for warning in candle_result.warnings])
                result.quality_score = min(result.quality_score, candle_result.quality_score)
            
            # 计算整体质量分数
            if data:
                result.quality_score = (valid_candles / len(data)) * result.quality_score
            
            result.is_valid = valid_candles == len(data)
            result.update_quality()
            
            result.metadata = {
                "total_candles": len(data),
                "valid_candles": valid_candles,
                "validation_level": level.value,
                "timeframe_detected": self._detect_timeframe(data)
            }
            
        except Exception as e:
            result.add_issue(f"K线验证失败: {str(e)}")
            result.quality_score = 0
            result.update_quality()
        
        return result
    
    def _validate_single_candle(self, candle: Dict, index: int, level: ValidationLevel) -> ValidationResult:
        """验证单根K线"""
        result = ValidationResult(is_valid=True, quality_score=100.0, quality_level=DataQuality.EXCELLENT)
        
        try:
            # 字段检查
            required_fields = ["open", "high", "low", "close", "volume"]
            for field in required_fields:
                if field not in candle:
                    result.add_issue(f"缺少字段: {field}")
                    result.quality_score -= 20
            
            if not result.is_valid:
                return result
            
            # 提取OHLCV数据
            o = float(candle["open"])
            h = float(candle["high"])
            l = float(candle["low"])
            c = float(candle["close"])
            v = float(candle["volume"])
            
            # OHLC关系检查
            if not (l <= o <= h and l <= c <= h):
                result.add_issue(f"OHLC价格关系异常: O={o}, H={h}, L={l}, C={c}")
                result.quality_score -= 30
            
            # 高低价关系检查
            if h < l:
                result.add_issue(f"最高价小于最低价: H={h}, L={l}")
                result.quality_score -= 25
            
            # 成交量检查
            if v < 0:
                result.add_issue(f"成交量为负数: {v}")
                result.quality_score -= 20
            elif v == 0:
                result.add_warning(f"成交量为零")
                result.quality_score -= 5
            
            # 价格异常检查
            prices = [o, h, l, c]
            if any(p <= 0 for p in prices):
                result.add_issue(f"存在零或负价格: OHLC={prices}")
                result.quality_score -= 25
            
            if any(p > 10000000 for p in prices):
                result.add_issue(f"价格过高异常: OHLC={prices}")
                result.quality_score -= 15
            
            # 时间戳检查 (如果存在)
            if "timestamp" in candle:
                timestamp = candle["timestamp"]
                if timestamp > 0:
                    candle_time = datetime.fromtimestamp(timestamp / 1000)
                    age = (datetime.utcnow() - candle_time).total_seconds()
                    
                    # 检查时间合理性
                    if age < -3600:  # 不能超前1小时
                        result.add_issue(f"时间戳过于超前: {age:.1f}秒")
                        result.quality_score -= 15
                    elif age > 86400 * 30:  # 不能超过30天
                        result.add_warning(f"历史数据时间较久: {age/86400:.1f}天")
                        result.quality_score -= 5
            
            # 严格模式下的额外检查
            if level in [ValidationLevel.STRICT, ValidationLevel.ENTERPRISE]:
                # 价格波动率检查
                if h > l:
                    volatility = (h - l) / l * 100
                    if volatility > 20:  # 单根K线波动超过20%
                        result.add_warning(f"单根K线波动过大: {volatility:.2f}%")
                        result.quality_score -= 5
                
                # 上下影线检查
                if o != c:  # 非十字星
                    body_size = abs(c - o)
                    upper_shadow = h - max(o, c)
                    lower_shadow = min(o, c) - l
                    
                    # 异常长影线检查
                    if body_size > 0:
                        upper_ratio = upper_shadow / body_size
                        lower_ratio = lower_shadow / body_size
                        
                        if upper_ratio > 10 or lower_ratio > 10:
                            result.add_warning(f"影线过长: 上影线比例={upper_ratio:.1f}, 下影线比例={lower_ratio:.1f}")
                            result.quality_score -= 3
            
            result.update_quality()
            
        except (ValueError, TypeError, KeyError) as e:
            result.add_issue(f"数据解析错误: {str(e)}")
            result.quality_score = 0
            result.update_quality()
        
        return result
    
    def _detect_timeframe(self, data: List[Dict]) -> str:
        """检测时间周期"""
        if len(data) < 2:
            return "unknown"
        
        try:
            timestamps = []
            for candle in data:
                if "timestamp" in candle:
                    timestamps.append(candle["timestamp"])
            
            if len(timestamps) >= 2:
                intervals = []
                for i in range(1, len(timestamps)):
                    interval = abs(timestamps[i] - timestamps[i-1]) / 1000  # 转换为秒
                    intervals.append(interval)
                
                avg_interval = statistics.median(intervals)
                
                # 判断时间周期
                if 50 <= avg_interval <= 70:
                    return "1m"
                elif 290 <= avg_interval <= 310:
                    return "5m"
                elif 890 <= avg_interval <= 910:
                    return "15m"
                elif 3500 <= avg_interval <= 3700:
                    return "1h"
                elif 14000 <= avg_interval <= 14800:
                    return "4h"
                elif 86000 <= avg_interval <= 87000:
                    return "1d"
                else:
                    return f"custom_{int(avg_interval)}s"
        
        except Exception:
            pass
        
        return "unknown"

class MarketDataValidator:
    """综合市场数据验证器"""
    
    def __init__(self):
        self.price_validator = PriceValidator()
        self.candle_validator = CandleValidator()
        self.metrics = DataQualityMetrics()
        
        # 验证统计
        self.validation_history = []
        self.max_history_length = 1000
    
    async def validate_market_data(self, data: Dict, data_type: str = "price", 
                                  level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """验证市场数据"""
        start_time = datetime.utcnow()
        
        try:
            if data_type == "price" or data_type == "ticker":
                result = self.price_validator.validate_price_data(data, level)
            elif data_type == "candle" or data_type == "kline":
                candle_list = data if isinstance(data, list) else [data]
                result = self.candle_validator.validate_candle_data(candle_list, level)
            else:
                result = ValidationResult(is_valid=False, quality_score=0, quality_level=DataQuality.INVALID)
                result.add_issue(f"不支持的数据类型: {data_type}")
            
            # 记录验证结果
            self.validation_history.append(result)
            if len(self.validation_history) > self.max_history_length:
                self.validation_history = self.validation_history[-self.max_history_length:]
            
            # 更新性能指标
            validation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            result.metadata["validation_time_ms"] = validation_time
            
            # 记录质量问题
            if not result.is_valid:
                logger.warning(f"数据验证失败 [{data_type}]: {', '.join(result.issues)}")
            elif result.warnings:
                logger.info(f"数据验证警告 [{data_type}]: {', '.join(result.warnings)}")
            
            return result
            
        except Exception as e:
            logger.error(f"数据验证异常: {e}")
            result = ValidationResult(is_valid=False, quality_score=0, quality_level=DataQuality.INVALID)
            result.add_issue(f"验证过程异常: {str(e)}")
            return result
    
    def get_quality_report(self, hours: int = 1) -> Dict[str, Any]:
        """获取数据质量报告"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_results = [r for r in self.validation_history if r.timestamp >= cutoff_time]
        
        if not recent_results:
            return {"message": "暂无数据质量记录"}
        
        # 更新指标
        self.metrics.update_stats(recent_results)
        
        # 质量分布
        quality_distribution = {}
        for quality in DataQuality:
            count = sum(1 for r in recent_results if r.quality_level == quality)
            quality_distribution[quality.value] = {
                "count": count,
                "percentage": count / len(recent_results) * 100
            }
        
        # 常见问题统计
        all_issues = []
        all_warnings = []
        for result in recent_results:
            all_issues.extend(result.issues)
            all_warnings.extend(result.warnings)
        
        issue_stats = {}
        for issue in all_issues:
            issue_stats[issue] = issue_stats.get(issue, 0) + 1
        
        warning_stats = {}
        for warning in all_warnings:
            warning_stats[warning] = warning_stats.get(warning, 0) + 1
        
        return {
            "time_range": f"最近{hours}小时",
            "summary": {
                "total_validations": self.metrics.total_records,
                "success_rate": f"{self.metrics.get_success_rate():.2f}%",
                "average_quality_score": f"{self.metrics.avg_quality_score:.2f}",
                "validation_performance": f"{self.metrics.throughput_per_second:.1f} 条/秒"
            },
            "quality_distribution": quality_distribution,
            "top_issues": dict(list(sorted(issue_stats.items(), key=lambda x: x[1], reverse=True))[:5]),
            "top_warnings": dict(list(sorted(warning_stats.items(), key=lambda x: x[1], reverse=True))[:5]),
            "performance": {
                "avg_validation_time_ms": self.metrics.validation_time_ms,
                "throughput_per_second": self.metrics.throughput_per_second
            },
            "last_update": self.metrics.last_update.isoformat()
        }

# 创建全局验证器实例
market_data_validator = MarketDataValidator()