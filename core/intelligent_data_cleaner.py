"""
智能数据清洗引擎
基于机器学习和统计方法的高级数据清洗和异常检测
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
import statistics
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_logger import get_logger, LogCategory

logger = get_logger()

@dataclass
class CleaningRule:
    """数据清洗规则"""
    rule_id: str
    rule_type: str  # "outlier", "missing", "duplicate", "format", "business"
    severity: str   # "low", "medium", "high", "critical"
    description: str
    enabled: bool = True
    auto_fix: bool = False
    threshold: float = 0.0
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CleaningResult:
    """数据清洗结果"""
    original_count: int
    cleaned_count: int
    removed_count: int
    modified_count: int
    issues_detected: List[Dict[str, Any]]
    quality_score: float
    processing_time: float
    rules_applied: List[str]

class StatisticalAnalyzer:
    """统计分析器"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.price_history = defaultdict(lambda: deque(maxlen=window_size))
        self.volume_history = defaultdict(lambda: deque(maxlen=window_size))
        
    def update_statistics(self, symbol: str, price: float, volume: float):
        """更新统计数据"""
        if price > 0:
            self.price_history[symbol].append(price)
        if volume > 0:
            self.volume_history[symbol].append(volume)
    
    def detect_price_outliers(self, symbol: str, price: float, method: str = "zscore") -> Tuple[bool, float]:
        """检测价格异常值"""
        history = list(self.price_history[symbol])
        if len(history) < 10:  # 样本不足
            return False, 0.0
        
        if method == "zscore":
            mean_price = statistics.mean(history)
            std_price = statistics.stdev(history)
            if std_price == 0:
                return False, 0.0
            zscore = abs((price - mean_price) / std_price)
            return zscore > 3.0, zscore
            
        elif method == "iqr":
            sorted_prices = sorted(history)
            n = len(sorted_prices)
            q1_idx = n // 4
            q3_idx = 3 * n // 4
            q1 = sorted_prices[q1_idx]
            q3 = sorted_prices[q3_idx]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            is_outlier = price < lower_bound or price > upper_bound
            deviation = max(abs(price - lower_bound), abs(price - upper_bound))
            return is_outlier, deviation
            
        return False, 0.0
    
    def detect_volume_anomaly(self, symbol: str, volume: float) -> Tuple[bool, float]:
        """检测交易量异常"""
        history = list(self.volume_history[symbol])
        if len(history) < 10:
            return False, 0.0
        
        median_volume = statistics.median(history)
        if median_volume == 0:
            return False, 0.0
        
        # 检测异常高交易量
        ratio = volume / median_volume
        return ratio > 50.0, ratio  # 超过中位数50倍视为异常
    
    def get_price_range(self, symbol: str, percentile: float = 0.95) -> Tuple[float, float]:
        """获取价格合理区间"""
        history = list(self.price_history[symbol])
        if len(history) < 10:
            return 0.0, float('inf')
        
        sorted_prices = sorted(history)
        n = len(sorted_prices)
        lower_idx = int(n * (1 - percentile) / 2)
        upper_idx = int(n * (1 + percentile) / 2)
        
        return sorted_prices[lower_idx], sorted_prices[upper_idx]

class BusinessRuleValidator:
    """业务规则验证器"""
    
    def __init__(self):
        self.exchange_trading_hours = {
            "binance": {"24/7": True},
            "okx": {"24/7": True}
        }
        
        self.min_prices = {
            "BTC/USDT": 1000.0,   # BTC最低合理价格
            "ETH/USDT": 100.0,    # ETH最低合理价格
        }
        
        self.max_prices = {
            "BTC/USDT": 200000.0,  # BTC最高合理价格
            "ETH/USDT": 50000.0,   # ETH最高合理价格
        }
    
    def validate_price_range(self, symbol: str, price: float) -> Tuple[bool, str]:
        """验证价格合理性"""
        min_price = self.min_prices.get(symbol, 0.0)
        max_price = self.max_prices.get(symbol, float('inf'))
        
        if price < min_price:
            return False, f"价格过低: {price} < {min_price}"
        if price > max_price:
            return False, f"价格过高: {price} > {max_price}"
        
        return True, "价格合理"
    
    def validate_kline_consistency(self, kline: Dict[str, float]) -> Tuple[bool, str]:
        """验证K线数据一致性"""
        open_price = kline.get("open", 0)
        high_price = kline.get("high", 0)
        low_price = kline.get("low", 0)
        close_price = kline.get("close", 0)
        volume = kline.get("volume", 0)
        
        # 基本数值检查
        if any(price <= 0 for price in [open_price, high_price, low_price, close_price]):
            return False, "K线价格存在非正数"
        
        if volume < 0:
            return False, "K线成交量为负数"
        
        # 高低价关系检查
        if high_price < max(open_price, close_price, low_price):
            return False, f"最高价{high_price}小于开盘价{open_price}或收盘价{close_price}或最低价{low_price}"
        
        if low_price > min(open_price, close_price, high_price):
            return False, f"最低价{low_price}大于开盘价{open_price}或收盘价{close_price}或最高价{high_price}"
        
        # 价格变动合理性检查（24小时内变动超过50%视为异常）
        price_change = abs(close_price - open_price) / open_price
        if price_change > 0.5:
            return False, f"单根K线价格变动过大: {price_change:.2%}"
        
        return True, "K线数据一致"
    
    def validate_timestamp(self, timestamp: Union[int, float], tolerance_seconds: int = 300) -> Tuple[bool, str]:
        """验证时间戳合理性"""
        try:
            if timestamp > 1e12:  # 毫秒时间戳
                dt = datetime.fromtimestamp(timestamp / 1000)
            else:  # 秒时间戳
                dt = datetime.fromtimestamp(timestamp)
            
            now = datetime.utcnow()
            time_diff = abs((now - dt).total_seconds())
            
            # 检查时间是否过于久远或未来
            if time_diff > 24 * 3600:  # 超过24小时
                return False, f"时间戳异常: 与当前时间相差{time_diff/3600:.1f}小时"
            
            return True, "时间戳正常"
            
        except (ValueError, OSError):
            return False, f"无效时间戳: {timestamp}"

class DataDuplicateDetector:
    """数据重复检测器"""
    
    def __init__(self, window_size: int = 1000):
        self.recent_data = defaultdict(lambda: deque(maxlen=window_size))
        
    def add_data(self, data_type: str, data_hash: str, timestamp: int):
        """添加数据指纹"""
        self.recent_data[data_type].append((data_hash, timestamp))
    
    def is_duplicate(self, data_type: str, data_hash: str, timestamp: int, 
                    tolerance_ms: int = 1000) -> bool:
        """检测数据是否重复"""
        recent = self.recent_data[data_type]
        
        for stored_hash, stored_timestamp in recent:
            if (stored_hash == data_hash and 
                abs(timestamp - stored_timestamp) <= tolerance_ms):
                return True
        
        return False
    
    def generate_data_hash(self, data: Dict[str, Any], key_fields: List[str] = None) -> str:
        """生成数据哈希值"""
        if key_fields:
            hash_data = {k: data.get(k) for k in key_fields if k in data}
        else:
            hash_data = data
        
        # 简单哈希实现
        hash_str = str(sorted(hash_data.items()))
        return str(hash(hash_str))

class IntelligentDataCleaner:
    """智能数据清洗引擎"""
    
    def __init__(self):
        self.statistical_analyzer = StatisticalAnalyzer()
        self.business_validator = BusinessRuleValidator()
        self.duplicate_detector = DataDuplicateDetector()
        
        # 清洗规则配置
        self.cleaning_rules = self._initialize_cleaning_rules()
        
        # 统计信息
        self.cleaning_stats = {
            "total_processed": 0,
            "total_cleaned": 0,
            "total_rejected": 0,
            "rule_hits": defaultdict(int)
        }
    
    def _initialize_cleaning_rules(self) -> List[CleaningRule]:
        """初始化清洗规则"""
        return [
            # 价格异常检测
            CleaningRule(
                rule_id="price_outlier_zscore",
                rule_type="outlier",
                severity="high",
                description="基于Z-Score的价格异常检测",
                auto_fix=False,
                threshold=3.0
            ),
            
            # 交易量异常检测
            CleaningRule(
                rule_id="volume_anomaly",
                rule_type="outlier",
                severity="medium",
                description="交易量异常检测",
                auto_fix=False,
                threshold=50.0
            ),
            
            # K线一致性验证
            CleaningRule(
                rule_id="kline_consistency",
                rule_type="business",
                severity="high",
                description="K线OHLC一致性验证",
                auto_fix=False
            ),
            
            # 重复数据检测
            CleaningRule(
                rule_id="duplicate_detection",
                rule_type="duplicate",
                severity="medium",
                description="重复数据检测和去除",
                auto_fix=True,
                parameters={"tolerance_ms": 1000}
            ),
            
            # 时间戳验证
            CleaningRule(
                rule_id="timestamp_validation",
                rule_type="format",
                severity="critical",
                description="时间戳合理性验证",
                auto_fix=False
            ),
            
            # 价格合理区间验证
            CleaningRule(
                rule_id="price_range_validation",
                rule_type="business", 
                severity="critical",
                description="价格合理区间验证",
                auto_fix=False
            )
        ]
    
    async def clean_market_data(self, data_batch: List[Dict[str, Any]]) -> CleaningResult:
        """清洗市场数据"""
        start_time = datetime.utcnow()
        original_count = len(data_batch)
        cleaned_data = []
        issues_detected = []
        removed_count = 0
        modified_count = 0
        
        for data in data_batch:
            try:
                is_valid, cleaning_issues, modified_data = await self._clean_single_market_data(data)
                
                if is_valid:
                    cleaned_data.append(modified_data or data)
                    if modified_data:
                        modified_count += 1
                else:
                    removed_count += 1
                
                issues_detected.extend(cleaning_issues)
                
            except Exception as e:
                logger.error(f"清洗市场数据失败: {e}")
                removed_count += 1
                issues_detected.append({
                    "rule_id": "processing_error",
                    "severity": "critical",
                    "description": f"数据处理异常: {str(e)}",
                    "data": data
                })
        
        # 计算质量评分
        quality_score = self._calculate_quality_score(
            original_count, len(cleaned_data), len(issues_detected)
        )
        
        # 更新统计
        self.cleaning_stats["total_processed"] += original_count
        self.cleaning_stats["total_cleaned"] += len(cleaned_data)
        self.cleaning_stats["total_rejected"] += removed_count
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return CleaningResult(
            original_count=original_count,
            cleaned_count=len(cleaned_data),
            removed_count=removed_count,
            modified_count=modified_count,
            issues_detected=issues_detected,
            quality_score=quality_score,
            processing_time=processing_time,
            rules_applied=[rule.rule_id for rule in self.cleaning_rules if rule.enabled]
        )
    
    async def _clean_single_market_data(self, data: Dict[str, Any]) -> Tuple[bool, List[Dict], Optional[Dict]]:
        """清洗单条市场数据"""
        issues = []
        modified_data = None
        is_valid = True
        
        symbol = data.get("symbol", "")
        price = float(data.get("price", 0))
        volume = float(data.get("volume_24h", 0))
        timestamp = data.get("timestamp", 0)
        
        # 1. 时间戳验证
        timestamp_valid, timestamp_msg = self.business_validator.validate_timestamp(timestamp)
        if not timestamp_valid:
            issues.append({
                "rule_id": "timestamp_validation",
                "severity": "critical",
                "description": timestamp_msg,
                "data": data
            })
            is_valid = False
        
        # 2. 价格范围验证
        price_valid, price_msg = self.business_validator.validate_price_range(symbol, price)
        if not price_valid:
            issues.append({
                "rule_id": "price_range_validation",
                "severity": "critical", 
                "description": price_msg,
                "data": data
            })
            is_valid = False
        
        # 3. 重复检测
        if is_valid:
            data_hash = self.duplicate_detector.generate_data_hash(
                data, ["symbol", "price", "timestamp"]
            )
            
            if self.duplicate_detector.is_duplicate("market_data", data_hash, timestamp):
                issues.append({
                    "rule_id": "duplicate_detection",
                    "severity": "medium",
                    "description": "检测到重复数据",
                    "data": data
                })
                # 重复数据自动过滤
                is_valid = False
            else:
                self.duplicate_detector.add_data("market_data", data_hash, timestamp)
        
        # 4. 统计异常检测
        if is_valid and symbol and price > 0:
            # 更新统计历史
            self.statistical_analyzer.update_statistics(symbol, price, volume)
            
            # 价格异常检测
            is_outlier, score = self.statistical_analyzer.detect_price_outliers(symbol, price)
            if is_outlier:
                issues.append({
                    "rule_id": "price_outlier_zscore",
                    "severity": "high",
                    "description": f"价格异常值检测，Z-Score: {score:.2f}",
                    "data": data
                })
                # 高严重性异常不自动修复，直接过滤
                if score > 5.0:
                    is_valid = False
            
            # 交易量异常检测
            volume_anomaly, volume_ratio = self.statistical_analyzer.detect_volume_anomaly(symbol, volume)
            if volume_anomaly:
                issues.append({
                    "rule_id": "volume_anomaly",
                    "severity": "medium",
                    "description": f"交易量异常，比例: {volume_ratio:.2f}x",
                    "data": data
                })
        
        # 更新规则命中统计
        for issue in issues:
            self.cleaning_stats["rule_hits"][issue["rule_id"]] += 1
        
        return is_valid, issues, modified_data
    
    async def clean_candle_data(self, data_batch: List[Dict[str, Any]]) -> CleaningResult:
        """清洗K线数据"""
        start_time = datetime.utcnow()
        original_count = len(data_batch)
        cleaned_data = []
        issues_detected = []
        removed_count = 0
        modified_count = 0
        
        for data in data_batch:
            try:
                is_valid, cleaning_issues, modified_data = await self._clean_single_candle_data(data)
                
                if is_valid:
                    cleaned_data.append(modified_data or data)
                    if modified_data:
                        modified_count += 1
                else:
                    removed_count += 1
                
                issues_detected.extend(cleaning_issues)
                
            except Exception as e:
                logger.error(f"清洗K线数据失败: {e}")
                removed_count += 1
        
        quality_score = self._calculate_quality_score(
            original_count, len(cleaned_data), len(issues_detected)
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return CleaningResult(
            original_count=original_count,
            cleaned_count=len(cleaned_data),
            removed_count=removed_count,
            modified_count=modified_count,
            issues_detected=issues_detected,
            quality_score=quality_score,
            processing_time=processing_time,
            rules_applied=[rule.rule_id for rule in self.cleaning_rules if rule.enabled]
        )
    
    async def _clean_single_candle_data(self, data: Dict[str, Any]) -> Tuple[bool, List[Dict], Optional[Dict]]:
        """清洗单条K线数据"""
        issues = []
        is_valid = True
        
        # K线一致性验证
        consistency_valid, consistency_msg = self.business_validator.validate_kline_consistency(data)
        if not consistency_valid:
            issues.append({
                "rule_id": "kline_consistency",
                "severity": "high",
                "description": consistency_msg,
                "data": data
            })
            is_valid = False
        
        # 时间戳验证
        timestamp = data.get("timestamp", 0)
        timestamp_valid, timestamp_msg = self.business_validator.validate_timestamp(timestamp)
        if not timestamp_valid:
            issues.append({
                "rule_id": "timestamp_validation",
                "severity": "critical",
                "description": timestamp_msg,
                "data": data
            })
            is_valid = False
        
        # 重复检测
        if is_valid:
            data_hash = self.duplicate_detector.generate_data_hash(
                data, ["symbol", "timeframe", "timestamp", "close"]
            )
            
            if self.duplicate_detector.is_duplicate("candle_data", data_hash, timestamp):
                issues.append({
                    "rule_id": "duplicate_detection", 
                    "severity": "medium",
                    "description": "检测到重复K线数据",
                    "data": data
                })
                is_valid = False
            else:
                self.duplicate_detector.add_data("candle_data", data_hash, timestamp)
        
        return is_valid, issues, None
    
    def _calculate_quality_score(self, original: int, cleaned: int, issues: int) -> float:
        """计算数据质量评分"""
        if original == 0:
            return 100.0
        
        # 基础质量分 = (清洗后数据量 / 原始数据量) * 100
        base_score = (cleaned / original) * 100
        
        # 问题惩罚分 = (问题数量 / 原始数据量) * 50
        issue_penalty = min((issues / original) * 50, 50)
        
        # 最终得分
        final_score = max(base_score - issue_penalty, 0.0)
        
        return round(final_score, 2)
    
    def get_cleaning_stats(self) -> Dict[str, Any]:
        """获取清洗统计信息"""
        total_processed = self.cleaning_stats["total_processed"]
        total_cleaned = self.cleaning_stats["total_cleaned"]
        total_rejected = self.cleaning_stats["total_rejected"]
        
        return {
            "total_processed": total_processed,
            "total_cleaned": total_cleaned,
            "total_rejected": total_rejected,
            "cleaning_rate": (total_cleaned / total_processed * 100) if total_processed > 0 else 0,
            "rejection_rate": (total_rejected / total_processed * 100) if total_processed > 0 else 0,
            "rule_hits": dict(self.cleaning_stats["rule_hits"]),
            "active_rules": len([r for r in self.cleaning_rules if r.enabled])
        }
    
    def update_rule_config(self, rule_id: str, **kwargs):
        """更新清洗规则配置"""
        for rule in self.cleaning_rules:
            if rule.rule_id == rule_id:
                for key, value in kwargs.items():
                    if hasattr(rule, key):
                        setattr(rule, key, value)
                logger.info(f"更新清洗规则 {rule_id}: {kwargs}")
                break

# 全局智能数据清洗引擎实例
intelligent_cleaner = IntelligentDataCleaner()