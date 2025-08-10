"""
延迟模拟器

模拟真实交易中的各种延迟：
- 网络延迟
- 订单处理延迟  
- 系统延迟
- 队列延迟
- 随机延迟
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import random
import numpy as np
from collections import deque

from ..backtest.trading_environment import Order
from core.unified_logger import get_logger, LogCategory

logger = get_logger()

class LatencyType(Enum):
    """延迟类型"""
    NETWORK = "network"         # 网络延迟
    PROCESSING = "processing"   # 处理延迟
    QUEUE = "queue"            # 队列延迟
    SYSTEM = "system"          # 系统延迟
    COMBINED = "combined"      # 组合延迟

@dataclass
class LatencyDistribution:
    """延迟分布参数"""
    mean_ms: float = 50.0      # 平均延迟(毫秒)
    std_ms: float = 20.0       # 标准差(毫秒)
    min_ms: float = 5.0        # 最小延迟(毫秒)
    max_ms: float = 500.0      # 最大延迟(毫秒)
    percentile_99_ms: float = 200.0  # 99分位延迟

@dataclass
class LatencyProfile:
    """延迟配置文件"""
    network_latency: LatencyDistribution
    processing_latency: LatencyDistribution  
    queue_latency: LatencyDistribution
    system_latency: LatencyDistribution
    
    # 延迟相关性参数
    correlation_factor: float = 0.3  # 不同类型延迟的相关性
    burst_probability: float = 0.05  # 突发延迟概率
    burst_multiplier: float = 3.0    # 突发延迟倍数

@dataclass
class LatencyResult:
    """延迟结果"""
    total_latency_ms: float
    network_ms: float
    processing_ms: float
    queue_ms: float
    system_ms: float
    is_burst: bool
    timestamp: datetime

class LatencySimulator:
    """延迟模拟器主类"""
    
    def __init__(self, latency_profile: Optional[LatencyProfile] = None):
        # 使用默认配置或自定义配置
        self.latency_profile = latency_profile or self._create_default_profile()
        
        # 延迟历史记录
        self.latency_history: deque = deque(maxlen=10000)
        
        # 当前网络状况
        self.network_condition = 1.0  # 1.0 = 正常，>1.0 = 网络拥堵
        self.system_load = 1.0        # 1.0 = 正常负载，>1.0 = 高负载
        
        # 队列状态
        self.current_queue_length = 0
        self.max_queue_length = 1000
        
        # 突发延迟状态
        self.burst_mode = False
        self.burst_end_time: Optional[datetime] = None
        
        # 统计数据
        self.total_requests = 0
        self.burst_count = 0
        
        logger.info(f"延迟模拟器初始化完成")
    
    def _create_default_profile(self) -> LatencyProfile:
        """创建默认延迟配置"""
        return LatencyProfile(
            network_latency=LatencyDistribution(
                mean_ms=30.0, std_ms=15.0, min_ms=5.0, max_ms=300.0, percentile_99_ms=150.0
            ),
            processing_latency=LatencyDistribution(
                mean_ms=10.0, std_ms=5.0, min_ms=1.0, max_ms=100.0, percentile_99_ms=50.0
            ),
            queue_latency=LatencyDistribution(
                mean_ms=5.0, std_ms=10.0, min_ms=0.0, max_ms=200.0, percentile_99_ms=80.0
            ),
            system_latency=LatencyDistribution(
                mean_ms=5.0, std_ms=3.0, min_ms=1.0, max_ms=50.0, percentile_99_ms=25.0
            ),
            correlation_factor=0.3,
            burst_probability=0.05,
            burst_multiplier=3.0
        )
    
    async def simulate_latency(self, order: Order, 
                             latency_type: LatencyType = LatencyType.COMBINED) -> LatencyResult:
        """模拟延迟"""
        try:
            self.total_requests += 1
            current_time = datetime.utcnow()
            
            # 检查突发延迟状态
            await self._check_burst_mode()
            
            # 根据延迟类型计算延迟
            if latency_type == LatencyType.NETWORK:
                latencies = await self._simulate_network_latency()
            elif latency_type == LatencyType.PROCESSING:
                latencies = await self._simulate_processing_latency(order)
            elif latency_type == LatencyType.QUEUE:
                latencies = await self._simulate_queue_latency()
            elif latency_type == LatencyType.SYSTEM:
                latencies = await self._simulate_system_latency()
            else:  # COMBINED
                latencies = await self._simulate_combined_latency(order)
            
            # 应用突发延迟
            if self.burst_mode:
                latencies = {k: v * self.latency_profile.burst_multiplier for k, v in latencies.items()}
                is_burst = True
                self.burst_count += 1
            else:
                is_burst = False
            
            # 计算总延迟
            total_latency = sum(latencies.values())
            
            # 创建结果
            result = LatencyResult(
                total_latency_ms=total_latency,
                network_ms=latencies.get('network', 0),
                processing_ms=latencies.get('processing', 0),
                queue_ms=latencies.get('queue', 0),
                system_ms=latencies.get('system', 0),
                is_burst=is_burst,
                timestamp=current_time
            )
            
            # 记录历史
            self.latency_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"模拟延迟失败: {e}")
            # 返回默认延迟
            return LatencyResult(
                total_latency_ms=50.0,
                network_ms=30.0,
                processing_ms=10.0,
                queue_ms=5.0,
                system_ms=5.0,
                is_burst=False,
                timestamp=datetime.utcnow()
            )
    
    async def _simulate_network_latency(self) -> Dict[str, float]:
        """模拟网络延迟"""
        try:
            dist = self.latency_profile.network_latency
            
            # 基础网络延迟
            base_latency = np.random.normal(dist.mean_ms, dist.std_ms)
            base_latency = max(dist.min_ms, min(base_latency, dist.max_ms))
            
            # 应用网络状况
            network_latency = base_latency * self.network_condition
            
            # 偶发的网络抖动
            if random.random() < 0.02:  # 2%概率
                jitter = random.uniform(10, 100)
                network_latency += jitter
            
            return {'network': network_latency}
            
        except Exception as e:
            logger.error(f"模拟网络延迟失败: {e}")
            return {'network': 30.0}
    
    async def _simulate_processing_latency(self, order: Order) -> Dict[str, float]:
        """模拟订单处理延迟"""
        try:
            dist = self.latency_profile.processing_latency
            
            # 基础处理延迟
            base_latency = np.random.normal(dist.mean_ms, dist.std_ms)
            base_latency = max(dist.min_ms, min(base_latency, dist.max_ms))
            
            # 订单类型影响
            if order.order_type.value == "market":
                type_multiplier = 0.8  # 市价订单处理较快
            elif order.order_type.value == "limit":
                type_multiplier = 1.0  # 限价订单正常处理
            else:
                type_multiplier = 1.2  # 复杂订单类型处理较慢
            
            # 订单大小影响
            order_value = order.quantity * (order.price or 1000)
            if order_value > 100000:  # 大订单
                size_multiplier = 1.5
            elif order_value > 10000:  # 中等订单
                size_multiplier = 1.2
            else:  # 小订单
                size_multiplier = 1.0
            
            processing_latency = base_latency * type_multiplier * size_multiplier
            
            return {'processing': processing_latency}
            
        except Exception as e:
            logger.error(f"模拟处理延迟失败: {e}")
            return {'processing': 10.0}
    
    async def _simulate_queue_latency(self) -> Dict[str, float]:
        """模拟队列延迟"""
        try:
            dist = self.latency_profile.queue_latency
            
            # 基于当前队列长度的延迟
            queue_factor = self.current_queue_length / self.max_queue_length
            queue_base_latency = dist.mean_ms * (1 + queue_factor * 5)
            
            # 添加随机性
            queue_latency = np.random.normal(queue_base_latency, dist.std_ms)
            queue_latency = max(dist.min_ms, min(queue_latency, dist.max_ms))
            
            # 更新队列长度 (简化模拟)
            self.current_queue_length = max(0, self.current_queue_length + random.randint(-5, 3))
            self.current_queue_length = min(self.current_queue_length, self.max_queue_length)
            
            return {'queue': queue_latency}
            
        except Exception as e:
            logger.error(f"模拟队列延迟失败: {e}")
            return {'queue': 5.0}
    
    async def _simulate_system_latency(self) -> Dict[str, float]:
        """模拟系统延迟"""
        try:
            dist = self.latency_profile.system_latency
            
            # 基础系统延迟
            base_latency = np.random.normal(dist.mean_ms, dist.std_ms)
            base_latency = max(dist.min_ms, min(base_latency, dist.max_ms))
            
            # 应用系统负载
            system_latency = base_latency * self.system_load
            
            # CPU密集型操作的额外延迟
            if random.random() < 0.05:  # 5%概率
                cpu_penalty = random.uniform(5, 20)
                system_latency += cpu_penalty
            
            return {'system': system_latency}
            
        except Exception as e:
            logger.error(f"模拟系统延迟失败: {e}")
            return {'system': 5.0}
    
    async def _simulate_combined_latency(self, order: Order) -> Dict[str, float]:
        """模拟组合延迟"""
        try:
            # 获取各组件延迟
            network_latencies = await self._simulate_network_latency()
            processing_latencies = await self._simulate_processing_latency(order)
            queue_latencies = await self._simulate_queue_latency()
            system_latencies = await self._simulate_system_latency()
            
            # 合并延迟
            latencies = {
                **network_latencies,
                **processing_latencies, 
                **queue_latencies,
                **system_latencies
            }
            
            # 应用相关性 - 当一个组件延迟高时，其他组件延迟也可能增加
            if self.latency_profile.correlation_factor > 0:
                max_latency = max(latencies.values())
                avg_latency = sum(latencies.values()) / len(latencies)
                
                if max_latency > avg_latency * 2:  # 检测到异常高延迟
                    correlation_boost = (max_latency - avg_latency) * self.latency_profile.correlation_factor
                    
                    for key in latencies:
                        if latencies[key] != max_latency:  # 不对已经最高的延迟再加成
                            latencies[key] += correlation_boost * random.uniform(0.3, 0.7)
            
            return latencies
            
        except Exception as e:
            logger.error(f"模拟组合延迟失败: {e}")
            return {'network': 30.0, 'processing': 10.0, 'queue': 5.0, 'system': 5.0}
    
    async def _check_burst_mode(self) -> None:
        """检查突发延迟模式"""
        try:
            current_time = datetime.utcnow()
            
            # 检查是否应该结束突发模式
            if self.burst_mode and self.burst_end_time and current_time > self.burst_end_time:
                self.burst_mode = False
                self.burst_end_time = None
                logger.info("突发延迟模式结束")
            
            # 检查是否应该开始突发模式
            if not self.burst_mode and random.random() < self.latency_profile.burst_probability:
                self.burst_mode = True
                # 突发延迟持续5-30秒
                burst_duration = random.uniform(5, 30)
                self.burst_end_time = current_time + timedelta(seconds=burst_duration)
                logger.warning(f"突发延迟模式开始，持续{burst_duration:.1f}秒")
            
        except Exception as e:
            logger.error(f"检查突发模式失败: {e}")
    
    async def apply_latency(self, latency_result: LatencyResult) -> None:
        """应用延迟 - 实际等待指定的时间"""
        try:
            if latency_result.total_latency_ms > 0:
                await asyncio.sleep(latency_result.total_latency_ms / 1000.0)
                
        except Exception as e:
            logger.error(f"应用延迟失败: {e}")
    
    def update_network_condition(self, condition: float) -> None:
        """更新网络状况"""
        self.network_condition = max(0.1, condition)  # 最低10%性能
        if condition > 1.5:
            logger.warning(f"网络状况恶化: {condition:.2f}x正常延迟")
        elif condition < 1.0:
            logger.info(f"网络状况良好: {condition:.2f}x正常延迟")
    
    def update_system_load(self, load: float) -> None:
        """更新系统负载"""
        self.system_load = max(0.1, load)
        if load > 1.5:
            logger.warning(f"系统负载较高: {load:.2f}x正常延迟")
        elif load < 1.0:
            logger.info(f"系统负载正常: {load:.2f}x正常延迟")
    
    def get_latency_statistics(self, window_minutes: int = 60) -> Dict[str, Any]:
        """获取延迟统计"""
        try:
            if not self.latency_history:
                return {}
            
            # 获取指定时间窗口内的数据
            cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
            recent_latencies = [
                l for l in self.latency_history 
                if l.timestamp > cutoff_time
            ]
            
            if not recent_latencies:
                return {}
            
            # 计算统计数据
            total_latencies = [l.total_latency_ms for l in recent_latencies]
            network_latencies = [l.network_ms for l in recent_latencies]
            processing_latencies = [l.processing_ms for l in recent_latencies]
            queue_latencies = [l.queue_ms for l in recent_latencies]
            system_latencies = [l.system_ms for l in recent_latencies]
            
            burst_count = sum(1 for l in recent_latencies if l.is_burst)
            
            return {
                'window_minutes': window_minutes,
                'total_requests': len(recent_latencies),
                'burst_requests': burst_count,
                'burst_rate': burst_count / len(recent_latencies),
                'total_latency': {
                    'mean_ms': np.mean(total_latencies),
                    'median_ms': np.median(total_latencies),
                    'std_ms': np.std(total_latencies),
                    'min_ms': np.min(total_latencies),
                    'max_ms': np.max(total_latencies),
                    'p95_ms': np.percentile(total_latencies, 95),
                    'p99_ms': np.percentile(total_latencies, 99)
                },
                'network_latency': {
                    'mean_ms': np.mean(network_latencies),
                    'p95_ms': np.percentile(network_latencies, 95)
                },
                'processing_latency': {
                    'mean_ms': np.mean(processing_latencies),
                    'p95_ms': np.percentile(processing_latencies, 95)
                },
                'queue_latency': {
                    'mean_ms': np.mean(queue_latencies),
                    'p95_ms': np.percentile(queue_latencies, 95)
                },
                'system_latency': {
                    'mean_ms': np.mean(system_latencies),
                    'p95_ms': np.percentile(system_latencies, 95)
                },
                'current_status': {
                    'network_condition': self.network_condition,
                    'system_load': self.system_load,
                    'queue_length': self.current_queue_length,
                    'burst_mode': self.burst_mode
                }
            }
            
        except Exception as e:
            logger.error(f"获取延迟统计失败: {e}")
            return {}
    
    def set_latency_profile(self, profile: LatencyProfile) -> None:
        """设置延迟配置"""
        self.latency_profile = profile
        logger.info("延迟配置已更新")
    
    def get_current_profile(self) -> LatencyProfile:
        """获取当前延迟配置"""
        return self.latency_profile
    
    def reset_statistics(self) -> None:
        """重置统计数据"""
        self.latency_history.clear()
        self.total_requests = 0
        self.burst_count = 0
        logger.info("延迟统计数据已重置")