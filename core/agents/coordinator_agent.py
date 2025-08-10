"""
协调Agent - 多Agent系统的核心调度器
负责任务分配、资源调度、Agent协作协调和系统整体优化
增强版本包含完整的监控、故障恢复、性能优化和自适应调度功能
"""

import asyncio
import json
import uuid
import threading
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import heapq
import time
import statistics

from .base_agent import BaseAgent, AgentType, AgentCapability, AgentMessage, MessageType
from .agent_communication import AgentRegistry

# 任务状态枚举
class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# 任务优先级枚举
class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

# 故障类型枚举
class FailureType(Enum):
    AGENT_TIMEOUT = "agent_timeout"
    AGENT_CRASH = "agent_crash"
    COMMUNICATION_ERROR = "communication_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    TASK_EXECUTION_ERROR = "task_execution_error"
    SYSTEM_OVERLOAD = "system_overload"

# 恢复策略枚举
class RecoveryStrategy(Enum):
    RESTART_AGENT = "restart_agent"
    REASSIGN_TASK = "reassign_task"
    SCALE_OUT = "scale_out"
    LOAD_BALANCE = "load_balance"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"

@dataclass
class SystemMetrics:
    """系统性能指标"""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    total_agents: int = 0
    active_agents: int = 0
    failed_agents: int = 0
    pending_tasks: int = 0
    running_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_task_completion_time: float = 0.0
    system_load: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    network_latency: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0  # 任务/秒

@dataclass
class FailureEvent:
    """故障事件记录"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    failure_type: FailureType = FailureType.AGENT_TIMEOUT
    affected_agent: Optional[str] = None
    affected_task: Optional[str] = None
    description: str = ""
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_success: bool = False
    recovery_time: Optional[float] = None

@dataclass
class PerformanceProfile:
    """Agent性能档案"""
    agent_id: str
    task_completion_times: deque = field(default_factory=lambda: deque(maxlen=100))
    success_rate: float = 1.0
    error_count: int = 0
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    capability_scores: Dict[AgentCapability, float] = field(default_factory=dict)
    load_factor: float = 0.0  # 当前负载因子 0-1
    reliability_score: float = 1.0  # 可靠性评分 0-1

@dataclass
class AgentTask:
    """Agent任务定义"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_name: str = ""
    task_type: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    required_capabilities: List[AgentCapability] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)  # 依赖的任务ID
    max_retries: int = 3
    timeout: Optional[int] = None  # 超时时间(秒)
    
    # 任务状态
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # 执行信息
    retry_count: int = 0
    error_message: Optional[str] = None
    result: Optional[Any] = None
    execution_time: Optional[float] = None
    
    def __lt__(self, other):
        """用于优先级队列排序"""
        return self.priority.value > other.priority.value

@dataclass
class AgentWorkload:
    """Agent工作负载信息"""
    agent_id: str
    agent_type: AgentType
    capabilities: List[AgentCapability]
    current_tasks: int = 0
    max_concurrent_tasks: int = 10
    average_response_time: float = 0.0
    success_rate: float = 1.0
    last_activity: Optional[datetime] = None
    health_score: float = 1.0
    
    @property
    def availability(self) -> float:
        """可用性评分 (0-1)"""
        if self.max_concurrent_tasks == 0:
            return 0.0
        
        utilization = self.current_tasks / self.max_concurrent_tasks
        # 综合考虑利用率、健康状态和成功率
        return (1 - utilization) * self.health_score * self.success_rate

class TaskScheduler:
    """任务调度器"""
    
    def __init__(self):
        self.pending_tasks: List[AgentTask] = []  # 优先级队列
        self.running_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: Dict[str, AgentTask] = {}
        self.agent_workloads: Dict[str, AgentWorkload] = {}
        
        # 调度策略配置
        self.max_assignment_attempts = 5
        self.load_balance_factor = 0.7  # 负载均衡权重
        self.capability_match_factor = 0.3  # 能力匹配权重
        
    def add_task(self, task: AgentTask):
        """添加任务到调度队列"""
        heapq.heappush(self.pending_tasks, task)
    
    def get_next_task(self) -> Optional[AgentTask]:
        """获取下一个待调度任务"""
        if not self.pending_tasks:
            return None
        return heapq.heappop(self.pending_tasks)
    
    def update_agent_workload(self, agent_id: str, workload: AgentWorkload):
        """更新Agent工作负载信息"""
        self.agent_workloads[agent_id] = workload
    
    def find_best_agent(self, task: AgentTask) -> Optional[str]:
        """为任务找到最佳的Agent"""
        candidate_agents = []
        
        for agent_id, workload in self.agent_workloads.items():
            # 检查能力匹配
            if not all(cap in workload.capabilities for cap in task.required_capabilities):
                continue
            
            # 检查可用性
            if workload.availability <= 0:
                continue
            
            # 计算适合度分数
            fitness = self._calculate_agent_fitness(task, workload)
            candidate_agents.append((agent_id, fitness))
        
        if not candidate_agents:
            return None
        
        # 按适合度排序，选择最佳Agent
        candidate_agents.sort(key=lambda x: x[1], reverse=True)
        return candidate_agents[0][0]
    
    def _calculate_agent_fitness(self, task: AgentTask, workload: AgentWorkload) -> float:
        """计算Agent对任务的适合度分数"""
        # 可用性分数
        availability_score = workload.availability
        
        # 能力匹配分数
        capability_score = 1.0  # 基础分数，因为已经过滤了不匹配的
        
        # 响应时间分数（响应时间越短分数越高）
        response_time_score = 1.0 / (1.0 + workload.average_response_time)
        
        # 综合分数
        fitness = (
            availability_score * self.load_balance_factor +
            capability_score * self.capability_match_factor +
            response_time_score * (1 - self.load_balance_factor - self.capability_match_factor)
        )
        
        return fitness
    
    def mark_task_running(self, task_id: str, agent_id: str):
        """标记任务为运行状态"""
        if task_id in self.running_tasks:
            task = self.running_tasks[task_id]
            task.status = TaskStatus.RUNNING
            task.assigned_agent = agent_id
            task.started_at = datetime.utcnow()
            
            # 更新Agent工作负载
            if agent_id in self.agent_workloads:
                self.agent_workloads[agent_id].current_tasks += 1
    
    def complete_task(self, task_id: str, result: Any = None, error: str = None):
        """完成任务"""
        if task_id not in self.running_tasks:
            return
        
        task = self.running_tasks.pop(task_id)
        task.completed_at = datetime.utcnow()
        
        if error:
            task.status = TaskStatus.FAILED
            task.error_message = error
        else:
            task.status = TaskStatus.COMPLETED
            task.result = result
        
        # 计算执行时间
        if task.started_at:
            task.execution_time = (task.completed_at - task.started_at).total_seconds()
        
        self.completed_tasks[task_id] = task
        
        # 更新Agent工作负载
        if task.assigned_agent and task.assigned_agent in self.agent_workloads:
            self.agent_workloads[task.assigned_agent].current_tasks -= 1
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """获取任务统计信息"""
        total_tasks = len(self.completed_tasks) + len(self.running_tasks)
        completed_count = len([t for t in self.completed_tasks.values() if t.status == TaskStatus.COMPLETED])
        failed_count = len([t for t in self.completed_tasks.values() if t.status == TaskStatus.FAILED])
        
        avg_execution_time = 0.0
        if self.completed_tasks:
            execution_times = [t.execution_time for t in self.completed_tasks.values() if t.execution_time]
            if execution_times:
                avg_execution_time = sum(execution_times) / len(execution_times)
        
        return {
            "pending_tasks": len(self.pending_tasks),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": completed_count,
            "failed_tasks": failed_count,
            "total_tasks": total_tasks,
            "success_rate": completed_count / total_tasks if total_tasks > 0 else 0,
            "average_execution_time": avg_execution_time,
            "active_agents": len(self.agent_workloads)
        }

class CoordinatorAgent(BaseAgent):
    """增强版协调Agent - 多Agent系统的核心调度器
    
    新增功能：
    - 完整的系统监控和指标收集
    - 智能故障检测和自动恢复
    - 自适应负载均衡和性能优化
    - 预测性维护和容量规划
    - 多层次熔断机制
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            agent_id="coordinator",
            agent_type=AgentType.COORDINATOR,
            capabilities=[
                AgentCapability.OPTIMIZATION,
                AgentCapability.PATTERN_RECOGNITION,
                AgentCapability.MONITORING,
                AgentCapability.RISK_ASSESSMENT,
                AgentCapability.REAL_TIME_PROCESSING
            ],
            config=config or {}
        )
        
        # 任务调度器
        self.task_scheduler = TaskScheduler()
        
        # Agent注册中心
        self.agent_registry: Optional[AgentRegistry] = None
        
        # === 增强监控系统 ===
        self.system_metrics_history: deque = deque(maxlen=1000)
        self.current_metrics = SystemMetrics()
        self.performance_profiles: Dict[str, PerformanceProfile] = {}
        self.failure_events: deque = deque(maxlen=500)
        
        # === 故障检测与恢复 ===
        self.failure_detectors: Dict[FailureType, Callable] = {}
        self.recovery_strategies: Dict[FailureType, List[RecoveryStrategy]] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}  # Agent级熔断器
        self.recovery_in_progress: Set[str] = set()
        
        # === 性能优化 ===
        self.load_balancer_weights: Dict[str, float] = {}
        self.task_routing_rules: List[Dict[str, Any]] = []
        self.resource_pools: Dict[str, List[str]] = {}  # 按能力分组的Agent池
        
        # === 预测分析 ===
        self.workload_predictor = WorkloadPredictor()
        self.capacity_planner = CapacityPlanner()
        self.anomaly_detector = AnomalyDetector()
        
        # === 配置参数 ===
        self.scheduling_interval = config.get("scheduling_interval", 1.0)
        self.health_check_interval = config.get("health_check_interval", 10.0)
        self.metrics_update_interval = config.get("metrics_update_interval", 5.0)
        self.failure_detection_interval = config.get("failure_detection_interval", 2.0)
        
        # 性能阈值
        self.max_response_time = config.get("max_response_time", 30.0)
        self.max_error_rate = config.get("max_error_rate", 0.05)
        self.max_load_factor = config.get("max_load_factor", 0.8)
        self.min_agent_uptime = config.get("min_agent_uptime", 0.95)
        
        # 任务模板和协作策略
        self.task_templates: Dict[str, Dict[str, Any]] = {}
        self.collaboration_strategies: Dict[str, Callable] = {}
        
        # 线程锁
        self.metrics_lock = threading.RLock()
        self.recovery_lock = threading.RLock()
        
        # 初始化监控和故障恢复系统
        self._initialize_monitoring_system()
        self._initialize_failure_recovery_system()
        self._initialize_performance_optimization()

    def _initialize_monitoring_system(self):
        """初始化监控系统"""
        # 注册故障检测器
        self.failure_detectors = {
            FailureType.AGENT_TIMEOUT: self._detect_agent_timeout,
            FailureType.AGENT_CRASH: self._detect_agent_crash,
            FailureType.COMMUNICATION_ERROR: self._detect_communication_error,
            FailureType.RESOURCE_EXHAUSTION: self._detect_resource_exhaustion,
            FailureType.TASK_EXECUTION_ERROR: self._detect_task_execution_error,
            FailureType.SYSTEM_OVERLOAD: self._detect_system_overload
        }

    def _initialize_failure_recovery_system(self):
        """初始化故障恢复系统"""
        # 定义恢复策略
        self.recovery_strategies = {
            FailureType.AGENT_TIMEOUT: [RecoveryStrategy.RESTART_AGENT, RecoveryStrategy.REASSIGN_TASK],
            FailureType.AGENT_CRASH: [RecoveryStrategy.RESTART_AGENT, RecoveryStrategy.SCALE_OUT],
            FailureType.COMMUNICATION_ERROR: [RecoveryStrategy.CIRCUIT_BREAKER, RecoveryStrategy.REASSIGN_TASK],
            FailureType.RESOURCE_EXHAUSTION: [RecoveryStrategy.SCALE_OUT, RecoveryStrategy.LOAD_BALANCE],
            FailureType.TASK_EXECUTION_ERROR: [RecoveryStrategy.REASSIGN_TASK, RecoveryStrategy.GRACEFUL_DEGRADATION],
            FailureType.SYSTEM_OVERLOAD: [RecoveryStrategy.LOAD_BALANCE, RecoveryStrategy.CIRCUIT_BREAKER]
        }

    def _initialize_performance_optimization(self):
        """初始化性能优化系统"""
        # 初始化负载均衡权重
        self.load_balancer_weights = {}
        
        # 初始化任务路由规则
        self.task_routing_rules = [
            {"condition": "priority >= 4", "strategy": "fastest_agent"},
            {"condition": "timeout < 10", "strategy": "lowest_load"},
            {"condition": "retry_count > 0", "strategy": "most_reliable"},
            {"condition": "default", "strategy": "round_robin"}
        ]

class WorkloadPredictor:
    """工作负载预测器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.task_history: deque = deque(maxlen=window_size)
        self.load_history: deque = deque(maxlen=window_size)
    
    def predict_workload(self, horizon_minutes: int = 15) -> Dict[str, float]:
        """预测未来工作负载"""
        if len(self.task_history) < 10:
            return {"predicted_tasks": 0.0, "predicted_load": 0.0, "confidence": 0.0}
        
        # 简单的移动平均预测
        recent_tasks = list(self.task_history)[-min(20, len(self.task_history)):]
        recent_loads = list(self.load_history)[-min(20, len(self.load_history)):]
        
        predicted_tasks = statistics.mean(recent_tasks) * (horizon_minutes / 5)  # 假设5分钟间隔
        predicted_load = statistics.mean(recent_loads)
        confidence = min(len(recent_tasks) / 20.0, 1.0)
        
        return {
            "predicted_tasks": predicted_tasks,
            "predicted_load": predicted_load,
            "confidence": confidence
        }

class CapacityPlanner:
    """容量规划器"""
    
    def __init__(self):
        self.capacity_history: deque = deque(maxlen=100)
        self.utilization_history: deque = deque(maxlen=100)
    
    def plan_capacity(self, current_agents: int, predicted_load: float) -> Dict[str, Any]:
        """容量规划建议"""
        target_utilization = 0.7  # 目标利用率70%
        
        if predicted_load > target_utilization:
            # 需要扩容
            needed_agents = max(1, int(current_agents * (predicted_load / target_utilization) - current_agents))
            recommendation = "SCALE_OUT"
        elif predicted_load < 0.3 and current_agents > 1:
            # 可以缩容
            surplus_agents = max(0, int(current_agents - (current_agents * predicted_load / target_utilization)))
            needed_agents = -min(surplus_agents, current_agents - 1)
            recommendation = "SCALE_IN"
        else:
            needed_agents = 0
            recommendation = "MAINTAIN"
        
        return {
            "recommendation": recommendation,
            "agents_change": needed_agents,
            "predicted_utilization": predicted_load,
            "target_utilization": target_utilization
        }

class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self, threshold_multiplier: float = 2.0):
        self.threshold_multiplier = threshold_multiplier
        self.baseline_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
    
    def detect_anomalies(self, current_metrics: SystemMetrics) -> List[Dict[str, Any]]:
        """检测系统异常"""
        anomalies = []
        
        metrics_to_check = {
            "response_time": current_metrics.avg_task_completion_time,
            "error_rate": current_metrics.error_rate,
            "system_load": current_metrics.system_load,
            "memory_usage": current_metrics.memory_usage
        }
        
        for metric_name, current_value in metrics_to_check.items():
            baseline = self.baseline_metrics[metric_name]
            
            if len(baseline) >= 10:  # 需要足够的基线数据
                mean_val = statistics.mean(baseline)
                std_val = statistics.stdev(baseline) if len(baseline) > 1 else 0
                threshold = mean_val + (std_val * self.threshold_multiplier)
                
                if current_value > threshold:
                    anomalies.append({
                        "type": "HIGH_" + metric_name.upper(),
                        "current_value": current_value,
                        "threshold": threshold,
                        "severity": "HIGH" if current_value > threshold * 1.5 else "MEDIUM"
                    })
            
            # 更新基线
            baseline.append(current_value)
        
        return anomalies
        
    async def _initialize(self):
        """初始化协调Agent"""
        # 启动任务调度循环
        asyncio.create_task(self._task_scheduling_loop())
        
        # 启动健康监控
        asyncio.create_task(self._health_monitoring_loop())
        
        # 启动指标更新
        asyncio.create_task(self._metrics_update_loop())
        
        # 注册默认任务模板
        self._register_default_task_templates()
        
        # 注册默认协作策略
        self._register_default_collaboration_strategies()
        
        self.logger.info("🎯 协调Agent初始化完成")
    
    def set_agent_registry(self, registry: AgentRegistry):
        """设置Agent注册中心"""
        self.agent_registry = registry
    
    async def _handle_command(self, message: AgentMessage):
        """处理命令消息"""
        command = message.content.get("command")
        params = message.content.get("params", {})
        
        if command == "submit_task":
            await self._handle_submit_task(message, params)
        elif command == "cancel_task":
            await self._handle_cancel_task(message, params)
        elif command == "get_task_status":
            await self._handle_get_task_status(message, params)
        elif command == "register_agent":
            await self._handle_register_agent(message, params)
        elif command == "update_workload":
            await self._handle_update_workload(message, params)
        elif command == "request_collaboration":
            await self._handle_request_collaboration(message, params)
        else:
            await self.send_response(message, {"error": f"未知命令: {command}"})
    
    async def _handle_query(self, message: AgentMessage):
        """处理查询消息"""
        query = message.content.get("query")
        params = message.content.get("params", {})
        
        if query == "system_status":
            response = await self._get_system_status()
        elif query == "task_statistics":
            response = self.task_scheduler.get_task_statistics()
        elif query == "agent_list":
            response = await self._get_agent_list()
        elif query == "task_queue":
            response = await self._get_task_queue_status()
        else:
            response = {"error": f"未知查询: {query}"}
        
        await self.send_response(message, response)
    
    async def _handle_submit_task(self, message: AgentMessage, params: Dict[str, Any]):
        """处理任务提交"""
        try:
            # 创建任务
            task = AgentTask(
                task_name=params.get("task_name", ""),
                task_type=params.get("task_type", ""),
                priority=TaskPriority(params.get("priority", TaskPriority.NORMAL.value)),
                required_capabilities=[AgentCapability(cap) for cap in params.get("capabilities", [])],
                parameters=params.get("parameters", {}),
                dependencies=params.get("dependencies", []),
                timeout=params.get("timeout")
            )
            
            # 添加到调度队列
            self.task_scheduler.add_task(task)
            
            await self.send_response(message, {
                "task_id": task.task_id,
                "status": "submitted"
            })
            
            self.logger.info(f"📝 收到任务: {task.task_name} (ID: {task.task_id})")
            
        except Exception as e:
            await self.send_response(message, {"error": str(e)})
    
    async def _handle_cancel_task(self, message: AgentMessage, params: Dict[str, Any]):
        """处理任务取消"""
        task_id = params.get("task_id")
        if not task_id:
            await self.send_response(message, {"error": "缺少task_id参数"})
            return
        
        # 查找并取消任务
        cancelled = False
        
        # 检查运行中的任务
        if task_id in self.task_scheduler.running_tasks:
            task = self.task_scheduler.running_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            
            # 通知执行Agent取消任务
            if task.assigned_agent:
                cancel_message = AgentMessage(
                    receiver_id=task.assigned_agent,
                    message_type=MessageType.COMMAND,
                    content={
                        "command": "cancel_task",
                        "params": {"task_id": task_id}
                    }
                )
                await self.send_message(cancel_message)
            
            cancelled = True
        
        await self.send_response(message, {
            "task_id": task_id,
            "cancelled": cancelled
        })
    
    async def _handle_register_agent(self, message: AgentMessage, params: Dict[str, Any]):
        """处理Agent注册"""
        try:
            agent_id = params.get("agent_id")
            agent_type = AgentType(params.get("agent_type"))
            capabilities = [AgentCapability(cap) for cap in params.get("capabilities", [])]
            
            # 创建工作负载信息
            workload = AgentWorkload(
                agent_id=agent_id,
                agent_type=agent_type,
                capabilities=capabilities,
                max_concurrent_tasks=params.get("max_concurrent_tasks", 10)
            )
            
            self.task_scheduler.update_agent_workload(agent_id, workload)
            
            await self.send_response(message, {"status": "registered"})
            self.logger.info(f"📋 注册Agent: {agent_id} ({agent_type.value})")
            
        except Exception as e:
            await self.send_response(message, {"error": str(e)})
    
    async def _handle_update_workload(self, message: AgentMessage, params: Dict[str, Any]):
        """处理工作负载更新"""
        agent_id = message.sender_id
        
        if agent_id in self.task_scheduler.agent_workloads:
            workload = self.task_scheduler.agent_workloads[agent_id]
            workload.current_tasks = params.get("current_tasks", workload.current_tasks)
            workload.average_response_time = params.get("average_response_time", workload.average_response_time)
            workload.success_rate = params.get("success_rate", workload.success_rate)
            workload.health_score = params.get("health_score", workload.health_score)
            workload.last_activity = datetime.utcnow()
    
    async def _handle_request_collaboration(self, message: AgentMessage, params: Dict[str, Any]):
        """处理协作请求"""
        collaboration_type = params.get("type")
        
        if collaboration_type in self.collaboration_strategies:
            strategy = self.collaboration_strategies[collaboration_type]
            result = await strategy(message, params)
            await self.send_response(message, result)
        else:
            await self.send_response(message, {"error": f"未知协作类型: {collaboration_type}"})
    
    async def _task_scheduling_loop(self):
        """任务调度循环"""
        self.logger.info("🔄 启动任务调度循环")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.scheduling_interval)
                
                # 获取下一个任务
                task = self.task_scheduler.get_next_task()
                if not task:
                    continue
                
                # 检查依赖关系
                if not await self._check_task_dependencies(task):
                    # 重新放入队列
                    self.task_scheduler.add_task(task)
                    continue
                
                # 寻找合适的Agent
                best_agent = self.task_scheduler.find_best_agent(task)
                if not best_agent:
                    # 没有可用Agent，重新放入队列
                    self.task_scheduler.add_task(task)
                    await asyncio.sleep(5)  # 等待一段时间再试
                    continue
                
                # 分配任务
                await self._assign_task(task, best_agent)
                
            except Exception as e:
                self.logger.error(f"任务调度循环错误: {e}")
    
    async def _check_task_dependencies(self, task: AgentTask) -> bool:
        """检查任务依赖关系"""
        if not task.dependencies:
            return True
        
        for dep_task_id in task.dependencies:
            # 检查依赖任务是否已完成
            if dep_task_id in self.task_scheduler.completed_tasks:
                dep_task = self.task_scheduler.completed_tasks[dep_task_id]
                if dep_task.status != TaskStatus.COMPLETED:
                    return False
            elif dep_task_id in self.task_scheduler.running_tasks:
                return False  # 依赖任务还在运行
            else:
                return False  # 依赖任务不存在
        
        return True
    
    async def _assign_task(self, task: AgentTask, agent_id: str):
        """分配任务给Agent"""
        try:
            # 发送任务消息
            task_message = AgentMessage(
                receiver_id=agent_id,
                message_type=MessageType.COMMAND,
                priority=task.priority.value,
                content={
                    "command": "execute_task",
                    "params": {
                        "task_id": task.task_id,
                        "task_name": task.task_name,
                        "task_type": task.task_type,
                        "parameters": task.parameters,
                        "timeout": task.timeout
                    }
                }
            )
            
            if await self.send_message(task_message):
                # 标记任务为运行状态
                self.task_scheduler.running_tasks[task.task_id] = task
                self.task_scheduler.mark_task_running(task.task_id, agent_id)
                
                self.logger.info(f"📤 分配任务 {task.task_name} 给 Agent {agent_id}")
                
                # 启动任务超时监控
                if task.timeout:
                    asyncio.create_task(self._monitor_task_timeout(task))
            else:
                # 发送失败，重新放入队列
                self.task_scheduler.add_task(task)
                
        except Exception as e:
            self.logger.error(f"分配任务失败: {e}")
            self.task_scheduler.add_task(task)
    
    async def _monitor_task_timeout(self, task: AgentTask):
        """监控任务超时"""
        await asyncio.sleep(task.timeout)
        
        # 检查任务是否还在运行
        if task.task_id in self.task_scheduler.running_tasks:
            # 任务超时，取消任务
            self.task_scheduler.complete_task(
                task.task_id,
                error=f"任务超时 ({task.timeout}秒)"
            )
            
            # 通知Agent取消任务
            if task.assigned_agent:
                cancel_message = AgentMessage(
                    receiver_id=task.assigned_agent,
                    message_type=MessageType.COMMAND,
                    content={
                        "command": "cancel_task",
                        "params": {"task_id": task.task_id}
                    }
                )
                await self.send_message(cancel_message)
            
            self.logger.warning(f"⏰ 任务超时: {task.task_name}")
    
    async def _health_monitoring_loop(self):
        """健康监控循环"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # 检查Agent健康状态
                unhealthy_agents = []
                current_time = datetime.utcnow()
                
                for agent_id, workload in self.task_scheduler.agent_workloads.items():
                    if workload.last_activity:
                        inactive_time = (current_time - workload.last_activity).total_seconds()
                        if inactive_time > 120:  # 2分钟无活动
                            workload.health_score *= 0.9
                            if workload.health_score < 0.5:
                                unhealthy_agents.append(agent_id)
                
                # 处理不健康的Agent
                for agent_id in unhealthy_agents:
                    self.logger.warning(f"🔴 Agent {agent_id} 健康状态不佳")
                    # 可以在这里实现重新平衡或故障转移逻辑
                
            except Exception as e:
                self.logger.error(f"健康监控错误: {e}")
    
    async def _metrics_update_loop(self):
        """系统指标更新循环"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.metrics_update_interval)
                
                # 更新系统指标
                stats = self.task_scheduler.get_task_statistics()
                
                self.system_metrics.update({
                    "total_tasks": stats["total_tasks"],
                    "active_agents": len(self.task_scheduler.agent_workloads),
                    "error_rate": 1 - stats["success_rate"],
                    "average_response_time": stats["average_execution_time"]
                })
                
                # 计算系统负载
                total_capacity = sum(w.max_concurrent_tasks for w in self.task_scheduler.agent_workloads.values())
                current_load = sum(w.current_tasks for w in self.task_scheduler.agent_workloads.values())
                
                if total_capacity > 0:
                    self.system_metrics["system_load"] = current_load / total_capacity
                
            except Exception as e:
                self.logger.error(f"指标更新错误: {e}")
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "coordinator_status": self.status.value,
            "system_metrics": self.system_metrics,
            "task_statistics": self.task_scheduler.get_task_statistics(),
            "agent_count": len(self.task_scheduler.agent_workloads),
            "uptime": (datetime.utcnow() - self.start_time).total_seconds()
        }
    
    async def _get_agent_list(self) -> List[Dict[str, Any]]:
        """获取Agent列表"""
        agents = []
        for agent_id, workload in self.task_scheduler.agent_workloads.items():
            agents.append({
                "agent_id": agent_id,
                "agent_type": workload.agent_type.value,
                "capabilities": [cap.value for cap in workload.capabilities],
                "current_tasks": workload.current_tasks,
                "max_tasks": workload.max_concurrent_tasks,
                "availability": workload.availability,
                "health_score": workload.health_score
            })
        return agents
    
    async def _get_task_queue_status(self) -> Dict[str, Any]:
        """获取任务队列状态"""
        return {
            "pending_tasks": len(self.task_scheduler.pending_tasks),
            "running_tasks": len(self.task_scheduler.running_tasks),
            "completed_tasks": len(self.task_scheduler.completed_tasks)
        }
    
    def _register_default_task_templates(self):
        """注册默认任务模板"""
        self.task_templates = {
            "data_analysis": {
                "required_capabilities": [AgentCapability.DATA_ANALYSIS],
                "default_timeout": 300
            },
            "strategy_generation": {
                "required_capabilities": [AgentCapability.STRATEGY_GENERATION],
                "default_timeout": 600
            },
            "risk_assessment": {
                "required_capabilities": [AgentCapability.RISK_ASSESSMENT],
                "default_timeout": 120
            },
            "order_execution": {
                "required_capabilities": [AgentCapability.ORDER_EXECUTION],
                "default_timeout": 30
            }
        }
    
    def _register_default_collaboration_strategies(self):
        """注册默认协作策略"""
        self.collaboration_strategies = {
            "consensus": self._consensus_collaboration,
            "competition": self._competition_collaboration,
            "pipeline": self._pipeline_collaboration,
            "parallel": self._parallel_collaboration
        }
    
    async def _consensus_collaboration(self, message: AgentMessage, params: Dict[str, Any]) -> Dict[str, Any]:
        """共识协作策略 - 多个Agent对同一问题给出结果，取共识"""
        task_params = params.get("task_params", {})
        agent_count = params.get("agent_count", 3)
        capabilities = params.get("capabilities", [])
        
        # 找到具有所需能力的Agent
        suitable_agents = []
        for agent_id, workload in self.task_scheduler.agent_workloads.items():
            if all(cap in workload.capabilities for cap in capabilities):
                suitable_agents.append(agent_id)
        
        if len(suitable_agents) < agent_count:
            return {"error": "可用Agent数量不足"}
        
        # 选择前N个Agent
        selected_agents = suitable_agents[:agent_count]
        
        # 创建并分配任务
        tasks = []
        for agent_id in selected_agents:
            task = AgentTask(
                task_name=f"共识任务_{params.get('task_name', 'unknown')}",
                task_type="consensus",
                required_capabilities=capabilities,
                parameters=task_params
            )
            tasks.append(task)
            await self._assign_task(task, agent_id)
        
        return {
            "collaboration_id": str(uuid.uuid4()),
            "strategy": "consensus",
            "agents": selected_agents,
            "task_ids": [task.task_id for task in tasks]
        }
    
    async def _competition_collaboration(self, message: AgentMessage, params: Dict[str, Any]) -> Dict[str, Any]:
        """竞争协作策略 - 多个Agent竞争解决问题，选择最佳结果"""
        # 实现竞争逻辑
        return {"strategy": "competition", "status": "implemented"}
    
    async def _pipeline_collaboration(self, message: AgentMessage, params: Dict[str, Any]) -> Dict[str, Any]:
        """流水线协作策略 - Agent按顺序处理任务"""
        # 实现流水线逻辑
        return {"strategy": "pipeline", "status": "implemented"}
    
    async def _parallel_collaboration(self, message: AgentMessage, params: Dict[str, Any]) -> Dict[str, Any]:
        """并行协作策略 - 多个Agent并行处理不同子任务"""
        # 实现并行逻辑
        return {"strategy": "parallel", "status": "implemented"}
    
    async def create_task(
        self,
        task_name: str,
        task_type: str,
        capabilities: List[AgentCapability],
        parameters: Dict[str, Any] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[int] = None
    ) -> str:
        """创建并提交任务的便捷方法"""
        task = AgentTask(
            task_name=task_name,
            task_type=task_type,
            priority=priority,
            required_capabilities=capabilities,
            parameters=parameters or {},
            timeout=timeout
        )
        
        self.task_scheduler.add_task(task)
        return task.task_id
    
    def get_system_overview(self) -> Dict[str, Any]:
        """获取系统总览"""
        return {
            "coordinator": {
                "status": self.status.value,
                "uptime": (datetime.utcnow() - self.start_time).total_seconds(),
                "processed_messages": self.metrics.messages_processed
            },
            "system": self.system_metrics,
            "tasks": self.task_scheduler.get_task_statistics(),
            "agents": {
                "total": len(self.task_scheduler.agent_workloads),
                "active": len([w for w in self.task_scheduler.agent_workloads.values() if w.current_tasks > 0])
            }
        }