"""
超级AI多Agent协作系统 - Agent基类
实现高性能异步Agent架构，支持智能协作和任务调度
"""

import asyncio
import uuid
import time
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

# Agent状态枚举
class AgentStatus(Enum):
    INITIALIZING = "initializing"
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    ERROR = "error"
    SHUTDOWN = "shutdown"

# Agent类型枚举
class AgentType(Enum):
    COORDINATOR = "coordinator"
    STRATEGY = "strategy"
    RISK = "risk"
    EXECUTION = "execution"
    MONITORING = "monitoring"
    DATA_PROCESSOR = "data_processor"
    ML_TRAINER = "ml_trainer"

# 消息类型枚举
class MessageType(Enum):
    COMMAND = "command"
    QUERY = "query"
    RESPONSE = "response"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    ERROR = "error"

# Agent能力枚举
class AgentCapability(Enum):
    DATA_ANALYSIS = "data_analysis"
    STRATEGY_GENERATION = "strategy_generation"
    RISK_ASSESSMENT = "risk_assessment"
    ORDER_EXECUTION = "order_execution"
    MARKET_MONITORING = "market_monitoring"
    PATTERN_RECOGNITION = "pattern_recognition"
    PREDICTION = "prediction"
    OPTIMIZATION = "optimization"

@dataclass
class AgentMessage:
    """Agent间通信消息"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""  # "*" for broadcast
    message_type: MessageType = MessageType.COMMAND
    priority: int = 1  # 1-10, 10最高优先级
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    correlation_id: Optional[str] = None  # 用于请求-响应关联
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type.value,
            "priority": self.priority,
            "timestamp": self.timestamp.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "correlation_id": self.correlation_id,
            "content": self.content,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        return cls(
            id=data["id"],
            sender_id=data["sender_id"],
            receiver_id=data["receiver_id"],
            message_type=MessageType(data["message_type"]),
            priority=data["priority"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data["expires_at"] else None,
            correlation_id=data.get("correlation_id"),
            content=data.get("content", {}),
            metadata=data.get("metadata", {})
        )

@dataclass
class AgentMetrics:
    """Agent性能指标"""
    messages_sent: int = 0
    messages_received: int = 0
    messages_processed: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_response_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    uptime: float = 0.0
    error_count: int = 0
    last_activity: Optional[datetime] = None

class BaseAgent(ABC):
    """
    Agent基类
    
    提供：
    - 异步消息处理
    - 生命周期管理
    - 性能监控
    - 错误处理和恢复
    - 健康检查
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        capabilities: List[AgentCapability],
        config: Optional[Dict[str, Any]] = None
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.config = config or {}
        
        # Agent状态
        self.status = AgentStatus.INITIALIZING
        self.start_time = datetime.utcnow()
        self.last_heartbeat = datetime.utcnow()
        
        # 消息处理
        self.message_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.response_handlers: Dict[str, Callable] = {}
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # 性能指标
        self.metrics = AgentMetrics()
        
        # 并发控制
        self.max_concurrent_tasks = config.get("max_concurrent_tasks", 10)
        self.task_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        
        # 线程池（用于CPU密集型任务）
        self.thread_pool = ThreadPoolExecutor(
            max_workers=config.get("thread_pool_size", 4),
            thread_name_prefix=f"Agent-{agent_id}"
        )
        
        # 任务管理
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_results: Dict[str, Any] = {}
        
        # 日志
        self.logger = logging.getLogger(f"Agent-{agent_id}")
        
        # 健康检查
        self.health_check_interval = config.get("health_check_interval", 30)
        self.heartbeat_timeout = config.get("heartbeat_timeout", 60)
        
        # Agent通信客户端（由子类设置）
        self.communication_client = None
        
        # 停止标志
        self._shutdown_event = asyncio.Event()
    
    async def initialize(self) -> bool:
        """初始化Agent"""
        try:
            self.logger.info(f"🚀 初始化Agent {self.agent_id} ({self.agent_type.value})")
            
            # 执行子类特定的初始化
            await self._initialize()
            
            # 启动消息处理循环
            asyncio.create_task(self._message_processing_loop())
            
            # 启动健康检查
            asyncio.create_task(self._health_check_loop())
            
            # 启动性能监控
            asyncio.create_task(self._metrics_update_loop())
            
            self.status = AgentStatus.IDLE
            self.logger.info(f"✅ Agent {self.agent_id} 初始化完成")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Agent {self.agent_id} 初始化失败: {e}")
            self.status = AgentStatus.ERROR
            return False
    
    @abstractmethod
    async def _initialize(self):
        """子类特定的初始化逻辑"""
        pass
    
    async def send_message(self, message: AgentMessage) -> bool:
        """发送消息"""
        try:
            if self.communication_client is None:
                self.logger.error("通信客户端未设置")
                return False
            
            message.sender_id = self.agent_id
            message.timestamp = datetime.utcnow()
            
            await self.communication_client.send_message(message)
            self.metrics.messages_sent += 1
            
            self.logger.debug(f"📤 发送消息 {message.id} 到 {message.receiver_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"发送消息失败: {e}")
            return False
    
    async def broadcast_message(self, message: AgentMessage) -> bool:
        """广播消息"""
        message.receiver_id = "*"
        return await self.send_message(message)
    
    async def send_command(
        self, 
        receiver_id: str, 
        command: str, 
        params: Dict[str, Any] = None,
        priority: int = 1,
        timeout: Optional[int] = None
    ) -> bool:
        """发送命令消息"""
        message = AgentMessage(
            receiver_id=receiver_id,
            message_type=MessageType.COMMAND,
            priority=priority,
            content={"command": command, "params": params or {}},
            expires_at=datetime.utcnow() + timedelta(seconds=timeout) if timeout else None
        )
        return await self.send_message(message)
    
    async def send_query(
        self,
        receiver_id: str,
        query: str,
        params: Dict[str, Any] = None,
        timeout: int = 30
    ) -> Optional[Any]:
        """发送查询并等待响应"""
        correlation_id = str(uuid.uuid4())
        
        # 创建响应等待
        response_future = asyncio.Future()
        self.response_handlers[correlation_id] = lambda response: response_future.set_result(response)
        
        message = AgentMessage(
            receiver_id=receiver_id,
            message_type=MessageType.QUERY,
            correlation_id=correlation_id,
            content={"query": query, "params": params or {}},
            expires_at=datetime.utcnow() + timedelta(seconds=timeout)
        )
        
        try:
            if await self.send_message(message):
                # 等待响应
                response = await asyncio.wait_for(response_future, timeout=timeout)
                return response
            else:
                return None
                
        except asyncio.TimeoutError:
            self.logger.warning(f"查询超时: {query}")
            return None
        finally:
            # 清理响应处理器
            self.response_handlers.pop(correlation_id, None)
    
    async def send_response(self, original_message: AgentMessage, response_data: Any):
        """发送响应消息"""
        if original_message.correlation_id is None:
            return False
        
        response = AgentMessage(
            receiver_id=original_message.sender_id,
            message_type=MessageType.RESPONSE,
            correlation_id=original_message.correlation_id,
            content={"response": response_data}
        )
        return await self.send_message(response)
    
    async def receive_message(self, message: AgentMessage):
        """接收消息（由通信客户端调用）"""
        if message.is_expired():
            self.logger.debug(f"收到过期消息: {message.id}")
            return
        
        # 放入消息队列
        try:
            await self.message_queue.put(message)
            self.metrics.messages_received += 1
        except asyncio.QueueFull:
            self.logger.warning("消息队列已满，丢弃消息")
    
    async def _message_processing_loop(self):
        """消息处理循环"""
        self.logger.info("📨 启动消息处理循环")
        
        while not self._shutdown_event.is_set():
            try:
                # 获取消息（带超时）
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                
                # 处理消息
                asyncio.create_task(self._process_message(message))
                
            except asyncio.TimeoutError:
                # 超时是正常的，继续循环
                continue
            except Exception as e:
                self.logger.error(f"消息处理循环错误: {e}")
                await asyncio.sleep(1)
    
    async def _process_message(self, message: AgentMessage):
        """处理单个消息"""
        async with self.task_semaphore:  # 控制并发
            start_time = time.time()
            
            try:
                self.status = AgentStatus.WORKING
                self.last_heartbeat = datetime.utcnow()
                
                if message.message_type == MessageType.COMMAND:
                    await self._handle_command(message)
                elif message.message_type == MessageType.QUERY:
                    await self._handle_query(message)
                elif message.message_type == MessageType.RESPONSE:
                    await self._handle_response(message)
                elif message.message_type == MessageType.EVENT:
                    await self._handle_event(message)
                elif message.message_type == MessageType.HEARTBEAT:
                    await self._handle_heartbeat(message)
                else:
                    self.logger.warning(f"未知消息类型: {message.message_type}")
                
                # 更新指标
                self.metrics.messages_processed += 1
                response_time = time.time() - start_time
                
                # 计算平均响应时间
                if self.metrics.average_response_time == 0:
                    self.metrics.average_response_time = response_time
                else:
                    self.metrics.average_response_time = (
                        self.metrics.average_response_time * 0.9 + response_time * 0.1
                    )
                
                self.status = AgentStatus.IDLE
                
            except Exception as e:
                self.logger.error(f"处理消息失败 {message.id}: {e}")
                self.metrics.error_count += 1
                self.status = AgentStatus.ERROR
                
                # 发送错误响应（如果需要）
                if message.message_type == MessageType.QUERY:
                    await self.send_response(message, {
                        "error": str(e),
                        "error_type": type(e).__name__
                    })
    
    @abstractmethod
    async def _handle_command(self, message: AgentMessage):
        """处理命令消息"""
        pass
    
    @abstractmethod
    async def _handle_query(self, message: AgentMessage):
        """处理查询消息"""
        pass
    
    async def _handle_response(self, message: AgentMessage):
        """处理响应消息"""
        correlation_id = message.correlation_id
        if correlation_id and correlation_id in self.response_handlers:
            handler = self.response_handlers[correlation_id]
            handler(message.content.get("response"))
    
    async def _handle_event(self, message: AgentMessage):
        """处理事件消息"""
        event_type = message.content.get("event_type")
        if event_type and event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    await handler(message)
                except Exception as e:
                    self.logger.error(f"事件处理器错误 {event_type}: {e}")
    
    async def _handle_heartbeat(self, message: AgentMessage):
        """处理心跳消息"""
        self.last_heartbeat = datetime.utcnow()
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """注册事件处理器"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.health_check_interval)
                
                # 检查心跳超时
                if datetime.utcnow() - self.last_heartbeat > timedelta(seconds=self.heartbeat_timeout):
                    self.logger.warning("心跳超时，Agent可能不健康")
                    self.status = AgentStatus.ERROR
                
                # 执行健康检查
                is_healthy = await self._perform_health_check()
                if not is_healthy:
                    self.logger.error("健康检查失败")
                    self.status = AgentStatus.ERROR
                
            except Exception as e:
                self.logger.error(f"健康检查循环错误: {e}")
    
    async def _perform_health_check(self) -> bool:
        """执行健康检查（子类可重写）"""
        return True
    
    async def _metrics_update_loop(self):
        """性能指标更新循环"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(10)  # 每10秒更新一次指标
                
                # 更新运行时间
                self.metrics.uptime = (datetime.utcnow() - self.start_time).total_seconds()
                self.metrics.last_activity = datetime.utcnow()
                
                # 更新系统资源使用情况
                try:
                    import psutil
                    process = psutil.Process()
                    self.metrics.cpu_usage = process.cpu_percent()
                    self.metrics.memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                except ImportError:
                    pass  # psutil未安装
                
            except Exception as e:
                self.logger.error(f"指标更新错误: {e}")
    
    async def execute_task(self, task_name: str, task_func: Callable, *args, **kwargs) -> str:
        """执行异步任务"""
        task_id = str(uuid.uuid4())
        
        async def task_wrapper():
            try:
                result = await task_func(*args, **kwargs)
                self.task_results[task_id] = {"status": "completed", "result": result}
                self.metrics.tasks_completed += 1
            except Exception as e:
                self.task_results[task_id] = {"status": "failed", "error": str(e)}
                self.metrics.tasks_failed += 1
            finally:
                self.running_tasks.pop(task_id, None)
        
        task = asyncio.create_task(task_wrapper())
        self.running_tasks[task_id] = task
        
        self.logger.info(f"🔄 启动任务 {task_name} (ID: {task_id})")
        return task_id
    
    async def execute_cpu_task(self, task_name: str, task_func: Callable, *args, **kwargs) -> str:
        """执行CPU密集型任务（在线程池中）"""
        task_id = str(uuid.uuid4())
        
        def task_wrapper():
            try:
                result = task_func(*args, **kwargs)
                self.task_results[task_id] = {"status": "completed", "result": result}
                self.metrics.tasks_completed += 1
            except Exception as e:
                self.task_results[task_id] = {"status": "failed", "error": str(e)}
                self.metrics.tasks_failed += 1
        
        # 在线程池中执行
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(self.thread_pool, task_wrapper)
        self.running_tasks[task_id] = task
        
        self.logger.info(f"🧮 启动CPU任务 {task_name} (ID: {task_id})")
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        if task_id in self.running_tasks:
            return {"status": "running"}
        elif task_id in self.task_results:
            return self.task_results[task_id]
        else:
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        metrics_dict = asdict(self.metrics)
        metrics_dict.update({
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "running_tasks": len(self.running_tasks),
            "queued_messages": self.message_queue.qsize()
        })
        return metrics_dict
    
    async def shutdown(self):
        """关闭Agent"""
        self.logger.info(f"🛑 关闭Agent {self.agent_id}")
        
        self.status = AgentStatus.SHUTDOWN
        self._shutdown_event.set()
        
        # 等待运行中的任务完成
        if self.running_tasks:
            self.logger.info(f"等待 {len(self.running_tasks)} 个任务完成...")
            await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
        
        # 关闭线程池
        self.thread_pool.shutdown(wait=True)
        
        # 执行子类特定的清理
        await self._cleanup()
        
        self.logger.info(f"✅ Agent {self.agent_id} 已关闭")
    
    async def _cleanup(self):
        """子类特定的清理逻辑"""
        pass
    
    def __repr__(self) -> str:
        return f"Agent({self.agent_id}, {self.agent_type.value}, {self.status.value})"