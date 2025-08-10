"""
N8N工作流管理器
实现与n8n平台的集成，自动化交易流程和数据处理
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_logger import get_logger, LogCategory

logger = get_logger()

class WorkflowStatus(Enum):
    """工作流状态"""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    WAITING = "waiting"
    CANCELED = "canceled"

class WorkflowTrigger(Enum):
    """工作流触发器类型"""
    MANUAL = "manual"
    WEBHOOK = "webhook"
    CRON = "cron"
    DATA_CHANGE = "data_change"
    PRICE_ALERT = "price_alert"
    MARKET_EVENT = "market_event"

@dataclass
class WorkflowExecution:
    """工作流执行记录"""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    trigger_type: WorkflowTrigger
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    input_data: Optional[Dict] = None
    output_data: Optional[Dict] = None
    error_message: Optional[str] = None
    retry_count: int = 0

@dataclass
class WorkflowDefinition:
    """工作流定义"""
    workflow_id: str
    name: str
    description: str
    trigger_type: WorkflowTrigger
    enabled: bool = True
    schedule: Optional[str] = None  # Cron表达式
    retry_attempts: int = 3
    timeout_seconds: int = 300
    webhook_path: Optional[str] = None
    nodes: List[Dict] = field(default_factory=list)

class N8NClient:
    """N8N API客户端"""
    
    def __init__(self, base_url: str = "http://localhost:5678", api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        headers = {}
        if self.api_key:
            headers["X-N8N-API-KEY"] = self.api_key
        
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_workflows(self) -> List[Dict]:
        """获取所有工作流"""
        try:
            async with self.session.get(f"{self.base_url}/api/v1/workflows") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', [])
                else:
                    logger.error(f"获取工作流失败: HTTP {response.status}")
                    return []
        except Exception as e:
            logger.error(f"获取工作流异常: {e}")
            return []
    
    async def get_workflow(self, workflow_id: str) -> Optional[Dict]:
        """获取单个工作流详情"""
        try:
            async with self.session.get(f"{self.base_url}/api/v1/workflows/{workflow_id}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"获取工作流 {workflow_id} 失败: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.error(f"获取工作流 {workflow_id} 异常: {e}")
            return None
    
    async def execute_workflow(self, workflow_id: str, input_data: Optional[Dict] = None) -> Optional[str]:
        """手动执行工作流"""
        try:
            payload = {"data": input_data} if input_data else {}
            
            async with self.session.post(
                f"{self.base_url}/api/v1/workflows/{workflow_id}/execute",
                json=payload
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('executionId')
                else:
                    logger.error(f"执行工作流 {workflow_id} 失败: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.error(f"执行工作流 {workflow_id} 异常: {e}")
            return None
    
    async def get_execution_status(self, execution_id: str) -> Optional[Dict]:
        """获取执行状态"""
        try:
            async with self.session.get(f"{self.base_url}/api/v1/executions/{execution_id}") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"获取执行状态 {execution_id} 失败: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.error(f"获取执行状态 {execution_id} 异常: {e}")
            return None
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """取消执行"""
        try:
            async with self.session.delete(f"{self.base_url}/api/v1/executions/{execution_id}/stop") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"取消执行 {execution_id} 异常: {e}")
            return False
    
    async def create_workflow(self, workflow_data: Dict) -> Optional[str]:
        """创建新工作流"""
        try:
            async with self.session.post(f"{self.base_url}/api/v1/workflows", json=workflow_data) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('id')
                else:
                    logger.error(f"创建工作流失败: HTTP {response.status}")
                    return None
        except Exception as e:
            logger.error(f"创建工作流异常: {e}")
            return None
    
    async def update_workflow(self, workflow_id: str, workflow_data: Dict) -> bool:
        """更新工作流"""
        try:
            async with self.session.put(f"{self.base_url}/api/v1/workflows/{workflow_id}", json=workflow_data) as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"更新工作流 {workflow_id} 异常: {e}")
            return False
    
    async def activate_workflow(self, workflow_id: str) -> bool:
        """激活工作流"""
        try:
            async with self.session.post(f"{self.base_url}/api/v1/workflows/{workflow_id}/activate") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"激活工作流 {workflow_id} 异常: {e}")
            return False
    
    async def deactivate_workflow(self, workflow_id: str) -> bool:
        """停用工作流"""
        try:
            async with self.session.post(f"{self.base_url}/api/v1/workflows/{workflow_id}/deactivate") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"停用工作流 {workflow_id} 异常: {e}")
            return False

class WorkflowErrorHandler:
    """工作流错误处理器"""
    
    def __init__(self):
        self.error_counts = {}
        self.error_history = []
        self.max_history = 1000
        
    def record_error(self, workflow_id: str, execution_id: str, error_message: str, error_data: Dict = None):
        """记录错误"""
        error_record = {
            "timestamp": datetime.utcnow(),
            "workflow_id": workflow_id,
            "execution_id": execution_id,
            "error_message": error_message,
            "error_data": error_data or {},
            "severity": self._classify_error_severity(error_message)
        }
        
        # 记录到历史
        self.error_history.append(error_record)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # 更新错误计数
        self.error_counts[workflow_id] = self.error_counts.get(workflow_id, 0) + 1
        
        # 记录日志
        severity = error_record["severity"]
        if severity == "critical":
            logger.critical(f"🚨 工作流严重错误 [{workflow_id}:{execution_id}]: {error_message}")
        elif severity == "high":
            logger.error(f"❌ 工作流高优先级错误 [{workflow_id}:{execution_id}]: {error_message}")
        elif severity == "medium":
            logger.warning(f"⚠️ 工作流中等错误 [{workflow_id}:{execution_id}]: {error_message}")
        else:
            logger.info(f"ℹ️ 工作流一般错误 [{workflow_id}:{execution_id}]: {error_message}")
    
    def _classify_error_severity(self, error_message: str) -> str:
        """错误严重程度分类"""
        error_lower = error_message.lower()
        
        if any(keyword in error_lower for keyword in ["timeout", "connection", "network", "database"]):
            return "high"
        elif any(keyword in error_lower for keyword in ["authentication", "permission", "unauthorized"]):
            return "critical"
        elif any(keyword in error_lower for keyword in ["validation", "format", "parse"]):
            return "medium"
        else:
            return "low"
    
    def get_error_stats(self, hours: int = 24) -> Dict[str, Any]:
        """获取错误统计"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_errors = [e for e in self.error_history if e["timestamp"] >= cutoff_time]
        
        severity_counts = {}
        workflow_errors = {}
        
        for error in recent_errors:
            # 按严重程度统计
            severity = error["severity"]
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # 按工作流统计
            workflow_id = error["workflow_id"]
            if workflow_id not in workflow_errors:
                workflow_errors[workflow_id] = {"count": 0, "last_error": None}
            workflow_errors[workflow_id]["count"] += 1
            workflow_errors[workflow_id]["last_error"] = error["error_message"]
        
        return {
            "time_range_hours": hours,
            "total_errors": len(recent_errors),
            "severity_breakdown": severity_counts,
            "workflow_breakdown": workflow_errors,
            "error_rate": len(recent_errors) / max(hours, 1)  # 错误/小时
        }

class N8NWorkflowManager:
    """N8N工作流管理器"""
    
    def __init__(self, n8n_base_url: str = "http://localhost:5678", api_key: Optional[str] = None):
        self.n8n_base_url = n8n_base_url
        self.api_key = api_key
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.error_handler = WorkflowErrorHandler()
        
        # 监控任务
        self.monitoring_task = None
        self.monitoring_enabled = False
        self.monitoring_interval = 30  # 秒
        
    async def initialize(self):
        """初始化工作流管理器"""
        try:
            logger.info("🔧 初始化N8N工作流管理器...")
            
            # 测试N8N连接
            await self._test_n8n_connection()
            
            # 加载现有工作流
            await self._load_workflows()
            
            # 初始化预定义工作流
            await self._initialize_predefined_workflows()
            
            # 启动监控
            self.monitoring_enabled = True
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            logger.info(f"✅ N8N工作流管理器初始化完成，已加载{len(self.workflows)}个工作流")
            
        except Exception as e:
            logger.error(f"❌ N8N工作流管理器初始化失败: {e}")
            raise
    
    async def _test_n8n_connection(self):
        """测试N8N连接"""
        try:
            async with N8NClient(self.n8n_base_url, self.api_key) as client:
                workflows = await client.get_workflows()
                logger.info(f"✅ N8N连接成功，发现{len(workflows)}个现有工作流")
        except Exception as e:
            logger.warning(f"⚠️ N8N连接测试失败，可能需要启动N8N服务: {e}")
    
    async def _load_workflows(self):
        """加载现有工作流"""
        try:
            async with N8NClient(self.n8n_base_url, self.api_key) as client:
                workflows = await client.get_workflows()
                
                for workflow_data in workflows:
                    workflow_def = WorkflowDefinition(
                        workflow_id=workflow_data.get('id', ''),
                        name=workflow_data.get('name', ''),
                        description=workflow_data.get('description', ''),
                        trigger_type=WorkflowTrigger.MANUAL,  # 默认手动触发
                        enabled=workflow_data.get('active', False),
                        nodes=workflow_data.get('nodes', [])
                    )
                    
                    self.workflows[workflow_def.workflow_id] = workflow_def
                    
        except Exception as e:
            logger.error(f"加载现有工作流失败: {e}")
    
    async def _initialize_predefined_workflows(self):
        """初始化预定义工作流"""
        # 定义交易系统常用工作流模板
        predefined_workflows = [
            {
                "name": "价格监控和告警",
                "description": "监控特定交易对价格变化并发送告警",
                "trigger_type": WorkflowTrigger.CRON,
                "schedule": "*/5 * * * *",  # 每5分钟执行
                "nodes": self._create_price_monitoring_nodes()
            },
            {
                "name": "数据质量检查",
                "description": "定期检查数据质量并生成报告",
                "trigger_type": WorkflowTrigger.CRON,
                "schedule": "0 */1 * * *",  # 每小时执行
                "nodes": self._create_data_quality_nodes()
            },
            {
                "name": "系统健康检查",
                "description": "监控系统各组件健康状态",
                "trigger_type": WorkflowTrigger.CRON,
                "schedule": "*/10 * * * *",  # 每10分钟执行
                "nodes": self._create_health_check_nodes()
            },
            {
                "name": "交易信号处理",
                "description": "处理AI生成的交易信号",
                "trigger_type": WorkflowTrigger.WEBHOOK,
                "webhook_path": "/webhook/trading-signal",
                "nodes": self._create_trading_signal_nodes()
            },
            {
                "name": "风险管理",
                "description": "实时风险评估和管理",
                "trigger_type": WorkflowTrigger.DATA_CHANGE,
                "nodes": self._create_risk_management_nodes()
            }
        ]
        
        # 创建预定义工作流
        for workflow_template in predefined_workflows:
            try:
                workflow_id = await self._create_workflow_from_template(workflow_template)
                if workflow_id:
                    logger.info(f"✅ 创建预定义工作流: {workflow_template['name']}")
            except Exception as e:
                logger.error(f"❌ 创建预定义工作流失败 {workflow_template['name']}: {e}")
    
    def _create_price_monitoring_nodes(self) -> List[Dict]:
        """创建价格监控节点"""
        return [
            {
                "name": "获取价格数据",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8000/api/v1/data/latest-prices",
                    "options": {"timeout": 10000}
                },
                "position": [250, 300]
            },
            {
                "name": "价格变化判断",
                "type": "n8n-nodes-base.function",
                "parameters": {
                    "functionCode": """
                    const data = items[0].json;
                    const btcPrice = parseFloat(data.prices['BTC/USDT'].binance?.price || 0);
                    const ethPrice = parseFloat(data.prices['ETH/USDT'].binance?.price || 0);
                    
                    // 价格告警阈值
                    const alerts = [];
                    if (btcPrice > 50000) alerts.push({coin: 'BTC', price: btcPrice, type: 'high'});
                    if (btcPrice < 40000) alerts.push({coin: 'BTC', price: btcPrice, type: 'low'});
                    if (ethPrice > 4000) alerts.push({coin: 'ETH', price: ethPrice, type: 'high'});
                    if (ethPrice < 3000) alerts.push({coin: 'ETH', price: ethPrice, type: 'low'});
                    
                    return alerts.map(alert => ({json: alert}));
                    """
                },
                "position": [450, 300]
            },
            {
                "name": "发送告警",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8000/api/v1/alerts/price",
                    "method": "POST",
                    "options": {"timeout": 5000}
                },
                "position": [650, 300]
            }
        ]
    
    def _create_data_quality_nodes(self) -> List[Dict]:
        """创建数据质量检查节点"""
        return [
            {
                "name": "获取数据质量报告",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8000/api/v1/data/quality-report",
                    "options": {"timeout": 30000}
                },
                "position": [250, 300]
            },
            {
                "name": "分析质量指标",
                "type": "n8n-nodes-base.function",
                "parameters": {
                    "functionCode": """
                    const report = items[0].json;
                    const issues = [];
                    
                    // 检查数据质量分数
                    if (report.overall_score < 80) {
                        issues.push({
                            type: 'low_quality_score',
                            score: report.overall_score,
                            severity: 'medium'
                        });
                    }
                    
                    // 检查错误率
                    const errorRate = report.error_rate || 0;
                    if (errorRate > 5) {
                        issues.push({
                            type: 'high_error_rate',
                            rate: errorRate,
                            severity: 'high'
                        });
                    }
                    
                    return issues.map(issue => ({json: issue}));
                    """
                },
                "position": [450, 300]
            },
            {
                "name": "记录质量问题",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8000/api/v1/logs/data-quality",
                    "method": "POST"
                },
                "position": [650, 300]
            }
        ]
    
    def _create_health_check_nodes(self) -> List[Dict]:
        """创建健康检查节点"""
        return [
            {
                "name": "系统健康检查",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8000/api/v1/ha/health",
                    "options": {"timeout": 15000}
                },
                "position": [250, 300]
            },
            {
                "name": "健康状态评估",
                "type": "n8n-nodes-base.function",
                "parameters": {
                    "functionCode": """
                    const health = items[0].json.data;
                    const alerts = [];
                    
                    // 检查整体状态
                    if (health.overall_status !== 'healthy') {
                        alerts.push({
                            type: 'system_unhealthy',
                            status: health.overall_status,
                            severity: 'critical'
                        });
                    }
                    
                    // 检查服务状态
                    for (const [service, info] of Object.entries(health.services)) {
                        if (info.availability < 80) {
                            alerts.push({
                                type: 'service_degraded',
                                service: service,
                                availability: info.availability,
                                severity: 'high'
                            });
                        }
                    }
                    
                    return alerts.map(alert => ({json: alert}));
                    """
                },
                "position": [450, 300]
            },
            {
                "name": "系统告警",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8000/api/v1/alerts/system",
                    "method": "POST"
                },
                "position": [650, 300]
            }
        ]
    
    def _create_trading_signal_nodes(self) -> List[Dict]:
        """创建交易信号处理节点"""
        return [
            {
                "name": "接收交易信号",
                "type": "n8n-nodes-base.webhook",
                "parameters": {
                    "path": "trading-signal",
                    "httpMethod": "POST"
                },
                "position": [250, 300]
            },
            {
                "name": "信号验证",
                "type": "n8n-nodes-base.function",
                "parameters": {
                    "functionCode": """
                    const signal = items[0].json;
                    
                    // 验证信号格式
                    const required_fields = ['symbol', 'action', 'confidence', 'timestamp'];
                    const missing_fields = required_fields.filter(field => !signal[field]);
                    
                    if (missing_fields.length > 0) {
                        throw new Error(`缺少必要字段: ${missing_fields.join(', ')}`);
                    }
                    
                    // 验证置信度
                    if (signal.confidence < 0.7) {
                        throw new Error(`信号置信度过低: ${signal.confidence}`);
                    }
                    
                    return items;
                    """
                },
                "position": [450, 300]
            },
            {
                "name": "执行交易决策",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8000/api/v1/trading/execute-signal",
                    "method": "POST"
                },
                "position": [650, 300]
            }
        ]
    
    def _create_risk_management_nodes(self) -> List[Dict]:
        """创建风险管理节点"""
        return [
            {
                "name": "获取持仓信息",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8000/api/v1/trading/positions",
                    "options": {"timeout": 10000}
                },
                "position": [250, 300]
            },
            {
                "name": "风险评估",
                "type": "n8n-nodes-base.function",
                "parameters": {
                    "functionCode": """
                    const positions = items[0].json.data;
                    const risks = [];
                    
                    let totalExposure = 0;
                    for (const position of positions) {
                        totalExposure += Math.abs(position.value || 0);
                        
                        // 单一持仓风险
                        const exposure_ratio = Math.abs(position.value) / position.account_balance;
                        if (exposure_ratio > 0.1) {  // 单仓超过10%
                            risks.push({
                                type: 'high_single_exposure',
                                symbol: position.symbol,
                                ratio: exposure_ratio,
                                severity: 'medium'
                            });
                        }
                    }
                    
                    // 总体风险
                    const total_ratio = totalExposure / positions[0]?.account_balance || 0;
                    if (total_ratio > 0.5) {  // 总风险超过50%
                        risks.push({
                            type: 'high_total_exposure',
                            ratio: total_ratio,
                            severity: 'high'
                        });
                    }
                    
                    return risks.map(risk => ({json: risk}));
                    """
                },
                "position": [450, 300]
            },
            {
                "name": "风险告警",
                "type": "n8n-nodes-base.httpRequest",
                "parameters": {
                    "url": "http://localhost:8000/api/v1/alerts/risk",
                    "method": "POST"
                },
                "position": [650, 300]
            }
        ]
    
    async def _create_workflow_from_template(self, template: Dict) -> Optional[str]:
        """从模板创建工作流"""
        workflow_data = {
            "name": template["name"],
            "description": template["description"],
            "active": True,
            "nodes": template["nodes"],
            "connections": self._generate_node_connections(template["nodes"])
        }
        
        try:
            async with N8NClient(self.n8n_base_url, self.api_key) as client:
                workflow_id = await client.create_workflow(workflow_data)
                
                if workflow_id:
                    # 创建工作流定义
                    workflow_def = WorkflowDefinition(
                        workflow_id=workflow_id,
                        name=template["name"],
                        description=template["description"],
                        trigger_type=template["trigger_type"],
                        enabled=True,
                        schedule=template.get("schedule"),
                        webhook_path=template.get("webhook_path"),
                        nodes=template["nodes"]
                    )
                    
                    self.workflows[workflow_id] = workflow_def
                    return workflow_id
                    
        except Exception as e:
            logger.error(f"从模板创建工作流失败: {e}")
            return None
    
    def _generate_node_connections(self, nodes: List[Dict]) -> Dict:
        """生成节点连接关系"""
        connections = {}
        
        # 简单的顺序连接
        for i in range(len(nodes) - 1):
            current_node = nodes[i]["name"]
            next_node = nodes[i + 1]["name"]
            
            connections[current_node] = {
                "main": [
                    [
                        {
                            "node": next_node,
                            "type": "main",
                            "index": 0
                        }
                    ]
                ]
            }
        
        return connections
    
    async def execute_workflow(self, workflow_id: str, input_data: Optional[Dict] = None) -> Optional[str]:
        """执行工作流"""
        try:
            if workflow_id not in self.workflows:
                logger.error(f"工作流不存在: {workflow_id}")
                return None
            
            workflow_def = self.workflows[workflow_id]
            
            # 检查工作流是否启用
            if not workflow_def.enabled:
                logger.warning(f"工作流已禁用: {workflow_id}")
                return None
            
            # 执行工作流
            async with N8NClient(self.n8n_base_url, self.api_key) as client:
                execution_id = await client.execute_workflow(workflow_id, input_data)
                
                if execution_id:
                    # 记录执行
                    execution = WorkflowExecution(
                        execution_id=execution_id,
                        workflow_id=workflow_id,
                        status=WorkflowStatus.RUNNING,
                        trigger_type=WorkflowTrigger.MANUAL,
                        start_time=datetime.utcnow(),
                        input_data=input_data
                    )
                    
                    self.executions[execution_id] = execution
                    logger.info(f"✅ 工作流执行启动: {workflow_def.name} [{execution_id}]")
                    
                return execution_id
                
        except Exception as e:
            logger.error(f"执行工作流失败 {workflow_id}: {e}")
            self.error_handler.record_error(workflow_id, "", str(e))
            return None
    
    async def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """获取执行状态"""
        if execution_id not in self.executions:
            return None
        
        execution = self.executions[execution_id]
        
        try:
            # 从N8N获取最新状态
            async with N8NClient(self.n8n_base_url, self.api_key) as client:
                status_data = await client.get_execution_status(execution_id)
                
                if status_data:
                    # 更新执行记录
                    execution.status = WorkflowStatus(status_data.get('status', 'running'))
                    if status_data.get('finishedAt'):
                        execution.end_time = datetime.fromisoformat(status_data['finishedAt'])
                        execution.duration = (execution.end_time - execution.start_time).total_seconds()
                    
                    execution.output_data = status_data.get('data')
                    execution.error_message = status_data.get('error', {}).get('message')
                    
                    # 如果有错误，记录
                    if execution.status == WorkflowStatus.ERROR and execution.error_message:
                        self.error_handler.record_error(
                            execution.workflow_id, 
                            execution_id, 
                            execution.error_message
                        )
                
        except Exception as e:
            logger.error(f"获取执行状态失败 {execution_id}: {e}")
        
        return execution
    
    async def _monitoring_loop(self):
        """监控循环"""
        logger.info("🔍 启动工作流监控循环")
        
        while self.monitoring_enabled:
            try:
                await asyncio.sleep(self.monitoring_interval)
                
                # 检查运行中的执行
                running_executions = [
                    exec_id for exec_id, execution in self.executions.items()
                    if execution.status == WorkflowStatus.RUNNING
                ]
                
                for exec_id in running_executions:
                    await self.get_execution_status(exec_id)
                
                # 清理旧的执行记录
                await self._cleanup_old_executions()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
        
        logger.info("🔍 工作流监控循环已停止")
    
    async def _cleanup_old_executions(self):
        """清理旧的执行记录"""
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        
        old_executions = [
            exec_id for exec_id, execution in self.executions.items()
            if execution.start_time < cutoff_time and execution.status not in [WorkflowStatus.RUNNING]
        ]
        
        for exec_id in old_executions:
            del self.executions[exec_id]
        
        if old_executions:
            logger.info(f"🧹 清理了{len(old_executions)}个旧执行记录")
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """获取工作流统计"""
        total_executions = len(self.executions)
        status_counts = {}
        
        for execution in self.executions.values():
            status = execution.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # 成功率
        success_count = status_counts.get(WorkflowStatus.SUCCESS.value, 0)
        success_rate = (success_count / total_executions * 100) if total_executions > 0 else 0
        
        # 错误统计
        error_stats = self.error_handler.get_error_stats()
        
        return {
            "total_workflows": len(self.workflows),
            "enabled_workflows": len([w for w in self.workflows.values() if w.enabled]),
            "total_executions": total_executions,
            "execution_status": status_counts,
            "success_rate": success_rate,
            "error_statistics": error_stats,
            "monitoring_enabled": self.monitoring_enabled
        }
    
    async def shutdown(self):
        """关闭工作流管理器"""
        try:
            self.monitoring_enabled = False
            
            if self.monitoring_task:
                self.monitoring_task.cancel()
                await self.monitoring_task
            
            logger.info("✅ N8N工作流管理器已关闭")
            
        except Exception as e:
            logger.error(f"关闭工作流管理器失败: {e}")

# 全局工作流管理器实例
n8n_workflow_manager = N8NWorkflowManager()