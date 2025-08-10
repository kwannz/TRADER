"""
Agent通信框架
基于Redis和WebSocket的高性能Agent间通信系统
支持发布/订阅、点对点消息、广播等通信模式
"""

import asyncio
import json
import redis.asyncio as redis
import websockets
from typing import Dict, List, Set, Optional, Callable, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import weakref

from .base_agent import AgentMessage, BaseAgent

@dataclass
class CommunicationConfig:
    """通信配置"""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    websocket_host: str = "localhost"
    websocket_port: int = 8765
    message_ttl: int = 300  # 消息生存时间(秒)
    max_retries: int = 3
    retry_delay: float = 1.0
    heartbeat_interval: int = 30
    compression: bool = True

class RedisMessageBroker:
    """基于Redis的消息代理"""
    
    def __init__(self, config: CommunicationConfig):
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.PubSub] = None
        self.subscribed_channels: Set[str] = set()
        self.logger = logging.getLogger("RedisMessageBroker")
        
    async def initialize(self) -> bool:
        """初始化Redis连接"""
        try:
            # 创建Redis连接
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                decode_responses=True,
                retry_on_timeout=True,
                retry_on_error=[redis.ConnectionError, redis.TimeoutError]
            )
            
            # 测试连接
            await self.redis_client.ping()
            
            # 创建发布/订阅对象
            self.pubsub = self.redis_client.pubsub()
            
            self.logger.info("✅ Redis消息代理初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Redis消息代理初始化失败: {e}")
            return False
    
    async def send_message(self, message: AgentMessage) -> bool:
        """发送消息"""
        if not self.redis_client:
            return False
            
        try:
            message_data = json.dumps(message.to_dict())
            
            if message.receiver_id == "*":
                # 广播消息
                channel = "agent_broadcast"
            else:
                # 点对点消息
                channel = f"agent_{message.receiver_id}"
            
            # 发布消息
            await self.redis_client.publish(channel, message_data)
            
            # 存储消息（用于持久化和重试）
            await self.redis_client.setex(
                f"message_{message.id}",
                self.config.message_ttl,
                message_data
            )
            
            self.logger.debug(f"📤 发送消息到频道 {channel}")
            return True
            
        except Exception as e:
            self.logger.error(f"发送消息失败: {e}")
            return False
    
    async def subscribe_agent(self, agent_id: str) -> bool:
        """订阅Agent消息"""
        if not self.pubsub:
            return False
            
        try:
            # 订阅点对点频道
            personal_channel = f"agent_{agent_id}"
            await self.pubsub.subscribe(personal_channel)
            self.subscribed_channels.add(personal_channel)
            
            # 订阅广播频道
            broadcast_channel = "agent_broadcast"
            if broadcast_channel not in self.subscribed_channels:
                await self.pubsub.subscribe(broadcast_channel)
                self.subscribed_channels.add(broadcast_channel)
            
            self.logger.info(f"📡 Agent {agent_id} 订阅消息频道")
            return True
            
        except Exception as e:
            self.logger.error(f"订阅失败: {e}")
            return False
    
    async def listen_messages(self, message_handler: Callable[[AgentMessage], None]):
        """监听消息"""
        if not self.pubsub:
            return
            
        self.logger.info("👂 开始监听消息...")
        
        while True:
            try:
                # 获取消息（带超时）
                message = await asyncio.wait_for(
                    self.pubsub.get_message(ignore_subscribe_messages=True),
                    timeout=1.0
                )
                
                if message and message['type'] == 'message':
                    try:
                        # 解析消息
                        message_data = json.loads(message['data'])
                        agent_message = AgentMessage.from_dict(message_data)
                        
                        # 调用处理器
                        await message_handler(agent_message)
                        
                    except Exception as e:
                        self.logger.error(f"处理消息失败: {e}")
                
            except asyncio.TimeoutError:
                # 超时是正常的，继续循环
                continue
            except Exception as e:
                self.logger.error(f"消息监听错误: {e}")
                await asyncio.sleep(1)
    
    async def get_message(self, message_id: str) -> Optional[AgentMessage]:
        """获取持久化消息"""
        if not self.redis_client:
            return None
            
        try:
            message_data = await self.redis_client.get(f"message_{message_id}")
            if message_data:
                return AgentMessage.from_dict(json.loads(message_data))
            return None
        except Exception as e:
            self.logger.error(f"获取消息失败: {e}")
            return None
    
    async def cleanup(self):
        """清理资源"""
        try:
            if self.pubsub:
                await self.pubsub.close()
            if self.redis_client:
                await self.redis_client.close()
        except Exception as e:
            self.logger.error(f"清理资源失败: {e}")

class WebSocketMessageBroker:
    """基于WebSocket的实时消息代理"""
    
    def __init__(self, config: CommunicationConfig):
        self.config = config
        self.websocket_server = None
        self.connected_agents: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.logger = logging.getLogger("WebSocketMessageBroker")
        
    async def start_server(self):
        """启动WebSocket服务器"""
        try:
            self.websocket_server = await websockets.serve(
                self.handle_client,
                self.config.websocket_host,
                self.config.websocket_port
            )
            
            self.logger.info(f"🌐 WebSocket服务器启动: ws://{self.config.websocket_host}:{self.config.websocket_port}")
            
        except Exception as e:
            self.logger.error(f"WebSocket服务器启动失败: {e}")
    
    async def handle_client(self, websocket, path):
        """处理WebSocket客户端连接"""
        agent_id = None
        try:
            self.logger.info(f"新的WebSocket连接: {websocket.remote_address}")
            
            # 等待Agent ID注册
            registration_msg = await websocket.recv()
            registration_data = json.loads(registration_msg)
            
            if registration_data.get("type") == "register":
                agent_id = registration_data.get("agent_id")
                if agent_id:
                    self.connected_agents[agent_id] = websocket
                    await websocket.send(json.dumps({"type": "registered", "agent_id": agent_id}))
                    self.logger.info(f"Agent {agent_id} 注册WebSocket连接")
                else:
                    await websocket.send(json.dumps({"type": "error", "message": "Invalid agent_id"}))
                    return
            else:
                await websocket.send(json.dumps({"type": "error", "message": "Registration required"}))
                return
            
            # 监听消息
            async for message in websocket:
                try:
                    message_data = json.loads(message)
                    if message_data.get("type") == "message":
                        agent_message = AgentMessage.from_dict(message_data["data"])
                        await self.route_message(agent_message)
                        
                except Exception as e:
                    self.logger.error(f"处理WebSocket消息失败: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            self.logger.error(f"WebSocket连接处理错误: {e}")
        finally:
            if agent_id and agent_id in self.connected_agents:
                del self.connected_agents[agent_id]
                self.logger.info(f"Agent {agent_id} 断开WebSocket连接")
    
    async def route_message(self, message: AgentMessage):
        """路由消息"""
        try:
            message_data = {
                "type": "message",
                "data": message.to_dict()
            }
            
            if message.receiver_id == "*":
                # 广播消息
                for websocket in self.connected_agents.values():
                    try:
                        await websocket.send(json.dumps(message_data))
                    except Exception:
                        pass  # 忽略发送失败的连接
            else:
                # 点对点消息
                websocket = self.connected_agents.get(message.receiver_id)
                if websocket:
                    await websocket.send(json.dumps(message_data))
                    
        except Exception as e:
            self.logger.error(f"路由消息失败: {e}")
    
    async def stop_server(self):
        """停止WebSocket服务器"""
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()

class AgentCommunicationClient:
    """Agent通信客户端"""
    
    def __init__(self, agent: BaseAgent, config: CommunicationConfig):
        self.agent = agent
        self.config = config
        self.redis_broker = RedisMessageBroker(config)
        self.websocket_client = None
        self.message_handlers: List[Callable] = []
        self.logger = logging.getLogger(f"AgentComm-{agent.agent_id}")
        self._running = False
        
    async def initialize(self) -> bool:
        """初始化通信客户端"""
        try:
            # 初始化Redis代理
            if not await self.redis_broker.initialize():
                return False
            
            # 订阅消息
            if not await self.redis_broker.subscribe_agent(self.agent.agent_id):
                return False
            
            # 启动消息监听
            asyncio.create_task(self._message_listening_loop())
            
            # 连接WebSocket（可选）
            asyncio.create_task(self._connect_websocket())
            
            # 启动心跳
            asyncio.create_task(self._heartbeat_loop())
            
            self._running = True
            self.logger.info("✅ Agent通信客户端初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 通信客户端初始化失败: {e}")
            return False
    
    async def send_message(self, message: AgentMessage) -> bool:
        """发送消息"""
        try:
            # 主要通过Redis发送
            success = await self.redis_broker.send_message(message)
            
            # 如果WebSocket可用，也通过WebSocket发送（实时性更好）
            if self.websocket_client:
                try:
                    message_data = {
                        "type": "message",
                        "data": message.to_dict()
                    }
                    await self.websocket_client.send(json.dumps(message_data))
                except Exception:
                    pass  # WebSocket发送失败不影响主要通信
            
            return success
            
        except Exception as e:
            self.logger.error(f"发送消息失败: {e}")
            return False
    
    async def _message_listening_loop(self):
        """消息监听循环"""
        await self.redis_broker.listen_messages(self._handle_received_message)
    
    async def _handle_received_message(self, message: AgentMessage):
        """处理接收到的消息"""
        try:
            # 过滤发给自己的消息
            if message.receiver_id == self.agent.agent_id or message.receiver_id == "*":
                await self.agent.receive_message(message)
                
        except Exception as e:
            self.logger.error(f"处理接收消息失败: {e}")
    
    async def _connect_websocket(self):
        """连接WebSocket服务器"""
        try:
            uri = f"ws://{self.config.websocket_host}:{self.config.websocket_port}"
            self.websocket_client = await websockets.connect(uri)
            
            # 注册Agent
            registration = {
                "type": "register",
                "agent_id": self.agent.agent_id
            }
            await self.websocket_client.send(json.dumps(registration))
            
            # 等待注册确认
            response = await self.websocket_client.recv()
            response_data = json.loads(response)
            
            if response_data.get("type") == "registered":
                self.logger.info("✅ WebSocket连接已建立")
            else:
                self.logger.error("WebSocket注册失败")
                self.websocket_client = None
                
        except Exception as e:
            self.logger.warning(f"WebSocket连接失败: {e}")
            self.websocket_client = None
    
    async def _heartbeat_loop(self):
        """心跳循环"""
        while self._running:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                # 发送心跳消息
                heartbeat = AgentMessage(
                    receiver_id="*",
                    message_type="heartbeat",
                    content={
                        "agent_id": self.agent.agent_id,
                        "status": self.agent.status.value,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                await self.send_message(heartbeat)
                
            except Exception as e:
                self.logger.error(f"心跳发送失败: {e}")
    
    async def cleanup(self):
        """清理资源"""
        try:
            self._running = False
            
            if self.websocket_client:
                await self.websocket_client.close()
            
            await self.redis_broker.cleanup()
            
        except Exception as e:
            self.logger.error(f"清理通信资源失败: {e}")

class AgentRegistry:
    """Agent注册中心"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.logger = logging.getLogger("AgentRegistry")
        
    async def register_agent(self, agent: BaseAgent) -> bool:
        """注册Agent"""
        try:
            agent_info = {
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type.value,
                "capabilities": [cap.value for cap in agent.capabilities],
                "status": agent.status.value,
                "registered_at": datetime.utcnow().isoformat(),
                "last_heartbeat": datetime.utcnow().isoformat()
            }
            
            # 存储Agent信息
            await self.redis_client.hset(
                "agent_registry",
                agent.agent_id,
                json.dumps(agent_info)
            )
            
            # 设置过期时间
            await self.redis_client.expire("agent_registry", 3600)
            
            self.logger.info(f"📝 注册Agent: {agent.agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"注册Agent失败: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """注销Agent"""
        try:
            await self.redis_client.hdel("agent_registry", agent_id)
            self.logger.info(f"🗑️ 注销Agent: {agent_id}")
            return True
        except Exception as e:
            self.logger.error(f"注销Agent失败: {e}")
            return False
    
    async def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """获取Agent信息"""
        try:
            agent_data = await self.redis_client.hget("agent_registry", agent_id)
            if agent_data:
                return json.loads(agent_data)
            return None
        except Exception as e:
            self.logger.error(f"获取Agent信息失败: {e}")
            return None
    
    async def list_agents(self) -> List[Dict[str, Any]]:
        """列出所有Agent"""
        try:
            agents_data = await self.redis_client.hgetall("agent_registry")
            agents = []
            for agent_data in agents_data.values():
                agents.append(json.loads(agent_data))
            return agents
        except Exception as e:
            self.logger.error(f"列出Agent失败: {e}")
            return []
    
    async def find_agents_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """根据能力查找Agent"""
        agents = await self.list_agents()
        return [
            agent for agent in agents
            if capability in agent.get("capabilities", [])
        ]
    
    async def update_agent_heartbeat(self, agent_id: str) -> bool:
        """更新Agent心跳"""
        try:
            agent_data = await self.redis_client.hget("agent_registry", agent_id)
            if agent_data:
                agent_info = json.loads(agent_data)
                agent_info["last_heartbeat"] = datetime.utcnow().isoformat()
                
                await self.redis_client.hset(
                    "agent_registry",
                    agent_id,
                    json.dumps(agent_info)
                )
                return True
            return False
        except Exception as e:
            self.logger.error(f"更新心跳失败: {e}")
            return False

# 全局通信管理器
class CommunicationManager:
    """通信管理器"""
    
    def __init__(self, config: CommunicationConfig):
        self.config = config
        self.redis_broker = RedisMessageBroker(config)
        self.websocket_broker = WebSocketMessageBroker(config)
        self.agent_registry = None
        self.logger = logging.getLogger("CommunicationManager")
        
    async def initialize(self) -> bool:
        """初始化通信管理器"""
        try:
            # 初始化Redis
            if not await self.redis_broker.initialize():
                return False
            
            # 创建Agent注册中心
            self.agent_registry = AgentRegistry(self.redis_broker.redis_client)
            
            # 启动WebSocket服务器
            await self.websocket_broker.start_server()
            
            self.logger.info("✅ 通信管理器初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 通信管理器初始化失败: {e}")
            return False
    
    def create_agent_client(self, agent: BaseAgent) -> AgentCommunicationClient:
        """为Agent创建通信客户端"""
        client = AgentCommunicationClient(agent, self.config)
        # 设置到Agent中
        agent.communication_client = client
        return client
    
    async def shutdown(self):
        """关闭通信管理器"""
        try:
            await self.websocket_broker.stop_server()
            await self.redis_broker.cleanup()
            self.logger.info("✅ 通信管理器已关闭")
        except Exception as e:
            self.logger.error(f"关闭通信管理器失败: {e}")

# 创建默认通信管理器实例
default_config = CommunicationConfig()
communication_manager = CommunicationManager(default_config)