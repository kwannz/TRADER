#!/usr/bin/env python3
"""
⏰ 加密货币24/7实时监控系统
Crypto Real-time Monitoring System

功能特性:
- 24/7不间断监控4大核心风险指标
- 多级预警系统集成
- 实时数据流处理
- 自动化响应机制
- 监控状态持久化
- 异常恢复能力
- 多渠道告警推送

作者: Claude Code Assistant
创建时间: 2025-08-09
"""

import asyncio
import aiohttp
import json
import time
import sqlite3
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import threading
import signal
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
import warnings
warnings.filterwarnings('ignore')

# 导入我们之前创建的模块
try:
    from enhanced_whale_detection import EnhancedWhaleDetection, WhaleActivity
    from enhanced_fear_greed_index import EnhancedFearGreedIndex, MarketSentiment
    from advanced_alert_system import AdvancedAlertSystem, AlertLevel
except ImportError:
    # 如果导入失败，使用简化版本
    logging.warning("无法导入增强模块，使用简化版本")
    
    class WhaleActivity(Enum):
        ACCUMULATION = "积累"
        DISTRIBUTION = "分配"
        COORDINATION = "协调"
        DORMANT = "休眠"
    
    class MarketSentiment(Enum):
        EXTREME_FEAR = "极度恐惧"
        FEAR = "恐惧"
        NEUTRAL = "中性"
        GREED = "贪婪"
        EXTREME_GREED = "极度贪婪"
    
    class AlertLevel(Enum):
        CRITICAL = "CRITICAL"
        HIGH = "HIGH"
        MEDIUM = "MEDIUM"
        LOW = "LOW"
        INFO = "INFO"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('realtime_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MonitorStatus(Enum):
    """监控状态"""
    STARTING = "启动中"
    RUNNING = "运行中"
    PAUSED = "暂停"
    STOPPING = "停止中"
    STOPPED = "已停止"
    ERROR = "错误"

class NotificationChannel(Enum):
    """通知渠道"""
    CONSOLE = "控制台"
    EMAIL = "邮件"
    WEBHOOK = "Webhook"
    FILE = "文件"

@dataclass
class MonitoringConfig:
    """监控配置"""
    # 基本配置
    update_interval: int = 60          # 更新间隔(秒)
    data_retention_days: int = 30      # 数据保留天数
    max_retries: int = 3               # 最大重试次数
    
    # 数据源配置
    price_api_url: str = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    funding_api_url: str = "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT"
    
    # 阈值配置
    whale_threshold: float = 100000    # 巨鲸交易阈值
    fear_greed_low: float = 25         # 恐惧阈值
    fear_greed_high: float = 75        # 贪婪阈值
    
    # 通知配置
    enable_console: bool = True
    enable_file: bool = True
    enable_email: bool = False
    enable_webhook: bool = False
    
    webhook_url: Optional[str] = None
    email_config: Optional[Dict] = None

@dataclass
class MarketSnapshot:
    """市场快照"""
    timestamp: datetime
    price: float
    volume: float
    funding_rate: float
    whale_activity: str
    fear_greed_index: float
    sentiment: str
    liquidity_risk: float
    alert_level: str
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class DataCollector:
    """数据收集器"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def fetch_market_data(self) -> Dict[str, Any]:
        """获取市场数据"""
        if not self.session:
            raise RuntimeError("DataCollector not properly initialized")
        
        market_data = {}
        
        try:
            # 获取价格数据
            async with self.session.get(self.config.price_api_url) as response:
                if response.status == 200:
                    price_data = await response.json()
                    market_data['price'] = float(price_data['price'])
                else:
                    logger.warning(f"价格API返回状态码: {response.status}")
                    market_data['price'] = None
        
        except Exception as e:
            logger.error(f"获取价格数据失败: {e}")
            market_data['price'] = None
        
        try:
            # 获取资金费率数据
            async with self.session.get(self.config.funding_api_url) as response:
                if response.status == 200:
                    funding_data = await response.json()
                    market_data['funding_rate'] = float(funding_data.get('lastFundingRate', 0))
                else:
                    logger.warning(f"资金费率API返回状态码: {response.status}")
                    market_data['funding_rate'] = None
                    
        except Exception as e:
            logger.error(f"获取资金费率数据失败: {e}")
            market_data['funding_rate'] = None
        
        # 模拟其他数据(实际应用中应从真实API获取)
        market_data.update({
            'volume': np.random.exponential(10000),
            'timestamp': datetime.now()
        })
        
        return market_data

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_path: str = "monitoring_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建市场快照表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    price REAL,
                    volume REAL,
                    funding_rate REAL,
                    whale_activity TEXT,
                    fear_greed_index REAL,
                    sentiment TEXT,
                    liquidity_risk REAL,
                    alert_level TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建告警记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_level TEXT NOT NULL,
                    indicator TEXT NOT NULL,
                    message TEXT NOT NULL,
                    value REAL,
                    threshold REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建监控状态表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS monitor_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    status TEXT NOT NULL,
                    start_time TEXT,
                    end_time TEXT,
                    error_message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            logger.info("数据库初始化完成")
    
    def save_snapshot(self, snapshot: MarketSnapshot) -> bool:
        """保存市场快照"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO market_snapshots 
                    (timestamp, price, volume, funding_rate, whale_activity, 
                     fear_greed_index, sentiment, liquidity_risk, alert_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    snapshot.timestamp.isoformat(),
                    snapshot.price,
                    snapshot.volume,
                    snapshot.funding_rate,
                    snapshot.whale_activity,
                    snapshot.fear_greed_index,
                    snapshot.sentiment,
                    snapshot.liquidity_risk,
                    snapshot.alert_level
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"保存快照失败: {e}")
            return False
    
    def save_alert(self, alert: Dict[str, Any]) -> bool:
        """保存告警记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO alerts 
                    (timestamp, alert_level, indicator, message, value, threshold)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    alert.get('timestamp', datetime.now().isoformat()),
                    alert.get('level', 'INFO'),
                    alert.get('indicator', ''),
                    alert.get('message', ''),
                    alert.get('value'),
                    alert.get('threshold')
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"保存告警失败: {e}")
            return False
    
    def get_recent_data(self, hours: int = 24) -> List[Dict]:
        """获取最近数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM market_snapshots 
                    WHERE datetime(timestamp) > datetime('now', '-{} hours')
                    ORDER BY timestamp DESC
                '''.format(hours))
                
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"获取历史数据失败: {e}")
            return []
    
    def cleanup_old_data(self, retention_days: int):
        """清理旧数据"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 清理旧的市场快照
                cursor.execute('''
                    DELETE FROM market_snapshots 
                    WHERE datetime(timestamp) < datetime('now', '-{} days')
                '''.format(retention_days))
                
                # 清理旧的告警记录
                cursor.execute('''
                    DELETE FROM alerts 
                    WHERE datetime(timestamp) < datetime('now', '-{} days')
                '''.format(retention_days))
                
                conn.commit()
                logger.info(f"清理了超过{retention_days}天的旧数据")
                
        except Exception as e:
            logger.error(f"清理数据失败: {e}")

class NotificationManager:
    """通知管理器"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
    
    async def send_notification(self, 
                              level: AlertLevel, 
                              message: str, 
                              data: Optional[Dict] = None):
        """发送通知"""
        notification = {
            'timestamp': datetime.now().isoformat(),
            'level': level.value,
            'message': message,
            'data': data or {}
        }
        
        tasks = []
        
        if self.config.enable_console:
            tasks.append(self._send_console_notification(notification))
        
        if self.config.enable_file:
            tasks.append(self._send_file_notification(notification))
        
        if self.config.enable_webhook and self.config.webhook_url:
            tasks.append(self._send_webhook_notification(notification))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_console_notification(self, notification: Dict):
        """发送控制台通知"""
        level_colors = {
            'CRITICAL': '\033[91m',  # 红色
            'HIGH': '\033[93m',      # 黄色
            'MEDIUM': '\033[94m',    # 蓝色
            'LOW': '\033[92m',       # 绿色
            'INFO': '\033[96m'       # 青色
        }
        
        reset_color = '\033[0m'
        level = notification['level']
        color = level_colors.get(level, '')
        
        print(f"\n{color}🚨 [{level}] {notification['message']}{reset_color}")
        print(f"⏰ 时间: {notification['timestamp']}")
        
        if notification['data']:
            print("📊 详细数据:")
            for key, value in notification['data'].items():
                print(f"   {key}: {value}")
    
    async def _send_file_notification(self, notification: Dict):
        """发送文件通知"""
        try:
            log_file = "alerts.log"
            with open(log_file, 'a', encoding='utf-8') as f:
                log_line = json.dumps(notification, ensure_ascii=False)
                f.write(f"{log_line}\n")
        except Exception as e:
            logger.error(f"文件通知失败: {e}")
    
    async def _send_webhook_notification(self, notification: Dict):
        """发送Webhook通知"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.webhook_url, 
                    json=notification,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Webhook响应状态码: {response.status}")
        except Exception as e:
            logger.error(f"Webhook通知失败: {e}")

class RiskAnalyzer:
    """风险分析器"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        
        # 初始化分析组件(如果可用)
        try:
            self.whale_detector = EnhancedWhaleDetection(
                min_whale_size=config.whale_threshold
            )
            self.fear_greed_analyzer = EnhancedFearGreedIndex()
        except:
            self.whale_detector = None
            self.fear_greed_analyzer = None
            logger.warning("使用简化版风险分析")
    
    def analyze_market_data(self, market_data: Dict[str, Any]) -> MarketSnapshot:
        """分析市场数据"""
        timestamp = market_data.get('timestamp', datetime.now())
        price = market_data.get('price', 0)
        volume = market_data.get('volume', 0)
        funding_rate = market_data.get('funding_rate', 0)
        
        # 简化版分析(实际应用中使用完整的分析逻辑)
        whale_activity = self._analyze_whale_activity(volume, price)
        fear_greed_index, sentiment = self._analyze_fear_greed(price, funding_rate)
        liquidity_risk = self._analyze_liquidity_risk(volume, price)
        alert_level = self._determine_alert_level(
            whale_activity, fear_greed_index, liquidity_risk
        )
        
        return MarketSnapshot(
            timestamp=timestamp,
            price=price,
            volume=volume,
            funding_rate=funding_rate,
            whale_activity=whale_activity,
            fear_greed_index=fear_greed_index,
            sentiment=sentiment,
            liquidity_risk=liquidity_risk,
            alert_level=alert_level
        )
    
    def _analyze_whale_activity(self, volume: float, price: float) -> str:
        """分析巨鲸活动"""
        # 简化版巨鲸分析
        avg_trade_size = price * volume / 100  # 简化计算
        
        if avg_trade_size > self.config.whale_threshold * 2:
            return WhaleActivity.COORDINATION.value
        elif avg_trade_size > self.config.whale_threshold:
            return WhaleActivity.ACCUMULATION.value
        else:
            return WhaleActivity.DORMANT.value
    
    def _analyze_fear_greed(self, price: float, funding_rate: float) -> tuple:
        """分析恐惧贪婪指数"""
        # 简化版恐惧贪婪分析
        base_score = 50
        
        # 基于资金费率调整
        if funding_rate > 0.001:  # 高资金费率表示贪婪
            score_adjustment = min(funding_rate * 10000, 30)
            base_score += score_adjustment
        elif funding_rate < -0.001:  # 负资金费率表示恐惧
            score_adjustment = min(abs(funding_rate) * 10000, 30)
            base_score -= score_adjustment
        
        # 随机波动模拟
        base_score += np.random.normal(0, 10)
        fear_greed_index = np.clip(base_score, 0, 100)
        
        # 确定情绪状态
        if fear_greed_index <= 20:
            sentiment = MarketSentiment.EXTREME_FEAR.value
        elif fear_greed_index <= 40:
            sentiment = MarketSentiment.FEAR.value
        elif fear_greed_index <= 60:
            sentiment = MarketSentiment.NEUTRAL.value
        elif fear_greed_index <= 80:
            sentiment = MarketSentiment.GREED.value
        else:
            sentiment = MarketSentiment.EXTREME_GREED.value
        
        return fear_greed_index, sentiment
    
    def _analyze_liquidity_risk(self, volume: float, price: float) -> float:
        """分析流动性风险"""
        # 简化版流动性风险分析
        if volume < 1000:  # 低成交量
            return 80.0
        elif volume > 50000:  # 高成交量
            return 20.0
        else:
            return 50.0
    
    def _determine_alert_level(self, 
                             whale_activity: str, 
                             fear_greed_index: float,
                             liquidity_risk: float) -> str:
        """确定告警级别"""
        
        # 极端情况
        if (fear_greed_index <= 10 or fear_greed_index >= 90 or
            whale_activity == WhaleActivity.COORDINATION.value or
            liquidity_risk >= 80):
            return AlertLevel.CRITICAL.value
        
        # 高风险情况
        elif (fear_greed_index <= 25 or fear_greed_index >= 75 or
              whale_activity == WhaleActivity.ACCUMULATION.value or
              liquidity_risk >= 60):
            return AlertLevel.HIGH.value
        
        # 中等风险
        elif (fear_greed_index <= 35 or fear_greed_index >= 65 or
              liquidity_risk >= 40):
            return AlertLevel.MEDIUM.value
        
        # 低风险
        elif liquidity_risk >= 30:
            return AlertLevel.LOW.value
        
        # 信息级别
        else:
            return AlertLevel.INFO.value

class RealTimeMonitor:
    """实时监控主类"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.status = MonitorStatus.STOPPED
        self.db_manager = DatabaseManager()
        self.notification_manager = NotificationManager(config)
        self.risk_analyzer = RiskAnalyzer(config)
        
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # 设置信号处理器用于优雅关闭
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("实时监控系统初始化完成")
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        logger.info(f"接收到信号 {signum}，开始优雅关闭...")
        asyncio.create_task(self.stop())
    
    async def start(self):
        """启动监控"""
        if self.status == MonitorStatus.RUNNING:
            logger.warning("监控系统已在运行中")
            return
        
        self.status = MonitorStatus.STARTING
        logger.info("🚀 启动加密货币24/7实时监控系统...")
        
        await self.notification_manager.send_notification(
            AlertLevel.INFO,
            "📡 实时监控系统启动",
            {
                'update_interval': f'{self.config.update_interval}秒',
                'whale_threshold': f'${self.config.whale_threshold:,}',
                'fear_greed_thresholds': f'{self.config.fear_greed_low}-{self.config.fear_greed_high}'
            }
        )
        
        self._running = True
        self.status = MonitorStatus.RUNNING
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
        try:
            await self._monitor_task
        except asyncio.CancelledError:
            logger.info("监控任务被取消")
        except Exception as e:
            logger.error(f"监控任务异常: {e}")
            self.status = MonitorStatus.ERROR
        finally:
            self.status = MonitorStatus.STOPPED
    
    async def stop(self):
        """停止监控"""
        if self.status != MonitorStatus.RUNNING:
            return
        
        self.status = MonitorStatus.STOPPING
        logger.info("🛑 停止实时监控系统...")
        
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        await self.notification_manager.send_notification(
            AlertLevel.INFO,
            "🛑 实时监控系统已停止"
        )
        
        self.status = MonitorStatus.STOPPED
        logger.info("监控系统已停止")
    
    async def pause(self):
        """暂停监控"""
        if self.status == MonitorStatus.RUNNING:
            self.status = MonitorStatus.PAUSED
            logger.info("⏸️ 监控系统已暂停")
    
    async def resume(self):
        """恢复监控"""
        if self.status == MonitorStatus.PAUSED:
            self.status = MonitorStatus.RUNNING
            logger.info("▶️ 监控系统已恢复")
    
    async def _monitor_loop(self):
        """主监控循环"""
        logger.info("开始主监控循环...")
        
        consecutive_errors = 0
        
        while self._running:
            try:
                if self.status == MonitorStatus.PAUSED:
                    await asyncio.sleep(1)
                    continue
                
                # 收集数据并分析
                async with DataCollector(self.config) as collector:
                    market_data = await collector.fetch_market_data()
                
                if market_data.get('price') is None:
                    consecutive_errors += 1
                    if consecutive_errors >= self.config.max_retries:
                        await self.notification_manager.send_notification(
                            AlertLevel.CRITICAL,
                            f"🚨 连续{consecutive_errors}次数据获取失败，可能存在网络问题"
                        )
                        consecutive_errors = 0  # 重置计数器
                else:
                    consecutive_errors = 0  # 重置错误计数
                
                # 分析市场数据
                snapshot = self.risk_analyzer.analyze_market_data(market_data)
                
                # 保存快照
                if not self.db_manager.save_snapshot(snapshot):
                    logger.warning("保存快照失败")
                
                # 检查是否需要发送告警
                await self._check_alerts(snapshot)
                
                # 定期清理旧数据
                if datetime.now().minute == 0:  # 每小时清理一次
                    self.db_manager.cleanup_old_data(self.config.data_retention_days)
                
                # 等待下次更新
                await asyncio.sleep(self.config.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                consecutive_errors += 1
                
                if consecutive_errors >= self.config.max_retries:
                    await self.notification_manager.send_notification(
                        AlertLevel.CRITICAL,
                        f"🚨 监控系统发生严重错误: {str(e)}"
                    )
                    break
                
                await asyncio.sleep(5)  # 错误后短暂等待
    
    async def _check_alerts(self, snapshot: MarketSnapshot):
        """检查告警条件"""
        alerts_to_send = []
        
        # 检查恐惧贪婪指数
        if snapshot.fear_greed_index <= self.config.fear_greed_low:
            alerts_to_send.append({
                'level': AlertLevel.HIGH,
                'message': f'😱 极度恐惧信号: 恐惧贪婪指数 {snapshot.fear_greed_index:.1f}',
                'data': {
                    'indicator': '恐惧贪婪指数',
                    'value': snapshot.fear_greed_index,
                    'threshold': self.config.fear_greed_low,
                    'recommendation': '可能是买入机会，但需谨慎'
                }
            })
        elif snapshot.fear_greed_index >= self.config.fear_greed_high:
            alerts_to_send.append({
                'level': AlertLevel.HIGH,
                'message': f'🤑 极度贪婪信号: 恐惧贪婪指数 {snapshot.fear_greed_index:.1f}',
                'data': {
                    'indicator': '恐惧贪婪指数',
                    'value': snapshot.fear_greed_index,
                    'threshold': self.config.fear_greed_high,
                    'recommendation': '可能是卖出时机，考虑获利了结'
                }
            })
        
        # 检查巨鲸活动
        if snapshot.whale_activity == WhaleActivity.COORDINATION.value:
            alerts_to_send.append({
                'level': AlertLevel.CRITICAL,
                'message': f'🐋 巨鲸协调活动检测到!',
                'data': {
                    'indicator': '巨鲸活动',
                    'value': snapshot.whale_activity,
                    'recommendation': '市场可能面临巨大波动，建议减仓观望'
                }
            })
        elif snapshot.whale_activity == WhaleActivity.ACCUMULATION.value:
            alerts_to_send.append({
                'level': AlertLevel.MEDIUM,
                'message': f'🐋 检测到巨鲸积累行为',
                'data': {
                    'indicator': '巨鲸活动',
                    'value': snapshot.whale_activity,
                    'recommendation': '可能有大资金看好后市，可考虑跟随'
                }
            })
        
        # 检查流动性风险
        if snapshot.liquidity_risk >= 80:
            alerts_to_send.append({
                'level': AlertLevel.HIGH,
                'message': f'💧 高流动性风险警告: {snapshot.liquidity_risk:.1f}%',
                'data': {
                    'indicator': '流动性风险',
                    'value': snapshot.liquidity_risk,
                    'threshold': 80,
                    'recommendation': '流动性不足，大额交易需谨慎'
                }
            })
        
        # 发送告警
        for alert in alerts_to_send:
            await self.notification_manager.send_notification(
                alert['level'], 
                alert['message'], 
                alert['data']
            )
            
            # 保存到数据库
            self.db_manager.save_alert({
                'timestamp': snapshot.timestamp.isoformat(),
                'level': alert['level'].value,
                'indicator': alert['data'].get('indicator', ''),
                'message': alert['message'],
                'value': alert['data'].get('value'),
                'threshold': alert['data'].get('threshold')
            })
    
    def get_status_report(self) -> Dict[str, Any]:
        """获取状态报告"""
        recent_data = self.db_manager.get_recent_data(1)  # 最近1小时
        
        return {
            'monitor_status': self.status.value,
            'config': asdict(self.config),
            'recent_snapshots': len(recent_data),
            'last_update': recent_data[0]['timestamp'] if recent_data else None,
            'database_path': self.db_manager.db_path
        }

async def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║             🚨 加密货币24/7实时监控系统 v2.0                    ║  
║                                                              ║
║  功能特性:                                                    ║
║  🔍 实时数据收集        📊 多维风险分析                       ║
║  🚨 智能告警系统        💾 数据持久化存储                     ║
║  📱 多渠道通知推送      🛡️ 异常恢复机制                      ║
║                                                              ║
║  作者: Claude Code Assistant                                 ║
║  版本: v2.0 | 创建时间: 2025-08-09                           ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 创建配置
    config = MonitoringConfig(
        update_interval=30,  # 30秒更新一次
        whale_threshold=100000,
        fear_greed_low=25,
        fear_greed_high=75,
        enable_console=True,
        enable_file=True
    )
    
    # 创建监控器
    monitor = RealTimeMonitor(config)
    
    try:
        print("⚡ 启动实时监控系统...")
        print("📋 监控配置:")
        print(f"   - 更新间隔: {config.update_interval}秒")
        print(f"   - 巨鲸阈值: ${config.whale_threshold:,}")
        print(f"   - 恐惧贪婪阈值: {config.fear_greed_low}-{config.fear_greed_high}")
        print("   - 通知渠道: 控制台 + 文件")
        print("\n🔥 系统运行中... (Ctrl+C 停止)\n")
        
        # 启动监控
        await monitor.start()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  接收到停止信号...")
    except Exception as e:
        logger.error(f"系统运行异常: {e}")
    finally:
        await monitor.stop()
        print("✅ 监控系统已安全关闭")

if __name__ == "__main__":
    # 运行主程序
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 再见!")
    except Exception as e:
        print(f"❌ 系统启动失败: {e}")
        sys.exit(1)