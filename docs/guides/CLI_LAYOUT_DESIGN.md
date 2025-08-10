# 量化交易系统 CLI 界面布局设计

## 1. 主仪表盘布局 (120x40 终端窗口)

### 完整布局预览
```
┌─────────────────────── 量化交易系统 v1.0 - AI驱动 ──────────────────────────────────────────────────────────────────────┐
│ 🔗 OKX: ✅ 连接 │ 🔗 Binance: ✅ 连接 │ 📊 账户: 500.00 USDT │ 💰 PnL: +12.50(+2.5%) │ ⏰ 2025-01-27 15:30:25 │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 📈 实时行情 (WebSocket Live)         │ 🤖 AI智能分析                     │ 📰 财经快讯                        │
│ ┌─ 主要币种 ──────────────────────┐  │ ┌─ 市场情绪 ───────────────────┐  │ ┌─ 实时新闻 ──────────────────┐   │
│ │ BTC/USDT  $45,123.50 ▲ +2.1%  │  │ │ 😊 情绪指数: 75/100 (乐观)   │  │ │ 🔥 比特币突破关键阻力位...   │   │
│ │ ETH/USDT  $2,890.25  ▼ -0.8%  │  │ │ 📈 AI预测: ↗️ 短期看涨       │  │ │ 📊 美联储会议纪要公布...     │   │
│ │ BNB/USDT  $315.80    ▲ +1.5%  │  │ │ ⚠️  恐慌指数: 32 (恐慌)      │  │ │ 💼 机构大量买入ETH...       │   │
│ │ SOL/USDT  $98.45     ▲ +3.2%  │  │ │ 🎯 推荐操作: 逢低买入         │  │ │ ───────────────────────── │   │
│ │ ──────────────────────────── │  │ │ ─────────────────────────── │  │ │ 更新: 15:30:20 (10秒前)    │   │
│ │ 📊 24H成交量: 2.5B USDT      │  │ │ 📊 持仓分析: 多头占比 65%     │  │ └─────────────────────────── │   │
│ └─────────────────────────────┘  │ └─────────────────────────────┘  │                                   │
│                                  │                                  │                                   │
├──────────────────────────────────┼──────────────────────────────────┼───────────────────────────────────┤
│ 🚀 策略运行状态                   │ 🛡️ 风控监控                      │ 📊 实时K线图 (ASCII)             │
│ ┌─ 网格策略 ─────────────────┐   │ ┌─ 风险指标 ─────────────────┐   │ ┌─ BTC/USDT 5分钟 ──────────┐  │
│ │ 状态: 🟢 运行中 (2h 15m)   │   │ │ 💸 最大亏损: 45/300 USDT   │   │ │        /\                  │  │
│ │ 收益: +5.2% (+26.0 USDT)  │   │ │ 📈 仓位使用: 60% (安全)     │   │ │    /\  /  \    /\           │  │
│ │ 持仓: 0.1 BTC             │   │ │ ⚡ 交易频率: 12/h (正常)    │   │ │   /  \/    \  /  \          │  │
│ │ 网格区间: $44k-$46k       │   │ │ 🎯 成功率: 68% (良好)       │   │ │  /          \/    \    /\   │  │
│ └───────────────────────────┘   │ └───────────────────────────┘   │ │ /                  \  /  \  │  │
│ ┌─ DCA策略 ─────────────────┐   │ ┌─ AI风险预警 ───────────────┐   │ │/                    \/    \ │  │
│ │ 状态: 🟡 暂停 (手动)       │   │ │ ✅ 清算风险: 低            │   │ │                           \│  │
│ │ 收益: +2.1% (+10.5 USDT)  │   │ │ 🔍 异常检测: 无            │   │ │ 45.8k  45.9k  46.0k  46.1k │  │
│ │ 持仓: 0.05 ETH            │   │ │ 📊 情绪波动: 稳定          │   │ └───────────────────────────┘  │
│ │ 下次买入: $2,850          │   │ │ 💡 建议: 继续执行策略       │   │                               │
│ └───────────────────────────┘   │ └───────────────────────────┘   │                               │
│ ┌─ AI策略 ──────────────────┐   │                                  │                               │
│ │ 状态: 🟢 运行中 (1h 5m)    │   │                                  │                               │
│ │ 收益: +8.5% (+42.5 USDT)  │   │                                  │                               │
│ │ 信号: Gemini生成 (多空)   │   │                                  │                               │
│ │ 置信度: 87% (高)          │   │                                  │                               │
│ └───────────────────────────┘   │                                  │                               │
├──────────────────────────────────┴──────────────────────────────────┴───────────────────────────────────┤
│ 📝 实时系统日志                                                                   [清空] [导出] [筛选]      │
│ 15:30:25 [AI] 🧠 DeepSeek分析: 市场情绪从恐慌转为乐观，建议增加仓位...                                     │
│ 15:30:22 [交易] 💰 网格策略 买入 0.001 BTC @ $45,120.50 (OKX)                                           │
│ 15:30:20 [新闻] 📰 金十快讯: 比特币突破关键阻力位$45,000，成交量激增                                     │
│ 15:30:18 [风控] ✅ 风险检查通过: 当前总仓位60%，在安全范围内                                             │
│ 15:30:15 [WebSocket] 🔄 Binance连接重新建立，数据流恢复正常                                             │
│ 15:30:12 [AI] 🎯 Gemini策略生成: 检测到突破信号，执行多头开仓                                           │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 🎮 快捷键: [1]仪表盘 [2]策略管理 [3]AI助手 [4]因子发现 [5]交易记录 [6]设置 [Q]退出 [H]帮助 [R]刷新数据       │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

## 2. WebSocket实时数据流架构

### 2.1 OKX WebSocket连接
```python
# OKX WebSocket配置
OKX_WS_CONFIG = {
    "public_url": "wss://ws.okx.com:8443/ws/v5/public",
    "private_url": "wss://ws.okx.com:8443/ws/v5/private",
    "channels": [
        "tickers",      # 24hr ticker数据
        "books",        # 深度数据
        "trades",       # 成交数据
        "funding-rate", # 资金费率
        "mark-price"    # 标记价格
    ],
    "symbols": ["BTC-USDT", "ETH-USDT", "BNB-USDT", "SOL-USDT"],
    "reconnect": True,
    "ping_interval": 30
}
```

### 2.2 Binance WebSocket连接
```python
# Binance WebSocket配置
BINANCE_WS_CONFIG = {
    "base_url": "wss://stream.binance.com:9443",
    "stream_url": "/ws/",
    "combined_url": "/stream?streams=",
    "streams": [
        "btcusdt@ticker",    # 24hr ticker
        "ethusdt@ticker",
        "bnbusdt@ticker", 
        "solusdt@ticker",
        "btcusdt@depth20",   # 深度数据
        "btcusdt@trade"      # 实时成交
    ],
    "reconnect": True,
    "ping_interval": 30
}
```

## 3. Rich Layout组件设计

### 3.1 主Layout结构
```python
from rich.layout import Layout
from rich.live import Live
from rich.console import Console

# 创建主布局
layout = Layout()

# 分割主区域
layout.split(
    Layout(name="header", size=3),      # 顶部状态栏
    Layout(name="body"),                # 主体内容
    Layout(name="footer", size=3)       # 底部快捷键
)

# 主体区域分割
layout["body"].split(
    Layout(name="top_row", ratio=2),    # 上半部分
    Layout(name="middle_row", ratio=2), # 中间部分  
    Layout(name="bottom_row", ratio=1)  # 下半部分(日志)
)

# 上半部分三列分割
layout["top_row"].split_row(
    Layout(name="market_data"),         # 实时行情
    Layout(name="ai_analysis"),         # AI分析
    Layout(name="news_feed")            # 新闻推送
)

# 中间部分三列分割
layout["middle_row"].split_row(
    Layout(name="strategies"),          # 策略状态
    Layout(name="risk_control"),        # 风控监控
    Layout(name="charts")               # K线图表
)
```

### 3.2 实时数据组件
```python
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from rich.text import Text

class MarketDataPanel:
    """实时行情面板"""
    def __init__(self):
        self.prices = {}
        self.changes = {}
        
    def create_table(self):
        table = Table(title="📈 实时行情", show_header=True, header_style="bold magenta")
        table.add_column("币种", style="cyan", width=12)
        table.add_column("价格", justify="right", style="green")
        table.add_column("24H变化", justify="right")
        table.add_column("成交量", justify="right", style="yellow")
        
        for symbol, data in self.prices.items():
            change_style = "green" if data['change'] >= 0 else "red"
            change_text = f"▲ +{data['change']:.1f}%" if data['change'] >= 0 else f"▼ {data['change']:.1f}%"
            
            table.add_row(
                symbol,
                f"${data['price']:,.2f}",
                Text(change_text, style=change_style),
                f"{data['volume']:.1f}M"
            )
        
        return Panel(table, border_style="blue")

class StrategyPanel:
    """策略状态面板"""
    def __init__(self):
        self.strategies = {}
    
    def create_panel(self):
        content = []
        for name, strategy in self.strategies.items():
            status_icon = "🟢" if strategy['status'] == 'running' else "🟡" if strategy['status'] == 'paused' else "🔴"
            profit_color = "green" if strategy['profit'] >= 0 else "red"
            
            content.extend([
                f"[bold]{name}[/bold]",
                f"状态: {status_icon} {strategy['status_text']}",
                f"收益: [bold {profit_color}]{strategy['profit']:+.1f}%[/bold {profit_color}]",
                f"持仓: {strategy['position']}",
                ""
            ])
        
        return Panel("\n".join(content), title="🚀 策略状态", border_style="green")
```

## 4. WebSocket数据处理流程

### 4.1 数据接收和解析
```python
import asyncio
import websockets
import json
from rich.live import Live

class WebSocketManager:
    """WebSocket连接管理器"""
    
    def __init__(self, exchange, config):
        self.exchange = exchange
        self.config = config
        self.websocket = None
        self.data_handlers = {}
        self.is_connected = False
    
    async def connect(self):
        """建立WebSocket连接"""
        try:
            self.websocket = await websockets.connect(self.config['url'])
            self.is_connected = True
            await self.subscribe_channels()
            return True
        except Exception as e:
            self.is_connected = False
            return False
    
    async def subscribe_channels(self):
        """订阅数据频道"""
        if self.exchange == "okx":
            subscribe_msg = {
                "op": "subscribe",
                "args": [
                    {"channel": "tickers", "instId": symbol}
                    for symbol in self.config['symbols']
                ]
            }
        elif self.exchange == "binance":
            # Binance使用不同的订阅格式
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": self.config['streams'],
                "id": 1
            }
        
        await self.websocket.send(json.dumps(subscribe_msg))
    
    async def listen(self, data_callback):
        """监听数据"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await data_callback(self.exchange, data)
        except websockets.exceptions.ConnectionClosed:
            self.is_connected = False
            # 触发重连
            await self.reconnect()

class RealTimeDataManager:
    """实时数据管理器"""
    
    def __init__(self):
        self.okx_ws = WebSocketManager("okx", OKX_WS_CONFIG)
        self.binance_ws = WebSocketManager("binance", BINANCE_WS_CONFIG)
        self.market_data = {}
        self.update_callbacks = []
    
    async def start_streams(self):
        """启动所有数据流"""
        tasks = [
            self.okx_ws.connect(),
            self.binance_ws.connect()
        ]
        
        # 启动监听任务
        listen_tasks = [
            self.okx_ws.listen(self.handle_data),
            self.binance_ws.listen(self.handle_data)
        ]
        
        await asyncio.gather(*tasks, *listen_tasks)
    
    async def handle_data(self, exchange, data):
        """处理接收到的数据"""
        if exchange == "okx":
            await self.process_okx_data(data)
        elif exchange == "binance":
            await self.process_binance_data(data)
        
        # 通知UI更新
        for callback in self.update_callbacks:
            await callback(self.market_data)
    
    async def process_okx_data(self, data):
        """处理OKX数据"""
        if data.get('arg', {}).get('channel') == 'tickers':
            for ticker in data.get('data', []):
                symbol = ticker['instId'].replace('-', '/')
                self.market_data[f"okx_{symbol}"] = {
                    'exchange': 'OKX',
                    'symbol': symbol,
                    'price': float(ticker['last']),
                    'change': float(ticker['change24h']),
                    'volume': float(ticker['vol24h']),
                    'timestamp': ticker['ts']
                }
    
    async def process_binance_data(self, data):
        """处理Binance数据"""
        if 'data' in data and 'e' in data['data']:
            ticker_data = data['data']
            if ticker_data['e'] == '24hrTicker':
                symbol = ticker_data['s'].replace('USDT', '/USDT')
                self.market_data[f"binance_{symbol}"] = {
                    'exchange': 'Binance',
                    'symbol': symbol,
                    'price': float(ticker_data['c']),
                    'change': float(ticker_data['P']),
                    'volume': float(ticker_data['v']),
                    'timestamp': ticker_data['E']
                }
```

## 5. 动态UI更新机制

### 5.1 Rich Live更新
```python
import asyncio
from rich.live import Live
from rich.console import Console

class LiveDashboard:
    """实时仪表盘"""
    
    def __init__(self):
        self.console = Console()
        self.data_manager = RealTimeDataManager()
        self.layout_manager = LayoutManager()
        self.is_running = False
    
    async def start_dashboard(self):
        """启动实时仪表盘"""
        self.is_running = True
        
        # 注册数据更新回调
        self.data_manager.update_callbacks.append(self.update_display)
        
        # 启动数据流
        data_task = asyncio.create_task(self.data_manager.start_streams())
        
        # 启动Live显示
        with Live(self.layout_manager.create_layout(), 
                  console=self.console, 
                  refresh_per_second=4,
                  screen=True) as live:
            
            while self.is_running:
                try:
                    # 每250ms刷新一次界面
                    await asyncio.sleep(0.25)
                    live.update(self.layout_manager.create_layout())
                except KeyboardInterrupt:
                    self.is_running = False
                    break
    
    async def update_display(self, market_data):
        """更新显示数据"""
        # 更新行情数据
        self.layout_manager.update_market_data(market_data)
        
        # 更新AI分析
        await self.update_ai_analysis()
        
        # 更新策略状态
        self.update_strategy_status()
    
    def handle_key_input(self, key):
        """处理按键输入"""
        key_actions = {
            '1': self.switch_to_dashboard,
            '2': self.switch_to_strategy_manager,
            '3': self.switch_to_ai_chat,
            '4': self.switch_to_factor_discovery,
            '5': self.switch_to_trading_history,
            '6': self.switch_to_settings,
            'q': self.quit_application,
            'r': self.refresh_data
        }
        
        if key in key_actions:
            key_actions[key]()
```

## 6. 连接状态监控

### 6.1 连接状态显示
```python
class ConnectionStatusManager:
    """连接状态管理"""
    
    def __init__(self):
        self.connections = {
            'okx': {'status': 'disconnected', 'last_ping': None},
            'binance': {'status': 'disconnected', 'last_ping': None},
            'coinglass': {'status': 'disconnected', 'last_ping': None},
            'jin10': {'status': 'disconnected', 'last_ping': None}
        }
    
    def get_status_display(self):
        """获取状态显示字符串"""
        status_texts = []
        for name, conn in self.connections.items():
            icon = "✅" if conn['status'] == 'connected' else "❌"
            status_texts.append(f"🔗 {name.upper()}: {icon} {conn['status']}")
        
        return " │ ".join(status_texts)
    
    async def monitor_connections(self):
        """监控连接状态"""
        while True:
            for name, ws_manager in [('okx', self.okx_ws), ('binance', self.binance_ws)]:
                if ws_manager.is_connected:
                    # 发送ping检查连接
                    try:
                        await ws_manager.websocket.send('{"op":"ping"}')
                        self.connections[name]['status'] = 'connected'
                        self.connections[name]['last_ping'] = asyncio.get_event_loop().time()
                    except:
                        self.connections[name]['status'] = 'disconnected'
                        # 触发重连
                        asyncio.create_task(ws_manager.reconnect())
                else:
                    self.connections[name]['status'] = 'disconnected'
            
            await asyncio.sleep(30)  # 每30秒检查一次
```

## 7. 启动入口

### 7.1 主程序入口
```python
#!/usr/bin/env python3
import asyncio
import sys
from rich.console import Console

async def main():
    """主程序入口"""
    console = Console()
    
    try:
        # 显示启动画面
        console.print("""
        ███████╗██╗   ██╗███████╗████████╗███████╗███╗   ███╗
        ██╔════╝╚██╗ ██╔╝██╔════╝╚══██╔══╝██╔════╝████╗ ████║
        ███████╗ ╚████╔╝ ███████╗   ██║   █████╗  ██╔████╔██║
        ╚════██║  ╚██╔╝  ╚════██║   ██║   ██╔══╝  ██║╚██╔╝██║
        ███████║   ██║   ███████║   ██║   ███████╗██║ ╚═╝ ██║
        ╚══════╝   ╚═╝   ╚══════╝   ╚═╝   ╚══════╝╚═╝     ╚═╝
        
        🚀 量化交易系统启动中... 连接WebSocket数据流...
        """, style="bold blue")
        
        # 创建仪表盘
        dashboard = LiveDashboard()
        
        # 启动系统
        await dashboard.start_dashboard()
        
    except KeyboardInterrupt:
        console.print("\n👋 感谢使用量化交易系统!", style="bold yellow")
    except Exception as e:
        console.print(f"❌ 系统错误: {e}", style="bold red")
        sys.exit(1)

if __name__ == "__main__":
    # 运行主程序
    asyncio.run(main())
```

这个详细的CLI布局设计包含了：

1. **完整的120x40终端布局**，合理分配屏幕空间
2. **OKX和Binance WebSocket实时连接**，支持多交易所数据流  
3. **Rich组件化设计**，模块化的界面组件
4. **实时数据更新机制**，250ms刷新频率
5. **连接状态监控**，自动重连和状态显示
6. **键盘快捷键支持**，高效操作体验

界面支持实时显示价格变化、策略状态、AI分析、新闻推送、因子发现等，完全满足专业量化交易的需求！

## 8. 因子发现界面设计

### 8.1 因子发现TUI界面布局
```
┌─────────────────────── 🔬 AI因子发现实验室 ────────────────────────────┐
│ 📊 因子库状态: 125个因子 │ 🎯 活跃因子: 23个 │ 📈 平均IC: 0.045 │
├─────────────────────────────────────────────────────────────────────┤
│ 🔍 操作面板              │ 📋 因子库列表                              │
│ ┌─ AI发现 ─────────┐    │ ┌─ 因子信息表格 ─────────────────────────┐ │
│ │ [DeepSeek发现]    │    │ │名称    │类型│IC均值│ICIR │状态│创建时间│ │
│ │ [Gemini优化]      │    │ │RSI_OPT │动量│0.052 │0.68 │活跃│1-27   │ │
│ │ [因子验证]        │    │ │MA_CROSS│趋势│0.041 │0.55 │测试│1-26   │ │
│ │ [组合优化]        │    │ │VOL_ADJ │波动│0.038 │0.49 │暂停│1-25   │ │
│ │ [A/B测试]         │    │ │AI_SENT │情绪│0.067 │0.89 │活跃│1-24   │ │
│ └───────────────────┘    │ │...     │... │...   │... │... │...    │ │
│                          │ └───────────────────────────────────────┘ │
├──────────────────────────┼─────────────────────────────────────────┤
│ 📊 因子详情              │ 📈 性能图表                              │
│ ┌─ AI_SENTIMENT ──────┐  │ ┌─ 收益曲线 ─────────────────────────┐   │
│ │ 类型: 情绪因子       │  │ │    ^                               │   │
│ │ 公式: sentiment_score│  │ │   /|\                              │   │
│ │      * volume_ratio  │  │ │  / | \     /\                     │   │
│ │ 描述: AI情绪与成交量 │  │ │ /  |  \   /  \        /\          │   │
│ │      结合的复合因子   │  │ │/   |   \ /    \      /  \         │   │
│ │ IC: 0.067 ±0.015    │  │ │    |    V      \    /    \        │   │
│ │ ICIR: 0.89          │  │ │    |             \  /      \       │   │
│ │ 胜率: 67%           │  │ │    └──────────────┴────────────    │   │
│ │ 最大回撤: -5.2%     │  │ │    1M   3M   6M   1Y              │   │
│ └─────────────────────┘  │ └───────────────────────────────────┘   │
├──────────────────────────┴─────────────────────────────────────────┤
│ 🤖 AI分析建议                                                       │
│ DeepSeek建议: 情绪因子在高波动期表现优异，建议增加权重至15%          │
│ Gemini优化: 可结合MACD信号增强因子稳定性，预计IC提升至0.075         │
│ 回测结果: 过去6个月累计收益+12.8%，最大回撤控制在5%以内             │
├─────────────────────────────────────────────────────────────────────┤
│ 🎮 快捷键: [D]发现 [O]优化 [T]测试 [B]回测 [C]组合 [ESC]返回        │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 因子发现工作流程界面
```
┌─────────────────────── 🔬 因子发现进行中... ────────────────────────┐
│ 步骤 1/5: 数据预处理          ████████████████████████████ 100%     │
│ 步骤 2/5: 技术指标计算        ████████████████████████████ 100%     │ 
│ 步骤 3/5: AI模式识别         ████████████████████▓▓▓▓▓▓▓▓ 75%      │
│ 步骤 4/5: 因子验证           ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 0%       │
│ 步骤 5/5: 性能回测           ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 0%       │
├─────────────────────────────────────────────────────────────────────┤
│ 🤖 DeepSeek正在分析价量数据...                                      │
│ 📊 已处理180天历史数据，发现23个潜在因子模式                        │
│ 🔍 正在评估因子"多时间框架RSI背离"的有效性...                       │
│                                                                     │
│ 💡 发现亮点:                                                       │
│ • 检测到强烈的价量背离信号 (置信度: 87%)                           │
│ • 发现新的波动率regime切换模式                                      │
│ • 情绪与技术指标出现有趣的非线性关系                                │
├─────────────────────────────────────────────────────────────────────┤
│ [取消发现] [查看日志] [暂停处理]                  预计剩余时间: 2分钟 │
└─────────────────────────────────────────────────────────────────────┘
```