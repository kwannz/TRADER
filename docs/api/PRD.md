# AI量化交易系统产品需求文档（Product Requirements Document）

## 1. 产品概述

### 1.1 产品定位
- **产品名称**：个人AI量化交易系统（Advanced AI Quant Trading Platform）
- **产品愿景**：打造集AI分析、因子发现、策略执行、风险控制于一体的下一代量化交易平台
- **目标用户**：具备编程技能的个人投资者、量化交易专业人士、金融科技从业者
- **核心价值**：通过AI驱动的因子发现和策略优化，大幅提升个人投资者的交易效率和收益能力

### 1.2 技术架构核心
- **系统核心**：Rust高性能计算引擎 + Python业务逻辑 + FastAPI接口层
- **CLI验证体系**：每个模块都通过专业CLI界面进行实时验证和调试
- **AI双引擎**：DeepSeek（情绪分析、因子发现）+ Gemini（策略生成、优化建议）
- **数据驱动**：实时WebSocket数据流 + MongoDB时序存储 + Redis缓存层

## 2. 用户分析

| 用户类型 | 用户特征 | 核心需求 | 使用场景 |
|:--------:|:--------:|:--------:|:--------:|
| 量化交易员 | 有编程基础，追求稳定收益 | AI因子发现、策略回测、风险控制 | 日常策略开发和优化 |
| 个人投资者 | 技术背景，小资金量交易 | 自动化交易、AI建议、实时监控 | 兼职交易和投资管理 |
| 金融科技开发者 | 资深程序员，需要可定制化 | API接口、模块化架构、数据访问 | 构建定制化交易系统 |

## 3. 核心系统架构

### 3.1 技术栈重构设计
```
┌─────────────── CLI验证调试层 ─────────────────┐
│ Rich/Textual CLI界面 - 每个模块独立验证调试   │
├─────────────── FastAPI接口层 ─────────────────┤
│ • REST API服务     • WebSocket实时数据        │
│ • 用户认证         • 接口限流保护              │  
├─────────────── Python业务逻辑层 ──────────────┤
│ • AI引擎集成       • 策略管理                 │
│ • 因子发现实验室   • 风控系统                 │
├─────────────── Rust高性能引擎 ────────────────┤
│ • 实时数据处理     • 技术指标计算              │
│ • 订单执行引擎     • Alpha因子计算             │
│ • 回测引擎         • 时序数据优化              │
├─────────────── 数据存储层 ────────────────────┤
│ • MongoDB(时序)    • Redis(缓存)              │
│ • PostgreSQL(配置) • 文件存储(模型权重)       │
└──────────────────────────────────────────────┘
```

### 3.2 模块化验证体系
**每个功能模块都具备独立的CLI验证界面**：
- **数据采集模块**：WebSocket连接状态、数据质量监控
- **AI分析模块**：情绪分析结果、策略生成过程可视化
- **因子发现模块**：实时因子挖掘进度、统计检验结果
- **策略执行模块**：订单状态追踪、仓位管理验证
- **风控模块**：风险指标监控、止损机制测试
- **回测模块**：策略性能评估、收益曲线展示

## 4. 页面架构与功能规划

### 4.1 核心页面清单
| 页面名称 | 页面类型 | 核心功能 | 用户价值 | CLI验证模式 | 优先级 |
|:--------:|:--------:|:--------:|:--------:|:-----------:|:------:|
| 实时仪表盘 | 主控制台 | Bloomberg风格数据监控 | 全局状态掌握 | 4Hz实时刷新验证 | P0 |
| 策略管理中心 | 功能页面 | CRUD策略、参数优化 | 策略全生命周期管理 | 策略创建调试界面 | P0 |
| AI因子发现实验室 | 创新功能 | AI驱动Alpha因子挖掘 | 发现隐藏交易机会 | 因子挖掘进度可视化 | P0 |
| AI智能助手 | 对话界面 | 自然语言交易分析 | 专业投资建议 | 对话流程验证 | P1 |
| 交易记录分析 | 数据页面 | 历史交易和绩效分析 | 投资效果评估 | 数据查询验证 | P1 |
| 风控监控中心 | 安全功能 | 实时风险监控告警 | 资金安全保护 | 风险模拟测试 | P0 |
| 系统设置面板 | 配置页面 | API配置、参数调整 | 个性化定制 | 配置验证工具 | P2 |

### 4.2 详细页面需求

#### 页面1：实时仪表盘（Bloomberg风格CLI）
- **页面目标**：提供交易者完整的市场概览和系统状态监控
- **核心功能**：
  - 实时行情流（WebSocket 4Hz刷新）
  - AI市场情绪分析（DeepSeek驱动）
  - 策略运行状态网格
  - 财经新闻和事件推送
  - 系统连接状态监控
- **CLI验证逻辑**：
  - WebSocket连接稳定性测试
  - 数据刷新频率验证（4Hz）
  - 内存占用监控
  - AI API响应时间测试
- **业务逻辑**：
  - 多交易所数据融合
  - AI情绪评分实时计算
  - 策略PnL动态更新
  - 异常状态自动告警
- **页面元素**：状态栏、三栏式布局、实时日志、快捷键栏

#### 页面2：AI因子发现实验室
- **页面目标**：通过AI技术自动发现和验证Alpha因子
- **核心功能**：
  - DeepSeek+Gemini双AI因子挖掘
  - Alpha101传统因子计算
  - 因子IC/ICIR统计验证
  - 因子组合优化建议
  - CTBench时序生成集成
- **CLI验证逻辑**：
  - AI模型调用状态监控
  - 因子计算进度条
  - 统计检验结果展示
  - 因子性能排名验证
- **业务逻辑**：
  - 多时间框架因子计算
  - 因子衰减分析
  - 协方差矩阵风险控制
  - 因子组合权重优化
- **页面元素**：操作面板、因子列表、详情图表、AI建议区域

#### 页面3：策略管理中心
- **页面目标**：策略全生命周期管理和性能监控
- **核心功能**：
  - 多种策略类型支持（网格、DCA、AI生成）
  - 可视化策略参数调整
  - 实时策略性能监控
  - 策略风险评估和优化建议
- **CLI验证逻辑**：
  - 策略创建流程验证
  - 参数边界检查
  - 回测结果验证
  - 策略启停状态测试
- **跳转逻辑**：从仪表盘策略卡片点击直接跳转到详情编辑

## 5. AI集成方案详细设计

### 5.1 DeepSeek集成架构
```python
class DeepSeekAnalyzer:
    """DeepSeek AI分析引擎"""
    
    async def analyze_market_sentiment(self, news_data: List[Dict]) -> Dict:
        """
        新闻情绪分析
        - 输入：财经新闻列表
        - 输出：情绪得分、置信度、关键事件
        - CLI验证：实时情绪波动图表
        """
        pass
    
    async def discover_alpha_factors(self, market_data: pd.DataFrame) -> List[Dict]:
        """
        AI因子发现
        - 输入：多维市场数据
        - 输出：新因子定义、预期效果
        - CLI验证：因子发现进度、统计显著性检验
        """
        pass
```

### 5.2 Gemini集成架构
```python
class GeminiOptimizer:
    """Gemini策略优化引擎"""
    
    async def generate_strategy_code(self, description: str) -> Dict:
        """
        自然语言生成策略代码
        - 输入：策略描述
        - 输出：Python代码、参数建议
        - CLI验证：代码语法检查、逻辑验证
        """
        pass
    
    async def optimize_factor_combination(self, factors: List) -> Dict:
        """
        因子组合优化
        - 输入：候选因子列表
        - 输出：最优权重、风险评估
        - CLI验证：优化过程可视化、结果对比
        """
        pass
```

## 6. Rust性能引擎设计

### 6.1 核心Rust模块
```rust
// 高频数据处理引擎
pub struct TickProcessor {
    buffer: RingBuffer<TickData>,
    indicators: TechnicalIndicators,
}

impl TickProcessor {
    // 实时技术指标计算（毫秒级延迟）
    pub fn update_indicators(&mut self, tick: TickData) -> IndicatorResult {
        // CLI验证：指标计算延迟监控
    }
    
    // Alpha101因子高速计算
    pub fn compute_alpha_factors(&self, lookback: usize) -> Vec<f64> {
        // CLI验证：因子计算性能基准测试
    }
}

// 订单执行引擎
pub struct OrderEngine {
    pending_orders: HashMap<OrderId, Order>,
    execution_latency: MovingAverage,
}

impl OrderEngine {
    // 低延迟订单执行
    pub async fn execute_order(&mut self, order: Order) -> ExecutionResult {
        // CLI验证：执行延迟分布图、成功率统计
    }
}
```

### 6.2 Python-Rust接口层
```python
import quantlib_rust  # Rust编译的Python扩展

class PerformanceEngine:
    """高性能计算引擎接口"""
    
    def __init__(self):
        self.rust_engine = quantlib_rust.TickProcessor()
        
    async def process_realtime_data(self, tick_data: Dict) -> Dict:
        """
        实时数据处理
        - Rust引擎：毫秒级技术指标计算
        - Python层：AI分析和策略逻辑
        - CLI验证：处理延迟监控、吞吐量测试
        """
        result = self.rust_engine.update_indicators(tick_data)
        return await self.ai_analysis(result)
```

## 7. CLI验证调试体系

### 7.1 模块化CLI验证架构
```
CLI验证体系架构：
┌─── 系统启动器 ────┐    ┌─── 实时监控面板 ────┐
│ • 环境检查       │    │ • WebSocket状态    │
│ • 依赖验证       │    │ • AI API延迟      │
│ • 配置加载       │    │ • 内存/CPU使用    │
│ • 服务启动       │    │ • 错误日志        │
└─────────────────┘    └───────────────────┘

┌─── 数据验证器 ────┐    ┌─── 策略调试器 ────┐
│ • 数据完整性     │    │ • 策略回测       │
│ • 异常检测       │    │ • 参数敏感性     │
│ • 延迟监控       │    │ • 风险指标       │
│ • 质量评分       │    │ • 执行轨迹       │
└─────────────────┘    └───────────────────┘

┌─── AI验证器 ──────┐    ┌─── 性能分析器 ────┐
│ • API调用状态    │    │ • Rust引擎性能   │
│ • 响应时间       │    │ • Python层延迟   │
│ • 结果一致性     │    │ • 内存分配       │
│ • 成本追踪       │    │ • GC停顿分析     │
└─────────────────┘    └───────────────────┘
```

### 7.2 验证界面设计规范
```python
class ModuleValidator:
    """模块验证基类"""
    
    def __init__(self, module_name: str):
        self.name = module_name
        self.console = Console()
        self.status_panel = StatusPanel()
        
    async def validate_module(self) -> ValidationResult:
        """
        通用验证流程：
        1. 前置条件检查
        2. 功能性测试
        3. 性能基准测试  
        4. 异常情况处理
        5. 结果可视化展示
        """
        with Live(self.status_panel, refresh_per_second=4):
            result = await self._run_validation_tests()
            self._display_results(result)
        return result
```

## 8. 用户故事

### P0核心功能：
- **作为量化交易员**，我希望通过AI自动发现新的Alpha因子，以便持续提升策略收益率
  - 业务规则：因子IC绝对值>0.02，ICIR>0.3才算有效因子
  
- **作为个人投资者**，我希望实时监控所有策略的运行状态和风险指标，以便及时调整投资决策
  - 业务规则：任何策略亏损超过设定阈值自动暂停，发送告警通知

- **作为技术人员**，我希望每个功能模块都有独立的CLI验证界面，以便快速定位和解决问题
  - 业务规则：每个模块的验证界面必须在10秒内完成健康检查

### P1重要功能：
- **作为交易策略开发者**，我希望用自然语言描述策略想法并自动生成代码，以便快速验证交易思路
  - 业务规则：生成的策略代码必须通过语法检查和基本逻辑验证

## 9. 技术约束与性能要求

### 9.1 系统性能指标
- **实时数据延迟**：< 50ms（Rust引擎处理）
- **AI分析响应**：< 2秒（网络I/O优化）  
- **因子计算速度**：1000个因子/秒（Alpha101基准）
- **策略回测速度**：1年历史数据 < 30秒
- **CLI界面刷新**：4Hz稳定刷新，无卡顿
- **内存占用**：< 2GB（包含AI模型缓存）

### 9.2 可靠性要求
- **系统可用性**：99.9%（自动重连机制）
- **数据一致性**：100%（事务性存储）
- **AI服务容错**：单个API故障不影响系统运行
- **WebSocket重连**：3秒内自动恢复连接

### 9.3 平台约束
- **操作系统**：支持macOS、Linux、Windows（Docker统一环境）
- **Python版本**：>=3.9（类型提示支持）
- **Rust版本**：>=1.70（稳定版async支持）
- **终端兼容**：iTerm2、Terminal、Windows Terminal、WSL
- **最小配置**：8GB RAM、4核CPU、100GB存储

## 10. CTBench集成方案

### 10.1 时间序列生成平台集成
```python
class CTBenchIntegration:
    """CTBench时间序列生成集成"""
    
    def __init__(self):
        self.models = {
            'TimeVAE': TimeVAEModel(),
            'QuantGAN': QuantGANModel(), 
            'DiffusionTS': DiffusionModel(),
            'FourierFlow': FlowModel()
        }
    
    async def generate_synthetic_data(
        self, 
        historical_data: pd.DataFrame,
        model_type: str = 'TimeVAE',
        n_samples: int = 1000
    ) -> pd.DataFrame:
        """
        生成合成时间序列数据用于策略回测
        - CLI验证：生成进度、数据质量分析
        - 输出：符合市场统计特征的合成数据
        """
        model = self.models[model_type]
        synthetic_data = await model.generate(historical_data, n_samples)
        return await self._validate_synthetic_quality(synthetic_data)
```

### 10.2 Alpha101因子集成
```python
class Alpha101Calculator:
    """Alpha101因子计算器"""
    
    async def calculate_all_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算全部101个Alpha因子
        - Rust引擎：高速向量化计算
        - Python层：结果验证和异常处理
        - CLI验证：计算进度、因子有效性统计
        """
        # 调用Rust引擎进行计算
        rust_result = quantlib_rust.compute_alpha101(data)
        return await self._validate_factor_results(rust_result)
```

## 11. 用户流程

### 主要操作路径：
1. **系统启动** → 2. **环境验证** → 3. **数据连接确认** → 4. **进入仪表盘**
2. **仪表盘监控** → 3. **发现异常** → 4. **切换到相应调试界面**
3. **策略开发** → 4. **AI因子发现** → 5. **回测验证** → 6. **部署运行**

### 页面流转图：
```
启动页面 → 主仪表盘 ←→ 策略管理 ←→ 因子发现实验室
    ↓           ↓              ↓              ↓
调试验证 → AI智能助手 ←→ 交易记录 ←→ 系统设置
```

## 12. CLI界面技术规范

### 12.1 界面框架选择
- **主框架**：Rich + Textual（Python生态最优选择）
- **实时刷新**：Rich.Live（4Hz稳定刷新）
- **交互组件**：Textual TUI（复杂表单和对话）
- **数据可视化**：ASCII艺术图表 + Rich Charts
- **颜色主题**：Bloomberg深色专业主题

### 12.2 界面布局标准
```python
# 标准120x40字符终端布局
TERMINAL_LAYOUT = {
    "header": {"height": 3, "content": "状态栏+时间"},
    "body": {"height": 34, "layout": "3x2_grid"}, 
    "footer": {"height": 3, "content": "快捷键+帮助"}
}

# 响应式布局适配
RESPONSIVE_BREAKPOINTS = {
    "small": (100, 30),    # 最小可用尺寸
    "medium": (120, 40),   # 标准尺寸  
    "large": (150, 50),    # 扩展显示
    "xlarge": (180, 60)    # 全功能显示
}
```

## 13. 产品约束

### 13.1 功能边界
- **资金规模**：500-50000 USDT（个人投资者定位）
- **交易所支持**：OKX、Binance（主流平台优先）
- **策略数量**：同时运行最多10个策略
- **数据存储**：180天历史数据，30天AI分析结果
- **API调用**：成本控制在月100美元以内

### 13.2 安全约束
- **API密钥**：本地加密存储，不上传云端
- **交易权限**：只支持现货交易，不支持期货杠杆
- **资金安全**：硬止损保护，最大单日亏损5%
- **数据安全**：所有数据本地化存储

### 13.3 合规约束
- **开源协议**：MIT许可，允许商业使用
- **数据使用**：遵循各交易所ToS，不滥用API
- **免责声明**：用户自担投资风险，系统仅提供技术工具

## 14. 成功指标

### 14.1 技术指标
- **系统稳定性**：连续运行7天无重大故障
- **数据准确性**：与交易所官方数据一致性>99.9%
- **响应性能**：CLI界面操作响应时间<100ms
- **资源效率**：系统空闲内存占用<1GB

### 14.2 用户价值指标
- **学习成本**：新用户30分钟内完成基础操作
- **开发效率**：从想法到策略上线时间<1小时
- **调试效率**：通过CLI验证快速定位问题<5分钟
- **策略效果**：AI生成因子IC均值>0.03

### 14.3 AI集成效果指标
- **因子发现质量**：每次发现3-5个有效因子
- **策略生成成功率**：自然语言转代码成功率>80%
- **AI建议准确性**：策略优化建议有效提升收益>10%

## 15. 一步到位实现规范与接口契约（给开发者的落地说明）

### 15.1 数据模型与存储索引（MongoDB/Redis/PostgreSQL）
- MongoDB 集合与主字段（均需记录 `created_at`, `updated_at`, `source`, `ingest_ts`，时间戳统一为毫秒）
  - `ticks`
    - 字段：`symbol`(str), `ts`(int64 epoch_ms), `price`(float), `size`(float), `side`('buy'|'sell'), `exchange`(str)
    - 索引：`{symbol:1, ts:1}`（复合升序）；TTL：可选（按业务保留期配置）
  - `candles_1m`（后续可扩展 `candles_{tf}`）
    - 字段：`symbol`, `tf`(str, 如 '1m'), `ts_open`(int64), `open` `high` `low` `close`(float), `volume`(float), `vwap`(float)
    - 索引：`{symbol:1, tf:1, ts_open:1}` 唯一
  - `orders`
    - 字段：`client_order_id`(str), `symbol`, `side`('buy'|'sell'), `type`('market'|'limit'), `price`(float|null), `qty`(float), `status`('new'|'partially_filled'|'filled'|'canceled'|'rejected'), `reason`(str|null), `ts_created`, `ts_updated`
    - 索引：`{client_order_id:1}` 唯一；`{symbol:1, ts_created:-1}`
  - `trades`
    - 字段：`trade_id`(str), `client_order_id`(str), `symbol`, `price`, `qty`, `fee`(float), `ts`
    - 索引：`{client_order_id:1, ts:1}`, `{symbol:1, ts:1}`
  - `positions`
    - 字段：`symbol`, `qty`, `avg_price`, `unrealized_pnl`, `realized_pnl`, `ts`
    - 索引：`{symbol:1}` 唯一
  - `strategies`
    - 字段：`strategy_id`(str), `name`(str), `type`('grid'|'dca'|'ai'|'custom'), `params`(object), `state`('created'|'initialized'|'running'|'paused'|'stopped'|'error'), `last_error`(str|null)
    - 索引：`{strategy_id:1}` 唯一；`{state:1}`
  - `factors`
    - 字段：`factor_id`(str), `name`(str), `universe`(list[str]), `window`(int), `definition`(object), `version`(str), `owner`(str)
    - 索引：`{factor_id:1}` 唯一；`{name:1, version:1}`
  - `factor_stats`
    - 字段：`factor_id`, `date`(int epoch_day), `ic`(float), `icir`(float), `coverage`(float), `decay`(object)
    - 索引：`{factor_id:1, date:1}` 唯一
  - `alerts`
    - 字段：`level`('info'|'warn'|'error'|'critical'), `category`('risk'|'system'|'strategy'), `message`(str), `context`(object), `ts`
    - 索引：`{level:1, ts:-1}`, `{category:1, ts:-1}`
  - `ai_requests`
    - 字段：`provider`('deepseek'|'gemini'), `task`('sentiment'|'factor_discovery'|'strategy_gen'|'optimize'), `prompt_hash`(str), `request`(object), `response`(object), `latency_ms`(int), `cost_usd`(float), `status`('ok'|'error'), `ts`
    - 索引：`{provider:1, task:1, ts:-1}`, `{prompt_hash:1}`

- Redis 键空间（用于热数据与速率/配额）
  - `ws:conn:{exchange}` → JSON 连接状态（TTL 10s）
  - `ratelimit:{route}:{api_key}` → 计数器（滑动窗口）
  - `strategy:state:{strategy_id}` → 运行状态快照
  - `ai:budget:month:{provider}` → 累计成本（美元）

- PostgreSQL（配置/审计）
  - 表 `config_kv(key text primary key, value jsonb, updated_at timestamptz)`
  - 表 `audit_log(id bigserial pk, actor text, action text, resource text, before jsonb, after jsonb, ts timestamptz)`

### 15.2 FastAPI REST API 路由表与契约
- 鉴权：Header `X-API-Key: <token>`；未提供或非法返回 401。速率限制：默认 60 req/min/Key。
- 通用响应包：`{ success: bool, data?: any, error?: { code: string, message: string } }`

- 健康与系统
  - `GET /health` → `{ success, data: { mongo: 'ok'|'down', redis: 'ok'|'down', uptime_s: int } }`
  - `GET /metrics` → 文本（Prometheus 格式）

- 市场数据
  - `GET /market/candles` 查询参数：`symbol`(必填), `tf`('1m'|'5m'|'1h'|...), `limit`(<=1000), `to`(epoch_ms 可选)
    - 返回 `data: [{ ts_open, open, high, low, close, volume, vwap }]`

- 策略管理
  - `GET /strategies` → 列表
  - `POST /strategies` Body：`{ name, type, params }` → 创建，返回 `strategy_id`
  - `GET /strategies/{id}` → 详情（含当前 `state`）
  - `POST /strategies/{id}/start` → 启动（要求 `state` ∈ {'initialized','paused','stopped'}）
  - `POST /strategies/{id}/pause` → 暂停（要求 `state` == 'running'）
  - `POST /strategies/{id}/stop` → 停止（允许幂等）
  - `GET /strategies/{id}/status` → 运行指标 `{ pnl, positions, orders_open, last_heartbeat_ts }`

- 回测
  - `POST /backtest/run` Body：`{ strategy: { type, params }, universe: [symbol], tf, start, end, fees_bps, slippage_bps, initial_cash }`
    - 返回：`{ equity_curve: [{ts, equity}], summary: { ann_return, max_dd, sharpe, calmar }, trades: [...] }`

- 因子计算与统计
  - `POST /factors/alpha101` Body：`{ universe, tf, start, end }` → 触发批量计算
    - 返回：`{ task_id }`；计算结果入库 `factors`/`factor_stats`
  - `GET /factors/stats?factor_id=...&from=...&to=...` → 返回 IC/ICIR 序列与聚合

- 风控
  - `POST /risk/limits` Body：`{ daily_loss_limit_pct, per_strategy_loss_limit_pct, kill_switch: bool }`
  - `GET /risk/status` → `{ global_state: 'normal'|'warning'|'halted', today_pnl_pct, breached_rules: [...] }`

- AI 集成
  - `POST /ai/factors/discover` Body：`{ universe, lookbacks, objectives }` → 返回 `{ factors: [{ name, definition, expected_ic }], cost_usd }`
  - `POST /ai/strategy/generate` Body：`{ description, constraints }` → `{ code, params, checks: { syntax_ok, risk_notes }, cost_usd }`

### 15.3 WebSocket 主题与消息协议
- 连接：`/ws/stream?token=<X-API-Key>`；心跳 10s。
- 订阅协议：发送 `{ action: 'subscribe', topics: [ 'market.ticks.BTC-USDT', 'strategy.status.{id}', 'risk.alerts' ] }`
- 主题命名：
  - `market.ticks.{symbol}`
  - `market.candles.{symbol}.{tf}`
  - `strategy.status.{strategy_id}`
  - `risk.alerts`
  - `system.metrics`
  - `ai.events`
- 消息统一包：`{ topic, ts, type, data }`；`type` 示例：`'tick'|'candle'|'strategy_status'|'alert'|'metric'|'ai_event'`。

### 15.4 策略与风控状态机
- 策略状态：`created → initialized → running → paused → stopped`；任意状态 → `error`（携带 `last_error`）。
- 允许的动作与前置：
  - `initialize`: 仅 `created`；加载参数/校验边界；写入持久化状态
  - `start`: `initialized|paused|stopped`；启动事件循环与订阅
  - `pause`: `running`；保留仓位，停止下单
  - `stop`: 任意；撤销未成交，关闭订阅
  - `recover`: `error`；基于 `last_error` 自动/手动恢复至 `initialized`
- 风控状态：`normal → warning → halted`
  - 触发：`today_pnl_pct <= -daily_loss_limit_pct` 或策略级别亏损越界 → `halted` 且触发 `kill_switch`（全局断路器）

### 15.5 Rust ↔ Python FFI 规范
- 技术：`pyo3 + maturin` 打包为 `quantlib_rust`；线程安全函数为 `#[pyfunction]`，GIL 最小化持有。
- 数据承载：优先 Arrow C Data Interface（零拷贝）或 `numpy.ndarray`；批量调用避免循环跨界。
- 函数清单：
  - `update_indicators(tick: dict) -> dict`：输入 `{ price, size, ... }`，输出指标字典
  - `compute_alpha101(ohlcv_arrow: bytes) -> arrow_table_bytes`：输入 Arrow 表（列 `ts_open, open, high, low, close, volume`），返回含 101 因子列的 Arrow 表
  - `backtest_bars(ohlcv_arrow: bytes, params_json: str) -> arrow_table_bytes`：返回含 `ts, equity, position, pnl` 的曲线表
- 错误：统一抛出 `PyErr`，消息包含 `code` 与可读 `message`；Python 层转为 `{ success:false, error:{...} }`。
- 并发：Rust 内部 Rayon/多线程；Python 侧通过 `asyncio.to_thread` 调度，禁用 GIL 热区。

### 15.6 回测规范
- 输入：统一交易日历；OHLCV 必须等间隔；缺口需前向填充并标注 `is_holiday`。
- 交易费用：`fees_bps`、滑点：`slippage_bps`；撮合：价格穿越成交，支持部分成交；订单类型：市价/限价。
- 资金：`initial_cash`、不允许负现金；仓位与风险约束与实盘一致。
- 输出：
  - `equity_curve`：`ts, equity`
  - 指标：`ann_return, max_drawdown, sharpe(annualized), calmar`
  - 交易明细：`ts, side, price, qty, fee, pnl`
- 可重复性：固定 `seed`；记录版本与参数哈希。

### 15.7 AI 契约（DeepSeek/Gemini）
- 任务与输入输出：
  - 情绪分析：`{ news: [{ ts, title, body, source }] }` → `{ score: [-1,1], confidence: [0,1], events: [...] }`
  - 因子发现：`{ market_features: Arrow 表描述, objectives }` → `{ factors: [{ name, definition(JSON), expected_ic }] }`
  - 策略生成：`{ description, constraints }` → `{ code, params, checks }`
- 超时与重试：`timeout_ms=1800`，指数退避重试 2 次；熔断窗口 60s。
- 成本预算：每次响应记录 `latency_ms` 与 `cost_usd` 至 `ai_requests`；月度上限 `$100`，触达即临时禁用并告警。
- 安全：在请求中移除密钥/账户等敏感字段；持久化仅存提示词哈希与必要上下文。

### 15.8 可观测性与健康检查
- 日志：结构化 JSON（字段：`ts, level, logger, msg, req_id, route, latency_ms, cost_usd(optional)`）。
- 指标（Prometheus）：
  - `api_request_latency_ms{route,method}` 摘要
  - `ws_messages_total{topic}` 计数
  - `ffi_call_latency_ms{fn}` 摘要
  - `ai_cost_usd_total{provider}` 计数
- 健康检查：`/health` 聚合 Mongo/Redis/AI 连通性与最近错误摘要。

### 15.9 CLI 区域→数据源映射（4Hz 刷新）
- Header：系统时钟、`/health` 状态、AI 配额余量（`ai:budget`）。
- Body 三列两行（示例）：
  - 左上：`market.ticks.*` 实时 Tape（最新 50 条）
  - 中上：策略网格（`GET /strategies` + `strategy.status.*`）
  - 右上：AI 情绪（`/ai/factors/discover` 结果摘要 + `ai.events`）
  - 左下：`risk.alerts` 实时告警
  - 中下：系统指标（`system.metrics`）
  - 右下：实时日志尾部（结构化字段投影）
- 退让策略：当刷新超 250ms，降低至 2Hz 并显示黄色状态标记。

### 15.10 配置与部署（Docker/ENV）
- 必需环境变量：
  - `MONGO_URI`, `REDIS_URL`, `POSTGRES_URL`
  - `API_KEY`（服务内部）、`AI_DEEPSEEK_KEY`, `AI_GEMINI_KEY`
  - `EXCHANGES`（逗号分隔）、`SYMBOLS`
  - `DAILY_LOSS_LIMIT_PCT`, `PER_STRATEGY_LOSS_LIMIT_PCT`, `KILL_SWITCH`
- 进程划分：`api`（FastAPI）、`worker`（订阅/策略/回测）、`rustlib`（随 Python 进程加载）。
- 启动自检：连接性、索引检查、必需 ENV 校验、AI 额度查询。

### 15.11 验收与基准（必须达成）
- 延迟：
  - FFI `compute_alpha101` 单批 10k bars → 往返 P50 < 25ms, P95 < 50ms（本机 M1/16GB 基准）
  - WS 广播到 CLI 渲染 P95 < 200ms（单主题）
- 吞吐：
  - `market.ticks` 2k msg/s 持续 60s 无丢包（本地回放）
- AI：
  - 因子发现单次总耗时 < 2s，缓存命中率 > 50%（典型会话）
- 风控：
  - 超阈即刻 `halted`，Kill-switch 在 1s 内生效，相关策略下单被拒绝
- 回测：
  - 1 年日线回测 < 30s；指标与已知用例一致性偏差 < 1%

---

**文档版本**：v2.0  
**创建日期**：2025-01-27  
**更新日期**：2025-01-27  
**技术架构**：Rust(性能引擎) + Python(业务逻辑) + FastAPI(接口层) + CLI(验证调试)  
**核心特色**：AI驱动因子发现 + 模块化CLI验证 + Bloomberg风格专业界面