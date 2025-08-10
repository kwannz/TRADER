# 统一日志系统使用指南

## 概述

项目现在拥有一套完整、高性能、功能丰富的统一日志系统，支持：

- ✅ **多种输出方式**：控制台、文件、JSON文件、数据库
- ✅ **结构化日志**：支持JSON格式和元数据
- ✅ **性能监控**：自动记录执行时间、内存使用等
- ✅ **上下文管理**：支持用户ID、会话ID、请求ID等上下文信息
- ✅ **异步日志**：高性能异步写入
- ✅ **日志分类**：15个预定义分类，便于管理
- ✅ **日志过滤**：按级别、分类、模块等过滤
- ✅ **配置化**：支持JSON配置文件
- ✅ **装饰器**：性能监控和错误捕获装饰器
- ✅ **兼容性**：兼容现有日志代码

## 快速开始

### 基础使用

```python
from core.unified_logger import get_logger, LogCategory

logger = get_logger()

# 基础日志记录
logger.info("系统启动", LogCategory.SYSTEM)
logger.error("连接失败", LogCategory.NETWORK, exception=Exception("timeout"))

# 便捷方法
logger.api_info("API请求成功", extra_data={"method": "GET", "url": "/api/test"})
logger.trading_error("下单失败", exception=Exception("余额不足"))
```

### 上下文管理

```python
# 使用上下文记录相关日志
with logger.context(user_id="user123", session_id="session456"):
    logger.info("用户登录", LogCategory.USER)
    logger.api_info("处理请求")
```

### 性能监控

```python
# 使用性能上下文
with logger.performance_context("数据处理"):
    # 执行耗时操作
    process_data()
    # 自动记录执行时间
```

### 装饰器

```python
from core.unified_logger import log_performance, log_errors

@log_performance(LogCategory.TRADING, "订单处理")
async def process_order(order_id: str):
    # 自动记录执行时间
    return await handle_order(order_id)

@log_errors(LogCategory.SYSTEM)
def risky_operation():
    # 自动捕获和记录异常
    raise ValueError("Something went wrong")
```

## 日志分类

系统预定义了15个日志分类：

| 分类 | 用途 | 示例 |
|------|------|------|
| `SYSTEM` | 系统级别日志 | 启动、关闭、配置 |
| `API` | API请求响应 | REST API、WebSocket |
| `TRADING` | 交易相关 | 下单、成交、持仓 |
| `AI` | AI引擎日志 | 模型推理、训练 |
| `DATABASE` | 数据库操作 | 查询、更新、连接 |
| `NETWORK` | 网络请求 | HTTP请求、连接 |
| `SECURITY` | 安全相关 | 认证、授权、风控 |
| `PERFORMANCE` | 性能监控 | 耗时、内存、CPU |
| `USER` | 用户操作 | 登录、操作记录 |
| `WORKFLOW` | 工作流 | 任务执行、状态变更 |
| `COINGLASS` | 数据收集 | Coinglass API |
| `BACKTEST` | 回测系统 | 策略回测、结果 |
| `STRATEGY` | 策略执行 | 信号生成、执行 |
| `RISK` | 风险管理 | 风控检查、告警 |
| `MONITORING` | 系统监控 | 健康检查、指标 |

## 日志级别

支持8个日志级别（从低到高）：

| 级别 | 用途 | 颜色 |
|------|------|------|
| `TRACE` | 详细跟踪信息 | 灰色 |
| `DEBUG` | 调试信息 | 青色 |
| `INFO` | 一般信息 | 绿色 |
| `SUCCESS` | 成功信息 | 亮绿色 |
| `WARNING` | 警告信息 | 黄色 |
| `ERROR` | 错误信息 | 红色 |
| `CRITICAL` | 严重错误 | 品红色 |
| `FATAL` | 致命错误 | 红色背景 |

## 便捷方法

```python
# 分类便捷方法
logger.api_info("API成功")
logger.api_error("API失败", exception=e)
logger.trading_info("交易成功")
logger.ai_info("AI推理完成")
logger.security_warning("安全警告")
logger.performance_info("性能信息", duration=100.5)

# 通用方法
logger.trace("跟踪信息", LogCategory.SYSTEM)
logger.debug("调试信息", LogCategory.SYSTEM)
logger.info("一般信息", LogCategory.SYSTEM)
logger.success("成功信息", LogCategory.SYSTEM)
logger.warning("警告信息", LogCategory.SYSTEM)
logger.error("错误信息", LogCategory.SYSTEM, exception=e)
logger.critical("严重错误", LogCategory.SYSTEM, exception=e)
logger.fatal("致命错误", LogCategory.SYSTEM, exception=e)
```

## 结构化数据

```python
# 记录复杂结构化数据
order_data = {
    "order_id": "12345",
    "symbol": "BTC/USDT",
    "side": "buy",
    "amount": 0.1,
    "price": 45000.0
}

logger.trading_info(
    "订单创建成功",
    extra_data=order_data,
    tags=["order", "btc", "spot"]
)
```

## 配置系统

### 配置文件

日志配置位于 `config/logging_config.json`：

```json
{
  "logging": {
    "level": "INFO",
    "console": {
      "enabled": true,
      "colorize": true
    },
    "file": {
      "enabled": true,
      "directory": "logs",
      "system_log": {
        "filename": "system.log",
        "max_bytes": 52428800,
        "backup_count": 5
      }
    }
  }
}
```

### 使用配置化logger

```python
from config.logger_factory import create_configured_logger

# 创建配置化的logger
logger = create_configured_logger("my_app")
```

## 输出方式

### 1. 控制台输出
彩色格式化输出，支持开关控制。

### 2. 文件输出
- **系统日志**：`logs/system.log`
- **错误日志**：`logs/errors.log` （仅ERROR及以上）
- **分类日志**：`logs/trading.log`, `logs/api.log` 等

### 3. JSON文件输出
结构化JSON格式：`logs/system_structured.jsonl`

### 4. 数据库输出
支持MongoDB + Redis存储（需配置）。

## 性能特性

- **异步写入**：不阻塞主线程
- **批量刷新**：提高写入效率
- **内存缓冲**：减少IO操作
- **自动轮转**：防止日志文件过大
- **压缩存储**：节省磁盘空间

## 兼容性

统一日志系统完全兼容现有代码：

```python
# 旧的使用方式仍然有效
from utils.logger import get_logger
logger = get_logger()

# 新的统一日志系统会自动接管
logger.info("这条日志会使用新系统")
```

## 日志文件说明

运行后会在 `logs/` 目录生成以下日志文件：

```
logs/
├── system.log              # 系统主日志（所有级别）
├── errors.log              # 错误专用日志（ERROR+）
├── trading.log             # 交易日志
├── api.log                 # API日志  
├── ai.log                  # AI引擎日志
├── performance.log         # 性能日志
├── system_structured.jsonl # 结构化JSON日志
└── trading_structured.jsonl # 交易结构化日志
```

## 最佳实践

### 1. 选择合适的日志级别
- 生产环境使用 `INFO` 或 `WARNING`
- 开发环境使用 `DEBUG`
- 故障排查使用 `TRACE`

### 2. 使用合适的分类
```python
# ✅ 正确
logger.trading_info("下单成功")
logger.api_error("API超时", exception=e)

# ❌ 避免
logger.info("下单成功", LogCategory.SYSTEM)
```

### 3. 添加上下文信息
```python
# ✅ 推荐
with logger.context(user_id="user123", order_id="order456"):
    logger.trading_info("处理订单")

# ✅ 也可以
logger.trading_info("处理订单", extra_data={"user_id": "user123"})
```

### 4. 使用装饰器
```python
# ✅ 自动性能监控
@log_performance(LogCategory.AI, "模型推理")
async def run_inference(data):
    return model.predict(data)

# ✅ 自动错误捕获
@log_errors(LogCategory.DATABASE)
def database_operation():
    # 可能抛出异常的操作
    pass
```

### 5. 异常处理
```python
try:
    risky_operation()
except Exception as e:
    logger.error("操作失败", LogCategory.SYSTEM, 
                exception=e, 
                extra_data={"context": "additional_info"})
    raise
```

## 高级功能

### 自定义输出处理器
```python
from core.unified_logger import LogOutput, UnifiedLogger

class CustomOutput(LogOutput):
    async def write(self, record):
        # 自定义写入逻辑
        pass

logger = UnifiedLogger("custom")
logger.add_output(CustomOutput())
```

### 过滤器
```python
from core.unified_logger import LogFilter, LogLevel, LogCategory

# 创建只记录错误的过滤器
error_filter = LogFilter(min_level=LogLevel.ERROR)

# 创建只记录交易相关的过滤器  
trading_filter = LogFilter(categories=[LogCategory.TRADING])
```

### 性能指标
日志记录会自动收集性能指标：
- 执行时间（毫秒）
- 内存使用量（MB）
- CPU使用率（%）
- 线程ID、进程ID
- 主机名、服务名

## 故障排查

### 1. 日志文件权限问题
确保 `logs/` 目录可写：
```bash
chmod 755 logs/
```

### 2. 配置文件问题
检查 `config/logging_config.json` 格式：
```bash
python -m json.tool config/logging_config.json
```

### 3. 数据库连接问题
检查MongoDB和Redis连接：
```python
# 设置环境变量
export MONGODB_URL="mongodb://localhost:27017/trader_logs"
export REDIS_URL="redis://localhost:6379"
```

## 测试

运行测试脚本验证功能：

```bash
python test_logger.py
```

测试包括：
- ✅ 基础日志功能
- ✅ 分类日志
- ✅ 上下文管理
- ✅ 性能监控
- ✅ 装饰器
- ✅ 异常处理
- ✅ 结构化数据

## 总结

统一日志系统为项目提供了：

1. **完整的日志解决方案** - 从控制台到数据库的全方位支持
2. **高性能** - 异步写入，不阻塞业务逻辑
3. **易于使用** - 简洁的API，丰富的便捷方法
4. **可配置** - JSON配置文件，灵活定制
5. **功能丰富** - 性能监控、上下文管理、装饰器支持
6. **向下兼容** - 不影响现有代码

这套日志系统将大大提升系统的可观测性、调试能力和运维效率！