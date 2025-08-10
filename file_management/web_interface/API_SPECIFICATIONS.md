# API接口规范文档

**项目**: QuantAnalyzer Pro - API接口设计规范  
**版本**: v2.0  
**创建日期**: 2025-08-10  

---

## 1. API设计原则

### 1.1 设计理念

- **RESTful架构**: 遵循REST设计原则，资源导向
- **版本控制**: 通过URL路径进行版本管理 (`/api/v2/`)
- **统一响应**: 标准化的请求和响应格式
- **向后兼容**: 维护V1 API的兼容性
- **安全优先**: JWT认证 + HTTPS传输
- **性能优化**: 分页、缓存、压缩支持

### 1.2 接口分层

```
/api/v2/
├── factors/          # 因子相关API
├── backtest/         # 回测相关API  
├── data/            # 数据相关API
├── portfolio/       # 组合相关API
├── ai/              # AI服务API
├── system/          # 系统相关API
└── users/           # 用户管理API
```

### 1.3 标准响应格式

```json
{
  "success": true|false,
  "data": {...},           // 成功时的数据
  "error": "...",          // 失败时的错误信息
  "message": "...",        // 描述信息
  "timestamp": "2025-08-10T12:00:00Z",
  "request_id": "uuid",    // 请求追踪ID
  "pagination": {          // 分页信息（可选）
    "page": 1,
    "per_page": 50,
    "total": 1000,
    "total_pages": 20
  }
}
```

---

## 2. 认证与授权

### 2.1 JWT认证

```http
POST /api/v2/auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "secure_password"
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "token_type": "Bearer",
    "expires_in": 1800,
    "user": {
      "id": "uuid",
      "username": "user@example.com",
      "role": "premium",
      "permissions": ["factor:read", "factor:write", "backtest:run"]
    }
  }
}
```

### 2.2 请求头格式

```http
Authorization: Bearer <access_token>
Content-Type: application/json
X-Request-ID: <uuid>
X-Client-Version: 2.0.0
```

---

## 3. 因子相关API

### 3.1 批量计算因子

```http
POST /api/v2/factors/calculate
Authorization: Bearer <token>
Content-Type: application/json
```

**请求体**:
```json
{
  "factors": [
    {
      "id": "rsi_14",
      "name": "RSI_14",
      "category": "technical", 
      "formula": "rsi(close, 14)",
      "parameters": {"period": 14}
    },
    {
      "id": "macd_12_26_9",
      "name": "MACD",
      "category": "technical",
      "formula": "macd(close, 12, 26, 9)",
      "parameters": {
        "fast_period": 12,
        "slow_period": 26,
        "signal_period": 9
      }
    }
  ],
  "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
  "start_date": "2024-01-01",
  "end_date": "2024-12-31", 
  "frequency": "daily",
  "cache_enabled": true
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "job_id": "calc_uuid_123",
    "results": {
      "RSI_14": {
        "factor_id": "rsi_14",
        "values": [45.2, 52.1, 38.9, 61.3, 42.8],
        "timestamps": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "metadata": {
          "calculation_time_ms": 1250,
          "data_points": 1000,
          "cache_hit": false
        }
      },
      "MACD": {
        "factor_id": "macd_12_26_9", 
        "values": [0.12, -0.05, 0.23, -0.18, 0.31],
        "timestamps": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "metadata": {
          "calculation_time_ms": 890,
          "data_points": 1000,
          "cache_hit": true
        }
      }
    },
    "performance": {
      "total_calculation_time_ms": 2140,
      "rust_engine_time_ms": 1850,
      "data_loading_time_ms": 290,
      "cache_hit_rate": 0.5,
      "processed_data_points": 2000
    }
  },
  "timestamp": "2025-08-10T12:00:00Z",
  "request_id": "req_uuid_456"
}
```

### 3.2 获取因子库

```http
GET /api/v2/factors/library?category=technical&min_ic=0.1&limit=50&offset=0
Authorization: Bearer <token>
```

**查询参数**:
- `category`: 因子类别 (technical, statistical, fundamental, sentiment)
- `min_ic`: 最小IC值过滤
- `min_sharpe`: 最小夏普比率过滤  
- `search`: 搜索关键词
- `sort_by`: 排序字段 (ic, sharpe_ratio, usage_count, created_at)
- `sort_order`: 排序方向 (asc, desc)
- `limit`: 返回数量限制 (1-1000)
- `offset`: 偏移量

**响应**:
```json
{
  "success": true,
  "data": {
    "factors": [
      {
        "id": "rsi_momentum_v2",
        "name": "RSI动量因子V2",
        "display_name": "RSI Momentum Factor V2",
        "category": "technical",
        "formula": "(rsi(close, 14) - 50) / 50 * momentum(close, 20)",
        "description": "结合RSI和动量的复合技术因子，适用于中短期趋势预测",
        "parameters": {
          "rsi_period": 14,
          "momentum_period": 20,
          "normalization": "z_score"
        },
        "performance": {
          "ic_mean": 0.156,
          "ic_std": 0.089,
          "ic_ir": 1.75,
          "sharpe_ratio": 2.12,
          "max_drawdown": -0.083,
          "win_rate": 0.642,
          "sample_size": 2500
        },
        "usage_stats": {
          "usage_count": 87,
          "last_used": "2025-08-09T15:30:00Z",
          "avg_calculation_time_ms": 45
        },
        "metadata": {
          "version": "2.1",
          "created_by": "system",
          "created_at": "2024-12-15T10:00:00Z",
          "updated_at": "2025-01-20T14:30:00Z",
          "tags": ["momentum", "rsi", "trend", "short_term"],
          "is_public": true,
          "approval_status": "approved"
        }
      }
    ]
  },
  "pagination": {
    "page": 1,
    "per_page": 50,
    "total": 156,
    "total_pages": 4,
    "has_next": true,
    "has_prev": false
  }
}
```

### 3.3 创建因子

```http
POST /api/v2/factors/create
Authorization: Bearer <token>
Content-Type: application/json
```

**请求体**:
```json
{
  "name": "Custom_Volatility_Factor",
  "display_name": "自定义波动率因子",
  "category": "statistical",
  "formula": "stddev(returns, 20) * sqrt(252) / mean(abs(returns), 20)",
  "description": "年化波动率标准化因子，用于衡量价格波动强度",
  "parameters": {
    "window": 20,
    "annualization_factor": 252,
    "normalization": "mean_abs"
  },
  "tags": ["volatility", "risk", "statistical"],
  "is_public": false
}
```

### 3.4 实时因子值

```http
GET /api/v2/factors/realtime/{factor_id}?symbols=BTCUSDT,ETHUSDT
Authorization: Bearer <token>
```

**响应**:
```json
{
  "success": true,
  "data": {
    "factor_id": "rsi_14",
    "factor_name": "RSI_14",
    "values": {
      "BTCUSDT": {
        "value": 45.7,
        "rank": 234,
        "quantile": 45,
        "zscore": -0.23,
        "timestamp": "2025-08-10T12:00:00Z"
      },
      "ETHUSDT": {
        "value": 62.1,
        "rank": 789,
        "quantile": 78,
        "zscore": 1.15,
        "timestamp": "2025-08-10T12:00:00Z"
      }
    },
    "market_context": {
      "total_symbols": 1000,
      "calculation_timestamp": "2025-08-10T12:00:00Z",
      "data_freshness_seconds": 15
    }
  }
}
```

---

## 4. 回测相关API

### 4.1 创建回测任务

```http
POST /api/v2/backtest/create
Authorization: Bearer <token>
Content-Type: application/json
```

**请求体**:
```json
{
  "name": "多因子动量策略回测",
  "description": "基于RSI、MACD、动量的多因子选股策略", 
  "strategy": {
    "id": "multi_factor_momentum",
    "name": "多因子动量策略",
    "type": "factor_based",
    "factors": ["rsi_14", "macd_12_26_9", "momentum_20"],
    "factor_weights": {
      "rsi_14": 0.4,
      "macd_12_26_9": 0.35,
      "momentum_20": 0.25
    },
    "rebalance_frequency": "weekly",
    "position_sizing": "equal_weight",
    "max_position_size": 0.05,
    "universe_filter": {
      "min_market_cap": 1000000000,
      "min_volume": 1000000,
      "exclude_sectors": ["utilities"]
    }
  },
  "symbols": [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT",
    "SOLUSDT", "DOTUSDT", "AVAXUSDT", "MATICUSDT", "LINKUSDT"
  ],
  "start_date": "2023-01-01",
  "end_date": "2024-12-31",
  "config": {
    "initial_capital": 1000000.0,
    "commission_rate": 0.0025,
    "slippage_bps": 2.0,
    "max_position_count": 20,
    "min_trade_amount": 1000,
    "benchmark": "BTC",
    "rebalance_cost": 0.001
  },
  "risk_management": {
    "stop_loss": -0.05,
    "take_profit": 0.10,
    "max_drawdown_limit": -0.15,
    "position_concentration_limit": 0.30,
    "sector_concentration_limit": 0.40
  }
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "job_id": "bt_uuid_789",
    "name": "多因子动量策略回测",
    "status": "queued",
    "priority": 5,
    "estimated_duration": "8-12分钟",
    "queue_position": 3,
    "resource_requirements": {
      "estimated_memory_mb": 512,
      "estimated_cpu_seconds": 180,
      "data_size_mb": 45
    },
    "created_at": "2025-08-10T12:00:00Z"
  }
}
```

### 4.2 获取回测状态

```http
GET /api/v2/backtest/status/{job_id}
Authorization: Bearer <token>
```

**响应**:
```json
{
  "success": true,
  "data": {
    "job_id": "bt_uuid_789",
    "name": "多因子动量策略回测",
    "status": "running",
    "progress": 65,
    "current_stage": "执行回测计算",
    "stages": [
      {
        "name": "数据加载",
        "status": "completed",
        "duration_seconds": 23
      },
      {
        "name": "因子计算", 
        "status": "completed",
        "duration_seconds": 156
      },
      {
        "name": "策略信号生成",
        "status": "completed", 
        "duration_seconds": 67
      },
      {
        "name": "执行回测计算",
        "status": "running",
        "progress": 65,
        "estimated_remaining_seconds": 89
      },
      {
        "name": "性能分析",
        "status": "pending"
      },
      {
        "name": "报告生成",
        "status": "pending"
      }
    ],
    "metrics": {
      "processed_days": 487,
      "total_days": 730,
      "generated_signals": 1205,
      "executed_trades": 894
    },
    "timestamps": {
      "created_at": "2025-08-10T12:00:00Z",
      "started_at": "2025-08-10T12:01:30Z",
      "estimated_completion": "2025-08-10T12:09:45Z"
    },
    "resource_usage": {
      "current_memory_mb": 387,
      "peak_memory_mb": 445,
      "cpu_time_seconds": 127
    }
  }
}
```

### 4.3 获取回测结果

```http
GET /api/v2/backtest/results/{job_id}
Authorization: Bearer <token>
```

**响应**:
```json
{
  "success": true,
  "data": {
    "job_id": "bt_uuid_789",
    "status": "completed",
    "summary": {
      "name": "多因子动量策略回测",
      "period": {
        "start_date": "2023-01-01",
        "end_date": "2024-12-31",
        "trading_days": 520,
        "duration_years": 2.0
      },
      "performance": {
        "total_return": 0.2847,
        "annualized_return": 0.1357,
        "excess_return": 0.0523,
        "max_drawdown": -0.0892,
        "volatility": 0.1654,
        "sharpe_ratio": 0.82,
        "sortino_ratio": 1.15,
        "calmar_ratio": 1.52,
        "information_ratio": 0.45,
        "beta": 0.87,
        "alpha": 0.0234,
        "win_rate": 0.567,
        "profit_factor": 1.34
      },
      "trading_stats": {
        "total_trades": 894,
        "winning_trades": 507,
        "losing_trades": 387,
        "avg_trade_return": 0.0032,
        "avg_winning_trade": 0.0187,
        "avg_losing_trade": -0.0142,
        "max_consecutive_wins": 12,
        "max_consecutive_losses": 8,
        "turnover_annual": 2.34
      },
      "costs": {
        "total_commission": 5680.50,
        "total_slippage": 3245.20,
        "total_trading_costs": 8925.70,
        "cost_as_percentage": 0.0089
      }
    },
    "time_series": {
      "daily_returns": [0.0012, -0.0045, 0.0089, 0.0156, -0.0023],
      "cumulative_returns": [0.0012, -0.0033, 0.0056, 0.0213, 0.0189],
      "portfolio_values": [1001200, 996700, 1005600, 1021300, 1018900],
      "drawdowns": [0.0000, -0.0045, -0.0011, 0.0000, -0.0012],
      "positions_count": [15, 16, 18, 17, 19],
      "timestamps": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
    },
    "factor_analysis": {
      "factor_contributions": {
        "rsi_14": 0.42,
        "macd_12_26_9": 0.35,
        "momentum_20": 0.23
      },
      "factor_performance": {
        "rsi_14": {
          "ic": 0.156,
          "ir": 1.23,
          "hit_rate": 0.634
        },
        "macd_12_26_9": {
          "ic": 0.132,
          "ir": 0.98,
          "hit_rate": 0.598
        },
        "momentum_20": {
          "ic": 0.189,
          "ir": 1.45,
          "hit_rate": 0.672
        }
      }
    },
    "risk_analysis": {
      "var_95": -0.0234,
      "cvar_95": -0.0378,
      "max_leverage": 1.0,
      "avg_leverage": 0.95,
      "correlation_with_benchmark": 0.67,
      "tracking_error": 0.0456,
      "downside_deviation": 0.0987
    },
    "sector_exposure": {
      "technology": 0.35,
      "financial": 0.22,
      "healthcare": 0.18,
      "consumer": 0.15,
      "industrial": 0.10
    },
    "top_positions": [
      {
        "symbol": "BTCUSDT",
        "avg_weight": 0.087,
        "total_return": 0.234,
        "contribution": 0.0204
      },
      {
        "symbol": "ETHUSDT", 
        "avg_weight": 0.079,
        "total_return": 0.189,
        "contribution": 0.0149
      }
    ]
  },
  "metadata": {
    "computation_time_seconds": 487,
    "data_points_processed": 1340000,
    "rust_engine_version": "1.2.3",
    "calculation_timestamp": "2025-08-10T12:08:15Z"
  }
}
```

### 4.4 策略对比分析

```http
POST /api/v2/backtest/compare
Authorization: Bearer <token>
Content-Type: application/json
```

**请求体**:
```json
{
  "strategies": [
    {
      "name": "多因子动量策略",
      "job_ids": ["bt_uuid_789", "bt_uuid_790"]
    },
    {
      "name": "技术指标策略", 
      "job_ids": ["bt_uuid_791"]
    },
    {
      "name": "基准策略",
      "job_ids": ["bt_uuid_792"]
    }
  ],
  "comparison_metrics": [
    "total_return", "sharpe_ratio", "max_drawdown", 
    "win_rate", "information_ratio"
  ],
  "period": {
    "start_date": "2023-01-01",
    "end_date": "2024-12-31"
  }
}
```

---

## 5. 数据相关API

### 5.1 获取市场数据

```http
GET /api/v2/data/market?symbols=BTCUSDT,ETHUSDT&start_date=2024-01-01&end_date=2024-01-31&frequency=daily
Authorization: Bearer <token>
```

**查询参数**:
- `symbols`: 股票代码列表，逗号分隔
- `start_date`: 开始日期 (YYYY-MM-DD)
- `end_date`: 结束日期 (YYYY-MM-DD) 
- `frequency`: 数据频率 (1m, 5m, 15m, 1h, 4h, daily)
- `fields`: 数据字段 (ohlcv, technical_indicators, all)
- `adjust`: 是否复权 (true, false)

**响应**:
```json
{
  "success": true,
  "data": {
    "symbols": ["BTCUSDT", "ETHUSDT"],
    "frequency": "daily",
    "period": {
      "start_date": "2024-01-01",
      "end_date": "2024-01-31",
      "trading_days": 31
    },
    "market_data": [
      {
        "symbol": "BTCUSDT",
        "timestamp": "2024-01-01T00:00:00Z",
        "date": "2024-01-01",
        "ohlcv": {
          "open": 42150.00,
          "high": 42850.00,
          "low": 41980.00,
          "close": 42456.00,
          "volume": 125430000,
          "amount": 5324567000,
          "vwap": 42387.5
        },
        "technical_indicators": {
          "ma_20": 41987.5,
          "ma_60": 40234.8,
          "rsi_14": 58.4,
          "macd": 123.5,
          "bb_upper": 43200.0,
          "bb_middle": 42000.0,
          "bb_lower": 40800.0,
          "volatility_20": 0.0234
        },
        "market_metrics": {
          "turnover_rate": 0.0456,
          "amplitude": 0.0205,
          "price_change": 306.0,
          "price_change_pct": 0.0073
        }
      }
    ],
    "metadata": {
      "data_source": "binance",
      "data_quality_score": 0.998,
      "total_records": 62,
      "cache_hit": true,
      "query_time_ms": 45
    }
  }
}
```

### 5.2 数据质量监控

```http
GET /api/v2/data/quality?table=market_data_daily&start_date=2024-01-01&end_date=2024-01-31
Authorization: Bearer <token>
```

**响应**:
```json
{
  "success": true,
  "data": {
    "table_name": "market_data_daily",
    "check_period": {
      "start_date": "2024-01-01", 
      "end_date": "2024-01-31"
    },
    "overall_score": 0.987,
    "quality_metrics": {
      "completeness": {
        "score": 0.995,
        "details": {
          "total_records": 155000,
          "missing_records": 775,
          "missing_rate": 0.005,
          "field_completeness": {
            "symbol": 1.000,
            "timestamp": 1.000,
            "close": 0.998,
            "volume": 0.992
          }
        }
      },
      "accuracy": {
        "score": 0.989,
        "details": {
          "ohlc_consistency": 0.996,
          "negative_prices": 0,
          "extreme_changes": 23,
          "logical_errors": 5
        }
      },
      "timeliness": {
        "score": 0.990,
        "details": {
          "latest_data_date": "2024-01-31",
          "delay_hours": 2.3,
          "expected_delay": 1.0
        }
      },
      "consistency": {
        "score": 0.994,
        "details": {
          "duplicate_records": 124,
          "cross_table_consistency": 0.998
        }
      }
    },
    "issues": [
      {
        "type": "accuracy",
        "severity": "medium",
        "count": 23,
        "description": "发现23个异常价格波动，单日涨跌幅超过30%",
        "affected_symbols": ["SYMBOL1", "SYMBOL2"],
        "suggested_action": "人工审核异常数据并标记"
      },
      {
        "type": "timeliness", 
        "severity": "low",
        "description": "数据更新延迟2.3小时，超过预期1小时",
        "suggested_action": "检查数据源API状态"
      }
    ],
    "trends": {
      "quality_score_7d": [0.991, 0.988, 0.995, 0.987, 0.992, 0.989, 0.987],
      "completeness_trend": "stable",
      "accuracy_trend": "declining",
      "timeliness_trend": "stable"
    }
  }
}
```

---

## 6. WebSocket实时数据API

### 6.1 WebSocket连接

```javascript
const ws = new WebSocket('wss://api.quantanalyzer.pro/ws/v2');

// 认证
ws.onopen = function() {
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'Bearer <access_token>'
  }));
};

// 订阅因子实时更新
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'factors',
  params: {
    factor_ids: ['rsi_14', 'macd_12_26_9'],
    symbols: ['BTCUSDT', 'ETHUSDT'],
    update_interval: 60 // 秒
  }
}));
```

### 6.2 实时因子数据推送

```json
{
  "type": "factor_update",
  "channel": "factors", 
  "data": {
    "factor_id": "rsi_14",
    "timestamp": "2025-08-10T12:00:00Z",
    "values": {
      "BTCUSDT": {
        "value": 45.7,
        "rank": 234,
        "quantile": 45,
        "change": -2.3,
        "change_pct": -0.048
      },
      "ETHUSDT": {
        "value": 62.1,
        "rank": 789,
        "quantile": 78,
        "change": 1.8,
        "change_pct": 0.030
      }
    },
    "market_context": {
      "total_symbols": 1000,
      "market_state": "normal_trading"
    }
  }
}
```

### 6.3 回测状态推送

```json
{
  "type": "backtest_update",
  "channel": "backtest",
  "data": {
    "job_id": "bt_uuid_789",
    "status": "running",
    "progress": 75,
    "current_stage": "性能分析",
    "intermediate_results": {
      "current_return": 0.0234,
      "current_drawdown": -0.0156,
      "completed_trades": 1205
    },
    "estimated_completion": "2025-08-10T12:05:30Z"
  }
}
```

---

## 7. 错误处理和状态码

### 7.1 HTTP状态码

| 状态码 | 含义 | 场景 |
|-------|------|------|
| 200 | 成功 | 请求成功处理 |
| 201 | 创建成功 | 资源创建成功 |
| 400 | 请求错误 | 参数错误、格式错误 |
| 401 | 未认证 | Token无效或过期 |
| 403 | 禁止访问 | 权限不足 |
| 404 | 资源不存在 | 接口或数据不存在 |
| 422 | 数据验证失败 | 业务逻辑验证失败 |
| 429 | 请求过多 | 触发限流 |
| 500 | 服务器错误 | 内部错误 |
| 503 | 服务不可用 | 系统维护或过载 |

### 7.2 错误响应格式

```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "请求参数验证失败",
    "details": {
      "field": "start_date",
      "reason": "start_date must be before end_date",
      "received": "2024-12-31",
      "expected": "date before 2024-01-01"
    }
  },
  "timestamp": "2025-08-10T12:00:00Z",
  "request_id": "req_uuid_456",
  "path": "/api/v2/factors/calculate"
}
```

### 7.3 业务错误代码

| 错误代码 | 描述 | HTTP状态码 |
|---------|------|------------|
| `VALIDATION_ERROR` | 参数验证失败 | 400 |
| `AUTHENTICATION_FAILED` | 认证失败 | 401 |
| `INSUFFICIENT_PERMISSIONS` | 权限不足 | 403 |
| `RESOURCE_NOT_FOUND` | 资源不存在 | 404 |
| `FACTOR_NOT_FOUND` | 因子不存在 | 404 |
| `BACKTEST_NOT_FOUND` | 回测任务不存在 | 404 |
| `DATA_NOT_AVAILABLE` | 数据不可用 | 422 |
| `CALCULATION_FAILED` | 计算失败 | 422 |
| `QUOTA_EXCEEDED` | 配额超限 | 429 |
| `RATE_LIMIT_EXCEEDED` | 请求频率超限 | 429 |
| `INTERNAL_ERROR` | 内部错误 | 500 |
| `SERVICE_UNAVAILABLE` | 服务不可用 | 503 |

---

## 8. 性能和限制

### 8.1 API限制

| 限制类型 | 免费版 | 高级版 | 企业版 |
|---------|-------|-------|-------|
| 请求频率 | 100次/分钟 | 1000次/分钟 | 无限制 |
| 月度请求总数 | 10,000 | 100,000 | 无限制 |
| 并发回测任务 | 1 | 5 | 20 |
| 因子计算并发 | 2 | 10 | 50 |
| 数据查询范围 | 1年 | 5年 | 无限制 |
| 实时订阅数 | 10 | 100 | 1000 |

### 8.2 响应时间SLA

| API类型 | 目标响应时间 | 可用性 |
|---------|-------------|--------|
| 因子计算 | < 3秒 | 99.9% |
| 数据查询 | < 1秒 | 99.9% |
| 回测创建 | < 2秒 | 99.5% |
| 实时数据 | < 100ms | 99.95% |
| 系统状态 | < 500ms | 99.99% |

### 8.3 数据分页

```json
{
  "pagination": {
    "page": 1,           // 当前页码
    "per_page": 50,      // 每页数量
    "total": 1000,       // 总记录数
    "total_pages": 20,   // 总页数
    "has_next": true,    // 是否有下一页
    "has_prev": false,   // 是否有上一页
    "next_page": 2,      // 下一页页码
    "prev_page": null    // 上一页页码
  },
  "links": {
    "first": "/api/v2/factors/library?page=1&per_page=50",
    "last": "/api/v2/factors/library?page=20&per_page=50", 
    "next": "/api/v2/factors/library?page=2&per_page=50",
    "prev": null
  }
}
```

---

## 9. 版本兼容性

### 9.1 API版本策略

- **v1 (Legacy)**: 维护模式，兼容现有客户端
- **v2 (Current)**: 当前版本，新功能开发
- **v3 (Future)**: 未来版本，重大升级

### 9.2 向后兼容保证

```http
# V1 API (兼容模式)
GET /api/v1/data/overview
# 自动路由到对应的V2端点并转换响应格式

# V2 API (推荐)  
GET /api/v2/data/overview
# 原生V2响应格式
```

### 9.3 废弃策略

```json
{
  "warning": {
    "type": "DEPRECATION_WARNING",
    "message": "此API端点将在2025年12月31日后停止支持",
    "deprecated_endpoint": "/api/v1/factors/generate",
    "replacement": "/api/v2/ai/factors/generate",
    "migration_guide": "https://docs.quantanalyzer.pro/migration/v1-to-v2"
  }
}
```

---

## 10. OpenAPI规范

### 10.1 OpenAPI文档

```yaml
openapi: 3.0.0
info:
  title: QuantAnalyzer Pro API
  description: AI-driven quantitative analysis platform API
  version: "2.0.0"
  contact:
    name: API Support
    url: https://support.quantanalyzer.pro
    email: api@quantanalyzer.pro
  license:
    name: Proprietary
servers:
  - url: https://api.quantanalyzer.pro/api/v2
    description: Production server
  - url: https://staging-api.quantanalyzer.pro/api/v2
    description: Staging server

components:
  securitySchemes:
    bearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      
  schemas:
    StandardResponse:
      type: object
      properties:
        success:
          type: boolean
        data:
          type: object
        error:
          type: string
        timestamp:
          type: string
          format: date-time
        request_id:
          type: string
          format: uuid
          
    FactorDefinition:
      type: object
      required:
        - id
        - name
        - category
        - formula
      properties:
        id:
          type: string
          example: "rsi_14"
        name:
          type: string
          example: "RSI_14"
        category:
          type: string
          enum: [technical, statistical, fundamental, sentiment]
        formula:
          type: string
          example: "rsi(close, 14)"
        parameters:
          type: object
        description:
          type: string

security:
  - bearerAuth: []

paths:
  /factors/calculate:
    post:
      summary: 批量计算因子
      tags: [Factors]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required:
                - factors
                - symbols
                - start_date
                - end_date
              properties:
                factors:
                  type: array
                  items:
                    $ref: '#/components/schemas/FactorDefinition'
                symbols:
                  type: array
                  items:
                    type: string
                start_date:
                  type: string
                  format: date
                end_date:
                  type: string
                  format: date
      responses:
        '200':
          description: 计算成功
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/StandardResponse'
        '400':
          description: 请求参数错误
        '401':
          description: 未认证
        '429':
          description: 请求过多
```

---

## 总结

本API接口规范文档提供了：

### 🚀 核心特性

1. **RESTful设计**: 标准REST架构，资源导向的URL设计
2. **版本控制**: V1/V2版本并行，平滑升级路径
3. **统一响应**: 标准化的请求响应格式
4. **实时支持**: WebSocket实时数据推送
5. **完整文档**: OpenAPI 3.0规范，自动生成文档

### 💡 设计优势

- **开发友好**: 清晰的接口设计和错误处理
- **性能优化**: 分页、缓存、压缩支持
- **安全可靠**: JWT认证 + HTTPS + 限流保护
- **扩展性强**: 模块化设计，易于扩展新功能

### 🛠 使用便利

- **SDK支持**: Python、JavaScript、Java等多语言SDK
- **交互式文档**: Swagger UI在线测试
- **代码生成**: 基于OpenAPI自动生成客户端代码
- **监控完善**: 请求追踪、性能监控、错误报告

该API规范为量化分析系统提供了完整、标准、高效的接口设计，支持各种客户端和第三方系统的集成。