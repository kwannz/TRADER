-- QuantAnalyzer Pro ClickHouse 数据库架构
-- 用于存储时间序列数据和高频量化数据

-- 创建数据库
CREATE DATABASE IF NOT EXISTS quant_data;

-- 使用数据库
USE quant_data;

-- 1. 市场OHLCV数据表
CREATE TABLE IF NOT EXISTS market_ohlcv (
    timestamp DateTime64(3, 'UTC'),
    symbol LowCardinality(String),
    exchange LowCardinality(String),
    interval LowCardinality(String),
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume Float64,
    quote_volume Float64,
    trades_count UInt64,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (toYYYYMM(timestamp), exchange)
ORDER BY (symbol, interval, timestamp)
TTL timestamp + INTERVAL 2 YEAR
SETTINGS index_granularity = 8192;

-- 2. 技术因子数据表
CREATE TABLE IF NOT EXISTS technical_factors (
    timestamp DateTime64(3, 'UTC'),
    symbol LowCardinality(String),
    interval LowCardinality(String),
    factor_name LowCardinality(String),
    factor_value Float64,
    parameters Map(String, Float64),
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (toYYYYMM(timestamp), factor_name)
ORDER BY (symbol, factor_name, interval, timestamp)
TTL timestamp + INTERVAL 1 YEAR
SETTINGS index_granularity = 8192;

-- 3. 基本面因子数据表
CREATE TABLE IF NOT EXISTS fundamental_factors (
    date Date,
    symbol LowCardinality(String),
    factor_name LowCardinality(String),
    factor_value Float64,
    period LowCardinality(String), -- quarterly, annual
    fiscal_year UInt16,
    fiscal_quarter UInt8,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (toYYYYMM(date), factor_name)
ORDER BY (symbol, factor_name, date)
SETTINGS index_granularity = 8192;

-- 4. 订单簿数据表（Level 2数据）
CREATE TABLE IF NOT EXISTS order_book (
    timestamp DateTime64(6, 'UTC'),
    symbol LowCardinality(String),
    exchange LowCardinality(String),
    side Enum8('bid' = 1, 'ask' = 2),
    price Float64,
    size Float64,
    level UInt8,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (toYYYYMMDD(timestamp), exchange)
ORDER BY (symbol, timestamp, side, level)
TTL timestamp + INTERVAL 7 DAY
SETTINGS index_granularity = 8192;

-- 5. 交易数据表（实时成交）
CREATE TABLE IF NOT EXISTS trades (
    timestamp DateTime64(6, 'UTC'),
    symbol LowCardinality(String),
    exchange LowCardinality(String),
    trade_id String,
    price Float64,
    size Float64,
    side Enum8('buy' = 1, 'sell' = 2),
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (toYYYYMMDD(timestamp), exchange)
ORDER BY (symbol, timestamp)
TTL timestamp + INTERVAL 30 DAY
SETTINGS index_granularity = 8192;

-- 6. 资金费率数据表
CREATE TABLE IF NOT EXISTS funding_rates (
    timestamp DateTime64(3, 'UTC'),
    symbol LowCardinality(String),
    exchange LowCardinality(String),
    funding_rate Float64,
    predicted_rate Float64,
    next_funding_time DateTime64(3, 'UTC'),
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (toYYYYMM(timestamp), exchange)
ORDER BY (symbol, exchange, timestamp)
TTL timestamp + INTERVAL 6 MONTH
SETTINGS index_granularity = 8192;

-- 7. 持仓量数据表
CREATE TABLE IF NOT EXISTS open_interest (
    timestamp DateTime64(3, 'UTC'),
    symbol LowCardinality(String),
    exchange LowCardinality(String),
    open_interest Float64,
    open_interest_value Float64,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (toYYYYMM(timestamp), exchange)
ORDER BY (symbol, exchange, timestamp)
TTL timestamp + INTERVAL 6 MONTH
SETTINGS index_granularity = 8192;

-- 8. 清算数据表
CREATE TABLE IF NOT EXISTS liquidations (
    timestamp DateTime64(3, 'UTC'),
    symbol LowCardinality(String),
    exchange LowCardinality(String),
    side Enum8('long' = 1, 'short' = 2),
    size Float64,
    price Float64,
    value Float64,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (toYYYYMMDD(timestamp), exchange)
ORDER BY (symbol, exchange, timestamp)
TTL timestamp + INTERVAL 3 MONTH
SETTINGS index_granularity = 8192;

-- 9. 回测结果数据表
CREATE TABLE IF NOT EXISTS backtest_results (
    backtest_id UUID,
    timestamp DateTime64(3, 'UTC'),
    symbol LowCardinality(String),
    strategy_name LowCardinality(String),
    position Float64,
    pnl Float64,
    cumulative_pnl Float64,
    trades_count UInt32,
    parameters Map(String, String),
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (backtest_id, symbol, timestamp)
TTL timestamp + INTERVAL 1 YEAR
SETTINGS index_granularity = 8192;

-- 10. 策略信号数据表
CREATE TABLE IF NOT EXISTS strategy_signals (
    timestamp DateTime64(3, 'UTC'),
    strategy_id UUID,
    symbol LowCardinality(String),
    signal Enum8('buy' = 1, 'sell' = 2, 'hold' = 3),
    confidence Float64,
    price Float64,
    factors Map(String, Float64),
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY (toYYYYMM(timestamp), strategy_id)
ORDER BY (symbol, timestamp)
TTL timestamp + INTERVAL 6 MONTH
SETTINGS index_granularity = 8192;

-- 创建物化视图用于实时聚合

-- 1. 1分钟OHLCV聚合视图
CREATE MATERIALIZED VIEW IF NOT EXISTS market_ohlcv_1m_mv TO market_ohlcv AS
SELECT
    toStartOfMinute(timestamp) as timestamp,
    symbol,
    exchange,
    '1m' as interval,
    argMin(open, timestamp) as open,
    max(high) as high,
    min(low) as low,
    argMax(close, timestamp) as close,
    sum(volume) as volume,
    sum(quote_volume) as quote_volume,
    sum(trades_count) as trades_count,
    now() as created_at
FROM trades
GROUP BY symbol, exchange, toStartOfMinute(timestamp);

-- 2. 因子统计聚合视图
CREATE MATERIALIZED VIEW IF NOT EXISTS factor_stats_daily_mv
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (symbol, factor_name, date) AS
SELECT
    toDate(timestamp) as date,
    symbol,
    factor_name,
    avg(factor_value) as avg_value,
    min(factor_value) as min_value,
    max(factor_value) as max_value,
    stddevPop(factor_value) as stddev_value,
    count() as count
FROM technical_factors
GROUP BY symbol, factor_name, toDate(timestamp);

-- 创建索引以优化查询性能

-- 1. 市场数据按交易所和时间索引
ALTER TABLE market_ohlcv ADD INDEX idx_exchange_time (exchange, timestamp) TYPE minmax GRANULARITY 3;

-- 2. 因子数据按因子名称索引
ALTER TABLE technical_factors ADD INDEX idx_factor_name (factor_name) TYPE set(1000) GRANULARITY 1;

-- 3. 交易数据按价格范围索引
ALTER TABLE trades ADD INDEX idx_price_range (price) TYPE minmax GRANULARITY 4;

-- 创建TTL策略优化存储
ALTER TABLE trades MODIFY TTL timestamp + INTERVAL 30 DAY TO VOLUME 'cold';
ALTER TABLE order_book MODIFY TTL timestamp + INTERVAL 7 DAY TO VOLUME 'cold';
ALTER TABLE liquidations MODIFY TTL timestamp + INTERVAL 3 MONTH TO VOLUME 'cold';