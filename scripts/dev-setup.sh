#!/bin/bash

# QuantAnalyzer Pro 开发环境设置脚本
set -e

echo "🚀 QuantAnalyzer Pro 开发环境设置"
echo "=================================="

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

# 检查Docker Compose是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose未安装，请先安装Docker Compose"
    exit 1
fi

# 创建数据目录
echo "📁 创建数据目录..."
mkdir -p data/{clickhouse-01,clickhouse-02,postgres-master,postgres-slave,redis-master,redis-slave-1,redis-slave-2,prometheus,grafana}

# 设置权限
echo "🔐 设置目录权限..."
sudo chown -R 101:101 data/clickhouse-*
sudo chown -R 999:999 data/postgres-*
sudo chown -R 999:999 data/redis-*
sudo chown -R 472:472 data/grafana
sudo chown -R 65534:65534 data/prometheus

# 启动基础设施服务
echo "🐳 启动基础设施服务..."
docker-compose up -d clickhouse-01 clickhouse-02 postgres-master redis-master

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 30

# 检查服务状态
echo "✅ 检查服务状态..."
docker-compose ps

# 初始化ClickHouse
echo "🗄️ 初始化ClickHouse..."
docker exec clickhouse-01 clickhouse-client --password=quant_password --query="
CREATE DATABASE IF NOT EXISTS quant_data;
CREATE TABLE IF NOT EXISTS quant_data.market_data (
    timestamp DateTime64(3),
    symbol String,
    open Float64,
    high Float64,
    low Float64,
    close Float64,
    volume Float64
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (symbol, timestamp);
"

# 初始化PostgreSQL
echo "🐘 初始化PostgreSQL..."
docker exec postgres-master psql -U quant_user -d quantanalyzer -c "
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS strategies (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    name VARCHAR(100) NOT NULL,
    description TEXT,
    code TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"

# 测试Redis连接
echo "🔴 测试Redis连接..."
docker exec redis-master redis-cli -a quant_password ping

echo "✅ 开发环境设置完成！"
echo ""
echo "📋 服务访问地址："
echo "  - ClickHouse (节点1): http://localhost:8123 (用户: default, 密码: quant_password)"
echo "  - ClickHouse (节点2): http://localhost:8124 (用户: default, 密码: quant_password)"
echo "  - PostgreSQL: localhost:5432 (用户: quant_user, 密码: quant_password)"
echo "  - Redis: localhost:6379 (密码: quant_password)"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana: http://localhost:3000 (用户: admin, 密码: quant_admin)"
echo ""
echo "🔧 下一步："
echo "  1. 运行 'cd rust-engine && cargo build --release' 编译Rust引擎"
echo "  2. 运行 'docker-compose up -d' 启动所有服务"
echo "  3. 访问 http://localhost 查看应用"