#!/bin/bash

# AI量化交易系统 - 快速启动脚本

echo "🚀 AI量化交易系统启动中..."
echo "=================================="

# 检查Python版本
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到Python3，请先安装Python"
    exit 1
fi

# 检查必要文件
required_files=("index.html" "styles.css" "app.js" "server.py")
missing_files=()

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        missing_files+=("$file")
    fi
done

if [[ ${#missing_files[@]} -gt 0 ]]; then
    echo "❌ 缺少必要文件: ${missing_files[*]}"
    exit 1
fi

echo "✅ 文件检查完成"
echo "🌐 启动前端服务器..."
echo ""

# 启动服务器
python3 server.py "$@"