#!/bin/bash

# Replit SSH连接脚本
# 使用方法: ./connect_replit.sh

# 设置SSH连接参数
SSH_KEY="~/.ssh/replit"
SSH_USER="f0cd19db-cb72-4cc1-89bd-a1caf15c5e2d"
SSH_HOST="f0cd19db-cb72-4cc1-89bd-a1caf15c5e2d-00-1q26osltsa01f.kirk.replit.dev"
SSH_PORT="22"

echo "🔗 连接到Replit开发环境..."
echo "用户: $SSH_USER"
echo "主机: $SSH_HOST"
echo "端口: $SSH_PORT"
echo "密钥: $SSH_KEY"
echo ""

# 执行SSH连接
ssh -i "$SSH_KEY" -p "$SSH_PORT" "$SSH_USER@$SSH_HOST"
