# AI量化交易系统 VPS部署指南

## 📋 系统要求

### 最低配置
- **CPU:** 2核心
- **内存:** 4GB RAM
- **存储:** 20GB SSD
- **网络:** 1Mbps稳定带宽
- **系统:** Ubuntu 20.04/22.04 LTS

### 推荐配置
- **CPU:** 4核心
- **内存:** 8GB RAM
- **存储:** 50GB SSD
- **网络:** 10Mbps稳定带宽

## 🛠️ 部署步骤

### 1. 系统准备

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装基础依赖
sudo apt install -y curl wget git build-essential software-properties-common

# 安装Python 3.9+
sudo apt install -y python3 python3-pip python3-venv python3-dev

# 安装MongoDB
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
sudo apt update
sudo apt install -y mongodb-org

# 启动MongoDB
sudo systemctl start mongod
sudo systemctl enable mongod

# 安装Redis
sudo apt install -y redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server

# 安装Nginx (可选，用于反向代理)
sudo apt install -y nginx
sudo systemctl start nginx
sudo systemctl enable nginx
```

### 2. 创建部署用户

```bash
# 创建专用用户
sudo adduser trader
sudo usermod -aG sudo trader

# 切换到trader用户
su - trader
```

### 3. 下载项目代码

```bash
# 下载项目 (替换为您的实际项目路径)
git clone <your-repo-url> /home/trader/quantum-trader
cd /home/trader/quantum-trader

# 或者直接上传代码包
# scp -r ./trader trader@your-vps-ip:/home/trader/quantum-trader
```

### 4. 配置Python环境

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 验证安装
python -c "import fastapi, uvicorn, pymongo, redis; print('依赖安装成功')"
```

### 5. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑环境变量
nano .env
```

**环境变量配置：**
```bash
# 服务配置
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=false

# 数据库配置
MONGODB_URL=mongodb://localhost:27017/quantum_trader
REDIS_URL=redis://localhost:6379/0

# API密钥 (请填入真实密钥)
OKX_API_KEY=your_okx_api_key
OKX_SECRET_KEY=your_okx_secret
OKX_PASSPHRASE=your_passphrase
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET_KEY=your_binance_secret

# AI服务配置
DEEPSEEK_API_KEY=your_deepseek_key
GEMINI_API_KEY=your_gemini_key

# 系统配置
ENVIRONMENT=production
LOG_LEVEL=INFO
```

### 6. 配置Systemd服务

创建系统服务文件：
```bash
sudo nano /etc/systemd/system/quantum-trader.service
```

**服务配置：**
```ini
[Unit]
Description=AI Quantum Trading System
After=network.target mongod.service redis.service
Wants=mongod.service redis.service

[Service]
Type=simple
User=trader
Group=trader
WorkingDirectory=/home/trader/quantum-trader
Environment=PATH=/home/trader/quantum-trader/venv/bin
ExecStart=/home/trader/quantum-trader/venv/bin/python scripts/start_server.py --port 8000
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=quantum-trader

# 安全设置
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=/home/trader/quantum-trader

[Install]
WantedBy=multi-user.target
```

**启动服务：**
```bash
# 重载systemd配置
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start quantum-trader

# 开机自启
sudo systemctl enable quantum-trader

# 查看服务状态
sudo systemctl status quantum-trader

# 查看日志
sudo journalctl -u quantum-trader -f
```

### 7. 配置Nginx反向代理 (可选)

```bash
sudo nano /etc/nginx/sites-available/quantum-trader
```

**Nginx配置：**
```nginx
server {
    listen 80;
    server_name your-domain.com;  # 替换为您的域名或IP

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # WebSocket支持
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }

    # 静态文件缓存
    location /static/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

**启用配置：**
```bash
# 启用站点
sudo ln -s /etc/nginx/sites-available/quantum-trader /etc/nginx/sites-enabled/

# 测试配置
sudo nginx -t

# 重载Nginx
sudo systemctl reload nginx
```

### 8. 配置防火墙

```bash
# 允许SSH
sudo ufw allow 22

# 允许HTTP/HTTPS
sudo ufw allow 80
sudo ufw allow 443

# 允许应用端口 (如果不使用Nginx)
sudo ufw allow 8000

# 启用防火墙
sudo ufw --force enable
```

### 9. 配置SSL证书 (推荐)

使用Let's Encrypt免费证书：
```bash
# 安装Certbot
sudo apt install -y certbot python3-certbot-nginx

# 获取证书 (替换为您的域名)
sudo certbot --nginx -d your-domain.com

# 自动续期
sudo crontab -e
# 添加以下行：
# 0 12 * * * /usr/bin/certbot renew --quiet
```

## 📊 部署验证

### 检查服务状态
```bash
# 检查系统服务
sudo systemctl status quantum-trader
sudo systemctl status mongod
sudo systemctl status redis-server
sudo systemctl status nginx

# 检查端口监听
sudo netstat -tlnp | grep -E "(8000|27017|6379|80|443)"

# 检查应用日志
sudo journalctl -u quantum-trader --since "1 hour ago"
```

### 健康检查
```bash
# API健康检查
curl http://localhost:8000/health

# 数据库连接测试
curl http://localhost:8000/api/system/status
```

## 🔧 运维管理

### 常用命令
```bash
# 重启服务
sudo systemctl restart quantum-trader

# 查看实时日志
sudo journalctl -u quantum-trader -f

# 查看性能指标
htop
df -h
free -h

# 备份数据库
mongodump --db quantum_trader --out /backup/$(date +%Y%m%d)
```

### 监控脚本
```bash
# 创建监控脚本
nano /home/trader/monitor.sh
```

```bash
#!/bin/bash
# 系统监控脚本

echo "=== 系统状态监控 ==="
echo "时间: $(date)"
echo "负载: $(uptime)"
echo "内存: $(free -h | grep Mem)"
echo "磁盘: $(df -h /)"
echo ""

# 检查服务状态
services=("quantum-trader" "mongod" "redis-server" "nginx")
for service in "${services[@]}"; do
    if systemctl is-active --quiet $service; then
        echo "✅ $service 运行正常"
    else
        echo "❌ $service 服务异常"
    fi
done

# 检查API响应
if curl -f -s http://localhost:8000/health > /dev/null; then
    echo "✅ API服务正常"
else
    echo "❌ API服务无响应"
fi
```

```bash
# 设置执行权限
chmod +x /home/trader/monitor.sh

# 设置定时任务
crontab -e
# 添加: */5 * * * * /home/trader/monitor.sh >> /var/log/system-monitor.log
```

## 🚨 故障排查

### 常见问题

1. **服务无法启动**
   ```bash
   # 检查日志
   sudo journalctl -u quantum-trader -n 50
   
   # 检查端口占用
   sudo netstat -tlnp | grep 8000
   ```

2. **数据库连接失败**
   ```bash
   # 检查MongoDB状态
   sudo systemctl status mongod
   
   # 检查连接
   mongosh --eval "db.runCommand('ping')"
   ```

3. **API密钥错误**
   ```bash
   # 检查环境变量
   source /home/trader/quantum-trader/venv/bin/activate
   python -c "import os; print(os.getenv('OKX_API_KEY'))"
   ```

### 性能优化

1. **内存优化**
   ```bash
   # 调整Python内存使用
   export PYTHONMALLOC=malloc
   ```

2. **数据库优化**
   ```javascript
   // MongoDB索引优化
   use quantum_trader
   db.market_data.createIndex({"symbol": 1, "timestamp": -1})
   db.trades.createIndex({"created_at": -1})
   ```

3. **Redis优化**
   ```bash
   # Redis内存优化
   sudo nano /etc/redis/redis.conf
   # 添加: maxmemory 1gb
   # 添加: maxmemory-policy allkeys-lru
   ```

## 📈 扩展部署

### 多实例部署
```bash
# 创建多个服务实例
sudo cp /etc/systemd/system/quantum-trader.service /etc/systemd/system/quantum-trader-2.service

# 修改端口配置
sudo nano /etc/systemd/system/quantum-trader-2.service
# ExecStart=... --port 8001
```

### 负载均衡配置
```nginx
upstream quantum_trader {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
}

server {
    location / {
        proxy_pass http://quantum_trader;
    }
}
```

## 🔐 安全配置

### SSH安全
```bash
# 修改SSH配置
sudo nano /etc/ssh/sshd_config
# PasswordAuthentication no
# PermitRootLogin no
# Port 2222

sudo systemctl restart ssh
```

### 应用安全
```bash
# 设置文件权限
chmod 600 /home/trader/quantum-trader/.env
chmod 700 /home/trader/quantum-trader/logs/
```

---

## ⚡ 快速部署脚本

您可以使用以下一键部署脚本：

```bash
# 下载并运行部署脚本
curl -fsSL https://raw.githubusercontent.com/your-repo/deploy.sh | bash
```

## 📞 技术支持

如遇问题，请检查：
1. 系统日志：`sudo journalctl -u quantum-trader -f`
2. 应用日志：`tail -f /home/trader/quantum-trader/logs/system.log`
3. 性能监控：`htop` 和 `iotop`

---
**部署完成后，访问 `http://your-vps-ip` 查看系统运行状态！** 🎉