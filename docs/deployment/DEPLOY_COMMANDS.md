# 🚀 VPS快速部署命令

## 一、上传项目到VPS

### 方法1: 使用scp上传
```bash
# 本地执行 - 压缩项目文件
cd /Users/zhaoleon/Desktop
tar -czf trader.tar.gz --exclude=venv --exclude=node_modules --exclude=.git trader/

# 上传到VPS
scp trader.tar.gz root@your-vps-ip:/root/

# VPS上解压
ssh root@your-vps-ip
cd /root
tar -xzf trader.tar.gz
```

### 方法2: 使用rsync同步（推荐）
```bash
# 本地执行 - 同步到VPS
rsync -avz --progress --exclude=venv --exclude=node_modules --exclude=.git \
  /Users/zhaoleon/Desktop/trader/ root@your-vps-ip:/root/trader/
```

### 方法3: Git克隆（如有代码仓库）
```bash
# VPS执行
ssh root@your-vps-ip
git clone https://github.com/your-username/trader.git /root/trader
cd /root/trader
```

## 二、VPS系统准备

### 连接VPS
```bash
ssh root@your-vps-ip
```

### 更新系统
```bash
apt update && apt upgrade -y
apt install -y curl wget git build-essential
```

## 三、执行一键部署

### 运行部署脚本
```bash
# 进入项目目录
cd /root/trader

# 赋予执行权限
chmod +x scripts/deploy_production.sh

# 执行部署
./scripts/deploy_production.sh
```

### 手动分步部署（如自动脚本失败）

#### 1. 安装Python环境
```bash
apt install -y python3 python3-pip python3-venv python3-dev
python3 -m pip install --upgrade pip
```

#### 2. 安装MongoDB
```bash
# 添加MongoDB GPG密钥
curl -fsSL https://pgp.mongodb.com/server-6.0.asc | gpg -o /usr/share/keyrings/mongodb-server-6.0.gpg --dearmor

# 添加MongoDB源
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-6.0.gpg ] https://repo.mongodb.org/apt/ubuntu $(lsb_release -cs)/mongodb-org/6.0 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-6.0.list

# 安装MongoDB
apt update
apt install -y mongodb-org

# 启动服务
systemctl start mongod
systemctl enable mongod
```

#### 3. 安装Redis
```bash
apt install -y redis-server
systemctl start redis-server
systemctl enable redis-server
```

#### 4. 安装Nginx
```bash
apt install -y nginx
systemctl start nginx
systemctl enable nginx
```

#### 5. 创建用户和虚拟环境
```bash
# 创建trader用户
adduser --disabled-password --gecos "" trader
usermod -aG sudo trader

# 设置项目目录
mkdir -p /home/trader/quantum-trader
cp -r /root/trader/* /home/trader/quantum-trader/
chown -R trader:trader /home/trader/quantum-trader

# 创建虚拟环境
su - trader
cd /home/trader/quantum-trader
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 四、配置应用

### 1. 配置环境变量
```bash
# 复制配置模板
cp .env.production .env

# 编辑配置文件
nano .env

# 填入必要的API密钥：
# OKX_API_KEY=your_okx_api_key
# OKX_SECRET_KEY=your_okx_secret_key
# OKX_PASSPHRASE=your_okx_passphrase
# BINANCE_API_KEY=your_binance_api_key
# BINANCE_SECRET_KEY=your_binance_secret_key
# DEEPSEEK_API_KEY=your_deepseek_api_key
```

### 2. 创建系统服务
```bash
# 创建systemd服务文件
cat > /etc/systemd/system/quantum-trader.service << 'EOF'
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

[Install]
WantedBy=multi-user.target
EOF

# 重载配置并启动服务
systemctl daemon-reload
systemctl enable quantum-trader
systemctl start quantum-trader
```

### 3. 配置Nginx反向代理
```bash
# 创建Nginx配置
cat > /etc/nginx/sites-available/quantum-trader << 'EOF'
server {
    listen 80;
    server_name _;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

# 启用配置
ln -s /etc/nginx/sites-available/quantum-trader /etc/nginx/sites-enabled/
rm /etc/nginx/sites-enabled/default
nginx -t
systemctl reload nginx
```

## 五、验证部署

### 检查服务状态
```bash
# 检查所有服务状态
systemctl status quantum-trader mongod redis-server nginx

# 检查端口监听
netstat -tlnp | grep -E "(8000|27017|6379|80)"
```

### 健康检查
```bash
# API健康检查
curl http://localhost/health
curl http://your-vps-ip/health

# 查看服务日志
journalctl -u quantum-trader -f
```

### 系统监控
```bash
# 查看系统资源
htop
df -h
free -h

# 查看应用日志
tail -f /home/trader/quantum-trader/logs/app.log
```

## 六、安全配置

### 配置防火墙
```bash
ufw --force reset
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp comment 'SSH'
ufw allow 80/tcp comment 'HTTP'
ufw allow 443/tcp comment 'HTTPS'
ufw --force enable
```

### SSL证书（可选）
```bash
# 安装Certbot
apt install -y certbot python3-certbot-nginx

# 获取证书（需要有域名）
certbot --nginx -d your-domain.com
```

## 七、日常维护命令

### 服务管理
```bash
# 重启应用
systemctl restart quantum-trader

# 查看实时日志
journalctl -u quantum-trader -f

# 重载Nginx配置
systemctl reload nginx
```

### 数据备份
```bash
# 创建备份脚本
cat > /home/trader/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/home/trader/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR
mongodump --db quantum_trader --out $BACKUP_DIR/mongo/
cp /home/trader/quantum-trader/.env $BACKUP_DIR/
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR/
rm -rf $BACKUP_DIR/
EOF

chmod +x /home/trader/backup.sh
./backup.sh
```

### 系统监控脚本
```bash
# 创建监控脚本
cat > /home/trader/monitor.sh << 'EOF'
#!/bin/bash
echo "=== 系统监控 $(date) ==="
echo "负载: $(uptime)"
echo "内存: $(free -h | grep Mem)"
echo "磁盘: $(df -h / | tail -1)"

services=("quantum-trader" "mongod" "redis-server" "nginx")
for svc in "${services[@]}"; do
    if systemctl is-active --quiet $svc; then
        echo "✅ $svc: 运行正常"
    else
        echo "❌ $svc: 服务异常"
    fi
done

if curl -f -s http://localhost:8000/health > /dev/null; then
    echo "✅ API: 响应正常"
else
    echo "❌ API: 无响应"
fi
EOF

chmod +x /home/trader/monitor.sh
```

## 八、故障排查

### 常见问题解决

#### 服务无法启动
```bash
# 查看详细日志
journalctl -u quantum-trader -n 100

# 检查端口占用
netstat -tlnp | grep 8000
```

#### 数据库连接失败
```bash
# 检查MongoDB状态
systemctl status mongod
mongosh --eval "db.runCommand('ping')"
```

#### 内存不足
```bash
# 查看内存使用
free -h
# 创建交换文件
dd if=/dev/zero of=/swapfile bs=1024 count=2048000
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

## 九、性能优化

### 系统优化
```bash
# 优化内核参数
cat >> /etc/sysctl.conf << 'EOF'
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
vm.swappiness = 10
fs.file-max = 65536
EOF

sysctl -p
```

### 数据库优化
```javascript
// MongoDB索引优化
use quantum_trader
db.market_data.createIndex({"symbol": 1, "timestamp": -1})
db.trades.createIndex({"created_at": -1})
```

## 十、完成验证

部署完成后访问以下地址验证：

- **主页**: http://your-vps-ip
- **健康检查**: http://your-vps-ip/health
- **API文档**: http://your-vps-ip/docs
- **系统状态**: http://your-vps-ip/metrics

---

## 🆘 紧急联系

如果遇到部署问题：
1. 检查 `/var/log/syslog` 系统日志
2. 运行 `/home/trader/monitor.sh` 检查状态
3. 查看应用日志 `journalctl -u quantum-trader -f`

**部署成功标志**: 访问 `http://your-vps-ip` 看到系统界面！ 🎉