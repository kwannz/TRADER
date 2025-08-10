#!/bin/bash
# AI量化交易系统 - 生产环境部署脚本
# 使用方法: ./scripts/deploy_production.sh [options]

set -e  # 遇到错误立即退出

# 配置变量
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY_USER="trader"
SERVICE_NAME="quantum-trader"
PYTHON_VERSION="3.9"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO:${NC} $1"
}

# 显示横幅
show_banner() {
    echo -e "${BLUE}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════════╗
║               🚀 AI量化交易系统 - 生产环境部署                    ║
║                                                                  ║
║  本脚本将完成以下任务：                                           ║
║  • 系统环境检查和准备                                            ║
║  • 依赖服务安装配置                                              ║
║  • 应用环境配置                                                  ║
║  • 系统服务创建                                                  ║
║  • 安全配置和优化                                                ║
╚══════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# 检查是否为root用户
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "请不要使用root用户运行此脚本"
        exit 1
    fi
    
    # 检查sudo权限
    if ! sudo -n true 2>/dev/null; then
        error "需要sudo权限，请确保当前用户在sudoers中"
        exit 1
    fi
}

# 检查操作系统
check_os() {
    if [[ ! -f /etc/os-release ]]; then
        error "不支持的操作系统"
        exit 1
    fi
    
    source /etc/os-release
    
    if [[ "$ID" != "ubuntu" && "$ID" != "debian" ]]; then
        error "此脚本仅支持Ubuntu和Debian系统"
        exit 1
    fi
    
    log "检测到操作系统: $PRETTY_NAME"
}

# 更新系统
update_system() {
    log "更新系统软件包..."
    sudo apt update && sudo apt upgrade -y
    
    log "安装基础工具..."
    sudo apt install -y \
        curl \
        wget \
        git \
        build-essential \
        software-properties-common \
        apt-transport-https \
        ca-certificates \
        gnupg \
        lsb-release \
        htop \
        iotop \
        unzip
}

# 安装Python
install_python() {
    log "检查Python版本..."
    
    if ! command -v python3 &> /dev/null; then
        log "安装Python 3..."
        sudo apt install -y python3 python3-pip python3-venv python3-dev
    fi
    
    CURRENT_PYTHON=$(python3 --version | cut -d ' ' -f 2)
    log "当前Python版本: $CURRENT_PYTHON"
    
    # 安装pip依赖
    log "升级pip..."
    python3 -m pip install --upgrade pip
}

# 安装MongoDB
install_mongodb() {
    log "安装MongoDB..."
    
    if ! command -v mongod &> /dev/null; then
        # 添加MongoDB官方GPG密钥
        curl -fsSL https://pgp.mongodb.com/server-6.0.asc | sudo gpg -o /usr/share/keyrings/mongodb-server-6.0.gpg --dearmor
        
        # 添加MongoDB源
        echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-6.0.gpg ] https://repo.mongodb.org/apt/ubuntu $(lsb_release -cs)/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
        
        # 安装MongoDB
        sudo apt update
        sudo apt install -y mongodb-org
    else
        log "MongoDB已安装"
    fi
    
    # 配置MongoDB
    sudo systemctl start mongod
    sudo systemctl enable mongod
    
    # 创建数据库和用户
    log "配置MongoDB数据库..."
    mongosh --eval "
        use quantum_trader
        db.createUser({
            user: 'trader_app',
            pwd: '$(openssl rand -base64 32)',
            roles: [{ role: 'readWrite', db: 'quantum_trader' }]
        })
    " 2>/dev/null || true
    
    log "MongoDB安装完成"
}

# 安装Redis
install_redis() {
    log "安装Redis..."
    
    if ! command -v redis-server &> /dev/null; then
        sudo apt install -y redis-server
    else
        log "Redis已安装"
    fi
    
    # 配置Redis
    sudo sed -i 's/^bind 127.0.0.1 ::1/bind 127.0.0.1/' /etc/redis/redis.conf
    sudo sed -i 's/^# requirepass foobared/requirepass $(openssl rand -base64 32)/' /etc/redis/redis.conf
    
    sudo systemctl start redis-server
    sudo systemctl enable redis-server
    
    log "Redis安装完成"
}

# 安装Nginx
install_nginx() {
    log "安装Nginx..."
    
    if ! command -v nginx &> /dev/null; then
        sudo apt install -y nginx
    else
        log "Nginx已安装"
    fi
    
    sudo systemctl start nginx
    sudo systemctl enable nginx
    
    log "Nginx安装完成"
}

# 创建部署用户
create_deploy_user() {
    if ! id "$DEPLOY_USER" &>/dev/null; then
        log "创建部署用户: $DEPLOY_USER"
        sudo adduser --disabled-password --gecos "" $DEPLOY_USER
        sudo usermod -aG sudo $DEPLOY_USER
    else
        log "用户 $DEPLOY_USER 已存在"
    fi
}

# 设置项目环境
setup_project_env() {
    log "设置项目环境..."
    
    PROJECT_DIR="/home/$DEPLOY_USER/$SERVICE_NAME"
    
    # 创建项目目录
    sudo -u $DEPLOY_USER mkdir -p $PROJECT_DIR
    
    # 复制项目文件
    log "复制项目文件..."
    sudo -u $DEPLOY_USER cp -r $PROJECT_ROOT/* $PROJECT_DIR/
    sudo -u $DEPLOY_USER cp $PROJECT_ROOT/.env.example $PROJECT_DIR/.env
    
    # 设置权限
    sudo chown -R $DEPLOY_USER:$DEPLOY_USER $PROJECT_DIR
    sudo chmod 700 $PROJECT_DIR
    sudo chmod 600 $PROJECT_DIR/.env
    
    # 创建虚拟环境
    log "创建Python虚拟环境..."
    sudo -u $DEPLOY_USER python3 -m venv $PROJECT_DIR/venv
    
    # 安装依赖
    log "安装Python依赖包..."
    sudo -u $DEPLOY_USER $PROJECT_DIR/venv/bin/pip install --upgrade pip
    sudo -u $DEPLOY_USER $PROJECT_DIR/venv/bin/pip install -r $PROJECT_DIR/requirements.txt
    
    # 创建日志目录
    sudo -u $DEPLOY_USER mkdir -p $PROJECT_DIR/logs
    sudo -u $DEPLOY_USER mkdir -p $PROJECT_DIR/data
    sudo -u $DEPLOY_USER mkdir -p $PROJECT_DIR/backups
    
    log "项目环境设置完成"
}

# 生成配置文件
generate_config() {
    log "生成生产环境配置..."
    
    PROJECT_DIR="/home/$DEPLOY_USER/$SERVICE_NAME"
    
    # 生成随机密钥
    DB_PASSWORD=$(openssl rand -base64 32)
    REDIS_PASSWORD=$(openssl rand -base64 32)
    SECRET_KEY=$(openssl rand -base64 64)
    
    # 创建.env文件
    sudo -u $DEPLOY_USER tee $PROJECT_DIR/.env > /dev/null << EOF
# 生产环境配置
ENVIRONMENT=production
DEBUG=false

# 服务配置
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY=$SECRET_KEY

# 数据库配置
MONGODB_URL=mongodb://trader_app:$DB_PASSWORD@localhost:27017/quantum_trader
REDIS_URL=redis://:$REDIS_PASSWORD@localhost:6379/0

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=/home/$DEPLOY_USER/$SERVICE_NAME/logs/app.log

# 系统配置
MAX_WORKERS=4
KEEP_ALIVE_TIMEOUT=5
WORKER_CONNECTIONS=1000

# API密钥 (请手动配置)
OKX_API_KEY=
OKX_SECRET_KEY=
OKX_PASSPHRASE=
BINANCE_API_KEY=
BINANCE_SECRET_KEY=
DEEPSEEK_API_KEY=
GEMINI_API_KEY=

# 风控配置
HARD_STOP_LOSS=1000
MAX_POSITION_SIZE=10000
RISK_LEVEL=conservative
EOF

    log "配置文件已生成: $PROJECT_DIR/.env"
    warn "请手动编辑 $PROJECT_DIR/.env 文件，填入真实的API密钥"
}

# 创建系统服务
create_systemd_service() {
    log "创建系统服务..."
    
    PROJECT_DIR="/home/$DEPLOY_USER/$SERVICE_NAME"
    
    # 创建systemd服务文件
    sudo tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null << EOF
[Unit]
Description=AI Quantum Trading System
After=network.target mongod.service redis.service
Wants=mongod.service redis.service
StartLimitBurst=5
StartLimitIntervalSec=10

[Service]
Type=simple
User=$DEPLOY_USER
Group=$DEPLOY_USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
EnvironmentFile=$PROJECT_DIR/.env
ExecStart=$PROJECT_DIR/venv/bin/python $PROJECT_DIR/scripts/start_server.py --port 8000
ExecReload=/bin/kill -HUP \$MAINPID

# 重启策略
Restart=always
RestartSec=5

# 输出到日志
StandardOutput=journal
StandardError=journal
SyslogIdentifier=$SERVICE_NAME

# 安全设置
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
ReadWritePaths=$PROJECT_DIR

# 资源限制
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
EOF

    # 重载systemd配置
    sudo systemctl daemon-reload
    sudo systemctl enable $SERVICE_NAME
    
    log "系统服务创建完成"
}

# 配置Nginx反向代理
configure_nginx() {
    log "配置Nginx反向代理..."
    
    # 备份默认配置
    sudo cp /etc/nginx/sites-available/default /etc/nginx/sites-available/default.backup
    
    # 创建新的配置
    sudo tee /etc/nginx/sites-available/$SERVICE_NAME > /dev/null << 'EOF'
# AI量化交易系统 Nginx配置
server {
    listen 80;
    listen [::]:80;
    server_name _;
    
    # 安全头
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    
    # 日志配置
    access_log /var/log/nginx/quantum-trader_access.log;
    error_log /var/log/nginx/quantum-trader_error.log;
    
    # 主应用代理
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # 超时设置
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
        
        # WebSocket支持
        proxy_buffering off;
    }
    
    # 健康检查
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        access_log off;
    }
    
    # 静态文件缓存
    location /static/ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        add_header X-Content-Type-Options nosniff;
    }
    
    # 限制文件上传大小
    client_max_body_size 100M;
    
    # Gzip压缩
    gzip on;
    gzip_vary on;
    gzip_min_length 10240;
    gzip_proxied expired no-cache no-store private must-revalidate auth;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/x-javascript
        application/javascript
        application/xml+rss
        application/json;
}
EOF

    # 启用配置
    sudo ln -sf /etc/nginx/sites-available/$SERVICE_NAME /etc/nginx/sites-enabled/$SERVICE_NAME
    sudo rm -f /etc/nginx/sites-enabled/default
    
    # 测试配置
    sudo nginx -t
    
    # 重载Nginx
    sudo systemctl reload nginx
    
    log "Nginx配置完成"
}

# 配置防火墙
configure_firewall() {
    log "配置防火墙..."
    
    # 安装ufw
    sudo apt install -y ufw
    
    # 设置默认策略
    sudo ufw --force reset
    sudo ufw default deny incoming
    sudo ufw default allow outgoing
    
    # 允许SSH
    sudo ufw allow 22/tcp comment 'SSH'
    
    # 允许HTTP/HTTPS
    sudo ufw allow 80/tcp comment 'HTTP'
    sudo ufw allow 443/tcp comment 'HTTPS'
    
    # 启用防火墙
    sudo ufw --force enable
    
    log "防火墙配置完成"
}

# 优化系统性能
optimize_system() {
    log "优化系统性能..."
    
    # 优化内核参数
    sudo tee /etc/sysctl.d/99-quantum-trader.conf > /dev/null << 'EOF'
# 网络优化
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 65536 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 5000
net.ipv4.tcp_congestion_control = bbr

# 文件描述符限制
fs.file-max = 65536

# 虚拟内存优化
vm.swappiness = 10
vm.vfs_cache_pressure = 50
EOF

    # 应用内核参数
    sudo sysctl -p /etc/sysctl.d/99-quantum-trader.conf
    
    # 设置用户限制
    sudo tee /etc/security/limits.d/quantum-trader.conf > /dev/null << EOF
$DEPLOY_USER soft nofile 65536
$DEPLOY_USER hard nofile 65536
$DEPLOY_USER soft nproc 4096
$DEPLOY_USER hard nproc 4096
EOF

    log "系统优化完成"
}

# 创建备份脚本
create_backup_script() {
    log "创建备份脚本..."
    
    PROJECT_DIR="/home/$DEPLOY_USER/$SERVICE_NAME"
    BACKUP_SCRIPT="$PROJECT_DIR/scripts/backup.sh"
    
    sudo -u $DEPLOY_USER tee $BACKUP_SCRIPT > /dev/null << EOF
#!/bin/bash
# 量化交易系统备份脚本

BACKUP_DIR="/home/$DEPLOY_USER/$SERVICE_NAME/backups"
DATE=\$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p \$BACKUP_DIR

# 备份MongoDB
echo "备份MongoDB数据库..."
mongodump --db quantum_trader --out \$BACKUP_DIR/mongo_\$DATE/

# 备份配置文件
echo "备份配置文件..."
cp /home/$DEPLOY_USER/$SERVICE_NAME/.env \$BACKUP_DIR/env_\$DATE
cp -r /home/$DEPLOY_USER/$SERVICE_NAME/config \$BACKUP_DIR/config_\$DATE/

# 备份日志
echo "备份日志文件..."
cp -r /home/$DEPLOY_USER/$SERVICE_NAME/logs \$BACKUP_DIR/logs_\$DATE/

# 压缩备份
echo "压缩备份文件..."
cd \$BACKUP_DIR
tar -czf backup_\$DATE.tar.gz mongo_\$DATE env_\$DATE config_\$DATE logs_\$DATE
rm -rf mongo_\$DATE env_\$DATE config_\$DATE logs_\$DATE

# 清理旧备份 (保留30天)
find \$BACKUP_DIR -name "backup_*.tar.gz" -mtime +30 -delete

echo "备份完成: \$BACKUP_DIR/backup_\$DATE.tar.gz"
EOF

    sudo -u $DEPLOY_USER chmod +x $BACKUP_SCRIPT
    
    # 添加定时任务
    (sudo -u $DEPLOY_USER crontab -l 2>/dev/null; echo "0 2 * * * $BACKUP_SCRIPT") | sudo -u $DEPLOY_USER crontab -
    
    log "备份脚本创建完成"
}

# 创建监控脚本
create_monitoring_script() {
    log "创建监控脚本..."
    
    PROJECT_DIR="/home/$DEPLOY_USER/$SERVICE_NAME"
    MONITOR_SCRIPT="$PROJECT_DIR/scripts/monitor.sh"
    
    sudo -u $DEPLOY_USER tee $MONITOR_SCRIPT > /dev/null << 'EOF'
#!/bin/bash
# 系统监控脚本

echo "=== AI量化交易系统监控报告 ==="
echo "时间: $(date)"
echo "主机: $(hostname)"
echo ""

# 系统负载
echo "=== 系统负载 ==="
uptime
echo ""

# 内存使用
echo "=== 内存使用 ==="
free -h
echo ""

# 磁盘使用
echo "=== 磁盘使用 ==="
df -h
echo ""

# 网络连接
echo "=== 网络连接 ==="
ss -tuln
echo ""

# 服务状态
echo "=== 服务状态 ==="
services=("quantum-trader" "mongod" "redis-server" "nginx")
for service in "${services[@]}"; do
    if systemctl is-active --quiet $service; then
        echo "✅ $service: 运行中"
    else
        echo "❌ $service: 停止"
    fi
done
echo ""

# 进程监控
echo "=== 进程监控 ==="
ps aux | grep -E "(python|mongod|redis|nginx)" | grep -v grep
echo ""

# API健康检查
echo "=== API健康检查 ==="
if curl -f -s http://localhost:8000/health > /dev/null; then
    echo "✅ API服务正常响应"
else
    echo "❌ API服务无响应"
fi
echo ""

# 日志监控
echo "=== 最近错误日志 ==="
sudo journalctl -u quantum-trader --since "1 hour ago" --grep "ERROR|CRITICAL" --no-pager -q || echo "无严重错误"
echo ""

echo "=== 监控报告结束 ==="
EOF

    sudo -u $DEPLOY_USER chmod +x $MONITOR_SCRIPT
    
    log "监控脚本创建完成"
}

# 安装完成验证
verify_installation() {
    log "验证安装..."
    
    # 检查服务状态
    services=("mongod" "redis-server" "nginx")
    for service in "${services[@]}"; do
        if systemctl is-active --quiet $service; then
            log "✅ $service 服务运行正常"
        else
            error "❌ $service 服务未运行"
        fi
    done
    
    # 检查端口监听
    ports=("27017" "6379" "80")
    for port in "${ports[@]}"; do
        if netstat -tlnp | grep -q ":$port "; then
            log "✅ 端口 $port 监听正常"
        else
            warn "⚠️  端口 $port 未监听"
        fi
    done
    
    log "验证完成"
}

# 显示部署完成信息
show_completion_info() {
    PROJECT_DIR="/home/$DEPLOY_USER/$SERVICE_NAME"
    
    echo -e "${GREEN}"
    cat << EOF

╔══════════════════════════════════════════════════════════════════╗
║                    🎉 部署完成！                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  项目目录: $PROJECT_DIR
║  配置文件: $PROJECT_DIR/.env
║  日志目录: $PROJECT_DIR/logs
║  备份目录: $PROJECT_DIR/backups
║                                                                  ║
║  服务管理命令:                                                    ║
║    启动: sudo systemctl start $SERVICE_NAME
║    停止: sudo systemctl stop $SERVICE_NAME
║    重启: sudo systemctl restart $SERVICE_NAME
║    状态: sudo systemctl status $SERVICE_NAME
║    日志: sudo journalctl -u $SERVICE_NAME -f
║                                                                  ║
║  监控命令:                                                        ║
║    系统监控: $PROJECT_DIR/scripts/monitor.sh
║    备份数据: $PROJECT_DIR/scripts/backup.sh
║                                                                  ║
║  ⚠️  重要提醒:                                                    ║
║    1. 请编辑配置文件填入API密钥                                   ║
║    2. 配置完成后启动服务                                          ║
║    3. 访问 http://your-server-ip 查看状态                        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

下一步操作:
1. 编辑配置: nano $PROJECT_DIR/.env
2. 启动服务: sudo systemctl start $SERVICE_NAME
3. 查看状态: sudo systemctl status $SERVICE_NAME

EOF
    echo -e "${NC}"
}

# 主函数
main() {
    show_banner
    
    # 参数解析
    SKIP_CHECKS=false
    INSTALL_OPTIONAL=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-checks)
                SKIP_CHECKS=true
                shift
                ;;
            --install-optional)
                INSTALL_OPTIONAL=true
                shift
                ;;
            --help|-h)
                echo "使用方法: $0 [options]"
                echo "  --skip-checks     跳过系统检查"
                echo "  --install-optional 安装可选组件"
                echo "  --help|-h         显示帮助"
                exit 0
                ;;
            *)
                error "未知参数: $1"
                exit 1
                ;;
        esac
    done
    
    # 执行部署步骤
    if [[ "$SKIP_CHECKS" != "true" ]]; then
        check_root
        check_os
    fi
    
    update_system
    install_python
    install_mongodb
    install_redis
    install_nginx
    create_deploy_user
    setup_project_env
    generate_config
    create_systemd_service
    configure_nginx
    configure_firewall
    optimize_system
    create_backup_script
    create_monitoring_script
    verify_installation
    show_completion_info
    
    log "🎉 生产环境部署完成！"
}

# 错误处理
trap 'error "部署过程中出现错误，行号: $LINENO"' ERR

# 运行主函数
main "$@"