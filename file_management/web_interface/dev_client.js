/**
 * AI量化交易系统 - 开发客户端热重载脚本
 * 用于实现前端自动刷新功能
 */

class DevClient {
    constructor() {
        this.ws = null;
        this.reconnectInterval = 3000;
        this.maxReconnectAttempts = 10;
        this.reconnectAttempts = 0;
        this.isReconnecting = false;
        
        this.init();
    }
    
    init() {
        this.connect();
        
        // 页面卸载时清理连接
        window.addEventListener('beforeunload', () => {
            if (this.ws) {
                this.ws.close();
            }
        });
        
        console.log('🔧 开发模式已启动 - 热重载功能激活');
    }
    
    connect() {
        try {
            // 尝试连接到开发WebSocket
            this.ws = new WebSocket('ws://localhost:8000/dev-ws');
            
            this.ws.onopen = () => {
                console.log('🔗 开发WebSocket已连接');
                this.reconnectAttempts = 0;
                this.isReconnecting = false;
                this.showDevNotification('🔧 开发模式已连接', 'success');
                
                // 发送ping保持连接
                this.startPing();
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                } catch (e) {
                    console.error('开发WebSocket消息解析错误:', e);
                }
            };
            
            this.ws.onclose = (event) => {
                console.log('🔌 开发WebSocket连接关闭');
                this.stopPing();
                
                if (!this.isReconnecting && this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnect();
                }
            };
            
            this.ws.onerror = (error) => {
                console.error('🚨 开发WebSocket错误:', error);
            };
            
        } catch (error) {
            console.error('开发WebSocket连接失败:', error);
            this.reconnect();
        }
    }
    
    handleMessage(data) {
        switch (data.type) {
            case 'dev_connected':
                console.log('✅', data.message);
                break;
                
            case 'reload_frontend':
                console.log('🔄 前端文件已更新，正在刷新页面...');
                this.showDevNotification('🔄 前端文件已更新，正在刷新...', 'info');
                
                // 延迟500ms刷新页面，让用户看到通知
                setTimeout(() => {
                    window.location.reload();
                }, 500);
                break;
                
            case 'backend_restarting':
                console.log('🔄 后端正在重启...');
                this.showDevNotification('🔄 后端正在重启...', 'warning');
                break;
                
            case 'backend_restarted':
                console.log('✅ 后端重启完成');
                this.showDevNotification('✅ 后端重启完成', 'success');
                
                // 重新连接主应用WebSocket
                setTimeout(() => {
                    if (window.app && window.app.connectRealTimeData) {
                        window.app.connectRealTimeData();
                    }
                }, 1000);
                break;
                
            case 'pong':
                // 心跳响应，无需处理
                break;
                
            default:
                console.log('开发消息:', data);
        }
    }
    
    reconnect() {
        if (this.isReconnecting) return;
        
        this.isReconnecting = true;
        this.reconnectAttempts++;
        
        console.log(`🔄 尝试重连开发WebSocket (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
        
        setTimeout(() => {
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.connect();
            } else {
                console.log('❌ 开发WebSocket重连失败，已达到最大重试次数');
                this.showDevNotification('❌ 开发模式连接失败', 'error');
            }
        }, this.reconnectInterval);
    }
    
    startPing() {
        this.pingInterval = setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000); // 每30秒ping一次
    }
    
    stopPing() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }
    
    showDevNotification(message, type = 'info') {
        // 创建开发通知元素
        const notification = document.createElement('div');
        notification.className = `dev-notification dev-notification-${type}`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${this.getNotificationColor(type)};
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 10000;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px;
            font-weight: 500;
            max-width: 300px;
            opacity: 0;
            transform: translateX(100%);
            transition: all 0.3s ease;
        `;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // 显示动画
        setTimeout(() => {
            notification.style.opacity = '1';
            notification.style.transform = 'translateX(0)';
        }, 100);
        
        // 自动移除
        setTimeout(() => {
            notification.style.opacity = '0';
            notification.style.transform = 'translateX(100%)';
            
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
    
    getNotificationColor(type) {
        const colors = {
            success: '#10b981',
            info: '#3b82f6',
            warning: '#f59e0b',
            error: '#ef4444'
        };
        return colors[type] || colors.info;
    }
}

// 页面加载完成后自动启动开发客户端
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.devClient = new DevClient();
    });
} else {
    window.devClient = new DevClient();
}

// 添加开发模式样式
const devStyles = document.createElement('style');
devStyles.textContent = `
    .dev-notification {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        user-select: none;
        cursor: default;
    }
    
    .dev-notification:hover {
        transform: scale(1.02) !important;
    }
    
    /* 开发模式指示器 */
    body::before {
        content: "🔧 开发模式";
        position: fixed;
        bottom: 10px;
        left: 10px;
        background: rgba(59, 130, 246, 0.9);
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-family: monospace;
        z-index: 9999;
        opacity: 0.7;
        pointer-events: none;
    }
`;
document.head.appendChild(devStyles);