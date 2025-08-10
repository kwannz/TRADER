/**
 * AI量化交易系统 - 工具函数库
 * 提供各种实用工具和辅助函数
 */

/**
 * 数值格式化工具
 */
class NumberFormatter {
    /**
     * 格式化价格显示
     */
    static formatPrice(price, decimals = 2) {
        if (typeof price !== 'number' || isNaN(price)) return '--';
        
        if (price >= 1000000) {
            return (price / 1000000).toFixed(decimals) + 'M';
        }
        if (price >= 1000) {
            return (price / 1000).toFixed(decimals) + 'K';
        }
        return price.toFixed(decimals);
    }

    /**
     * 格式化百分比
     */
    static formatPercentage(value, decimals = 2) {
        if (typeof value !== 'number' || isNaN(value)) return '--';
        const sign = value >= 0 ? '+' : '';
        return `${sign}${(value * 100).toFixed(decimals)}%`;
    }

    /**
     * 格式化成交量
     */
    static formatVolume(volume) {
        if (typeof volume !== 'number' || isNaN(volume)) return '--';
        
        if (volume >= 1e9) {
            return (volume / 1e9).toFixed(2) + 'B';
        }
        if (volume >= 1e6) {
            return (volume / 1e6).toFixed(2) + 'M';
        }
        if (volume >= 1e3) {
            return (volume / 1e3).toFixed(2) + 'K';
        }
        return volume.toFixed(2);
    }

    /**
     * 格式化货币
     */
    static formatCurrency(amount, currency = 'USDT', decimals = 2) {
        if (typeof amount !== 'number' || isNaN(amount)) return '--';
        return `${this.formatPrice(amount, decimals)} ${currency}`;
    }
}

/**
 * 时间格式化工具
 */
class TimeFormatter {
    /**
     * 格式化时间戳为可读时间
     */
    static formatTime(timestamp, includeSeconds = true) {
        const date = new Date(timestamp);
        const options = {
            hour: '2-digit',
            minute: '2-digit',
            ...(includeSeconds && { second: '2-digit' }),
            hour12: false
        };
        return date.toLocaleTimeString('zh-CN', options);
    }

    /**
     * 格式化日期
     */
    static formatDate(timestamp, format = 'YYYY-MM-DD') {
        const date = new Date(timestamp);
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        
        return format
            .replace('YYYY', year)
            .replace('MM', month)
            .replace('DD', day);
    }

    /**
     * 相对时间显示
     */
    static formatRelativeTime(timestamp) {
        const now = Date.now();
        const diff = now - timestamp;
        const seconds = Math.floor(diff / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);

        if (days > 0) return `${days}天前`;
        if (hours > 0) return `${hours}小时前`;
        if (minutes > 0) return `${minutes}分钟前`;
        return `${seconds}秒前`;
    }

    /**
     * 格式化持续时间
     */
    static formatDuration(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;

        if (hours > 0) {
            return `${hours}h ${minutes}m`;
        }
        if (minutes > 0) {
            return `${minutes}m ${secs}s`;
        }
        return `${secs}s`;
    }
}

/**
 * DOM操作工具
 */
class DOMUtils {
    /**
     * 创建元素
     */
    static createElement(tag, className = '', attributes = {}) {
        const element = document.createElement(tag);
        if (className) element.className = className;
        
        Object.entries(attributes).forEach(([key, value]) => {
            element.setAttribute(key, value);
        });
        
        return element;
    }

    /**
     * 添加CSS类（带动画）
     */
    static addClass(element, className, delay = 0) {
        setTimeout(() => {
            element.classList.add(className);
        }, delay);
    }

    /**
     * 移除CSS类（带动画）
     */
    static removeClass(element, className, delay = 0) {
        setTimeout(() => {
            element.classList.remove(className);
        }, delay);
    }

    /**
     * 切换CSS类
     */
    static toggleClass(element, className) {
        element.classList.toggle(className);
    }

    /**
     * 平滑滚动到元素
     */
    static scrollToElement(element, behavior = 'smooth') {
        element.scrollIntoView({ behavior, block: 'center' });
    }

    /**
     * 复制到剪贴板
     */
    static async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch (err) {
            console.error('复制失败:', err);
            return false;
        }
    }
}

/**
 * 颜色工具
 */
class ColorUtils {
    /**
     * 根据数值获取颜色
     */
    static getColorByValue(value, type = 'pnl') {
        switch (type) {
            case 'pnl':
                return value >= 0 ? '#22c55e' : '#ef4444';
            case 'sentiment':
                if (value > 0.3) return '#22c55e';
                if (value < -0.3) return '#ef4444';
                return '#f59e0b';
            case 'risk':
                if (value < 0.3) return '#22c55e';
                if (value < 0.7) return '#f59e0b';
                return '#ef4444';
            default:
                return '#64748b';
        }
    }

    /**
     * 十六进制转RGBA
     */
    static hexToRgba(hex, alpha = 1) {
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r}, ${g}, ${b}, ${alpha})`;
    }

    /**
     * 生成渐变色
     */
    static generateGradient(color1, color2, direction = '135deg') {
        return `linear-gradient(${direction}, ${color1}, ${color2})`;
    }
}

/**
 * 数据验证工具
 */
class ValidationUtils {
    /**
     * 验证API Key格式
     */
    static validateApiKey(key) {
        if (!key || typeof key !== 'string') return false;
        return key.length >= 32 && /^[a-zA-Z0-9_-]+$/.test(key);
    }

    /**
     * 验证交易对格式
     */
    static validateTradingPair(pair) {
        if (!pair || typeof pair !== 'string') return false;
        return /^[A-Z]{3,10}-[A-Z]{3,10}$/.test(pair);
    }

    /**
     * 验证价格
     */
    static validatePrice(price) {
        if (typeof price !== 'number' || isNaN(price)) return false;
        return price > 0 && price < Number.MAX_SAFE_INTEGER;
    }

    /**
     * 验证邮箱
     */
    static validateEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    /**
     * 验证手机号
     */
    static validatePhone(phone) {
        const phoneRegex = /^1[3-9]\d{9}$/;
        return phoneRegex.test(phone);
    }
}

/**
 * 本地存储工具
 */
class StorageUtils {
    /**
     * 获取本地存储
     */
    static get(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (err) {
            console.error('读取本地存储失败:', err);
            return defaultValue;
        }
    }

    /**
     * 设置本地存储
     */
    static set(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
            return true;
        } catch (err) {
            console.error('写入本地存储失败:', err);
            return false;
        }
    }

    /**
     * 移除本地存储
     */
    static remove(key) {
        try {
            localStorage.removeItem(key);
            return true;
        } catch (err) {
            console.error('删除本地存储失败:', err);
            return false;
        }
    }

    /**
     * 清空本地存储
     */
    static clear() {
        try {
            localStorage.clear();
            return true;
        } catch (err) {
            console.error('清空本地存储失败:', err);
            return false;
        }
    }

    /**
     * 获取存储大小
     */
    static getSize() {
        let total = 0;
        for (let key in localStorage) {
            if (localStorage.hasOwnProperty(key)) {
                total += localStorage[key].length + key.length;
            }
        }
        return total;
    }
}

/**
 * 防抖和节流工具
 */
class ThrottleUtils {
    /**
     * 防抖函数
     */
    static debounce(func, wait, immediate = false) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                timeout = null;
                if (!immediate) func(...args);
            };
            const callNow = immediate && !timeout;
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
            if (callNow) func(...args);
        };
    }

    /**
     * 节流函数
     */
    static throttle(func, limit) {
        let inThrottle;
        return function executedFunction(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }
}

/**
 * 网络请求工具
 */
class HTTPUtils {
    /**
     * 请求拦截器
     */
    static async request(url, options = {}) {
        const defaultOptions = {
            timeout: 10000,
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        };

        const finalOptions = { ...defaultOptions, ...options };
        
        // 添加超时控制
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), finalOptions.timeout);
        
        try {
            const response = await fetch(url, {
                ...finalOptions,
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            clearTimeout(timeoutId);
            if (error.name === 'AbortError') {
                throw new Error('请求超时');
            }
            throw error;
        }
    }

    /**
     * GET请求
     */
    static get(url, params = {}) {
        const urlWithParams = new URL(url);
        Object.entries(params).forEach(([key, value]) => {
            urlWithParams.searchParams.append(key, value);
        });
        
        return this.request(urlWithParams.toString());
    }

    /**
     * POST请求
     */
    static post(url, data = {}) {
        return this.request(url, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    /**
     * PUT请求
     */
    static put(url, data = {}) {
        return this.request(url, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }

    /**
     * DELETE请求
     */
    static delete(url) {
        return this.request(url, { method: 'DELETE' });
    }
}

/**
 * 事件总线
 */
class EventBus {
    constructor() {
        this.events = {};
    }

    /**
     * 订阅事件
     */
    on(event, callback) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(callback);
        
        // 返回取消订阅函数
        return () => this.off(event, callback);
    }

    /**
     * 取消订阅
     */
    off(event, callback) {
        if (this.events[event]) {
            this.events[event] = this.events[event].filter(cb => cb !== callback);
        }
    }

    /**
     * 触发事件
     */
    emit(event, data) {
        if (this.events[event]) {
            this.events[event].forEach(callback => callback(data));
        }
    }

    /**
     * 一次性事件监听
     */
    once(event, callback) {
        const onceCallback = (data) => {
            callback(data);
            this.off(event, onceCallback);
        };
        this.on(event, onceCallback);
    }
}

/**
 * 性能监控工具
 */
class PerformanceUtils {
    /**
     * 测量函数执行时间
     */
    static measure(func, name = 'anonymous') {
        return function(...args) {
            const start = performance.now();
            const result = func.apply(this, args);
            const end = performance.now();
            console.log(`⏱️ ${name} 执行时间: ${(end - start).toFixed(2)}ms`);
            return result;
        };
    }

    /**
     * 测量异步函数执行时间
     */
    static async measureAsync(func, name = 'anonymous') {
        return async function(...args) {
            const start = performance.now();
            const result = await func.apply(this, args);
            const end = performance.now();
            console.log(`⏱️ ${name} 执行时间: ${(end - start).toFixed(2)}ms`);
            return result;
        };
    }

    /**
     * 获取内存使用情况
     */
    static getMemoryUsage() {
        if (performance.memory) {
            return {
                used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024),
                total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024),
                limit: Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024)
            };
        }
        return null;
    }

    /**
     * 监控FPS
     */
    static monitorFPS(callback) {
        let frames = 0;
        let lastTime = performance.now();

        function countFrames() {
            frames++;
            const currentTime = performance.now();
            
            if (currentTime >= lastTime + 1000) {
                const fps = Math.round(frames * 1000 / (currentTime - lastTime));
                callback(fps);
                frames = 0;
                lastTime = currentTime;
            }
            
            requestAnimationFrame(countFrames);
        }
        
        countFrames();
    }
}

/**
 * 加密工具
 */
class CryptoUtils {
    /**
     * 生成随机字符串
     */
    static generateRandomString(length = 32) {
        const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
        let result = '';
        for (let i = 0; i < length; i++) {
            result += chars.charAt(Math.floor(Math.random() * chars.length));
        }
        return result;
    }

    /**
     * 简单哈希函数（用于非加密场景）
     */
    static simpleHash(str) {
        let hash = 0;
        if (str.length === 0) return hash;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // 转换为32位整数
        }
        return hash;
    }

    /**
     * Base64编码
     */
    static base64Encode(str) {
        return btoa(unescape(encodeURIComponent(str)));
    }

    /**
     * Base64解码
     */
    static base64Decode(str) {
        return decodeURIComponent(escape(atob(str)));
    }
}

// 创建全局事件总线实例
window.eventBus = new EventBus();

// 导出所有工具类
window.utils = {
    NumberFormatter,
    TimeFormatter,
    DOMUtils,
    ColorUtils,
    ValidationUtils,
    StorageUtils,
    ThrottleUtils,
    HTTPUtils,
    EventBus,
    PerformanceUtils,
    CryptoUtils
};

// 开发模式下添加调试工具
if (typeof window !== 'undefined' && window.location.hostname === 'localhost') {
    window.debugUtils = () => {
        console.log('🛠️ 工具函数库已加载');
        console.log('可用工具:', Object.keys(window.utils));
        console.log('全局事件总线:', window.eventBus);
    };
}