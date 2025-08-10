/**
 * AI量化交易系统 - 增强版主应用JavaScript
 * v2.0 现代化前端架构 - TypeScript风格的JavaScript
 */

class TradingSystemApp {
    constructor() {
        // 核心配置
        this.config = {
            apiUrl: 'http://localhost:8001/api/v1',
            wsUrl: 'ws://localhost:8001/ws',
            updateInterval: 1000,
            retryAttempts: 3,
            retryDelay: 2000,
            maxHistoryLength: 1000,
            animationDuration: 300
        };

        // 应用状态
        this.state = {
            currentPage: 'dashboard',
            isConnected: false,
            isInitialized: false,
            theme: 'dark',
            user: null,
            systemStatus: 'initializing'
        };

        // 数据存储
        this.data = {
            marketData: new Map(),
            strategies: new Map(),
            trades: new Map(),
            aiAnalysis: new Map(),
            systemMetrics: new Map(),
            chartData: new Map()
        };

        // 核心实例
        this.websockets = new Map();
        this.charts = new Map();
        this.intervals = new Map();
        this.subscriptions = new Set();
        
        // DOM元素缓存
        this.elements = {};
        
        this.init();
    }

    /**
     * 初始化应用
     */
    async init() {
        console.log('🚀 AI量化交易系统 v2.0 初始化中...');
        
        try {
            // 显示初始化状态
            this.showLoading('系统初始化中...');
            
            // 缓存DOM元素
            this.cacheElements();
            
            // 初始化图标系统
            await this.initLucideIcons();
            
            // 设置事件监听器
            this.setupEventListeners();
            
            // 初始化图表系统
            await this.initChartSystem();
            
            // 连接实时数据
            await this.connectRealTimeData();
            
            // 加载初始数据
            await this.loadInitialData();
            
            // 启动更新循环
            this.startUpdateCycles();
            
            // 标记为已初始化
            this.state.isInitialized = true;
            this.state.systemStatus = 'running';
            
            // 隐藏加载遮罩
            this.hideLoading();
            
            // 显示成功消息
            this.showToast('🎉 系统初始化完成', 'success');
            console.log('✅ AI量化交易系统初始化完成');
            
        } catch (error) {
            console.error('❌ 系统初始化失败:', error);
            this.handleInitializationError(error);
        }
    }

    /**
     * 缓存重要DOM元素
     */
    cacheElements() {
        this.elements = {
            // 导航相关
            navButtons: document.querySelectorAll('.nav-btn'),
            pages: document.querySelectorAll('.page'),
            
            // 状态指示器
            connectionStatus: document.getElementById('connection-status'),
            currentTime: document.getElementById('current-time'),
            
            // 市场数据
            marketTable: document.getElementById('market-data-body'),
            
            // AI分析
            sentimentScore: document.getElementById('sentiment-score'),
            predictionTrend: document.getElementById('prediction-trend'),
            predictionConfidence: document.getElementById('prediction-confidence'),
            recommendation: document.getElementById('recommendation'),
            
            // 投资组合
            portfolioValue: document.getElementById('portfolio-value'),
            portfolioChange: document.getElementById('portfolio-change'),
            
            // 系统监控
            overallRiskScore: document.getElementById('overall-risk-score'),
            dailyPnl: document.getElementById('daily-pnl'),
            
            // 容器
            toastContainer: document.getElementById('toast-container'),
            loadingOverlay: document.getElementById('loading-overlay'),
            
            // 图表容器
            sentimentChart: document.getElementById('sentiment-chart'),
            portfolioChart: document.getElementById('portfolio-chart'),
            pnlChart: document.getElementById('pnl-curve-chart')
        };
    }

    /**
     * 初始化Lucide图标系统
     */
    async initLucideIcons() {
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
            console.log('✓ Lucide图标系统已初始化');
        } else {
            console.warn('⚠️ Lucide图标库未加载');
        }
    }

    /**
     * 设置所有事件监听器
     */
    setupEventListeners() {
        // 导航事件
        this.setupNavigationEvents();
        
        // 窗口事件
        this.setupWindowEvents();
        
        // 表单事件
        this.setupFormEvents();
        
        // 快捷键事件
        this.setupKeyboardEvents();
        
        console.log('✓ 事件监听器已设置');
    }

    /**
     * 设置导航事件
     */
    setupNavigationEvents() {
        this.elements.navButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                const page = button.dataset.page;
                this.navigateToPage(page);
            });
        });
    }

    /**
     * 设置窗口事件
     */
    setupWindowEvents() {
        // 窗口大小调整
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                this.handleResize();
            }, 150);
        });
        
        // 页面可见性变化
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.pauseDataUpdates();
            } else {
                this.resumeDataUpdates();
            }
        });
        
        // 在线/离线状态
        window.addEventListener('online', () => this.handleConnectionChange(true));
        window.addEventListener('offline', () => this.handleConnectionChange(false));
    }

    /**
     * 设置表单事件
     */
    setupFormEvents() {
        // Tab切换器
        document.querySelectorAll('.tab-switcher').forEach(switcher => {
            switcher.addEventListener('click', (e) => {
                if (e.target.classList.contains('tab-btn')) {
                    this.switchTab(e.target);
                }
            });
        });
        
        // 设置面板切换
        document.querySelectorAll('.settings-menu .menu-item').forEach(item => {
            item.addEventListener('click', (e) => {
                this.switchSettingsPanel(e.target);
            });
        });
    }

    /**
     * 设置键盘快捷键
     */
    setupKeyboardEvents() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + 数字键切换页面
            if ((e.ctrlKey || e.metaKey) && e.key >= '1' && e.key <= '7') {
                e.preventDefault();
                const pages = ['dashboard', 'factor-lab', 'strategy', 'ai-assistant', 'trades', 'risk', 'settings'];
                const pageIndex = parseInt(e.key) - 1;
                if (pages[pageIndex]) {
                    this.navigateToPage(pages[pageIndex]);
                }
            }
            
            // ESC键关闭模态框
            if (e.key === 'Escape') {
                this.closeAllModals();
            }
        });
    }

    /**
     * 初始化图表系统
     */
    async initChartSystem() {
        try {
            // 默认图表配置
            const defaultConfig = {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#e2e8f0',
                            font: { family: 'Inter' }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: '#334155' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        grid: { color: '#334155' },
                        ticks: { color: '#94a3b8' }
                    }
                }
            };

            // 初始化各个图表
            await this.initSentimentChart(defaultConfig);
            await this.initPortfolioChart(defaultConfig);
            await this.initPnLChart(defaultConfig);
            
            console.log('✓ 图表系统已初始化');
        } catch (error) {
            console.error('图表初始化失败:', error);
            this.showToast('图表初始化失败', 'error');
        }
    }

    /**
     * 初始化情绪分析图表
     */
    async initSentimentChart(defaultConfig) {
        const canvas = this.elements.sentimentChart;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        this.charts.set('sentiment', new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: '市场情绪',
                    data: [],
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                ...defaultConfig,
                scales: {
                    ...defaultConfig.scales,
                    y: {
                        ...defaultConfig.scales.y,
                        min: -1,
                        max: 1
                    }
                }
            }
        }));
    }

    /**
     * 初始化投资组合图表
     */
    async initPortfolioChart(defaultConfig) {
        const canvas = this.elements.portfolioChart;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        this.charts.set('portfolio', new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['BTC', 'ETH', 'USDT'],
                datasets: [{
                    data: [45.2, 28.7, 26.1],
                    backgroundColor: [
                        '#f59e0b',
                        '#3b82f6',
                        '#10b981'
                    ],
                    borderColor: '#1e2130',
                    borderWidth: 2
                }]
            },
            options: {
                ...defaultConfig,
                plugins: {
                    ...defaultConfig.plugins,
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#e2e8f0',
                            font: { family: 'Inter', size: 12 },
                            padding: 15
                        }
                    }
                }
            }
        }));
    }

    /**
     * 初始化PnL曲线图表
     */
    async initPnLChart(defaultConfig) {
        const canvas = this.elements.pnlChart;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        this.charts.set('pnl', new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'PnL',
                    data: [],
                    borderColor: '#22c55e',
                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                ...defaultConfig,
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        }));
    }

    /**
     * 连接实时数据源
     */
    async connectRealTimeData() {
        try {
            // 连接市场数据WebSocket
            await this.connectWebSocket('market-data', `${this.config.wsUrl}/market-data`);
            
            // 连接AI分析WebSocket  
            await this.connectWebSocket('ai-analysis', `${this.config.wsUrl}/ai-analysis`);
            
            console.log('✓ 实时数据连接已建立');
        } catch (error) {
            console.error('实时数据连接失败:', error);
            this.showToast('实时数据连接失败，将使用模拟数据', 'warning');
            
            // 启动模拟数据
            this.startSimulatedData();
        }
    }

    /**
     * 连接WebSocket
     */
    async connectWebSocket(name, url) {
        return new Promise((resolve, reject) => {
            const ws = new WebSocket(url);
            
            ws.onopen = () => {
                this.websockets.set(name, ws);
                this.state.isConnected = true;
                this.updateConnectionStatus(true);
                resolve(ws);
            };
            
            ws.onmessage = (event) => {
                this.handleWebSocketMessage(name, event);
            };
            
            ws.onerror = (error) => {
                console.error(`WebSocket错误 (${name}):`, error);
                reject(error);
            };
            
            ws.onclose = () => {
                this.websockets.delete(name);
                this.state.isConnected = false;
                this.updateConnectionStatus(false);
                
                // 自动重连
                setTimeout(() => {
                    this.connectWebSocket(name, url);
                }, this.config.retryDelay);
            };
        });
    }

    /**
     * 处理WebSocket消息
     */
    handleWebSocketMessage(source, event) {
        try {
            const data = JSON.parse(event.data);
            
            switch (source) {
                case 'market-data':
                    this.handleMarketDataUpdate(data);
                    break;
                case 'ai-analysis':
                    this.handleAIAnalysisUpdate(data);
                    break;
                default:
                    console.warn('未知WebSocket消息源:', source);
            }
        } catch (error) {
            console.error('WebSocket消息解析失败:', error);
        }
    }

    /**
     * 处理市场数据更新
     */
    handleMarketDataUpdate(data) {
        if (data.type === 'market_update') {
            // 更新市场数据
            this.updateMarketData(data.data);
            
            // 更新UI
            this.renderMarketTable();
            this.updatePriceDisplay();
        }
    }

    /**
     * 处理AI分析更新
     */
    handleAIAnalysisUpdate(data) {
        if (data.type === 'ai_analysis') {
            // 更新AI分析数据
            this.data.aiAnalysis.set('current', data.data);
            
            // 更新UI
            this.updateAIAnalysisDisplay(data.data);
        }
    }

    /**
     * 加载初始数据
     */
    async loadInitialData() {
        try {
            // 并行加载各种初始数据
            const [marketData, strategies, aiAnalysis] = await Promise.all([
                this.fetchMarketData(),
                this.fetchStrategies(),
                this.fetchAIAnalysis()
            ]);

            // 存储数据
            this.data.marketData.set('current', marketData);
            this.data.strategies.set('list', strategies);
            this.data.aiAnalysis.set('current', aiAnalysis);

            // 更新UI
            this.renderInitialUI();
            
            console.log('✓ 初始数据加载完成');
        } catch (error) {
            console.error('初始数据加载失败:', error);
            this.showToast('部分数据加载失败，正在重试...', 'warning');
            
            // 启动模拟数据作为后备
            this.startSimulatedData();
        }
    }

    /**
     * 启动更新循环
     */
    startUpdateCycles() {
        // 时钟更新
        this.intervals.set('clock', setInterval(() => {
            this.updateClock();
        }, 1000));

        // 系统状态更新
        this.intervals.set('status', setInterval(() => {
            this.updateSystemStatus();
        }, 5000));

        // 数据刷新
        this.intervals.set('data', setInterval(() => {
            this.refreshData();
        }, this.config.updateInterval));
        
        console.log('✓ 更新循环已启动');
    }

    /**
     * 页面导航
     */
    navigateToPage(pageName) {
        // 移除所有活动状态
        this.elements.navButtons.forEach(btn => btn.classList.remove('active'));
        this.elements.pages.forEach(page => page.classList.remove('active'));
        
        // 设置新的活动状态
        const targetButton = document.querySelector(`[data-page="${pageName}"]`);
        const targetPage = document.getElementById(`${pageName}-page`);
        
        if (targetButton && targetPage) {
            targetButton.classList.add('active');
            targetPage.classList.add('active');
            this.state.currentPage = pageName;
            
            // 页面特殊处理
            this.handlePageSwitch(pageName);
        }
    }

    /**
     * 处理页面切换
     */
    handlePageSwitch(pageName) {
        switch (pageName) {
            case 'dashboard':
                this.refreshDashboard();
                break;
            case 'factor-lab':
                this.initFactorLab();
                break;
            case 'strategy':
                this.refreshStrategies();
                break;
            case 'trades':
                this.refreshTrades();
                break;
            case 'risk':
                this.refreshRiskAnalysis();
                break;
        }
    }

    /**
     * 显示Toast通知
     */
    showToast(message, type = 'info', duration = 3000) {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <span class="toast-message">${message}</span>
                <button class="toast-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        
        this.elements.toastContainer.appendChild(toast);
        
        // 自动动画显示
        setTimeout(() => toast.classList.add('show'), 10);
        
        // 自动关闭
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    /**
     * 显示加载状态
     */
    showLoading(message = '加载中...') {
        const overlay = this.elements.loadingOverlay;
        if (overlay) {
            overlay.querySelector('p').textContent = message;
            overlay.classList.add('active');
        }
    }

    /**
     * 隐藏加载状态
     */
    hideLoading() {
        const overlay = this.elements.loadingOverlay;
        if (overlay) {
            overlay.classList.remove('active');
        }
    }

    /**
     * 更新时钟显示
     */
    updateClock() {
        if (this.elements.currentTime) {
            const now = new Date();
            this.elements.currentTime.textContent = now.toLocaleTimeString('zh-CN');
        }
    }

    /**
     * 更新连接状态
     */
    updateConnectionStatus(isConnected) {
        const indicator = this.elements.connectionStatus;
        if (indicator) {
            indicator.className = `status-indicator ${isConnected ? 'online' : 'offline'}`;
        }
    }

    /**
     * API请求封装
     */
    async apiRequest(endpoint, options = {}) {
        const url = `${this.config.apiUrl}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        };
        
        const finalOptions = { ...defaultOptions, ...options };
        
        try {
            const response = await fetch(url, finalOptions);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`API请求失败 (${endpoint}):`, error);
            throw error;
        }
    }

    /**
     * 获取市场数据
     */
    async fetchMarketData() {
        return this.apiRequest('/market/latest');
    }

    /**
     * 获取策略列表
     */
    async fetchStrategies() {
        return this.apiRequest('/strategies');
    }

    /**
     * 获取AI分析
     */
    async fetchAIAnalysis() {
        return this.apiRequest('/ai/sentiment');
    }

    /**
     * 启动模拟数据（开发/测试用）
     */
    startSimulatedData() {
        console.log('🧪 启动模拟数据生成');
        
        // 模拟市场数据
        this.intervals.set('simulated-market', setInterval(() => {
            const mockData = this.generateMockMarketData();
            this.handleMarketDataUpdate({
                type: 'market_update',
                data: mockData
            });
        }, 2000));
        
        // 模拟AI分析
        this.intervals.set('simulated-ai', setInterval(() => {
            const mockAnalysis = this.generateMockAIAnalysis();
            this.handleAIAnalysisUpdate({
                type: 'ai_analysis',
                data: mockAnalysis
            });
        }, 5000));
    }

    /**
     * 生成模拟市场数据
     */
    generateMockMarketData() {
        const symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT'];
        const prices = {};
        
        symbols.forEach(symbol => {
            const basePrice = symbol.startsWith('BTC') ? 45000 : 
                            symbol.startsWith('ETH') ? 2800 : 
                            symbol.startsWith('BNB') ? 300 : 100;
            
            prices[symbol] = {
                price: basePrice + (Math.random() - 0.5) * basePrice * 0.02,
                change24h: (Math.random() - 0.5) * 0.1,
                volume24h: Math.random() * 1000000
            };
        });
        
        return { prices };
    }

    /**
     * 生成模拟AI分析
     */
    generateMockAIAnalysis() {
        return {
            sentiment: {
                score: (Math.random() - 0.5) * 2,
                trend: Math.random() > 0.5 ? 'bullish' : 'bearish',
                confidence: 0.6 + Math.random() * 0.4
            },
            prediction: {
                direction: Math.random() > 0.5 ? 'up' : 'down',
                confidence: 0.5 + Math.random() * 0.5
            }
        };
    }

    /**
     * 处理初始化错误
     */
    handleInitializationError(error) {
        this.state.systemStatus = 'error';
        this.showToast('系统初始化失败，请检查网络连接', 'error', 5000);
        
        // 尝试降级模式
        setTimeout(() => {
            this.startSimulatedData();
            this.showToast('已切换到模拟模式', 'warning');
        }, 2000);
    }

    /**
     * 销毁应用（清理资源）
     */
    destroy() {
        // 关闭WebSocket连接
        this.websockets.forEach(ws => ws.close());
        this.websockets.clear();
        
        // 清除定时器
        this.intervals.forEach(interval => clearInterval(interval));
        this.intervals.clear();
        
        // 销毁图表
        this.charts.forEach(chart => chart.destroy());
        this.charts.clear();
        
        console.log('🧹 应用资源已清理');
    }
}

// 全局错误处理
window.addEventListener('error', (event) => {
    console.error('全局错误:', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('未处理的Promise拒绝:', event.reason);
    event.preventDefault();
});

// 创建全局应用实例
window.tradingApp = new TradingSystemApp();

// 开发者工具
if (process?.env?.NODE_ENV === 'development') {
    window.debugApp = () => {
        console.log('应用状态:', window.tradingApp.state);
        console.log('数据存储:', window.tradingApp.data);
        console.log('图表实例:', window.tradingApp.charts);
    };
}