/**
 * AI量化数据分析与因子研究平台 - 主应用
 * 专注于数据分析、因子生成和研究功能
 */

class DataAnalysisApp {
    constructor() {
        // 核心配置
        this.config = {
            apiUrl: 'http://localhost:8002/api/v1',
            wsUrl: 'ws://localhost:8002/ws',
            updateInterval: 2000,
            maxDataPoints: 10000,
            aiCostLimit: 100,
            dataRetentionDays: 365
        };

        // 应用状态
        this.state = {
            currentPage: 'data-overview',
            isDataConnected: false,
            aiEnginesStatus: new Map(),
            activeFactors: new Map(),
            backtestResults: new Map(),
            currentProject: null,
            generatingFactors: false
        };

        // 数据存储
        this.data = {
            marketData: new Map(),
            factors: new Map(),
            backtestResults: new Map(),
            aiGeneratedFactors: new Map(),
            dataQuality: new Map(),
            reports: new Map()
        };

        // 核心实例
        this.websockets = new Map();
        this.charts = new Map();
        this.intervals = new Map();
        
        // DOM元素缓存
        this.elements = {};
        
        this.init();
    }

    /**
     * 初始化数据分析应用
     */
    async init() {
        console.log('📊 AI量化数据分析平台初始化中...');
        
        try {
            // 显示加载状态
            this.showLoading('数据分析系统初始化中...');
            
            // 缓存DOM元素
            this.cacheElements();
            
            // 初始化图标系统
            await this.initLucideIcons();
            
            // 设置事件监听器
            this.setupEventListeners();
            
            // 初始化图表系统
            await this.initChartSystem();
            
            // 连接数据源
            await this.connectDataSources();
            
            // 加载初始数据
            await this.loadInitialData();
            
            // 启动数据更新循环
            this.startDataUpdates();
            
            // 标记为已初始化
            this.state.isInitialized = true;
            
            // 隐藏加载遮罩
            this.hideLoading();
            
            // 显示成功消息
            this.showToast('🎉 数据分析系统初始化完成', 'success');
            console.log('✅ 数据分析平台初始化完成');
            
        } catch (error) {
            console.error('❌ 系统初始化失败:', error);
            this.handleInitializationError(error);
        }
    }

    /**
     * 缓存DOM元素
     */
    cacheElements() {
        this.elements = {
            // 导航相关
            navButtons: document.querySelectorAll('.nav-btn'),
            pages: document.querySelectorAll('.page'),
            
            // 状态指示器
            dataConnectionStatus: document.getElementById('data-connection-status'),
            currentTime: document.getElementById('current-time'),
            
            // 数据概览
            totalSymbols: document.getElementById('total-symbols'),
            totalDatapoints: document.getElementById('total-datapoints'),
            activeFactors: document.getElementById('active-factors'),
            updateFrequency: document.getElementById('update-frequency'),
            
            // 数据流
            dataStream: document.getElementById('data-stream'),
            
            // 图表容器
            marketHeatmap: document.getElementById('market-heatmap'),
            factorPreviewChart: document.getElementById('factor-preview-chart'),
            backtestPerformanceChart: document.getElementById('backtest-performance-chart'),
            dataSourceStatsChart: document.getElementById('data-source-stats-chart'),
            
            // AI因子生成
            generatedFactors: document.getElementById('generated-factors'),
            
            // 其他
            toastContainer: document.getElementById('toast-container'),
            loadingOverlay: document.getElementById('loading-overlay')
        };
    }

    /**
     * 初始化Lucide图标系统
     */
    async initLucideIcons() {
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
            console.log('✓ Lucide图标系统已初始化');
        }
    }

    /**
     * 设置事件监听器
     */
    setupEventListeners() {
        // 页面导航
        this.elements.navButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                const page = button.dataset.page;
                this.navigateToPage(page);
            });
        });

        // 窗口事件
        window.addEventListener('resize', this.debounce(() => {
            this.handleResize();
        }, 200));

        // 键盘快捷键
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key >= '1' && e.key <= '8') {
                e.preventDefault();
                const pages = [
                    'data-overview', 'factor-research', 'factor-generation',
                    'backtest-lab', 'factor-library', 'data-sources',
                    'analysis-reports', 'system-config'
                ];
                const pageIndex = parseInt(e.key) - 1;
                if (pages[pageIndex]) {
                    this.navigateToPage(pages[pageIndex]);
                }
            }
        });

        console.log('✓ 事件监听器已设置');
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
            await this.initFactorPreviewChart(defaultConfig);
            await this.initBacktestChart(defaultConfig);
            await this.initDataSourceChart(defaultConfig);
            await this.initMarketHeatmap();
            
            console.log('✓ 图表系统已初始化');
        } catch (error) {
            console.error('图表初始化失败:', error);
        }
    }

    /**
     * 初始化因子预览图表
     */
    async initFactorPreviewChart(defaultConfig) {
        const canvas = this.elements.factorPreviewChart;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        this.charts.set('factor-preview', new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: '因子值',
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
                        title: {
                            display: true,
                            text: '因子值',
                            color: '#e2e8f0'
                        }
                    }
                }
            }
        }));
    }

    /**
     * 初始化回测图表
     */
    async initBacktestChart(defaultConfig) {
        const canvas = this.elements.backtestPerformanceChart;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        this.charts.set('backtest', new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: '策略收益',
                    data: [],
                    borderColor: '#22c55e',
                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                    borderWidth: 2,
                    fill: false
                }, {
                    label: '基准收益',
                    data: [],
                    borderColor: '#64748b',
                    backgroundColor: 'rgba(100, 116, 139, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    borderDash: [5, 5]
                }]
            },
            options: {
                ...defaultConfig,
                scales: {
                    ...defaultConfig.scales,
                    y: {
                        ...defaultConfig.scales.y,
                        title: {
                            display: true,
                            text: '累计收益率 (%)',
                            color: '#e2e8f0'
                        }
                    }
                }
            }
        }));
    }

    /**
     * 初始化数据源统计图表
     */
    async initDataSourceChart(defaultConfig) {
        const canvas = this.elements.dataSourceStatsChart;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        this.charts.set('data-source', new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Binance', 'OKX', 'Coinglass', '其他'],
                datasets: [{
                    data: [45, 30, 15, 10],
                    backgroundColor: [
                        '#f59e0b',
                        '#3b82f6',
                        '#10b981',
                        '#64748b'
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
     * 初始化市场热力图
     */
    async initMarketHeatmap() {
        const container = this.elements.marketHeatmap;
        if (!container || typeof echarts === 'undefined') return;

        const heatmapChart = echarts.init(container, 'dark');
        
        // 模拟热力图数据
        const data = [];
        const symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'XRP', 'SOL', 'DOT', 'AVAX'];
        
        symbols.forEach((symbol, i) => {
            symbols.forEach((symbol2, j) => {
                data.push([i, j, Math.random() * 0.2 - 0.1]);
            });
        });

        const option = {
            tooltip: {
                trigger: 'item',
                formatter: function(params) {
                    return `${symbols[params.data[0]]} vs ${symbols[params.data[1]]}: ${(params.data[2] * 100).toFixed(2)}%`;
                }
            },
            grid: {
                height: '80%',
                top: '10%',
                left: '10%'
            },
            xAxis: {
                type: 'category',
                data: symbols,
                axisLabel: { color: '#94a3b8' }
            },
            yAxis: {
                type: 'category',
                data: symbols,
                axisLabel: { color: '#94a3b8' }
            },
            visualMap: {
                min: -0.1,
                max: 0.1,
                calculable: true,
                orient: 'horizontal',
                left: 'center',
                bottom: '5%',
                inRange: {
                    color: ['#ef4444', '#f59e0b', '#22c55e']
                },
                textStyle: { color: '#94a3b8' }
            },
            series: [{
                name: '24h变化',
                type: 'heatmap',
                data: data,
                label: {
                    show: false
                },
                emphasis: {
                    itemStyle: {
                        shadowBlur: 10,
                        shadowColor: 'rgba(0, 0, 0, 0.5)'
                    }
                }
            }]
        };

        heatmapChart.setOption(option);
        this.charts.set('heatmap', heatmapChart);
    }

    /**
     * 连接数据源
     */
    async connectDataSources() {
        try {
            console.log('🔗 连接数据源...');
            
            // 模拟数据源连接
            await this.simulateDataSourceConnection();
            
            // 更新连接状态
            this.updateDataConnectionStatus(true);
            
            console.log('✓ 数据源连接成功');
        } catch (error) {
            console.error('数据源连接失败:', error);
            this.updateDataConnectionStatus(false);
        }
    }

    /**
     * 模拟数据源连接
     */
    async simulateDataSourceConnection() {
        return new Promise(resolve => {
            setTimeout(() => {
                this.state.isDataConnected = true;
                resolve();
            }, 1000);
        });
    }

    /**
     * 加载初始数据
     */
    async loadInitialData() {
        try {
            console.log('📥 加载初始数据...');
            
            // 加载数据概览统计
            await this.loadDataOverviewStats();
            
            // 加载AI引擎状态
            await this.loadAIEngineStatus();
            
            // 加载因子库数据
            await this.loadFactorLibrary();
            
            console.log('✓ 初始数据加载完成');
        } catch (error) {
            console.error('初始数据加载失败:', error);
        }
    }

    /**
     * 加载数据概览统计
     */
    async loadDataOverviewStats() {
        // 模拟统计数据
        const stats = {
            totalSymbols: 1247,
            totalDatapoints: '2.4M',
            activeFactors: 156,
            updateFrequency: '2.3Hz'
        };

        // 更新UI
        if (this.elements.totalSymbols) {
            this.elements.totalSymbols.textContent = stats.totalSymbols.toLocaleString();
        }
        if (this.elements.totalDatapoints) {
            this.elements.totalDatapoints.textContent = stats.totalDatapoints;
        }
        if (this.elements.activeFactors) {
            this.elements.activeFactors.textContent = stats.activeFactors.toLocaleString();
        }
        if (this.elements.updateFrequency) {
            this.elements.updateFrequency.textContent = stats.updateFrequency;
        }
    }

    /**
     * 加载AI引擎状态
     */
    async loadAIEngineStatus() {
        const engines = [
            {
                name: 'DeepSeek',
                status: 'online',
                calls: 1247,
                responseTime: '1.2s'
            },
            {
                name: 'Gemini',
                status: 'online',
                calls: 856,
                responseTime: '2.1s'
            }
        ];

        engines.forEach(engine => {
            this.state.aiEnginesStatus.set(engine.name, engine);
        });
    }

    /**
     * 加载因子库
     */
    async loadFactorLibrary() {
        // 模拟因子数据
        const factors = [
            {
                id: 'factor_001',
                name: 'RSI动量因子',
                category: '技术指标',
                ic: 0.156,
                icir: 1.23,
                winRate: 0.642,
                source: 'ai',
                created: '2024-01-27'
            },
            {
                id: 'factor_002',
                name: '成交量价格背离因子',
                category: '成交量',
                ic: 0.132,
                icir: 0.98,
                winRate: 0.617,
                source: 'manual',
                created: '2024-01-26'
            }
        ];

        factors.forEach(factor => {
            this.data.factors.set(factor.id, factor);
        });
    }

    /**
     * 启动数据更新循环
     */
    startDataUpdates() {
        // 时钟更新
        this.intervals.set('clock', setInterval(() => {
            this.updateClock();
        }, 1000));

        // 数据流更新
        this.intervals.set('dataStream', setInterval(() => {
            this.updateDataStream();
        }, 500));

        // 数据质量监控
        this.intervals.set('dataQuality', setInterval(() => {
            this.updateDataQuality();
        }, 5000));

        console.log('✓ 数据更新循环已启动');
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
     * 更新实时数据流
     */
    updateDataStream() {
        const dataStream = this.elements.dataStream;
        if (!dataStream) return;

        // 生成模拟数据流项目
        const streamItem = document.createElement('div');
        streamItem.className = 'stream-item';
        
        const symbols = ['BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'SOL-USDT'];
        const symbol = symbols[Math.floor(Math.random() * symbols.length)];
        const price = (Math.random() * 50000 + 20000).toFixed(2);
        const change = (Math.random() - 0.5) * 0.1;
        const changeClass = change >= 0 ? 'positive' : 'negative';
        
        streamItem.innerHTML = `
            <span class="stream-time">${new Date().toLocaleTimeString()}</span>
            <span class="stream-symbol">${symbol}</span>
            <span class="stream-price">$${price}</span>
            <span class="stream-change ${changeClass}">${change >= 0 ? '+' : ''}${(change * 100).toFixed(2)}%</span>
        `;

        dataStream.insertBefore(streamItem, dataStream.firstChild);

        // 限制显示数量
        while (dataStream.children.length > 50) {
            dataStream.removeChild(dataStream.lastChild);
        }

        // 添加动画
        setTimeout(() => {
            streamItem.classList.add('fade-in');
        }, 10);
    }

    /**
     * 更新数据质量指标
     */
    updateDataQuality() {
        // 模拟数据质量变化
        const qualities = {
            completeness: Math.random() * 2 + 97,
            accuracy: Math.random() * 1 + 98.5,
            timeliness: Math.random() * 5 + 95
        };

        // 更新进度条
        document.querySelectorAll('.quality-item').forEach((item, index) => {
            const progressFill = item.querySelector('.progress-fill');
            const progressText = item.querySelector('.progress-text');
            
            if (progressFill && progressText) {
                const keys = Object.keys(qualities);
                const value = qualities[keys[index]];
                
                progressFill.style.width = `${value}%`;
                progressText.textContent = `${value.toFixed(1)}%`;
            }
        });
    }

    /**
     * 更新数据连接状态
     */
    updateDataConnectionStatus(isConnected) {
        const indicator = this.elements.dataConnectionStatus;
        if (indicator) {
            indicator.className = `status-indicator ${isConnected ? 'online' : 'offline'}`;
        }
        this.state.isDataConnected = isConnected;
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
            case 'data-overview':
                this.refreshDataOverview();
                break;
            case 'factor-research':
                this.initFactorResearch();
                break;
            case 'factor-generation':
                this.initAIFactorGeneration();
                break;
            case 'backtest-lab':
                this.initBacktestLab();
                break;
            case 'factor-library':
                this.refreshFactorLibrary();
                break;
            case 'data-sources':
                this.refreshDataSources();
                break;
            case 'analysis-reports':
                this.refreshAnalysisReports();
                break;
            case 'system-config':
                this.initSystemConfig();
                break;
        }
    }

    /**
     * 刷新数据概览
     */
    refreshDataOverview() {
        console.log('🔄 刷新数据概览');
        // 重新加载概览数据
        this.loadDataOverviewStats();
    }

    /**
     * 初始化因子研究
     */
    initFactorResearch() {
        console.log('🔬 初始化因子研究');
        // 初始化因子研究相关功能
    }

    /**
     * 初始化AI因子生成
     */
    initAIFactorGeneration() {
        console.log('🧠 初始化AI因子生成');
        // 初始化AI生成界面
    }

    /**
     * 初始化回测实验室
     */
    initBacktestLab() {
        console.log('🧪 初始化回测实验室');
        // 初始化回测功能
    }

    /**
     * 刷新因子库
     */
    refreshFactorLibrary() {
        console.log('📚 刷新因子库');
        // 重新加载因子库数据
        this.loadFactorLibrary();
    }

    /**
     * 刷新数据源
     */
    refreshDataSources() {
        console.log('🔌 刷新数据源');
        // 重新检查数据源状态
    }

    /**
     * 刷新分析报告
     */
    refreshAnalysisReports() {
        console.log('📊 刷新分析报告');
        // 重新加载报告列表
    }

    /**
     * 初始化系统配置
     */
    initSystemConfig() {
        console.log('⚙️ 初始化系统配置');
        // 初始化配置界面
    }

    /**
     * 开始AI因子生成
     */
    async startAIGeneration() {
        if (this.state.generatingFactors) {
            this.showToast('AI正在生成中，请稍候...', 'warning');
            return;
        }

        try {
            this.state.generatingFactors = true;
            
            // 显示生成进度
            this.showGeneratingProgress();
            
            // 模拟AI生成过程
            await this.simulateAIGeneration();
            
            this.state.generatingFactors = false;
            this.showToast('AI因子生成完成！', 'success');
            
        } catch (error) {
            console.error('AI生成失败:', error);
            this.state.generatingFactors = false;
            this.showToast('AI生成失败，请重试', 'error');
        }
    }

    /**
     * 显示生成进度
     */
    showGeneratingProgress() {
        const container = this.elements.generatedFactors;
        if (!container) return;

        // 添加生成中的占位符
        const generatingCard = document.createElement('div');
        generatingCard.className = 'factor-card generating';
        generatingCard.innerHTML = `
            <div class="generating-indicator">
                <div class="loading-spinner"></div>
                <div class="generating-text">AI正在分析数据并生成因子...</div>
            </div>
        `;
        
        container.appendChild(generatingCard);
    }

    /**
     * 模拟AI生成过程
     */
    async simulateAIGeneration() {
        return new Promise(resolve => {
            setTimeout(() => {
                // 移除生成中的占位符
                const generatingCard = document.querySelector('.factor-card.generating');
                if (generatingCard) {
                    generatingCard.remove();
                }
                
                // 添加生成结果
                this.addGeneratedFactor();
                resolve();
            }, 3000);
        });
    }

    /**
     * 添加生成的因子
     */
    addGeneratedFactor() {
        const container = this.elements.generatedFactors;
        if (!container) return;

        const factorCard = document.createElement('div');
        factorCard.className = 'factor-card fade-enter';
        
        const factorNames = ['趋势强度因子', '波动率均值回归因子', '资金流向因子', '情绪反转因子'];
        const randomName = factorNames[Math.floor(Math.random() * factorNames.length)];
        const randomScore = (Math.random() * 3 + 6).toFixed(1);
        
        factorCard.innerHTML = `
            <div class="factor-header">
                <div class="factor-name">${randomName} #${Date.now().toString().slice(-3)}</div>
                <div class="factor-score">
                    <span class="score-label">AI评分</span>
                    <span class="score-value">${randomScore}</span>
                </div>
            </div>
            <div class="factor-content">
                <div class="factor-formula">
                    <code>
                        // AI生成的因子公式<br>
                        factor = (close - sma(close, 20)) / std(returns, 10)<br>
                        signal = tanh(factor * 2.5)
                    </code>
                </div>
                <div class="factor-description">
                    <p>基于价格动量和波动率特征的多层次因子，结合了技术分析和统计学方法。</p>
                </div>
            </div>
            <div class="factor-actions">
                <button class="btn btn-secondary small">
                    <i data-lucide="bar-chart"></i>
                    预览
                </button>
                <button class="btn btn-secondary small">
                    <i data-lucide="play"></i>
                    回测
                </button>
                <button class="btn btn-primary small">
                    <i data-lucide="save"></i>
                    保存到因子库
                </button>
            </div>
        `;
        
        container.appendChild(factorCard);
        
        // 添加进入动画
        setTimeout(() => {
            factorCard.classList.add('fade-enter-active');
        }, 10);
    }

    /**
     * 窗口大小调整处理
     */
    handleResize() {
        // 重新调整图表大小
        this.charts.forEach(chart => {
            if (chart.resize) {
                chart.resize();
            }
        });
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
        
        setTimeout(() => toast.classList.add('show'), 10);
        
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
     * 防抖函数
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * 处理初始化错误
     */
    handleInitializationError(error) {
        this.hideLoading();
        this.showToast('系统初始化失败，启用模拟模式', 'error', 5000);
        
        // 启用模拟模式
        this.enableSimulationMode();
    }

    /**
     * 启用模拟模式
     */
    enableSimulationMode() {
        console.log('🧪 启用模拟模式');
        
        // 模拟数据更新
        this.startDataUpdates();
        
        // 标记为已初始化
        this.state.isInitialized = true;
        
        this.showToast('已启用模拟模式', 'warning');
    }

    /**
     * 销毁应用
     */
    destroy() {
        // 清除定时器
        this.intervals.forEach(interval => clearInterval(interval));
        this.intervals.clear();
        
        // 销毁图表
        this.charts.forEach(chart => {
            if (chart.destroy) {
                chart.destroy();
            }
        });
        this.charts.clear();
        
        console.log('🧹 数据分析应用资源已清理');
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
window.dataAnalysisApp = new DataAnalysisApp();

// 绑定全局函数
window.startAIGeneration = () => {
    window.dataAnalysisApp.startAIGeneration();
};

// 开发者工具
if (typeof window !== 'undefined' && window.location.hostname === 'localhost') {
    window.debugDataApp = () => {
        console.log('应用状态:', window.dataAnalysisApp.state);
        console.log('数据存储:', window.dataAnalysisApp.data);
        console.log('图表实例:', window.dataAnalysisApp.charts);
    };
}