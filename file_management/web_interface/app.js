/**
 * AI量化数据分析平台 - 现代化JavaScript应用
 * 全新设计的数据分析和因子研究平台
 */

class ModernDataAnalysisApp {
    constructor() {
        // 核心配置
        this.config = {
            apiUrl: 'http://localhost:8003/api/v1',
            wsUrl: 'ws://localhost:8003/ws',
            updateInterval: 2000,
            maxDataPoints: 10000,
            aiCostLimit: 100,
            dataRetentionDays: 365
        };

        // 应用状态
        this.state = {
            currentPage: 'dashboard',
            isDataConnected: false,
            aiEnginesStatus: new Map(),
            activeFactors: new Map(),
            backtestResults: new Map(),
            currentProject: null,
            generatingFactors: false,
            isInitialized: false
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
     * 初始化现代化应用
     */
    async init() {
        console.log('🎨 智能因子实验室初始化中...');
        
        try {
            // 显示加载状态
            this.showLoading('系统初始化中...');
            
            // 缓存DOM元素
            this.cacheElements();
            
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
            this.showToast('🎉 智能因子实验室初始化完成', 'success');
            console.log('✅ 现代化数据分析平台初始化完成');
            
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
            navButtons: document.querySelectorAll('.nav-button'),
            pages: document.querySelectorAll('.page'),
            
            // 统计显示
            totalSymbols: document.getElementById('total-symbols'),
            totalDatapoints: document.getElementById('total-datapoints'),
            activeFactors: document.getElementById('active-factors'),
            updateFrequency: document.getElementById('update-frequency'),
            
            // 图表容器
            marketOverviewChart: document.getElementById('market-overview-chart'),
            factorHeatmap: document.getElementById('factor-heatmap'),
            factorPreviewChart: document.getElementById('factor-preview-chart'),
            
            // 因子生成
            generatedFactors: document.getElementById('generated-factors'),
            generationHistory: document.getElementById('generation-history'),
            
            // 其他
            toastContainer: document.getElementById('toast-container'),
            loadingOverlay: document.getElementById('loading-overlay')
        };
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

        // 移动端触摸优化
        this.setupTouchOptimization();

        // 键盘快捷键
        document.addEventListener('keydown', (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key >= '1' && e.key <= '8') {
                e.preventDefault();
                const pages = [
                    'dashboard', 'factor-workshop', 'backtest-center',
                    'factor-market', 'data-center', 'analysis-workbench',
                    'report-center', 'settings'
                ];
                const pageIndex = parseInt(e.key) - 1;
                if (pages[pageIndex]) {
                    this.navigateToPage(pages[pageIndex]);
                }
            }
        });

        console.log('✓ 现代化事件监听器已设置');
    }

    /**
     * 初始化图表系统
     */
    async initChartSystem() {
        try {
            // 现代化图表配置
            const modernChartConfig = {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#4a5568',
                            font: { 
                                family: 'Inter',
                                size: 12,
                                weight: 500
                            },
                            padding: 20
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(255, 255, 255, 0.95)',
                        titleColor: '#2d3748',
                        bodyColor: '#4a5568',
                        borderColor: '#e2e8f0',
                        borderWidth: 1,
                        cornerRadius: 12,
                        padding: 12
                    }
                },
                scales: {
                    x: {
                        grid: { 
                            color: '#f7fafc',
                            borderColor: '#e2e8f0'
                        },
                        ticks: { 
                            color: '#718096',
                            font: { family: 'Inter' }
                        }
                    },
                    y: {
                        grid: { 
                            color: '#f7fafc',
                            borderColor: '#e2e8f0'
                        },
                        ticks: { 
                            color: '#718096',
                            font: { family: 'Inter' }
                        }
                    }
                }
            };

            // 初始化市场概览图表
            await this.initMarketOverviewChart(modernChartConfig);
            
            // 初始化因子预览图表
            await this.initFactorPreviewChart(modernChartConfig);
            
            // 初始化因子热力图
            await this.initFactorHeatmap();
            
            console.log('✓ 现代化图表系统已初始化');
        } catch (error) {
            console.error('图表初始化失败:', error);
        }
    }

    /**
     * 初始化市场概览图表
     */
    async initMarketOverviewChart(config) {
        const canvas = this.elements.marketOverviewChart;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        this.charts.set('market-overview', new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.generateTimeLabels(30),
                datasets: [{
                    label: 'BTC-USDT',
                    data: this.generateRandomData(30, 45000, 50000),
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: '#667eea',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 4
                }, {
                    label: 'ETH-USDT',
                    data: this.generateRandomData(30, 2800, 3200),
                    borderColor: '#f093fb',
                    backgroundColor: 'rgba(240, 147, 251, 0.1)',
                    borderWidth: 3,
                    fill: false,
                    tension: 0.4,
                    pointBackgroundColor: '#f093fb',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 4
                }]
            },
            options: {
                ...config,
                scales: {
                    ...config.scales,
                    y: {
                        ...config.scales.y,
                        title: {
                            display: true,
                            text: '价格 (USDT)',
                            color: '#4a5568',
                            font: { weight: 600 }
                        }
                    }
                }
            }
        }));
    }

    /**
     * 初始化因子预览图表
     */
    async initFactorPreviewChart(config) {
        const canvas = this.elements.factorPreviewChart;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        this.charts.set('factor-preview', new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.generateTimeLabels(20),
                datasets: [{
                    label: '因子值',
                    data: this.generateRandomData(20, -0.5, 0.5),
                    borderColor: '#4facfe',
                    backgroundColor: 'rgba(79, 172, 254, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: '#4facfe',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 5
                }]
            },
            options: {
                ...config,
                scales: {
                    ...config.scales,
                    y: {
                        ...config.scales.y,
                        title: {
                            display: true,
                            text: '因子值',
                            color: '#4a5568',
                            font: { weight: 600 }
                        }
                    }
                }
            }
        }));
    }

    /**
     * 初始化因子热力图
     */
    async initFactorHeatmap() {
        const container = this.elements.factorHeatmap;
        if (!container || typeof echarts === 'undefined') return;

        const heatmapChart = echarts.init(container);
        
        // 生成现代化热力图数据
        const factors = ['RSI', 'MACD', 'BB', 'VOL', 'MOM', 'MEAN'];
        const symbols = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT'];
        const data = [];
        
        factors.forEach((factor, i) => {
            symbols.forEach((symbol, j) => {
                data.push([i, j, Math.random() * 0.4 - 0.2]);
            });
        });

        const option = {
            tooltip: {
                trigger: 'item',
                backgroundColor: 'rgba(255, 255, 255, 0.95)',
                borderColor: '#e2e8f0',
                borderWidth: 1,
                textStyle: { color: '#2d3748' },
                formatter: function(params) {
                    return `${factors[params.data[0]]} × ${symbols[params.data[1]]}<br/>相关性: ${(params.data[2]).toFixed(3)}`;
                }
            },
            grid: {
                height: '80%',
                top: '10%',
                left: '15%',
                right: '5%',
                bottom: '10%'
            },
            xAxis: {
                type: 'category',
                data: factors,
                axisLabel: { 
                    color: '#718096',
                    fontSize: 12,
                    fontFamily: 'Inter'
                },
                axisLine: { lineStyle: { color: '#e2e8f0' } }
            },
            yAxis: {
                type: 'category',
                data: symbols,
                axisLabel: { 
                    color: '#718096',
                    fontSize: 12,
                    fontFamily: 'Inter'
                },
                axisLine: { lineStyle: { color: '#e2e8f0' } }
            },
            visualMap: {
                min: -0.2,
                max: 0.2,
                calculable: true,
                orient: 'horizontal',
                left: 'center',
                bottom: '2%',
                inRange: {
                    color: ['#f5576c', '#ffffff', '#4facfe']
                },
                textStyle: { 
                    color: '#718096',
                    fontFamily: 'Inter'
                }
            },
            series: [{
                name: '因子相关性',
                type: 'heatmap',
                data: data,
                label: {
                    show: true,
                    formatter: function(params) {
                        return params.data[2].toFixed(2);
                    },
                    color: '#2d3748',
                    fontSize: 10
                },
                emphasis: {
                    itemStyle: {
                        shadowBlur: 20,
                        shadowColor: 'rgba(0, 0, 0, 0.2)'
                    }
                }
            }]
        };

        heatmapChart.setOption(option);
        this.charts.set('factor-heatmap', heatmapChart);
    }

    /**
     * 连接数据源
     */
    async connectDataSources() {
        try {
            console.log('🔗 连接现代化数据源...');
            
            // 模拟数据源连接
            await this.simulateDataSourceConnection();
            
            console.log('✓ 数据源连接成功');
        } catch (error) {
            console.error('数据源连接失败:', error);
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
            
            // 并行加载各种数据
            await Promise.all([
                this.loadDashboardStats(),
                this.loadAIEngineStatus(),
                this.loadFactorLibrary(),
                this.loadDataSources()
            ]);
            
            console.log('✓ 初始数据加载完成');
        } catch (error) {
            console.error('初始数据加载失败:', error);
        }
    }

    /**
     * 加载仪表盘统计
     */
    /**
     * API请求封装 - 带重试和错误处理
     */
    async makeApiRequest(endpoint, options = {}, retries = 3) {
        const url = `${this.config.apiUrl}${endpoint}`;
        
        for (let attempt = 1; attempt <= retries; attempt++) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 10000); // 10秒超时

                const response = await fetch(url, {
                    headers: {
                        'Content-Type': 'application/json',
                        ...options.headers
                    },
                    signal: controller.signal,
                    ...options
                });

                clearTimeout(timeoutId);

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                
                // 如果重连成功，更新连接状态
                if (attempt > 1) {
                    console.log(`✅ API重连成功，尝试次数: ${attempt}`);
                    this.showToast('API连接已恢复', 'success');
                }
                
                return data;
                
            } catch (error) {
                console.warn(`API请求失败 ${endpoint} (尝试 ${attempt}/${retries}):`, error.message);
                
                if (attempt === retries) {
                    console.error(`API请求最终失败 ${endpoint}:`, error);
                    this.handleApiError(endpoint, error);
                    throw error;
                }
                
                // 等待重试 (指数退避)
                const delay = Math.min(1000 * Math.pow(2, attempt - 1), 5000);
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
    }

    /**
     * 处理API错误
     */
    handleApiError(endpoint, error) {
        const isNetworkError = error.name === 'TypeError' || error.message.includes('Failed to fetch');
        const isTimeoutError = error.name === 'AbortError';
        
        if (isNetworkError) {
            this.showToast('网络连接失败，请检查网络状态', 'error');
        } else if (isTimeoutError) {
            this.showToast('请求超时，服务器响应缓慢', 'warning');
        } else {
            this.showToast(`API请求失败: ${error.message}`, 'error');
        }

        // 触发重连机制
        this.scheduleReconnect();
    }

    /**
     * 定时重连机制
     */
    scheduleReconnect() {
        if (this.reconnectTimer) {
            return; // 已经在重连中
        }

        this.reconnectTimer = setTimeout(async () => {
            try {
                console.log('🔄 尝试重新连接API...');
                await this.makeApiRequest('/health');
                console.log('✅ API连接已恢复');
                this.showToast('API连接已恢复', 'success');
                this.reconnectTimer = null;
            } catch (error) {
                console.log('❌ API重连失败，30秒后再次尝试');
                this.reconnectTimer = null;
                this.scheduleReconnect(); // 继续重连
            }
        }, 30000); // 30秒后重连
    }

    /**
     * 健康检查
     */
    async performHealthCheck() {
        try {
            const response = await this.makeApiRequest('/health', {}, 1); // 只尝试一次
            console.log('✅ API健康检查通过');
            return true;
        } catch (error) {
            console.warn('❌ API健康检查失败');
            return false;
        }
    }

    async loadDashboardStats() {
        try {
            console.log('🔄 从API加载仪表盘统计数据...');
            
            const response = await this.makeApiRequest('/data/overview');
            
            if (response.success && response.data) {
                const { statistics } = response.data;
                
                const stats = {
                    totalSymbols: statistics.total_symbols,
                    totalDatapoints: this.formatDataSize(statistics.total_datapoints),
                    activeFactors: statistics.active_factors,
                    updateFrequency: `${statistics.update_frequency}Hz`
                };

                // 动画更新统计数据
                this.animateStatValue(this.elements.totalSymbols, stats.totalSymbols);
                this.animateStatValue(this.elements.totalDatapoints, stats.totalDatapoints, false);
                this.animateStatValue(this.elements.activeFactors, stats.activeFactors);
                this.animateStatValue(this.elements.updateFrequency, stats.updateFrequency, false);

                // 更新数据质量指标
                this.updateDataQualityMetrics(statistics);

                console.log('✅ 仪表盘统计数据加载成功');
            }
        } catch (error) {
            console.warn('API连接失败，使用模拟数据');
            this.loadFallbackStats();
        }
    }

    /**
     * 格式化数据大小
     */
    formatDataSize(size) {
        if (size >= 1000000) {
            return `${(size / 1000000).toFixed(1)}M`;
        } else if (size >= 1000) {
            return `${(size / 1000).toFixed(1)}K`;
        }
        return size.toString();
    }

    /**
     * 更新数据质量指标
     */
    updateDataQualityMetrics(statistics) {
        const qualityMetrics = {
            completeness: statistics.data_completeness || 98.5,
            accuracy: statistics.data_accuracy || 99.2,
            timeliness: statistics.data_timeliness || 96.8
        };

        // 更新进度条
        document.querySelectorAll('.quality-item').forEach((item, index) => {
            const progressFill = item.querySelector('.progress-fill');
            const progressText = item.querySelector('.progress-text');
            
            if (progressFill && progressText) {
                const keys = Object.keys(qualityMetrics);
                const value = qualityMetrics[keys[index]];
                
                if (value) {
                    progressFill.style.width = `${value}%`;
                    progressText.textContent = `${value.toFixed(1)}%`;
                }
            }
        });
    }

    /**
     * 备用统计数据加载
     */
    loadFallbackStats() {
        const stats = {
            totalSymbols: Math.floor(Math.random() * 100 + 1200),
            totalDatapoints: `${(Math.random() * 0.5 + 2.2).toFixed(1)}M`,
            activeFactors: Math.floor(Math.random() * 20 + 150),
            updateFrequency: `${(Math.random() * 0.5 + 2.0).toFixed(1)}Hz`
        };

        // 动画更新统计数据
        this.animateStatValue(this.elements.totalSymbols, stats.totalSymbols);
        this.animateStatValue(this.elements.totalDatapoints, stats.totalDatapoints, false);
        this.animateStatValue(this.elements.activeFactors, stats.activeFactors);
        this.animateStatValue(this.elements.updateFrequency, stats.updateFrequency, false);
    }

    /**
     * 动画更新统计值
     */
    animateStatValue(element, targetValue, isNumber = true) {
        if (!element) return;

        if (isNumber) {
            let currentValue = 0;
            const increment = targetValue / 30;
            const timer = setInterval(() => {
                currentValue += increment;
                if (currentValue >= targetValue) {
                    currentValue = targetValue;
                    clearInterval(timer);
                }
                element.textContent = Math.floor(currentValue).toLocaleString();
            }, 50);
        } else {
            element.textContent = targetValue;
        }
    }

    /**
     * 加载AI引擎状态
     */
    async loadAIEngineStatus() {
        try {
            console.log('🔄 从API加载AI引擎状态...');
            
            const response = await this.makeApiRequest('/ai/engines');
            
            if (response.success && response.data) {
                const { engines } = response.data;
                
                engines.forEach(engine => {
                    this.state.aiEnginesStatus.set(engine.name, {
                        name: engine.name,
                        status: engine.status,
                        calls: engine.daily_calls,
                        responseTime: `${engine.avg_response_time}s`,
                        cost: engine.cost_today
                    });
                });

                // 更新UI显示
                this.updateAIEngineStatusUI(engines, response.data);

                console.log('✅ AI引擎状态加载成功');
            }
        } catch (error) {
            console.warn('AI引擎状态API失败，使用模拟数据');
            this.loadFallbackAIEngineStatus();
        }
    }

    /**
     * 更新AI引擎状态UI
     */
    updateAIEngineStatusUI(engines, data) {
        const aiStatusContainer = document.getElementById('ai-engines-status');
        if (!aiStatusContainer) return;

        let html = '';
        engines.forEach(engine => {
            const statusClass = engine.status === 'online' ? 'status-online' : 'status-warning';
            html += `
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
                    <span>${engine.name}</span>
                    <span class="status-indicator ${statusClass}">
                        <i class="fas fa-brain"></i>
                        ${engine.status === 'online' ? '运行中' : '离线'}
                    </span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.875rem; color: var(--text-muted); margin-bottom: 1rem;">
                    <span>调用次数: ${engine.daily_calls}</span>
                    <span>响应时间: ${engine.avg_response_time}s</span>
                </div>
            `;
        });

        // 添加成本信息
        html += `
            <div style="text-align: center; margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border-color);">
                <div style="font-size: 0.875rem; color: var(--text-muted);">今日调用成本</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: var(--text-primary);">$${data.total_cost_today.toFixed(2)}</div>
                <div style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem;">
                    月度预算: $${data.monthly_budget} (已用: $${data.monthly_used.toFixed(2)})
                </div>
            </div>
        `;

        aiStatusContainer.innerHTML = html;
    }

    /**
     * 备用AI引擎状态
     */
    loadFallbackAIEngineStatus() {
        const engines = [
            {
                name: 'DeepSeek',
                status: 'online',
                calls: Math.floor(Math.random() * 500 + 1000),
                responseTime: `${(Math.random() * 0.5 + 1.0).toFixed(1)}s`
            },
            {
                name: 'Gemini',
                status: 'online',
                calls: Math.floor(Math.random() * 400 + 800),
                responseTime: `${(Math.random() * 0.8 + 1.5).toFixed(1)}s`
            }
        ];

        engines.forEach(engine => {
            this.state.aiEnginesStatus.set(engine.name, engine);
        });
    }

    /**
     * 启动数据更新循环
     */
    startDataUpdates() {
        // 定期更新统计数据
        this.intervals.set('statsUpdate', setInterval(() => {
            this.updateStats();
        }, 5000));

        // 定期更新图表数据
        this.intervals.set('chartUpdate', setInterval(() => {
            this.updateCharts();
        }, 3000));

        console.log('✓ 数据更新循环已启动');
    }

    /**
     * 更新统计数据
     */
    updateStats() {
        // 轻微调整统计数据以模拟实时更新
        this.loadDashboardStats();
    }

    /**
     * 更新图表数据
     */
    updateCharts() {
        const marketChart = this.charts.get('market-overview');
        if (marketChart) {
            // 更新市场概览图表
            marketChart.data.datasets.forEach(dataset => {
                dataset.data.shift();
                const lastValue = dataset.data[dataset.data.length - 1];
                const newValue = lastValue * (0.98 + Math.random() * 0.04);
                dataset.data.push(newValue);
            });
            
            // 更新时间标签
            marketChart.data.labels.shift();
            marketChart.data.labels.push(new Date().toLocaleTimeString());
            
            marketChart.update('none');
        }

        const factorChart = this.charts.get('factor-preview');
        if (factorChart) {
            // 更新因子预览图表
            factorChart.data.datasets[0].data.shift();
            factorChart.data.datasets[0].data.push((Math.random() - 0.5) * 0.8);
            
            factorChart.data.labels.shift();
            factorChart.data.labels.push(new Date().toLocaleTimeString());
            
            factorChart.update('none');
        }
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
            
            // 页面切换动画
            targetPage.style.opacity = '0';
            targetPage.style.transform = 'translateY(20px)';
            
            setTimeout(() => {
                targetPage.style.opacity = '1';
                targetPage.style.transform = 'translateY(0)';
            }, 50);
            
            // 页面特殊处理
            this.handlePageSwitch(pageName);
            
            console.log(`📄 切换到页面: ${pageName}`);
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
            case 'factor-workshop':
                this.initFactorWorkshop();
                break;
            case 'backtest-center':
                this.initBacktestCenter();
                break;
            case 'factor-market':
                this.initFactorMarket();
                break;
            case 'data-center':
                this.initDataCenter();
                break;
            case 'analysis-workbench':
                this.initAnalysisWorkbench();
                break;
            case 'report-center':
                this.initReportCenter();
                break;
            case 'settings':
                this.initSettings();
                break;
        }
    }

    /**
     * 刷新仪表盘
     */
    refreshDashboard() {
        console.log('🔄 刷新仪表盘');
        this.loadDashboardStats();
    }

    /**
     * 初始化因子工坊
     */
    initFactorWorkshop() {
        console.log('🧪 初始化智能因子工坊');
        // 因子工坊特殊初始化逻辑
    }

    /**
     * 初始化回测中心
     */
    initBacktestCenter() {
        console.log('🚀 初始化策略回测中心');
        
        // 初始化回测性能图表
        this.initBacktestPerformanceChart();
        
        // 绑定回测历史项点击事件
        this.bindBacktestHistoryEvents();
    }

    /**
     * 初始化回测性能图表
     */
    initBacktestPerformanceChart() {
        const canvas = document.getElementById('backtest-performance-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        
        // 生成模拟回测数据
        const days = 30;
        const labels = [];
        const strategyReturns = [];
        const benchmarkReturns = [];
        
        let strategyValue = 100000;
        let benchmarkValue = 100000;
        
        for (let i = 0; i < days; i++) {
            const date = new Date();
            date.setDate(date.getDate() - (days - i));
            labels.push(date.toLocaleDateString());
            
            // 策略收益（更好的表现）
            const strategyReturn = (Math.random() - 0.45) * 0.02; // 略微向上偏移
            strategyValue *= (1 + strategyReturn);
            strategyReturns.push(((strategyValue - 100000) / 100000 * 100).toFixed(2));
            
            // 基准收益
            const benchmarkReturn = (Math.random() - 0.5) * 0.015;
            benchmarkValue *= (1 + benchmarkReturn);
            benchmarkReturns.push(((benchmarkValue - 100000) / 100000 * 100).toFixed(2));
        }

        this.charts.set('backtest-performance', new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: '策略收益',
                    data: strategyReturns,
                    borderColor: '#22c55e',
                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                    borderWidth: 3,
                    fill: false,
                    tension: 0.4,
                    pointBackgroundColor: '#22c55e',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 4
                }, {
                    label: '基准收益 (BTC)',
                    data: benchmarkReturns,
                    borderColor: '#64748b',
                    backgroundColor: 'rgba(100, 116, 139, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    borderDash: [5, 5],
                    tension: 0.4,
                    pointBackgroundColor: '#64748b',
                    pointBorderColor: '#ffffff',
                    pointBorderWidth: 2,
                    pointRadius: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#4a5568',
                            font: { 
                                family: 'Inter',
                                size: 12,
                                weight: 500
                            },
                            padding: 20
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(255, 255, 255, 0.95)',
                        titleColor: '#2d3748',
                        bodyColor: '#4a5568',
                        borderColor: '#e2e8f0',
                        borderWidth: 1,
                        cornerRadius: 12,
                        padding: 12,
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: ${context.parsed.y}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { 
                            color: '#f7fafc',
                            borderColor: '#e2e8f0'
                        },
                        ticks: { 
                            color: '#718096',
                            font: { family: 'Inter' }
                        }
                    },
                    y: {
                        grid: { 
                            color: '#f7fafc',
                            borderColor: '#e2e8f0'
                        },
                        ticks: { 
                            color: '#718096',
                            font: { family: 'Inter' },
                            callback: function(value) {
                                return value + '%';
                            }
                        },
                        title: {
                            display: true,
                            text: '累计收益率 (%)',
                            color: '#4a5568',
                            font: { weight: 600 }
                        }
                    }
                }
            }
        }));
    }

    /**
     * 绑定回测历史事件
     */
    bindBacktestHistoryEvents() {
        document.querySelectorAll('.backtest-history-item').forEach(item => {
            item.addEventListener('click', () => {
                // 显示详细分析结果
                this.showBacktestResults();
                
                // 添加选中效果
                document.querySelectorAll('.backtest-history-item').forEach(i => 
                    i.style.background = 'transparent'
                );
                item.style.background = 'rgba(102, 126, 234, 0.05)';
            });
        });
    }

    /**
     * 显示回测结果
     */
    showBacktestResults() {
        const resultsCard = document.getElementById('backtest-results');
        if (resultsCard) {
            resultsCard.style.display = 'block';
            
            // 滚动到结果区域
            resultsCard.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start' 
            });
        }
    }

    /**
     * 开始回测
     */
    startBacktest() {
        console.log('🚀 启动策略回测');
        
        // 显示进度卡片
        const progressCard = document.getElementById('backtest-progress');
        if (progressCard) {
            progressCard.style.display = 'block';
            
            // 模拟回测进度
            this.simulateBacktestProgress();
        }
        
        this.showToast('🚀 回测已启动，正在处理...', 'info');
    }

    /**
     * 模拟回测进度
     */
    simulateBacktestProgress() {
        const progressFill = document.querySelector('.progress-bar-fill');
        const currentStep = document.getElementById('backtest-current-step');
        const eta = document.getElementById('backtest-eta');
        
        const steps = [
            { text: '数据准备中...', duration: 1000 },
            { text: '因子计算中...', duration: 2000 },
            { text: '信号生成中...', duration: 1500 },
            { text: '回测执行中...', duration: 2500 },
            { text: '性能分析中...', duration: 1000 },
            { text: '报告生成中...', duration: 500 }
        ];
        
        let currentStepIndex = 0;
        let totalDuration = steps.reduce((sum, step) => sum + step.duration, 0);
        let elapsed = 0;
        
        const updateProgress = () => {
            if (currentStepIndex < steps.length) {
                const step = steps[currentStepIndex];
                
                if (currentStep) currentStep.textContent = step.text;
                
                setTimeout(() => {
                    elapsed += step.duration;
                    const progress = (elapsed / totalDuration) * 100;
                    
                    if (progressFill) {
                        progressFill.style.width = `${progress}%`;
                    }
                    
                    if (eta) {
                        const remaining = Math.ceil((totalDuration - elapsed) / 1000);
                        eta.textContent = `预计还需 ${remaining} 秒`;
                    }
                    
                    currentStepIndex++;
                    if (currentStepIndex < steps.length) {
                        updateProgress();
                    } else {
                        // 回测完成
                        setTimeout(() => {
                            this.completeBacktest();
                        }, 500);
                    }
                }, step.duration);
            }
        };
        
        updateProgress();
    }

    /**
     * 完成回测
     */
    completeBacktest() {
        // 隐藏进度卡片
        const progressCard = document.getElementById('backtest-progress');
        if (progressCard) {
            progressCard.style.display = 'none';
        }
        
        // 显示结果
        this.showBacktestResults();
        
        // 更新回测历史（添加新的回测记录）
        this.addNewBacktestRecord();
        
        this.showToast('✅ 回测完成！结果已生成', 'success');
    }

    /**
     * 添加新的回测记录
     */
    addNewBacktestRecord() {
        const historyContainer = document.getElementById('backtest-history');
        if (!historyContainer) return;
        
        const newRecord = document.createElement('div');
        newRecord.className = 'backtest-history-item';
        newRecord.style.cssText = 'display: flex; align-items: center; padding: 1rem; border: 1px solid var(--border-color); border-radius: var(--radius-md); margin-bottom: 0.75rem; cursor: pointer; transition: all 0.3s ease; background: rgba(102, 126, 234, 0.05);';
        
        const now = new Date();
        const dateStr = now.toLocaleDateString() + ' ' + now.toLocaleTimeString().slice(0, 5);
        const return_rate = (Math.random() * 20 + 15).toFixed(1);
        
        newRecord.innerHTML = `
            <div style="flex: 1;">
                <div style="font-weight: 600; margin-bottom: 0.25rem;">多因子量化策略 v2.1</div>
                <div style="font-size: 0.875rem; color: var(--text-muted);">${dateStr}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-weight: 600; color: #22c55e;">+${return_rate}%</div>
                <div style="font-size: 0.875rem; color: var(--text-muted);">年化收益</div>
            </div>
            <div style="margin-left: 1rem;">
                <div class="status-indicator status-online">
                    <i class="fas fa-check"></i>
                    完成
                </div>
            </div>
        `;
        
        // 添加点击事件
        newRecord.addEventListener('click', () => {
            this.showBacktestResults();
            document.querySelectorAll('.backtest-history-item').forEach(i => 
                i.style.background = 'transparent'
            );
            newRecord.style.background = 'rgba(102, 126, 234, 0.05)';
        });
        
        historyContainer.insertBefore(newRecord, historyContainer.firstChild);
        
        // 添加进入动画
        setTimeout(() => {
            newRecord.style.opacity = '1';
            newRecord.style.transform = 'translateX(0)';
        }, 100);
    }

    /**
     * 初始化因子市场
     */
    initFactorMarket() {
        console.log('🛒 初始化因子市场');
        
        // 绑定搜索和过滤事件
        this.bindFactorMarketEvents();
        
        // 加载因子库数据
        this.loadFactorLibrary();
    }

    /**
     * 绑定因子市场事件
     */
    bindFactorMarketEvents() {
        // 搜索框事件
        const searchInput = document.querySelector('#factor-market-page input[placeholder="搜索因子..."]');
        if (searchInput) {
            searchInput.addEventListener('input', this.debounce((e) => {
                this.filterFactors(e.target.value);
            }, 300));
        }

        // 分类过滤事件
        const categorySelect = document.querySelector('#factor-market-page select');
        if (categorySelect) {
            categorySelect.addEventListener('change', (e) => {
                this.filterFactorsByCategory(e.target.value);
            });
        }
    }

    /**
     * 过滤因子
     */
    filterFactors(searchTerm) {
        const factorCards = document.querySelectorAll('#factor-library-grid .factor-card');
        const term = searchTerm.toLowerCase();

        factorCards.forEach(card => {
            const factorName = card.querySelector('.factor-name').textContent.toLowerCase();
            const description = card.querySelector('div[style*="line-height: 1.4"]').textContent.toLowerCase();
            
            const matches = factorName.includes(term) || description.includes(term);
            card.style.display = matches ? 'block' : 'none';
        });

        this.showToast(`找到 ${[...factorCards].filter(card => card.style.display !== 'none').length} 个相关因子`, 'info');
    }

    /**
     * 按分类过滤因子
     */
    filterFactorsByCategory(category) {
        const factorCards = document.querySelectorAll('#factor-library-grid .factor-card');
        
        factorCards.forEach(card => {
            if (category === '全部分类') {
                card.style.display = 'block';
            } else {
                const categories = card.querySelectorAll('.status-indicator');
                let hasCategory = false;
                
                categories.forEach(indicator => {
                    if (indicator.textContent.trim() === category) {
                        hasCategory = true;
                    }
                });
                
                card.style.display = hasCategory ? 'block' : 'none';
            }
        });
    }

    /**
     * 使用因子
     */
    useFactor(factorId) {
        console.log(`🎯 使用因子: ${factorId}`);
        
        // 模拟添加到回测配置
        this.showToast('✅ 因子已添加到策略配置中', 'success');
        
        // 可以跳转到回测页面
        setTimeout(() => {
            if (confirm('是否跳转到回测中心进行策略配置？')) {
                this.navigateToPage('backtest-center');
            }
        }, 1000);
    }

    /**
     * 添加新因子
     */
    addNewFactor() {
        console.log('➕ 添加新因子');
        
        // 可以打开一个模态框或跳转到因子创建页面
        this.showToast('💡 请前往"智能因子工坊"创建新因子', 'info');
        
        setTimeout(() => {
            if (confirm('是否跳转到智能因子工坊？')) {
                this.navigateToPage('factor-workshop');
            }
        }, 1500);
    }

    /**
     * 初始化数据中心
     */
    initDataCenter() {
        console.log('🗄️ 初始化数据中心');
        
        // 初始化数据源分布图表
        this.initDataSourceDistributionChart();
        
        // 开始实时监控数据更新
        this.startDataSourceMonitoring();
    }

    /**
     * 初始化数据源分布图表
     */
    initDataSourceDistributionChart() {
        const canvas = document.getElementById('data-source-distribution-chart');
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        
        this.charts.set('data-source-distribution', new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Binance', 'OKX', 'Coinglass', '其他数据源'],
                datasets: [{
                    data: [45, 30, 15, 10],
                    backgroundColor: [
                        '#f7931e',
                        '#0066cc', 
                        '#10b981',
                        '#64748b'
                    ],
                    borderColor: '#ffffff',
                    borderWidth: 3,
                    hoverOffset: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#4a5568',
                            font: { 
                                family: 'Inter',
                                size: 12,
                                weight: 500
                            },
                            padding: 15,
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(255, 255, 255, 0.95)',
                        titleColor: '#2d3748',
                        bodyColor: '#4a5568',
                        borderColor: '#e2e8f0',
                        borderWidth: 1,
                        cornerRadius: 12,
                        padding: 12,
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${context.parsed}%`;
                            }
                        }
                    }
                }
            }
        }));
    }

    /**
     * 开始数据源监控
     */
    startDataSourceMonitoring() {
        // 定期更新数据源状态
        this.intervals.set('dataSourceMonitoring', setInterval(() => {
            this.updateDataSourceStatus();
        }, 10000)); // 每10秒更新一次
    }

    /**
     * 更新数据源状态
     */
    updateDataSourceStatus() {
        // 模拟更新延迟数据
        const dataSources = document.querySelectorAll('.data-source-card');
        
        dataSources.forEach((card, index) => {
            const latencyElements = card.querySelectorAll('div[style*="font-size: 1.2rem"]');
            if (latencyElements.length >= 4) {
                // 更新延迟
                const currentLatency = latencyElements[0];
                let newLatency;
                
                if (index === 0) { // Binance
                    newLatency = Math.floor(Math.random() * 50 + 80) + 'ms';
                } else if (index === 1) { // OKX
                    newLatency = Math.floor(Math.random() * 40 + 60) + 'ms';
                } else { // Coinglass
                    newLatency = (Math.random() * 1 + 1.5).toFixed(1) + 's';
                    currentLatency.style.color = '#f59e0b'; // 保持警告色
                }
                
                currentLatency.textContent = newLatency;
                
                // 更新最后更新时间
                const lastUpdateElement = latencyElements[3];
                const minutesAgo = Math.floor(Math.random() * 5 + 1);
                lastUpdateElement.textContent = `${minutesAgo}分钟前`;
            }
        });
    }

    /**
     * 刷新数据源状态
     */
    refreshDataSources() {
        console.log('🔄 刷新数据源状态');
        
        this.showToast('🔄 正在刷新数据源状态...', 'info');
        
        // 模拟刷新过程
        setTimeout(() => {
            this.updateDataSourceStatus();
            this.showToast('✅ 数据源状态已更新', 'success');
        }, 1500);
    }

    /**
     * 添加数据源
     */
    addDataSource() {
        console.log('➕ 添加新数据源');
        
        // 模拟添加数据源的过程
        const dataSources = [
            'Coinbase Pro API',
            'Kraken WebSocket', 
            'Bybit Market Data',
            'FTX REST API'
        ];
        
        const randomSource = dataSources[Math.floor(Math.random() * dataSources.length)];
        
        this.showToast(`📡 正在连接 ${randomSource}...`, 'info');
        
        setTimeout(() => {
            this.showToast(`✅ ${randomSource} 连接成功！`, 'success');
            
            // 可以在这里动态添加新的数据源卡片
            // this.addDataSourceCard(randomSource);
        }, 2000);
    }

    /**
     * 初始化分析工作台
     */
    initAnalysisWorkbench() {
        console.log('🔬 初始化分析工作台');
        // 分析工作台特殊初始化逻辑
    }

    /**
     * 初始化报告中心
     */
    initReportCenter() {
        console.log('📊 初始化报告中心');
        // 报告中心特殊初始化逻辑
    }

    /**
     * 生成报告
     */
    generateReport() {
        console.log('📝 开始生成报告');
        
        // 获取选择的报告类型
        const reportType = document.getElementById('report-type')?.value || 'factor-analysis';
        const reportTypeMap = {
            'factor-analysis': '因子分析报告',
            'backtest-summary': '回测汇总报告', 
            'performance-analysis': '业绩分析报告',
            'risk-assessment': '风险评估报告',
            'market-overview': '市场概览报告',
            'data-quality': '数据质量报告'
        };
        
        const reportName = reportTypeMap[reportType] || '智能报告';
        
        this.showToast(`🎯 开始生成${reportName}...`, 'info');
        
        // 显示进度条
        this.showReportProgress();
        
        // 模拟报告生成过程
        this.simulateReportGeneration(reportName);
    }

    /**
     * 显示报告生成进度
     */
    showReportProgress() {
        const progressCard = document.getElementById('report-progress');
        if (progressCard) {
            progressCard.style.display = 'block';
        }
    }

    /**
     * 模拟报告生成过程
     */
    simulateReportGeneration(reportName) {
        const progressBar = document.getElementById('report-progress-bar');
        const currentStep = document.getElementById('report-current-step');
        const progressPercent = document.getElementById('report-progress-percent');
        
        const steps = [
            { text: '正在收集数据...', duration: 2000 },
            { text: '正在分析数据...', duration: 3000 },
            { text: '正在生成图表...', duration: 2000 },
            { text: '正在应用AI洞察...', duration: 2500 },
            { text: '正在生成报告...', duration: 1500 },
            { text: '正在优化格式...', duration: 1000 }
        ];
        
        let currentStepIndex = 0;
        let totalDuration = steps.reduce((sum, step) => sum + step.duration, 0);
        let elapsed = 0;
        
        const updateProgress = () => {
            if (currentStepIndex < steps.length) {
                const step = steps[currentStepIndex];
                
                if (currentStep) currentStep.textContent = step.text;
                
                setTimeout(() => {
                    elapsed += step.duration;
                    const progress = (elapsed / totalDuration) * 100;
                    
                    if (progressBar) {
                        progressBar.style.width = `${progress}%`;
                    }
                    
                    if (progressPercent) {
                        progressPercent.textContent = `${Math.round(progress)}%`;
                    }
                    
                    currentStepIndex++;
                    if (currentStepIndex < steps.length) {
                        updateProgress();
                    } else {
                        // 报告生成完成
                        setTimeout(() => {
                            this.completeReportGeneration(reportName);
                        }, 500);
                    }
                }, step.duration);
            }
        };
        
        updateProgress();
    }

    /**
     * 完成报告生成
     */
    completeReportGeneration(reportName) {
        // 隐藏进度条
        const progressCard = document.getElementById('report-progress');
        if (progressCard) {
            progressCard.style.display = 'none';
        }
        
        // 添加新生成的报告到历史列表
        this.addNewReport(reportName);
        
        this.showToast(`✅ ${reportName}生成完成！`, 'success');
    }

    /**
     * 添加新报告到历史列表
     */
    addNewReport(reportName) {
        const reportsList = document.getElementById('reports-list');
        if (!reportsList) return;
        
        const newReport = document.createElement('div');
        newReport.className = 'report-item';
        newReport.style.cssText = 'display: flex; align-items: center; padding: 1rem; border: 1px solid var(--border-color); border-radius: var(--radius-md); margin-bottom: 0.75rem; cursor: pointer; transition: all 0.3s ease; background: rgba(102, 126, 234, 0.05);';
        
        const now = new Date();
        const dateStr = now.toLocaleDateString() + ' ' + now.toLocaleTimeString().slice(0, 5);
        const fileSize = (Math.random() * 3 + 1.5).toFixed(1);
        
        // 根据报告类型选择图标和颜色
        const iconMap = {
            '因子分析报告': { icon: 'fas fa-chart-bar', gradient: 'var(--primary-gradient)' },
            '回测汇总报告': { icon: 'fas fa-rocket', gradient: 'var(--success-gradient)' },
            '业绩分析报告': { icon: 'fas fa-chart-line', gradient: 'var(--secondary-gradient)' },
            '风险评估报告': { icon: 'fas fa-shield-alt', gradient: 'var(--warning-gradient)' },
            '市场概览报告': { icon: 'fas fa-globe', gradient: 'var(--primary-gradient)' },
            '数据质量报告': { icon: 'fas fa-database', gradient: 'var(--success-gradient)' }
        };
        
        const config = iconMap[reportName] || iconMap['因子分析报告'];
        
        newReport.innerHTML = `
            <div style="width: 50px; height: 50px; background: ${config.gradient}; border-radius: var(--radius-md); display: flex; align-items: center; justify-content: center; color: white; margin-right: 1rem;">
                <i class="${config.icon}"></i>
            </div>
            <div style="flex: 1;">
                <div style="font-weight: 600; margin-bottom: 0.25rem;">${reportName}</div>
                <div style="font-size: 0.875rem; color: var(--text-muted); margin-bottom: 0.25rem;">PDF • ${fileSize}MB • 刚刚生成</div>
                <div style="font-size: 0.75rem; color: var(--text-muted);">${dateStr} 生成</div>
            </div>
            <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                <button class="btn btn-secondary" style="padding: 0.4rem 0.8rem; font-size: 0.8rem;">
                    <i class="fas fa-download"></i>
                    下载
                </button>
                <button class="btn btn-secondary" style="padding: 0.4rem 0.8rem; font-size: 0.8rem;">
                    <i class="fas fa-eye"></i>
                    预览
                </button>
            </div>
        `;
        
        reportsList.insertBefore(newReport, reportsList.firstChild);
        
        // 添加进入动画
        setTimeout(() => {
            newReport.style.background = 'transparent';
        }, 2000);
    }

    /**
     * 刷新报告列表
     */
    refreshReports() {
        console.log('🔄 刷新报告列表');
        this.showToast('🔄 正在刷新报告列表...', 'info');
        
        // 模拟刷新过程
        setTimeout(() => {
            this.showToast('✅ 报告列表已更新', 'success');
        }, 1000);
    }

    /**
     * 创建报告模板
     */
    createTemplate() {
        console.log('➕ 创建新报告模板');
        this.showToast('📝 报告模板编辑器正在开发中...', 'info');
    }

    /**
     * 初始化设置
     */
    initSettings() {
        console.log('⚙️ 初始化个人设置');
        // 设置特殊初始化逻辑
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
            this.showToast('🤖 启动AI因子生成...', 'info');
            
            // 显示生成进度
            this.showGeneratingProgress();
            
            // 调用真实的AI生成API
            await this.callAIGenerationAPI();
            
            this.state.generatingFactors = false;
            this.showToast('✨ AI因子生成完成！', 'success');
            
        } catch (error) {
            console.error('AI生成失败:', error);
            this.state.generatingFactors = false;
            this.showToast('❌ AI生成失败，请重试', 'error');
            
            // 移除生成中的卡片
            const generatingCard = document.querySelector('.factor-card.generating');
            if (generatingCard) {
                generatingCard.remove();
            }
        }
    }

    /**
     * 调用AI生成API
     */
    async callAIGenerationAPI() {
        try {
            console.log('🔄 调用AI因子生成API...');
            
            const response = await this.makeApiRequest('/factors/generate');
            
            if (response.success && response.data) {
                const { factor, generation_cost } = response.data;
                
                // 移除生成中的占位符
                const generatingCard = document.querySelector('.factor-card.generating');
                if (generatingCard) {
                    generatingCard.remove();
                }
                
                // 添加生成的因子
                this.addGeneratedFactorFromAPI(factor, generation_cost);

                console.log('✅ AI因子生成API调用成功');
            }
        } catch (error) {
            console.warn('AI生成API失败，使用模拟生成');
            await this.simulateAIGeneration();
        }
    }

    /**
     * 从API添加生成的因子
     */
    addGeneratedFactorFromAPI(factor, cost) {
        const container = this.elements.generatedFactors;
        if (!container) return;

        const factorCard = document.createElement('div');
        factorCard.className = 'factor-card fade-enter';
        
        factorCard.innerHTML = `
            <div class="factor-header">
                <div class="factor-name">${factor.name}</div>
                <div class="factor-score">
                    <span class="score-label">AI评分</span>
                    <span class="score-value">${factor.ai_score}</span>
                </div>
            </div>
            <div class="factor-content">
                <div class="factor-formula">
                    <code>${factor.formula}</code>
                </div>
                <div class="factor-description">
                    <p>${factor.description}</p>
                </div>
                <div class="factor-metrics">
                    <div class="metric">
                        <span class="metric-label">预期IC:</span>
                        <span class="metric-value">${factor.estimated_ic}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">预期Sharpe:</span>
                        <span class="metric-value">${factor.estimated_sharpe}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">生成成本:</span>
                        <span class="metric-value">$${cost}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">AI引擎:</span>
                        <span class="metric-value">${factor.engine}</span>
                    </div>
                </div>
            </div>
            <div class="factor-actions">
                <button class="btn btn-secondary small" onclick="previewFactor('${factor.id}')">
                    <i class="fas fa-chart-line"></i>
                    预览
                </button>
                <button class="btn btn-secondary small" onclick="backtestFactor('${factor.id}')">
                    <i class="fas fa-play"></i>
                    回测
                </button>
                <button class="btn btn-primary small" onclick="saveFactorToLibrary('${factor.id}')">
                    <i class="fas fa-save"></i>
                    保存到因子库
                </button>
            </div>
        `;
        
        container.appendChild(factorCard);
        
        // 添加进入动画
        setTimeout(() => {
            factorCard.classList.add('fade-enter-active');
        }, 10);

        // 存储因子数据
        this.data.aiGeneratedFactors.set(factor.id, factor);
    }

    /**
     * 因子预览功能
     */
    async previewFactor(factorId) {
        try {
            console.log(`🔄 预览因子: ${factorId}`);
            
            const factor = this.data.aiGeneratedFactors.get(factorId);
            if (!factor) {
                this.showToast('因子数据不存在', 'error');
                return;
            }

            this.showToast(`正在生成因子 "${factor.name}" 的预览...`, 'info');
            
            // 这里可以调用API获取因子的历史表现数据并显示图表
            // 暂时显示一个模态框或切换到因子详情页面
            this.showFactorPreviewModal(factor);

        } catch (error) {
            console.error('预览因子失败:', error);
            this.showToast('预览因子失败', 'error');
        }
    }

    /**
     * 显示因子预览模态框
     */
    showFactorPreviewModal(factor) {
        // 这里实现一个简单的预览提示
        this.showToast(`因子预览: ${factor.name} (IC: ${factor.estimated_ic}, Sharpe: ${factor.estimated_sharpe})`, 'info', 5000);
    }

    /**
     * 因子回测功能
     */
    async backtestFactor(factorId) {
        try {
            console.log(`🔄 回测因子: ${factorId}`);
            
            const factor = this.data.aiGeneratedFactors.get(factorId);
            if (!factor) {
                this.showToast('因子数据不存在', 'error');
                return;
            }

            this.showToast(`正在启动因子 "${factor.name}" 的回测...`, 'info');
            
            // 调用回测API
            const backtestConfig = {
                factor_id: factorId,
                factor_formula: factor.formula,
                start_date: '2024-01-01',
                end_date: '2024-01-27',
                universe: 'crypto_top100',
                rebalance_frequency: 'daily'
            };

            const response = await this.makeApiRequest('/backtest/start', {
                method: 'POST',
                body: JSON.stringify(backtestConfig)
            });

            if (response.success) {
                this.showToast(`回测已启动，ID: ${response.data.backtest_id}`, 'success');
                
                // 切换到回测页面显示结果
                this.navigateToPage('backtest-center');
                
                // 监控回测进度
                this.monitorBacktestProgress(response.data.backtest_id);
            }

        } catch (error) {
            console.error('回测因子失败:', error);
            this.showToast('启动回测失败', 'error');
        }
    }

    /**
     * 监控回测进度
     */
    async monitorBacktestProgress(backtestId) {
        const progressInterval = setInterval(async () => {
            try {
                const response = await this.makeApiRequest(`/backtest/status/${backtestId}`);
                
                if (response.success) {
                    const status = response.data.status;
                    
                    if (status === 'completed') {
                        clearInterval(progressInterval);
                        this.showToast('回测完成！', 'success');
                        this.loadBacktestResults(backtestId);
                    } else if (status === 'failed') {
                        clearInterval(progressInterval);
                        this.showToast('回测失败', 'error');
                    } else {
                        console.log(`回测进度: ${status}`);
                    }
                }
            } catch (error) {
                console.error('获取回测状态失败:', error);
                clearInterval(progressInterval);
            }
        }, 2000);

        // 10分钟后停止监控
        setTimeout(() => {
            clearInterval(progressInterval);
        }, 600000);
    }

    /**
     * 加载回测结果
     */
    async loadBacktestResults(backtestId) {
        try {
            const response = await this.makeApiRequest('/backtest/results');
            
            if (response.success && response.data) {
                this.displayBacktestResults(response.data);
            }
        } catch (error) {
            console.error('加载回测结果失败:', error);
        }
    }

    /**
     * 显示回测结果
     */
    displayBacktestResults(results) {
        // 更新回测页面的图表和指标
        console.log('显示回测结果:', results);
        
        // 这里可以更新UI显示回测结果
        this.showToast(`回测完成: 收益率 ${(results.metrics.total_return * 100).toFixed(2)}%`, 'success', 8000);
    }

    /**
     * 保存因子到因子库
     */
    async saveFactorToLibrary(factorId) {
        try {
            console.log(`🔄 保存因子到库: ${factorId}`);
            
            const factor = this.data.aiGeneratedFactors.get(factorId);
            if (!factor) {
                this.showToast('因子数据不存在', 'error');
                return;
            }

            const saveData = {
                name: factor.name,
                formula: factor.formula,
                description: factor.description,
                category: '技术指标',
                ai_score: factor.ai_score,
                estimated_ic: factor.estimated_ic,
                estimated_sharpe: factor.estimated_sharpe,
                creation_source: 'ai',
                engine: factor.engine
            };

            const response = await this.makeApiRequest('/factors/create', {
                method: 'POST',
                body: JSON.stringify(saveData)
            });

            if (response.success) {
                this.showToast(`因子 "${factor.name}" 已保存到因子库`, 'success');
                
                // 将因子添加到本地因子库数据中
                this.data.factors.set(response.data.factor_id, {
                    id: response.data.factor_id,
                    ...saveData,
                    created: new Date().toISOString().split('T')[0]
                });

                // 移除生成的因子卡片或标记为已保存
                const factorCard = document.querySelector(`[onclick*="${factorId}"]`);
                if (factorCard && factorCard.closest('.factor-card')) {
                    const card = factorCard.closest('.factor-card');
                    card.style.opacity = '0.6';
                    card.querySelector('.btn-primary').textContent = '已保存';
                    card.querySelector('.btn-primary').disabled = true;
                }
            }

        } catch (error) {
            console.error('保存因子失败:', error);
            this.showToast('保存因子到库失败', 'error');
        }
    }

    /**
     * 启动回测
     */
    async startBacktest(strategyConfig) {
        try {
            console.log('🔄 启动策略回测:', strategyConfig);
            
            this.showToast('正在启动策略回测...', 'info');
            
            const response = await this.makeApiRequest('/backtest/start', {
                method: 'POST',
                body: JSON.stringify(strategyConfig)
            });

            if (response.success) {
                this.showToast(`策略回测已启动: ${response.data.backtest_id}`, 'success');
                this.monitorBacktestProgress(response.data.backtest_id);
            }

        } catch (error) {
            console.error('启动回测失败:', error);
            this.showToast('启动策略回测失败', 'error');
        }
    }

    /**
     * 加载因子库
     */
    async loadFactorLibrary() {
        try {
            console.log('🔄 从API加载因子库...');
            
            const response = await this.makeApiRequest('/factors/library');
            
            if (response.success && response.data) {
                const { factors, categories } = response.data;
                
                // 存储因子数据
                factors.forEach(factor => {
                    this.data.factors.set(factor.id, factor);
                });

                // 更新因子库UI
                this.updateFactorLibraryUI(factors, categories);

                console.log('✅ 因子库加载成功');
            }
        } catch (error) {
            console.warn('因子库API失败，使用模拟数据');
            this.loadFallbackFactorLibrary();
        }
    }

    /**
     * 更新因子库UI
     */
    updateFactorLibraryUI(factors, categories) {
        // 更新因子库页面的内容
        const factorLibraryContainer = document.getElementById('factor-library-container');
        if (!factorLibraryContainer) return;

        let html = `
            <div class="factor-library-header">
                <h3>因子库总览</h3>
                <div class="factor-stats">
                    <span>共 ${factors.length} 个因子</span>
                    <span>分类: ${Object.keys(categories).length} 种</span>
                </div>
            </div>
            <div class="factor-categories">
        `;

        // 按分类显示因子
        Object.entries(categories).forEach(([category, count]) => {
            html += `
                <div class="category-card">
                    <div class="category-name">${category}</div>
                    <div class="category-count">${count} 个因子</div>
                </div>
            `;
        });

        html += '</div><div class="factor-list">';

        // 显示因子列表
        factors.slice(0, 10).forEach(factor => {
            const statusClass = factor.creation_source === 'ai' ? 'status-ai' : 'status-manual';
            html += `
                <div class="factor-item">
                    <div class="factor-info">
                        <div class="factor-name">${factor.name}</div>
                        <div class="factor-meta">
                            <span class="factor-category">${factor.category}</span>
                            <span class="factor-source ${statusClass}">${factor.creation_source === 'ai' ? 'AI生成' : '手动创建'}</span>
                        </div>
                        <div class="factor-description">${factor.description}</div>
                    </div>
                    <div class="factor-metrics">
                        <div class="metric">
                            <span class="metric-label">IC:</span>
                            <span class="metric-value">${factor.ic}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">ICIR:</span>
                            <span class="metric-value">${factor.icir}</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">胜率:</span>
                            <span class="metric-value">${(factor.win_rate * 100).toFixed(1)}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Sharpe:</span>
                            <span class="metric-value">${factor.sharpe}</span>
                        </div>
                    </div>
                    <div class="factor-actions">
                        <button class="btn btn-small btn-secondary" onclick="viewFactorDetails('${factor.id}')">查看详情</button>
                        <button class="btn btn-small btn-primary" onclick="backtestFactor('${factor.id}')">回测</button>
                    </div>
                </div>
            `;
        });

        html += '</div>';
        factorLibraryContainer.innerHTML = html;
    }

    /**
     * 备用因子库数据
     */
    loadFallbackFactorLibrary() {
        const factors = [
            {
                id: 'factor_001',
                name: 'RSI动量因子',
                category: '技术指标',
                ic: 0.156,
                icir: 1.23,
                win_rate: 0.642,
                sharpe: 1.85,
                creation_source: 'ai',
                description: '基于RSI指标构建的动量因子'
            },
            {
                id: 'factor_002',
                name: '成交量价格背离因子',
                category: '成交量',
                ic: 0.132,
                icir: 0.98,
                win_rate: 0.617,
                sharpe: 1.45,
                creation_source: 'manual',
                description: '检测价格与成交量背离的反转信号因子'
            }
        ];

        factors.forEach(factor => {
            this.data.factors.set(factor.id, factor);
        });
    }

    /**
     * 加载数据源状态
     */
    async loadDataSources() {
        try {
            console.log('🔄 从API加载数据源状态...');
            
            const response = await this.makeApiRequest('/data/sources');
            
            if (response.success && response.data) {
                const { sources } = response.data;
                
                // 更新数据源状态UI
                this.updateDataSourcesUI(sources);

                console.log('✅ 数据源状态加载成功');
            }
        } catch (error) {
            console.warn('数据源状态API失败，使用模拟数据');
            this.loadFallbackDataSources();
        }
    }

    /**
     * 更新数据源状态UI
     */
    updateDataSourcesUI(sources) {
        const dataSourcesContainer = document.getElementById('data-sources-status');
        if (!dataSourcesContainer) return;

        let html = '';
        sources.forEach(source => {
            const statusClass = source.status === 'online' ? 'status-online' : 
                               source.status === 'warning' ? 'status-warning' : 'status-offline';
            
            html += `
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
                    <span>${source.name}</span>
                    <span class="status-indicator ${statusClass}">
                        <i class="fas fa-circle"></i>
                        ${source.status === 'online' ? '在线' : source.status === 'warning' ? '延迟' : '离线'}
                    </span>
                </div>
                <div style="font-size: 0.875rem; color: var(--text-muted); margin-bottom: 1rem;">
                    延迟: ${source.latency} | 日交易量: ${source.daily_volume} | 更新: ${source.last_update}
                </div>
            `;
        });

        dataSourcesContainer.innerHTML = html;
    }

    /**
     * 备用数据源数据
     */
    loadFallbackDataSources() {
        // 使用现有的静态数据源状态
        console.log('使用备用数据源状态');
    }

    /**
     * 生成分析报告
     */
    async generateAnalysisReport(reportConfig) {
        try {
            console.log('🔄 生成分析报告:', reportConfig);
            
            this.showToast('正在生成分析报告...', 'info');
            
            const response = await this.makeApiRequest('/reports/generate', {
                method: 'POST',
                body: JSON.stringify(reportConfig)
            });

            if (response.success) {
                this.showToast(`报告生成已启动: ${response.data.report_id}`, 'success');
                
                // 监控报告生成进度
                this.monitorReportProgress(response.data.report_id);
            }

        } catch (error) {
            console.error('生成报告失败:', error);
            this.showToast('生成分析报告失败', 'error');
        }
    }

    /**
     * 监控报告生成进度
     */
    async monitorReportProgress(reportId) {
        const progressInterval = setInterval(async () => {
            try {
                const response = await this.makeApiRequest(`/reports/status/${reportId}`);
                
                if (response.success) {
                    const status = response.data.status;
                    
                    if (status === 'completed') {
                        clearInterval(progressInterval);
                        this.showToast('报告生成完成！', 'success');
                        this.loadReportsList();
                    } else if (status === 'failed') {
                        clearInterval(progressInterval);
                        this.showToast('报告生成失败', 'error');
                    } else {
                        console.log(`报告生成进度: ${status}`);
                    }
                }
            } catch (error) {
                console.error('获取报告状态失败:', error);
                clearInterval(progressInterval);
            }
        }, 3000);

        // 10分钟后停止监控
        setTimeout(() => {
            clearInterval(progressInterval);
        }, 600000);
    }

    /**
     * 加载报告列表
     */
    async loadReportsList() {
        try {
            console.log('🔄 加载报告列表...');
            
            const response = await this.makeApiRequest('/reports');
            
            if (response.success && response.data) {
                const { reports } = response.data;
                
                // 更新报告列表UI
                this.updateReportsListUI(reports);

                console.log('✅ 报告列表加载成功');
            }
        } catch (error) {
            console.warn('报告列表API失败，使用模拟数据');
            this.loadFallbackReportsList();
        }
    }

    /**
     * 更新报告列表UI
     */
    updateReportsListUI(reports) {
        const reportsContainer = document.getElementById('reports-list-container');
        if (!reportsContainer) return;

        let html = '<div class="reports-list">';

        reports.forEach(report => {
            const statusClass = report.status === 'completed' ? 'status-online' : 'status-warning';
            html += `
                <div class="report-item">
                    <div class="report-info">
                        <div class="report-title">${report.title}</div>
                        <div class="report-meta">
                            <span class="report-type">${report.type}</span>
                            <span class="report-date">${report.created_date}</span>
                            <span class="report-status ${statusClass}">${report.status}</span>
                        </div>
                        <div class="report-summary">${report.summary}</div>
                    </div>
                    <div class="report-actions">
                        <span class="report-size">${report.file_size}</span>
                        <button class="btn btn-small btn-primary" onclick="downloadReport('${report.id}')">下载</button>
                        <button class="btn btn-small btn-secondary" onclick="viewReport('${report.id}')">预览</button>
                    </div>
                </div>
            `;
        });

        html += '</div>';
        reportsContainer.innerHTML = html;
    }

    /**
     * 备用报告列表
     */
    loadFallbackReportsList() {
        const reports = [
            {
                id: 'report_001',
                title: '技术指标因子分析报告',
                type: 'factor_analysis',
                status: 'completed',
                created_date: '2024-01-27',
                summary: '分析了156个技术指标因子的有效性',
                file_size: '2.3MB'
            }
        ];

        this.updateReportsListUI(reports);
    }

    /**
     * 保存系统配置
     */
    async saveSystemConfig(config) {
        try {
            console.log('🔄 保存系统配置:', config);
            
            this.showToast('正在保存系统配置...', 'info');
            
            const response = await this.makeApiRequest('/system/config', {
                method: 'POST',
                body: JSON.stringify(config)
            });

            if (response.success) {
                this.showToast('系统配置保存成功', 'success');
                
                // 更新本地配置
                Object.assign(this.config, config);
            }

        } catch (error) {
            console.error('保存配置失败:', error);
            this.showToast('保存系统配置失败', 'error');
        }
    }

    /**
     * 加载系统配置
     */
    async loadSystemConfig() {
        try {
            console.log('🔄 加载系统配置...');
            
            const response = await this.makeApiRequest('/system/config');
            
            if (response.success && response.data) {
                // 更新本地配置
                Object.assign(this.config, response.data);
                
                // 更新配置UI
                this.updateConfigUI(response.data);

                console.log('✅ 系统配置加载成功');
            }
        } catch (error) {
            console.warn('配置加载API失败，使用默认配置');
        }
    }

    /**
     * 更新配置UI
     */
    updateConfigUI(config) {
        // 更新各种配置输入框的值
        const configElements = {
            'api-url': config.apiUrl,
            'ws-url': config.wsUrl,
            'update-interval': config.updateInterval,
            'max-data-points': config.maxDataPoints,
            'ai-cost-limit': config.aiCostLimit,
            'data-retention-days': config.dataRetentionDays
        };

        Object.entries(configElements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.value = value;
            }
        });
    }

    /**
     * 显示生成进度
     */
    showGeneratingProgress() {
        const container = this.elements.generatedFactors;
        if (!container) return;

        const generatingCard = document.createElement('div');
        generatingCard.className = 'factor-card generating';
        generatingCard.innerHTML = `
            <div style="text-align: center; padding: 2rem;">
                <div class="loading-spinner" style="margin: 0 auto 1rem; width: 40px; height: 40px;"></div>
                <div style="color: var(--text-muted); font-size: 0.95rem;">AI正在分析数据并生成因子...</div>
                <div style="color: var(--text-muted); font-size: 0.875rem; margin-top: 0.5rem;">预计需要 15-30 秒</div>
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
            }, Math.random() * 15000 + 10000); // 10-25秒随机时间
        });
    }

    /**
     * 添加生成的因子
     */
    addGeneratedFactor() {
        const container = this.elements.generatedFactors;
        if (!container) return;

        const factorCard = document.createElement('div');
        factorCard.className = 'factor-card';
        factorCard.style.opacity = '0';
        factorCard.style.transform = 'translateY(20px)';
        
        const factorNames = [
            '动量反转因子', '波动率突破因子', '成交量异动因子', 
            '情绪反转因子', '价格偏离因子', '资金流向因子'
        ];
        const randomName = factorNames[Math.floor(Math.random() * factorNames.length)];
        const randomScore = (Math.random() * 2 + 7).toFixed(1);
        const factorId = Date.now().toString().slice(-4);
        
        factorCard.innerHTML = `
            <div class="factor-header">
                <div class="factor-name">${randomName} #${factorId}</div>
                <div class="factor-score">${randomScore}/10</div>
            </div>
            <div style="margin-bottom: 1rem;">
                <div style="background: #f8fafc; padding: 1rem; border-radius: 0.5rem; font-family: 'Monaco', 'Consolas', monospace; font-size: 0.875rem; color: #2d3748; margin-bottom: 0.75rem;">
                    // AI生成的因子公式<br>
                    factor = tanh((close / sma(close, 20) - 1) * atr(20))<br>
                    signal = sign(factor) * pow(abs(factor), 0.7)
                </div>
                <div style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.5;">
                    基于价格动量和波动率特征的多层次因子，结合了技术分析和统计学方法，适用于中短期趋势预测。
                </div>
            </div>
            <div style="display: flex; gap: 0.5rem; flex-wrap: wrap;">
                <button class="btn btn-secondary" style="padding: 0.5rem 1rem; font-size: 0.875rem;">
                    <i class="fas fa-chart-line"></i>
                    预览
                </button>
                <button class="btn btn-secondary" style="padding: 0.5rem 1rem; font-size: 0.875rem;">
                    <i class="fas fa-play"></i>
                    回测
                </button>
                <button class="btn btn-primary" style="padding: 0.5rem 1rem; font-size: 0.875rem;">
                    <i class="fas fa-save"></i>
                    保存
                </button>
            </div>
        `;
        
        container.appendChild(factorCard);
        
        // 添加进入动画
        setTimeout(() => {
            factorCard.style.transition = 'all 0.3s ease';
            factorCard.style.opacity = '1';
            factorCard.style.transform = 'translateY(0)';
        }, 100);
    }

    /**
     * 生成时间标签
     */
    generateTimeLabels(count) {
        const labels = [];
        const now = new Date();
        for (let i = count - 1; i >= 0; i--) {
            const time = new Date(now - i * 60000); // 每分钟一个点
            labels.push(time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
        }
        return labels;
    }

    /**
     * 生成随机数据
     */
    generateRandomData(count, min, max) {
        const data = [];
        let lastValue = (min + max) / 2;
        
        for (let i = 0; i < count; i++) {
            const change = (Math.random() - 0.5) * (max - min) * 0.1;
            lastValue += change;
            lastValue = Math.max(min, Math.min(max, lastValue));
            data.push(parseFloat(lastValue.toFixed(2)));
        }
        
        return data;
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
        
        const iconMap = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };
        
        toast.innerHTML = `
            <div style="display: flex; align-items: center; gap: 0.75rem;">
                <span style="font-size: 1.2rem;">${iconMap[type] || 'ℹ️'}</span>
                <span style="flex: 1;">${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" style="background: none; border: none; color: var(--text-muted); cursor: pointer; font-size: 1.2rem;">×</button>
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
        this.showToast('系统初始化失败，启用模拟模式', 'warning', 5000);
        
        // 启用模拟模式
        this.enableSimulationMode();
    }

    /**
     * 启用模拟模式
     */
    enableSimulationMode() {
        console.log('🎭 启用模拟模式');
        
        // 模拟数据更新
        this.startDataUpdates();
        
        // 标记为已初始化
        this.state.isInitialized = true;
        
        this.showToast('已启用模拟模式', 'info');
    }

    /**
     * 打开移动端菜单
     */
    openMobileMenu() {
        const sidebar = document.getElementById('sidebar');
        const overlay = document.querySelector('.mobile-overlay');
        
        if (sidebar && overlay) {
            sidebar.classList.add('open');
            overlay.classList.add('active');
        }
    }

    /**
     * 关闭移动端菜单
     */
    closeMobileMenu() {
        const sidebar = document.getElementById('sidebar');
        const overlay = document.querySelector('.mobile-overlay');
        
        if (sidebar && overlay) {
            sidebar.classList.remove('open');
            overlay.classList.remove('active');
        }
    }

    /**
     * 切换移动端菜单
     */
    toggleMobileMenu() {
        const sidebar = document.getElementById('sidebar');
        
        if (sidebar && sidebar.classList.contains('open')) {
            this.closeMobileMenu();
        } else {
            this.openMobileMenu();
        }
    }

    /**
     * 设置移动端触摸优化
     */
    setupTouchOptimization() {
        // 检测移动设备
        const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        
        if (isMobile) {
            // 禁用iOS safari的bounce效果
            document.body.addEventListener('touchstart', (e) => {
                if (e.target === document.body) {
                    e.preventDefault();
                }
            });

            // 添加触摸友好的样式类
            document.body.classList.add('mobile-device');

            // 优化触摸延迟
            document.addEventListener('touchstart', () => {}, true);
        }

        // 处理双指缩放
        document.addEventListener('touchmove', (e) => {
            if (e.touches.length > 1) {
                e.preventDefault();
            }
        }, { passive: false });

        console.log('✓ 移动端触摸优化已设置');
    }

    /**
     * 导航页面优化版
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
            
            // 添加页面切换动画
            setTimeout(() => {
                targetPage.classList.add('active');
                this.state.currentPage = pageName;
                
                // 页面特殊处理
                this.handlePageSwitch(pageName);
            }, 50);

            // 在移动端自动关闭侧边栏
            if (window.innerWidth <= 1024) {
                this.closeMobileMenu();
            }
        }
    }

    /**
     * 窗口大小调整处理 - 增强版
     */
    handleResize() {
        // 重新调整图表大小
        this.charts.forEach(chart => {
            if (chart.resize) {
                chart.resize();
            }
        });

        // 移动端适配处理
        if (window.innerWidth > 1024) {
            this.closeMobileMenu();
            const sidebar = document.getElementById('sidebar');
            if (sidebar) {
                sidebar.classList.remove('open');
            }
        }

        // 更新视口高度变量（用于移动端地址栏问题）
        document.documentElement.style.setProperty('--vh', `${window.innerHeight * 0.01}px`);
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
        
        console.log('🧹 现代化应用资源已清理');
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
window.modernDataApp = new ModernDataAnalysisApp();

// 绑定全局函数
window.startAIGeneration = () => {
    window.modernDataApp.startAIGeneration();
};

window.navigateToPage = (page) => {
    window.modernDataApp.navigateToPage(page);
};

window.startBacktest = () => {
    window.modernDataApp.startBacktest();
};

window.exportBacktestReport = () => {
    window.modernDataApp.showToast('📊 报告导出功能正在开发中...', 'info');
};

window.saveStrategy = () => {
    window.modernDataApp.showToast('💾 策略已保存到因子库', 'success');
};

window.useFactor = (factorId) => {
    window.modernDataApp.useFactor(factorId);
};

window.addNewFactor = () => {
    window.modernDataApp.addNewFactor();
};

window.refreshDataSources = () => {
    window.modernDataApp.refreshDataSources();
};

window.addDataSource = () => {
    window.modernDataApp.addDataSource();
};

window.generateReport = () => {
    window.modernDataApp.generateReport();
};

window.refreshReports = () => {
    window.modernDataApp.refreshReports();
};

window.createTemplate = () => {
    window.modernDataApp.createTemplate();
};

// 开发者工具
if (typeof window !== 'undefined' && window.location.hostname === 'localhost') {
    window.debugModernApp = () => {
        console.log('🎨 现代化应用状态:', window.modernDataApp.state);
        console.log('📊 数据存储:', window.modernDataApp.data);
        console.log('📈 图表实例:', window.modernDataApp.charts);
    };
    
    console.log('%c🎨 智能因子实验室', 'color: #667eea; font-size: 24px; font-weight: bold;');
    console.log('%c现代化设计 | AI驱动 | 开发模式已启用', 'color: #718096; font-size: 14px;');
    console.log('%c使用 debugModernApp() 查看应用状态', 'color: #4a5568; font-size: 12px;');
}