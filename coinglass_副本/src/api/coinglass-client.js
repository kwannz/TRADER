const axios = require('axios');
const { EventEmitter } = require('events');

class CoinGlassClient extends EventEmitter {
  constructor(apiKey, options = {}) {
    super();
    
    this.apiKey = apiKey;
    this.baseURL = 'https://open-api-v4.coinglass.com';
    this.requestsPerSecond = options.requestsPerSecond || 10; // 默认每秒10个请求
    this.retryAttempts = options.retryAttempts || 3;
    this.retryDelay = options.retryDelay || 1000; // 1秒
    
    // 创建axios实例
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: options.timeout || 30000,
      headers: {
        'accept': 'application/json',
        'CG-API-KEY': this.apiKey
      }
    });

    // 请求拦截器 - 添加请求限流
    this.setupRequestInterceptor();
    
    // 响应拦截器 - 处理错误和重试
    this.setupResponseInterceptor();
    
    // 初始化请求队列
    this.requestQueue = [];
    this.isProcessingQueue = false;
    this.lastRequestTime = 0;
  }

  setupRequestInterceptor() {
    this.client.interceptors.request.use(
      (config) => {
        config.metadata = { startTime: new Date() };
        this.emit('request', { url: config.url, method: config.method });
        return config;
      },
      (error) => {
        this.emit('requestError', error);
        return Promise.reject(error);
      }
    );
  }

  setupResponseInterceptor() {
    this.client.interceptors.response.use(
      (response) => {
        const endTime = new Date();
        const duration = endTime - response.config.metadata.startTime;
        
        this.emit('response', {
          url: response.config.url,
          status: response.status,
          duration: duration
        });

        // 检查API响应码
        if (response.data && response.data.code !== "0") {
          const error = new Error(`API Error: ${response.data.msg || 'Unknown error'}`);
          error.code = response.data.code;
          error.apiMessage = response.data.msg;
          throw error;
        }

        return response;
      },
      (error) => {
        this.emit('responseError', error);
        return Promise.reject(error);
      }
    );
  }

  async makeRequest(endpoint, params = {}, options = {}) {
    return new Promise((resolve, reject) => {
      this.requestQueue.push({
        endpoint,
        params,
        options,
        resolve,
        reject,
        attempts: 0
      });

      if (!this.isProcessingQueue) {
        this.processQueue();
      }
    });
  }

  async processQueue() {
    if (this.requestQueue.length === 0) {
      this.isProcessingQueue = false;
      return;
    }

    this.isProcessingQueue = true;
    const request = this.requestQueue.shift();

    try {
      // 实现速率限制
      const now = Date.now();
      const timeSinceLastRequest = now - this.lastRequestTime;
      const minInterval = 1000 / this.requestsPerSecond;

      if (timeSinceLastRequest < minInterval) {
        await new Promise(resolve => setTimeout(resolve, minInterval - timeSinceLastRequest));
      }

      this.lastRequestTime = Date.now();

      // 发送请求
      const response = await this.client.get(request.endpoint, {
        params: request.params,
        ...request.options
      });

      request.resolve(response.data);

    } catch (error) {
      // 重试逻辑
      if (request.attempts < this.retryAttempts) {
        request.attempts++;
        
        // 将请求重新放入队列
        this.requestQueue.unshift(request);
        
        // 等待后重试
        await new Promise(resolve => setTimeout(resolve, this.retryDelay * request.attempts));
        
      } else {
        request.reject(error);
      }
    }

    // 继续处理下一个请求
    setTimeout(() => this.processQueue(), 0);
  }

  // ===================
  // 基础信息API
  // ===================

  /**
   * 获取支持的币种列表
   */
  async getSupportedCoins() {
    return this.makeRequest('/api/futures/supported-coins');
  }

  /**
   * 获取支持的交易所和交易对
   */
  async getSupportedExchangePairs() {
    return this.makeRequest('/api/futures/supported-exchange-pairs');
  }

  // ===================
  // 合约市场数据API (需要升级计划)
  // ===================

  /**
   * 获取合约市场数据 (需要升级计划)
   */
  async getContractMarkets(symbol = 'all') {
    try {
      return await this.makeRequest('/api/futures/coins-markets', { symbol });
    } catch (error) {
      if (error.apiMessage === 'Upgrade plan') {
        console.warn('合约市场数据需要升级API计划');
        return { data: [], message: '需要升级API计划' };
      }
      throw error;
    }
  }

  /**
   * 获取交易对市场数据
   */
  async getPairMarkets(symbol) {
    if (!symbol) {
      throw new Error('symbol参数是必需的');
    }
    return this.makeRequest('/api/futures/pairs-markets', { symbol });
  }

  // ===================
  // 持仓数据API (目前不可用)
  // ===================

  /**
   * 获取持仓数据 (目前端点不可用)
   */
  async getOpenInterest(params = {}) {
    console.warn('持仓数据API端点当前不可用');
    return { data: [], message: 'API端点不可用' };
  }

  /**
   * 获取持仓历史数据 (目前有服务器错误)
   */
  async getOpenInterestHistory(params = {}) {
    console.warn('持仓历史数据API当前有服务器错误');
    return { data: [], message: 'API服务器错误' };
  }

  // ===================
  // 资金费率API (目前不可用)
  // ===================

  /**
   * 获取资金费率数据 (目前端点不可用)
   */
  async getFundingRates(params = {}) {
    console.warn('资金费率API端点当前不可用');
    return { data: [], message: 'API端点不可用' };
  }

  /**
   * 获取资金费率历史数据 (目前端点不可用)
   */
  async getFundingRatesHistory(params = {}) {
    console.warn('资金费率历史API端点当前不可用');
    return { data: [], message: 'API端点不可用' };
  }

  // ===================
  // 多空比API (目前不可用)
  // ===================

  /**
   * 获取多空比数据 (目前端点不可用)
   */
  async getLongShortRatio(params = {}) {
    console.warn('多空比API端点当前不可用');
    return { data: [], message: 'API端点不可用' };
  }

  /**
   * 获取账户多空比 (目前端点不可用)
   */
  async getLongShortAccounts(params = {}) {
    console.warn('账户多空比API端点当前不可用');
    return { data: [], message: 'API端点不可用' };
  }

  /**
   * 获取大户持仓多空比 (目前端点不可用)
   */
  async getTopPositionsLongShort(params = {}) {
    console.warn('大户持仓多空比API端点当前不可用');
    return { data: [], message: 'API端点不可用' };
  }

  // ===================
  // 爆仓数据API (目前不可用)
  // ===================

  /**
   * 获取币种爆仓数据 (目前端点不可用)
   */
  async getLiquidationByCoin(params = {}) {
    console.warn('币种爆仓数据API端点当前不可用');
    return { data: [], message: 'API端点不可用' };
  }

  /**
   * 获取交易所爆仓数据 (目前端点不可用)
   */
  async getLiquidationByExchange(params = {}) {
    console.warn('交易所爆仓数据API端点当前不可用');
    return { data: [], message: 'API端点不可用' };
  }

  // ===================
  // ETF数据API (目前不可用)
  // ===================

  /**
   * 获取BTC ETF净流入数据 (目前端点不可用)
   */
  async getBTCETFNetflow() {
    console.warn('BTC ETF数据API端点当前不可用');
    return { data: [], message: 'API端点不可用' };
  }

  /**
   * 获取ETH ETF净流入数据 (目前端点不可用)
   */
  async getETHETFNetflow() {
    console.warn('ETH ETF数据API端点当前不可用');
    return { data: [], message: 'API端点不可用' };
  }

  // ===================
  // 技术指标API (目前不可用)
  // ===================

  /**
   * 获取AHR999指标 (目前端点不可用)
   */
  async getAHR999() {
    console.warn('AHR999指标API端点当前不可用');
    return { data: [], message: 'API端点不可用' };
  }

  /**
   * 获取彩虹图指标 (目前端点不可用)
   */
  async getRainbowChart() {
    console.warn('彩虹图指标API端点当前不可用');
    return { data: [], message: 'API端点不可用' };
  }

  /**
   * 获取恐惧贪婪指数 (目前端点不可用)
   */
  async getFearGreedIndex() {
    console.warn('恐惧贪婪指数API端点当前不可用');
    return { data: [], message: 'API端点不可用' };
  }

  // ===================
  // 工具方法
  // ===================

  /**
   * 测试API连接
   */
  async testConnection() {
    try {
      const result = await this.getSupportedCoins();
      return {
        success: true,
        message: '连接成功',
        coinCount: result.data ? result.data.length : 0
      };
    } catch (error) {
      return {
        success: false,
        message: error.message,
        error: error
      };
    }
  }

  /**
   * 获取API状态信息
   */
  getStatus() {
    return {
      baseURL: this.baseURL,
      requestsPerSecond: this.requestsPerSecond,
      queueLength: this.requestQueue.length,
      isProcessingQueue: this.isProcessingQueue
    };
  }
}

module.exports = CoinGlassClient; 