import axios from 'axios';
import { config } from '../config/index.js';
import logger from '../utils/logger.js';
const { log } = logger;

class CoinGlassAPI {
  constructor() {
    this.baseURL = config.api.baseUrl;
    this.apiKey = config.api.key;
    this.timeout = config.api.timeout;
    this.maxRetries = config.api.maxRetries;
    this.retryDelay = config.api.retryDelay;
    
    // 创建axios实例
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: this.timeout,
      headers: {
        'CG-API-KEY': this.apiKey,
        'Content-Type': 'application/json',
        'User-Agent': 'CoinGlass-Data-Collector/1.0'
      }
    });

    // 请求拦截器
    this.client.interceptors.request.use(
      (config) => {
        config.startTime = Date.now();
        log.debug(`发起API请求: ${config.method?.toUpperCase()} ${config.url}`, {
          params: config.params,
          data: config.data
        });
        return config;
      },
      (error) => {
        log.error('请求拦截器错误', error);
        return Promise.reject(error);
      }
    );

    // 响应拦截器
    this.client.interceptors.response.use(
      (response) => {
        const duration = Date.now() - response.config.startTime;
        log.apiRequest(
          response.config.method?.toUpperCase(),
          response.config.url,
          response.status,
          duration,
          { dataSize: JSON.stringify(response.data).length }
        );
        return response;
      },
      (error) => {
        const duration = error.config ? Date.now() - error.config.startTime : 0;
        log.error('API请求失败', {
          method: error.config?.method?.toUpperCase(),
          url: error.config?.url,
          error: error.message,
          duration,
          status: error.response?.status
        });
        return Promise.reject(error);
      }
    );

    // 请求限流器
    this.requestQueue = [];
    this.requestTimes = [];
    this.isProcessing = false;
  }

  // 请求限流控制
  async rateLimit() {
    const now = Date.now();
    const { requests, period } = config.api.rateLimit;
    
    // 清理过期的请求时间
    this.requestTimes = this.requestTimes.filter(time => now - time < period);
    
    // 如果超过限制，等待
    if (this.requestTimes.length >= requests) {
      const oldestRequest = Math.min(...this.requestTimes);
      const waitTime = period - (now - oldestRequest);
      log.warn(`请求限流，等待 ${waitTime}ms`);
      await new Promise(resolve => setTimeout(resolve, waitTime));
    }
    
    this.requestTimes.push(now);
  }

  // 带重试的请求方法
  async request(config, retryCount = 0) {
    try {
      await this.rateLimit();
      const response = await this.client(config);
      
      // 检查API响应状态
      if (response.data && response.data.code !== 0) {
        throw new Error(`API响应错误: ${response.data.msg || 'Unknown error'}`);
      }
      
      return response.data;
    } catch (error) {
      if (retryCount < this.maxRetries) {
        const delay = this.retryDelay * Math.pow(2, retryCount); // 指数退避
        log.warn(`请求失败，${delay}ms后重试 (${retryCount + 1}/${this.maxRetries})`, {
          url: config.url,
          error: error.message
        });
        
        await new Promise(resolve => setTimeout(resolve, delay));
        return this.request(config, retryCount + 1);
      } else {
        log.error('请求最终失败，已达到最大重试次数', error, {
          url: config.url,
          retryCount
        });
        throw error;
      }
    }
  }

  // GET请求
  async get(endpoint, params = {}) {
    return this.request({
      method: 'GET',
      url: endpoint,
      params
    });
  }

  // POST请求
  async post(endpoint, data = {}) {
    return this.request({
      method: 'POST',
      url: endpoint,
      data
    });
  }

  // ===== 合约数据API =====
  
  // 获取支持的币种
  async getSupportedCoins() {
    return this.get(config.endpoints.contract.supportedCoins);
  }

  // 获取支持的交易所
  async getSupportedExchanges() {
    return this.get(config.endpoints.contract.supportedExchanges);
  }

  // 获取币种市场数据
  async getCoinMarkets(params = {}) {
    return this.get(config.endpoints.contract.coinMarkets, params);
  }

  // 获取交易对市场数据
  async getPairMarkets(params = {}) {
    return this.get(config.endpoints.contract.pairMarkets, params);
  }

  // 获取币种涨跌幅
  async getCoinChange(params = {}) {
    return this.get(config.endpoints.contract.coinChange, params);
  }

  // 获取K线历史数据
  async getKlineHistory(params = {}) {
    return this.get(config.endpoints.contract.klineHistory, params);
  }

  // 获取持仓历史数据
  async getOpenInterestHistory(params = {}) {
    return this.get(config.endpoints.contract.openInterestHistory, params);
  }

  // 获取聚合持仓数据
  async getAggregateOpenInterest(params = {}) {
    return this.get(config.endpoints.contract.aggregateOpenInterest, params);
  }

  // 获取资金费率历史
  async getFundingRateHistory(params = {}) {
    return this.get(config.endpoints.contract.fundingRateHistory, params);
  }

  // 获取币种资金费率
  async getCoinFundingRates(params = {}) {
    return this.get(config.endpoints.contract.coinFundingRates, params);
  }

  // 获取多空比数据
  async getAccountLongShortRatio(params = {}) {
    return this.get(config.endpoints.contract.accountLongShortRatio, params);
  }

  // 获取大户多空比
  async getBigTraderLongShortRatio(params = {}) {
    return this.get(config.endpoints.contract.bigTraderLongShortRatio, params);
  }

  // 获取持仓多空比
  async getPositionLongShortRatio(params = {}) {
    return this.get(config.endpoints.contract.positionLongShortRatio, params);
  }

  // 获取爆仓历史数据
  async getPairLiquidationHistory(params = {}) {
    return this.get(config.endpoints.contract.pairLiquidationHistory, params);
  }

  // 获取币种爆仓历史
  async getCoinLiquidationHistory(params = {}) {
    return this.get(config.endpoints.contract.coinLiquidationHistory, params);
  }

  // 获取爆仓订单
  async getLiquidationOrders(params = {}) {
    return this.get(config.endpoints.contract.liquidationOrders, params);
  }

  // 获取爆仓热力图
  async getPairLiquidationHeatmap(params = {}) {
    return this.get(config.endpoints.contract.pairLiquidationHeatmap, params);
  }

  // 获取币种爆仓热力图
  async getCoinLiquidationHeatmap(params = {}) {
    return this.get(config.endpoints.contract.coinLiquidationHeatmap, params);
  }

  // 获取订单簿历史
  async getOrderbookHistory(params = {}) {
    return this.get(config.endpoints.contract.orderbookHistory, params);
  }

  // 获取聚合订单簿
  async getAggregateOrderbook(params = {}) {
    return this.get(config.endpoints.contract.aggregateOrderbook, params);
  }

  // 获取大额订单簿
  async getLargeOrderbook(params = {}) {
    return this.get(config.endpoints.contract.largeOrderbook, params);
  }

  // 获取主动买卖数据
  async getPairActiveTrade(params = {}) {
    return this.get(config.endpoints.contract.pairActiveTrade, params);
  }

  // 获取币种主动买卖
  async getCoinActiveTrade(params = {}) {
    return this.get(config.endpoints.contract.coinActiveTrade, params);
  }

  // ===== 现货数据API =====
  
  // 获取现货支持币种
  async getSpotSupportedCoins() {
    return this.get(config.endpoints.spot.supportedCoins);
  }

  // 获取现货市场数据
  async getSpotCoinMarkets(params = {}) {
    return this.get(config.endpoints.spot.coinMarkets, params);
  }

  // 获取现货价格历史
  async getSpotPriceHistory(params = {}) {
    return this.get(config.endpoints.spot.priceHistory, params);
  }

  // ===== 期权数据API =====
  
  // 获取期权最大痛点
  async getOptionMaxPain(params = {}) {
    return this.get(config.endpoints.option.maxPain, params);
  }

  // 获取期权数据
  async getOptionData(params = {}) {
    return this.get(config.endpoints.option.optionData, params);
  }

  // ===== ETF数据API =====
  
  // 获取比特币ETF列表
  async getBitcoinEtfList() {
    return this.get(config.endpoints.etf.bitcoinEtfList);
  }

  // 获取ETF流入流出
  async getEtfFlow(params = {}) {
    return this.get(config.endpoints.etf.etfFlow, params);
  }

  // 获取ETF净资产
  async getEtfNetAssets(params = {}) {
    return this.get(config.endpoints.etf.etfNetAssets, params);
  }

  // 获取以太坊ETF列表
  async getEthereumEtfList() {
    return this.get(config.endpoints.etf.ethereumEtfList);
  }

  // ===== 链上数据API =====
  
  // 获取交易所余额
  async getExchangeBalance(params = {}) {
    return this.get(config.endpoints.onchain.exchangeBalance, params);
  }

  // 获取ERC20转账
  async getErc20Transfers(params = {}) {
    return this.get(config.endpoints.onchain.erc20Transfers, params);
  }

  // ===== 指标数据API =====
  
  // 获取恐惧贪婪指数
  async getFearGreedIndex(params = {}) {
    return this.get(config.endpoints.indicator.fearGreedIndex, params);
  }

  // 获取RSI列表
  async getRsiList(params = {}) {
    return this.get(config.endpoints.indicator.rsiList, params);
  }

  // 获取稳定币市值
  async getStablecoinMarketCap(params = {}) {
    return this.get(config.endpoints.indicator.stablecoinMarketCap, params);
  }

  // 获取比特币AHR999指标
  async getAhr999(params = {}) {
    return this.get(config.endpoints.indicator.ahr999, params);
  }

  // 获取彩虹图指标
  async getRainbowChart(params = {}) {
    return this.get(config.endpoints.indicator.rainbowChart, params);
  }

  // ===== 批量数据获取 =====
  
  // 批量获取多个端点数据
  async getBatchData(endpoints, params = {}) {
    const requests = endpoints.map(endpoint => 
      this.get(endpoint, params[endpoint] || {})
    );
    
    try {
      const results = await Promise.allSettled(requests);
      return results.map((result, index) => ({
        endpoint: endpoints[index],
        success: result.status === 'fulfilled',
        data: result.status === 'fulfilled' ? result.value : null,
        error: result.status === 'rejected' ? result.reason.message : null
      }));
    } catch (error) {
      log.error('批量请求失败', error);
      throw error;
    }
  }
}

// 创建单例实例
export const coinglassApi = new CoinGlassAPI();
export default coinglassApi; 