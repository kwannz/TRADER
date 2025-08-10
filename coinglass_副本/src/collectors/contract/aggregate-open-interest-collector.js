import { BaseCollector } from '../base-collector.js';
import { coinglassApi } from '../../services/coinglassApi.js';
import logger from "../../utils/logger.js";
const { log } = logger;

export class AggregateOpenInterestCollector extends BaseCollector {
  constructor() {
    super('aggregate-open-interest-collector', 'aggregated_open_interest');
    this.interval = 5; // 5分钟收集一次
    this.priority = 'high';
    this.supportedSymbols = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOGE', 'MATIC', 'DOT', 'LTC', 'LINK', 'UNI', 'ARB', 'OP'];
  }

  async collectData() {
    const allData = [];
    const timestamp = new Date();

    log.info(`${this.name}: 开始收集聚合持仓数据`);

    try {
      // 收集聚合持仓数据
      for (const symbol of this.supportedSymbols) {
        try {
          log.debug(`收集 ${symbol} 聚合持仓数据`);
          
          const aggregateOI = await coinglassApi.getAggregateOpenInterest({
            symbol: symbol.toLowerCase()
          });

          if (aggregateOI?.data) {
            const aggregatedData = this.transformAggregateData(aggregateOI.data, symbol, timestamp);
            if (aggregatedData) {
              allData.push(aggregatedData);
            }
          }

          // 添加小延迟避免API限制
          await this.delay(200);

        } catch (error) {
          log.warn(`收集 ${symbol} 聚合持仓数据失败: ${error.message}`);
          this.recordError(error);
        }
      }

      // 收集各交易所持仓数据以计算聚合
      for (const symbol of ['BTC', 'ETH']) { // 主要币种的详细数据
        try {
          const exchangeOI = await coinglassApi.getExchangeOpenInterest({
            symbol: symbol.toLowerCase()
          });

          if (exchangeOI?.data && Array.isArray(exchangeOI.data)) {
            const aggregatedByExchange = this.aggregateByExchange(exchangeOI.data, symbol, timestamp);
            if (aggregatedByExchange) {
              allData.push(aggregatedByExchange);
            }
          }

          await this.delay(200);

        } catch (error) {
          log.warn(`收集 ${symbol} 交易所持仓数据失败: ${error.message}`);
        }
      }

      log.info(`${this.name}: 收集完成，共获取 ${allData.length} 条聚合持仓数据`);
      return allData;

    } catch (error) {
      log.error(`${this.name}: 数据收集失败`, error);
      throw error;
    }
  }

  transformAggregateData(data, symbol, timestamp) {
    try {
      if (!data || typeof data !== 'object') {
        return null;
      }

      return {
        symbol: symbol,
        total_open_interest_usd: this.parseNumber(data.totalOpenInterestUsd),
        total_open_interest_native: this.parseNumber(data.totalOpenInterest),
        exchanges: this.extractExchangeData(data.exchanges),
        dominant_exchange: this.findDominantExchange(data.exchanges),
        timestamp: timestamp,
        created_at: timestamp
      };
    } catch (error) {
      log.warn(`转换聚合持仓数据失败: ${error.message}`, { data, symbol });
      return null;
    }
  }

  aggregateByExchange(exchangeData, symbol, timestamp) {
    try {
      if (!Array.isArray(exchangeData) || exchangeData.length === 0) {
        return null;
      }

      let totalOIUsd = 0;
      let totalOINative = 0;
      const exchanges = [];

      for (const item of exchangeData) {
        const oiUsd = this.parseNumber(item.openInterestUsd);
        const oiNative = this.parseNumber(item.openInterest);

        if (oiUsd > 0) {
          totalOIUsd += oiUsd;
          totalOINative += oiNative;

          exchanges.push({
            exchange: item.exchangeName || 'Unknown',
            open_interest_usd: oiUsd,
            open_interest_native: oiNative,
            percentage: 0 // 稍后计算
          });
        }
      }

      // 计算各交易所百分比
      if (totalOIUsd > 0) {
        exchanges.forEach(exchange => {
          exchange.percentage = parseFloat(((exchange.open_interest_usd / totalOIUsd) * 100).toFixed(2));
        });
      }

      // 按持仓量排序
      exchanges.sort((a, b) => b.open_interest_usd - a.open_interest_usd);

      return {
        symbol: symbol,
        total_open_interest_usd: totalOIUsd,
        total_open_interest_native: totalOINative,
        exchanges: exchanges.slice(0, 10), // 只保留前10个交易所
        dominant_exchange: exchanges.length > 0 ? exchanges[0].exchange : null,
        timestamp: timestamp,
        created_at: timestamp
      };

    } catch (error) {
      log.warn(`聚合交易所数据失败: ${error.message}`, { exchangeData, symbol });
      return null;
    }
  }

  extractExchangeData(exchangesData) {
    if (!Array.isArray(exchangesData)) {
      return [];
    }

    return exchangesData.map(item => ({
      exchange: item.exchangeName || item.exchange || 'Unknown',
      open_interest_usd: this.parseNumber(item.openInterestUsd),
      open_interest_native: this.parseNumber(item.openInterest),
      percentage: this.parseNumber(item.percentage)
    })).filter(item => item.open_interest_usd > 0);
  }

  findDominantExchange(exchangesData) {
    if (!Array.isArray(exchangesData) || exchangesData.length === 0) {
      return null;
    }

    const sortedExchanges = exchangesData
      .filter(item => this.parseNumber(item.openInterestUsd) > 0)
      .sort((a, b) => this.parseNumber(b.openInterestUsd) - this.parseNumber(a.openInterestUsd));

    return sortedExchanges.length > 0 ? 
      (sortedExchanges[0].exchangeName || sortedExchanges[0].exchange || 'Unknown') : 
      null;
  }

  parseNumber(value) {
    if (typeof value === 'number') return value;
    if (typeof value === 'string') {
      const parsed = parseFloat(value);
      return isNaN(parsed) ? 0 : parsed;
    }
    return 0;
  }

  async validateData(data) {
    const errors = [];

    for (const item of data) {
      // 验证必需字段
      if (!item.symbol) {
        errors.push(`缺少币种符号: ${JSON.stringify(item)}`);
        continue;
      }

      // 验证持仓量数据
      if (typeof item.total_open_interest_usd !== 'number' || item.total_open_interest_usd < 0) {
        errors.push(`持仓量数据异常: ${item.symbol}, value: ${item.total_open_interest_usd}`);
      }

      // 验证交易所数据结构
      if (!Array.isArray(item.exchanges)) {
        errors.push(`交易所数据格式错误: ${item.symbol}`);
        continue;
      }

      // 验证百分比总和
      const totalPercentage = item.exchanges.reduce((sum, ex) => sum + (ex.percentage || 0), 0);
      if (Math.abs(totalPercentage - 100) > 5 && item.exchanges.length > 0) { // 允许5%误差
        errors.push(`百分比总和异常: ${item.symbol}, total: ${totalPercentage}%`);
      }
    }

    if (errors.length > 0) {
      log.warn(`${this.name}: 数据验证发现 ${errors.length} 个问题`, errors.slice(0, 5));
    }

    return {
      isValid: errors.length === 0,
      errors: errors,
      validCount: data.length - errors.length,
      totalCount: data.length
    };
  }

  getCollectorInfo() {
    return {
      name: this.name,
      collectionName: this.collectionName,
      description: '收集聚合持仓数据，包括总持仓量和各交易所分布',
      interval: `${this.interval}分钟`,
      supportedSymbols: this.supportedSymbols,
      apiEndpoints: [
        '/api/futures/aggregate-open-interest',
        '/api/futures/exchange-open-interest'
      ],
      dataFields: [
        'symbol', 'total_open_interest_usd', 'total_open_interest_native',
        'exchanges', 'dominant_exchange', 'timestamp'
      ]
    };
  }
}