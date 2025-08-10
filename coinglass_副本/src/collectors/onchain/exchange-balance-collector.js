import { BaseCollector } from '../base-collector.js';
import { coinglassApi } from '../../services/coinglassApi.js';
import logger from "../../utils/logger.js";
const { log } = logger;

export class ExchangeBalanceCollector extends BaseCollector {
  constructor() {
    super('exchange-balance-collector', 'exchange_balances');
    this.interval = 60; // 60分钟收集一次
    this.priority = 'medium';
    this.supportedSymbols = ['BTC', 'ETH', 'USDT', 'USDC', 'BNB', 'SOL', 'XRP', 'ADA'];
    this.exchanges = ['Binance', 'Coinbase', 'Kraken', 'Huobi', 'OKX', 'Bitfinex', 'Gemini'];
  }

  async collectData() {
    const allData = [];
    const timestamp = new Date();

    log.info(`${this.name}: 开始收集交易所余额数据`);

    try {
      // 收集各币种的交易所余额
      for (const symbol of this.supportedSymbols) {
        try {
          log.debug(`收集 ${symbol} 交易所余额数据`);
          
          const exchangeBalance = await coinglassApi.getExchangeBalance({
            symbol: symbol.toLowerCase()
          });

          if (exchangeBalance?.data && Array.isArray(exchangeBalance.data)) {
            for (const item of exchangeBalance.data) {
              const balanceData = this.transformExchangeBalanceData(item, symbol, timestamp);
              if (balanceData) {
                allData.push(balanceData);
              }
            }
          }

          await this.delay(400);

        } catch (error) {
          log.warn(`收集 ${symbol} 交易所余额失败: ${error.message}`);
          this.recordError(error);
        }
      }

      // 收集交易所余额图表数据（历史趋势）
      for (const symbol of ['BTC', 'ETH']) { // 主要币种的历史数据
        try {
          log.debug(`收集 ${symbol} 交易所余额图表数据`);
          
          const balanceChart = await coinglassApi.getExchangeBalanceChart({
            symbol: symbol.toLowerCase(),
            time_type: '1d'
          });

          if (balanceChart?.data && Array.isArray(balanceChart.data)) {
            for (const item of balanceChart.data) {
              const chartData = this.transformBalanceChartData(item, symbol, timestamp);
              if (chartData) {
                allData.push(chartData);
              }
            }
          }

          await this.delay(500);

        } catch (error) {
          log.warn(`收集 ${symbol} 交易所余额图表失败: ${error.message}`);
        }
      }

      // 收集ERC20转账数据
      try {
        log.debug('收集ERC20转账数据');
        
        const erc20Transfers = await coinglassApi.getErc20Transfers({
          limit: 100
        });

        if (erc20Transfers?.data && Array.isArray(erc20Transfers.data)) {
          for (const item of erc20Transfers.data) {
            const transferData = this.transformTransferData(item, timestamp);
            if (transferData) {
              allData.push(transferData);
            }
          }
        }

      } catch (error) {
        log.warn(`收集ERC20转账数据失败: ${error.message}`);
      }

      log.info(`${this.name}: 收集完成，共获取 ${allData.length} 条交易所余额数据`);
      return allData;

    } catch (error) {
      log.error(`${this.name}: 数据收集失败`, error);
      throw error;
    }
  }

  transformExchangeBalanceData(item, symbol, timestamp) {
    try {
      if (!item || typeof item !== 'object') {
        return null;
      }

      const balance = this.parseNumber(item.balance);
      const balanceChange24h = this.parseNumber(item.balanceChange24h);
      const inFlow24h = this.parseNumber(item.inFlow24h);
      const outFlow24h = this.parseNumber(item.outFlow24h);

      return {
        exchange: item.exchangeName || item.exchange || 'Unknown',
        symbol: symbol,
        balance: balance,
        balance_change_24h: balanceChange24h,
        balance_change_percent_24h: balance > 0 ? (balanceChange24h / balance * 100) : 0,
        in_flow_24h: inFlow24h,
        out_flow_24h: outFlow24h,
        net_flow_24h: inFlow24h - outFlow24h,
        timestamp: item.time ? new Date(item.time * 1000) : timestamp,
        created_at: timestamp
      };

    } catch (error) {
      log.warn(`转换交易所余额数据失败: ${error.message}`, { item, symbol });
      return null;
    }
  }

  transformBalanceChartData(item, symbol, timestamp) {
    try {
      if (!item || typeof item !== 'object') {
        return null;
      }

      const balance = this.parseNumber(item.balance);
      const change = this.parseNumber(item.change);

      return {
        exchange: 'AGGREGATE', // 图表数据通常是聚合的
        symbol: symbol,
        balance: balance,
        balance_change_24h: change,
        balance_change_percent_24h: balance > 0 ? (change / balance * 100) : 0,
        in_flow_24h: 0, // 图表数据通常不包含流入流出详情
        out_flow_24h: 0,
        net_flow_24h: change,
        timestamp: item.time ? new Date(item.time * 1000) : timestamp,
        created_at: timestamp
      };

    } catch (error) {
      log.warn(`转换余额图表数据失败: ${error.message}`, { item, symbol });
      return null;
    }
  }

  transformTransferData(item, timestamp) {
    try {
      if (!item || typeof item !== 'object') {
        return null;
      }

      const amount = this.parseNumber(item.amount);
      const valueUsd = this.parseNumber(item.valueUsd);

      // 判断是否为交易所相关转账
      const fromExchange = this.identifyExchange(item.from);
      const toExchange = this.identifyExchange(item.to);

      if (!fromExchange && !toExchange) {
        return null; // 跳过非交易所相关的转账
      }

      return {
        exchange: fromExchange || toExchange || 'Unknown',
        symbol: item.symbol || 'ETH',
        balance: amount,
        balance_change_24h: fromExchange ? -amount : amount, // 流出为负，流入为正
        balance_change_percent_24h: 0, // 无法计算百分比变化
        in_flow_24h: toExchange ? amount : 0,
        out_flow_24h: fromExchange ? amount : 0,
        net_flow_24h: (toExchange ? amount : 0) - (fromExchange ? amount : 0),
        timestamp: item.time ? new Date(item.time * 1000) : timestamp,
        created_at: timestamp
      };

    } catch (error) {
      log.warn(`转换转账数据失败: ${error.message}`, { item });
      return null;
    }
  }

  identifyExchange(address) {
    if (!address || typeof address !== 'string') {
      return null;
    }

    // 简单的交易所地址识别（实际应用中需要更完整的地址库）
    const exchangePatterns = {
      'Binance': ['0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE'],
      'Coinbase': ['0x71660c4005BA85c37ccec55d0C4493E66Fe775d3'],
      'Kraken': ['0x2910543af39aba0cd09dbb2d50200b3e800a63d2'],
      'Huobi': ['0x6F50c6bff08Ec925232937b204b0aE900C676B39'],
      'OKX': ['0x98eC059Dc3aDFBdd63429454aEB0c990fbA4a128']
    };

    for (const [exchange, addresses] of Object.entries(exchangePatterns)) {
      for (const pattern of addresses) {
        if (address.toLowerCase() === pattern.toLowerCase()) {
          return exchange;
        }
      }
    }

    return null;
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
      if (!item.exchange || !item.symbol) {
        errors.push(`缺少必需字段: ${JSON.stringify(item)}`);
        continue;
      }

      // 验证余额数据
      if (typeof item.balance !== 'number' || item.balance < 0) {
        errors.push(`余额数据异常: ${item.exchange}-${item.symbol}, balance: ${item.balance}`);
      }

      // 验证百分比变化
      if (typeof item.balance_change_percent_24h !== 'number' || 
          Math.abs(item.balance_change_percent_24h) > 1000) { // 允许最大1000%变化
        errors.push(`余额变化百分比异常: ${item.exchange}-${item.symbol}, percent: ${item.balance_change_percent_24h}`);
      }

      // 验证流入流出一致性
      const calculatedNetFlow = item.in_flow_24h - item.out_flow_24h;
      if (Math.abs(calculatedNetFlow - item.net_flow_24h) > 0.001) {
        errors.push(`净流量计算不一致: ${item.exchange}-${item.symbol}, calculated: ${calculatedNetFlow}, stored: ${item.net_flow_24h}`);
      }

      // 验证时间戳
      if (!item.timestamp || isNaN(new Date(item.timestamp).getTime())) {
        errors.push(`时间戳无效: ${item.exchange}-${item.symbol}`);
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
      description: '收集各交易所的链上余额数据，包括余额变化和资金流动',
      interval: `${this.interval}分钟`,
      supportedSymbols: this.supportedSymbols,
      supportedExchanges: this.exchanges,
      apiEndpoints: [
        '/api/onchain/exchange-balance',
        '/api/onchain/exchange-balance-chart',
        '/api/onchain/erc20-transfers'
      ],
      dataFields: [
        'exchange', 'symbol', 'balance', 'balance_change_24h', 
        'balance_change_percent_24h', 'in_flow_24h', 'out_flow_24h', 
        'net_flow_24h', 'timestamp'
      ]
    };
  }
}