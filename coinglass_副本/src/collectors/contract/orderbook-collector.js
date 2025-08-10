import { BaseCollector } from '../base-collector.js';
import { coinglassApi } from '../../services/coinglassApi.js';
import logger from "../../utils/logger.js";
const { log } = logger;

export class OrderbookCollector extends BaseCollector {
  constructor() {
    super('orderbook-collector', 'orderbook');
    this.interval = 2; // 2分钟收集一次
    this.priority = 'high';
    this.supportedSymbols = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'AVAX', 'DOGE', 'MATIC', 'DOT'];
    this.exchanges = ['Binance', 'OKX', 'Bybit', 'Deribit', 'Bitget'];
  }

  async collectData() {
    const allData = [];
    const timestamp = new Date();

    log.info(`${this.name}: 开始收集订单簿数据`);

    try {
      // 收集聚合订单簿数据
      for (const symbol of this.supportedSymbols) {
        try {
          log.debug(`收集 ${symbol} 聚合订单簿数据`);
          
          const aggregateOrderbook = await coinglassApi.getAggregateOrderbook({
            symbol: symbol.toLowerCase()
          });

          if (aggregateOrderbook?.data && Array.isArray(aggregateOrderbook.data)) {
            for (const item of aggregateOrderbook.data) {
              const orderbookData = this.transformAggregateOrderbookData(item, symbol, timestamp);
              if (orderbookData) {
                allData.push(orderbookData);
              }
            }
          }

          // 添加小延迟避免API限制
          await this.delay(100);

        } catch (error) {
          log.warn(`收集 ${symbol} 订单簿数据失败: ${error.message}`);
          this.recordError(error);
        }
      }

      // 收集各交易所订单簿历史数据
      for (const symbol of ['BTC', 'ETH']) { // 限制为主要币种以节省API调用
        for (const exchange of this.exchanges) {
          try {
            log.debug(`收集 ${symbol}-${exchange} 订单簿历史数据`);
            
            const orderbookHistory = await coinglassApi.getOrderbookHistory({
              symbol: symbol.toLowerCase(),
              exchange: exchange,
              time_type: '1h'
            });

            if (orderbookHistory?.data && Array.isArray(orderbookHistory.data)) {
              for (const item of orderbookHistory.data) {
                const orderbookData = this.transformOrderbookHistoryData(item, symbol, exchange, timestamp);
                if (orderbookData) {
                  allData.push(orderbookData);
                }
              }
            }

            await this.delay(150);

          } catch (error) {
            log.warn(`收集 ${symbol}-${exchange} 订单簿历史失败: ${error.message}`);
          }
        }
      }

      log.info(`${this.name}: 收集完成，共获取 ${allData.length} 条订单簿数据`);
      return allData;

    } catch (error) {
      log.error(`${this.name}: 数据收集失败`, error);
      throw error;
    }
  }

  transformAggregateOrderbookData(item, symbol, timestamp) {
    try {
      if (!item || typeof item !== 'object') {
        return null;
      }

      return {
        symbol: symbol,
        exchange: 'AGGREGATE', // 聚合数据标记
        bids: this.transformOrderbookSide(item.bids),
        asks: this.transformOrderbookSide(item.asks),
        spread: this.calculateSpread(item.bids, item.asks),
        timestamp: item.time ? new Date(item.time * 1000) : timestamp,
        created_at: timestamp
      };
    } catch (error) {
      log.warn(`转换聚合订单簿数据失败: ${error.message}`, { item, symbol });
      return null;
    }
  }

  transformOrderbookHistoryData(item, symbol, exchange, timestamp) {
    try {
      if (!item || typeof item !== 'object') {
        return null;
      }

      return {
        symbol: symbol,
        exchange: exchange,
        bids: this.transformOrderbookSide(item.bids),
        asks: this.transformOrderbookSide(item.asks),
        spread: this.calculateSpread(item.bids, item.asks),
        timestamp: item.time ? new Date(item.time * 1000) : timestamp,
        created_at: timestamp
      };
    } catch (error) {
      log.warn(`转换订单簿历史数据失败: ${error.message}`, { item, symbol, exchange });
      return null;
    }
  }

  transformOrderbookSide(orders) {
    if (!Array.isArray(orders)) {
      return [];
    }

    return orders.slice(0, 20).map(order => { // 只保留前20档
      if (Array.isArray(order) && order.length >= 2) {
        return {
          price: parseFloat(order[0]) || 0,
          amount: parseFloat(order[1]) || 0,
          total: parseFloat(order[2]) || 0
        };
      }
      return null;
    }).filter(order => order !== null);
  }

  calculateSpread(bids, asks) {
    try {
      if (!Array.isArray(bids) || !Array.isArray(asks) || bids.length === 0 || asks.length === 0) {
        return 0;
      }

      const bestBid = Array.isArray(bids[0]) ? parseFloat(bids[0][0]) : 0;
      const bestAsk = Array.isArray(asks[0]) ? parseFloat(asks[0][0]) : 0;

      if (bestBid > 0 && bestAsk > 0) {
        return ((bestAsk - bestBid) / bestBid * 100).toFixed(4);
      }

      return 0;
    } catch (error) {
      return 0;
    }
  }

  async validateData(data) {
    const errors = [];

    for (const item of data) {
      // 验证必需字段
      if (!item.symbol || !item.exchange) {
        errors.push(`缺少必需字段: ${JSON.stringify(item)}`);
        continue;
      }

      // 验证订单簿数据结构
      if (!Array.isArray(item.bids) || !Array.isArray(item.asks)) {
        errors.push(`订单簿数据格式错误: ${item.symbol}-${item.exchange}`);
        continue;
      }

      // 验证价格合理性
      if (item.bids.length > 0 && item.asks.length > 0) {
        const bestBid = item.bids[0]?.price || 0;
        const bestAsk = item.asks[0]?.price || 0;
        
        if (bestBid <= 0 || bestAsk <= 0 || bestBid >= bestAsk) {
          errors.push(`订单簿价格异常: ${item.symbol}-${item.exchange}, bid: ${bestBid}, ask: ${bestAsk}`);
        }
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
      description: '收集合约订单簿深度数据，包括聚合订单簿和各交易所历史数据',
      interval: `${this.interval}分钟`,
      supportedSymbols: this.supportedSymbols,
      supportedExchanges: this.exchanges,
      apiEndpoints: [
        '/api/futures/aggregate-orderbook',
        '/api/futures/orderbook-history'
      ],
      dataFields: [
        'symbol', 'exchange', 'bids', 'asks', 'spread', 'timestamp'
      ]
    };
  }
}