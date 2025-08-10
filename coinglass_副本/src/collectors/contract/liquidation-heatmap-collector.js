import { BaseCollector } from '../base-collector.js';
import { coinglassApi } from '../../services/coinglassApi.js';
import logger from "../../utils/logger.js";
const { log } = logger;

export class LiquidationHeatmapCollector extends BaseCollector {
  constructor() {
    super('liquidation-heatmap-collector', 'liquidation_heatmaps');
    this.interval = 10; // 10分钟收集一次
    this.priority = 'medium';
    this.supportedSymbols = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP'];
    this.exchanges = ['Binance', 'OKX', 'Bybit', 'Deribit'];
  }

  async collectData() {
    const allData = [];
    const timestamp = new Date();

    log.info(`${this.name}: 开始收集爆仓热力图数据`);

    try {
      // 收集币种爆仓热力图
      for (const symbol of this.supportedSymbols) {
        try {
          log.debug(`收集 ${symbol} 爆仓热力图数据`);
          
          const coinHeatmap = await coinglassApi.getCoinLiquidationHeatmap({
            symbol: symbol.toLowerCase()
          });

          if (coinHeatmap?.data && Array.isArray(coinHeatmap.data)) {
            for (const item of coinHeatmap.data) {
              const heatmapData = this.transformCoinHeatmapData(item, symbol, timestamp);
              if (heatmapData) {
                allData.push(heatmapData);
              }
            }
          }

          await this.delay(300);

        } catch (error) {
          log.warn(`收集 ${symbol} 币种爆仓热力图失败: ${error.message}`);
          this.recordError(error);
        }
      }

      // 收集交易对爆仓热力图
      for (const symbol of ['BTC', 'ETH']) { // 主要币种
        for (const exchange of this.exchanges) {
          try {
            log.debug(`收集 ${symbol}-${exchange} 交易对爆仓热力图`);
            
            const pairHeatmap = await coinglassApi.getPairLiquidationHeatmap({
              symbol: symbol.toLowerCase(),
              exchange: exchange
            });

            if (pairHeatmap?.data && Array.isArray(pairHeatmap.data)) {
              for (const item of pairHeatmap.data) {
                const heatmapData = this.transformPairHeatmapData(item, symbol, exchange, timestamp);
                if (heatmapData) {
                  allData.push(heatmapData);
                }
              }
            }

            await this.delay(400);

          } catch (error) {
            log.warn(`收集 ${symbol}-${exchange} 交易对爆仓热力图失败: ${error.message}`);
          }
        }
      }

      log.info(`${this.name}: 收集完成，共获取 ${allData.length} 条爆仓热力图数据`);
      return allData;

    } catch (error) {
      log.error(`${this.name}: 数据收集失败`, error);
      throw error;
    }
  }

  transformCoinHeatmapData(item, symbol, timestamp) {
    try {
      if (!item || typeof item !== 'object') {
        return null;
      }

      return {
        symbol: symbol,
        exchange: 'AGGREGATE', // 币种级别的聚合数据
        price_level: this.parseNumber(item.price),
        long_liquidation_usd: this.parseNumber(item.longLiquidation),
        short_liquidation_usd: this.parseNumber(item.shortLiquidation),
        total_liquidation_usd: this.parseNumber(item.totalLiquidation),
        cumulative_long: this.parseNumber(item.cumulativeLong),
        cumulative_short: this.parseNumber(item.cumulativeShort),
        timestamp: item.time ? new Date(item.time * 1000) : timestamp,
        created_at: timestamp
      };
    } catch (error) {
      log.warn(`转换币种爆仓热力图数据失败: ${error.message}`, { item, symbol });
      return null;
    }
  }

  transformPairHeatmapData(item, symbol, exchange, timestamp) {
    try {
      if (!item || typeof item !== 'object') {
        return null;
      }

      return {
        symbol: symbol,
        exchange: exchange,
        price_level: this.parseNumber(item.price),
        long_liquidation_usd: this.parseNumber(item.longLiquidation),
        short_liquidation_usd: this.parseNumber(item.shortLiquidation),
        total_liquidation_usd: this.parseNumber(item.longLiquidation) + this.parseNumber(item.shortLiquidation),
        cumulative_long: this.parseNumber(item.cumulativeLong),
        cumulative_short: this.parseNumber(item.cumulativeShort),
        timestamp: item.time ? new Date(item.time * 1000) : timestamp,
        created_at: timestamp
      };
    } catch (error) {
      log.warn(`转换交易对爆仓热力图数据失败: ${error.message}`, { item, symbol, exchange });
      return null;
    }
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
      if (!item.symbol || !item.exchange) {
        errors.push(`缺少必需字段: ${JSON.stringify(item)}`);
        continue;
      }

      // 验证价格级别
      if (typeof item.price_level !== 'number' || item.price_level <= 0) {
        errors.push(`价格级别异常: ${item.symbol}-${item.exchange}, price: ${item.price_level}`);
      }

      // 验证爆仓数据
      if (item.long_liquidation_usd < 0 || item.short_liquidation_usd < 0) {
        errors.push(`爆仓数据异常: ${item.symbol}-${item.exchange}, long: ${item.long_liquidation_usd}, short: ${item.short_liquidation_usd}`);
      }

      // 验证总量一致性
      const calculatedTotal = item.long_liquidation_usd + item.short_liquidation_usd;
      if (Math.abs(calculatedTotal - item.total_liquidation_usd) > 0.01) {
        errors.push(`总爆仓量不一致: ${item.symbol}-${item.exchange}, calculated: ${calculatedTotal}, stored: ${item.total_liquidation_usd}`);
      }

      // 验证时间戳
      if (!item.timestamp || isNaN(new Date(item.timestamp).getTime())) {
        errors.push(`时间戳无效: ${item.symbol}-${item.exchange}`);
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
      description: '收集爆仓热力图数据，显示不同价格级别的爆仓分布',
      interval: `${this.interval}分钟`,
      supportedSymbols: this.supportedSymbols,
      supportedExchanges: this.exchanges,
      apiEndpoints: [
        '/api/futures/coin-liquidation-heatmap',
        '/api/futures/pair-liquidation-heatmap'
      ],
      dataFields: [
        'symbol', 'exchange', 'price_level', 'long_liquidation_usd', 
        'short_liquidation_usd', 'total_liquidation_usd', 'cumulative_long', 
        'cumulative_short', 'timestamp'
      ]
    };
  }
}