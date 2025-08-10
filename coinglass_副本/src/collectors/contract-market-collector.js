const BaseCollector = require('./base-collector');

class ContractMarketCollector extends BaseCollector {
  constructor(coinGlassClient) {
    super('ContractMarketCollector', 'contract_markets', coinGlassClient, {
      batchSize: 500,
      retryAttempts: 3
    });
    
    this.supportedSymbols = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP', 'DOGE', 'MATIC', 'DOT', 'AVAX'];
  }

  async performCollection() {
    const allMarketData = [];
    let totalCount = 0;

    try {
      // 检查客户端是否可用
      if (!this.coinGlassClient) {
        throw new Error('CoinGlass client not available');
      }

      // 获取所有市场数据
      const response = await this.coinGlassClient.getContractMarkets('all');
      
      if (response.data && Array.isArray(response.data)) {
        for (const marketData of response.data) {
          const processedData = this.processMarketData(marketData);
          if (processedData) {
            allMarketData.push(processedData);
            totalCount++;
          }
        }
      }

      // 保存数据
      const saveResult = await this.saveData(allMarketData);
      
      return {
        count: totalCount,
        saved: saveResult.insertedCount,
        duplicatesRemoved: saveResult.duplicatesRemoved || 0
      };
      
    } catch (error) {
      throw new Error(`Failed to collect contract market data: ${error.message}`);
    }
  }

  processMarketData(data) {
    try {
      return {
        symbol: this.safeParseString(data.symbol),
        exchange: this.safeParseString(data.exchange || 'Unknown'),
        price: this.safeParseNumber(data.price),
        price_change_24h: this.safeParseNumber(data.price_change_24h),
        price_change_percent_24h: this.safeParseNumber(data.price_change_percent_24h),
        volume_24h: this.safeParseNumber(data.volume_24h),
        volume_change_percent_24h: this.safeParseNumber(data.volume_change_percent_24h),
        market_cap: this.safeParseNumber(data.market_cap),
        funding_rate: this.safeParseNumber(data.funding_rate),
        open_interest: this.safeParseNumber(data.open_interest),
        oi_change_24h: this.safeParseNumber(data.oi_change_24h),
        timestamp: this.formatTimestamp(data.timestamp || new Date())
      };
    } catch (error) {
      return null;
    }
  }

  generateDuplicateKey(item) {
    const timestamp = Math.floor(new Date(item.timestamp).getTime() / (5 * 60 * 1000)) * (5 * 60 * 1000);
    return `${item.symbol}-${item.exchange}-${timestamp}`;
  }
}

module.exports = ContractMarketCollector; 