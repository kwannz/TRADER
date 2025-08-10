const BaseCollector = require('./base-collector');

class OpenInterestCollector extends BaseCollector {
  constructor(coinGlassClient) {
    super('OpenInterestCollector', 'open_interest', coinGlassClient, {
      batchSize: 300,
      retryAttempts: 3
    });
    
    this.symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP'];
    this.timeframes = ['1h', '4h', '24h'];
  }

  async performCollection() {
    const allData = [];
    let totalCount = 0;

    try {
      // 检查客户端是否可用
      if (!this.coinGlassClient) {
        console.log('CoinGlass client not available, skipping open interest collection');
        return { count: 0, saved: 0, duplicatesRemoved: 0 };
      }

      for (const symbol of this.symbols) {
        for (const timeframe of this.timeframes) {
          try {
            const response = await this.coinGlassClient.getOpenInterest(symbol, 'all', timeframe);
            
            if (response.data && Array.isArray(response.data)) {
              for (const item of response.data) {
                const processedData = this.processOpenInterestData(item, timeframe);
                if (processedData) {
                  allData.push(processedData);
                  totalCount++;
                }
              }
            }
            
            // 避免过于频繁的请求
            await this.sleep(200);
          } catch (error) {
            console.log(`Failed to get open interest for ${symbol} ${timeframe}: ${error.message}`);
          }
        }
      }

      const saveResult = await this.saveData(allData);
      
      return {
        count: totalCount,
        saved: saveResult.insertedCount,
        duplicatesRemoved: saveResult.duplicatesRemoved || 0
      };
      
    } catch (error) {
      throw new Error(`Failed to collect open interest data: ${error.message}`);
    }
  }

  processOpenInterestData(data, timeframe) {
    try {
      return {
        symbol: this.safeParseString(data.symbol),
        exchange: this.safeParseString(data.exchange || 'Unknown'),
        open_interest_usd: this.safeParseNumber(data.open_interest_usd || data.openInterestUsd),
        open_interest_native: this.safeParseNumber(data.open_interest_native || data.openInterest),
        cash_margin_oi: this.safeParseNumber(data.cash_margin_oi),
        crypto_margin_oi: this.safeParseNumber(data.crypto_margin_oi),
        timeframe: timeframe,
        timestamp: this.formatTimestamp(data.timestamp || new Date())
      };
    } catch (error) {
      return null;
    }
  }

  generateDuplicateKey(item) {
    const timestamp = Math.floor(new Date(item.timestamp).getTime() / (15 * 60 * 1000)) * (15 * 60 * 1000);
    return `${item.symbol}-${item.exchange}-${item.timeframe}-${timestamp}`;
  }
}

module.exports = OpenInterestCollector; 