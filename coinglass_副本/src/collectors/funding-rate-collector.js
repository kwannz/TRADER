const BaseCollector = require('./base-collector');

class FundingRateCollector extends BaseCollector {
  constructor(coinGlassClient) {
    super('FundingRateCollector', 'funding_rates', coinGlassClient, {
      batchSize: 200,
      retryAttempts: 3
    });
    
    this.symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA'];
  }

  async performCollection() {
    const allData = [];
    let totalCount = 0;

    try {
      // 检查客户端是否可用
      if (!this.coinGlassClient) {
        console.log('CoinGlass client not available, skipping funding rate collection');
        return { count: 0, saved: 0, duplicatesRemoved: 0 };
      }

      for (const symbol of this.symbols) {
        try {
          const response = await this.coinGlassClient.getFundingRates(symbol);
          
          if (response.data && Array.isArray(response.data)) {
            for (const item of response.data) {
              const processedData = this.processFundingRateData(item);
              if (processedData) {
                allData.push(processedData);
                totalCount++;
              }
            }
          }
          
          await this.sleep(300);
        } catch (error) {
          console.log(`Failed to get funding rates for ${symbol}: ${error.message}`);
        }
      }

      const saveResult = await this.saveData(allData);
      
      return {
        count: totalCount,
        saved: saveResult.insertedCount,
        duplicatesRemoved: saveResult.duplicatesRemoved || 0
      };
      
    } catch (error) {
      throw new Error(`Failed to collect funding rate data: ${error.message}`);
    }
  }

  processFundingRateData(data) {
    try {
      return {
        symbol: this.safeParseString(data.symbol),
        exchange: this.safeParseString(data.exchange || 'Unknown'),
        funding_rate: this.safeParseNumber(data.funding_rate || data.fundingRate),
        next_funding_time: this.formatTimestamp(data.next_funding_time || data.nextFundingTime),
        funding_rate_8h_avg: this.safeParseNumber(data.funding_rate_8h_avg || data.avg8h),
        oi_weighted_funding: this.safeParseNumber(data.oi_weighted_funding),
        volume_weighted_funding: this.safeParseNumber(data.volume_weighted_funding),
        timestamp: this.formatTimestamp(data.timestamp || new Date())
      };
    } catch (error) {
      return null;
    }
  }

  generateDuplicateKey(item) {
    const timestamp = Math.floor(new Date(item.timestamp).getTime() / (60 * 60 * 1000)) * (60 * 60 * 1000);
    return `${item.symbol}-${item.exchange}-${timestamp}`;
  }
}

module.exports = FundingRateCollector; 