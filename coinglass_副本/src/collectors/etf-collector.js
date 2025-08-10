const BaseCollector = require('./base-collector');

class ETFCollector extends BaseCollector {
  constructor(coinGlassClient) {
    super('ETFCollector', 'etf_data', coinGlassClient, {
      batchSize: 100,
      retryAttempts: 3
    });
    
    this.etfTypes = ['btc', 'eth'];
  }

  async performCollection() {
    const allData = [];
    let totalCount = 0;

    try {
      // 检查客户端是否可用
      if (!this.coinGlassClient) {
        console.log('CoinGlass client not available, skipping ETF collection');
        return { count: 0, saved: 0, duplicatesRemoved: 0 };
      }

      for (const etfType of this.etfTypes) {
        try {
          const response = await this.coinGlassClient.getETFFlow(etfType);
          
          if (response.data && Array.isArray(response.data)) {
            for (const item of response.data) {
              const processedData = this.processETFData(item, etfType);
              if (processedData) {
                allData.push(processedData);
                totalCount++;
              }
            }
          }
          
          await this.sleep(500);
        } catch (error) {
          console.log(`Failed to get ETF data for ${etfType}: ${error.message}`);
        }
      }

      const saveResult = await this.saveData(allData);
      
      return {
        count: totalCount,
        saved: saveResult.insertedCount,
        duplicatesRemoved: saveResult.duplicatesRemoved || 0
      };
      
    } catch (error) {
      throw new Error(`Failed to collect ETF data: ${error.message}`);
    }
  }

  processETFData(data, type) {
    try {
      return {
        name: this.safeParseString(data.name || data.etf_name),
        type: type,
        flows_usd: this.safeParseNumber(data.flows_usd || data.netFlow || data.flow),
        net_assets: this.safeParseNumber(data.net_assets || data.aum),
        premium_discount: this.safeParseNumber(data.premium_discount || data.premium),
        price: this.safeParseNumber(data.price || data.nav),
        volume: this.safeParseNumber(data.volume),
        date: this.formatTimestamp(data.date || data.timestamp || new Date()),
        timestamp: this.formatTimestamp(data.timestamp || new Date())
      };
    } catch (error) {
      return null;
    }
  }

  generateDuplicateKey(item) {
    const date = new Date(item.date);
    const dateKey = `${date.getFullYear()}-${(date.getMonth() + 1).toString().padStart(2, '0')}-${date.getDate().toString().padStart(2, '0')}`;
    return `${item.name}-${item.type}-${dateKey}`;
  }
}

module.exports = ETFCollector; 