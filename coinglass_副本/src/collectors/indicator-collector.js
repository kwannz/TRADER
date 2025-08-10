const BaseCollector = require('./base-collector');

class IndicatorCollector extends BaseCollector {
  constructor(coinGlassClient) {
    super('IndicatorCollector', 'indicators', coinGlassClient, {
      batchSize: 50,
      retryAttempts: 3
    });
    
    this.indicators = [
      { name: 'fearGreed', method: 'getFearGreedIndex' },
      { name: 'ahr999', method: 'getAHR999' },
      { name: 'rainbow', method: 'getRainbowChart' },
      { name: 'rsi', method: 'getRSI' }
    ];
  }

  async performCollection() {
    const allData = [];
    let totalCount = 0;

    try {
      // 检查客户端是否可用
      if (!this.coinGlassClient) {
        console.log('CoinGlass client not available, skipping indicator collection');
        return { count: 0, saved: 0, duplicatesRemoved: 0 };
      }

      for (const indicator of this.indicators) {
        try {
          const response = await this.coinGlassClient[indicator.method]();
          
          if (response.data && Array.isArray(response.data)) {
            for (const item of response.data) {
              const processedData = this.processIndicatorData(item, indicator.name);
              if (processedData) {
                allData.push(processedData);
                totalCount++;
              }
            }
          }
          
          await this.sleep(500);
        } catch (error) {
          console.log(`Failed to get indicator ${indicator.name}: ${error.message}`);
        }
      }

      const saveResult = await this.saveData(allData);
      
      return {
        count: totalCount,
        saved: saveResult.insertedCount,
        duplicatesRemoved: saveResult.duplicatesRemoved || 0
      };
      
    } catch (error) {
      throw new Error(`Failed to collect indicator data: ${error.message}`);
    }
  }

  processIndicatorData(data, type) {
    try {
      let processedData;
      switch (type) {
        case 'fearGreed':
          processedData = this.processFearGreedData(data);
          break;
        case 'ahr999':
          processedData = this.processAHR999Data(data);
          break;
        case 'rainbow':
          processedData = this.processRainbowData(data);
          break;
        case 'rsi':
          processedData = this.processRSIData(data);
          break;
        default:
          return null;
      }
      return processedData;
    } catch (error) {
      return null;
    }
  }

  processFearGreedData(data) {
    try {
      return {
        name: 'Fear_Greed_Index',
        symbol: 'MARKET',
        value: this.safeParseNumber(data.value || data.fgi),
        signal: this.getSignalFromValue(data.value, 'fear_greed'),
        description: this.safeParseString(data.classification || data.desc),
        timestamp: this.formatTimestamp(data.timestamp || new Date())
      };
    } catch (error) {
      return null;
    }
  }

  processAHR999Data(data) {
    try {
      return {
        name: 'AHR999',
        symbol: 'BTC',
        value: this.safeParseNumber(data.value || data.ahr999),
        signal: this.getSignalFromValue(data.value, 'ahr999'),
        description: this.safeParseString(data.description || data.level),
        timestamp: this.formatTimestamp(data.timestamp || new Date())
      };
    } catch (error) {
      return null;
    }
  }

  processRainbowData(data) {
    try {
      return {
        name: 'Rainbow_Chart',
        symbol: 'BTC',
        value: this.safeParseNumber(data.value || data.rainbow_value),
        signal: this.getSignalFromValue(data.value, 'rainbow'),
        description: this.safeParseString(data.description || data.color_band),
        timestamp: this.formatTimestamp(data.timestamp || new Date())
      };
    } catch (error) {
      return null;
    }
  }

  processRSIData(data) {
    try {
      return {
        name: 'RSI',
        symbol: 'BTC',
        value: this.safeParseNumber(data.value || data.rsi),
        signal: this.getSignalFromValue(data.value, 'rsi'),
        description: this.getRSIDescription(data.value),
        timestamp: this.formatTimestamp(data.timestamp || new Date())
      };
    } catch (error) {
      return null;
    }
  }

  getSignalFromValue(value, type) {
    const numValue = this.safeParseNumber(value);
    
    switch (type) {
      case 'fear_greed':
        if (numValue <= 25) return 'Extreme Fear';
        if (numValue <= 45) return 'Fear';
        if (numValue <= 55) return 'Neutral';
        if (numValue <= 75) return 'Greed';
        return 'Extreme Greed';
      
      case 'ahr999':
        if (numValue < 0.45) return 'Strong Buy';
        if (numValue < 1.2) return 'Buy';
        if (numValue < 5) return 'Hold';
        return 'Sell';
      
      case 'rsi':
        if (numValue <= 30) return 'Oversold';
        if (numValue >= 70) return 'Overbought';
        return 'Neutral';
      
      default:
        return 'Neutral';
    }
  }

  getRSIDescription(value) {
    const numValue = this.safeParseNumber(value);
    if (numValue <= 30) return 'RSI indicates oversold conditions';
    if (numValue >= 70) return 'RSI indicates overbought conditions';
    return 'RSI in neutral zone';
  }

  generateDuplicateKey(item) {
    const timestamp = Math.floor(new Date(item.timestamp).getTime() / (4 * 60 * 60 * 1000)) * (4 * 60 * 60 * 1000);
    return `${item.name}-${item.symbol}-${timestamp}`;
  }
}

module.exports = IndicatorCollector; 