import { BaseCollector } from '../base-collector.js';
import { coinglassApi } from '../../services/coinglassApi.js';
import logger from "../../utils/logger.js";
const { log } = logger;

export class MacroIndicatorCollector extends BaseCollector {
  constructor() {
    super('macro-indicator-collector', 'macro_indicators');
    this.interval = 60; // 60分钟收集一次
    this.priority = 'medium';
    
    // 定义支持的宏观指标
    this.indicators = [
      { name: 'ahr999', endpoint: 'getAhr999', symbol: 'BTC', category: 'macro' },
      { name: 'rainbowChart', endpoint: 'getRainbowChart', symbol: 'BTC', category: 'macro' },
      { name: 'puellMultiple', endpoint: 'getPuellMultiple', symbol: 'BTC', category: 'onchain' },
      { name: 'piCycleTop', endpoint: 'getPiCycleTop', symbol: 'BTC', category: 'macro' },
      { name: 'goldenRatio', endpoint: 'getGoldenRatio', symbol: 'BTC', category: 'macro' },
      { name: 'profitDays', endpoint: 'getProfitDays', symbol: 'BTC', category: 'macro' },
      { name: 'bubbleIndex', endpoint: 'getBubbleIndex', symbol: 'BTC', category: 'macro' },
      { name: 'twoYearMA', endpoint: 'getTwoYearMA', symbol: 'BTC', category: 'technical' },
      { name: 'fearGreedIndex', endpoint: 'getFearGreedIndex', symbol: null, category: 'macro' }
    ];
  }

  async collectData() {
    const allData = [];
    const timestamp = new Date();

    log.info(`${this.name}: 开始收集宏观指标数据`);

    try {
      // 收集各个宏观指标
      for (const indicator of this.indicators) {
        try {
          log.debug(`收集 ${indicator.name} 指标数据`);
          
          const indicatorData = await this.collectIndicator(indicator, timestamp);
          if (indicatorData) {
            allData.push(...indicatorData);
          }

          // 添加延迟避免API限制
          await this.delay(500);

        } catch (error) {
          log.warn(`收集 ${indicator.name} 指标失败: ${error.message}`);
          this.recordError(error);
        }
      }

      // 收集RSI指标列表
      try {
        log.debug('收集RSI指标列表');
        const rsiData = await coinglassApi.getRsiList();
        
        if (rsiData?.data && Array.isArray(rsiData.data)) {
          for (const item of rsiData.data) {
            const rsiIndicator = this.transformRSIData(item, timestamp);
            if (rsiIndicator) {
              allData.push(rsiIndicator);
            }
          }
        }
      } catch (error) {
        log.warn(`收集RSI指标失败: ${error.message}`);
      }

      // 收集稳定币市值数据
      try {
        log.debug('收集稳定币市值数据');
        const stablecoinData = await coinglassApi.getStablecoinMarketCap();
        
        if (stablecoinData?.data && Array.isArray(stablecoinData.data)) {
          for (const item of stablecoinData.data) {
            const stablecoinIndicator = this.transformStablecoinData(item, timestamp);
            if (stablecoinIndicator) {
              allData.push(stablecoinIndicator);
            }
          }
        }
      } catch (error) {
        log.warn(`收集稳定币市值数据失败: ${error.message}`);
      }

      log.info(`${this.name}: 收集完成，共获取 ${allData.length} 条宏观指标数据`);
      return allData;

    } catch (error) {
      log.error(`${this.name}: 数据收集失败`, error);
      throw error;
    }
  }

  async collectIndicator(indicator, timestamp) {
    try {
      const methodName = indicator.endpoint;
      if (typeof coinglassApi[methodName] !== 'function') {
        log.warn(`API方法 ${methodName} 不存在`);
        return null;
      }

      const params = indicator.symbol ? { symbol: indicator.symbol.toLowerCase() } : {};
      const response = await coinglassApi[methodName](params);

      if (!response?.data) {
        return null;
      }

      return this.transformIndicatorData(response.data, indicator, timestamp);

    } catch (error) {
      log.warn(`收集指标 ${indicator.name} 失败: ${error.message}`);
      return null;
    }
  }

  transformIndicatorData(data, indicator, timestamp) {
    try {
      const results = [];

      if (Array.isArray(data)) {
        // 处理数组数据
        for (const item of data) {
          const transformed = this.transformSingleIndicator(item, indicator, timestamp);
          if (transformed) {
            results.push(transformed);
          }
        }
      } else if (typeof data === 'object' && data !== null) {
        // 处理单个对象数据
        const transformed = this.transformSingleIndicator(data, indicator, timestamp);
        if (transformed) {
          results.push(transformed);
        }
      }

      return results;

    } catch (error) {
      log.warn(`转换指标数据失败: ${error.message}`, { indicator: indicator.name, data });
      return null;
    }
  }

  transformSingleIndicator(item, indicator, timestamp) {
    try {
      const value = this.extractValue(item);
      const signal = this.determineSignal(indicator.name, value);
      const signalStrength = this.calculateSignalStrength(indicator.name, value);
      const normalizedValue = this.normalizeValue(indicator.name, value);

      return {
        name: indicator.name,
        symbol: indicator.symbol,
        value: value,
        normalized_value: normalizedValue,
        signal: signal,
        signal_strength: signalStrength,
        description: this.getIndicatorDescription(indicator.name, value, signal),
        category: indicator.category,
        timestamp: item.time ? new Date(item.time * 1000) : timestamp,
        created_at: timestamp
      };

    } catch (error) {
      log.warn(`转换单个指标失败: ${error.message}`, { indicator: indicator.name, item });
      return null;
    }
  }

  transformRSIData(item, timestamp) {
    try {
      const value = this.parseNumber(item.rsi);
      const signal = this.determineRSISignal(value);

      return {
        name: 'RSI',
        symbol: item.symbol || 'BTC',
        value: value,
        normalized_value: value, // RSI已经是0-100的标准化值
        signal: signal,
        signal_strength: this.calculateRSISignalStrength(value),
        description: `RSI指标: ${value.toFixed(2)}`,
        category: 'technical',
        timestamp: timestamp,
        created_at: timestamp
      };
    } catch (error) {
      log.warn(`转换RSI数据失败: ${error.message}`, { item });
      return null;
    }
  }

  transformStablecoinData(item, timestamp) {
    try {
      const marketCap = this.parseNumber(item.marketCap);
      const change24h = this.parseNumber(item.change24h);

      return {
        name: 'stablecoinMarketCap',
        symbol: item.coin || 'USDT',
        value: marketCap,
        normalized_value: this.normalizeStablecoinMarketCap(marketCap),
        signal: change24h > 0 ? 'bullish' : change24h < 0 ? 'bearish' : 'neutral',
        signal_strength: Math.min(Math.abs(change24h / marketCap * 100) * 10, 10),
        description: `${item.coin} 市值: ${this.formatNumber(marketCap)}`,
        category: 'macro',
        timestamp: timestamp,
        created_at: timestamp
      };
    } catch (error) {
      log.warn(`转换稳定币数据失败: ${error.message}`, { item });
      return null;
    }
  }

  extractValue(item) {
    // 尝试从不同可能的字段提取数值
    if (typeof item.value === 'number') return item.value;
    if (typeof item.index === 'number') return item.index;
    if (typeof item.ratio === 'number') return item.ratio;
    if (typeof item.multiple === 'number') return item.multiple;
    if (typeof item.price === 'number') return item.price;
    
    // 尝试解析字符串
    const numFields = ['value', 'index', 'ratio', 'multiple', 'price'];
    for (const field of numFields) {
      if (typeof item[field] === 'string') {
        const parsed = parseFloat(item[field]);
        if (!isNaN(parsed)) return parsed;
      }
    }

    return 0;
  }

  determineSignal(indicatorName, value) {
    switch (indicatorName) {
      case 'ahr999':
        if (value < 0.45) return 'strong_buy';
        if (value < 1.2) return 'buy';
        if (value > 5) return 'sell';
        return 'neutral';
      
      case 'rainbowChart':
        if (value < 0.2) return 'buy';
        if (value > 0.8) return 'sell';
        return 'neutral';
      
      case 'fearGreedIndex':
        if (value < 25) return 'buy'; // 极度恐惧
        if (value > 75) return 'sell'; // 极度贪婪
        return 'neutral';
      
      default:
        return 'neutral';
    }
  }

  determineRSISignal(value) {
    if (value < 30) return 'buy'; // 超卖
    if (value > 70) return 'sell'; // 超买
    return 'neutral';
  }

  calculateSignalStrength(indicatorName, value) {
    // 返回0-10的信号强度
    switch (indicatorName) {
      case 'ahr999':
        if (value < 0.45) return 9;
        if (value < 1.2) return 7;
        if (value > 5) return 8;
        return 3;
      
      case 'fearGreedIndex':
        const distance = Math.min(value, 100 - value);
        return Math.max(10 - distance / 5, 0);
      
      default:
        return 5;
    }
  }

  calculateRSISignalStrength(value) {
    if (value < 20 || value > 80) return 9;
    if (value < 30 || value > 70) return 7;
    return 3;
  }

  normalizeValue(indicatorName, value) {
    // 将指标值标准化到0-100
    switch (indicatorName) {
      case 'ahr999':
        return Math.min(value * 10, 100); // AHR999通常在0-10范围
      
      case 'fearGreedIndex':
        return value; // 已经是0-100
      
      default:
        return Math.min(Math.max(value * 100, 0), 100);
    }
  }

  normalizeStablecoinMarketCap(marketCap) {
    // 稳定币市值标准化 (假设最大值为2000亿)
    return Math.min(marketCap / 200000000000 * 100, 100);
  }

  getIndicatorDescription(name, value, signal) {
    switch (name) {
      case 'ahr999':
        return `AHR999指标: ${value.toFixed(3)} (${signal})`;
      case 'fearGreedIndex':
        return `恐惧贪婪指数: ${value.toFixed(0)} (${signal})`;
      default:
        return `${name}: ${value.toFixed(4)}`;
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

  formatNumber(num) {
    if (num >= 1e9) return `$${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `$${(num / 1e6).toFixed(2)}M`;
    if (num >= 1e3) return `$${(num / 1e3).toFixed(2)}K`;
    return `$${num.toFixed(2)}`;
  }

  async validateData(data) {
    const errors = [];

    for (const item of data) {
      // 验证必需字段
      if (!item.name || !item.category) {
        errors.push(`缺少必需字段: ${JSON.stringify(item)}`);
        continue;
      }

      // 验证数值字段
      if (typeof item.value !== 'number' || isNaN(item.value)) {
        errors.push(`指标值异常: ${item.name}, value: ${item.value}`);
      }

      if (typeof item.normalized_value !== 'number' || item.normalized_value < 0 || item.normalized_value > 100) {
        errors.push(`标准化值异常: ${item.name}, normalized_value: ${item.normalized_value}`);
      }

      if (typeof item.signal_strength !== 'number' || item.signal_strength < 0 || item.signal_strength > 10) {
        errors.push(`信号强度异常: ${item.name}, signal_strength: ${item.signal_strength}`);
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
      description: '收集各种宏观指标数据，包括AHR999、恐惧贪婪指数、RSI等',
      interval: `${this.interval}分钟`,
      supportedIndicators: this.indicators.map(i => i.name),
      apiEndpoints: this.indicators.map(i => `/api/indicator/${i.name}`),
      dataFields: [
        'name', 'symbol', 'value', 'normalized_value', 'signal', 
        'signal_strength', 'description', 'category', 'timestamp'
      ]
    };
  }
}