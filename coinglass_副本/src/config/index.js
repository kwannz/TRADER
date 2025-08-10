import dotenv from 'dotenv';
import path from 'path';

// 加载环境变量
dotenv.config();

export const config = {
  // CoinGlass API配置
  api: {
    key: process.env.COINGLASS_API_KEY || '51e89d90bf31473384e7e6c61b75afe7',
    baseUrl: process.env.COINGLASS_BASE_URL || 'https://open-api-v4.coinglass.com',
    timeout: parseInt(process.env.REQUEST_TIMEOUT) || 30000,
    maxRetries: parseInt(process.env.MAX_RETRIES) || 3,
    retryDelay: 1000, // 重试延迟（毫秒）
    rateLimit: {
      requests: 100, // 每分钟请求数
      period: 60 * 1000 // 时间窗口（毫秒）
    }
  },

  // JSON文件存储配置
  storage: {
    baseDir: process.env.STORAGE_BASE_DIR || './data',
    backupDir: process.env.STORAGE_BACKUP_DIR || './data/backup',
    compression: process.env.STORAGE_COMPRESSION || 'gzip',
    options: {
      prettyPrint: process.env.NODE_ENV === 'development',
      backup: true,
      maxFileSize: '100MB'
    }
  },

  // 日志配置
  logging: {
    level: process.env.LOG_LEVEL || 'info',
    file: process.env.LOG_FILE || 'logs/app.log',
    maxSize: '20m',
    maxFiles: 5,
    datePattern: 'YYYY-MM-DD'
  },

  // 数据收集配置
  collection: {
    intervalMinutes: parseInt(process.env.COLLECTION_INTERVAL_MINUTES) || 5,
    batchSize: 100, // 批量处理大小
    concurrency: 5, // 并发请求数
    retentionDays: parseInt(process.env.DATA_RETENTION_DAYS) || 365,
    
    // 各类数据的收集频率（分钟）
    frequencies: {
      contractMarket: 1,     // 合约市场数据
      fundingRate: 5,        // 资金费率
      openInterest: 5,       // 持仓量
      longShortRatio: 5,     // 多空比
      liquidation: 1,        // 爆仓数据
      orderbook: 2,          // 订单簿
      activeTrade: 2,        // 主动买卖
      spotMarket: 1,         // 现货市场
      option: 30,            // 期权数据
      etf: 60,               // ETF数据
      exchangeBalance: 60,   // 交易所余额
      indicator: 60,         // 技术指标
      fearGreedIndex: 1440   // 恐惧贪婪指数（每天）
    }
  },

  // API端点配置
  endpoints: {
    // 合约数据端点
    contract: {
      // 交易市场
      supportedCoins: '/api/futures/supported-coins',
      supportedExchanges: '/api/futures/supported-exchanges',
      coinMarkets: '/api/futures/coin-markets',
      pairMarkets: '/api/futures/pair-markets',
      coinChange: '/api/futures/coin-change',
      klineHistory: '/api/futures/kline-history',
      
      // 持仓
      openInterestHistory: '/api/futures/open-interest-history',
      aggregateOpenInterest: '/api/futures/aggregate-open-interest',
      stablecoinOpenInterest: '/api/futures/stablecoin-open-interest',
      coinOpenInterest: '/api/futures/coin-open-interest',
      exchangeOpenInterest: '/api/futures/exchange-open-interest',
      exchangeOpenInterestHistory: '/api/futures/exchange-open-interest-history',
      
      // 资金费率
      fundingRateHistory: '/api/futures/funding-rate-history',
      positionWeightedFunding: '/api/futures/position-weighted-funding',
      volumeWeightedFunding: '/api/futures/volume-weighted-funding',
      coinFundingRates: '/api/futures/coin-funding-rates',
      cumulativeFundingRates: '/api/futures/cumulative-funding-rates',
      fundingArbitrage: '/api/futures/funding-arbitrage',
      
      // 多空比
      accountLongShortRatio: '/api/futures/account-long-short-ratio',
      bigTraderLongShortRatio: '/api/futures/big-trader-long-short-ratio',
      positionLongShortRatio: '/api/futures/position-long-short-ratio',
      activeBuySellRatio: '/api/futures/active-buy-sell-ratio',
      
      // 爆仓
      pairLiquidationHistory: '/api/futures/pair-liquidation-history',
      coinLiquidationHistory: '/api/futures/coin-liquidation-history',
      exchangeLiquidationList: '/api/futures/exchange-liquidation-list',
      coinLiquidationList: '/api/futures/coin-liquidation-list',
      liquidationOrders: '/api/futures/liquidation-orders',
      pairLiquidationHeatmap: '/api/futures/pair-liquidation-heatmap',
      coinLiquidationHeatmap: '/api/futures/coin-liquidation-heatmap',
      liquidationMap: '/api/futures/liquidation-map',
      
      // 订单簿
      orderbookHistory: '/api/futures/orderbook-history',
      aggregateOrderbook: '/api/futures/aggregate-orderbook',
      orderbookHeatmap: '/api/futures/orderbook-heatmap',
      largeOrderbook: '/api/futures/large-orderbook',
      largeOrderbookHistory: '/api/futures/large-orderbook-history',
      
      // 主动买卖
      pairActiveTrade: '/api/futures/pair-active-trade',
      coinActiveTrade: '/api/futures/coin-active-trade'
    },

    // 现货数据端点
    spot: {
      supportedCoins: '/api/spot/supported-coins',
      supportedExchanges: '/api/spot/supported-exchanges',
      coinMarkets: '/api/spot/coin-markets',
      pairMarkets: '/api/spot/pair-markets',
      priceHistory: '/api/spot/price-history',
      orderbookHistory: '/api/spot/orderbook-history',
      aggregateOrderbook: '/api/spot/aggregate-orderbook',
      orderbookHeatmap: '/api/spot/orderbook-heatmap',
      largeOrderbook: '/api/spot/large-orderbook',
      largeOrderbookHistory: '/api/spot/large-orderbook-history',
      pairActiveTrade: '/api/spot/pair-active-trade',
      coinActiveTrade: '/api/spot/coin-active-trade'
    },

    // 期权数据端点
    option: {
      maxPain: '/api/option/max-pain',
      optionData: '/api/option/option-data',
      exchangeOpenInterest: '/api/option/exchange-open-interest',
      exchangeVolume: '/api/option/exchange-volume'
    },

    // 链上数据端点
    onchain: {
      exchangeReserves: '/api/onchain/exchange-reserves',
      exchangeBalance: '/api/onchain/exchange-balance',
      exchangeBalanceChart: '/api/onchain/exchange-balance-chart',
      erc20Transfers: '/api/onchain/erc20-transfers'
    },

    // ETF数据端点
    etf: {
      // 比特币ETF
      bitcoinEtfList: '/api/etf/bitcoin-etf-list',
      hongkongEtfFlow: '/api/etf/hongkong-etf-flow',
      etfNetAssets: '/api/etf/etf-net-assets',
      etfFlow: '/api/etf/etf-flow',
      etfPremium: '/api/etf/etf-premium',
      bitcoinEtfHistory: '/api/etf/bitcoin-etf-history',
      etfPrice: '/api/etf/etf-price',
      bitcoinEtfDetail: '/api/etf/bitcoin-etf-detail',
      etfAum: '/api/etf/etf-aum',
      
      // 以太坊ETF
      ethereumEtfNetAssets: '/api/etf/ethereum-etf-net-assets',
      ethereumEtfList: '/api/etf/ethereum-etf-list',
      ethereumEtfFlow: '/api/etf/ethereum-etf-flow',
      
      // 灰度基金
      grayscaleHoldings: '/api/etf/grayscale-holdings',
      grayscalePremium: '/api/etf/grayscale-premium'
    },

    // 指标端点
    indicator: {
      // 合约指标
      rsiList: '/api/indicator/rsi-list',
      contractBasis: '/api/indicator/contract-basis',
      
      // 现货指标
      coinbasePremium: '/api/indicator/coinbase-premium',
      bitfinexMargin: '/api/indicator/bitfinex-margin',
      lendingRates: '/api/indicator/lending-rates',
      
      // 宏观指标
      ahr999: '/api/indicator/ahr999',
      puellMultiple: '/api/indicator/puell-multiple',
      bitcoinSupply: '/api/indicator/bitcoin-supply',
      piCycleTop: '/api/indicator/pi-cycle-top',
      goldenRatio: '/api/indicator/golden-ratio',
      profitDays: '/api/indicator/profit-days',
      rainbowChart: '/api/indicator/rainbow-chart',
      fearGreedIndex: '/api/indicator/fear-greed-index',
      stablecoinMarketCap: '/api/indicator/stablecoin-market-cap',
      bubbleIndex: '/api/indicator/bubble-index',
      bullMarketSignals: '/api/indicator/bull-market-signals',
      twoYearMA: '/api/indicator/two-year-ma',
      weeklyMAHeatmap: '/api/indicator/weekly-ma-heatmap'
    }
  },

  // 系统配置
  system: {
    timezone: 'Asia/Shanghai',
    dateFormat: 'YYYY-MM-DD HH:mm:ss',
    maxMemoryUsage: '1GB',
    healthCheckInterval: 60000, // 健康检查间隔（毫秒）
  }
};

// 验证配置
export function validateConfig() {
  const required = [
    'api.key',
    'api.baseUrl',
    'storage.baseDir'
  ];

  for (const key of required) {
    const value = key.split('.').reduce((obj, k) => obj?.[k], config);
    if (!value) {
      throw new Error(`缺少必需的配置项: ${key}`);
    }
  }

  console.log('配置验证通过');
  return true;
}

// 获取特定配置
export function getConfig(key) {
  return key.split('.').reduce((obj, k) => obj?.[k], config);
}

// 开发/生产环境配置覆盖
if (process.env.NODE_ENV === 'production') {
  config.logging.level = 'warn';
  config.collection.intervalMinutes = 1;
} else if (process.env.NODE_ENV === 'development') {
  config.logging.level = 'debug';
  config.collection.intervalMinutes = 10;
}

export default config; 