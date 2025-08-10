/**
 * CoinGlass 数据模型定义
 * 基于 CoinGlass API v4 的所有数据类型
 */

// 合约市场数据
const contractMarketSchema = {
  symbol: String,
  exchange: String,
  price: Number,
  price_change_24h: Number,
  price_change_percent_24h: Number,
  volume_24h: Number,
  volume_change_percent_24h: Number,
  market_cap: Number,
  funding_rate: Number,
  open_interest: Number,
  oi_change_24h: Number,
  timestamp: Date,
  created_at: Date
};

// 持仓数据（Open Interest）
const openInterestSchema = {
  symbol: String,
  exchange: String,
  open_interest_usd: Number,
  open_interest_native: Number,
  cash_margin_oi: Number,
  crypto_margin_oi: Number,
  timestamp: Date,
  timeframe: String, // '1h', '4h', '24h'
  created_at: Date
};

// 资金费率数据
const fundingRateSchema = {
  symbol: String,
  exchange: String,
  funding_rate: Number,
  next_funding_time: Date,
  funding_rate_8h_avg: Number,
  oi_weighted_funding: Number,
  volume_weighted_funding: Number,
  timestamp: Date,
  created_at: Date
};

// 多空比数据
const longShortRatioSchema = {
  symbol: String,
  exchange: String,
  long_ratio: Number,
  short_ratio: Number,
  long_account_ratio: Number,
  short_account_ratio: Number,
  taker_buy_ratio: Number,
  taker_sell_ratio: Number,
  timestamp: Date,
  created_at: Date
};

// 爆仓数据
const liquidationSchema = {
  symbol: String,
  exchange: String,
  side: String, // 'long' or 'short'
  amount_usd: Number,
  price: Number,
  timestamp: Date,
  created_at: Date
};

// 爆仓历史统计
const liquidationHistorySchema = {
  symbol: String,
  exchange: String,
  timeframe: String, // '5m', '1h', '4h', '24h'
  long_liquidation_usd: Number,
  short_liquidation_usd: Number,
  total_liquidation_usd: Number,
  timestamp: Date,
  created_at: Date
};

// 主动买卖数据
const activeBuySellSchema = {
  symbol: String,
  exchange: String,
  buy_volume: Number,
  sell_volume: Number,
  buy_ratio: Number,
  sell_ratio: Number,
  net_flow: Number,
  timestamp: Date,
  timeframe: String,
  created_at: Date
};

// ETF数据
const etfSchema = {
  name: String,
  type: String, // 'bitcoin', 'ethereum'
  flows_usd: Number,
  net_assets: Number,
  premium_discount: Number,
  price: Number,
  volume: Number,
  date: Date,
  created_at: Date
};

// 恐惧贪婪指数
const fearGreedIndexSchema = {
  value: Number,
  classification: String,
  timestamp: Date,
  created_at: Date
};

// 技术指标
const indicatorSchema = {
  name: String, // 'RSI', 'AHR999', 'Rainbow', etc.
  symbol: String,
  value: Number,
  signal: String,
  description: String,
  timestamp: Date,
  created_at: Date
};

// 稳定币市值
const stablecoinMarketCapSchema = {
  coin: String, // 'USDT', 'USDC', 'BUSD'
  market_cap: Number,
  market_cap_change_24h: Number,
  market_cap_change_percent_24h: Number,
  timestamp: Date,
  created_at: Date
};

// 交易所余额
const exchangeBalanceSchema = {
  exchange: String,
  symbol: String,
  balance: Number,
  balance_change_24h: Number,
  balance_change_percent_24h: Number,
  in_flow_24h: Number,
  out_flow_24h: Number,
  net_flow_24h: Number,
  timestamp: Date,
  created_at: Date
};

// 期权数据
const optionSchema = {
  symbol: String,
  exchange: String,
  strike_price: Number,
  expiry_date: Date,
  option_type: String, // 'call', 'put'
  open_interest: Number,
  volume: Number,
  implied_volatility: Number,
  max_pain: Number,
  timestamp: Date,
  created_at: Date
};

// 大额订单
const largeOrderSchema = {
  symbol: String,
  exchange: String,
  side: String, // 'buy', 'sell'
  amount: Number,
  price: Number,
  amount_usd: Number,
  timestamp: Date,
  created_at: Date
};

// 订单簿数据
const orderbookSchema = {
  symbol: String,
  exchange: String,
  bids: [{
    price: Number,
    amount: Number,
    total: Number
  }],
  asks: [{
    price: Number,
    amount: Number,
    total: Number
  }],
  spread: Number,
  timestamp: Date,
  created_at: Date
};

// 巨鲸持仓警报
const whalePositionSchema = {
  exchange: String,
  symbol: String,
  trader_id: String,
  position_type: String, // 'long', 'short'
  position_size: Number,
  entry_price: Number,
  current_price: Number,
  pnl: Number,
  leverage: Number,
  timestamp: Date,
  created_at: Date
};

// 价格历史数据
const priceHistorySchema = {
  symbol: String,
  exchange: String,
  timeframe: String, // '1m', '5m', '1h', '4h', '1d'
  open: Number,
  high: Number,
  low: Number,
  close: Number,
  volume: Number,
  timestamp: Date,
  created_at: Date
};

// 支持的币种列表
const supportedCoinsSchema = {
  symbol: String,
  name: String,
  category: String,
  is_active: Boolean,
  supported_exchanges: [String],
  last_updated: Date,
  created_at: Date
};

// 聚合持仓数据
const aggregatedOpenInterestSchema = {
  symbol: String,
  total_open_interest_usd: Number,
  total_open_interest_native: Number,
  exchanges: [{
    exchange: String,
    open_interest_usd: Number,
    open_interest_native: Number,
    percentage: Number
  }],
  dominant_exchange: String,
  timestamp: Date,
  created_at: Date
};

// 爆仓热力图数据
const liquidationHeatmapSchema = {
  symbol: String,
  exchange: String,
  price_level: Number,
  long_liquidation_usd: Number,
  short_liquidation_usd: Number,
  total_liquidation_usd: Number,
  cumulative_long: Number,
  cumulative_short: Number,
  timestamp: Date,
  created_at: Date
};

// 宏观指标数据
const macroIndicatorSchema = {
  name: String, // 'ahr999', 'rainbowChart', 'puellMultiple', etc.
  symbol: String,
  value: Number,
  normalized_value: Number, // 0-100标准化值
  signal: String, // 'buy', 'sell', 'neutral'
  signal_strength: Number, // 信号强度 0-10
  description: String,
  category: String, // 'macro', 'onchain', 'technical'
  timestamp: Date,
  created_at: Date
};

// 现货市场数据
const spotMarketSchema = {
  symbol: String,
  exchange: String,
  price: Number,
  price_change_24h: Number,
  price_change_percent_24h: Number,
  volume_24h: Number,
  volume_change_percent_24h: Number,
  high_24h: Number,
  low_24h: Number,
  market_cap: Number,
  timestamp: Date,
  created_at: Date
};

// 期权聚合数据
const optionAggregateSchema = {
  symbol: String,
  expiry_date: Date,
  max_pain: Number,
  put_call_ratio: Number,
  total_call_oi: Number,
  total_put_oi: Number,
  total_call_volume: Number,
  total_put_volume: Number,
  implied_volatility: Number,
  delta_neutral_level: Number,
  timestamp: Date,
  created_at: Date
};

// 集合索引配置
const indexConfigs = {
  contract_markets: [
    { keys: { symbol: 1, exchange: 1, timestamp: -1 }, options: { background: true } },
    { keys: { timestamp: -1 }, options: { background: true } },
    { keys: { created_at: 1 }, options: { expireAfterSeconds: 365 * 24 * 60 * 60 } }
  ],
  open_interest: [
    { keys: { symbol: 1, exchange: 1, timestamp: -1 }, options: { background: true } },
    { keys: { timestamp: -1 }, options: { background: true } },
    { keys: { created_at: 1 }, options: { expireAfterSeconds: 365 * 24 * 60 * 60 } }
  ],
  funding_rates: [
    { keys: { symbol: 1, exchange: 1, timestamp: -1 }, options: { background: true } },
    { keys: { timestamp: -1 }, options: { background: true } },
    { keys: { created_at: 1 }, options: { expireAfterSeconds: 365 * 24 * 60 * 60 } }
  ],
  long_short_ratios: [
    { keys: { symbol: 1, exchange: 1, timestamp: -1 }, options: { background: true } },
    { keys: { timestamp: -1 }, options: { background: true } },
    { keys: { created_at: 1 }, options: { expireAfterSeconds: 365 * 24 * 60 * 60 } }
  ],
  liquidations: [
    { keys: { symbol: 1, exchange: 1, timestamp: -1 }, options: { background: true } },
    { keys: { timestamp: -1 }, options: { background: true } },
    { keys: { side: 1, timestamp: -1 }, options: { background: true } },
    { keys: { created_at: 1 }, options: { expireAfterSeconds: 90 * 24 * 60 * 60 } }
  ],
  liquidation_history: [
    { keys: { symbol: 1, timeframe: 1, timestamp: -1 }, options: { background: true } },
    { keys: { timestamp: -1 }, options: { background: true } },
    { keys: { created_at: 1 }, options: { expireAfterSeconds: 365 * 24 * 60 * 60 } }
  ],
  active_buy_sell: [
    { keys: { symbol: 1, exchange: 1, timestamp: -1 }, options: { background: true } },
    { keys: { timestamp: -1 }, options: { background: true } },
    { keys: { created_at: 1 }, options: { expireAfterSeconds: 365 * 24 * 60 * 60 } }
  ],
  etf_data: [
    { keys: { name: 1, date: -1 }, options: { background: true } },
    { keys: { type: 1, date: -1 }, options: { background: true } },
    { keys: { created_at: 1 }, options: { expireAfterSeconds: 365 * 24 * 60 * 60 } }
  ],
  fear_greed_index: [
    { keys: { timestamp: -1 }, options: { background: true } },
    { keys: { created_at: 1 }, options: { expireAfterSeconds: 365 * 24 * 60 * 60 } }
  ],
  indicators: [
    { keys: { name: 1, symbol: 1, timestamp: -1 }, options: { background: true } },
    { keys: { timestamp: -1 }, options: { background: true } },
    { keys: { created_at: 1 }, options: { expireAfterSeconds: 365 * 24 * 60 * 60 } }
  ],
  stablecoin_market_cap: [
    { keys: { coin: 1, timestamp: -1 }, options: { background: true } },
    { keys: { timestamp: -1 }, options: { background: true } },
    { keys: { created_at: 1 }, options: { expireAfterSeconds: 365 * 24 * 60 * 60 } }
  ],
  exchange_balances: [
    { keys: { exchange: 1, symbol: 1, timestamp: -1 }, options: { background: true } },
    { keys: { timestamp: -1 }, options: { background: true } },
    { keys: { created_at: 1 }, options: { expireAfterSeconds: 365 * 24 * 60 * 60 } }
  ],
  options: [
    { keys: { symbol: 1, exchange: 1, timestamp: -1 }, options: { background: true } },
    { keys: { expiry_date: 1, timestamp: -1 }, options: { background: true } },
    { keys: { created_at: 1 }, options: { expireAfterSeconds: 365 * 24 * 60 * 60 } }
  ],
  large_orders: [
    { keys: { symbol: 1, exchange: 1, timestamp: -1 }, options: { background: true } },
    { keys: { timestamp: -1 }, options: { background: true } },
    { keys: { amount_usd: -1, timestamp: -1 }, options: { background: true } },
    { keys: { created_at: 1 }, options: { expireAfterSeconds: 30 * 24 * 60 * 60 } }
  ],
  orderbook: [
    { keys: { symbol: 1, exchange: 1, timestamp: -1 }, options: { background: true } },
    { keys: { timestamp: -1 }, options: { background: true } },
    { keys: { created_at: 1 }, options: { expireAfterSeconds: 7 * 24 * 60 * 60 } }
  ],
  whale_positions: [
    { keys: { exchange: 1, symbol: 1, timestamp: -1 }, options: { background: true } },
    { keys: { trader_id: 1, timestamp: -1 }, options: { background: true } },
    { keys: { created_at: 1 }, options: { expireAfterSeconds: 90 * 24 * 60 * 60 } }
  ],
  price_history: [
    { keys: { symbol: 1, exchange: 1, timeframe: 1, timestamp: -1 }, options: { background: true } },
    { keys: { timestamp: -1 }, options: { background: true } },
    { keys: { created_at: 1 }, options: { expireAfterSeconds: 365 * 24 * 60 * 60 } }
  ],
  supported_coins: [
    { keys: { symbol: 1 }, options: { unique: true, background: true } },
    { keys: { is_active: 1 }, options: { background: true } }
  ],
  aggregated_open_interest: [
    { keys: { symbol: 1, timestamp: -1 }, options: { background: true } },
    { keys: { timestamp: -1 }, options: { background: true } },
    { keys: { created_at: 1 }, options: { expireAfterSeconds: 365 * 24 * 60 * 60 } }
  ],
  liquidation_heatmaps: [
    { keys: { symbol: 1, exchange: 1, timestamp: -1 }, options: { background: true } },
    { keys: { price_level: 1, timestamp: -1 }, options: { background: true } },
    { keys: { timestamp: -1 }, options: { background: true } },
    { keys: { created_at: 1 }, options: { expireAfterSeconds: 30 * 24 * 60 * 60 } }
  ],
  macro_indicators: [
    { keys: { name: 1, timestamp: -1 }, options: { background: true } },
    { keys: { category: 1, timestamp: -1 }, options: { background: true } },
    { keys: { signal: 1, timestamp: -1 }, options: { background: true } },
    { keys: { created_at: 1 }, options: { expireAfterSeconds: 365 * 24 * 60 * 60 } }
  ],
  spot_markets: [
    { keys: { symbol: 1, exchange: 1, timestamp: -1 }, options: { background: true } },
    { keys: { timestamp: -1 }, options: { background: true } },
    { keys: { created_at: 1 }, options: { expireAfterSeconds: 365 * 24 * 60 * 60 } }
  ],
  option_aggregates: [
    { keys: { symbol: 1, expiry_date: 1, timestamp: -1 }, options: { background: true } },
    { keys: { timestamp: -1 }, options: { background: true } },
    { keys: { created_at: 1 }, options: { expireAfterSeconds: 365 * 24 * 60 * 60 } }
  ]
};

module.exports = {
  schemas: {
    contractMarketSchema,
    openInterestSchema,
    fundingRateSchema,
    longShortRatioSchema,
    liquidationSchema,
    liquidationHistorySchema,
    activeBuySellSchema,
    etfSchema,
    fearGreedIndexSchema,
    indicatorSchema,
    stablecoinMarketCapSchema,
    exchangeBalanceSchema,
    optionSchema,
    largeOrderSchema,
    orderbookSchema,
    whalePositionSchema,
    priceHistorySchema,
    supportedCoinsSchema,
    aggregatedOpenInterestSchema,
    liquidationHeatmapSchema,
    macroIndicatorSchema,
    spotMarketSchema,
    optionAggregateSchema
  },
  indexConfigs
}; 