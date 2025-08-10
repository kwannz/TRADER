"""
Crypto Data Reader
加密货币专用数据读取器 - 整合多交易所数据源
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
import json

# 异步HTTP客户端
try:
    import aiohttp
    import ccxt.async_support as ccxt
except ImportError:
    aiohttp = None
    ccxt = None

from .unified_data_reader import DatabaseHandler


class CryptoExchangeConnector:
    """加密货币交易所连接器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("CryptoExchangeConnector")
        
        # 支持的交易所
        self.exchanges = {}
        self._initialize_exchanges()
    
    def _initialize_exchanges(self):
        """初始化交易所连接"""
        if not ccxt:
            self.logger.warning("ccxt not available, exchange data will be simulated")
            return
        
        exchange_configs = self.config.get('exchanges', {})
        
        # Binance
        if 'binance' in exchange_configs:
            self.exchanges['binance'] = ccxt.binance({
                'apiKey': exchange_configs['binance'].get('api_key'),
                'secret': exchange_configs['binance'].get('secret'),
                'sandbox': exchange_configs['binance'].get('sandbox', True),
                'rateLimit': 1000,
                'enableRateLimit': True
            })
        
        # Coinbase Pro
        if 'coinbase' in exchange_configs:
            self.exchanges['coinbase'] = ccxt.coinbase({
                'apiKey': exchange_configs['coinbase'].get('api_key'),
                'secret': exchange_configs['coinbase'].get('secret'),
                'password': exchange_configs['coinbase'].get('passphrase'),
                'rateLimit': 1000,
                'enableRateLimit': True
            })
        
        # OKX
        if 'okx' in exchange_configs:
            self.exchanges['okx'] = ccxt.okx({
                'apiKey': exchange_configs['okx'].get('api_key'),
                'secret': exchange_configs['okx'].get('secret'),
                'password': exchange_configs['okx'].get('passphrase'),
                'rateLimit': 1000,
                'enableRateLimit': True
            })
        
        self.logger.info(f"Initialized {len(self.exchanges)} exchange connections")
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         limit: int = 1000) -> pd.DataFrame:
        """
        获取OHLCV数据
        
        Args:
            symbol: 交易对符号 (如 'BTC/USDT')
            timeframe: 时间框架 ('1m', '5m', '15m', '1h', '4h', '1d')
            start_time: 开始时间
            end_time: 结束时间
            limit: 数据条数限制
            
        Returns:
            OHLCV数据DataFrame
        """
        if not self.exchanges:
            return self._generate_mock_crypto_data(symbol, timeframe, start_time, end_time, limit)
        
        try:
            # 优先使用Binance
            exchange = self.exchanges.get('binance') or list(self.exchanges.values())[0]
            
            # 转换时间为时间戳
            since = None
            if start_time:
                since = int(start_time.timestamp() * 1000)
            
            # 获取数据
            ohlcv_data = await exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                limit=limit
            )
            
            # 转换为DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # 过滤时间范围
            if end_time:
                df = df[df.index <= end_time]
            
            self.logger.info(f"Fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching {symbol} data: {str(e)}")
            return self._generate_mock_crypto_data(symbol, timeframe, start_time, end_time, limit)
    
    def _generate_mock_crypto_data(self, symbol: str, timeframe: str,
                                  start_time: Optional[datetime],
                                  end_time: Optional[datetime],
                                  limit: int) -> pd.DataFrame:
        """生成模拟加密货币数据"""
        self.logger.info(f"Generating mock data for {symbol}")
        
        # 时间范围
        if not start_time:
            start_time = datetime.now() - timedelta(days=30)
        if not end_time:
            end_time = datetime.now()
        
        # 时间频率转换
        freq_map = {
            '1m': '1T',
            '5m': '5T', 
            '15m': '15T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        freq = freq_map.get(timeframe, '1H')
        
        # 生成时间索引
        time_index = pd.date_range(start=start_time, end=end_time, freq=freq)[:limit]
        
        # 基础价格设定
        base_prices = {
            'BTC/USDT': 45000,
            'ETH/USDT': 3000,
            'BNB/USDT': 400,
            'ADA/USDT': 0.5,
            'SOL/USDT': 100,
            'MATIC/USDT': 1.2,
            'AVAX/USDT': 35,
            'ATOM/USDT': 12
        }
        
        base_price = base_prices.get(symbol, 100)
        
        # 生成价格序列（几何布朗运动 + 跳跃过程）
        np.random.seed(42)
        n_points = len(time_index)
        
        # 基础参数
        dt = 1/24 if timeframe == '1h' else 1/(24*60)  # 时间步长
        mu = 0.0001  # 漂移率（略微上涨）
        sigma = 0.02 if symbol.startswith('BTC') else 0.03  # 波动率
        
        # 几何布朗运动
        dW = np.random.normal(0, np.sqrt(dt), n_points)
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * dW
        
        # 添加跳跃（加密市场特色）
        jump_intensity = 0.05  # 5%概率发生跳跃
        jump_size = np.random.normal(0, 0.1, n_points)  # 跳跃大小
        jumps = np.random.binomial(1, jump_intensity, n_points) * jump_size
        
        # 价格路径
        log_returns = drift + diffusion + jumps
        log_prices = np.log(base_price) + np.cumsum(log_returns)
        close_prices = np.exp(log_prices)
        
        # 生成OHLC数据
        data = []
        for i, price in enumerate(close_prices):
            if i == 0:
                open_price = base_price
            else:
                open_price = close_prices[i-1]
            
            # 日内波动
            intraday_vol = np.random.uniform(0.005, 0.02)  # 0.5%-2%日内波动
            high_price = max(open_price, price) * (1 + intraday_vol)
            low_price = min(open_price, price) * (1 - intraday_vol)
            
            # 成交量（对数正态分布）
            base_volume = 1000000 if symbol.startswith('BTC') else 500000
            volume = np.random.lognormal(np.log(base_volume), 0.5)
            
            data.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': price,
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=time_index)
        return df
    
    async def fetch_funding_rates(self, symbol: str, start_time: Optional[datetime] = None,
                                end_time: Optional[datetime] = None) -> pd.Series:
        """
        获取资金费率数据
        
        Args:
            symbol: 永续合约符号
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            资金费率时间序列
        """
        if not self.exchanges:
            return self._generate_mock_funding_rates(symbol, start_time, end_time)
        
        try:
            exchange = self.exchanges.get('binance')
            if not exchange:
                return self._generate_mock_funding_rates(symbol, start_time, end_time)
            
            # 获取资金费率历史
            funding_history = await exchange.fetch_funding_rate_history(symbol)
            
            # 转换为Series
            funding_df = pd.DataFrame(funding_history)
            funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'], unit='ms')
            funding_series = pd.Series(
                funding_df['fundingRate'].values,
                index=funding_df['timestamp'],
                name='funding_rate'
            )
            
            # 过滤时间范围
            if start_time:
                funding_series = funding_series[funding_series.index >= start_time]
            if end_time:
                funding_series = funding_series[funding_series.index <= end_time]
            
            return funding_series
            
        except Exception as e:
            self.logger.error(f"Error fetching funding rates for {symbol}: {str(e)}")
            return self._generate_mock_funding_rates(symbol, start_time, end_time)
    
    def _generate_mock_funding_rates(self, symbol: str, start_time: Optional[datetime],
                                   end_time: Optional[datetime]) -> pd.Series:
        """生成模拟资金费率数据"""
        if not start_time:
            start_time = datetime.now() - timedelta(days=30)
        if not end_time:
            end_time = datetime.now()
        
        # 8小时一次的资金费率
        time_index = pd.date_range(start=start_time, end=end_time, freq='8H')
        
        # 生成资金费率（通常在-0.01到0.01之间）
        np.random.seed(hash(symbol) % 1000)
        n_points = len(time_index)
        
        # 基础资金费率趋势
        base_rate = 0.0001  # 0.01%基础费率
        
        # 随机游走
        innovations = np.random.normal(0, 0.0005, n_points)  # 0.05%标准差
        funding_rates = base_rate + np.cumsum(innovations)
        
        # 限制在合理范围内
        funding_rates = np.clip(funding_rates, -0.01, 0.01)
        
        return pd.Series(funding_rates, index=time_index, name='funding_rate')
    
    async def close_connections(self):
        """关闭交易所连接"""
        for exchange_name, exchange in self.exchanges.items():
            try:
                await exchange.close()
                self.logger.info(f"Closed {exchange_name} connection")
            except Exception as e:
                self.logger.error(f"Error closing {exchange_name}: {str(e)}")


class CryptoDataReader:
    """加密货币数据读取器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("CryptoDataReader")
        
        # 初始化组件
        try:
            self.db_handler = DatabaseHandler(config)
        except Exception as e:
            self.logger.warning(f"Database handler initialization failed: {e}")
            self.db_handler = None
            
        self.exchange_connector = CryptoExchangeConnector(config)
        
        # 支持的加密货币列表
        self.supported_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
            'MATIC/USDT', 'AVAX/USDT', 'ATOM/USDT', 'DOT/USDT', 'LINK/USDT'
        ]
        
        self.logger.info(f"CryptoDataReader initialized with {len(self.supported_symbols)} supported symbols")
    
    async def get_crypto_ohlcv(self, symbols: List[str], start_date: str, end_date: str,
                              timeframe: str = '1h') -> Dict[str, pd.Series]:
        """
        获取加密货币OHLCV数据
        
        Args:
            symbols: 交易对列表
            start_date: 开始日期
            end_date: 结束日期
            timeframe: 时间框架
            
        Returns:
            多币种OHLCV数据字典
        """
        start_time = pd.to_datetime(start_date)
        end_time = pd.to_datetime(end_date)
        
        crypto_data = {}
        
        # 并发获取多个币种数据
        tasks = []
        for symbol in symbols:
            task = self.exchange_connector.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )
            tasks.append((symbol, task))
        
        # 执行并发任务
        for symbol, task in tasks:
            try:
                df = await task
                
                if not df.empty:
                    # 转换为MultiIndex Series格式
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in df.columns:
                            # 创建MultiIndex
                            multi_index = pd.MultiIndex.from_product(
                                [df.index, [symbol.replace('/', '')]],
                                names=['date', 'symbol']
                            )
                            
                            series_data = []
                            for timestamp in df.index:
                                series_data.append(df.loc[timestamp, col])
                            
                            crypto_data[f"{symbol.replace('/', '')}_{col}"] = pd.Series(
                                series_data, 
                                index=pd.MultiIndex.from_tuples(
                                    [(timestamp, symbol.replace('/', '')) for timestamp in df.index],
                                    names=['date', 'symbol']
                                )
                            )
                
                self.logger.info(f"Successfully loaded {symbol} data: {len(df)} records")
                
            except Exception as e:
                self.logger.error(f"Failed to load {symbol}: {str(e)}")
        
        return crypto_data
    
    async def get_crypto_market_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """
        获取综合市场数据
        
        Args:
            symbols: 币种列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            综合市场数据
        """
        market_data = {
            'ohlcv': {},
            'funding_rates': {},
            'market_cap': {},
            'social_sentiment': {}
        }
        
        # 1. 获取OHLCV数据
        ohlcv_data = await self.get_crypto_ohlcv(symbols, start_date, end_date)
        market_data['ohlcv'] = ohlcv_data
        
        # 2. 获取资金费率数据（永续合约）
        start_time = pd.to_datetime(start_date)
        end_time = pd.to_datetime(end_date)
        
        for symbol in symbols:
            if symbol.endswith('/USDT'):  # 永续合约
                try:
                    funding_rates = await self.exchange_connector.fetch_funding_rates(
                        symbol, start_time, end_time
                    )
                    market_data['funding_rates'][symbol] = funding_rates
                except Exception as e:
                    self.logger.warning(f"Could not fetch funding rates for {symbol}: {str(e)}")
        
        # 3. 生成模拟市值数据
        market_data['market_cap'] = self._generate_market_cap_data(symbols, start_date, end_date)
        
        # 4. 生成模拟社交情绪数据
        market_data['social_sentiment'] = self._generate_sentiment_data(symbols, start_date, end_date)
        
        return market_data
    
    def _generate_market_cap_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.Series]:
        """生成模拟市值数据"""
        market_cap_data = {}
        
        # 市值基准（十亿美元）
        market_cap_base = {
            'BTC': 800,
            'ETH': 400,
            'BNB': 60,
            'ADA': 15,
            'SOL': 40,
            'MATIC': 8,
            'AVAX': 12,
            'ATOM': 3
        }
        
        dates = pd.date_range(start=start_date, end=end_date, freq='1D')
        
        for symbol in symbols:
            base_symbol = symbol.replace('/USDT', '')
            base_cap = market_cap_base.get(base_symbol, 1) * 1e9  # 转换为美元
            
            # 随机游走生成市值变化
            np.random.seed(hash(symbol) % 1000)
            returns = np.random.normal(0, 0.03, len(dates))  # 3%日波动
            market_caps = base_cap * np.exp(np.cumsum(returns))
            
            market_cap_data[base_symbol] = pd.Series(market_caps, index=dates)
        
        return market_cap_data
    
    def _generate_sentiment_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.Series]:
        """生成模拟社交情绪数据"""
        sentiment_data = {}
        
        dates = pd.date_range(start=start_date, end=end_date, freq='1D')
        
        for symbol in symbols:
            base_symbol = symbol.replace('/USDT', '')
            
            # 情绪分数生成（-1到1）
            np.random.seed(hash(symbol) % 1000 + 1000)
            
            # 基础情绪趋势
            base_sentiment = 0.1  # 略微积极
            
            # 添加随机波动和趋势
            noise = np.random.normal(0, 0.2, len(dates))
            trend = np.sin(np.arange(len(dates)) * 2 * np.pi / 30) * 0.3  # 30天周期
            
            sentiments = np.clip(base_sentiment + trend + noise, -1, 1)
            
            sentiment_data[base_symbol] = pd.Series(sentiments, index=dates)
        
        return sentiment_data
    
    def get_supported_symbols(self) -> List[str]:
        """获取支持的交易对列表"""
        return self.supported_symbols.copy()
    
    def get_crypto_pairs_by_base(self, base_currency: str = 'USDT') -> List[str]:
        """
        获取特定基准货币的交易对
        
        Args:
            base_currency: 基准货币 (如 'USDT', 'BTC', 'ETH')
            
        Returns:
            交易对列表
        """
        return [symbol for symbol in self.supported_symbols if symbol.endswith(f'/{base_currency}')]
    
    async def close(self):
        """关闭连接"""
        await self.exchange_connector.close_connections()


class CryptoMarketDataManager:
    """加密货币市场数据管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("CryptoMarketDataManager")
        self.crypto_reader = CryptoDataReader(config)
    
    async def get_multi_timeframe_data(self, symbols: List[str], start_date: str, end_date: str,
                                     timeframes: List[str] = ['1h', '4h', '1d']) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        获取多时间框架数据
        
        Args:
            symbols: 币种列表
            start_date: 开始日期
            end_date: 结束日期
            timeframes: 时间框架列表
            
        Returns:
            多时间框架数据字典
        """
        multi_tf_data = {}
        
        for timeframe in timeframes:
            self.logger.info(f"Fetching {timeframe} data...")
            
            # 获取该时间框架的数据
            tf_data = await self.crypto_reader.get_crypto_ohlcv(
                symbols, start_date, end_date, timeframe
            )
            
            multi_tf_data[timeframe] = tf_data
        
        return multi_tf_data
    
    async def get_cross_exchange_data(self, symbol: str, exchanges: List[str],
                                    start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        获取跨交易所数据用于套利分析
        
        Args:
            symbol: 交易对
            exchanges: 交易所列表
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            跨交易所数据字典
        """
        cross_exchange_data = {}
        
        # 这里可以扩展支持多交易所
        # 目前先返回模拟数据
        for exchange in exchanges:
            self.logger.info(f"Fetching {symbol} data from {exchange}")
            
            # 生成略有差异的模拟数据
            base_data = await self.crypto_reader.exchange_connector.fetch_ohlcv(
                symbol, '1h', pd.to_datetime(start_date), pd.to_datetime(end_date)
            )
            
            # 添加交易所特定的价格差异
            price_adjustment = np.random.normal(1, 0.002, len(base_data))  # 0.2%差异
            adjusted_data = base_data.copy()
            for col in ['open', 'high', 'low', 'close']:
                adjusted_data[col] *= price_adjustment
            
            cross_exchange_data[exchange] = adjusted_data
        
        return cross_exchange_data
    
    async def close(self):
        """关闭数据管理器"""
        await self.crypto_reader.close()


# 创建全局实例
crypto_data_manager = CryptoMarketDataManager({})