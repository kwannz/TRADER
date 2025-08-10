"""
真实数据管理器
集成OKX和Binance的实时数据流和历史数据
替代模拟数据系统
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_logger import get_logger, LogCategory
from .websocket_client import OKXWebSocketClient, BinanceWebSocketClient, WebSocketManager
from .enhanced_websocket_client import enhanced_binance_client
from .historical_data_fetcher import HistoricalDataManager
from .data_manager import data_manager
from .coinglass_collector import coinglass_collector_manager
from .coinglass_analyzer import coinglass_analyzer
from .data_validator import market_data_validator, ValidationLevel
from .optimized_data_pipeline import optimized_pipeline

logger = get_logger()

class RealDataManager:
    """真实数据管理器"""
    
    def __init__(self):
        self.websocket_manager = WebSocketManager()
        self.historical_manager = HistoricalDataManager()
        
        # 交易对配置
        self.trading_pairs = {
            "BTC/USDT": {"okx": "BTC-USDT", "binance": "BTCUSDT"},
            "ETH/USDT": {"okx": "ETH-USDT", "binance": "ETHUSDT"}
        }
        
        # 数据回调
        self.tick_callbacks: List[Callable] = []
        self.candle_callbacks: List[Callable] = []
        
        # 运行状态
        self.is_running = False
        
        # Coinglass相关
        self.coinglass_enabled = bool(os.getenv("COINGLASS_API_KEY", ""))
        self.coinglass_update_interval = int(os.getenv("COINGLASS_UPDATE_INTERVAL", "300"))
        
    async def initialize(self):
        """初始化真实数据管理器"""
        try:
            logger.info("初始化真实数据管理器...")
            
            # 初始化数据库连接
            if not data_manager._initialized:
                await data_manager.initialize()
            
            # 初始化优化数据流水线
            await optimized_pipeline.start()
            logger.info("✅ 优化数据流水线已启动")
            
            # 注册数据流水线回调
            optimized_pipeline.register_callback("market_data", self._save_market_data_batch)
            optimized_pipeline.register_callback("candle_data", self._save_candle_data_batch)
            optimized_pipeline.register_callback("coinglass_data", self._save_coinglass_data_batch)
            
            # 创建WebSocket客户端
            okx_client = OKXWebSocketClient()
            
            # 使用增强版Binance客户端
            binance_client = enhanced_binance_client
            
            # 设置优化回调（使用流水线）
            okx_client.add_callback("ticker", self._handle_ticker_data_optimized)
            okx_client.add_callback("candle", self._handle_candle_data_optimized)
            binance_client.add_callback("ticker", self._handle_ticker_data_optimized)
            binance_client.add_callback("candle", self._handle_candle_data_optimized)
            
            # 添加到管理器
            self.websocket_manager.add_client("okx", okx_client)
            self.websocket_manager.add_client("binance", binance_client)
            
            # 初始化Coinglass分析器
            if self.coinglass_enabled:
                await coinglass_analyzer.initialize()
                logger.info("Coinglass分析器已初始化")
            
            logger.info("真实数据管理器初始化完成")
            
        except Exception as e:
            logger.error(f"真实数据管理器初始化失败: {e}")
            raise
    
    async def start_real_data_stream(self):
        """启动真实数据流"""
        if self.is_running:
            logger.warning("真实数据流已在运行")
            return
            
        try:
            self.is_running = True
            logger.info("启动真实数据流...")
            
            # 连接到WebSocket
            okx_client = self.websocket_manager.get_client("okx")
            binance_client = self.websocket_manager.get_client("binance")
            
            if okx_client:
                # 订阅OKX数据
                okx_symbols = [pair["okx"] for pair in self.trading_pairs.values()]
                await okx_client.connect()
                await okx_client.subscribe_tickers(okx_symbols)
                await okx_client.subscribe_candles(okx_symbols, "1m")
                logger.info(f"已订阅OKX数据流: {okx_symbols}")
            
            if binance_client:
                # 订阅Binance数据
                binance_symbols = [pair["binance"] for pair in self.trading_pairs.values()]
                await binance_client.connect()
                await binance_client.subscribe_tickers(binance_symbols)
                await binance_client.subscribe_candles(binance_symbols, "1m")
                logger.info(f"已订阅Binance数据流: {binance_symbols}")
            
            # 启动WebSocket监听
            await self.websocket_manager.start_all()
            
            # 启动Coinglass数据收集
            if self.coinglass_enabled:
                await coinglass_collector_manager.start_collection()
                logger.info("Coinglass数据收集已启动")
            
        except Exception as e:
            logger.error(f"启动真实数据流失败: {e}")
            self.is_running = False
            raise
    
    async def stop_real_data_stream(self):
        """停止真实数据流"""
        try:
            self.is_running = False
            await self.websocket_manager.stop_all()
            
            # 停止Coinglass数据收集
            if self.coinglass_enabled:
                await coinglass_collector_manager.stop_collection()
                logger.info("Coinglass数据收集已停止")
            
            logger.info("真实数据流已停止")
            
        except Exception as e:
            logger.error(f"停止真实数据流失败: {e}")
    
    async def fetch_historical_training_data(self, days_back: int = 30):
        """获取历史训练数据"""
        try:
            logger.info(f"开始获取{days_back}天历史数据用于AI训练...")
            
            # 获取主要交易对的历史数据
            symbols = list(self.trading_pairs.keys())
            exchanges = ["okx", "binance"]
            timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
            
            training_data = await self.historical_manager.fetch_training_data(
                symbols=symbols,
                exchanges=exchanges,
                timeframes=timeframes,
                days_back=days_back
            )
            
            # 创建训练数据集文件
            dataset_file = await self.historical_manager.create_training_dataset(
                symbols=symbols,
                target_file="data/btc_eth_training_dataset.csv"
            )
            
            logger.info(f"历史数据获取完成，训练数据集保存至: {dataset_file}")
            return training_data
            
        except Exception as e:
            logger.error(f"获取历史训练数据失败: {e}")
            raise
    
    async def _handle_ticker_data(self, ticker_data: Dict):
        """处理行情数据"""
        try:
            # 标准化数据格式
            standardized_data = {
                "symbol": ticker_data.get("symbol", ""),
                "exchange": ticker_data.get("exchange", ""),
                "price": ticker_data.get("price", 0),
                "volume_24h": ticker_data.get("volume_24h", 0),
                "change_24h": ticker_data.get("change_24h", 0),
                "change_24h_pct": ticker_data.get("change_24h_pct", 0),
                "high_24h": ticker_data.get("high_24h", 0),
                "low_24h": ticker_data.get("low_24h", 0),
                "timestamp": ticker_data.get("timestamp", int(datetime.utcnow().timestamp() * 1000)),
                "source": "real_data"
            }
            
            # 数据验证
            validation_result = await market_data_validator.validate_market_data(
                standardized_data, 
                data_type="ticker", 
                level=ValidationLevel.STANDARD
            )
            
            # 记录验证结果
            if not validation_result.is_valid:
                logger.warning(f"⚠️ 行情数据验证失败 {standardized_data.get('symbol', 'UNKNOWN')}: {validation_result.issues}")
                return
            elif validation_result.warnings:
                logger.debug(f"📊 行情数据质量警告 {standardized_data.get('symbol', 'UNKNOWN')}: {validation_result.warnings}")
            
            # 添加质量分数到数据中
            standardized_data["data_quality_score"] = validation_result.quality_score
            standardized_data["data_quality_level"] = validation_result.quality_level.value
            
            # 通知回调函数
            for callback in self.tick_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(standardized_data)
                    else:
                        callback(standardized_data)
                except Exception as e:
                    logger.error(f"Ticker回调执行失败: {e}")
                    
        except Exception as e:
            logger.error(f"处理行情数据失败: {e}")
    
    async def _handle_candle_data(self, symbol: str, candle_data: List[Dict]):
        """处理K线数据"""
        try:
            # 数据验证
            validation_result = await market_data_validator.validate_market_data(
                candle_data, 
                data_type="candle", 
                level=ValidationLevel.STANDARD
            )
            
            # 记录验证结果
            if not validation_result.is_valid:
                logger.warning(f"⚠️ K线数据验证失败 {symbol}: {validation_result.issues}")
                return
            elif validation_result.warnings:
                logger.debug(f"📊 K线数据质量警告 {symbol}: {validation_result.warnings}")
            
            # 为每个K线添加质量分数
            validated_candles = []
            for i, candle in enumerate(candle_data):
                enhanced_candle = candle.copy()
                enhanced_candle["data_quality_score"] = validation_result.quality_score
                enhanced_candle["data_quality_level"] = validation_result.quality_level.value
                enhanced_candle["symbol"] = symbol
                validated_candles.append(enhanced_candle)
            
            # 通知回调函数
            for callback in self.candle_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(symbol, validated_candles)
                    else:
                        callback(symbol, validated_candles)
                except Exception as e:
                    logger.error(f"K线回调执行失败: {e}")
                    
        except Exception as e:
            logger.error(f"处理K线数据失败: {e}")
    
    def add_tick_callback(self, callback: Callable):
        """添加行情数据回调"""
        self.tick_callbacks.append(callback)
        
    def add_candle_callback(self, callback: Callable):
        """添加K线数据回调"""
        self.candle_callbacks.append(callback)
    
    def get_connection_status(self) -> Dict[str, bool]:
        """获取连接状态"""
        return self.websocket_manager.get_connection_status()
    
    async def get_latest_prices(self) -> Dict[str, Any]:
        """获取最新价格"""
        try:
            prices = {}
            
            for symbol in self.trading_pairs:
                # 从增强缓存管理器获取最新价格
                if data_manager.cache_manager:
                    # 检查是否使用增强缓存管理器
                    if hasattr(data_manager.cache_manager, 'cache') and hasattr(data_manager.cache_manager.cache, 'l1_cache'):
                        # 使用增强缓存管理器的MarketDataCache接口
                        from .enhanced_cache_manager import MarketDataCache
                        market_cache = MarketDataCache(data_manager.cache_manager)
                        
                        okx_data = await market_cache.get_market_data("OKX", self.trading_pairs[symbol]['okx'])
                        binance_data = await market_cache.get_market_data("Binance", self.trading_pairs[symbol]['binance'])
                    else:
                        # 使用传统缓存方法
                        okx_data = await data_manager.cache_manager.get_market_data(f"OKX:{self.trading_pairs[symbol]['okx']}")
                        binance_data = await data_manager.cache_manager.get_market_data(f"Binance:{self.trading_pairs[symbol]['binance']}")
                    
                    prices[symbol] = {
                        "okx": okx_data,
                        "binance": binance_data
                    }
            
            return prices
            
        except Exception as e:
            logger.error(f"获取最新价格失败: {e}")
            return {}
    
    async def _handle_ticker_data_optimized(self, ticker_data: Dict):
        """优化版行情数据处理器"""
        try:
            # 标准化数据格式
            standardized_data = {
                "symbol": ticker_data.get("symbol", ""),
                "exchange": ticker_data.get("exchange", ""),
                "price": ticker_data.get("price", 0),
                "volume_24h": ticker_data.get("volume_24h", 0),
                "change_24h": ticker_data.get("change_24h", 0),
                "change_24h_pct": ticker_data.get("change_24h_pct", 0),
                "high_24h": ticker_data.get("high_24h", 0),
                "low_24h": ticker_data.get("low_24h", 0),
                "timestamp": ticker_data.get("timestamp", int(datetime.utcnow().timestamp() * 1000)),
                "source": "real_data"
            }
            
            # 直接发送到优化流水线（内部包含验证逻辑）
            await optimized_pipeline.add_data("market_data", standardized_data)
            
        except Exception as e:
            logger.error(f"优化行情数据处理失败: {e}")
    
    async def _handle_candle_data_optimized(self, candle_data: Dict):
        """优化版K线数据处理器"""
        try:
            # 标准化K线数据格式
            standardized_data = {
                "symbol": candle_data.get("symbol", ""),
                "exchange": candle_data.get("exchange", ""),
                "timeframe": candle_data.get("timeframe", "1m"),
                "timestamp": candle_data.get("timestamp", int(datetime.utcnow().timestamp() * 1000)),
                "open": candle_data.get("open", 0),
                "high": candle_data.get("high", 0),
                "low": candle_data.get("low", 0),
                "close": candle_data.get("close", 0),
                "volume": candle_data.get("volume", 0),
                "source": "real_data"
            }
            
            # 发送到优化流水线
            await optimized_pipeline.add_data("candle_data", standardized_data)
            
        except Exception as e:
            logger.error(f"优化K线数据处理失败: {e}")
    
    async def _save_market_data_batch(self, processed_data: List[Dict]):
        """批量保存市场数据"""
        try:
            # 分类数据并批量存储到缓存
            if data_manager.cache_manager:
                # 批量缓存操作
                cache_data = {}
                for data in processed_data:
                    symbol = data.get("symbol", "")
                    exchange = data.get("exchange", "")
                    if symbol and exchange:
                        cache_key = f"market:{exchange}:{symbol}"
                        cache_data[cache_key] = {
                            "price": data.get("price", 0),
                            "volume_24h": data.get("volume_24h", 0),
                            "change_24h_pct": data.get("change_24h_pct", 0),
                            "timestamp": data.get("timestamp", 0)
                        }
                
                if cache_data:
                    await data_manager.cache_manager.mset(cache_data, ttl=60)
                    logger.debug(f"批量缓存市场数据: {len(cache_data)}条")
            
            # 执行原有回调
            for callback in self.tick_callbacks:
                try:
                    for data in processed_data:
                        await callback(data)
                except Exception as e:
                    logger.error(f"市场数据回调执行失败: {e}")
            
        except Exception as e:
            logger.error(f"批量保存市场数据失败: {e}")
    
    async def _save_candle_data_batch(self, processed_data: List[Dict]):
        """批量保存K线数据"""
        try:
            # 按交易对和时间框架分组
            grouped_data = defaultdict(list)
            for data in processed_data:
                symbol = data.get("symbol", "")
                timeframe = data.get("timeframe", "1m")
                if symbol and timeframe:
                    grouped_data[(symbol, timeframe)].append(data)
            
            # 批量插入数据库
            for (symbol, timeframe), candles in grouped_data.items():
                try:
                    await data_manager.time_series_manager.insert_kline_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        data=candles
                    )
                except Exception as e:
                    logger.error(f"保存K线数据失败 {symbol} {timeframe}: {e}")
            
            # 执行原有回调
            for callback in self.candle_callbacks:
                try:
                    for data in processed_data:
                        await callback(data)
                except Exception as e:
                    logger.error(f"K线数据回调执行失败: {e}")
            
            logger.debug(f"批量保存K线数据: {len(processed_data)}条")
            
        except Exception as e:
            logger.error(f"批量保存K线数据失败: {e}")
    
    async def _save_coinglass_data_batch(self, processed_data: List[Dict]):
        """批量保存Coinglass数据"""
        try:
            # 按数据类型分组保存
            fear_greed_data = []
            funding_rate_data = []
            open_interest_data = []
            
            for data in processed_data:
                data_type = data.get("data_type", "")
                if data_type == "fear_greed":
                    fear_greed_data.append(data)
                elif data_type == "funding_rate":
                    funding_rate_data.append(data)
                elif data_type == "open_interest":
                    open_interest_data.append(data)
            
            # 批量保存到数据库
            if fear_greed_data:
                await data_manager.db.fear_greed_index.insert_many(fear_greed_data)
            if funding_rate_data:
                await data_manager.db.funding_rates.insert_many(funding_rate_data)
            if open_interest_data:
                await data_manager.db.open_interest.insert_many(open_interest_data)
            
            logger.debug(f"批量保存Coinglass数据: FGI={len(fear_greed_data)}, FR={len(funding_rate_data)}, OI={len(open_interest_data)}")
            
        except Exception as e:
            logger.error(f"批量保存Coinglass数据失败: {e}")
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """获取数据流水线性能指标"""
        return optimized_pipeline.get_metrics()
    
    async def get_coinglass_analysis(self) -> Dict[str, Any]:
        """获取Coinglass分析数据"""
        if not self.coinglass_enabled:
            return {"enabled": False, "message": "Coinglass未启用"}
        
        try:
            # 获取综合分析信号
            composite_signal = await coinglass_analyzer.generate_composite_signal()
            
            return {
                "enabled": True,
                "composite_signal": {
                    "overall_score": composite_signal.overall_score,
                    "signal_strength": composite_signal.signal_strength,
                    "market_regime": composite_signal.market_regime,
                    "risk_assessment": composite_signal.risk_assessment,
                    "confidence": composite_signal.confidence,
                    "key_factors": composite_signal.key_factors,
                    "timestamp": composite_signal.timestamp.isoformat()
                },
                "sentiment": {
                    "score": composite_signal.sentiment_signal.sentiment_score,
                    "trend": composite_signal.sentiment_signal.trend,
                    "strength": composite_signal.sentiment_signal.strength,
                    "confidence": composite_signal.sentiment_signal.confidence
                },
                "funding_rates": {
                    "overall_rate": composite_signal.funding_signal.overall_rate,
                    "trend": composite_signal.funding_signal.rate_trend,
                    "market_heat": composite_signal.funding_signal.market_heat,
                    "divergence": composite_signal.funding_signal.divergence_score
                },
                "open_interest": {
                    "total": composite_signal.oi_signal.total_oi,
                    "trend": composite_signal.oi_signal.oi_trend,
                    "change_24h": composite_signal.oi_signal.oi_change_24h,
                    "risk_level": composite_signal.oi_signal.risk_level
                },
                "etf_flows": {
                    "btc_flow": composite_signal.etf_signal.btc_flow,
                    "eth_flow": composite_signal.etf_signal.eth_flow,
                    "trend": composite_signal.etf_signal.flow_trend,
                    "institutional_sentiment": composite_signal.etf_signal.institutional_sentiment
                }
            }
            
        except Exception as e:
            logger.error(f"获取Coinglass分析失败: {e}")
            return {"enabled": True, "error": str(e)}
    
    async def get_data_quality_report(self, hours: int = 1) -> Dict[str, Any]:
        """获取数据质量报告"""
        try:
            return market_data_validator.get_quality_report(hours)
        except Exception as e:
            logger.error(f"获取数据质量报告失败: {e}")
            return {"error": str(e)}
    
    async def migrate_coinglass_historical_data(self, source_path: str) -> Dict[str, Any]:
        """迁移Coinglass历史数据"""
        if not self.coinglass_enabled:
            return {"success": False, "enabled": False, "message": "Coinglass未启用"}
        
        try:
            from .coinglass_data_migrator import migrate_coinglass_data
            result = await migrate_coinglass_data(source_path)
            logger.info(f"Coinglass数据迁移完成: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Coinglass数据迁移失败: {e}")
            return {"success": False, "error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 数据库连接状态
            db_status = await data_manager.health_check()
            
            # WebSocket连接状态
            ws_status = self.get_connection_status()
            
            # Coinglass状态
            coinglass_status = {}
            if self.coinglass_enabled:
                coinglass_status = coinglass_collector_manager.get_status()
            
            return {
                "database": db_status,
                "websockets": ws_status,
                "coinglass": {
                    "enabled": self.coinglass_enabled,
                    "status": coinglass_status
                },
                "running": self.is_running,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {"error": str(e)}

# 全局真实数据管理器实例
real_data_manager = RealDataManager()