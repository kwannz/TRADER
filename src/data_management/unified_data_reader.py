"""
Unified Data Reader
统一数据读取器 - 集成PandaFactor数据管理能力
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import asyncio

# MongoDB相关导入
try:
    from pymongo import MongoClient
    from pymongo.database import Database
    from pymongo.collection import Collection
except ImportError:
    MongoClient = None
    Database = None
    Collection = None


class DatabaseHandler:
    """
    数据库处理器 - 基于PandaFactor的数据库接口适配
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("DatabaseHandler")
        
        # MongoDB连接参数
        self.mongo_host = config.get("MONGO_HOST", "localhost")
        self.mongo_port = config.get("MONGO_PORT", 27017)
        self.mongo_username = config.get("MONGO_USERNAME")
        self.mongo_password = config.get("MONGO_PASSWORD")
        self.mongo_db = config.get("MONGO_DB", "quantitative_trading")
        
        self.mongo_client: Optional[MongoClient] = None
        self._connect_to_mongodb()
    
    def _connect_to_mongodb(self):
        """连接到MongoDB"""
        if not MongoClient:
            self.logger.warning("pymongo not available, MongoDB features disabled")
            return
        
        try:
            # 构建连接字符串
            if self.mongo_username and self.mongo_password:
                connection_string = f"mongodb://{self.mongo_username}:{self.mongo_password}@{self.mongo_host}:{self.mongo_port}/"
            else:
                connection_string = f"mongodb://{self.mongo_host}:{self.mongo_port}/"
            
            self.mongo_client = MongoClient(
                connection_string,
                serverSelectionTimeoutMS=5000,  # 5秒超时
                maxPoolSize=50
            )
            
            # 测试连接
            self.mongo_client.server_info()
            self.logger.info(f"Successfully connected to MongoDB at {self.mongo_host}:{self.mongo_port}")
            
        except Exception as e:
            self.logger.warning(f"Failed to connect to MongoDB: {str(e)}")
            self.mongo_client = None
    
    def get_mongo_collection(self, database_name: str, collection_name: str) -> Optional[Collection]:
        """获取MongoDB集合"""
        if not self.mongo_client:
            return None
        
        try:
            database = self.mongo_client[database_name]
            return database[collection_name]
        except Exception as e:
            self.logger.error(f"Error accessing collection {collection_name}: {str(e)}")
            return None
    
    def mongo_find(self, database_name: str, collection_name: str, 
                   query: Dict[str, Any], projection: Dict[str, Any] = None,
                   limit: int = None, sort: List[Tuple[str, int]] = None) -> List[Dict[str, Any]]:
        """执行MongoDB查询"""
        collection = self.get_mongo_collection(database_name, collection_name)
        if not collection:
            return []
        
        try:
            cursor = collection.find(query, projection)
            
            if sort:
                cursor = cursor.sort(sort)
            if limit:
                cursor = cursor.limit(limit)
            
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"MongoDB query error: {str(e)}")
            return []
    
    def mongo_aggregate(self, database_name: str, collection_name: str,
                       pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行MongoDB聚合查询"""
        collection = self.get_mongo_collection(database_name, collection_name)
        if not collection:
            return []
        
        try:
            return list(collection.aggregate(pipeline))
        except Exception as e:
            self.logger.error(f"MongoDB aggregate error: {str(e)}")
            return []
    
    def get_distinct_values(self, database_name: str, collection_name: str,
                          field: str, query: Dict[str, Any] = None) -> List[Any]:
        """获取字段的唯一值"""
        collection = self.get_mongo_collection(database_name, collection_name)
        if not collection:
            return []
        
        try:
            return collection.distinct(field, query or {})
        except Exception as e:
            self.logger.error(f"MongoDB distinct error: {str(e)}")
            return []


class FactorDataReader:
    """
    因子数据读取器 - 基于PandaFactor的FactorReader适配
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("FactorDataReader")
        self.db_handler = DatabaseHandler(config)
        
        # 缓存所有股票代码
        self.all_symbols = self._get_all_symbols()
        
        self.logger.info(f"FactorDataReader initialized with {len(self.all_symbols)} symbols")
    
    def _get_all_symbols(self) -> List[str]:
        """获取所有股票代码"""
        try:
            symbols = self.db_handler.get_distinct_values(
                self.config.get("MONGO_DB", "quantitative_trading"),
                "stock_market",
                "symbol"
            )
            return symbols or []
        except Exception as e:
            self.logger.warning(f"Failed to get symbols from database: {str(e)}")
            # 返回模拟数据
            return ['000001', '000002', '000858', '002415', '300059', '600036', '600519', '600887']
    
    def get_base_factors(self, symbols: List[str], start_date: str, end_date: str,
                        factors: List[str] = None, index_component: Optional[str] = None,
                        asset_type: str = 'stock') -> Optional[pd.DataFrame]:
        """
        获取基础因子数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            factors: 因子列表
            index_component: 指数成分股筛选
            asset_type: 资产类型 ('stock', 'future')
            
        Returns:
            包含所有因子的DataFrame
        """
        if factors is None:
            factors = ['open', 'close', 'high', 'low', 'volume', 'market_cap', 'turnover', 'amount']
        
        # 转换为小写
        factors = [f.lower() for f in factors]
        base_factors = ["open", "close", "high", "low", "volume", "market_cap", "turnover", "amount"]
        requested_base_factors = [f for f in factors if f in base_factors]
        
        if not requested_base_factors:
            self.logger.warning("No valid base factors requested")
            return None
        
        try:
            # 构建查询条件
            query = {
                "date": {"$gte": start_date, "$lte": end_date}
            }
            
            if symbols:
                query["symbol"] = {"$in": symbols}
            
            if index_component:
                query['index_component'] = {"$eq": index_component}
            
            # 构建投影
            projection = {field: 1 for field in ['date', 'symbol'] + requested_base_factors}
            projection['_id'] = 0
            
            # 选择集合
            collection_name = "future_market" if asset_type == 'future' else "factor_base"
            
            if asset_type == 'future':
                # 期货特殊处理
                query["$expr"] = {
                    "$eq": [
                        "$symbol",
                        {"$concat": ["$underlying_symbol", "88"]}
                    ]
                }
            
            # 执行查询
            records = self.db_handler.mongo_find(
                self.config.get("MONGO_DB", "quantitative_trading"),
                collection_name,
                query,
                projection
            )
            
            if not records:
                self.logger.warning("No data found for the specified parameters")
                return self._generate_mock_data(symbols, start_date, end_date, factors)
            
            # 转换为DataFrame
            df = pd.DataFrame(records)
            
            # 确保date列为datetime类型
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            self.logger.info(f"Successfully loaded {len(df)} records from database")
            return df
            
        except Exception as e:
            self.logger.warning(f"Database query failed: {str(e)}, using mock data")
            return self._generate_mock_data(symbols, start_date, end_date, factors)
    
    def _generate_mock_data(self, symbols: List[str], start_date: str, end_date: str,
                           factors: List[str]) -> pd.DataFrame:
        """生成模拟数据"""
        self.logger.info("Generating mock data for testing")
        
        # 创建日期范围
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 生成数据
        data_records = []
        np.random.seed(42)  # 确保可重复性
        
        for date in dates:
            for symbol in symbols:
                record = {
                    'date': date,
                    'symbol': symbol
                }
                
                # 生成基础价格
                base_price = 100 + np.random.randn() * 20
                
                if 'close' in factors:
                    record['close'] = max(10, base_price + np.random.randn() * 5)
                
                if 'open' in factors:
                    record['open'] = record.get('close', base_price) * (1 + np.random.normal(0, 0.01))
                
                if 'high' in factors:
                    record['high'] = max(record.get('close', base_price), record.get('open', base_price)) * np.random.uniform(1.0, 1.05)
                
                if 'low' in factors:
                    record['low'] = min(record.get('close', base_price), record.get('open', base_price)) * np.random.uniform(0.95, 1.0)
                
                if 'volume' in factors:
                    record['volume'] = max(1000, int(np.random.lognormal(13, 0.5)))
                
                if 'amount' in factors:
                    price = record.get('close', base_price)
                    volume = record.get('volume', 1000000)
                    record['amount'] = price * volume
                
                if 'market_cap' in factors:
                    price = record.get('close', base_price)
                    record['market_cap'] = price * np.random.uniform(1e8, 1e12)  # 总股本随机
                
                if 'turnover' in factors:
                    record['turnover'] = np.random.uniform(0.001, 0.1)  # 换手率
                
                data_records.append(record)
        
        df = pd.DataFrame(data_records)
        self.logger.info(f"Generated {len(df)} mock records")
        return df
    
    def get_custom_factor(self, user_id: str, factor_name: str, start_date: str, end_date: str,
                         symbol_type: str = 'stock') -> Optional[pd.DataFrame]:
        """
        获取自定义因子数据
        
        Args:
            user_id: 用户ID
            factor_name: 因子名称
            start_date: 开始日期
            end_date: 结束日期
            symbol_type: 股票类型
            
        Returns:
            因子数据DataFrame
        """
        try:
            # 首先检查是否有预计算的因子表
            collection_name = f"factor_{factor_name}_{user_id}"
            
            # 查询预计算数据
            query = {
                "date": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }
            
            records = self.db_handler.mongo_find(
                self.config.get("MONGO_DB", "quantitative_trading"),
                collection_name,
                query
            )
            
            if records:
                df = pd.DataFrame(records)
                df = df.set_index(['date', 'symbol'])
                df = df.drop(columns=['_id'], errors='ignore')
                self.logger.info(f"Found {len(df)} custom factor records in {collection_name}")
                return df
            
            # 如果没有预计算数据，从用户因子定义中获取
            return self._get_factor_definition_and_calculate(user_id, factor_name, start_date, end_date)
            
        except Exception as e:
            self.logger.error(f"Error getting custom factor {factor_name} for user {user_id}: {str(e)}")
            return None
    
    def _get_factor_definition_and_calculate(self, user_id: str, factor_name: str,
                                          start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """获取因子定义并计算"""
        try:
            # 查询用户因子定义
            query = {
                "user_id": str(user_id),
                "factor_name": factor_name
            }
            
            records = self.db_handler.mongo_find(
                self.config.get("MONGO_DB", "quantitative_trading"),
                "user_factors",
                query
            )
            
            if not records:
                self.logger.warning(f"No factor definition found for {factor_name} by user {user_id}")
                return None
            
            factor_def = records[0]
            code_type = factor_def.get("code_type", "formula")
            code = factor_def.get("code", "")
            params = factor_def.get("params", {})
            
            # 获取股票池
            stock_pool = params.get('stock_pool', '000985')  # 默认全市场
            include_st = params.get('include_st', True)
            
            # TODO: 这里应该集成因子计算引擎
            # from ..factor_engine import unified_interface
            # result = unified_interface.calculate_custom_factor(code_type, code, start_date, end_date)
            
            self.logger.info(f"Factor definition found: {code_type} type, code length: {len(code)}")
            return None  # 暂时返回None，等待因子引擎集成
            
        except Exception as e:
            self.logger.error(f"Error calculating custom factor: {str(e)}")
            return None
    
    def get_symbols_by_index(self, index_code: str, date: Optional[str] = None) -> List[str]:
        """
        根据指数代码获取成分股
        
        Args:
            index_code: 指数代码 ('000300', '000905', '000852' 等)
            date: 指定日期，None表示最新
            
        Returns:
            股票代码列表
        """
        try:
            # 构建查询条件
            query = {}
            
            # 指数成分映射
            index_mapping = {
                "000300": "100",  # 沪深300
                "000905": "010",  # 中证500
                "000852": "001"   # 中证1000
            }
            
            if index_code in index_mapping:
                query["index_component"] = index_mapping[index_code]
            
            if date:
                query["date"] = date
            
            # 获取符合条件的股票代码
            symbols = self.db_handler.get_distinct_values(
                self.config.get("MONGO_DB", "quantitative_trading"),
                "stock_market",
                "symbol",
                query
            )
            
            self.logger.info(f"Found {len(symbols)} symbols for index {index_code}")
            return symbols
            
        except Exception as e:
            self.logger.error(f"Error getting symbols for index {index_code}: {str(e)}")
            return []
    
    def get_trading_calendar(self, start_date: str, end_date: str) -> List[str]:
        """
        获取交易日历
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            交易日列表
        """
        try:
            # 查询交易日历表
            query = {
                "date": {"$gte": start_date, "$lte": end_date},
                "is_trading_day": True
            }
            
            records = self.db_handler.mongo_find(
                self.config.get("MONGO_DB", "quantitative_trading"),
                "trading_calendar",
                query,
                {"date": 1, "_id": 0},
                sort=[("date", 1)]
            )
            
            if records:
                trading_days = [record["date"] for record in records]
                self.logger.info(f"Found {len(trading_days)} trading days")
                return trading_days
            else:
                # 如果没有交易日历数据，生成工作日
                dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 工作日
                trading_days = [date.strftime('%Y-%m-%d') for date in dates]
                self.logger.warning(f"No trading calendar found, generated {len(trading_days)} business days")
                return trading_days
                
        except Exception as e:
            self.logger.error(f"Error getting trading calendar: {str(e)}")
            # 返回工作日作为备选
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
            return [date.strftime('%Y-%m-%d') for date in dates]


class UnifiedDataReader:
    """
    统一数据读取器 - 整合多种数据源
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("UnifiedDataReader")
        
        # 初始化数据读取器
        self.factor_reader = FactorDataReader(self.config)
        
        self.logger.info("Unified Data Reader initialized")
    
    def get_market_data(self, symbols: List[str], start_date: str, end_date: str,
                       factors: List[str] = None, **kwargs) -> Dict[str, pd.Series]:
        """
        获取市场数据，返回MultiIndex Series格式
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期  
            factors: 因子列表
            **kwargs: 其他参数
            
        Returns:
            因子数据字典，键为因子名，值为MultiIndex Series
        """
        try:
            # 获取DataFrame格式数据
            df = self.factor_reader.get_base_factors(symbols, start_date, end_date, factors, **kwargs)
            
            if df is None or df.empty:
                self.logger.warning("No data returned from factor reader")
                return {}
            
            # 转换为MultiIndex Series格式
            data_dict = {}
            
            # 确保date和symbol列存在
            if 'date' not in df.columns or 'symbol' not in df.columns:
                self.logger.error("Missing date or symbol columns in data")
                return {}
            
            # 设置MultiIndex
            df_indexed = df.set_index(['date', 'symbol'])
            
            # 转换每个因子为Series
            for factor in df_indexed.columns:
                series = df_indexed[factor].copy()
                series.index.names = ['date', 'symbol']
                data_dict[factor] = series
            
            self.logger.info(f"Successfully converted data for {len(data_dict)} factors")
            return data_dict
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {str(e)}")
            return {}
    
    def get_custom_factor_data(self, user_id: str, factor_name: str, 
                              start_date: str, end_date: str) -> Optional[pd.Series]:
        """
        获取自定义因子数据
        
        Args:
            user_id: 用户ID
            factor_name: 因子名称
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            因子数据Series，MultiIndex格式
        """
        try:
            df = self.factor_reader.get_custom_factor(user_id, factor_name, start_date, end_date)
            
            if df is None or df.empty:
                return None
            
            # 如果DataFrame有多列，取第一列作为因子值
            if len(df.columns) > 1:
                factor_col = [col for col in df.columns if col != '_id'][0]
                series = df[factor_col]
            else:
                series = df.iloc[:, 0]
            
            series.name = factor_name
            return series
            
        except Exception as e:
            self.logger.error(f"Error getting custom factor {factor_name}: {str(e)}")
            return None
    
    def get_index_constituents(self, index_code: str, date: Optional[str] = None) -> List[str]:
        """获取指数成分股"""
        return self.factor_reader.get_symbols_by_index(index_code, date)
    
    def get_trading_days(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日列表"""
        return self.factor_reader.get_trading_calendar(start_date, end_date)
    
    def get_available_symbols(self) -> List[str]:
        """获取所有可用的股票代码"""
        return self.factor_reader.all_symbols
    
    def validate_data_availability(self, symbols: List[str], start_date: str, 
                                  end_date: str, factors: List[str]) -> Dict[str, Any]:
        """
        验证数据可用性
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            factors: 因子列表
            
        Returns:
            数据可用性报告
        """
        report = {
            "request": {
                "symbols": len(symbols),
                "date_range": f"{start_date} to {end_date}",
                "factors": factors
            },
            "availability": {},
            "missing_data": {},
            "recommendations": []
        }
        
        try:
            # 获取数据样本
            data = self.get_market_data(symbols[:5], start_date, end_date, factors)  # 只检查前5个股票
            
            for factor, series in data.items():
                total_expected = 5 * len(pd.date_range(start_date, end_date, freq='D'))
                actual_count = len(series.dropna())
                coverage = actual_count / total_expected if total_expected > 0 else 0
                
                report["availability"][factor] = {
                    "coverage_ratio": coverage,
                    "available_records": actual_count,
                    "expected_records": total_expected
                }
                
                if coverage < 0.5:
                    report["missing_data"][factor] = "Low data coverage"
                    report["recommendations"].append(f"Check data source for factor {factor}")
            
        except Exception as e:
            report["error"] = str(e)
        
        return report


# 创建全局统一数据读取器实例
unified_data_reader = UnifiedDataReader()