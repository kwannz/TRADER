"""
Coinglass历史数据迁移工具
将现有的JSON格式数据迁移到MongoDB数据库
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.unified_logger import get_logger, LogCategory
from .data_manager import data_manager

logger = get_logger()

class CoinglassDataMigrator:
    """Coinglass数据迁移器"""
    
    def __init__(self, source_path: str):
        self.source_path = Path(source_path)
        self.migration_stats = {
            "files_processed": 0,
            "records_migrated": 0,
            "errors": 0,
            "collections_created": set()
        }
    
    async def migrate_all_data(self) -> Dict[str, Any]:
        """迁移所有Coinglass数据"""
        logger.info(f"开始从{self.source_path}迁移Coinglass数据到MongoDB...")
        
        # 确保数据库连接
        if not data_manager._initialized:
            await data_manager.initialize()
        
        start_time = datetime.utcnow()
        
        try:
            # 迁移指标数据
            await self._migrate_indicators()
            
            # 迁移期货数据
            await self._migrate_futures_data()
            
            # 迁移ETF数据
            await self._migrate_etf_data()
            
            # 迁移现货数据
            await self._migrate_spot_data()
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            result = {
                "success": True,
                "duration": duration,
                "stats": {
                    **self.migration_stats,
                    "collections_created": list(self.migration_stats["collections_created"])
                }
            }
            
            logger.info(f"数据迁移完成: {result}")
            return result
            
        except Exception as e:
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"数据迁移失败: {e}")
            
            return {
                "success": False,
                "duration": duration,
                "error": str(e),
                "stats": {
                    **self.migration_stats,
                    "collections_created": list(self.migration_stats["collections_created"])
                }
            }
    
    async def _migrate_indicators(self):
        """迁移指标数据"""
        indicators_path = self.source_path / "data" / "coinglass" / "indicators"
        
        if not indicators_path.exists():
            logger.warning(f"指标数据路径不存在: {indicators_path}")
            return
        
        # 迁移恐惧贪婪指数
        await self._migrate_fear_greed_index(indicators_path)
    
    async def _migrate_fear_greed_index(self, indicators_path: Path):
        """迁移恐惧贪婪指数数据"""
        fear_greed_files = [
            "fear-greed-current.json",
            "fear-greed-history.json",
            "fear-greed-index.json"
        ]
        
        collection_name = "fear_greed_index"
        collection = data_manager.db[collection_name]
        self.migration_stats["collections_created"].add(collection_name)
        
        all_records = []
        
        for filename in fear_greed_files:
            file_path = indicators_path / filename
            if file_path.exists():
                try:
                    records = await self._process_fear_greed_file(file_path)
                    all_records.extend(records)
                    self.migration_stats["files_processed"] += 1
                    logger.info(f"处理恐惧贪婪指数文件: {filename}, {len(records)}条记录")
                    
                except Exception as e:
                    logger.error(f"处理文件{filename}失败: {e}")
                    self.migration_stats["errors"] += 1
        
        # 批量保存数据
        if all_records:
            await self._bulk_insert_with_dedup(collection, all_records, ["date", "value"])
            logger.info(f"恐惧贪婪指数数据迁移完成: {len(all_records)}条记录")
    
    async def _process_fear_greed_file(self, file_path: Path) -> List[Dict]:
        """处理恐惧贪婪指数文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            records = []
            
            # 处理不同的数据格式
            if isinstance(data, dict):
                # 处理嵌套的data.data.data格式
                data_content = data
                if "data" in data and isinstance(data["data"], dict):
                    data_content = data["data"]
                    if "data" in data_content and isinstance(data_content["data"], dict):
                        data_content = data_content["data"]
                
                if "data_list" in data_content:
                    # 处理data_list格式
                    data_list = data_content["data_list"]
                    timestamp_list = data_content.get("timestamp_list", [])
                    
                    for i, value in enumerate(data_list):
                        if isinstance(value, (int, float)):
                            timestamp = None
                            if i < len(timestamp_list):
                                timestamp = datetime.fromtimestamp(timestamp_list[i])
                            
                            ts = timestamp or datetime.utcnow()
                            record = {
                                "value": int(value),
                                "timestamp": ts,
                                "date": ts.strftime("%Y-%m-%d"),
                                "classification": self._classify_fear_greed(value),
                                "source_file": file_path.name,
                                "migrated_at": datetime.utcnow()
                            }
                            records.append(record)
                
                elif "data" in data_content and isinstance(data_content["data"], list):
                    # 处理历史数据列表格式
                    for item in data_content["data"]:
                        if isinstance(item, dict):
                            record = {
                                "value": item.get("value", 0),
                                "date": item.get("date", ""),
                                "timestamp": self._parse_date(item.get("date", "")),
                                "classification": item.get("classification") or self._classify_fear_greed(item.get("value", 0)),
                                "source_file": file_path.name,
                                "migrated_at": datetime.utcnow()
                            }
                            records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"处理恐惧贪婪指数文件失败: {e}")
            return []
    
    async def _migrate_futures_data(self):
        """迁移期货数据"""
        futures_path = self.source_path / "data" / "coinglass" / "futures"
        
        if not futures_path.exists():
            logger.warning(f"期货数据路径不存在: {futures_path}")
            return
        
        # 迁移资金费率数据
        await self._migrate_funding_rates(futures_path)
        
        # 迁移持仓数据
        await self._migrate_open_interest(futures_path)
        
        # 迁移爆仓数据
        await self._migrate_liquidation_data(futures_path)
    
    async def _migrate_funding_rates(self, futures_path: Path):
        """迁移资金费率数据"""
        funding_files = list(futures_path.glob("*-funding-rate.json"))
        
        collection_name = "funding_rates"
        collection = data_manager.db[collection_name]
        self.migration_stats["collections_created"].add(collection_name)
        
        all_records = []
        
        for file_path in funding_files:
            try:
                symbol = file_path.stem.split('-')[0].upper()
                records = await self._process_funding_rate_file(file_path, symbol)
                all_records.extend(records)
                self.migration_stats["files_processed"] += 1
                logger.info(f"处理资金费率文件: {file_path.name}, {len(records)}条记录")
                
            except Exception as e:
                logger.error(f"处理文件{file_path.name}失败: {e}")
                self.migration_stats["errors"] += 1
        
        if all_records:
            await self._bulk_insert_with_dedup(collection, all_records, ["symbol", "exchange", "timestamp"])
            logger.info(f"资金费率数据迁移完成: {len(all_records)}条记录")
    
    async def _process_funding_rate_file(self, file_path: Path, symbol: str) -> List[Dict]:
        """处理资金费率文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            records = []
            
            if isinstance(data, dict) and "data" in data:
                if isinstance(data["data"], list):
                    for item in data["data"]:
                        if isinstance(item, dict):
                            record = {
                                "symbol": symbol,
                                "exchange": item.get("exchange", ""),
                                "funding_rate": float(item.get("rate", 0)),
                                "next_funding_time": item.get("next_funding_time", ""),
                                "timestamp": self._parse_timestamp(item.get("timestamp")),
                                "date": datetime.utcnow().strftime("%Y-%m-%d"),
                                "source_file": file_path.name,
                                "migrated_at": datetime.utcnow()
                            }
                            records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"处理资金费率文件失败: {e}")
            return []
    
    async def _migrate_open_interest(self, futures_path: Path):
        """迁移持仓数据"""
        oi_files = list(futures_path.glob("*-open-interest.json"))
        
        collection_name = "open_interest"
        collection = data_manager.db[collection_name]
        self.migration_stats["collections_created"].add(collection_name)
        
        all_records = []
        
        for file_path in oi_files:
            try:
                symbol = file_path.stem.split('-')[0].upper()
                records = await self._process_open_interest_file(file_path, symbol)
                all_records.extend(records)
                self.migration_stats["files_processed"] += 1
                logger.info(f"处理持仓数据文件: {file_path.name}, {len(records)}条记录")
                
            except Exception as e:
                logger.error(f"处理文件{file_path.name}失败: {e}")
                self.migration_stats["errors"] += 1
        
        if all_records:
            await self._bulk_insert_with_dedup(collection, all_records, ["symbol", "exchange", "timestamp"])
            logger.info(f"持仓数据迁移完成: {len(all_records)}条记录")
    
    async def _process_open_interest_file(self, file_path: Path, symbol: str) -> List[Dict]:
        """处理持仓数据文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            records = []
            
            if isinstance(data, dict) and "data" in data:
                if isinstance(data["data"], list):
                    for item in data["data"]:
                        if isinstance(item, dict):
                            record = {
                                "symbol": symbol,
                                "exchange": item.get("exchange", ""),
                                "open_interest": float(item.get("open_interest", 0)),
                                "open_interest_value": float(item.get("open_interest_value", 0)),
                                "change_24h": float(item.get("change_24h", 0)),
                                "change_24h_percent": float(item.get("change_24h_percent", 0)),
                                "timestamp": self._parse_timestamp(item.get("timestamp")),
                                "date": datetime.utcnow().strftime("%Y-%m-%d"),
                                "source_file": file_path.name,
                                "migrated_at": datetime.utcnow()
                            }
                            records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"处理持仓数据文件失败: {e}")
            return []
    
    async def _migrate_liquidation_data(self, futures_path: Path):
        """迁移爆仓数据"""
        liq_files = list(futures_path.glob("*-liquidation.json"))
        
        collection_name = "liquidations"
        collection = data_manager.db[collection_name]
        self.migration_stats["collections_created"].add(collection_name)
        
        all_records = []
        
        for file_path in liq_files:
            try:
                symbol = file_path.stem.split('-')[0].upper()
                records = await self._process_liquidation_file(file_path, symbol)
                all_records.extend(records)
                self.migration_stats["files_processed"] += 1
                logger.info(f"处理爆仓数据文件: {file_path.name}, {len(records)}条记录")
                
            except Exception as e:
                logger.error(f"处理文件{file_path.name}失败: {e}")
                self.migration_stats["errors"] += 1
        
        if all_records:
            await self._bulk_insert_with_dedup(collection, all_records, ["symbol", "period", "timestamp"])
            logger.info(f"爆仓数据迁移完成: {len(all_records)}条记录")
    
    async def _process_liquidation_file(self, file_path: Path, symbol: str) -> List[Dict]:
        """处理爆仓数据文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            records = []
            
            if isinstance(data, dict) and "data" in data:
                for period, period_data in data["data"].items():
                    if isinstance(period_data, dict):
                        record = {
                            "symbol": symbol,
                            "period": period,
                            "total_liquidation": float(period_data.get("total", 0)),
                            "long_liquidation": float(period_data.get("long", 0)),
                            "short_liquidation": float(period_data.get("short", 0)),
                            "long_percentage": float(period_data.get("long_pct", 0)),
                            "short_percentage": float(period_data.get("short_pct", 0)),
                            "timestamp": datetime.utcnow(),
                            "date": datetime.utcnow().strftime("%Y-%m-%d"),
                            "source_file": file_path.name,
                            "migrated_at": datetime.utcnow()
                        }
                        records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"处理爆仓数据文件失败: {e}")
            return []
    
    async def _migrate_etf_data(self):
        """迁移ETF数据"""
        etf_path = self.source_path / "data" / "coinglass" / "etf"
        
        if not etf_path.exists():
            logger.warning(f"ETF数据路径不存在: {etf_path}")
            return
        
        collection_name = "etf_flows"
        collection = data_manager.db[collection_name]
        self.migration_stats["collections_created"].add(collection_name)
        
        etf_files = list(etf_path.glob("*.json"))
        all_records = []
        
        for file_path in etf_files:
            try:
                records = await self._process_etf_file(file_path)
                all_records.extend(records)
                self.migration_stats["files_processed"] += 1
                logger.info(f"处理ETF数据文件: {file_path.name}, {len(records)}条记录")
                
            except Exception as e:
                logger.error(f"处理文件{file_path.name}失败: {e}")
                self.migration_stats["errors"] += 1
        
        if all_records:
            await self._bulk_insert_with_dedup(collection, all_records, ["asset", "date"])
            logger.info(f"ETF数据迁移完成: {len(all_records)}条记录")
    
    async def _process_etf_file(self, file_path: Path) -> List[Dict]:
        """处理ETF文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            records = []
            asset = "BTC" if "bitcoin" in file_path.name.lower() else "ETH"
            
            if isinstance(data, dict) and "data" in data:
                if isinstance(data["data"], list):
                    for item in data["data"]:
                        if isinstance(item, dict):
                            record = {
                                "asset": asset,
                                "total_netflow": float(item.get("total", 0)),
                                "daily_netflow": float(item.get("daily", 0)),
                                "weekly_netflow": float(item.get("weekly", 0)),
                                "monthly_netflow": float(item.get("monthly", 0)),
                                "date": item.get("date", ""),
                                "timestamp": self._parse_date(item.get("date", "")),
                                "source_file": file_path.name,
                                "migrated_at": datetime.utcnow()
                            }
                            records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"处理ETF文件失败: {e}")
            return []
    
    async def _migrate_spot_data(self):
        """迁移现货数据"""
        spot_path = self.source_path / "data" / "coinglass" / "spot"
        
        if not spot_path.exists():
            logger.warning(f"现货数据路径不存在: {spot_path}")
            return
        
        logger.info("现货数据迁移暂时跳过（数据格式较复杂）")
    
    async def _bulk_insert_with_dedup(self, collection, records: List[Dict], dedup_keys: List[str]):
        """批量插入数据并去重"""
        if not records:
            return
        
        try:
            # 直接批量插入，忽略重复键错误
            result = await collection.insert_many(records, ordered=False)
            inserted_count = len(result.inserted_ids)
            self.migration_stats["records_migrated"] += inserted_count
            logger.debug(f"批量插入{collection.name}: {inserted_count}条记录")
            
        except Exception as e:
            # 如果有重复键错误，尝试逐个插入
            logger.warning(f"批量插入{collection.name}部分失败，尝试逐个插入: {e}")
            inserted_count = 0
            for record in records:
                try:
                    await collection.insert_one(record)
                    inserted_count += 1
                except Exception:
                    # 跳过重复记录
                    pass
            
            self.migration_stats["records_migrated"] += inserted_count
            logger.debug(f"逐个插入{collection.name}: {inserted_count}条记录")
    
    def _classify_fear_greed(self, value: int) -> str:
        """分类恐惧贪婪指数"""
        if value <= 25:
            return "Extreme Fear"
        elif value <= 45:
            return "Fear"
        elif value <= 55:
            return "Neutral"
        elif value <= 75:
            return "Greed"
        else:
            return "Extreme Greed"
    
    def _parse_date(self, date_str: str) -> datetime:
        """解析日期字符串"""
        if not date_str:
            return datetime.utcnow()
        
        try:
            # 尝试多种日期格式
            formats = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ"]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            # 如果都不匹配，返回当前时间
            return datetime.utcnow()
            
        except Exception:
            return datetime.utcnow()
    
    def _parse_timestamp(self, timestamp) -> datetime:
        """解析时间戳"""
        if not timestamp:
            return datetime.utcnow()
        
        try:
            if isinstance(timestamp, str):
                return self._parse_date(timestamp)
            elif isinstance(timestamp, (int, float)):
                # 处理毫秒级时间戳
                if timestamp > 1e12:
                    timestamp = timestamp / 1000
                return datetime.fromtimestamp(timestamp)
            else:
                return datetime.utcnow()
                
        except Exception:
            return datetime.utcnow()

async def migrate_coinglass_data(source_path: str) -> Dict[str, Any]:
    """迁移Coinglass数据的便捷函数"""
    migrator = CoinglassDataMigrator(source_path)
    return await migrator.migrate_all_data()

if __name__ == "__main__":
    # 命令行运行数据迁移
    import argparse
    
    parser = argparse.ArgumentParser(description="迁移Coinglass历史数据到MongoDB")
    parser.add_argument("--source", required=True, help="Coinglass数据源路径")
    
    args = parser.parse_args()
    
    async def main():
        result = await migrate_coinglass_data(args.source)
        print(f"迁移结果: {json.dumps(result, indent=2, default=str)}")
    
    asyncio.run(main())