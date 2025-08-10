"""
数据库验证器

测试MongoDB和Redis数据库连接和性能
"""

import time
from typing import Dict, Any, List

class DatabaseValidator:
    """数据库连接验证器"""
    
    def __init__(self):
        self.mongodb_available = self._check_mongodb()
        self.redis_available = self._check_redis()
    
    def _check_mongodb(self) -> bool:
        """检查MongoDB可用性"""
        try:
            import pymongo
            client = pymongo.MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=3000)
            client.admin.command('ping')
            client.close()
            return True
        except Exception:
            return False
    
    def _check_redis(self) -> bool:
        """检查Redis可用性"""
        try:
            import redis
            client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            client.ping()
            client.close()
            return True
        except Exception:
            return False
    
    def test_mongodb_connection(self) -> Dict[str, Any]:
        """测试MongoDB连接"""
        start_time = time.time()
        
        try:
            if not self.mongodb_available:
                return {
                    "status": "FAIL",
                    "message": "MongoDB连接失败",
                    "details": {"reason": "服务不可用或连接超时"},
                    "metrics": {"test_duration": (time.time() - start_time) * 1000}
                }
            
            import pymongo
            
            # 连接性能测试
            connection_times = []
            for _ in range(5):
                conn_start = time.time()
                client = pymongo.MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
                client.admin.command('ping')
                client.close()
                conn_time = (time.time() - conn_start) * 1000
                connection_times.append(conn_time)
            
            avg_connection_time = sum(connection_times) / len(connection_times)
            
            return {
                "status": "PASS",
                "message": "MongoDB连接测试通过",
                "details": {
                    "connection_attempts": len(connection_times),
                    "average_connection_time": avg_connection_time,
                    "max_connection_time": max(connection_times),
                    "min_connection_time": min(connection_times),
                    "all_connections_successful": True
                },
                "metrics": {
                    "connection_time": avg_connection_time,
                    "connection_stability": max(connection_times) - min(connection_times)
                }
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"MongoDB连接测试失败: {e}",
                "details": {"error_type": type(e).__name__},
                "metrics": {"test_duration": (time.time() - start_time) * 1000}
            }
    
    def test_redis_connection(self) -> Dict[str, Any]:
        """测试Redis连接"""
        start_time = time.time()
        
        try:
            if not self.redis_available:
                return {
                    "status": "FAIL", 
                    "message": "Redis连接失败",
                    "details": {"reason": "服务不可用或连接超时"},
                    "metrics": {"test_duration": (time.time() - start_time) * 1000}
                }
            
            import redis
            
            # 连接性能测试
            connection_times = []
            for _ in range(5):
                conn_start = time.time()
                client = redis.Redis(host='localhost', port=6379, decode_responses=True)
                client.ping()
                client.close()
                conn_time = (time.time() - conn_start) * 1000
                connection_times.append(conn_time)
            
            avg_connection_time = sum(connection_times) / len(connection_times)
            
            return {
                "status": "PASS",
                "message": "Redis连接测试通过",
                "details": {
                    "connection_attempts": len(connection_times),
                    "average_connection_time": avg_connection_time,
                    "max_connection_time": max(connection_times),
                    "min_connection_time": min(connection_times),
                    "all_connections_successful": True
                },
                "metrics": {
                    "connection_time": avg_connection_time,
                    "connection_stability": max(connection_times) - min(connection_times)
                }
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Redis连接测试失败: {e}",
                "details": {"error_type": type(e).__name__},
                "metrics": {"test_duration": (time.time() - start_time) * 1000}
            }
    
    def test_mongodb_operations(self) -> Dict[str, Any]:
        """测试MongoDB操作性能"""
        start_time = time.time()
        
        try:
            if not self.mongodb_available:
                return {
                    "status": "SKIP",
                    "message": "MongoDB不可用，跳过操作测试",
                    "details": {},
                    "metrics": {"test_duration": (time.time() - start_time) * 1000}
                }
            
            import pymongo
            from datetime import datetime
            
            client = pymongo.MongoClient('mongodb://localhost:27017/')
            db = client['test_trading_system']
            collection = db['test_operations']
            
            # 测试写入性能
            write_times = []
            test_documents = []
            
            for i in range(10):
                write_start = time.time()
                
                doc = {
                    "symbol": "BTC/USDT",
                    "price": 45000 + i * 100,
                    "timestamp": datetime.utcnow(),
                    "test_id": i
                }
                
                result = collection.insert_one(doc)
                test_documents.append(result.inserted_id)
                
                write_time = (time.time() - write_start) * 1000
                write_times.append(write_time)
            
            # 测试读取性能
            read_times = []
            for doc_id in test_documents:
                read_start = time.time()
                collection.find_one({"_id": doc_id})
                read_time = (time.time() - read_start) * 1000
                read_times.append(read_time)
            
            # 清理测试数据
            collection.delete_many({"test_id": {"$exists": True}})
            client.close()
            
            avg_write_time = sum(write_times) / len(write_times)
            avg_read_time = sum(read_times) / len(read_times)
            
            return {
                "status": "PASS",
                "message": "MongoDB操作性能测试通过",
                "details": {
                    "documents_written": len(test_documents),
                    "documents_read": len(read_times),
                    "average_write_time": avg_write_time,
                    "average_read_time": avg_read_time,
                    "write_throughput": 1000 / avg_write_time,
                    "read_throughput": 1000 / avg_read_time
                },
                "metrics": {
                    "write_time": avg_write_time,
                    "query_time": avg_read_time
                }
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"MongoDB操作测试失败: {e}",
                "details": {"error_type": type(e).__name__},
                "metrics": {"test_duration": (time.time() - start_time) * 1000}
            }
    
    def test_redis_operations(self) -> Dict[str, Any]:
        """测试Redis操作性能"""
        start_time = time.time()
        
        try:
            if not self.redis_available:
                return {
                    "status": "SKIP",
                    "message": "Redis不可用，跳过操作测试",
                    "details": {},
                    "metrics": {"test_duration": (time.time() - start_time) * 1000}
                }
            
            import redis
            import json
            
            client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            
            # 测试SET操作性能
            set_times = []
            test_keys = []
            
            for i in range(20):
                set_start = time.time()
                
                key = f"test:price:{i}"
                value = json.dumps({
                    "symbol": "BTC/USDT",
                    "price": 45000 + i * 100,
                    "timestamp": time.time()
                })
                
                client.set(key, value, ex=60)  # 60秒过期
                test_keys.append(key)
                
                set_time = (time.time() - set_start) * 1000
                set_times.append(set_time)
            
            # 测试GET操作性能
            get_times = []
            for key in test_keys:
                get_start = time.time()
                value = client.get(key)
                if value:
                    json.loads(value)
                get_time = (time.time() - get_start) * 1000
                get_times.append(get_time)
            
            # 清理测试数据
            client.delete(*test_keys)
            client.close()
            
            avg_set_time = sum(set_times) / len(set_times)
            avg_get_time = sum(get_times) / len(get_times)
            
            return {
                "status": "PASS",
                "message": "Redis操作性能测试通过",
                "details": {
                    "keys_set": len(test_keys),
                    "keys_retrieved": len(get_times),
                    "average_set_time": avg_set_time,
                    "average_get_time": avg_get_time,
                    "set_throughput": 1000 / avg_set_time,
                    "get_throughput": 1000 / avg_get_time
                },
                "metrics": {
                    "write_time": avg_set_time,
                    "query_time": avg_get_time
                }
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Redis操作测试失败: {e}",
                "details": {"error_type": type(e).__name__},
                "metrics": {"test_duration": (time.time() - start_time) * 1000}
            }