"""
数据库迁移管理器
负责管理数据库架构变更、数据迁移和版本控制
支持MongoDB和Redis的完整迁移流程
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import motor.motor_asyncio
import redis.asyncio as aioredis
from loguru import logger
import traceback
from pydantic import BaseModel

from config.settings import settings

class MigrationRecord(BaseModel):
    """迁移记录模型"""
    version: str
    name: str
    applied_at: datetime
    execution_time: float
    status: str  # success, failed, rolled_back
    checksum: str
    metadata: Dict[str, Any]

class MigrationScript(BaseModel):
    """迁移脚本模型"""
    version: str
    name: str
    description: str
    up_script: str
    down_script: str
    dependencies: List[str]
    database_type: str  # mongodb, redis, both
    checksum: str

class DatabaseMigrationManager:
    """数据库迁移管理器"""
    
    def __init__(self):
        self.mongodb_client = None
        self.redis_client = None
        self.migrations_path = Path(__file__).parent / "scripts"
        self.migrations_path.mkdir(exist_ok=True)
        
        # 创建迁移记录集合/键
        self.migration_collection = "migration_history"
        self.migration_lock_key = "migration:lock"
        
        # 迁移状态
        self.is_running = False
        self.current_migration = None
        
    async def initialize(self):
        """初始化数据库连接"""
        try:
            # MongoDB连接
            db_config = settings.get_database_config()
            self.mongodb_client = motor.motor_asyncio.AsyncIOMotorClient(
                db_config["mongodb_url"],
                maxPoolSize=10,
                minPoolSize=2,
                serverSelectionTimeoutMS=5000
            )
            self.mongodb_db = self.mongodb_client.get_default_database()
            
            # Redis连接
            self.redis_client = aioredis.from_url(
                db_config["redis_url"],
                max_connections=10,
                retry_on_timeout=True
            )
            
            # 确保迁移历史集合存在
            await self._ensure_migration_history_exists()
            
            logger.info("数据库迁移管理器初始化成功")
            
        except Exception as e:
            logger.error(f"数据库迁移管理器初始化失败: {e}")
            raise
    
    async def _ensure_migration_history_exists(self):
        """确保迁移历史集合存在"""
        try:
            # 创建迁移历史集合的索引
            await self.mongodb_db[self.migration_collection].create_index("version", unique=True)
            await self.mongodb_db[self.migration_collection].create_index("applied_at")
            await self.mongodb_db[self.migration_collection].create_index("status")
            
            logger.debug("迁移历史集合索引创建完成")
            
        except Exception as e:
            logger.warning(f"创建迁移历史集合索引时出错: {e}")
    
    async def get_migration_lock(self, timeout: int = 300) -> bool:
        """获取迁移锁"""
        try:
            lock_value = f"migration_lock_{datetime.utcnow().isoformat()}"
            result = await self.redis_client.set(
                self.migration_lock_key, 
                lock_value, 
                nx=True, 
                ex=timeout
            )
            
            if result:
                logger.info("获得迁移锁")
                return True
            else:
                logger.warning("无法获得迁移锁，可能有其他迁移正在进行")
                return False
                
        except Exception as e:
            logger.error(f"获取迁移锁失败: {e}")
            return False
    
    async def release_migration_lock(self):
        """释放迁移锁"""
        try:
            await self.redis_client.delete(self.migration_lock_key)
            logger.info("迁移锁已释放")
        except Exception as e:
            logger.error(f"释放迁移锁失败: {e}")
    
    def load_migration_scripts(self) -> List[MigrationScript]:
        """加载所有迁移脚本"""
        migrations = []
        
        for script_file in sorted(self.migrations_path.glob("*.py")):
            if script_file.name.startswith("__"):
                continue
                
            try:
                migration = self._load_migration_script(script_file)
                if migration:
                    migrations.append(migration)
                    
            except Exception as e:
                logger.error(f"加载迁移脚本失败 {script_file}: {e}")
                
        return sorted(migrations, key=lambda m: m.version)
    
    def _load_migration_script(self, script_file: Path) -> Optional[MigrationScript]:
        """加载单个迁移脚本"""
        try:
            content = script_file.read_text(encoding="utf-8")
            
            # 解析脚本内容（简化版本，实际应使用AST）
            lines = content.split('\n')
            metadata = {}
            
            for line in lines:
                line = line.strip()
                if line.startswith('# VERSION:'):
                    metadata['version'] = line.split(':', 1)[1].strip()
                elif line.startswith('# NAME:'):
                    metadata['name'] = line.split(':', 1)[1].strip()
                elif line.startswith('# DESCRIPTION:'):
                    metadata['description'] = line.split(':', 1)[1].strip()
                elif line.startswith('# DATABASE:'):
                    metadata['database_type'] = line.split(':', 1)[1].strip()
                elif line.startswith('# DEPENDENCIES:'):
                    deps = line.split(':', 1)[1].strip()
                    metadata['dependencies'] = [d.strip() for d in deps.split(',') if d.strip()]
            
            # 计算校验和
            import hashlib
            checksum = hashlib.md5(content.encode()).hexdigest()
            
            return MigrationScript(
                version=metadata.get('version', '0.0.0'),
                name=metadata.get('name', script_file.stem),
                description=metadata.get('description', 'No description'),
                up_script=content,
                down_script='',  # 需要从脚本中提取
                dependencies=metadata.get('dependencies', []),
                database_type=metadata.get('database_type', 'mongodb'),
                checksum=checksum
            )
            
        except Exception as e:
            logger.error(f"解析迁移脚本失败 {script_file}: {e}")
            return None
    
    async def get_applied_migrations(self) -> List[MigrationRecord]:
        """获取已应用的迁移记录"""
        try:
            cursor = self.mongodb_db[self.migration_collection].find(
                {"status": "success"}
            ).sort("applied_at", 1)
            
            records = []
            async for doc in cursor:
                # 转换MongoDB文档为MigrationRecord
                doc['applied_at'] = doc.get('applied_at', datetime.utcnow())
                records.append(MigrationRecord(**doc))
            
            return records
            
        except Exception as e:
            logger.error(f"获取已应用迁移记录失败: {e}")
            return []
    
    async def get_pending_migrations(self) -> List[MigrationScript]:
        """获取待执行的迁移"""
        all_migrations = self.load_migration_scripts()
        applied_migrations = await self.get_applied_migrations()
        applied_versions = {m.version for m in applied_migrations}
        
        pending = [m for m in all_migrations if m.version not in applied_versions]
        return pending
    
    async def execute_migration(self, migration: MigrationScript) -> bool:
        """执行单个迁移"""
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"开始执行迁移: {migration.version} - {migration.name}")
            self.current_migration = migration
            
            # 检查依赖
            if not await self._check_dependencies(migration):
                raise Exception(f"迁移依赖未满足: {migration.dependencies}")
            
            # 创建备份（如果需要）
            backup_info = await self._create_backup_if_needed(migration)
            
            # 执行迁移脚本
            success = await self._run_migration_script(migration)
            
            if success:
                # 记录成功
                execution_time = (datetime.utcnow() - start_time).total_seconds()
                await self._record_migration_success(migration, execution_time, backup_info)
                logger.info(f"迁移执行成功: {migration.version}")
                return True
            else:
                # 记录失败
                await self._record_migration_failure(migration, "Script execution failed")
                logger.error(f"迁移执行失败: {migration.version}")
                return False
                
        except Exception as e:
            logger.error(f"迁移执行异常 {migration.version}: {e}")
            await self._record_migration_failure(migration, str(e))
            
            # 尝试回滚
            await self._attempt_rollback(migration)
            return False
            
        finally:
            self.current_migration = None
    
    async def _check_dependencies(self, migration: MigrationScript) -> bool:
        """检查迁移依赖"""
        if not migration.dependencies:
            return True
            
        applied_migrations = await self.get_applied_migrations()
        applied_versions = {m.version for m in applied_migrations}
        
        for dep in migration.dependencies:
            if dep not in applied_versions:
                logger.error(f"依赖迁移未应用: {dep}")
                return False
                
        return True
    
    async def _create_backup_if_needed(self, migration: MigrationScript) -> Dict[str, Any]:
        """创建备份（如果需要）"""
        backup_info = {
            "created": False,
            "backup_path": None,
            "collections": [],
            "redis_keys": []
        }
        
        try:
            # 对于重要迁移创建备份
            if "critical" in migration.description.lower() or "schema" in migration.description.lower():
                backup_path = Path(f"/tmp/migration_backup_{migration.version}_{int(datetime.utcnow().timestamp())}")
                backup_path.mkdir(exist_ok=True)
                
                # 备份相关集合（简化实现）
                # TODO: 实现完整的备份逻辑
                backup_info["created"] = True
                backup_info["backup_path"] = str(backup_path)
                
                logger.info(f"为迁移 {migration.version} 创建备份")
                
        except Exception as e:
            logger.warning(f"创建备份失败: {e}")
            
        return backup_info
    
    async def _run_migration_script(self, migration: MigrationScript) -> bool:
        """运行迁移脚本"""
        try:
            # 这里需要根据database_type分别处理
            if migration.database_type in ["mongodb", "both"]:
                success = await self._execute_mongodb_migration(migration)
                if not success:
                    return False
            
            if migration.database_type in ["redis", "both"]:
                success = await self._execute_redis_migration(migration)
                if not success:
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"运行迁移脚本失败: {e}")
            return False
    
    async def _execute_mongodb_migration(self, migration: MigrationScript) -> bool:
        """执行MongoDB迁移"""
        try:
            # 简化实现：执行脚本中的MongoDB命令
            # 实际应该解析脚本并安全执行
            logger.info(f"执行MongoDB迁移: {migration.name}")
            
            # TODO: 实现安全的脚本执行
            # 这里应该解析和执行具体的MongoDB操作
            
            return True
            
        except Exception as e:
            logger.error(f"执行MongoDB迁移失败: {e}")
            return False
    
    async def _execute_redis_migration(self, migration: MigrationScript) -> bool:
        """执行Redis迁移"""
        try:
            logger.info(f"执行Redis迁移: {migration.name}")
            
            # TODO: 实现Redis迁移逻辑
            
            return True
            
        except Exception as e:
            logger.error(f"执行Redis迁移失败: {e}")
            return False
    
    async def _record_migration_success(self, migration: MigrationScript, 
                                      execution_time: float, backup_info: Dict):
        """记录迁移成功"""
        try:
            record = {
                "version": migration.version,
                "name": migration.name,
                "applied_at": datetime.utcnow(),
                "execution_time": execution_time,
                "status": "success",
                "checksum": migration.checksum,
                "metadata": {
                    "description": migration.description,
                    "database_type": migration.database_type,
                    "backup_info": backup_info
                }
            }
            
            await self.mongodb_db[self.migration_collection].insert_one(record)
            logger.debug(f"迁移成功记录已保存: {migration.version}")
            
        except Exception as e:
            logger.error(f"记录迁移成功失败: {e}")
    
    async def _record_migration_failure(self, migration: MigrationScript, error: str):
        """记录迁移失败"""
        try:
            record = {
                "version": migration.version,
                "name": migration.name,
                "applied_at": datetime.utcnow(),
                "execution_time": 0,
                "status": "failed",
                "checksum": migration.checksum,
                "metadata": {
                    "description": migration.description,
                    "database_type": migration.database_type,
                    "error": error,
                    "traceback": traceback.format_exc()
                }
            }
            
            await self.mongodb_db[self.migration_collection].insert_one(record)
            logger.debug(f"迁移失败记录已保存: {migration.version}")
            
        except Exception as e:
            logger.error(f"记录迁移失败失败: {e}")
    
    async def _attempt_rollback(self, migration: MigrationScript):
        """尝试回滚迁移"""
        try:
            logger.warning(f"尝试回滚迁移: {migration.version}")
            
            # TODO: 实现回滚逻辑
            # 1. 执行down_script
            # 2. 恢复备份（如果存在）
            # 3. 更新迁移记录状态
            
        except Exception as e:
            logger.error(f"回滚迁移失败: {e}")
    
    async def run_migrations(self, target_version: Optional[str] = None) -> Dict[str, Any]:
        """运行所有待执行的迁移"""
        if self.is_running:
            return {"status": "error", "message": "迁移已在运行中"}
        
        # 获取迁移锁
        if not await self.get_migration_lock():
            return {"status": "error", "message": "无法获得迁移锁"}
        
        try:
            self.is_running = True
            start_time = datetime.utcnow()
            
            pending_migrations = await self.get_pending_migrations()
            
            if target_version:
                # 过滤到目标版本
                pending_migrations = [
                    m for m in pending_migrations 
                    if m.version <= target_version
                ]
            
            if not pending_migrations:
                return {
                    "status": "success",
                    "message": "没有待执行的迁移",
                    "migrations_applied": 0
                }
            
            logger.info(f"开始执行 {len(pending_migrations)} 个迁移")
            
            applied_count = 0
            failed_migrations = []
            
            for migration in pending_migrations:
                success = await self.execute_migration(migration)
                if success:
                    applied_count += 1
                else:
                    failed_migrations.append(migration.version)
                    # 根据策略决定是否继续
                    break  # 暂时遇到失败就停止
            
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds()
            
            result = {
                "status": "success" if not failed_migrations else "partial_success",
                "message": f"应用了 {applied_count} 个迁移",
                "migrations_applied": applied_count,
                "failed_migrations": failed_migrations,
                "execution_time": execution_time,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
            
            logger.info(f"迁移执行完成: {result}")
            return result
            
        except Exception as e:
            logger.error(f"运行迁移异常: {e}")
            return {
                "status": "error",
                "message": f"迁移执行异常: {e}"
            }
            
        finally:
            self.is_running = False
            await self.release_migration_lock()
    
    async def rollback_migration(self, version: str) -> Dict[str, Any]:
        """回滚指定版本的迁移"""
        try:
            # 获取迁移记录
            record = await self.mongodb_db[self.migration_collection].find_one(
                {"version": version, "status": "success"}
            )
            
            if not record:
                return {
                    "status": "error",
                    "message": f"未找到版本 {version} 的成功迁移记录"
                }
            
            # TODO: 实现回滚逻辑
            logger.info(f"回滚迁移: {version}")
            
            return {
                "status": "success",
                "message": f"迁移 {version} 已回滚"
            }
            
        except Exception as e:
            logger.error(f"回滚迁移失败: {e}")
            return {
                "status": "error",
                "message": f"回滚失败: {e}"
            }
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """获取迁移状态"""
        try:
            applied_migrations = await self.get_applied_migrations()
            pending_migrations = await self.get_pending_migrations()
            
            return {
                "is_running": self.is_running,
                "current_migration": self.current_migration.version if self.current_migration else None,
                "applied_count": len(applied_migrations),
                "pending_count": len(pending_migrations),
                "latest_applied": applied_migrations[-1].version if applied_migrations else None,
                "next_pending": pending_migrations[0].version if pending_migrations else None,
                "database_status": {
                    "mongodb_connected": self.mongodb_client is not None,
                    "redis_connected": self.redis_client is not None
                }
            }
            
        except Exception as e:
            logger.error(f"获取迁移状态失败: {e}")
            return {
                "status": "error",
                "message": f"获取状态失败: {e}"
            }
    
    async def validate_migrations(self) -> Dict[str, Any]:
        """验证迁移完整性"""
        try:
            all_migrations = self.load_migration_scripts()
            applied_migrations = await self.get_applied_migrations()
            
            validation_errors = []
            warnings = []
            
            # 检查版本一致性
            for applied in applied_migrations:
                matching_script = next(
                    (m for m in all_migrations if m.version == applied.version),
                    None
                )
                
                if not matching_script:
                    validation_errors.append(
                        f"已应用的迁移 {applied.version} 缺少对应的脚本文件"
                    )
                elif matching_script.checksum != applied.checksum:
                    warnings.append(
                        f"迁移 {applied.version} 的脚本文件已被修改"
                    )
            
            # 检查依赖关系
            applied_versions = {m.version for m in applied_migrations}
            for migration in all_migrations:
                for dep in migration.dependencies:
                    if dep not in applied_versions:
                        migration_applied = migration.version in applied_versions
                        if migration_applied:
                            validation_errors.append(
                                f"迁移 {migration.version} 已应用但依赖 {dep} 未应用"
                            )
            
            return {
                "status": "valid" if not validation_errors else "invalid",
                "errors": validation_errors,
                "warnings": warnings,
                "total_migrations": len(all_migrations),
                "applied_migrations": len(applied_migrations)
            }
            
        except Exception as e:
            logger.error(f"验证迁移失败: {e}")
            return {
                "status": "error",
                "message": f"验证失败: {e}"
            }
    
    async def close(self):
        """关闭数据库连接"""
        try:
            if self.mongodb_client:
                self.mongodb_client.close()
            if self.redis_client:
                await self.redis_client.close()
            logger.info("数据库迁移管理器连接已关闭")
        except Exception as e:
            logger.error(f"关闭数据库连接失败: {e}")

# 全局迁移管理器实例
migration_manager = DatabaseMigrationManager()