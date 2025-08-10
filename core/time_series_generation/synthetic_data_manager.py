"""
合成数据管理系统
管理CTBench生成的合成金融数据，提供数据存储、检索、版本控制和质量管理功能
"""

import numpy as np
import pandas as pd
import h5py
import sqlite3
import pickle
import json
import os
import shutil
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import logging
import threading
from pathlib import Path
import warnings
import asyncio
from concurrent.futures import ThreadPoolExecutor
import zipfile
import tarfile

from .ctbench_evaluator import EvaluationMetrics
from .model_orchestrator import GenerationResult

warnings.filterwarnings('ignore')

class DatasetType(Enum):
    """数据集类型"""
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"
    PRODUCTION = "production"
    EXPERIMENTAL = "experimental"

class DataFormat(Enum):
    """数据格式"""
    HDF5 = "hdf5"
    CSV = "csv"
    NUMPY = "numpy"
    PARQUET = "parquet"
    PICKLE = "pickle"

class DataQuality(Enum):
    """数据质量等级"""
    EXCELLENT = "excellent"  # 综合得分 > 0.9
    GOOD = "good"           # 综合得分 > 0.7
    FAIR = "fair"           # 综合得分 > 0.5
    POOR = "poor"           # 综合得分 <= 0.5

@dataclass
class DatasetMetadata:
    """数据集元数据"""
    dataset_id: str
    name: str
    description: str
    
    # 基本信息
    dataset_type: DatasetType
    data_format: DataFormat
    created_at: datetime
    updated_at: datetime
    
    # 数据统计
    num_samples: int
    num_features: int
    sequence_length: int
    file_size: int  # 字节
    
    # 生成信息
    generator_model: str
    generation_config: Dict[str, Any]
    generation_time: float
    
    # 质量信息
    quality_score: float
    quality_grade: DataQuality
    evaluation_metrics: Optional[Dict[str, Any]] = None
    
    # 使用统计
    download_count: int = 0
    last_accessed: Optional[datetime] = None
    usage_count: int = 0
    
    # 版本信息
    version: str = "1.0.0"
    parent_version: Optional[str] = None
    
    # 标签和分类
    tags: List[str] = field(default_factory=list)
    market_regimes: List[str] = field(default_factory=list)
    
    # 文件路径
    file_path: str = ""
    metadata_path: str = ""
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            self.created_at = datetime.fromisoformat(self.created_at)
        if isinstance(self.updated_at, str):
            self.updated_at = datetime.fromisoformat(self.updated_at)
        if self.last_accessed and isinstance(self.last_accessed, str):
            self.last_accessed = datetime.fromisoformat(self.last_accessed)

@dataclass
class DatasetQuery:
    """数据集查询条件"""
    dataset_type: Optional[DatasetType] = None
    quality_grade: Optional[List[DataQuality]] = None
    min_samples: Optional[int] = None
    max_samples: Optional[int] = None
    sequence_length: Optional[int] = None
    generator_model: Optional[str] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    tags: Optional[List[str]] = None
    market_regimes: Optional[List[str]] = None
    min_quality_score: Optional[float] = None
    limit: int = 100
    offset: int = 0

class DatasetStorage:
    """数据集存储管理器"""
    
    def __init__(self, storage_root: str):
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.data_dir = self.storage_root / "data"
        self.metadata_dir = self.storage_root / "metadata"
        self.archive_dir = self.storage_root / "archive"
        self.temp_dir = self.storage_root / "temp"
        
        for dir_path in [self.data_dir, self.metadata_dir, self.archive_dir, self.temp_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger("DatasetStorage")
    
    def save_dataset(self, dataset_id: str, data: np.ndarray, 
                    metadata: DatasetMetadata, data_format: DataFormat) -> str:
        """保存数据集"""
        try:
            # 确定文件路径
            filename = self._generate_filename(dataset_id, data_format)
            file_path = self.data_dir / filename
            
            # 保存数据
            if data_format == DataFormat.HDF5:
                with h5py.File(file_path, 'w') as f:
                    f.create_dataset('data', data=data)
                    f.attrs['dataset_id'] = dataset_id
                    f.attrs['created_at'] = metadata.created_at.isoformat()
            
            elif data_format == DataFormat.NUMPY:
                np.save(file_path, data)
            
            elif data_format == DataFormat.CSV:
                if data.ndim == 3:
                    # 3D数据需要重塑为2D
                    reshaped_data = data.reshape(-1, data.shape[-1])
                    df = pd.DataFrame(reshaped_data)
                else:
                    df = pd.DataFrame(data)
                df.to_csv(file_path, index=False)
            
            elif data_format == DataFormat.PARQUET:
                if data.ndim == 3:
                    reshaped_data = data.reshape(-1, data.shape[-1])
                    df = pd.DataFrame(reshaped_data)
                else:
                    df = pd.DataFrame(data)
                df.to_parquet(file_path)
            
            elif data_format == DataFormat.PICKLE:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            
            else:
                raise ValueError(f"不支持的数据格式: {data_format}")
            
            # 更新元数据路径和文件大小
            metadata.file_path = str(file_path)
            metadata.file_size = file_path.stat().st_size
            
            # 保存元数据
            metadata_path = self.metadata_dir / f"{dataset_id}.json"
            self._save_metadata(metadata, metadata_path)
            metadata.metadata_path = str(metadata_path)
            
            self.logger.info(f"数据集已保存: {dataset_id} -> {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"保存数据集失败 {dataset_id}: {e}")
            raise
    
    def load_dataset(self, dataset_id: str, metadata: DatasetMetadata) -> np.ndarray:
        """加载数据集"""
        try:
            file_path = Path(metadata.file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"数据文件不存在: {file_path}")
            
            data_format = metadata.data_format
            
            if data_format == DataFormat.HDF5:
                with h5py.File(file_path, 'r') as f:
                    data = f['data'][:]
            
            elif data_format == DataFormat.NUMPY:
                data = np.load(file_path)
            
            elif data_format == DataFormat.CSV:
                df = pd.read_csv(file_path)
                data = df.values
                # 如果需要，重塑回3D
                if metadata.sequence_length > 1:
                    data = data.reshape(-1, metadata.sequence_length, metadata.num_features)
            
            elif data_format == DataFormat.PARQUET:
                df = pd.read_parquet(file_path)
                data = df.values
                if metadata.sequence_length > 1:
                    data = data.reshape(-1, metadata.sequence_length, metadata.num_features)
            
            elif data_format == DataFormat.PICKLE:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
            
            else:
                raise ValueError(f"不支持的数据格式: {data_format}")
            
            # 更新访问统计
            metadata.last_accessed = datetime.utcnow()
            metadata.usage_count += 1
            
            self.logger.info(f"数据集已加载: {dataset_id}")
            return data
            
        except Exception as e:
            self.logger.error(f"加载数据集失败 {dataset_id}: {e}")
            raise
    
    def delete_dataset(self, dataset_id: str, metadata: DatasetMetadata) -> bool:
        """删除数据集"""
        try:
            # 删除数据文件
            file_path = Path(metadata.file_path)
            if file_path.exists():
                file_path.unlink()
            
            # 删除元数据文件
            metadata_path = Path(metadata.metadata_path)
            if metadata_path.exists():
                metadata_path.unlink()
            
            self.logger.info(f"数据集已删除: {dataset_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"删除数据集失败 {dataset_id}: {e}")
            return False
    
    def archive_dataset(self, dataset_id: str, metadata: DatasetMetadata) -> str:
        """归档数据集"""
        try:
            # 创建归档文件
            archive_path = self.archive_dir / f"{dataset_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.tar.gz"
            
            with tarfile.open(archive_path, 'w:gz') as tar:
                # 添加数据文件
                tar.add(metadata.file_path, arcname=f"{dataset_id}_data")
                # 添加元数据文件
                tar.add(metadata.metadata_path, arcname=f"{dataset_id}_metadata.json")
            
            self.logger.info(f"数据集已归档: {dataset_id} -> {archive_path}")
            return str(archive_path)
            
        except Exception as e:
            self.logger.error(f"归档数据集失败 {dataset_id}: {e}")
            raise
    
    def _generate_filename(self, dataset_id: str, data_format: DataFormat) -> str:
        """生成文件名"""
        extensions = {
            DataFormat.HDF5: ".h5",
            DataFormat.NUMPY: ".npy",
            DataFormat.CSV: ".csv",
            DataFormat.PARQUET: ".parquet",
            DataFormat.PICKLE: ".pkl"
        }
        
        ext = extensions.get(data_format, ".dat")
        return f"{dataset_id}{ext}"
    
    def _save_metadata(self, metadata: DatasetMetadata, metadata_path: Path):
        """保存元数据"""
        # 转换为字典并序列化datetime
        metadata_dict = asdict(metadata)
        
        # 处理datetime字段
        for key in ['created_at', 'updated_at', 'last_accessed']:
            if key in metadata_dict and metadata_dict[key]:
                if isinstance(metadata_dict[key], datetime):
                    metadata_dict[key] = metadata_dict[key].isoformat()
        
        # 处理枚举类型
        metadata_dict['dataset_type'] = metadata_dict['dataset_type'].value
        metadata_dict['data_format'] = metadata_dict['data_format'].value
        metadata_dict['quality_grade'] = metadata_dict['quality_grade'].value
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)

class DatasetDatabase:
    """数据集数据库管理器"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.RLock()
        
        # 初始化数据库
        self._init_database()
        self.logger = logging.getLogger("DatasetDatabase")
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    dataset_type TEXT NOT NULL,
                    data_format TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    num_samples INTEGER NOT NULL,
                    num_features INTEGER NOT NULL,
                    sequence_length INTEGER NOT NULL,
                    file_size INTEGER NOT NULL,
                    generator_model TEXT NOT NULL,
                    generation_config TEXT,
                    generation_time REAL NOT NULL,
                    quality_score REAL NOT NULL,
                    quality_grade TEXT NOT NULL,
                    evaluation_metrics TEXT,
                    download_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    usage_count INTEGER DEFAULT 0,
                    version TEXT NOT NULL,
                    parent_version TEXT,
                    tags TEXT,
                    market_regimes TEXT,
                    file_path TEXT NOT NULL,
                    metadata_path TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_dataset_type ON datasets(dataset_type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_quality_score ON datasets(quality_score)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON datasets(created_at)
            """)
            
            conn.commit()
    
    def insert_dataset(self, metadata: DatasetMetadata) -> bool:
        """插入数据集记录"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO datasets VALUES (
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                        )
                    """, (
                        metadata.dataset_id,
                        metadata.name,
                        metadata.description,
                        metadata.dataset_type.value,
                        metadata.data_format.value,
                        metadata.created_at.isoformat(),
                        metadata.updated_at.isoformat(),
                        metadata.num_samples,
                        metadata.num_features,
                        metadata.sequence_length,
                        metadata.file_size,
                        metadata.generator_model,
                        json.dumps(metadata.generation_config),
                        metadata.generation_time,
                        metadata.quality_score,
                        metadata.quality_grade.value,
                        json.dumps(metadata.evaluation_metrics) if metadata.evaluation_metrics else None,
                        metadata.download_count,
                        metadata.last_accessed.isoformat() if metadata.last_accessed else None,
                        metadata.usage_count,
                        metadata.version,
                        metadata.parent_version,
                        json.dumps(metadata.tags),
                        json.dumps(metadata.market_regimes),
                        metadata.file_path,
                        metadata.metadata_path
                    ))
                    conn.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"插入数据集记录失败 {metadata.dataset_id}: {e}")
            return False
    
    def get_dataset(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """获取数据集记录"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(
                        "SELECT * FROM datasets WHERE dataset_id = ?",
                        (dataset_id,)
                    )
                    row = cursor.fetchone()
                    
                    if row:
                        return self._row_to_metadata(row)
                    return None
                    
        except Exception as e:
            self.logger.error(f"获取数据集记录失败 {dataset_id}: {e}")
            return None
    
    def query_datasets(self, query: DatasetQuery) -> List[DatasetMetadata]:
        """查询数据集"""
        try:
            conditions = []
            params = []
            
            # 构建查询条件
            if query.dataset_type:
                conditions.append("dataset_type = ?")
                params.append(query.dataset_type.value)
            
            if query.quality_grade:
                placeholders = ','.join(['?'] * len(query.quality_grade))
                conditions.append(f"quality_grade IN ({placeholders})")
                params.extend([grade.value for grade in query.quality_grade])
            
            if query.min_samples:
                conditions.append("num_samples >= ?")
                params.append(query.min_samples)
            
            if query.max_samples:
                conditions.append("num_samples <= ?")
                params.append(query.max_samples)
            
            if query.sequence_length:
                conditions.append("sequence_length = ?")
                params.append(query.sequence_length)
            
            if query.generator_model:
                conditions.append("generator_model = ?")
                params.append(query.generator_model)
            
            if query.created_after:
                conditions.append("created_at >= ?")
                params.append(query.created_after.isoformat())
            
            if query.created_before:
                conditions.append("created_at <= ?")
                params.append(query.created_before.isoformat())
            
            if query.min_quality_score:
                conditions.append("quality_score >= ?")
                params.append(query.min_quality_score)
            
            # 构建SQL查询
            sql = "SELECT * FROM datasets"
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            
            sql += " ORDER BY quality_score DESC, created_at DESC"
            sql += f" LIMIT {query.limit} OFFSET {query.offset}"
            
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(sql, params)
                    rows = cursor.fetchall()
                    
                    results = []
                    for row in rows:
                        metadata = self._row_to_metadata(row)
                        
                        # 标签过滤
                        if query.tags:
                            if not any(tag in metadata.tags for tag in query.tags):
                                continue
                        
                        # 市场状态过滤
                        if query.market_regimes:
                            if not any(regime in metadata.market_regimes for regime in query.market_regimes):
                                continue
                        
                        results.append(metadata)
                    
                    return results
                    
        except Exception as e:
            self.logger.error(f"查询数据集失败: {e}")
            return []
    
    def update_usage_stats(self, dataset_id: str):
        """更新使用统计"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        UPDATE datasets 
                        SET usage_count = usage_count + 1,
                            last_accessed = ?
                        WHERE dataset_id = ?
                    """, (datetime.utcnow().isoformat(), dataset_id))
                    conn.commit()
                    
        except Exception as e:
            self.logger.error(f"更新使用统计失败 {dataset_id}: {e}")
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """删除数据集记录"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "DELETE FROM datasets WHERE dataset_id = ?",
                        (dataset_id,)
                    )
                    conn.commit()
                    return cursor.rowcount > 0
                    
        except Exception as e:
            self.logger.error(f"删除数据集记录失败 {dataset_id}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    # 总数统计
                    cursor = conn.execute("SELECT COUNT(*) FROM datasets")
                    total_datasets = cursor.fetchone()[0]
                    
                    # 类型统计
                    cursor = conn.execute("""
                        SELECT dataset_type, COUNT(*) 
                        FROM datasets 
                        GROUP BY dataset_type
                    """)
                    type_stats = dict(cursor.fetchall())
                    
                    # 质量统计
                    cursor = conn.execute("""
                        SELECT quality_grade, COUNT(*) 
                        FROM datasets 
                        GROUP BY quality_grade
                    """)
                    quality_stats = dict(cursor.fetchall())
                    
                    # 存储统计
                    cursor = conn.execute("SELECT SUM(file_size) FROM datasets")
                    total_size = cursor.fetchone()[0] or 0
                    
                    return {
                        "total_datasets": total_datasets,
                        "type_distribution": type_stats,
                        "quality_distribution": quality_stats,
                        "total_storage_size": total_size
                    }
                    
        except Exception as e:
            self.logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def _row_to_metadata(self, row: sqlite3.Row) -> DatasetMetadata:
        """将数据库行转换为元数据对象"""
        return DatasetMetadata(
            dataset_id=row['dataset_id'],
            name=row['name'],
            description=row['description'],
            dataset_type=DatasetType(row['dataset_type']),
            data_format=DataFormat(row['data_format']),
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            num_samples=row['num_samples'],
            num_features=row['num_features'],
            sequence_length=row['sequence_length'],
            file_size=row['file_size'],
            generator_model=row['generator_model'],
            generation_config=json.loads(row['generation_config']) if row['generation_config'] else {},
            generation_time=row['generation_time'],
            quality_score=row['quality_score'],
            quality_grade=DataQuality(row['quality_grade']),
            evaluation_metrics=json.loads(row['evaluation_metrics']) if row['evaluation_metrics'] else None,
            download_count=row['download_count'],
            last_accessed=datetime.fromisoformat(row['last_accessed']) if row['last_accessed'] else None,
            usage_count=row['usage_count'],
            version=row['version'],
            parent_version=row['parent_version'],
            tags=json.loads(row['tags']) if row['tags'] else [],
            market_regimes=json.loads(row['market_regimes']) if row['market_regimes'] else [],
            file_path=row['file_path'],
            metadata_path=row['metadata_path']
        )

class SyntheticDataManager:
    """合成数据管理器"""
    
    def __init__(self, storage_root: str = "data/synthetic_datasets",
                 db_path: str = "data/datasets.db"):
        
        self.storage = DatasetStorage(storage_root)
        self.database = DatasetDatabase(db_path)
        
        # 线程池用于异步操作
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # 缓存
        self.metadata_cache: Dict[str, DatasetMetadata] = {}
        self.cache_lock = threading.RLock()
        
        self.logger = logging.getLogger("SyntheticDataManager")
    
    def create_dataset_from_generation(self, generation_result: GenerationResult,
                                     name: str, description: str,
                                     dataset_type: DatasetType = DatasetType.EXPERIMENTAL,
                                     data_format: DataFormat = DataFormat.HDF5,
                                     tags: List[str] = None,
                                     market_regimes: List[str] = None,
                                     evaluation_metrics: EvaluationMetrics = None) -> str:
        """从生成结果创建数据集"""
        
        # 生成数据集ID
        dataset_id = self._generate_dataset_id(name, generation_result.model_id)
        
        # 计算质量得分和等级
        quality_score = evaluation_metrics.overall_quality_score if evaluation_metrics else 0.5
        quality_grade = self._calculate_quality_grade(quality_score)
        
        # 创建元数据
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            name=name,
            description=description,
            dataset_type=dataset_type,
            data_format=data_format,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            num_samples=generation_result.samples.shape[0],
            num_features=generation_result.samples.shape[-1],
            sequence_length=generation_result.samples.shape[1] if generation_result.samples.ndim > 2 else 1,
            file_size=0,  # 将在保存时更新
            generator_model=generation_result.model_id,
            generation_config=generation_result.metadata or {},
            generation_time=generation_result.generation_time,
            quality_score=quality_score,
            quality_grade=quality_grade,
            evaluation_metrics=asdict(evaluation_metrics) if evaluation_metrics else None,
            tags=tags or [],
            market_regimes=market_regimes or []
        )
        
        try:
            # 保存数据和元数据
            file_path = self.storage.save_dataset(dataset_id, generation_result.samples, metadata, data_format)
            
            # 插入数据库记录
            success = self.database.insert_dataset(metadata)
            if not success:
                # 如果数据库插入失败，清理文件
                self.storage.delete_dataset(dataset_id, metadata)
                raise RuntimeError("数据库插入失败")
            
            # 更新缓存
            with self.cache_lock:
                self.metadata_cache[dataset_id] = metadata
            
            self.logger.info(f"数据集创建成功: {dataset_id}")
            return dataset_id
            
        except Exception as e:
            self.logger.error(f"创建数据集失败: {e}")
            raise
    
    def load_dataset(self, dataset_id: str) -> Tuple[np.ndarray, DatasetMetadata]:
        """加载数据集"""
        # 从缓存或数据库获取元数据
        metadata = self.get_metadata(dataset_id)
        if not metadata:
            raise ValueError(f"数据集不存在: {dataset_id}")
        
        # 加载数据
        data = self.storage.load_dataset(dataset_id, metadata)
        
        # 更新使用统计
        self.database.update_usage_stats(dataset_id)
        
        return data, metadata
    
    def get_metadata(self, dataset_id: str) -> Optional[DatasetMetadata]:
        """获取数据集元数据"""
        # 先从缓存查找
        with self.cache_lock:
            if dataset_id in self.metadata_cache:
                return self.metadata_cache[dataset_id]
        
        # 从数据库查找
        metadata = self.database.get_dataset(dataset_id)
        if metadata:
            with self.cache_lock:
                self.metadata_cache[dataset_id] = metadata
        
        return metadata
    
    def search_datasets(self, query: DatasetQuery) -> List[DatasetMetadata]:
        """搜索数据集"""
        return self.database.query_datasets(query)
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """删除数据集"""
        try:
            # 获取元数据
            metadata = self.get_metadata(dataset_id)
            if not metadata:
                return False
            
            # 删除文件
            storage_success = self.storage.delete_dataset(dataset_id, metadata)
            
            # 删除数据库记录
            db_success = self.database.delete_dataset(dataset_id)
            
            # 清理缓存
            with self.cache_lock:
                self.metadata_cache.pop(dataset_id, None)
            
            success = storage_success and db_success
            if success:
                self.logger.info(f"数据集删除成功: {dataset_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"删除数据集失败 {dataset_id}: {e}")
            return False
    
    def archive_dataset(self, dataset_id: str) -> Optional[str]:
        """归档数据集"""
        try:
            metadata = self.get_metadata(dataset_id)
            if not metadata:
                return None
            
            archive_path = self.storage.archive_dataset(dataset_id, metadata)
            self.logger.info(f"数据集归档成功: {dataset_id} -> {archive_path}")
            
            return archive_path
            
        except Exception as e:
            self.logger.error(f"归档数据集失败 {dataset_id}: {e}")
            return None
    
    def update_dataset_tags(self, dataset_id: str, tags: List[str]) -> bool:
        """更新数据集标签"""
        try:
            metadata = self.get_metadata(dataset_id)
            if not metadata:
                return False
            
            metadata.tags = tags
            metadata.updated_at = datetime.utcnow()
            
            success = self.database.insert_dataset(metadata)
            if success:
                # 更新缓存
                with self.cache_lock:
                    self.metadata_cache[dataset_id] = metadata
            
            return success
            
        except Exception as e:
            self.logger.error(f"更新数据集标签失败 {dataset_id}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取管理器统计信息"""
        db_stats = self.database.get_statistics()
        
        # 添加缓存统计
        with self.cache_lock:
            cache_stats = {
                "cached_datasets": len(self.metadata_cache)
            }
        
        return {**db_stats, **cache_stats}
    
    def cleanup_expired_datasets(self, days_threshold: int = 30) -> int:
        """清理过期数据集"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
            
            # 查询长时间未使用的数据集
            query = DatasetQuery(
                dataset_type=DatasetType.EXPERIMENTAL,  # 只清理实验数据
                created_before=cutoff_date,
                limit=1000
            )
            
            expired_datasets = self.search_datasets(query)
            
            cleaned_count = 0
            for metadata in expired_datasets:
                if (metadata.last_accessed is None or 
                    metadata.last_accessed < cutoff_date):
                    
                    if self.delete_dataset(metadata.dataset_id):
                        cleaned_count += 1
            
            self.logger.info(f"清理了 {cleaned_count} 个过期数据集")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"清理过期数据集失败: {e}")
            return 0
    
    def export_dataset_catalog(self, output_path: str) -> bool:
        """导出数据集目录"""
        try:
            # 获取所有数据集
            query = DatasetQuery(limit=10000)
            all_datasets = self.search_datasets(query)
            
            # 准备导出数据
            catalog_data = []
            for metadata in all_datasets:
                catalog_data.append({
                    "dataset_id": metadata.dataset_id,
                    "name": metadata.name,
                    "description": metadata.description,
                    "type": metadata.dataset_type.value,
                    "format": metadata.data_format.value,
                    "samples": metadata.num_samples,
                    "features": metadata.num_features,
                    "sequence_length": metadata.sequence_length,
                    "quality_score": metadata.quality_score,
                    "quality_grade": metadata.quality_grade.value,
                    "generator_model": metadata.generator_model,
                    "created_at": metadata.created_at.isoformat(),
                    "file_size_mb": metadata.file_size / (1024 * 1024),
                    "tags": metadata.tags,
                    "market_regimes": metadata.market_regimes
                })
            
            # 导出为JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(catalog_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"数据集目录已导出: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出数据集目录失败: {e}")
            return False
    
    def _generate_dataset_id(self, name: str, model_id: str) -> str:
        """生成数据集ID"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        content = f"{name}_{model_id}_{timestamp}"
        hash_value = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"dataset_{hash_value}_{timestamp}"
    
    def _calculate_quality_grade(self, quality_score: float) -> DataQuality:
        """计算质量等级"""
        if quality_score > 0.9:
            return DataQuality.EXCELLENT
        elif quality_score > 0.7:
            return DataQuality.GOOD
        elif quality_score > 0.5:
            return DataQuality.FAIR
        else:
            return DataQuality.POOR

# 全局数据管理器实例
default_data_manager = SyntheticDataManager()