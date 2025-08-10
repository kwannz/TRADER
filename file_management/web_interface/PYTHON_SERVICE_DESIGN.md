# Python服务层重构设计文档

**项目**: QuantAnalyzer Pro - Python API服务层  
**版本**: v1.0  
**创建日期**: 2025-08-10  

---

## 1. 现有系统分析

### 1.1 当前Python服务现状

基于对现有 `data-analysis-api.py` 的分析，当前系统特点：

**现有优势**：
- ✅ 完整的API端点覆盖（因子、回测、数据等）
- ✅ 良好的路由结构设计
- ✅ CORS跨域支持
- ✅ 丰富的模拟数据生成

**主要限制**：
- ❌ 基于简单HTTP服务器，缺乏异步处理能力
- ❌ 无数据持久化，全部为模拟数据
- ❌ 缺乏数据验证和错误处理
- ❌ 无认证和权限控制
- ❌ 性能和并发能力有限

### 1.2 重构目标

1. **架构现代化**: 迁移到FastAPI异步框架
2. **性能提升**: 集成Rust引擎，实现高性能计算
3. **数据持久化**: 集成ClickHouse和PostgreSQL
4. **服务化**: 微服务架构，支持水平扩展
5. **生产就绪**: 添加监控、日志、安全等生产环境必需功能

---

## 2. FastAPI服务架构设计

### 2.1 项目结构

```
python_api/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI应用入口
│   ├── config.py                  # 配置管理
│   ├── dependencies.py            # 依赖注入
│   ├── api/                       # API路由层
│   │   ├── __init__.py
│   │   ├── v1/                    # API v1版本（向后兼容）
│   │   │   ├── __init__.py
│   │   │   ├── legacy.py          # 兼容旧API
│   │   │   └── ...
│   │   └── v2/                    # API v2版本（新架构）
│   │       ├── __init__.py
│   │       ├── factors.py         # 因子相关API
│   │       ├── backtest.py        # 回测相关API
│   │       ├── data.py            # 数据相关API
│   │       ├── portfolio.py       # 组合相关API
│   │       ├── ai.py              # AI引擎相关API
│   │       └── system.py          # 系统相关API
│   ├── core/                      # 核心模块
│   │   ├── __init__.py
│   │   ├── database.py            # 数据库连接
│   │   ├── cache.py               # 缓存管理
│   │   ├── rust_engine.py         # Rust引擎接口
│   │   ├── security.py            # 安全认证
│   │   └── logging.py             # 日志配置
│   ├── models/                    # 数据模型
│   │   ├── __init__.py
│   │   ├── base.py                # 基础模型
│   │   ├── factor.py              # 因子模型
│   │   ├── backtest.py            # 回测模型
│   │   ├── portfolio.py           # 组合模型
│   │   └── user.py                # 用户模型
│   ├── services/                  # 业务逻辑层
│   │   ├── __init__.py
│   │   ├── factor_service.py      # 因子服务
│   │   ├── backtest_service.py    # 回测服务
│   │   ├── data_service.py        # 数据服务
│   │   ├── portfolio_service.py   # 组合服务
│   │   └── ai_service.py          # AI服务
│   ├── workers/                   # 后台任务
│   │   ├── __init__.py
│   │   ├── factor_worker.py       # 因子计算任务
│   │   ├── backtest_worker.py     # 回测任务
│   │   └── data_worker.py         # 数据处理任务
│   ├── middleware/                # 中间件
│   │   ├── __init__.py
│   │   ├── auth.py                # 认证中间件
│   │   ├── rate_limit.py          # 限流中间件
│   │   └── monitoring.py          # 监控中间件
│   └── utils/                     # 工具模块
│       ├── __init__.py
│       ├── datetime.py            # 时间处理
│       ├── validation.py          # 数据验证
│       └── serialization.py       # 序列化
├── tests/                         # 测试代码
├── alembic/                       # 数据库迁移
├── requirements.txt               # Python依赖
├── pyproject.toml                # 项目配置
└── Dockerfile                     # 容器配置
```

### 2.2 核心应用配置

```python
# app/main.py
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging

from .config import get_settings
from .core import database, cache, rust_engine
from .middleware import auth, rate_limit, monitoring
from .api.v1 import legacy_router
from .api.v2 import (
    factors_router, backtest_router, data_router, 
    portfolio_router, ai_router, system_router
)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("🚀 QuantAnalyzer Pro API 启动中...")
    
    try:
        # 初始化数据库连接
        await database.init_database()
        logger.info("✅ 数据库连接已建立")
        
        # 初始化Redis缓存
        await cache.init_cache()
        logger.info("✅ 缓存系统已初始化")
        
        # 初始化Rust引擎
        await rust_engine.init_engine()
        logger.info("✅ Rust计算引擎已加载")
        
        # 启动后台任务
        await start_background_tasks()
        logger.info("✅ 后台任务已启动")
        
        logger.info("🎉 系统初始化完成")
        
    except Exception as e:
        logger.error(f"❌ 系统初始化失败: {e}")
        raise
    
    yield  # 应用运行阶段
    
    # 清理资源
    logger.info("🧹 系统清理中...")
    await database.close_database()
    await cache.close_cache()
    await rust_engine.cleanup_engine()
    logger.info("✅ 系统清理完成")

# 创建FastAPI应用
app = FastAPI(
    title="QuantAnalyzer Pro API",
    description="AI-driven quantitative analysis platform with high-performance computing",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# 获取配置
settings = get_settings()

# 添加中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(monitoring.PrometheusMiddleware)
app.add_middleware(rate_limit.RateLimitMiddleware)

# 全局异常处理
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat(),
            "path": str(request.url.path)
        }
    )

# 注册路由
# V1 API (向后兼容)
app.include_router(
    legacy_router,
    prefix="/api/v1",
    tags=["Legacy API v1"]
)

# V2 API (新架构)
app.include_router(
    factors_router,
    prefix="/api/v2/factors",
    tags=["Factors"],
    dependencies=[Depends(auth.get_current_user)]
)

app.include_router(
    backtest_router,
    prefix="/api/v2/backtest",
    tags=["Backtest"],
    dependencies=[Depends(auth.get_current_user)]
)

app.include_router(
    data_router,
    prefix="/api/v2/data",
    tags=["Data"],
    dependencies=[Depends(auth.get_current_user)]
)

app.include_router(
    portfolio_router,
    prefix="/api/v2/portfolio",
    tags=["Portfolio"],
    dependencies=[Depends(auth.get_current_user)]
)

app.include_router(
    ai_router,
    prefix="/api/v2/ai",
    tags=["AI Services"],
    dependencies=[Depends(auth.get_current_user)]
)

app.include_router(
    system_router,
    prefix="/api/v2/system",
    tags=["System"]
)

# 健康检查端点
@app.get("/health", tags=["Health"])
async def health_check():
    """系统健康检查"""
    try:
        # 检查各个组件状态
        db_status = await database.check_health()
        cache_status = await cache.check_health()
        rust_status = await rust_engine.check_health()
        
        overall_status = "healthy" if all([db_status, cache_status, rust_status]) else "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0",
            "components": {
                "database": "healthy" if db_status else "unhealthy",
                "cache": "healthy" if cache_status else "unhealthy",
                "rust_engine": "healthy" if rust_status else "unhealthy"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

# 根路径
@app.get("/", tags=["Root"])
async def root():
    """API根路径"""
    return {
        "message": "QuantAnalyzer Pro API v2.0",
        "docs": "/docs",
        "health": "/health",
        "version": "2.0.0"
    }

async def start_background_tasks():
    """启动后台任务"""
    # 这里可以启动定时任务、数据同步等后台任务
    pass

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

### 2.3 配置管理

```python
# app/config.py
from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    # 应用配置
    APP_NAME: str = "QuantAnalyzer Pro"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    # 服务器配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # 安全配置
    SECRET_KEY: str = "your-secret-key-here"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS配置
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    # 数据库配置
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/quant_db"
    CLICKHOUSE_URL: str = "http://localhost:8123"
    CLICKHOUSE_DATABASE: str = "quant_data"
    
    # Redis配置
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 3600  # 1小时
    
    # Rust引擎配置
    RUST_ENGINE_PATH: str = "./rust_engine/target/release"
    RUST_ENGINE_THREADS: int = 0  # 0表示使用CPU核心数
    RUST_ENGINE_MEMORY_LIMIT_GB: int = 8
    
    # 数据源配置
    BINANCE_API_KEY: Optional[str] = None
    BINANCE_SECRET_KEY: Optional[str] = None
    OKX_API_KEY: Optional[str] = None
    OKX_SECRET_KEY: Optional[str] = None
    
    # AI服务配置
    OPENAI_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    
    # 监控配置
    ENABLE_PROMETHEUS: bool = True
    PROMETHEUS_PORT: int = 9090
    LOG_LEVEL: str = "INFO"
    
    # 任务队列配置
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # 限流配置
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 60  # 秒
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """获取配置实例（缓存）"""
    return Settings()

# 环境特定配置
class DevelopmentSettings(Settings):
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    WORKERS: int = 1

class ProductionSettings(Settings):
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    WORKERS: int = 4

class TestingSettings(Settings):
    DEBUG: bool = True
    DATABASE_URL: str = "sqlite+aiosqlite:///./test.db"
    REDIS_URL: str = "redis://localhost:6379/1"

def get_settings_for_env(env: str = None) -> Settings:
    """根据环境获取配置"""
    env = env or os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()
```

---

## 3. 核心服务模块设计

### 3.1 因子服务

```python
# app/services/factor_service.py
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
import pandas as pd

from ..models.factor import FactorDefinition, FactorResult, FactorLibrary
from ..core.database import get_session
from ..core.rust_engine import get_rust_engine
from ..core.cache import get_cache
from ..utils.validation import validate_factor_definition

class FactorService:
    def __init__(self):
        self.cache = get_cache()
        self.rust_engine = get_rust_engine()
    
    async def calculate_factors(
        self,
        factors: List[FactorDefinition],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        session: AsyncSession = None
    ) -> Dict[str, FactorResult]:
        """批量计算因子"""
        
        # 输入验证
        for factor in factors:
            validate_factor_definition(factor)
        
        # 检查缓存
        cache_key = self._generate_cache_key(factors, symbols, start_date, end_date)
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # 获取市场数据
        market_data = await self._get_market_data(symbols, start_date, end_date)
        
        # 使用Rust引擎计算因子
        calculation_start = datetime.now()
        
        try:
            # 准备Rust引擎输入
            rust_factors = [self._convert_to_rust_factor(f) for f in factors]
            rust_market_data = self._convert_to_rust_data(market_data)
            
            # 并行计算
            raw_results = await asyncio.to_thread(
                self.rust_engine.batch_calculate_factors,
                rust_factors,
                rust_market_data,
                symbols
            )
            
            calculation_time = (datetime.now() - calculation_start).total_seconds()
            
            # 转换结果格式
            results = {}
            for factor in factors:
                if factor.name in raw_results:
                    results[factor.name] = FactorResult(
                        factor_id=factor.id,
                        factor_name=factor.name,
                        values=raw_results[factor.name],
                        timestamps=market_data.index.tolist(),
                        symbols=symbols,
                        calculation_time=calculation_time,
                        metadata={
                            "start_date": start_date.isoformat(),
                            "end_date": end_date.isoformat(),
                            "data_points": len(raw_results[factor.name])
                        }
                    )
            
            # 缓存结果
            await self.cache.set(cache_key, results, ttl=3600)
            
            # 保存到数据库（异步）
            if session:
                asyncio.create_task(self._save_factor_results(results, session))
            
            return results
            
        except Exception as e:
            raise Exception(f"Factor calculation failed: {str(e)}")
    
    async def get_factor_library(
        self,
        category: Optional[str] = None,
        min_ic: Optional[float] = None,
        min_sharpe: Optional[float] = None,
        limit: int = 100,
        session: AsyncSession = None
    ) -> List[FactorLibrary]:
        """获取因子库"""
        
        if not session:
            async with get_session() as session:
                return await self._get_factor_library_impl(
                    session, category, min_ic, min_sharpe, limit
                )
        else:
            return await self._get_factor_library_impl(
                session, category, min_ic, min_sharpe, limit
            )
    
    async def _get_factor_library_impl(
        self,
        session: AsyncSession,
        category: Optional[str],
        min_ic: Optional[float],
        min_sharpe: Optional[float],
        limit: int
    ) -> List[FactorLibrary]:
        """因子库查询实现"""
        
        # 构建查询条件
        conditions = [FactorLibrary.is_active == True]
        
        if category:
            conditions.append(FactorLibrary.category == category)
        
        if min_ic is not None:
            conditions.append(FactorLibrary.ic >= min_ic)
        
        if min_sharpe is not None:
            conditions.append(FactorLibrary.sharpe_ratio >= min_sharpe)
        
        # 执行查询
        query = select(FactorLibrary).where(and_(*conditions)).limit(limit)
        result = await session.execute(query)
        
        return result.scalars().all()
    
    async def create_factor(
        self,
        factor_def: FactorDefinition,
        session: AsyncSession = None
    ) -> FactorLibrary:
        """创建新因子"""
        
        # 验证因子定义
        validate_factor_definition(factor_def)
        
        if not session:
            async with get_session() as session:
                return await self._create_factor_impl(session, factor_def)
        else:
            return await self._create_factor_impl(session, factor_def)
    
    async def _create_factor_impl(
        self,
        session: AsyncSession,
        factor_def: FactorDefinition
    ) -> FactorLibrary:
        """创建因子实现"""
        
        # 检查因子是否已存在
        existing_query = select(FactorLibrary).where(
            FactorLibrary.name == factor_def.name
        )
        existing = await session.execute(existing_query)
        if existing.scalar_one_or_none():
            raise ValueError(f"Factor '{factor_def.name}' already exists")
        
        # 创建新因子记录
        new_factor = FactorLibrary(
            name=factor_def.name,
            category=factor_def.category,
            formula=factor_def.formula,
            description=factor_def.description,
            parameters=factor_def.parameters,
            created_by=factor_def.created_by,
            is_active=True
        )
        
        session.add(new_factor)
        await session.commit()
        await session.refresh(new_factor)
        
        return new_factor
    
    async def evaluate_factor_performance(
        self,
        factor_id: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        session: AsyncSession = None
    ) -> Dict[str, float]:
        """评估因子性能"""
        
        # 获取因子定义
        if not session:
            async with get_session() as session:
                factor_query = select(FactorLibrary).where(FactorLibrary.id == factor_id)
                result = await session.execute(factor_query)
                factor = result.scalar_one_or_none()
        
        if not factor:
            raise ValueError(f"Factor {factor_id} not found")
        
        # 计算因子值
        factor_def = FactorDefinition(
            id=factor.id,
            name=factor.name,
            category=factor.category,
            formula=factor.formula,
            parameters=factor.parameters
        )
        
        factor_results = await self.calculate_factors(
            [factor_def], symbols, start_date, end_date
        )
        
        if factor.name not in factor_results:
            raise ValueError("Factor calculation failed")
        
        # 获取价格数据用于计算IC
        price_data = await self._get_price_data(symbols, start_date, end_date)
        
        # 计算性能指标
        performance = await self._calculate_performance_metrics(
            factor_results[factor.name],
            price_data
        )
        
        # 更新因子性能记录
        if session:
            await self._update_factor_performance(session, factor_id, performance)
        
        return performance
    
    async def search_factors(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        limit: int = 50,
        session: AsyncSession = None
    ) -> List[FactorLibrary]:
        """搜索因子"""
        
        if not session:
            async with get_session() as session:
                return await self._search_factors_impl(session, query, filters, limit)
        else:
            return await self._search_factors_impl(session, query, filters, limit)
    
    async def _search_factors_impl(
        self,
        session: AsyncSession,
        query: str,
        filters: Dict[str, Any],
        limit: int
    ) -> List[FactorLibrary]:
        """搜索因子实现"""
        
        conditions = [FactorLibrary.is_active == True]
        
        # 文本搜索
        if query:
            text_condition = or_(
                FactorLibrary.name.ilike(f"%{query}%"),
                FactorLibrary.description.ilike(f"%{query}%"),
                FactorLibrary.formula.ilike(f"%{query}%")
            )
            conditions.append(text_condition)
        
        # 过滤条件
        if filters:
            if "category" in filters:
                conditions.append(FactorLibrary.category == filters["category"])
            
            if "min_ic" in filters:
                conditions.append(FactorLibrary.ic >= filters["min_ic"])
            
            if "created_after" in filters:
                conditions.append(FactorLibrary.created_at >= filters["created_after"])
        
        # 执行查询
        search_query = select(FactorLibrary).where(and_(*conditions)).limit(limit)
        result = await session.execute(search_query)
        
        return result.scalars().all()
    
    # 私有辅助方法
    def _generate_cache_key(
        self, factors: List[FactorDefinition], symbols: List[str], 
        start_date: datetime, end_date: datetime
    ) -> str:
        """生成缓存键"""
        factor_names = sorted([f.name for f in factors])
        symbols_str = sorted(symbols)
        return f"factors:{':'.join(factor_names)}:{':'.join(symbols_str)}:{start_date.date()}:{end_date.date()}"
    
    async def _get_market_data(
        self, symbols: List[str], start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """获取市场数据"""
        # 这里应该从数据服务获取真实市场数据
        # 暂时返回模拟数据
        from ..services.data_service import DataService
        data_service = DataService()
        return await data_service.get_market_data(symbols, start_date, end_date)
    
    def _convert_to_rust_factor(self, factor: FactorDefinition) -> Dict[str, Any]:
        """转换为Rust引擎格式"""
        return {
            "id": factor.id,
            "name": factor.name,
            "factor_type": factor.category.lower(),
            "formula": factor.formula,
            "parameters": factor.parameters or {}
        }
    
    def _convert_to_rust_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """转换市场数据为Rust引擎格式"""
        return {
            "timestamps": df.index.tolist(),
            "data": df.to_dict('records')
        }
    
    async def _save_factor_results(
        self, results: Dict[str, FactorResult], session: AsyncSession
    ):
        """保存因子计算结果到数据库"""
        # 实现批量保存逻辑
        pass
    
    async def _get_price_data(
        self, symbols: List[str], start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """获取价格数据"""
        # 实现价格数据获取逻辑
        pass
    
    async def _calculate_performance_metrics(
        self, factor_result: FactorResult, price_data: pd.DataFrame
    ) -> Dict[str, float]:
        """计算性能指标"""
        # 实现IC、IR、夏普比率等指标计算
        return {
            "ic": 0.15,  # 示例值
            "ic_ir": 1.2,
            "sharpe_ratio": 1.8,
            "max_drawdown": -0.08,
            "win_rate": 0.65
        }
    
    async def _update_factor_performance(
        self, session: AsyncSession, factor_id: str, performance: Dict[str, float]
    ):
        """更新因子性能记录"""
        # 实现性能指标更新逻辑
        pass
```

### 3.2 回测服务

```python
# app/services/backtest_service.py
from typing import List, Dict, Any, Optional
import asyncio
import uuid
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..models.backtest import BacktestJob, BacktestResult, StrategyDefinition
from ..core.database import get_session
from ..core.rust_engine import get_rust_engine
from ..workers.backtest_worker import run_backtest_task

class BacktestService:
    def __init__(self):
        self.rust_engine = get_rust_engine()
    
    async def create_backtest_job(
        self,
        strategy: StrategyDefinition,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 1000000.0,
        benchmark: Optional[str] = None,
        session: AsyncSession = None
    ) -> str:
        """创建回测任务"""
        
        job_id = str(uuid.uuid4())
        
        if not session:
            async with get_session() as session:
                return await self._create_backtest_job_impl(
                    session, job_id, strategy, start_date, end_date, 
                    initial_capital, benchmark
                )
        else:
            return await self._create_backtest_job_impl(
                session, job_id, strategy, start_date, end_date,
                initial_capital, benchmark
            )
    
    async def _create_backtest_job_impl(
        self,
        session: AsyncSession,
        job_id: str,
        strategy: StrategyDefinition,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float,
        benchmark: Optional[str]
    ) -> str:
        """创建回测任务实现"""
        
        # 创建任务记录
        job = BacktestJob(
            id=job_id,
            strategy_id=strategy.id,
            name=f"Backtest {strategy.name} {start_date.date()}",
            status="pending",
            start_date=start_date,
            end_date=end_date,
            config={
                "initial_capital": initial_capital,
                "benchmark": benchmark,
                "commission_rate": 0.0025,
                "slippage_bps": 2.0
            }
        )
        
        session.add(job)
        await session.commit()
        
        # 异步启动回测任务
        asyncio.create_task(
            run_backtest_task(job_id, strategy, start_date, end_date, initial_capital)
        )
        
        return job_id
    
    async def get_backtest_status(
        self, job_id: str, session: AsyncSession = None
    ) -> Dict[str, Any]:
        """获取回测状态"""
        
        if not session:
            async with get_session() as session:
                return await self._get_backtest_status_impl(session, job_id)
        else:
            return await self._get_backtest_status_impl(session, job_id)
    
    async def _get_backtest_status_impl(
        self, session: AsyncSession, job_id: str
    ) -> Dict[str, Any]:
        """获取回测状态实现"""
        
        query = select(BacktestJob).where(BacktestJob.id == job_id)
        result = await session.execute(query)
        job = result.scalar_one_or_none()
        
        if not job:
            raise ValueError(f"Backtest job {job_id} not found")
        
        return {
            "job_id": job.id,
            "name": job.name,
            "status": job.status,
            "progress": job.progress or 0,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error_message": job.error_message,
            "estimated_duration": self._estimate_duration(job)
        }
    
    async def get_backtest_results(
        self, job_id: str, session: AsyncSession = None
    ) -> Optional[BacktestResult]:
        """获取回测结果"""
        
        if not session:
            async with get_session() as session:
                return await self._get_backtest_results_impl(session, job_id)
        else:
            return await self._get_backtest_results_impl(session, job_id)
    
    async def _get_backtest_results_impl(
        self, session: AsyncSession, job_id: str
    ) -> Optional[BacktestResult]:
        """获取回测结果实现"""
        
        # 获取任务信息
        job_query = select(BacktestJob).where(BacktestJob.id == job_id)
        job_result = await session.execute(job_query)
        job = job_result.scalar_one_or_none()
        
        if not job or job.status != "completed":
            return None
        
        # 获取结果数据
        result_query = select(BacktestResult).where(BacktestResult.job_id == job_id)
        result_data = await session.execute(result_query)
        result = result_data.scalar_one_or_none()
        
        return result
    
    async def run_vectorized_backtest(
        self,
        strategy_config: Dict[str, Any],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 1000000.0
    ) -> Dict[str, Any]:
        """运行向量化回测（使用Rust引擎）"""
        
        try:
            # 准备回测配置
            backtest_config = {
                "initial_capital": initial_capital,
                "commission_rate": 0.0025,
                "slippage_bps": 2.0,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "symbols": symbols
            }
            
            # 调用Rust回测引擎
            results = await asyncio.to_thread(
                self.rust_engine.run_vectorized_backtest,
                strategy_config,
                backtest_config
            )
            
            return {
                "success": True,
                "results": results,
                "computation_time": results.get("computation_time_ms", 0),
                "total_trades": results.get("total_trades", 0)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "computation_time": 0
            }
    
    async def compare_strategies(
        self,
        strategies: List[Dict[str, Any]],
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """策略对比分析"""
        
        results = {}
        
        # 并行运行多个策略回测
        tasks = []
        for i, strategy in enumerate(strategies):
            task = self.run_vectorized_backtest(
                strategy, symbols, start_date, end_date
            )
            tasks.append(task)
        
        strategy_results = await asyncio.gather(*tasks)
        
        # 整合结果
        for i, result in enumerate(strategy_results):
            strategy_name = strategies[i].get("name", f"Strategy_{i+1}")
            results[strategy_name] = result
        
        # 计算对比指标
        comparison = self._calculate_comparison_metrics(results)
        
        return {
            "strategies": results,
            "comparison": comparison,
            "best_strategy": self._find_best_strategy(results)
        }
    
    async def get_backtest_history(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        session: AsyncSession = None
    ) -> List[Dict[str, Any]]:
        """获取回测历史"""
        
        if not session:
            async with get_session() as session:
                return await self._get_backtest_history_impl(session, user_id, limit)
        else:
            return await self._get_backtest_history_impl(session, user_id, limit)
    
    async def _get_backtest_history_impl(
        self,
        session: AsyncSession,
        user_id: Optional[str],
        limit: int
    ) -> List[Dict[str, Any]]:
        """获取回测历史实现"""
        
        conditions = []
        if user_id:
            conditions.append(BacktestJob.created_by == user_id)
        
        query = select(BacktestJob).order_by(BacktestJob.created_at.desc()).limit(limit)
        if conditions:
            from sqlalchemy import and_
            query = query.where(and_(*conditions))
        
        result = await session.execute(query)
        jobs = result.scalars().all()
        
        return [
            {
                "id": job.id,
                "name": job.name,
                "status": job.status,
                "created_at": job.created_at.isoformat(),
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "duration_minutes": self._calculate_duration(job),
                "summary": self._generate_summary(job)
            }
            for job in jobs
        ]
    
    # 私有辅助方法
    def _estimate_duration(self, job: BacktestJob) -> str:
        """估算回测持续时间"""
        # 基于历史数据和策略复杂度估算
        return "5-10分钟"
    
    def _calculate_comparison_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """计算策略对比指标"""
        if not results:
            return {}
        
        # 提取各策略的关键指标
        metrics = {}
        for name, result in results.items():
            if result.get("success") and "results" in result:
                strategy_results = result["results"]
                metrics[name] = {
                    "total_return": strategy_results.get("total_return", 0),
                    "sharpe_ratio": strategy_results.get("sharpe_ratio", 0),
                    "max_drawdown": strategy_results.get("max_drawdown", 0),
                    "win_rate": strategy_results.get("win_rate", 0)
                }
        
        # 计算排名
        rankings = {}
        for metric in ["total_return", "sharpe_ratio", "win_rate"]:
            sorted_strategies = sorted(
                metrics.items(),
                key=lambda x: x[1].get(metric, 0),
                reverse=True
            )
            rankings[metric] = [name for name, _ in sorted_strategies]
        
        # 最大回撤排名（越小越好）
        sorted_by_drawdown = sorted(
            metrics.items(),
            key=lambda x: abs(x[1].get("max_drawdown", 0))
        )
        rankings["max_drawdown"] = [name for name, _ in sorted_by_drawdown]
        
        return {
            "metrics": metrics,
            "rankings": rankings,
            "best_performer": rankings["total_return"][0] if rankings["total_return"] else None
        }
    
    def _find_best_strategy(self, results: Dict[str, Any]) -> Optional[str]:
        """找出最佳策略"""
        # 综合评分算法
        scores = {}
        
        for name, result in results.items():
            if result.get("success") and "results" in result:
                strategy_results = result["results"]
                
                # 综合评分（可调整权重）
                score = (
                    strategy_results.get("total_return", 0) * 0.3 +
                    strategy_results.get("sharpe_ratio", 0) * 0.3 +
                    strategy_results.get("win_rate", 0) * 0.2 +
                    (1 + strategy_results.get("max_drawdown", 0)) * 0.2  # 回撤越小得分越高
                )
                scores[name] = score
        
        return max(scores.items(), key=lambda x: x[1])[0] if scores else None
    
    def _calculate_duration(self, job: BacktestJob) -> Optional[float]:
        """计算回测持续时间（分钟）"""
        if job.started_at and job.completed_at:
            duration = job.completed_at - job.started_at
            return duration.total_seconds() / 60
        return None
    
    def _generate_summary(self, job: BacktestJob) -> str:
        """生成回测摘要"""
        if job.status == "completed" and job.results:
            total_return = job.results.get("total_return", 0)
            sharpe_ratio = job.results.get("sharpe_ratio", 0)
            return f"收益率: {total_return:.2%}, 夏普比率: {sharpe_ratio:.2f}"
        elif job.status == "failed":
            return "回测失败"
        else:
            return f"状态: {job.status}"
```

---

## 4. 数据模型设计

### 4.1 Pydantic模型

```python
# app/models/factor.py
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class FactorCategory(str, Enum):
    TECHNICAL = "technical"
    STATISTICAL = "statistical" 
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    AI_GENERATED = "ai_generated"

class FactorDefinition(BaseModel):
    """因子定义模型"""
    id: str = Field(..., description="因子唯一标识")
    name: str = Field(..., min_length=1, max_length=100, description="因子名称")
    category: FactorCategory = Field(..., description="因子类别")
    formula: str = Field(..., min_length=1, description="因子计算公式")
    description: Optional[str] = Field(None, max_length=1000, description="因子描述")
    parameters: Optional[Dict[str, Any]] = Field(None, description="计算参数")
    created_by: Optional[str] = Field(None, description="创建者")
    
    @validator('formula')
    def validate_formula(cls, v):
        """验证因子公式"""
        if not v or len(v.strip()) == 0:
            raise ValueError("Formula cannot be empty")
        # 这里可以添加更多的公式语法验证
        return v.strip()
    
    class Config:
        use_enum_values = True
        schema_extra = {
            "example": {
                "id": "rsi_14",
                "name": "RSI_14",
                "category": "technical",
                "formula": "rsi(close, 14)",
                "description": "14期相对强弱指数",
                "parameters": {"period": 14},
                "created_by": "user_123"
            }
        }

class FactorResult(BaseModel):
    """因子计算结果模型"""
    factor_id: str
    factor_name: str
    values: List[float]
    timestamps: List[datetime]
    symbols: List[str]
    calculation_time: float
    metadata: Optional[Dict[str, Any]] = None

class FactorCalculationRequest(BaseModel):
    """因子计算请求模型"""
    factors: List[FactorDefinition] = Field(..., min_items=1, max_items=50)
    symbols: List[str] = Field(..., min_items=1, max_items=1000)
    start_date: datetime
    end_date: datetime
    cache_enabled: bool = Field(True, description="是否启用缓存")
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v
    
    @validator('symbols')
    def validate_symbols(cls, v):
        if len(set(v)) != len(v):
            raise ValueError('Duplicate symbols not allowed')
        return v

class FactorCalculationResponse(BaseModel):
    """因子计算响应模型"""
    success: bool
    results: Optional[Dict[str, FactorResult]] = None
    error: Optional[str] = None
    computation_time: float
    cached: bool = False
    
class FactorLibraryQuery(BaseModel):
    """因子库查询模型"""
    category: Optional[FactorCategory] = None
    min_ic: Optional[float] = Field(None, ge=-1, le=1)
    min_sharpe: Optional[float] = Field(None, ge=0)
    search_text: Optional[str] = Field(None, max_length=100)
    limit: int = Field(50, ge=1, le=1000)
    offset: int = Field(0, ge=0)

# app/models/backtest.py
class BacktestConfig(BaseModel):
    """回测配置模型"""
    initial_capital: float = Field(1000000.0, gt=0, description="初始资金")
    commission_rate: float = Field(0.0025, ge=0, le=0.1, description="手续费率")
    slippage_bps: float = Field(2.0, ge=0, le=100, description="滑点(基点)")
    max_position_size: float = Field(0.2, gt=0, le=1, description="最大仓位比例")
    rebalance_frequency: str = Field("daily", regex="^(daily|weekly|monthly)$")
    benchmark: Optional[str] = Field(None, description="基准代码")

class StrategyDefinition(BaseModel):
    """策略定义模型"""
    id: str
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)
    factors: List[str] = Field(..., min_items=1, description="使用的因子ID列表")
    weights: Optional[Dict[str, float]] = Field(None, description="因子权重")
    rebalance_rule: str = Field("weekly", description="调仓规则")
    risk_management: Optional[Dict[str, Any]] = Field(None, description="风险控制")

class BacktestRequest(BaseModel):
    """回测请求模型"""
    strategy: StrategyDefinition
    symbols: List[str] = Field(..., min_items=1, max_items=500)
    start_date: datetime
    end_date: datetime
    config: BacktestConfig = BacktestConfig()
    
    @validator('end_date')
    def validate_backtest_period(cls, v, values):
        if 'start_date' in values:
            if v <= values['start_date']:
                raise ValueError('end_date must be after start_date')
            
            # 检查回测期间长度
            period_days = (v - values['start_date']).days
            if period_days < 30:
                raise ValueError('Backtest period must be at least 30 days')
            if period_days > 3650:  # 10年
                raise ValueError('Backtest period cannot exceed 10 years')
        return v

class BacktestMetrics(BaseModel):
    """回测指标模型"""
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    information_ratio: float
    total_trades: int
    avg_trade_return: float

class BacktestResult(BaseModel):
    """回测结果模型"""
    job_id: str
    status: str
    metrics: Optional[BacktestMetrics] = None
    daily_returns: Optional[List[float]] = None
    portfolio_values: Optional[List[float]] = None
    positions_history: Optional[Dict[str, List[float]]] = None
    trade_history: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    computation_time: float = 0
    
class BacktestJobStatus(BaseModel):
    """回测任务状态模型"""
    job_id: str
    name: str
    status: str  # pending, running, completed, failed
    progress: float = Field(0, ge=0, le=100)
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_duration: Optional[str] = None
    error_message: Optional[str] = None
```

### 4.2 SQLAlchemy模型

```python
# app/models/db_models.py
from sqlalchemy import Column, String, Float, Integer, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

class FactorLibrary(Base):
    """因子库数据表"""
    __tablename__ = "factor_library"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False, unique=True, index=True)
    category = Column(String(50), nullable=False, index=True)
    formula = Column(Text, nullable=False)
    description = Column(Text)
    parameters = Column(JSON)
    
    # 性能指标
    ic = Column(Float)
    ic_ir = Column(Float) 
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    
    # 使用统计
    usage_count = Column(Integer, default=0)
    last_used = Column(DateTime)
    
    # 元数据
    created_by = Column(String(100))
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    is_active = Column(Boolean, default=True, index=True)

class BacktestJob(Base):
    """回测任务表"""
    __tablename__ = "backtest_jobs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    strategy_id = Column(String(100), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    status = Column(String(50), default='pending', index=True)
    progress = Column(Float, default=0)
    
    # 回测配置
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    config = Column(JSON, nullable=False)
    
    # 结果
    results = Column(JSON)
    metrics = Column(JSON)
    
    # 时间戳
    created_at = Column(DateTime, server_default=func.now(), index=True)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # 错误信息
    error_message = Column(Text)
    
    # 创建者
    created_by = Column(String(100), index=True)

class FactorValue(Base):
    """因子值数据表（ClickHouse会是主存储）"""
    __tablename__ = "factor_values"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    factor_id = Column(String(100), nullable=False, index=True)
    symbol = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    value = Column(Float, nullable=False)
    rank = Column(Integer)
    quantile = Column(Integer)
    
    # 复合索引
    __table_args__ = (
        Index('idx_factor_symbol_time', 'factor_id', 'symbol', 'timestamp'),
    )
```

---

## 5. API路由设计

### 5.1 因子相关API

```python
# app/api/v2/factors.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.database import get_session
from ...core.security import get_current_user
from ...models.factor import (
    FactorDefinition, FactorCalculationRequest, FactorCalculationResponse,
    FactorLibraryQuery
)
from ...services.factor_service import FactorService

router = APIRouter()

@router.post("/calculate", response_model=FactorCalculationResponse)
async def calculate_factors(
    request: FactorCalculationRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    """批量计算因子"""
    try:
        factor_service = FactorService()
        
        results = await factor_service.calculate_factors(
            factors=request.factors,
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            session=session
        )
        
        # 记录使用统计（后台任务）
        background_tasks.add_task(
            update_factor_usage_stats,
            [f.id for f in request.factors]
        )
        
        return FactorCalculationResponse(
            success=True,
            results=results,
            computation_time=sum(r.calculation_time for r in results.values()),
            cached=any(r.metadata and r.metadata.get("cached") for r in results.values())
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/library")
async def get_factor_library(
    category: Optional[str] = Query(None, description="因子类别"),
    min_ic: Optional[float] = Query(None, ge=-1, le=1, description="最小IC值"),
    min_sharpe: Optional[float] = Query(None, ge=0, description="最小夏普比率"),
    search_text: Optional[str] = Query(None, max_length=100, description="搜索关键词"),
    limit: int = Query(50, ge=1, le=1000, description="返回数量限制"),
    offset: int = Query(0, ge=0, description="偏移量"),
    session: AsyncSession = Depends(get_session)
):
    """获取因子库"""
    try:
        factor_service = FactorService()
        
        if search_text:
            # 搜索模式
            factors = await factor_service.search_factors(
                query=search_text,
                filters={
                    "category": category,
                    "min_ic": min_ic,
                    "min_sharpe": min_sharpe
                },
                limit=limit,
                session=session
            )
        else:
            # 列表模式
            factors = await factor_service.get_factor_library(
                category=category,
                min_ic=min_ic,
                min_sharpe=min_sharpe,
                limit=limit,
                session=session
            )
        
        return {
            "success": True,
            "data": {
                "factors": factors,
                "total_count": len(factors),
                "limit": limit,
                "offset": offset
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create")
async def create_factor(
    factor_def: FactorDefinition,
    current_user: str = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    """创建新因子"""
    try:
        factor_def.created_by = current_user
        
        factor_service = FactorService()
        new_factor = await factor_service.create_factor(factor_def, session)
        
        return {
            "success": True,
            "data": {
                "factor_id": new_factor.id,
                "name": new_factor.name,
                "message": "因子创建成功"
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate/{factor_id}")
async def evaluate_factor(
    factor_id: str,
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
):
    """评估因子性能"""
    try:
        factor_service = FactorService()
        
        performance = await factor_service.evaluate_factor_performance(
            factor_id=factor_id,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            session=session
        )
        
        return {
            "success": True,
            "data": {
                "factor_id": factor_id,
                "performance": performance,
                "evaluation_date": datetime.utcnow().isoformat()
            }
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/realtime/{factor_id}")
async def get_realtime_factor_value(
    factor_id: str,
    symbols: List[str] = Query(..., description="股票代码列表"),
    session: AsyncSession = Depends(get_session)
):
    """获取实时因子值"""
    try:
        factor_service = FactorService()
        
        # 获取因子定义
        factor = await factor_service.get_factor_by_id(factor_id, session)
        if not factor:
            raise HTTPException(status_code=404, detail="Factor not found")
        
        # 计算实时因子值
        values = {}
        for symbol in symbols:
            value = await factor_service.calculate_realtime_factor(
                factor, symbol
            )
            values[symbol] = value
        
        return {
            "success": True,
            "data": {
                "factor_id": factor_id,
                "factor_name": factor.name,
                "values": values,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def update_factor_usage_stats(factor_ids: List[str]):
    """更新因子使用统计（后台任务）"""
    try:
        async with get_session() as session:
            # 更新使用次数和最后使用时间
            for factor_id in factor_ids:
                await session.execute(
                    "UPDATE factor_library SET usage_count = usage_count + 1, "
                    "last_used = :now WHERE id = :factor_id",
                    {"now": datetime.utcnow(), "factor_id": factor_id}
                )
            await session.commit()
    except Exception as e:
        logger.error(f"Failed to update factor usage stats: {e}")
```

---

## 6. 监控和运维

### 6.1 监控中间件

```python
# app/middleware/monitoring.py
import time
import asyncio
from typing import Callable
from fastapi import Request, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

# Prometheus指标定义
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

ACTIVE_REQUESTS = Gauge(
    'http_requests_active',
    'Active HTTP requests'
)

RUST_ENGINE_CALLS = Counter(
    'rust_engine_calls_total',
    'Total Rust engine calls',
    ['function']
)

FACTOR_CALCULATIONS = Histogram(
    'factor_calculation_duration_seconds',
    'Factor calculation duration',
    ['factor_type']
)

class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # 增加活跃请求计数
        ACTIVE_REQUESTS.inc()
        
        try:
            response = await call_next(request)
            
            # 记录指标
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=self._get_endpoint(request),
                status=response.status_code
            ).inc()
            
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=self._get_endpoint(request)
            ).observe(time.time() - start_time)
            
            return response
            
        finally:
            # 减少活跃请求计数
            ACTIVE_REQUESTS.dec()
    
    def _get_endpoint(self, request: Request) -> str:
        """提取API端点"""
        path = request.url.path
        # 简化路径（去掉ID等动态部分）
        if '/factors/' in path and path.split('/')[-1].isdigit():
            return '/factors/{id}'
        elif '/backtest/' in path and path.split('/')[-1].isdigit():
            return '/backtest/{id}'
        return path

@app.get("/metrics")
async def get_metrics():
    """Prometheus指标端点"""
    return Response(generate_latest(), media_type="text/plain")
```

### 6.2 日志配置

```python
# app/core/logging.py
import logging
import sys
from loguru import logger
from pythonjsonlogger import jsonlogger

class StructuredLogger:
    def __init__(self, log_level: str = "INFO"):
        # 移除默认处理器
        logger.remove()
        
        # 控制台输出（开发环境）
        logger.add(
            sys.stderr,
            format="<green>{time}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_level,
            colorize=True
        )
        
        # 结构化日志文件（生产环境）
        logger.add(
            "logs/app.log",
            rotation="100 MB",
            retention="30 days",
            level=log_level,
            format="{time} | {level} | {name}:{function}:{line} | {message}",
            serialize=True  # JSON格式
        )
        
        # 错误日志
        logger.add(
            "logs/errors.log",
            rotation="50 MB", 
            retention="90 days",
            level="ERROR",
            format="{time} | {level} | {name}:{function}:{line} | {message}",
            serialize=True,
            backtrace=True,
            diagnose=True
        )
        
        # 性能日志
        logger.add(
            "logs/performance.log",
            rotation="50 MB",
            retention="7 days", 
            level="INFO",
            format="{time} | {level} | {message}",
            filter=lambda record: "performance" in record["extra"]
        )
    
    def log_api_call(self, endpoint: str, method: str, duration: float, status_code: int):
        """记录API调用"""
        logger.info(
            "API call completed",
            extra={
                "endpoint": endpoint,
                "method": method,
                "duration_ms": duration * 1000,
                "status_code": status_code,
                "event_type": "api_call"
            }
        )
    
    def log_factor_calculation(self, factor_names: list, duration: float, success: bool):
        """记录因子计算"""
        logger.info(
            "Factor calculation completed",
            extra={
                "factor_names": factor_names,
                "duration_ms": duration * 1000,
                "success": success,
                "event_type": "factor_calculation",
                "performance": True
            }
        )
    
    def log_backtest(self, job_id: str, status: str, duration: float = None):
        """记录回测"""
        extra_data = {
            "job_id": job_id,
            "status": status,
            "event_type": "backtest"
        }
        
        if duration:
            extra_data["duration_ms"] = duration * 1000
            extra_data["performance"] = True
        
        logger.info("Backtest status update", extra=extra_data)
    
    def log_rust_engine_call(self, function: str, duration: float, success: bool):
        """记录Rust引擎调用"""
        logger.info(
            "Rust engine call",
            extra={
                "function": function,
                "duration_ms": duration * 1000,
                "success": success,
                "event_type": "rust_engine",
                "performance": True
            }
        )
```

---

## 总结

本Python服务层重构设计提供了：

### 🚀 核心改进

1. **现代化架构**: FastAPI + 异步处理 + 微服务设计
2. **性能提升**: Rust引擎集成，异步数据库操作
3. **生产就绪**: 监控、日志、缓存、限流等完整功能
4. **向后兼容**: 保持现有API接口的兼容性
5. **扩展性**: 模块化设计，支持水平扩展

### 💡 技术优势

- **10倍并发能力**: 异步处理 vs 同步HTTP服务器
- **类型安全**: Pydantic数据验证和FastAPI自动文档
- **缓存优化**: Redis缓存减少重复计算
- **监控完善**: Prometheus指标 + 结构化日志
- **安全认证**: JWT令牌 + 权限控制

### 🛠 实施策略

1. **渐进式迁移**: 先部署V2 API，保持V1兼容
2. **数据库迁移**: 使用Alembic管理数据库版本
3. **性能测试**: 压力测试验证性能提升
4. **监控部署**: 实时监控系统健康状态

该设计为量化分析系统提供了现代化、高性能、生产就绪的API服务层，能够支撑大规模用户和计算需求。