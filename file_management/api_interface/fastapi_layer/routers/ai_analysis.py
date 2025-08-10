"""
AI分析路由模块
处理AI智能分析、市场预测、策略推荐等API
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import asyncio
import json
import numpy as np
from core.real_data_manager import real_data_manager

router = APIRouter()

class MarketAnalysisRequest(BaseModel):
    symbols: List[str] = Field(..., description="要分析的交易对列表")
    timeframe: str = Field("1h", description="时间周期：1m, 5m, 15m, 1h, 4h, 1d")
    analysis_type: str = Field("comprehensive", description="分析类型：technical, sentiment, comprehensive")

class TechnicalSignal(BaseModel):
    indicator: str
    signal: str  # buy, sell, neutral
    strength: float  # 0-100
    value: float
    description: str

class MarketSentiment(BaseModel):
    score: float  # -100 to 100
    trend: str    # bullish, bearish, neutral
    confidence: float  # 0-100
    factors: List[str]

class AIAnalysisResult(BaseModel):
    symbol: str
    timeframe: str
    analysis_type: str
    overall_signal: str  # buy, sell, hold
    confidence: float
    technical_signals: List[TechnicalSignal]
    sentiment: MarketSentiment
    price_prediction: Dict[str, float]
    risk_assessment: str
    generated_at: datetime

class CustomFactorRequest(BaseModel):
    """自定义因子请求"""
    name: str = Field(..., description="因子名称")
    description: str = Field(..., description="因子描述")
    formula: str = Field(..., description="计算公式")
    data_sources: List[str] = Field(..., description="数据源列表")
    parameters: Dict[str, Any] = Field({}, description="因子参数")
    category: str = Field("technical", description="因子类别")
    
    @validator('name')
    def validate_name(cls, v):
        if not v.replace('_', '').isalnum():
            raise ValueError("因子名称只能包含字母、数字和下划线")
        return v

class ModelTrainingRequest(BaseModel):
    """模型训练请求"""
    model_name: str = Field(..., description="模型名称")
    model_type: str = Field("regression", description="模型类型")
    features: List[str] = Field(..., description="特征列表")
    target: str = Field(..., description="目标变量")
    training_period: int = Field(30, description="训练周期（天）")
    validation_split: float = Field(0.2, description="验证集比例", ge=0.1, le=0.4)
    hyperparameters: Dict[str, Any] = Field({}, description="超参数")

class ModelPredictionRequest(BaseModel):
    """模型预测请求"""
    model_id: str = Field(..., description="模型ID")
    symbols: List[str] = Field(..., description="交易品种")
    features: Optional[Dict[str, Any]] = Field(None, description="输入特征")

class FactorAnalysisRequest(BaseModel):
    """因子分析请求"""
    factors: List[str] = Field(..., description="因子列表")
    symbols: List[str] = Field(..., description="交易品种")
    start_date: datetime = Field(..., description="开始日期")
    end_date: datetime = Field(..., description="结束日期")
    analysis_type: str = Field("correlation", description="分析类型")

class BacktestFactorRequest(BaseModel):
    """因子回测请求"""
    factor_name: str = Field(..., description="因子名称")
    symbols: List[str] = Field(..., description="测试品种")
    start_date: datetime = Field(..., description="开始日期") 
    end_date: datetime = Field(..., description="结束日期")
    rebalance_frequency: str = Field("daily", description="调仓频率")

@router.post("/market-analysis", response_model=Dict[str, AIAnalysisResult])
async def analyze_market(
    request: MarketAnalysisRequest,
):
    """执行AI市场分析"""
    try:
        results = {}
        
        for symbol in request.symbols:
            # 获取市场数据
            market_data = await real_data_manager.get_latest_prices()
            coinglass_data = await real_data_manager.get_coinglass_analysis()
            
            # 构建技术信号（模拟）
            technical_signals = [
                TechnicalSignal(
                    indicator="RSI",
                    signal="neutral",
                    strength=45.0,
                    value=52.3,
                    description="RSI指标显示中性区域"
                ),
                TechnicalSignal(
                    indicator="MACD", 
                    signal="buy",
                    strength=65.0,
                    value=0.0023,
                    description="MACD金叉信号"
                ),
                TechnicalSignal(
                    indicator="Bollinger Bands",
                    signal="neutral",
                    strength=30.0,
                    value=0.5,
                    description="价格在布林带中轨附近"
                )
            ]
            
            # 构建市场情绪（基于Coinglass数据）
            sentiment_data = coinglass_data.get("sentiment", {})
            sentiment = MarketSentiment(
                score=sentiment_data.get("score", 0),
                trend=sentiment_data.get("trend", "neutral"),
                confidence=sentiment_data.get("confidence", 0) * 100,
                factors=["恐惧贪婪指数", "资金费率", "持仓量变化"]
            )
            
            # 计算综合信号
            bullish_signals = len([s for s in technical_signals if s.signal == "buy"])
            bearish_signals = len([s for s in technical_signals if s.signal == "sell"])
            
            if bullish_signals > bearish_signals:
                overall_signal = "buy"
                confidence = min(75.0, bullish_signals * 25)
            elif bearish_signals > bullish_signals:
                overall_signal = "sell"
                confidence = min(75.0, bearish_signals * 25)
            else:
                overall_signal = "hold"
                confidence = 50.0
            
            # 价格预测（模拟）
            current_price = 50000.0  # 模拟当前价格
            price_prediction = {
                "1h": current_price * (1 + 0.001),
                "4h": current_price * (1 + 0.005),
                "24h": current_price * (1 + 0.02),
                "7d": current_price * (1 + 0.05)
            }
            
            # 风险评估
            risk_level = coinglass_data.get("composite_signal", {}).get("risk_assessment", "moderate")
            
            results[symbol] = AIAnalysisResult(
                symbol=symbol,
                timeframe=request.timeframe,
                analysis_type=request.analysis_type,
                overall_signal=overall_signal,
                confidence=confidence,
                technical_signals=technical_signals,
                sentiment=sentiment,
                price_prediction=price_prediction,
                risk_assessment=risk_level,
                generated_at=datetime.utcnow()
            )
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI分析失败: {str(e)}")

@router.get("/sentiment")
async def get_market_sentiment():
    """获取市场情绪分析"""
    try:
        coinglass_data = await real_data_manager.get_coinglass_analysis()
        
        # 构建综合情绪分析
        sentiment_analysis = {
            "overall_sentiment": {
                "score": coinglass_data.get("sentiment", {}).get("score", 0),
                "trend": coinglass_data.get("sentiment", {}).get("trend", "neutral"),
                "strength": coinglass_data.get("sentiment", {}).get("strength", "weak"),
                "confidence": coinglass_data.get("sentiment", {}).get("confidence", 0)
            },
            "fear_greed_index": {
                "value": abs(coinglass_data.get("sentiment", {}).get("score", 0)),
                "classification": "Neutral",
                "trend": coinglass_data.get("sentiment", {}).get("trend", "neutral")
            },
            "funding_rates": coinglass_data.get("funding_rates", {}),
            "open_interest": coinglass_data.get("open_interest", {}),
            "etf_flows": coinglass_data.get("etf_flows", {}),
            "composite_signal": coinglass_data.get("composite_signal", {}),
            "last_updated": datetime.utcnow().isoformat()
        }
        
        return {
            "success": True,
            "data": sentiment_analysis,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取市场情绪失败: {str(e)}")

@router.get("/signals")
async def get_trading_signals(
    symbols: Optional[str] = Query(None, description="交易对列表，逗号分隔"),
    timeframe: str = Query("1h", description="时间周期"),
    signal_type: str = Query("all", description="信号类型：technical, sentiment, all")
):
    """获取交易信号"""
    try:
        # 解析交易对
        if symbols:
            symbol_list = [s.strip() for s in symbols.split(",")]
        else:
            symbol_list = ["BTC/USDT", "ETH/USDT"]
        
        signals = {}
        
        for symbol in symbol_list:
            # 获取技术信号
            technical_signals = []
            if signal_type in ["technical", "all"]:
                technical_signals = [
                    {
                        "indicator": "RSI_14",
                        "signal": "neutral",
                        "value": 52.3,
                        "strength": 45.0
                    },
                    {
                        "indicator": "MACD",
                        "signal": "buy",
                        "value": 0.0023,
                        "strength": 65.0
                    },
                    {
                        "indicator": "SMA_20_50",
                        "signal": "buy",
                        "value": 1.02,
                        "strength": 55.0
                    }
                ]
            
            # 获取情绪信号
            sentiment_signals = {}
            if signal_type in ["sentiment", "all"]:
                coinglass_data = await real_data_manager.get_coinglass_analysis()
                sentiment_signals = {
                    "fear_greed": {
                        "value": coinglass_data.get("sentiment", {}).get("score", 0),
                        "signal": coinglass_data.get("sentiment", {}).get("trend", "neutral")
                    },
                    "funding_rate": {
                        "value": coinglass_data.get("funding_rates", {}).get("overall_rate", 0),
                        "signal": coinglass_data.get("funding_rates", {}).get("trend", "stable")
                    }
                }
            
            signals[symbol] = {
                "symbol": symbol,
                "timeframe": timeframe,
                "technical_signals": technical_signals,
                "sentiment_signals": sentiment_signals,
                "generated_at": datetime.utcnow().isoformat()
            }
        
        return {
            "success": True,
            "data": signals,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取交易信号失败: {str(e)}")

@router.post("/predict-price")
async def predict_price(
    symbol: str = Query(..., description="交易对"),
    horizon: str = Query("1h", description="预测时间：1h, 4h, 24h, 7d"),
):
    """AI价格预测"""
    try:
        # 获取当前市场数据
        market_data = await real_data_manager.get_latest_prices()
        coinglass_data = await real_data_manager.get_coinglass_analysis()
        
        # 模拟AI价格预测
        current_price = 50000.0  # 模拟当前价格
        
        # 基于市场情绪调整预测
        sentiment_score = coinglass_data.get("sentiment", {}).get("score", 0)
        sentiment_multiplier = 1 + (sentiment_score / 1000)  # 微调
        
        horizon_map = {
            "1h": {"change": 0.001, "confidence": 85},
            "4h": {"change": 0.005, "confidence": 75},
            "24h": {"change": 0.02, "confidence": 65},
            "7d": {"change": 0.08, "confidence": 45}
        }
        
        prediction_data = horizon_map.get(horizon, horizon_map["24h"])
        predicted_change = prediction_data["change"] * sentiment_multiplier
        predicted_price = current_price * (1 + predicted_change)
        
        prediction = {
            "symbol": symbol,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "predicted_change": predicted_change * 100,
            "horizon": horizon,
            "confidence": prediction_data["confidence"],
            "factors": [
                f"市场情绪得分: {sentiment_score}",
                f"技术指标综合: 中性偏多",
                f"市场状态: {coinglass_data.get('composite_signal', {}).get('market_regime', 'sideways')}"
            ],
            "risk_level": coinglass_data.get("composite_signal", {}).get("risk_assessment", "moderate"),
            "generated_at": datetime.utcnow().isoformat()
        }
        
        return {
            "success": True,
            "data": prediction,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"价格预测失败: {str(e)}")

@router.get("/analysis-history")
async def get_analysis_history(
    limit: int = Query(20, ge=1, le=100, description="返回数量限制"),
    analysis_type: Optional[str] = Query(None, description="分析类型过滤"),
):
    """获取AI分析历史"""
    try:
        # 模拟历史分析记录
        history = []
        for i in range(limit):
            created_time = datetime.utcnow() - timedelta(hours=i)
            history.append({
                "id": f"analysis_{i+1:04d}",
                "analysis_type": "market_sentiment",
                "symbols": ["BTC/USDT", "ETH/USDT"],
                "result_summary": {
                    "overall_signal": "buy" if i % 3 == 0 else "hold",
                    "confidence": 65 + (i % 30),
                    "sentiment_score": 25 - (i % 50)
                },
                "created_at": created_time.isoformat(),
                "execution_time_ms": 1200 + (i % 500)
            })
        
        return {
            "success": True,
            "data": history,
            "total": len(history),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取分析历史失败: {str(e)}")

@router.post("/refresh-analysis")
async def refresh_analysis(
    background_tasks: BackgroundTasks,
    force: bool = Query(False, description="是否强制刷新"),
):
    """刷新AI分析缓存"""
    try:
        # 添加后台任务来刷新分析
        background_tasks.add_task(refresh_coinglass_analysis)
        
        return {
            "success": True,
            "message": "AI分析刷新任务已启动",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"刷新分析失败: {str(e)}")

async def refresh_coinglass_analysis():
    """后台任务：刷新Coinglass分析"""
    try:
        # 清除分析缓存
        from core.coinglass_analyzer import coinglass_analyzer
        coinglass_analyzer.analysis_cache.clear()
        
        # 重新获取分析
        await real_data_manager.get_coinglass_analysis()
        
    except Exception as e:
        print(f"刷新Coinglass分析失败: {e}")

# ============ 自定义因子和模型训练API ============

# 全局存储（实际应该使用数据库）
custom_factors_db = {}
trained_models_db = {}

@router.post("/factors/create")
async def create_custom_factor(
    factor_request: CustomFactorRequest,
):
    """创建自定义因子"""
    try:
        factor_id = f"factor_{len(custom_factors_db) + 1:04d}"
        
        # 验证公式安全性（简化）
        forbidden_keywords = ["import", "exec", "eval", "__", "open", "file"]
        if any(keyword in factor_request.formula.lower() for keyword in forbidden_keywords):
            raise HTTPException(status_code=400, detail="因子公式包含禁用关键词")
        
        # 保存因子配置
        factor_config = {
            "id": factor_id,
            "name": factor_request.name,
            "description": factor_request.description,
            "formula": factor_request.formula,
            "data_sources": factor_request.data_sources,
            "parameters": factor_request.parameters,
            "category": factor_request.category,
            "created_by": "user",
            "created_at": datetime.utcnow(),
            "status": "active"
        }
        
        custom_factors_db[factor_id] = factor_config
        
        return {
            "success": True,
            "message": "自定义因子创建成功",
            "data": {
                "factor_id": factor_id,
                "factor_name": factor_request.name
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建因子失败: {str(e)}")

@router.get("/factors")
async def list_custom_factors(
    category: Optional[str] = Query(None, description="因子类别过滤"),
):
    """获取自定义因子列表"""
    try:
        factors = list(custom_factors_db.values())
        
        # 按类别过滤
        if category:
            factors = [f for f in factors if f["category"] == category]
        
        # 隐藏敏感信息
        for factor in factors:
            factor.pop("formula", None)  # 不显示具体公式
        
        return {
            "success": True,
            "data": factors,
            "total": len(factors),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取因子列表失败: {str(e)}")

@router.get("/factors/{factor_id}")
async def get_factor_details(
    factor_id: str,
):
    """获取因子详情"""
    try:
        if factor_id not in custom_factors_db:
            raise HTTPException(status_code=404, detail="因子不存在")
        
        factor = custom_factors_db[factor_id].copy()
        
        # 始终显示因子详情（个人使用）
        
        return {
            "success": True,
            "data": factor,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取因子详情失败: {str(e)}")

@router.post("/factors/calculate")
async def calculate_factor_values(
    factor_id: str,
    symbols: List[str],
    start_date: datetime,
    end_date: datetime,
):
    """计算因子值"""
    try:
        if factor_id not in custom_factors_db:
            raise HTTPException(status_code=404, detail="因子不存在")
        
        factor = custom_factors_db[factor_id]
        
        # 模拟因子计算
        results = {}
        for symbol in symbols:
            # 生成模拟时间序列数据
            days = (end_date - start_date).days
            dates = [start_date + timedelta(days=i) for i in range(days + 1)]
            
            # 模拟因子值（正态分布）
            np.random.seed(hash(symbol + factor_id) % 1000)
            factor_values = np.random.normal(0, 1, len(dates)).tolist()
            
            results[symbol] = [
                {
                    "date": date.isoformat(),
                    "value": round(value, 6)
                }
                for date, value in zip(dates, factor_values)
            ]
        
        return {
            "success": True,
            "data": {
                "factor_id": factor_id,
                "factor_name": factor["name"],
                "calculation_period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "results": results,
                "total_points": sum(len(values) for values in results.values())
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"计算因子值失败: {str(e)}")

@router.post("/factors/analyze")
async def analyze_factor_performance(
    analysis_request: FactorAnalysisRequest,
):
    """因子表现分析"""
    try:
        # 验证因子存在
        for factor_name in analysis_request.factors:
            factor_exists = any(
                f["name"] == factor_name for f in custom_factors_db.values()
            )
            if not factor_exists:
                raise HTTPException(status_code=404, detail=f"因子 {factor_name} 不存在")
        
        # 模拟因子分析结果
        analysis_results = {}
        
        if analysis_request.analysis_type == "correlation":
            # 相关性分析
            correlation_matrix = {}
            for i, factor1 in enumerate(analysis_request.factors):
                correlation_matrix[factor1] = {}
                for j, factor2 in enumerate(analysis_request.factors):
                    # 模拟相关系数
                    if i == j:
                        corr = 1.0
                    else:
                        np.random.seed(hash(factor1 + factor2) % 1000)
                        corr = round(np.random.uniform(-0.8, 0.8), 3)
                    correlation_matrix[factor1][factor2] = corr
            
            analysis_results = {
                "analysis_type": "correlation",
                "correlation_matrix": correlation_matrix,
                "high_correlations": [
                    {"factor1": "momentum", "factor2": "trend", "correlation": 0.75},
                    {"factor1": "volatility", "factor2": "risk", "correlation": -0.68}
                ]
            }
            
        elif analysis_request.analysis_type == "ic":
            # 信息系数分析
            ic_results = {}
            for factor in analysis_request.factors:
                np.random.seed(hash(factor) % 1000)
                ic_results[factor] = {
                    "mean_ic": round(np.random.uniform(-0.05, 0.15), 4),
                    "ic_std": round(np.random.uniform(0.1, 0.3), 4),
                    "ic_ir": round(np.random.uniform(-0.5, 1.5), 4),
                    "positive_ic_ratio": round(np.random.uniform(0.4, 0.7), 3)
                }
            
            analysis_results = {
                "analysis_type": "information_coefficient",
                "ic_analysis": ic_results,
                "summary": {
                    "best_factor": max(ic_results.keys(), key=lambda x: ic_results[x]["ic_ir"]),
                    "avg_ic": round(np.mean([r["mean_ic"] for r in ic_results.values()]), 4)
                }
            }
        
        return {
            "success": True,
            "data": {
                "analysis_request": analysis_request.dict(),
                "results": analysis_results,
                "computed_at": datetime.utcnow().isoformat()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"因子分析失败: {str(e)}")

@router.post("/models/train")
async def train_model(
    training_request: ModelTrainingRequest,
    background_tasks: BackgroundTasks,
):
    """训练机器学习模型"""
    try:
        model_id = f"model_{len(trained_models_db) + 1:04d}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # 验证特征是否存在
        for feature in training_request.features:
            feature_exists = any(
                f["name"] == feature for f in custom_factors_db.values()
            )
            if not feature_exists and feature not in ["price", "volume", "returns"]:  # 允许内置特征
                raise HTTPException(status_code=404, detail=f"特征 {feature} 不存在")
        
        # 创建训练任务
        model_config = {
            "id": model_id,
            "name": training_request.model_name,
            "type": training_request.model_type,
            "features": training_request.features,
            "target": training_request.target,
            "training_period": training_request.training_period,
            "validation_split": training_request.validation_split,
            "hyperparameters": training_request.hyperparameters,
            "created_by": "user",
            "created_at": datetime.utcnow(),
            "status": "training",
            "progress": 0.0
        }
        
        trained_models_db[model_id] = model_config
        
        # 启动后台训练任务
        background_tasks.add_task(train_model_background, model_id, training_request)
        
        return {
            "success": True,
            "message": "模型训练任务已启动",
            "data": {
                "model_id": model_id,
                "estimated_completion": datetime.utcnow() + timedelta(minutes=10)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动模型训练失败: {str(e)}")

async def train_model_background(model_id: str, training_request: ModelTrainingRequest):
    """后台模型训练任务"""
    try:
        model_config = trained_models_db[model_id]
        
        # 模拟训练过程
        for progress in range(0, 101, 10):
            model_config["progress"] = progress
            await asyncio.sleep(0.5)  # 模拟训练时间
        
        # 模拟训练结果
        model_config.update({
            "status": "completed",
            "progress": 100.0,
            "training_results": {
                "train_score": 0.85,
                "validation_score": 0.78,
                "test_score": 0.72,
                "feature_importance": {
                    feature: round(np.random.uniform(0.05, 0.3), 3)
                    for feature in training_request.features
                },
                "training_time_seconds": 180
            },
            "completed_at": datetime.utcnow()
        })
        
    except Exception as e:
        trained_models_db[model_id]["status"] = "failed"
        trained_models_db[model_id]["error"] = str(e)

@router.get("/models")
async def list_trained_models(
    status: Optional[str] = Query(None, description="状态过滤"),
):
    """获取训练模型列表"""
    try:
        models = list(trained_models_db.values())
        
        # 按状态过滤
        if status:
            models = [m for m in models if m["status"] == status]
        
        return {
            "success": True,
            "data": models,
            "total": len(models),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")

@router.get("/models/{model_id}")
async def get_model_details(
    model_id: str,
):
    """获取模型详情"""
    try:
        if model_id not in trained_models_db:
            raise HTTPException(status_code=404, detail="模型不存在")
        
        model = trained_models_db[model_id]
        
        return {
            "success": True,
            "data": model,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型详情失败: {str(e)}")

@router.post("/models/predict")
async def model_predict(
    prediction_request: ModelPredictionRequest,
):
    """使用模型进行预测"""
    try:
        model_id = prediction_request.model_id
        
        if model_id not in trained_models_db:
            raise HTTPException(status_code=404, detail="模型不存在")
        
        model = trained_models_db[model_id]
        
        if model["status"] != "completed":
            raise HTTPException(status_code=400, detail=f"模型状态为 {model['status']}，无法进行预测")
        
        # 模拟预测过程
        predictions = {}
        for symbol in prediction_request.symbols:
            np.random.seed(hash(model_id + symbol) % 1000)
            
            # 模拟预测结果
            prediction_value = round(np.random.uniform(-0.05, 0.05), 4)  # 预测收益率
            confidence = round(np.random.uniform(0.6, 0.9), 3)
            
            predictions[symbol] = {
                "predicted_value": prediction_value,
                "confidence": confidence,
                "prediction_type": model["target"],
                "feature_contributions": {
                    feature: round(np.random.uniform(-0.02, 0.02), 4)
                    for feature in model["features"]
                }
            }
        
        return {
            "success": True,
            "data": {
                "model_id": model_id,
                "model_name": model["name"],
                "predictions": predictions,
                "prediction_timestamp": datetime.utcnow().isoformat()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型预测失败: {str(e)}")

@router.post("/factors/backtest")
async def backtest_factor(
    backtest_request: BacktestFactorRequest,
    background_tasks: BackgroundTasks,
):
    """因子回测"""
    try:
        # 验证因子存在
        factor_exists = any(
            f["name"] == backtest_request.factor_name for f in custom_factors_db.values()
        )
        if not factor_exists:
            raise HTTPException(status_code=404, detail=f"因子 {backtest_request.factor_name} 不存在")
        
        backtest_id = f"backtest_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # 启动后台回测任务
        background_tasks.add_task(run_factor_backtest, backtest_id, backtest_request)
        
        return {
            "success": True,
            "message": "因子回测任务已启动",
            "data": {
                "backtest_id": backtest_id,
                "factor_name": backtest_request.factor_name,
                "estimated_completion": datetime.utcnow() + timedelta(minutes=5)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"启动因子回测失败: {str(e)}")

async def run_factor_backtest(backtest_id: str, backtest_request: BacktestFactorRequest):
    """后台因子回测任务"""
    try:
        # 模拟回测计算
        await asyncio.sleep(3)  # 模拟计算时间
        
        # 生成模拟回测结果
        np.random.seed(hash(backtest_id) % 1000)
        
        # 模拟回测指标
        total_return = round(np.random.uniform(0.05, 0.25), 4)
        annual_return = round(np.random.uniform(0.08, 0.35), 4)
        volatility = round(np.random.uniform(0.15, 0.3), 4)
        sharpe_ratio = round(annual_return / volatility, 4)
        max_drawdown = round(np.random.uniform(0.05, 0.2), 4)
        
        # 模拟净值曲线
        days = (backtest_request.end_date - backtest_request.start_date).days
        dates = [backtest_request.start_date + timedelta(days=i) for i in range(days + 1)]
        
        cumulative_returns = [1.0]
        for i in range(1, len(dates)):
            daily_return = np.random.normal(annual_return / 252, volatility / np.sqrt(252))
            cumulative_returns.append(cumulative_returns[-1] * (1 + daily_return))
        
        nav_curve = [
            {
                "date": date.isoformat(),
                "nav": round(nav, 6)
            }
            for date, nav in zip(dates, cumulative_returns)
        ]
        
        backtest_results = {
            "backtest_id": backtest_id,
            "factor_name": backtest_request.factor_name,
            "performance_metrics": {
                "total_return": total_return,
                "annual_return": annual_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "calmar_ratio": round(annual_return / max_drawdown, 4)
            },
            "nav_curve": nav_curve,
            "symbols_tested": backtest_request.symbols,
            "backtest_period": {
                "start_date": backtest_request.start_date.isoformat(),
                "end_date": backtest_request.end_date.isoformat()
            },
            "completed_at": datetime.utcnow().isoformat()
        }
        
        # 在实际实现中，这里应该保存到数据库
        print(f"因子回测 {backtest_id} 完成")
        
    except Exception as e:
        print(f"因子回测失败: {e}")

@router.post("/models/upload")
async def upload_model_file(
    file: UploadFile = File(...),
    model_name: str = Query(..., description="模型名称"),
):
    """上传预训练模型文件"""
    try:
        # 验证文件类型
        allowed_extensions = [".pkl", ".joblib", ".h5", ".onnx"]
        if not any(file.filename.endswith(ext) for ext in allowed_extensions):
            raise HTTPException(status_code=400, detail="不支持的文件类型")
        
        # 验证文件大小（限制为100MB）
        file_content = await file.read()
        if len(file_content) > 100 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="文件大小超过限制（100MB）")
        
        model_id = f"uploaded_model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # 模拟文件保存
        model_config = {
            "id": model_id,
            "name": model_name,
            "type": "uploaded",
            "filename": file.filename,
            "file_size": len(file_content),
            "upload_by": "user",
            "uploaded_at": datetime.utcnow(),
            "status": "active"
        }
        
        trained_models_db[model_id] = model_config
        
        return {
            "success": True,
            "message": "模型文件上传成功",
            "data": {
                "model_id": model_id,
                "filename": file.filename,
                "file_size": len(file_content)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传模型文件失败: {str(e)}")