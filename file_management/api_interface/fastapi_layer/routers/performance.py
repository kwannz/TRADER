"""
性能分析API路由
提供投资组合绩效分析、风险评估、收益归因等功能
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import numpy as np

router = APIRouter()

class PerformancePeriod(str, Enum):
    """性能分析周期"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"

class ReportType(str, Enum):
    """报告类型"""
    SUMMARY = "summary"
    DETAILED = "detailed"
    RISK_ANALYSIS = "risk_analysis"
    ATTRIBUTION = "attribution"
    COMPARISON = "comparison"

class PerformanceAnalysisRequest(BaseModel):
    """性能分析请求"""
    portfolio_id: Optional[str] = Field(None, description="投资组合ID")
    start_date: datetime = Field(..., description="分析开始日期")
    end_date: datetime = Field(..., description="分析结束日期")
    benchmark: Optional[str] = Field("BTC", description="基准")
    metrics: List[str] = Field(["return", "volatility", "sharpe", "drawdown"], description="分析指标")
    period: PerformancePeriod = Field(PerformancePeriod.DAILY, description="分析周期")
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError("结束日期必须晚于开始日期")
        return v

class RiskAnalysisRequest(BaseModel):
    """风险分析请求"""
    portfolio_id: Optional[str] = Field(None, description="投资组合ID")
    analysis_period: int = Field(252, description="分析期间（天数）")
    confidence_level: float = Field(0.95, description="置信水平", ge=0.5, le=0.99)
    risk_models: List[str] = Field(["var", "cvar", "beta"], description="风险模型")

class AttributionAnalysisRequest(BaseModel):
    """归因分析请求"""
    portfolio_id: str = Field(..., description="投资组合ID") 
    benchmark_id: str = Field(..., description="基准ID")
    start_date: datetime = Field(..., description="分析开始日期")
    end_date: datetime = Field(..., description="分析结束日期")
    attribution_method: str = Field("brinson", description="归因方法")

class ComparisonRequest(BaseModel):
    """对比分析请求"""
    portfolios: List[str] = Field(..., description="投资组合ID列表")
    benchmark: Optional[str] = Field(None, description="基准ID")
    start_date: datetime = Field(..., description="对比开始日期")
    end_date: datetime = Field(..., description="对比结束日期")
    metrics: List[str] = Field(["return", "risk", "sharpe"], description="对比指标")

# 模拟性能数据存储
performance_data_db = {}
portfolio_data_db = {
    "portfolio_001": {
        "name": "量化策略组合",
        "inception_date": datetime(2024, 1, 1),
        "total_return": 0.25,
        "benchmark": "BTC"
    },
    "portfolio_002": {
        "name": "套利策略组合", 
        "inception_date": datetime(2024, 1, 1),
        "total_return": 0.15,
        "benchmark": "ETH"
    }
}

def generate_mock_returns(days: int, annual_return: float = 0.12, volatility: float = 0.2) -> List[float]:
    """生成模拟收益率序列"""
    np.random.seed(42)
    daily_return = annual_return / 252
    daily_vol = volatility / np.sqrt(252)
    returns = np.random.normal(daily_return, daily_vol, days)
    return returns.tolist()

def calculate_performance_metrics(returns: List[float]) -> Dict[str, float]:
    """计算性能指标"""
    returns_array = np.array(returns)
    
    # 基础指标
    total_return = (1 + returns_array).prod() - 1
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1
    volatility = returns_array.std() * np.sqrt(252)
    
    # 风险调整收益
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    # 最大回撤
    cumulative_returns = (1 + returns_array).cumprod()
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # Calmar比率
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0
    
    # 胜率
    win_rate = len(returns_array[returns_array > 0]) / len(returns_array)
    
    return {
        "total_return": round(total_return, 4),
        "annual_return": round(annual_return, 4),
        "volatility": round(volatility, 4),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "max_drawdown": round(max_drawdown, 4),
        "calmar_ratio": round(calmar_ratio, 4),
        "win_rate": round(win_rate, 4),
        "total_trades": len(returns),
        "avg_return": round(returns_array.mean(), 6),
        "skewness": round(float(np.nan_to_num(((returns_array - returns_array.mean()) ** 3).mean() / (returns_array.std() ** 3))), 4),
        "kurtosis": round(float(np.nan_to_num(((returns_array - returns_array.mean()) ** 4).mean() / (returns_array.std() ** 4) - 3)), 4)
    }

@router.post("/analyze")
async def analyze_performance(
    request: PerformanceAnalysisRequest,
):
    """执行性能分析"""
    try:
        # 计算分析天数
        analysis_days = (request.end_date - request.start_date).days
        if analysis_days <= 0:
            raise HTTPException(status_code=400, detail="分析期间必须大于0天")
        
        # 生成模拟收益数据
        portfolio_returns = generate_mock_returns(analysis_days, 0.15, 0.25)
        benchmark_returns = generate_mock_returns(analysis_days, 0.12, 0.20)
        
        # 计算投资组合指标
        portfolio_metrics = calculate_performance_metrics(portfolio_returns)
        benchmark_metrics = calculate_performance_metrics(benchmark_returns)
        
        # 相对性能指标
        excess_returns = np.array(portfolio_returns) - np.array(benchmark_returns)
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
        
        # 生成时间序列数据
        dates = [request.start_date + timedelta(days=i) for i in range(analysis_days)]
        portfolio_cumulative = (1 + np.array(portfolio_returns)).cumprod()
        benchmark_cumulative = (1 + np.array(benchmark_returns)).cumprod()
        
        performance_series = [
            {
                "date": date.isoformat(),
                "portfolio_value": round(portfolio_cum, 6),
                "benchmark_value": round(benchmark_cum, 6),
                "portfolio_return": round(portfolio_ret, 6),
                "benchmark_return": round(benchmark_ret, 6)
            }
            for date, portfolio_cum, benchmark_cum, portfolio_ret, benchmark_ret
            in zip(dates, portfolio_cumulative, benchmark_cumulative, portfolio_returns, benchmark_returns)
        ]
        
        result = {
            "analysis_period": {
                "start_date": request.start_date.isoformat(),
                "end_date": request.end_date.isoformat(),
                "days": analysis_days
            },
            "portfolio_metrics": portfolio_metrics,
            "benchmark_metrics": benchmark_metrics,
            "relative_metrics": {
                "excess_return": round(portfolio_metrics["total_return"] - benchmark_metrics["total_return"], 4),
                "tracking_error": round(tracking_error, 4),
                "information_ratio": round(information_ratio, 4),
                "beta": round(np.corrcoef(portfolio_returns, benchmark_returns)[0,1] * 
                            (np.std(portfolio_returns) / np.std(benchmark_returns)), 4),
                "alpha": round((portfolio_metrics["annual_return"] - 
                             benchmark_metrics["annual_return"] * np.corrcoef(portfolio_returns, benchmark_returns)[0,1] * 
                             (np.std(portfolio_returns) / np.std(benchmark_returns))), 4)
            },
            "performance_series": performance_series[-50:] if len(performance_series) > 50 else performance_series,  # 限制返回数据量
            "total_data_points": len(performance_series)
        }
        
        return {
            "success": True,
            "data": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"性能分析失败: {str(e)}")

@router.post("/risk-analysis")
async def analyze_risk(
    request: RiskAnalysisRequest,
):
    """风险分析"""
    try:
        # 生成模拟收益数据
        returns = generate_mock_returns(request.analysis_period, 0.15, 0.25)
        returns_array = np.array(returns)
        
        # VaR计算
        var_95 = np.percentile(returns_array, (1 - request.confidence_level) * 100)
        var_99 = np.percentile(returns_array, 1)
        
        # CVaR计算
        cvar_95 = returns_array[returns_array <= var_95].mean()
        cvar_99 = returns_array[returns_array <= var_99].mean()
        
        # 波动率分析
        rolling_vol = []
        window = 30  # 30天滚动窗口
        for i in range(window, len(returns_array)):
            rolling_vol.append(returns_array[i-window:i].std() * np.sqrt(252))
        
        # 最大回撤分析
        cumulative_returns = (1 + returns_array).cumprod()
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        
        # 回撤持续期分析
        drawdown_periods = []
        in_drawdown = False
        start_idx = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                start_idx = i
            elif dd >= 0 and in_drawdown:
                in_drawdown = False
                drawdown_periods.append({
                    "start_date": (datetime.utcnow() - timedelta(days=len(returns_array)-start_idx)).isoformat(),
                    "end_date": (datetime.utcnow() - timedelta(days=len(returns_array)-i)).isoformat(),
                    "duration_days": i - start_idx,
                    "max_drawdown": round(drawdown[start_idx:i].min(), 4)
                })
        
        # 风险贡献分析（模拟）
        risk_contributions = {
            "market_risk": 0.65,
            "specific_risk": 0.25,
            "liquidity_risk": 0.08,
            "operational_risk": 0.02
        }
        
        risk_analysis = {
            "var_analysis": {
                "confidence_level": request.confidence_level,
                "daily_var_95": round(var_95, 4),
                "daily_var_99": round(var_99, 4),
                "monthly_var_95": round(var_95 * np.sqrt(21), 4),
                "annual_var_95": round(var_95 * np.sqrt(252), 4)
            },
            "cvar_analysis": {
                "daily_cvar_95": round(cvar_95, 4),
                "daily_cvar_99": round(cvar_99, 4),
                "monthly_cvar_95": round(cvar_95 * np.sqrt(21), 4)
            },
            "volatility_analysis": {
                "current_volatility": round(returns_array[-30:].std() * np.sqrt(252), 4),
                "average_volatility": round(returns_array.std() * np.sqrt(252), 4),
                "volatility_trend": "increasing" if len(rolling_vol) > 0 and rolling_vol[-1] > np.mean(rolling_vol) else "decreasing",
                "rolling_volatility": rolling_vol[-30:] if len(rolling_vol) > 30 else rolling_vol
            },
            "drawdown_analysis": {
                "current_drawdown": round(drawdown[-1], 4),
                "max_drawdown": round(drawdown.min(), 4),
                "average_drawdown": round(drawdown[drawdown < 0].mean(), 4),
                "drawdown_periods": drawdown_periods[-5:],  # 最近5次回撤
                "recovery_time": len([x for x in drawdown if x < 0])  # 简化恢复时间
            },
            "risk_contributions": risk_contributions,
            "stress_test_results": {
                "market_crash_2020": round(var_95 * 3, 4),
                "crypto_winter_2022": round(var_95 * 2.5, 4),
                "liquidity_crisis": round(var_95 * 2, 4)
            }
        }
        
        return {
            "success": True,
            "data": {
                "analysis_period_days": request.analysis_period,
                "risk_analysis": risk_analysis,
                "summary": {
                    "risk_level": "medium" if abs(var_95) < 0.03 else "high",
                    "risk_score": min(100, abs(var_95) * 1000),
                    "key_risks": ["市场风险", "波动率风险", "流动性风险"]
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"风险分析失败: {str(e)}")

@router.post("/attribution")
async def analyze_attribution(
    request: AttributionAnalysisRequest,
):
    """归因分析"""
    try:
        analysis_days = (request.end_date - request.start_date).days
        
        # 模拟归因分析结果
        np.random.seed(hash(request.portfolio_id) % 1000)
        
        # 行业/资产配置归因
        asset_allocation = {
            "BTC": {"weight": 0.6, "return": 0.12, "benchmark_weight": 0.5, "benchmark_return": 0.10},
            "ETH": {"weight": 0.25, "return": 0.15, "benchmark_weight": 0.3, "benchmark_return": 0.12},
            "Others": {"weight": 0.15, "return": 0.08, "benchmark_weight": 0.2, "benchmark_return": 0.09}
        }
        
        # 计算配置效应和选择效应
        allocation_effects = {}
        selection_effects = {}
        interaction_effects = {}
        
        benchmark_return = sum(asset["benchmark_weight"] * asset["benchmark_return"] 
                             for asset in asset_allocation.values())
        
        for asset, data in asset_allocation.items():
            # 配置效应 = (组合权重 - 基准权重) * 基准收益
            allocation_effect = (data["weight"] - data["benchmark_weight"]) * data["benchmark_return"]
            
            # 选择效应 = 基准权重 * (组合收益 - 基准收益)
            selection_effect = data["benchmark_weight"] * (data["return"] - data["benchmark_return"])
            
            # 交互效应 = (组合权重 - 基准权重) * (组合收益 - 基准收益)
            interaction_effect = (data["weight"] - data["benchmark_weight"]) * (data["return"] - data["benchmark_return"])
            
            allocation_effects[asset] = round(allocation_effect, 4)
            selection_effects[asset] = round(selection_effect, 4)
            interaction_effects[asset] = round(interaction_effect, 4)
        
        # 总体归因
        total_allocation = sum(allocation_effects.values())
        total_selection = sum(selection_effects.values())
        total_interaction = sum(interaction_effects.values())
        
        portfolio_return = sum(data["weight"] * data["return"] for data in asset_allocation.values())
        excess_return = portfolio_return - benchmark_return
        
        attribution_result = {
            "period": {
                "start_date": request.start_date.isoformat(),
                "end_date": request.end_date.isoformat(),
                "days": analysis_days
            },
            "returns": {
                "portfolio_return": round(portfolio_return, 4),
                "benchmark_return": round(benchmark_return, 4),
                "excess_return": round(excess_return, 4)
            },
            "attribution_breakdown": {
                "asset_allocation": allocation_effects,
                "security_selection": selection_effects,
                "interaction": interaction_effects
            },
            "attribution_summary": {
                "total_allocation_effect": round(total_allocation, 4),
                "total_selection_effect": round(total_selection, 4),
                "total_interaction_effect": round(total_interaction, 4),
                "explained_excess_return": round(total_allocation + total_selection + total_interaction, 4)
            },
            "asset_weights": {
                asset: {"portfolio": data["weight"], "benchmark": data["benchmark_weight"]}
                for asset, data in asset_allocation.items()
            },
            "performance_contribution": {
                asset: round(data["weight"] * data["return"], 4)
                for asset, data in asset_allocation.items()
            }
        }
        
        return {
            "success": True,
            "data": attribution_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"归因分析失败: {str(e)}")

@router.post("/comparison")
async def compare_portfolios(
    request: ComparisonRequest,
):
    """投资组合对比分析"""
    try:
        analysis_days = (request.end_date - request.start_date).days
        comparison_results = {}
        
        # 为每个投资组合生成分析
        for i, portfolio_id in enumerate(request.portfolios):
            # 使用不同的随机种子为每个组合生成不同的数据
            np.random.seed(hash(portfolio_id) % 1000)
            
            portfolio_returns = generate_mock_returns(analysis_days, 0.10 + i * 0.05, 0.15 + i * 0.05)
            metrics = calculate_performance_metrics(portfolio_returns)
            
            # 计算累计净值
            cumulative_returns = (1 + np.array(portfolio_returns)).cumprod()
            
            comparison_results[portfolio_id] = {
                "name": portfolio_data_db.get(portfolio_id, {}).get("name", f"组合{i+1}"),
                "metrics": metrics,
                "final_value": round(cumulative_returns[-1], 6),
                "monthly_returns": [
                    round(ret, 4) for ret in portfolio_returns[::21]  # 每21天取一个点作为月收益
                ][:12]  # 最多12个月
            }
        
        # 基准对比（如果提供）
        benchmark_data = None
        if request.benchmark:
            np.random.seed(42)  # 基准使用固定种子
            benchmark_returns = generate_mock_returns(analysis_days, 0.12, 0.18)
            benchmark_metrics = calculate_performance_metrics(benchmark_returns)
            benchmark_cumulative = (1 + np.array(benchmark_returns)).cumprod()
            
            benchmark_data = {
                "name": f"{request.benchmark} 基准",
                "metrics": benchmark_metrics,
                "final_value": round(benchmark_cumulative[-1], 6),
                "monthly_returns": [round(ret, 4) for ret in benchmark_returns[::21]][:12]
            }
        
        # 排名分析
        portfolios_by_return = sorted(
            comparison_results.items(), 
            key=lambda x: x[1]["metrics"]["total_return"], 
            reverse=True
        )
        
        portfolios_by_sharpe = sorted(
            comparison_results.items(),
            key=lambda x: x[1]["metrics"]["sharpe_ratio"],
            reverse=True
        )
        
        # 相关性分析
        portfolio_ids = list(comparison_results.keys())
        correlation_matrix = {}
        for i, id1 in enumerate(portfolio_ids):
            correlation_matrix[id1] = {}
            for j, id2 in enumerate(portfolio_ids):
                if i == j:
                    corr = 1.0
                else:
                    np.random.seed(hash(id1 + id2) % 1000)
                    corr = round(np.random.uniform(0.3, 0.9), 3)
                correlation_matrix[id1][id2] = corr
        
        comparison_result = {
            "analysis_period": {
                "start_date": request.start_date.isoformat(),
                "end_date": request.end_date.isoformat(),
                "days": analysis_days
            },
            "portfolios": comparison_results,
            "benchmark": benchmark_data,
            "rankings": {
                "by_total_return": [
                    {"portfolio_id": pid, "name": data["name"], "value": data["metrics"]["total_return"]}
                    for pid, data in portfolios_by_return
                ],
                "by_sharpe_ratio": [
                    {"portfolio_id": pid, "name": data["name"], "value": data["metrics"]["sharpe_ratio"]}
                    for pid, data in portfolios_by_sharpe
                ]
            },
            "correlation_matrix": correlation_matrix,
            "summary_stats": {
                "best_performer": portfolios_by_return[0][0] if portfolios_by_return else None,
                "best_risk_adjusted": portfolios_by_sharpe[0][0] if portfolios_by_sharpe else None,
                "avg_return": round(np.mean([data["metrics"]["total_return"] for data in comparison_results.values()]), 4),
                "avg_volatility": round(np.mean([data["metrics"]["volatility"] for data in comparison_results.values()]), 4)
            }
        }
        
        return {
            "success": True,
            "data": comparison_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"对比分析失败: {str(e)}")

@router.get("/reports/{report_type}")
async def generate_performance_report(
    report_type: ReportType,
    portfolio_id: Optional[str] = Query(None, description="投资组合ID"),
    start_date: Optional[datetime] = Query(None, description="报告开始日期"),
    end_date: Optional[datetime] = Query(None, description="报告结束日期"),
):
    """生成性能报告"""
    try:
        # 设置默认日期范围
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        analysis_days = (end_date - start_date).days
        
        # 生成模拟数据
        returns = generate_mock_returns(analysis_days)
        metrics = calculate_performance_metrics(returns)
        
        if report_type == ReportType.SUMMARY:
            report_data = {
                "report_type": "summary",
                "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                "key_metrics": {
                    "total_return": metrics["total_return"],
                    "annual_return": metrics["annual_return"],
                    "volatility": metrics["volatility"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "max_drawdown": metrics["max_drawdown"]
                },
                "performance_summary": f"在{analysis_days}天的分析期间内，投资组合实现了{metrics['total_return']*100:.2f}%的总回报，年化收益率为{metrics['annual_return']*100:.2f}%，夏普比率为{metrics['sharpe_ratio']:.2f}。"
            }
        
        elif report_type == ReportType.DETAILED:
            cumulative_returns = (1 + np.array(returns)).cumprod()
            dates = [start_date + timedelta(days=i) for i in range(analysis_days)]
            
            report_data = {
                "report_type": "detailed",
                "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                "metrics": metrics,
                "daily_returns": [
                    {"date": date.isoformat(), "return": round(ret, 6), "cumulative": round(cum, 6)}
                    for date, ret, cum in zip(dates[-30:], returns[-30:], cumulative_returns[-30:])
                ],
                "monthly_breakdown": [
                    {"month": f"2024-{i:02d}", "return": round(np.random.uniform(-0.05, 0.15), 4)}
                    for i in range(1, 13)
                ]
            }
        
        elif report_type == ReportType.RISK_ANALYSIS:
            var_95 = np.percentile(returns, 5)
            report_data = {
                "report_type": "risk_analysis",
                "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
                "risk_metrics": {
                    "volatility": metrics["volatility"],
                    "var_95": round(var_95, 4),
                    "max_drawdown": metrics["max_drawdown"],
                    "skewness": metrics["skewness"],
                    "kurtosis": metrics["kurtosis"]
                },
                "risk_assessment": "中等风险" if abs(var_95) < 0.03 else "高风险"
            }
        
        return {
            "success": True,
            "data": report_data,
            "generated_at": datetime.utcnow().isoformat(),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成报告失败: {str(e)}")

@router.get("/portfolios")
async def list_portfolios():
    """获取投资组合列表"""
    try:
        portfolios = []
        for portfolio_id, data in portfolio_data_db.items():
            # 生成简单的性能指标
            returns = generate_mock_returns(30, np.random.uniform(0.08, 0.20), np.random.uniform(0.15, 0.3))
            recent_metrics = calculate_performance_metrics(returns)
            
            portfolios.append({
                "id": portfolio_id,
                "name": data["name"],
                "inception_date": data["inception_date"].isoformat(),
                "benchmark": data["benchmark"],
                "recent_performance": {
                    "30d_return": recent_metrics["total_return"],
                    "volatility": recent_metrics["volatility"],
                    "sharpe_ratio": recent_metrics["sharpe_ratio"]
                },
                "total_return": data["total_return"]
            })
        
        return {
            "success": True,
            "data": portfolios,
            "total": len(portfolios),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取投资组合列表失败: {str(e)}")

@router.get("/benchmarks")
async def list_benchmarks():
    """获取可用基准列表"""
    benchmarks = [
        {"id": "BTC", "name": "Bitcoin", "description": "比特币基准"},
        {"id": "ETH", "name": "Ethereum", "description": "以太坊基准"},
        {"id": "CRYPTO_INDEX", "name": "加密货币指数", "description": "综合加密货币指数"},
        {"id": "DEFI_INDEX", "name": "DeFi指数", "description": "去中心化金融指数"}
    ]
    
    return {
        "success": True,
        "data": benchmarks,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/schedule-report")
async def schedule_performance_report(
    portfolio_id: str,
    report_type: ReportType,
    schedule_frequency: str = Query(..., description="报告频率: daily, weekly, monthly"),
    email_recipients: List[str] = [],
    background_tasks: BackgroundTasks = None,
):
    """定期性能报告计划"""
    try:
        report_schedule = {
            "id": f"schedule_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "portfolio_id": portfolio_id,
            "report_type": report_type,
            "frequency": schedule_frequency,
            "recipients": email_recipients,
            "created_by": "user",
            "created_at": datetime.utcnow(),
            "status": "active",
            "next_run": datetime.utcnow() + timedelta(days=1)  # 简化调度
        }
        
        return {
            "success": True,
            "message": "定期报告已设置",
            "data": report_schedule,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"设置定期报告失败: {str(e)}")

@router.get("/metrics/definitions")
async def get_metrics_definitions():
    """获取性能指标定义"""
    metrics_definitions = {
        "total_return": {
            "name": "总收益率",
            "description": "投资期间的总回报率",
            "formula": "(期末价值 - 期初价值) / 期初价值",
            "unit": "百分比"
        },
        "annual_return": {
            "name": "年化收益率", 
            "description": "年化后的收益率",
            "formula": "(1 + 总收益率) ^ (252 / 交易天数) - 1",
            "unit": "百分比"
        },
        "volatility": {
            "name": "波动率",
            "description": "收益率的标准差，衡量风险",
            "formula": "日收益率标准差 * √252",
            "unit": "百分比"
        },
        "sharpe_ratio": {
            "name": "夏普比率",
            "description": "风险调整后的收益指标",
            "formula": "年化收益率 / 年化波动率",
            "unit": "比值"
        },
        "max_drawdown": {
            "name": "最大回撤",
            "description": "从峰值到谷值的最大下跌幅度",
            "formula": "min((当前价值 - 峰值) / 峰值)",
            "unit": "百分比"
        },
        "information_ratio": {
            "name": "信息比率",
            "description": "相对基准的超额收益与跟踪误差的比值",
            "formula": "超额收益 / 跟踪误差",
            "unit": "比值"
        },
        "calmar_ratio": {
            "name": "Calmar比率",
            "description": "年化收益率与最大回撤的比值",
            "formula": "年化收益率 / |最大回撤|",
            "unit": "比值"
        }
    }
    
    return {
        "success": True,
        "data": metrics_definitions,
        "timestamp": datetime.utcnow().isoformat()
    }