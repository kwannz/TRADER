"""
AI引擎核心模块
统一管理DeepSeek和Gemini AI服务
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from loguru import logger

from config.settings import settings
from services.ai_clients.deepseek_client import deepseek_client
from services.ai_clients.gemini_client import gemini_client
from .data_manager import data_manager

class AIEngine:
    """AI引擎主类 - 统一AI服务接口"""
    
    def __init__(self):
        self.deepseek = deepseek_client
        self.gemini = gemini_client
        self.data_manager = data_manager
        self._initialized = False
        self._analysis_cache = {}
        self._cache_ttl = 1800  # 缓存30分钟
        
    async def initialize(self):
        """初始化AI引擎"""
        try:
            logger.info("正在初始化AI引擎...")
            
            # 测试AI服务连通性
            await self._test_ai_services()
            
            self._initialized = True
            logger.info("AI引擎初始化完成")
            
        except Exception as e:
            logger.error(f"AI引擎初始化失败: {e}")
            raise
    
    async def _test_ai_services(self):
        """测试AI服务连通性"""
        try:
            # 测试DeepSeek
            test_news = [{"title": "测试新闻", "content": "测试内容"}]
            deepseek_result = await self.deepseek.analyze_sentiment(test_news)
            if deepseek_result:
                logger.info("DeepSeek服务连接正常")
            else:
                logger.warning("DeepSeek服务连接异常")
            
            # 测试Gemini  
            test_requirements = {"strategy_type": "test", "symbols": ["BTC-USDT"]}
            gemini_result = await self.gemini.generate_strategy(test_requirements)
            if gemini_result:
                logger.info("Gemini服务连接正常")
            else:
                logger.warning("Gemini服务连接异常")
                
        except Exception as e:
            logger.warning(f"AI服务连通性测试失败: {e}")
    
    # ============ 情绪分析服务 ============
    async def analyze_market_sentiment(self, force_refresh: bool = False) -> Dict[str, Any]:
        """分析市场情绪（综合新闻和社交媒体）"""
        try:
            cache_key = "market_sentiment"
            
            # 检查缓存
            if not force_refresh and cache_key in self._analysis_cache:
                cached_data = self._analysis_cache[cache_key]
                if datetime.fromisoformat(cached_data["timestamp"]) > datetime.utcnow() - timedelta(seconds=self._cache_ttl):
                    logger.debug("使用缓存的市场情绪分析")
                    return cached_data
            
            # 获取最近新闻
            recent_news = await self.data_manager.get_recent_news(hours=6, limit=20)
            
            if not recent_news:
                logger.warning("没有获取到新闻数据，使用默认情绪分析")
                return self._get_default_sentiment()
            
            # 使用DeepSeek分析情绪
            sentiment_result = await self.deepseek.analyze_sentiment(recent_news)
            
            # 缓存结果
            self._analysis_cache[cache_key] = sentiment_result
            
            return sentiment_result
            
        except Exception as e:
            logger.error(f"市场情绪分析失败: {e}")
            return self._get_default_sentiment()
    
    # ============ 市场预测服务 ============
    async def predict_market_movement(self, symbols: List[str], 
                                    timeframe: str = "1h") -> Dict[str, Any]:
        """预测市场走势"""
        try:
            cache_key = f"market_prediction_{'-'.join(symbols)}_{timeframe}"
            
            # 检查缓存
            if cache_key in self._analysis_cache:
                cached_data = self._analysis_cache[cache_key]
                if datetime.fromisoformat(cached_data["timestamp"]) > datetime.utcnow() - timedelta(seconds=900):  # 15分钟缓存
                    return cached_data
            
            # 收集市场数据
            market_data = {}
            technical_indicators = {}
            
            for symbol in symbols:
                # 从缓存获取行情数据
                ticker_data = await self.data_manager.cache_manager.get_market_data(f"OKX:{symbol}")
                if ticker_data:
                    market_data[symbol] = ticker_data
                
                # 获取技术指标（简化版）
                historical_data = await self.data_manager.time_series_manager.get_kline_data(
                    symbol, timeframe, limit=100
                )
                
                if not historical_data.empty:
                    # 计算简单技术指标
                    technical_indicators[symbol] = self._calculate_simple_indicators(historical_data)
            
            # 使用DeepSeek预测
            prediction_result = await self.deepseek.predict_market_trend(market_data, technical_indicators)
            
            # 缓存结果
            self._analysis_cache[cache_key] = prediction_result
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"市场预测失败: {e}")
            return self._get_default_prediction()
    
    # ============ 策略生成服务 ============
    async def generate_trading_strategy(self, requirements: Dict) -> Dict[str, Any]:
        """生成交易策略"""
        try:
            # 添加系统约束
            enhanced_requirements = {
                **requirements,
                "max_capital": min(requirements.get("max_capital", 500), settings.exchange.initial_balance * 0.8),
                "risk_level": requirements.get("risk_level", "medium"),
                "hard_stop_loss": settings.exchange.hard_stop_loss
            }
            
            # 使用Gemini生成策略
            strategy_result = await self.gemini.generate_strategy(enhanced_requirements)
            
            # 保存策略到数据库
            if strategy_result and "code" in strategy_result:
                try:
                    strategy_id = await self.data_manager.save_strategy({
                        "name": strategy_result["strategy_name"],
                        "type": strategy_result["strategy_type"],
                        "code": strategy_result["code"],
                        "description": strategy_result["description"],
                        "parameters": strategy_result.get("parameters", {}),
                        "status": "draft",
                        "generated_by": "AI",
                        "ai_source": "gemini"
                    })
                    strategy_result["strategy_id"] = strategy_id
                    logger.info(f"AI策略已保存: {strategy_id}")
                except Exception as e:
                    logger.warning(f"保存AI策略失败: {e}")
            
            return strategy_result
            
        except Exception as e:
            logger.error(f"策略生成失败: {e}")
            return self._get_default_strategy()
    
    async def optimize_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """优化现有策略"""
        try:
            # 获取策略信息
            strategies = await self.data_manager.get_strategies()
            target_strategy = None
            
            for strategy in strategies:
                if strategy["_id"] == strategy_id:
                    target_strategy = strategy
                    break
            
            if not target_strategy:
                raise ValueError(f"未找到策略: {strategy_id}")
            
            # 获取策略表现数据
            trades = await self.data_manager.get_trades(strategy_id=strategy_id, limit=100)
            performance_data = self._calculate_strategy_performance(trades)
            
            # 使用Gemini优化策略
            optimization_result = await self.gemini.optimize_strategy(
                target_strategy["code"], 
                performance_data
            )
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"策略优化失败: {e}")
            return {"error": str(e)}
    
    # ============ 风险评估服务 ============
    async def assess_portfolio_risk(self, portfolio_data: Dict) -> Dict[str, Any]:
        """评估投资组合风险"""
        try:
            # 获取市场条件数据
            market_conditions = {
                "volatility": await self._calculate_market_volatility(),
                "liquidity": await self._assess_market_liquidity(),
                "sentiment_score": (await self.analyze_market_sentiment()).get("sentiment_score", 0.0)
            }
            
            # 使用DeepSeek评估风险
            risk_result = await self.deepseek.assess_risk(portfolio_data, market_conditions)
            
            # 添加系统风险检查
            risk_result = await self._enhance_risk_assessment(risk_result, portfolio_data)
            
            return risk_result
            
        except Exception as e:
            logger.error(f"风险评估失败: {e}")
            return self._get_default_risk_assessment()
    
    # ============ 因子发现服务 ============
    async def discover_alpha_factors(self, symbols: List[str]) -> Dict[str, Any]:
        """发现Alpha因子"""
        try:
            # 收集历史价格数据
            market_data = {}
            price_history = {"prices": [], "volumes": [], "time_range": "180天"}
            
            for symbol in symbols:
                historical_data = await self.data_manager.time_series_manager.get_kline_data(
                    symbol, "1h", limit=4320  # 180天 * 24小时
                )
                
                if not historical_data.empty:
                    price_history["prices"].extend(historical_data["close"].tolist())
                    price_history["volumes"].extend(historical_data["volume"].tolist())
            
            # 计算相关性矩阵
            if len(symbols) > 1:
                correlation_data = await self._calculate_correlation_matrix(symbols)
                market_data["correlation_matrix"] = correlation_data
            
            # 使用DeepSeek发现因子
            factor_result = await self.deepseek.discover_factors(market_data, price_history)
            
            # 保存发现的因子
            if factor_result and "discovered_factors" in factor_result:
                for factor in factor_result["discovered_factors"]:
                    try:
                        await self.data_manager.save_factor({
                            "name": factor["name"],
                            "formula": factor["formula"],
                            "description": factor["description"],
                            "expected_ic": factor.get("expected_ic", "未知"),
                            "status": "testing",
                            "created_by": "AI",
                            "ai_source": "deepseek"
                        })
                    except Exception as e:
                        logger.warning(f"保存因子失败 {factor['name']}: {e}")
            
            return factor_result
            
        except Exception as e:
            logger.error(f"因子发现失败: {e}")
            return self._get_default_factor_discovery()
    
    # ============ 智能助手服务 ============
    async def chat_with_assistant(self, user_message: str, 
                                 context: Optional[Dict] = None) -> Dict[str, Any]:
        """与AI助手对话"""
        try:
            if context is None:
                context = {}
                
            # 获取当前系统状态
            system_context = await self._get_system_context()
            context.update(system_context)
            
            # 使用Gemini助手
            chat_result = await self.gemini.chat_assistant(user_message, context)
            
            return chat_result
            
        except Exception as e:
            logger.error(f"AI助手对话失败: {e}")
            return {
                "response": "抱歉，我现在无法回答您的问题。",
                "suggestions": ["请稍后再试"],
                "risk_warning": "请谨慎进行交易操作",
                "follow_up_questions": [],
                "confidence": "低",
                "timestamp": datetime.utcnow().isoformat(),
                "source": "ai_engine_fallback"
            }
    
    # ============ 多模态分析服务 ============
    async def analyze_chart_pattern(self, chart_data: str, 
                                   symbol: str) -> Dict[str, Any]:
        """分析图表模式"""
        try:
            # 获取市场背景信息
            market_context = {}
            
            # 从缓存获取当前市场数据
            ticker_data = await self.data_manager.cache_manager.get_market_data(f"OKX:{symbol}")
            if ticker_data:
                market_context = {
                    "current_price": ticker_data.get("price"),
                    "price_change_24h": ticker_data.get("change_24h_pct"),
                    "volume_24h": ticker_data.get("volume_24h"),
                }
            
            # 获取技术指标
            historical_data = await self.data_manager.time_series_manager.get_kline_data(
                symbol, "1h", limit=100
            )
            
            if not historical_data.empty:
                market_context["technical_indicators"] = self._calculate_simple_indicators(historical_data)
            
            # 获取市场情绪
            sentiment = await self.analyze_market_sentiment()
            market_context["news_sentiment"] = sentiment.get("sentiment_score", 0.0)
            
            # 使用Gemini分析
            analysis_result = await self.gemini.analyze_market_multimodal(chart_data, market_context)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"图表模式分析失败: {e}")
            return self._get_default_chart_analysis()
    
    # ============ 辅助方法 ============
    def _calculate_simple_indicators(self, df) -> Dict[str, Any]:
        """计算简单技术指标"""
        try:
            if df.empty:
                return {}
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # 移动平均
            ma_20 = df['close'].rolling(window=20).mean()
            ma_50 = df['close'].rolling(window=50).mean()
            
            return {
                "rsi": float(rsi.iloc[-1]) if not rsi.empty else 50.0,
                "ma_20": float(ma_20.iloc[-1]) if not ma_20.empty else 0.0,
                "ma_50": float(ma_50.iloc[-1]) if not ma_50.empty else 0.0,
                "price": float(df['close'].iloc[-1]) if not df.empty else 0.0
            }
        except Exception as e:
            logger.warning(f"技术指标计算失败: {e}")
            return {}
    
    def _calculate_strategy_performance(self, trades: List[Dict]) -> Dict[str, Any]:
        """计算策略表现"""
        try:
            if not trades:
                return {}
            
            # 简单性能计算
            total_pnl = sum(float(trade.get("pnl", 0)) for trade in trades)
            win_trades = sum(1 for trade in trades if float(trade.get("pnl", 0)) > 0)
            total_trades = len(trades)
            
            return {
                "return_rate": round(total_pnl * 100 / settings.exchange.initial_balance, 2),
                "win_rate": round(win_trades * 100 / total_trades, 2) if total_trades > 0 else 0,
                "max_drawdown": 10.0,  # 简化
                "sharpe_ratio": 1.0,   # 简化
                "trade_count": total_trades,
                "issues": "需要更多历史数据"
            }
        except Exception as e:
            logger.warning(f"策略性能计算失败: {e}")
            return {}
    
    async def _calculate_market_volatility(self) -> float:
        """计算市场波动率"""
        try:
            # 简化的波动率计算
            return 0.25  # 25%年化波动率
        except:
            return 0.25
    
    async def _assess_market_liquidity(self) -> str:
        """评估市场流动性"""
        try:
            return "正常"
        except:
            return "未知"
    
    async def _calculate_correlation_matrix(self, symbols: List[str]) -> Dict:
        """计算相关性矩阵"""
        try:
            # 简化实现
            return {"correlation": "中等相关"}
        except:
            return {}
    
    async def _enhance_risk_assessment(self, risk_result: Dict, 
                                     portfolio_data: Dict) -> Dict[str, Any]:
        """增强风险评估"""
        try:
            # 系统级风险检查
            total_value = portfolio_data.get("total_value", 0)
            if total_value <= settings.exchange.hard_stop_loss:
                risk_result["urgent_action_needed"] = True
                risk_result["risk_level"] = "extreme"
                risk_result["main_risks"].append("接近硬止损线")
            
            return risk_result
        except:
            return risk_result
    
    async def _get_system_context(self) -> Dict[str, Any]:
        """获取系统上下文信息"""
        try:
            strategies = await self.data_manager.get_strategies(status="active")
            recent_trades = await self.data_manager.get_trades(limit=10)
            
            return {
                "account_balance": 500.0,  # 简化
                "active_strategies": len(strategies),
                "daily_pnl": sum(float(trade.get("pnl", 0)) for trade in recent_trades),
                "market_status": "正常"
            }
        except:
            return {}
    
    # 默认返回值
    def _get_default_sentiment(self) -> Dict[str, Any]:
        return {
            "sentiment_score": 0.0,
            "confidence": 0.5,
            "key_factors": ["数据不足"],
            "market_impact": "中性",
            "recommendation": "观望",
            "timestamp": datetime.utcnow().isoformat(),
            "source": "ai_engine_fallback"
        }
    
    def _get_default_prediction(self) -> Dict[str, Any]:
        return {
            "trend_direction": "sideways",
            "confidence": 0.5,
            "time_horizon": "1-4小时",
            "key_levels": {"support": 0, "resistance": 0},
            "risk_level": "medium",
            "reasoning": "数据不足",
            "action_suggestion": "观望",
            "timestamp": datetime.utcnow().isoformat(),
            "source": "ai_engine_fallback"
        }
    
    def _get_default_strategy(self) -> Dict[str, Any]:
        return {
            "strategy_name": "默认观望策略",
            "strategy_type": "保守",
            "code": "# 暂无策略代码",
            "description": "默认策略",
            "timestamp": datetime.utcnow().isoformat(),
            "source": "ai_engine_fallback"
        }
    
    def _get_default_risk_assessment(self) -> Dict[str, Any]:
        return {
            "risk_score": 50,
            "risk_level": "medium",
            "main_risks": ["市场不确定性"],
            "urgent_action_needed": False,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "ai_engine_fallback"
        }
    
    def _get_default_factor_discovery(self) -> Dict[str, Any]:
        return {
            "discovered_factors": [],
            "factor_combinations": [],
            "research_direction": "需要更多数据",
            "confidence": 0.3,
            "timestamp": datetime.utcnow().isoformat(),
            "source": "ai_engine_fallback"
        }
    
    def _get_default_chart_analysis(self) -> Dict[str, Any]:
        return {
            "technical_analysis": "数据不足",
            "pattern_recognition": "无明显模式",
            "support_resistance": {"support": "未知", "resistance": "未知"},
            "trend_analysis": "横盘",
            "volume_analysis": "正常",
            "trading_signals": ["观望"],
            "confidence_level": "低",
            "risk_warning": "请谨慎操作",
            "timestamp": datetime.utcnow().isoformat(),
            "source": "ai_engine_fallback"
        }
    
    async def close(self):
        """关闭AI引擎"""
        try:
            await self.deepseek.close()
            await self.gemini.close()
            logger.info("AI引擎已关闭")
        except Exception as e:
            logger.error(f"关闭AI引擎失败: {e}")

# 全局AI引擎实例
ai_engine = AIEngine()