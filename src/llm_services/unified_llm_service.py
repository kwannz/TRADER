"""
Unified LLM Service
统一LLM服务 - 融合AI因子发现、PandaFactor专业助手和CTBench智能分析
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from datetime import datetime
import json
import pandas as pd

# LLM相关导入
try:
    import openai
    from openai import AsyncOpenAI
except ImportError:
    openai = None
    AsyncOpenAI = None


class FactorDevelopmentAssistant:
    """
    因子开发助手 - 基于PandaFactor的专业因子开发LLM助手
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("FactorDevelopmentAssistant")
        
        # 初始化OpenAI客户端
        if AsyncOpenAI:
            self.client = AsyncOpenAI(
                api_key=config.get("LLM_API_KEY", ""),
                base_url=config.get("LLM_BASE_URL", "")
            )
        else:
            self.client = None
            self.logger.warning("OpenAI not available, LLM features disabled")
        
        self.model = config.get("LLM_MODEL", "gpt-3.5-turbo")
        
        # 专业因子开发系统提示词
        self.system_prompt = """You are PandaAI Factor Development Assistant, a specialized AI designed to help with quantitative factor development and optimization.

I will ONLY answer questions related to factor development, coding, and optimization. If asked about unrelated topics, I will politely remind users that I'm specialized in factor development.

I WILL ALWAYS RESPOND IN CHINESE regardless of the input language.

I can assist with:
- Writing and optimizing factor code in both formula and Python modes
- Explaining built-in functions for factor development
- Providing examples of factor implementations
- Debugging factor code
- Suggesting improvements to factor logic

My knowledge includes these factor types and functions:

1. Basic Factors:
   - Price factors: CLOSE, OPEN, HIGH, LOW
   - Volume factors: VOLUME, AMOUNT, TURNOVER
   - Market cap factors: MARKET_CAP

2. Factor Development Methods:
   - Formula Mode: Mathematical expressions with built-in functions
   - Python Mode: Custom factor classes implementing the calculate method

3. Built-in Function Libraries with Parameters:
   - Basic calculation:
     * RANK(series) - Cross-sectional ranking, normalized to [-0.5, 0.5]
     * RETURNS(close, period=1) - Calculate returns
     * STDDEV(series, window=20) - Calculate rolling standard deviation
     * CORRELATION(series1, series2, window=20) - Calculate rolling correlation
     * IF(condition, true_value, false_value) - Conditional selection
     * MIN(series1, series2) - Take minimum values
     * MAX(series1, series2) - Take maximum values
     * ABS(series) - Calculate absolute values
     * LOG(series) - Calculate natural logarithm
     * POWER(series, power) - Calculate power

   - Time series:
     * DELAY(series, period=1) - Series delay, returns value from N periods ago
     * SUM(series, window=20) - Calculate moving sum
     * TS_MEAN(series, window=20) - Calculate moving average
     * TS_MIN(series, window=20) - Calculate moving minimum
     * TS_MAX(series, window=20) - Calculate moving maximum
     * TS_RANK(series, window=20) - Calculate time series ranking
     * MA(series, window) - Simple moving average
     * EMA(series, window) - Exponential moving average
     * SMA(series, window, M=1) - Smoothed moving average
     * WMA(series, window) - Weighted moving average

   - Technical indicators:
     * MACD(close, SHORT=12, LONG=26, M=9) - Calculate MACD
     * KDJ(close, high, low, N=9, M1=3, M2=3) - Calculate KDJ
     * RSI(close, N=24) - Calculate Relative Strength Index
     * BOLL(close, N=20, P=2) - Calculate Bollinger Bands
     * CCI(close, high, low, N=14) - Calculate Commodity Channel Index
     * ATR(close, high, low, N=20) - Calculate Average True Range

   - Core utilities:
     * RD(S, D=3) - Round to D decimal places
     * REF(S, N=1) - Shift entire series down by N
     * DIFF(S, N=1) - Calculate difference between values
     * CROSS(S1, S2) - Check for upward cross
     * FILTER(S, N) - Filter signals, only keep first signal in N periods

4. Examples:

   - Formula Mode Examples:
     * Simple momentum: "RANK((CLOSE / DELAY(CLOSE, 20)) - 1)"
     * Volume-price correlation: "CORRELATION(CLOSE, VOLUME, 20)"
     * Complex example: "RANK((CLOSE / DELAY(CLOSE, 20)) - 1) * STDDEV((CLOSE / DELAY(CLOSE, 1)) - 1, 20) * IF(CLOSE > DELAY(CLOSE, 1), 1, -1)"

   - Python Mode Examples:
     * Basic momentum factor:
```python
class MomentumFactor(BaseFactor):
    def calculate(self, factors):
        close = factors['close']
        # Calculate 20-day returns
        returns = (close / DELAY(close, 20)) - 1
        return RANK(returns)
```

     * Complex multi-signal factor:
```python
class ComplexFactor(BaseFactor):
    def calculate(self, factors):
        close = factors['close']
        volume = factors['volume']
        
        # Calculate returns
        returns = (close / DELAY(close, 20)) - 1
        # Calculate volatility
        volatility = STDDEV((close / DELAY(close, 1)) - 1, 20)
        # Calculate volume ratio
        volume_ratio = volume / DELAY(volume, 1)
        # Calculate momentum signal
        momentum = RANK(returns)
        # Calculate volatility signal
        vol_signal = IF(volatility > DELAY(volatility, 1), 1, -1)
        # Combine signals
        result = momentum * vol_signal * (volume_ratio / SUM(volume_ratio, 10))
        return result
```

IMPORTANT: I will not reference functions that don't exist in the system. I will avoid using future data, as the competition rules require out-of-sample running, calculating factor values daily, and placing orders the next day to calculate returns.

For all questions unrelated to factor development, I will politely remind users that I can only help with factor development topics."""

    async def chat(self, messages: List[Dict[str, str]], stream: bool = False) -> Union[str, AsyncGenerator[str, None]]:
        """
        与因子开发助手对话
        
        Args:
            messages: 对话消息列表
            stream: 是否使用流式响应
            
        Returns:
            响应内容或流式生成器
        """
        if not self.client:
            return "LLM服务不可用，请检查配置"
        
        try:
            # 准备消息
            formatted_messages = [{"role": "system", "content": self.system_prompt}]
            formatted_messages.extend(messages)
            
            if stream:
                return self._stream_chat(formatted_messages)
            else:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=formatted_messages,
                    temperature=0.7,
                    max_tokens=2000
                )
                return response.choices[0].message.content
                
        except Exception as e:
            self.logger.error(f"Factor development chat error: {str(e)}")
            return f"对话出错: {str(e)}"
    
    async def _stream_chat(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """流式对话响应"""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"流式对话出错: {str(e)}"

    async def generate_factor_formula(self, requirements: str) -> Dict[str, Any]:
        """
        根据需求生成因子公式
        
        Args:
            requirements: 因子需求描述
            
        Returns:
            包含因子公式和说明的字典
        """
        prompt = f"""请根据以下需求生成一个因子公式：

需求描述：{requirements}

请提供：
1. 公式模式的因子表达式
2. 因子的详细解释
3. 参数说明
4. 适用场景

请用JSON格式回复，包含formula、explanation、parameters、scenarios四个字段。"""

        try:
            response = await self.chat([{"role": "user", "content": prompt}])
            # 尝试解析JSON响应
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {
                    "formula": "RANK((CLOSE / DELAY(CLOSE, 20)) - 1)",
                    "explanation": response,
                    "parameters": {"period": 20},
                    "scenarios": "动量策略"
                }
        except Exception as e:
            self.logger.error(f"Generate factor formula error: {str(e)}")
            return {
                "error": f"生成因子公式失败: {str(e)}"
            }

    async def optimize_factor(self, factor_code: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        优化现有因子
        
        Args:
            factor_code: 当前因子代码
            performance_data: 因子性能数据
            
        Returns:
            优化建议和新因子代码
        """
        prompt = f"""请帮我优化以下因子：

当前因子代码：
{factor_code}

性能数据：
{json.dumps(performance_data, indent=2, ensure_ascii=False)}

请分析当前因子的问题并提供优化建议，包括：
1. 问题分析
2. 优化后的因子公式
3. 改进说明
4. 预期效果

请用JSON格式回复。"""

        try:
            response = await self.chat([{"role": "user", "content": prompt}])
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {
                    "analysis": response,
                    "optimized_formula": factor_code,
                    "improvements": "无法解析优化建议",
                    "expected_effect": "未知"
                }
        except Exception as e:
            self.logger.error(f"Optimize factor error: {str(e)}")
            return {
                "error": f"因子优化失败: {str(e)}"
            }

    async def debug_factor_code(self, code: str, error_message: str) -> str:
        """
        调试因子代码
        
        Args:
            code: 出错的因子代码
            error_message: 错误消息
            
        Returns:
            调试建议
        """
        prompt = f"""请帮我调试以下因子代码：

代码：
{code}

错误信息：
{error_message}

请分析错误原因并提供修复建议。"""

        try:
            return await self.chat([{"role": "user", "content": prompt}])
        except Exception as e:
            return f"调试失败: {str(e)}"


class AIFactorDiscovery:
    """
    AI因子发现服务 - 智能发现和生成新因子
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("AIFactorDiscovery")
        
        if AsyncOpenAI:
            self.client = AsyncOpenAI(
                api_key=config.get("LLM_API_KEY", ""),
                base_url=config.get("LLM_BASE_URL", "")
            )
        else:
            self.client = None
        
        self.model = config.get("LLM_MODEL", "gpt-3.5-turbo")
    
    async def discover_factors_from_market_data(self, market_data: Dict[str, pd.DataFrame],
                                              target_returns: pd.Series) -> List[Dict[str, Any]]:
        """
        基于市场数据发现潜在因子
        
        Args:
            market_data: 市场数据
            target_returns: 目标收益率
            
        Returns:
            发现的因子列表
        """
        # 分析市场数据特征
        data_summary = self._analyze_market_data(market_data, target_returns)
        
        prompt = f"""作为量化因子发现专家，请基于以下市场数据特征发现潜在的因子：

市场数据摘要：
{data_summary}

请发现3-5个有潜力的因子，每个因子包括：
1. 因子公式
2. 理论基础
3. 预期效果
4. 风险提示

请用JSON格式回复，包含factors数组。"""

        try:
            if not self.client:
                return []
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            try:
                result = json.loads(content)
                return result.get("factors", [])
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse AI factor discovery response as JSON")
                return []
                
        except Exception as e:
            self.logger.error(f"AI factor discovery error: {str(e)}")
            return []
    
    def _analyze_market_data(self, market_data: Dict[str, pd.DataFrame],
                           target_returns: pd.Series) -> str:
        """分析市场数据特征"""
        analysis = []
        
        for name, data in market_data.items():
            if isinstance(data, pd.DataFrame):
                analysis.append(f"{name}: {data.shape[0]}行 × {data.shape[1]}列")
                if 'close' in data.columns:
                    returns = data['close'].pct_change()
                    analysis.append(f"  - 平均收益率: {returns.mean():.4f}")
                    analysis.append(f"  - 收益率波动率: {returns.std():.4f}")
        
        # 目标收益率分析
        analysis.append(f"目标收益率 - 均值: {target_returns.mean():.4f}, 标准差: {target_returns.std():.4f}")
        
        return "\n".join(analysis)


class CTBenchAnalysisAssistant:
    """
    CTBench分析助手 - 协助时间序列生成和压力测试分析
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("CTBenchAnalysisAssistant")
        
        if AsyncOpenAI:
            self.client = AsyncOpenAI(
                api_key=config.get("LLM_API_KEY", ""),
                base_url=config.get("LLM_BASE_URL", "")
            )
        else:
            self.client = None
        
        self.model = config.get("LLM_MODEL", "gpt-3.5-turbo")
    
    async def analyze_synthetic_data(self, real_data: pd.DataFrame,
                                   synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析合成数据质量
        
        Args:
            real_data: 真实市场数据
            synthetic_data: 合成数据
            
        Returns:
            分析结果
        """
        # 计算统计特征对比
        real_stats = self._calculate_data_stats(real_data)
        synthetic_stats = self._calculate_data_stats(synthetic_data)
        
        prompt = f"""请分析以下真实数据与合成数据的质量对比：

真实数据统计特征：
{json.dumps(real_stats, indent=2, ensure_ascii=False)}

合成数据统计特征：
{json.dumps(synthetic_stats, indent=2, ensure_ascii=False)}

请提供：
1. 数据质量评估
2. 相似度分析
3. 差异点识别
4. 改进建议

请用JSON格式回复。"""

        try:
            if not self.client:
                return {"error": "LLM服务不可用"}
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"analysis": content}
                
        except Exception as e:
            self.logger.error(f"Synthetic data analysis error: {str(e)}")
            return {"error": f"分析失败: {str(e)}"}
    
    def _calculate_data_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """计算数据统计特征"""
        stats = {}
        
        if 'close' in data.columns:
            close = data['close']
            returns = close.pct_change().dropna()
            
            stats.update({
                "price_stats": {
                    "mean": close.mean(),
                    "std": close.std(),
                    "min": close.min(),
                    "max": close.max()
                },
                "return_stats": {
                    "mean": returns.mean(),
                    "std": returns.std(),
                    "skewness": returns.skew(),
                    "kurtosis": returns.kurtosis()
                }
            })
        
        if 'volume' in data.columns:
            volume = data['volume']
            stats["volume_stats"] = {
                "mean": volume.mean(),
                "std": volume.std(),
                "median": volume.median()
            }
        
        return stats


class UnifiedLLMService:
    """
    统一LLM服务 - 整合所有AI辅助功能
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("UnifiedLLMService")
        
        # 初始化各个助手
        self.factor_assistant = FactorDevelopmentAssistant(self.config)
        self.ai_discovery = AIFactorDiscovery(self.config)
        self.ctbench_assistant = CTBenchAnalysisAssistant(self.config)
        
        self.logger.info("Unified LLM Service initialized")
    
    # ================== 因子开发助手接口 ==================
    
    async def chat_with_factor_assistant(self, message: str, 
                                       conversation_history: List[Dict[str, str]] = None) -> str:
        """与因子开发助手对话"""
        messages = conversation_history or []
        messages.append({"role": "user", "content": message})
        return await self.factor_assistant.chat(messages)
    
    async def generate_factor(self, requirements: str) -> Dict[str, Any]:
        """生成因子公式"""
        return await self.factor_assistant.generate_factor_formula(requirements)
    
    async def optimize_factor(self, factor_code: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """优化因子"""
        return await self.factor_assistant.optimize_factor(factor_code, performance_data)
    
    async def debug_factor(self, code: str, error: str) -> str:
        """调试因子代码"""
        return await self.factor_assistant.debug_factor_code(code, error)
    
    # ================== AI因子发现接口 ==================
    
    async def discover_factors(self, market_data: Dict[str, pd.DataFrame],
                             target_returns: pd.Series) -> List[Dict[str, Any]]:
        """AI因子发现"""
        return await self.ai_discovery.discover_factors_from_market_data(market_data, target_returns)
    
    # ================== CTBench分析接口 ==================
    
    async def analyze_synthetic_data_quality(self, real_data: pd.DataFrame,
                                           synthetic_data: pd.DataFrame) -> Dict[str, Any]:
        """分析合成数据质量"""
        return await self.ctbench_assistant.analyze_synthetic_data(real_data, synthetic_data)
    
    # ================== 统一服务接口 ==================
    
    async def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            "timestamp": datetime.now().isoformat(),
            "services": {
                "factor_assistant": self.factor_assistant.client is not None,
                "ai_discovery": self.ai_discovery.client is not None,
                "ctbench_assistant": self.ctbench_assistant.client is not None
            },
            "model": self.config.get("LLM_MODEL", "unknown"),
            "base_url": self.config.get("LLM_BASE_URL", "unknown")
        }
    
    def get_available_functions(self) -> List[str]:
        """获取可用功能列表"""
        return [
            "chat_with_factor_assistant",
            "generate_factor", 
            "optimize_factor",
            "debug_factor",
            "discover_factors",
            "analyze_synthetic_data_quality"
        ]


# 创建全局统一LLM服务实例
unified_llm_service = UnifiedLLMService()