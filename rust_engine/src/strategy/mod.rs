//! 策略执行模块
//! 
//! 提供高性能的交易策略执行引擎

use serde::{Deserialize, Serialize};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum StrategyError {
    #[error("Invalid strategy configuration: {0}")]
    InvalidConfig(String),
    #[error("Execution error: {0}")]
    ExecutionError(String),
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
}

/// 交易策略类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StrategyType {
    Grid,           // 网格策略
    DCA,            // 定投策略
    MACD,           // MACD策略
    RSI,            // RSI策略
    MeanReversion,  // 均值回归
    Momentum,       // 动量策略
    AIGenerated,    // AI生成策略
    Manual,         // 手动策略
}

impl std::fmt::Display for StrategyType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            StrategyType::Grid => write!(f, "grid"),
            StrategyType::DCA => write!(f, "dca"),
            StrategyType::MACD => write!(f, "macd"),
            StrategyType::RSI => write!(f, "rsi"),
            StrategyType::MeanReversion => write!(f, "mean_reversion"),
            StrategyType::Momentum => write!(f, "momentum"),
            StrategyType::AIGenerated => write!(f, "ai_generated"),
            StrategyType::Manual => write!(f, "manual"),
        }
    }
}

/// 交易信号类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalType {
    Buy,
    Sell,
    Hold,
    StrongBuy,
    StrongSell,
}

impl std::fmt::Display for SignalType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            SignalType::Buy => write!(f, "BUY"),
            SignalType::Sell => write!(f, "SELL"),
            SignalType::Hold => write!(f, "HOLD"),
            SignalType::StrongBuy => write!(f, "STRONG_BUY"),
            SignalType::StrongSell => write!(f, "STRONG_SELL"),
        }
    }
}

/// 策略配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Strategy {
    pub id: String,
    pub name: String,
    pub strategy_type: StrategyType,
    pub symbol: String,
    pub config: HashMap<String, f64>,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Strategy {
    /// 从Python字典创建策略
    pub fn from_py_dict(dict: &PyDict) -> PyResult<Self> {
        let strategy_type_str: String = dict.get_item("type")
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("missing 'type'"))?
            .extract()?;
        
        let strategy_type = match strategy_type_str.as_str() {
            "grid" => StrategyType::Grid,
            "dca" => StrategyType::DCA,
            "macd" => StrategyType::MACD,
            "rsi" => StrategyType::RSI,
            "mean_reversion" => StrategyType::MeanReversion,
            "momentum" => StrategyType::Momentum,
            "ai_generated" => StrategyType::AIGenerated,
            "manual" => StrategyType::Manual,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("invalid strategy type")),
        };
        
        let config_dict = dict.get_item("config")
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("missing 'config'"))?
            .downcast::<PyDict>()?;
        
        let mut config = HashMap::new();
        for item in config_dict.iter() {
            let (key, value) = item;
            let key_str: String = key.extract()?;
            let value_float: f64 = value.extract()?;
            config.insert(key_str, value_float);
        }
        
        Ok(Strategy {
            id: dict.get_item("id").unwrap_or(&"".into()).extract()?,
            name: dict.get_item("name").unwrap_or(&"".into()).extract()?,
            strategy_type,
            symbol: dict.get_item("symbol").unwrap_or(&"BTC/USDT".into()).extract()?,
            config,
            enabled: dict.get_item("enabled").unwrap_or(&true.into()).extract()?,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        })
    }
}

/// 交易信号
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeSignal {
    pub signal_type: SignalType,
    pub confidence: f64,
    pub price: f64,
    pub quantity: f64,
    pub timestamp: DateTime<Utc>,
    pub reason: String,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub metadata: HashMap<String, String>,
}

impl TradeSignal {
    /// 转换为Python对象
    pub fn to_py_object(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("signal_type", self.signal_type.to_string())?;
        dict.set_item("confidence", self.confidence)?;
        dict.set_item("price", self.price)?;
        dict.set_item("quantity", self.quantity)?;
        dict.set_item("timestamp", self.timestamp.to_rfc3339())?;
        dict.set_item("reason", &self.reason)?;
        dict.set_item("stop_loss", self.stop_loss)?;
        dict.set_item("take_profit", self.take_profit)?;
        
        let metadata_dict = PyDict::new(py);
        for (key, value) in &self.metadata {
            metadata_dict.set_item(key, value)?;
        }
        dict.set_item("metadata", metadata_dict)?;
        
        Ok(dict.into())
    }
}

/// 策略执行器
pub struct StrategyExecutor {
    executor_id: String,
    total_executions: u64,
    created_at: DateTime<Utc>,
}

impl StrategyExecutor {
    /// 创建新的策略执行器
    pub fn new() -> Self {
        Self {
            executor_id: format!("se_{}", Utc::now().timestamp()),
            total_executions: 0,
            created_at: Utc::now(),
        }
    }
    
    /// 执行策略
    pub fn execute(
        &mut self,
        strategy: &Strategy,
        market_data: &crate::data_processor::MarketData,
    ) -> Result<TradeSignal, StrategyError> {
        if !strategy.enabled {
            return Err(StrategyError::ExecutionError("Strategy is disabled".to_string()));
        }
        
        let signal = match strategy.strategy_type {
            StrategyType::Grid => self.execute_grid_strategy(strategy, market_data)?,
            StrategyType::DCA => self.execute_dca_strategy(strategy, market_data)?,
            StrategyType::MACD => self.execute_macd_strategy(strategy, market_data)?,
            StrategyType::RSI => self.execute_rsi_strategy(strategy, market_data)?,
            StrategyType::MeanReversion => self.execute_mean_reversion_strategy(strategy, market_data)?,
            StrategyType::Momentum => self.execute_momentum_strategy(strategy, market_data)?,
            StrategyType::AIGenerated => self.execute_ai_strategy(strategy, market_data)?,
            StrategyType::Manual => self.execute_manual_strategy(strategy, market_data)?,
        };
        
        self.total_executions += 1;
        Ok(signal)
    }
    
    /// 网格策略执行
    fn execute_grid_strategy(
        &self,
        strategy: &Strategy,
        market_data: &crate::data_processor::MarketData,
    ) -> Result<TradeSignal, StrategyError> {
        let grid_size = strategy.config.get("grid_size").unwrap_or(&0.5);
        let upper_price = strategy.config.get("upper_price").unwrap_or(&50000.0);
        let lower_price = strategy.config.get("lower_price").unwrap_or(&40000.0);
        let investment = strategy.config.get("investment").unwrap_or(&1000.0);
        
        if market_data.close_prices.is_empty() {
            return Err(StrategyError::InsufficientData("No close prices available".to_string()));
        }
        
        let current_price = market_data.close_prices[market_data.close_prices.len() - 1];
        let grid_levels = ((upper_price - lower_price) / grid_size) as usize;
        let quantity_per_grid = investment / grid_levels as f64;
        
        let mut signal_type = SignalType::Hold;
        let mut confidence = 0.5;
        let mut reason = "Grid analysis".to_string();
        
        // 简化的网格逻辑
        if current_price <= lower_price + grid_size {
            signal_type = SignalType::Buy;
            confidence = 0.8;
            reason = "Price near grid lower bound".to_string();
        } else if current_price >= upper_price - grid_size {
            signal_type = SignalType::Sell;
            confidence = 0.8;
            reason = "Price near grid upper bound".to_string();
        }
        
        let mut metadata = HashMap::new();
        metadata.insert("strategy_type".to_string(), "grid".to_string());
        metadata.insert("grid_levels".to_string(), grid_levels.to_string());
        metadata.insert("current_grid_level".to_string(), 
            ((current_price - lower_price) / grid_size).floor().to_string());
        
        Ok(TradeSignal {
            signal_type,
            confidence,
            price: current_price,
            quantity: quantity_per_grid / current_price,
            timestamp: Utc::now(),
            reason,
            stop_loss: Some(lower_price * 0.95),
            take_profit: Some(upper_price * 1.05),
            metadata,
        })
    }
    
    /// DCA策略执行
    fn execute_dca_strategy(
        &self,
        strategy: &Strategy,
        market_data: &crate::data_processor::MarketData,
    ) -> Result<TradeSignal, StrategyError> {
        let investment_amount = strategy.config.get("investment_amount").unwrap_or(&100.0);
        let frequency_hours = strategy.config.get("frequency_hours").unwrap_or(&24.0) as usize;
        
        if market_data.close_prices.is_empty() {
            return Err(StrategyError::InsufficientData("No close prices available".to_string()));
        }
        
        let current_price = market_data.close_prices[market_data.close_prices.len() - 1];
        
        // DCA策略总是产生买入信号
        let mut metadata = HashMap::new();
        metadata.insert("strategy_type".to_string(), "dca".to_string());
        metadata.insert("frequency_hours".to_string(), frequency_hours.to_string());
        metadata.insert("investment_amount".to_string(), investment_amount.to_string());
        
        Ok(TradeSignal {
            signal_type: SignalType::Buy,
            confidence: 0.9, // DCA策略高置信度
            price: current_price,
            quantity: investment_amount / current_price,
            timestamp: Utc::now(),
            reason: "DCA periodic investment".to_string(),
            stop_loss: None, // DCA策略通常不设止损
            take_profit: None,
            metadata,
        })
    }
    
    /// MACD策略执行
    fn execute_macd_strategy(
        &self,
        strategy: &Strategy,
        market_data: &crate::data_processor::MarketData,
    ) -> Result<TradeSignal, StrategyError> {
        let fast_period = strategy.config.get("fast_period").unwrap_or(&12.0) as usize;
        let slow_period = strategy.config.get("slow_period").unwrap_or(&26.0) as usize;
        let signal_period = strategy.config.get("signal_period").unwrap_or(&9.0) as usize;
        
        if market_data.close_prices.len() < slow_period.max(signal_period) {
            return Err(StrategyError::InsufficientData("Insufficient data for MACD calculation".to_string()));
        }
        
        let (macd_line, signal_line, histogram) = crate::utils::indicators::macd(
            &market_data.close_prices,
            fast_period,
            slow_period,
            signal_period
        );
        
        if macd_line.is_empty() || signal_line.is_empty() || histogram.is_empty() {
            return Err(StrategyError::ExecutionError("MACD calculation failed".to_string()));
        }
        
        let current_price = market_data.close_prices[market_data.close_prices.len() - 1];
        let current_macd = macd_line[macd_line.len() - 1];
        let current_signal = signal_line[signal_line.len() - 1];
        let current_histogram = histogram[histogram.len() - 1];
        
        let mut signal_type = SignalType::Hold;
        let mut confidence = 0.5;
        let mut reason = "MACD neutral".to_string();
        
        // MACD交叉信号
        if current_macd > current_signal && current_histogram > 0.0 {
            signal_type = SignalType::Buy;
            confidence = 0.75;
            reason = "MACD bullish crossover".to_string();
        } else if current_macd < current_signal && current_histogram < 0.0 {
            signal_type = SignalType::Sell;
            confidence = 0.75;
            reason = "MACD bearish crossover".to_string();
        }
        
        let mut metadata = HashMap::new();
        metadata.insert("strategy_type".to_string(), "macd".to_string());
        metadata.insert("current_macd".to_string(), current_macd.to_string());
        metadata.insert("current_signal".to_string(), current_signal.to_string());
        metadata.insert("current_histogram".to_string(), current_histogram.to_string());
        
        Ok(TradeSignal {
            signal_type,
            confidence,
            price: current_price,
            quantity: 0.1, // 固定数量
            timestamp: Utc::now(),
            reason,
            stop_loss: Some(current_price * 0.95),
            take_profit: Some(current_price * 1.10),
            metadata,
        })
    }
    
    /// RSI策略执行
    fn execute_rsi_strategy(
        &self,
        strategy: &Strategy,
        market_data: &crate::data_processor::MarketData,
    ) -> Result<TradeSignal, StrategyError> {
        let rsi_period = strategy.config.get("rsi_period").unwrap_or(&14.0) as usize;
        let oversold_threshold = strategy.config.get("oversold_threshold").unwrap_or(&30.0);
        let overbought_threshold = strategy.config.get("overbought_threshold").unwrap_or(&70.0);
        
        if market_data.close_prices.len() <= rsi_period {
            return Err(StrategyError::InsufficientData("Insufficient data for RSI calculation".to_string()));
        }
        
        let rsi_values = crate::utils::indicators::relative_strength_index(
            &market_data.close_prices,
            rsi_period
        );
        
        if rsi_values.is_empty() {
            return Err(StrategyError::ExecutionError("RSI calculation failed".to_string()));
        }
        
        let current_price = market_data.close_prices[market_data.close_prices.len() - 1];
        let current_rsi = rsi_values[rsi_values.len() - 1];
        
        let mut signal_type = SignalType::Hold;
        let mut confidence = 0.5;
        let mut reason = "RSI neutral".to_string();
        
        // RSI超买超卖信号
        if current_rsi < *oversold_threshold {
            signal_type = SignalType::Buy;
            confidence = 0.8;
            reason = format!("RSI oversold at {:.2}", current_rsi);
        } else if current_rsi > *overbought_threshold {
            signal_type = SignalType::Sell;
            confidence = 0.8;
            reason = format!("RSI overbought at {:.2}", current_rsi);
        }
        
        let mut metadata = HashMap::new();
        metadata.insert("strategy_type".to_string(), "rsi".to_string());
        metadata.insert("current_rsi".to_string(), current_rsi.to_string());
        metadata.insert("oversold_threshold".to_string(), oversold_threshold.to_string());
        metadata.insert("overbought_threshold".to_string(), overbought_threshold.to_string());
        
        Ok(TradeSignal {
            signal_type,
            confidence,
            price: current_price,
            quantity: 0.1,
            timestamp: Utc::now(),
            reason,
            stop_loss: Some(current_price * 0.95),
            take_profit: Some(current_price * 1.08),
            metadata,
        })
    }
    
    /// 均值回归策略执行
    fn execute_mean_reversion_strategy(
        &self,
        strategy: &Strategy,
        market_data: &crate::data_processor::MarketData,
    ) -> Result<TradeSignal, StrategyError> {
        let lookback_period = strategy.config.get("lookback_period").unwrap_or(&20.0) as usize;
        let threshold = strategy.config.get("threshold").unwrap_or(&2.0);
        
        if market_data.close_prices.len() < lookback_period {
            return Err(StrategyError::InsufficientData("Insufficient data for mean reversion calculation".to_string()));
        }
        
        let alpha_factor = crate::utils::alpha_factors::mean_reversion_factor(
            &market_data.close_prices,
            lookback_period,
            *threshold
        );
        
        let current_price = market_data.close_prices[market_data.close_prices.len() - 1];
        
        let mut signal_type = SignalType::Hold;
        let mut confidence = 0.5;
        let mut reason = "Mean reversion neutral".to_string();
        
        if alpha_factor > 0.5 {
            signal_type = SignalType::Buy;
            confidence = alpha_factor.min(1.0);
            reason = format!("Mean reversion buy signal: {:.3}", alpha_factor);
        } else if alpha_factor < -0.5 {
            signal_type = SignalType::Sell;
            confidence = (-alpha_factor).min(1.0);
            reason = format!("Mean reversion sell signal: {:.3}", alpha_factor);
        }
        
        let mut metadata = HashMap::new();
        metadata.insert("strategy_type".to_string(), "mean_reversion".to_string());
        metadata.insert("alpha_factor".to_string(), alpha_factor.to_string());
        metadata.insert("lookback_period".to_string(), lookback_period.to_string());
        
        Ok(TradeSignal {
            signal_type,
            confidence,
            price: current_price,
            quantity: 0.1,
            timestamp: Utc::now(),
            reason,
            stop_loss: Some(current_price * 0.95),
            take_profit: Some(current_price * 1.05),
            metadata,
        })
    }
    
    /// 动量策略执行
    fn execute_momentum_strategy(
        &self,
        strategy: &Strategy,
        market_data: &crate::data_processor::MarketData,
    ) -> Result<TradeSignal, StrategyError> {
        let momentum_period = strategy.config.get("momentum_period").unwrap_or(&10.0) as usize;
        let rsi_period = strategy.config.get("rsi_period").unwrap_or(&14.0) as usize;
        
        if market_data.close_prices.len() < momentum_period.max(rsi_period) {
            return Err(StrategyError::InsufficientData("Insufficient data for momentum calculation".to_string()));
        }
        
        let momentum_rsi_factor = crate::utils::alpha_factors::momentum_rsi_factor(
            &market_data.close_prices,
            rsi_period,
            momentum_period
        );
        
        let current_price = market_data.close_prices[market_data.close_prices.len() - 1];
        
        let mut signal_type = SignalType::Hold;
        let mut confidence = 0.5;
        let mut reason = "Momentum neutral".to_string();
        
        if momentum_rsi_factor > 0.3 {
            signal_type = SignalType::Buy;
            confidence = momentum_rsi_factor.min(1.0);
            reason = format!("Positive momentum signal: {:.3}", momentum_rsi_factor);
        } else if momentum_rsi_factor < -0.3 {
            signal_type = SignalType::Sell;
            confidence = (-momentum_rsi_factor).min(1.0);
            reason = format!("Negative momentum signal: {:.3}", momentum_rsi_factor);
        }
        
        let mut metadata = HashMap::new();
        metadata.insert("strategy_type".to_string(), "momentum".to_string());
        metadata.insert("momentum_rsi_factor".to_string(), momentum_rsi_factor.to_string());
        metadata.insert("momentum_period".to_string(), momentum_period.to_string());
        
        Ok(TradeSignal {
            signal_type,
            confidence,
            price: current_price,
            quantity: 0.1,
            timestamp: Utc::now(),
            reason,
            stop_loss: Some(current_price * 0.94),
            take_profit: Some(current_price * 1.12),
            metadata,
        })
    }
    
    /// AI生成策略执行
    fn execute_ai_strategy(
        &self,
        strategy: &Strategy,
        market_data: &crate::data_processor::MarketData,
    ) -> Result<TradeSignal, StrategyError> {
        // AI策略会综合多个因子
        let alpha_factors = crate::utils::alpha_factors::calculate_alpha_batch(
            &market_data.close_prices,
            &market_data.high_prices,
            &market_data.low_prices,
            &market_data.volumes
        );
        
        let current_price = market_data.close_prices[market_data.close_prices.len() - 1];
        
        // 获取综合Alpha得分
        let combined_alpha = alpha_factors.get("combined_alpha").unwrap_or(&0.0);
        let volatility_breakout = alpha_factors.get("volatility_breakout").unwrap_or(&0.0);
        let momentum_rsi = alpha_factors.get("momentum_rsi").unwrap_or(&0.0);
        
        // AI策略决策逻辑
        let ai_score = combined_alpha + volatility_breakout * 0.5 + momentum_rsi * 0.3;
        
        let mut signal_type = SignalType::Hold;
        let mut confidence = 0.5;
        let mut reason = "AI analysis neutral".to_string();
        
        if ai_score > 0.2 {
            signal_type = SignalType::Buy;
            confidence = ai_score.min(1.0);
            reason = format!("AI bullish signal: {:.3}", ai_score);
        } else if ai_score < -0.2 {
            signal_type = SignalType::Sell;
            confidence = (-ai_score).min(1.0);
            reason = format!("AI bearish signal: {:.3}", ai_score);
        }
        
        let mut metadata = HashMap::new();
        metadata.insert("strategy_type".to_string(), "ai_generated".to_string());
        metadata.insert("ai_score".to_string(), ai_score.to_string());
        metadata.insert("combined_alpha".to_string(), combined_alpha.to_string());
        metadata.insert("volatility_breakout".to_string(), volatility_breakout.to_string());
        metadata.insert("momentum_rsi".to_string(), momentum_rsi.to_string());
        
        Ok(TradeSignal {
            signal_type,
            confidence,
            price: current_price,
            quantity: 0.1,
            timestamp: Utc::now(),
            reason,
            stop_loss: Some(current_price * 0.93),
            take_profit: Some(current_price * 1.15),
            metadata,
        })
    }
    
    /// 手动策略执行（占位符）
    fn execute_manual_strategy(
        &self,
        _strategy: &Strategy,
        market_data: &crate::data_processor::MarketData,
    ) -> Result<TradeSignal, StrategyError> {
        let current_price = market_data.close_prices[market_data.close_prices.len() - 1];
        
        let mut metadata = HashMap::new();
        metadata.insert("strategy_type".to_string(), "manual".to_string());
        
        Ok(TradeSignal {
            signal_type: SignalType::Hold,
            confidence: 1.0,
            price: current_price,
            quantity: 0.0,
            timestamp: Utc::now(),
            reason: "Manual strategy - awaiting user input".to_string(),
            stop_loss: None,
            take_profit: None,
            metadata,
        })
    }
    
    /// 获取执行器统计信息
    pub fn get_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        stats.insert("executor_id".to_string(), self.executor_id.clone());
        stats.insert("total_executions".to_string(), self.total_executions.to_string());
        stats.insert("created_at".to_string(), self.created_at.to_rfc3339());
        stats.insert("uptime_seconds".to_string(), 
            (Utc::now() - self.created_at).num_seconds().to_string());
        stats
    }
}

impl Default for StrategyExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_processor::MarketData;
    
    #[test]
    fn test_strategy_executor() {
        let mut executor = StrategyExecutor::new();
        
        let strategy = Strategy {
            id: "test_strategy".to_string(),
            name: "Test Strategy".to_string(),
            strategy_type: StrategyType::RSI,
            symbol: "BTC/USDT".to_string(),
            config: {
                let mut config = HashMap::new();
                config.insert("rsi_period".to_string(), 14.0);
                config.insert("oversold_threshold".to_string(), 30.0);
                config.insert("overbought_threshold".to_string(), 70.0);
                config
            },
            enabled: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        
        let market_data = MarketData {
            symbol: "BTC/USDT".to_string(),
            timestamp: Utc::now(),
            open_prices: vec![45000.0; 20],
            high_prices: vec![45500.0; 20],
            low_prices: vec![44500.0; 20],
            close_prices: vec![45000.0; 20],
            volumes: vec![100.0; 20],
            interval: "1m".to_string(),
            count: 20,
        };
        
        let result = executor.execute(&strategy, &market_data);
        assert!(result.is_ok());
        
        let signal = result.unwrap();
        assert!(signal.confidence >= 0.0 && signal.confidence <= 1.0);
    }
}