//! 数据处理模块
//! 
//! 提供高性能的市场数据处理功能，包括K线数据处理、WebSocket流处理等

pub mod kline;
pub mod websocket;
pub mod factor;

use serde::{Deserialize, Serialize};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DataProcessorError {
    #[error("Invalid data format: {0}")]
    InvalidFormat(String),
    #[error("Processing error: {0}")]
    ProcessingError(String),
    #[error("WebSocket error: {0}")]
    WebSocketError(String),
}

/// K线数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KlineData {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub interval: String,
}

impl KlineData {
    /// 从Python字典创建KlineData
    pub fn from_py_dict(dict: &PyDict) -> PyResult<Self> {
        let symbol = dict.get_item("symbol")
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("missing 'symbol'"))?
            .extract::<String>()?;
        
        let timestamp_str = dict.get_item("timestamp")
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("missing 'timestamp'"))?
            .extract::<String>()?;
        
        let timestamp = DateTime::parse_from_rfc3339(&timestamp_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("invalid timestamp: {}", e)))?
            .with_timezone(&Utc);
        
        Ok(KlineData {
            symbol,
            timestamp,
            open: dict.get_item("open").unwrap().extract()?,
            high: dict.get_item("high").unwrap().extract()?,
            low: dict.get_item("low").unwrap().extract()?,
            close: dict.get_item("close").unwrap().extract()?,
            volume: dict.get_item("volume").unwrap().extract()?,
            interval: dict.get_item("interval").unwrap_or(&"1m".into()).extract()?,
        })
    }
    
    /// 转换为Python对象
    pub fn to_py_object(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("symbol", &self.symbol)?;
        dict.set_item("timestamp", self.timestamp.to_rfc3339())?;
        dict.set_item("open", self.open)?;
        dict.set_item("high", self.high)?;
        dict.set_item("low", self.low)?;
        dict.set_item("close", self.close)?;
        dict.set_item("volume", self.volume)?;
        dict.set_item("interval", &self.interval)?;
        
        Ok(dict.into())
    }
}

/// 市场数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketData {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub open_prices: Vec<f64>,
    pub high_prices: Vec<f64>,
    pub low_prices: Vec<f64>,
    pub close_prices: Vec<f64>,
    pub volumes: Vec<f64>,
    pub interval: String,
    pub count: usize,
}

impl MarketData {
    /// 从Python字典创建MarketData
    pub fn from_py_dict(dict: &PyDict) -> PyResult<Self> {
        let symbol = dict.get_item("symbol").unwrap().extract()?;
        let timestamp_str = dict.get_item("timestamp").unwrap().extract::<String>()?;
        let timestamp = DateTime::parse_from_rfc3339(&timestamp_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("invalid timestamp: {}", e)))?
            .with_timezone(&Utc);
        
        let open_prices: Vec<f64> = dict.get_item("open_prices").unwrap().extract()?;
        let high_prices: Vec<f64> = dict.get_item("high_prices").unwrap().extract()?;
        let low_prices: Vec<f64> = dict.get_item("low_prices").unwrap().extract()?;
        let close_prices: Vec<f64> = dict.get_item("close_prices").unwrap().extract()?;
        let volumes: Vec<f64> = dict.get_item("volumes").unwrap().extract()?;
        
        Ok(MarketData {
            symbol,
            timestamp,
            count: close_prices.len(),
            open_prices,
            high_prices,
            low_prices,
            close_prices,
            volumes,
            interval: dict.get_item("interval").unwrap_or(&"1m".into()).extract()?,
        })
    }
}

/// 处理后的数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedData {
    pub symbol: String,
    pub processed_at: DateTime<Utc>,
    pub indicators: HashMap<String, Vec<f64>>,
    pub alpha_factors: HashMap<String, f64>,
    pub signals: Vec<String>,
    pub metadata: HashMap<String, String>,
    pub processing_time_ms: f64,
}

impl ProcessedData {
    /// 转换为Python对象
    pub fn to_py_object(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("symbol", &self.symbol)?;
        dict.set_item("processed_at", self.processed_at.to_rfc3339())?;
        dict.set_item("processing_time_ms", self.processing_time_ms)?;
        
        // 指标数据
        let indicators_dict = PyDict::new(py);
        for (key, values) in &self.indicators {
            indicators_dict.set_item(key, values)?;
        }
        dict.set_item("indicators", indicators_dict)?;
        
        // Alpha因子
        let factors_dict = PyDict::new(py);
        for (key, value) in &self.alpha_factors {
            factors_dict.set_item(key, value)?;
        }
        dict.set_item("alpha_factors", factors_dict)?;
        
        // 信号
        dict.set_item("signals", &self.signals)?;
        
        // 元数据
        let metadata_dict = PyDict::new(py);
        for (key, value) in &self.metadata {
            metadata_dict.set_item(key, value)?;
        }
        dict.set_item("metadata", metadata_dict)?;
        
        Ok(dict.into())
    }
}

/// 数据处理器主类
pub struct DataProcessor {
    processor_id: String,
    created_at: DateTime<Utc>,
    total_processed: u64,
}

impl DataProcessor {
    /// 创建新的数据处理器
    pub fn new() -> Self {
        Self {
            processor_id: format!("dp_{}", chrono::Utc::now().timestamp()),
            created_at: Utc::now(),
            total_processed: 0,
        }
    }
    
    /// 批量处理K线数据
    pub fn process_batch(&mut self, klines: Vec<KlineData>) -> Result<ProcessedData, DataProcessorError> {
        let start_time = std::time::Instant::now();
        
        if klines.is_empty() {
            return Err(DataProcessorError::InvalidFormat("Empty kline data".to_string()));
        }
        
        let symbol = klines[0].symbol.clone();
        
        // 提取价格数据
        let close_prices: Vec<f64> = klines.iter().map(|k| k.close).collect();
        let open_prices: Vec<f64> = klines.iter().map(|k| k.open).collect();
        let high_prices: Vec<f64> = klines.iter().map(|k| k.high).collect();
        let low_prices: Vec<f64> = klines.iter().map(|k| k.low).collect();
        let volumes: Vec<f64> = klines.iter().map(|k| k.volume).collect();
        
        // 计算技术指标
        let mut indicators = HashMap::new();
        
        if close_prices.len() >= 20 {
            indicators.insert("sma_20".to_string(), 
                crate::utils::indicators::simple_moving_average(&close_prices, 20));
        }
        
        if close_prices.len() >= 12 {
            indicators.insert("ema_12".to_string(), 
                crate::utils::indicators::exponential_moving_average(&close_prices, 12));
        }
        
        if close_prices.len() >= 14 {
            indicators.insert("rsi_14".to_string(), 
                crate::utils::indicators::relative_strength_index(&close_prices, 14));
        }
        
        if close_prices.len() >= 26 {
            let (macd_line, signal_line, histogram) = crate::utils::indicators::macd(&close_prices, 12, 26, 9);
            indicators.insert("macd_line".to_string(), macd_line);
            indicators.insert("macd_signal".to_string(), signal_line);
            indicators.insert("macd_histogram".to_string(), histogram);
        }
        
        // 计算Alpha因子
        let mut alpha_factors = HashMap::new();
        alpha_factors.insert("alpha001".to_string(), 
            crate::utils::alpha_factors::alpha_001(&close_prices, &volumes));
        alpha_factors.insert("alpha002".to_string(), 
            crate::utils::alpha_factors::alpha_002(&close_prices, &volumes));
        alpha_factors.insert("momentum_rsi".to_string(), 
            crate::utils::alpha_factors::momentum_rsi_factor(&close_prices, 14, 20));
        
        // 生成交易信号
        let mut signals = Vec::new();
        
        // 简单的信号生成逻辑
        if let (Some(rsi), Some(sma)) = (indicators.get("rsi_14"), indicators.get("sma_20")) {
            if let (Some(&last_rsi), Some(&last_close), Some(&last_sma)) = 
                (rsi.last(), close_prices.last(), sma.last()) {
                
                if last_rsi < 30.0 && last_close > last_sma {
                    signals.push("BUY_RSI_OVERSOLD".to_string());
                } else if last_rsi > 70.0 && last_close < last_sma {
                    signals.push("SELL_RSI_OVERBOUGHT".to_string());
                }
            }
        }
        
        // 构造元数据
        let mut metadata = HashMap::new();
        metadata.insert("processor_id".to_string(), self.processor_id.clone());
        metadata.insert("data_points".to_string(), klines.len().to_string());
        metadata.insert("timeframe".to_string(), klines[0].interval.clone());
        
        let processing_time = start_time.elapsed().as_nanos() as f64 / 1_000_000.0; // 转换为毫秒
        self.total_processed += klines.len() as u64;
        
        Ok(ProcessedData {
            symbol,
            processed_at: Utc::now(),
            indicators,
            alpha_factors,
            signals,
            metadata,
            processing_time_ms: processing_time,
        })
    }
    
    /// 获取处理器统计信息
    pub fn get_stats(&self) -> HashMap<String, String> {
        let mut stats = HashMap::new();
        stats.insert("processor_id".to_string(), self.processor_id.clone());
        stats.insert("created_at".to_string(), self.created_at.to_rfc3339());
        stats.insert("total_processed".to_string(), self.total_processed.to_string());
        stats.insert("uptime_seconds".to_string(), 
            (Utc::now() - self.created_at).num_seconds().to_string());
        stats
    }
}

impl Default for DataProcessor {
    fn default() -> Self {
        Self::new()
    }
}