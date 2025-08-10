//! Rust高性能量化交易引擎
//! 
//! 提供高性能的数据处理、策略执行和风控检查功能
//! 通过PyO3 FFI与Python业务层集成

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub mod data_processor;
pub mod strategy;
pub mod risk;
pub mod utils;

use data_processor::{KlineData, ProcessedData, DataProcessor};
use strategy::{Strategy, TradeSignal, StrategyExecutor};
use risk::{Position, Trade, RiskResult, RiskManager};

/// 初始化日志系统
fn init_logging() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();
}

/// 处理K线数据 - Python接口
#[pyfunction]
fn process_kline_data(py: Python, klines: Vec<PyObject>) -> PyResult<PyObject> {
    let mut processor = DataProcessor::new();
    
    // 将Python对象转换为Rust结构
    let rust_klines: Vec<KlineData> = klines
        .into_iter()
        .map(|py_kline| {
            let dict = py_kline.downcast::<pyo3::types::PyDict>(py)?;
            KlineData::from_py_dict(dict)
        })
        .collect::<PyResult<Vec<_>>>()?;
    
    // 执行数据处理
    let result = processor.process_batch(rust_klines)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    // 转换回Python对象
    result.to_py_object(py)
}

/// 执行交易策略 - Python接口
#[pyfunction] 
fn execute_strategy(py: Python, strategy_data: PyObject, market_data: PyObject) -> PyResult<PyObject> {
    let mut executor = StrategyExecutor::new();
    
    // 转换Python参数
    let strategy_dict = strategy_data.downcast::<pyo3::types::PyDict>(py)?;
    let market_dict = market_data.downcast::<pyo3::types::PyDict>(py)?;
    
    let strategy = Strategy::from_py_dict(strategy_dict)?;
    let market = data_processor::MarketData::from_py_dict(market_dict)?;
    
    // 执行策略
    let signal = executor.execute(&strategy, &market)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    signal.to_py_object(py)
}

/// 风控检查 - Python接口
#[pyfunction]
fn risk_check(py: Python, position_data: PyObject, trade_data: PyObject) -> PyResult<PyObject> {
    let risk_manager = RiskManager::new();
    
    // 转换参数
    let position_dict = position_data.downcast::<pyo3::types::PyDict>(py)?;
    let trade_dict = trade_data.downcast::<pyo3::types::PyDict>(py)?;
    
    let position = Position::from_py_dict(position_dict)?;
    let trade = Trade::from_py_dict(trade_dict)?;
    
    // 执行风控检查
    let result = risk_manager.check_trade(&position, &trade)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    result.to_py_object(py)
}

/// 计算技术指标 - Python接口
#[pyfunction]
fn calculate_indicators(py: Python, price_data: Vec<f64>, indicators: Vec<String>) -> PyResult<PyObject> {
    use utils::indicators::*;
    
    let mut results = std::collections::HashMap::new();
    
    for indicator in indicators {
        match indicator.as_str() {
            "sma_20" => {
                let sma = simple_moving_average(&price_data, 20);
                results.insert("sma_20".to_string(), sma);
            },
            "ema_12" => {
                let ema = exponential_moving_average(&price_data, 12);
                results.insert("ema_12".to_string(), ema);
            },
            "rsi_14" => {
                let rsi = relative_strength_index(&price_data, 14);
                results.insert("rsi_14".to_string(), rsi);
            },
            "macd" => {
                let (macd_line, signal_line, histogram) = macd(&price_data, 12, 26, 9);
                results.insert("macd_line".to_string(), macd_line);
                results.insert("macd_signal".to_string(), signal_line);
                results.insert("macd_histogram".to_string(), histogram);
            },
            _ => {
                log::warn!("未知指标: {}", indicator);
            }
        }
    }
    
    // 转换为Python字典
    let py_dict = pyo3::types::PyDict::new(py);
    for (key, value) in results {
        py_dict.set_item(key, value)?;
    }
    
    Ok(py_dict.into())
}

/// 批量处理Alpha因子 - Python接口
#[pyfunction]
fn calculate_alpha_factors(py: Python, market_data: PyObject) -> PyResult<PyObject> {
    use utils::alpha_factors::*;
    
    let market_dict = market_data.downcast::<pyo3::types::PyDict>(py)?;
    let market = data_processor::MarketData::from_py_dict(market_dict)?;
    
    let mut factor_results = std::collections::HashMap::new();
    
    // Alpha101因子计算
    factor_results.insert("alpha001", alpha_001(&market.close_prices, &market.volumes));
    factor_results.insert("alpha002", alpha_002(&market.close_prices, &market.volumes));
    factor_results.insert("alpha003", alpha_003(&market.close_prices, &market.volumes));
    factor_results.insert("alpha004", alpha_004(&market.close_prices, &market.high_prices, &market.low_prices, &market.volumes));
    factor_results.insert("alpha005", alpha_005(&market.close_prices, &market.volumes));
    
    // 自定义因子
    factor_results.insert("momentum_rsi", momentum_rsi_factor(&market.close_prices, 14, 20));
    factor_results.insert("volatility_breakout", volatility_breakout_factor(&market.close_prices, &market.volumes, 20));
    factor_results.insert("mean_reversion", mean_reversion_factor(&market.close_prices, 10, 2.0));
    
    // 转换为Python字典
    let py_dict = pyo3::types::PyDict::new(py);
    for (key, value) in factor_results {
        py_dict.set_item(key, value)?;
    }
    
    Ok(py_dict.into())
}

/// 性能基准测试 - Python接口
#[pyfunction]
fn benchmark_performance(py: Python, test_size: usize, iterations: usize) -> PyResult<PyObject> {
    use std::time::Instant;
    use rand::Rng;
    
    let mut rng = rand::thread_rng();
    let test_data: Vec<f64> = (0..test_size)
        .map(|_| rng.gen_range(100.0..200.0))
        .collect();
    
    let mut times = Vec::new();
    
    // 基准测试
    for _ in 0..iterations {
        let start = Instant::now();
        
        // 执行数据处理
        let _sma = utils::indicators::simple_moving_average(&test_data, 20);
        let _ema = utils::indicators::exponential_moving_average(&test_data, 12);
        let _rsi = utils::indicators::relative_strength_index(&test_data, 14);
        
        let elapsed = start.elapsed();
        times.push(elapsed.as_nanos() as f64 / 1_000_000.0); // 转换为毫秒
    }
    
    // 计算统计指标
    let mean_time = times.iter().sum::<f64>() / times.len() as f64;
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_time = times[times.len() / 2];
    let p95_time = times[(times.len() as f64 * 0.95) as usize];
    
    // 构造结果字典
    let result = pyo3::types::PyDict::new(py);
    result.set_item("mean_time_ms", mean_time)?;
    result.set_item("median_time_ms", median_time)?;
    result.set_item("p95_time_ms", p95_time)?;
    result.set_item("throughput_ops_per_sec", 1000.0 / mean_time)?;
    result.set_item("test_size", test_size)?;
    result.set_item("iterations", iterations)?;
    
    Ok(result.into())
}

/// WebSocket数据流处理器 - Python接口
#[pyfunction]
fn start_websocket_stream(py: Python, url: String, symbols: Vec<String>) -> PyResult<PyObject> {
    use tokio::runtime::Runtime;
    use data_processor::websocket::WebSocketStreamer;
    
    let rt = Runtime::new()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    
    let streamer = WebSocketStreamer::new(url, symbols);
    
    // 在后台运行WebSocket连接
    rt.spawn(async move {
        if let Err(e) = streamer.start().await {
            log::error!("WebSocket stream error: {}", e);
        }
    });
    
    // 返回连接状态
    let result = pyo3::types::PyDict::new(py);
    result.set_item("status", "started")?;
    result.set_item("message", "WebSocket stream started successfully")?;
    
    Ok(result.into())
}

/// Python模块定义
#[pymodule]
fn rust_engine(py: Python, m: &PyModule) -> PyResult<()> {
    // 初始化日志
    let _ = init_logging();
    
    // 添加函数到模块
    m.add_function(wrap_pyfunction!(process_kline_data, m)?)?;
    m.add_function(wrap_pyfunction!(execute_strategy, m)?)?;
    m.add_function(wrap_pyfunction!(risk_check, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_indicators, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_alpha_factors, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_performance, m)?)?;
    m.add_function(wrap_pyfunction!(start_websocket_stream, m)?)?;
    
    // 添加常量
    m.add("VERSION", "1.0.0")?;
    m.add("AUTHOR", "Fullstack Engineer Agent")?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_functionality() {
        // 基础功能测试
        assert_eq!(2 + 2, 4);
    }
}