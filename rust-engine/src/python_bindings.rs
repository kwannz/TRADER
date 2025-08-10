//! Python绑定模块

use pyo3::prelude::*;
use numpy::PyArray1;
use crate::factors::technical;

/// Python模块入口点
#[pymodule]
fn quant_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    // 注册因子计算函数
    m.add_function(wrap_pyfunction!(py_calculate_rsi, m)?)?;
    m.add_function(wrap_pyfunction!(py_calculate_macd, m)?)?;
    m.add_function(wrap_pyfunction!(py_calculate_ma, m)?)?;
    m.add_function(wrap_pyfunction!(py_calculate_sma, m)?)?;
    m.add_function(wrap_pyfunction!(py_calculate_ema, m)?)?;
    m.add_function(wrap_pyfunction!(py_calculate_bollinger_bands, m)?)?;
    
    // 添加版本信息
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}

/// Python包装的RSI计算函数
#[pyfunction]
fn py_calculate_rsi<'py>(
    py: Python<'py>,
    prices: &PyArray1<f64>,
    period: usize,
) -> PyResult<&'py PyArray1<f64>> {
    let prices_slice = unsafe { prices.as_slice()? };
    let result = technical::rsi(prices_slice, period);
    Ok(PyArray1::from_vec(py, result))
}

/// Python包装的MACD计算函数
#[pyfunction]
fn py_calculate_macd<'py>(
    py: Python<'py>,
    prices: &PyArray1<f64>,
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> PyResult<(&'py PyArray1<f64>, &'py PyArray1<f64>, &'py PyArray1<f64>)> {
    let prices_slice = unsafe { prices.as_slice()? };
    let (macd, signal, histogram) = technical::macd(
        prices_slice, 
        fast_period, 
        slow_period, 
        signal_period
    );
    
    Ok((
        PyArray1::from_vec(py, macd),
        PyArray1::from_vec(py, signal),
        PyArray1::from_vec(py, histogram),
    ))
}

/// Python包装的移动平均计算函数
#[pyfunction]
fn py_calculate_ma<'py>(
    py: Python<'py>,
    prices: &PyArray1<f64>,
    period: usize,
    ma_type: &str,
) -> PyResult<&'py PyArray1<f64>> {
    let prices_slice = unsafe { prices.as_slice()? };
    let result = match ma_type {
        "sma" | "simple" => technical::sma(prices_slice, period),
        "ema" | "exponential" => technical::ema(prices_slice, period),
        _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid MA type")),
    };
    Ok(PyArray1::from_vec(py, result))
}

/// Python包装的SMA计算函数
#[pyfunction]
fn py_calculate_sma<'py>(
    py: Python<'py>,
    prices: &PyArray1<f64>,
    period: usize,
) -> PyResult<&'py PyArray1<f64>> {
    let prices_slice = unsafe { prices.as_slice()? };
    let result = technical::sma(prices_slice, period);
    Ok(PyArray1::from_vec(py, result))
}

/// Python包装的EMA计算函数
#[pyfunction]
fn py_calculate_ema<'py>(
    py: Python<'py>,
    prices: &PyArray1<f64>,
    period: usize,
) -> PyResult<&'py PyArray1<f64>> {
    let prices_slice = unsafe { prices.as_slice()? };
    let result = technical::ema(prices_slice, period);
    Ok(PyArray1::from_vec(py, result))
}

/// Python包装的布林带计算函数
#[pyfunction]
fn py_calculate_bollinger_bands<'py>(
    py: Python<'py>,
    prices: &PyArray1<f64>,
    period: usize,
    std_dev: f64,
) -> PyResult<(&'py PyArray1<f64>, &'py PyArray1<f64>, &'py PyArray1<f64>)> {
    let prices_slice = unsafe { prices.as_slice()? };
    let (upper, middle, lower) = technical::bollinger_bands(
        prices_slice, 
        period, 
        std_dev
    );
    
    Ok((
        PyArray1::from_vec(py, upper),
        PyArray1::from_vec(py, middle),
        PyArray1::from_vec(py, lower),
    ))
}