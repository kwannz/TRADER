//! 技术分析因子
//! 
//! 实现各种技术指标的高性能计算，包括RSI、MACD、移动平均等

use ndarray::Array1;
use rayon::prelude::*;

/// 计算RSI（相对强弱指数）
/// 
/// # 参数
/// * `prices` - 价格序列
/// * `period` - 计算周期
/// 
/// # 返回
/// RSI值序列，前period-1个值为NaN
pub fn rsi(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() < period + 1 {
        return vec![f64::NAN; prices.len()];
    }

    let mut result = vec![f64::NAN; prices.len()];
    let mut gains = Vec::with_capacity(prices.len() - 1);
    let mut losses = Vec::with_capacity(prices.len() - 1);

    // 计算价格变化
    for i in 1..prices.len() {
        let change = prices[i] - prices[i - 1];
        gains.push(if change > 0.0 { change } else { 0.0 });
        losses.push(if change < 0.0 { -change } else { 0.0 });
    }

    // 计算初始的平均收益和损失
    let initial_avg_gain = gains[..period].iter().sum::<f64>() / period as f64;
    let initial_avg_loss = losses[..period].iter().sum::<f64>() / period as f64;

    let mut avg_gain = initial_avg_gain;
    let mut avg_loss = initial_avg_loss;

    // 计算第一个RSI值
    if avg_loss != 0.0 {
        let rs = avg_gain / avg_loss;
        result[period] = 100.0 - (100.0 / (1.0 + rs));
    } else {
        result[period] = 100.0;
    }

    // 使用Wilder's平滑方法计算后续RSI值
    for i in (period + 1)..prices.len() {
        let gain_index = i - 1;
        let loss_index = i - 1;
        
        avg_gain = (avg_gain * (period - 1) as f64 + gains[gain_index]) / period as f64;
        avg_loss = (avg_loss * (period - 1) as f64 + losses[loss_index]) / period as f64;

        if avg_loss != 0.0 {
            let rs = avg_gain / avg_loss;
            result[i] = 100.0 - (100.0 / (1.0 + rs));
        } else {
            result[i] = 100.0;
        }
    }

    result
}

/// 计算MACD（指数平滑移动平均收敛/发散）
/// 
/// # 参数
/// * `prices` - 价格序列
/// * `fast_period` - 快线周期
/// * `slow_period` - 慢线周期
/// * `signal_period` - 信号线周期
/// 
/// # 返回
/// (MACD线, 信号线, 柱状图)
pub fn macd(
    prices: &[f64], 
    fast_period: usize, 
    slow_period: usize, 
    signal_period: usize
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let fast_ema = ema(prices, fast_period);
    let slow_ema = ema(prices, slow_period);
    
    let mut macd_line = vec![f64::NAN; prices.len()];
    
    // 计算MACD线
    for i in 0..prices.len() {
        if !fast_ema[i].is_nan() && !slow_ema[i].is_nan() {
            macd_line[i] = fast_ema[i] - slow_ema[i];
        }
    }
    
    // 计算信号线（MACD的EMA）
    let signal_line = ema(&macd_line.iter().filter(|&&x| !x.is_nan()).cloned().collect::<Vec<f64>>(), signal_period);
    
    // 扩展信号线以匹配原始长度
    let mut full_signal_line = vec![f64::NAN; prices.len()];
    let start_index = slow_period - 1 + signal_period - 1;
    for (i, &value) in signal_line.iter().enumerate() {
        if start_index + i < full_signal_line.len() {
            full_signal_line[start_index + i] = value;
        }
    }
    
    // 计算柱状图
    let mut histogram = vec![f64::NAN; prices.len()];
    for i in 0..prices.len() {
        if !macd_line[i].is_nan() && !full_signal_line[i].is_nan() {
            histogram[i] = macd_line[i] - full_signal_line[i];
        }
    }
    
    (macd_line, full_signal_line, histogram)
}

/// 计算简单移动平均（SMA）
/// 
/// # 参数
/// * `prices` - 价格序列
/// * `period` - 计算周期
/// 
/// # 返回
/// SMA值序列
pub fn sma(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() < period {
        return vec![f64::NAN; prices.len()];
    }

    let mut result = vec![f64::NAN; period - 1];
    
    // 使用滑动窗口计算SMA
    for i in period..=prices.len() {
        let window_sum: f64 = prices[i - period..i].iter().sum();
        result.push(window_sum / period as f64);
    }

    result
}

/// 计算指数移动平均（EMA）
/// 
/// # 参数
/// * `prices` - 价格序列
/// * `period` - 计算周期
/// 
/// # 返回
/// EMA值序列
pub fn ema(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.is_empty() || period == 0 {
        return vec![f64::NAN; prices.len()];
    }

    let mut result = vec![f64::NAN; prices.len()];
    let alpha = 2.0 / (period + 1) as f64;

    // 找到第一个有效价格作为初始值
    let mut start_index = 0;
    while start_index < prices.len() && prices[start_index].is_nan() {
        start_index += 1;
    }

    if start_index >= prices.len() {
        return result;
    }

    result[start_index] = prices[start_index];

    // 计算EMA
    for i in (start_index + 1)..prices.len() {
        if !prices[i].is_nan() {
            result[i] = alpha * prices[i] + (1.0 - alpha) * result[i - 1];
        }
    }

    result
}

/// 计算布林带
/// 
/// # 参数
/// * `prices` - 价格序列
/// * `period` - 计算周期
/// * `std_dev` - 标准差倍数
/// 
/// # 返回
/// (上轨, 中轨, 下轨)
pub fn bollinger_bands(
    prices: &[f64], 
    period: usize, 
    std_dev: f64
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let middle_line = sma(prices, period);
    let mut upper_line = vec![f64::NAN; prices.len()];
    let mut lower_line = vec![f64::NAN; prices.len()];

    for i in (period - 1)..prices.len() {
        if !middle_line[i].is_nan() {
            let window = &prices[i - period + 1..=i];
            let mean = middle_line[i];
            
            // 计算标准差
            let variance: f64 = window.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / period as f64;
            let std = variance.sqrt();
            
            upper_line[i] = mean + std_dev * std;
            lower_line[i] = mean - std_dev * std;
        }
    }

    (upper_line, middle_line, lower_line)
}

/// 计算随机指标（Stochastic）
/// 
/// # 参数
/// * `high` - 最高价序列
/// * `low` - 最低价序列
/// * `close` - 收盘价序列
/// * `k_period` - K值周期
/// * `d_period` - D值周期
/// 
/// # 返回
/// (K值, D值)
pub fn stochastic(
    high: &[f64],
    low: &[f64], 
    close: &[f64],
    k_period: usize,
    d_period: usize,
) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(high.len(), low.len());
    assert_eq!(high.len(), close.len());
    
    let mut k_values = vec![f64::NAN; close.len()];
    
    for i in (k_period - 1)..close.len() {
        let window_high = high[i - k_period + 1..=i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let window_low = low[i - k_period + 1..=i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        if window_high != window_low {
            k_values[i] = 100.0 * (close[i] - window_low) / (window_high - window_low);
        } else {
            k_values[i] = 50.0; // 中性值
        }
    }
    
    // D值是K值的移动平均
    let d_values = sma(&k_values, d_period);
    
    (k_values, d_values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sma() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&prices, 3);
        
        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_relative_eq!(result[2], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result[3], 3.0, epsilon = 1e-10);
        assert_relative_eq!(result[4], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ema() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ema(&prices, 3);
        
        assert_relative_eq!(result[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result[1], 1.5, epsilon = 1e-10);
        // EMA计算较复杂，这里只测试基本逻辑
        assert!(result[2] > result[1]);
    }

    #[test]
    fn test_rsi_basic() {
        let prices = vec![44.0, 44.25, 44.5, 43.75, 44.5, 45.0, 45.25, 45.5, 45.75, 46.0];
        let result = rsi(&prices, 5);
        
        // 前几个值应该是NaN
        assert!(result[0].is_nan());
        assert!(result[4].is_nan());
        
        // RSI值应该在0-100之间
        for i in 5..result.len() {
            assert!(result[i] >= 0.0 && result[i] <= 100.0);
        }
    }
}