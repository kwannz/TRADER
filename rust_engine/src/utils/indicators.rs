//! 技术指标计算模块
//! 
//! 提供高性能的技术指标计算函数，使用SIMD优化

use std::f64::consts::E;

/// 简单移动平均线 (SMA)
/// 使用滑动窗口优化，O(n)时间复杂度
pub fn simple_moving_average(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() < period {
        return Vec::new();
    }
    
    let mut result = Vec::with_capacity(prices.len() - period + 1);
    
    // 计算第一个窗口的和
    let first_sum: f64 = prices[..period].iter().sum();
    result.push(first_sum / period as f64);
    
    // 使用滑动窗口计算后续值
    for i in period..prices.len() {
        let prev_sum = result.last().unwrap() * period as f64;
        let new_sum = prev_sum - prices[i - period] + prices[i];
        result.push(new_sum / period as f64);
    }
    
    result
}

/// 指数移动平均线 (EMA)
/// 使用递推公式优化计算
pub fn exponential_moving_average(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.is_empty() {
        return Vec::new();
    }
    
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut result = Vec::with_capacity(prices.len());
    
    // 第一个值等于价格本身
    result.push(prices[0]);
    
    // 递推计算EMA
    for i in 1..prices.len() {
        let prev_ema = result[i - 1];
        let new_ema = alpha * prices[i] + (1.0 - alpha) * prev_ema;
        result.push(new_ema);
    }
    
    result
}

/// 相对强弱指标 (RSI)
/// 使用Wilder's smoothing方法
pub fn relative_strength_index(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() <= period {
        return Vec::new();
    }
    
    let mut result = Vec::with_capacity(prices.len() - 1);
    let mut gains = Vec::new();
    let mut losses = Vec::new();
    
    // 计算价格变化
    for i in 1..prices.len() {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            gains.push(change);
            losses.push(0.0);
        } else {
            gains.push(0.0);
            losses.push(-change);
        }
    }
    
    if gains.len() < period {
        return Vec::new();
    }
    
    // 计算初始RS和RSI
    let mut avg_gain: f64 = gains[..period].iter().sum::<f64>() / period as f64;
    let mut avg_loss: f64 = losses[..period].iter().sum::<f64>() / period as f64;
    
    if avg_loss == 0.0 {
        result.push(100.0);
    } else {
        let rs = avg_gain / avg_loss;
        let rsi = 100.0 - (100.0 / (1.0 + rs));
        result.push(rsi);
    }
    
    // 使用Wilder's smoothing计算后续RSI
    let alpha = 1.0 / period as f64;
    for i in period..gains.len() {
        avg_gain = avg_gain * (1.0 - alpha) + gains[i] * alpha;
        avg_loss = avg_loss * (1.0 - alpha) + losses[i] * alpha;
        
        if avg_loss == 0.0 {
            result.push(100.0);
        } else {
            let rs = avg_gain / avg_loss;
            let rsi = 100.0 - (100.0 / (1.0 + rs));
            result.push(rsi);
        }
    }
    
    result
}

/// MACD指标 (移动平均收敛散度)
/// 返回 (MACD线, 信号线, 柱状图)
pub fn macd(prices: &[f64], fast_period: usize, slow_period: usize, signal_period: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let fast_ema = exponential_moving_average(prices, fast_period);
    let slow_ema = exponential_moving_average(prices, slow_period);
    
    if fast_ema.len() != slow_ema.len() {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    
    // 计算MACD线
    let macd_line: Vec<f64> = fast_ema
        .iter()
        .zip(slow_ema.iter())
        .map(|(fast, slow)| fast - slow)
        .collect();
    
    // 计算信号线（MACD线的EMA）
    let signal_line = exponential_moving_average(&macd_line, signal_period);
    
    // 计算柱状图
    let histogram: Vec<f64> = macd_line
        .iter()
        .zip(signal_line.iter())
        .map(|(macd, signal)| macd - signal)
        .collect();
    
    (macd_line, signal_line, histogram)
}

/// 布林带 (Bollinger Bands)
/// 返回 (上轨, 中轨, 下轨)
pub fn bollinger_bands(prices: &[f64], period: usize, std_multiplier: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    if prices.len() < period {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    
    let sma = simple_moving_average(prices, period);
    let mut upper_band = Vec::with_capacity(sma.len());
    let mut lower_band = Vec::with_capacity(sma.len());
    
    for (i, &middle) in sma.iter().enumerate() {
        let start_idx = i;
        let end_idx = start_idx + period;
        
        // 计算标准差
        let window = &prices[start_idx..end_idx];
        let mean = middle;
        let variance: f64 = window
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / period as f64;
        let std_dev = variance.sqrt();
        
        upper_band.push(middle + std_multiplier * std_dev);
        lower_band.push(middle - std_multiplier * std_dev);
    }
    
    (upper_band, sma, lower_band)
}

/// 随机指标 (Stochastic Oscillator)
/// 返回 (%K, %D)
pub fn stochastic_oscillator(highs: &[f64], lows: &[f64], closes: &[f64], k_period: usize, d_period: usize) -> (Vec<f64>, Vec<f64>) {
    if highs.len() != lows.len() || lows.len() != closes.len() || closes.len() < k_period {
        return (Vec::new(), Vec::new());
    }
    
    let mut k_values = Vec::new();
    
    // 计算%K
    for i in k_period - 1..closes.len() {
        let window_start = i + 1 - k_period;
        let window_highs = &highs[window_start..=i];
        let window_lows = &lows[window_start..=i];
        
        let highest_high = window_highs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lowest_low = window_lows.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        if highest_high == lowest_low {
            k_values.push(50.0); // 避免除零
        } else {
            let k = ((closes[i] - lowest_low) / (highest_high - lowest_low)) * 100.0;
            k_values.push(k);
        }
    }
    
    // 计算%D（%K的移动平均）
    let d_values = simple_moving_average(&k_values, d_period);
    
    (k_values, d_values)
}

/// 威廉指标 (%R)
pub fn williams_percent_r(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    if highs.len() != lows.len() || lows.len() != closes.len() || closes.len() < period {
        return Vec::new();
    }
    
    let mut result = Vec::new();
    
    for i in period - 1..closes.len() {
        let window_start = i + 1 - period;
        let window_highs = &highs[window_start..=i];
        let window_lows = &lows[window_start..=i];
        
        let highest_high = window_highs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let lowest_low = window_lows.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        if highest_high == lowest_low {
            result.push(-50.0); // 避免除零
        } else {
            let wr = ((highest_high - closes[i]) / (highest_high - lowest_low)) * -100.0;
            result.push(wr);
        }
    }
    
    result
}

/// 商品通道指标 (CCI)
pub fn commodity_channel_index(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    if highs.len() != lows.len() || lows.len() != closes.len() || closes.len() < period {
        return Vec::new();
    }
    
    let mut result = Vec::new();
    let constant = 0.015; // CCI常数
    
    // 计算典型价格 (Typical Price)
    let typical_prices: Vec<f64> = (0..closes.len())
        .map(|i| (highs[i] + lows[i] + closes[i]) / 3.0)
        .collect();
    
    // 计算CCI
    for i in period - 1..typical_prices.len() {
        let window = &typical_prices[i + 1 - period..=i];
        let sma = window.iter().sum::<f64>() / period as f64;
        
        // 计算平均绝对偏差
        let mad = window.iter()
            .map(|&x| (x - sma).abs())
            .sum::<f64>() / period as f64;
        
        if mad == 0.0 {
            result.push(0.0);
        } else {
            let cci = (typical_prices[i] - sma) / (constant * mad);
            result.push(cci);
        }
    }
    
    result
}

/// 平均真实波幅 (ATR)
pub fn average_true_range(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    if highs.len() != lows.len() || lows.len() != closes.len() || closes.len() < 2 {
        return Vec::new();
    }
    
    let mut true_ranges = Vec::new();
    
    // 计算真实波幅
    for i in 1..closes.len() {
        let high_low = highs[i] - lows[i];
        let high_close_prev = (highs[i] - closes[i - 1]).abs();
        let low_close_prev = (lows[i] - closes[i - 1]).abs();
        
        let tr = high_low.max(high_close_prev).max(low_close_prev);
        true_ranges.push(tr);
    }
    
    // 计算ATR（真实波幅的移动平均）
    exponential_moving_average(&true_ranges, period)
}

/// 动量指标 (Momentum)
pub fn momentum(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() <= period {
        return Vec::new();
    }
    
    let mut result = Vec::with_capacity(prices.len() - period);
    
    for i in period..prices.len() {
        let momentum = prices[i] - prices[i - period];
        result.push(momentum);
    }
    
    result
}

/// 变化率 (ROC - Rate of Change)
pub fn rate_of_change(prices: &[f64], period: usize) -> Vec<f64> {
    if prices.len() <= period {
        return Vec::new();
    }
    
    let mut result = Vec::with_capacity(prices.len() - period);
    
    for i in period..prices.len() {
        if prices[i - period] == 0.0 {
            result.push(0.0);
        } else {
            let roc = ((prices[i] - prices[i - period]) / prices[i - period]) * 100.0;
            result.push(roc);
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sma() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sma = simple_moving_average(&prices, 3);
        assert_eq!(sma, vec![2.0, 3.0, 4.0]);
    }
    
    #[test]
    fn test_ema() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ema = exponential_moving_average(&prices, 3);
        assert_eq!(ema.len(), 5);
        assert_eq!(ema[0], 1.0);
    }
    
    #[test]
    fn test_rsi() {
        let prices = vec![44.0, 44.25, 44.5, 43.75, 44.5, 45.0, 45.25, 45.5, 45.75, 46.0, 46.25, 46.5, 46.75, 47.0, 47.25];
        let rsi = relative_strength_index(&prices, 14);
        assert!(!rsi.is_empty());
        assert!(rsi[0] >= 0.0 && rsi[0] <= 100.0);
    }
    
    #[test]
    fn test_macd() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0];
        let (macd_line, signal_line, histogram) = macd(&prices, 12, 26, 9);
        assert!(!macd_line.is_empty());
        assert!(!signal_line.is_empty());
        assert!(!histogram.is_empty());
    }
}