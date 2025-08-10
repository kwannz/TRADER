//! Alpha因子计算模块
//! 
//! 实现Alpha101因子和自定义Alpha因子

use std::collections::HashMap;

/// Alpha001: (-1 * correlation(rank(delta(log(volume), 1)), rank(((close - open) / open)), 6))
pub fn alpha_001(close: &[f64], volume: &[f64]) -> f64 {
    if close.len() < 7 || volume.len() != close.len() {
        return 0.0;
    }
    
    // 计算 log(volume) 的差分
    let log_volume_delta: Vec<f64> = (1..volume.len())
        .map(|i| volume[i].ln() - volume[i-1].ln())
        .collect();
    
    // 计算 (close - open) / open，这里假设 open 约等于前一个 close
    let open_close_ratio: Vec<f64> = (1..close.len())
        .map(|i| (close[i] - close[i-1]) / close[i-1])
        .collect();
    
    if log_volume_delta.len() < 6 || open_close_ratio.len() < 6 {
        return 0.0;
    }
    
    // 计算最近6期的相关系数
    let recent_log_vol_delta = &log_volume_delta[log_volume_delta.len()-6..];
    let recent_open_close_ratio = &open_close_ratio[open_close_ratio.len()-6..];
    
    let correlation = calculate_correlation(recent_log_vol_delta, recent_open_close_ratio);
    
    -1.0 * correlation
}

/// Alpha002: (-1 * delta((((close - low) - (high - close)) / (high - low)), 1))
pub fn alpha_002(close: &[f64], _volume: &[f64]) -> f64 {
    if close.len() < 2 {
        return 0.0;
    }
    
    // 这里简化处理，假设 high 和 low 基于 close 价格的波动
    let high = close.iter().map(|&x| x * 1.01).collect::<Vec<_>>();
    let low = close.iter().map(|&x| x * 0.99).collect::<Vec<_>>();
    
    let current_idx = close.len() - 1;
    let prev_idx = close.len() - 2;
    
    let current_val = ((close[current_idx] - low[current_idx]) - (high[current_idx] - close[current_idx])) / (high[current_idx] - low[current_idx]);
    let prev_val = ((close[prev_idx] - low[prev_idx]) - (high[prev_idx] - close[prev_idx])) / (high[prev_idx] - low[prev_idx]);
    
    -1.0 * (current_val - prev_val)
}

/// Alpha003: SUM((close=delay(close,1)?0:close-(close>delay(close,1)?min(low,delay(close,1)):max(high,delay(close,1)))),6)
pub fn alpha_003(close: &[f64], _volume: &[f64]) -> f64 {
    if close.len() < 7 {
        return 0.0;
    }
    
    let high = close.iter().map(|&x| x * 1.01).collect::<Vec<_>>();
    let low = close.iter().map(|&x| x * 0.99).collect::<Vec<_>>();
    
    let mut sum = 0.0;
    let start_idx = close.len() - 6;
    
    for i in start_idx..close.len() {
        if i == 0 {
            continue;
        }
        
        let current_close = close[i];
        let prev_close = close[i - 1];
        
        if current_close == prev_close {
            // close = delay(close, 1) 的情况
            sum += 0.0;
        } else if current_close > prev_close {
            // close > delay(close, 1) 的情况
            let min_val = low[i].min(prev_close);
            sum += current_close - min_val;
        } else {
            // close < delay(close, 1) 的情况
            let max_val = high[i].max(prev_close);
            sum += current_close - max_val;
        }
    }
    
    sum
}

/// Alpha004: ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume / adv20) < 1)) ? 1 : (-1 * 1))))
pub fn alpha_004(close: &[f64], high: &[f64], low: &[f64], volume: &[f64]) -> f64 {
    if close.len() < 20 || volume.len() < 20 {
        return 0.0;
    }
    
    let len = close.len();
    
    // 计算 sum(close, 8) / 8
    let sum8_avg = close[len-8..].iter().sum::<f64>() / 8.0;
    
    // 计算 stddev(close, 8)
    let stddev8 = {
        let mean = sum8_avg;
        let variance = close[len-8..].iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / 8.0;
        variance.sqrt()
    };
    
    // 计算 sum(close, 2) / 2
    let sum2_avg = close[len-2..].iter().sum::<f64>() / 2.0;
    
    // 计算 adv20 (20日平均成交量)
    let adv20 = volume[len-20..].iter().sum::<f64>() / 20.0;
    
    // 当前成交量
    let current_volume = volume[len-1];
    
    // 条件判断
    if (sum8_avg + stddev8) < sum2_avg {
        -1.0
    } else if sum2_avg < (sum8_avg - stddev8) {
        1.0
    } else if (1.0 < (current_volume / adv20)) || ((current_volume / adv20) < 1.0) {
        1.0
    } else {
        -1.0
    }
}

/// Alpha005: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
pub fn alpha_005(close: &[f64], volume: &[f64]) -> f64 {
    if close.len() < 10 || volume.len() != close.len() {
        return 0.0;
    }
    
    // 简化：假设 vwap 约等于 close，open 约等于前一个 close
    let vwap = close; // 简化处理
    let open: Vec<f64> = std::iter::once(close[0])
        .chain(close[..close.len()-1].iter().cloned())
        .collect();
    
    let len = close.len();
    
    // 计算 sum(vwap, 10) / 10
    let vwap_avg_10 = vwap[len-10..].iter().sum::<f64>() / 10.0;
    
    // open - (sum(vwap, 10) / 10)
    let open_minus_vwap_avg = open[len-1] - vwap_avg_10;
    
    // close - vwap
    let close_minus_vwap = close[len-1] - vwap[len-1];
    
    // 这里简化 rank 函数处理
    let rank1 = open_minus_vwap_avg;
    let rank2 = close_minus_vwap.abs();
    
    rank1 * (-1.0 * rank2)
}

/// 自定义因子：动量RSI因子
pub fn momentum_rsi_factor(close: &[f64], rsi_period: usize, momentum_period: usize) -> f64 {
    if close.len() < rsi_period.max(momentum_period) + 1 {
        return 0.0;
    }
    
    // 计算 RSI
    let rsi = crate::utils::indicators::relative_strength_index(close, rsi_period);
    if rsi.is_empty() {
        return 0.0;
    }
    
    // 计算动量
    let momentum = crate::utils::indicators::momentum(close, momentum_period);
    if momentum.is_empty() {
        return 0.0;
    }
    
    let current_rsi = rsi[rsi.len() - 1];
    let current_momentum = momentum[momentum.len() - 1];
    
    // 综合因子：RSI偏离中线的程度 * 动量强度
    let rsi_deviation = (current_rsi - 50.0) / 50.0; // 标准化到 [-1, 1]
    let momentum_normalized = current_momentum / close[close.len()-1]; // 动量相对强度
    
    rsi_deviation * momentum_normalized
}

/// 自定义因子：波动率突破因子
pub fn volatility_breakout_factor(close: &[f64], volume: &[f64], period: usize) -> f64 {
    if close.len() < period + 1 || volume.len() != close.len() {
        return 0.0;
    }
    
    let len = close.len();
    
    // 计算历史波动率
    let returns: Vec<f64> = (1..len)
        .map(|i| (close[i] / close[i-1]).ln())
        .collect();
    
    if returns.len() < period {
        return 0.0;
    }
    
    let recent_returns = &returns[returns.len()-period..];
    let mean_return = recent_returns.iter().sum::<f64>() / period as f64;
    let volatility = {
        let variance = recent_returns.iter()
            .map(|x| (x - mean_return).powi(2))
            .sum::<f64>() / period as f64;
        variance.sqrt()
    };
    
    // 当前价格变化
    let current_return = returns[returns.len() - 1];
    
    // 成交量放大
    let recent_volume = volume[len-period..].iter().sum::<f64>() / period as f64;
    let current_volume = volume[len-1];
    let volume_ratio = current_volume / recent_volume;
    
    // 突破因子：价格突破程度 * 成交量放大
    let price_breakout = current_return.abs() / volatility; // 多少倍标准差
    
    if price_breakout > 2.0 && volume_ratio > 1.5 {
        if current_return > 0.0 { 1.0 } else { -1.0 } * price_breakout * volume_ratio.min(3.0)
    } else {
        0.0
    }
}

/// 自定义因子：均值回归因子
pub fn mean_reversion_factor(close: &[f64], period: usize, threshold: f64) -> f64 {
    if close.len() < period + 1 {
        return 0.0;
    }
    
    let len = close.len();
    
    // 计算移动平均
    let recent_prices = &close[len-period..];
    let mean_price = recent_prices.iter().sum::<f64>() / period as f64;
    
    // 当前价格偏离程度
    let current_price = close[len-1];
    let deviation = (current_price - mean_price) / mean_price;
    
    // 计算价格标准差
    let std_dev = {
        let variance = recent_prices.iter()
            .map(|x| (x - mean_price).powi(2))
            .sum::<f64>() / period as f64;
        variance.sqrt()
    };
    
    let normalized_deviation = deviation / (std_dev / mean_price);
    
    // 均值回归信号：偏离超过阈值时产生反向信号
    if normalized_deviation.abs() > threshold {
        -1.0 * normalized_deviation.signum() * normalized_deviation.abs().min(3.0)
    } else {
        0.0
    }
}

/// 计算两个序列的相关系数
fn calculate_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    
    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;
    
    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }
    
    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// 批量计算多个Alpha因子
pub fn calculate_alpha_batch(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    volume: &[f64]
) -> HashMap<String, f64> {
    let mut results = HashMap::new();
    
    // Alpha101 因子
    results.insert("alpha001".to_string(), alpha_001(close, volume));
    results.insert("alpha002".to_string(), alpha_002(close, volume));
    results.insert("alpha003".to_string(), alpha_003(close, volume));
    results.insert("alpha004".to_string(), alpha_004(close, high, low, volume));
    results.insert("alpha005".to_string(), alpha_005(close, volume));
    
    // 自定义因子
    results.insert("momentum_rsi".to_string(), momentum_rsi_factor(close, 14, 10));
    results.insert("volatility_breakout".to_string(), volatility_breakout_factor(close, volume, 20));
    results.insert("mean_reversion".to_string(), mean_reversion_factor(close, 20, 2.0));
    
    // 计算因子的组合得分
    let factor_values: Vec<f64> = results.values().cloned().collect();
    let combined_score = if !factor_values.is_empty() {
        let sum = factor_values.iter().sum::<f64>();
        let count = factor_values.len() as f64;
        sum / count
    } else {
        0.0
    };
    
    results.insert("combined_alpha".to_string(), combined_score);
    
    results
}

/// 因子标准化
pub fn normalize_factor(factor_value: f64, historical_values: &[f64]) -> f64 {
    if historical_values.is_empty() {
        return 0.0;
    }
    
    let mean = historical_values.iter().sum::<f64>() / historical_values.len() as f64;
    let variance = historical_values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / historical_values.len() as f64;
    
    let std_dev = variance.sqrt();
    
    if std_dev == 0.0 {
        0.0
    } else {
        (factor_value - mean) / std_dev
    }
}

/// 因子有效性检验
pub fn factor_validity_test(factor_values: &[f64], returns: &[f64]) -> (f64, f64) {
    if factor_values.len() != returns.len() || factor_values.len() < 10 {
        return (0.0, 0.0);
    }
    
    // 计算IC (信息系数)
    let ic = calculate_correlation(factor_values, returns);
    
    // 计算RankIC
    let mut factor_ranks: Vec<(f64, usize)> = factor_values.iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();
    factor_ranks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
    let mut return_ranks: Vec<(f64, usize)> = returns.iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();
    return_ranks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
    // 计算排名
    let mut factor_rank_values = vec![0.0; factor_values.len()];
    let mut return_rank_values = vec![0.0; returns.len()];
    
    for (rank, (_, idx)) in factor_ranks.iter().enumerate() {
        factor_rank_values[*idx] = rank as f64;
    }
    
    for (rank, (_, idx)) in return_ranks.iter().enumerate() {
        return_rank_values[*idx] = rank as f64;
    }
    
    let rank_ic = calculate_correlation(&factor_rank_values, &return_rank_values);
    
    (ic, rank_ic)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_alpha_001() {
        let close = vec![100.0, 101.0, 102.0, 103.0, 104.0, 103.0, 102.0, 101.0];
        let volume = vec![1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1300.0, 1200.0, 1100.0];
        
        let result = alpha_001(&close, &volume);
        assert!(result.is_finite());
    }
    
    #[test]
    fn test_momentum_rsi_factor() {
        let close = vec![100.0, 101.0, 102.0, 103.0, 104.0, 103.0, 102.0, 101.0, 100.0, 99.0, 98.0, 97.0, 98.0, 99.0, 100.0, 101.0];
        
        let result = momentum_rsi_factor(&close, 14, 10);
        assert!(result.is_finite());
    }
    
    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let corr = calculate_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 1e-10); // 完全正相关
    }
    
    #[test]
    fn test_factor_validity() {
        let factors = vec![0.1, 0.2, -0.1, 0.3, -0.2, 0.0, 0.1, -0.1];
        let returns = vec![0.01, 0.02, -0.01, 0.03, -0.02, 0.00, 0.01, -0.01];
        
        let (ic, rank_ic) = factor_validity_test(&factors, &returns);
        assert!(ic.is_finite());
        assert!(rank_ic.is_finite());
    }
}