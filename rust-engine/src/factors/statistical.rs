//! 统计因子
//! 
//! 实现各种统计分析因子，如相关性、回归等

use ndarray::Array1;
use statrs::statistics::*;

/// 计算价格收益率
/// 
/// # 参数
/// * `prices` - 价格序列
/// * `method` - 计算方法："simple"或"log"
/// 
/// # 返回
/// 收益率序列
pub fn returns(prices: &[f64], method: &str) -> Vec<f64> {
    if prices.len() < 2 {
        return Vec::new();
    }

    let mut result = Vec::with_capacity(prices.len() - 1);
    
    match method {
        "simple" => {
            for i in 1..prices.len() {
                if prices[i - 1] != 0.0 {
                    result.push((prices[i] - prices[i - 1]) / prices[i - 1]);
                } else {
                    result.push(f64::NAN);
                }
            }
        },
        "log" => {
            for i in 1..prices.len() {
                if prices[i] > 0.0 && prices[i - 1] > 0.0 {
                    result.push((prices[i] / prices[i - 1]).ln());
                } else {
                    result.push(f64::NAN);
                }
            }
        },
        _ => panic!("Invalid method. Use 'simple' or 'log'"),
    }

    result
}

/// 计算滚动相关系数
/// 
/// # 参数
/// * `x` - 第一个序列
/// * `y` - 第二个序列
/// * `window` - 滚动窗口大小
/// 
/// # 返回
/// 滚动相关系数序列
pub fn rolling_correlation(x: &[f64], y: &[f64], window: usize) -> Vec<f64> {
    assert_eq!(x.len(), y.len());
    
    if x.len() < window {
        return vec![f64::NAN; x.len()];
    }

    let mut result = vec![f64::NAN; window - 1];
    
    for i in window..=x.len() {
        let x_window = &x[i - window..i];
        let y_window = &y[i - window..i];
        
        let corr = correlation(x_window, y_window);
        result.push(corr);
    }

    result
}

/// 计算两个序列的相关系数
fn correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return f64::NAN;
    }

    let mean_x = x.iter().sum::<f64>() / x.len() as f64;
    let mean_y = y.iter().sum::<f64>() / y.len() as f64;

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
        f64::NAN
    } else {
        numerator / denominator
    }
}

/// 计算滚动贝塔系数
/// 
/// # 参数
/// * `asset_returns` - 资产收益率
/// * `market_returns` - 市场收益率
/// * `window` - 滚动窗口大小
/// 
/// # 返回
/// 滚动贝塔系数序列
pub fn rolling_beta(
    asset_returns: &[f64], 
    market_returns: &[f64], 
    window: usize
) -> Vec<f64> {
    assert_eq!(asset_returns.len(), market_returns.len());
    
    if asset_returns.len() < window {
        return vec![f64::NAN; asset_returns.len()];
    }

    let mut result = vec![f64::NAN; window - 1];
    
    for i in window..=asset_returns.len() {
        let asset_window = &asset_returns[i - window..i];
        let market_window = &market_returns[i - window..i];
        
        let beta = calculate_beta(asset_window, market_window);
        result.push(beta);
    }

    result
}

/// 计算贝塔系数
fn calculate_beta(asset_returns: &[f64], market_returns: &[f64]) -> f64 {
    if asset_returns.len() != market_returns.len() || asset_returns.len() < 2 {
        return f64::NAN;
    }

    let market_mean = market_returns.iter().sum::<f64>() / market_returns.len() as f64;
    let asset_mean = asset_returns.iter().sum::<f64>() / asset_returns.len() as f64;

    let mut covariance = 0.0;
    let mut market_variance = 0.0;

    for i in 0..market_returns.len() {
        let market_diff = market_returns[i] - market_mean;
        let asset_diff = asset_returns[i] - asset_mean;
        
        covariance += market_diff * asset_diff;
        market_variance += market_diff * market_diff;
    }

    if market_variance == 0.0 {
        f64::NAN
    } else {
        covariance / market_variance
    }
}

/// 计算滚动夏普比率
/// 
/// # 参数
/// * `returns` - 收益率序列
/// * `risk_free_rate` - 无风险利率
/// * `window` - 滚动窗口大小
/// 
/// # 返回
/// 滚动夏普比率序列
pub fn rolling_sharpe_ratio(
    returns: &[f64], 
    risk_free_rate: f64, 
    window: usize
) -> Vec<f64> {
    if returns.len() < window {
        return vec![f64::NAN; returns.len()];
    }

    let mut result = vec![f64::NAN; window - 1];
    
    for i in window..=returns.len() {
        let window_returns = &returns[i - window..i];
        
        let mean_return = window_returns.iter().sum::<f64>() / window_returns.len() as f64;
        let excess_return = mean_return - risk_free_rate;
        
        // 计算标准差
        let variance = window_returns.iter()
            .map(|&r| (r - mean_return).powi(2))
            .sum::<f64>() / (window_returns.len() - 1) as f64;
        let std_dev = variance.sqrt();
        
        let sharpe = if std_dev == 0.0 {
            f64::NAN
        } else {
            excess_return / std_dev
        };
        
        result.push(sharpe);
    }

    result
}

/// 计算最大回撤
/// 
/// # 参数
/// * `prices` - 价格序列
/// 
/// # 返回
/// 最大回撤值
pub fn max_drawdown(prices: &[f64]) -> f64 {
    if prices.is_empty() {
        return 0.0;
    }

    let mut max_price = prices[0];
    let mut max_dd = 0.0;

    for &price in prices.iter() {
        if price > max_price {
            max_price = price;
        }
        
        let drawdown = (max_price - price) / max_price;
        if drawdown > max_dd {
            max_dd = drawdown;
        }
    }

    max_dd
}

/// 计算滚动波动率
/// 
/// # 参数
/// * `returns` - 收益率序列
/// * `window` - 滚动窗口大小
/// * `annualize` - 是否年化（假设252个交易日）
/// 
/// # 返回
/// 滚动波动率序列
pub fn rolling_volatility(
    returns: &[f64], 
    window: usize, 
    annualize: bool
) -> Vec<f64> {
    if returns.len() < window {
        return vec![f64::NAN; returns.len()];
    }

    let mut result = vec![f64::NAN; window - 1];
    let annualize_factor = if annualize { 252.0_f64.sqrt() } else { 1.0 };
    
    for i in window..=returns.len() {
        let window_returns = &returns[i - window..i];
        
        let mean_return = window_returns.iter().sum::<f64>() / window_returns.len() as f64;
        let variance = window_returns.iter()
            .map(|&r| (r - mean_return).powi(2))
            .sum::<f64>() / (window_returns.len() - 1) as f64;
        
        let volatility = variance.sqrt() * annualize_factor;
        result.push(volatility);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simple_returns() {
        let prices = vec![100.0, 105.0, 110.0, 102.0];
        let result = returns(&prices, "simple");
        
        assert_relative_eq!(result[0], 0.05, epsilon = 1e-10);
        assert_relative_eq!(result[1], 105.0/105.0 - 1.0 + 110.0/105.0 - 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let result = correlation(&x, &y);
        
        // 完全正相关应该接近1.0
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_max_drawdown() {
        let prices = vec![100.0, 110.0, 105.0, 95.0, 120.0];
        let result = max_drawdown(&prices);
        
        // 最大回撤应该是从110到95，即(110-95)/110 ≈ 0.136
        assert_relative_eq!(result, 15.0/110.0, epsilon = 1e-10);
    }
}