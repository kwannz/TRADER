//! 工具模块
//! 
//! 提供通用的工具函数和辅助功能

use std::collections::HashMap;

/// 数学工具函数
pub mod math {
    /// 计算序列的均值
    pub fn mean(data: &[f64]) -> f64 {
        if data.is_empty() {
            return f64::NAN;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }
    
    /// 计算序列的标准差
    pub fn std_dev(data: &[f64]) -> f64 {
        if data.len() < 2 {
            return f64::NAN;
        }
        
        let mean_val = mean(data);
        let variance = data.iter()
            .map(|&x| (x - mean_val).powi(2))
            .sum::<f64>() / (data.len() - 1) as f64;
            
        variance.sqrt()
    }
    
    /// 计算序列的分位数
    pub fn quantile(data: &[f64], q: f64) -> f64 {
        if data.is_empty() || q < 0.0 || q > 1.0 {
            return f64::NAN;
        }
        
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = q * (sorted_data.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        
        if lower == upper {
            sorted_data[lower]
        } else {
            let weight = index - lower as f64;
            sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
        }
    }
    
    /// 计算中位数
    pub fn median(data: &[f64]) -> f64 {
        quantile(data, 0.5)
    }
    
    /// 计算序列的偏度 (Skewness)
    pub fn skewness(data: &[f64]) -> f64 {
        if data.len() < 3 {
            return f64::NAN;
        }
        
        let mean_val = mean(data);
        let std_val = std_dev(data);
        
        if std_val == 0.0 {
            return f64::NAN;
        }
        
        let n = data.len() as f64;
        let sum_cubed = data.iter()
            .map(|&x| ((x - mean_val) / std_val).powi(3))
            .sum::<f64>();
            
        (n / ((n - 1.0) * (n - 2.0))) * sum_cubed
    }
    
    /// 计算序列的峰度 (Kurtosis)
    pub fn kurtosis(data: &[f64]) -> f64 {
        if data.len() < 4 {
            return f64::NAN;
        }
        
        let mean_val = mean(data);
        let std_val = std_dev(data);
        
        if std_val == 0.0 {
            return f64::NAN;
        }
        
        let n = data.len() as f64;
        let sum_fourth = data.iter()
            .map(|&x| ((x - mean_val) / std_val).powi(4))
            .sum::<f64>();
            
        let numerator = n * (n + 1.0) * sum_fourth;
        let denominator = (n - 1.0) * (n - 2.0) * (n - 3.0);
        let correction = 3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0));
        
        (numerator / denominator) - correction
    }
}

/// 性能监控工具
pub mod performance {
    use std::time::{Duration, Instant};
    
    /// 简单的性能计时器
    pub struct Timer {
        start: Instant,
        name: String,
    }
    
    impl Timer {
        pub fn new(name: &str) -> Self {
            Self {
                start: Instant::now(),
                name: name.to_string(),
            }
        }
        
        pub fn elapsed(&self) -> Duration {
            self.start.elapsed()
        }
        
        pub fn elapsed_ms(&self) -> u128 {
            self.elapsed().as_millis()
        }
        
        pub fn elapsed_us(&self) -> u128 {
            self.elapsed().as_micros()
        }
        
        pub fn print_elapsed(&self) {
            println!("{}: {:?}", self.name, self.elapsed());
        }
    }
    
    impl Drop for Timer {
        fn drop(&mut self) {
            self.print_elapsed();
        }
    }
}

/// 缓存工具
pub mod cache {
    use std::collections::HashMap;
    use std::hash::Hash;
    
    /// 简单的LRU缓存
    pub struct LruCache<K, V> {
        capacity: usize,
        map: HashMap<K, V>,
        order: Vec<K>,
    }
    
    impl<K: Clone + Eq + Hash, V> LruCache<K, V> {
        pub fn new(capacity: usize) -> Self {
            Self {
                capacity,
                map: HashMap::new(),
                order: Vec::new(),
            }
        }
        
        pub fn get(&mut self, key: &K) -> Option<&V> {
            if let Some(value) = self.map.get(key) {
                // 移动到最前面
                self.order.retain(|k| k != key);
                self.order.push(key.clone());
                Some(value)
            } else {
                None
            }
        }
        
        pub fn put(&mut self, key: K, value: V) {
            if self.map.contains_key(&key) {
                // 更新现有键
                self.map.insert(key.clone(), value);
                self.order.retain(|k| k != &key);
                self.order.push(key);
            } else {
                // 添加新键
                if self.map.len() >= self.capacity {
                    // 移除最久未使用的项
                    if let Some(oldest) = self.order.first().cloned() {
                        self.map.remove(&oldest);
                        self.order.remove(0);
                    }
                }
                self.map.insert(key.clone(), value);
                self.order.push(key);
            }
        }
        
        pub fn len(&self) -> usize {
            self.map.len()
        }
        
        pub fn is_empty(&self) -> bool {
            self.map.is_empty()
        }
    }
}

/// 数据验证工具
pub mod validation {
    /// 检查价格数据的有效性
    pub fn validate_prices(prices: &[f64]) -> Result<(), String> {
        if prices.is_empty() {
            return Err("价格数据为空".to_string());
        }
        
        for (i, &price) in prices.iter().enumerate() {
            if price.is_nan() {
                return Err(format!("索引{}处的价格为NaN", i));
            }
            if price < 0.0 {
                return Err(format!("索引{}处的价格为负数: {}", i, price));
            }
        }
        
        Ok(())
    }
    
    /// 检查周期参数的有效性
    pub fn validate_period(period: usize, data_length: usize) -> Result<(), String> {
        if period == 0 {
            return Err("周期不能为0".to_string());
        }
        if period > data_length {
            return Err(format!("周期({})超过数据长度({})", period, data_length));
        }
        
        Ok(())
    }
    
    /// 检查参数范围
    pub fn validate_range(value: f64, min: f64, max: f64, name: &str) -> Result<(), String> {
        if value < min || value > max {
            return Err(format!("{}({})超出范围[{}, {}]", name, value, min, max));
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_math_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = math::mean(&data);
        assert_relative_eq!(result, 3.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_math_std_dev() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = math::std_dev(&data);
        // 期望标准差约为1.58
        assert!((result - 1.58).abs() < 0.1);
    }
    
    #[test]
    fn test_lru_cache() {
        let mut cache = cache::LruCache::new(2);
        
        cache.put("a", 1);
        cache.put("b", 2);
        
        assert_eq!(cache.get(&"a"), Some(&1));
        assert_eq!(cache.get(&"b"), Some(&2));
        
        // 添加第三个元素，应该移除最久未使用的
        cache.put("c", 3);
        assert_eq!(cache.get(&"a"), None); // 'a' 应该被移除
        assert_eq!(cache.get(&"b"), Some(&2));
        assert_eq!(cache.get(&"c"), Some(&3));
    }
}