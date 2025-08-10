//! 工具模块
//! 
//! 提供数学计算、技术指标、Alpha因子等工具函数

pub mod math;
pub mod time;
pub mod indicators;
pub mod alpha_factors;

use chrono::{DateTime, Utc};
use std::collections::VecDeque;

/// 高性能滑动窗口计算器
pub struct SlidingWindow<T> {
    window: VecDeque<T>,
    capacity: usize,
    sum: f64,
}

impl SlidingWindow<f64> {
    /// 创建新的滑动窗口
    pub fn new(capacity: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(capacity),
            capacity,
            sum: 0.0,
        }
    }
    
    /// 添加新值并返回当前平均值
    pub fn push(&mut self, value: f64) -> Option<f64> {
        if self.window.len() == self.capacity {
            if let Some(old_value) = self.window.pop_front() {
                self.sum -= old_value;
            }
        }
        
        self.window.push_back(value);
        self.sum += value;
        
        if self.window.len() == self.capacity {
            Some(self.sum / self.capacity as f64)
        } else {
            None
        }
    }
    
    /// 获取当前窗口大小
    pub fn len(&self) -> usize {
        self.window.len()
    }
    
    /// 检查窗口是否为空
    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }
    
    /// 获取当前平均值
    pub fn mean(&self) -> Option<f64> {
        if self.window.is_empty() {
            None
        } else {
            Some(self.sum / self.window.len() as f64)
        }
    }
}

/// 性能计数器
#[derive(Debug, Clone)]
pub struct PerformanceCounter {
    pub name: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub iterations: u64,
    pub total_time_ms: f64,
}

impl PerformanceCounter {
    /// 创建新的性能计数器
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            start_time: Utc::now(),
            end_time: None,
            iterations: 0,
            total_time_ms: 0.0,
        }
    }
    
    /// 开始计时
    pub fn start(&mut self) {
        self.start_time = Utc::now();
    }
    
    /// 结束计时并记录一次迭代
    pub fn stop(&mut self) {
        self.end_time = Some(Utc::now());
        if let Some(end) = self.end_time {
            let duration = (end - self.start_time).num_microseconds().unwrap_or(0) as f64 / 1000.0;
            self.total_time_ms += duration;
            self.iterations += 1;
        }
    }
    
    /// 获取平均执行时间（毫秒）
    pub fn average_time_ms(&self) -> f64 {
        if self.iterations > 0 {
            self.total_time_ms / self.iterations as f64
        } else {
            0.0
        }
    }
    
    /// 获取每秒操作数
    pub fn ops_per_second(&self) -> f64 {
        let avg_time_s = self.average_time_ms() / 1000.0;
        if avg_time_s > 0.0 {
            1.0 / avg_time_s
        } else {
            0.0
        }
    }
}

/// 内存池管理器
pub struct ObjectPool<T> {
    objects: Vec<T>,
    create_fn: Box<dyn Fn() -> T + Send + Sync>,
    reset_fn: Box<dyn Fn(&mut T) + Send + Sync>,
}

impl<T> ObjectPool<T>
where
    T: Send + Sync,
{
    /// 创建新的对象池
    pub fn new<F, R>(create_fn: F, reset_fn: R) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
        R: Fn(&mut T) + Send + Sync + 'static,
    {
        Self {
            objects: Vec::new(),
            create_fn: Box::new(create_fn),
            reset_fn: Box::new(reset_fn),
        }
    }
    
    /// 从池中获取对象
    pub fn get(&mut self) -> T {
        match self.objects.pop() {
            Some(mut obj) => {
                (self.reset_fn)(&mut obj);
                obj
            }
            None => (self.create_fn)(),
        }
    }
    
    /// 将对象返回到池中
    pub fn put(&mut self, obj: T) {
        self.objects.push(obj);
    }
    
    /// 获取池中对象数量
    pub fn len(&self) -> usize {
        self.objects.len()
    }
    
    /// 检查池是否为空
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

/// 线程安全的统计收集器
#[derive(Debug, Clone)]
pub struct StatsCollector {
    pub count: u64,
    pub sum: f64,
    pub sum_squares: f64,
    pub min: f64,
    pub max: f64,
}

impl StatsCollector {
    /// 创建新的统计收集器
    pub fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            sum_squares: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }
    
    /// 添加数值
    pub fn add(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.sum_squares += value * value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }
    
    /// 获取平均值
    pub fn mean(&self) -> f64 {
        if self.count > 0 {
            self.sum / self.count as f64
        } else {
            0.0
        }
    }
    
    /// 获取标准差
    pub fn std_dev(&self) -> f64 {
        if self.count > 1 {
            let mean = self.mean();
            let variance = (self.sum_squares - self.count as f64 * mean * mean) / (self.count - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        }
    }
    
    /// 获取变异系数
    pub fn coefficient_of_variation(&self) -> f64 {
        let mean = self.mean();
        if mean != 0.0 {
            self.std_dev() / mean.abs()
        } else {
            0.0
        }
    }
}

impl Default for StatsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// 高性能哈希映射工具
use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};

/// 快速哈希器（用于性能关键场景）
pub struct FastHasher {
    hash: u64,
}

impl Default for FastHasher {
    fn default() -> Self {
        Self { hash: 0 }
    }
}

impl Hasher for FastHasher {
    fn finish(&self) -> u64 {
        self.hash
    }
    
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.hash = self.hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
    }
}

/// 快速哈希构建器
#[derive(Default)]
pub struct FastHashBuilder;

impl BuildHasher for FastHashBuilder {
    type Hasher = FastHasher;
    
    fn build_hasher(&self) -> Self::Hasher {
        FastHasher::default()
    }
}

/// 高性能HashMap类型别名
pub type FastHashMap<K, V> = HashMap<K, V, FastHashBuilder>;

/// 创建快速HashMap
pub fn fast_hashmap<K, V>() -> FastHashMap<K, V> {
    HashMap::with_hasher(FastHashBuilder)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sliding_window() {
        let mut window = SlidingWindow::new(3);
        
        assert_eq!(window.push(1.0), None);
        assert_eq!(window.push(2.0), None);
        assert_eq!(window.push(3.0), Some(2.0)); // (1+2+3)/3 = 2.0
        assert_eq!(window.push(4.0), Some(3.0)); // (2+3+4)/3 = 3.0
    }
    
    #[test]
    fn test_stats_collector() {
        let mut stats = StatsCollector::new();
        
        stats.add(1.0);
        stats.add(2.0);
        stats.add(3.0);
        
        assert_eq!(stats.mean(), 2.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 3.0);
        assert!(stats.std_dev() > 0.0);
    }
    
    #[test]
    fn test_fast_hashmap() {
        let mut map = fast_hashmap();
        map.insert("key1", "value1");
        map.insert("key2", "value2");
        
        assert_eq!(map.get("key1"), Some(&"value1"));
        assert_eq!(map.len(), 2);
    }
}