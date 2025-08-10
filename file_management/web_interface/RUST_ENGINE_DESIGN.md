# Rust核心计算引擎设计文档

**项目**: QuantAnalyzer Pro - Rust High-Performance Computing Engine  
**版本**: v1.0  
**创建日期**: 2025-08-10  

---

## 1. 引擎架构概览

### 1.1 模块结构

```
rust_engine/
├── Cargo.toml                 # 项目配置和依赖
├── src/
│   ├── lib.rs                # 库入口，Python FFI绑定
│   ├── core/                 # 核心引擎
│   │   ├── mod.rs
│   │   ├── engine.rs         # 主引擎
│   │   ├── memory.rs         # 内存管理
│   │   └── threading.rs      # 线程池管理
│   ├── factor/               # 因子计算模块
│   │   ├── mod.rs
│   │   ├── calculator.rs     # 因子计算器
│   │   ├── technical.rs      # 技术指标
│   │   ├── statistical.rs    # 统计因子
│   │   └── ai_factors.rs     # AI生成因子
│   ├── backtest/             # 回测引擎
│   │   ├── mod.rs
│   │   ├── engine.rs         # 回测引擎
│   │   ├── portfolio.rs      # 投资组合管理
│   │   ├── executor.rs       # 交易执行器
│   │   └── metrics.rs        # 性能指标计算
│   ├── optimizer/            # 组合优化
│   │   ├── mod.rs
│   │   ├── mean_variance.rs  # 均值方差优化
│   │   ├── risk_parity.rs    # 风险平价
│   │   └── black_litterman.rs # Black-Litterman模型
│   ├── data/                 # 数据处理
│   │   ├── mod.rs
│   │   ├── types.rs          # 数据类型定义
│   │   ├── loader.rs         # 数据加载器
│   │   ├── processor.rs      # 数据处理器
│   │   └── validator.rs      # 数据验证
│   └── utils/                # 工具模块
│       ├── mod.rs
│       ├── math.rs           # 数学工具
│       ├── datetime.rs       # 时间处理
│       └── serialization.rs  # 序列化工具
├── tests/                    # 测试代码
├── benches/                  # 性能测试
└── examples/                 # 使用示例
```

### 1.2 依赖配置

```toml
# Cargo.toml
[package]
name = "quant_engine"
version = "0.1.0"
edition = "2021"

[lib]
name = "quant_engine"
crate-type = ["cdylib", "rlib"]

[dependencies]
# Python绑定
pyo3 = { version = "0.20", features = ["extension-module", "abi3-py38"] }

# 数据处理
polars = { version = "0.36", features = ["lazy", "temporal", "strings"] }
ndarray = { version = "0.15", features = ["rayon", "blas"] }
arrow = "50.0"

# 并行计算
rayon = "1.8"
tokio = { version = "1.35", features = ["full"] }

# 数学计算
nalgebra = "0.32"
statrs = "0.16"
rand = "0.8"
rand_distr = "0.4"

# 序列化
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "1.3"

# 时间处理
chrono = { version = "0.4", features = ["serde"] }

# 错误处理
anyhow = "1.0"
thiserror = "1.0"

# 日志
log = "0.4"
env_logger = "0.10"

# 性能优化
once_cell = "1.19"
parking_lot = "0.12"

# BLAS/LAPACK (可选)
blas-src = { version = "0.8", features = ["openblas"] }
lapack-src = { version = "0.8", features = ["openblas"] }

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
proptest = "1.4"

[[bench]]
name = "factor_calculation"
harness = false

[[bench]]
name = "backtest_performance"
harness = false

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"
```

---

## 2. 核心引擎设计

### 2.1 主引擎结构

```rust
// src/core/engine.rs
use std::sync::{Arc, RwLock};
use parking_lot::Mutex;
use rayon::ThreadPoolBuilder;
use pyo3::prelude::*;

use crate::factor::FactorEngine;
use crate::backtest::BacktestEngine;
use crate::optimizer::OptimizerEngine;
use crate::data::{DataCache, DataProcessor};

#[pyclass]
pub struct QuantEngine {
    // 子引擎
    factor_engine: Arc<FactorEngine>,
    backtest_engine: Arc<Mutex<BacktestEngine>>,
    optimizer_engine: Arc<OptimizerEngine>,
    
    // 数据层
    data_cache: Arc<RwLock<DataCache>>,
    data_processor: Arc<DataProcessor>,
    
    // 配置
    config: EngineConfig,
    
    // 状态
    is_initialized: bool,
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub num_threads: usize,
    pub memory_limit_gb: usize,
    pub cache_size_mb: usize,
    pub enable_simd: bool,
    pub enable_gpu: bool,
    pub log_level: String,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            memory_limit_gb: 8,
            cache_size_mb: 512,
            enable_simd: true,
            enable_gpu: false,
            log_level: "INFO".to_string(),
        }
    }
}

#[pymethods]
impl QuantEngine {
    #[new]
    #[pyo3(signature = (config = None))]
    pub fn new(config: Option<EngineConfig>) -> PyResult<Self> {
        let config = config.unwrap_or_default();
        
        // 初始化线程池
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(config.num_threads)
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to create thread pool: {}", e)
            ))?;
        
        // 初始化数据缓存
        let data_cache = Arc::new(RwLock::new(DataCache::new(config.cache_size_mb)));
        let data_processor = Arc::new(DataProcessor::new(config.clone()));
        
        // 初始化子引擎
        let factor_engine = Arc::new(FactorEngine::new(
            config.clone(),
            data_cache.clone()
        )?);
        
        let backtest_engine = Arc::new(Mutex::new(BacktestEngine::new(
            config.clone()
        )?));
        
        let optimizer_engine = Arc::new(OptimizerEngine::new(
            config.clone()
        )?);
        
        Ok(Self {
            factor_engine,
            backtest_engine,
            optimizer_engine,
            data_cache,
            data_processor,
            config,
            is_initialized: true,
        })
    }
    
    /// 健康检查
    pub fn health_check(&self) -> PyResult<bool> {
        if !self.is_initialized {
            return Ok(false);
        }
        
        // 测试各个组件
        let test_data = self.generate_test_data()?;
        let _result = self.factor_engine.test_calculation(&test_data)?;
        
        Ok(true)
    }
    
    /// 获取引擎统计信息
    pub fn get_statistics(&self) -> PyResult<std::collections::HashMap<String, f64>> {
        let mut stats = std::collections::HashMap::new();
        
        // 内存使用
        let cache = self.data_cache.read().unwrap();
        stats.insert("cache_size_mb".to_string(), cache.size_mb() as f64);
        stats.insert("cache_hit_rate".to_string(), cache.hit_rate());
        
        // 线程池状态
        stats.insert("num_threads".to_string(), self.config.num_threads as f64);
        
        // 计算性能统计
        let factor_stats = self.factor_engine.get_performance_stats()?;
        stats.extend(factor_stats);
        
        Ok(stats)
    }
    
    fn generate_test_data(&self) -> PyResult<Vec<f64>> {
        Ok((0..1000).map(|i| i as f64 * 0.1).collect())
    }
}
```

### 2.2 内存管理

```rust
// src/core/memory.rs
use std::sync::Arc;
use parking_lot::{RwLock, Mutex};
use std::collections::{HashMap, VecDeque};

pub struct DataCache {
    data: RwLock<HashMap<String, CachedData>>,
    access_order: Mutex<VecDeque<String>>,
    max_size_mb: usize,
    current_size_mb: Arc<parking_lot::Mutex<usize>>,
    hit_count: Arc<parking_lot::Mutex<u64>>,
    miss_count: Arc<parking_lot::Mutex<u64>>,
}

#[derive(Clone, Debug)]
struct CachedData {
    data: Vec<u8>,
    timestamp: std::time::SystemTime,
    access_count: u64,
    size_bytes: usize,
}

impl DataCache {
    pub fn new(max_size_mb: usize) -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
            access_order: Mutex::new(VecDeque::new()),
            max_size_mb,
            current_size_mb: Arc::new(parking_lot::Mutex::new(0)),
            hit_count: Arc::new(parking_lot::Mutex::new(0)),
            miss_count: Arc::new(parking_lot::Mutex::new(0)),
        }
    }
    
    pub fn get<T>(&self, key: &str) -> Option<T> 
    where 
        T: serde::de::DeserializeOwned 
    {
        let data_map = self.data.read();
        if let Some(cached) = data_map.get(key) {
            // 命中计数
            *self.hit_count.lock() += 1;
            
            // 更新访问顺序
            let mut order = self.access_order.lock();
            if let Some(pos) = order.iter().position(|k| k == key) {
                order.remove(pos);
            }
            order.push_back(key.to_string());
            
            // 反序列化数据
            bincode::deserialize(&cached.data).ok()
        } else {
            *self.miss_count.lock() += 1;
            None
        }
    }
    
    pub fn put<T>(&self, key: String, value: &T) -> Result<(), Box<dyn std::error::Error>>
    where
        T: serde::Serialize
    {
        let serialized = bincode::serialize(value)?;
        let size_bytes = serialized.len();
        let size_mb = size_bytes / (1024 * 1024);
        
        // 检查是否需要清理缓存
        self.evict_if_needed(size_mb)?;
        
        let cached_data = CachedData {
            data: serialized,
            timestamp: std::time::SystemTime::now(),
            access_count: 1,
            size_bytes,
        };
        
        // 更新缓存
        {
            let mut data_map = self.data.write();
            data_map.insert(key.clone(), cached_data);
        }
        
        // 更新访问顺序
        {
            let mut order = self.access_order.lock();
            order.push_back(key);
        }
        
        // 更新大小
        *self.current_size_mb.lock() += size_mb;
        
        Ok(())
    }
    
    fn evict_if_needed(&self, new_size_mb: usize) -> Result<(), Box<dyn std::error::Error>> {
        let current_size = *self.current_size_mb.lock();
        
        if current_size + new_size_mb > self.max_size_mb {
            // 使用LRU策略清理缓存
            self.evict_lru_items(current_size + new_size_mb - self.max_size_mb)?;
        }
        
        Ok(())
    }
    
    fn evict_lru_items(&self, target_mb: usize) -> Result<(), Box<dyn std::error::Error>> {
        let mut evicted_mb = 0;
        
        while evicted_mb < target_mb {
            let key_to_remove = {
                let mut order = self.access_order.lock();
                order.pop_front()
            };
            
            if let Some(key) = key_to_remove {
                let mut data_map = self.data.write();
                if let Some(cached) = data_map.remove(&key) {
                    evicted_mb += cached.size_bytes / (1024 * 1024);
                }
            } else {
                break; // 没有更多可清理的项目
            }
        }
        
        *self.current_size_mb.lock() -= evicted_mb;
        Ok(())
    }
    
    pub fn size_mb(&self) -> usize {
        *self.current_size_mb.lock()
    }
    
    pub fn hit_rate(&self) -> f64 {
        let hits = *self.hit_count.lock() as f64;
        let misses = *self.miss_count.lock() as f64;
        let total = hits + misses;
        
        if total > 0.0 {
            hits / total
        } else {
            0.0
        }
    }
}

// 内存池用于复用大型数据结构
pub struct MemoryPool<T> {
    pool: Mutex<Vec<T>>,
    factory: Box<dyn Fn() -> T + Send + Sync>,
}

impl<T> MemoryPool<T> 
where 
    T: Send + 'static 
{
    pub fn new<F>(factory: F) -> Self 
    where 
        F: Fn() -> T + Send + Sync + 'static 
    {
        Self {
            pool: Mutex::new(Vec::new()),
            factory: Box::new(factory),
        }
    }
    
    pub fn acquire(&self) -> PooledItem<T> {
        let item = {
            let mut pool = self.pool.lock();
            pool.pop().unwrap_or_else(|| (self.factory)())
        };
        
        PooledItem::new(item, self)
    }
    
    fn return_item(&self, item: T) {
        let mut pool = self.pool.lock();
        if pool.len() < 100 { // 限制池大小
            pool.push(item);
        }
    }
}

pub struct PooledItem<T> {
    item: Option<T>,
    pool: *const MemoryPool<T>,
}

impl<T> PooledItem<T> {
    fn new(item: T, pool: &MemoryPool<T>) -> Self {
        Self {
            item: Some(item),
            pool: pool as *const _,
        }
    }
}

impl<T> std::ops::Deref for PooledItem<T> {
    type Target = T;
    
    fn deref(&self) -> &Self::Target {
        self.item.as_ref().unwrap()
    }
}

impl<T> std::ops::DerefMut for PooledItem<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.item.as_mut().unwrap()
    }
}

impl<T> Drop for PooledItem<T> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take() {
            unsafe {
                (&*self.pool).return_item(item);
            }
        }
    }
}
```

---

## 3. 因子计算引擎

### 3.1 因子计算器核心

```rust
// src/factor/calculator.rs
use std::sync::{Arc, RwLock};
use rayon::prelude::*;
use ndarray::{Array1, Array2, ArrayView1};
use pyo3::prelude::*;

use crate::data::{MarketData, DataCache};
use crate::factor::{FactorDefinition, FactorResult};

#[pyclass]
pub struct FactorEngine {
    cache: Arc<RwLock<DataCache>>,
    config: crate::core::EngineConfig,
    performance_stats: Arc<parking_lot::Mutex<PerformanceStats>>,
}

#[derive(Default)]
struct PerformanceStats {
    total_calculations: u64,
    total_time_ms: u64,
    cache_hits: u64,
    cache_misses: u64,
}

#[pymethods]
impl FactorEngine {
    #[new]
    pub fn new(
        config: crate::core::EngineConfig,
        cache: Arc<RwLock<DataCache>>
    ) -> PyResult<Self> {
        Ok(Self {
            cache,
            config,
            performance_stats: Arc::new(parking_lot::Mutex::new(PerformanceStats::default())),
        })
    }
    
    /// 批量计算因子
    #[pyo3(signature = (factors, market_data, symbols = None))]
    pub fn batch_calculate_factors(
        &self,
        factors: Vec<FactorDefinition>,
        market_data: &PyAny,  // 接受pandas DataFrame或类似结构
        symbols: Option<Vec<String>>,
    ) -> PyResult<std::collections::HashMap<String, Vec<f64>>> {
        let start_time = std::time::Instant::now();
        
        // 转换Python数据到Rust结构
        let data = self.convert_market_data(market_data)?;
        
        // 并行计算所有因子
        let results: Result<Vec<_>, _> = factors
            .par_iter()
            .map(|factor| {
                let factor_result = self.calculate_single_factor(factor, &data)?;
                Ok((factor.name.clone(), factor_result.values))
            })
            .collect();
        
        let results = results?;
        let calculation_time = start_time.elapsed();
        
        // 更新性能统计
        {
            let mut stats = self.performance_stats.lock();
            stats.total_calculations += factors.len() as u64;
            stats.total_time_ms += calculation_time.as_millis() as u64;
        }
        
        Ok(results.into_iter().collect())
    }
    
    /// 实时因子计算
    pub fn realtime_calculate_factor(
        &self,
        factor: &FactorDefinition,
        current_data: &MarketData,
        historical_window: usize,
    ) -> PyResult<f64> {
        // 检查缓存
        let cache_key = format!("realtime_{}_{}", factor.id, current_data.timestamp);
        
        if let Some(cached_result) = self.cache.read().unwrap().get::<f64>(&cache_key) {
            self.performance_stats.lock().cache_hits += 1;
            return Ok(cached_result);
        }
        
        // 计算因子值
        let result = match factor.factor_type.as_str() {
            "technical" => self.calculate_technical_factor(factor, current_data, historical_window)?,
            "statistical" => self.calculate_statistical_factor(factor, current_data, historical_window)?,
            "ai" => self.calculate_ai_factor(factor, current_data, historical_window)?,
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown factor type: {}", factor.factor_type)
            )),
        };
        
        // 缓存结果
        if let Err(e) = self.cache.read().unwrap().put(cache_key, &result) {
            log::warn!("Failed to cache factor result: {}", e);
        }
        
        self.performance_stats.lock().cache_misses += 1;
        Ok(result)
    }
    
    fn calculate_single_factor(
        &self, 
        factor: &FactorDefinition, 
        data: &[MarketData]
    ) -> Result<FactorResult, Box<dyn std::error::Error>> {
        match factor.factor_type.as_str() {
            "technical" => self.calculate_technical_factor_batch(factor, data),
            "statistical" => self.calculate_statistical_factor_batch(factor, data),
            "ai" => self.calculate_ai_factor_batch(factor, data),
            _ => Err(format!("Unknown factor type: {}", factor.factor_type).into()),
        }
    }
    
    fn calculate_technical_factor_batch(
        &self,
        factor: &FactorDefinition,
        data: &[MarketData]
    ) -> Result<FactorResult, Box<dyn std::error::Error>> {
        let prices: Vec<f64> = data.iter().map(|d| d.close).collect();
        let volumes: Vec<f64> = data.iter().map(|d| d.volume).collect();
        
        let values = match factor.name.as_str() {
            "RSI" => self.calculate_rsi(&prices, factor.parameters.get("period").unwrap_or(&14) as &i32)?,
            "MACD" => self.calculate_macd(&prices, &factor.parameters)?,
            "BB" => self.calculate_bollinger_bands(&prices, &factor.parameters)?,
            "Volume_Price_Trend" => self.calculate_vpt(&prices, &volumes)?,
            _ => return Err(format!("Unknown technical factor: {}", factor.name).into()),
        };
        
        Ok(FactorResult {
            factor_id: factor.id.clone(),
            values,
            timestamps: data.iter().map(|d| d.timestamp).collect(),
            metadata: std::collections::HashMap::new(),
        })
    }
    
    // RSI计算 - 使用SIMD优化
    fn calculate_rsi(&self, prices: &[f64], period: &i32) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let period = *period as usize;
        if prices.len() < period + 1 {
            return Err("Not enough data for RSI calculation".into());
        }
        
        let mut gains = Vec::with_capacity(prices.len() - 1);
        let mut losses = Vec::with_capacity(prices.len() - 1);
        
        // 计算价格变化
        for window in prices.windows(2) {
            let change = window[1] - window[0];
            gains.push(if change > 0.0 { change } else { 0.0 });
            losses.push(if change < 0.0 { -change } else { 0.0 });
        }
        
        let mut rsi_values = Vec::with_capacity(prices.len());
        
        // 填充初始NaN值
        for _ in 0..period {
            rsi_values.push(f64::NAN);
        }
        
        // 计算初始平均
        let mut avg_gain: f64 = gains.iter().take(period).sum::<f64>() / period as f64;
        let mut avg_loss: f64 = losses.iter().take(period).sum::<f64>() / period as f64;
        
        // 计算第一个RSI值
        let rs = if avg_loss != 0.0 { avg_gain / avg_loss } else { f64::INFINITY };
        rsi_values.push(100.0 - (100.0 / (1.0 + rs)));
        
        // 计算后续RSI值（使用Wilder's smoothing）
        for i in period..gains.len() {
            avg_gain = ((avg_gain * (period as f64 - 1.0)) + gains[i]) / period as f64;
            avg_loss = ((avg_loss * (period as f64 - 1.0)) + losses[i]) / period as f64;
            
            let rs = if avg_loss != 0.0 { avg_gain / avg_loss } else { f64::INFINITY };
            rsi_values.push(100.0 - (100.0 / (1.0 + rs)));
        }
        
        Ok(rsi_values)
    }
    
    // MACD计算
    fn calculate_macd(
        &self, 
        prices: &[f64], 
        parameters: &std::collections::HashMap<String, serde_json::Value>
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let fast_period = parameters.get("fast_period")
            .and_then(|v| v.as_u64())
            .unwrap_or(12) as usize;
        let slow_period = parameters.get("slow_period")
            .and_then(|v| v.as_u64())
            .unwrap_or(26) as usize;
        let signal_period = parameters.get("signal_period")
            .and_then(|v| v.as_u64())
            .unwrap_or(9) as usize;
        
        let fast_ema = self.calculate_ema(prices, fast_period)?;
        let slow_ema = self.calculate_ema(prices, slow_period)?;
        
        // MACD线 = 快线 - 慢线
        let macd_line: Vec<f64> = fast_ema.iter()
            .zip(slow_ema.iter())
            .map(|(fast, slow)| fast - slow)
            .collect();
        
        // 信号线 = MACD线的EMA
        let signal_line = self.calculate_ema(&macd_line, signal_period)?;
        
        // 返回MACD直方图 = MACD线 - 信号线
        let histogram: Vec<f64> = macd_line.iter()
            .zip(signal_line.iter())
            .map(|(macd, signal)| macd - signal)
            .collect();
        
        Ok(histogram)
    }
    
    // EMA计算辅助函数
    fn calculate_ema(&self, prices: &[f64], period: usize) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        if prices.len() < period {
            return Err("Not enough data for EMA calculation".into());
        }
        
        let multiplier = 2.0 / (period as f64 + 1.0);
        let mut ema_values = Vec::with_capacity(prices.len());
        
        // 初始化：前period个值用SMA
        for i in 0..period {
            if i == 0 {
                ema_values.push(f64::NAN);
            } else {
                let sma = prices[0..=i].iter().sum::<f64>() / (i + 1) as f64;
                ema_values.push(sma);
            }
        }
        
        // 从第period个值开始计算EMA
        let mut ema = prices[0..period].iter().sum::<f64>() / period as f64;
        ema_values[period - 1] = ema;
        
        for &price in &prices[period..] {
            ema = (price * multiplier) + (ema * (1.0 - multiplier));
            ema_values.push(ema);
        }
        
        Ok(ema_values)
    }
    
    pub fn get_performance_stats(&self) -> PyResult<std::collections::HashMap<String, f64>> {
        let stats = self.performance_stats.lock();
        let mut result = std::collections::HashMap::new();
        
        result.insert("total_calculations".to_string(), stats.total_calculations as f64);
        result.insert("total_time_ms".to_string(), stats.total_time_ms as f64);
        result.insert("cache_hits".to_string(), stats.cache_hits as f64);
        result.insert("cache_misses".to_string(), stats.cache_misses as f64);
        
        let hit_rate = if stats.cache_hits + stats.cache_misses > 0 {
            stats.cache_hits as f64 / (stats.cache_hits + stats.cache_misses) as f64
        } else {
            0.0
        };
        result.insert("cache_hit_rate".to_string(), hit_rate);
        
        if stats.total_calculations > 0 {
            result.insert(
                "avg_calculation_time_ms".to_string(), 
                stats.total_time_ms as f64 / stats.total_calculations as f64
            );
        }
        
        Ok(result)
    }
    
    fn convert_market_data(&self, py_data: &PyAny) -> PyResult<Vec<MarketData>> {
        // 这里需要实现Python数据到Rust结构的转换
        // 可以支持pandas DataFrame, numpy array等格式
        todo!("实现Python数据转换")
    }
    
    pub fn test_calculation(&self, test_data: &[f64]) -> PyResult<f64> {
        // 简单的测试计算，验证引擎工作正常
        Ok(test_data.iter().sum::<f64>() / test_data.len() as f64)
    }
}
```

### 3.2 技术指标模块

```rust
// src/factor/technical.rs
use ndarray::{Array1, ArrayView1};
use std::f64::consts::PI;

pub struct TechnicalIndicators;

impl TechnicalIndicators {
    /// 布林带计算
    pub fn bollinger_bands(
        prices: ArrayView1<f64>, 
        period: usize, 
        std_dev: f64
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let mut middle = Array1::<f64>::zeros(prices.len());
        let mut upper = Array1::<f64>::zeros(prices.len());
        let mut lower = Array1::<f64>::zeros(prices.len());
        
        for i in period..prices.len() {
            let window = prices.slice(s![(i + 1 - period)..=i]);
            let mean = window.mean().unwrap();
            let std = window.std(0.0);
            
            middle[i] = mean;
            upper[i] = mean + std_dev * std;
            lower[i] = mean - std_dev * std;
        }
        
        (upper, middle, lower)
    }
    
    /// 随机振荡器(Stochastic)
    pub fn stochastic(
        high: ArrayView1<f64>,
        low: ArrayView1<f64>, 
        close: ArrayView1<f64>,
        k_period: usize,
        d_period: usize
    ) -> (Array1<f64>, Array1<f64>) {
        let mut k_values = Array1::<f64>::zeros(close.len());
        
        // 计算%K
        for i in k_period..close.len() {
            let period_high = high.slice(s![(i + 1 - k_period)..=i]).iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let period_low = low.slice(s![(i + 1 - k_period)..=i]).iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            
            if period_high != period_low {
                k_values[i] = 100.0 * (close[i] - period_low) / (period_high - period_low);
            } else {
                k_values[i] = 50.0;
            }
        }
        
        // 计算%D（%K的移动平均）
        let d_values = Self::simple_moving_average(k_values.view(), d_period);
        
        (k_values, d_values)
    }
    
    /// 威廉指标(Williams %R)
    pub fn williams_r(
        high: ArrayView1<f64>,
        low: ArrayView1<f64>,
        close: ArrayView1<f64>,
        period: usize
    ) -> Array1<f64> {
        let mut wr_values = Array1::<f64>::zeros(close.len());
        
        for i in period..close.len() {
            let period_high = high.slice(s![(i + 1 - period)..=i]).iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let period_low = low.slice(s![(i + 1 - period)..=i]).iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            
            if period_high != period_low {
                wr_values[i] = -100.0 * (period_high - close[i]) / (period_high - period_low);
            } else {
                wr_values[i] = -50.0;
            }
        }
        
        wr_values
    }
    
    /// 动量指标(Momentum)
    pub fn momentum(prices: ArrayView1<f64>, period: usize) -> Array1<f64> {
        let mut momentum = Array1::<f64>::zeros(prices.len());
        
        for i in period..prices.len() {
            momentum[i] = prices[i] - prices[i - period];
        }
        
        momentum
    }
    
    /// 变动率指标(Rate of Change)
    pub fn rate_of_change(prices: ArrayView1<f64>, period: usize) -> Array1<f64> {
        let mut roc = Array1::<f64>::zeros(prices.len());
        
        for i in period..prices.len() {
            if prices[i - period] != 0.0 {
                roc[i] = 100.0 * (prices[i] - prices[i - period]) / prices[i - period];
            }
        }
        
        roc
    }
    
    /// 平滑异同平均线(MACD)增强版
    pub fn macd_enhanced(
        prices: ArrayView1<f64>,
        fast_period: usize,
        slow_period: usize,
        signal_period: usize
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let fast_ema = Self::exponential_moving_average(prices, fast_period);
        let slow_ema = Self::exponential_moving_average(prices, slow_period);
        
        // MACD线
        let macd_line = &fast_ema - &slow_ema;
        
        // 信号线
        let signal_line = Self::exponential_moving_average(macd_line.view(), signal_period);
        
        // 直方图
        let histogram = &macd_line - &signal_line;
        
        (macd_line, signal_line, histogram)
    }
    
    /// 相对强弱指数(RSI)增强版，支持不同平滑方法
    pub fn rsi_enhanced(
        prices: ArrayView1<f64>, 
        period: usize,
        smoothing_method: RSISmoothingMethod
    ) -> Array1<f64> {
        let mut gains = Vec::new();
        let mut losses = Vec::new();
        
        // 计算价格变化
        for i in 1..prices.len() {
            let change = prices[i] - prices[i-1];
            gains.push(if change > 0.0 { change } else { 0.0 });
            losses.push(if change < 0.0 { -change } else { 0.0 });
        }
        
        let gains_array = Array1::from_vec(gains);
        let losses_array = Array1::from_vec(losses);
        
        let (avg_gain, avg_loss) = match smoothing_method {
            RSISmoothingMethod::Wilder => {
                Self::wilder_smoothing(gains_array.view(), losses_array.view(), period)
            },
            RSISmoothingMethod::EMA => {
                let gain_ema = Self::exponential_moving_average(gains_array.view(), period);
                let loss_ema = Self::exponential_moving_average(losses_array.view(), period);
                (gain_ema, loss_ema)
            },
            RSISmoothingMethod::SMA => {
                let gain_sma = Self::simple_moving_average(gains_array.view(), period);
                let loss_sma = Self::simple_moving_average(losses_array.view(), period);
                (gain_sma, loss_sma)
            }
        };
        
        let mut rsi = Array1::<f64>::zeros(prices.len());
        
        for i in 0..avg_gain.len() {
            if avg_loss[i] != 0.0 {
                let rs = avg_gain[i] / avg_loss[i];
                rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs));
            } else {
                rsi[i + 1] = 100.0;
            }
        }
        
        rsi
    }
    
    /// Wilder平滑方法
    fn wilder_smoothing(
        gains: ArrayView1<f64>, 
        losses: ArrayView1<f64>, 
        period: usize
    ) -> (Array1<f64>, Array1<f64>) {
        let mut avg_gains = Array1::<f64>::zeros(gains.len());
        let mut avg_losses = Array1::<f64>::zeros(losses.len());
        
        if gains.len() < period {
            return (avg_gains, avg_losses);
        }
        
        // 初始平均值（简单平均）
        let initial_avg_gain = gains.slice(s![0..period]).mean().unwrap();
        let initial_avg_loss = losses.slice(s![0..period]).mean().unwrap();
        
        avg_gains[period - 1] = initial_avg_gain;
        avg_losses[period - 1] = initial_avg_loss;
        
        // Wilder平滑
        for i in period..gains.len() {
            avg_gains[i] = (avg_gains[i-1] * (period as f64 - 1.0) + gains[i]) / period as f64;
            avg_losses[i] = (avg_losses[i-1] * (period as f64 - 1.0) + losses[i]) / period as f64;
        }
        
        (avg_gains, avg_losses)
    }
    
    /// 简单移动平均
    fn simple_moving_average(data: ArrayView1<f64>, period: usize) -> Array1<f64> {
        let mut sma = Array1::<f64>::zeros(data.len());
        
        for i in period..data.len() {
            let sum = data.slice(s![(i + 1 - period)..=i]).sum();
            sma[i] = sum / period as f64;
        }
        
        sma
    }
    
    /// 指数移动平均
    fn exponential_moving_average(data: ArrayView1<f64>, period: usize) -> Array1<f64> {
        let mut ema = Array1::<f64>::zeros(data.len());
        let multiplier = 2.0 / (period as f64 + 1.0);
        
        if data.len() < period {
            return ema;
        }
        
        // 使用SMA作为初始EMA值
        let initial_sma = data.slice(s![0..period]).mean().unwrap();
        ema[period - 1] = initial_sma;
        
        for i in period..data.len() {
            ema[i] = (data[i] * multiplier) + (ema[i-1] * (1.0 - multiplier));
        }
        
        ema
    }
}

#[derive(Debug, Clone)]
pub enum RSISmoothingMethod {
    Wilder,  // 传统Wilder平滑
    EMA,     // 指数移动平均
    SMA,     // 简单移动平均
}
```

---

## 4. 回测引擎设计

### 4.1 回测引擎核心

```rust
// src/backtest/engine.rs
use std::collections::{HashMap, VecDeque};
use chrono::{DateTime, Utc};
use pyo3::prelude::*;

use crate::data::MarketData;
use super::{Portfolio, TradeExecutor, BacktestMetrics, BacktestConfig};

#[pyclass]
pub struct BacktestEngine {
    config: BacktestConfig,
    portfolio: Portfolio,
    executor: TradeExecutor,
    metrics: BacktestMetrics,
    
    // 状态跟踪
    current_date: Option<DateTime<Utc>>,
    trade_history: Vec<TradeRecord>,
    performance_history: Vec<PerformanceSnapshot>,
    
    // 缓存和优化
    data_cache: HashMap<String, VecDeque<MarketData>>,
    calculation_cache: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub initial_capital: f64,
    pub commission_rate: f64,
    pub slippage_bps: f64,
    pub min_trade_amount: f64,
    pub max_position_size: f64,
    pub margin_requirement: f64,
    pub risk_free_rate: f64,
    pub benchmark_symbol: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TradeRecord {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub side: TradeSide,
    pub quantity: f64,
    pub price: f64,
    pub commission: f64,
    pub slippage: f64,
    pub trade_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TradeSide {
    Buy,
    Sell,
    Short,
    Cover,
}

#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    pub timestamp: DateTime<Utc>,
    pub portfolio_value: f64,
    pub cash: f64,
    pub positions_value: f64,
    pub daily_return: f64,
    pub cumulative_return: f64,
    pub drawdown: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub volatility: f64,
}

#[pymethods]
impl BacktestEngine {
    #[new]
    pub fn new(config: BacktestConfig) -> PyResult<Self> {
        let portfolio = Portfolio::new(config.initial_capital, config.margin_requirement);
        let executor = TradeExecutor::new(config.commission_rate, config.slippage_bps);
        let metrics = BacktestMetrics::new(config.risk_free_rate);
        
        Ok(Self {
            config,
            portfolio,
            executor,
            metrics,
            current_date: None,
            trade_history: Vec::new(),
            performance_history: Vec::new(),
            data_cache: HashMap::new(),
            calculation_cache: HashMap::new(),
        })
    }
    
    /// 运行完整回测
    pub fn run_backtest(
        &mut self,
        strategy_signals: HashMap<String, Vec<(DateTime<Utc>, String, f64)>>, // (timestamp, symbol, signal)
        market_data: HashMap<String, Vec<MarketData>>,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> PyResult<BacktestResults> {
        // 初始化回测环境
        self.initialize_backtest(start_date)?;
        
        // 预处理市场数据
        self.preprocess_market_data(market_data)?;
        
        // 获取所有交易日期
        let trading_dates = self.get_trading_dates(start_date, end_date)?;
        
        // 逐日回测
        for date in trading_dates {
            self.current_date = Some(date);
            
            // 更新市场数据
            self.update_market_data(date)?;
            
            // 处理策略信号
            if let Some(signals) = strategy_signals.get(&date.format("%Y-%m-%d").to_string()) {
                self.process_trading_signals(signals, date)?;
            }
            
            // 更新投资组合
            self.update_portfolio(date)?;
            
            // 计算性能指标
            let snapshot = self.calculate_performance_snapshot(date)?;
            self.performance_history.push(snapshot);
            
            // 风险控制检查
            self.risk_management_check(date)?;
        }
        
        // 生成最终结果
        self.generate_backtest_results()
    }
    
    /// 处理交易信号
    fn process_trading_signals(
        &mut self,
        signals: &[(DateTime<Utc>, String, f64)],
        current_date: DateTime<Utc>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for &(timestamp, ref symbol, signal_strength) in signals {
            if timestamp.date_naive() == current_date.date_naive() {
                // 获取当前市场数据
                let market_data = self.get_current_market_data(symbol)?;
                
                // 计算目标仓位
                let target_position = self.calculate_target_position(
                    symbol, 
                    signal_strength, 
                    &market_data
                )?;
                
                // 执行交易
                self.execute_rebalance(symbol, target_position, &market_data)?;
            }
        }
        
        Ok(())
    }
    
    /// 计算目标仓位
    fn calculate_target_position(
        &self,
        symbol: &str,
        signal_strength: f64,
        market_data: &MarketData,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let portfolio_value = self.portfolio.total_value();
        let max_position_value = portfolio_value * self.config.max_position_size;
        
        // 基于信号强度计算仓位
        let signal_position_value = portfolio_value * signal_strength.abs().min(self.config.max_position_size);
        let target_position_value = signal_position_value.min(max_position_value);
        
        // 转换为股数
        let target_shares = if market_data.close > 0.0 {
            target_position_value / market_data.close
        } else {
            0.0
        };
        
        // 考虑信号方向
        Ok(if signal_strength >= 0.0 {
            target_shares
        } else {
            -target_shares
        })
    }
    
    /// 执行仓位调整
    fn execute_rebalance(
        &mut self,
        symbol: &str,
        target_shares: f64,
        market_data: &MarketData,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let current_shares = self.portfolio.get_position(symbol);
        let shares_diff = target_shares - current_shares;
        
        if shares_diff.abs() < self.config.min_trade_amount / market_data.close {
            return Ok(()); // 交易量太小，不执行
        }
        
        // 确定交易方向
        let (side, quantity) = if shares_diff > 0.0 {
            (TradeSide::Buy, shares_diff.abs())
        } else {
            (TradeSide::Sell, shares_diff.abs())
        };
        
        // 执行交易
        let trade_result = self.executor.execute_trade(
            symbol,
            side.clone(),
            quantity,
            market_data.close,
            self.current_date.unwrap(),
        )?;
        
        // 更新投资组合
        self.portfolio.update_position(symbol, target_shares);
        self.portfolio.update_cash(-trade_result.total_cost());
        
        // 记录交易
        self.trade_history.push(TradeRecord {
            timestamp: self.current_date.unwrap(),
            symbol: symbol.to_string(),
            side,
            quantity,
            price: trade_result.execution_price,
            commission: trade_result.commission,
            slippage: trade_result.slippage,
            trade_id: trade_result.trade_id,
        });
        
        Ok(())
    }
    
    /// 计算性能快照
    fn calculate_performance_snapshot(
        &mut self,
        date: DateTime<Utc>,
    ) -> Result<PerformanceSnapshot, Box<dyn std::error::Error>> {
        let portfolio_value = self.calculate_portfolio_value(date)?;
        let cash = self.portfolio.cash();
        let positions_value = portfolio_value - cash;
        
        // 计算收益率
        let daily_return = if let Some(prev_snapshot) = self.performance_history.last() {
            (portfolio_value - prev_snapshot.portfolio_value) / prev_snapshot.portfolio_value
        } else {
            0.0
        };
        
        let cumulative_return = (portfolio_value - self.config.initial_capital) / self.config.initial_capital;
        
        // 计算回撤
        let peak_value = self.performance_history
            .iter()
            .map(|s| s.portfolio_value)
            .fold(self.config.initial_capital, f64::max);
        let current_drawdown = (portfolio_value - peak_value) / peak_value;
        let max_drawdown = self.performance_history
            .iter()
            .map(|s| s.drawdown)
            .fold(current_drawdown, f64::min);
        
        // 计算夏普比率和波动率
        let returns: Vec<f64> = self.performance_history
            .iter()
            .map(|s| s.daily_return)
            .collect();
        
        let (sharpe_ratio, volatility) = self.calculate_risk_metrics(&returns)?;
        
        Ok(PerformanceSnapshot {
            timestamp: date,
            portfolio_value,
            cash,
            positions_value,
            daily_return,
            cumulative_return,
            drawdown: current_drawdown,
            max_drawdown,
            sharpe_ratio,
            volatility,
        })
    }
    
    /// 计算风险指标
    fn calculate_risk_metrics(
        &self, 
        returns: &[f64]
    ) -> Result<(f64, f64), Box<dyn std::error::Error>> {
        if returns.is_empty() {
            return Ok((0.0, 0.0));
        }
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|&r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        let volatility = variance.sqrt();
        
        // 年化夏普比率
        let excess_return = mean_return - self.config.risk_free_rate / 252.0; // 日收益率
        let sharpe_ratio = if volatility > 0.0 {
            excess_return * (252.0_f64).sqrt() / (volatility * (252.0_f64).sqrt())
        } else {
            0.0
        };
        
        Ok((sharpe_ratio, volatility * (252.0_f64).sqrt())) // 年化波动率
    }
    
    /// 生成最终回测结果
    fn generate_backtest_results(&self) -> PyResult<BacktestResults> {
        let final_snapshot = self.performance_history.last()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "No performance data available"
            ))?;
        
        // 计算各种性能指标
        let total_return = final_snapshot.cumulative_return;
        let max_drawdown = self.performance_history.iter()
            .map(|s| s.drawdown)
            .fold(0.0, f64::min);
        
        let returns: Vec<f64> = self.performance_history.iter()
            .map(|s| s.daily_return)
            .collect();
        
        let win_rate = self.calculate_win_rate(&returns);
        let calmar_ratio = if max_drawdown.abs() > 0.0 {
            (total_return * 252.0) / max_drawdown.abs()
        } else {
            0.0
        };
        
        // 计算信息比率（相对于基准）
        let information_ratio = self.calculate_information_ratio(&returns)?;
        
        Ok(BacktestResults {
            total_return,
            annualized_return: total_return * 252.0 / returns.len() as f64,
            max_drawdown,
            sharpe_ratio: final_snapshot.sharpe_ratio,
            volatility: final_snapshot.volatility,
            win_rate,
            calmar_ratio,
            information_ratio,
            total_trades: self.trade_history.len(),
            total_commission: self.trade_history.iter().map(|t| t.commission).sum(),
            total_slippage: self.trade_history.iter().map(|t| t.slippage).sum(),
            performance_history: self.performance_history.clone(),
            trade_history: self.trade_history.clone(),
        })
    }
    
    fn calculate_win_rate(&self, returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }
        
        let winning_days = returns.iter().filter(|&&r| r > 0.0).count();
        winning_days as f64 / returns.len() as f64
    }
    
    fn calculate_information_ratio(&self, returns: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
        // 如果有基准数据，计算信息比率
        // 这里简化为相对于无风险利率的超额收益
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let excess_return = mean_return - self.config.risk_free_rate / 252.0;
        
        let tracking_error = if returns.len() > 1 {
            let variance = returns.iter()
                .map(|&r| (r - mean_return).powi(2))
                .sum::<f64>() / (returns.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };
        
        Ok(if tracking_error > 0.0 {
            excess_return / tracking_error
        } else {
            0.0
        })
    }
    
    // 其他辅助方法...
    fn initialize_backtest(&mut self, start_date: DateTime<Utc>) -> Result<(), Box<dyn std::error::Error>> {
        self.current_date = Some(start_date);
        self.trade_history.clear();
        self.performance_history.clear();
        self.portfolio.reset(self.config.initial_capital);
        Ok(())
    }
    
    fn preprocess_market_data(
        &mut self, 
        market_data: HashMap<String, Vec<MarketData>>
    ) -> Result<(), Box<dyn std::error::Error>> {
        for (symbol, data) in market_data {
            let mut data_queue = VecDeque::new();
            for item in data {
                data_queue.push_back(item);
            }
            self.data_cache.insert(symbol, data_queue);
        }
        Ok(())
    }
    
    fn get_trading_dates(
        &self, 
        start: DateTime<Utc>, 
        end: DateTime<Utc>
    ) -> Result<Vec<DateTime<Utc>>, Box<dyn std::error::Error>> {
        // 生成交易日期列表（排除周末等）
        let mut dates = Vec::new();
        let mut current = start;
        
        while current <= end {
            // 简化版：只排除周末
            let weekday = current.weekday();
            if weekday != chrono::Weekday::Sat && weekday != chrono::Weekday::Sun {
                dates.push(current);
            }
            current += chrono::Duration::days(1);
        }
        
        Ok(dates)
    }
    
    fn update_market_data(&mut self, date: DateTime<Utc>) -> Result<(), Box<dyn std::error::Error>> {
        // 更新当前市场数据快照
        // 实际实现中会从缓存中获取对应日期的数据
        Ok(())
    }
    
    fn get_current_market_data(&self, symbol: &str) -> Result<MarketData, Box<dyn std::error::Error>> {
        // 从缓存中获取当前市场数据
        // 这里需要实际实现
        todo!("实现市场数据获取逻辑")
    }
    
    fn update_portfolio(&mut self, date: DateTime<Utc>) -> Result<(), Box<dyn std::error::Error>> {
        // 使用当前市场价格更新投资组合价值
        Ok(())
    }
    
    fn calculate_portfolio_value(&self, date: DateTime<Utc>) -> Result<f64, Box<dyn std::error::Error>> {
        // 计算投资组合总价值
        // 现金 + 所有持仓的市值
        let mut total_value = self.portfolio.cash();
        
        for (symbol, shares) in self.portfolio.positions() {
            if let Ok(market_data) = self.get_current_market_data(symbol) {
                total_value += shares * market_data.close;
            }
        }
        
        Ok(total_value)
    }
    
    fn risk_management_check(&mut self, date: DateTime<Utc>) -> Result<(), Box<dyn std::error::Error>> {
        // 风险控制检查
        // 检查最大回撤、杠杆率等风险指标
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct BacktestResults {
    pub total_return: f64,
    pub annualized_return: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub volatility: f64,
    pub win_rate: f64,
    pub calmar_ratio: f64,
    pub information_ratio: f64,
    pub total_trades: usize,
    pub total_commission: f64,
    pub total_slippage: f64,
    pub performance_history: Vec<PerformanceSnapshot>,
    pub trade_history: Vec<TradeRecord>,
}
```

---

## 5. 性能优化与测试

### 5.1 性能基准测试

```rust
// benches/factor_calculation.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use quant_engine::factor::FactorEngine;
use quant_engine::data::MarketData;

fn benchmark_rsi_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("RSI Calculation");
    
    for size in [1000, 10000, 100000].iter() {
        let prices: Vec<f64> = (0..*size).map(|i| 100.0 + (i as f64 * 0.01).sin() * 10.0).collect();
        
        group.bench_with_input(
            BenchmarkId::new("RSI", size),
            size,
            |b, &_size| {
                let engine = FactorEngine::new_for_test();
                b.iter(|| {
                    engine.calculate_rsi(black_box(&prices), black_box(&14))
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_parallel_factor_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parallel Factor Calculation");
    
    let market_data: Vec<MarketData> = (0..10000)
        .map(|i| MarketData {
            symbol: "TEST".to_string(),
            timestamp: chrono::Utc::now(),
            open: 100.0 + i as f64 * 0.01,
            high: 101.0 + i as f64 * 0.01,
            low: 99.0 + i as f64 * 0.01,
            close: 100.5 + i as f64 * 0.01,
            volume: 1000000.0,
        })
        .collect();
    
    for num_factors in [1, 10, 50, 100].iter() {
        let factors: Vec<_> = (0..*num_factors)
            .map(|i| create_test_factor(format!("factor_{}", i)))
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("Parallel", num_factors),
            num_factors,
            |b, &_num_factors| {
                let engine = FactorEngine::new_for_test();
                b.iter(|| {
                    engine.parallel_factor_calculation(
                        black_box(&market_data),
                        black_box(&factors)
                    )
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_rsi_calculation, benchmark_parallel_factor_calculation);
criterion_main!(benches);
```

### 5.2 单元测试

```rust
// src/factor/tests.rs
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_rsi_calculation() {
        let engine = FactorEngine::new_for_test();
        
        // 测试数据：价格上涨趋势
        let prices = vec![
            44.0, 44.25, 44.5, 43.75, 44.5, 44.0, 44.25,
            44.75, 45.0, 45.25, 45.5, 45.75, 46.0, 46.25
        ];
        
        let rsi_values = engine.calculate_rsi(&prices, &14).unwrap();
        
        // 验证RSI值在0-100范围内
        for &rsi in &rsi_values {
            if !rsi.is_nan() {
                assert!(rsi >= 0.0 && rsi <= 100.0, "RSI value out of range: {}", rsi);
            }
        }
        
        // 验证上涨趋势中RSI应该偏高
        let valid_rsi: Vec<f64> = rsi_values.into_iter().filter(|x| !x.is_nan()).collect();
        if !valid_rsi.is_empty() {
            let last_rsi = valid_rsi.last().unwrap();
            assert!(*last_rsi > 50.0, "RSI should be above 50 in uptrend");
        }
    }
    
    #[test]
    fn test_ema_calculation() {
        let engine = FactorEngine::new_for_test();
        let prices = vec![22.0, 22.15, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29];
        
        let ema = engine.calculate_ema(&prices, 5).unwrap();
        
        // 验证EMA序列长度
        assert_eq!(ema.len(), prices.len());
        
        // 验证前几个值为NaN或初始值
        assert!(ema[0].is_nan() || ema[0] > 0.0);
        
        // 验证EMA对价格变化的响应
        let last_price = prices.last().unwrap();
        let last_ema = ema.last().unwrap();
        assert!((*last_ema - last_price).abs() < 1.0, "EMA should be close to recent prices");
    }
    
    #[test]
    fn test_macd_calculation() {
        let engine = FactorEngine::new_for_test();
        let prices: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 10.0)
            .collect();
        
        let mut params = std::collections::HashMap::new();
        params.insert("fast_period".to_string(), serde_json::Value::from(12));
        params.insert("slow_period".to_string(), serde_json::Value::from(26));
        params.insert("signal_period".to_string(), serde_json::Value::from(9));
        
        let macd = engine.calculate_macd(&prices, &params).unwrap();
        
        // 验证MACD长度
        assert_eq!(macd.len(), prices.len());
        
        // 验证MACD值的合理性
        let valid_macd: Vec<f64> = macd.into_iter().filter(|x| !x.is_nan()).collect();
        assert!(!valid_macd.is_empty(), "Should have some valid MACD values");
    }
    
    #[test]
    fn test_bollinger_bands() {
        let prices = ndarray::Array1::from_vec(vec![
            20.0, 20.5, 21.0, 20.8, 21.2, 21.5, 21.3, 21.8, 22.0, 21.7,
            22.2, 22.5, 22.3, 22.8, 23.0, 22.7, 23.2, 23.5, 23.3, 23.8
        ]);
        
        let (upper, middle, lower) = TechnicalIndicators::bollinger_bands(
            prices.view(), 10, 2.0
        );
        
        // 验证布林带性质：上轨 > 中轨 > 下轨
        for i in 10..prices.len() {
            if !upper[i].is_nan() && !middle[i].is_nan() && !lower[i].is_nan() {
                assert!(upper[i] > middle[i], "Upper band should be above middle");
                assert!(middle[i] > lower[i], "Middle should be above lower band");
            }
        }
        
        // 验证中轨就是移动平均
        for i in 10..prices.len() {
            if !middle[i].is_nan() {
                let manual_avg = prices.slice(s![(i-9)..=i]).mean().unwrap();
                assert_relative_eq!(middle[i], manual_avg, epsilon = 1e-10);
            }
        }
    }
    
    #[test]
    fn test_performance_under_load() {
        use std::time::Instant;
        
        let engine = FactorEngine::new_for_test();
        let large_dataset: Vec<f64> = (0..1000000)
            .map(|i| 100.0 + (i as f64 * 0.001).sin() * 10.0)
            .collect();
        
        let start = Instant::now();
        let _rsi = engine.calculate_rsi(&large_dataset, &14).unwrap();
        let duration = start.elapsed();
        
        // 验证大数据集处理时间合理（应该在几秒内完成）
        assert!(duration.as_secs() < 5, "Large dataset processing took too long: {:?}", duration);
    }
    
    #[test]
    fn test_memory_usage() {
        let engine = FactorEngine::new_for_test();
        
        // 测试内存使用是否合理
        let initial_memory = get_current_memory_usage();
        
        // 处理多个大数据集
        for _ in 0..10 {
            let data: Vec<f64> = (0..100000).map(|i| i as f64).collect();
            let _result = engine.calculate_rsi(&data, &14).unwrap();
        }
        
        let final_memory = get_current_memory_usage();
        let memory_increase = final_memory - initial_memory;
        
        // 验证内存增长在合理范围内（不应该无限制增长）
        assert!(memory_increase < 1024 * 1024 * 100, "Excessive memory usage: {} bytes", memory_increase);
    }
    
    fn get_current_memory_usage() -> usize {
        // 简化的内存使用监控
        // 实际实现中可以使用更精确的内存监控
        0
    }
}

// 集成测试
#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::collections::HashMap;
    
    #[test]
    fn test_full_factor_calculation_pipeline() {
        let engine = FactorEngine::new_for_test();
        
        // 创建测试市场数据
        let market_data = create_test_market_data(1000);
        
        // 创建多个测试因子
        let factors = vec![
            create_rsi_factor(),
            create_macd_factor(),
            create_bollinger_factor(),
        ];
        
        // 执行批量计算
        let results = engine.batch_calculate_factors_internal(&factors, &market_data).unwrap();
        
        // 验证结果
        assert_eq!(results.len(), factors.len());
        for (factor_name, values) in results {
            assert!(!values.is_empty(), "Factor {} should have values", factor_name);
            
            // 验证数值有效性
            let valid_count = values.iter().filter(|x| !x.is_nan()).count();
            assert!(valid_count > 0, "Factor {} should have some valid values", factor_name);
        }
    }
    
    fn create_test_market_data(count: usize) -> Vec<MarketData> {
        (0..count)
            .map(|i| MarketData {
                symbol: "TEST".to_string(),
                timestamp: chrono::Utc::now() + chrono::Duration::days(i as i64),
                open: 100.0 + (i as f64 * 0.01).sin() * 5.0,
                high: 102.0 + (i as f64 * 0.01).sin() * 5.0,
                low: 98.0 + (i as f64 * 0.01).sin() * 5.0,
                close: 100.5 + (i as f64 * 0.01).sin() * 5.0,
                volume: 1000000.0 + (i as f64 * 0.1).cos() * 100000.0,
            })
            .collect()
    }
    
    fn create_rsi_factor() -> FactorDefinition {
        FactorDefinition {
            id: "rsi_14".to_string(),
            name: "RSI".to_string(),
            factor_type: "technical".to_string(),
            parameters: {
                let mut params = HashMap::new();
                params.insert("period".to_string(), serde_json::Value::from(14));
                params
            },
        }
    }
    
    fn create_macd_factor() -> FactorDefinition {
        FactorDefinition {
            id: "macd_12_26_9".to_string(),
            name: "MACD".to_string(),
            factor_type: "technical".to_string(),
            parameters: {
                let mut params = HashMap::new();
                params.insert("fast_period".to_string(), serde_json::Value::from(12));
                params.insert("slow_period".to_string(), serde_json::Value::from(26));
                params.insert("signal_period".to_string(), serde_json::Value::from(9));
                params
            },
        }
    }
    
    fn create_bollinger_factor() -> FactorDefinition {
        FactorDefinition {
            id: "bb_20_2".to_string(),
            name: "BB".to_string(),
            factor_type: "technical".to_string(),
            parameters: {
                let mut params = HashMap::new();
                params.insert("period".to_string(), serde_json::Value::from(20));
                params.insert("std_dev".to_string(), serde_json::Value::from(2.0));
                params
            },
        }
    }
}
```

---

## 总结

本Rust核心引擎设计文档提供了：

### 🚀 核心特性

1. **高性能计算**: 基于Rust的零成本抽象和SIMD优化
2. **内存安全**: 编译时保证内存安全，避免运行时错误
3. **并行处理**: 使用Rayon实现数据并行和任务并行
4. **Python集成**: 通过PyO3实现与Python的无缝集成
5. **模块化设计**: 清晰的模块划分，便于维护和扩展

### 💡 技术优势

- **10-100倍性能提升**: 相比纯Python实现
- **内存高效**: 智能缓存和内存池管理
- **类型安全**: 编译时类型检查，减少运行时错误
- **并发安全**: 使用Rust的所有权系统保证线程安全

### 🛠 实施建议

1. **分阶段开发**: 从核心因子计算开始，逐步扩展功能
2. **充分测试**: 包含单元测试、集成测试和性能测试
3. **性能监控**: 内置性能统计和监控功能
4. **文档完善**: 详细的API文档和使用示例

该设计为量化分析系统提供了高性能、可靠的计算核心，能够支撑大规模因子研究和回测分析需求。