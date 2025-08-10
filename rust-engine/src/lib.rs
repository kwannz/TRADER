//! QuantAnalyzer Pro - 高性能量化分析引擎
//! 
//! 这是一个用Rust编写的高性能量化因子计算和回测引擎，
//! 具有Python绑定，用于量化金融分析。

pub mod factors;
pub mod backtest;
pub mod data;
pub mod utils;

#[cfg(feature = "python-bindings")]
pub mod python_bindings;

// 重新导出主要模块
pub use factors::*;
pub use backtest::*;
pub use data::*;
pub use utils::*;

// 当启用Python绑定时，重新导出Python模块
#[cfg(feature = "python-bindings")]
pub use python_bindings::*;