//! 风控管理模块
//! 
//! 提供实时风险控制和仓位管理功能

use serde::{Deserialize, Serialize};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RiskError {
    #[error("Risk limit exceeded: {0}")]
    RiskLimitExceeded(String),
    #[error("Invalid position: {0}")]
    InvalidPosition(String),
    #[error("Calculation error: {0}")]
    CalculationError(String),
}

/// 仓位信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub side: String,        // "long" or "short"
    pub size: f64,           // 仓位大小
    pub entry_price: f64,    // 开仓价格
    pub current_price: f64,  // 当前价格
    pub unrealized_pnl: f64, // 未实现盈亏
    pub margin_used: f64,    // 使用保证金
    pub leverage: f64,       // 杠杆倍数
    pub timestamp: DateTime<Utc>,
}

impl Position {
    /// 从Python字典创建Position
    pub fn from_py_dict(dict: &PyDict) -> PyResult<Self> {
        let timestamp_str = dict.get_item("timestamp")
            .unwrap_or(&Utc::now().to_rfc3339().into())
            .extract::<String>()?;
        
        let timestamp = DateTime::parse_from_rfc3339(&timestamp_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("invalid timestamp: {}", e)))?
            .with_timezone(&Utc);
        
        Ok(Position {
            symbol: dict.get_item("symbol").unwrap_or(&"BTC/USDT".into()).extract()?,
            side: dict.get_item("side").unwrap_or(&"long".into()).extract()?,
            size: dict.get_item("size").unwrap_or(&0.0.into()).extract()?,
            entry_price: dict.get_item("entry_price").unwrap_or(&0.0.into()).extract()?,
            current_price: dict.get_item("current_price").unwrap_or(&0.0.into()).extract()?,
            unrealized_pnl: dict.get_item("unrealized_pnl").unwrap_or(&0.0.into()).extract()?,
            margin_used: dict.get_item("margin_used").unwrap_or(&0.0.into()).extract()?,
            leverage: dict.get_item("leverage").unwrap_or(&1.0.into()).extract()?,
            timestamp,
        })
    }
    
    /// 计算未实现盈亏
    pub fn calculate_unrealized_pnl(&mut self) {
        match self.side.as_str() {
            "long" => {
                self.unrealized_pnl = (self.current_price - self.entry_price) * self.size;
            },
            "short" => {
                self.unrealized_pnl = (self.entry_price - self.current_price) * self.size;
            },
            _ => {
                self.unrealized_pnl = 0.0;
            }
        }
    }
    
    /// 获取仓位价值
    pub fn get_position_value(&self) -> f64 {
        self.size * self.current_price
    }
    
    /// 获取收益率
    pub fn get_return_rate(&self) -> f64 {
        if self.entry_price == 0.0 || self.size == 0.0 {
            return 0.0;
        }
        
        let position_value = self.size * self.entry_price;
        if position_value == 0.0 {
            return 0.0;
        }
        
        self.unrealized_pnl / position_value
    }
}

/// 交易订单
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub symbol: String,
    pub side: String,        // "buy" or "sell"
    pub order_type: String,  // "market", "limit", "stop"
    pub quantity: f64,       // 交易数量
    pub price: f64,          // 交易价格
    pub leverage: f64,       // 杠杆倍数
    pub stop_loss: Option<f64>,    // 止损价格
    pub take_profit: Option<f64>,  // 止盈价格
    pub timestamp: DateTime<Utc>,
}

impl Trade {
    /// 从Python字典创建Trade
    pub fn from_py_dict(dict: &PyDict) -> PyResult<Self> {
        let timestamp_str = dict.get_item("timestamp")
            .unwrap_or(&Utc::now().to_rfc3339().into())
            .extract::<String>()?;
        
        let timestamp = DateTime::parse_from_rfc3339(&timestamp_str)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("invalid timestamp: {}", e)))?
            .with_timezone(&Utc);
        
        let stop_loss: Option<f64> = dict.get_item("stop_loss")
            .and_then(|item| item.extract().ok());
        
        let take_profit: Option<f64> = dict.get_item("take_profit")
            .and_then(|item| item.extract().ok());
        
        Ok(Trade {
            symbol: dict.get_item("symbol").unwrap_or(&"BTC/USDT".into()).extract()?,
            side: dict.get_item("side").unwrap_or(&"buy".into()).extract()?,
            order_type: dict.get_item("order_type").unwrap_or(&"market".into()).extract()?,
            quantity: dict.get_item("quantity").unwrap_or(&0.0.into()).extract()?,
            price: dict.get_item("price").unwrap_or(&0.0.into()).extract()?,
            leverage: dict.get_item("leverage").unwrap_or(&1.0.into()).extract()?,
            stop_loss,
            take_profit,
            timestamp,
        })
    }
    
    /// 获取交易名义价值
    pub fn get_notional_value(&self) -> f64 {
        self.quantity * self.price * self.leverage
    }
    
    /// 获取所需保证金
    pub fn get_required_margin(&self) -> f64 {
        (self.quantity * self.price) / self.leverage
    }
}

/// 风控结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskResult {
    pub allowed: bool,
    pub reason: String,
    pub risk_level: String,  // "low", "medium", "high", "critical"
    pub suggested_quantity: Option<f64>,
    pub warnings: Vec<String>,
    pub risk_metrics: HashMap<String, f64>,
    pub timestamp: DateTime<Utc>,
}

impl RiskResult {
    /// 转换为Python对象
    pub fn to_py_object(&self, py: Python) -> PyResult<PyObject> {
        let dict = PyDict::new(py);
        dict.set_item("allowed", self.allowed)?;
        dict.set_item("reason", &self.reason)?;
        dict.set_item("risk_level", &self.risk_level)?;
        dict.set_item("suggested_quantity", self.suggested_quantity)?;
        dict.set_item("warnings", &self.warnings)?;
        dict.set_item("timestamp", self.timestamp.to_rfc3339())?;
        
        let metrics_dict = PyDict::new(py);
        for (key, value) in &self.risk_metrics {
            metrics_dict.set_item(key, value)?;
        }
        dict.set_item("risk_metrics", metrics_dict)?;
        
        Ok(dict.into())
    }
}

/// 风控配置
#[derive(Debug, Clone)]
pub struct RiskConfig {
    pub max_position_size: f64,      // 最大仓位大小 (USDT)
    pub max_leverage: f64,           // 最大杠杆倍数
    pub max_daily_loss: f64,         // 日最大亏损 (USDT)
    pub max_drawdown: f64,           // 最大回撤比例
    pub position_size_limit: f64,    // 单币种仓位限制比例
    pub correlation_limit: f64,      // 相关性限制
    pub var_confidence_level: f64,   // VaR置信水平
    pub stress_test_threshold: f64,  // 压力测试阈值
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_position_size: 10000.0,    // 1万USDT
            max_leverage: 3.0,             // 3倍杠杆
            max_daily_loss: 300.0,         // 日最大亏损300 USDT
            max_drawdown: 0.15,            // 15%最大回撤
            position_size_limit: 0.3,      // 单币种最大30%仓位
            correlation_limit: 0.7,        // 相关性限制70%
            var_confidence_level: 0.95,    // 95%置信水平
            stress_test_threshold: 0.2,    // 20%压力测试阈值
        }
    }
}

/// 风控管理器
pub struct RiskManager {
    config: RiskConfig,
    total_capital: f64,
    positions: HashMap<String, Position>,
    daily_pnl: f64,
    created_at: DateTime<Utc>,
}

impl RiskManager {
    /// 创建新的风控管理器
    pub fn new() -> Self {
        Self::with_config(RiskConfig::default(), 5000.0) // 默认5000 USDT资金
    }
    
    /// 使用指定配置创建风控管理器
    pub fn with_config(config: RiskConfig, initial_capital: f64) -> Self {
        Self {
            config,
            total_capital: initial_capital,
            positions: HashMap::new(),
            daily_pnl: 0.0,
            created_at: Utc::now(),
        }
    }
    
    /// 检查交易风险
    pub fn check_trade(&self, position: &Position, trade: &Trade) -> Result<RiskResult, RiskError> {
        let mut warnings = Vec::new();
        let mut risk_metrics = HashMap::new();
        let mut allowed = true;
        let mut risk_level = "low".to_string();
        let mut reason = "Trade approved".to_string();
        let mut suggested_quantity = Some(trade.quantity);
        
        // 1. 杠杆检查
        if trade.leverage > self.config.max_leverage {
            allowed = false;
            reason = format!("Leverage {:.1}x exceeds maximum {:.1}x", trade.leverage, self.config.max_leverage);
            risk_level = "critical".to_string();
        } else if trade.leverage > self.config.max_leverage * 0.8 {
            warnings.push("High leverage warning".to_string());
            risk_level = "high".to_string();
        }
        risk_metrics.insert("leverage".to_string(), trade.leverage);
        
        // 2. 仓位大小检查
        let notional_value = trade.get_notional_value();
        if notional_value > self.config.max_position_size {
            let max_allowed_quantity = self.config.max_position_size / (trade.price * trade.leverage);
            suggested_quantity = Some(max_allowed_quantity);
            
            if notional_value > self.config.max_position_size * 1.5 {
                allowed = false;
                reason = format!("Position size {:.2} USDT exceeds maximum {:.2} USDT", 
                    notional_value, self.config.max_position_size);
                risk_level = "critical".to_string();
            } else {
                warnings.push("Large position size".to_string());
                risk_level = risk_level.max("medium".to_string());
            }
        }
        risk_metrics.insert("notional_value".to_string(), notional_value);
        risk_metrics.insert("max_position_size".to_string(), self.config.max_position_size);
        
        // 3. 保证金检查
        let required_margin = trade.get_required_margin();
        let available_margin = self.calculate_available_margin(position);
        if required_margin > available_margin {
            allowed = false;
            reason = format!("Insufficient margin: required {:.2}, available {:.2}", 
                required_margin, available_margin);
            risk_level = "critical".to_string();
        } else if required_margin > available_margin * 0.8 {
            warnings.push("Low available margin".to_string());
            risk_level = risk_level.max("high".to_string());
        }
        risk_metrics.insert("required_margin".to_string(), required_margin);
        risk_metrics.insert("available_margin".to_string(), available_margin);
        
        // 4. 日亏损限制检查
        let potential_loss = self.calculate_potential_loss(trade);
        if self.daily_pnl + potential_loss < -self.config.max_daily_loss {
            allowed = false;
            reason = format!("Daily loss limit exceeded: current {:.2}, potential {:.2}, limit {:.2}", 
                self.daily_pnl, potential_loss, -self.config.max_daily_loss);
            risk_level = "critical".to_string();
        } else if self.daily_pnl < -self.config.max_daily_loss * 0.7 {
            warnings.push("Approaching daily loss limit".to_string());
            risk_level = risk_level.max("high".to_string());
        }
        risk_metrics.insert("daily_pnl".to_string(), self.daily_pnl);
        risk_metrics.insert("potential_loss".to_string(), potential_loss);
        risk_metrics.insert("max_daily_loss".to_string(), self.config.max_daily_loss);
        
        // 5. 最大回撤检查
        let current_drawdown = self.calculate_current_drawdown(position);
        if current_drawdown > self.config.max_drawdown {
            allowed = false;
            reason = format!("Maximum drawdown exceeded: current {:.2}%, limit {:.2}%", 
                current_drawdown * 100.0, self.config.max_drawdown * 100.0);
            risk_level = "critical".to_string();
        } else if current_drawdown > self.config.max_drawdown * 0.8 {
            warnings.push("High drawdown warning".to_string());
            risk_level = risk_level.max("high".to_string());
        }
        risk_metrics.insert("current_drawdown".to_string(), current_drawdown);
        risk_metrics.insert("max_drawdown".to_string(), self.config.max_drawdown);
        
        // 6. 仓位集中度检查
        let position_concentration = self.calculate_position_concentration(trade);
        if position_concentration > self.config.position_size_limit {
            let max_allowed = self.config.position_size_limit * self.total_capital;
            let adjusted_quantity = max_allowed / (trade.price * trade.leverage);
            suggested_quantity = Some(adjusted_quantity.min(trade.quantity));
            
            if position_concentration > self.config.position_size_limit * 1.2 {
                allowed = false;
                reason = format!("Position concentration {:.1}% exceeds limit {:.1}%", 
                    position_concentration * 100.0, self.config.position_size_limit * 100.0);
                risk_level = "critical".to_string();
            } else {
                warnings.push("High position concentration".to_string());
                risk_level = risk_level.max("medium".to_string());
            }
        }
        risk_metrics.insert("position_concentration".to_string(), position_concentration);
        
        // 7. VaR (Value at Risk) 计算
        let var_estimate = self.calculate_var(trade, self.config.var_confidence_level);
        risk_metrics.insert("var_estimate".to_string(), var_estimate);
        
        if var_estimate > self.total_capital * 0.05 { // VaR超过总资金的5%
            warnings.push("High VaR estimate".to_string());
            risk_level = risk_level.max("high".to_string());
        }
        
        // 8. 流动性风险检查
        let liquidity_risk = self.assess_liquidity_risk(trade);
        risk_metrics.insert("liquidity_risk".to_string(), liquidity_risk);
        
        if liquidity_risk > 0.7 {
            warnings.push("High liquidity risk".to_string());
            risk_level = risk_level.max("medium".to_string());
        }
        
        // 风险等级最终确定
        if !warnings.is_empty() && risk_level == "low" {
            risk_level = "medium".to_string();
        }
        
        Ok(RiskResult {
            allowed,
            reason,
            risk_level,
            suggested_quantity,
            warnings,
            risk_metrics,
            timestamp: Utc::now(),
        })
    }
    
    /// 计算可用保证金
    fn calculate_available_margin(&self, current_position: &Position) -> f64 {
        let mut used_margin = 0.0;
        
        // 计算所有持仓使用的保证金
        for position in self.positions.values() {
            used_margin += position.margin_used;
        }
        
        // 加上当前仓位的保证金
        used_margin += current_position.margin_used;
        
        // 可用保证金 = 总资金 - 已使用保证金 - 未实现亏损
        let total_unrealized_pnl: f64 = self.positions.values()
            .map(|p| p.unrealized_pnl)
            .sum::<f64>() + current_position.unrealized_pnl;
        
        let available = self.total_capital - used_margin + total_unrealized_pnl.min(0.0);
        available.max(0.0)
    }
    
    /// 计算潜在损失
    fn calculate_potential_loss(&self, trade: &Trade) -> f64 {
        // 简化计算：假设最大损失为仓位价值的一定比例
        let position_value = trade.quantity * trade.price;
        let max_loss_rate = match trade.leverage {
            x if x <= 1.0 => 0.1,  // 无杠杆最大亏损10%
            x if x <= 3.0 => 0.15, // 低杠杆最大亏损15%
            x if x <= 5.0 => 0.25, // 中杠杆最大亏损25%
            _ => 0.4,              // 高杠杆最大亏损40%
        };
        
        -(position_value * max_loss_rate)
    }
    
    /// 计算当前回撤
    fn calculate_current_drawdown(&self, _position: &Position) -> f64 {
        let total_unrealized_pnl: f64 = self.positions.values()
            .map(|p| p.unrealized_pnl)
            .sum();
        
        let current_equity = self.total_capital + total_unrealized_pnl + self.daily_pnl;
        let peak_equity = self.total_capital; // 简化：假设初始资金为峰值
        
        if peak_equity > 0.0 {
            ((peak_equity - current_equity) / peak_equity).max(0.0)
        } else {
            0.0
        }
    }
    
    /// 计算仓位集中度
    fn calculate_position_concentration(&self, trade: &Trade) -> f64 {
        let trade_value = trade.get_notional_value();
        let symbol_total_value = self.positions.get(&trade.symbol)
            .map(|p| p.get_position_value())
            .unwrap_or(0.0) + trade_value;
        
        symbol_total_value / self.total_capital
    }
    
    /// 计算VaR (Value at Risk)
    fn calculate_var(&self, trade: &Trade, confidence_level: f64) -> f64 {
        // 简化的VaR计算：基于历史波动率
        let position_value = trade.quantity * trade.price;
        
        // 假设日波动率（这里应该从历史数据计算）
        let daily_volatility = match trade.symbol.as_str() {
            s if s.contains("BTC") => 0.04,  // 比特币日波动率4%
            s if s.contains("ETH") => 0.05,  // 以太坊日波动率5%
            _ => 0.03,                       // 其他币种3%
        };
        
        // 使用正态分布逆函数计算VaR
        let z_score = match confidence_level {
            x if (x - 0.95).abs() < 0.01 => 1.645, // 95%置信度
            x if (x - 0.99).abs() < 0.01 => 2.326, // 99%置信度
            _ => 1.96, // 默认97.5%置信度
        };
        
        position_value * daily_volatility * z_score * trade.leverage
    }
    
    /// 评估流动性风险
    fn assess_liquidity_risk(&self, trade: &Trade) -> f64 {
        // 简化的流动性风险评估
        let position_size = trade.get_notional_value();
        
        // 根据交易对和仓位大小评估流动性风险
        let base_risk = match trade.symbol.as_str() {
            s if s.contains("BTC") || s.contains("ETH") => 0.1, // 主流币种低风险
            _ => 0.3, // 其他币种中等风险
        };
        
        // 仓位越大，流动性风险越高
        let size_risk = if position_size > 50000.0 {
            0.5
        } else if position_size > 10000.0 {
            0.3
        } else {
            0.1
        };
        
        (base_risk + size_risk).min(1.0)
    }
    
    /// 更新仓位信息
    pub fn update_position(&mut self, symbol: String, position: Position) {
        self.positions.insert(symbol, position);
    }
    
    /// 更新日盈亏
    pub fn update_daily_pnl(&mut self, pnl_change: f64) {
        self.daily_pnl += pnl_change;
    }
    
    /// 获取风险统计信息
    pub fn get_risk_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        
        let total_unrealized_pnl: f64 = self.positions.values()
            .map(|p| p.unrealized_pnl)
            .sum();
        
        let current_equity = self.total_capital + total_unrealized_pnl + self.daily_pnl;
        let total_margin_used: f64 = self.positions.values()
            .map(|p| p.margin_used)
            .sum();
        
        stats.insert("total_capital".to_string(), self.total_capital);
        stats.insert("current_equity".to_string(), current_equity);
        stats.insert("unrealized_pnl".to_string(), total_unrealized_pnl);
        stats.insert("daily_pnl".to_string(), self.daily_pnl);
        stats.insert("margin_used".to_string(), total_margin_used);
        stats.insert("margin_ratio".to_string(), total_margin_used / self.total_capital);
        stats.insert("equity_ratio".to_string(), current_equity / self.total_capital);
        stats.insert("position_count".to_string(), self.positions.len() as f64);
        
        stats
    }
}

impl Default for RiskManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_position_calculation() {
        let mut position = Position {
            symbol: "BTC/USDT".to_string(),
            side: "long".to_string(),
            size: 0.1,
            entry_price: 45000.0,
            current_price: 46000.0,
            unrealized_pnl: 0.0,
            margin_used: 1500.0,
            leverage: 3.0,
            timestamp: Utc::now(),
        };
        
        position.calculate_unrealized_pnl();
        assert_eq!(position.unrealized_pnl, 100.0); // (46000 - 45000) * 0.1 = 100
        
        let return_rate = position.get_return_rate();
        assert!((return_rate - 0.022222).abs() < 0.0001); // 约2.22%收益率
    }
    
    #[test]
    fn test_risk_check() {
        let risk_manager = RiskManager::new();
        
        let position = Position {
            symbol: "BTC/USDT".to_string(),
            side: "long".to_string(),
            size: 0.0,
            entry_price: 45000.0,
            current_price: 45000.0,
            unrealized_pnl: 0.0,
            margin_used: 0.0,
            leverage: 1.0,
            timestamp: Utc::now(),
        };
        
        let trade = Trade {
            symbol: "BTC/USDT".to_string(),
            side: "buy".to_string(),
            order_type: "market".to_string(),
            quantity: 0.1,
            price: 45000.0,
            leverage: 2.0,
            stop_loss: Some(43000.0),
            take_profit: Some(47000.0),
            timestamp: Utc::now(),
        };
        
        let result = risk_manager.check_trade(&position, &trade);
        assert!(result.is_ok());
        
        let risk_result = result.unwrap();
        assert!(risk_result.allowed);
        assert!(!risk_result.risk_metrics.is_empty());
    }
    
    #[test]
    fn test_excessive_leverage() {
        let risk_manager = RiskManager::new();
        
        let position = Position {
            symbol: "BTC/USDT".to_string(),
            side: "long".to_string(),
            size: 0.0,
            entry_price: 45000.0,
            current_price: 45000.0,
            unrealized_pnl: 0.0,
            margin_used: 0.0,
            leverage: 1.0,
            timestamp: Utc::now(),
        };
        
        let trade = Trade {
            symbol: "BTC/USDT".to_string(),
            side: "buy".to_string(),
            order_type: "market".to_string(),
            quantity: 0.1,
            price: 45000.0,
            leverage: 10.0, // 超过默认最大杠杆3倍
            stop_loss: None,
            take_profit: None,
            timestamp: Utc::now(),
        };
        
        let result = risk_manager.check_trade(&position, &trade);
        assert!(result.is_ok());
        
        let risk_result = result.unwrap();
        assert!(!risk_result.allowed);
        assert_eq!(risk_result.risk_level, "critical");
    }
}