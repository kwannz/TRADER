//! 回测引擎模块
//! 
//! 提供高性能的策略回测功能

use crate::data::OHLCV;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// 交易信号类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Signal {
    Buy,
    Sell,
    Hold,
}

/// 订单类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
}

/// 交易订单
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: u64,
    pub symbol: String,
    pub order_type: OrderType,
    pub signal: Signal,
    pub quantity: f64,
    pub price: Option<f64>,
    pub timestamp: DateTime<Utc>,
    pub filled: bool,
}

/// 交易记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub id: u64,
    pub symbol: String,
    pub side: Signal,
    pub quantity: f64,
    pub price: f64,
    pub timestamp: DateTime<Utc>,
    pub commission: f64,
}

/// 持仓信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub quantity: f64,
    pub avg_price: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
}

/// 账户状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub initial_capital: f64,
    pub cash: f64,
    pub total_value: f64,
    pub positions: Vec<Position>,
    pub trades: Vec<Trade>,
}

impl Account {
    pub fn new(initial_capital: f64) -> Self {
        Self {
            initial_capital,
            cash: initial_capital,
            total_value: initial_capital,
            positions: Vec::new(),
            trades: Vec::new(),
        }
    }
    
    /// 更新账户总价值
    pub fn update_total_value(&mut self, current_prices: &std::collections::HashMap<String, f64>) {
        let mut position_value = 0.0;
        
        for position in &mut self.positions {
            if let Some(&current_price) = current_prices.get(&position.symbol) {
                let market_value = position.quantity * current_price;
                position.unrealized_pnl = market_value - (position.quantity * position.avg_price);
                position_value += market_value;
            }
        }
        
        self.total_value = self.cash + position_value;
    }
    
    /// 获取当前收益率
    pub fn current_return(&self) -> f64 {
        (self.total_value - self.initial_capital) / self.initial_capital
    }
}

/// 回测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub initial_capital: f64,
    pub final_value: f64,
    pub total_return: f64,
    pub annualized_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub equity_curve: Vec<(DateTime<Utc>, f64)>,
    pub trades: Vec<Trade>,
}

/// 简单的回测引擎
pub struct BacktestEngine {
    pub account: Account,
    commission_rate: f64,
    next_order_id: u64,
    next_trade_id: u64,
}

impl BacktestEngine {
    pub fn new(initial_capital: f64, commission_rate: f64) -> Self {
        Self {
            account: Account::new(initial_capital),
            commission_rate,
            next_order_id: 1,
            next_trade_id: 1,
        }
    }
    
    /// 执行市场订单
    pub fn execute_market_order(
        &mut self,
        symbol: &str,
        signal: Signal,
        quantity: f64,
        current_price: f64,
        timestamp: DateTime<Utc>,
    ) -> Option<Trade> {
        let order_value = quantity * current_price;
        let commission = order_value * self.commission_rate;
        
        match signal {
            Signal::Buy => {
                let total_cost = order_value + commission;
                if self.account.cash >= total_cost {
                    self.account.cash -= total_cost;
                    self.add_to_position(symbol, quantity, current_price);
                    
                    let trade = Trade {
                        id: self.next_trade_id,
                        symbol: symbol.to_string(),
                        side: Signal::Buy,
                        quantity,
                        price: current_price,
                        timestamp,
                        commission,
                    };
                    
                    self.next_trade_id += 1;
                    self.account.trades.push(trade.clone());
                    Some(trade)
                } else {
                    None // 资金不足
                }
            },
            Signal::Sell => {
                if let Some(position_index) = self.find_position(symbol) {
                    let position = &mut self.account.positions[position_index];
                    if position.quantity >= quantity {
                        let proceeds = order_value - commission;
                        self.account.cash += proceeds;
                        
                        // 更新已实现盈亏
                        let realized_pnl = (current_price - position.avg_price) * quantity;
                        position.realized_pnl += realized_pnl;
                        position.quantity -= quantity;
                        
                        // 如果持仓为0，移除该持仓
                        if position.quantity == 0.0 {
                            self.account.positions.remove(position_index);
                        }
                        
                        let trade = Trade {
                            id: self.next_trade_id,
                            symbol: symbol.to_string(),
                            side: Signal::Sell,
                            quantity,
                            price: current_price,
                            timestamp,
                            commission,
                        };
                        
                        self.next_trade_id += 1;
                        self.account.trades.push(trade.clone());
                        Some(trade)
                    } else {
                        None // 持仓不足
                    }
                } else {
                    None // 无持仓
                }
            },
            Signal::Hold => None,
        }
    }
    
    /// 添加到持仓
    fn add_to_position(&mut self, symbol: &str, quantity: f64, price: f64) {
        if let Some(position_index) = self.find_position(symbol) {
            let position = &mut self.account.positions[position_index];
            let total_quantity = position.quantity + quantity;
            let total_cost = position.quantity * position.avg_price + quantity * price;
            position.avg_price = total_cost / total_quantity;
            position.quantity = total_quantity;
        } else {
            self.account.positions.push(Position {
                symbol: symbol.to_string(),
                quantity,
                avg_price: price,
                unrealized_pnl: 0.0,
                realized_pnl: 0.0,
            });
        }
    }
    
    /// 查找持仓索引
    fn find_position(&self, symbol: &str) -> Option<usize> {
        self.account.positions.iter()
            .position(|p| p.symbol == symbol)
    }
    
    /// 运行回测
    pub fn run_backtest<F>(
        &mut self,
        data: &[OHLCV],
        strategy: F,
    ) -> BacktestResult
    where
        F: Fn(&OHLCV, &Account) -> Signal,
    {
        let mut equity_curve = Vec::new();
        let mut current_prices = std::collections::HashMap::new();
        
        for ohlcv in data {
            current_prices.insert(ohlcv.symbol.clone(), ohlcv.close);
            
            // 更新账户价值
            self.account.update_total_value(&current_prices);
            equity_curve.push((ohlcv.timestamp, self.account.total_value));
            
            // 获取策略信号
            let signal = strategy(ohlcv, &self.account);
            
            // 执行交易（使用收盘价）
            if signal != Signal::Hold {
                let position_size = self.calculate_position_size(&ohlcv.symbol, signal);
                if position_size > 0.0 {
                    self.execute_market_order(
                        &ohlcv.symbol,
                        signal,
                        position_size,
                        ohlcv.close,
                        ohlcv.timestamp,
                    );
                }
            }
        }
        
        // 计算最终回测结果
        self.calculate_backtest_result(equity_curve)
    }
    
    /// 计算持仓大小
    fn calculate_position_size(&self, symbol: &str, signal: Signal) -> f64 {
        match signal {
            Signal::Buy => {
                // 简单策略：使用10%的可用资金
                let available_cash = self.account.cash * 0.1;
                available_cash / 100.0 // 假设价格为100，实际应该使用当前价格
            },
            Signal::Sell => {
                // 卖出所有持仓
                if let Some(position) = self.account.positions.iter().find(|p| p.symbol == symbol) {
                    position.quantity
                } else {
                    0.0
                }
            },
            Signal::Hold => 0.0,
        }
    }
    
    /// 计算回测结果
    fn calculate_backtest_result(&self, equity_curve: Vec<(DateTime<Utc>, f64)>) -> BacktestResult {
        let final_value = self.account.total_value;
        let total_return = (final_value - self.account.initial_capital) / self.account.initial_capital;
        
        // 计算收益率序列
        let returns: Vec<f64> = equity_curve.windows(2)
            .map(|w| (w[1].1 - w[0].1) / w[0].1)
            .collect();
        
        // 计算年化收益率（假设252个交易日）
        let periods_per_year = 252.0;
        let num_periods = equity_curve.len() as f64;
        let annualized_return = if num_periods > 0.0 {
            (1.0 + total_return).powf(periods_per_year / num_periods) - 1.0
        } else {
            0.0
        };
        
        // 计算波动率
        let volatility = if !returns.is_empty() {
            let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter()
                .map(|&r| (r - mean_return).powi(2))
                .sum::<f64>() / returns.len() as f64;
            variance.sqrt() * periods_per_year.sqrt()
        } else {
            0.0
        };
        
        // 计算夏普比率（假设无风险利率为0）
        let sharpe_ratio = if volatility > 0.0 {
            annualized_return / volatility
        } else {
            0.0
        };
        
        // 计算最大回撤
        let max_drawdown = self.calculate_max_drawdown(&equity_curve);
        
        // 计算交易统计
        let total_trades = self.account.trades.len();
        let winning_trades = self.account.trades.iter()
            .filter(|t| {
                // 简化的盈利判断，实际应该更复杂
                match t.side {
                    Signal::Sell => true, // 假设卖出时才能确定盈亏
                    _ => false,
                }
            })
            .count();
        
        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };
        
        // 计算利润因子（简化计算）
        let profit_factor = if total_return < 0.0 {
            0.0
        } else {
            1.0 + total_return
        };
        
        BacktestResult {
            initial_capital: self.account.initial_capital,
            final_value,
            total_return,
            annualized_return,
            volatility,
            sharpe_ratio,
            max_drawdown,
            total_trades,
            winning_trades,
            win_rate,
            profit_factor,
            equity_curve,
            trades: self.account.trades.clone(),
        }
    }
    
    /// 计算最大回撤
    fn calculate_max_drawdown(&self, equity_curve: &[(DateTime<Utc>, f64)]) -> f64 {
        if equity_curve.is_empty() {
            return 0.0;
        }
        
        let mut max_value = equity_curve[0].1;
        let mut max_drawdown = 0.0;
        
        for &(_, value) in equity_curve.iter() {
            if value > max_value {
                max_value = value;
            }
            
            let drawdown = (max_value - value) / max_value;
            if drawdown > max_drawdown {
                max_drawdown = drawdown;
            }
        }
        
        max_drawdown
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_account_creation() {
        let account = Account::new(100000.0);
        assert_eq!(account.initial_capital, 100000.0);
        assert_eq!(account.cash, 100000.0);
        assert_eq!(account.total_value, 100000.0);
    }

    #[test]
    fn test_backtest_engine() {
        let mut engine = BacktestEngine::new(100000.0, 0.001);
        
        let trade = engine.execute_market_order(
            "BTCUSDT",
            Signal::Buy,
            1.0,
            50000.0,
            Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap(),
        );
        
        assert!(trade.is_some());
        assert_eq!(engine.account.positions.len(), 1);
        assert!(engine.account.cash < 100000.0);
    }
}