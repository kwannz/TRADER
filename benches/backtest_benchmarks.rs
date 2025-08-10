//! 回测引擎性能基准测试

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use quant_engine::backtest::{BacktestEngine, Signal};
use quant_engine::data::OHLCV;
use chrono::{DateTime, Utc, TimeZone};

fn generate_market_data(size: usize) -> Vec<OHLCV> {
    let mut data = Vec::with_capacity(size);
    let mut price = 100.0;
    let start_time = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
    
    for i in 0..size {
        let noise = ((i as f64 * 1.23456) % 1.0 - 0.5) * 0.02;
        price *= 1.0 + noise;
        price = price.max(1.0);
        
        let timestamp = start_time + chrono::Duration::minutes(i as i64);
        let open = price;
        let high = price * 1.01;
        let low = price * 0.99;
        let close = price * (1.0 + noise * 0.5);
        let volume = 1000.0 + (i as f64 * 17.0) % 2000.0;
        
        data.push(OHLCV::new(
            timestamp,
            "BTCUSDT".to_string(),
            open,
            high,
            low,
            close,
            volume,
        ));
    }
    
    data
}

fn simple_moving_average_strategy(ohlcv: &OHLCV, account: &quant_engine::backtest::Account) -> Signal {
    // 简单的移动平均策略示例
    // 这里只是一个示例，实际策略会更复杂
    if ohlcv.close > ohlcv.open {
        if account.cash > ohlcv.close * 10.0 {
            Signal::Buy
        } else {
            Signal::Hold
        }
    } else if ohlcv.close < ohlcv.open * 0.98 {
        Signal::Sell
    } else {
        Signal::Hold
    }
}

fn bench_backtest_engine(c: &mut Criterion) {
    let mut group = c.benchmark_group("Backtest_Engine");
    
    for size in [1000, 5000, 10000].iter() {
        let data = generate_market_data(*size);
        
        group.bench_with_input(
            BenchmarkId::new("simple_strategy", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut engine = BacktestEngine::new(100000.0, 0.001);
                    engine.run_backtest(black_box(&data), simple_moving_average_strategy)
                })
            }
        );
    }
    group.finish();
}

fn bench_order_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("Order_Execution");
    
    let mut engine = BacktestEngine::new(100000.0, 0.001);
    let timestamp = Utc.with_ymd_and_hms(2023, 1, 1, 0, 0, 0).unwrap();
    
    group.bench_function("market_buy_orders", |b| {
        b.iter(|| {
            for i in 0..1000 {
                let price = 100.0 + (i as f64 % 10.0);
                engine.execute_market_order(
                    "BTCUSDT",
                    Signal::Buy,
                    0.1,
                    price,
                    timestamp,
                );
            }
        })
    });
    
    group.finish();
}

fn bench_portfolio_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("Portfolio_Updates");
    
    let data = generate_market_data(1000);
    let mut engine = BacktestEngine::new(100000.0, 0.001);
    
    // 先执行一些交易以建立持仓
    for (i, ohlcv) in data.iter().take(100).enumerate() {
        if i % 10 == 0 {
            engine.execute_market_order(
                &ohlcv.symbol,
                Signal::Buy,
                0.1,
                ohlcv.close,
                ohlcv.timestamp,
            );
        }
    }
    
    group.bench_function("update_portfolio_values", |b| {
        b.iter(|| {
            let mut prices = std::collections::HashMap::new();
            for ohlcv in &data {
                prices.insert(ohlcv.symbol.clone(), ohlcv.close);
                engine.account.update_total_value(black_box(&prices));
            }
        })
    });
    
    group.finish();
}

// 测试不同复杂度的策略
fn complex_strategy(ohlcv: &OHLCV, _account: &quant_engine::backtest::Account) -> Signal {
    // 模拟一个复杂策略，涉及多个计算
    let sma_short = if ohlcv.close > 50.0 { ohlcv.close * 0.95 } else { ohlcv.close * 1.05 };
    let sma_long = ohlcv.close * 0.98;
    let rsi_like = ((ohlcv.close - ohlcv.open) / ohlcv.open * 100.0 + 50.0).min(100.0).max(0.0);
    
    if sma_short > sma_long && rsi_like < 30.0 {
        Signal::Buy
    } else if sma_short < sma_long && rsi_like > 70.0 {
        Signal::Sell
    } else {
        Signal::Hold
    }
}

fn bench_strategy_complexity(c: &mut Criterion) {
    let mut group = c.benchmark_group("Strategy_Complexity");
    
    let data = generate_market_data(5000);
    
    group.bench_function("simple_strategy", |b| {
        b.iter(|| {
            let mut engine = BacktestEngine::new(100000.0, 0.001);
            engine.run_backtest(black_box(&data), simple_moving_average_strategy)
        })
    });
    
    group.bench_function("complex_strategy", |b| {
        b.iter(|| {
            let mut engine = BacktestEngine::new(100000.0, 0.001);
            engine.run_backtest(black_box(&data), complex_strategy)
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_backtest_engine,
    bench_order_execution,
    bench_portfolio_updates,
    bench_strategy_complexity
);
criterion_main!(benches);