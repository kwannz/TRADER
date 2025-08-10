//! 因子计算性能基准测试

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use quant_engine::factors::technical;

fn generate_test_data(size: usize) -> Vec<f64> {
    let mut prices = Vec::with_capacity(size);
    let mut price = 100.0;
    
    // 生成随机游走价格数据
    for i in 0..size {
        let noise = ((i as f64 * 1.23456) % 1.0 - 0.5) * 0.02; // 伪随机噪声
        price *= 1.0 + noise;
        prices.push(price.max(1.0));
    }
    
    prices
}

fn bench_rsi(c: &mut Criterion) {
    let mut group = c.benchmark_group("RSI");
    
    for size in [1000, 5000, 10000, 50000].iter() {
        let prices = generate_test_data(*size);
        group.bench_with_input(BenchmarkId::new("calculate", size), size, |b, _| {
            b.iter(|| technical::rsi(black_box(&prices), black_box(14)))
        });
    }
    group.finish();
}

fn bench_sma(c: &mut Criterion) {
    let mut group = c.benchmark_group("SMA");
    
    for size in [1000, 5000, 10000, 50000].iter() {
        let prices = generate_test_data(*size);
        for period in [20, 50, 200].iter() {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("size_{}_period_{}", size, period)), 
                size, 
                |b, _| {
                    b.iter(|| technical::sma(black_box(&prices), black_box(*period)))
                }
            );
        }
    }
    group.finish();
}

fn bench_ema(c: &mut Criterion) {
    let mut group = c.benchmark_group("EMA");
    
    for size in [1000, 5000, 10000, 50000].iter() {
        let prices = generate_test_data(*size);
        for period in [12, 26, 50].iter() {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!("size_{}_period_{}", size, period)),
                size,
                |b, _| {
                    b.iter(|| technical::ema(black_box(&prices), black_box(*period)))
                }
            );
        }
    }
    group.finish();
}

fn bench_macd(c: &mut Criterion) {
    let mut group = c.benchmark_group("MACD");
    
    for size in [1000, 5000, 10000, 50000].iter() {
        let prices = generate_test_data(*size);
        group.bench_with_input(BenchmarkId::new("calculate", size), size, |b, _| {
            b.iter(|| technical::macd(black_box(&prices), black_box(12), black_box(26), black_box(9)))
        });
    }
    group.finish();
}

fn bench_bollinger_bands(c: &mut Criterion) {
    let mut group = c.benchmark_group("Bollinger_Bands");
    
    for size in [1000, 5000, 10000, 50000].iter() {
        let prices = generate_test_data(*size);
        group.bench_with_input(BenchmarkId::new("calculate", size), size, |b, _| {
            b.iter(|| technical::bollinger_bands(black_box(&prices), black_box(20), black_box(2.0)))
        });
    }
    group.finish();
}

fn bench_stochastic(c: &mut Criterion) {
    let mut group = c.benchmark_group("Stochastic");
    
    for size in [1000, 5000, 10000].iter() {
        let prices = generate_test_data(*size);
        
        // 生成高低价数据
        let high: Vec<f64> = prices.iter().map(|&p| p * 1.02).collect();
        let low: Vec<f64> = prices.iter().map(|&p| p * 0.98).collect();
        
        group.bench_with_input(BenchmarkId::new("calculate", size), size, |b, _| {
            b.iter(|| technical::stochastic(
                black_box(&high), 
                black_box(&low), 
                black_box(&prices), 
                black_box(14), 
                black_box(3)
            ))
        });
    }
    group.finish();
}

// 综合性能测试
fn bench_mixed_indicators(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mixed_Indicators");
    
    let size = 10000;
    let prices = generate_test_data(size);
    
    group.bench_function("all_indicators", |b| {
        b.iter(|| {
            let _rsi = technical::rsi(black_box(&prices), 14);
            let _sma_20 = technical::sma(black_box(&prices), 20);
            let _ema_12 = technical::ema(black_box(&prices), 12);
            let _macd = technical::macd(black_box(&prices), 12, 26, 9);
            let _bb = technical::bollinger_bands(black_box(&prices), 20, 2.0);
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_rsi,
    bench_sma,
    bench_ema,
    bench_macd,
    bench_bollinger_bands,
    bench_stochastic,
    bench_mixed_indicators
);
criterion_main!(benches);