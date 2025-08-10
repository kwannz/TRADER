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
    let prices = generate_test_data(10000);
    c.bench_function("RSI", |b| {
        b.iter(|| technical::rsi(black_box(&prices), black_box(14)))
    });
}

fn bench_sma(c: &mut Criterion) {
    let prices = generate_test_data(10000);
    c.bench_function("SMA", |b| {
        b.iter(|| technical::sma(black_box(&prices), black_box(20)))
    });
}

fn bench_ema(c: &mut Criterion) {
    let prices = generate_test_data(10000);
    c.bench_function("EMA", |b| {
        b.iter(|| technical::ema(black_box(&prices), black_box(20)))
    });
}

fn bench_macd(c: &mut Criterion) {
    let prices = generate_test_data(10000);
    c.bench_function("MACD", |b| {
        b.iter(|| technical::macd(black_box(&prices), black_box(12), black_box(26), black_box(9)))
    });
}

criterion_group!(benches, bench_rsi, bench_sma, bench_ema, bench_macd);
criterion_main!(benches);