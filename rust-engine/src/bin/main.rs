//! QuantAnalyzer Pro - 主程序入口
//! 
//! 量化分析引擎的命令行工具

use clap::{App, Arg, SubCommand};
use quant_engine::factors::technical;
use quant_engine::data::{OHLCV, MarketData};
use serde_json;
use std::fs;
use chrono::{DateTime, Utc};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let matches = App::new("QuantAnalyzer Pro")
        .version(env!("CARGO_PKG_VERSION"))
        .author("QuantAnalyzer Team")
        .about("高性能量化分析引擎")
        .subcommand(
            SubCommand::with_name("calculate")
                .about("计算技术指标")
                .arg(
                    Arg::with_name("factor")
                        .short("f")
                        .long("factor")
                        .value_name("FACTOR")
                        .help("要计算的因子类型 (rsi, macd, sma, ema)")
                        .required(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("input")
                        .short("i")
                        .long("input")
                        .value_name("FILE")
                        .help("输入数据文件 (JSON格式)")
                        .required(true)
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("period")
                        .short("p")
                        .long("period")
                        .value_name("PERIOD")
                        .help("计算周期")
                        .default_value("14")
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("output")
                        .short("o")
                        .long("output")
                        .value_name("FILE")
                        .help("输出文件路径")
                        .takes_value(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("benchmark")
                .about("运行性能基准测试")
                .arg(
                    Arg::with_name("factor")
                        .short("f")
                        .long("factor")
                        .value_name("FACTOR")
                        .help("要测试的因子 (all, rsi, macd, sma, ema)")
                        .default_value("all")
                        .takes_value(true),
                )
                .arg(
                    Arg::with_name("size")
                        .short("s")
                        .long("size")
                        .value_name("SIZE")
                        .help("测试数据大小")
                        .default_value("10000")
                        .takes_value(true),
                ),
        )
        .subcommand(
            SubCommand::with_name("validate")
                .about("验证数据文件格式")
                .arg(
                    Arg::with_name("input")
                        .short("i")
                        .long("input")
                        .value_name("FILE")
                        .help("要验证的数据文件")
                        .required(true)
                        .takes_value(true),
                ),
        )
        .get_matches();

    match matches.subcommand() {
        ("calculate", Some(sub_matches)) => {
            let factor = sub_matches.value_of("factor").unwrap();
            let input_file = sub_matches.value_of("input").unwrap();
            let period: usize = sub_matches.value_of("period").unwrap().parse()?;
            let output_file = sub_matches.value_of("output");

            calculate_factor(factor, input_file, period, output_file)?;
        }
        ("benchmark", Some(sub_matches)) => {
            let factor = sub_matches.value_of("factor").unwrap();
            let size: usize = sub_matches.value_of("size").unwrap().parse()?;

            run_benchmark(factor, size)?;
        }
        ("validate", Some(sub_matches)) => {
            let input_file = sub_matches.value_of("input").unwrap();
            validate_data_file(input_file)?;
        }
        _ => {
            println!("使用 --help 查看可用命令");
        }
    }

    Ok(())
}

/// 计算技术因子
fn calculate_factor(
    factor: &str,
    input_file: &str,
    period: usize,
    output_file: Option<&str>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("正在读取数据文件: {}", input_file);
    
    let data_content = fs::read_to_string(input_file)?;
    let market_data: MarketData = serde_json::from_str(&data_content)?;
    
    if market_data.data.is_empty() {
        return Err("数据文件为空".into());
    }

    let prices = market_data.close_prices();
    println!("数据点数量: {}", prices.len());

    let result = match factor {
        "rsi" => {
            println!("计算RSI，周期: {}", period);
            let timer = quant_engine::utils::performance::Timer::new("RSI计算");
            let rsi_values = technical::rsi(&prices, period);
            drop(timer);
            serde_json::to_string_pretty(&rsi_values)?
        }
        "sma" => {
            println!("计算SMA，周期: {}", period);
            let timer = quant_engine::utils::performance::Timer::new("SMA计算");
            let sma_values = technical::sma(&prices, period);
            drop(timer);
            serde_json::to_string_pretty(&sma_values)?
        }
        "ema" => {
            println!("计算EMA，周期: {}", period);
            let timer = quant_engine::utils::performance::Timer::new("EMA计算");
            let ema_values = technical::ema(&prices, period);
            drop(timer);
            serde_json::to_string_pretty(&ema_values)?
        }
        "macd" => {
            println!("计算MACD，快线: 12，慢线: 26，信号线: 9");
            let timer = quant_engine::utils::performance::Timer::new("MACD计算");
            let (macd, signal, histogram) = technical::macd(&prices, 12, 26, 9);
            drop(timer);
            let macd_result = serde_json::json!({
                "macd": macd,
                "signal": signal,
                "histogram": histogram
            });
            serde_json::to_string_pretty(&macd_result)?
        }
        _ => {
            return Err(format!("不支持的因子类型: {}", factor).into());
        }
    };

    match output_file {
        Some(file) => {
            fs::write(file, &result)?;
            println!("结果已保存到: {}", file);
        }
        None => {
            println!("计算结果:");
            println!("{}", result);
        }
    }

    Ok(())
}

/// 运行性能基准测试
fn run_benchmark(factor: &str, size: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("生成测试数据，大小: {}", size);
    
    // 生成模拟价格数据
    let mut prices = Vec::with_capacity(size);
    let mut price = 100.0;
    
    for _ in 0..size {
        price += (rand::random::<f64>() - 0.5) * 2.0; // 随机游走
        prices.push(price.max(1.0)); // 确保价格为正数
    }

    match factor {
        "rsi" | "all" => {
            println!("\n=== RSI基准测试 ===");
            let timer = quant_engine::utils::performance::Timer::new("RSI计算");
            let _result = technical::rsi(&prices, 14);
            let elapsed = timer.elapsed_ms();
            println!("RSI计算时间: {}ms", elapsed);
            println!("每秒处理数据点: {:.0}", size as f64 * 1000.0 / elapsed as f64);
        }
        _ => {}
    }

    match factor {
        "sma" | "all" => {
            println!("\n=== SMA基准测试 ===");
            let timer = quant_engine::utils::performance::Timer::new("SMA计算");
            let _result = technical::sma(&prices, 20);
            let elapsed = timer.elapsed_ms();
            println!("SMA计算时间: {}ms", elapsed);
            println!("每秒处理数据点: {:.0}", size as f64 * 1000.0 / elapsed as f64);
        }
        _ => {}
    }

    match factor {
        "ema" | "all" => {
            println!("\n=== EMA基准测试 ===");
            let timer = quant_engine::utils::performance::Timer::new("EMA计算");
            let _result = technical::ema(&prices, 20);
            let elapsed = timer.elapsed_ms();
            println!("EMA计算时间: {}ms", elapsed);
            println!("每秒处理数据点: {:.0}", size as f64 * 1000.0 / elapsed as f64);
        }
        _ => {}
    }

    match factor {
        "macd" | "all" => {
            println!("\n=== MACD基准测试 ===");
            let timer = quant_engine::utils::performance::Timer::new("MACD计算");
            let _result = technical::macd(&prices, 12, 26, 9);
            let elapsed = timer.elapsed_ms();
            println!("MACD计算时间: {}ms", elapsed);
            println!("每秒处理数据点: {:.0}", size as f64 * 1000.0 / elapsed as f64);
        }
        _ => {}
    }

    Ok(())
}

/// 验证数据文件格式
fn validate_data_file(input_file: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("验证数据文件: {}", input_file);
    
    let data_content = fs::read_to_string(input_file)?;
    let market_data: MarketData = serde_json::from_str(&data_content)?;
    
    println!("数据点数量: {}", market_data.data.len());
    
    if market_data.data.is_empty() {
        println!("⚠️  数据文件为空");
        return Ok(());
    }
    
    // 验证数据完整性
    let mut valid_count = 0;
    let mut invalid_count = 0;
    
    for (i, ohlcv) in market_data.data.iter().enumerate() {
        if ohlcv.open.is_nan() || ohlcv.high.is_nan() || ohlcv.low.is_nan() || 
           ohlcv.close.is_nan() || ohlcv.volume.is_nan() ||
           ohlcv.open < 0.0 || ohlcv.high < 0.0 || ohlcv.low < 0.0 || 
           ohlcv.close < 0.0 || ohlcv.volume < 0.0 {
            invalid_count += 1;
            if invalid_count <= 5 { // 只显示前5个错误
                println!("❌ 无效数据点 #{}: {:?}", i, ohlcv);
            }
        } else {
            valid_count += 1;
        }
    }
    
    println!("✅ 有效数据点: {}", valid_count);
    if invalid_count > 0 {
        println!("❌ 无效数据点: {}", invalid_count);
    } else {
        println!("🎉 所有数据点均有效！");
    }
    
    // 显示数据范围
    if let (Some(first), Some(last)) = (market_data.data.first(), market_data.data.last()) {
        println!("时间范围: {} 到 {}", first.timestamp, last.timestamp);
        println!("价格范围: {:.2} - {:.2}", 
            market_data.data.iter().map(|d| d.low).fold(f64::INFINITY, f64::min),
            market_data.data.iter().map(|d| d.high).fold(f64::NEG_INFINITY, f64::max)
        );
    }
    
    Ok(())
}

// 简单的随机数生成（用于基准测试）
mod rand {
    use std::cell::RefCell;
    
    thread_local! {
        static RNG: RefCell<u64> = RefCell::new(1);
    }
    
    pub fn random<T: From<f64>>() -> T {
        RNG.with(|rng| {
            let mut seed = rng.borrow_mut();
            *seed = (*seed).wrapping_mul(1103515245).wrapping_add(12345);
            T::from((*seed as f64 / u64::MAX as f64))
        })
    }
}