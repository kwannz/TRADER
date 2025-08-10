#!/usr/bin/env python3
"""
Crypto PandaFactor Demo
加密货币专用因子演示程序
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from src.factor_engine.crypto_specialized.crypto_factor_utils import CryptoFactorUtils, CryptoDataProcessor

def print_header(title):
    """打印标题"""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print('='*60)

def print_section(title):
    """打印小节标题"""
    print(f"\n💰 {title}")
    print('-'*40)

def main():
    """加密货币因子演示主程序"""
    print_header("Crypto PandaFactor Professional - 加密货币专用因子演示")
    
    # 初始化工具
    crypto_utils = CryptoFactorUtils()
    data_processor = CryptoDataProcessor()
    
    print("✅ 初始化完成 - 加密货币专用因子工具已就绪")
    print(f"📊 支持 {len([attr for attr in dir(crypto_utils) if not attr.startswith('_') and callable(getattr(crypto_utils, attr))])} 个专用加密因子")
    
    # 生成模拟数据
    print_section("生成模拟加密货币市场数据")
    
    # 时间范围：最近30天的小时数据
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    dates = pd.date_range(start=start_date, end=end_date, freq='1H')[:500]  # 限制500个数据点
    
    np.random.seed(42)  # 固定随机种子确保可重现
    
    # BTC价格数据（几何布朗运动 + 跳跃）
    base_price = 45000
    n_points = len(dates)
    
    # 价格路径生成
    mu = 0.0001  # 微小上升趋势
    sigma = 0.02  # 2%波动率
    dt = 1/24     # 小时时间步长
    
    dW = np.random.normal(0, np.sqrt(dt), n_points)
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * dW
    
    # 添加跳跃（加密市场特色）
    jump_intensity = 0.03
    jump_size = np.random.normal(0, 0.08, n_points)
    jumps = np.random.binomial(1, jump_intensity, n_points) * jump_size
    
    log_returns = drift + diffusion + jumps
    log_prices = np.log(base_price) + np.cumsum(log_returns)
    prices = np.exp(log_prices)
    
    # 生成相关数据
    volume = np.random.lognormal(13, 0.5, n_points)  # 成交量
    amount = prices * volume  # 成交额
    
    # 资金费率数据（8小时一次）
    funding_dates = pd.date_range(start=start_date, end=end_date, freq='8H')[:62]  
    funding_rates = np.random.normal(0.0001, 0.0003, len(funding_dates))
    funding_rates = np.clip(funding_rates, -0.01, 0.01)  # 限制在合理范围
    
    # 创建Series
    price_series = pd.Series(prices, index=dates, name='BTC_price')
    volume_series = pd.Series(volume, index=dates, name='volume')  
    amount_series = pd.Series(amount, index=dates, name='amount')
    funding_series = pd.Series(funding_rates, index=funding_dates, name='funding_rate')
    
    print(f"✅ 生成数据完成:")
    print(f"   📈 价格数据: {len(price_series)} 个小时数据点")
    print(f"   📊 成交量: {volume_series.mean():.0f} 平均成交量")
    print(f"   💰 资金费率: {len(funding_series)} 个费率数据点")
    print(f"   📅 时间范围: {start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}")
    
    # 演示1: 资金费率动量分析
    print_section("演示1: 资金费率动量分析")
    
    funding_momentum = crypto_utils.FUNDING_RATE_MOMENTUM(funding_series, window=24)
    
    print("📈 资金费率动量分析结果:")
    print(f"   当前资金费率: {funding_series.iloc[-1]:+.4f}% ({'多头付费' if funding_series.iloc[-1] > 0 else '空头付费'})")
    print(f"   动量指标: {funding_momentum.iloc[-1]:.3f}")
    print(f"   极端信号: {int((abs(funding_momentum) > 1.5).sum())} 次")
    
    if funding_momentum.iloc[-1] > 1.0:
        print("   💡 建议: 资金费率极端偏多，考虑空头机会")
    elif funding_momentum.iloc[-1] < -1.0:
        print("   💡 建议: 资金费率极端偏空，考虑多头机会")
    else:
        print("   💡 建议: 资金费率正常区间，观察等待")
    
    # 演示2: 巨鲸交易检测
    print_section("演示2: 巨鲸交易检测")
    
    whale_alerts = crypto_utils.WHALE_ALERT(volume_series, amount_series, threshold_std=2.5)
    significant_whales = whale_alerts[abs(whale_alerts) > 1.0]
    
    print("🐋 巨鲸交易检测结果:")
    print(f"   检测到巨鲸交易: {len(significant_whales)} 次")
    
    if len(significant_whales) > 0:
        print("   📊 最近巨鲸活动:")
        for i, (timestamp, alert_value) in enumerate(significant_whales.tail(3).items()):
            impact = "🔴 高影响" if abs(alert_value) > 2 else "🟡 中等影响"
            print(f"      {timestamp.strftime('%m-%d %H:%M')} | 强度: {alert_value:.2f} | {impact}")
    else:
        print("   ✅ 近期无显著巨鲸交易活动")
    
    # 演示3: 恐惧贪婪指数
    print_section("演示3: 恐惧贪婪指数计算")
    
    fear_greed = crypto_utils.FEAR_GREED_INDEX(price_series, volume_series)
    current_fg = fear_greed.iloc[-1]
    
    print("😰 恐惧贪婪指数分析:")
    print(f"   当前指数: {current_fg:.1f}/100")
    
    if current_fg > 75:
        emotion = "极度贪婪"
        advice = "考虑减仓，市场过热"
        emoji = "🔴"
    elif current_fg > 60:
        emotion = "贪婪"  
        advice = "谨慎操作，观察反转信号"
        emoji = "🟠"
    elif current_fg > 40:
        emotion = "中性"
        advice = "保持观察，等待明确方向"
        emoji = "🟡"
    elif current_fg > 25:
        emotion = "恐惧"
        advice = "关注抄底机会"
        emoji = "🔵"
    else:
        emotion = "极度恐惧"
        advice = "优质抄底时机"
        emoji = "🟢"
    
    print(f"   市场情绪: {emoji} {emotion}")
    print(f"   操作建议: {advice}")
    
    # 演示4: 市场制度识别
    print_section("演示4: 市场制度识别")
    
    market_regimes = data_processor.detect_market_regime(price_series, volume_series)
    current_regime = market_regimes.iloc[-1] if not pd.isna(market_regimes.iloc[-1]) else "未知"
    
    regime_desc = {
        'bull_quiet': '🟢 牛市-低波动 (稳健上涨)',
        'bull_volatile': '🟡 牛市-高波动 (剧烈上涨)',
        'bear_quiet': '🔵 熊市-低波动 (缓慢下跌)',
        'bear_volatile': '🔴 熊市-高波动 (快速下跌)',
        'sideways': '⚪ 横盘整理 (震荡行情)'
    }
    
    print("📊 市场制度分析:")
    print(f"   当前制度: {regime_desc.get(current_regime, '❓ 未识别')}")
    
    # 制度分布
    regime_counts = market_regimes.value_counts()
    print("   历史分布:")
    for regime, count in regime_counts.items():
        if pd.notna(regime):
            pct = count / len(market_regimes) * 100
            print(f"      {regime_desc.get(regime, regime)}: {pct:.1f}%")
    
    # 演示5: 清算风险分析
    print_section("演示5: 清算瀑布风险评估")
    
    # 生成模拟持仓量数据
    open_interest = pd.Series(
        np.random.lognormal(15, 0.3, len(dates)), 
        index=dates, 
        name='open_interest'
    )
    
    # 需要将资金费率重采样到小时级别用于清算风险计算
    funding_resampled = funding_series.reindex(dates, method='ffill')
    
    liquidation_risk = crypto_utils.LIQUIDATION_CASCADE_RISK(
        price_series, open_interest, funding_resampled, window=72
    )
    
    current_risk = liquidation_risk.iloc[-1]
    
    print("💥 清算风险分析:")
    print(f"   当前风险指数: {current_risk:.3f}")
    
    if current_risk > 0.8:
        risk_level = "🔴 极高风险"
        advice = "避免高杠杆，准备应对清算潮"
    elif current_risk > 0.6:
        risk_level = "🟠 高风险"
        advice = "减少杠杆倍数，控制仓位"
    elif current_risk > 0.4:
        risk_level = "🟡 中等风险" 
        advice = "正常操作，关注资金费率"
    else:
        risk_level = "🟢 低风险"
        advice = "相对安全，可考虑适度杠杆"
    
    print(f"   风险等级: {risk_level}")
    print(f"   操作建议: {advice}")
    
    # 演示6: 数据清洗功能
    print_section("演示6: 加密货币数据清洗")
    
    # 创建包含异常值的测试数据
    test_data = pd.DataFrame({
        'open': prices * 0.999,
        'high': prices * 1.002, 
        'low': prices * 0.998,
        'close': prices,
        'volume': volume
    })
    
    # 人为添加一些异常值
    test_data.iloc[100:105, :4] *= 2  # 价格异常跳跃
    test_data.iloc[200, 4] = 0  # 零成交量
    
    print(f"📊 清洗前数据质量:")
    print(f"   异常价格跳跃: {((test_data['close'].pct_change().abs() > 0.5).sum())} 个")
    print(f"   零成交量点: {(test_data['volume'] == 0).sum()} 个")
    
    # 清洗数据
    cleaned_data = data_processor.clean_crypto_data(test_data)
    
    print(f"✅ 清洗后数据质量:")
    print(f"   异常价格跳跃: {((cleaned_data['close'].pct_change().abs() > 0.5).sum())} 个")  
    print(f"   零成交量点: {(cleaned_data['volume'] == 0).sum()} 个")
    print(f"   OHLC逻辑一致性: {'✅ 正常' if (cleaned_data['high'] >= cleaned_data[['open', 'close']].max(axis=1)).all() else '❌ 异常'}")
    
    # 综合分析报告
    print_section("综合分析报告")
    
    print("🎯 市场状态综合评估:")
    print(f"   📈 价格趋势: {price_series.iloc[-1]/price_series.iloc[0]-1:+.2%} (期间收益)")
    print(f"   😰 市场情绪: {emotion} ({current_fg:.0f}/100)")
    print(f"   📊 市场制度: {regime_desc.get(current_regime, '未知').split(' ')[1] if current_regime != '未知' else '未知'}")
    print(f"   💰 资金费率: {'偏多头' if funding_series.iloc[-1] > 0 else '偏空头'} ({funding_series.iloc[-1]:+.4f}%)")
    print(f"   🐋 巨鲸活跃: {'活跃' if len(significant_whales) > 5 else '平静'} ({len(significant_whales)} 次检测)")
    print(f"   💥 清算风险: {risk_level.split(' ')[1]} ({current_risk:.2f})")
    
    print("\n🎉 加密货币专用因子演示完成！")
    print("\n💡 下一步建议:")
    print("   1. 使用 'python src/cli/crypto_cli.py' 启动完整CLI")
    print("   2. 尝试创建自定义加密因子组合")  
    print("   3. 接入实时交易所数据进行实盘分析")
    print("   4. 结合AI助手优化交易策略")
    print("\n📚 更多功能请查看:")
    print("   • 跨链套利分析 (CROSS_CHAIN_CORRELATION)")
    print("   • DeFi生态关联 (DEFI_TVL_CORRELATION)")
    print("   • 收益农场压力 (YIELD_FARMING_PRESSURE)")
    print("   • 矿工投降指标 (MINER_CAPITULATION)")
    

if __name__ == "__main__":
    main()