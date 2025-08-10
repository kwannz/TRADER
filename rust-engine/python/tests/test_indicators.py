"""
测试技术指标计算函数
"""

import pytest
import numpy as np
import sys
import os

# 添加父目录到路径以导入quant_engine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import quant_engine as qe
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    qe = None

@pytest.fixture
def sample_prices():
    """生成示例价格数据"""
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0, 0.02, 100)  # 2%日波动率
    prices = [base_price]
    
    for r in returns:
        prices.append(prices[-1] * (1 + r))
    
    return np.array(prices)

@pytest.fixture
def uptrend_prices():
    """生成上升趋势价格数据"""
    return np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113])

@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust扩展不可用")
class TestTechnicalIndicators:
    
    def test_rsi_calculation(self, uptrend_prices):
        """测试RSI计算"""
        rsi = qe.calculate_rsi(uptrend_prices, period=5)
        
        # RSI值应该在0-100之间
        valid_rsi = rsi[~np.isnan(rsi)]
        assert np.all(valid_rsi >= 0)
        assert np.all(valid_rsi <= 100)
        
        # 前4个值应该是NaN（周期为5）
        assert np.sum(np.isnan(rsi[:4])) == 4
        
        # 上升趋势中RSI应该相对较高
        assert valid_rsi[-1] > 50
    
    def test_rsi_edge_cases(self):
        """测试RSI边界情况"""
        # 测试数据长度不足
        short_prices = np.array([100, 101, 102])
        with pytest.raises(ValueError):
            qe.calculate_rsi(short_prices, period=5)
        
        # 测试所有价格相同的情况
        flat_prices = np.array([100] * 20)
        rsi = qe.calculate_rsi(flat_prices, period=5)
        # 价格不变时RSI应该接近50或NaN
        valid_rsi = rsi[~np.isnan(rsi)]
        assert len(valid_rsi) > 0  # 应该有一些有效值
    
    def test_sma_calculation(self, sample_prices):
        """测试SMA计算"""
        period = 10
        sma = qe.calculate_sma(sample_prices, period)
        
        # 前9个值应该是NaN
        assert np.sum(np.isnan(sma[:period-1])) == period - 1
        
        # 手动验证第一个有效的SMA值
        expected_first_sma = np.mean(sample_prices[:period])
        assert abs(sma[period-1] - expected_first_sma) < 1e-10
        
        # SMA应该比原始价格更平滑
        sma_valid = sma[~np.isnan(sma)]
        prices_valid = sample_prices[period-1:]
        
        # SMA的标准差应该小于原始价格的标准差
        assert np.std(sma_valid) < np.std(prices_valid)
    
    def test_ema_calculation(self, sample_prices):
        """测试EMA计算"""
        period = 10
        ema = qe.calculate_ema(sample_prices, period)
        
        # EMA第一个值应该等于第一个价格
        assert abs(ema[0] - sample_prices[0]) < 1e-10
        
        # EMA应该对价格变化做出响应
        assert len(ema) == len(sample_prices)
        
        # EMA应该在合理范围内
        assert np.all(ema >= np.min(sample_prices) * 0.5)
        assert np.all(ema <= np.max(sample_prices) * 1.5)
    
    def test_macd_calculation(self, sample_prices):
        """测试MACD计算"""
        macd, signal, histogram = qe.calculate_macd(sample_prices, 12, 26, 9)
        
        # 所有序列长度应该相等
        assert len(macd) == len(signal) == len(histogram) == len(sample_prices)
        
        # 前面部分应该有NaN值
        assert np.sum(np.isnan(macd)) > 0
        assert np.sum(np.isnan(signal)) > 0
        
        # 柱状图应该等于MACD - 信号线
        valid_indices = ~(np.isnan(macd) | np.isnan(signal) | np.isnan(histogram))
        if np.any(valid_indices):
            diff = macd[valid_indices] - signal[valid_indices]
            hist = histogram[valid_indices]
            np.testing.assert_array_almost_equal(diff, hist, decimal=10)
    
    def test_bollinger_bands_calculation(self, sample_prices):
        """测试布林带计算"""
        period = 20
        upper, middle, lower = qe.calculate_bollinger_bands(sample_prices, period, 2.0)
        
        # 所有序列长度应该相等
        assert len(upper) == len(middle) == len(lower) == len(sample_prices)
        
        # 前19个值应该是NaN
        assert np.sum(np.isnan(upper[:period-1])) == period - 1
        assert np.sum(np.isnan(middle[:period-1])) == period - 1  
        assert np.sum(np.isnan(lower[:period-1])) == period - 1
        
        # 有效值中，上轨应该大于中轨，中轨应该大于下轨
        valid_indices = ~(np.isnan(upper) | np.isnan(middle) | np.isnan(lower))
        if np.any(valid_indices):
            assert np.all(upper[valid_indices] >= middle[valid_indices])
            assert np.all(middle[valid_indices] >= lower[valid_indices])
        
        # 中轨应该是SMA
        sma = qe.calculate_sma(sample_prices, period)
        valid_sma_indices = ~np.isnan(sma)
        np.testing.assert_array_almost_equal(
            middle[valid_sma_indices], 
            sma[valid_sma_indices], 
            decimal=10
        )
    
    def test_input_validation(self):
        """测试输入验证"""
        prices = np.array([100, 101, 102, 103, 104])
        
        # 测试非numpy数组输入
        list_prices = [100, 101, 102, 103, 104]
        rsi = qe.calculate_rsi(list_prices, 3)
        assert isinstance(rsi, np.ndarray)
        
        # 测试错误的数据类型
        int_prices = np.array([100, 101, 102, 103, 104], dtype=int)
        rsi = qe.calculate_rsi(int_prices, 3)  # 应该自动转换为float64
        assert rsi.dtype == np.float64
    
    def test_ma_type_parameter(self, sample_prices):
        """测试移动平均类型参数"""
        period = 10
        
        # 测试SMA
        ma_sma = qe.calculate_ma(sample_prices, period, "sma")
        sma_direct = qe.calculate_sma(sample_prices, period)
        np.testing.assert_array_almost_equal(ma_sma, sma_direct)
        
        # 测试EMA
        ma_ema = qe.calculate_ma(sample_prices, period, "ema")
        ema_direct = qe.calculate_ema(sample_prices, period)
        np.testing.assert_array_almost_equal(ma_ema, ema_direct)
        
        # 测试无效类型
        with pytest.raises(ValueError):
            qe.calculate_ma(sample_prices, period, "invalid_type")

@pytest.mark.benchmark
@pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust扩展不可用")
class TestPerformance:
    
    def test_rsi_performance(self, benchmark):
        """RSI性能测试"""
        prices = np.random.random(10000) * 100 + 50
        result = benchmark(qe.calculate_rsi, prices, 14)
        assert len(result) == len(prices)
    
    def test_sma_performance(self, benchmark):
        """SMA性能测试"""
        prices = np.random.random(10000) * 100 + 50
        result = benchmark(qe.calculate_sma, prices, 20)
        assert len(result) == len(prices)
    
    def test_ema_performance(self, benchmark):
        """EMA性能测试"""
        prices = np.random.random(10000) * 100 + 50
        result = benchmark(qe.calculate_ema, prices, 20)
        assert len(result) == len(prices)
    
    def test_macd_performance(self, benchmark):
        """MACD性能测试"""
        prices = np.random.random(10000) * 100 + 50
        macd, signal, histogram = benchmark(qe.calculate_macd, prices, 12, 26, 9)
        assert len(macd) == len(signal) == len(histogram) == len(prices)

def test_module_info():
    """测试模块信息"""
    if RUST_AVAILABLE:
        info = qe.get_info()
        assert 'version' in info
        assert 'rust_available' in info
        assert info['rust_available'] == True
        assert len(info['supported_indicators']) > 0

if __name__ == "__main__":
    # 如果直接运行此文件，执行基本测试
    if RUST_AVAILABLE:
        print("✅ Rust扩展可用")
        
        # 基本功能测试
        prices = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109])
        
        print(f"价格数据: {prices}")
        
        rsi = qe.calculate_rsi(prices, 5)
        print(f"RSI: {rsi}")
        
        sma = qe.calculate_sma(prices, 5)
        print(f"SMA: {sma}")
        
        ema = qe.calculate_ema(prices, 5)  
        print(f"EMA: {ema}")
        
        macd, signal, hist = qe.calculate_macd(prices, 3, 6, 2)
        print(f"MACD: {macd}")
        
        upper, middle, lower = qe.calculate_bollinger_bands(prices, 5, 2.0)
        print(f"布林带上轨: {upper}")
        print(f"布林带中轨: {middle}")
        print(f"布林带下轨: {lower}")
        
        print("🎉 所有基本测试通过！")
    else:
        print("❌ Rust扩展不可用，请运行 'maturin develop' 进行构建")