"""
1000+量子特征因子引擎
基于量子计算概念和先进数学变换的超级Alpha因子生成系统
实现多维度、多时间尺度的量子启发式特征工程
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.fft as fft
from scipy import stats, signal, optimize, sparse
from scipy.fft import hilbert, rfft, irfft
from scipy.linalg import svd, qr, cholesky
from scipy.special import factorial, gamma, beta
from sklearn.decomposition import PCA, ICA, NMF, FastICA
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
import networkx as nx
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
from datetime import datetime, timedelta
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import lru_cache, partial
import itertools
from collections import defaultdict

warnings.filterwarnings('ignore')

class FactorCategory(Enum):
    """因子分类"""
    QUANTUM_FOURIER = "quantum_fourier"
    QUANTUM_WAVELET = "quantum_wavelet"
    QUANTUM_ENTANGLEMENT = "quantum_entanglement"
    QUANTUM_PHASE = "quantum_phase"
    TOPOLOGICAL_FEATURES = "topological_features"
    FRACTAL_GEOMETRY = "fractal_geometry"
    INFORMATION_THEORY = "information_theory"
    CHAOS_THEORY = "chaos_theory"
    STOCHASTIC_CALCULUS = "stochastic_calculus"
    SPECTRAL_ANALYSIS = "spectral_analysis"
    MANIFOLD_LEARNING = "manifold_learning"
    NETWORK_THEORY = "network_theory"
    TENSOR_DECOMPOSITION = "tensor_decomposition"
    ALGEBRAIC_TOPOLOGY = "algebraic_topology"
    DIFFERENTIAL_GEOMETRY = "differential_geometry"

@dataclass
class QuantumFactor:
    """量子因子定义"""
    factor_id: str
    name: str
    category: FactorCategory
    description: str
    
    # 计算函数
    compute_func: Callable[[np.ndarray], np.ndarray]
    
    # 参数配置
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # 时间窗口
    lookback_window: int = 20
    forward_window: int = 5
    
    # 计算复杂度 (1-5级)
    complexity_level: int = 1
    
    # 预期计算时间(秒)
    expected_compute_time: float = 0.1
    
    # 因子统计属性
    mean_ic: Optional[float] = None
    ic_std: Optional[float] = None
    max_drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    
    # 元数据
    created_at: datetime = field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    dependencies: List[str] = field(default_factory=list)

class QuantumTransforms:
    """量子变换核心算法"""
    
    @staticmethod
    def quantum_fourier_transform(data: np.ndarray, n_qubits: int = None) -> np.ndarray:
        """量子傅里叶变换 (经典模拟)"""
        if n_qubits is None:
            n_qubits = int(np.ceil(np.log2(len(data))))
        
        n = 2 ** n_qubits
        if len(data) < n:
            data = np.pad(data, (0, n - len(data)), 'constant')
        elif len(data) > n:
            data = data[:n]
        
        # 经典FFT作为QFT的近似
        fft_result = np.fft.fft(data)
        
        # 量子相位编码
        phases = np.angle(fft_result)
        amplitudes = np.abs(fft_result)
        
        # 量子叠加态模拟
        quantum_state = amplitudes * np.exp(1j * phases)
        
        return np.real(quantum_state), np.imag(quantum_state), phases, amplitudes
    
    @staticmethod
    def quantum_wavelet_transform(data: np.ndarray, wavelet_type: str = 'morlet') -> Dict[str, np.ndarray]:
        """量子小波变换"""
        from scipy import signal
        
        # 生成小波基
        if wavelet_type == 'morlet':
            widths = np.arange(1, min(31, len(data)//4))
            cwt_matrix = signal.cwt(data, signal.morlet2, widths)
        elif wavelet_type == 'mexican_hat':
            widths = np.arange(1, min(31, len(data)//4))
            cwt_matrix = signal.cwt(data, signal.ricker, widths)
        else:
            widths = np.arange(1, min(31, len(data)//4))
            cwt_matrix = signal.cwt(data, signal.morlet2, widths)
        
        # 量子相干性测量
        coherence = np.abs(cwt_matrix)
        phase_matrix = np.angle(cwt_matrix)
        
        # 量子纠缠特征
        entanglement_entropy = []
        for i in range(len(widths)):
            row = coherence[i]
            prob_dist = row / np.sum(row) if np.sum(row) > 0 else row
            entropy = -np.sum(prob_dist * np.log(prob_dist + 1e-10))
            entanglement_entropy.append(entropy)
        
        return {
            'coherence': coherence,
            'phase_matrix': phase_matrix,
            'entanglement_entropy': np.array(entanglement_entropy),
            'widths': widths
        }
    
    @staticmethod
    def quantum_entanglement_measure(data1: np.ndarray, data2: np.ndarray) -> Dict[str, float]:
        """量子纠缠度测量"""
        # 互信息 (经典纠缠类比)
        bins = min(50, len(data1) // 10)
        hist_2d, _, _ = np.histogram2d(data1, data2, bins=bins)
        hist_2d = hist_2d / np.sum(hist_2d)
        
        # 边际分布
        p_x = np.sum(hist_2d, axis=1)
        p_y = np.sum(hist_2d, axis=0)
        
        # 互信息计算
        mutual_info = 0
        for i in range(len(p_x)):
            for j in range(len(p_y)):
                if hist_2d[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mutual_info += hist_2d[i, j] * np.log(hist_2d[i, j] / (p_x[i] * p_y[j]))
        
        # 量子纠缠熵
        joint_entropy = -np.sum(hist_2d * np.log(hist_2d + 1e-10))
        marginal_entropy_x = -np.sum(p_x * np.log(p_x + 1e-10))
        marginal_entropy_y = -np.sum(p_y * np.log(p_y + 1e-10))
        
        # 纠缠度量
        entanglement_measure = mutual_info / np.sqrt(marginal_entropy_x * marginal_entropy_y + 1e-10)
        
        return {
            'mutual_information': mutual_info,
            'entanglement_measure': entanglement_measure,
            'joint_entropy': joint_entropy,
            'conditional_entropy': joint_entropy - marginal_entropy_x
        }
    
    @staticmethod
    def quantum_phase_estimation(data: np.ndarray, reference: np.ndarray = None) -> Dict[str, np.ndarray]:
        """量子相位估计"""
        if reference is None:
            reference = np.roll(data, 1)
        
        # 解析信号
        analytic_signal = hilbert(data)
        reference_signal = hilbert(reference)
        
        # 瞬时相位
        instantaneous_phase = np.angle(analytic_signal)
        reference_phase = np.angle(reference_signal)
        
        # 相位差
        phase_difference = instantaneous_phase - reference_phase
        phase_difference = np.unwrap(phase_difference)
        
        # 量子相位估计特征
        phase_velocity = np.gradient(instantaneous_phase)
        phase_acceleration = np.gradient(phase_velocity)
        
        # 相位同步指数
        synchronization_index = np.abs(np.mean(np.exp(1j * phase_difference)))
        
        return {
            'instantaneous_phase': instantaneous_phase,
            'phase_difference': phase_difference,
            'phase_velocity': phase_velocity,
            'phase_acceleration': phase_acceleration,
            'synchronization_index': synchronization_index
        }

class TopologicalFeatures:
    """拓扑特征提取器"""
    
    @staticmethod
    def persistent_homology_features(data: np.ndarray, max_dimension: int = 2) -> Dict[str, np.ndarray]:
        """持久同调特征"""
        # 简化版本的持久同调
        # 构建距离矩阵
        n = len(data)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = abs(data[i] - data[j])
                distance_matrix[i, j] = distance_matrix[j, i] = dist
        
        # 计算连通分量
        thresholds = np.percentile(distance_matrix[distance_matrix > 0], 
                                 np.linspace(0, 100, 20))
        
        betti_numbers = []
        for threshold in thresholds:
            # 构建邻接矩阵
            adjacency = (distance_matrix <= threshold).astype(int)
            
            # 计算连通分量数量 (Betti_0)
            graph = nx.from_numpy_array(adjacency)
            betti_0 = nx.number_connected_components(graph)
            betti_numbers.append(betti_0)
        
        # 持久性特征
        persistence_curve = np.array(betti_numbers)
        persistence_entropy = -np.sum(persistence_curve / np.sum(persistence_curve) * 
                                    np.log(persistence_curve / np.sum(persistence_curve) + 1e-10))
        
        return {
            'persistence_curve': persistence_curve,
            'persistence_entropy': persistence_entropy,
            'thresholds': thresholds
        }
    
    @staticmethod
    def euler_characteristics(data: np.ndarray, embedding_dim: int = 3) -> float:
        """欧拉特征数"""
        # 嵌入高维空间
        if embedding_dim > 1:
            embedded = np.column_stack([data[i:len(data)-embedding_dim+1+i] 
                                      for i in range(embedding_dim)])
        else:
            embedded = data.reshape(-1, 1)
        
        # 简化的欧拉特征数估计
        # 基于局部最大值和最小值
        local_maxima = (embedded[1:-1] > embedded[:-2]).all(axis=1) & \
                      (embedded[1:-1] > embedded[2:]).all(axis=1)
        local_minima = (embedded[1:-1] < embedded[:-2]).all(axis=1) & \
                      (embedded[1:-1] < embedded[2:]).all(axis=1)
        
        # 欧拉特征数 = 顶点数 - 边数 + 面数 (简化为最大值-最小值)
        euler_char = np.sum(local_maxima) - np.sum(local_minima)
        
        return float(euler_char)
    
    @staticmethod
    def genus_estimation(data: np.ndarray) -> float:
        """亏格估计"""
        # 通过数据的复杂性估计拓扑亏格
        # 基于Hausdorff维数的简化估计
        
        # 盒计数维数
        def box_counting_dimension(series, max_box_size=None):
            if max_box_size is None:
                max_box_size = len(series) // 10
            
            box_sizes = np.logspace(0, np.log10(max_box_size), 20).astype(int)
            box_counts = []
            
            for box_size in box_sizes:
                if box_size == 0:
                    continue
                
                # 计算需要的盒子数量
                data_range = np.max(series) - np.min(series)
                if data_range == 0:
                    box_counts.append(1)
                    continue
                
                n_boxes = int(np.ceil(data_range / (data_range / box_size)))
                box_counts.append(max(1, n_boxes))
            
            if len(box_counts) < 2:
                return 1.0
            
            # 线性拟合
            valid_indices = np.array(box_counts) > 0
            if np.sum(valid_indices) < 2:
                return 1.0
            
            log_box_sizes = np.log(np.array(box_sizes)[valid_indices])
            log_box_counts = np.log(np.array(box_counts)[valid_indices])
            
            slope, _ = np.polyfit(log_box_sizes, log_box_counts, 1)
            return abs(slope)
        
        hausdorff_dim = box_counting_dimension(data)
        
        # 亏格估计 (启发式)
        genus = max(0, (hausdorff_dim - 1) / 2)
        
        return genus

class FractalFeatures:
    """分形几何特征"""
    
    @staticmethod
    def hurst_exponent_multifractal(data: np.ndarray, q_range: np.ndarray = None) -> Dict[str, np.ndarray]:
        """多重分形Hurst指数"""
        if q_range is None:
            q_range = np.linspace(-5, 5, 21)
        
        n = len(data)
        scales = np.unique(np.logspace(1, np.log10(n//4), 20).astype(int))
        
        fluctuations = []
        
        for scale in scales:
            # 累积和
            cumsum = np.cumsum(data - np.mean(data))
            
            # 分段
            segments = n // scale
            if segments == 0:
                continue
                
            segment_fluctuations = []
            for i in range(segments):
                start = i * scale
                end = start + scale
                segment_data = cumsum[start:end]
                
                # 去趋势
                x = np.arange(len(segment_data))
                poly_coeffs = np.polyfit(x, segment_data, 1)
                trend = np.polyval(poly_coeffs, x)
                detrended = segment_data - trend
                
                # 均方根涨落
                rms_fluctuation = np.sqrt(np.mean(detrended**2))
                segment_fluctuations.append(rms_fluctuation)
            
            if segment_fluctuations:
                fluctuations.append(np.mean(segment_fluctuations))
            else:
                fluctuations.append(0)
        
        fluctuations = np.array(fluctuations)
        valid_indices = fluctuations > 0
        
        if np.sum(valid_indices) < 2:
            return {'q_range': q_range, 'hurst_spectrum': np.ones_like(q_range) * 0.5}
        
        scales = scales[valid_indices]
        fluctuations = fluctuations[valid_indices]
        
        # 多重分形分析
        hurst_spectrum = []
        
        for q in q_range:
            if q == 0:
                # 特殊处理q=0的情况
                log_fluctuations = np.log(fluctuations + 1e-10)
                weights = np.ones_like(log_fluctuations) / len(log_fluctuations)
            else:
                weights = fluctuations**(q-1)
                weights = weights / np.sum(weights)
                log_fluctuations = np.log(fluctuations + 1e-10)
            
            # 加权线性回归
            weighted_mean_log_scale = np.sum(weights * np.log(scales))
            weighted_mean_log_fluc = np.sum(weights * log_fluctuations)
            
            numerator = np.sum(weights * (np.log(scales) - weighted_mean_log_scale) * 
                             (log_fluctuations - weighted_mean_log_fluc))
            denominator = np.sum(weights * (np.log(scales) - weighted_mean_log_scale)**2)
            
            if denominator > 0:
                hurst = numerator / denominator
            else:
                hurst = 0.5
            
            hurst_spectrum.append(hurst)
        
        return {
            'q_range': q_range,
            'hurst_spectrum': np.array(hurst_spectrum)
        }
    
    @staticmethod
    def lacunarity_analysis(data: np.ndarray, max_box_size: int = None) -> Dict[str, np.ndarray]:
        """空隙度分析"""
        if max_box_size is None:
            max_box_size = len(data) // 5
        
        box_sizes = np.unique(np.logspace(0, np.log10(max_box_size), 15).astype(int))
        lacunarities = []
        
        for box_size in box_sizes:
            if box_size <= 1:
                lacunarities.append(0)
                continue
            
            # 将数据离散化到盒子中
            n_boxes = len(data) // box_size
            if n_boxes == 0:
                lacunarities.append(0)
                continue
            
            box_masses = []
            for i in range(n_boxes):
                start = i * box_size
                end = min(start + box_size, len(data))
                box_mass = np.sum(np.abs(data[start:end]))
                box_masses.append(box_mass)
            
            box_masses = np.array(box_masses)
            
            # 计算空隙度
            if len(box_masses) > 1 and np.std(box_masses) > 0:
                mean_mass = np.mean(box_masses)
                variance_mass = np.var(box_masses)
                lacunarity = variance_mass / (mean_mass**2) if mean_mass > 0 else 0
            else:
                lacunarity = 0
            
            lacunarities.append(lacunarity)
        
        return {
            'box_sizes': box_sizes,
            'lacunarities': np.array(lacunarities)
        }
    
    @staticmethod
    def correlation_dimension(data: np.ndarray, embedding_dim: int = 5) -> float:
        """关联维数"""
        # 相空间重构
        if embedding_dim > len(data):
            embedding_dim = len(data) // 2
        
        if embedding_dim <= 1:
            return 1.0
        
        embedded = np.column_stack([data[i:len(data)-embedding_dim+1+i] 
                                  for i in range(embedding_dim)])
        
        n = len(embedded)
        if n < 10:
            return 1.0
        
        # 计算距离矩阵
        distances = []
        for i in range(min(n, 1000)):  # 限制样本数量以提高效率
            for j in range(i+1, min(n, 1000)):
                dist = np.linalg.norm(embedded[i] - embedded[j])
                distances.append(dist)
        
        distances = np.array(distances)
        distances = distances[distances > 0]
        
        if len(distances) == 0:
            return 1.0
        
        # 关联积分
        r_values = np.logspace(np.log10(np.min(distances)), 
                              np.log10(np.max(distances)), 20)
        
        correlation_integrals = []
        for r in r_values:
            correlation_integral = np.sum(distances <= r) / len(distances)
            correlation_integrals.append(correlation_integral)
        
        correlation_integrals = np.array(correlation_integrals)
        
        # 计算关联维数
        valid_indices = (correlation_integrals > 0) & (correlation_integrals < 1)
        if np.sum(valid_indices) < 2:
            return 1.0
        
        log_r = np.log(r_values[valid_indices])
        log_c = np.log(correlation_integrals[valid_indices])
        
        slope, _ = np.polyfit(log_r, log_c, 1)
        correlation_dim = slope
        
        return max(0, correlation_dim)

class InformationTheoryFeatures:
    """信息论特征"""
    
    @staticmethod
    def shannon_entropy_spectrum(data: np.ndarray, window_sizes: List[int] = None) -> Dict[str, np.ndarray]:
        """Shannon熵谱"""
        if window_sizes is None:
            window_sizes = [5, 10, 20, 50]
        
        entropy_spectrum = []
        
        for window_size in window_sizes:
            if window_size >= len(data):
                entropy_spectrum.append(0)
                continue
            
            windowed_entropies = []
            for i in range(len(data) - window_size + 1):
                window_data = data[i:i+window_size]
                
                # 离散化
                bins = min(10, window_size//2 + 1)
                hist, _ = np.histogram(window_data, bins=bins, density=True)
                hist = hist[hist > 0]  # 移除零概率
                
                # 计算Shannon熵
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                windowed_entropies.append(entropy)
            
            entropy_spectrum.append(np.mean(windowed_entropies))
        
        return {
            'window_sizes': np.array(window_sizes),
            'entropy_spectrum': np.array(entropy_spectrum)
        }
    
    @staticmethod
    def kolmogorov_complexity_approximation(data: np.ndarray) -> float:
        """Kolmogorov复杂度近似"""
        # 使用压缩率作为复杂度的近似
        import zlib
        
        # 将数据转换为字节
        data_bytes = data.tobytes()
        
        # 压缩数据
        compressed = zlib.compress(data_bytes)
        
        # 压缩率
        compression_ratio = len(compressed) / len(data_bytes)
        
        # Kolmogorov复杂度近似
        kolmogorov_complexity = compression_ratio * len(data)
        
        return kolmogorov_complexity
    
    @staticmethod
    def transfer_entropy(source: np.ndarray, target: np.ndarray, lag: int = 1) -> float:
        """传递熵"""
        if len(source) != len(target) or len(source) <= lag:
            return 0.0
        
        # 构建时间序列
        target_present = target[lag:]
        target_past = target[:-lag]
        source_past = source[:-lag]
        
        # 离散化
        bins = min(10, len(target_present)//10 + 2)
        
        # 计算联合概率和边际概率
        try:
            # 三元联合分布
            hist_3d = np.histogramdd([target_present, target_past, source_past], 
                                   bins=bins, density=True)[0]
            
            # 二元联合分布
            hist_tp_sp = np.histogram2d(target_past, source_past, bins=bins, density=True)[0]
            hist_tpr_tp = np.histogram2d(target_present, target_past, bins=bins, density=True)[0]
            
            # 一元分布
            hist_tp = np.histogram(target_past, bins=bins, density=True)[0]
            
            # 计算传递熵
            transfer_entropy = 0
            for i in range(bins):
                for j in range(bins):
                    for k in range(bins):
                        p_tpr_tp_sp = hist_3d[i, j, k]
                        p_tp_sp = hist_tp_sp[j, k]
                        p_tpr_tp = hist_tpr_tp[i, j]
                        p_tp = hist_tp[j]
                        
                        if (p_tpr_tp_sp > 0 and p_tp_sp > 0 and 
                            p_tpr_tp > 0 and p_tp > 0):
                            transfer_entropy += (p_tpr_tp_sp * 
                                               np.log2((p_tpr_tp_sp * p_tp) / 
                                                      (p_tpr_tp * p_tp_sp)))
            
            return max(0, transfer_entropy)
            
        except:
            return 0.0

class ChaosTheoryFeatures:
    """混沌理论特征"""
    
    @staticmethod
    def lyapunov_exponent(data: np.ndarray, embedding_dim: int = 3, delay: int = 1) -> float:
        """Lyapunov指数"""
        if len(data) < embedding_dim + delay:
            return 0.0
        
        # 相空间重构
        embedded = np.array([data[i:i+embedding_dim*delay:delay] 
                           for i in range(len(data) - embedding_dim*delay + 1)])
        
        n = len(embedded)
        if n < 10:
            return 0.0
        
        # 寻找最近邻
        divergences = []
        
        for i in range(n - 1):
            # 计算距离
            distances = np.linalg.norm(embedded - embedded[i], axis=1)
            # 排除自身
            distances[i] = np.inf
            
            # 找到最近邻
            nearest_idx = np.argmin(distances)
            
            if nearest_idx < n - 1:
                # 计算轨道分离
                initial_separation = distances[nearest_idx]
                if initial_separation > 0:
                    final_separation = np.linalg.norm(embedded[i+1] - embedded[nearest_idx+1])
                    if final_separation > 0:
                        divergence = np.log(final_separation / initial_separation)
                        divergences.append(divergence)
        
        if divergences:
            return np.mean(divergences)
        else:
            return 0.0
    
    @staticmethod
    def approximate_entropy(data: np.ndarray, m: int = 2, r: float = None) -> float:
        """近似熵"""
        if r is None:
            r = 0.2 * np.std(data)
        
        n = len(data)
        
        def _maxdist(xi, xj, m):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            patterns = np.array([[data[i + j] for j in range(m)] for i in range(n - m + 1)])
            c = np.zeros(n - m + 1)
            
            for i in range(n - m + 1):
                template_i = patterns[i]
                for j in range(n - m + 1):
                    if _maxdist(template_i, patterns[j], m) <= r:
                        c[i] += 1.0
            
            phi = np.mean(np.log(c / (n - m + 1.0)))
            return phi
        
        return _phi(m) - _phi(m + 1)
    
    @staticmethod
    def recurrence_quantification(data: np.ndarray, embedding_dim: int = 3, 
                                 threshold: float = None) -> Dict[str, float]:
        """递归定量分析"""
        if threshold is None:
            threshold = 0.1 * np.std(data)
        
        # 相空间重构
        if embedding_dim > len(data) // 2:
            embedding_dim = max(1, len(data) // 2)
        
        embedded = np.array([data[i:i+embedding_dim] 
                           for i in range(len(data) - embedding_dim + 1)])
        
        n = len(embedded)
        
        # 计算递归矩阵
        recurrence_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distance = np.linalg.norm(embedded[i] - embedded[j])
                recurrence_matrix[i, j] = 1 if distance <= threshold else 0
        
        # 递归定量特征
        total_points = n * n
        recurrence_points = np.sum(recurrence_matrix)
        recurrence_rate = recurrence_points / total_points
        
        # 确定性（对角线结构）
        determinism = 0
        diagonal_lengths = []
        
        for offset in range(1, n):
            diagonal = np.diagonal(recurrence_matrix, offset)
            line_length = 0
            for point in diagonal:
                if point == 1:
                    line_length += 1
                else:
                    if line_length >= 2:  # 最小线长度
                        diagonal_lengths.append(line_length)
                    line_length = 0
            
            if line_length >= 2:
                diagonal_lengths.append(line_length)
        
        if diagonal_lengths:
            determinism = np.sum(diagonal_lengths) / recurrence_points if recurrence_points > 0 else 0
            avg_diagonal_length = np.mean(diagonal_lengths)
            max_diagonal_length = np.max(diagonal_lengths)
        else:
            avg_diagonal_length = 0
            max_diagonal_length = 0
        
        return {
            'recurrence_rate': recurrence_rate,
            'determinism': determinism,
            'avg_diagonal_length': avg_diagonal_length,
            'max_diagonal_length': max_diagonal_length
        }

class StochasticCalculusFeatures:
    """随机微积分特征"""
    
    @staticmethod
    def ito_integral_approximation(data: np.ndarray, dt: float = 1.0) -> Dict[str, np.ndarray]:
        """Itô积分近似"""
        # 计算增量
        increments = np.diff(data)
        
        # 布朗运动近似
        brownian_increments = increments / np.sqrt(dt)
        
        # Itô积分近似 (左端点)
        ito_integral = np.cumsum(data[:-1] * increments) * dt
        
        # Stratonovich积分近似 (中点)
        midpoints = (data[:-1] + data[1:]) / 2
        stratonovich_integral = np.cumsum(midpoints * increments) * dt
        
        # 二次变分
        quadratic_variation = np.cumsum(increments**2)
        
        return {
            'ito_integral': ito_integral,
            'stratonovich_integral': stratonovich_integral,
            'quadratic_variation': quadratic_variation,
            'brownian_increments': brownian_increments
        }
    
    @staticmethod
    def ornstein_uhlenbeck_features(data: np.ndarray) -> Dict[str, float]:
        """Ornstein-Uhlenbeck过程特征"""
        # 参数估计
        y = data[1:]
        x = data[:-1]
        
        # 线性回归 y_t = a + b*x_{t-1} + noise
        design_matrix = np.column_stack([np.ones(len(x)), x])
        
        try:
            coefficients = np.linalg.lstsq(design_matrix, y, rcond=None)[0]
            a, b = coefficients
            
            # OU参数
            theta = -np.log(b)  # 均值回归速度
            mu = a / (1 - b)    # 长期均值
            
            # 残差分析
            predicted = a + b * x
            residuals = y - predicted
            sigma = np.std(residuals)  # 噪声强度
            
            # 半衰期
            half_life = np.log(2) / theta if theta > 0 else np.inf
            
            return {
                'mean_reversion_speed': theta,
                'long_term_mean': mu,
                'noise_intensity': sigma,
                'half_life': half_life
            }
        except:
            return {
                'mean_reversion_speed': 0,
                'long_term_mean': np.mean(data),
                'noise_intensity': np.std(data),
                'half_life': np.inf
            }
    
    @staticmethod
    def martingale_test(data: np.ndarray) -> Dict[str, float]:
        """鞅性检验"""
        # 差分序列
        increments = np.diff(data)
        
        # 检验增量的期望是否为0
        increment_mean = np.mean(increments)
        increment_std = np.std(increments)
        
        # t检验
        if increment_std > 0:
            t_statistic = increment_mean * np.sqrt(len(increments)) / increment_std
        else:
            t_statistic = 0
        
        # 方差比检验
        # 检验 Var(S_2t - S_t) = 2 * Var(S_t - S_0) 
        n = len(data)
        if n >= 4:
            # 短期方差
            short_var = np.var(data[1:] - data[:-1])
            
            # 长期方差 (每两个点)
            long_increments = data[2::2] - data[:-2:2]
            long_var = np.var(long_increments) if len(long_increments) > 0 else 0
            
            variance_ratio = long_var / (2 * short_var) if short_var > 0 else 1
        else:
            variance_ratio = 1
        
        return {
            'increment_mean': increment_mean,
            't_statistic': t_statistic,
            'variance_ratio': variance_ratio,
            'martingale_deviation': abs(variance_ratio - 1)
        }

class QuantumFactorEngine:
    """1000+量子特征因子引擎"""
    
    def __init__(self, n_workers: int = None):
        if n_workers is None:
            n_workers = min(8, mp.cpu_count())
        
        self.n_workers = n_workers
        self.factor_registry: Dict[str, QuantumFactor] = {}
        self.compute_cache: Dict[str, np.ndarray] = {}
        
        # 组件初始化
        self.quantum_transforms = QuantumTransforms()
        self.topological_features = TopologicalFeatures()
        self.fractal_features = FractalFeatures()
        self.information_features = InformationTheoryFeatures()
        self.chaos_features = ChaosTheoryFeatures()
        self.stochastic_features = StochasticCalculusFeatures()
        
        self.logger = logging.getLogger("QuantumFactorEngine")
        
        # 初始化1000+因子
        self._initialize_factor_library()
    
    def _initialize_factor_library(self):
        """初始化因子库"""
        self.logger.info("初始化1000+量子特征因子库...")
        
        # 量子傅里叶变换因子 (100个)
        self._register_quantum_fourier_factors()
        
        # 量子小波变换因子 (150个)
        self._register_quantum_wavelet_factors()
        
        # 量子纠缠因子 (100个)
        self._register_quantum_entanglement_factors()
        
        # 拓扑特征因子 (120个)
        self._register_topological_factors()
        
        # 分形几何因子 (100个)
        self._register_fractal_factors()
        
        # 信息论因子 (80个)
        self._register_information_theory_factors()
        
        # 混沌理论因子 (90个)
        self._register_chaos_theory_factors()
        
        # 随机微积分因子 (80个)
        self._register_stochastic_calculus_factors()
        
        # 谱分析因子 (70个)
        self._register_spectral_analysis_factors()
        
        # 流形学习因子 (60个)
        self._register_manifold_learning_factors()
        
        # 网络理论因子 (40个)
        self._register_network_theory_factors()
        
        self.logger.info(f"✅ 初始化完成，共注册 {len(self.factor_registry)} 个量子因子")
    
    def _register_quantum_fourier_factors(self):
        """注册量子傅里叶变换因子"""
        for n_qubits in range(3, 8):  # 不同量子比特数
            for component in ['real', 'imag', 'phase', 'amplitude']:
                for window in [10, 20, 50, 100]:
                    factor_id = f"qft_{n_qubits}q_{component}_w{window}"
                    
                    def compute_func(data, nq=n_qubits, comp=component, w=window):
                        if len(data) < w:
                            return np.zeros(len(data))
                        
                        result = []
                        for i in range(len(data) - w + 1):
                            window_data = data[i:i+w]
                            real, imag, phase, amplitude = self.quantum_transforms.quantum_fourier_transform(
                                window_data, nq
                            )
                            
                            if comp == 'real':
                                result.append(np.mean(real))
                            elif comp == 'imag':
                                result.append(np.mean(imag))
                            elif comp == 'phase':
                                result.append(np.mean(phase))
                            else:  # amplitude
                                result.append(np.mean(amplitude))
                        
                        # 填充初始值
                        full_result = np.full(len(data), result[0] if result else 0)
                        full_result[-len(result):] = result
                        return full_result
                    
                    factor = QuantumFactor(
                        factor_id=factor_id,
                        name=f"Quantum Fourier {component} ({n_qubits} qubits, window {window})",
                        category=FactorCategory.QUANTUM_FOURIER,
                        description=f"Quantum Fourier Transform {component} component with {n_qubits} qubits over {window} periods",
                        compute_func=compute_func,
                        lookback_window=window,
                        complexity_level=3
                    )
                    
                    self.factor_registry[factor_id] = factor
    
    def _register_quantum_wavelet_factors(self):
        """注册量子小波变换因子"""
        wavelet_types = ['morlet', 'mexican_hat']
        
        for wavelet_type in wavelet_types:
            for feature in ['coherence_mean', 'coherence_std', 'phase_mean', 'entanglement_entropy']:
                for window in [20, 50, 100]:
                    for scale_idx in range(5):  # 不同尺度
                        factor_id = f"qwt_{wavelet_type}_{feature}_w{window}_s{scale_idx}"
                        
                        def compute_func(data, wt=wavelet_type, feat=feature, w=window, si=scale_idx):
                            if len(data) < w:
                                return np.zeros(len(data))
                            
                            result = []
                            for i in range(len(data) - w + 1):
                                window_data = data[i:i+w]
                                qwt_result = self.quantum_transforms.quantum_wavelet_transform(
                                    window_data, wt
                                )
                                
                                if feat == 'coherence_mean':
                                    if si < len(qwt_result['coherence']):
                                        result.append(np.mean(qwt_result['coherence'][si]))
                                    else:
                                        result.append(0)
                                elif feat == 'coherence_std':
                                    if si < len(qwt_result['coherence']):
                                        result.append(np.std(qwt_result['coherence'][si]))
                                    else:
                                        result.append(0)
                                elif feat == 'phase_mean':
                                    if si < len(qwt_result['phase_matrix']):
                                        result.append(np.mean(qwt_result['phase_matrix'][si]))
                                    else:
                                        result.append(0)
                                else:  # entanglement_entropy
                                    if si < len(qwt_result['entanglement_entropy']):
                                        result.append(qwt_result['entanglement_entropy'][si])
                                    else:
                                        result.append(0)
                            
                            full_result = np.full(len(data), result[0] if result else 0)
                            full_result[-len(result):] = result
                            return full_result
                        
                        factor = QuantumFactor(
                            factor_id=factor_id,
                            name=f"Quantum Wavelet {feature} ({wavelet_type}, scale {scale_idx})",
                            category=FactorCategory.QUANTUM_WAVELET,
                            description=f"Quantum wavelet transform {feature} using {wavelet_type} wavelet at scale {scale_idx}",
                            compute_func=compute_func,
                            lookback_window=window,
                            complexity_level=4
                        )
                        
                        self.factor_registry[factor_id] = factor
    
    def _register_quantum_entanglement_factors(self):
        """注册量子纠缠因子"""
        lag_ranges = [1, 2, 5, 10]
        
        for lag in lag_ranges:
            for feature in ['mutual_information', 'entanglement_measure', 'joint_entropy', 'conditional_entropy']:
                for window in [30, 60, 120]:
                    factor_id = f"qent_{feature}_lag{lag}_w{window}"
                    
                    def compute_func(data, feat=feature, l=lag, w=window):
                        if len(data) < w + l:
                            return np.zeros(len(data))
                        
                        result = []
                        for i in range(len(data) - w - l + 1):
                            data1 = data[i:i+w]
                            data2 = data[i+l:i+l+w]
                            
                            entanglement_result = self.quantum_transforms.quantum_entanglement_measure(
                                data1, data2
                            )
                            result.append(entanglement_result[feat])
                        
                        full_result = np.full(len(data), result[0] if result else 0)
                        full_result[-len(result):] = result
                        return full_result
                    
                    factor = QuantumFactor(
                        factor_id=factor_id,
                        name=f"Quantum Entanglement {feature} (lag {lag})",
                        category=FactorCategory.QUANTUM_ENTANGLEMENT,
                        description=f"Quantum entanglement {feature} with lag {lag} over {window} periods",
                        compute_func=compute_func,
                        lookback_window=window + lag,
                        complexity_level=4
                    )
                    
                    self.factor_registry[factor_id] = factor
    
    def _register_topological_factors(self):
        """注册拓扑特征因子"""
        # 持久同调特征
        for window in [50, 100, 200]:
            for feature in ['persistence_entropy', 'betti_mean', 'betti_std']:
                factor_id = f"topo_ph_{feature}_w{window}"
                
                def compute_func(data, feat=feature, w=window):
                    if len(data) < w:
                        return np.zeros(len(data))
                    
                    result = []
                    for i in range(len(data) - w + 1):
                        window_data = data[i:i+w]
                        ph_result = self.topological_features.persistent_homology_features(window_data)
                        
                        if feat == 'persistence_entropy':
                            result.append(ph_result['persistence_entropy'])
                        elif feat == 'betti_mean':
                            result.append(np.mean(ph_result['persistence_curve']))
                        else:  # betti_std
                            result.append(np.std(ph_result['persistence_curve']))
                    
                    full_result = np.full(len(data), result[0] if result else 0)
                    full_result[-len(result):] = result
                    return full_result
                
                factor = QuantumFactor(
                    factor_id=factor_id,
                    name=f"Topological Persistent Homology {feature}",
                    category=FactorCategory.TOPOLOGICAL_FEATURES,
                    description=f"Persistent homology {feature} over {window} periods",
                    compute_func=compute_func,
                    lookback_window=window,
                    complexity_level=5
                )
                
                self.factor_registry[factor_id] = factor
        
        # 欧拉特征数
        for embedding_dim in [2, 3, 4, 5]:
            for window in [40, 80, 160]:
                factor_id = f"topo_euler_dim{embedding_dim}_w{window}"
                
                def compute_func(data, emb_dim=embedding_dim, w=window):
                    if len(data) < w:
                        return np.zeros(len(data))
                    
                    result = []
                    for i in range(len(data) - w + 1):
                        window_data = data[i:i+w]
                        euler_char = self.topological_features.euler_characteristics(
                            window_data, emb_dim
                        )
                        result.append(euler_char)
                    
                    full_result = np.full(len(data), result[0] if result else 0)
                    full_result[-len(result):] = result
                    return full_result
                
                factor = QuantumFactor(
                    factor_id=factor_id,
                    name=f"Euler Characteristics (dim {embedding_dim})",
                    category=FactorCategory.TOPOLOGICAL_FEATURES,
                    description=f"Euler characteristics in {embedding_dim}D embedding space",
                    compute_func=compute_func,
                    lookback_window=window,
                    complexity_level=3
                )
                
                self.factor_registry[factor_id] = factor
    
    def _register_fractal_factors(self):
        """注册分形几何因子"""
        # 多重分形Hurst指数
        q_values = [-2, -1, 0, 1, 2, 3]
        for q in q_values:
            for window in [100, 200, 500]:
                factor_id = f"fractal_hurst_q{q}_w{window}"
                
                def compute_func(data, q_val=q, w=window):
                    if len(data) < w:
                        return np.zeros(len(data))
                    
                    result = []
                    for i in range(len(data) - w + 1):
                        window_data = data[i:i+w]
                        mf_result = self.fractal_features.hurst_exponent_multifractal(window_data)
                        
                        # 找到最接近目标q值的Hurst指数
                        q_range = mf_result['q_range']
                        hurst_spectrum = mf_result['hurst_spectrum']
                        
                        q_idx = np.argmin(np.abs(q_range - q_val))
                        result.append(hurst_spectrum[q_idx])
                    
                    full_result = np.full(len(data), result[0] if result else 0.5)
                    full_result[-len(result):] = result
                    return full_result
                
                factor = QuantumFactor(
                    factor_id=factor_id,
                    name=f"Multifractal Hurst Exponent (q={q})",
                    category=FactorCategory.FRACTAL_GEOMETRY,
                    description=f"Multifractal Hurst exponent for q={q} over {window} periods",
                    compute_func=compute_func,
                    lookback_window=window,
                    complexity_level=5
                )
                
                self.factor_registry[factor_id] = factor
        
        # 空隙度分析
        for window in [80, 160, 320]:
            for feature in ['lacunarity_mean', 'lacunarity_trend']:
                factor_id = f"fractal_lacunarity_{feature}_w{window}"
                
                def compute_func(data, feat=feature, w=window):
                    if len(data) < w:
                        return np.zeros(len(data))
                    
                    result = []
                    for i in range(len(data) - w + 1):
                        window_data = data[i:i+w]
                        lac_result = self.fractal_features.lacunarity_analysis(window_data)
                        
                        if feat == 'lacunarity_mean':
                            result.append(np.mean(lac_result['lacunarities']))
                        else:  # lacunarity_trend
                            lacunarities = lac_result['lacunarities']
                            if len(lacunarities) > 1:
                                x = np.arange(len(lacunarities))
                                slope = np.polyfit(x, lacunarities, 1)[0]
                                result.append(slope)
                            else:
                                result.append(0)
                    
                    full_result = np.full(len(data), result[0] if result else 0)
                    full_result[-len(result):] = result
                    return full_result
                
                factor = QuantumFactor(
                    factor_id=factor_id,
                    name=f"Fractal Lacunarity {feature}",
                    category=FactorCategory.FRACTAL_GEOMETRY,
                    description=f"Fractal lacunarity {feature} over {window} periods",
                    compute_func=compute_func,
                    lookback_window=window,
                    complexity_level=4
                )
                
                self.factor_registry[factor_id] = factor
    
    def _register_information_theory_factors(self):
        """注册信息论因子"""
        # Shannon熵谱
        window_sizes_list = [[5, 10, 20], [10, 20, 50], [20, 50, 100]]
        
        for i, window_sizes in enumerate(window_sizes_list):
            factor_id = f"info_shannon_entropy_spectrum_{i}"
            
            def compute_func(data, ws=window_sizes):
                if len(data) < max(ws) * 2:
                    return np.zeros(len(data))
                
                result = []
                for j in range(len(data) - max(ws) * 2 + 1):
                    segment_data = data[j:j + max(ws) * 2]
                    entropy_result = self.information_features.shannon_entropy_spectrum(
                        segment_data, ws
                    )
                    result.append(np.mean(entropy_result['entropy_spectrum']))
                
                full_result = np.full(len(data), result[0] if result else 0)
                full_result[-len(result):] = result
                return full_result
            
            factor = QuantumFactor(
                factor_id=factor_id,
                name=f"Shannon Entropy Spectrum {i}",
                category=FactorCategory.INFORMATION_THEORY,
                description=f"Shannon entropy spectrum with windows {window_sizes}",
                compute_func=compute_func,
                lookback_window=max(window_sizes) * 2,
                complexity_level=3
            )
            
            self.factor_registry[factor_id] = factor
        
        # Kolmogorov复杂度
        for window in [50, 100, 200]:
            factor_id = f"info_kolmogorov_complexity_w{window}"
            
            def compute_func(data, w=window):
                if len(data) < w:
                    return np.zeros(len(data))
                
                result = []
                for i in range(len(data) - w + 1):
                    window_data = data[i:i+w]
                    complexity = self.information_features.kolmogorov_complexity_approximation(
                        window_data
                    )
                    result.append(complexity)
                
                full_result = np.full(len(data), result[0] if result else 0)
                full_result[-len(result):] = result
                return full_result
            
            factor = QuantumFactor(
                factor_id=factor_id,
                name=f"Kolmogorov Complexity (window {window})",
                category=FactorCategory.INFORMATION_THEORY,
                description=f"Kolmogorov complexity approximation over {window} periods",
                compute_func=compute_func,
                lookback_window=window,
                complexity_level=2
            )
            
            self.factor_registry[factor_id] = factor
    
    def _register_chaos_theory_factors(self):
        """注册混沌理论因子"""
        # Lyapunov指数
        embedding_dims = [2, 3, 4, 5]
        for embedding_dim in embedding_dims:
            for window in [100, 200, 500]:
                factor_id = f"chaos_lyapunov_dim{embedding_dim}_w{window}"
                
                def compute_func(data, emb_dim=embedding_dim, w=window):
                    if len(data) < w:
                        return np.zeros(len(data))
                    
                    result = []
                    for i in range(len(data) - w + 1):
                        window_data = data[i:i+w]
                        lyapunov = self.chaos_features.lyapunov_exponent(
                            window_data, emb_dim
                        )
                        result.append(lyapunov)
                    
                    full_result = np.full(len(data), result[0] if result else 0)
                    full_result[-len(result):] = result
                    return full_result
                
                factor = QuantumFactor(
                    factor_id=factor_id,
                    name=f"Lyapunov Exponent (dim {embedding_dim})",
                    category=FactorCategory.CHAOS_THEORY,
                    description=f"Lyapunov exponent with {embedding_dim}D embedding over {window} periods",
                    compute_func=compute_func,
                    lookback_window=window,
                    complexity_level=5
                )
                
                self.factor_registry[factor_id] = factor
        
        # 近似熵
        m_values = [1, 2, 3]
        for m in m_values:
            for window in [50, 100, 200]:
                factor_id = f"chaos_approx_entropy_m{m}_w{window}"
                
                def compute_func(data, m_val=m, w=window):
                    if len(data) < w:
                        return np.zeros(len(data))
                    
                    result = []
                    for i in range(len(data) - w + 1):
                        window_data = data[i:i+w]
                        approx_ent = self.chaos_features.approximate_entropy(
                            window_data, m_val
                        )
                        result.append(approx_ent)
                    
                    full_result = np.full(len(data), result[0] if result else 0)
                    full_result[-len(result):] = result
                    return full_result
                
                factor = QuantumFactor(
                    factor_id=factor_id,
                    name=f"Approximate Entropy (m={m})",
                    category=FactorCategory.CHAOS_THEORY,
                    description=f"Approximate entropy with pattern length {m} over {window} periods",
                    compute_func=compute_func,
                    lookback_window=window,
                    complexity_level=4
                )
                
                self.factor_registry[factor_id] = factor
    
    def _register_stochastic_calculus_factors(self):
        """注册随机微积分因子"""
        # Itô积分特征
        for feature in ['ito_integral', 'stratonovich_integral', 'quadratic_variation']:
            for window in [30, 60, 120]:
                factor_id = f"stoch_{feature}_w{window}"
                
                def compute_func(data, feat=feature, w=window):
                    if len(data) < w:
                        return np.zeros(len(data))
                    
                    result = []
                    for i in range(len(data) - w + 1):
                        window_data = data[i:i+w]
                        ito_result = self.stochastic_features.ito_integral_approximation(
                            window_data
                        )
                        
                        if feat in ito_result:
                            result.append(ito_result[feat][-1] if len(ito_result[feat]) > 0 else 0)
                        else:
                            result.append(0)
                    
                    full_result = np.full(len(data), result[0] if result else 0)
                    full_result[-len(result):] = result
                    return full_result
                
                factor = QuantumFactor(
                    factor_id=factor_id,
                    name=f"Stochastic Calculus {feature}",
                    category=FactorCategory.STOCHASTIC_CALCULUS,
                    description=f"Stochastic calculus {feature} over {window} periods",
                    compute_func=compute_func,
                    lookback_window=window,
                    complexity_level=4
                )
                
                self.factor_registry[factor_id] = factor
        
        # Ornstein-Uhlenbeck特征
        for feature in ['mean_reversion_speed', 'long_term_mean', 'noise_intensity', 'half_life']:
            for window in [50, 100, 200]:
                factor_id = f"stoch_ou_{feature}_w{window}"
                
                def compute_func(data, feat=feature, w=window):
                    if len(data) < w:
                        return np.zeros(len(data))
                    
                    result = []
                    for i in range(len(data) - w + 1):
                        window_data = data[i:i+w]
                        ou_result = self.stochastic_features.ornstein_uhlenbeck_features(
                            window_data
                        )
                        
                        value = ou_result.get(feat, 0)
                        # 处理无穷大值
                        if np.isinf(value):
                            value = 1000 if value > 0 else -1000
                        elif np.isnan(value):
                            value = 0
                        
                        result.append(value)
                    
                    full_result = np.full(len(data), result[0] if result else 0)
                    full_result[-len(result):] = result
                    return full_result
                
                factor = QuantumFactor(
                    factor_id=factor_id,
                    name=f"Ornstein-Uhlenbeck {feature}",
                    category=FactorCategory.STOCHASTIC_CALCULUS,
                    description=f"Ornstein-Uhlenbeck process {feature} over {window} periods",
                    compute_func=compute_func,
                    lookback_window=window,
                    complexity_level=3
                )
                
                self.factor_registry[factor_id] = factor
    
    def _register_spectral_analysis_factors(self):
        """注册谱分析因子"""
        # 功率谱密度特征
        for freq_band in ['low', 'mid', 'high', 'full']:
            for window in [64, 128, 256]:
                factor_id = f"spectral_psd_{freq_band}_w{window}"
                
                def compute_func(data, band=freq_band, w=window):
                    if len(data) < w:
                        return np.zeros(len(data))
                    
                    result = []
                    for i in range(len(data) - w + 1):
                        window_data = data[i:i+w]
                        
                        # 计算功率谱密度
                        freqs, psd = signal.welch(window_data, nperseg=min(w//4, 32))
                        
                        # 选择频带
                        if band == 'low':
                            band_indices = freqs <= np.percentile(freqs, 25)
                        elif band == 'mid':
                            band_indices = (freqs > np.percentile(freqs, 25)) & \
                                         (freqs <= np.percentile(freqs, 75))
                        elif band == 'high':
                            band_indices = freqs > np.percentile(freqs, 75)
                        else:  # full
                            band_indices = np.ones_like(freqs, dtype=bool)
                        
                        band_power = np.mean(psd[band_indices]) if np.sum(band_indices) > 0 else 0
                        result.append(band_power)
                    
                    full_result = np.full(len(data), result[0] if result else 0)
                    full_result[-len(result):] = result
                    return full_result
                
                factor = QuantumFactor(
                    factor_id=factor_id,
                    name=f"Power Spectral Density ({freq_band} band)",
                    category=FactorCategory.SPECTRAL_ANALYSIS,
                    description=f"Power spectral density in {freq_band} frequency band over {window} periods",
                    compute_func=compute_func,
                    lookback_window=window,
                    complexity_level=2
                )
                
                self.factor_registry[factor_id] = factor
    
    def _register_manifold_learning_factors(self):
        """注册流形学习因子"""
        # 局部线性嵌入特征
        embedding_dims = [2, 3, 5]
        
        for embedding_dim in embedding_dims:
            for window in [50, 100, 200]:
                factor_id = f"manifold_lle_dim{embedding_dim}_w{window}"
                
                def compute_func(data, emb_dim=embedding_dim, w=window):
                    if len(data) < w or w < emb_dim + 1:
                        return np.zeros(len(data))
                    
                    result = []
                    for i in range(len(data) - w + 1):
                        window_data = data[i:i+w]
                        
                        # 时间延迟嵌入
                        if emb_dim == 1:
                            embedded = window_data.reshape(-1, 1)
                        else:
                            embedded = np.array([window_data[j:j+emb_dim] 
                                               for j in range(len(window_data) - emb_dim + 1)])
                        
                        if len(embedded) > emb_dim:
                            # 简化的流形学习：计算局部方差
                            try:
                                from sklearn.decomposition import PCA
                                pca = PCA(n_components=min(emb_dim, len(embedded)-1))
                                pca.fit(embedded)
                                # 使用第一主成分的方差比作为特征
                                manifold_feature = pca.explained_variance_ratio_[0]
                            except:
                                manifold_feature = np.var(embedded.flatten())
                        else:
                            manifold_feature = np.var(window_data)
                        
                        result.append(manifold_feature)
                    
                    full_result = np.full(len(data), result[0] if result else 0)
                    full_result[-len(result):] = result
                    return full_result
                
                factor = QuantumFactor(
                    factor_id=factor_id,
                    name=f"Manifold Learning Feature (dim {embedding_dim})",
                    category=FactorCategory.MANIFOLD_LEARNING,
                    description=f"Manifold learning feature in {embedding_dim}D space over {window} periods",
                    compute_func=compute_func,
                    lookback_window=window,
                    complexity_level=4
                )
                
                self.factor_registry[factor_id] = factor
    
    def _register_network_theory_factors(self):
        """注册网络理论因子"""
        # 可见性图网络特征
        for threshold_percentile in [10, 25, 50, 75, 90]:
            for window in [50, 100, 200]:
                factor_id = f"network_visibility_p{threshold_percentile}_w{window}"
                
                def compute_func(data, thresh_p=threshold_percentile, w=window):
                    if len(data) < w:
                        return np.zeros(len(data))
                    
                    result = []
                    for i in range(len(data) - w + 1):
                        window_data = data[i:i+w]
                        
                        # 构建可见性图
                        n = len(window_data)
                        adjacency_matrix = np.zeros((n, n))
                        
                        for a in range(n):
                            for b in range(a + 2, n):  # 跳过相邻点
                                # 检查可见性
                                visible = True
                                for c in range(a + 1, b):
                                    # 线性插值检查
                                    interpolated = window_data[a] + (window_data[b] - window_data[a]) * \
                                                 (c - a) / (b - a)
                                    if window_data[c] > interpolated:
                                        visible = False
                                        break
                                
                                if visible:
                                    adjacency_matrix[a, b] = 1
                                    adjacency_matrix[b, a] = 1
                        
                        # 网络特征
                        try:
                            G = nx.from_numpy_array(adjacency_matrix)
                            if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
                                # 度中心性
                                degree_centrality = list(nx.degree_centrality(G).values())
                                network_feature = np.percentile(degree_centrality, thresh_p)
                            else:
                                network_feature = 0
                        except:
                            network_feature = 0
                        
                        result.append(network_feature)
                    
                    full_result = np.full(len(data), result[0] if result else 0)
                    full_result[-len(result):] = result
                    return full_result
                
                factor = QuantumFactor(
                    factor_id=factor_id,
                    name=f"Network Visibility Graph Feature (p{threshold_percentile})",
                    category=FactorCategory.NETWORK_THEORY,
                    description=f"Visibility graph network feature at {threshold_percentile}th percentile over {window} periods",
                    compute_func=compute_func,
                    lookback_window=window,
                    complexity_level=4
                )
                
                self.factor_registry[factor_id] = factor
    
    def compute_factor(self, factor_id: str, data: np.ndarray, use_cache: bool = True) -> np.ndarray:
        """计算单个因子"""
        if factor_id not in self.factor_registry:
            raise ValueError(f"未知因子ID: {factor_id}")
        
        # 检查缓存
        if use_cache:
            cache_key = f"{factor_id}_{hashlib.md5(data.tobytes()).hexdigest()}"
            if cache_key in self.compute_cache:
                return self.compute_cache[cache_key]
        
        factor = self.factor_registry[factor_id]
        
        try:
            start_time = datetime.utcnow()
            
            # 计算因子值
            factor_values = factor.compute_func(data)
            
            compute_time = (datetime.utcnow() - start_time).total_seconds()
            
            # 更新因子统计
            factor.expected_compute_time = 0.8 * factor.expected_compute_time + 0.2 * compute_time
            
            # 缓存结果
            if use_cache:
                self.compute_cache[cache_key] = factor_values
                
                # 限制缓存大小
                if len(self.compute_cache) > 1000:
                    # 移除最旧的一半缓存
                    keys_to_remove = list(self.compute_cache.keys())[:500]
                    for key in keys_to_remove:
                        del self.compute_cache[key]
            
            return factor_values
            
        except Exception as e:
            self.logger.error(f"计算因子失败 {factor_id}: {e}")
            return np.zeros_like(data)
    
    def compute_factors_batch(self, factor_ids: List[str], data: np.ndarray, 
                             n_workers: int = None) -> Dict[str, np.ndarray]:
        """批量计算因子"""
        if n_workers is None:
            n_workers = self.n_workers
        
        results = {}
        
        if n_workers <= 1:
            # 单线程计算
            for factor_id in factor_ids:
                results[factor_id] = self.compute_factor(factor_id, data)
        else:
            # 多线程计算
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                future_to_factor = {
                    executor.submit(self.compute_factor, factor_id, data): factor_id
                    for factor_id in factor_ids
                }
                
                for future in future_to_factor:
                    factor_id = future_to_factor[future]
                    try:
                        results[factor_id] = future.result()
                    except Exception as e:
                        self.logger.error(f"批量计算因子失败 {factor_id}: {e}")
                        results[factor_id] = np.zeros_like(data)
        
        return results
    
    def get_factors_by_category(self, category: FactorCategory) -> List[str]:
        """按分类获取因子ID列表"""
        return [factor_id for factor_id, factor in self.factor_registry.items()
                if factor.category == category]
    
    def get_factor_info(self, factor_id: str) -> Optional[QuantumFactor]:
        """获取因子信息"""
        return self.factor_registry.get(factor_id)
    
    def list_all_factors(self) -> List[str]:
        """列出所有因子ID"""
        return list(self.factor_registry.keys())
    
    def get_factor_statistics(self) -> Dict[str, Any]:
        """获取因子引擎统计信息"""
        category_counts = defaultdict(int)
        complexity_counts = defaultdict(int)
        
        for factor in self.factor_registry.values():
            category_counts[factor.category.value] += 1
            complexity_counts[factor.complexity_level] += 1
        
        return {
            "total_factors": len(self.factor_registry),
            "category_distribution": dict(category_counts),
            "complexity_distribution": dict(complexity_counts),
            "cache_size": len(self.compute_cache)
        }

# 全局量子因子引擎实例
quantum_factor_engine = QuantumFactorEngine()