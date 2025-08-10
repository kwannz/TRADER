# CTBench 加密货币时间序列生成基准测试平台 - 详细开发计划

## 第一部分：项目定义与目标

### 1.1 项目背景与动机

#### 问题陈述
当前金融时间序列生成（TSG）领域存在三个关键缺陷：
1. **领域局限性**：现有基准测试（如TSGBench、FinTSB）主要针对传统股票市场，缺乏加密货币数据支持
2. **任务范围狭窄**：过度关注分类和预测任务，忽视了交易实用性评估
3. **评估体系不完整**：缺少加密货币特定的金融评估指标，无法反映真实交易效用

#### 加密货币市场独特挑战
- **24/7连续交易**：无休市时间，需要处理连续数据流
- **极端波动性**：价格波动远超传统资产，日内波动可达10-20%
- **无内在价值锚定**：缺乏基本面信息，纯粹依赖价格和成交量
- **流动性不规则**：不同代币流动性差异巨大，影响交易执行

### 1.2 项目目标

#### 主要目标
构建首个专门针对加密货币市场的时间序列生成基准测试框架，实现：
- 提供高质量的加密货币时间序列数据集（452个代币，5年历史数据）
- 实现8个代表性TSG模型，覆盖5大方法论家族
- 建立双任务评估体系（预测效用 + 统计套利）
- 开发13个评估指标，全面衡量模型表现
- 提供可操作的模型选择指南

#### 成功标准
- 数据质量：缺失率<1%，异常值处理完善
- 模型覆盖：实现论文中所有8个模型，训练收敛率>95%
- 性能基准：单模型训练时间<1小时，推理延迟<1秒
- 可用性：提供完整API文档，支持Docker一键部署

## 第二部分：技术架构详细设计

### 2.1 系统整体架构

```
CTBench Platform Architecture
│
├── 【数据层 Data Layer】
│   ├── Raw Data Storage (PostgreSQL)
│   │   ├── OHLCV数据表（价格、成交量）
│   │   ├── 元数据表（代币信息、市值分类）
│   │   └── 数据质量表（缺失记录、异常标记）
│   │
│   ├── Feature Store (Redis + Parquet)
│   │   ├── Alpha101因子缓存
│   │   ├── 技术指标缓存
│   │   └── 市场微观结构特征
│   │
│   └── Data Pipeline (Apache Airflow)
│       ├── 数据采集任务（每小时执行）
│       ├── 数据清洗任务（实时触发）
│       └── 特征计算任务（批处理）
│
├── 【模型层 Model Layer】
│   ├── Model Registry
│   │   ├── 模型元数据管理
│   │   ├── 版本控制（Git LFS）
│   │   └── 模型权重存储（S3/MinIO）
│   │
│   ├── Training Infrastructure
│   │   ├── 分布式训练（PyTorch DDP）
│   │   ├── 超参数优化（Optuna）
│   │   └── 实验追踪（MLflow）
│   │
│   └── Inference Service
│       ├── 模型服务（TorchServe）
│       ├── 批量推理（Ray）
│       └── 在线推理（FastAPI）
│
├── 【任务层 Task Layer】
│   ├── Task Orchestrator
│   │   ├── 任务调度器
│   │   ├── 资源分配器
│   │   └── 结果聚合器
│   │
│   ├── Predictive Utility Engine
│   │   ├── 预测模型训练
│   │   ├── 交易信号生成
│   │   └── 组合构建器
│   │
│   └── Statistical Arbitrage Engine
│       ├── 残差计算器
│       ├── OU过程拟合
│       └── 套利信号生成
│
├── 【评估层 Evaluation Layer】
│   ├── Metrics Calculator
│   │   ├── 误差指标（MSE, MAE）
│   │   ├── 排序指标（IC, IR）
│   │   ├── 交易指标（CAGR, Sharpe）
│   │   ├── 风险指标（MDD, VaR, ES）
│   │   └── 效率指标（训练/推理时间）
│   │
│   └── Visualization Engine
│       ├── 性能雷达图
│       ├── 权益曲线
│       └── 排名分析
│
└── 【接口层 Interface Layer】
    ├── REST API (FastAPI)
    ├── Python SDK
    ├── Web Dashboard (React + D3.js)
    └── CLI Tools
```

### 2.2 核心组件详细设计

#### 2.2.1 数据采集与处理组件

```python
# 数据采集器详细实现
class CryptoDataCollector:
    """
    加密货币数据采集器
    负责从多个数据源采集OHLCV数据
    """
    
    def __init__(self, config: Dict):
        self.exchanges = {
            'binance': BinanceClient(config['binance']),
            'coinbase': CoinbaseClient(config['coinbase']),  # 扩展支持
            'kraken': KrakenClient(config['kraken'])  # 扩展支持
        }
        self.db_connection = PostgreSQLConnection(config['database'])
        self.redis_cache = RedisCache(config['redis'])
        self.retry_policy = RetryPolicy(max_attempts=3, backoff=2.0)
        
    async def collect_hourly_data(self, 
                                  symbols: List[str], 
                                  start_time: datetime, 
                                  end_time: datetime) -> pd.DataFrame:
        """
        异步采集小时级别数据
        
        参数:
            symbols: 代币符号列表，如['BTC/USDT', 'ETH/USDT']
            start_time: 开始时间
            end_time: 结束时间
            
        返回:
            DataFrame: 包含OHLCV数据的DataFrame
        """
        tasks = []
        for symbol in symbols:
            task = self._fetch_symbol_data(symbol, start_time, end_time)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return self._merge_results(results)
    
    async def _fetch_symbol_data(self, symbol: str, start: datetime, end: datetime):
        """
        获取单个交易对的数据，带重试机制
        """
        for attempt in range(self.retry_policy.max_attempts):
            try:
                # 首先检查缓存
                cached_data = await self.redis_cache.get(f"ohlcv:{symbol}:{start}:{end}")
                if cached_data:
                    return cached_data
                
                # 从交易所API获取
                data = await self.exchanges['binance'].fetch_ohlcv(
                    symbol=symbol,
                    timeframe='1h',
                    since=int(start.timestamp() * 1000),
                    limit=int((end - start).total_seconds() / 3600)
                )
                
                # 数据验证
                validated_data = self._validate_data(data)
                
                # 存入缓存
                await self.redis_cache.set(
                    f"ohlcv:{symbol}:{start}:{end}",
                    validated_data,
                    expire=3600
                )
                
                return validated_data
                
            except Exception as e:
                if attempt == self.retry_policy.max_attempts - 1:
                    raise DataCollectionError(f"Failed to fetch {symbol}: {str(e)}")
                await asyncio.sleep(self.retry_policy.backoff ** attempt)
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据验证和清洗
        - 检查缺失值
        - 检测异常值（价格跳变>50%）
        - 验证时间连续性
        """
        # 缺失值处理
        if data.isnull().any().any():
            data = self._handle_missing_values(data)
        
        # 异常值检测
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            pct_change = data[col].pct_change()
            outliers = abs(pct_change) > 0.5  # 50%跳变视为异常
            if outliers.any():
                data.loc[outliers, col] = np.nan
                data[col] = data[col].interpolate(method='linear')
        
        # 时间连续性检查
        expected_index = pd.date_range(
            start=data.index[0],
            end=data.index[-1],
            freq='H'
        )
        data = data.reindex(expected_index, method='ffill')
        
        return data
```

#### 2.2.2 特征工程模块

```python
class FeatureEngineer:
    """
    特征工程模块
    计算Alpha101因子和技术指标
    """
    
    def __init__(self):
        self.alpha101_calculator = Alpha101Calculator()
        self.technical_indicators = TechnicalIndicators()
        self.microstructure_features = MicrostructureFeatures()
        
    def compute_features(self, 
                        price_data: pd.DataFrame,
                        volume_data: pd.DataFrame,
                        window_sizes: List[int] = [20, 50, 100]) -> pd.DataFrame:
        """
        计算完整特征集
        
        参数:
            price_data: 价格数据 (n_assets × n_times)
            volume_data: 成交量数据
            window_sizes: 滚动窗口大小列表
            
        返回:
            DataFrame: (n_assets × n_times × n_features)
        """
        features = {}
        
        # 1. 计算收益率相关特征
        returns = self._calculate_returns(price_data)
        features['returns'] = returns
        features['log_returns'] = np.log(price_data).diff()
        
        # 2. Alpha101因子（选择关键的30个）
        alpha_factors = self.alpha101_calculator.compute_factors(
            open_prices=price_data['open'],
            high_prices=price_data['high'],
            low_prices=price_data['low'],
            close_prices=price_data['close'],
            volumes=volume_data,
            returns=returns
        )
        features.update(alpha_factors)
        
        # 3. 技术指标
        for window in window_sizes:
            # 布林带
            bb_upper, bb_middle, bb_lower = self.technical_indicators.bollinger_bands(
                price_data['close'], window=window
            )
            features[f'bb_ratio_{window}'] = (price_data['close'] - bb_middle) / (bb_upper - bb_lower)
            
            # RSI
            features[f'rsi_{window}'] = self.technical_indicators.rsi(
                price_data['close'], window=window
            )
            
            # 移动平均
            features[f'sma_{window}'] = price_data['close'].rolling(window).mean()
            features[f'ema_{window}'] = price_data['close'].ewm(span=window).mean()
            
        # 4. 市场微观结构特征
        features['volatility'] = returns.rolling(20).std() * np.sqrt(24 * 365)  # 年化波动率
        features['skewness'] = returns.rolling(50).skew()
        features['kurtosis'] = returns.rolling(50).kurt()
        features['volume_ratio'] = volume_data / volume_data.rolling(20).mean()
        
        # 5. 横截面特征
        features['cross_sectional_rank'] = price_data['close'].rank(axis=0, pct=True)
        features['relative_strength'] = price_data['close'] / price_data['close'].mean(axis=0)
        
        return pd.concat(features, axis=1)
```

### 2.3 模型实现详细规范

#### 2.3.1 基础模型接口定义

```python
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn

class BaseTSGModel(ABC):
    """
    时间序列生成模型基类
    所有TSG模型必须继承此类并实现相应方法
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模型
        
        参数:
            config: 模型配置字典，包含:
                - n_assets: 资产数量
                - seq_length: 序列长度
                - feature_dim: 特征维度
                - latent_dim: 潜在空间维度
                - device: 计算设备
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.loss_history = []
        
    @abstractmethod
    def build_model(self) -> nn.Module:
        """构建模型架构"""
        pass
    
    @abstractmethod
    def train_step(self, 
                   real_data: torch.Tensor, 
                   epoch: int) -> Dict[str, float]:
        """
        单步训练
        
        参数:
            real_data: 真实数据 (batch_size, n_assets, seq_length, feature_dim)
            epoch: 当前训练轮次
            
        返回:
            Dict: 包含各项损失的字典
        """
        pass
    
    def train(self, 
             train_loader: torch.utils.data.DataLoader,
             n_epochs: int,
             callbacks: Optional[List] = None) -> Dict[str, List]:
        """
        完整训练流程
        
        参数:
            train_loader: 训练数据加载器
            n_epochs: 训练轮数
            callbacks: 回调函数列表（早停、模型保存等）
            
        返回:
            Dict: 训练历史记录
        """
        self.model = self.build_model().to(self.device)
        
        for epoch in range(n_epochs):
            epoch_losses = []
            
            for batch_idx, real_data in enumerate(train_loader):
                real_data = real_data.to(self.device)
                losses = self.train_step(real_data, epoch)
                epoch_losses.append(losses)
                
                # 执行回调
                if callbacks:
                    for callback in callbacks:
                        callback.on_batch_end(batch_idx, losses)
            
            # 记录epoch级别的损失
            avg_losses = self._average_losses(epoch_losses)
            self.loss_history.append(avg_losses)
            
            # 执行epoch回调
            if callbacks:
                for callback in callbacks:
                    callback.on_epoch_end(epoch, avg_losses)
            
            # 日志输出
            if epoch % 10 == 0:
                self._print_progress(epoch, n_epochs, avg_losses)
        
        return {'loss_history': self.loss_history}
    
    @abstractmethod
    def generate(self, 
                n_samples: int,
                noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        生成合成数据
        
        参数:
            n_samples: 生成样本数量
            noise: 可选的噪声输入
            
        返回:
            Tensor: 生成的数据 (n_samples, n_assets, seq_length, feature_dim)
        """
        pass
    
    @abstractmethod
    def reconstruct(self, 
                   data: torch.Tensor) -> torch.Tensor:
        """
        重建输入数据
        
        参数:
            data: 输入数据
            
        返回:
            Tensor: 重建的数据
        """
        pass
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'loss_history': self.loss_history
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model = self.build_model().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        self.loss_history = checkpoint['loss_history']
```

#### 2.3.2 具体模型实现示例 - TimeVAE

```python
class TimeVAE(BaseTSGModel):
    """
    TimeVAE实现
    时序感知的变分自编码器
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.beta = config.get('beta', 1.0)  # KL散度权重
        
    def build_model(self) -> nn.Module:
        """构建TimeVAE模型架构"""
        return TimeVAENetwork(
            input_dim=self.config['feature_dim'],
            hidden_dim=self.config['hidden_dim'],
            latent_dim=self.config['latent_dim'],
            seq_length=self.config['seq_length'],
            n_assets=self.config['n_assets']
        )
    
    def train_step(self, real_data: torch.Tensor, epoch: int) -> Dict[str, float]:
        """VAE训练步骤"""
        self.optimizer.zero_grad()
        
        # 前向传播
        reconstructed, mu, log_var = self.model(real_data)
        
        # 计算损失
        reconstruction_loss = nn.MSELoss()(reconstructed, real_data)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss / real_data.size(0)  # 平均到batch
        
        # 总损失
        total_loss = reconstruction_loss + self.beta * kl_loss
        
        # 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'kl_loss': kl_loss.item()
        }
    
    def generate(self, n_samples: int, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """从潜在空间生成数据"""
        self.model.eval()
        with torch.no_grad():
            if noise is None:
                noise = torch.randn(
                    n_samples, 
                    self.config['latent_dim']
                ).to(self.device)
            
            generated = self.model.decode(noise)
            return generated
    
    def reconstruct(self, data: torch.Tensor) -> torch.Tensor:
        """重建输入数据"""
        self.model.eval()
        with torch.no_grad():
            reconstructed, _, _ = self.model(data)
            return reconstructed


class TimeVAENetwork(nn.Module):
    """TimeVAE网络架构"""
    
    def __init__(self, input_dim, hidden_dim, latent_dim, seq_length, n_assets):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim * n_assets, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 潜在空间映射
        self.fc_mu = nn.Linear(hidden_dim * 4 * seq_length, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim * 4 * seq_length, latent_dim)
        
        # 解码器
        self.fc_decode = nn.Linear(latent_dim, hidden_dim * 4 * seq_length)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim * 4, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, input_dim * n_assets, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
    def encode(self, x):
        """编码到潜在空间"""
        batch_size = x.size(0)
        x = x.view(batch_size, -1, x.size(-2))  # (batch, channels, seq_length)
        h = self.encoder(x)
        h = h.view(batch_size, -1)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        """重参数化技巧"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """从潜在空间解码"""
        h = self.fc_decode(z)
        h = h.view(z.size(0), -1, self.seq_length)
        x_hat = self.decoder(h)
        return x_hat.view(z.size(0), self.n_assets, self.seq_length, -1)
    
    def forward(self, x):
        """前向传播"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var
```

### 2.4 任务执行引擎

#### 2.4.1 预测效用任务实现

```python
class PredictiveUtilityTask:
    """
    预测效用任务
    评估生成数据的预测能力和交易价值
    """
    
    def __init__(self, 
                 tsg_model: BaseTSGModel,
                 forecaster_type: str = 'xgboost',
                 strategy: str = 'csm'):
        """
        初始化任务
        
        参数:
            tsg_model: 时间序列生成模型
            forecaster_type: 预测器类型 ('xgboost', 'lightgbm', 'catboost')
            strategy: 交易策略 ('csm', 'lotq', 'pw')
        """
        self.tsg_model = tsg_model
        self.forecaster = self._init_forecaster(forecaster_type)
        self.strategy = self._init_strategy(strategy)
        self.feature_engineer = FeatureEngineer()
        self.performance_tracker = PerformanceTracker()
        
    def _init_forecaster(self, forecaster_type: str):
        """初始化预测器"""
        if forecaster_type == 'xgboost':
            return XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        elif forecaster_type == 'lightgbm':
            return LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.01,
                subsample=0.8,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown forecaster type: {forecaster_type}")
    
    def run_experiment(self, 
                       train_data: np.ndarray,
                       test_data: np.ndarray,
                       n_synthetic_samples: int = 1000) -> Dict[str, Any]:
        """
        运行完整实验
        
        参数:
            train_data: 训练数据 (n_assets, train_length, n_features)
            test_data: 测试数据 (n_assets, test_length, n_features)
            n_synthetic_samples: 生成样本数量
            
        返回:
            Dict: 实验结果，包含所有评估指标
        """
        results = {}
        
        # ========== 训练阶段 ==========
        print("Phase 1: Training TSG Model...")
        
        # 1. 训练TSG模型
        self.tsg_model.train(train_data)
        
        # 2. 生成合成数据
        print("Phase 2: Generating Synthetic Data...")
        synthetic_data = self.tsg_model.generate(n_samples=n_synthetic_samples)
        
        # 3. 提取特征
        print("Phase 3: Feature Engineering...")
        synthetic_features = self.feature_engineer.compute_features(
            price_data=synthetic_data[:, :, :, 0],  # Close prices
            volume_data=synthetic_data[:, :, :, 1]   # Volumes
        )
        
        # 4. 准备训练数据
        X_train, y_train = self._prepare_forecasting_data(
            synthetic_features,
            horizon=1  # 预测下一小时
        )
        
        # 5. 训练预测模型
        print("Phase 4: Training Forecaster...")
        self.forecaster.fit(X_train, y_train)
        
        # ========== 交易阶段 ==========
        print("Phase 5: Trading Simulation...")
        
        # 1. 在测试数据上提取特征
        test_features = self.feature_engineer.compute_features(
            price_data=test_data[:, :, :, 0],
            volume_data=test_data[:, :, :, 1]
        )
        
        # 2. 准备测试数据
        X_test, y_test = self._prepare_forecasting_data(
            test_features,
            horizon=1
        )
        
        # 3. 生成预测
        predictions = self.forecaster.predict(X_test)
        
        # 4. 构建投资组合
        portfolio_weights = self.strategy.compute_weights(
            predictions=predictions.reshape(-1, test_data.shape[0]),  # (time, n_assets)
            current_prices=test_data[:, :, 0, 0]  # Close prices
        )
        
        # 5. 回测交易
        backtest_results = self._run_backtest(
            weights=portfolio_weights,
            returns=self._calculate_returns(test_data[:, :, 0, 0]),
            transaction_cost=0.0003  # 0.03%手续费
        )
        
        # ========== 计算评估指标 ==========
        print("Phase 6: Computing Metrics...")
        
        # 预测误差指标
        results['mse'] = mean_squared_error(y_test, predictions)
        results['mae'] = mean_absolute_error(y_test, predictions)
        
        # 排序指标
        results['ic'] = self._compute_ic(y_test, predictions)
        results['ir'] = self._compute_ir(y_test, predictions)
        
        # 交易指标
        results['cagr'] = backtest_results['cagr']
        results['sharpe'] = backtest_results['sharpe_ratio']
        results['max_drawdown'] = backtest_results['max_drawdown']
        results['var_95'] = backtest_results['var_95']
        results['es_95'] = backtest_results['es_95']
        
        # 保存详细结果
        results['equity_curve'] = backtest_results['equity_curve']
        results['daily_returns'] = backtest_results['daily_returns']
        results['portfolio_weights'] = portfolio_weights
        
        return results
    
    def _prepare_forecasting_data(self, 
                                  features: np.ndarray,
                                  horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备预测数据
        
        参数:
            features: 特征数据 (n_assets, n_times, n_features)
            horizon: 预测时间跨度
            
        返回:
            X: 特征矩阵 (n_samples, n_features)
            y: 目标向量 (n_samples,)
        """
        n_assets, n_times, n_features = features.shape
        
        X = []
        y = []
        
        for t in range(n_times - horizon):
            # 当前时刻的特征
            X.append(features[:, t, :].flatten())
            
            # 未来收益率作为目标
            future_returns = features[:, t + horizon, 0]  # 假设第一个特征是收益率
            y.append(future_returns)
        
        return np.array(X), np.array(y).flatten()
    
    def _run_backtest(self,
                      weights: np.ndarray,
                      returns: np.ndarray,
                      transaction_cost: float = 0.0) -> Dict[str, float]:
        """
        运行回测
        
        参数:
            weights: 投资组合权重 (n_times, n_assets)
            returns: 资产收益率 (n_times, n_assets)
            transaction_cost: 交易成本
            
        返回:
            Dict: 回测结果
        """
        n_times, n_assets = weights.shape
        
        # 初始化
        equity = [10000.0]  # 初始资金
        daily_returns = []
        
        for t in range(1, n_times):
            # 计算投资组合收益
            portfolio_return = np.sum(weights[t] * returns[t])
            
            # 计算换手率
            turnover = np.sum(np.abs(weights[t] - weights[t-1]))
            
            # 扣除交易成本
            net_return = portfolio_return - transaction_cost * turnover
            
            # 更新权益
            equity.append(equity[-1] * (1 + net_return))
            daily_returns.append(net_return)
        
        # 计算性能指标
        equity = np.array(equity)
        daily_returns = np.array(daily_returns)
        
        # CAGR
        total_return = (equity[-1] / equity[0]) - 1
        n_years = n_times / (24 * 365)  # 小时数转年数
        cagr = (1 + total_return) ** (1 / n_years) - 1
        
        # Sharpe Ratio
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(24 * 365)
        
        # Maximum Drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown)
        
        # VaR和ES
        var_95 = np.percentile(daily_returns, 5)
        es_95 = np.mean(daily_returns[daily_returns <= var_95])
        
        return {
            'equity_curve': equity,
            'daily_returns': daily_returns,
            'cagr': cagr,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'es_95': es_95
        }
```

#### 2.4.2 统计套利任务实现

```python
class StatisticalArbitrageTask:
    """
    统计套利任务
    评估重建残差的可交易性
    """
    
    def __init__(self, tsg_model: BaseTSGModel):
        self.tsg_model = tsg_model
        self.ou_processes = {}  # 存储每个资产的OU过程参数
        
    def run_experiment(self,
                       train_data: np.ndarray,
                       test_data: np.ndarray,
                       transaction_cost: float = 0.0003) -> Dict[str, Any]:
        """
        运行统计套利实验
        
        参数:
            train_data: 训练数据
            test_data: 测试数据
            transaction_cost: 交易成本
            
        返回:
            Dict: 实验结果
        """
        results = {}
        
        # ========== 训练阶段 ==========
        print("Phase 1: Training and Reconstruction...")
        
        # 1. 训练TSG模型
        self.tsg_model.train(train_data)
        
        # 2. 重建训练数据
        train_reconstructed = self.tsg_model.reconstruct(train_data)
        
        # 3. 计算训练残差
        train_residuals = train_data - train_reconstructed
        
        # 4. 拟合OU过程
        print("Phase 2: Fitting OU Processes...")
        self._fit_ou_processes(train_residuals)
        
        # ========== 交易阶段 ==========
        print("Phase 3: Trading Signal Generation...")
        
        # 1. 重建测试数据
        test_reconstructed = self.tsg_model.reconstruct(test_data)
        
        # 2. 计算测试残差
        test_residuals = test_data - test_reconstructed
        
        # 3. 生成交易信号
        trading_signals = self._generate_trading_signals(test_residuals)
        
        # 4. 计算投资组合权重
        portfolio_weights = self._compute_portfolio_weights(trading_signals)
        
        # 5. 回测
        print("Phase 4: Backtesting...")
        test_returns = self._calculate_returns(test_data[:, :, 0])
        backtest_results = self._run_backtest(
            weights=portfolio_weights,
            returns=test_returns,
            transaction_cost=transaction_cost
        )
        
        # ========== 评估指标 ==========
        results.update(backtest_results)
        
        # 添加套利特定指标
        results['signal_quality'] = self._assess_signal_quality(trading_signals)
        results['mean_reversion_speed'] = np.mean(list(self.ou_processes.values()))
        
        return results
    
    def _fit_ou_processes(self, residuals: np.ndarray):
        """
        拟合Ornstein-Uhlenbeck过程
        dX_t = θ(μ - X_t)dt + σdW_t
        
        参数:
            residuals: 残差数据 (n_assets, n_times)
        """
        n_assets, n_times = residuals.shape[:2]
        
        for i in range(n_assets):
            asset_residuals = residuals[i, :, 0]  # 使用close价格的残差
            
            # 使用最大似然估计拟合OU参数
            dt = 1 / 24  # 小时数据，一天24小时
            
            # 计算参数
            X = asset_residuals[:-1]
            Y = asset_residuals[1:]
            
            # 线性回归: Y = a + b*X
            # 其中 a = θ*μ*dt, b = 1 - θ*dt
            b, a = np.polyfit(X, Y, 1)
            
            theta = -(np.log(b) / dt)
            mu = a / (theta * dt)
            
            # 估计波动率
            residual_var = np.var(Y - (a + b * X))
            sigma = np.sqrt(2 * theta * residual_var / (1 - np.exp(-2 * theta * dt)))
            
            self.ou_processes[i] = {
                'theta': theta,
                'mu': mu,
                'sigma': sigma
            }
    
    def _generate_trading_signals(self, residuals: np.ndarray) -> np.ndarray:
        """
        生成交易信号
        
        参数:
            residuals: 残差数据
            
        返回:
            signals: 交易信号 (-1: 做空, 0: 空仓, 1: 做多)
        """
        n_assets, n_times = residuals.shape[:2]
        signals = np.zeros((n_times, n_assets))
        
        for i in range(n_assets):
            params = self.ou_processes[i]
            asset_residuals = residuals[i, :, 0]
            
            # 计算z-score
            z_scores = (asset_residuals - params['mu']) / (params['sigma'] / np.sqrt(2 * params['theta']))
            
            # 生成信号
            # z-score > 2: 做空（预期均值回归）
            # z-score < -2: 做多（预期均值回归）
            # |z-score| < 0.5: 平仓
            
            for t in range(n_times):
                if z_scores[t] > 2:
                    signals[t, i] = -1
                elif z_scores[t] < -2:
                    signals[t, i] = 1
                elif abs(z_scores[t]) < 0.5:
                    signals[t, i] = 0
                elif t > 0:
                    signals[t, i] = signals[t-1, i]  # 保持前一时刻的仓位
        
        return signals
```

## 第三部分：实施计划

### 3.1 项目时间线（16周详细计划）

#### 第1-2周：环境搭建与数据基础设施
**目标**: 建立完整的开发环境和数据管道

**任务清单**:
- [ ] 配置开发环境（Python 3.9, CUDA 11.8, Docker）
- [ ] 搭建PostgreSQL数据库，设计数据表结构
- [ ] 配置Redis缓存系统
- [ ] 实现Binance API数据采集器
- [ ] 开发数据清洗和验证模块
- [ ] 建立数据质量监控系统
- [ ] 实现异常检测和处理机制
- [ ] 创建数据备份和恢复策略

**交付物**:
- 可运行的数据采集脚本
- 包含452个代币5年历史数据的数据库
- 数据质量报告

#### 第3-4周：特征工程与基础框架
**目标**: 完成特征计算系统和模型基础架构

**任务清单**:
- [ ] 实现Alpha101因子计算器（30个核心因子）
- [ ] 开发技术指标库（布林带、RSI、MA等）
- [ ] 实现市场微观结构特征提取
- [ ] 设计BaseTSGModel抽象类
- [ ] 实现模型训练框架
- [ ] 开发模型保存/加载机制
- [ ] 创建实验追踪系统（MLflow集成）
- [ ] 编写单元测试

**交付物**:
- 完整的特征工程模块
- 基础模型框架
- 测试覆盖率>80%

#### 第5-6周：VAE模型实现
**目标**: 完成VAE系列模型的实现和优化

**任务清单**:
- [ ] 实现TimeVAE编码器/解码器架构
- [ ] 开发时序卷积层
- [ ] 实现重参数化技巧
- [ ] 开发KoVAE的Koopman算子模块
- [ ] 实现潜在动态建模
- [ ] 优化训练稳定性（梯度裁剪、学习率调度）
- [ ] 进行超参数调优
- [ ] 验证生成质量

**交付物**:
- TimeVAE完整实现
- KoVAE完整实现
- 模型训练日志和收敛曲线

#### 第7-8周：GAN模型实现
**目标**: 完成GAN系列模型的实现

**任务清单**:
- [ ] 实现Quant-GAN生成器和判别器
- [ ] 开发交易效用函数优化
- [ ] 实现COSCI-GAN的因果自注意力机制
- [ ] 开发协调生成机制
- [ ] 实施WGAN-GP训练策略
- [ ] 解决模式崩塌问题
- [ ] 实现早停和模型选择策略
- [ ] 性能基准测试

**交付物**:
- Quant-GAN完整实现
- COSCI-GAN完整实现
- 训练稳定性分析报告

#### 第9-10周：高级模型实现
**目标**: 完成扩散、流和混合型模型

**任务清单**:
- [ ] 实现Diffusion-TS的前向/反向过程
- [ ] 开发噪声调度器
- [ ] 实现FIDE的因子化条件机制
- [ ] 开发注意力驱动的分数网络
- [ ] 实现Fourier-Flow的频域耦合层
- [ ] 开发可逆变换
- [ ] 实现LS4的状态空间模型
- [ ] 集成所有模型到统一框架

**交付物**:
- 4个高级模型的完整实现
- 模型对比分析报告

#### 第11-12周：任务系统实现
**目标**: 完成双任务评估系统

**任务清单**:
- [ ] 实现预测效用任务完整流程
- [ ] 集成XGBoost预测器
- [ ] 实现三种交易策略（CSM、LOTQ、PW）
- [ ] 开发统计套利任务
- [ ] 实现OU过程拟合
- [ ] 开发交易信号生成器
- [ ] 实现投资组合优化
- [ ] 开发回测引擎

**交付物**:
- 完整的任务执行系统
- 策略回测报告

#### 第13周：评估体系与系统集成
**目标**: 完成评估指标计算和系统集成

**任务清单**:
- [ ] 实现13个评估指标
- [ ] 开发可视化模块（雷达图、权益曲线）
- [ ] 系统集成测试
- [ ] 端到端流程验证
- [ ] 性能优化（并行计算、缓存优化）
- [ ] Bug修复
- [ ] 压力测试
- [ ] 生成基准测试报告

**交付物**:
- 完整的评估系统
- 系统集成测试报告

#### 第14-15周：API开发与界面
**目标**: 开发用户接口和可视化界面

**任务清单**:
- [ ] 开发RESTful API（FastAPI）
- [ ] 实现认证和授权机制
- [ ] 开发Python SDK
- [ ] 创建React前端框架
- [ ] 实现D3.js可视化组件
- [ ] 开发实时数据流处理（WebSocket）
- [ ] 创建用户Dashboard
- [ ] 实现实验管理界面

**交付物**:
- 完整的API文档
- Web界面原型

#### 第16周：部署与文档
**目标**: 完成生产部署和文档编写

**任务清单**:
- [ ] Docker镜像构建
- [ ] Kubernetes部署配置
- [ ] CI/CD管道设置
- [ ] 监控系统配置（Prometheus + Grafana）
- [ ] 编写API文档（OpenAPI规范）
- [ ] 编写用户指南
- [ ] 创建教程和示例
- [ ] 准备发布材料

**交付物**:
- 生产就绪的部署包
- 完整的文档集

### 3.2 资源需求

#### 硬件需求
- **开发服务器**: 
  - CPU: Intel Xeon 8480C或同等级别
  - GPU: NVIDIA H100 80GB × 2
  - 内存: 256GB RAM
  - 存储: 4TB NVMe SSD

- **数据库服务器**:
  - CPU: 16核心
  - 内存: 64GB RAM
  - 存储: 2TB SSD

- **生产集群**:
  - Kubernetes节点: 4个（每个8核16GB）
  - 负载均衡器: 1个

#### 软件需求
- Python 3.9+及相关库
- PostgreSQL 14+
- Redis 6+
- Docker 20+
- Kubernetes 1.25+
- CUDA 11.8+

#### 团队配置（6人团队）
1. **技术负责人** (1人)
   - 整体架构设计
   - 技术决策
   - 代码审查

2. **算法工程师** (2人)
   - 模型实现
   - 算法优化
   - 实验设计

3. **数据工程师** (1人)
   - 数据管道
   - 特征工程
   - 数据质量保证

4. **后端开发** (1人)
   - API开发
   - 系统集成
   - 性能优化

5. **全栈开发** (1人)
   - 前端界面
   - 可视化组件
   - 用户体验

### 3.3 风险管理计划

#### 技术风险及缓解措施

| 风险类型 | 风险描述 | 影响等级 | 缓解措施 |
|---------|---------|---------|---------|
| 数据质量 | 历史数据缺失或异常 | 高 | 多源验证、插值算法、异常检测 |
| 模型收敛 | GAN训练不稳定 | 高 | WGAN-GP、谱归一化、早停策略 |
| 性能瓶颈 | 大规模数据处理慢 | 中 | 分布式处理、GPU加速、缓存优化 |
| 过拟合 | 模型泛化能力差 | 高 | 交叉验证、正则化、Dropout |
| 系统复杂度 | 集成困难 | 中 | 模块化设计、接口标准化、持续集成 |

#### 业务风险及缓解措施

| 风险类型 | 风险描述 | 影响等级 | 缓解措施 |
|---------|---------|---------|---------|
| 市场变化 | 加密市场结构突变 | 高 | 滚动窗口训练、在线学习 |
| 交易成本 | 实际费用高于预期 | 中 | 多级费用测试、滑点建模 |
| 监管风险 | 加密货币政策变化 | 低 | 合规性审查、灵活架构 |
| 竞争风险 | 类似项目出现 | 中 | 快速迭代、差异化功能 |

### 3.4 质量保证计划

#### 代码质量标准
- **代码规范**: PEP 8 (Python), ESLint (JavaScript)
- **测试覆盖率**: >85%
- **代码审查**: 所有PR需要至少1人审查
- **文档标准**: 所有公共API必须有docstring

#### 测试策略
1. **单元测试**: 所有核心函数
2. **集成测试**: 模块间接口
3. **系统测试**: 端到端流程
4. **性能测试**: 负载和压力测试
5. **回归测试**: 每次发布前

#### 持续集成/部署
- **CI工具**: GitHub Actions
- **测试自动化**: pytest + coverage
- **部署自动化**: ArgoCD
- **监控**: Prometheus + Grafana

## 第四部分：预期成果与影响

### 4.1 技术成果
- 8个高质量TSG模型实现
- 完整的加密货币时间序列数据集
- 可扩展的评估框架
- 生产级API和SDK

### 4.2 学术贡献
- 首个加密货币TSG基准
- 双任务评估方法论
- 模型选择最佳实践指南
- 开源代码和数据

### 4.3 商业价值
- 支持量化交易策略开发
- 风险管理工具
- 市场模拟平台
- 数据增强服务

### 4.4 社区影响
- 推动TSG研究在金融领域应用
- 建立行业标准
- 培育开源生态
- 知识共享平台

## 附录A：技术栈详细说明

### 核心依赖
```yaml
# requirements.yaml
python: ">=3.9,<3.11"
pytorch: ">=2.0"
numpy: ">=1.21"
pandas: ">=1.5"
scikit-learn: ">=1.0"
xgboost: ">=1.7"
lightgbm: ">=3.3"
optuna: ">=3.0"
mlflow: ">=2.0"
fastapi: ">=0.95"
redis-py: ">=4.5"
psycopg2: ">=2.9"
plotly: ">=5.13"
dash: ">=2.8"
```

### 开发工具
```yaml
# dev-requirements.yaml
pytest: ">=7.2"
pytest-cov: ">=4.0"
black: ">=23.0"
flake8: ">=6.0"
mypy: ">=1.0"
pre-commit: ">=3.0"
jupyter: ">=1.0"
tensorboard: ">=2.11"
```

## 附录B：数据库架构

```sql
-- 主要数据表结构
CREATE TABLE crypto_ohlcv (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DECIMAL(20, 8),
    high DECIMAL(20, 8),
    low DECIMAL(20, 8),
    close DECIMAL(20, 8),
    volume DECIMAL(20, 8),
    UNIQUE(symbol, timestamp)
);

CREATE TABLE features (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    feature_name VARCHAR(50) NOT NULL,
    feature_value DECIMAL(20, 8),
    FOREIGN KEY (symbol, timestamp) 
        REFERENCES crypto_ohlcv(symbol, timestamp)
);

CREATE TABLE model_experiments (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50),
    model_version VARCHAR(20),
    train_start TIMESTAMP,
    train_end TIMESTAMP,
    test_start TIMESTAMP,
    test_end TIMESTAMP,
    hyperparameters JSONB,
    metrics JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 索引优化
CREATE INDEX idx_ohlcv_symbol_time ON crypto_ohlcv(symbol, timestamp);
CREATE INDEX idx_features_lookup ON features(symbol, timestamp, feature_name);
CREATE INDEX idx_experiments_model ON model_experiments(model_name, created_at);
```

## 附录C：API规范示例

```yaml
openapi: 3.0.0
info:
  title: CTBench API
  version: 1.0.0
  description: Cryptocurrency Time Series Generation Benchmark API

paths:
  /models:
    get:
      summary: List available TSG models
      responses:
        200:
          description: List of models
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: '#/components/schemas/Model'
  
  /models/{model_id}/generate:
    post:
      summary: Generate synthetic time series
      parameters:
        - name: model_id
          in: path
          required: true
          schema:
            type: string
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                n_samples:
                  type: integer
                  default: 100
                seq_length:
                  type: integer
                  default: 500
      responses:
        200:
          description: Generated data
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                  metadata:
                    type: object

components:
  schemas:
    Model:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
        type:
          type: string
          enum: [gan, vae, diffusion, flow, mixed]
        parameters:
          type: object
```

---

这份详细的开发计划现在包含了：
1. 清晰的项目背景和目标定义
2. 完整的技术架构和组件设计
3. 详细的代码实现示例
4. 16周的具体实施计划
5. 资源需求和团队配置
6. 风险管理和质量保证计划
7. 预期成果和影响分析
8. 技术栈和数据库设计
9. API规范示例

每个部分都经过仔细设计，确保计划的可执行性和完整性。