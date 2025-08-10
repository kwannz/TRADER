# AI量化交易系统全栈架构文档 (CTBench集成版)

## 1. 项目概述

### 1.1 系统定位
- **项目名称**：Advanced AI Quant Trading Platform with CTBench Integration
- **核心定位**：集成时间序列生成的下一代AI量化交易平台
- **技术栈**：Rich/Textual CLI + FastAPI + Rust Engine + CTBench TSG + MongoDB/Redis/PostgreSQL
- **架构模式**：CLI前端 + WebSocket实时流 + Rust高性能引擎 + AI时间序列生成平台
- **部署方案**：Docker容器化部署 + GPU加速计算集群 + Poetry依赖管理

### 1.2 系统特色
- **双引擎驱动**：Rust性能引擎 + CTBench AI生成引擎
- **三重AI集成**：DeepSeek(情绪分析) + Gemini(策略生成) + CTBench(数据生成)
- **银行级风控**：黑天鹅检测 + 场景生成 + 压力测试
- **专业级界面**：Bloomberg风格CLI + 实时4Hz刷新

## 2. 增强项目结构

```
ai-quant-trader-enhanced/
├── frontend/                     # Rich/Textual CLI前端
│   ├── src/
│   │   ├── components/          # CLI组件库
│   │   │   ├── dashboard.py     # 主仪表盘
│   │   │   ├── strategy_manager.py # 策略管理
│   │   │   ├── risk_monitor.py  # 风控监控
│   │   │   ├── ai_assistant.py  # AI助手
│   │   │   ├── factor_lab.py    # 因子发现实验室
│   │   │   └── data_lab.py      # 【新增】数据实验室
│   │   │   └── model_trainer.py # 【新增】模型训练界面
│   │   ├── services/            # 服务层
│   │   └── main.py              # CLI主程序

├── backend/                      # FastAPI后端服务
│   ├── src/
│   │   ├── api/v1/
│   │   │   ├── ctbench.py       # 【新增】CTBench API端点
│   │   │   ├── data_generation.py # 【新增】数据生成API
│   │   │   ├── synthetic_data.py  # 【新增】合成数据管理
│   │   │   └── scenario_testing.py # 【新增】场景测试API
│   │   │
│   │   ├── ctbench/             # 【新增】CTBench核心模块
│   │   │   ├── models/          # TSG模型实现
│   │   │   │   ├── base_model.py    # 基础模型接口
│   │   │   │   ├── timevae.py       # TimeVAE实现
│   │   │   │   ├── quantgan.py      # Quant-GAN实现
│   │   │   │   ├── diffusion_ts.py  # Diffusion-TS实现
│   │   │   │   ├── fourier_flow.py  # Fourier Flow实现
│   │   │   │   └── model_registry.py # 模型注册中心
│   │   │   │
│   │   │   ├── tasks/           # 任务执行引擎
│   │   │   │   ├── predictive_utility.py # 预测效用任务
│   │   │   │   └── statistical_arbitrage.py # 统计套利任务
│   │   │   │
│   │   │   ├── evaluation/      # 评估指标系统
│   │   │   │   ├── metrics.py       # 13个评估指标
│   │   │   │   ├── visualizer.py    # 可视化引擎
│   │   │   │   └── benchmarker.py   # 基准测试器
│   │   │   │
│   │   │   ├── data_generation/ # 数据生成服务
│   │   │   │   ├── synthetic_data_manager.py # 合成数据管理器
│   │   │   │   ├── scenario_generator.py     # 情景生成器
│   │   │   │   └── quality_assessor.py      # 数据质量评估
│   │   │   │
│   │   │   └── training/        # 模型训练基础设施
│   │   │       ├── trainer.py       # 分布式训练器
│   │   │       ├── optimizer.py     # 超参数优化
│   │   │       └── callbacks.py     # 训练回调
│   │   │
│   │   ├── core/                # 核心业务逻辑
│   │   │   ├── enhanced_risk_manager.py  # 【增强】集成场景测试
│   │   │   ├── data_augmentation.py      # 【新增】数据增强服务
│   │   │   └── stress_testing.py         # 【新增】压力测试引擎
│   │   │
│   │   └── services/            # 外部服务
│   │       ├── ai_providers/    # AI服务提供者
│   │       │   ├── deepseek_enhanced.py  # 【增强】集成数据评估
│   │       │   └── gemini_enhanced.py    # 【增强】集成策略增强
│   │       │
│   │       └── model_serving/   # 【新增】模型服务
│   │           ├── model_server.py       # TorchServe集成
│   │           ├── inference_engine.py   # 推理引擎
│   │           └── batch_processor.py    # 批量处理器

├── rust-engine/                  # Rust高性能计算引擎
│   ├── src/
│   │   ├── ctbench_accelerator/ # 【新增】CTBench加速模块
│   │   │   ├── mod.rs
│   │   │   ├── alpha101_fast.rs     # 高速Alpha101计算
│   │   │   ├── feature_engine.rs   # 特征工程加速
│   │   │   └── tensor_ops.rs       # 张量操作优化
│   │   │
│   │   ├── indicators/          # 技术指标计算
│   │   │   ├── technical.rs     # 【增强】支持合成数据
│   │   │   └── risk_metrics.rs  # 【增强】场景风险指标
│   │   │
│   │   └── synthetic/           # 【新增】合成数据处理
│   │       ├── mod.rs
│   │       ├── data_validator.rs   # 数据验证器
│   │       └── quality_metrics.rs # 质量指标计算

├── ctbench-training/             # 【新增】独立训练环境
│   ├── experiments/             # 实验配置
│   │   ├── timevae_config.yaml
│   │   ├── quantgan_config.yaml
│   │   └── benchmark_suite.yaml
│   │
│   ├── training/                # GPU密集型训练脚本
│   │   ├── train_timevae.py
│   │   ├── train_quantgan.py
│   │   ├── distributed_trainer.py
│   │   └── hyperopt_tuner.py
│   │
│   ├── benchmarks/              # 基准测试
│   │   ├── performance_tests.py
│   │   ├── quality_benchmarks.py
│   │   └── comparison_suite.py
│   │
│   └── notebooks/               # 研究笔记本
│       ├── model_analysis.ipynb
│       ├── data_exploration.ipynb
│       └── results_visualization.ipynb

├── docker/                       # Docker配置
│   ├── frontend.Dockerfile
│   ├── backend.Dockerfile
│   ├── training.Dockerfile      # 【新增】训练环境镜像
│   ├── gpu-cluster.Dockerfile   # 【新增】GPU集群镜像
│   └── docker-compose.enhanced.yml # 【新增】增强版编排

├── monitoring/                   # 监控配置
│   ├── prometheus.yml
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   ├── trading_dashboard.json
│   │   │   ├── ctbench_dashboard.json  # 【新增】CTBench监控
│   │   │   └── model_performance.json  # 【新增】模型性能
│   │   └── datasources/

├── scripts/                      # 辅助脚本
│   ├── setup_enhanced.sh        # 【增强】环境设置脚本
│   ├── train_models.sh          # 【新增】模型训练脚本
│   ├── benchmark_suite.sh       # 【新增】基准测试脚本
│   └── deploy_gpu_cluster.sh    # 【新增】GPU集群部署

└── docs/                         # 项目文档
    ├── CTBENCH_INTEGRATION.md   # 【新增】CTBench集成指南
    ├── MODEL_TRAINING_GUIDE.md  # 【新增】模型训练指南
    ├── SYNTHETIC_DATA_USAGE.md  # 【新增】合成数据使用说明
    └── GPU_DEPLOYMENT.md        # 【新增】GPU部署指南
```

## 3. CTBench集成架构详细设计

### 3.1 时间序列生成模型集成

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np

class BaseTSGModel(ABC):
    """
    时间序列生成模型基类
    所有CTBench模型的统一接口
    """
    
    def __init__(self, config: Dict):
        """
        初始化TSG模型
        
        参数:
            config: 模型配置字典
                - n_assets: 资产数量 (默认10个主流加密货币)
                - seq_length: 序列长度 (默认500小时)
                - feature_dim: 特征维度 (OHLCV + 技术指标)
                - latent_dim: 潜在空间维度
                - device: 计算设备
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.loss_history = []
        
        # 集成到交易系统的特有属性
        self.market_regime_adapter = MarketRegimeAdapter()
        self.risk_aware_generator = RiskAwareGenerator()
        
    @abstractmethod
    def build_model(self) -> nn.Module:
        """构建模型架构"""
        pass
    
    @abstractmethod
    def train_step(self, real_data: torch.Tensor, epoch: int) -> Dict[str, float]:
        """单步训练"""
        pass
    
    @abstractmethod
    def generate(self, n_samples: int, **kwargs) -> torch.Tensor:
        """生成合成数据"""
        pass
    
    def generate_trading_scenarios(self, scenario_type: str, n_scenarios: int = 100) -> Dict:
        """
        为交易系统生成特定场景数据
        
        参数:
            scenario_type: 'normal', 'stress', 'black_swan', 'bull_market', 'bear_market'
            n_scenarios: 生成场景数量
            
        返回:
            Dict: {
                'scenarios': 场景数据,
                'metadata': 场景元信息,
                'risk_metrics': 风险指标,
                'usage_recommendations': 使用建议
            }
        """
        if scenario_type == 'black_swan':
            return self._generate_black_swan_scenarios(n_scenarios)
        elif scenario_type == 'stress':
            return self._generate_stress_scenarios(n_scenarios)
        elif scenario_type == 'bull_market':
            return self._generate_bull_market_scenarios(n_scenarios)
        elif scenario_type == 'bear_market':
            return self._generate_bear_market_scenarios(n_scenarios)
        else:
            return self._generate_normal_scenarios(n_scenarios)
    
    def _generate_black_swan_scenarios(self, n_scenarios: int) -> Dict:
        """生成黑天鹅事件场景"""
        # 设置极端参数
        extreme_config = self.config.copy()
        extreme_config.update({
            'volatility_multiplier': 5.0,    # 5倍波动率
            'correlation_breakdown': True,    # 相关性崩溃
            'liquidity_crisis': True,         # 流动性危机
            'flash_crash_prob': 0.1           # 10%闪崩概率
        })
        
        scenarios = self.generate(
            n_samples=n_scenarios, 
            config_override=extreme_config
        )
        
        return {
            'scenarios': scenarios,
            'metadata': {
                'scenario_type': 'black_swan',
                'volatility_range': '5-20x normal',
                'duration': '1-7 days',
                'recovery_period': '1-4 weeks'
            },
            'risk_metrics': self._calculate_scenario_risk_metrics(scenarios),
            'usage_recommendations': [
                '用于压力测试现有策略',
                '评估极端情况下的风控效果',
                '优化应急响应机制',
                '测试断路器触发条件'
            ]
        }
    
    def assess_generation_quality(self, 
                                  synthetic_data: torch.Tensor,
                                  real_data: torch.Tensor) -> Dict[str, float]:
        """
        评估生成数据质量
        使用加密货币特定的指标
        """
        quality_metrics = {}
        
        # 1. 统计特性相似度
        quality_metrics['price_distribution_similarity'] = self._compare_distributions(
            synthetic_data[:, :, 0],  # Close prices
            real_data[:, :, 0]
        )
        
        # 2. 波动率聚类检测
        quality_metrics['volatility_clustering'] = self._measure_volatility_clustering(
            synthetic_data
        )
        
        # 3. 长尾特性保持
        quality_metrics['tail_preservation'] = self._measure_tail_preservation(
            synthetic_data, real_data
        )
        
        # 4. 跨资产相关性
        quality_metrics['correlation_preservation'] = self._measure_correlation_preservation(
            synthetic_data, real_data
        )
        
        # 5. 24/7交易特性
        quality_metrics['continuous_trading_patterns'] = self._measure_continuous_patterns(
            synthetic_data
        )
        
        # 6. 交易信号有效性
        quality_metrics['trading_signal_validity'] = self._assess_trading_signals(
            synthetic_data
        )
        
        # 综合质量评分
        quality_metrics['overall_quality'] = np.mean([
            quality_metrics['price_distribution_similarity'],
            quality_metrics['volatility_clustering'],
            quality_metrics['tail_preservation'],
            quality_metrics['correlation_preservation'],
            quality_metrics['continuous_trading_patterns'],
            quality_metrics['trading_signal_validity']
        ])
        
        return quality_metrics
```

### 3.2 数据生成服务集成

```python
class SyntheticDataManager:
    """
    合成数据管理器
    将CTBench集成到交易系统的核心组件
    """
    
    def __init__(self):
        self.tsg_models = {
            'timevae': TimeVAE(self._get_model_config('timevae')),
            'quantgan': QuantGAN(self._get_model_config('quantgan')),
            'diffusion_ts': DiffusionTS(self._get_model_config('diffusion_ts')),
            'fourier_flow': FourierFlow(self._get_model_config('fourier_flow'))
        }
        
        self.model_registry = ModelRegistry()
        self.quality_assessor = DataQualityAssessor()
        self.scenario_generator = ScenarioGenerator()
        self.rust_accelerator = RustAccelerator()  # Rust加速器
        
        # 与现有系统的集成点
        self.risk_manager = None  # 将在运行时注入
        self.feature_engineer = None
        self.mongo_client = None
        self.redis_client = None
    
    async def generate_augmented_training_data(self,
                                             strategy_type: str,
                                             base_data: np.ndarray,
                                             augmentation_ratio: float = 0.3) -> Dict:
        """
        为策略训练生成增强数据
        
        参数:
            strategy_type: 策略类型 ('grid', 'dca', 'momentum', 'mean_reversion')
            base_data: 基础历史数据
            augmentation_ratio: 增强比例 (0.3 = 30%合成数据)
            
        返回:
            Dict: 包含原始数据和增强数据的组合
        """
        print(f"🔬 为{strategy_type}策略生成增强训练数据...")
        
        # 1. 根据策略类型选择最适合的TSG模型
        best_model = await self._select_optimal_model_for_strategy(strategy_type, base_data)
        print(f"📊 选择模型: {best_model}")
        
        # 2. 训练TSG模型
        training_results = await self._train_model_on_data(best_model, base_data)
        
        # 3. 生成增强数据
        n_synthetic_samples = int(len(base_data) * augmentation_ratio)
        synthetic_data = await self.tsg_models[best_model].generate(
            n_samples=n_synthetic_samples,
            conditioning_data=base_data[-100:],  # 使用最近100个数据点作为条件
            strategy_specific_config=self._get_strategy_specific_config(strategy_type)
        )
        
        # 4. 质量评估
        quality_metrics = await self.quality_assessor.assess(
            synthetic_data=synthetic_data,
            real_data=base_data,
            strategy_context=strategy_type
        )
        
        # 5. 数据混合
        if quality_metrics['overall_quality'] > 0.7:  # 质量阈值
            augmented_data = self._blend_data(base_data, synthetic_data, augmentation_ratio)
            
            # 6. 保存到缓存
            await self._cache_augmented_data(strategy_type, augmented_data, quality_metrics)
            
            return {
                'status': 'success',
                'augmented_data': augmented_data,
                'original_size': len(base_data),
                'synthetic_size': len(synthetic_data),
                'quality_metrics': quality_metrics,
                'model_used': best_model,
                'recommendations': self._get_usage_recommendations(quality_metrics)
            }
        else:
            return {
                'status': 'low_quality',
                'message': f"生成数据质量过低 ({quality_metrics['overall_quality']:.2f}), 建议使用原始数据",
                'quality_metrics': quality_metrics
            }
    
    async def generate_stress_test_scenarios(self,
                                           current_portfolio: Dict,
                                           test_types: List[str]) -> Dict:
        """
        为风控系统生成压力测试场景
        
        参数:
            current_portfolio: 当前投资组合配置
            test_types: 测试类型列表 ['black_swan', 'flash_crash', 'correlation_breakdown']
            
        返回:
            Dict: 压力测试场景数据
        """
        print("🧪 生成风控压力测试场景...")
        
        scenarios = {}
        
        for test_type in test_types:
            print(f"  生成 {test_type} 场景...")
            
            # 选择适合的模型
            if test_type == 'black_swan':
                model = self.tsg_models['quantgan']  # GAN更适合极端事件
            elif test_type == 'flash_crash':
                model = self.tsg_models['diffusion_ts']  # 扩散模型适合急速变化
            else:
                model = self.tsg_models['timevae']  # VAE适合相关性建模
            
            # 生成场景数据
            scenario_data = await model.generate_trading_scenarios(
                scenario_type=test_type,
                n_scenarios=50,
                portfolio_context=current_portfolio
            )
            
            # 使用Rust引擎加速风险计算
            risk_metrics = await self.rust_accelerator.calculate_portfolio_risk(
                portfolio=current_portfolio,
                scenario_data=scenario_data['scenarios'],
                metrics=['var', 'expected_shortfall', 'max_drawdown']
            )
            
            scenarios[test_type] = {
                'scenario_data': scenario_data,
                'risk_metrics': risk_metrics,
                'impact_analysis': self._analyze_portfolio_impact(
                    current_portfolio, risk_metrics
                )
            }
        
        # 生成压力测试报告
        stress_test_report = self._generate_stress_test_report(scenarios, current_portfolio)
        
        # 保存到数据库
        await self._save_stress_test_results(stress_test_report)
        
        return {
            'scenarios': scenarios,
            'report': stress_test_report,
            'recommendations': self._get_risk_management_recommendations(scenarios)
        }
    
    async def _select_optimal_model_for_strategy(self, 
                                                strategy_type: str, 
                                                data: np.ndarray) -> str:
        """
        根据策略类型和数据特性选择最优TSG模型
        """
        data_characteristics = await self._analyze_data_characteristics(data)
        
        # 模型选择规则
        if strategy_type == 'grid':
            # 网格策略需要稳定的价格区间，选择VAE
            return 'timevae'
        elif strategy_type == 'momentum':
            # 动量策略需要趋势延续，选择Flow模型
            return 'fourier_flow'
        elif strategy_type == 'mean_reversion':
            # 均值回归需要周期性模式，选择GAN
            return 'quantgan'
        elif data_characteristics['volatility'] > 0.5:
            # 高波动数据选择Diffusion模型
            return 'diffusion_ts'
        else:
            # 默认选择TimeVAE
            return 'timevae'
```

### 3.3 CLI界面集成

```python
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from textual.screen import Screen
from textual.widgets import Header, Footer, Static, Button, DataTable

class DataLaboratory(Screen):
    """
    数据实验室界面
    CTBench模型训练和数据生成的GUI
    """
    
    CSS = """
    #data_lab_grid {
        layout: grid;
        grid-size: 2 2;
        grid-gutter: 1;
        height: 100%;
    }
    
    .model_panel {
        border: solid #7209B7;
        background: #1B263B;
        padding: 1;
    }
    
    .generation_panel {
        border: solid #52B788;
        background: #1B263B;
        padding: 1;
    }
    
    .training_panel {
        border: solid #F2CC8F;
        background: #1B263B;
        padding: 1;
    }
    
    .results_panel {
        border: solid #277DA1;
        background: #1B263B;
        padding: 1;
    }
    """
    
    BINDINGS = [
        ("1", "select_timevae", "TimeVAE"),
        ("2", "select_quantgan", "Quant-GAN"),
        ("3", "select_diffusion", "Diffusion-TS"),
        ("4", "select_fourier_flow", "Fourier Flow"),
        ("t", "start_training", "开始训练"),
        ("g", "generate_data", "生成数据"),
        ("s", "stress_test", "压力测试"),
        ("q", "quit_lab", "退出实验室"),
    ]
    
    def __init__(self):
        super().__init__()
        self.selected_model = 'timevae'
        self.training_progress = 0
        self.generation_status = "待机"
        self.model_metrics = {}
        
        # 集成服务
        self.synthetic_data_manager = SyntheticDataManager()
        self.model_trainer = ModelTrainer()
        
    def compose(self) -> ComposeResult:
        """组合数据实验室界面"""
        yield Header()
        
        with Container(id="data_lab_grid"):
            # 左上：模型选择和状态
            yield Static(self._create_model_selection_panel(), classes="model_panel")
            
            # 右上：数据生成控制
            yield Static(self._create_data_generation_panel(), classes="generation_panel")
            
            # 左下：训练监控
            yield Static(self._create_training_monitor_panel(), classes="training_panel")
            
            # 右下：结果展示
            yield Static(self._create_results_panel(), classes="results_panel")
        
        yield Footer()
    
    def _create_model_selection_panel(self) -> Panel:
        """创建模型选择面板"""
        
        models_table = Table.grid(padding=1)
        models_table.add_column(style="cyan", width=15)
        models_table.add_column(width=30)
        models_table.add_column(style="green", width=10)
        
        models_info = {
            'timevae': {'name': 'TimeVAE', 'type': '变分自编码器', 'status': '✅已训练'},
            'quantgan': {'name': 'Quant-GAN', 'type': '生成对抗网络', 'status': '🔄训练中'},
            'diffusion_ts': {'name': 'Diffusion-TS', 'type': '扩散模型', 'status': '⏸️暂停'},
            'fourier_flow': {'name': 'Fourier Flow', 'type': '标准化流', 'status': '❌未训练'}
        }
        
        for model_id, info in models_info.items():
            # 当前选择的模型用不同颜色标识
            name_style = "bold yellow" if model_id == self.selected_model else "white"
            
            models_table.add_row(
                f"[{name_style}]{info['name']}[/{name_style}]",
                f"[blue]{info['type']}[/blue]",
                info['status']
            )
        
        # 模型详细信息
        if self.selected_model in self.model_metrics:
            metrics = self.model_metrics[self.selected_model]
            details = f"""
📊 模型详情:
• 训练轮数: {metrics.get('epochs', 0)}
• 损失值: {metrics.get('loss', 0.0):.4f}  
• 质量评分: {metrics.get('quality_score', 0.0):.2f}
• 训练时间: {metrics.get('training_time', '0min')}
"""
        else:
            details = "\n📋 选择模型查看详情"
        
        content = Group(
            Text(f"🧬 当前模型: [bold yellow]{self.selected_model.upper()}[/bold yellow]", style="bold purple"),
            "",
            models_table,
            "",
            details
        )
        
        return Panel(
            content,
            title="🤖 TSG模型选择",
            border_style="purple"
        )
    
    def _create_data_generation_panel(self) -> Panel:
        """创建数据生成控制面板"""
        
        generation_controls = Table.grid(padding=1)
        generation_controls.add_column(style="cyan", width=18)
        generation_controls.add_column(width=25)
        
        generation_controls.add_row("🎯 生成类型:", "[yellow]策略增强数据[/yellow]")
        generation_controls.add_row("📈 目标策略:", "[blue]网格交易策略[/blue]")
        generation_controls.add_row("📊 样本数量:", "[green]1000个场景[/green]")
        generation_controls.add_row("⚡ 加速模式:", "[purple]Rust引擎[/purple]")
        generation_controls.add_row("🔄 当前状态:", f"[bold]{self.generation_status}[/bold]")
        
        # 生成选项
        options_text = """
[bold green]🚀 G - 生成增强数据[/bold green]
  └─ 为当前策略生成训练数据增强

[bold yellow]🧪 S - 压力测试场景[/bold yellow]
  └─ 生成黑天鹅/闪崩等极端场景

[bold blue]📊 Q - 质量评估[/bold blue]
  └─ 评估已生成数据的质量指标

[bold red]💾 E - 导出数据[/bold red]
  └─ 导出生成的数据到文件
        """
        
        content = Group(
            Text("📡 数据生成控制中心", style="bold green"),
            "",
            generation_controls,
            "",
            options_text
        )
        
        return Panel(
            content,
            title="🎛️ 数据生成",
            border_style="green"
        )
    
    def _create_training_monitor_panel(self) -> Panel:
        """创建训练监控面板"""
        
        # 训练进度条
        progress_bar = self._create_progress_bar(self.training_progress)
        
        # 训练指标表
        training_metrics = Table.grid(padding=1)
        training_metrics.add_column(style="cyan", width=15)
        training_metrics.add_column(width=20)
        
        training_metrics.add_row("⏱️ 已用时间:", "[blue]45分30秒[/blue]")
        training_metrics.add_row("🔥 当前轮次:", "[yellow]150/500[/yellow]")
        training_metrics.add_row("📉 训练损失:", "[green]0.0234[/green]")
        training_metrics.add_row("📈 验证损失:", "[green]0.0267[/green]")
        training_metrics.add_row("🎯 学习率:", "[purple]0.0001[/purple]")
        training_metrics.add_row("💾 检查点:", "[blue]已保存[/blue]")
        
        # GPU使用情况
        gpu_info = """
🖥️ GPU状态:
  • GPU 0: RTX 4090 - 85% 使用率
  • 显存: 18.2GB / 24GB  
  • 温度: 76°C
  • 功耗: 380W
"""
        
        # 训练日志（最近几行）
        recent_logs = """
📝 最近日志:
[15:30:42] Epoch 150 - Loss: 0.0234 ⬇️
[15:30:41] Validation - Loss: 0.0267 ⬆️
[15:30:40] Saved checkpoint at epoch 150 💾
[15:30:35] Learning rate reduced to 0.0001 📉
        """
        
        content = Group(
            Text("🏋️ 模型训练监控", style="bold yellow"),
            "",
            progress_bar,
            "",
            training_metrics,
            "",
            gpu_info,
            "",
            recent_logs
        )
        
        return Panel(
            content,
            title="📈 训练监控",
            border_style="yellow"
        )
    
    def _create_results_panel(self) -> Panel:
        """创建结果展示面板"""
        
        # 质量评估结果
        quality_results = Table()
        quality_results.add_column("指标", style="cyan")
        quality_results.add_column("数值", style="yellow")
        quality_results.add_column("评级", style="green")
        
        quality_results.add_row("价格分布相似度", "0.89", "⭐⭐⭐⭐")
        quality_results.add_row("波动率聚类", "0.92", "⭐⭐⭐⭐⭐")
        quality_results.add_row("长尾特性保持", "0.85", "⭐⭐⭐⭐")
        quality_results.add_row("相关性保持", "0.91", "⭐⭐⭐⭐⭐")
        quality_results.add_row("交易信号有效性", "0.87", "⭐⭐⭐⭐")
        quality_results.add_row("综合质量评分", "0.89", "⭐⭐⭐⭐")
        
        # 生成数据统计
        generation_stats = """
📊 生成数据统计:
• 总样本数: 1,000个场景
• 时间跨度: 每场景500小时  
• 资产数量: 10个主流币种
• 特征维度: OHLCV + 30个技术指标
• 文件大小: 2.3GB
• 生成耗时: 12分钟
        """
        
        # 使用建议
        recommendations = """
💡 使用建议:
✅ 数据质量优良，适合策略训练
✅ 可用于回测验证和风险评估
⚠️ 建议与真实数据混合使用 (70:30)
⚠️ 极端场景数据谨慎用于实盘
        """
        
        content = Group(
            Text("📋 数据质量评估结果", style="bold blue"),
            "",
            quality_results,
            "",
            generation_stats,
            "",
            recommendations
        )
        
        return Panel(
            content,
            title="📊 结果分析",
            border_style="blue"
        )
    
    async def action_start_training(self):
        """开始模型训练"""
        self.notify("🚀 开始训练TSG模型...", severity="information")
        
        # 启动异步训练任务
        training_task = asyncio.create_task(
            self._train_selected_model()
        )
        
        # 更新界面状态
        self.generation_status = "训练中"
        await self.refresh_data_lab()
    
    async def action_generate_data(self):
        """生成合成数据"""
        self.notify("📡 开始生成合成数据...", severity="information")
        
        try:
            # 调用合成数据管理器
            results = await self.synthetic_data_manager.generate_augmented_training_data(
                strategy_type='grid',
                base_data=self._get_current_market_data(),
                augmentation_ratio=0.3
            )
            
            if results['status'] == 'success':
                self.notify(
                    f"✅ 数据生成完成！质量评分: {results['quality_metrics']['overall_quality']:.2f}", 
                    severity="success"
                )
                self.generation_status = "生成完成"
            else:
                self.notify("❌ 数据质量不达标，建议重新训练模型", severity="warning")
                self.generation_status = "质量不达标"
                
        except Exception as e:
            self.notify(f"❌ 生成失败: {str(e)}", severity="error")
            self.generation_status = "生成失败"
        
        await self.refresh_data_lab()
    
    async def action_stress_test(self):
        """生成压力测试场景"""
        self.notify("🧪 生成压力测试场景...", severity="information")
        
        try:
            current_portfolio = await self._get_current_portfolio_config()
            
            stress_results = await self.synthetic_data_manager.generate_stress_test_scenarios(
                current_portfolio=current_portfolio,
                test_types=['black_swan', 'flash_crash', 'correlation_breakdown']
            )
            
            self.notify("✅ 压力测试场景生成完成", severity="success")
            
            # 显示风险警报（如果有）
            critical_risks = [
                risk for risk in stress_results['recommendations'] 
                if risk['severity'] == 'critical'
            ]
            
            if critical_risks:
                self.notify(
                    f"⚠️ 发现{len(critical_risks)}个严重风险点，请查看详细报告", 
                    severity="warning"
                )
            
        except Exception as e:
            self.notify(f"❌ 压力测试失败: {str(e)}", severity="error")
```

### 3.4 API端点集成

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "\u5206\u6790CTBench\u4e0e\u73b0\u6709\u7cfb\u7edf\u7684\u96c6\u6210\u70b9", "status": "completed", "id": "14"}, {"content": "\u8bbe\u8ba1CTBench\u96c6\u6210\u67b6\u6784", "status": "completed", "id": "15"}, {"content": "\u66f4\u65b0\u5168\u6808\u67b6\u6784\u6587\u6863", "status": "completed", "id": "16"}, {"content": "\u751f\u6210CTBench\u96c6\u6210\u4ee3\u7801", "status": "in_progress", "id": "17"}]