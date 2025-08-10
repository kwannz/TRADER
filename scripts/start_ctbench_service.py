#!/usr/bin/env python3
"""
CTBench Service Startup Script
CTBench服务启动脚本
"""

import sys
import os
import asyncio
import logging
from pathlib import Path
import json
import argparse
from typing import Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.integration.model_service import CTBenchModelService
from src.integration.risk_control.enhanced_risk_manager import EnhancedRiskManager

def setup_logging(config: dict):
    """设置日志"""
    log_config = config.get('logging', {})
    
    # 创建日志目录
    log_file = log_config.get('file', 'logs/ctbench.log')
    log_dir = Path(log_file).parent
    log_dir.mkdir(exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('ctbench_startup')
    logger.info("CTBench服务启动脚本开始运行")
    return logger

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        return {}

def display_startup_banner():
    """显示启动横幅"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║       ▄████▄  ▄▄▄█████▓ ▄▄▄▄   ▓█████ ███▄    █  ▄████▄     ║
    ║      ▒██▀ ▀█  ▓  ██▒ ▓▒▓█████▄ ▓█   ▀ ██ ▀█   █ ▒██▀ ▀█     ║
    ║      ▒▓█    ▄ ▒ ▓██░ ▒░▒██▒ ▄██▒███  ▓██  ▀█ ██▒▒▓█    ▄    ║
    ║      ▒▓▓▄ ▄██▒░ ▓██▓ ░ ▒██░█▀  ▒▓█  ▄▓██▒  ▐▌██▒▒▓▓▄ ▄██▒   ║
    ║      ▒ ▓███▀ ░  ▒██▒ ░ ░▓█  ▀█▓░▒████▒██░   ▓██░▒ ▓███▀ ░   ║
    ║      ░ ░▒ ▒  ░  ▒ ░░   ░▒▓███▀▒░░ ▒░ ░ ▒░   ▒ ▒ ░ ░▒ ▒  ░   ║
    ║        ░  ▒       ░    ▒░▒   ░  ░ ░  ░ ░░   ░ ▒░  ░  ▒      ║
    ║      ░          ░       ░    ░    ░     ░   ░ ░ ░           ║
    ║      ░ ░                ░         ░  ░        ░ ░ ░         ║
    ║      ░                       ░                  ░           ║
    ║                                                              ║
    ║              时序生成模型基准平台 v1.0                        ║
    ║            Cryptocurrency Time Series Benchmark              ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print("🚀 正在启动CTBench服务...")
    print("=" * 64)

async def initialize_models(ctbench_service: CTBenchModelService, 
                          config: dict, logger: logging.Logger):
    """初始化模型"""
    model_configs = config.get('ctbench_models', {})
    
    logger.info("开始初始化CTBench模型...")
    
    for model_type, model_config in model_configs.items():
        try:
            logger.info(f"正在初始化 {model_type} 模型...")
            
            # 更新模型配置
            ctbench_service.synthetic_manager.model_configs[model_type] = model_config
            
            # 初始化模型
            success = ctbench_service.synthetic_manager.initialize_model(model_type)
            
            if success:
                logger.info(f"✓ {model_type} 模型初始化成功")
            else:
                logger.warning(f"✗ {model_type} 模型初始化失败")
                
        except Exception as e:
            logger.error(f"初始化 {model_type} 模型时出错: {e}")

async def run_health_check(ctbench_service: CTBenchModelService,
                         risk_manager: EnhancedRiskManager,
                         logger: logging.Logger) -> bool:
    """运行健康检查"""
    logger.info("开始系统健康检查...")
    
    try:
        # 检查CTBench服务
        service_stats = ctbench_service.get_service_stats()
        if service_stats['is_running']:
            logger.info("✓ CTBench服务运行正常")
        else:
            logger.error("✗ CTBench服务未运行")
            return False
            
        # 检查模型状态
        model_status = ctbench_service.synthetic_manager.get_model_status()
        initialized_models = [name for name, info in model_status.items() if info['initialized']]
        
        if initialized_models:
            logger.info(f"✓ 已初始化模型: {', '.join(initialized_models)}")
        else:
            logger.warning("! 没有已初始化的模型")
            
        # 生成测试数据
        if initialized_models:
            test_model = initialized_models[0]
            logger.info(f"正在使用 {test_model} 进行测试生成...")
            
            result = ctbench_service.synthetic_manager.generate_synthetic_data(
                test_model, 10  # 生成10个测试样本
            )
            
            if result['success']:
                logger.info(f"✓ 测试生成成功，形状: {result['shape']}")
            else:
                logger.warning(f"! 测试生成失败: {result.get('error', '未知错误')}")
                
        logger.info("✓ 系统健康检查完成")
        return True
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return False

async def start_services(config: dict, logger: logging.Logger):
    """启动服务"""
    try:
        # 初始化CTBench服务
        logger.info("正在初始化CTBench模型服务...")
        config_path = str(project_root / "config" / "ctbench_config.json")
        ctbench_service = CTBenchModelService(config_path)
        
        # 初始化风险管理器
        logger.info("正在初始化增强风险管理器...")
        risk_manager = EnhancedRiskManager(config.get('risk_management', {}))
        await risk_manager.initialize()
        
        # 初始化模型
        await initialize_models(ctbench_service, config, logger)
        
        # 运行健康检查
        health_ok = await run_health_check(ctbench_service, risk_manager, logger)
        
        if not health_ok:
            logger.error("健康检查失败，服务可能无法正常运行")
            
        # 启动CTBench服务
        logger.info("正在启动CTBench模型服务...")
        service_task = asyncio.create_task(ctbench_service.start_service())
        
        logger.info("🎉 CTBench服务启动完成!")
        logger.info("=" * 50)
        logger.info("服务信息:")
        logger.info(f"  - 配置文件: {config_path}")
        logger.info(f"  - 日志文件: {config.get('logging', {}).get('file', 'logs/ctbench.log')}")
        logger.info(f"  - 工作目录: {project_root}")
        
        # 显示可用的模型
        model_status = ctbench_service.synthetic_manager.get_model_status()
        initialized_models = [name for name, info in model_status.items() if info['initialized']]
        if initialized_models:
            logger.info(f"  - 可用模型: {', '.join(initialized_models)}")
        
        logger.info("=" * 50)
        logger.info("使用 Ctrl+C 停止服务")
        
        # 等待服务任务
        await service_task
        
    except KeyboardInterrupt:
        logger.info("\n收到停止信号，正在关闭服务...")
        await ctbench_service.stop_service()
        logger.info("✓ 服务已停止")
        
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='CTBench Service Launcher')
    parser.add_argument(
        '--config', 
        default=str(project_root / "config" / "ctbench_config.json"),
        help='配置文件路径'
    )
    parser.add_argument(
        '--no-banner', 
        action='store_true',
        help='不显示启动横幅'
    )
    parser.add_argument(
        '--health-check-only',
        action='store_true',
        help='仅运行健康检查'
    )
    
    args = parser.parse_args()
    
    # 显示启动横幅
    if not args.no_banner:
        display_startup_banner()
    
    # 加载配置
    config = load_config(args.config)
    if not config:
        print("无法加载配置文件，使用默认配置")
        config = {}
        
    # 设置日志
    logger = setup_logging(config)
    
    # 运行服务
    try:
        if args.health_check_only:
            logger.info("仅运行健康检查模式")
            # 这里可以添加健康检查逻辑
        else:
            asyncio.run(start_services(config, logger))
    except KeyboardInterrupt:
        logger.info("服务被用户中断")
    except Exception as e:
        logger.error(f"服务运行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()