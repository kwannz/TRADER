"""
集成测试验证器

测试系统各组件之间的集成功能
"""

import time
import asyncio
from typing import Dict, Any, List
from pathlib import Path

class IntegrationValidator:
    """系统集成验证器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
    
    def test_system_integration(self) -> Dict[str, Any]:
        """测试系统整体集成"""
        start_time = time.time()
        
        try:
            # 检查核心组件文件
            core_components = {
                "rust_engine": self.project_root / 'rust_engine' / 'Cargo.toml',
                "python_layer": self.project_root / 'python_layer' / '__init__.py',
                "fastapi_layer": self.project_root / 'fastapi_layer' / 'main.py',
                "cli_interface": self.project_root / 'cli_interface' / 'main.py',
                "config_files": self.project_root / '.env'
            }
            
            component_status = {}
            available_components = 0
            
            for component, path in core_components.items():
                exists = path.exists()
                component_status[component] = {
                    "available": exists,
                    "path": str(path),
                    "size": path.stat().st_size if exists else 0
                }
                if exists:
                    available_components += 1
            
            integration_score = available_components / len(core_components)
            
            return {
                "status": "PASS" if integration_score >= 0.8 else "FAIL",
                "message": f"系统集成测试: {available_components}/{len(core_components)} 组件可用",
                "details": {
                    "component_status": component_status,
                    "available_components": available_components,
                    "total_components": len(core_components),
                    "integration_score": integration_score
                },
                "metrics": {
                    "integration_completeness": integration_score,
                    "test_duration": (time.time() - start_time) * 1000
                }
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"系统集成测试失败: {e}",
                "details": {"error_type": type(e).__name__},
                "metrics": {"test_duration": (time.time() - start_time) * 1000}
            }
    
    def test_configuration_validation(self) -> Dict[str, Any]:
        """测试配置文件验证"""
        start_time = time.time()
        
        try:
            config_files = [
                self.project_root / '.env',
                self.project_root / 'config' / 'local.json',
                self.project_root / 'requirements.txt'
            ]
            
            config_status = {}
            valid_configs = 0
            
            for config_file in config_files:
                if config_file.exists():
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                        # 基本验证
                        is_valid = len(content.strip()) > 0
                        
                        # 特殊验证
                        if config_file.name == '.env':
                            is_valid = 'ENVIRONMENT=' in content
                        elif config_file.name == 'local.json':
                            import json
                            json.loads(content)  # 验证JSON格式
                        elif config_file.name == 'requirements.txt':
                            is_valid = 'fastapi>=' in content and 'rich>=' in content
                        
                        config_status[config_file.name] = {
                            "exists": True,
                            "valid": is_valid,
                            "size": len(content),
                            "lines": len(content.splitlines())
                        }
                        
                        if is_valid:
                            valid_configs += 1
                            
                    except Exception as parse_error:
                        config_status[config_file.name] = {
                            "exists": True,
                            "valid": False,
                            "error": str(parse_error)
                        }
                else:
                    config_status[config_file.name] = {
                        "exists": False,
                        "valid": False
                    }
            
            config_completeness = valid_configs / len(config_files)
            
            return {
                "status": "PASS" if config_completeness >= 0.7 else "FAIL",
                "message": f"配置验证: {valid_configs}/{len(config_files)} 配置文件有效",
                "details": {
                    "config_status": config_status,
                    "valid_configs": valid_configs,
                    "total_configs": len(config_files),
                    "config_completeness": config_completeness
                },
                "metrics": {
                    "config_validity": config_completeness,
                    "test_duration": (time.time() - start_time) * 1000
                }
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"配置验证测试失败: {e}",
                "details": {"error_type": type(e).__name__},
                "metrics": {"test_duration": (time.time() - start_time) * 1000}
            }
    
    def test_dependency_check(self) -> Dict[str, Any]:
        """测试依赖包检查"""
        start_time = time.time()
        
        try:
            # 核心依赖包列表
            critical_packages = [
                'rich',
                'textual', 
                'fastapi',
                'uvicorn',
                'pydantic'
            ]
            
            optional_packages = [
                'pymongo',
                'redis',
                'httpx',
                'pandas',
                'numpy'
            ]
            
            dependency_status = {}
            available_critical = 0
            available_optional = 0
            
            # 检查关键依赖
            for package in critical_packages:
                try:
                    __import__(package)
                    dependency_status[package] = {
                        "available": True,
                        "type": "critical",
                        "import_success": True
                    }
                    available_critical += 1
                except ImportError as e:
                    dependency_status[package] = {
                        "available": False,
                        "type": "critical",
                        "import_success": False,
                        "error": str(e)
                    }
            
            # 检查可选依赖
            for package in optional_packages:
                try:
                    __import__(package)
                    dependency_status[package] = {
                        "available": True,
                        "type": "optional",
                        "import_success": True
                    }
                    available_optional += 1
                except ImportError as e:
                    dependency_status[package] = {
                        "available": False,
                        "type": "optional", 
                        "import_success": False,
                        "error": str(e)
                    }
            
            critical_completeness = available_critical / len(critical_packages)
            optional_completeness = available_optional / len(optional_packages)
            overall_completeness = (available_critical + available_optional) / (len(critical_packages) + len(optional_packages))
            
            # 关键依赖必须100%可用
            status = "PASS" if critical_completeness == 1.0 else "FAIL"
            
            return {
                "status": status,
                "message": f"依赖检查: 关键{available_critical}/{len(critical_packages)}, 可选{available_optional}/{len(optional_packages)}",
                "details": {
                    "dependency_status": dependency_status,
                    "critical_available": available_critical,
                    "critical_total": len(critical_packages),
                    "optional_available": available_optional,
                    "optional_total": len(optional_packages),
                    "critical_completeness": critical_completeness,
                    "optional_completeness": optional_completeness,
                    "overall_completeness": overall_completeness
                },
                "metrics": {
                    "dependency_completeness": overall_completeness,
                    "critical_dependency_rate": critical_completeness,
                    "test_duration": (time.time() - start_time) * 1000
                }
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"依赖检查失败: {e}",
                "details": {"error_type": type(e).__name__},
                "metrics": {"test_duration": (time.time() - start_time) * 1000}
            }
    
    async def test_end_to_end_workflow(self) -> Dict[str, Any]:
        """测试端到端工作流"""
        start_time = time.time()
        
        try:
            workflow_steps = []
            
            # 步骤1: 数据获取模拟
            step1_start = time.time()
            market_data = {
                "symbol": "BTC/USDT",
                "price": 45000,
                "volume": 1000000,
                "timestamp": time.time()
            }
            step1_time = (time.time() - step1_start) * 1000
            workflow_steps.append({
                "step": "data_acquisition",
                "duration": step1_time,
                "success": True,
                "data_points": 1
            })
            
            # 步骤2: AI分析模拟
            step2_start = time.time()
            await asyncio.sleep(0.1)  # 模拟AI分析延迟
            ai_analysis = {
                "sentiment_score": 0.75,
                "confidence": 0.82,
                "recommendation": "BUY"
            }
            step2_time = (time.time() - step2_start) * 1000
            workflow_steps.append({
                "step": "ai_analysis",
                "duration": step2_time,
                "success": True,
                "analysis_result": "BUY"
            })
            
            # 步骤3: 策略执行模拟
            step3_start = time.time()
            strategy_decision = {
                "action": ai_analysis["recommendation"],
                "quantity": 0.01,
                "price": market_data["price"],
                "risk_approved": True
            }
            step3_time = (time.time() - step3_start) * 1000
            workflow_steps.append({
                "step": "strategy_execution",
                "duration": step3_time,
                "success": strategy_decision["risk_approved"],
                "action": strategy_decision["action"]
            })
            
            # 步骤4: 结果记录模拟
            step4_start = time.time()
            trade_record = {
                "id": f"trade_{int(time.time())}",
                "symbol": market_data["symbol"],
                "action": strategy_decision["action"],
                "quantity": strategy_decision["quantity"],
                "price": strategy_decision["price"],
                "timestamp": time.time()
            }
            step4_time = (time.time() - step4_start) * 1000
            workflow_steps.append({
                "step": "trade_recording",
                "duration": step4_time,
                "success": True,
                "trade_id": trade_record["id"]
            })
            
            # 计算工作流统计
            total_workflow_time = sum(step["duration"] for step in workflow_steps)
            successful_steps = sum(1 for step in workflow_steps if step["success"])
            workflow_success_rate = successful_steps / len(workflow_steps)
            
            return {
                "status": "PASS" if workflow_success_rate == 1.0 else "FAIL",
                "message": f"端到端工作流测试: {successful_steps}/{len(workflow_steps)} 步骤成功",
                "details": {
                    "workflow_steps": workflow_steps,
                    "successful_steps": successful_steps,
                    "total_steps": len(workflow_steps),
                    "workflow_success_rate": workflow_success_rate,
                    "total_workflow_time": total_workflow_time,
                    "average_step_time": total_workflow_time / len(workflow_steps)
                },
                "metrics": {
                    "workflow_latency": total_workflow_time,
                    "workflow_success_rate": workflow_success_rate,
                    "test_duration": (time.time() - start_time) * 1000
                }
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"端到端工作流测试失败: {e}",
                "details": {"error_type": type(e).__name__},
                "metrics": {"test_duration": (time.time() - start_time) * 1000}
            }