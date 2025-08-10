"""
CLI模块验证测试器

提供完整的模块验证、测试和调试功能
支持实时结果显示和详细报告生成
"""

import asyncio
import time
import sys
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess

# Rich组件用于美观的CLI输出
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TimeElapsedColumn,
    TimeRemainingColumn
)
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.tree import Tree
from rich.align import Align
from rich.rule import Rule

# 验证器模块导入 - 修复相对导入问题
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'validators'))

from rust_engine_test import RustEngineValidator
from python_layer_test import PythonLayerValidator  
from fastapi_test import FastAPIValidator
from database_test import DatabaseValidator
from integration_test import IntegrationValidator

console = Console()

@dataclass
class ValidationResult:
    """验证结果数据结构"""
    module_name: str
    test_name: str
    status: str  # "PASS", "FAIL", "SKIP", "ERROR"
    duration_ms: float
    message: str
    details: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: datetime
    error_trace: Optional[str] = None

@dataclass
class ModuleTestReport:
    """模块测试报告"""
    module_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    total_duration_ms: float
    results: List[ValidationResult]
    status: str  # "PASS", "FAIL", "PARTIAL"

class CLITester:
    """
    CLI模块测试器主类
    
    负责协调所有模块的验证测试
    """
    
    def __init__(self):
        self.console = Console(record=True)
        self.validators: Dict[str, Any] = {}
        self.test_results: Dict[str, ModuleTestReport] = {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # 配置选项
        self.verbose = False
        self.parallel = True
        self.timeout_seconds = 300  # 5分钟总超时
        self.module_timeout = 60    # 单模块60秒超时
        
        # 性能基准
        self.performance_thresholds = {
            "rust_engine": {
                "data_processing_time": 100,  # ms
                "strategy_execution_time": 200,  # ms
                "risk_check_time": 50,  # ms
            },
            "python_layer": {
                "ai_response_time": 2000,  # ms
                "data_query_time": 500,  # ms
                "model_load_time": 1000,  # ms
            },
            "fastapi": {
                "api_response_time": 100,  # ms
                "websocket_latency": 50,  # ms
                "auth_time": 200,  # ms
            },
            "database": {
                "connection_time": 1000,  # ms
                "query_time": 100,  # ms
                "write_time": 200,  # ms
            }
        }
        
    def add_validator(self, validator_instance) -> None:
        """添加验证器实例"""
        module_name = validator_instance.__class__.__name__.replace('Validator', '').lower()
        self.validators[module_name] = validator_instance
        
    async def run_all_validations(self, modules: Optional[List[str]] = None) -> Dict[str, ModuleTestReport]:
        """运行所有模块验证"""
        self.start_time = datetime.utcnow()
        
        try:
            # 确定要测试的模块
            test_modules = modules or list(self.validators.keys())
            
            with self.console.status("[bold green]初始化测试环境...") as status:
                await asyncio.sleep(0.5)
                status.update("[bold blue]开始模块验证...")
                
                # 显示测试概览
                self._show_test_overview(test_modules)
                
                # 并行或串行执行测试
                if self.parallel and len(test_modules) > 1:
                    await self._run_parallel_tests(test_modules)
                else:
                    await self._run_sequential_tests(test_modules)
            
            self.end_time = datetime.utcnow()
            
            # 显示最终结果
            self._show_final_results()
            
            return self.test_results
            
        except KeyboardInterrupt:
            self.console.print("\n[yellow]测试被用户中断[/yellow]")
            return self.test_results
        except Exception as e:
            self.console.print(f"\n[red]测试执行错误: {e}[/red]")
            return self.test_results

    async def _run_parallel_tests(self, modules: List[str]) -> None:
        """并行执行测试"""
        tasks = []
        
        # 创建进度显示
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=True
        )
        
        with progress:
            # 为每个模块创建进度任务
            progress_tasks = {}
            for module in modules:
                task_id = progress.add_task(f"[cyan]测试 {module}...", total=100)
                progress_tasks[module] = task_id
            
            # 创建异步任务
            for module in modules:
                task = asyncio.create_task(
                    self._run_module_test_with_progress(module, progress, progress_tasks[module])
                )
                tasks.append(task)
            
            # 等待所有任务完成
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_sequential_tests(self, modules: List[str]) -> None:
        """顺序执行测试"""
        for i, module in enumerate(modules, 1):
            self.console.print(f"\n[bold cyan]═══ 测试模块 {i}/{len(modules)}: {module} ═══[/bold cyan]")
            await self._run_module_test(module)

    async def _run_module_test_with_progress(self, module: str, progress: Progress, task_id) -> None:
        """带进度显示的模块测试"""
        try:
            # 更新进度：开始测试
            progress.update(task_id, completed=10, description=f"[cyan]初始化 {module}...")
            
            validator = self.validators.get(module)
            if not validator:
                raise ValueError(f"未找到模块验证器: {module}")
            
            # 更新进度：运行测试
            progress.update(task_id, completed=30, description=f"[yellow]运行 {module} 测试...")
            
            module_start_time = time.time()
            test_results = []
            
            # 获取测试方法
            test_methods = [method for method in dir(validator) if method.startswith('test_')]
            total_methods = len(test_methods)
            
            for i, method_name in enumerate(test_methods):
                method = getattr(validator, method_name)
                
                # 更新子进度
                sub_progress = 30 + (i / total_methods) * 60
                progress.update(task_id, completed=sub_progress, 
                               description=f"[yellow]{module}: {method_name}...")
                
                # 执行测试方法
                result = await self._run_single_test(validator, method_name, method)
                test_results.append(result)
            
            # 更新进度：完成
            progress.update(task_id, completed=100, description=f"[green]✅ {module} 完成")
            
            # 生成模块报告
            module_duration = (time.time() - module_start_time) * 1000
            self.test_results[module] = self._generate_module_report(module, test_results, module_duration)
            
        except Exception as e:
            progress.update(task_id, completed=100, description=f"[red]❌ {module} 失败")
            
            # 创建错误报告
            error_result = ValidationResult(
                module_name=module,
                test_name="module_initialization",
                status="ERROR",
                duration_ms=0,
                message=str(e),
                details={"error_type": type(e).__name__},
                metrics={},
                timestamp=datetime.utcnow(),
                error_trace=traceback.format_exc()
            )
            
            self.test_results[module] = ModuleTestReport(
                module_name=module,
                total_tests=1,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                error_tests=1,
                total_duration_ms=0,
                results=[error_result],
                status="FAIL"
            )

    async def _run_module_test(self, module: str) -> None:
        """运行单个模块测试"""
        try:
            validator = self.validators.get(module)
            if not validator:
                self.console.print(f"[red]❌ 未找到模块验证器: {module}[/red]")
                return
            
            module_start_time = time.time()
            test_results = []
            
            # 获取所有测试方法
            test_methods = [method for method in dir(validator) if method.startswith('test_')]
            
            self.console.print(f"[dim]找到 {len(test_methods)} 个测试用例[/dim]")
            
            # 执行每个测试方法
            for method_name in test_methods:
                method = getattr(validator, method_name)
                result = await self._run_single_test(validator, method_name, method)
                test_results.append(result)
                
                # 实时显示结果
                self._show_test_result(result)
            
            # 生成模块报告
            module_duration = (time.time() - module_start_time) * 1000
            self.test_results[module] = self._generate_module_report(module, test_results, module_duration)
            
        except Exception as e:
            self.console.print(f"[red]❌ 模块测试失败: {e}[/red]")

    async def _run_single_test(self, validator, method_name: str, method: Callable) -> ValidationResult:
        """执行单个测试方法"""
        test_start_time = time.time()
        
        try:
            # 执行测试方法（支持同步和异步）
            if asyncio.iscoroutinefunction(method):
                result = await asyncio.wait_for(method(), timeout=self.module_timeout)
            else:
                # 在线程池中执行同步方法
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    result = await loop.run_in_executor(executor, method)
            
            test_duration = (time.time() - test_start_time) * 1000
            
            # 解析测试结果
            if isinstance(result, dict):
                status = result.get("status", "PASS")
                message = result.get("message", "测试通过")
                details = result.get("details", {})
                metrics = result.get("metrics", {})
            else:
                status = "PASS" if result else "FAIL"
                message = "测试通过" if result else "测试失败"
                details = {}
                metrics = {}
            
            return ValidationResult(
                module_name=validator.__class__.__name__.replace('Validator', '').lower(),
                test_name=method_name,
                status=status,
                duration_ms=test_duration,
                message=message,
                details=details,
                metrics=metrics,
                timestamp=datetime.utcnow()
            )
            
        except asyncio.TimeoutError:
            test_duration = (time.time() - test_start_time) * 1000
            return ValidationResult(
                module_name=validator.__class__.__name__.replace('Validator', '').lower(),
                test_name=method_name,
                status="FAIL",
                duration_ms=test_duration,
                message=f"测试超时（>{self.module_timeout}秒）",
                details={"timeout": self.module_timeout},
                metrics={},
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            test_duration = (time.time() - test_start_time) * 1000
            return ValidationResult(
                module_name=validator.__class__.__name__.replace('Validator', '').lower(),
                test_name=method_name,
                status="ERROR",
                duration_ms=test_duration,
                message=str(e),
                details={"error_type": type(e).__name__},
                metrics={},
                timestamp=datetime.utcnow(),
                error_trace=traceback.format_exc()
            )

    def _generate_module_report(self, module: str, results: List[ValidationResult], duration_ms: float) -> ModuleTestReport:
        """生成模块测试报告"""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.status == "PASS")
        failed_tests = sum(1 for r in results if r.status == "FAIL")
        skipped_tests = sum(1 for r in results if r.status == "SKIP")
        error_tests = sum(1 for r in results if r.status == "ERROR")
        
        # 确定整体状态
        if error_tests > 0 or failed_tests > 0:
            status = "FAIL"
        elif passed_tests == total_tests:
            status = "PASS"
        else:
            status = "PARTIAL"
        
        return ModuleTestReport(
            module_name=module,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            error_tests=error_tests,
            total_duration_ms=duration_ms,
            results=results,
            status=status
        )

    def _show_test_overview(self, modules: List[str]) -> None:
        """显示测试概览"""
        table = Table(title="[bold cyan]CLI模块验证测试概览[/bold cyan]", show_header=True)
        table.add_column("模块", style="cyan", no_wrap=True)
        table.add_column("验证器", style="magenta")
        table.add_column("测试数量", justify="center", style="green")
        table.add_column("预计耗时", justify="center", style="yellow")
        
        for module in modules:
            validator = self.validators.get(module)
            if validator:
                validator_name = validator.__class__.__name__
                test_count = len([m for m in dir(validator) if m.startswith('test_')])
                estimated_time = f"{test_count * 2}s"
            else:
                validator_name = "❌ 未找到"
                test_count = 0
                estimated_time = "0s"
            
            table.add_row(module, validator_name, str(test_count), estimated_time)
        
        self.console.print(table)
        self.console.print()

    def _show_test_result(self, result: ValidationResult) -> None:
        """显示单个测试结果"""
        status_icon = {
            "PASS": "✅",
            "FAIL": "❌", 
            "SKIP": "⏭️",
            "ERROR": "💥"
        }.get(result.status, "❓")
        
        status_color = {
            "PASS": "green",
            "FAIL": "red",
            "SKIP": "yellow",
            "ERROR": "magenta"
        }.get(result.status, "white")
        
        duration_color = "green" if result.duration_ms < 1000 else "yellow" if result.duration_ms < 5000 else "red"
        
        self.console.print(
            f"{status_icon} [{status_color}]{result.test_name}[/{status_color}] "
            f"[{duration_color}]{result.duration_ms:.1f}ms[/{duration_color}] "
            f"[dim]{result.message}[/dim]"
        )
        
        # 显示性能指标
        if result.metrics and self.verbose:
            for key, value in result.metrics.items():
                self.console.print(f"    📊 {key}: {value}")

    def _show_final_results(self) -> None:
        """显示最终测试结果"""
        if not self.test_results:
            self.console.print("[red]没有测试结果[/red]")
            return
        
        # 计算总体统计
        total_modules = len(self.test_results)
        passed_modules = sum(1 for report in self.test_results.values() if report.status == "PASS")
        failed_modules = sum(1 for report in self.test_results.values() if report.status == "FAIL")
        partial_modules = sum(1 for report in self.test_results.values() if report.status == "PARTIAL")
        
        total_tests = sum(report.total_tests for report in self.test_results.values())
        total_passed = sum(report.passed_tests for report in self.test_results.values())
        total_failed = sum(report.failed_tests for report in self.test_results.values())
        total_errors = sum(report.error_tests for report in self.test_results.values())
        
        total_duration = (self.end_time - self.start_time).total_seconds() if self.end_time and self.start_time else 0
        
        # 创建结果面板
        self.console.print("\n" + "="*80)
        
        # 总体状态
        if failed_modules == 0 and total_errors == 0:
            overall_status = "[bold green]✅ 全部通过[/bold green]"
        elif passed_modules > 0:
            overall_status = "[bold yellow]⚠️ 部分通过[/bold yellow]"
        else:
            overall_status = "[bold red]❌ 测试失败[/bold red]"
        
        summary_panel = Panel(
            f"""
{overall_status}

📊 [bold]模块统计[/bold]:
   • 总模块数: {total_modules}
   • 通过: [green]{passed_modules}[/green]
   • 失败: [red]{failed_modules}[/red]
   • 部分通过: [yellow]{partial_modules}[/yellow]

🧪 [bold]测试统计[/bold]:
   • 总测试数: {total_tests}
   • 通过: [green]{total_passed}[/green]
   • 失败: [red]{total_failed}[/red]
   • 错误: [magenta]{total_errors}[/magenta]

⏱️ [bold]性能统计[/bold]:
   • 总耗时: {total_duration:.2f}s
   • 平均每测试: {(total_duration*1000/max(1,total_tests)):.1f}ms
   • 测试速率: {(total_tests/max(0.1,total_duration)):.1f} tests/s
""".strip(),
            title="[bold cyan]CLI验证测试报告[/bold cyan]",
            border_style="cyan" if failed_modules == 0 else "red",
            padding=(1, 2)
        )
        
        self.console.print(summary_panel)
        
        # 详细模块结果表格
        self._show_detailed_results_table()
        
        # 性能分析
        self._show_performance_analysis()
        
        # 失败测试详情
        if total_failed > 0 or total_errors > 0:
            self._show_failure_details()

    def _show_detailed_results_table(self) -> None:
        """显示详细结果表格"""
        table = Table(title="[bold]模块详细结果[/bold]", show_header=True)
        table.add_column("模块", style="cyan", no_wrap=True)
        table.add_column("状态", justify="center")
        table.add_column("通过/总计", justify="center", style="green")
        table.add_column("耗时", justify="center", style="yellow") 
        table.add_column("成功率", justify="center", style="blue")
        table.add_column("平均响应", justify="center", style="magenta")
        
        for module, report in self.test_results.items():
            # 状态图标和颜色
            if report.status == "PASS":
                status = "[green]✅ 通过[/green]"
            elif report.status == "FAIL":
                status = "[red]❌ 失败[/red]"
            else:
                status = "[yellow]⚠️ 部分[/yellow]"
            
            # 通过率
            pass_rate = f"{report.passed_tests}/{report.total_tests}"
            
            # 成功率百分比
            success_rate = f"{(report.passed_tests/max(1,report.total_tests)*100):.1f}%"
            
            # 平均响应时间
            avg_response = f"{(report.total_duration_ms/max(1,report.total_tests)):.1f}ms"
            
            # 总耗时
            duration = f"{report.total_duration_ms:.0f}ms"
            
            table.add_row(module, status, pass_rate, duration, success_rate, avg_response)
        
        self.console.print("\n")
        self.console.print(table)

    def _show_performance_analysis(self) -> None:
        """显示性能分析"""
        self.console.print("\n[bold cyan]📈 性能分析[/bold cyan]")
        
        perf_issues = []
        perf_good = []
        
        for module, report in self.test_results.items():
            module_thresholds = self.performance_thresholds.get(module, {})
            
            for result in report.results:
                for metric_name, metric_value in result.metrics.items():
                    threshold = module_thresholds.get(metric_name)
                    if threshold:
                        if metric_value > threshold:
                            perf_issues.append(f"⚠️ {module}.{result.test_name}.{metric_name}: {metric_value:.1f}ms (阈值: {threshold}ms)")
                        else:
                            perf_good.append(f"✅ {module}.{result.test_name}.{metric_name}: {metric_value:.1f}ms")
        
        if perf_issues:
            self.console.print("[yellow]性能警告:[/yellow]")
            for issue in perf_issues:
                self.console.print(f"  {issue}")
        
        if perf_good and self.verbose:
            self.console.print("[green]性能良好:[/green]")
            for good in perf_good[:5]:  # 只显示前5个
                self.console.print(f"  {good}")

    def _show_failure_details(self) -> None:
        """显示失败测试详情"""
        self.console.print("\n[bold red]❌ 失败测试详情[/bold red]")
        
        failure_count = 0
        for module, report in self.test_results.items():
            for result in report.results:
                if result.status in ["FAIL", "ERROR"]:
                    failure_count += 1
                    
                    # 创建失败详情面板
                    details_content = f"""
[red]模块:[/red] {result.module_name}
[red]测试:[/red] {result.test_name}
[red]状态:[/red] {result.status}
[red]耗时:[/red] {result.duration_ms:.1f}ms
[red]错误信息:[/red] {result.message}
"""
                    
                    if result.details:
                        details_content += f"[red]详细信息:[/red] {json.dumps(result.details, indent=2)}\n"
                    
                    if result.error_trace and self.verbose:
                        details_content += f"[red]错误堆栈:[/red]\n{result.error_trace}"
                    
                    panel = Panel(
                        details_content.strip(),
                        title=f"[red]失败 #{failure_count}[/red]",
                        border_style="red",
                        padding=(0, 1)
                    )
                    
                    self.console.print(panel)
                    
                    if failure_count >= 5:  # 限制显示数量
                        remaining = sum(
                            len([r for r in rep.results if r.status in ["FAIL", "ERROR"]])
                            for rep in self.test_results.values()
                        ) - 5
                        if remaining > 0:
                            self.console.print(f"[dim]... 还有 {remaining} 个失败测试（使用 --verbose 查看全部）[/dim]")
                        break

    def generate_report(self, output_file: Optional[str] = None) -> str:
        """生成测试报告"""
        report_data = {
            "summary": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "total_duration_seconds": (
                    (self.end_time - self.start_time).total_seconds() 
                    if self.end_time and self.start_time else 0
                ),
                "total_modules": len(self.test_results),
                "passed_modules": sum(1 for r in self.test_results.values() if r.status == "PASS"),
                "failed_modules": sum(1 for r in self.test_results.values() if r.status == "FAIL"),
                "total_tests": sum(r.total_tests for r in self.test_results.values()),
                "passed_tests": sum(r.passed_tests for r in self.test_results.values()),
                "failed_tests": sum(r.failed_tests for r in self.test_results.values()),
                "error_tests": sum(r.error_tests for r in self.test_results.values()),
            },
            "modules": {}
        }
        
        # 添加模块详细信息
        for module, report in self.test_results.items():
            report_data["modules"][module] = {
                "status": report.status,
                "total_tests": report.total_tests,
                "passed_tests": report.passed_tests,
                "failed_tests": report.failed_tests,
                "error_tests": report.error_tests,
                "duration_ms": report.total_duration_ms,
                "results": [asdict(result) for result in report.results]
            }
        
        report_json = json.dumps(report_data, ensure_ascii=False, indent=2, default=str)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_json)
            self.console.print(f"[green]✅ 测试报告已保存到: {output_file}[/green]")
        
        return report_json

    async def run_continuous_validation(self, interval_seconds: int = 60) -> None:
        """连续验证模式"""
        self.console.print(f"[cyan]开始连续验证模式，间隔: {interval_seconds}秒[/cyan]")
        
        try:
            while True:
                self.console.print(f"\n[yellow]{'='*20} {datetime.now().strftime('%H:%M:%S')} {'='*20}[/yellow]")
                
                await self.run_all_validations()
                
                self.console.print(f"[dim]等待 {interval_seconds} 秒后继续...[/dim]")
                await asyncio.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            self.console.print("\n[yellow]连续验证已停止[/yellow]")

# ============ 命令行接口 ============

async def main():
    """主程序入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CLI模块验证测试器")
    parser.add_argument("--modules", nargs="+", help="指定要测试的模块")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--parallel", action="store_true", default=True, help="并行执行测试")
    parser.add_argument("--sequential", action="store_true", help="顺序执行测试")
    parser.add_argument("--output", "-o", help="输出报告文件路径")
    parser.add_argument("--continuous", "-c", type=int, metavar="SECONDS", help="连续验证模式")
    parser.add_argument("--timeout", type=int, default=300, help="总超时时间（秒）")
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = CLITester()
    tester.verbose = args.verbose
    tester.parallel = args.parallel and not args.sequential
    tester.timeout_seconds = args.timeout
    
    # 添加验证器
    tester.add_validator(RustEngineValidator())
    tester.add_validator(PythonLayerValidator())
    tester.add_validator(FastAPIValidator())
    tester.add_validator(DatabaseValidator())
    tester.add_validator(IntegrationValidator())
    
    # 执行测试
    if args.continuous:
        await tester.run_continuous_validation(args.continuous)
    else:
        await tester.run_all_validations(args.modules)
        
        # 生成报告
        if args.output:
            tester.generate_report(args.output)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]程序被用户中断[/yellow]")
    except Exception as e:
        console.print(f"\n[red]程序执行错误: {e}[/red]")
        sys.exit(1)