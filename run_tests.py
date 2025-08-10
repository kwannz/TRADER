#!/usr/bin/env python3
"""
测试运行脚本
支持不同类型的测试和覆盖率检测
"""

import sys
import subprocess
import argparse
from pathlib import Path
import time
import json

class TestRunner:
    """测试运行器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_dir = self.project_root / 'tests'
        self.coverage_dir = self.test_dir / 'htmlcov'
        self.reports_dir = self.test_dir / 'reports'
        
        # 确保报告目录存在
        self.reports_dir.mkdir(exist_ok=True)
    
    def run_command(self, cmd, description=""):
        """运行命令并返回结果"""
        print(f"🚀 {description}")
        print(f"📜 执行: {' '.join(cmd)}")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                check=False
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ 成功 ({duration:.2f}s)")
                if result.stdout.strip():
                    print("📤 输出:")
                    print(result.stdout)
            else:
                print(f"❌ 失败 ({duration:.2f}s) - 退出码: {result.returncode}")
                if result.stderr.strip():
                    print("📤 错误:")
                    print(result.stderr)
                if result.stdout.strip():
                    print("📤 输出:")
                    print(result.stdout)
            
            return result
            
        except Exception as e:
            print(f"❌ 执行失败: {e}")
            return None
    
    def check_dependencies(self):
        """检查测试依赖"""
        print("🔍 检查测试依赖...")
        
        required_packages = ['pytest', 'coverage']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✅ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"  ❌ {package} (缺失)")
        
        if missing_packages:
            print(f"\n📦 需要安装缺失的包:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
        
        print("✅ 所有测试依赖已满足")
        return True
    
    def run_unit_tests(self, verbose=True, coverage=True):
        """运行单元测试"""
        cmd = ['python', '-m', 'pytest']
        
        if verbose:
            cmd.append('-v')
        
        if coverage:
            cmd.extend(['--cov=.', '--cov-report=html', '--cov-report=term-missing'])
        
        # 只运行单元测试
        cmd.extend(['tests/unit/', '-m', 'not slow'])
        
        return self.run_command(cmd, "运行单元测试")
    
    def run_integration_tests(self, verbose=True):
        """运行集成测试"""
        cmd = ['python', '-m', 'pytest']
        
        if verbose:
            cmd.append('-v')
        
        # 运行集成测试
        cmd.extend(['tests/integration/', '-m', 'not slow'])
        
        return self.run_command(cmd, "运行集成测试")
    
    def run_all_tests(self, verbose=True, coverage=True, fail_under=80):
        """运行所有测试"""
        cmd = ['python', '-m', 'pytest']
        
        if verbose:
            cmd.extend(['-v', '--tb=short'])
        
        if coverage:
            cmd.extend([
                '--cov=.',
                '--cov-report=html',
                '--cov-report=term-missing',
                '--cov-report=xml',
                f'--cov-fail-under={fail_under}'
            ])
        
        # 运行所有测试，排除慢速测试
        cmd.extend(['tests/', '-m', 'not slow'])
        
        return self.run_command(cmd, f"运行所有测试 (目标覆盖率: {fail_under}%)")
    
    def run_coverage_only(self):
        """只运行覆盖率检测"""
        cmd = ['python', '-m', 'coverage', 'run', '-m', 'pytest', 'tests/']
        result1 = self.run_command(cmd, "运行覆盖率收集")
        
        if result1 and result1.returncode == 0:
            # 生成覆盖率报告
            cmd = ['python', '-m', 'coverage', 'report', '-m']
            result2 = self.run_command(cmd, "生成覆盖率报告")
            
            # 生成HTML报告
            cmd = ['python', '-m', 'coverage', 'html']
            result3 = self.run_command(cmd, "生成HTML覆盖率报告")
            
            return all(r and r.returncode == 0 for r in [result1, result2, result3])
        
        return False
    
    def run_specific_tests(self, test_pattern, verbose=True):
        """运行特定测试"""
        cmd = ['python', '-m', 'pytest']
        
        if verbose:
            cmd.append('-v')
        
        cmd.extend(['-k', test_pattern])
        
        return self.run_command(cmd, f"运行匹配 '{test_pattern}' 的测试")
    
    def generate_test_report(self):
        """生成测试报告"""
        print("📊 生成测试报告...")
        
        # 运行测试并生成JSON报告
        cmd = [
            'python', '-m', 'pytest',
            '--json-report',
            f'--json-report-file={self.reports_dir}/test-report.json',
            'tests/'
        ]
        
        result = self.run_command(cmd, "生成JSON测试报告")
        
        if result and result.returncode == 0:
            report_path = self.reports_dir / 'test-report.json'
            if report_path.exists():
                print(f"📄 测试报告已生成: {report_path}")
                
                # 显示报告摘要
                try:
                    with open(report_path) as f:
                        report_data = json.load(f)
                    
                    summary = report_data.get('summary', {})
                    print("\n📈 测试摘要:")
                    print(f"  总计: {summary.get('total', 0)}")
                    print(f"  通过: {summary.get('passed', 0)}")
                    print(f"  失败: {summary.get('failed', 0)}")
                    print(f"  跳过: {summary.get('skipped', 0)}")
                    print(f"  耗时: {summary.get('duration', 0):.2f}s")
                    
                except Exception as e:
                    print(f"⚠️ 无法解析报告: {e}")
        
        return result and result.returncode == 0
    
    def clean_reports(self):
        """清理测试报告"""
        print("🧹 清理测试报告...")
        
        import shutil
        
        dirs_to_clean = [
            self.coverage_dir,
            self.reports_dir,
            self.project_root / '.pytest_cache',
            self.project_root / '.coverage'
        ]
        
        for dir_path in dirs_to_clean:
            if dir_path.exists():
                if dir_path.is_dir():
                    shutil.rmtree(dir_path)
                    print(f"  🗑️ 已删除目录: {dir_path}")
                else:
                    dir_path.unlink()
                    print(f"  🗑️ 已删除文件: {dir_path}")
        
        print("✅ 清理完成")
    
    def show_coverage_summary(self):
        """显示覆盖率摘要"""
        coverage_file = self.project_root / '.coverage'
        
        if not coverage_file.exists():
            print("❌ 未找到覆盖率数据，请先运行测试")
            return False
        
        cmd = ['python', '-m', 'coverage', 'report']
        result = self.run_command(cmd, "显示覆盖率摘要")
        
        return result and result.returncode == 0

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='AI量化交易系统测试运行器')
    
    parser.add_argument('--type', choices=['unit', 'integration', 'all', 'coverage'], 
                       default='all', help='测试类型')
    parser.add_argument('--pattern', help='测试模式匹配')
    parser.add_argument('--coverage', action='store_true', default=True, help='启用覆盖率检测')
    parser.add_argument('--no-coverage', dest='coverage', action='store_false', help='禁用覆盖率检测')
    parser.add_argument('--fail-under', type=int, default=80, help='最低覆盖率要求')
    parser.add_argument('--verbose', '-v', action='store_true', default=True, help='详细输出')
    parser.add_argument('--quiet', '-q', dest='verbose', action='store_false', help='静默模式')
    parser.add_argument('--clean', action='store_true', help='清理测试报告')
    parser.add_argument('--report', action='store_true', help='生成测试报告')
    parser.add_argument('--summary', action='store_true', help='显示覆盖率摘要')
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    print("🧪 AI量化交易系统 - 测试运行器")
    print("=" * 50)
    
    # 清理操作
    if args.clean:
        runner.clean_reports()
        return
    
    # 显示摘要
    if args.summary:
        runner.show_coverage_summary()
        return
    
    # 检查依赖
    if not runner.check_dependencies():
        sys.exit(1)
    
    success = False
    
    # 运行测试
    if args.pattern:
        # 运行特定模式的测试
        result = runner.run_specific_tests(args.pattern, args.verbose)
        success = result and result.returncode == 0
        
    elif args.type == 'unit':
        # 单元测试
        result = runner.run_unit_tests(args.verbose, args.coverage)
        success = result and result.returncode == 0
        
    elif args.type == 'integration':
        # 集成测试
        result = runner.run_integration_tests(args.verbose)
        success = result and result.returncode == 0
        
    elif args.type == 'coverage':
        # 仅覆盖率检测
        success = runner.run_coverage_only()
        
    else:
        # 运行所有测试
        result = runner.run_all_tests(args.verbose, args.coverage, args.fail_under)
        success = result and result.returncode == 0
    
    # 生成报告
    if args.report:
        runner.generate_test_report()
    
    # 最终结果
    print("\n" + "=" * 50)
    if success:
        print("🎉 测试运行成功！")
        if args.coverage and runner.coverage_dir.exists():
            print(f"📊 HTML覆盖率报告: {runner.coverage_dir}/index.html")
    else:
        print("❌ 测试运行失败！")
        sys.exit(1)

if __name__ == '__main__':
    main()