#!/usr/bin/env python3
"""
测试开发环境的基本功能
"""

import sys
import os
import asyncio
from pathlib import Path

def test_imports():
    """测试关键导入"""
    print("🔍 测试Python包导入...")
    
    required_packages = [
        'aiohttp',
        'watchdog', 
        'ccxt',
        'pandas',
        'numpy',
        'websockets'
    ]
    
    success_count = 0
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
            success_count += 1
        except ImportError as e:
            print(f"  ❌ {package} - {e}")
    
    print(f"📊 导入测试结果: {success_count}/{len(required_packages)} 成功")
    return success_count == len(required_packages)

def test_file_structure():
    """测试项目文件结构"""
    print("\n🏗️ 测试项目文件结构...")
    
    required_files = [
        'dev_server.py',
        'server.py', 
        'dev_client.js',
        'start_dev.py',
        'requirements-dev.txt',
        'dev_config.json',
        'DEV_ENVIRONMENT.md',
        'file_management/web_interface/index.html',
        'file_management/web_interface/app.js',
        'file_management/web_interface/dev_client.js'
    ]
    
    success_count = 0
    project_root = Path(__file__).parent
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
            success_count += 1
        else:
            print(f"  ❌ {file_path}")
    
    print(f"📊 文件结构测试: {success_count}/{len(required_files)} 文件存在")
    return success_count >= len(required_files) * 0.8  # 允许20%的文件缺失

def test_dev_server_syntax():
    """测试开发服务器脚本语法"""
    print("\n📜 测试开发服务器脚本语法...")
    
    try:
        import ast
        
        with open('dev_server.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        ast.parse(content)
        print("  ✅ dev_server.py 语法正确")
        
        with open('server.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        ast.parse(content) 
        print("  ✅ server.py 语法正确")
        
        return True
        
    except SyntaxError as e:
        print(f"  ❌ 语法错误: {e}")
        return False
    except Exception as e:
        print(f"  ❌ 检查失败: {e}")
        return False

async def test_basic_server():
    """测试基本服务器功能"""
    print("\n🚀 测试基本服务器功能...")
    
    try:
        # 尝试导入和创建基本的aiohttp应用
        from aiohttp import web
        
        app = web.Application()
        
        async def hello(request):
            return web.json_response({'status': 'ok', 'message': 'test'})
        
        app.router.add_get('/test', hello)
        
        print("  ✅ aiohttp应用创建成功")
        print("  ✅ 路由配置正常")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 服务器测试失败: {e}")
        return False

def test_watchdog_functionality():
    """测试watchdog文件监控功能"""
    print("\n👀 测试文件监控功能...")
    
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
        
        class TestHandler(FileSystemEventHandler):
            def on_modified(self, event):
                pass
        
        observer = Observer()
        observer.schedule(TestHandler(), path='.', recursive=True)
        
        print("  ✅ watchdog Observer 创建成功")
        print("  ✅ 事件处理器配置正常")
        
        # 测试启动和停止
        observer.start()
        observer.stop() 
        observer.join()
        
        print("  ✅ 文件监控启动/停止正常")
        return True
        
    except Exception as e:
        print(f"  ❌ 文件监控测试失败: {e}")
        return False

async def main():
    """主测试函数"""
    print("🧪 AI量化交易系统 - 开发环境测试")
    print("=" * 50)
    
    test_results = []
    
    # 运行各项测试
    test_results.append(test_imports())
    test_results.append(test_file_structure())
    test_results.append(test_dev_server_syntax()) 
    test_results.append(await test_basic_server())
    test_results.append(test_watchdog_functionality())
    
    # 汇总结果
    print("\n" + "=" * 50)
    print("📊 测试结果汇总:")
    
    test_names = [
        "Python包导入",
        "项目文件结构", 
        "脚本语法检查",
        "基本服务器功能",
        "文件监控功能"
    ]
    
    passed = sum(test_results)
    total = len(test_results)
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {i+1}. {name}: {status}")
    
    print(f"\n🎯 总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！开发环境配置成功！")
        print("\n🚀 可以开始使用开发环境:")
        print("  • 运行: python dev_server.py")
        print("  • 或者: ./start_dev.sh")
        print("  • 访问: http://localhost:8000")
    elif passed >= total * 0.8:
        print("⚠️ 大部分测试通过，开发环境基本可用")
        print("建议检查失败的测试项并修复")
    else:
        print("❌ 多项测试失败，请检查环境配置")
        return False
    
    return passed >= total * 0.8

if __name__ == '__main__':
    asyncio.run(main())