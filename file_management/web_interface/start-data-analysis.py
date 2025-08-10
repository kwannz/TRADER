#!/usr/bin/env python3
"""
AI量化数据分析平台 - 一键启动脚本
同时启动前端服务器和数据分析API服务器
"""

import os
import sys
import time
import threading
import webbrowser
from pathlib import Path

def start_frontend_server():
    """启动前端服务器"""
    print("🌐 启动前端服务器...")
    os.system("python -m http.server 8080")

def start_api_server():
    """启动API服务器"""
    print("🔗 启动数据分析API服务器...")
    os.system("python data-analysis-api.py")

def main():
    """主函数"""
    # 切换到脚本目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("\n" + "="*70)
    print("🚀 AI量化数据分析平台启动器")
    print("="*70)
    print("📊 专注于数据分析、因子研究和AI生成")
    print("-"*70)
    
    # 启动API服务器
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()
    
    # 等待API服务器启动
    print("⏳ 等待API服务器启动...")
    time.sleep(3)
    
    # 启动前端服务器
    frontend_thread = threading.Thread(target=start_frontend_server, daemon=True)
    frontend_thread.start()
    
    # 等待前端服务器启动
    print("⏳ 等待前端服务器启动...")
    time.sleep(2)
    
    # 显示启动信息
    print("\n✅ 所有服务启动完成!")
    print("-"*70)
    print("🌐 前端服务: http://localhost:8080")
    print("📊 数据分析界面: http://localhost:8080/data-analysis-index.html")
    print("🔗 API服务: http://localhost:8002/api/v1/health")
    print("-"*70)
    print("🎯 主要功能:")
    print("  ✓ 数据概览 - 实时数据源状态和统计")
    print("  ✓ 因子研究 - 自定义因子开发环境")
    print("  ✓ AI因子生成 - DeepSeek/Gemini AI生成因子")
    print("  ✓ 回测实验室 - 因子回测和评估")
    print("  ✓ 因子库 - 因子存储和管理")
    print("  ✓ 数据源管理 - 数据源配置和监控")
    print("  ✓ 分析报告 - 自动生成分析报告")
    print("  ✓ 系统配置 - API密钥和参数设置")
    print("-"*70)
    print("🎨 界面特色:")
    print("  • Bloomberg风格专业界面")
    print("  • 实时数据流显示")
    print("  • 交互式图表和热力图")
    print("  • AI因子评分和建议")
    print("  • 响应式移动端适配")
    print("-"*70)
    print("💡 使用提示:")
    print("  - 使用快捷键 Ctrl/Cmd + 1-8 切换页面")
    print("  - 所有数据会自动刷新和更新")
    print("  - AI生成需要API密钥配置")
    print("  - 支持导出因子和报告")
    print("="*70)
    
    # 自动打开浏览器
    try:
        print("🔄 正在自动打开浏览器...")
        time.sleep(1)
        webbrowser.open('http://localhost:8080/data-analysis-index.html')
    except:
        pass
    
    print("\n📱 请在浏览器中访问: http://localhost:8080/data-analysis-index.html")
    print("🛑 按 Ctrl+C 停止所有服务")
    print("="*70)
    
    try:
        # 保持主线程运行
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 正在关闭所有服务...")
        print("✅ 数据分析平台已停止")
        sys.exit(0)

if __name__ == '__main__':
    main()