#!/bin/bash

# AI量化交易系统 - 开发环境启动脚本 (Unix/Linux/macOS)

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# 显示带颜色的消息
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# 显示标题
print_title() {
    echo
    print_message $CYAN "🚀 AI量化交易系统 - 开发环境启动器"
    print_message $CYAN "=================================="
    echo
    print_message $BLUE "📂 项目路径: $PROJECT_ROOT"
    print_message $BLUE "🐍 Python: $(which python3 2>/dev/null || which python || echo '未找到')"
    print_message $BLUE "💻 系统: $(uname -s) $(uname -r)"
    print_message $BLUE "👤 用户: $(whoami)"
    echo "--------------------------------"
}

# 检查Python
check_python() {
    print_message $YELLOW "🔍 检查Python环境..."
    
    # 尝试找到Python3
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_message $RED "❌ 未找到Python解释器"
        print_message $RED "请安装Python 3.8+:"
        print_message $RED "  macOS: brew install python3"
        print_message $RED "  Ubuntu: sudo apt install python3 python3-pip"
        print_message $RED "  CentOS: sudo yum install python3 python3-pip"
        exit 1
    fi
    
    # 检查Python版本
    PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_message $GREEN "✅ Python版本: $PYTHON_VERSION"
    
    # 检查版本是否满足要求
    if $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
        print_message $GREEN "✅ Python版本符合要求 (>=3.8)"
    else
        print_message $RED "❌ Python版本过低，需要3.8+"
        exit 1
    fi
}

# 检查依赖
check_dependencies() {
    print_message $YELLOW "📦 检查Python依赖包..."
    
    # 运行Python依赖检查脚本
    if $PYTHON_CMD -c "
import sys
required_packages = ['aiohttp', 'watchdog', 'ccxt', 'pandas', 'numpy', 'websockets']
missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        missing.append(pkg)
        print(f'❌ {pkg} (缺失)')

if missing:
    print(f'\n需要安装 {len(missing)} 个依赖包')
    sys.exit(1)
else:
    print('✅ 所有依赖包已安装')
"; then
        print_message $GREEN "✅ 所有Python依赖已满足"
    else
        print_message $YELLOW "📦 发现缺失的依赖包"
        
        # 询问是否自动安装
        read -p "是否自动安装缺失的依赖包? (y/N): " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_message $YELLOW "📦 正在安装依赖包..."
            
            if $PYTHON_CMD -m pip install aiohttp watchdog ccxt pandas numpy websockets; then
                print_message $GREEN "✅ 依赖包安装成功!"
            else
                print_message $RED "❌ 依赖包安装失败"
                exit 1
            fi
        else
            print_message $YELLOW "请手动安装依赖包:"
            print_message $YELLOW "  $PYTHON_CMD -m pip install aiohttp watchdog ccxt pandas numpy websockets"
            exit 1
        fi
    fi
}

# 检查项目文件
check_project_files() {
    print_message $YELLOW "🏗️ 检查项目结构..."
    
    local files=(
        "dev_server.py"
        "server.py" 
        "dev_client.js"
        "file_management/web_interface/index.html"
        "file_management/web_interface/app.js"
    )
    
    local missing_files=0
    
    for file in "${files[@]}"; do
        if [[ -f "$file" ]]; then
            print_message $GREEN "  ✅ $file"
        else
            print_message $YELLOW "  ⚠️ $file (可选)"
            ((missing_files++))
        fi
    done
    
    if [[ $missing_files -gt 0 ]]; then
        print_message $YELLOW "⚠️ 注意: $missing_files 个文件缺失，但不影响基本功能"
    fi
}

# 显示使用说明
show_usage() {
    echo
    print_message $CYAN "=================================="
    print_message $CYAN "🔧 开发环境使用说明"
    print_message $CYAN "=================================="
    echo
    print_message $GREEN "📖 功能特性:"
    echo "  • 🔥 热重载: 修改代码自动刷新"
    echo "  • 📱 实时预览: 浏览器自动更新"
    echo "  • 🛠️ 开发工具: 完整的调试支持"
    echo "  • 🔌 API测试: 内置开发API"
    echo
    print_message $GREEN "🎯 操作指南:"
    echo "  • 修改 .py 文件 → 后端自动重启"
    echo "  • 修改 .html/.css/.js → 浏览器自动刷新"  
    echo "  • 查看控制台 → 实时开发日志"
    echo "  • 按 Ctrl+C → 停止开发服务器"
    echo
    print_message $GREEN "🌐 访问地址:"
    echo "  • 📊 前端界面: http://localhost:8000"
    echo "  • 🔧 开发状态: http://localhost:8000/api/dev/status"
    echo "  • 📈 市场数据: http://localhost:8000/api/market"
    echo
    print_message $GREEN "🛠️ 开发提示:"
    echo "  • 页面左下角显示'开发模式'标识"
    echo "  • 右上角显示代码更新通知"
    echo "  • 开发者工具查看WebSocket连接状态"
    echo
    print_message $CYAN "=================================="
}

# 启动开发服务器
start_dev_server() {
    local mode=${1:-"hot"}
    
    print_message $GREEN "🚀 启动开发环境 ($mode 模式)..."
    
    case $mode in
        "hot")
            if [[ -f "dev_server.py" ]]; then
                print_message $BLUE "📜 执行: $PYTHON_CMD dev_server.py"
                $PYTHON_CMD dev_server.py
            else
                print_message $RED "❌ dev_server.py 文件不存在"
                exit 1
            fi
            ;;
        "enhanced")
            if [[ -f "server.py" ]]; then
                print_message $BLUE "📜 执行: $PYTHON_CMD server.py --dev"
                $PYTHON_CMD server.py --dev
            else
                print_message $RED "❌ server.py 文件不存在"
                exit 1
            fi
            ;;
        *)
            print_message $RED "❌ 未知模式: $mode"
            print_message $YELLOW "可用模式: hot, enhanced"
            exit 1
            ;;
    esac
}

# 清理函数
cleanup() {
    echo
    print_message $YELLOW "🛑 正在停止开发服务器..."
    print_message $GREEN "✅ 开发服务器已停止"
}

# 设置信号处理
trap cleanup SIGINT SIGTERM

# 主函数
main() {
    # 解析命令行参数
    local mode="hot"
    local skip_deps=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                mode="$2"
                shift 2
                ;;
            --skip-deps)
                skip_deps=true
                shift
                ;;
            --help|-h)
                echo "用法: $0 [选项]"
                echo "选项:"
                echo "  --mode MODE      启动模式: hot, enhanced (默认: hot)"
                echo "  --skip-deps      跳过依赖检查"  
                echo "  --help|-h        显示帮助"
                exit 0
                ;;
            *)
                print_message $RED "未知参数: $1"
                exit 1
                ;;
        esac
    done
    
    # 显示标题
    print_title
    
    # 检查环境
    check_python
    
    if [[ $skip_deps != true ]]; then
        check_dependencies
    else
        print_message $YELLOW "⚠️ 已跳过依赖检查"
    fi
    
    check_project_files
    
    # 显示使用说明
    show_usage
    
    # 等待用户确认
    echo
    read -p "按 Enter 键启动开发服务器 (或 Ctrl+C 退出)..."
    
    # 启动开发服务器
    start_dev_server "$mode"
}

# 运行主函数
main "$@"