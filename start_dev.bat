@echo off
setlocal EnableDelayedExpansion

REM AI量化交易系统 - 开发环境启动脚本 (Windows)
title AI量化交易系统 - 开发环境

REM 设置项目根目录
set PROJECT_ROOT=%~dp0
cd /d "%PROJECT_ROOT%"

REM 颜色定义 (Windows 10+)
set RED=[91m
set GREEN=[92m
set YELLOW=[93m
set BLUE=[94m
set PURPLE=[95m
set CYAN=[96m
set NC=[0m

REM 显示标题
echo.
echo %CYAN%🚀 AI量化交易系统 - 开发环境启动器%NC%
echo %CYAN%==================================%NC%
echo.
echo %BLUE%📂 项目路径: %PROJECT_ROOT%%NC%
echo %BLUE%💻 系统: Windows%NC%
echo %BLUE%👤 用户: %USERNAME%%NC%
echo --------------------------------

REM 检查Python
echo %YELLOW%🔍 检查Python环境...%NC%

REM 尝试找到Python
python --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=python
    goto :check_version
)

python3 --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=python3
    goto :check_version
)

py --version >nul 2>&1
if %errorlevel% == 0 (
    set PYTHON_CMD=py
    goto :check_version
)

echo %RED%❌ 未找到Python解释器%NC%
echo %RED%请从 https://python.org 下载并安装Python 3.8+%NC%
echo %RED%安装时请勾选 "Add Python to PATH"%NC%
pause
exit /b 1

:check_version
REM 获取Python版本
for /f "tokens=2" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set PYTHON_VERSION=%%i
echo %GREEN%✅ Python版本: %PYTHON_VERSION%%NC%

REM 检查版本是否满足要求
%PYTHON_CMD% -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" 2>nul
if %errorlevel% neq 0 (
    echo %RED%❌ Python版本过低，需要3.8+%NC%
    pause
    exit /b 1
)
echo %GREEN%✅ Python版本符合要求 (>=3.8)%NC%

REM 检查依赖包
echo %YELLOW%📦 检查Python依赖包...%NC%

%PYTHON_CMD% -c "
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
"

if %errorlevel% neq 0 (
    echo %YELLOW%📦 发现缺失的依赖包%NC%
    echo.
    set /p INSTALL_DEPS="是否自动安装缺失的依赖包? (y/N): "
    
    if /i "!INSTALL_DEPS!" == "y" (
        echo %YELLOW%📦 正在安装依赖包...%NC%
        %PYTHON_CMD% -m pip install aiohttp watchdog ccxt pandas numpy websockets
        
        if !errorlevel! == 0 (
            echo %GREEN%✅ 依赖包安装成功!%NC%
        ) else (
            echo %RED%❌ 依赖包安装失败%NC%
            pause
            exit /b 1
        )
    ) else (
        echo %YELLOW%请手动安装依赖包:%NC%
        echo %YELLOW%  %PYTHON_CMD% -m pip install aiohttp watchdog ccxt pandas numpy websockets%NC%
        pause
        exit /b 1
    )
) else (
    echo %GREEN%✅ 所有Python依赖已满足%NC%
)

REM 检查项目文件
echo %YELLOW%🏗️ 检查项目结构...%NC%

set MISSING_FILES=0

if exist "dev_server.py" (
    echo   %GREEN%✅ dev_server.py%NC%
) else (
    echo   %YELLOW%⚠️ dev_server.py (可选)%NC%
    set /a MISSING_FILES+=1
)

if exist "server.py" (
    echo   %GREEN%✅ server.py%NC%
) else (
    echo   %YELLOW%⚠️ server.py (可选)%NC%
    set /a MISSING_FILES+=1
)

if exist "dev_client.js" (
    echo   %GREEN%✅ dev_client.js%NC%
) else (
    echo   %YELLOW%⚠️ dev_client.js (可选)%NC%
    set /a MISSING_FILES+=1
)

if exist "file_management\web_interface\index.html" (
    echo   %GREEN%✅ file_management\web_interface\index.html%NC%
) else (
    echo   %YELLOW%⚠️ file_management\web_interface\index.html (可选)%NC%
    set /a MISSING_FILES+=1
)

if exist "file_management\web_interface\app.js" (
    echo   %GREEN%✅ file_management\web_interface\app.js%NC%
) else (
    echo   %YELLOW%⚠️ file_management\web_interface\app.js (可选)%NC%
    set /a MISSING_FILES+=1
)

if %MISSING_FILES% gtr 0 (
    echo %YELLOW%⚠️ 注意: %MISSING_FILES% 个文件缺失，但不影响基本功能%NC%
)

REM 显示使用说明
echo.
echo %CYAN%==================================%NC%
echo %CYAN%🔧 开发环境使用说明%NC%
echo %CYAN%==================================%NC%
echo.
echo %GREEN%📖 功能特性:%NC%
echo   • 🔥 热重载: 修改代码自动刷新
echo   • 📱 实时预览: 浏览器自动更新
echo   • 🛠️ 开发工具: 完整的调试支持
echo   • 🔌 API测试: 内置开发API
echo.
echo %GREEN%🎯 操作指南:%NC%
echo   • 修改 .py 文件 → 后端自动重启
echo   • 修改 .html/.css/.js → 浏览器自动刷新
echo   • 查看控制台 → 实时开发日志
echo   • 按 Ctrl+C → 停止开发服务器
echo.
echo %GREEN%🌐 访问地址:%NC%
echo   • 📊 前端界面: http://localhost:8000
echo   • 🔧 开发状态: http://localhost:8000/api/dev/status
echo   • 📈 市场数据: http://localhost:8000/api/market
echo.
echo %GREEN%🛠️ 开发提示:%NC%
echo   • 页面左下角显示'开发模式'标识
echo   • 右上角显示代码更新通知
echo   • 开发者工具查看WebSocket连接状态
echo.
echo %CYAN%==================================%NC%

REM 等待用户确认
echo.
pause

REM 启动开发服务器
echo %GREEN%🚀 启动热重载开发环境...%NC%

if exist "dev_server.py" (
    echo %BLUE%📜 执行: %PYTHON_CMD% dev_server.py%NC%
    %PYTHON_CMD% dev_server.py
) else (
    echo %YELLOW%⚠️ dev_server.py 不存在，尝试启动增强模式...%NC%
    if exist "server.py" (
        echo %BLUE%📜 执行: %PYTHON_CMD% server.py --dev%NC%
        %PYTHON_CMD% server.py --dev
    ) else (
        echo %RED%❌ 找不到服务器脚本%NC%
        pause
        exit /b 1
    )
)

REM 清理
echo.
echo %YELLOW%🛑 开发服务器已停止%NC%
pause