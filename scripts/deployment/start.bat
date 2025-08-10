@echo off
chcp 65001 >nul

echo 🚀 AI量化交易系统启动中...
echo ==================================

REM 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未找到Python，请先安装Python
    pause
    exit /b 1
)

REM 检查必要文件
set missing_files=
if not exist "index.html" set missing_files=%missing_files% index.html
if not exist "styles.css" set missing_files=%missing_files% styles.css
if not exist "app.js" set missing_files=%missing_files% app.js
if not exist "server.py" set missing_files=%missing_files% server.py

if not "%missing_files%"=="" (
    echo ❌ 缺少必要文件: %missing_files%
    pause
    exit /b 1
)

echo ✅ 文件检查完成
echo 🌐 启动前端服务器...
echo.

REM 启动服务器
python server.py %*

pause