# AI量化交易系统 - 测试和开发命令

.PHONY: help test test-unit test-integration test-coverage test-fast clean-test install-deps dev start-dev

# 默认目标
help:
	@echo "🚀 AI量化交易系统 - 可用命令:"
	@echo ""
	@echo "📦 依赖管理:"
	@echo "  install-deps     安装所有依赖"
	@echo "  install-dev      安装开发依赖"
	@echo "  install-test     安装测试依赖"
	@echo ""
	@echo "🧪 测试命令:"
	@echo "  test            运行所有测试 (80%覆盖率)"
	@echo "  test-unit       运行单元测试"
	@echo "  test-integration 运行集成测试"
	@echo "  test-coverage   运行覆盖率检测"
	@echo "  test-fast       快速测试 (跳过慢速测试)"
	@echo "  test-report     生成测试报告"
	@echo ""
	@echo "🔧 开发命令:"
	@echo "  dev             启动热重载开发服务器"
	@echo "  start-dev       启动开发环境 (使用Python脚本)"
	@echo "  check-env       检查开发环境"
	@echo ""
	@echo "🧹 清理命令:"
	@echo "  clean           清理所有生成的文件"
	@echo "  clean-test      清理测试文件"
	@echo "  clean-cache     清理Python缓存"

# 依赖安装
install-deps:
	@echo "📦 安装项目依赖..."
	python -m pip install --break-system-packages -r requirements-dev.txt

install-dev:
	@echo "📦 安装开发依赖..."
	python -m pip install --break-system-packages -r requirements-dev.txt

install-test:
	@echo "📦 安装测试依赖..."
	python -m pip install --break-system-packages pytest pytest-asyncio pytest-cov coverage

# 测试命令
test: install-test
	@echo "🧪 运行所有测试 (80%覆盖率要求)..."
	python run_tests.py --type all --fail-under 80

test-unit: install-test
	@echo "🧪 运行单元测试..."
	python run_tests.py --type unit

test-integration: install-test
	@echo "🧪 运行集成测试..."
	python run_tests.py --type integration

test-coverage: install-test
	@echo "📊 运行覆盖率检测..."
	python run_tests.py --type coverage

test-fast: install-test
	@echo "⚡ 快速测试 (跳过慢速测试)..."
	python -m pytest tests/ -m "not slow" -x --tb=short

test-report: install-test
	@echo "📄 生成测试报告..."
	python run_tests.py --report

test-summary:
	@echo "📊 显示覆盖率摘要..."
	python run_tests.py --summary

# 开发命令
dev: install-dev
	@echo "🔥 启动热重载开发服务器..."
	python dev_server.py

start-dev: install-dev
	@echo "🚀 启动开发环境..."
	python start_dev.py

check-env:
	@echo "🔍 检查开发环境..."
	python test_dev_env.py

server:
	@echo "🌐 启动生产服务器..."
	python server.py

server-dev:
	@echo "🔧 启动开发模式服务器..."
	python server.py --dev

# 代码质量检查 (如果可用)
lint:
	@echo "🔍 代码风格检查..."
	-python -m flake8 . --max-line-length=100 --ignore=E203,W503 || echo "⚠️ flake8 not installed"

format:
	@echo "🎨 代码格式化..."
	-python -m black . --line-length=100 || echo "⚠️ black not installed"

# 清理命令
clean: clean-test clean-cache
	@echo "🧹 清理完成"

clean-test:
	@echo "🧹 清理测试文件..."
	python run_tests.py --clean

clean-cache:
	@echo "🧹 清理Python缓存..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

# Docker相关 (可选)
docker-build:
	@echo "🐳 构建Docker镜像..."
	docker build -t ai-trader:dev .

docker-run:
	@echo "🐳 运行Docker容器..."
	docker run -p 8000:8000 ai-trader:dev

# 文档生成 (可选)
docs:
	@echo "📚 生成文档..."
	@echo "文档功能待实现"

# 数据库相关 (如果需要)
db-setup:
	@echo "🗄️ 设置数据库..."
	@echo "数据库设置功能待实现"

# 安全检查 (可选)
security-check:
	@echo "🔐 安全检查..."
	-python -m bandit -r . -x tests/ || echo "⚠️ bandit not installed"

# 性能测试 (可选)
benchmark:
	@echo "⚡ 性能测试..."
	python -m pytest tests/ -m "benchmark" --benchmark-only

# 查看测试覆盖率
coverage-html: test-coverage
	@echo "🌐 打开HTML覆盖率报告..."
	@if [ -f "tests/htmlcov/index.html" ]; then \
		echo "📊 覆盖率报告: tests/htmlcov/index.html"; \
		python -c "import webbrowser; webbrowser.open('tests/htmlcov/index.html')" 2>/dev/null || echo "请手动打开 tests/htmlcov/index.html"; \
	else \
		echo "❌ HTML覆盖率报告不存在，请先运行 make test-coverage"; \
	fi

# CI/CD 相关
ci: clean install-test test
	@echo "🤖 CI流水线完成"

# 全面检查
full-check: clean install-deps lint test-coverage
	@echo "✅ 全面检查完成"