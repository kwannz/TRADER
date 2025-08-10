# Comprehensive Verbose Logging System

## Overview

The trader system now includes a comprehensive verbose logging system designed for debugging, testing, and monitoring. It extends the existing unified logger with detailed tracing, performance monitoring, and testing capabilities.

## Features

### 🔍 Verbose Logging
- **Function Entry/Exit Tracing**: Automatically log when functions are entered and exited
- **Parameter & Return Value Logging**: Log function parameters and return values 
- **Execution Time Tracking**: Measure and log function execution times
- **Memory Usage Monitoring**: Track memory consumption during execution
- **Call Stack Tracing**: Log call stacks for debugging complex execution flows

### 🛠️ Debug Tools
- **Debug Checkpoints**: Set checkpoints in code with contextual data
- **Variable Logging**: Log variable values during execution
- **Breakpoint System**: Set conditional breakpoints for debugging
- **Performance Profiling**: Profile code blocks and functions
- **System Monitoring**: Monitor CPU, memory, disk, and network usage

### 🧪 Testing Support
- **Test Context Management**: Manage test execution with automatic timing
- **Assertion Logging**: Log test assertions with pass/fail status
- **Test Report Generation**: Generate comprehensive test reports
- **Enhanced Test Runner**: Run test suites with verbose logging

### 🔐 Security Features
- **Sensitive Data Masking**: Automatically mask passwords, tokens, and API keys
- **Configurable Masking Patterns**: Customize which data patterns to mask
- **Data Truncation**: Limit log message sizes to prevent information leakage

## Quick Start

### Basic Usage

```python
from core.verbose_logger import get_verbose_logger, trace_execution
from core.debug_utils import checkpoint, log_variable

# Get the verbose logger instance
verbose_logger = get_verbose_logger()

# Use debug checkpoints
checkpoint("processing_start")
checkpoint("data_loaded", {"records": 1000})

# Log variables
user_id = "12345"
log_variable("user_id", user_id)

# Trace function execution
@trace_execution(log_params=True, log_returns=True)
def process_data(data):
    # Function will be automatically traced
    return data.upper()

result = process_data("hello world")
```

### Testing with Verbose Logging

```python
from core.verbose_logger import get_verbose_logger
from core.debug_utils import TestRunner

# Create test runner
test_runner = TestRunner()

class SampleTest:
    def test_basic_functionality(self):
        test_runner.assert_equals(2 + 2, 4, "basic_math")
        test_runner.assert_true(True, "boolean_test")
    
    def test_with_context(self):
        verbose_logger = get_verbose_logger()
        with verbose_logger.test_context("sample_test", {"param": "value"}):
            # Test code here
            verbose_logger.log_assertion("test_assertion", True)

# Run tests
results = test_runner.run_test_suite(SampleTest)
```

### Performance Profiling

```python
from core.debug_utils import debug_profiler, debug_profile

# Using context manager
with debug_profiler.profile("data_processing"):
    # Code to profile
    process_large_dataset()

# Using decorator
@debug_profile("expensive_operation")
def expensive_function():
    # This function will be profiled
    return calculate_complex_result()

# Get profiling results
stats = debug_profiler.get_profile_stats("data_processing")
print(f"Average duration: {stats['avg_duration_ms']:.2f}ms")
```

## Configuration

### Logging Configuration (config/logging_config.json)

The verbose logging system uses an enhanced configuration file:

```json
{
  "logging": {
    "level": "DEBUG",
    "verbose": true,
    "debug_mode": true,
    "testing_mode": false,
    "trace_execution": true,
    "profile_performance": true,
    "log_all_function_calls": true
  },
  "verbose_settings": {
    "log_entry_exit": true,
    "log_parameters": true,
    "log_return_values": true,
    "log_exceptions_detailed": true,
    "log_memory_usage": true,
    "log_execution_time": true,
    "log_thread_info": true,
    "log_call_stack": true,
    "max_parameter_length": 500,
    "max_return_value_length": 500,
    "sensitive_data_masking": true,
    "mask_patterns": ["password", "token", "key", "secret", "api_key"]
  },
  "debug_settings": {
    "trace_all_modules": false,
    "trace_specific_modules": ["core", "trading", "ai_engine", "strategy"],
    "exclude_modules": ["urllib", "requests", "asyncio", "logging"],
    "execution_flow_tracking": true
  },
  "testing_settings": {
    "log_test_execution": true,
    "log_assertions": true,
    "log_test_data": true,
    "log_setup_teardown": true,
    "test_timing": true
  }
}
```

## Advanced Features

### Function Tracing Decorators

```python
from core.verbose_logger import trace_execution, trace_async_execution
from core.unified_logger import LogCategory

# Sync function tracing
@trace_execution(category=LogCategory.TRADING, log_params=True, log_returns=True)
def execute_trade(symbol, quantity, price):
    # Function implementation
    return trade_result

# Async function tracing
@trace_async_execution(category=LogCategory.AI, log_params=True, log_returns=True)
async def analyze_market_data(data):
    # Async function implementation
    await process_data(data)
    return analysis_result
```

### System Monitoring

```python
from core.debug_utils import start_system_monitoring, stop_system_monitoring, system_monitor

# Start monitoring system resources
start_system_monitoring(interval=1.0)  # Monitor every second

# Your application code here...

# Get current system stats
stats = system_monitor.get_current_stats()
print(f"CPU: {stats['cpu_percent']}%")
print(f"Memory: {stats['memory']['percent']}%")

# Stop monitoring
stop_system_monitoring()

# Export monitoring data
system_monitor.export_metrics("logs/system_metrics.json")
```

### Debug Breakpoints

```python
from core.debug_utils import debug_breakpoint

# Set a simple breakpoint
debug_breakpoint.set_breakpoint("data_validation")

# Set a conditional breakpoint
def should_break():
    return error_count > 10

debug_breakpoint.set_breakpoint("error_threshold", should_break)

# Hit breakpoint during execution
if debug_breakpoint.hit_breakpoint("data_validation", {"record_id": 12345}):
    # Breakpoint was hit, do debugging actions
    pass
```

### Data Export and Analysis

```python
from core.debug_utils import get_debug_summary, export_all_debug_data

# Get comprehensive debug summary
summary = get_debug_summary()
print(f"Functions traced: {summary['function_stats']}")
print(f"Performance data: {summary['profiler_stats']}")

# Export all debug data
export_directory = export_all_debug_data("logs/debug_export")
print(f"Debug data exported to: {export_directory}")

# Export specific verbose logger data
verbose_logger = get_verbose_logger()
verbose_logger.export_debug_data("logs/verbose_debug.json")
```

## Log Levels and Categories

### Log Levels
- **TRACE**: Most detailed logging for deep debugging
- **DEBUG**: Debug information for development
- **INFO**: General information messages
- **SUCCESS**: Success operation messages
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages for handled exceptions
- **CRITICAL**: Critical errors requiring immediate attention
- **FATAL**: Fatal errors that cause application termination

### Log Categories
- **SYSTEM**: General system operations
- **API**: API requests and responses
- **TRADING**: Trading-related operations
- **AI**: AI engine operations
- **DATABASE**: Database operations
- **NETWORK**: Network communications
- **SECURITY**: Security-related events
- **PERFORMANCE**: Performance monitoring
- **USER**: User interactions
- **WORKFLOW**: Workflow execution
- **COINGLASS**: Coinglass data operations
- **BACKTEST**: Backtesting operations
- **STRATEGY**: Strategy execution
- **RISK**: Risk management
- **MONITORING**: System monitoring

## Log Output Formats

### Console Output
```
[2025-08-10 14:17:34.199] INFO     system       main:run:188                                   | 🚀 Starting AI量化交易系统 v1.0.0 [correlation_id=abc123 | duration=45.67ms | tags=startup,application]
```

### JSON Structured Output
```json
{
  "@timestamp": "2025-08-10T14:17:34.199Z",
  "level": "INFO",
  "category": "system",
  "message": "🚀 Starting AI量化交易系统 v1.0.0",
  "location": {
    "module": "main",
    "function": "run",
    "line": 188,
    "filename": "/path/to/main.py"
  },
  "metadata": {
    "correlation_id": "abc123",
    "user_id": null,
    "session_id": null
  },
  "performance": {
    "execution_time_ms": 45.67,
    "memory_usage_mb": 128.5
  },
  "tags": ["startup", "application"]
}
```

## Performance Impact

The verbose logging system is designed to have minimal performance impact:

- **Async Logging**: Log operations are performed asynchronously
- **Configurable Tracing**: Only trace specific modules to reduce overhead
- **Data Truncation**: Limit log message sizes to prevent memory issues
- **Lazy Evaluation**: Only format log messages when actually needed
- **Efficient Filtering**: Filter logs early to avoid unnecessary processing

## File Structure

```
logs/
├── system.log                    # General system logs
├── debug.log                     # Debug-specific logs
├── trace.log                     # Execution trace logs
├── execution_trace.log           # Function execution logs
├── errors.log                    # Error-only logs
├── system_structured.jsonl       # Structured JSON logs
├── debug_structured.jsonl        # Debug structured logs
├── performance_structured.jsonl  # Performance structured logs
└── debug_export/                 # Exported debug data
    ├── verbose_debug_20250810.json
    ├── system_metrics_20250810.json
    └── debug_summary_20250810.json
```

## Integration Examples

### In Main Application

```python
# main.py
from core.verbose_logger import setup_verbose_logging, log_startup_sequence
from core.debug_utils import start_system_monitoring

def main():
    # Initialize verbose logging
    setup_verbose_logging()
    
    # Log application startup
    log_startup_sequence("AI Trader", "1.0.0")
    
    # Start system monitoring
    start_system_monitoring()
    
    # Your application logic here...
    
if __name__ == "__main__":
    main()
```

### In Core Modules

```python
# core/trading_engine.py
from .verbose_logger import trace_execution, get_verbose_logger
from .debug_utils import checkpoint, log_variable
from .unified_logger import LogCategory

class TradingEngine:
    def __init__(self):
        self.verbose_logger = get_verbose_logger()
    
    @trace_execution(category=LogCategory.TRADING)
    def execute_order(self, order):
        checkpoint("order_received", {"order_id": order.id})
        
        # Process order
        result = self._process_order(order)
        
        log_variable("execution_result", result)
        checkpoint("order_completed")
        
        return result
```

### In Test Files

```python
# tests/test_trading.py
from core.debug_utils import TestRunner
from core.verbose_logger import get_verbose_logger

class TradingEngineTest:
    def setUp(self):
        self.test_runner = TestRunner()
        self.verbose_logger = get_verbose_logger()
    
    def test_order_execution(self):
        with self.verbose_logger.test_context("order_execution_test"):
            # Test implementation
            result = execute_test_order()
            self.test_runner.assert_equals(result.status, "executed")
```

## Troubleshooting

### Common Issues

1. **No logs appearing**: Check if log level is set correctly in configuration
2. **Performance issues**: Reduce tracing scope by configuring `trace_specific_modules`
3. **Large log files**: Enable log rotation and adjust file size limits
4. **Sensitive data leaking**: Verify `sensitive_data_masking` is enabled and patterns are configured

### Debug Commands

```python
# Check logging configuration
verbose_logger = get_verbose_logger()
print(verbose_logger.config)

# Get current statistics
stats = verbose_logger.get_performance_summary()
print(f"Total function calls: {stats['total_function_calls']}")

# Reset statistics
verbose_logger.reset_stats()

# Export debug data for analysis
verbose_logger.export_debug_data("debug_analysis.json")
```

## Best Practices

1. **Use appropriate log levels**: Don't log everything at DEBUG level
2. **Configure module-specific tracing**: Only trace modules you need to debug
3. **Mask sensitive data**: Always enable sensitive data masking in production
4. **Monitor performance impact**: Regularly check if logging affects application performance
5. **Export debug data regularly**: Export debug data for offline analysis
6. **Use test contexts**: Wrap test code in test contexts for better logging
7. **Set meaningful checkpoints**: Use descriptive names for debug checkpoints
8. **Clean up regularly**: Rotate and clean up log files to prevent disk space issues

## API Reference

### VerboseLogger Class

- `trace_function_entry(func_name, parameters, caller_info)`
- `trace_function_exit(func_name, return_value, duration_ms, caller_info)`
- `trace_exception(func_name, exception, duration_ms, caller_info)`
- `debug_checkpoint(checkpoint_name, data=None)`
- `log_assertion(assertion_name, condition, expected=None, actual=None)`
- `start_test(test_name, test_data=None)`
- `end_test(test_name, success, duration_ms=None)`
- `get_function_stats()`
- `get_debug_markers()`
- `get_performance_summary()`
- `export_debug_data(filepath)`

### Utility Functions

- `get_verbose_logger()`: Get global verbose logger instance
- `setup_verbose_logging(config_path)`: Setup verbose logging with config
- `trace_execution(category, log_params, log_returns)`: Function tracing decorator
- `trace_async_execution(category, log_params, log_returns)`: Async function tracing decorator
- `checkpoint(name, data)`: Quick debug checkpoint
- `log_variable(name, value, category)`: Log variable value
- `start_system_monitoring(interval)`: Start system monitoring
- `get_debug_summary()`: Get comprehensive debug summary
- `export_all_debug_data(directory)`: Export all debug data

This comprehensive verbose logging system provides everything needed for debugging, testing, and monitoring the trader application with detailed insights into system behavior and performance.