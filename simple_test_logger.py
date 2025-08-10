#!/usr/bin/env python3
"""
Simple test script to verify verbose logging functionality
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.verbose_logger import (
    get_verbose_logger, trace_execution, trace_async_execution,
    log_startup_sequence, log_api_request
)
from core.debug_utils import (
    debug_profile, debug_trace, checkpoint, log_variable,
    debug_profiler, get_debug_summary
)
from core.unified_logger import LogCategory
import asyncio

def main():
    print("🧪 Simple Verbose Logger Test")
    print("=" * 40)
    
    # Initialize verbose logger
    verbose_logger = get_verbose_logger()
    
    # Test 1: Basic logging
    print("\n1. Testing basic logging...")
    log_startup_sequence("TestApp", "1.0.0")
    log_api_request("GET", "/api/test", 200, 123.45)
    
    # Test 2: Debug checkpoint
    print("\n2. Testing debug checkpoints...")
    checkpoint("test_start")
    checkpoint("test_data", {"key": "value", "count": 42})
    
    # Test 3: Variable logging
    print("\n3. Testing variable logging...")
    test_var = "Hello World"
    log_variable("test_var", test_var)
    
    # Test 4: Function tracing
    print("\n4. Testing function tracing...")
    
    @trace_execution(log_params=True, log_returns=True)
    def sample_function(x: int, y: str = "default") -> str:
        time.sleep(0.01)  # Simulate work
        return f"Result: {x} - {y}"
    
    result = sample_function(10, "test")
    print(f"Function returned: {result}")
    
    # Test 5: Debug profiling
    print("\n5. Testing debug profiling...")
    
    @debug_profile("profile_test")
    def profiled_function(iterations: int) -> int:
        total = 0
        for i in range(iterations):
            total += i * i
        return total
    
    result = profiled_function(1000)
    print(f"Profiled function returned: {result}")
    
    # Test 6: Async function tracing
    print("\n6. Testing async function tracing...")
    
    @trace_async_execution(log_params=True, log_returns=True)
    async def async_sample_function(delay: float, message: str) -> dict:
        await asyncio.sleep(delay)
        return {"message": message.upper(), "delay": delay}
    
    async_result = asyncio.run(async_sample_function(0.01, "hello world"))
    print(f"Async function returned: {async_result}")
    
    # Test 7: Error handling
    print("\n7. Testing error handling...")
    
    @trace_execution()
    def error_function():
        raise ValueError("Test error for logging")
    
    try:
        error_function()
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    # Test 8: Get statistics
    print("\n8. Getting performance statistics...")
    stats = verbose_logger.get_function_stats()
    print(f"Functions traced: {len(stats)}")
    
    perf_summary = verbose_logger.get_performance_summary()
    print(f"Total function calls: {perf_summary['total_function_calls']}")
    print(f"Total errors: {perf_summary['total_errors']}")
    
    profiler_stats = debug_profiler.get_all_stats()
    print(f"Profiled functions: {len(profiler_stats)}")
    
    # Test 9: Debug summary
    print("\n9. Getting debug summary...")
    debug_summary = get_debug_summary()
    print(f"Debug markers: {debug_summary['debug_markers']}")
    print(f"Function stats count: {len(debug_summary['function_stats'])}")
    
    # Test 10: Export debug data
    print("\n10. Testing debug data export...")
    export_success = verbose_logger.export_debug_data("logs/simple_test_export.json")
    print(f"Export successful: {export_success}")
    
    print("\n✅ All tests completed!")
    print(f"📊 Final stats: {len(stats)} functions, {perf_summary['total_function_calls']} calls")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)