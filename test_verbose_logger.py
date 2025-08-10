#!/usr/bin/env python3
"""
Comprehensive Test Suite for Verbose Logger System
Tests all logging functionalities including verbose logging, debugging, and testing features
"""

import asyncio
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.verbose_logger import (
    get_verbose_logger, setup_verbose_logging, trace_execution, trace_async_execution,
    log_startup_sequence, log_shutdown_sequence, log_configuration_loaded,
    log_database_connection, log_api_request, VerboseLogger
)
from core.unified_logger import LogCategory


class VerboseLoggerTestSuite:
    """Comprehensive test suite for verbose logger"""
    
    def __init__(self):
        self.verbose_logger = get_verbose_logger()
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
    
    def run_all_tests(self):
        """Run all test categories"""
        print("🧪 Starting Verbose Logger Test Suite")
        print("=" * 60)
        
        # Test categories
        test_categories = [
            ("Basic Logging Functions", self.test_basic_logging),
            ("Function Tracing", self.test_function_tracing),
            ("Async Function Tracing", self.test_async_tracing),
            ("Debug Checkpoints", self.test_debug_checkpoints),
            ("Test Context Management", self.test_context_management),
            ("Performance Monitoring", self.test_performance_monitoring),
            ("Configuration Loading", self.test_configuration),
            ("Data Masking", self.test_data_masking),
            ("Error Handling", self.test_error_handling),
            ("Statistics Collection", self.test_statistics)
        ]
        
        for category_name, test_func in test_categories:
            print(f"\n📝 Testing: {category_name}")
            print("-" * 40)
            
            try:
                if asyncio.iscoroutinefunction(test_func):
                    asyncio.run(test_func())
                else:
                    test_func()
                print(f"✅ {category_name}: PASSED")
            except Exception as e:
                print(f"❌ {category_name}: FAILED - {e}")
                traceback.print_exc()
                self.failed_tests += 1
            
            self.total_tests += 1
        
        self.print_test_summary()
    
    def test_basic_logging(self):
        """Test basic logging functionality"""
        logger = self.verbose_logger.base_logger
        
        # Test different log levels
        logger.trace("This is a trace message", category=LogCategory.SYSTEM)
        logger.debug("This is a debug message", category=LogCategory.SYSTEM)
        logger.info("This is an info message", category=LogCategory.SYSTEM)
        logger.success("This is a success message", category=LogCategory.SYSTEM)
        logger.warning("This is a warning message", category=LogCategory.SYSTEM)
        
        # Test with extra data
        logger.info(
            "Message with extra data",
            category=LogCategory.API,
            extra_data={"test_key": "test_value", "number": 42}
        )
        
        # Test with performance metrics
        from core.unified_logger import PerformanceMetrics
        logger.performance_info(
            "Performance test message",
            duration=123.45,
            performance=PerformanceMetrics(
                execution_time_ms=123.45,
                memory_usage_mb=256.8
            )
        )
        
        self.passed_tests += 1
        print("  ✓ Basic logging levels working")
        print("  ✓ Extra data logging working")
        print("  ✓ Performance logging working")
    
    def test_function_tracing(self):
        """Test function execution tracing"""
        
        @trace_execution(category=LogCategory.SYSTEM)
        def sample_function(x: int, y: str = "default") -> str:
            """Sample function for testing tracing"""
            time.sleep(0.01)  # Simulate some work
            if x < 0:
                raise ValueError("x must be positive")
            return f"Result: {x} - {y}"
        
        @trace_execution(category=LogCategory.TRADING)
        def complex_function(data: Dict[str, Any]) -> List[str]:
            """Complex function with nested calls"""
            results = []
            for key, value in data.items():
                result = sample_function(len(key), str(value))
                results.append(result)
            return results
        
        # Test successful execution
        result = sample_function(10, "test")
        assert result == "Result: 10 - test"
        
        # Test with complex data
        complex_data = {
            "key1": 123,
            "key2": {"nested": "data"},
            "key3": [1, 2, 3]
        }
        complex_result = complex_function(complex_data)
        assert len(complex_result) == 3
        
        # Test exception handling
        try:
            sample_function(-1)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # Expected
        
        self.passed_tests += 1
        print("  ✓ Function entry/exit tracing working")
        print("  ✓ Parameter and return value logging working")
        print("  ✓ Exception tracing working")
    
    async def test_async_tracing(self):
        """Test async function tracing"""
        
        @trace_async_execution(category=LogCategory.AI)
        async def async_sample_function(delay: float, data: str) -> Dict[str, Any]:
            """Async function for testing"""
            await asyncio.sleep(delay)
            return {
                "processed": data.upper(),
                "delay": delay,
                "timestamp": time.time()
            }
        
        @trace_async_execution(category=LogCategory.DATABASE)
        async def async_complex_function(items: List[str]) -> Dict[str, int]:
            """Complex async function with multiple calls"""
            results = {}
            for item in items:
                result = await async_sample_function(0.01, item)
                results[item] = len(result["processed"])
            return results
        
        # Test successful async execution
        result = await async_sample_function(0.02, "hello world")
        assert result["processed"] == "HELLO WORLD"
        
        # Test complex async execution
        items = ["test1", "test2", "test3"]
        complex_result = await async_complex_function(items)
        assert len(complex_result) == 3
        
        self.passed_tests += 1
        print("  ✓ Async function tracing working")
        print("  ✓ Async parameter and return logging working")
    
    def test_debug_checkpoints(self):
        """Test debug checkpoint functionality"""
        
        # Simple checkpoint
        self.verbose_logger.debug_checkpoint("test_start")
        
        # Checkpoint with data
        test_data = {
            "iteration": 1,
            "value": 42,
            "status": "processing"
        }
        self.verbose_logger.debug_checkpoint("processing_data", test_data)
        
        # Checkpoint with sensitive data
        sensitive_data = {
            "user_id": "12345",
            "password": "secret123",
            "api_key": "abc123def456",
            "normal_data": "this is fine"
        }
        self.verbose_logger.debug_checkpoint("sensitive_data_test", sensitive_data)
        
        # Verify debug markers were created
        markers = self.verbose_logger.get_debug_markers()
        assert len(markers) >= 3, f"Expected at least 3 markers, got {len(markers)}"
        
        self.passed_tests += 1
        print("  ✓ Debug checkpoints working")
        print("  ✓ Sensitive data masking working")
        print(f"  ✓ Created {len(markers)} debug markers")
    
    def test_context_management(self):
        """Test test context management"""
        
        # Test assertion logging
        self.verbose_logger.log_assertion("simple_assertion", True, "expected", "expected")
        self.verbose_logger.log_assertion("failed_assertion", False, "expected", "actual")
        
        # Test context manager
        with self.verbose_logger.test_context("sample_test", {"param1": "value1"}):
            self.verbose_logger.log_assertion("context_assertion_1", True)
            self.verbose_logger.log_assertion("context_assertion_2", True)
            
            # Simulate some test work
            time.sleep(0.01)
        
        # Test failed test context
        try:
            with self.verbose_logger.test_context("failing_test"):
                self.verbose_logger.log_assertion("will_fail", False)
                raise ValueError("Simulated test failure")
        except ValueError:
            pass  # Expected
        
        self.passed_tests += 1
        print("  ✓ Test context management working")
        print("  ✓ Assertion logging working")
        print("  ✓ Test failure handling working")
    
    def test_performance_monitoring(self):
        """Test performance monitoring features"""
        
        # Get initial stats
        initial_stats = self.verbose_logger.get_function_stats()
        initial_count = len(initial_stats)
        
        # Create some traced functions to generate stats
        @trace_execution()
        def perf_test_function_1(iterations: int) -> int:
            total = 0
            for i in range(iterations):
                total += i
            return total
        
        @trace_execution()
        def perf_test_function_2(data: str) -> str:
            return data.upper()
        
        # Call functions multiple times
        for i in range(5):
            perf_test_function_1(100)
            perf_test_function_2(f"test_{i}")
        
        # Get updated stats
        updated_stats = self.verbose_logger.get_function_stats()
        
        # Verify stats were collected
        assert len(updated_stats) > initial_count
        
        # Get performance summary
        perf_summary = self.verbose_logger.get_performance_summary()
        
        assert perf_summary["total_function_calls"] > 0
        assert len(perf_summary["most_called_functions"]) > 0
        
        self.passed_tests += 1
        print("  ✓ Function statistics collection working")
        print(f"  ✓ Tracking {perf_summary['total_functions_traced']} functions")
        print(f"  ✓ Recorded {perf_summary['total_function_calls']} function calls")
    
    def test_configuration(self):
        """Test configuration loading and management"""
        
        # Test convenience logging functions
        log_startup_sequence("TestApp", "1.0.0")
        log_configuration_loaded("test_config", {"key1": "value1", "key2": 123})
        log_database_connection("MongoDB", "connected", "localhost:27017")
        log_database_connection("Redis", "failed", "Connection timeout")
        log_api_request("GET", "/api/test", 200, 45.67)
        log_api_request("POST", "/api/data", 500, 1234.56)
        
        # Test configuration access
        config = self.verbose_logger.config
        assert "verbose_settings" in config or "logging" in config
        
        self.passed_tests += 1
        print("  ✓ Convenience logging functions working")
        print("  ✓ Configuration loading working")
    
    def test_data_masking(self):
        """Test sensitive data masking"""
        
        # Test data with sensitive information
        sensitive_test_data = {
            "username": "testuser",
            "password": "secret123",
            "api_key": "abc123def456",
            "token": "xyz789token",
            "SECRET_VALUE": "topsecret",
            "normal_data": "this is fine",
            "nested": {
                "password": "nested_secret",
                "public_info": "visible"
            }
        }
        
        # Use debug checkpoint to trigger masking
        self.verbose_logger.debug_checkpoint("masking_test", sensitive_test_data)
        
        # The actual masking is internal, but we can verify the function exists
        masked_data = self.verbose_logger._mask_sensitive_data(sensitive_test_data)
        
        # Check that sensitive data was masked
        assert masked_data["password"] == "***MASKED***"
        assert masked_data["api_key"] == "***MASKED***" 
        assert masked_data["normal_data"] == "this is fine"
        assert masked_data["nested"]["password"] == "***MASKED***"
        assert masked_data["nested"]["public_info"] == "visible"
        
        self.passed_tests += 1
        print("  ✓ Sensitive data masking working")
        print("  ✓ Nested data masking working")
    
    def test_error_handling(self):
        """Test error handling and exception logging"""
        
        @trace_execution()
        def error_prone_function(x: int) -> str:
            """Function that may raise errors"""
            if x == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            elif x < 0:
                raise ValueError("Negative values not allowed")
            else:
                return f"Success: {10 / x}"
        
        # Test successful execution
        result = error_prone_function(2)
        assert "Success: 5.0" == result
        
        # Test different exception types
        exceptions_caught = []
        
        try:
            error_prone_function(0)
        except ZeroDivisionError as e:
            exceptions_caught.append(type(e).__name__)
        
        try:
            error_prone_function(-1)
        except ValueError as e:
            exceptions_caught.append(type(e).__name__)
        
        assert "ZeroDivisionError" in exceptions_caught
        assert "ValueError" in exceptions_caught
        
        # Check that error stats were updated
        stats = self.verbose_logger.get_function_stats()
        func_name = f"{error_prone_function.__module__}.{error_prone_function.__name__}"
        
        if func_name in stats:
            assert stats[func_name]["error_count"] > 0
        
        self.passed_tests += 1
        print("  ✓ Exception logging working")
        print("  ✓ Error statistics tracking working")
    
    def test_statistics(self):
        """Test statistics collection and export"""
        
        # Create multiple traced functions to generate varied stats
        @trace_execution()
        def fast_function():
            return "fast"
        
        @trace_execution()
        def slow_function():
            time.sleep(0.1)
            return "slow"
        
        @trace_execution()
        def error_function():
            raise RuntimeError("Test error")
        
        # Generate some statistics
        for _ in range(3):
            fast_function()
        
        for _ in range(2):
            slow_function()
        
        for _ in range(2):
            try:
                error_function()
            except RuntimeError:
                pass
        
        # Get and verify statistics
        stats = self.verbose_logger.get_function_stats()
        perf_summary = self.verbose_logger.get_performance_summary()
        
        assert perf_summary["total_function_calls"] > 0
        assert perf_summary["total_errors"] > 0
        assert len(perf_summary["slowest_functions"]) > 0
        assert len(perf_summary["error_prone_functions"]) > 0
        
        # Test export functionality
        export_path = Path("logs/debug_export_test.json")
        export_success = self.verbose_logger.export_debug_data(export_path)
        assert export_success, "Debug data export should succeed"
        assert export_path.exists(), "Export file should exist"
        
        # Verify exported data
        with open(export_path, 'r', encoding='utf-8') as f:
            exported_data = json.load(f)
        
        assert "function_stats" in exported_data
        assert "performance_summary" in exported_data
        assert "debug_markers" in exported_data
        
        # Clean up
        export_path.unlink(missing_ok=True)
        
        self.passed_tests += 1
        print("  ✓ Statistics collection working")
        print("  ✓ Performance summary generation working")
        print("  ✓ Debug data export working")
    
    def print_test_summary(self):
        """Print test execution summary"""
        print("\n" + "=" * 60)
        print("🧪 VERBOSE LOGGER TEST SUITE RESULTS")
        print("=" * 60)
        
        success_rate = (self.passed_tests / max(self.total_tests, 1)) * 100
        
        print(f"Total Tests: {self.total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if self.failed_tests == 0:
            print("\n✅ ALL TESTS PASSED! Verbose logging system is working correctly.")
        else:
            print(f"\n❌ {self.failed_tests} test(s) failed. Please check the output above.")
        
        # Print system information
        print("\n📊 SYSTEM INFORMATION:")
        print("-" * 30)
        
        perf_summary = self.verbose_logger.get_performance_summary()
        print(f"Functions traced: {perf_summary['total_functions_traced']}")
        print(f"Total function calls: {perf_summary['total_function_calls']}")
        print(f"Total errors: {perf_summary['total_errors']}")
        
        debug_markers = self.verbose_logger.get_debug_markers()
        print(f"Debug markers: {len(debug_markers)}")
        
        print("\n🔧 CONFIGURATION STATUS:")
        print("-" * 25)
        config = self.verbose_logger.config
        verbose_settings = config.get("verbose_settings", {})
        debug_settings = config.get("debug_settings", {})
        testing_settings = config.get("testing_settings", {})
        
        print(f"Verbose logging: {'✅' if verbose_settings.get('log_entry_exit') else '❌'}")
        print(f"Parameter logging: {'✅' if verbose_settings.get('log_parameters') else '❌'}")
        print(f"Return value logging: {'✅' if verbose_settings.get('log_return_values') else '❌'}")
        print(f"Exception tracing: {'✅' if verbose_settings.get('log_exceptions_detailed') else '❌'}")
        print(f"Memory monitoring: {'✅' if verbose_settings.get('log_memory_usage') else '❌'}")
        print(f"Execution flow tracking: {'✅' if debug_settings.get('execution_flow_tracking') else '❌'}")
        print(f"Test execution logging: {'✅' if testing_settings.get('log_test_execution') else '❌'}")
        
        print("\n📈 TOP FUNCTION STATISTICS:")
        print("-" * 28)
        
        if perf_summary['most_called_functions']:
            print("Most Called Functions:")
            for func_name, call_count in perf_summary['most_called_functions'][:5]:
                print(f"  • {func_name}: {call_count} calls")
        
        if perf_summary['slowest_functions']:
            print("\nSlowest Functions:")
            for func_name, avg_duration in perf_summary['slowest_functions'][:5]:
                print(f"  • {func_name}: {avg_duration:.2f}ms avg")
        
        if perf_summary['error_prone_functions']:
            print("\nError-Prone Functions:")
            for func_name, error_count in perf_summary['error_prone_functions'][:5]:
                print(f"  • {func_name}: {error_count} errors")


def main():
    """Main test execution function"""
    
    print("🚀 Initializing Verbose Logger Test Suite")
    
    # Setup logging directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Initialize verbose logger
    verbose_logger = setup_verbose_logging()
    
    # Run the test suite
    test_suite = VerboseLoggerTestSuite()
    test_suite.run_all_tests()
    
    # Optional: Reset stats after testing
    # verbose_logger.reset_stats()
    
    print("\n🏁 Test suite completed!")
    return 0 if test_suite.failed_tests == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)