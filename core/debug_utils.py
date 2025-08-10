"""
Debug and Testing Utilities
Provides debugging tools, test helpers, and development utilities
"""

import asyncio
import inspect
import json
import os
import sys
import time
import traceback
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from pathlib import Path
from contextlib import contextmanager, asynccontextmanager
from collections import defaultdict

from .verbose_logger import get_verbose_logger, trace_execution, trace_async_execution
from .unified_logger import LogCategory, LogLevel


class DebugProfiler:
    """Performance profiler for debugging"""
    
    def __init__(self):
        self.profiles = defaultdict(list)
        self.active_profiles = {}
        
    def start_profile(self, name: str) -> str:
        """Start profiling a code block"""
        profile_id = f"{name}_{int(time.time() * 1000)}"
        self.active_profiles[profile_id] = {
            'name': name,
            'start_time': time.time(),
            'start_memory': self._get_memory_usage()
        }
        return profile_id
    
    def end_profile(self, profile_id: str) -> Dict[str, Any]:
        """End profiling and return results"""
        if profile_id not in self.active_profiles:
            return {}
        
        start_data = self.active_profiles.pop(profile_id)
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        result = {
            'name': start_data['name'],
            'duration_ms': (end_time - start_data['start_time']) * 1000,
            'memory_delta_mb': end_memory - start_data['start_memory'] if end_memory and start_data['start_memory'] else 0,
            'start_time': start_data['start_time'],
            'end_time': end_time
        }
        
        self.profiles[start_data['name']].append(result)
        return result
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return None
    
    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling code blocks"""
        profile_id = self.start_profile(name)
        verbose_logger = get_verbose_logger()
        
        try:
            yield
        finally:
            result = self.end_profile(profile_id)
            verbose_logger.base_logger.debug(
                f"⏱️  PROFILE: {name} completed in {result.get('duration_ms', 0):.2f}ms",
                category=LogCategory.PERFORMANCE,
                extra_data=result,
                tags=["profiling", "performance"]
            )
    
    def get_profile_stats(self, name: str) -> Dict[str, Any]:
        """Get aggregated statistics for a profile"""
        if name not in self.profiles:
            return {}
        
        profiles = self.profiles[name]
        durations = [p['duration_ms'] for p in profiles]
        memory_deltas = [p['memory_delta_mb'] for p in profiles if p['memory_delta_mb']]
        
        return {
            'name': name,
            'call_count': len(profiles),
            'total_duration_ms': sum(durations),
            'avg_duration_ms': sum(durations) / len(durations),
            'min_duration_ms': min(durations),
            'max_duration_ms': max(durations),
            'avg_memory_delta_mb': sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0,
            'last_called': profiles[-1]['end_time']
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all profiles"""
        return {name: self.get_profile_stats(name) for name in self.profiles.keys()}


class TestRunner:
    """Advanced test runner with verbose logging"""
    
    def __init__(self):
        self.verbose_logger = get_verbose_logger()
        self.test_results = []
        self.current_suite = None
        self.profiler = DebugProfiler()
        
    def run_test_suite(self, test_class, test_methods: Optional[List[str]] = None):
        """Run a complete test suite"""
        self.current_suite = test_class.__name__
        
        # Get test methods
        if test_methods is None:
            test_methods = [
                method for method in dir(test_class) 
                if method.startswith('test_') and callable(getattr(test_class, method))
            ]
        
        # Initialize test instance
        test_instance = test_class()
        
        # Setup phase
        self._run_phase(test_instance, 'setUp', 'setup')
        
        # Run tests
        passed = 0
        failed = 0
        
        for test_method in test_methods:
            with self.verbose_logger.test_context(f"{self.current_suite}.{test_method}"):
                try:
                    with self.profiler.profile(f"test_{test_method}"):
                        method = getattr(test_instance, test_method)
                        if asyncio.iscoroutinefunction(method):
                            asyncio.run(method())
                        else:
                            method()
                    
                    passed += 1
                    self.test_results.append({
                        'suite': self.current_suite,
                        'method': test_method,
                        'status': 'PASSED',
                        'timestamp': datetime.utcnow()
                    })
                    
                except Exception as e:
                    failed += 1
                    self.test_results.append({
                        'suite': self.current_suite,
                        'method': test_method,
                        'status': 'FAILED',
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                        'timestamp': datetime.utcnow()
                    })
        
        # Teardown phase
        self._run_phase(test_instance, 'tearDown', 'teardown')
        
        return {'passed': passed, 'failed': failed, 'total': len(test_methods)}
    
    def _run_phase(self, test_instance, method_name: str, phase_name: str):
        """Run a test phase (setup/teardown)"""
        if hasattr(test_instance, method_name):
            self.verbose_logger.base_logger.debug(
                f"🔧 Running {phase_name} for {self.current_suite}",
                category=LogCategory.SYSTEM,
                tags=["testing", phase_name]
            )
            
            method = getattr(test_instance, method_name)
            try:
                if asyncio.iscoroutinefunction(method):
                    asyncio.run(method())
                else:
                    method()
            except Exception as e:
                self.verbose_logger.base_logger.error(
                    f"❌ {phase_name} failed for {self.current_suite}: {e}",
                    category=LogCategory.SYSTEM,
                    exception=e,
                    tags=["testing", phase_name, "error"]
                )
    
    def assert_equals(self, actual, expected, message: str = ""):
        """Enhanced assertion with verbose logging"""
        test_name = f"assert_equals_{message}" if message else "assert_equals"
        condition = actual == expected
        
        self.verbose_logger.log_assertion(test_name, condition, expected, actual)
        
        if not condition:
            raise AssertionError(f"Expected {expected}, but got {actual}. {message}")
    
    def assert_true(self, condition: bool, message: str = ""):
        """Assert true with logging"""
        test_name = f"assert_true_{message}" if message else "assert_true"
        self.verbose_logger.log_assertion(test_name, condition, True, condition)
        
        if not condition:
            raise AssertionError(f"Expected True, but got {condition}. {message}")
    
    def assert_false(self, condition: bool, message: str = ""):
        """Assert false with logging"""
        test_name = f"assert_false_{message}" if message else "assert_false"
        self.verbose_logger.log_assertion(test_name, not condition, False, condition)
        
        if condition:
            raise AssertionError(f"Expected False, but got {condition}. {message}")
    
    def assert_raises(self, exception_type, callable_obj, *args, **kwargs):
        """Assert that a callable raises a specific exception"""
        test_name = f"assert_raises_{exception_type.__name__}"
        
        try:
            callable_obj(*args, **kwargs)
            self.verbose_logger.log_assertion(test_name, False, f"{exception_type.__name__} raised", "No exception")
            raise AssertionError(f"Expected {exception_type.__name__} to be raised")
        except exception_type:
            self.verbose_logger.log_assertion(test_name, True, f"{exception_type.__name__} raised", f"{exception_type.__name__} raised")
        except Exception as e:
            self.verbose_logger.log_assertion(test_name, False, f"{exception_type.__name__} raised", f"{type(e).__name__} raised")
            raise AssertionError(f"Expected {exception_type.__name__}, but got {type(e).__name__}: {e}")
    
    def get_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        passed_tests = [r for r in self.test_results if r['status'] == 'PASSED']
        failed_tests = [r for r in self.test_results if r['status'] == 'FAILED']
        
        profile_stats = self.profiler.get_all_stats()
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'summary': {
                'total_tests': len(self.test_results),
                'passed': len(passed_tests),
                'failed': len(failed_tests),
                'success_rate': len(passed_tests) / len(self.test_results) * 100 if self.test_results else 0
            },
            'failed_tests': failed_tests,
            'performance_profiles': profile_stats,
            'test_details': self.test_results
        }


class DebugBreakpoint:
    """Debug breakpoint utility"""
    
    def __init__(self):
        self.breakpoints = {}
        self.hit_counts = defaultdict(int)
        self.verbose_logger = get_verbose_logger()
    
    def set_breakpoint(self, name: str, condition: Optional[Callable[[], bool]] = None):
        """Set a debug breakpoint"""
        self.breakpoints[name] = {
            'condition': condition,
            'created': datetime.utcnow(),
            'hit_count': 0
        }
        
        self.verbose_logger.debug_checkpoint(f"breakpoint_set_{name}")
    
    def hit_breakpoint(self, name: str, context: Optional[Dict[str, Any]] = None):
        """Hit a breakpoint and log debug information"""
        if name not in self.breakpoints:
            return False
        
        breakpoint = self.breakpoints[name]
        
        # Check condition if provided
        if breakpoint['condition'] and not breakpoint['condition']():
            return False
        
        self.hit_counts[name] += 1
        breakpoint['hit_count'] = self.hit_counts[name]
        
        self.verbose_logger.debug_checkpoint(
            f"breakpoint_hit_{name}",
            {
                'hit_count': self.hit_counts[name],
                'context': context
            }
        )
        
        self.verbose_logger.base_logger.warning(
            f"🔴 BREAKPOINT HIT: {name} (#{self.hit_counts[name]})",
            category=LogCategory.SYSTEM,
            extra_data={
                'breakpoint_name': name,
                'hit_count': self.hit_counts[name],
                'context': context
            },
            tags=["debug", "breakpoint"]
        )
        
        return True
    
    def remove_breakpoint(self, name: str):
        """Remove a breakpoint"""
        if name in self.breakpoints:
            del self.breakpoints[name]
            self.verbose_logger.debug_checkpoint(f"breakpoint_removed_{name}")


class SystemMonitor:
    """System resource monitoring for debugging"""
    
    def __init__(self):
        self.verbose_logger = get_verbose_logger()
        self.monitoring = False
        self.monitor_thread = None
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_usage': [],
            'network_io': []
        }
    
    def start_monitoring(self, interval: float = 1.0):
        """Start system monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,), 
            daemon=True
        )
        self.monitor_thread.start()
        
        self.verbose_logger.base_logger.info(
            f"📊 Started system monitoring (interval: {interval}s)",
            category=LogCategory.MONITORING,
            tags=["monitoring", "start"]
        )
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        self.verbose_logger.base_logger.info(
            "📊 Stopped system monitoring",
            category=LogCategory.MONITORING,
            tags=["monitoring", "stop"]
        )
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        try:
            import psutil
        except ImportError:
            self.verbose_logger.base_logger.warning(
                "psutil not available, system monitoring disabled",
                category=LogCategory.MONITORING
            )
            return
        
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent()
                self.metrics['cpu_usage'].append({
                    'timestamp': datetime.utcnow(),
                    'value': cpu_percent
                })
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.metrics['memory_usage'].append({
                    'timestamp': datetime.utcnow(),
                    'value': memory.percent,
                    'used_mb': memory.used / 1024 / 1024,
                    'available_mb': memory.available / 1024 / 1024
                })
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.metrics['disk_usage'].append({
                    'timestamp': datetime.utcnow(),
                    'value': (disk.used / disk.total) * 100,
                    'used_gb': disk.used / 1024 / 1024 / 1024,
                    'free_gb': disk.free / 1024 / 1024 / 1024
                })
                
                # Network I/O
                net_io = psutil.net_io_counters()
                self.metrics['network_io'].append({
                    'timestamp': datetime.utcnow(),
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv
                })
                
                # Keep only recent data (last 1000 points)
                for metric_type in self.metrics:
                    if len(self.metrics[metric_type]) > 1000:
                        self.metrics[metric_type] = self.metrics[metric_type][-1000:]
                
                time.sleep(interval)
                
            except Exception as e:
                self.verbose_logger.base_logger.error(
                    f"System monitoring error: {e}",
                    category=LogCategory.MONITORING,
                    exception=e
                )
                time.sleep(interval)
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        try:
            import psutil
            
            return {
                'cpu_percent': psutil.cpu_percent(),
                'memory': {
                    'percent': psutil.virtual_memory().percent,
                    'used_mb': psutil.virtual_memory().used / 1024 / 1024,
                    'available_mb': psutil.virtual_memory().available / 1024 / 1024
                },
                'disk': {
                    'percent': (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100,
                    'used_gb': psutil.disk_usage('/').used / 1024 / 1024 / 1024,
                    'free_gb': psutil.disk_usage('/').free / 1024 / 1024 / 1024
                },
                'network': {
                    'bytes_sent': psutil.net_io_counters().bytes_sent,
                    'bytes_recv': psutil.net_io_counters().bytes_recv
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def export_metrics(self, filepath: Union[str, Path]) -> bool:
        """Export monitoring metrics to file"""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            export_data = {
                'export_timestamp': datetime.utcnow().isoformat(),
                'monitoring_period': {
                    'start': self.metrics['cpu_usage'][0]['timestamp'].isoformat() if self.metrics['cpu_usage'] else None,
                    'end': self.metrics['cpu_usage'][-1]['timestamp'].isoformat() if self.metrics['cpu_usage'] else None,
                    'data_points': len(self.metrics['cpu_usage'])
                },
                'metrics': self.metrics,
                'current_stats': self.get_current_stats()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            return True
        except Exception as e:
            self.verbose_logger.base_logger.error(
                f"Failed to export monitoring metrics: {e}",
                category=LogCategory.MONITORING,
                exception=e
            )
            return False


# Global instances
debug_profiler = DebugProfiler()
test_runner = TestRunner()
debug_breakpoint = DebugBreakpoint()
system_monitor = SystemMonitor()


# Convenience decorators
def debug_profile(name: Optional[str] = None):
    """Decorator to profile function execution"""
    def decorator(func):
        profile_name = name or f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                with debug_profiler.profile(profile_name):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def wrapper(*args, **kwargs):
                with debug_profiler.profile(profile_name):
                    return func(*args, **kwargs)
            return wrapper
    
    return decorator


def debug_trace(log_params: bool = True, log_returns: bool = False):
    """Decorator to trace function calls for debugging"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            return trace_async_execution(LogCategory.SYSTEM, log_params, log_returns)(func)
        else:
            return trace_execution(LogCategory.SYSTEM, log_params, log_returns)(func)
    
    return decorator


# Convenience functions
def checkpoint(name: str, data: Optional[Dict[str, Any]] = None):
    """Quick debug checkpoint"""
    verbose_logger = get_verbose_logger()
    verbose_logger.debug_checkpoint(name, data)


def breakpoint_here(name: str, condition: Optional[Callable[[], bool]] = None, 
                   context: Optional[Dict[str, Any]] = None):
    """Set and immediately hit a breakpoint"""
    debug_breakpoint.set_breakpoint(name, condition)
    debug_breakpoint.hit_breakpoint(name, context)


def log_variable(var_name: str, value: Any, category: LogCategory = LogCategory.SYSTEM):
    """Log a variable value"""
    verbose_logger = get_verbose_logger()
    verbose_logger.base_logger.debug(
        f"📊 VARIABLE: {var_name} = {value}",
        category=category,
        extra_data={'variable_name': var_name, 'variable_value': value},
        tags=["debug", "variable"]
    )


def start_system_monitoring(interval: float = 1.0):
    """Start system resource monitoring"""
    system_monitor.start_monitoring(interval)


def stop_system_monitoring():
    """Stop system resource monitoring"""
    system_monitor.stop_monitoring()


def get_debug_summary() -> Dict[str, Any]:
    """Get comprehensive debug summary"""
    verbose_logger = get_verbose_logger()
    
    return {
        'timestamp': datetime.utcnow().isoformat(),
        'function_stats': verbose_logger.get_function_stats(),
        'performance_summary': verbose_logger.get_performance_summary(),
        'debug_markers': len(verbose_logger.get_debug_markers()),
        'profiler_stats': debug_profiler.get_all_stats(),
        'breakpoint_hits': dict(debug_breakpoint.hit_counts),
        'system_stats': system_monitor.get_current_stats(),
        'test_results': test_runner.get_test_report() if test_runner.test_results else None
    }


def export_all_debug_data(directory: Union[str, Path] = "logs/debug_export"):
    """Export all debug data to files"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    # Export verbose logger data
    verbose_logger = get_verbose_logger()
    verbose_logger.export_debug_data(directory / f"verbose_debug_{timestamp}.json")
    
    # Export system metrics
    system_monitor.export_metrics(directory / f"system_metrics_{timestamp}.json")
    
    # Export comprehensive summary
    summary = get_debug_summary()
    with open(directory / f"debug_summary_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"Debug data exported to: {directory}")
    return directory