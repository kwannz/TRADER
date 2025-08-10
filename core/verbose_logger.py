"""
Verbose Logger - Enhanced logging with detailed tracing, debugging, and testing support
Extends the unified logger with comprehensive verbose logging capabilities
"""

import asyncio
import inspect
import json
import os
import sys
import time
import traceback
import threading
import functools
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, deque

# Import existing unified logger
from .unified_logger import (
    UnifiedLogger, LogLevel, LogCategory, LogRecord, LogMetadata, 
    PerformanceMetrics, get_logger as get_unified_logger
)

@dataclass
class ExecutionTrace:
    """Execution trace information"""
    function_name: str
    module: str
    filename: str
    line_number: int
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None
    return_value: Optional[Any] = None
    exception: Optional[Exception] = None
    memory_before: Optional[float] = None
    memory_after: Optional[float] = None
    thread_id: Optional[str] = None
    call_stack: Optional[List[str]] = None

@dataclass 
class DebugContext:
    """Debug context information"""
    test_name: Optional[str] = None
    test_phase: Optional[str] = None  # setup, execution, teardown
    assertion_count: int = 0
    debug_markers: List[str] = field(default_factory=list)
    breakpoints: List[Tuple[str, int]] = field(default_factory=list)
    coverage_data: Dict[str, Any] = field(default_factory=dict)

class VerboseLogger:
    """Enhanced verbose logger with comprehensive debugging and testing support"""
    
    def __init__(self, name: str = "verbose_logger", config_path: str = "config/logging_config.json"):
        self.name = name
        self.config_path = Path(config_path)
        self.base_logger = get_unified_logger()
        
        # Load configuration
        self.config = self._load_config()
        
        # Execution tracking
        self.execution_traces: deque = deque(maxlen=10000)
        self.function_stats: Dict[str, Dict] = defaultdict(lambda: {
            'call_count': 0,
            'total_duration': 0,
            'avg_duration': 0,
            'max_duration': 0,
            'min_duration': float('inf'),
            'error_count': 0,
            'last_called': None
        })
        
        # Debug context
        self.debug_context = DebugContext()
        self.debug_markers = []
        self.call_stack = []
        
        # Performance monitoring
        self.performance_metrics = {
            'memory_snapshots': deque(maxlen=1000),
            'cpu_usage': deque(maxlen=1000),
            'function_timings': defaultdict(list)
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize based on configuration
        self._initialize_verbose_features()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load logging configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                return config
            else:
                return self._get_default_config()
        except Exception as e:
            print(f"Failed to load logging config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default verbose logging configuration"""
        return {
            "verbose_settings": {
                "log_entry_exit": True,
                "log_parameters": True,
                "log_return_values": True,
                "log_exceptions_detailed": True,
                "log_memory_usage": True,
                "log_execution_time": True,
                "log_thread_info": True,
                "log_call_stack": True,
                "max_parameter_length": 500,
                "max_return_value_length": 500,
                "sensitive_data_masking": True,
                "mask_patterns": ["password", "token", "key", "secret", "api_key"]
            },
            "debug_settings": {
                "trace_all_modules": False,
                "trace_specific_modules": ["core", "trading", "ai_engine", "strategy"],
                "exclude_modules": ["urllib", "requests", "asyncio", "logging"],
                "execution_flow_tracking": True
            },
            "testing_settings": {
                "log_test_execution": True,
                "log_assertions": True,
                "log_test_data": True,
                "log_setup_teardown": True,
                "test_timing": True
            }
        }
    
    def _initialize_verbose_features(self):
        """Initialize verbose logging features based on configuration"""
        verbose_config = self.config.get("verbose_settings", {})
        
        if verbose_config.get("log_memory_usage", True):
            self._start_memory_monitoring()
        
        if verbose_config.get("log_execution_time", True):
            self._start_performance_monitoring()
    
    def _start_memory_monitoring(self):
        """Start background memory monitoring"""
        def monitor_memory():
            try:
                import psutil
                while True:
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    self.performance_metrics['memory_snapshots'].append({
                        'timestamp': datetime.utcnow(),
                        'memory_mb': memory_mb
                    })
                    time.sleep(5)  # Monitor every 5 seconds
            except Exception:
                pass
        
        thread = threading.Thread(target=monitor_memory, daemon=True)
        thread.start()
    
    def _start_performance_monitoring(self):
        """Start background performance monitoring"""
        def monitor_performance():
            try:
                import psutil
                while True:
                    cpu_percent = psutil.cpu_percent()
                    self.performance_metrics['cpu_usage'].append({
                        'timestamp': datetime.utcnow(),
                        'cpu_percent': cpu_percent
                    })
                    time.sleep(1)  # Monitor every second
            except Exception:
                pass
        
        thread = threading.Thread(target=monitor_performance, daemon=True)
        thread.start()
    
    def _should_trace_module(self, module_name: str) -> bool:
        """Check if module should be traced based on configuration"""
        debug_config = self.config.get("debug_settings", {})
        
        # Check exclusions first
        exclude_modules = debug_config.get("exclude_modules", [])
        for exclude in exclude_modules:
            if exclude in module_name:
                return False
        
        # Check if tracing all modules
        if debug_config.get("trace_all_modules", False):
            return True
        
        # Check specific modules
        trace_modules = debug_config.get("trace_specific_modules", [])
        return any(module in module_name for module in trace_modules)
    
    def _mask_sensitive_data(self, data: Any) -> Any:
        """Mask sensitive data in logs"""
        verbose_config = self.config.get("verbose_settings", {})
        
        if not verbose_config.get("sensitive_data_masking", True):
            return data
        
        mask_patterns = verbose_config.get("mask_patterns", [])
        
        if isinstance(data, dict):
            masked = {}
            for key, value in data.items():
                if any(pattern.lower() in key.lower() for pattern in mask_patterns):
                    masked[key] = "***MASKED***"
                else:
                    masked[key] = self._mask_sensitive_data(value)
            return masked
        elif isinstance(data, (list, tuple)):
            return [self._mask_sensitive_data(item) for item in data]
        elif isinstance(data, str):
            # Check if string contains sensitive patterns
            lower_data = data.lower()
            if any(pattern.lower() in lower_data for pattern in mask_patterns):
                return "***MASKED***"
            return data
        else:
            return data
    
    def _truncate_data(self, data: Any, max_length: int) -> str:
        """Truncate data to specified length"""
        try:
            data_str = str(data)
            if len(data_str) <= max_length:
                return data_str
            return data_str[:max_length] + "... [TRUNCATED]"
        except Exception:
            return "[REPR_ERROR]"
    
    def _get_current_memory(self) -> Optional[float]:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return None
    
    def _get_call_stack(self) -> List[str]:
        """Get current call stack"""
        try:
            stack = traceback.extract_stack()[:-2]  # Exclude current frame
            return [f"{frame.filename}:{frame.name}:{frame.lineno}" for frame in stack]
        except Exception:
            return []
    
    def trace_function_entry(self, func_name: str, parameters: Dict[str, Any], 
                           caller_info: Tuple[str, str, int, str]):
        """Trace function entry"""
        verbose_config = self.config.get("verbose_settings", {})
        
        if not verbose_config.get("log_entry_exit", True):
            return
        
        module, function, line, filename = caller_info
        
        if not self._should_trace_module(module):
            return
        
        # Mask and truncate parameters
        masked_params = self._mask_sensitive_data(parameters)
        max_length = verbose_config.get("max_parameter_length", 500)
        truncated_params = {
            k: self._truncate_data(v, max_length) for k, v in masked_params.items()
        }
        
        extra_data = {
            "function_name": func_name,
            "parameters": truncated_params,
            "caller_module": module,
            "caller_function": function,
            "caller_line": line
        }
        
        if verbose_config.get("log_memory_usage", True):
            extra_data["memory_before_mb"] = self._get_current_memory()
        
        if verbose_config.get("log_thread_info", True):
            extra_data["thread_id"] = threading.current_thread().name
        
        if verbose_config.get("log_call_stack", True):
            extra_data["call_stack"] = self._get_call_stack()[-5:]  # Last 5 frames
        
        self.base_logger.trace(
            f"→ ENTER {func_name}({', '.join(f'{k}={v}' for k, v in truncated_params.items())})",
            category=LogCategory.SYSTEM,
            extra_data=extra_data,
            tags=["function_trace", "entry"]
        )
    
    def trace_function_exit(self, func_name: str, return_value: Any, duration_ms: float,
                          caller_info: Tuple[str, str, int, str]):
        """Trace function exit"""
        verbose_config = self.config.get("verbose_settings", {})
        
        if not verbose_config.get("log_entry_exit", True):
            return
        
        module, function, line, filename = caller_info
        
        if not self._should_trace_module(module):
            return
        
        extra_data = {
            "function_name": func_name,
            "duration_ms": duration_ms,
            "caller_module": module,
            "caller_function": function,
            "caller_line": line
        }
        
        if verbose_config.get("log_return_values", True):
            masked_return = self._mask_sensitive_data(return_value)
            max_length = verbose_config.get("max_return_value_length", 500)
            extra_data["return_value"] = self._truncate_data(masked_return, max_length)
        
        if verbose_config.get("log_memory_usage", True):
            extra_data["memory_after_mb"] = self._get_current_memory()
        
        self.base_logger.trace(
            f"← EXIT {func_name} [{duration_ms:.2f}ms]",
            category=LogCategory.SYSTEM,
            extra_data=extra_data,
            tags=["function_trace", "exit"],
            performance=PerformanceMetrics(execution_time_ms=duration_ms)
        )
        
        # Update function statistics
        with self._lock:
            stats = self.function_stats[func_name]
            stats['call_count'] += 1
            stats['total_duration'] += duration_ms
            stats['avg_duration'] = stats['total_duration'] / stats['call_count']
            stats['max_duration'] = max(stats['max_duration'], duration_ms)
            stats['min_duration'] = min(stats['min_duration'], duration_ms)
            stats['last_called'] = datetime.utcnow()
    
    def trace_exception(self, func_name: str, exception: Exception, duration_ms: float,
                       caller_info: Tuple[str, str, int, str]):
        """Trace function exception"""
        verbose_config = self.config.get("verbose_settings", {})
        
        module, function, line, filename = caller_info
        
        if not self._should_trace_module(module):
            return
        
        extra_data = {
            "function_name": func_name,
            "duration_ms": duration_ms,
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "caller_module": module,
            "caller_function": function,
            "caller_line": line
        }
        
        if verbose_config.get("log_exceptions_detailed", True):
            extra_data["traceback"] = traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )
        
        self.base_logger.error(
            f"✗ EXCEPTION in {func_name}: {type(exception).__name__}: {exception}",
            category=LogCategory.SYSTEM,
            exception=exception,
            extra_data=extra_data,
            tags=["function_trace", "exception"],
            performance=PerformanceMetrics(execution_time_ms=duration_ms)
        )
        
        # Update error statistics
        with self._lock:
            self.function_stats[func_name]['error_count'] += 1
    
    def debug_checkpoint(self, checkpoint_name: str, data: Optional[Dict[str, Any]] = None):
        """Create a debug checkpoint with optional data"""
        self.debug_markers.append({
            'timestamp': datetime.utcnow(),
            'checkpoint': checkpoint_name,
            'data': self._mask_sensitive_data(data) if data else None,
            'memory_mb': self._get_current_memory(),
            'thread_id': threading.current_thread().name
        })
        
        self.base_logger.debug(
            f"🔍 DEBUG CHECKPOINT: {checkpoint_name}",
            category=LogCategory.SYSTEM,
            extra_data={
                'checkpoint_name': checkpoint_name,
                'checkpoint_data': data,
                'memory_mb': self._get_current_memory()
            },
            tags=["debug", "checkpoint"]
        )
    
    def log_assertion(self, assertion_name: str, condition: bool, 
                     expected: Any = None, actual: Any = None):
        """Log test assertion with detailed information"""
        testing_config = self.config.get("testing_settings", {})
        
        if not testing_config.get("log_assertions", True):
            return
        
        self.debug_context.assertion_count += 1
        
        status = "✓ PASS" if condition else "✗ FAIL"
        message = f"{status} ASSERTION: {assertion_name}"
        
        extra_data = {
            'assertion_name': assertion_name,
            'condition': condition,
            'assertion_number': self.debug_context.assertion_count,
            'test_name': self.debug_context.test_name
        }
        
        if expected is not None:
            extra_data['expected'] = self._truncate_data(expected, 500)
        if actual is not None:
            extra_data['actual'] = self._truncate_data(actual, 500)
        
        level = LogLevel.DEBUG if condition else LogLevel.ERROR
        self.base_logger.log(
            level,
            LogCategory.SYSTEM,
            message,
            extra_data=extra_data,
            tags=["testing", "assertion", "pass" if condition else "fail"]
        )
    
    def start_test(self, test_name: str, test_data: Optional[Dict[str, Any]] = None):
        """Start test execution logging"""
        testing_config = self.config.get("testing_settings", {})
        
        if not testing_config.get("log_test_execution", True):
            return
        
        self.debug_context.test_name = test_name
        self.debug_context.test_phase = "execution"
        self.debug_context.assertion_count = 0
        
        extra_data = {
            'test_name': test_name,
            'test_phase': 'start',
            'test_data': self._mask_sensitive_data(test_data) if test_data else None
        }
        
        self.base_logger.info(
            f"🧪 TEST START: {test_name}",
            category=LogCategory.SYSTEM,
            extra_data=extra_data,
            tags=["testing", "start"]
        )
    
    def end_test(self, test_name: str, success: bool, duration_ms: float = None):
        """End test execution logging"""
        testing_config = self.config.get("testing_settings", {})
        
        if not testing_config.get("log_test_execution", True):
            return
        
        status = "✓ PASSED" if success else "✗ FAILED"
        message = f"🧪 TEST END: {test_name} - {status}"
        
        extra_data = {
            'test_name': test_name,
            'test_phase': 'end',
            'test_success': success,
            'assertion_count': self.debug_context.assertion_count,
            'duration_ms': duration_ms
        }
        
        level = LogLevel.SUCCESS if success else LogLevel.ERROR
        self.base_logger.log(
            level,
            LogCategory.SYSTEM,
            message,
            extra_data=extra_data,
            tags=["testing", "end", "passed" if success else "failed"],
            performance=PerformanceMetrics(execution_time_ms=duration_ms) if duration_ms else None
        )
        
        # Reset test context
        self.debug_context.test_name = None
        self.debug_context.test_phase = None
        self.debug_context.assertion_count = 0
    
    @contextmanager
    def test_context(self, test_name: str, test_data: Optional[Dict[str, Any]] = None):
        """Test execution context manager"""
        start_time = time.time()
        success = True
        
        try:
            self.start_test(test_name, test_data)
            yield
        except Exception as e:
            success = False
            self.base_logger.error(
                f"🧪 TEST ERROR: {test_name}",
                category=LogCategory.SYSTEM,
                exception=e,
                extra_data={'test_name': test_name},
                tags=["testing", "error"]
            )
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            self.end_test(test_name, success, duration_ms)
    
    def get_function_stats(self) -> Dict[str, Dict]:
        """Get function execution statistics"""
        with self._lock:
            return dict(self.function_stats)
    
    def get_debug_markers(self) -> List[Dict]:
        """Get all debug markers"""
        return list(self.debug_markers)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary"""
        stats = self.get_function_stats()
        
        # Calculate top slowest functions
        slowest_functions = sorted(
            [(name, data['avg_duration']) for name, data in stats.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Calculate most called functions
        most_called = sorted(
            [(name, data['call_count']) for name, data in stats.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Calculate error-prone functions
        error_prone = sorted(
            [(name, data['error_count']) for name, data in stats.items() if data['error_count'] > 0],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            'total_functions_traced': len(stats),
            'total_function_calls': sum(data['call_count'] for data in stats.values()),
            'total_errors': sum(data['error_count'] for data in stats.values()),
            'slowest_functions': slowest_functions,
            'most_called_functions': most_called,
            'error_prone_functions': error_prone,
            'memory_snapshots_count': len(self.performance_metrics['memory_snapshots']),
            'cpu_usage_samples': len(self.performance_metrics['cpu_usage'])
        }
    
    def reset_stats(self):
        """Reset all statistics and traces"""
        with self._lock:
            self.function_stats.clear()
            self.execution_traces.clear()
            self.debug_markers.clear()
            self.performance_metrics['memory_snapshots'].clear()
            self.performance_metrics['cpu_usage'].clear()
            self.performance_metrics['function_timings'].clear()
    
    def export_debug_data(self, filepath: Union[str, Path]) -> bool:
        """Export all debug data to a JSON file"""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            debug_data = {
                'export_timestamp': datetime.utcnow().isoformat(),
                'function_stats': self.get_function_stats(),
                'debug_markers': self.get_debug_markers(),
                'performance_summary': self.get_performance_summary(),
                'memory_snapshots': list(self.performance_metrics['memory_snapshots'])[-100:],  # Last 100
                'cpu_usage': list(self.performance_metrics['cpu_usage'])[-100:],  # Last 100
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(debug_data, f, indent=2, ensure_ascii=False, default=str)
            
            return True
        except Exception as e:
            self.base_logger.error(f"Failed to export debug data: {e}", exception=e)
            return False


def trace_execution(category: LogCategory = LogCategory.SYSTEM, 
                   log_params: bool = True, 
                   log_returns: bool = True):
    """Decorator for tracing function execution with verbose logging"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            verbose_logger = get_verbose_logger()
            
            # Get caller information
            frame = inspect.currentframe()
            caller_frame = frame.f_back
            caller_info = (
                caller_frame.f_globals.get('__name__', 'unknown'),
                caller_frame.f_code.co_name,
                caller_frame.f_lineno,
                caller_frame.f_code.co_filename
            )
            
            func_name = f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            # Prepare parameters for logging
            params = {}
            if log_params:
                # Get function signature
                try:
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    params = dict(bound_args.arguments)
                except Exception:
                    params = {'args': args, 'kwargs': kwargs}
            
            # Log function entry
            verbose_logger.trace_function_entry(func_name, params, caller_info)
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                # Log successful exit
                if log_returns:
                    verbose_logger.trace_function_exit(func_name, result, duration_ms, caller_info)
                else:
                    verbose_logger.trace_function_exit(func_name, "[RETURN_NOT_LOGGED]", duration_ms, caller_info)
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                verbose_logger.trace_exception(func_name, e, duration_ms, caller_info)
                raise
        
        return wrapper
    return decorator


def trace_async_execution(category: LogCategory = LogCategory.SYSTEM,
                         log_params: bool = True,
                         log_returns: bool = True):
    """Decorator for tracing async function execution with verbose logging"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            verbose_logger = get_verbose_logger()
            
            # Get caller information
            frame = inspect.currentframe()
            caller_frame = frame.f_back
            caller_info = (
                caller_frame.f_globals.get('__name__', 'unknown'),
                caller_frame.f_code.co_name,
                caller_frame.f_lineno,
                caller_frame.f_code.co_filename
            )
            
            func_name = f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            # Prepare parameters for logging
            params = {}
            if log_params:
                try:
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()
                    params = dict(bound_args.arguments)
                except Exception:
                    params = {'args': args, 'kwargs': kwargs}
            
            # Log function entry
            verbose_logger.trace_function_entry(func_name, params, caller_info)
            
            try:
                # Execute async function
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                # Log successful exit
                if log_returns:
                    verbose_logger.trace_function_exit(func_name, result, duration_ms, caller_info)
                else:
                    verbose_logger.trace_function_exit(func_name, "[RETURN_NOT_LOGGED]", duration_ms, caller_info)
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                verbose_logger.trace_exception(func_name, e, duration_ms, caller_info)
                raise
        
        return wrapper
    return decorator


# Global verbose logger instance
_global_verbose_logger: Optional[VerboseLogger] = None
_verbose_logger_lock = threading.Lock()


def get_verbose_logger() -> VerboseLogger:
    """Get global verbose logger instance"""
    global _global_verbose_logger
    
    if _global_verbose_logger is None:
        with _verbose_logger_lock:
            if _global_verbose_logger is None:
                _global_verbose_logger = VerboseLogger()
    
    return _global_verbose_logger


def setup_verbose_logging(config_path: str = "config/logging_config.json") -> VerboseLogger:
    """Setup and configure verbose logging"""
    global _global_verbose_logger
    
    with _verbose_logger_lock:
        _global_verbose_logger = VerboseLogger(config_path=config_path)
    
    return _global_verbose_logger


# Convenience functions for common logging patterns
def log_startup_sequence(app_name: str, version: str):
    """Log application startup sequence"""
    logger = get_verbose_logger()
    logger.base_logger.info(
        f"🚀 Starting {app_name} v{version}",
        category=LogCategory.SYSTEM,
        tags=["startup", "application"]
    )

def log_shutdown_sequence(app_name: str):
    """Log application shutdown sequence"""
    logger = get_verbose_logger()
    logger.base_logger.info(
        f"🛑 Shutting down {app_name}",
        category=LogCategory.SYSTEM,
        tags=["shutdown", "application"]
    )

def log_configuration_loaded(config_name: str, config_data: Dict[str, Any]):
    """Log configuration loading"""
    logger = get_verbose_logger()
    logger.base_logger.info(
        f"⚙️  Loaded configuration: {config_name}",
        category=LogCategory.SYSTEM,
        extra_data={"config_name": config_name, "config_keys": list(config_data.keys())},
        tags=["configuration", "startup"]
    )

def log_database_connection(database_type: str, status: str, details: Optional[str] = None):
    """Log database connection status"""
    logger = get_verbose_logger()
    emoji = "✅" if status == "connected" else "❌"
    logger.base_logger.info(
        f"{emoji} Database {database_type}: {status}",
        category=LogCategory.DATABASE,
        extra_data={"database_type": database_type, "status": status, "details": details},
        tags=["database", "connection"]
    )

def log_api_request(method: str, url: str, status_code: int, duration_ms: float):
    """Log API request with timing"""
    logger = get_verbose_logger()
    status_emoji = "✅" if 200 <= status_code < 400 else "❌"
    logger.base_logger.info(
        f"{status_emoji} {method} {url} [{status_code}] [{duration_ms:.2f}ms]",
        category=LogCategory.API,
        extra_data={
            "method": method,
            "url": url,
            "status_code": status_code,
            "duration_ms": duration_ms
        },
        performance=PerformanceMetrics(execution_time_ms=duration_ms),
        tags=["api", "request"]
    )

# Module initialization
verbose_logger = get_verbose_logger()