"""
日志中间件
记录API请求和响应信息
"""

import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    """日志中间件"""
    
    def __init__(self, app):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        """处理请求并记录日志"""
        # 生成请求ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # 记录请求开始时间
        start_time = time.time()
        
        # 获取客户端信息
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")
        
        # 记录请求开始
        logger.info(
            f"Request started - {request.method} {request.url.path} | "
            f"Client: {client_ip} | Request ID: {request_id}"
        )
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 记录请求完成
            logger.info(
                f"Request completed - {request.method} {request.url.path} | "
                f"Status: {response.status_code} | "
                f"Duration: {process_time:.3f}s | "
                f"Request ID: {request_id}"
            )
            
            # 添加响应头
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(round(process_time, 3))
            
            return response
            
        except Exception as e:
            # 计算处理时间
            process_time = time.time() - start_time
            
            # 记录错误
            logger.error(
                f"Request failed - {request.method} {request.url.path} | "
                f"Error: {str(e)} | "
                f"Duration: {process_time:.3f}s | "
                f"Request ID: {request_id}",
                exc_info=True
            )
            
            # 重新抛出异常
            raise