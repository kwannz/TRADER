"""
认证中间件
处理JWT令牌验证和用户认证
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import jwt
import os
from datetime import datetime

class AuthMiddleware(BaseHTTPMiddleware):
    """认证中间件"""
    
    def __init__(self, app):
        super().__init__(app)
        self.secret_key = os.getenv("JWT_SECRET_KEY", "quantum_trader_secret_key_2025")
        self.algorithm = "HS256"
        
        # 不需要认证的路径
        self.public_paths = {
            "/",
            "/docs",
            "/redoc", 
            "/openapi.json",
            "/health",
            "/api/v1/auth/login",
            "/api/v1/market/latest",
            "/api/v1/market/coinglass",
            "/api/v1/market/health",
            "/api/v1/market/stream-status",
            "/api/v1/ai/sentiment",
            "/api/v1/ai/signals",
            "/api/v1/system/info",
            "/api/v1/system/status",
            "/api/v1/system/health",
            "/api/v1/system/metrics"
        }
    
    async def dispatch(self, request: Request, call_next):
        """处理请求"""
        # 跳过公共路径
        if request.url.path in self.public_paths:
            return await call_next(request)
        
        # 跳过WebSocket连接
        if request.headers.get("upgrade") == "websocket":
            return await call_next(request)
        
        # 跳过OPTIONS请求（CORS预检）
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # 检查Authorization头
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={
                    "success": False,
                    "error": {
                        "type": "authentication_required",
                        "message": "需要认证令牌"
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        # 提取令牌
        token = auth_header.split(" ")[1]
        
        try:
            # 验证JWT令牌
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username = payload.get("sub")
            
            if not username:
                raise jwt.InvalidTokenError("令牌中缺少用户信息")
            
            # 将用户信息添加到请求状态中
            request.state.current_user = username
            
            # 继续处理请求
            response = await call_next(request)
            return response
            
        except jwt.ExpiredSignatureError:
            return JSONResponse(
                status_code=401,
                content={
                    "success": False,
                    "error": {
                        "type": "token_expired",
                        "message": "认证令牌已过期"
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        except jwt.InvalidTokenError:
            return JSONResponse(
                status_code=401,
                content={
                    "success": False,
                    "error": {
                        "type": "invalid_token",
                        "message": "无效的认证令牌"
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error": {
                        "type": "auth_error",
                        "message": f"认证处理错误: {str(e)}"
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
            )