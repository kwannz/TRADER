"""
自定义CORS中间件
处理跨域请求
"""

from fastapi import Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import PlainTextResponse

class CustomCORSMiddleware(BaseHTTPMiddleware):
    """自定义CORS中间件"""
    
    def __init__(
        self,
        app,
        allow_origins=None,
        allow_methods=None,
        allow_headers=None,
        allow_credentials=False,
        expose_headers=None,
        max_age=600
    ):
        super().__init__(app)
        self.allow_origins = allow_origins or ["*"]
        self.allow_methods = allow_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"]
        self.allow_headers = allow_headers or ["*"]
        self.allow_credentials = allow_credentials
        self.expose_headers = expose_headers or []
        self.max_age = max_age
    
    async def dispatch(self, request: Request, call_next):
        """处理CORS"""
        origin = request.headers.get("origin")
        
        # 检查Origin是否被允许
        if origin and self.allow_origins != ["*"]:
            if origin not in self.allow_origins:
                return PlainTextResponse(
                    "CORS policy violation", 
                    status_code=403
                )
        
        # 处理预检请求
        if request.method == "OPTIONS":
            response = Response()
            self._add_cors_headers(response, origin)
            return response
        
        # 处理正常请求
        response = await call_next(request)
        self._add_cors_headers(response, origin)
        
        return response
    
    def _add_cors_headers(self, response: Response, origin: str = None):
        """添加CORS响应头"""
        if origin and self.allow_origins != ["*"]:
            if origin in self.allow_origins:
                response.headers["Access-Control-Allow-Origin"] = origin
        else:
            response.headers["Access-Control-Allow-Origin"] = "*"
        
        if self.allow_credentials:
            response.headers["Access-Control-Allow-Credentials"] = "true"
        
        response.headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)
        response.headers["Access-Control-Allow-Headers"] = ", ".join(self.allow_headers)
        
        if self.expose_headers:
            response.headers["Access-Control-Expose-Headers"] = ", ".join(self.expose_headers)
        
        response.headers["Access-Control-Max-Age"] = str(self.max_age)