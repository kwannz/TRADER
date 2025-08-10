"""
认证依赖注入
提供当前用户信息
"""

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import os

security = HTTPBearer()

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "quantum_trader_secret_key_2025")
ALGORITHM = "HS256"

def get_current_user(request: Request) -> str:
    """从请求状态获取当前用户"""
    if hasattr(request.state, 'current_user'):
        return request.state.current_user
    
    raise HTTPException(
        status_code=401,
        detail="用户未认证"
    )

def get_current_user_optional(request: Request) -> str:
    """获取当前用户（可选）"""
    if hasattr(request.state, 'current_user'):
        return request.state.current_user
    return None

async def verify_admin_user(current_user: str = Depends(get_current_user)) -> str:
    """验证管理员用户"""
    # 这里可以添加管理员权限检查逻辑
    admin_users = ["admin"]  # 可以从数据库或配置获取
    
    if current_user not in admin_users:
        raise HTTPException(
            status_code=403,
            detail="需要管理员权限"
        )
    
    return current_user

def verify_token_manually(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """手动验证JWT令牌（用于特殊情况）"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        
        if username is None:
            raise HTTPException(
                status_code=401,
                detail="无效的认证凭据"
            )
        
        return username
        
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=401,
            detail="无效的认证凭据"
        )