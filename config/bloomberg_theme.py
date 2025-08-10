"""
Bloomberg Terminal风格主题配置
深蓝色系专业金融终端配色方案
"""

from rich.theme import Theme
from rich.color import Color
from textual.design import ColorSystem

# Bloomberg深蓝色系配色
BLOOMBERG_COLORS = {
    # 主色调 (Bloomberg深蓝系)
    "primary_bg": "#0D1B2A",        # 深海蓝，专业沉稳
    "secondary_bg": "#1B263B",      # 中等蓝，面板背景
    "border_color": "#415A77",      # 蓝灰，边框分隔
    
    # 辅助色彩 (功能性配色)
    "text_primary": "#E0E1DD",      # 米白，主要文字
    "text_secondary": "#9DB4C0",    # 浅蓝灰，次要信息
    "text_emphasis": "#52B788",     # 翠绿，重要数据
    "brand_blue": "#277DA1",        # 彭博蓝，品牌识别
    
    # 功能色彩 (状态指示)
    "success": "#52B788",           # 翠绿 - 成功/盈利
    "profit": "#2D6A4F",           # 深绿 - 上涨趋势
    "warning": "#F2CC8F",          # 金黄 - 警告/暂停
    "danger": "#E07A5F",           # 暖橙红 - 错误/亏损
    "loss": "#C1121F",             # 深红 - 下跌趋势
    "ai_active": "#7209B7",        # 紫色 - AI活跃状态
    
    # 中性色 (灰度系统)
    "dark_gray": "#2D3748",        # 卡片背景
    "medium_gray": "#4A5568",      # 禁用状态
    "light_gray": "#718096",       # 辅助线条
    "ultra_light": "#CBD5E0",      # 分割线
}

# Rich主题配置
BLOOMBERG_RICH_THEME = Theme({
    # 基础样式
    "default": f"{BLOOMBERG_COLORS['text_primary']} on {BLOOMBERG_COLORS['primary_bg']}",
    "primary": f"bold {BLOOMBERG_COLORS['brand_blue']}",
    "secondary": f"{BLOOMBERG_COLORS['text_secondary']}",
    "emphasis": f"bold {BLOOMBERG_COLORS['text_emphasis']}",
    
    # 状态样式
    "success": f"bold {BLOOMBERG_COLORS['success']}",
    "profit": f"bold {BLOOMBERG_COLORS['profit']}",
    "warning": f"bold {BLOOMBERG_COLORS['warning']}",
    "danger": f"bold {BLOOMBERG_COLORS['danger']}",
    "loss": f"bold {BLOOMBERG_COLORS['loss']}",
    "ai": f"bold {BLOOMBERG_COLORS['ai_active']}",
    
    # UI组件样式
    "panel_title": f"bold underline {BLOOMBERG_COLORS['brand_blue']}",
    "panel_border": f"{BLOOMBERG_COLORS['border_color']}",
    "table_header": f"bold {BLOOMBERG_COLORS['text_emphasis']} on {BLOOMBERG_COLORS['secondary_bg']}",
    "table_row_even": f"{BLOOMBERG_COLORS['text_primary']} on {BLOOMBERG_COLORS['primary_bg']}",
    "table_row_odd": f"{BLOOMBERG_COLORS['text_primary']} on {BLOOMBERG_COLORS['secondary_bg']}",
    
    # 数据显示样式
    "price_up": f"bold {BLOOMBERG_COLORS['profit']}",
    "price_down": f"bold {BLOOMBERG_COLORS['loss']}",
    "price_neutral": f"{BLOOMBERG_COLORS['text_primary']}",
    "volume": f"{BLOOMBERG_COLORS['text_secondary']}",
    "percentage": f"bold {BLOOMBERG_COLORS['text_emphasis']}",
    
    # 策略状态
    "strategy_running": f"bold {BLOOMBERG_COLORS['success']}",
    "strategy_paused": f"bold {BLOOMBERG_COLORS['warning']}",
    "strategy_stopped": f"bold {BLOOMBERG_COLORS['danger']}",
    
    # 日志样式
    "log_info": f"{BLOOMBERG_COLORS['text_primary']}",
    "log_warning": f"{BLOOMBERG_COLORS['warning']}",
    "log_error": f"bold {BLOOMBERG_COLORS['danger']}",
    "log_success": f"{BLOOMBERG_COLORS['success']}",
    "log_ai": f"{BLOOMBERG_COLORS['ai_active']}",
    
    # 按钮样式
    "button_primary": f"bold {BLOOMBERG_COLORS['primary_bg']} on {BLOOMBERG_COLORS['brand_blue']}",
    "button_secondary": f"bold {BLOOMBERG_COLORS['text_primary']} on {BLOOMBERG_COLORS['border_color']}",
    "button_danger": f"bold {BLOOMBERG_COLORS['primary_bg']} on {BLOOMBERG_COLORS['danger']}",
    
    # 输入框样式
    "input_field": f"{BLOOMBERG_COLORS['text_primary']} on {BLOOMBERG_COLORS['secondary_bg']}",
    "input_border": f"{BLOOMBERG_COLORS['border_color']}",
    "input_focus": f"{BLOOMBERG_COLORS['brand_blue']}",
})

# Textual色彩系统配置
BLOOMBERG_COLOR_SYSTEM = ColorSystem(
    primary=BLOOMBERG_COLORS["brand_blue"],
    secondary=BLOOMBERG_COLORS["border_color"],
    accent=BLOOMBERG_COLORS["text_emphasis"],
    foreground=BLOOMBERG_COLORS["text_primary"],
    background=BLOOMBERG_COLORS["primary_bg"],
    surface=BLOOMBERG_COLORS["secondary_bg"],
    panel=BLOOMBERG_COLORS["secondary_bg"],
    boost=BLOOMBERG_COLORS["success"],
    warning=BLOOMBERG_COLORS["warning"],
    error=BLOOMBERG_COLORS["danger"],
    success=BLOOMBERG_COLORS["success"],
    dark=True,
)

# ASCII艺术样式
ASCII_ART_STYLES = {
    "logo": f"bold color({BLOOMBERG_COLORS['brand_blue']})",
    "separator": f"color({BLOOMBERG_COLORS['border_color']})",
    "chart_line": f"color({BLOOMBERG_COLORS['text_emphasis']})",
    "chart_up": f"color({BLOOMBERG_COLORS['profit']})",
    "chart_down": f"color({BLOOMBERG_COLORS['loss']})",
}

# 图表配置
CHART_CONFIG = {
    "background_color": BLOOMBERG_COLORS["primary_bg"],
    "grid_color": BLOOMBERG_COLORS["border_color"],
    "text_color": BLOOMBERG_COLORS["text_primary"],
    "profit_line": BLOOMBERG_COLORS["profit"],
    "loss_line": BLOOMBERG_COLORS["loss"],
    "neutral_line": BLOOMBERG_COLORS["text_secondary"],
    "volume_bar": BLOOMBERG_COLORS["text_secondary"],
}

# 状态指示器配置
STATUS_INDICATORS = {
    "connected": "✅",
    "disconnected": "❌", 
    "reconnecting": "🔄",
    "running": "🟢",
    "paused": "🟡",
    "stopped": "🔴",
    "ai_thinking": "🧠",
    "profit": "📈",
    "loss": "📉",
    "warning": "⚠️",
    "info": "ℹ️",
}

# 快捷键提示样式
SHORTCUT_STYLES = {
    "key": f"bold color({BLOOMBERG_COLORS['text_emphasis']}) on color({BLOOMBERG_COLORS['secondary_bg']})",
    "description": f"color({BLOOMBERG_COLORS['text_secondary']})",
    "separator": f"color({BLOOMBERG_COLORS['border_color']})",
}

def get_theme():
    """获取Bloomberg主题配置"""
    return BLOOMBERG_RICH_THEME

def get_color_system():
    """获取Textual色彩系统"""
    return BLOOMBERG_COLOR_SYSTEM

def get_color(name: str) -> str:
    """根据名称获取颜色值"""
    return BLOOMBERG_COLORS.get(name, BLOOMBERG_COLORS["text_primary"])

def get_status_color(status: str) -> str:
    """根据状态获取对应颜色"""
    status_map = {
        "success": "success",
        "running": "success", 
        "connected": "success",
        "profit": "profit",
        "warning": "warning",
        "paused": "warning",
        "error": "danger",
        "danger": "danger",
        "stopped": "danger",
        "loss": "loss",
        "ai": "ai_active",
        "info": "text_secondary",
    }
    color_key = status_map.get(status.lower(), "text_primary")
    return BLOOMBERG_COLORS[color_key]

# 导出配置
__all__ = [
    "BLOOMBERG_COLORS",
    "BLOOMBERG_RICH_THEME", 
    "BLOOMBERG_COLOR_SYSTEM",
    "ASCII_ART_STYLES",
    "CHART_CONFIG",
    "STATUS_INDICATORS",
    "SHORTCUT_STYLES",
    "get_theme",
    "get_color_system", 
    "get_color",
    "get_status_color",
]