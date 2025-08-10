"""
Bloomberg Terminal主题

提供专业的Bloomberg风格配色方案和UI主题
"""

from rich.theme import Theme
from rich.style import Style
from textual.theme import Theme as TextualTheme
from typing import Dict, Any

class BloombergTheme:
    """Bloomberg Terminal主题配置"""
    
    # Bloomberg经典配色方案
    COLORS = {
        # 主色系
        "bg_primary": "#0D1B2A",      # 深海蓝背景
        "bg_secondary": "#1B263B",    # 中等蓝面板
        "bg_tertiary": "#415A77",     # 浅蓝边框
        
        # 文本色系
        "text_primary": "#E0E1DD",    # 主要文本（米白）
        "text_secondary": "#9DB4C0",  # 次要文本（浅蓝灰）
        "text_highlight": "#52B788",  # 高亮文本（翠绿）
        "text_brand": "#277DA1",      # 品牌蓝
        
        # 功能色系
        "success": "#52B788",         # 成功/盈利（翠绿）
        "profit": "#2D6A4F",          # 上涨（深绿）
        "warning": "#F2CC8F",         # 警告/暂停（金黄）
        "danger": "#E07A5F",          # 错误/亏损（暖橙红）
        "loss": "#C1121F",            # 下跌（深红）
        "ai_active": "#7209B7",       # AI活跃（紫色）
        
        # 中性色系
        "gray_dark": "#2D3748",       # 深灰
        "gray_medium": "#4A5568",     # 中灰
        "gray_light": "#718096",      # 浅灰
        "gray_lighter": "#CBD5E0",    # 极浅灰
        
        # 特殊用途
        "border": "#415A77",          # 边框色
        "focus": "#277DA1",           # 聚焦色
        "selection": "#52B788",       # 选择色
    }
    
    @classmethod
    def get_theme(cls) -> Theme:
        """获取Rich主题"""
        return Theme({
            # 基础样式
            "default": f"color {cls.COLORS['text_primary']} on {cls.COLORS['bg_primary']}",
            "primary": f"bold {cls.COLORS['text_primary']}",
            "secondary": f"{cls.COLORS['text_secondary']}",
            "highlight": f"bold {cls.COLORS['text_highlight']}",
            "brand": f"bold {cls.COLORS['text_brand']}",
            
            # 状态样式
            "success": f"bold {cls.COLORS['success']}",
            "profit": f"bold {cls.COLORS['profit']}",
            "warning": f"bold {cls.COLORS['warning']}",
            "danger": f"bold {cls.COLORS['danger']}",
            "loss": f"bold {cls.COLORS['loss']}",
            "ai": f"bold {cls.COLORS['ai_active']}",
            
            # UI组件样式
            "panel": f"{cls.COLORS['text_primary']} on {cls.COLORS['bg_secondary']}",
            "panel_title": f"bold {cls.COLORS['text_highlight']}",
            "border": f"{cls.COLORS['border']}",
            "focus": f"bold {cls.COLORS['focus']}",
            "selection": f"bold {cls.COLORS['selection']} on {cls.COLORS['bg_secondary']}",
            
            # 数据样式
            "price": f"bold {cls.COLORS['text_primary']}",
            "price_up": f"bold {cls.COLORS['profit']}",
            "price_down": f"bold {cls.COLORS['loss']}",
            "volume": f"{cls.COLORS['text_secondary']}",
            "percentage": f"bold {cls.COLORS['text_highlight']}",
            
            # 表格样式
            "table_header": f"bold {cls.COLORS['text_brand']} on {cls.COLORS['bg_secondary']}",
            "table_row": f"{cls.COLORS['text_primary']}",
            "table_row_alt": f"{cls.COLORS['text_primary']} on {cls.COLORS['gray_dark']}",
            
            # 按钮样式
            "button": f"bold {cls.COLORS['text_primary']} on {cls.COLORS['bg_secondary']}",
            "button_hover": f"bold {cls.COLORS['text_primary']} on {cls.COLORS['focus']}",
            "button_active": f"bold {cls.COLORS['bg_primary']} on {cls.COLORS['selection']}",
            
            # 输入框样式
            "input": f"{cls.COLORS['text_primary']} on {cls.COLORS['bg_secondary']}",
            "input_focus": f"{cls.COLORS['text_primary']} on {cls.COLORS['bg_secondary']} underline",
            
            # 日志样式
            "log_info": f"{cls.COLORS['text_secondary']}",
            "log_success": f"{cls.COLORS['success']}",
            "log_warning": f"{cls.COLORS['warning']}",
            "log_error": f"{cls.COLORS['danger']}",
            "log_timestamp": f"dim {cls.COLORS['text_secondary']}",
            
            # 图表样式
            "chart_axis": f"{cls.COLORS['text_secondary']}",
            "chart_line": f"{cls.COLORS['text_highlight']}",
            "chart_candle_up": f"{cls.COLORS['profit']}",
            "chart_candle_down": f"{cls.COLORS['loss']}",
            
            # 状态栏样式
            "status_bar": f"{cls.COLORS['text_primary']} on {cls.COLORS['bg_secondary']}",
            "status_connection": f"bold {cls.COLORS['success']}",
            "status_disconnection": f"bold {cls.COLORS['danger']}",
            "status_time": f"{cls.COLORS['text_secondary']}",
            
            # 快捷键样式
            "shortcut": f"bold {cls.COLORS['text_brand']}",
            "shortcut_desc": f"{cls.COLORS['text_secondary']}",
            "shortcut_exit": f"bold {cls.COLORS['danger']}",
        })
    
    @classmethod
    def get_textual_theme(cls) -> TextualTheme:
        """获取Textual主题"""
        return TextualTheme(
            name="bloomberg",
            primary=cls.COLORS["text_brand"],
            secondary=cls.COLORS["text_secondary"],
            accent=cls.COLORS["text_highlight"],
            foreground=cls.COLORS["text_primary"],
            background=cls.COLORS["bg_primary"],
            surface=cls.COLORS["bg_secondary"],
            panel=cls.COLORS["bg_tertiary"],
            boost=cls.COLORS["selection"],
            warning=cls.COLORS["warning"],
            error=cls.COLORS["danger"],
            success=cls.COLORS["success"],
            dark=True,
        )
    
    @classmethod
    def get_css_variables(cls) -> Dict[str, str]:
        """获取CSS变量定义"""
        return {f"--{key.replace('_', '-')}": value for key, value in cls.COLORS.items()}
    
    @classmethod
    def get_component_styles(cls) -> Dict[str, Dict[str, Any]]:
        """获取组件样式定义"""
        return {
            # 主容器样式
            "Screen": {
                "background": cls.COLORS["bg_primary"],
                "color": cls.COLORS["text_primary"],
            },
            
            # 面板样式
            "Static": {
                "background": "transparent",
                "color": cls.COLORS["text_primary"],
            },
            
            "Container": {
                "background": "transparent",
            },
            
            # 输入组件样式
            "Input": {
                "background": cls.COLORS["bg_secondary"],
                "color": cls.COLORS["text_primary"],
                "border": f"solid {cls.COLORS['border']}",
                "border-title-color": cls.COLORS["text_brand"],
            },
            
            "Input:focus": {
                "border": f"solid {cls.COLORS['focus']}",
                "background": cls.COLORS["bg_secondary"],
            },
            
            # 按钮样式
            "Button": {
                "background": cls.COLORS["bg_secondary"],
                "color": cls.COLORS["text_primary"],
                "border": f"solid {cls.COLORS['border']}",
            },
            
            "Button:hover": {
                "background": cls.COLORS["focus"],
                "border": f"solid {cls.COLORS['focus']}",
            },
            
            "Button:focus": {
                "background": cls.COLORS["selection"],
                "color": cls.COLORS["bg_primary"],
                "border": f"solid {cls.COLORS['selection']}",
            },
            
            # 表格样式
            "DataTable": {
                "background": cls.COLORS["bg_primary"],
                "color": cls.COLORS["text_primary"],
                "scrollbar-background": cls.COLORS["bg_secondary"],
                "scrollbar-color": cls.COLORS["border"],
            },
            
            "DataTable > .datatable--header": {
                "background": cls.COLORS["bg_secondary"],
                "color": cls.COLORS["text_brand"],
                "text-style": "bold",
            },
            
            "DataTable > .datatable--cursor": {
                "background": cls.COLORS["selection"],
                "color": cls.COLORS["bg_primary"],
            },
            
            # 选项卡样式
            "TabbedContent": {
                "background": cls.COLORS["bg_primary"],
            },
            
            "Tabs": {
                "background": cls.COLORS["bg_secondary"],
            },
            
            "Tab": {
                "background": "transparent",
                "color": cls.COLORS["text_secondary"],
                "border": "none",
            },
            
            "Tab:hover": {
                "background": cls.COLORS["bg_tertiary"],
                "color": cls.COLORS["text_primary"],
            },
            
            "Tab.-active": {
                "background": cls.COLORS["focus"],
                "color": cls.COLORS["text_primary"],
                "text-style": "bold",
            },
            
            # 进度条样式
            "ProgressBar": {
                "background": cls.COLORS["bg_secondary"],
                "color": cls.COLORS["success"],
                "bar-color": cls.COLORS["success"],
            },
            
            # 日志样式
            "Log": {
                "background": cls.COLORS["bg_primary"],
                "color": cls.COLORS["text_primary"],
                "scrollbar-background": cls.COLORS["bg_secondary"],
                "scrollbar-color": cls.COLORS["border"],
            },
            
            # 选择器样式
            "Select": {
                "background": cls.COLORS["bg_secondary"],
                "color": cls.COLORS["text_primary"],
                "border": f"solid {cls.COLORS['border']}",
            },
            
            "Select:focus": {
                "border": f"solid {cls.COLORS['focus']}",
            },
            
            "OptionList": {
                "background": cls.COLORS["bg_secondary"],
                "color": cls.COLORS["text_primary"],
                "scrollbar-background": cls.COLORS["bg_tertiary"],
                "scrollbar-color": cls.COLORS["border"],
            },
            
            "OptionList > .option-list--option": {
                "background": "transparent",
                "color": cls.COLORS["text_primary"],
            },
            
            "OptionList > .option-list--option-highlighted": {
                "background": cls.COLORS["focus"],
                "color": cls.COLORS["text_primary"],
            },
            
            "OptionList > .option-list--option-selected": {
                "background": cls.COLORS["selection"],
                "color": cls.COLORS["bg_primary"],
                "text-style": "bold",
            },
        }
    
    @classmethod
    def generate_css(cls) -> str:
        """生成完整的CSS样式表"""
        css_vars = "\n".join([f"    {key}: {value};" for key, value in cls.get_css_variables().items()])
        
        css_content = f"""
/* Bloomberg Terminal主题 - 自动生成 */
:root {{
{css_vars}
}}

/* 全局样式 */
Screen {{
    background: $bg-primary;
    color: $text-primary;
    layout: vertical;
}}

/* 主容器 */
#main-container {{
    height: 100%;
    layout: vertical;
}}

/* 状态栏 */
#status-bar {{
    dock: top;
    height: 3;
    background: $bg-secondary;
    color: $text-primary;
    content-align: left middle;
    padding: 0 2;
}}

/* 内容区域 */
#content-area {{
    height: 1fr;
    background: $bg-primary;
    padding: 1;
}}

/* 快捷键栏 */
#shortcut-bar {{
    dock: bottom;
    height: 3;
    background: $bg-secondary;
    layout: horizontal;
    align: center middle;
}}

.shortcut-key {{
    margin: 0 2;
    color: $text-brand;
    text-style: bold;
}}

.exit-key {{
    color: $danger;
}}

/* 面板样式 */
.panel {{
    background: $bg-secondary;
    border: solid $border;
    border-title-color: $text-highlight;
    padding: 1;
}}

.panel-header {{
    background: $bg-tertiary;
    color: $text-brand;
    text-style: bold;
    height: 3;
    content-align: center middle;
}}

/* 数据表格 */
.data-table {{
    background: $bg-primary;
    scrollbar-background: $bg-secondary;
    scrollbar-color: $border;
}}

.table-header {{
    background: $bg-secondary;
    color: $text-brand;
    text-style: bold;
}}

.table-row-profit {{
    color: $profit;
    text-style: bold;
}}

.table-row-loss {{
    color: $loss;
    text-style: bold;
}}

/* 图表区域 */
.chart-container {{
    background: $bg-primary;
    border: solid $border;
    padding: 1;
}}

.chart-title {{
    color: $text-highlight;
    text-style: bold;
    text-align: center;
}}

/* AI相关样式 */
.ai-status {{
    color: $ai-active;
    text-style: bold;
}}

.ai-thinking {{
    color: $ai-active;
    text-style: italic;
}}

/* 连接状态 */
.connection-ok {{
    color: $success;
    text-style: bold;
}}

.connection-error {{
    color: $danger;
    text-style: bold;
}}

/* 动画效果 */
.pulse {{
    text-style: bold;
}}

.blink {{
    text-style: blink;
}}

/* 响应式布局 */
@media (max-width: 120) {{
    #shortcut-bar {{
        display: none;
    }}
    
    .panel {{
        padding: 0;
    }}
}}

/* 高对比度模式 */
.high-contrast {{
    background: #000000;
    color: #FFFFFF;
}}

.high-contrast .panel {{
    border: solid #FFFFFF;
    background: #222222;
}}
"""
        
        return css_content
    
    @classmethod
    def get_color_palette(cls) -> Dict[str, str]:
        """获取颜色调色板（用于图表等）"""
        return {
            "primary_series": [
                cls.COLORS["text_highlight"],
                cls.COLORS["text_brand"], 
                cls.COLORS["ai_active"],
                cls.COLORS["warning"],
                cls.COLORS["success"]
            ],
            "profit_loss": [cls.COLORS["profit"], cls.COLORS["loss"]],
            "status": [
                cls.COLORS["success"],
                cls.COLORS["warning"],
                cls.COLORS["danger"]
            ],
            "gradients": {
                "profit": [cls.COLORS["success"], cls.COLORS["profit"]],
                "loss": [cls.COLORS["danger"], cls.COLORS["loss"]],
                "neutral": [cls.COLORS["text_secondary"], cls.COLORS["text_primary"]]
            }
        }

# 便捷函数
def get_bloomberg_theme() -> Theme:
    """获取Bloomberg主题（便捷函数）"""
    return BloombergTheme.get_theme()

def get_bloomberg_textual_theme() -> TextualTheme:
    """获取Bloomberg Textual主题（便捷函数）"""
    return BloombergTheme.get_textual_theme()

def generate_bloomberg_css(output_path: str = None) -> str:
    """生成Bloomberg CSS文件"""
    css_content = BloombergTheme.generate_css()
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(css_content)
    
    return css_content