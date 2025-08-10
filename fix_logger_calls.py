#!/usr/bin/env python3
"""
批量修复logger调用脚本
"""

import os
import re
from pathlib import Path

def fix_logger_calls(file_path):
    """修复单个文件中的logger调用"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 修复 get_logger(__name__) 调用
        content = re.sub(r'get_logger\(__name__\)', 'get_logger()', content)
        
        # 检查是否有统一logger导入，如果没有则替换导入
        if 'from .unified_logger import' not in content and 'from core.unified_logger import' not in content:
            # 替换旧的logger导入
            imports_to_replace = [
                (r'from \.\.python_layer\.utils\.logger import get_logger', 'from .unified_logger import get_logger, LogCategory'),
                (r'from python_layer\.utils\.logger import get_logger', 'from core.unified_logger import get_logger, LogCategory'),
                (r'from \.\.utils\.logger import get_logger', 'from .unified_logger import get_logger, LogCategory'),
                (r'from utils\.logger import get_logger', 'from core.unified_logger import get_logger, LogCategory'),
            ]
            
            for old_import, new_import in imports_to_replace:
                content = re.sub(old_import, new_import, content)
        
        # 如果内容有变化，写回文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"修复文件 {file_path} 时出错: {e}")
        return False

def main():
    """主函数"""
    core_dir = Path('/Users/zhaoleon/Desktop/trader/core')
    fixed_files = []
    
    # 递归查找所有Python文件
    for py_file in core_dir.rglob('*.py'):
        if fix_logger_calls(py_file):
            fixed_files.append(py_file)
            print(f"✅ 修复文件: {py_file}")
    
    if fixed_files:
        print(f"\n🎉 总共修复了 {len(fixed_files)} 个文件")
    else:
        print("\n📝 没有找到需要修复的文件")

if __name__ == '__main__':
    main()