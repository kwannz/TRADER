"""
Unified Factor Engine
统一因子引擎 - 融合PandaFactor传统算子与AI增强功能
"""

# Traditional components
from .traditional.factor_utils import PandaFactorUtils, panda_factor_utils
from .traditional.factor_base import (
    BaseFactor, FactorSeries, FormulaFactor, PythonFactor, 
    FactorLibrary, factor_library
)

# Unified interface
from .unified_interface.factor_interface import UnifiedFactorInterface, unified_interface

# AI enhanced components (to be integrated in future phases)
# from .ai_enhanced.ai_factor_discovery import AIFactorDiscovery

__all__ = [
    # Traditional components
    'PandaFactorUtils',
    'panda_factor_utils',
    'BaseFactor',
    'FactorSeries',
    'FormulaFactor',
    'PythonFactor',
    'FactorLibrary',
    'factor_library',
    
    # Unified interface
    'UnifiedFactorInterface',
    'unified_interface',
    
    # AI enhanced (future)
    # 'AIFactorDiscovery'
]

__version__ = "1.0.0"