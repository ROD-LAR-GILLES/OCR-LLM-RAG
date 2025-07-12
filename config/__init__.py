# config/__init__.py
"""
Módulo de configuración para el sistema OCR mejorado.
"""

from .enhanced_config import (
    EnhancedSystemConfig,
    OCRConfig,
    PreprocessingConfig,
    QualityConfig,
    ProcessingConfig,
    QUALITY_PROFILES,
    load_config_from_file,
    save_config_to_file
)

__all__ = [
    'EnhancedSystemConfig',
    'OCRConfig',
    'PreprocessingConfig',
    'QualityConfig',
    'ProcessingConfig',
    'QUALITY_PROFILES',
    'load_config_from_file',
    'save_config_to_file'
]
