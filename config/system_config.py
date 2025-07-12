# config/enhanced_config.py
"""
Configuraciones para el sistema OCR mejorado.

Este módulo centraliza todas las configuraciones del sistema mejorado,
incluyendo parámetros de calidad, preprocesamiento y métricas.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path


@dataclass
class OCRConfig:
    """Configuración para el motor de OCR."""
    language: str = "spa"
    dpi: int = 300
    confidence_threshold: float = 60.0
    enable_preprocessing: bool = True
    enable_quality_analysis: bool = True
    max_retry_attempts: int = 2


@dataclass
class PreprocessingConfig:
    """Configuración para el preprocesamiento de imágenes."""
    enable_skew_correction: bool = True
    enable_noise_reduction: bool = True
    enable_contrast_enhancement: bool = True
    enable_adaptive_binarization: bool = True
    quality_threshold_for_intensive_processing: float = 0.7


@dataclass
class QualityConfig:
    """Configuración para análisis de calidad."""
    min_acceptable_quality: float = 60.0
    high_quality_threshold: float = 80.0
    enable_detailed_analysis: bool = True
    generate_quality_reports: bool = True
    export_confidence_maps: bool = False


@dataclass
class ProcessingConfig:
    """Configuración general de procesamiento."""
    enable_auto_retry: bool = True
    parallel_processing: bool = False  # Para futuras implementaciones
    max_processing_time_minutes: int = 30
    preserve_intermediate_files: bool = False
    enable_table_extraction: bool = True  # Nueva opción para controlar extracción de tablas
    table_quality_threshold: float = 0.5  # Umbral de calidad para incluir tablas


@dataclass
class EnhancedSystemConfig:
    """Configuración completa del sistema mejorado."""
    ocr: OCRConfig = None
    preprocessing: PreprocessingConfig = None
    quality: QualityConfig = None
    processing: ProcessingConfig = None
    
    def __post_init__(self):
        """Inicializa configuraciones por defecto si no se proporcionan."""
        if self.ocr is None:
            self.ocr = OCRConfig()
        if self.preprocessing is None:
            self.preprocessing = PreprocessingConfig()
        if self.quality is None:
            self.quality = QualityConfig()
        if self.processing is None:
            self.processing = ProcessingConfig()
    
    @classmethod
    def create_high_quality_config(cls) -> 'EnhancedSystemConfig':
        """
        Crea una configuración optimizada para máxima calidad.
        
        Returns:
            EnhancedSystemConfig: Configuración para alta calidad
        """
        return cls(
            ocr=OCRConfig(
                dpi=600,
                confidence_threshold=80.0,
                enable_preprocessing=True,
                enable_quality_analysis=True,
                max_retry_attempts=3
            ),
            preprocessing=PreprocessingConfig(
                enable_skew_correction=True,
                enable_noise_reduction=True,
                enable_contrast_enhancement=True,
                enable_adaptive_binarization=True,
                quality_threshold_for_intensive_processing=0.6
            ),
            quality=QualityConfig(
                min_acceptable_quality=80.0,
                high_quality_threshold=90.0,
                enable_detailed_analysis=True,
                generate_quality_reports=True,
                export_confidence_maps=True
            ),
            processing=ProcessingConfig(
                enable_auto_retry=True,
                parallel_processing=False,
                max_processing_time_minutes=60,
                preserve_intermediate_files=True
            )
        )
    
    @classmethod
    def create_fast_config(cls) -> 'EnhancedSystemConfig':
        """
        Crea una configuración optimizada para velocidad.
        
        Returns:
            EnhancedSystemConfig: Configuración para procesamiento rápido
        """
        return cls(
            ocr=OCRConfig(
                dpi=150,
                confidence_threshold=50.0,
                enable_preprocessing=False,
                enable_quality_analysis=False,
                max_retry_attempts=1
            ),
            preprocessing=PreprocessingConfig(
                enable_skew_correction=False,
                enable_noise_reduction=False,
                enable_contrast_enhancement=True,
                enable_adaptive_binarization=False,
                quality_threshold_for_intensive_processing=0.9
            ),
            quality=QualityConfig(
                min_acceptable_quality=40.0,
                high_quality_threshold=70.0,
                enable_detailed_analysis=False,
                generate_quality_reports=False,
                export_confidence_maps=False
            ),
            processing=ProcessingConfig(
                enable_auto_retry=False,
                parallel_processing=True,
                max_processing_time_minutes=10,
                preserve_intermediate_files=False
            )
        )
    
    @classmethod
    def create_balanced_config(cls) -> 'EnhancedSystemConfig':
        """
        Crea una configuración balanceada entre calidad y velocidad.
        
        Returns:
            EnhancedSystemConfig: Configuración balanceada (por defecto)
        """
        return cls()  # Usa los valores por defecto
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte la configuración a diccionario para serialización.
        
        Returns:
            Dict: Configuración serializada
        """
        return {
            'ocr': {
                'language': self.ocr.language,
                'dpi': self.ocr.dpi,
                'confidence_threshold': self.ocr.confidence_threshold,
                'enable_preprocessing': self.ocr.enable_preprocessing,
                'enable_quality_analysis': self.ocr.enable_quality_analysis,
                'max_retry_attempts': self.ocr.max_retry_attempts
            },
            'preprocessing': {
                'enable_skew_correction': self.preprocessing.enable_skew_correction,
                'enable_noise_reduction': self.preprocessing.enable_noise_reduction,
                'enable_contrast_enhancement': self.preprocessing.enable_contrast_enhancement,
                'enable_adaptive_binarization': self.preprocessing.enable_adaptive_binarization,
                'quality_threshold_for_intensive_processing': self.preprocessing.quality_threshold_for_intensive_processing
            },
            'quality': {
                'min_acceptable_quality': self.quality.min_acceptable_quality,
                'high_quality_threshold': self.quality.high_quality_threshold,
                'enable_detailed_analysis': self.quality.enable_detailed_analysis,
                'generate_quality_reports': self.quality.generate_quality_reports,
                'export_confidence_maps': self.quality.export_confidence_maps
            },
            'processing': {
                'enable_auto_retry': self.processing.enable_auto_retry,
                'parallel_processing': self.processing.parallel_processing,
                'max_processing_time_minutes': self.processing.max_processing_time_minutes,
                'preserve_intermediate_files': self.processing.preserve_intermediate_files,
                'enable_table_extraction': self.processing.enable_table_extraction,
                'table_quality_threshold': self.processing.table_quality_threshold
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'EnhancedSystemConfig':
        """
        Crea una configuración desde un diccionario.
        
        Args:
            config_dict: Diccionario con configuraciones
            
        Returns:
            EnhancedSystemConfig: Configuración creada
        """
        ocr_config = OCRConfig(**config_dict.get('ocr', {}))
        preprocessing_config = PreprocessingConfig(**config_dict.get('preprocessing', {}))
        quality_config = QualityConfig(**config_dict.get('quality', {}))
        processing_config = ProcessingConfig(**config_dict.get('processing', {}))
        
        return cls(
            ocr=ocr_config,
            preprocessing=preprocessing_config,
            quality=quality_config,
            processing=processing_config
        )


def load_config_from_file(config_path: Path) -> EnhancedSystemConfig:
    """
    Carga configuración desde un archivo YAML.
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        EnhancedSystemConfig: Configuración cargada
    """
    try:
        import yaml
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return EnhancedSystemConfig.from_dict(config_dict)
        
    except Exception as e:
        print(f"Error cargando configuración desde {config_path}: {e}")
        print("Usando configuración por defecto...")
        return EnhancedSystemConfig.create_balanced_config()


def save_config_to_file(config: EnhancedSystemConfig, config_path: Path) -> bool:
    """
    Guarda configuración en un archivo YAML.
    
    Args:
        config: Configuración a guardar
        config_path: Ruta donde guardar
        
    Returns:
        bool: True si se guardó exitosamente
    """
    try:
        import yaml
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)
        
        return True
        
    except Exception as e:
        print(f"Error guardando configuración en {config_path}: {e}")
        return False


# Configuraciones predefinidas para casos comunes
QUALITY_PROFILES = {
    'maximum_quality': EnhancedSystemConfig.create_high_quality_config(),
    'fast_processing': EnhancedSystemConfig.create_fast_config(),
    'balanced': EnhancedSystemConfig.create_balanced_config()
}
