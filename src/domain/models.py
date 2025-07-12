# domain/models.py
"""
Modelos de dominio que representan las entidades principales del negocio.

Este módulo contiene las entidades puras del dominio, libres de dependencias
externas y frameworks específicos. Los modelos definen la estructura y
comportamiento esencial de los conceptos del negocio.

Principios aplicados:
- Domain-Driven Design: Modelos que reflejan el lenguaje del negocio
- Single Responsibility: Cada modelo tiene una responsabilidad clara
- Immutability: Modelos inmutables para garantizar consistencia
- Type Safety: Uso de type hints para claridad y validación
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Any, Dict, Optional
from datetime import datetime


@dataclass
class ProcessingMetrics:
    """
    Métricas de procesamiento y calidad para operaciones de OCR.
    
    Contiene información detallada sobre el rendimiento y calidad
    del procesamiento de documentos.
    """
    page_number: Optional[int] = None
    confidence_score: float = 0.0
    processing_time: float = 0.0
    image_quality_score: float = 0.0
    preprocessing_applied: bool = False
    quality_metrics: Dict[str, Any] = field(default_factory=dict)
    total_processing_time: float = 0.0
    average_confidence: float = 0.0
    document_quality: Dict[str, Any] = field(default_factory=dict)
    
    def add_page_metrics(self, page_metrics: 'ProcessingMetrics') -> None:
        """Acumula métricas de una página al total del documento."""
        self.total_processing_time += page_metrics.processing_time
        
        # Calcular promedio de confianza (implementación simple)
        if self.average_confidence == 0.0:
            self.average_confidence = page_metrics.confidence_score
        else:
            # Promedio ponderado simple
            self.average_confidence = (self.average_confidence + page_metrics.confidence_score) / 2


@dataclass
class OCRResult:
    """
    Resultado completo de una operación de OCR con métricas.
    
    Encapsula tanto el texto extraído como todas las métricas
    de calidad y rendimiento del procesamiento.
    """
    text: str
    metrics: ProcessingMetrics
    page_count: int
    processing_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def quality_score(self) -> float:
        """Puntuación de calidad general del resultado."""
        return self.metrics.average_confidence
    
    @property
    def is_high_quality(self) -> bool:
        """Indica si el resultado es de alta calidad."""
        return self.quality_score >= 80.0


@dataclass
class Document:
    """
    Entidad que representa un documento procesado en el sistema.
    
    Esta entidad encapsula toda la información extraída de un documento PDF
    después del procesamiento completo, incluyendo tanto texto como datos
    estructurados (tablas).
    
    Responsabilidades:
    - Mantener la integridad de los datos extraídos
    - Proporcionar una representación unificada del documento
    - Servir como modelo de datos para persistencia y transferencia
    - Mantener la trazabilidad hacia el archivo original
    
    Casos de uso:
    - Transferencia de datos entre capas de la aplicación
    - Serialización/deserialización para APIs REST
    - Persistencia en bases de datos
    - Análisis y transformación de contenido
    
    Future Extensions:
    - metadata: Dict con información adicional (fecha procesamiento, versión OCR, etc.)
    - pages: List[Page] para análisis granular por página
    - confidence_scores: Métricas de confianza del OCR
    - language: Idioma detectado del documento
    """
    
    name: str
    """
    Nombre identificador del documento.
    
    Generalmente corresponde al nombre del archivo original sin extensión,
    pero puede ser modificado por el usuario para mayor claridad semántica.
    Debe ser único dentro del contexto de procesamiento.
    """
    
    source_path: Path
    """
    Ruta al archivo PDF original en el sistema de archivos.
    
    Mantiene la trazabilidad hacia el documento fuente, permitiendo
    reprocesamiento si es necesario y validación de integridad.
    """
    
    extracted_text: str
    """
    Texto completo extraído del documento mediante OCR.
    
    Contiene todo el contenido textual identificado en el documento,
    preservando la estructura de párrafos y separación entre páginas.
    El texto puede contener errores típicos de OCR que requieren validación.
    """
    
    tables: List[Any]
    """
    Lista de tablas extraídas como DataFrames de pandas.
    
    Cada elemento representa una tabla detectada en el documento,
    manteniendo la estructura de filas y columnas originales.
    Las tablas están listas para análisis, exportación o transformación.
    """
    
    # Nuevos campos para métricas avanzadas
    ocr_result: Optional[OCRResult] = None
    """Resultado detallado del OCR con métricas de calidad."""
    
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    """Metadatos adicionales del procesamiento."""
    
    def __post_init__(self) -> None:
        """
        Validaciones post-inicialización del modelo.
        
        Ejecuta validaciones de integridad después de la creación
        del objeto para garantizar que los datos son consistentes
        y válidos.
        
        Validaciones aplicadas:
        - Nombre no vacío y válido para nombres de archivo
        - Texto no nulo (puede estar vacío si OCR falló)
        - Lista de tablas inicializada (aunque esté vacía)
        - Archivo fuente existe y es accesible
        
        Raises:
            ValueError: Si alguna validación falla
            FileNotFoundError: Si el archivo fuente no existe
        """
        # Validación: nombre debe ser válido para crear archivos
        if not self.name or not self.name.strip():
            raise ValueError("Document name cannot be empty")
            
        # Validación: texto debe estar inicializado (aunque esté vacío)
        if self.extracted_text is None:
            raise ValueError("Document text cannot be None")
            
        # Validación: lista de tablas debe estar inicializada
        if self.tables is None:
            self.tables = []
            
        # Validación: archivo fuente debe existir
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source file not found: {self.source_path}")
    
    @property
    def has_tables(self) -> bool:
        """
        Indica si el documento contiene tablas extraídas.
        
        Returns:
            bool: True si se detectaron y extrajeron tablas, False en caso contrario
        """
        return len(self.tables) > 0
    
    @property
    def table_count(self) -> int:
        """
        Número de tablas extraídas del documento.
        
        Returns:
            int: Cantidad de tablas detectadas en el documento
        """
        return len(self.tables)
    
    @property
    def word_count(self) -> int:
        """
        Aproximación del número de palabras en el texto extraído.
        
        Útil para:
        - Métricas de procesamiento
        - Estimación de tiempo de lectura
        - Validación de calidad de OCR
        
        Returns:
            int: Número aproximado de palabras en el texto
        """
        return len(self.extracted_text.split()) if self.extracted_text else 0
    
    @property
    def quality_score(self) -> Optional[float]:
        """
        Puntuación de calidad del OCR aplicado al documento.
        
        Returns:
            float: Puntuación de confianza promedio (0-100) o None si no disponible
        """
        if self.ocr_result:
            return self.ocr_result.quality_score
        return None
    
    @property
    def is_high_quality(self) -> bool:
        """
        Indica si el documento fue procesado con alta calidad.
        
        Returns:
            bool: True si la calidad es superior al 80%, False en caso contrario
        """
        quality = self.quality_score
        return quality is not None and quality >= 80.0
    
    @property
    def processing_summary(self) -> Dict[str, Any]:
        """
        Resumen de métricas de procesamiento del documento.
        
        Returns:
            Dict: Diccionario con métricas clave del procesamiento
        """
        summary = {
            'word_count': self.word_count,
            'table_count': self.table_count,
            'has_ocr_metrics': self.ocr_result is not None
        }
        
        if self.ocr_result:
            summary.update({
                'quality_score': self.quality_score,
                'processing_time': self.ocr_result.processing_time,
                'page_count': self.ocr_result.page_count,
                'is_high_quality': self.is_high_quality
            })
        
        return summary