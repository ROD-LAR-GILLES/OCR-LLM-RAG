# domain/rules.py
"""
Reglas de negocio y validaciones del dominio.

Este módulo está preparado para contener las reglas de negocio específicas
del dominio OCR-CLI, siguiendo los principios de Domain-Driven Design.

Tipos de reglas que podrían implementarse aquí:

1. VALIDACIONES DE DOCUMENTOS:
   - Tamaño máximo de archivo PDF
   - Formatos de PDF soportados
   - Validación de integridad del documento
   - Detección de documentos protegidos con contraseña

2. REGLAS DE PROCESAMIENTO:
   - Políticas de calidad mínima para OCR
   - Filtros de confianza para texto extraído
   - Reglas de fusión de páginas duplicadas
   - Validación de consistencia de tablas

3. REGLAS DE NEGOCIO:
   - Límites de procesamiento concurrente
   - Políticas de retención de documentos
   - Clasificación automática de tipos de documento
   - Reglas de anonimización de datos sensibles

Ejemplo de implementación futura:

class DocumentValidationRules:
    MAX_FILE_SIZE_MB = 50
    SUPPORTED_FORMATS = ['.pdf']
    MIN_OCR_CONFIDENCE = 0.7
    
    @staticmethod
    def validate_file_size(file_path: Path) -> bool:
        size_mb = file_path.stat().st_size / (1024 * 1024)
        return size_mb <= DocumentValidationRules.MAX_FILE_SIZE_MB
    
    @staticmethod
    def validate_ocr_quality(confidence_score: float) -> bool:
        return confidence_score >= DocumentValidationRules.MIN_OCR_CONFIDENCE

class TableExtractionRules:
    MIN_ROWS_FOR_TABLE = 2
    MIN_COLUMNS_FOR_TABLE = 2
    MAX_EMPTY_CELLS_PERCENTAGE = 0.5
    
    @staticmethod
    def is_valid_table(df: pd.DataFrame) -> bool:
        if len(df) < TableExtractionRules.MIN_ROWS_FOR_TABLE:
            return False
        if len(df.columns) < TableExtractionRules.MIN_COLUMNS_FOR_TABLE:
            return False
        empty_cells = df.isnull().sum().sum()
        total_cells = len(df) * len(df.columns)
        empty_percentage = empty_cells / total_cells
        return empty_percentage <= TableExtractionRules.MAX_EMPTY_CELLS_PERCENTAGE

Principios aplicados:
- Domain-Driven Design: Reglas que reflejan el conocimiento del negocio
- Single Responsibility: Cada regla tiene una responsabilidad específica
- Testability: Reglas fácilmente testeable de forma unitaria
- Configuration: Parámetros externalizables para diferentes entornos
"""

# Placeholder para futuras reglas de negocio
# Este archivo se mantendrá vacío hasta que se identifiquen
# reglas específicas del dominio que requieran implementación