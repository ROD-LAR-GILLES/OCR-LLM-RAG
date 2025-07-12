# application/ports.py
"""
Definición de puertos (interfaces/contratos) para la arquitectura hexagonal.

Este módulo define las abstracciones que deben implementar los adaptadores
externos, siguiendo el principio de Inversión de Dependencias. Los puertos
establecen el contrato que deben cumplir las implementaciones concretas
sin acoplar la lógica de negocio a tecnologías específicas.

Principios aplicados:
- Dependency Inversion Principle (DIP): Las capas de alto nivel no dependen de detalles
- Interface Segregation Principle (ISP): Interfaces pequeñas y específicas
- Single Responsibility Principle (SRP): Cada puerto tiene una responsabilidad clara
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Any, Dict, Optional


class OCRPort(ABC):
    """
    Puerto (interfaz) para servicios de Reconocimiento Óptico de Caracteres.
    
    Define el contrato que deben cumplir todos los adaptadores de OCR,
    permitiendo intercambiar implementaciones (Tesseract, EasyOCR, Google Vision, etc.)
    sin modificar la lógica de negocio.
    
    Casos de uso:
    - Extraer texto de documentos escaneados
    - Procesar PDFs que contienen imágenes con texto
    - Digitalizar documentos físicos
    
    Implementaciones futuras previstas:
    - TesseractAdapter: OCR local con Tesseract (actual)
    - EasyOCRAdapter: OCR con deep learning para mejor precisión
    - GoogleVisionAdapter: OCR en la nube con Google Cloud Vision API
    - AmazonTextractAdapter: OCR avanzado con AWS Textract
    - AzureComputerVisionAdapter: OCR con Microsoft Azure
    """
    
    @abstractmethod
    def extract_text(self, pdf_path: Path) -> str:
        """
        Extrae todo el texto contenido en un documento PDF mediante OCR.
        
        Este método debe manejar:
        - PDFs con múltiples páginas
        - Diferentes calidades de imagen
        - Texto en múltiples idiomas (según configuración del adaptador)
        - Orientaciones de texto (horizontal, vertical, rotado)
        
        Args:
            pdf_path (Path): Ruta absoluta al archivo PDF a procesar
            
        Returns:
            str: Texto completo extraído del documento. Las páginas deben
                 estar separadas por saltos de línea para facilitar la lectura.
                 
        Raises:
            FileNotFoundError: Si el archivo PDF no existe
            OCRError: Si el proceso de OCR falla (definir en implementaciones)
            UnsupportedFormatError: Si el formato PDF no es compatible
            
        Note:
            - El texto resultante debe ser UTF-8 válido
            - Los espacios en blanco deben preservarse para mantener formato
            - Los caracteres especiales deben manejarse correctamente
        """
        ...


class TableExtractorPort(ABC):
    """
    Puerto (interfaz) para servicios de extracción de tablas de documentos.
    
    Define el contrato para extraer datos tabulares de PDFs, permitiendo
    intercambiar diferentes motores de análisis de tablas según las
    necesidades específicas del documento.
    
    Casos de uso:
    - Extraer datos financieros de reportes
    - Procesar formularios estructurados
    - Analizar tablas de inventarios o catálogos
    - Digitalizar hojas de cálculo impresas
    
    Implementaciones futuras previstas:
    - PdfPlumberAdapter: Análisis estructural de PDFs nativos (actual)
    - CamelotAdapter: Especializado en tablas complejas
    - TabulaAdapter: Extracción basada en Java/Tabula
    - OCRTableAdapter: Tablas desde imágenes usando OCR + ML
    """
    
    @abstractmethod
    def extract_tables(self, pdf_path: Path) -> List[Any]:
        """
        Extrae todas las tablas detectadas en un documento PDF.
        
        Este método debe:
        - Detectar automáticamente estructuras tabulares
        - Preservar la jerarquía de filas y columnas
        - Manejar celdas combinadas cuando sea posible
        - Extraer encabezados de tabla
        
        Args:
            pdf_path (Path): Ruta absoluta al archivo PDF a procesar
            
        Returns:
            List[Any]: Lista de estructuras de datos que representan tablas.
                      Típicamente pandas.DataFrame, pero el puerto permite
                      flexibilidad en el tipo de retorno según la implementación.
                      Lista vacía si no se detectan tablas.
                      
        Raises:
            FileNotFoundError: Si el archivo PDF no existe
            TableExtractionError: Si la extracción de tablas falla
            CorruptedFileError: Si el PDF está dañado o protegido
            
        Note:
            - El orden de las tablas en la lista debe corresponder a su aparición en el PDF
            - Las celdas vacías deben representarse de forma consistente (None, "", etc.)
            - Los tipos de datos deben inferirse cuando sea posible (números, fechas, texto)
        """
        ...


class StoragePort(ABC):
    """
    Puerto (interfaz) para servicios de persistencia de resultados procesados.
    
    Define el contrato para almacenar los resultados del procesamiento de documentos,
    permitiendo múltiples estrategias de persistencia según el entorno y requisitos.
    
    Casos de uso:
    - Almacenamiento local para desarrollo y pruebas
    - Persistencia en bases de datos para aplicaciones empresariales
    - Almacenamiento en la nube para escalabilidad
    - Sistemas de archivos distribuidos para big data
    
    Implementaciones futuras previstas:
    - FileStorage: Sistema de archivos local (actual)
    - DatabaseStorage: Persistencia en PostgreSQL/MongoDB
    - S3Storage: Almacenamiento en Amazon S3
    - AzureBlobStorage: Almacenamiento en Azure Blob Storage
    - ElasticsearchStorage: Índices para búsqueda full-text
    """
    
    @abstractmethod
    def save(self, name: str, text: str, tables: List[Any], original: Path) -> List[str]:
        """
        Persiste los resultados del procesamiento de un documento.
        
        Este método debe:
        - Almacenar el texto extraído de forma recuperable
        - Persistir las tablas en formato estructurado
        - Mantener referencia al documento original
        - Generar identificadores únicos para evitar colisiones
        - Manejar transacciones para garantizar consistencia
        - Retornar la lista de archivos/recursos generados
        
        Args:
            name (str): Nombre identificador del documento (sin extensión)
            text (str): Texto completo extraído por OCR
            tables (List[Any]): Lista de tablas extraídas del documento
            original (Path): Ruta al archivo PDF original para referencia
            
        Returns:
            List[str]: Lista de rutas/identificadores de los archivos generados
            
        Raises:
            StorageError: Si hay problemas en la persistencia
            DuplicateError: Si ya existe un documento con el mismo nombre
            InsufficientSpaceError: Si no hay espacio suficiente
            PermissionError: Si no hay permisos de escritura
            
        Note:
            - Los datos deben persistirse de forma atómica cuando sea posible
            - Los metadatos (fecha de procesamiento, versión, etc.) deben incluirse
            - El formato de almacenamiento debe ser recoverable y portable
        """
        ...


class LLMDocumentPort(ABC):
    """
    Puerto (interfaz) para servicios de procesamiento de documentos optimizado para LLMs.
    
    Define el contrato para extraer y estructurar contenido de documentos
    de forma optimizada para modelos de lenguaje, generando formato markdown
    estructurado que preserva la jerarquía y contexto del documento original.
    
    Casos de uso:
    - Preparación de documentos para RAG (Retrieval-Augmented Generation)
    - Extracción de contenido para sistemas de embeddings vectoriales
    - Conversión de documentos técnicos para análisis por IA
    - Indexación de documentos para búsqueda semántica
    - Preparación de datasets de entrenamiento para LLMs
    
    Implementaciones futuras previstas:
    - PyMuPDF4LLMAdapter: Extracción optimizada con pymupdf4llm (actual)
    - UnstructuredAdapter: Procesamiento con librería Unstructured
    - LlamaParseAdapter: Extracción con LlamaParse de LlamaIndex
    - OpenAIDocumentAdapter: Procesamiento con GPT-4 Vision
    - AnthropicClaudeAdapter: Extracción con Claude Vision
    """
    
    @abstractmethod
    def extract_structured_content(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extrae contenido estructurado optimizado para procesamiento por LLMs.
        
        Este método debe generar contenido en formato markdown estructurado
        que preserve la jerarquía del documento y sea óptimo para:
        - Chunking inteligente para sistemas RAG
        - Generación de embeddings vectoriales
        - Análisis por modelos de lenguaje
        - Búsqueda semántica y retrieval
        
        Args:
            pdf_path (Path): Ruta absoluta al archivo PDF a procesar
            
        Returns:
            Dict[str, Any]: Diccionario con contenido estructurado:
                - 'markdown': str - Contenido principal en formato markdown
                - 'metadata': Dict - Metadatos del documento (título, autor, etc.)
                - 'structure': Dict - Información sobre la estructura (secciones, etc.)
                - 'stats': Dict - Estadísticas del contenido (palabras, páginas, etc.)
                - 'chunks': List[str] - Contenido dividido en chunks (opcional)
                
        Raises:
            FileNotFoundError: Si el archivo PDF no existe
            LLMProcessingError: Si el procesamiento específico para LLM falla
            UnsupportedDocumentError: Si el documento no es compatible
            
        Note:
            - El markdown debe preservar la jerarquía de encabezados
            - Las tablas deben mantenerse en formato markdown
            - Los metadatos deben incluir información relevante para RAG
            - El chunking debe ser semánticamente coherente
        """
        ...

    @abstractmethod
    def get_chunk_strategy(self) -> Dict[str, Any]:
        """
        Retorna la estrategia de chunking utilizada por el adaptador.
        
        Returns:
            Dict[str, Any]: Configuración de chunking:
                - 'method': str - Método de chunking (semantic, fixed, page, etc.)
                - 'max_tokens': int - Tamaño máximo de chunk en tokens
                - 'overlap': int - Solapamiento entre chunks
                - 'preserve_structure': bool - Si preserva estructura de encabezados
        """
        ...