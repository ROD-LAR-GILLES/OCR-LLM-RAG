# application/use_cases.py
"""
Casos de uso (interactors) que orquestan la lógica de negocio.

Este módulo contiene los casos de uso de la aplicación, implementando
la lógica de negocio pura sin depender de detalles técnicos específicos.
Los casos de uso coordinan los puertos (interfaces) para ejecutar
flujos de trabajo completos.

Principios aplicados:
- Single Responsibility: Cada caso de uso tiene una responsabilidad específica
- Dependency Injection: Recibe dependencias via constructor
- Clean Architecture: Aislamiento de la lógica de negocio
- Command Pattern: Casos de uso como comandos ejecutables
"""
from pathlib import Path
from typing import Tuple, List, Any, Optional, Dict, Literal

from src.application.ports import OCRPort, TableExtractorPort, StoragePort
from src.domain.models import Document
from src.adapters.llm_pymupdf4llm_adapter import PyMuPDF4LLMAdapter


class ProcessDocument:
    """
    Caso de uso principal para el procesamiento completo de documentos PDF.
    
    Este caso de uso orquesta todo el flujo de procesamiento de un documento:
    1. Extracción de texto mediante OCR
    2. Identificación y extracción de tablas
    3. Persistencia de resultados en formato estructurado
    
    Responsabilidades:
    - Coordinar la secuencia de procesamiento
    - Manejar errores en cualquier etapa del proceso
    - Garantizar la integridad de los datos procesados
    - Proporcionar feedback sobre el resultado del procesamiento
    
    Ventajas del patrón Caso de Uso:
    - Testeable: Fácil crear mocks para cada dependencia
    - Flexible: Cambiar implementaciones sin afectar la lógica
    - Reutilizable: Mismo caso de uso para CLI, API REST, etc.
    - Mantenible: Lógica de negocio separada de detalles técnicos
    
    Flujo de procesamiento:
    PDF -> [OCR] -> Texto plano -> [Storage] -> Archivos de salida
        -> [Table Extraction] -> DataFrames -> [Storage] -> JSON/ASCII
    """

    def __init__(
        self,
        ocr: OCRPort,
        table_extractor: TableExtractorPort,
        storage: StoragePort,
    ) -> None:
        """
        Inicializa el caso de uso con las dependencias inyectadas.
        
        Este constructor implementa el patrón de Inyección de Dependencias,
        permitiendo que el caso de uso trabaje con cualquier implementación
        de los puertos sin conocer los detalles específicos.
        
        Args:
            ocr (OCRPort): Servicio de reconocimiento óptico de caracteres.
                          Puede ser Tesseract, EasyOCR, Google Vision, etc.
                          
            table_extractor (TableExtractorPort): Servicio de extracción de tablas.
                                                  Puede ser pdfplumber, Camelot, Tabula, etc.
                                                  
            storage (StoragePort): Servicio de persistencia de resultados.
                                  Puede ser filesystem, database, cloud storage, etc.
        
        Note:
            La inyección de dependencias permite:
            - Testing: Usar mocks en lugar de implementaciones reales
            - Flexibility: Cambiar implementaciones sin modificar código
            - Configuration: Elegir implementaciones según el entorno (dev/prod)
        """
        self.ocr = ocr
        self.table_extractor = table_extractor
        self.storage = storage

    def __call__(self, pdf_path: Path) -> Tuple[str, List[str]]:
        """
        Ejecuta el procesamiento completo de un documento PDF.
        
        Este método implementa el patrón Command, permitiendo que el caso
        de uso sea ejecutado como una función callable. El procesamiento
        sigue una secuencia determinística que garantiza la integridad
        de los datos.
        
        NUEVA LÓGICA: Ahora crea una carpeta organizada por documento
        y retorna las rutas reales de los archivos generados.
        
        Flujo de ejecución:
        1. Validación inicial del archivo PDF
        2. Extracción de texto mediante OCR (puede tomar varios minutos)
        3. Extracción paralela de tablas (análisis estructural)
        4. Persistencia atómica de todos los resultados en carpeta dedicada
        5. Retorno de las rutas reales de archivos generados
        
        Args:
            pdf_path (Path): Ruta absoluta al archivo PDF a procesar.
                            Debe existir y ser legible.
        
        Returns:
            Tuple[str, List[str]]: Tupla con:
                - str: Ruta al archivo de texto principal generado
                - List[str]: Lista de todas las rutas de archivos generados
                            (organizados en carpeta por documento)
        
        Raises:
            FileNotFoundError: Si el archivo PDF no existe
            ProcessingError: Si alguna etapa del procesamiento falla
            StorageError: Si hay problemas al persistir los resultados
            
        Example:
            >>> processor = ProcessDocument(
            ...     ocr=TesseractAdapter(),
            ...     table_extractor=PdfPlumberAdapter(),
            ...     storage=FileStorage(Path("./output"))
            ... )
            >>> text_file, all_files = processor(Path("document.pdf"))
            >>> print(f"Texto extraído en: {text_file}")
            >>> print(f"Archivos generados: {all_files}")
            # Salida esperada:
            # Texto extraído en: /app/resultado/document/texto_completo.txt
            # Archivos generados: ['/app/resultado/document/texto_completo.txt', 
            #                      '/app/resultado/document/tabla_1.json', ...]
            
        Performance Notes:
            - OCR es la operación más costosa (O(n) con número de páginas)
            - Extracción de tablas es más rápida (análisis estructural)
            - El tiempo total depende de: resolución DPI, número de páginas, complejidad
        """
        # ETAPA 1: Extracción de texto mediante OCR
        # Esta es típicamente la operación más lenta del proceso
        # El tiempo depende de: número de páginas, resolución DPI, complejidad del texto
        text: str = self.ocr.extract_text(pdf_path)
        
        # ETAPA 2: Extracción de tablas estructuradas
        # Análisis paralelo e independiente del OCR
        # Más rápido que OCR pues analiza estructura vectorial del PDF
        tables: List[Any] = self.table_extractor.extract_tables(pdf_path)

        # ETAPA 3: Persistencia atómica de resultados en carpeta organizada
        # Guarda todos los resultados de forma consistente en una carpeta dedicada
        # Si falla aquí, no se pierde el trabajo de OCR/tablas ya realizado
        archivos_generados: List[str] = self.storage.save(pdf_path.stem, text, tables, pdf_path)

        # ETAPA 4: Identificación del archivo principal
        # El archivo de texto completo es el resultado principal
        texto_principal = next(
            (archivo for archivo in archivos_generados if archivo.endswith("texto_completo.txt")),
            archivos_generados[0] if archivos_generados else ""
        )
        
        return texto_principal, archivos_generados


class ExtractDocumentUseCase:
    """
    Caso de uso para extraer contenido de documentos PDF utilizando
    diferentes estrategias de extracción.
    """
    
    def __init__(self, pymupdf_adapter: PyMuPDF4LLMAdapter):
        self.pymupdf_adapter = pymupdf_adapter
    
    def execute(
        self,
        pdf_path: str,
        output_format: Literal["markdown", "llama_documents"] = "markdown",
        pages: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Ejecuta la extracción del documento según el formato solicitado.
        
        Args:
            pdf_path: Ruta al archivo PDF a procesar
            output_format: Formato de salida deseado ("markdown" o "llama_documents")
            pages: Lista opcional de páginas a extraer (solo válido para markdown)
            
        Returns:
            Diccionario con el resultado de la extracción y metadatos
        """
        result = {
            "source": pdf_path,
            "format": output_format,
        }
        
        if output_format == "markdown":
            content = self.pymupdf_adapter.extract_markdown(pdf_path, pages)
            result["content"] = content
            
        elif output_format == "llama_documents":
            documents = self.pymupdf_adapter.extract_llama_documents(pdf_path)
            result["documents"] = documents
            result["document_count"] = len(documents)
            
        else:
            raise ValueError(f"Formato de salida no soportado: {output_format}")
            
        return result