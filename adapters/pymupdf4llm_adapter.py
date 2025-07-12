# adapters/pymupdf4llm_adapter.py
"""
Adaptador de extracción de contenido basado en pymupdf4llm.

Este módulo implementa el puerto OCRPort utilizando pymupdf4llm,
una librería especializada en extraer contenido de PDFs de forma
optimizada para modelos de lenguaje (LLMs), generando markdown
estructurado y preservando el formato del documento.

Principios aplicados:
- Single Responsibility: Solo se encarga de extraer contenido con pymupdf4llm
- Dependency Inversion: Implementa el contrato OCRPort
- Error Handling: Manejo robusto de errores específicos de pymupdf4llm
"""
from pathlib import Path
from typing import Optional, Dict, Any
import logging

try:
    import pymupdf4llm
except ImportError:
    raise ImportError(
        "pymupdf4llm no está instalado. "
        "Instálalo con: pip install pymupdf4llm"
    )

from application.ports import OCRPort


class PyMuPDF4LLMAdapter(OCRPort):
    """
    Adaptador para extracción de contenido optimizada para LLMs usando pymupdf4llm.
    
    pymupdf4llm es una librería especializada que convierte PDFs en markdown
    estructurado, preservando la jerarquía del documento y optimizando
    el contenido para procesamiento por modelos de lenguaje.
    
    Ventajas de pymupdf4llm:
    - **Markdown estructurado**: Salida en formato markdown con jerarquía
    - **Optimizado para LLMs**: Formato ideal para procesamiento por IA
    - **Preserva estructura**: Mantiene encabezados, listas, tablas
    - **Alta velocidad**: Procesamiento directo sin conversión a imagen
    - **Metadatos ricos**: Incluye información sobre el documento
    - **Soporte completo**: Maneja texto, imágenes y elementos complejos
    
    Limitaciones:
    - **Solo PDFs nativos**: No procesa documentos escaneados (requiere OCR adicional)
    - **Dependiente de estructura**: Funciona mejor con PDFs bien estructurados
    - **Formato fijo**: Salida siempre en markdown (no texto plano)
    
    Casos de uso ideales:
    - Preparación de documentos para RAG (Retrieval-Augmented Generation)
    - Análisis de documentos con LLMs
    - Extracción de contenido para sistemas de embeddings
    - Conversión de documentos técnicos a formato procesable por IA
    - Indexación de documentos para búsqueda semántica
    
    Configuraciones disponibles:
    - page_chunks: Dividir en chunks por página
    - write_images: Extraer y referenciar imágenes
    - embed_images: Incrustar imágenes en base64
    - table_strategy: Estrategia para procesar tablas
    - ignore_code: Ignorar bloques de código
    """

    def __init__(
        self,
        page_chunks: bool = False,
        write_images: bool = False,
        embed_images: bool = False,
        table_strategy: str = "lines_strict",
        ignore_code: bool = False,
        **kwargs
    ) -> None:
        """
        Inicializa el adaptador con configuraciones específicas de pymupdf4llm.
        
        Args:
            page_chunks (bool): Si True, divide el contenido en chunks por página.
                               Útil para documentos largos que se procesarán por LLMs
                               con límites de tokens.
                               
            write_images (bool): Si True, extrae imágenes a archivos separados
                                y las referencia en el markdown. Útil cuando las
                                imágenes son importantes para el análisis.
                                
            embed_images (bool): Si True, incrusta imágenes como base64 en el markdown.
                                Alternativa a write_images para incluir imágenes
                                directamente en el texto.
                                
            table_strategy (str): Estrategia para procesar tablas.
                                 - "lines_strict": Detecta tablas usando líneas (más preciso)
                                 - "lines": Detecta tablas usando líneas (más flexible)
                                 - "text": Usa espaciado de texto para detectar tablas
                                 - "explicit": Solo tablas marcadas explícitamente
                                 
            ignore_code (bool): Si True, ignora bloques de código y elementos
                               similares. Útil para documentos técnicos donde
                               el código no es relevante para el análisis.
                               
            **kwargs: Parámetros adicionales específicos de pymupdf4llm
        
        Note:
            Las configuraciones son específicas de pymupdf4llm y pueden cambiar
            según la versión de la librería. Consulta la documentación oficial
            para opciones avanzadas.
        """
        self.page_chunks = page_chunks
        self.write_images = write_images
        self.embed_images = embed_images
        self.table_strategy = table_strategy
        self.ignore_code = ignore_code
        self.extra_config = kwargs
        
        # Configurar logging específico para pymupdf4llm
        self.logger = logging.getLogger(__name__)

    def extract_text(self, pdf_path: Path) -> str:
        """
        Extrae contenido del PDF en formato markdown optimizado para LLMs.
        
        Este método utiliza pymupdf4llm para convertir el PDF en markdown
        estructurado, preservando la jerarquía del documento y optimizando
        el formato para procesamiento por modelos de lenguaje.
        
        Proceso de extracción:
        1. Validación del archivo PDF
        2. Configuración de parámetros de pymupdf4llm
        3. Extracción de contenido con preservación de estructura
        4. Post-procesamiento y limpieza del markdown
        5. Retorno del contenido estructurado
        
        Args:
            pdf_path (Path): Ruta absoluta al archivo PDF a procesar
            
        Returns:
            str: Contenido del PDF en formato markdown estructurado.
                 Incluye encabezados, párrafos, listas, tablas y metadatos
                 preservando la jerarquía original del documento.
                 
        Raises:
            FileNotFoundError: Si el archivo PDF no existe
            PyMuPDF4LLMError: Si hay problemas específicos con pymupdf4llm
            PermissionError: Si no hay permisos para leer el archivo
            CorruptedPDFError: Si el PDF está dañado o es ilegible
            UnsupportedPDFError: Si el PDF tiene características no soportadas
            
        Example:
            >>> adapter = PyMuPDF4LLMAdapter(
            ...     page_chunks=False,
            ...     table_strategy="lines_strict"
            ... )
            >>> content = adapter.extract_text(Path("document.pdf"))
            >>> print(content[:200])
            # Documento Técnico
            
            ## Introducción
            
            Este documento presenta...
            
            ### Tabla de Contenidos
            
            | Sección | Página |
            |---------|--------|
            | Intro   | 1      |
            ...
            
        Performance Notes:
            - pymupdf4llm es significativamente más rápido que OCR tradicional
            - El tiempo de procesamiento es O(n) con el número de páginas
            - La extracción de imágenes puede incrementar el tiempo de procesamiento
            - Los PDFs complejos (muchas tablas/imágenes) tardan más
        """
        # ETAPA 1: Validación del archivo de entrada
        if not pdf_path.exists():
            raise FileNotFoundError(f"El archivo PDF no existe: {pdf_path}")
            
        if not pdf_path.is_file():
            raise ValueError(f"La ruta no apunta a un archivo: {pdf_path}")
            
        if pdf_path.suffix.lower() != '.pdf':
            raise ValueError(f"El archivo no es un PDF: {pdf_path}")

        try:
            # ETAPA 2: Configuración de parámetros para pymupdf4llm
            # Construir configuración dinámica basada en parámetros del adaptador
            extraction_config = {
                "page_chunks": self.page_chunks,
                "write_images": self.write_images,
                "embed_images": self.embed_images,
                "table_strategy": self.table_strategy,
                "ignore_code": self.ignore_code,
                **self.extra_config  # Parámetros adicionales específicos
            }
            
            self.logger.info(
                f"Iniciando extracción con pymupdf4llm: {pdf_path.name}, "
                f"configuración: {extraction_config}"
            )

            # ETAPA 3: Extracción del contenido con pymupdf4llm
            # to_markdown() es el método principal que convierte PDF a markdown
            # estructurado optimizado para LLMs
            markdown_content: str = pymupdf4llm.to_markdown(
                doc=str(pdf_path),  # pymupdf4llm acepta string path
                **extraction_config
            )

            # ETAPA 4: Post-procesamiento del contenido extraído
            # Limpiar y normalizar el markdown generado
            processed_content = self._post_process_markdown(markdown_content)
            
            # ETAPA 5: Logging y estadísticas
            content_stats = self._calculate_content_stats(processed_content)
            self.logger.info(
                f"Extracción completada: {content_stats['words']} palabras, "
                f"{content_stats['lines']} líneas, "
                f"{content_stats['characters']} caracteres"
            )
            
            return processed_content

        except Exception as e:
            # Manejo específico de errores de pymupdf4llm
            self.logger.error(f"Error en extracción con pymupdf4llm: {str(e)}")
            
            # Re-raise con contexto específico según el tipo de error
            if "password" in str(e).lower():
                raise PermissionError(
                    f"El PDF está protegido con contraseña: {pdf_path.name}"
                ) from e
            elif "corrupt" in str(e).lower() or "damaged" in str(e).lower():
                raise ValueError(
                    f"El PDF está dañado o corrupto: {pdf_path.name}"
                ) from e
            elif "unsupported" in str(e).lower():
                raise ValueError(
                    f"El PDF contiene características no soportadas: {pdf_path.name}"
                ) from e
            else:
                # Error genérico de pymupdf4llm
                raise RuntimeError(
                    f"Error al procesar PDF con pymupdf4llm: {pdf_path.name}. "
                    f"Detalle: {str(e)}"
                ) from e

    def _post_process_markdown(self, content: str) -> str:
        """
        Post-procesa el contenido markdown generado por pymupdf4llm.
        
        Realiza limpieza y normalización del markdown para optimizar
        la legibilidad y consistencia del formato.
        
        Args:
            content (str): Contenido markdown crudo de pymupdf4llm
            
        Returns:
            str: Contenido markdown procesado y normalizado
        """
        if not content or not content.strip():
            return ""
            
        # Normalizar saltos de línea
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        
        # Eliminar espacios en blanco excesivos al final de las líneas
        lines = [line.rstrip() for line in content.split('\n')]
        
        # Eliminar líneas vacías excesivas (máximo 2 consecutivas)
        processed_lines = []
        empty_count = 0
        
        for line in lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:  # Permitir máximo 2 líneas vacías consecutivas
                    processed_lines.append(line)
            else:
                empty_count = 0
                processed_lines.append(line)
        
        # Asegurar que el documento termine con una sola línea vacía
        while processed_lines and processed_lines[-1].strip() == '':
            processed_lines.pop()
        processed_lines.append('')
        
        return '\n'.join(processed_lines)

    def _calculate_content_stats(self, content: str) -> Dict[str, int]:
        """
        Calcula estadísticas básicas del contenido extraído.
        
        Args:
            content (str): Contenido markdown extraído
            
        Returns:
            Dict[str, int]: Diccionario con estadísticas:
                - characters: Número total de caracteres
                - words: Número aproximado de palabras
                - lines: Número de líneas
                - paragraphs: Número aproximado de párrafos
        """
        if not content:
            return {"characters": 0, "words": 0, "lines": 0, "paragraphs": 0}
            
        lines = content.split('\n')
        words = len(content.split())
        characters = len(content)
        
        # Contar párrafos aproximadamente (líneas no vacías que no son encabezados)
        paragraphs = sum(
            1 for line in lines 
            if line.strip() and not line.strip().startswith('#')
        )
        
        return {
            "characters": characters,
            "words": words,
            "lines": len(lines),
            "paragraphs": paragraphs
        }

    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Retorna un resumen de la configuración actual del adaptador.
        
        Útil para logging, debugging y documentación de la configuración
        utilizada en el procesamiento.
        
        Returns:
            Dict[str, Any]: Configuración actual del adaptador
        """
        return {
            "adapter_type": "PyMuPDF4LLM",
            "page_chunks": self.page_chunks,
            "write_images": self.write_images,
            "embed_images": self.embed_images,
            "table_strategy": self.table_strategy,
            "ignore_code": self.ignore_code,
            "extra_config": self.extra_config
        }
