# adapters/storage_filesystem.py
"""
Adaptador de almacenamiento basado en sistema de archivos local.

Este módulo implementa el puerto StoragePort para persistir los resultados
del procesamiento OCR en el sistema de archivos local, generando múltiples
formatos de salida para diferentes casos de uso.
"""
import shutil
from pathlib import Path
from typing import List, Any, Tuple
import pandas as pd
from tabulate import tabulate
from datetime import datetime

from src.application.ports import StoragePort


class FileStorage(StoragePort):
    """
    Adaptador de almacenamiento que persiste resultados en el sistema de archivos.
    
    NUEVA FUNCIONALIDAD: Crea una carpeta dedicada por cada documento procesado
    para mejor organización y evitar conflictos de archivos.
    
    Esta implementación genera múltiples formatos de salida organizados por documento:
    
    Estructura de directorios:
    resultado/
    ├── documento1/
    │   ├── texto_completo.txt           <- Texto plano extraído por OCR
    │   ├── documento1.md                <- Documento Markdown con texto y tablas
    │   └── documento1_original.pdf      <- Copia del archivo original
    └── documento2/
        ├── texto_completo.txt
        ├── documento2.md
        └── documento2_original.pdf
    
    Ventajas de la organización por carpetas:
    - Evita conflictos de nombres entre documentos
    - Facilita el backup y archivado por documento
    - Permite procesamiento de múltiples archivos con el mismo nombre
    - Estructura más clara para herramientas de automatización
    - Mejor integración con sistemas de versionado
    
    Formatos generados por documento:
    1. texto_completo.txt - Texto plano extraído por OCR (legible por humanos)
    2. [nombre].md - Documento Markdown estructurado con texto y tablas (documentación)
    3. [nombre]_original.pdf - Copia del archivo original (trazabilidad)
    
    Ventajas del almacenamiento en archivos:
    - Simple y rápido de implementar
    - No requiere infraestructura adicional (BD, cloud)
    - Fácil integración con herramientas de línea de comandos
    - Formatos estándar legibles por múltiples aplicaciones
    - Backup y versionado simple con herramientas estándar
    
    Limitaciones:
    - No soporta consultas complejas
    - Sin control de concurrencia
    - Escalabilidad limitada para grandes volúmenes
    - Sin índices para búsqueda rápida
    """

    def __init__(self, out_dir: Path) -> None:
        """
        Inicializa el adaptador de almacenamiento con directorio de salida.
        
        Args:
            out_dir (Path): Directorio donde se guardarán los archivos procesados.
                           Se crea automáticamente si no existe.
                           
        Note:
            - parents=True crea directorios padre si no existen
            - exist_ok=True evita errores si el directorio ya existe
        """
        self.out_dir = out_dir
        # Crea la estructura de directorios de forma segura
        # parents=True equivale a 'mkdir -p' en Unix
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def save(self, name: str, text: str, tables: List[Any], original: Path) -> List[str]:
        """
        Persiste el contenido en carpeta dedicada con múltiples formatos.
        
        NUEVA IMPLEMENTACIÓN: Organiza archivos por documento en carpetas separadas
        
        Args:
            name (str): Nombre del documento (sin extensión)
            text (str): Texto extraído (plain text o markdown según el adaptador)
            tables (List[Any]): Lista de tablas (DataFrames o datos estructurados)
            original (Path): Ruta al archivo PDF original
            
        Returns:
            List[str]: Lista de rutas absolutas de archivos generados
        """
        # ETAPA 1: Crear carpeta dedicada para el documento
        document_dir = self.out_dir / name
        document_dir.mkdir(parents=True, exist_ok=True)
        
        archivos_generados = []
        
        try:
            # ETAPA 2: Detectar el tipo de contenido y guardar apropiadamente
            if self._is_markdown_content(text):
                # Para contenido LLM (markdown estructurado)
                archivos_generados.extend(
                    self._save_llm_content(document_dir, name, text, tables, original)
                )
            else:
                # Para contenido OCR tradicional (texto plano)
                archivos_generados.extend(
                    self._save_traditional_content(document_dir, name, text, tables, original)
                )
            
            return archivos_generados
            
        except Exception as e:
            # Si hay error, limpiar carpeta parcialmente creada
            if document_dir.exists() and not any(document_dir.iterdir()):
                document_dir.rmdir()
            raise RuntimeError(f"Error guardando archivos para {name}: {str(e)}") from e

    def _is_markdown_content(self, text: str) -> bool:
        """
        Detecta si el contenido es markdown estructurado.
        
        Args:
            text (str): Contenido a analizar
            
        Returns:
            bool: True si es contenido markdown
        """
        markdown_indicators = [
            text.strip().startswith('#'),  # Encabezados markdown
            '##' in text,                  # Múltiples niveles de encabezado
            '|' in text and '---' in text, # Tablas markdown
            text.count('\n#') > 2          # Múltiples secciones
        ]
        
        return any(markdown_indicators)

    def _save_llm_content(
        self, 
        document_dir: Path, 
        name: str, 
        markdown_text: str, 
        tables: List[Any], 
        original: Path
    ) -> List[str]:
        """
        Guarda contenido optimizado para LLMs (markdown estructurado).
        
        Args:
            document_dir (Path): Directorio del documento
            name (str): Nombre del documento
            markdown_text (str): Contenido markdown estructurado
            tables (List[Any]): Tablas extraídas (si las hay)
            original (Path): Archivo original
            
        Returns:
            List[str]: Archivos generados para contenido LLM
        """
        archivos_generados = []
        
        # 1. Archivo markdown principal (contenido completo)
        markdown_file = document_dir / f"{name}.md"
        self._save_markdown_with_metadata(markdown_file, markdown_text, original, name)
        archivos_generados.append(str(markdown_file))
        
        # 2. Archivo de texto plano (para compatibilidad)
        text_file = document_dir / "contenido_completo.txt"
        plain_text = self._markdown_to_plain_text(markdown_text)
        self._write_text_file(text_file, plain_text)
        archivos_generados.append(str(text_file))
        
        # 3. Copia del archivo original (trazabilidad)
        original_copy = document_dir / f"{name}_original.pdf"
        shutil.copy2(original, original_copy)
        archivos_generados.append(str(original_copy))
        
        # 4. Metadatos del procesamiento LLM
        metadata_file = document_dir / f"{name}_metadata.json"
        self._save_llm_metadata(metadata_file, markdown_text, original, tables)
        archivos_generados.append(str(metadata_file))
        
        return archivos_generados

    def _save_traditional_content(
        self, 
        document_dir: Path, 
        name: str, 
        text: str, 
        tables: List[Any], 
        original: Path
    ) -> List[str]:
        """
        Guarda contenido OCR tradicional (texto plano + tablas).
        
        Args:
            document_dir (Path): Directorio del documento
            name (str): Nombre del documento
            text (str): Texto extraído por OCR
            tables (List[Any]): Tablas extraídas
            original (Path): Archivo original
            
        Returns:
            List[str]: Archivos generados para contenido tradicional
        """
        archivos_generados = []
        
        # 1. Archivo de texto completo
        text_file = document_dir / "texto_completo.txt"
        self._write_text_file(text_file, text)
        archivos_generados.append(str(text_file))
        
        # 2. Documento Markdown combinado
        markdown_file = document_dir / f"{name}.md"
        combined_content = self._generate_combined_markdown(text, tables, original, name)
        self._write_text_file(markdown_file, combined_content)
        archivos_generados.append(str(markdown_file))
        
        # 3. Copia del archivo original
        original_copy = document_dir / f"{name}_original.pdf"
        shutil.copy2(original, original_copy)
        archivos_generados.append(str(original_copy))
        
        # 4. Tablas individuales si existen
        if tables:
            table_files = self._save_individual_tables(document_dir, name, tables)
            archivos_generados.extend(table_files)
        
        return archivos_generados

    def _save_markdown_with_metadata(
        self, 
        file_path: Path, 
        markdown_content: str, 
        original: Path, 
        name: str
    ) -> None:
        """
        Guarda contenido markdown con metadatos enriquecidos.
        
        Args:
            file_path (Path): Ruta del archivo markdown
            markdown_content (str): Contenido principal
            original (Path): Archivo original
            name (str): Nombre del documento
        """
        # Agregar frontmatter YAML para metadatos
        frontmatter = f"""---
title: "{name}"
source_file: "{original.name}"
processed_date: "{datetime.now().isoformat()}"
processor: "pymupdf4llm"
format: "markdown_structured"
optimized_for: "llm_processing"
---

"""
        
        full_content = frontmatter + markdown_content
        self._write_text_file(file_path, full_content)

    def _save_llm_metadata(
        self, 
        file_path: Path, 
        content: str, 
        original: Path, 
        tables: List[Any]
    ) -> None:
        """
        Guarda metadatos específicos para sistemas LLM.
        
        Args:
            file_path (Path): Ruta del archivo de metadatos
            content (str): Contenido procesado
            original (Path): Archivo original
            tables (List[Any]): Tablas extraídas
        """
        import json
        
        # Calcular estadísticas básicas
        lines = content.split('\n')
        words = content.split()
        
        metadata = {
            "processing_info": {
                "processor": "pymupdf4llm",
                "processed_date": datetime.now().isoformat(),
                "source_file": str(original),
                "optimization": "llm_ready"
            },
            "content_stats": {
                "characters": len(content),
                "words": len(words),
                "lines": len(lines),
                "estimated_tokens": len(words) * 1.3,
                "estimated_reading_time_minutes": len(words) / 200
            },
            "structure_info": {
                "heading_count": content.count('\n#'),
                "table_count": len(tables) if tables else 0,
                "has_images": "![" in content,
                "markdown_formatted": True
            },
            "llm_recommendations": {
                "suitable_for_rag": len(words) > 100,
                "chunking_recommended": len(words) > 1000,
                "embedding_ready": True,
                "context_preserved": "##" in content
            }
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def _markdown_to_plain_text(self, markdown: str) -> str:
        """
        Convierte markdown a texto plano para compatibilidad.
        
        Args:
            markdown (str): Contenido markdown
            
        Returns:
            str: Texto plano sin formato markdown
        """
        import re
        
        # Remover encabezados markdown
        text = re.sub(r'^#+\s*', '', markdown, flags=re.MULTILINE)
        
        # Remover formato de tablas básico
        text = re.sub(r'\|.*\|', lambda m: m.group().replace('|', ' '), text)
        text = re.sub(r'-+\s*\|-+', '', text)
        
        # Remover enlaces markdown
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remover formato de texto
        text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^\*]+)\*', r'\1', text)      # Italic
        
        # Limpiar líneas vacías excesivas
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        
        return text.strip()

    # ...existing code...