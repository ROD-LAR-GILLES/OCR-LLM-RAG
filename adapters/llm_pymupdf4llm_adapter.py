# adapters/llm_pymupdf4llm_adapter.py
"""
Adaptador LLM basado en pymupdf4llm para procesamiento optimizado.

Este módulo implementa el puerto LLMDocumentPort utilizando pymupdf4llm,
proporcionando extracción de contenido estructurado específicamente
optimizado para modelos de lenguaje y sistemas RAG.

Diferencias con el adaptador OCR tradicional:
- Genera markdown estructurado en lugar de texto plano
- Incluye metadatos ricos para sistemas RAG
- Proporciona chunking semántico inteligente
- Optimiza el formato para embeddings vectoriales
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json
from datetime import datetime

try:
    import pymupdf4llm
    import pymupdf
except ImportError:
    raise ImportError(
        "pymupdf4llm y/o pymupdf no están instalados. "
        "Instálalos con: pip install pymupdf4llm pymupdf"
    )

from application.ports import LLMDocumentPort


class LLMPyMuPDF4LLMAdapter(LLMDocumentPort):
    """
    Adaptador LLM especializado para pymupdf4llm con funcionalidades avanzadas.
    
    Este adaptador está específicamente diseñado para sistemas que requieren
    contenido optimizado para LLMs, incluyendo:
    - Formato markdown estructurado
    - Metadatos ricos para contexto
    - Chunking semánticamente coherente
    - Estadísticas detalladas del contenido
    - Configuración flexible para diferentes casos de uso
    
    Casos de uso principales:
    - **RAG Systems**: Preparación de documentos para Retrieval-Augmented Generation
    - **Vector Databases**: Contenido optimizado para embeddings vectoriales
    - **LLM Training**: Datasets estructurados para entrenamiento de modelos
    - **Semantic Search**: Indexación optimizada para búsqueda semántica
    - **Document Analysis**: Análisis automatizado por modelos de lenguaje
    
    Configuraciones especializadas:
    - Chunk size adaptativo basado en estructura semántica
    - Preservación de contexto entre chunks
    - Extracción de metadatos enriquecidos
    - Formato optimizado para embeddings
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        preserve_structure: bool = True,
        extract_images: bool = False,
        include_metadata: bool = True,
        semantic_chunking: bool = True,
        table_strategy: str = "lines_strict",
        **kwargs
    ) -> None:
        """
        Inicializa el adaptador LLM con configuraciones específicas.
        
        Args:
            chunk_size (int): Tamaño máximo de chunk en caracteres.
                             Para sistemas RAG, se recomienda 1000-2000 caracteres.
                             
            chunk_overlap (int): Solapamiento entre chunks para preservar contexto.
                                Típicamente 10-20% del chunk_size.
                                
            preserve_structure (bool): Si True, mantiene la estructura de encabezados
                                      en cada chunk para mejor contexto.
                                      
            extract_images (bool): Si True, incluye descripción de imágenes.
                                  Útil para documentos con contenido visual relevante.
                                  
            include_metadata (bool): Si True, incluye metadatos enriquecidos.
                                    Esencial para sistemas RAG y búsqueda semántica.
                                    
            semantic_chunking (bool): Si True, usa chunking basado en estructura
                                     semántica en lugar de tamaño fijo.
                                     
            table_strategy (str): Estrategia para procesamiento de tablas.
                                 Opciones: "lines_strict", "lines", "text", "explicit"
                                 
            **kwargs: Parámetros adicionales para pymupdf4llm
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.preserve_structure = preserve_structure
        self.extract_images = extract_images
        self.include_metadata = include_metadata
        self.semantic_chunking = semantic_chunking
        self.table_strategy = table_strategy
        self.extra_config = kwargs
        
        # Configurar logging
        self.logger = logging.getLogger(__name__)

    def extract_structured_content(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extrae contenido estructurado optimizado para LLMs.
        
        Implementa el pipeline completo de procesamiento:
        1. Extracción de contenido con pymupdf4llm
        2. Generación de metadatos enriquecidos
        3. Análisis de estructura del documento
        4. Chunking semánticamente coherente
        5. Optimización para sistemas RAG
        
        Args:
            pdf_path (Path): Ruta al PDF a procesar
            
        Returns:
            Dict[str, Any]: Contenido estructurado completo para LLMs
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {pdf_path}")

        try:
            self.logger.info(f"Iniciando procesamiento LLM para: {pdf_path.name}")
            
            # ETAPA 1: Extracción de contenido principal
            markdown_content = self._extract_markdown_content(pdf_path)
            
            # ETAPA 2: Extracción de metadatos enriquecidos
            metadata = self._extract_rich_metadata(pdf_path)
            
            # ETAPA 3: Análisis de estructura del documento
            structure = self._analyze_document_structure(markdown_content)
            
            # ETAPA 4: Generación de estadísticas detalladas
            stats = self._calculate_detailed_stats(markdown_content, pdf_path)
            
            # ETAPA 5: Chunking semánticamente coherente
            chunks = self._generate_semantic_chunks(markdown_content, structure)
            
            # ETAPA 6: Construcción del resultado estructurado
            result = {
                "markdown": markdown_content,
                "metadata": metadata,
                "structure": structure,
                "stats": stats,
                "chunks": chunks,
                "processing_info": {
                    "adapter": "LLMPyMuPDF4LLMAdapter",
                    "timestamp": datetime.now().isoformat(),
                    "config": self._get_processing_config()
                }
            }
            
            self.logger.info(
                f"Procesamiento completado: {len(chunks)} chunks generados, "
                f"{stats['words']} palabras totales"
            )
            
            return result

        except Exception as e:
            self.logger.error(f"Error en procesamiento LLM: {str(e)}")
            raise RuntimeError(f"Error al procesar {pdf_path.name}: {str(e)}") from e

    def _extract_markdown_content(self, pdf_path: Path) -> str:
        """
        Extrae el contenido principal en formato markdown.
        
        Args:
            pdf_path (Path): Ruta al archivo PDF
            
        Returns:
            str: Contenido en formato markdown estructurado
        """
        config = {
            "page_chunks": False,  # Manejamos chunking manualmente
            "write_images": self.extract_images,
            "embed_images": False,  # Para LLM preferimos referencias
            "table_strategy": self.table_strategy,
            **self.extra_config
        }
        
        return pymupdf4llm.to_markdown(str(pdf_path), **config)

    def _extract_rich_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extrae metadatos enriquecidos del documento.
        
        Args:
            pdf_path (Path): Ruta al archivo PDF
            
        Returns:
            Dict[str, Any]: Metadatos enriquecidos para sistemas RAG
        """
        metadata = {
            "filename": pdf_path.name,
            "filepath": str(pdf_path),
            "file_size": pdf_path.stat().st_size,
            "modified_date": datetime.fromtimestamp(pdf_path.stat().st_mtime).isoformat()
        }
        
        if not self.include_metadata:
            return metadata
            
        try:
            # Extraer metadatos del PDF usando PyMuPDF
            with pymupdf.open(str(pdf_path)) as doc:
                pdf_metadata = doc.metadata
                
                metadata.update({
                    "title": pdf_metadata.get("title", ""),
                    "author": pdf_metadata.get("author", ""),
                    "subject": pdf_metadata.get("subject", ""),
                    "keywords": pdf_metadata.get("keywords", ""),
                    "creator": pdf_metadata.get("creator", ""),
                    "producer": pdf_metadata.get("producer", ""),
                    "creation_date": pdf_metadata.get("creationDate", ""),
                    "modification_date": pdf_metadata.get("modDate", ""),
                    "page_count": doc.page_count,
                    "is_encrypted": doc.is_encrypted,
                    "has_links": any(page.get_links() for page in doc),
                    "has_annotations": any(page.annots() for page in doc)
                })
                
        except Exception as e:
            self.logger.warning(f"No se pudieron extraer metadatos PDF: {str(e)}")
            
        return metadata

    def _analyze_document_structure(self, content: str) -> Dict[str, Any]:
        """
        Analiza la estructura jerárquica del documento.
        
        Args:
            content (str): Contenido markdown del documento
            
        Returns:
            Dict[str, Any]: Información sobre la estructura del documento
        """
        lines = content.split('\n')
        
        # Detectar encabezados y su jerarquía
        headings = []
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                headings.append({
                    "level": level,
                    "title": title,
                    "line_number": i,
                    "id": f"heading_{len(headings)}"
                })
        
        # Detectar tablas
        table_count = content.count('|')  # Aproximación simple
        
        # Detectar listas
        list_items = sum(1 for line in lines if line.strip().startswith(('-', '*', '+')))
        
        return {
            "headings": headings,
            "heading_count": len(headings),
            "max_heading_level": max((h["level"] for h in headings), default=0),
            "table_count": table_count,
            "list_items": list_items,
            "sections": self._identify_sections(headings),
            "document_type": self._classify_document_type(headings, content)
        }

    def _identify_sections(self, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identifica secciones principales del documento.
        
        Args:
            headings (List[Dict]): Lista de encabezados detectados
            
        Returns:
            List[Dict]: Secciones identificadas con sus características
        """
        sections = []
        current_section = None
        
        for heading in headings:
            if heading["level"] <= 2:  # Encabezados principales
                if current_section:
                    sections.append(current_section)
                    
                current_section = {
                    "title": heading["title"],
                    "level": heading["level"],
                    "start_line": heading["line_number"],
                    "subsections": []
                }
            elif current_section and heading["level"] <= 4:
                current_section["subsections"].append({
                    "title": heading["title"],
                    "level": heading["level"],
                    "line_number": heading["line_number"]
                })
        
        if current_section:
            sections.append(current_section)
            
        return sections

    def _classify_document_type(self, headings: List[Dict], content: str) -> str:
        """
        Clasifica el tipo de documento basado en su estructura.
        
        Args:
            headings (List[Dict]): Encabezados del documento
            content (str): Contenido completo
            
        Returns:
            str: Tipo de documento clasificado
        """
        heading_titles = [h["title"].lower() for h in headings]
        content_lower = content.lower()
        
        # Patrones para diferentes tipos de documentos
        if any(word in content_lower for word in ["abstract", "introduction", "methodology", "results", "conclusion"]):
            return "academic_paper"
        elif any(word in content_lower for word in ["manual", "guide", "tutorial", "instructions"]):
            return "manual"
        elif any(word in content_lower for word in ["report", "analysis", "findings", "recommendations"]):
            return "report"
        elif any(word in content_lower for word in ["specification", "requirements", "design", "architecture"]):
            return "technical_document"
        elif len(headings) < 3 and "table of contents" not in content_lower:
            return "simple_document"
        else:
            return "structured_document"

    def _calculate_detailed_stats(self, content: str, pdf_path: Path) -> Dict[str, Any]:
        """
        Calcula estadísticas detalladas del contenido.
        
        Args:
            content (str): Contenido markdown
            pdf_path (Path): Ruta al archivo PDF
            
        Returns:
            Dict[str, Any]: Estadísticas detalladas para análisis
        """
        lines = content.split('\n')
        words = content.split()
        
        # Estadísticas básicas
        stats = {
            "characters": len(content),
            "words": len(words),
            "lines": len(lines),
            "paragraphs": len([line for line in lines if line.strip() and not line.startswith('#')]),
            "estimated_reading_time_minutes": len(words) / 200,  # ~200 palabras por minuto
            "estimated_tokens": len(words) * 1.3,  # Aproximación: 1 palabra ≈ 1.3 tokens
        }
        
        # Estadísticas avanzadas
        if len(words) > 0:
            stats.update({
                "avg_words_per_line": len(words) / len([l for l in lines if l.strip()]),
                "avg_chars_per_word": len(''.join(words)) / len(words),
                "unique_words": len(set(word.lower().strip('.,!?;:') for word in words)),
                "vocabulary_richness": len(set(word.lower() for word in words)) / len(words)
            })
        
        return stats

    def _generate_semantic_chunks(self, content: str, structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Genera chunks semánticamente coherentes.
        
        Args:
            content (str): Contenido markdown completo
            structure (Dict): Estructura del documento
            
        Returns:
            List[Dict]: Chunks con metadata contextual
        """
        if not self.semantic_chunking:
            return self._generate_fixed_chunks(content)
            
        chunks = []
        lines = content.split('\n')
        current_chunk = []
        current_size = 0
        current_context = {"section": None, "subsection": None}
        
        for i, line in enumerate(lines):
            line_size = len(line)
            
            # Actualizar contexto basado en encabezados
            if line.strip().startswith('#'):
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                
                if level <= 2:
                    current_context["section"] = title
                    current_context["subsection"] = None
                elif level <= 4:
                    current_context["subsection"] = title
                    
                # Si el chunk actual es muy grande y encontramos un encabezado, cortamos
                if current_size > self.chunk_size * 0.7 and current_chunk:
                    chunks.append(self._create_chunk(current_chunk, current_context.copy(), len(chunks)))
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(line)
            current_size += line_size
            
            # Cortar chunk si excede el tamaño máximo
            if current_size >= self.chunk_size:
                chunks.append(self._create_chunk(current_chunk, current_context.copy(), len(chunks)))
                
                # Aplicar overlap
                if self.chunk_overlap > 0:
                    overlap_lines = self._get_overlap_lines(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_lines
                    current_size = sum(len(line) for line in overlap_lines)
                else:
                    current_chunk = []
                    current_size = 0
        
        # Agregar último chunk si hay contenido
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, current_context, len(chunks)))
            
        return chunks

    def _create_chunk(self, lines: List[str], context: Dict[str, Any], chunk_id: int) -> Dict[str, Any]:
        """
        Crea un chunk con metadata contextual.
        
        Args:
            lines (List[str]): Líneas del chunk
            context (Dict): Contexto actual (sección, subsección)
            chunk_id (int): ID único del chunk
            
        Returns:
            Dict[str, Any]: Chunk con metadata
        """
        content = '\n'.join(lines)
        words = content.split()
        
        chunk = {
            "id": chunk_id,
            "content": content,
            "word_count": len(words),
            "char_count": len(content),
            "section": context.get("section"),
            "subsection": context.get("subsection"),
            "estimated_tokens": len(words) * 1.3,
            "preview": content[:100] + "..." if len(content) > 100 else content
        }
        
        # Agregar contexto de encabezado si preserve_structure está habilitado
        if self.preserve_structure and context.get("section"):
            header_context = f"# {context['section']}\n"
            if context.get("subsection"):
                header_context += f"## {context['subsection']}\n"
            chunk["content_with_context"] = header_context + content
        
        return chunk

    def _generate_fixed_chunks(self, content: str) -> List[Dict[str, Any]]:
        """
        Genera chunks de tamaño fijo como fallback.
        
        Args:
            content (str): Contenido completo
            
        Returns:
            List[Dict]: Chunks de tamaño fijo
        """
        chunks = []
        words = content.split()
        
        # Calcular palabras por chunk aproximadamente
        words_per_chunk = self.chunk_size // 6  # ~6 caracteres por palabra
        overlap_words = self.chunk_overlap // 6
        
        for i in range(0, len(words), words_per_chunk - overlap_words):
            chunk_words = words[i:i + words_per_chunk]
            chunk_content = ' '.join(chunk_words)
            
            chunks.append({
                "id": len(chunks),
                "content": chunk_content,
                "word_count": len(chunk_words),
                "char_count": len(chunk_content),
                "section": None,
                "subsection": None,
                "estimated_tokens": len(chunk_words) * 1.3,
                "preview": chunk_content[:100] + "..." if len(chunk_content) > 100 else chunk_content
            })
            
        return chunks

    def _get_overlap_lines(self, lines: List[str], overlap_size: int) -> List[str]:
        """
        Obtiene líneas para overlap entre chunks.
        
        Args:
            lines (List[str]): Líneas del chunk actual
            overlap_size (int): Tamaño de overlap en caracteres
            
        Returns:
            List[str]: Líneas para overlap
        """
        if not lines:
            return []
            
        overlap_lines = []
        current_size = 0
        
        # Tomar líneas desde el final hasta alcanzar el tamaño de overlap
        for line in reversed(lines):
            if current_size + len(line) > overlap_size:
                break
            overlap_lines.insert(0, line)
            current_size += len(line)
            
        return overlap_lines

    def get_chunk_strategy(self) -> Dict[str, Any]:
        """
        Retorna la estrategia de chunking configurada.
        
        Returns:
            Dict[str, Any]: Configuración de chunking actual
        """
        return {
            "method": "semantic" if self.semantic_chunking else "fixed",
            "max_size": self.chunk_size,
            "overlap": self.chunk_overlap,
            "preserve_structure": self.preserve_structure,
            "unit": "characters"
        }

    def _get_processing_config(self) -> Dict[str, Any]:
        """
        Retorna la configuración de procesamiento actual.
        
        Returns:
            Dict[str, Any]: Configuración completa del adaptador
        """
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "preserve_structure": self.preserve_structure,
            "extract_images": self.extract_images,
            "include_metadata": self.include_metadata,
            "semantic_chunking": self.semantic_chunking,
            "table_strategy": self.table_strategy,
            "extra_config": self.extra_config
        }
