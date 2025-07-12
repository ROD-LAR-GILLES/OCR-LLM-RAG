# application/llm_use_cases.py
"""
Casos de uso específicos para procesamiento LLM de documentos.

Este módulo contiene casos de uso especializados para extraer y procesar
contenido de documentos de forma optimizada para modelos de lenguaje,
sistemas RAG y aplicaciones de IA.

Diferencias con use_cases.py tradicional:
- Enfoque en formato markdown estructurado
- Metadatos enriquecidos para contexto
- Chunking semánticamente coherente
- Optimización para embeddings vectoriales
- Preparación para sistemas RAG
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import logging

from application.ports import LLMDocumentPort, StoragePort


class ProcessDocumentForLLM:
    """
    Caso de uso principal para procesamiento de documentos optimizado para LLMs.
    
    Este caso de uso orquesta el flujo completo de procesamiento específico
    para aplicaciones que utilizan modelos de lenguaje:
    
    1. Extracción de contenido estructurado optimizado para LLMs
    2. Generación de metadatos enriquecidos para contexto
    3. Chunking semánticamente coherente para sistemas RAG
    4. Persistencia en formatos optimizados para IA
    5. Generación de estadísticas para análisis
    
    Casos de uso principales:
    - **RAG Systems**: Preparación de documentos para Retrieval-Augmented Generation
    - **Vector Databases**: Contenido optimizado para embeddings
    - **LLM Training**: Datasets estructurados para entrenamiento
    - **Semantic Search**: Indexación optimizada para búsqueda semántica
    - **Document Analysis**: Análisis automatizado por LLMs
    
    Ventajas sobre el procesamiento tradicional:
    - Preserva la estructura semántica del documento
    - Genera chunks coherentes para mejor contexto
    - Incluye metadatos ricos para sistemas RAG
    - Optimiza el formato para modelos de lenguaje
    - Proporciona estadísticas avanzadas para análisis
    """

    def __init__(
        self,
        llm_processor: LLMDocumentPort,
        storage: StoragePort,
    ) -> None:
        """
        Inicializa el caso de uso con las dependencias inyectadas.
        
        Args:
            llm_processor (LLMDocumentPort): Servicio de procesamiento LLM.
                                           Extrae contenido optimizado para modelos de lenguaje.
                                           
            storage (StoragePort): Servicio de persistencia de resultados.
                                  Guarda el contenido en formatos optimizados para IA.
        
        Note:
            La inyección de dependencias permite flexibilidad total:
            - Testing: Usar mocks en lugar de implementaciones reales
            - Configuración: Elegir implementaciones según el caso de uso
            - Escalabilidad: Intercambiar componentes según necesidades
        """
        self.llm_processor = llm_processor
        self.storage = storage
        self.logger = logging.getLogger(__name__)

    def __call__(self, pdf_path: Path, output_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ejecuta el procesamiento completo de un documento para LLMs.
        
        Implementa el pipeline optimizado para aplicaciones de IA:
        1. Procesamiento con pymupdf4llm para contenido estructurado
        2. Extracción de metadatos enriquecidos para contexto RAG
        3. Chunking semánticamente coherente para mejor retrieval
        4. Persistencia en múltiples formatos (markdown, JSON, chunks)
        5. Generación de estadísticas y métricas de calidad
        
        Args:
            pdf_path (Path): Ruta absoluta al archivo PDF a procesar
            output_config (Optional[Dict]): Configuración de salida personalizada:
                - save_chunks_separately: bool - Guardar chunks en archivos separados
                - include_embeddings_metadata: bool - Incluir metadata para embeddings
                - export_json: bool - Exportar resultado completo en JSON
                - generate_summary: bool - Generar resumen del documento
        
        Returns:
            Dict[str, Any]: Resultado completo del procesamiento:
                - 'success': bool - Indica si el procesamiento fue exitoso
                - 'document_id': str - ID único del documento procesado
                - 'content': Dict - Contenido estructurado completo
                - 'files_generated': List[str] - Rutas de archivos generados
                - 'processing_stats': Dict - Estadísticas del procesamiento
                - 'chunk_info': Dict - Información sobre chunks generados
                - 'recommendations': List[str] - Recomendaciones para uso en LLMs
                
        Raises:
            FileNotFoundError: Si el archivo PDF no existe
            LLMProcessingError: Si hay errores en el procesamiento específico para LLM
            StorageError: Si hay problemas al persistir los resultados
            
        Example:
            >>> # Configuración básica
            >>> processor = ProcessDocumentForLLM(
            ...     llm_processor=LLMPyMuPDF4LLMAdapter(
            ...         chunk_size=1000,
            ...         semantic_chunking=True
            ...     ),
            ...     storage=FileStorage(Path("./output"))
            ... )
            >>> 
            >>> # Procesamiento con configuración personalizada
            >>> config = {
            ...     "save_chunks_separately": True,
            ...     "include_embeddings_metadata": True,
            ...     "export_json": True
            ... }
            >>> result = processor(Path("document.pdf"), config)
            >>> 
            >>> print(f"Documento procesado: {result['document_id']}")
            >>> print(f"Chunks generados: {result['chunk_info']['total_chunks']}")
            >>> print(f"Archivos creados: {len(result['files_generated'])}")
            
        Performance Notes:
            - pymupdf4llm es ~10x más rápido que OCR tradicional
            - El chunking semántico agrega ~20% de tiempo pero mejora la calidad
            - Los metadatos enriquecidos agregan procesamiento mínimo
            - La persistencia en múltiples formatos puede duplicar el tiempo de I/O
        """
        try:
            self.logger.info(f"Iniciando procesamiento LLM para: {pdf_path.name}")
            
            # Configuración de salida por defecto
            config = {
                "save_chunks_separately": True,
                "include_embeddings_metadata": True,
                "export_json": True,
                "generate_summary": False,
                **(output_config or {})
            }
            
            # ETAPA 1: Procesamiento con LLM adapter
            self.logger.info("Extrayendo contenido estructurado...")
            structured_content = self.llm_processor.extract_structured_content(pdf_path)
            
            # ETAPA 2: Enriquecimiento del contenido
            self.logger.info("Enriqueciendo contenido...")
            enriched_content = self._enrich_content(structured_content, pdf_path, config)
            
            # ETAPA 3: Persistencia optimizada
            self.logger.info("Guardando resultados...")
            files_generated = self._save_llm_optimized_content(
                pdf_path.stem, 
                enriched_content, 
                pdf_path, 
                config
            )
            
            # ETAPA 4: Generación de estadísticas y recomendaciones
            processing_stats = self._calculate_processing_stats(enriched_content)
            recommendations = self._generate_llm_recommendations(enriched_content)
            
            # ETAPA 5: Construcción del resultado
            result = {
                "success": True,
                "document_id": self._generate_document_id(pdf_path),
                "content": enriched_content,
                "files_generated": files_generated,
                "processing_stats": processing_stats,
                "chunk_info": self._extract_chunk_info(enriched_content),
                "recommendations": recommendations,
                "processing_config": config
            }
            
            self.logger.info(
                f"Procesamiento LLM completado: {len(files_generated)} archivos generados, "
                f"{len(enriched_content.get('chunks', []))} chunks creados"
            )
            
            return result

        except Exception as e:
            self.logger.error(f"Error en procesamiento LLM: {str(e)}")
            
            # Retornar resultado de error estructurado
            return {
                "success": False,
                "document_id": self._generate_document_id(pdf_path),
                "error": str(e),
                "error_type": type(e).__name__,
                "files_generated": [],
                "processing_stats": {},
                "recommendations": [
                    "Revisar que el archivo PDF no esté dañado",
                    "Verificar que pymupdf4llm esté instalado correctamente",
                    "Intentar con configuraciones más permisivas"
                ]
            }

    def _enrich_content(
        self, 
        content: Dict[str, Any], 
        pdf_path: Path, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enriquece el contenido con metadata adicional para LLMs.
        
        Args:
            content (Dict): Contenido estructurado del LLM processor
            pdf_path (Path): Ruta al archivo original
            config (Dict): Configuración de procesamiento
            
        Returns:
            Dict[str, Any]: Contenido enriquecido con metadata adicional
        """
        enriched = content.copy()
        
        # Agregar información del contexto de procesamiento
        enriched["processing_context"] = {
            "source_file": str(pdf_path),
            "processing_method": "pymupdf4llm",
            "chunk_strategy": self.llm_processor.get_chunk_strategy(),
            "optimization_target": "llm_processing"
        }
        
        # Generar metadata para embeddings si está habilitado
        if config.get("include_embeddings_metadata", False):
            enriched["embeddings_metadata"] = self._generate_embeddings_metadata(content)
        
        # Generar resumen si está habilitado
        if config.get("generate_summary", False):
            enriched["document_summary"] = self._generate_document_summary(content)
            
        return enriched

    def _generate_embeddings_metadata(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera metadata específico para sistemas de embeddings.
        
        Args:
            content (Dict): Contenido estructurado
            
        Returns:
            Dict[str, Any]: Metadata optimizado para embeddings
        """
        chunks = content.get("chunks", [])
        
        return {
            "chunk_count": len(chunks),
            "avg_chunk_size": sum(c.get("char_count", 0) for c in chunks) / len(chunks) if chunks else 0,
            "recommended_embedding_model": self._recommend_embedding_model(content),
            "chunk_overlap_ratio": self.llm_processor.get_chunk_strategy().get("overlap", 0) / 
                                 self.llm_processor.get_chunk_strategy().get("max_size", 1),
            "content_density": self._calculate_content_density(content),
            "semantic_coherence_score": self._estimate_semantic_coherence(chunks)
        }

    def _recommend_embedding_model(self, content: Dict[str, Any]) -> str:
        """
        Recomienda un modelo de embeddings basado en las características del contenido.
        
        Args:
            content (Dict): Contenido analizado
            
        Returns:
            str: Recomendación de modelo de embeddings
        """
        stats = content.get("stats", {})
        avg_words = stats.get("words", 0) / max(len(content.get("chunks", [])), 1)
        doc_type = content.get("structure", {}).get("document_type", "unknown")
        
        if doc_type == "academic_paper":
            return "sentence-transformers/allenai-specter"
        elif doc_type == "technical_document":
            return "sentence-transformers/all-MiniLM-L6-v2"
        elif avg_words > 500:
            return "text-embedding-3-large"  # OpenAI para documentos largos
        else:
            return "text-embedding-3-small"  # OpenAI para documentos cortos

    def _calculate_content_density(self, content: Dict[str, Any]) -> float:
        """
        Calcula la densidad de contenido (información por caractér).
        
        Args:
            content (Dict): Contenido analizado
            
        Returns:
            float: Score de densidad de contenido (0.0 - 1.0)
        """
        stats = content.get("stats", {})
        chars = stats.get("characters", 1)
        words = stats.get("words", 0)
        unique_words = stats.get("unique_words", 0)
        
        # Factores que contribuyen a la densidad
        word_density = words / chars if chars > 0 else 0
        vocabulary_richness = unique_words / words if words > 0 else 0
        structure_richness = len(content.get("structure", {}).get("headings", [])) / words * 100 if words > 0 else 0
        
        # Combinar factores (normalizado a 0-1)
        density = min(1.0, (word_density * 10 + vocabulary_richness + structure_richness) / 3)
        return round(density, 3)

    def _estimate_semantic_coherence(self, chunks: List[Dict[str, Any]]) -> float:
        """
        Estima la coherencia semántica entre chunks.
        
        Args:
            chunks (List[Dict]): Lista de chunks generados
            
        Returns:
            float: Score de coherencia estimado (0.0 - 1.0)
        """
        if len(chunks) < 2:
            return 1.0
            
        # Métrica simple basada en:
        # 1. Consistencia en el tamaño de chunks
        # 2. Presencia de contexto estructural
        # 3. Distribución de contenido
        
        sizes = [c.get("char_count", 0) for c in chunks]
        avg_size = sum(sizes) / len(sizes)
        size_variance = sum((s - avg_size) ** 2 for s in sizes) / len(sizes)
        size_consistency = 1.0 - min(1.0, size_variance / (avg_size ** 2))
        
        # Penalizar chunks muy pequeños o muy grandes
        size_penalty = sum(1 for s in sizes if s < avg_size * 0.3 or s > avg_size * 2) / len(sizes)
        
        # Bonus por preservación de estructura
        structure_bonus = sum(1 for c in chunks if c.get("section")) / len(chunks) * 0.2
        
        coherence = max(0.0, min(1.0, size_consistency - size_penalty + structure_bonus))
        return round(coherence, 3)

    def _generate_document_summary(self, content: Dict[str, Any]) -> Dict[str, str]:
        """
        Genera un resumen básico del documento.
        
        Args:
            content (Dict): Contenido estructurado
            
        Returns:
            Dict[str, str]: Resumen del documento
        """
        markdown = content.get("markdown", "")
        structure = content.get("structure", {})
        stats = content.get("stats", {})
        
        # Extraer primer párrafo como introducción
        lines = markdown.split('\n')
        intro = ""
        for line in lines:
            if line.strip() and not line.startswith('#') and len(line.strip()) > 50:
                intro = line.strip()[:200] + "..."
                break
        
        # Generar resumen de estructura
        headings = structure.get("headings", [])
        main_sections = [h["title"] for h in headings if h.get("level", 0) <= 2][:5]
        
        return {
            "introduction": intro,
            "main_sections": ", ".join(main_sections) if main_sections else "Sin secciones identificadas",
            "document_type": structure.get("document_type", "unknown"),
            "estimated_reading_time": f"{stats.get('estimated_reading_time_minutes', 0):.1f} minutos",
            "complexity": "Alta" if len(headings) > 10 else "Media" if len(headings) > 3 else "Baja"
        }

    def _save_llm_optimized_content(
        self, 
        name: str, 
        content: Dict[str, Any], 
        original_path: Path, 
        config: Dict[str, Any]
    ) -> List[str]:
        """
        Guarda el contenido en formatos optimizados para LLMs.
        
        Args:
            name (str): Nombre base del documento
            content (Dict): Contenido estructurado
            original_path (Path): Ruta al archivo original
            config (Dict): Configuración de salida
            
        Returns:
            List[str]: Lista de archivos generados
        """
        files_generated = []
        
        # Guardar contenido principal como markdown
        markdown_content = content.get("markdown", "")
        files_generated.extend(
            self.storage.save(name, markdown_content, [], original_path)
        )
        
        # Guardar chunks por separado si está habilitado
        if config.get("save_chunks_separately", False):
            chunks = content.get("chunks", [])
            for i, chunk in enumerate(chunks):
                chunk_name = f"{name}_chunk_{i+1:03d}"
                chunk_content = chunk.get("content", "")
                chunk_files = self.storage.save(chunk_name, chunk_content, [], original_path)
                files_generated.extend(chunk_files)
        
        # Exportar JSON completo si está habilitado
        if config.get("export_json", False):
            json_content = json.dumps(content, indent=2, ensure_ascii=False)
            json_name = f"{name}_complete_data"
            json_files = self.storage.save(json_name, json_content, [], original_path)
            files_generated.extend(json_files)
            
        return files_generated

    def _calculate_processing_stats(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula estadísticas del procesamiento.
        
        Args:
            content (Dict): Contenido procesado
            
        Returns:
            Dict[str, Any]: Estadísticas del procesamiento
        """
        chunks = content.get("chunks", [])
        stats = content.get("stats", {})
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(c.get("char_count", 0) for c in chunks) / len(chunks) if chunks else 0,
            "min_chunk_size": min((c.get("char_count", 0) for c in chunks), default=0),
            "max_chunk_size": max((c.get("char_count", 0) for c in chunks), default=0),
            "total_tokens_estimated": sum(c.get("estimated_tokens", 0) for c in chunks),
            "processing_efficiency": stats.get("words", 0) / max(stats.get("characters", 1), 1),
            "content_density": self._calculate_content_density(content),
            "semantic_coherence": self._estimate_semantic_coherence(chunks)
        }

    def _extract_chunk_info(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrae información detallada sobre los chunks.
        
        Args:
            content (Dict): Contenido con chunks
            
        Returns:
            Dict[str, Any]: Información sobre chunks
        """
        chunks = content.get("chunks", [])
        
        # Agrupar chunks por sección
        sections = {}
        for chunk in chunks:
            section = chunk.get("section", "Sin sección")
            if section not in sections:
                sections[section] = []
            sections[section].append(chunk["id"])
        
        return {
            "total_chunks": len(chunks),
            "chunks_with_section": sum(1 for c in chunks if c.get("section")),
            "chunks_with_subsection": sum(1 for c in chunks if c.get("subsection")),
            "sections_distribution": {k: len(v) for k, v in sections.items()},
            "avg_tokens_per_chunk": sum(c.get("estimated_tokens", 0) for c in chunks) / len(chunks) if chunks else 0,
            "chunk_strategy": self.llm_processor.get_chunk_strategy()
        }

    def _generate_llm_recommendations(self, content: Dict[str, Any]) -> List[str]:
        """
        Genera recomendaciones para uso del contenido con LLMs.
        
        Args:
            content (Dict): Contenido analizado
            
        Returns:
            List[str]: Lista de recomendaciones
        """
        recommendations = []
        
        chunks = content.get("chunks", [])
        stats = content.get("stats", {})
        structure = content.get("structure", {})
        
        # Recomendaciones basadas en número de chunks
        if len(chunks) > 50:
            recommendations.append(
                "Documento largo: Considera usar un sistema de retrieval (RAG) "
                "para seleccionar chunks relevantes antes de enviarse al LLM"
            )
        elif len(chunks) < 5:
            recommendations.append(
                "Documento corto: Puede procesarse completo por la mayoría de LLMs"
            )
        
        # Recomendaciones basadas en estructura
        if len(structure.get("headings", [])) > 15:
            recommendations.append(
                "Documento bien estructurado: Los chunks preservan jerarquía, "
                "ideal para procesamiento contextual"
            )
        
        # Recomendaciones basadas en tipo de documento
        doc_type = structure.get("document_type", "")
        if doc_type == "academic_paper":
            recommendations.append(
                "Documento académico: Usa modelos especializados como Specter "
                "para embeddings científicos"
            )
        elif doc_type == "technical_document":
            recommendations.append(
                "Documento técnico: Considera usar prompts especializados "
                "para análisis técnico"
            )
        
        # Recomendaciones basadas en tokens
        total_tokens = sum(c.get("estimated_tokens", 0) for c in chunks)
        if total_tokens > 100000:
            recommendations.append(
                "Documento muy largo: Implementa estrategias de summarización "
                "o procesamiento por partes"
            )
        
        return recommendations

    def _generate_document_id(self, pdf_path: Path) -> str:
        """
        Genera un ID único para el documento.
        
        Args:
            pdf_path (Path): Ruta al archivo PDF
            
        Returns:
            str: ID único del documento
        """
        import hashlib
        from datetime import datetime
        
        # Combinar nombre, tamaño y timestamp para ID único
        content = f"{pdf_path.name}_{pdf_path.stat().st_size}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
