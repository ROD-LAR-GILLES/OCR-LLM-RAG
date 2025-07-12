# application/llm_controllers.py
"""
Controladores específicos para procesamiento LLM de documentos.

Este módulo implementa controladores especializados para casos de uso
relacionados con modelos de lenguaje, sistemas RAG y aplicaciones de IA,
proporcionando una interfaz limpia entre la lógica de negocio y las
interfaces de usuario.

Diferencias con controllers.py tradicional:
- Enfoque específico en preparación de contenido para LLMs
- Configuraciones optimizadas para sistemas RAG
- Manejo de chunks y embeddings
- Soporte para múltiples formatos de salida orientados a IA
"""
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass

from src.application.llm_use_cases import ProcessDocumentForLLM
from src.application.ports import LLMDocumentPort, StoragePort
from adapters.llm_pymupdf4llm_adapter import PyMuPDF4LLMAdapter
from adapters.storage_filesystem import FileStorage


@dataclass
class LLMProcessingConfig:
    """
    Configuración para procesamiento LLM de documentos.
    
    Esta clase encapsula todas las opciones de configuración disponibles
    para el procesamiento optimizado para modelos de lenguaje.
    
    Attributes:
        chunk_size (int): Tamaño máximo de cada chunk en caracteres
        chunk_overlap (int): Solapamiento entre chunks en caracteres
        semantic_chunking (bool): Si usar chunking basado en estructura semántica
        preserve_structure (bool): Si preservar jerarquía de encabezados en chunks
        extract_images (bool): Si incluir información sobre imágenes
        include_metadata (bool): Si incluir metadatos enriquecidos
        save_chunks_separately (bool): Si guardar cada chunk en archivo separado
        export_json (bool): Si exportar datos completos en JSON
        generate_summary (bool): Si generar resumen del documento
        table_strategy (str): Estrategia para procesamiento de tablas
        optimization_target (str): Objetivo de optimización ('rag', 'embeddings', 'analysis')
    """
    
    # Configuración de chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    semantic_chunking: bool = True
    preserve_structure: bool = True
    
    # Configuración de contenido
    extract_images: bool = False
    include_metadata: bool = True
    table_strategy: str = "lines_strict"
    
    # Configuración de salida
    save_chunks_separately: bool = True
    export_json: bool = True
    generate_summary: bool = False
    
    # Configuración de optimización
    optimization_target: str = "rag"  # 'rag', 'embeddings', 'analysis'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la configuración a diccionario."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "semantic_chunking": self.semantic_chunking,
            "preserve_structure": self.preserve_structure,
            "extract_images": self.extract_images,
            "include_metadata": self.include_metadata,
            "table_strategy": self.table_strategy,
            "save_chunks_separately": self.save_chunks_separately,
            "export_json": self.export_json,
            "generate_summary": self.generate_summary,
            "optimization_target": self.optimization_target
        }
    
    @classmethod
    def for_rag_system(cls, chunk_size: int = 1000) -> 'LLMProcessingConfig':
        """
        Configuración optimizada para sistemas RAG.
        
        Args:
            chunk_size (int): Tamaño de chunk optimizado para el modelo LLM
            
        Returns:
            LLMProcessingConfig: Configuración para RAG
        """
        return cls(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.2),  # 20% overlap
            semantic_chunking=True,
            preserve_structure=True,
            include_metadata=True,
            save_chunks_separately=True,
            export_json=True,
            optimization_target="rag"
        )
    
    @classmethod
    def for_embeddings(cls, chunk_size: int = 512) -> 'LLMProcessingConfig':
        """
        Configuración optimizada para generación de embeddings.
        
        Args:
            chunk_size (int): Tamaño optimizado para modelos de embeddings
            
        Returns:
            LLMProcessingConfig: Configuración para embeddings
        """
        return cls(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.1),  # 10% overlap menor
            semantic_chunking=True,
            preserve_structure=False,  # Menos estructura para embeddings
            include_metadata=True,
            save_chunks_separately=True,
            export_json=False,  # Menos archivos para embeddings
            optimization_target="embeddings"
        )
    
    @classmethod
    def for_analysis(cls, chunk_size: int = 2000) -> 'LLMProcessingConfig':
        """
        Configuración optimizada para análisis por LLMs.
        
        Args:
            chunk_size (int): Tamaño más grande para análisis contextual
            
        Returns:
            LLMProcessingConfig: Configuración para análisis
        """
        return cls(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.15),  # 15% overlap
            semantic_chunking=True,
            preserve_structure=True,
            include_metadata=True,
            extract_images=True,  # Incluir imágenes para análisis completo
            save_chunks_separately=False,  # Menos fragmentación
            export_json=True,
            generate_summary=True,  # Resumen para análisis
            optimization_target="analysis"
        )


class LLMDocumentController:
    """
    Controlador principal para procesamiento LLM de documentos.
    
    Este controlador orquesta el procesamiento de documentos específicamente
    optimizado para modelos de lenguaje, sistemas RAG y aplicaciones de IA.
    Proporciona una interfaz simple para casos de uso complejos.
    
    Responsabilidades:
    - Configurar adaptadores según el caso de uso
    - Orquestar el procesamiento completo
    - Manejar errores específicos de LLM
    - Proporcionar feedback detallado sobre el procesamiento
    - Optimizar configuraciones según el objetivo
    
    Casos de uso soportados:
    - Preparación de documentos para sistemas RAG
    - Generación de datasets para embeddings
    - Análisis de documentos por LLMs
    - Indexación para búsqueda semántica
    """

    def __init__(self, output_dir: Path):
        """
        Inicializa el controlador con el directorio de salida.
        
        Args:
            output_dir (Path): Directorio donde se guardarán los resultados
        """
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)

    def process_document(
        self, 
        pdf_path: Path, 
        config: Optional[LLMProcessingConfig] = None
    ) -> Dict[str, Any]:
        """
        Procesa un documento PDF optimizado para LLMs.
        
        Args:
            pdf_path (Path): Ruta al documento PDF
            config (Optional[LLMProcessingConfig]): Configuración de procesamiento.
                                                   Si None, usa configuración por defecto para RAG.
            
        Returns:
            Dict[str, Any]: Resultado del procesamiento con métricas y archivos generados
            
        Example:
            >>> controller = LLMDocumentController(Path("./output"))
            >>> 
            >>> # Configuración para RAG
            >>> rag_config = LLMProcessingConfig.for_rag_system(chunk_size=1000)
            >>> result = controller.process_document(Path("document.pdf"), rag_config)
            >>> 
            >>> print(f"Chunks generados: {result['chunk_info']['total_chunks']}")
            >>> print(f"Recomendaciones: {result['recommendations']}")
        """
        # Usar configuración por defecto si no se proporciona
        if config is None:
            config = LLMProcessingConfig.for_rag_system()
            
        self.logger.info(f"Procesando documento para LLM: {pdf_path.name}")
        self.logger.info(f"Configuración: {config.optimization_target}")
        
        try:
            # ETAPA 1: Crear adaptadores específicos
            llm_adapter = self._create_llm_adapter(config)
            storage_adapter = self._create_storage_adapter()
            
            # ETAPA 2: Crear caso de uso
            processor = ProcessDocumentForLLM(
                llm_processor=llm_adapter,
                storage=storage_adapter
            )
            
            # ETAPA 3: Ejecutar procesamiento
            result = processor(pdf_path, config.to_dict())
            
            # ETAPA 4: Enriquecer resultado con información del controlador
            result["controller_info"] = {
                "processing_config": config.to_dict(),
                "adapter_type": "PyMuPDF4LLMAdapter",
                "output_directory": str(self.output_dir)
            }
            
            return result

        except Exception as e:
            self.logger.error(f"Error en controlador LLM: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "recommendations": [
                    "Verificar que el archivo PDF sea válido",
                    "Comprobar que pymupdf4llm esté instalado",
                    "Intentar con una configuración más simple"
                ]
            }

    def process_multiple_documents(
        self, 
        pdf_paths: List[Path], 
        config: Optional[LLMProcessingConfig] = None
    ) -> Dict[str, Any]:
        """
        Procesa múltiples documentos en lote.
        
        Args:
            pdf_paths (List[Path]): Lista de rutas a documentos PDF
            config (Optional[LLMProcessingConfig]): Configuración común para todos
            
        Returns:
            Dict[str, Any]: Resultado agregado del procesamiento en lote
        """
        if config is None:
            config = LLMProcessingConfig.for_rag_system()
            
        self.logger.info(f"Procesando {len(pdf_paths)} documentos en lote")
        
        results = []
        successful = 0
        failed = 0
        
        for pdf_path in pdf_paths:
            try:
                result = self.process_document(pdf_path, config)
                results.append(result)
                
                if result.get("success", False):
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                self.logger.error(f"Error procesando {pdf_path.name}: {str(e)}")
                results.append({
                    "success": False,
                    "document_path": str(pdf_path),
                    "error": str(e)
                })
                failed += 1
        
        # Calcular estadísticas agregadas
        total_chunks = sum(r.get("chunk_info", {}).get("total_chunks", 0) for r in results)
        total_files = sum(len(r.get("files_generated", [])) for r in results)
        
        return {
            "batch_success": True,
            "total_documents": len(pdf_paths),
            "successful": successful,
            "failed": failed,
            "results": results,
            "aggregate_stats": {
                "total_chunks_generated": total_chunks,
                "total_files_generated": total_files,
                "success_rate": successful / len(pdf_paths) if pdf_paths else 0
            },
            "config_used": config.to_dict()
        }

    def get_recommended_config(self, use_case: str, **kwargs) -> LLMProcessingConfig:
        """
        Obtiene configuración recomendada para un caso de uso específico.
        
        Args:
            use_case (str): Caso de uso ('rag', 'embeddings', 'analysis')
            **kwargs: Parámetros específicos para personalizar la configuración
            
        Returns:
            LLMProcessingConfig: Configuración optimizada para el caso de uso
        """
        if use_case.lower() == "rag":
            return LLMProcessingConfig.for_rag_system(**kwargs)
        elif use_case.lower() == "embeddings":
            return LLMProcessingConfig.for_embeddings(**kwargs)
        elif use_case.lower() == "analysis":
            return LLMProcessingConfig.for_analysis(**kwargs)
        else:
            self.logger.warning(f"Caso de uso desconocido: {use_case}. Usando configuración RAG.")
            return LLMProcessingConfig.for_rag_system(**kwargs)

    def _create_llm_adapter(self, config: LLMProcessingConfig) -> LLMDocumentPort:
        """
        Crea el adaptador LLM con la configuración especificada.
        
        Args:
            config (LLMProcessingConfig): Configuración para el adaptador
            
        Returns:
            LLMDocumentPort: Adaptador configurado
        """
        return PyMuPDF4LLMAdapter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            preserve_structure=config.preserve_structure,
            extract_images=config.extract_images,
            include_metadata=config.include_metadata,
            semantic_chunking=config.semantic_chunking,
            table_strategy=config.table_strategy
        )

    def _create_storage_adapter(self) -> StoragePort:
        """
        Crea el adaptador de almacenamiento.
        
        Returns:
            StoragePort: Adaptador de almacenamiento configurado
        """
        return FileStorage(self.output_dir)

    def validate_pdf_for_llm_processing(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Valida si un PDF es adecuado para procesamiento LLM.
        
        Args:
            pdf_path (Path): Ruta al archivo PDF
            
        Returns:
            Dict[str, Any]: Resultado de la validación con recomendaciones
        """
        validation_result = {
            "is_valid": True,
            "warnings": [],
            "recommendations": [],
            "estimated_processing_time": "desconocido"
        }
        
        try:
            if not pdf_path.exists():
                validation_result["is_valid"] = False
                validation_result["warnings"].append("El archivo no existe")
                return validation_result
                
            # Verificar tamaño del archivo
            file_size = pdf_path.stat().st_size
            if file_size > 50 * 1024 * 1024:  # 50MB
                validation_result["warnings"].append("Archivo muy grande (>50MB)")
                validation_result["recommendations"].append(
                    "Considera dividir el documento o usar chunks más pequeños"
                )
            
            # Estimación básica de tiempo de procesamiento
            estimated_minutes = max(1, file_size / (5 * 1024 * 1024))  # ~5MB por minuto
            validation_result["estimated_processing_time"] = f"{estimated_minutes:.1f} minutos"
            
            # Verificar si es PDF nativo (no escaneado)
            # Esta es una verificación básica - pymupdf4llm puede manejar ambos tipos
            validation_result["recommendations"].append(
                "pymupdf4llm funciona mejor con PDFs nativos. "
                "Si es un documento escaneado, considera usar OCR adicional."
            )
            
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["warnings"].append(f"Error validando archivo: {str(e)}")
            
        return validation_result

    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del directorio de salida.
        
        Returns:
            Dict[str, Any]: Estadísticas de archivos procesados
        """
        try:
            if not self.output_dir.exists():
                return {"total_files": 0, "total_size": 0, "recent_files": []}
                
            files = list(self.output_dir.rglob("*"))
            file_files = [f for f in files if f.is_file()]
            
            total_size = sum(f.stat().st_size for f in file_files)
            recent_files = sorted(file_files, key=lambda x: x.stat().st_mtime, reverse=True)[:10]
            
            return {
                "total_files": len(file_files),
                "total_size": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "recent_files": [str(f.name) for f in recent_files],
                "output_directory": str(self.output_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Error obteniendo estadísticas: {str(e)}")
            return {"error": str(e)}
