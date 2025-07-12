"""
Controladores de aplicación para OCR-CLI.

Este módulo contiene la lógica de coordinación de alto nivel que orquesta
los casos de uso sin depender de detalles de interfaz específicos.
"""

import time
from pathlib import Path
from typing import Tuple, List, Dict, Any

from adapters.ocr_tesseract import TesseractAdapter
from adapters.ocr_tesseract_opencv import TesseractOpenCVAdapter
from adapters.table_pdfplumber import PdfPlumberAdapter
from adapters.storage_filesystem import FileStorage
from src.application.use_cases import ProcessDocument
from utils.menu_logic import OCRConfig


class DocumentController:
    """
    Controlador para operaciones de procesamiento de documentos.
    
    Encapsula la lógica de coordinación entre adaptadores y casos de uso,
    permitiendo que sea reutilizable desde diferentes interfaces.
    """
    
    def __init__(self, pdf_dir: Path, output_dir: Path):
        """
        Inicializa el controlador con directorios de trabajo.
        
        Args:
            pdf_dir (Path): Directorio donde se encuentran los PDFs
            output_dir (Path): Directorio donde guardar resultados
        """
        self.pdf_dir = pdf_dir
        self.output_dir = output_dir
    
    def process_document(
        self, 
        filename: str, 
        ocr_config: OCRConfig
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Procesa un documento PDF usando la configuración especificada.
        
        Args:
            filename (str): Nombre del archivo PDF a procesar
            ocr_config (OCRConfig): Configuración del motor OCR
            
        Returns:
            Tuple[bool, Dict]: (éxito, información_del_procesamiento)
            
        Example:
            >>> controller = DocumentController(Path("/pdfs"), Path("/output"))
            >>> config = OCRConfig("basic")
            >>> success, info = controller.process_document("doc.pdf", config)
            >>> print(info["processing_time"])
            2.34
        """
        pdf_path = self.pdf_dir / filename
        
        # Validar que el archivo existe
        if not pdf_path.exists():
            return False, {
                "error": f"Archivo {filename} no encontrado",
                "filename": filename,
                "processing_time": 0
            }
        
        try:
            # Configurar adaptadores basado en la configuración
            ocr_adapter = self._create_ocr_adapter(ocr_config)
            table_adapter = PdfPlumberAdapter()
            storage_adapter = FileStorage(self.output_dir)
            
            # Crear caso de uso
            processor = ProcessDocument(
                ocr=ocr_adapter,
                table_extractor=table_adapter,
                storage=storage_adapter
            )
            
            # Medir tiempo de procesamiento
            start_time = time.time()
            
            # Ejecutar procesamiento
            try:
                result = processor(pdf_path)
                
                # Verificar si es una tupla con 2 elementos
                if isinstance(result, tuple) and len(result) == 2:
                    texto_principal, archivos_generados = result
                else:
                    raise ValueError(f"ProcessDocument retornó {type(result)} en lugar de tupla de 2 elementos: {result}")
                    
            except Exception as debug_error:
                raise Exception(f"Error en processor(): {debug_error}")
            
            processing_time = time.time() - start_time
            
            return True, {
                "filename": filename,
                "processing_time": processing_time,
                "main_text_file": texto_principal,
                "generated_files": archivos_generados,
                "files_count": len(archivos_generados),
                "ocr_config": ocr_config,
                "error": None
            }
            
        except Exception as e:
            processing_time = time.time() - start_time if 'start_time' in locals() else 0
            return False, {
                "filename": filename,
                "processing_time": processing_time,
                "error": str(e),
                "error_type": type(e).__name__,
                "ocr_config": ocr_config
            }
    
    def _create_ocr_adapter(self, config: OCRConfig):
        """
        Crea el adaptador OCR apropiado basado en la configuración.
        
        Args:
            config (OCRConfig): Configuración del motor OCR
            
        Returns:
            OCRPort: Adaptador OCR configurado
        """
        if config.engine_type == "basic":
            return TesseractAdapter()
        elif config.engine_type == "opencv":
            return TesseractOpenCVAdapter(
                enable_deskewing=config.enable_deskewing,
                enable_denoising=config.enable_denoising,
                enable_contrast_enhancement=config.enable_contrast_enhancement
            )
        else:
            raise ValueError(f"Tipo de motor OCR no soportado: {config.engine_type}")
    
    def get_processing_capabilities(self) -> Dict[str, Any]:
        """
        Retorna información sobre las capacidades de procesamiento disponibles.
        
        Returns:
            Dict: Información sobre motores OCR disponibles y configuraciones
        """
        return {
            "ocr_engines": {
                "basic": {
                    "name": "Tesseract básico",
                    "description": "OCR rápido para documentos de alta calidad",
                    "performance": "Alto",
                    "quality": "Bueno"
                },
                "opencv": {
                    "name": "Tesseract + OpenCV",
                    "description": "OCR avanzado con preprocesamiento",
                    "performance": "Medio",
                    "quality": "Excelente",
                    "features": [
                        "Corrección de inclinación",
                        "Eliminación de ruido",
                        "Mejora de contraste",
                        "Binarización adaptativa"
                    ]
                }
            },
            "supported_formats": ["PDF"],
            "output_formats": ["TXT", "JSON", "ASCII"],
            "directories": {
                "input": str(self.pdf_dir),
                "output": str(self.output_dir)
            }
        }
