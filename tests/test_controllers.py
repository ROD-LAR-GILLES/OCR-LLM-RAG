"""
Tests para el controlador de documentos.

Demuestra cómo testear la lógica de procesamiento sin depender de I/O real.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from tempfile import TemporaryDirectory

from application.controllers import DocumentController
from utils.menu_logic import OCRConfig


class TestDocumentController(unittest.TestCase):
    """Tests para la clase DocumentController."""
    
    def setUp(self):
        """Configuración inicial para cada test."""
        self.temp_dir = TemporaryDirectory()
        self.pdf_dir = Path(self.temp_dir.name) / "pdfs"
        self.output_dir = Path(self.temp_dir.name) / "output"
        
        # Crear directorios
        self.pdf_dir.mkdir()
        self.output_dir.mkdir()
        
        # Crear archivo PDF de prueba
        self.test_pdf = self.pdf_dir / "test.pdf"
        self.test_pdf.write_text("fake pdf content")
        
        self.controller = DocumentController(self.pdf_dir, self.output_dir)
    
    def tearDown(self):
        """Limpieza después de cada test."""
        self.temp_dir.cleanup()
    
    @patch('application.controllers.ProcessDocument')
    @patch('application.controllers.TesseractAdapter')
    @patch('application.controllers.PdfPlumberAdapter')
    @patch('application.controllers.FileStorage')
    def test_process_document_success_basic_ocr(
        self, 
        mock_storage, 
        mock_table_adapter, 
        mock_tesseract_adapter, 
        mock_process_document
    ):
        """Test procesamiento exitoso con OCR básico."""
        # Configurar mocks
        mock_processor_instance = Mock()
        mock_process_document.return_value = mock_processor_instance
        mock_processor_instance.return_value = ("texto_principal.txt", ["archivo1.txt", "archivo2.json"])
        
        # Configuración básica
        config = OCRConfig("basic")
        
        # Ejecutar
        success, info = self.controller.process_document("test.pdf", config)
        
        # Verificar resultado
        assert success is True
        assert info["filename"] == "test.pdf"
        assert info["main_text_file"] == "texto_principal.txt"
        assert info["generated_files"] == ["archivo1.txt", "archivo2.json"]
        assert info["files_count"] == 2
        assert info["error"] is None
        assert info["processing_time"] > 0
        
        # Verificar que se usó TesseractAdapter
        mock_tesseract_adapter.assert_called_once()
        mock_process_document.assert_called_once()
    
    @patch('application.controllers.ProcessDocument')
    @patch('application.controllers.TesseractOpenCVAdapter')
    @patch('application.controllers.PdfPlumberAdapter')
    @patch('application.controllers.FileStorage')
    def test_process_document_success_opencv_ocr(
        self, 
        mock_storage, 
        mock_table_adapter, 
        mock_opencv_adapter, 
        mock_process_document
    ):
        """Test procesamiento exitoso con OCR OpenCV."""
        # Configurar mocks
        mock_processor_instance = Mock()
        mock_process_document.return_value = mock_processor_instance
        mock_processor_instance.return_value = ("texto.txt", ["archivo.txt"])
        
        # Configuración OpenCV
        config = OCRConfig(
            "opencv", 
            enable_deskewing=True, 
            enable_denoising=False, 
            enable_contrast_enhancement=True
        )
        
        # Ejecutar
        success, info = self.controller.process_document("test.pdf", config)
        
        # Verificar resultado
        assert success is True
        assert info["ocr_config"] == config
        
        # Verificar que se usó TesseractOpenCVAdapter con configuración correcta
        mock_opencv_adapter.assert_called_once_with(
            enable_deskewing=True,
            enable_denoising=False,
            enable_contrast_enhancement=True
        )
    
    def test_process_document_file_not_found(self):
        """Test con archivo que no existe."""
        config = OCRConfig("basic")
        
        success, info = self.controller.process_document("nonexistent.pdf", config)
        
        assert success is False
        assert "no encontrado" in info["error"]
        assert info["filename"] == "nonexistent.pdf"
        assert info["processing_time"] == 0
    
    @patch('application.controllers.ProcessDocument')
    @patch('application.controllers.TesseractAdapter')
    @patch('application.controllers.PdfPlumberAdapter')
    @patch('application.controllers.FileStorage')
    def test_process_document_processing_error(
        self, 
        mock_storage, 
        mock_table_adapter, 
        mock_tesseract_adapter, 
        mock_process_document
    ):
        """Test cuando el procesamiento falla."""
        # Configurar mock para que lance excepción
        mock_processor_instance = Mock()
        mock_process_document.return_value = mock_processor_instance
        mock_processor_instance.side_effect = Exception("Error de procesamiento")
        
        config = OCRConfig("basic")
        
        success, info = self.controller.process_document("test.pdf", config)
        
        assert success is False
        assert info["error"] == "Error de procesamiento"
        assert info["error_type"] == "Exception"
        assert info["processing_time"] > 0
    
    def test_create_ocr_adapter_basic(self):
        """Test creación de adaptador básico."""
        config = OCRConfig("basic")
        
        with patch('application.controllers.TesseractAdapter') as mock_adapter:
            adapter = self.controller._create_ocr_adapter(config)
            mock_adapter.assert_called_once()
    
    def test_create_ocr_adapter_opencv(self):
        """Test creación de adaptador OpenCV."""
        config = OCRConfig(
            "opencv", 
            enable_deskewing=False, 
            enable_denoising=True, 
            enable_contrast_enhancement=False
        )
        
        with patch('application.controllers.TesseractOpenCVAdapter') as mock_adapter:
            adapter = self.controller._create_ocr_adapter(config)
            mock_adapter.assert_called_once_with(
                enable_deskewing=False,
                enable_denoising=True,
                enable_contrast_enhancement=False
            )
    
    def test_create_ocr_adapter_invalid_type(self):
        """Test con tipo de motor inválido."""
        config = OCRConfig("invalid_type")
        
        with self.assertRaises(ValueError):
            self.controller._create_ocr_adapter(config)
    
    def test_get_processing_capabilities(self):
        """Test obtención de capacidades de procesamiento."""
        capabilities = self.controller.get_processing_capabilities()
        
        assert "ocr_engines" in capabilities
        assert "basic" in capabilities["ocr_engines"]
        assert "opencv" in capabilities["ocr_engines"]
        assert "supported_formats" in capabilities
        assert "PDF" in capabilities["supported_formats"]
        assert "output_formats" in capabilities
        assert capabilities["directories"]["input"] == str(self.pdf_dir)
        assert capabilities["directories"]["output"] == str(self.output_dir)


if __name__ == "__main__":
    unittest.main()
