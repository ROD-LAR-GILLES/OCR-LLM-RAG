"""
Tests para la lógica de menús.

Demuestra cómo testear la lógica de menús sin simulación de I/O.
"""

import pytest
from utils.menu_logic import (
    create_pdf_menu_options,
    validate_menu_selection,
    get_selected_pdf,
    is_exit_selection,
    create_ocr_config_from_user_choices,
    validate_ocr_engine_choice,
    OCRConfig
)


class TestCreatePdfMenuOptions:
    """Tests para la creación de opciones de menú."""
    
    def test_create_menu_options_empty_list(self):
        """Test con lista vacía de PDFs."""
        options = create_pdf_menu_options([])
        
        assert len(options) == 1  # Solo opción de salida
        assert options[0].text == "1. Salir"
        assert options[0].value == "exit"
    
    def test_create_menu_options_with_pdfs(self):
        """Test con archivos PDF."""
        pdf_files = ["doc1.pdf", "doc2.pdf"]
        options = create_pdf_menu_options(pdf_files)
        
        assert len(options) == 3  # 2 PDFs + Salir
        assert options[0].text == "1. doc1.pdf"
        assert options[0].value == "doc1.pdf"
        assert options[1].text == "2. doc2.pdf"
        assert options[1].value == "doc2.pdf"
        assert options[2].text == "3. Salir"
        assert options[2].value == "exit"


class TestValidateMenuSelection:
    """Tests para la validación de selección de menú."""
    
    def test_validate_menu_selection_valid_range(self):
        """Test con selecciones válidas."""
        assert validate_menu_selection(1, 3) is True
        assert validate_menu_selection(2, 3) is True
        assert validate_menu_selection(3, 3) is True
    
    def test_validate_menu_selection_invalid_range(self):
        """Test con selecciones inválidas."""
        assert validate_menu_selection(0, 3) is False
        assert validate_menu_selection(4, 3) is False
        assert validate_menu_selection(-1, 3) is False


class TestGetSelectedPdf:
    """Tests para obtener el PDF seleccionado."""
    
    def test_get_selected_pdf_valid_selection(self):
        """Test con selección válida."""
        pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
        
        assert get_selected_pdf(pdf_files, 1) == "doc1.pdf"
        assert get_selected_pdf(pdf_files, 2) == "doc2.pdf"
        assert get_selected_pdf(pdf_files, 3) == "doc3.pdf"
    
    def test_get_selected_pdf_invalid_selection(self):
        """Test con selección inválida."""
        pdf_files = ["doc1.pdf", "doc2.pdf"]
        
        with pytest.raises(ValueError):
            get_selected_pdf(pdf_files, 0)
        
        with pytest.raises(ValueError):
            get_selected_pdf(pdf_files, 3)
    
    def test_get_selected_pdf_empty_list(self):
        """Test con lista vacía."""
        with pytest.raises(ValueError):
            get_selected_pdf([], 1)


class TestIsExitSelection:
    """Tests para detectar selección de salida."""
    
    def test_is_exit_selection_true(self):
        """Test cuando es selección de salida."""
        assert is_exit_selection(3, 2) is True  # 3 es salir cuando hay 2 PDFs
        assert is_exit_selection(1, 0) is True  # 1 es salir cuando no hay PDFs
    
    def test_is_exit_selection_false(self):
        """Test cuando no es selección de salida."""
        assert is_exit_selection(1, 2) is False  # 1 es PDF cuando hay 2 PDFs
        assert is_exit_selection(2, 2) is False  # 2 es PDF cuando hay 2 PDFs


class TestCreateOcrConfigFromUserChoices:
    """Tests para crear configuración OCR."""
    
    def test_create_basic_ocr_config(self):
        """Test para configuración básica."""
        config = create_ocr_config_from_user_choices(1)
        
        assert config.engine_type == "basic"
        assert config.enable_deskewing is False
        assert config.enable_denoising is False
        assert config.enable_contrast_enhancement is False
    
    def test_create_opencv_ocr_config_default(self):
        """Test para configuración OpenCV por defecto."""
        config = create_ocr_config_from_user_choices(2)
        
        assert config.engine_type == "opencv"
        assert config.enable_deskewing is True
        assert config.enable_denoising is True
        assert config.enable_contrast_enhancement is True
    
    def test_create_opencv_ocr_config_custom(self):
        """Test para configuración OpenCV personalizada."""
        config = create_ocr_config_from_user_choices(
            2, 
            enable_deskewing=False,
            enable_denoising=True,
            enable_contrast=False
        )
        
        assert config.engine_type == "opencv"
        assert config.enable_deskewing is False
        assert config.enable_denoising is True
        assert config.enable_contrast_enhancement is False
    
    def test_create_ocr_config_invalid_choice(self):
        """Test con opción inválida."""
        with pytest.raises(ValueError):
            create_ocr_config_from_user_choices(0)
        
        with pytest.raises(ValueError):
            create_ocr_config_from_user_choices(4)


class TestValidateOcrEngineChoice:
    """Tests para validar elección de motor OCR."""
    
    def test_validate_ocr_engine_choice_valid(self):
        """Test con opciones válidas."""
        assert validate_ocr_engine_choice(1) is True
        assert validate_ocr_engine_choice(2) is True
        assert validate_ocr_engine_choice(3) is True
    
    def test_validate_ocr_engine_choice_invalid(self):
        """Test con opciones inválidas."""
        assert validate_ocr_engine_choice(0) is False
        assert validate_ocr_engine_choice(4) is False
        assert validate_ocr_engine_choice(-1) is False
