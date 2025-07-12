"""
Tests para las utilidades de archivos.

Demuestra cómo la separación de I/O permite testing automatizado
sin necesidad de simulación de entrada/salida.
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from utils.file_utils import discover_pdf_files, validate_pdf_exists, get_file_info


class TestDiscoverPdfFiles:
    """Tests para la función de descubrimiento de archivos PDF."""
    
    def test_discover_pdf_files_empty_directory(self):
        """Test con directorio vacío."""
        with TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            result = discover_pdf_files(directory)
            assert result == []
    
    def test_discover_pdf_files_with_pdfs(self):
        """Test con archivos PDF presentes."""
        with TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            
            # Crear archivos PDF de prueba
            (directory / "documento1.pdf").touch()
            (directory / "documento2.pdf").touch()
            (directory / "archivo.txt").touch()  # No es PDF
            
            result = discover_pdf_files(directory)
            
            assert len(result) == 2
            assert "documento1.pdf" in result
            assert "documento2.pdf" in result
            assert "archivo.txt" not in result
            assert result == sorted(result)  # Debe estar ordenado
    
    def test_discover_pdf_files_nonexistent_directory(self):
        """Test con directorio que no existe."""
        nonexistent_dir = Path("/directorio/que/no/existe")
        
        with pytest.raises(FileNotFoundError):
            discover_pdf_files(nonexistent_dir)
    
    def test_discover_pdf_files_not_a_directory(self):
        """Test cuando el path no es un directorio."""
        with TemporaryDirectory() as tmpdir:
            # Crear un archivo, no un directorio
            file_path = Path(tmpdir) / "archivo.txt"
            file_path.touch()
            
            with pytest.raises(NotADirectoryError):
                discover_pdf_files(file_path)


class TestValidatePdfExists:
    """Tests para la validación de existencia de archivos PDF."""
    
    def test_validate_existing_pdf(self):
        """Test con PDF que existe."""
        with TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            pdf_file = directory / "test.pdf"
            pdf_file.touch()
            
            result = validate_pdf_exists(directory, "test.pdf")
            assert result is True
    
    def test_validate_nonexistent_pdf(self):
        """Test con PDF que no existe."""
        with TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            
            result = validate_pdf_exists(directory, "nonexistent.pdf")
            assert result is False
    
    def test_validate_non_pdf_file(self):
        """Test con archivo que no es PDF."""
        with TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            text_file = directory / "document.txt"
            text_file.touch()
            
            result = validate_pdf_exists(directory, "document.txt")
            assert result is False


class TestGetFileInfo:
    """Tests para obtener información de archivos."""
    
    def test_get_file_info_existing_file(self):
        """Test con archivo existente."""
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.pdf"
            file_path.write_text("contenido de prueba")
            
            result = get_file_info(file_path)
            
            assert result["name"] == "test.pdf"
            assert result["size_bytes"] > 0
            assert result["size_mb"] >= 0
            assert "modified" in result
            assert result["is_readable"] is True
    
    def test_get_file_info_nonexistent_file(self):
        """Test con archivo que no existe."""
        nonexistent_file = Path("/archivo/que/no/existe.pdf")
        
        with pytest.raises(FileNotFoundError):
            get_file_info(nonexistent_file)
