"""
Pruebas de integración para la funcionalidad de pymupdf4llm.
"""
import os
import pytest
from pathlib import Path

# Añadir la ruta del proyecto al PYTHONPATH para importar desde src
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.adapters.llm_pymupdf4llm_adapter import PyMuPDF4LLMAdapter
from src.application.use_cases import ExtractDocumentUseCase


# Ruta a los archivos de prueba
TEST_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
# Asegúrate de tener un PDF de muestra en esta carpeta
SAMPLE_PDF = os.path.join(TEST_ASSETS_DIR, "sample.pdf")


@pytest.fixture
def pymupdf_adapter():
    return PyMuPDF4LLMAdapter()


@pytest.fixture
def extract_document_use_case(pymupdf_adapter):
    return ExtractDocumentUseCase(pymupdf_adapter)


@pytest.mark.skipif(not os.path.exists(SAMPLE_PDF), reason="Archivo de prueba no encontrado")
def test_extract_markdown(pymupdf_adapter):
    """Prueba la extracción de contenido en formato Markdown."""
    markdown = pymupdf_adapter.extract_markdown(SAMPLE_PDF)
    
    # Verifica que el resultado sea una cadena no vacía
    assert isinstance(markdown, str)
    assert len(markdown) > 0
    
    # Verifica características típicas del Markdown
    assert "# " in markdown or "## " in markdown or "### " in markdown or "*" in markdown


@pytest.mark.skipif(not os.path.exists(SAMPLE_PDF), reason="Archivo de prueba no encontrado")
def test_extract_llama_documents(pymupdf_adapter):
    """Prueba la extracción de documentos para LlamaIndex."""
    documents = pymupdf_adapter.extract_llama_documents(SAMPLE_PDF)
    
    # Verifica que el resultado sea una lista no vacía
    assert isinstance(documents, list)
    assert len(documents) > 0


@pytest.mark.skipif(not os.path.exists(SAMPLE_PDF), reason="Archivo de prueba no encontrado")
def test_extract_document_use_case(extract_document_use_case):
    """Prueba el caso de uso de extracción de documentos."""
    # Prueba con formato Markdown
    result_md = extract_document_use_case.execute(SAMPLE_PDF, output_format="markdown")
    assert result_md["format"] == "markdown"
    assert "content" in result_md
    assert isinstance(result_md["content"], str)
    
    # Prueba con formato de documentos LlamaIndex
    result_llama = extract_document_use_case.execute(SAMPLE_PDF, output_format="llama_documents")
    assert result_llama["format"] == "llama_documents"
    assert "documents" in result_llama
    assert isinstance(result_llama["documents"], list)
