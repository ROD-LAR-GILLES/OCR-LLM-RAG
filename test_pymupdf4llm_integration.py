# test_pymupdf4llm_integration.py
"""
Script de prueba para validar la integración de pymupdf4llm.

Este script verifica que todos los componentes funcionen correctamente
y proporciona ejemplos de uso de la nueva funcionalidad LLM.
"""
import sys
from pathlib import Path
import json
import tempfile
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Verifica que todas las dependencias estén disponibles."""
    print("Verificando imports...")
    
    try:
        import pymupdf4llm
        print("[OK] pymupdf4llm disponible")
    except ImportError as e:
        print(f"[ERROR] pymupdf4llm no disponible: {e}")
        return False
    
    try:
        from adapters.llm_pymupdf4llm_adapter import LLMPyMuPDF4LLMAdapter
        print("[OK] LLMPyMuPDF4LLMAdapter disponible")
    except ImportError as e:
        print(f"[ERROR] LLMPyMuPDF4LLMAdapter no disponible: {e}")
        return False
    
    try:
        from application.llm_controllers import LLMDocumentController, LLMProcessingConfig
        print("[OK] LLMDocumentController disponible")
    except ImportError as e:
        print(f"[ERROR] LLMDocumentController no disponible: {e}")
        return False
    
    return True

def test_basic_adapter():
    """Prueba básica del adaptador pymupdf4llm."""
    print("\n[TEST] Probando adaptador básico...")
    
    try:
        from adapters.llm_pymupdf4llm_adapter import LLMPyMuPDF4LLMAdapter
        
        # Crear adaptador con configuración básica
        adapter = LLMPyMuPDF4LLMAdapter(
            chunk_size=500,
            semantic_chunking=True
        )
        
        # Verificar configuración
        config = adapter.get_chunk_strategy()
        print(f"[OK] Configuración del adaptador: {config}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error en adaptador: {e}")
        return False

def test_controller_configs():
    """Prueba las configuraciones predefinidas del controlador."""
    print("\n[TEST] Probando configuraciones predefinidas...")
    
    try:
        from application.llm_controllers import LLMDocumentController, LLMProcessingConfig
        
        # Crear controlador con directorio temporal
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = LLMDocumentController(Path(temp_dir))
            
            # Probar configuraciones predefinidas
            configs = {
                "rag": controller.get_recommended_config("rag"),
                "embeddings": controller.get_recommended_config("embeddings"),
                "analysis": controller.get_recommended_config("analysis")
            }
            
            for name, config in configs.items():
                print(f"[OK] Configuración {name}: chunk_size={config.chunk_size}, overlap={config.chunk_overlap}")
            
            return True
            
    except Exception as e:
        print(f"[ERROR] Error en configuraciones: {e}")
        return False

def test_with_sample_pdf():
    """Prueba con un PDF de muestra si está disponible."""
    print("\n[TEST] Buscando PDF de muestra para prueba...")
    
    # Buscar PDF de muestra en directorios comunes
    sample_paths = [
        Path("pdfs"),
        Path("./pdfs"),
        Path("../pdfs"),
        Path("samples"),
        Path("test_files")
    ]
    
    sample_pdf = None
    for path in sample_paths:
        if path.exists():
            pdf_files = list(path.glob("*.pdf"))
            if pdf_files:
                sample_pdf = pdf_files[0]
                break
    
    if not sample_pdf:
        print("[WARNING] No se encontró PDF de muestra. Saltando prueba con archivo real.")
        return True
    
    print(f"📄 Usando PDF de muestra: {sample_pdf.name}")
    
    try:
        from application.llm_controllers import LLMDocumentController, LLMProcessingConfig
        
        # Crear controlador con directorio temporal
        with tempfile.TemporaryDirectory() as temp_dir:
            controller = LLMDocumentController(Path(temp_dir))
            
            # Configuración básica para prueba
            config = LLMProcessingConfig.for_rag_system(chunk_size=500)
            
            # Procesar documento
            print("[PROCESSING] Procesando documento...")
            result = controller.process_document(sample_pdf, config)
            
            if result.get("success", False):
                print("[OK] Procesamiento exitoso!")
                print(f"   • Chunks generados: {result.get('chunk_info', {}).get('total_chunks', 0)}")
                print(f"   • Archivos creados: {len(result.get('files_generated', []))}")
                
                # Mostrar ejemplo de chunk
                chunks = result.get('content', {}).get('chunks', [])
                if chunks:
                    first_chunk = chunks[0]
                    preview = first_chunk.get('preview', '')
                    print(f"   • Primer chunk (preview): {preview}")
                
                return True
            else:
                print(f"❌ Procesamiento falló: {result.get('error', 'Error desconocido')}")
                return False
                
    except Exception as e:
        print(f"[ERROR] Error procesando PDF: {e}")
        return False

def test_chunk_strategy():
    """Prueba las estrategias de chunking."""
    print("\n[TEST] Probando estrategias de chunking...")
    
    try:
        from adapters.llm_pymupdf4llm_adapter import LLMPyMuPDF4LLMAdapter
        
        # Probar diferentes configuraciones
        configs = [
            {"chunk_size": 500, "semantic_chunking": True, "name": "Semántico pequeño"},
            {"chunk_size": 1000, "semantic_chunking": True, "name": "Semántico medio"},
            {"chunk_size": 1000, "semantic_chunking": False, "name": "Fijo medio"},
        ]
        
        for config in configs:
            adapter = LLMPyMuPDF4LLMAdapter(
                chunk_size=config["chunk_size"],
                semantic_chunking=config["semantic_chunking"]
            )
            
            strategy = adapter.get_chunk_strategy()
            print(f"[OK] {config['name']}: {strategy}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error en estrategias de chunking: {e}")
        return False

def test_metadata_generation():
    """Prueba la generación de metadatos."""
    print("\n[TEST] Probando generación de metadatos...")
    
    try:
        from adapters.llm_pymupdf4llm_adapter import LLMPyMuPDF4LLMAdapter
        
        adapter = LLMPyMuPDF4LLMAdapter()
        
        # Simular contenido para análisis
        mock_content = """
        # Título Principal
        
        ## Introducción
        
        Este es un documento de ejemplo con múltiples secciones.
        
        ### Subsección
        
        Contenido adicional con información relevante.
        
        | Tabla | Ejemplo |
        |-------|---------|
        | A     | 1       |
        | B     | 2       |
        """
        
        # Probar análisis de estructura
        structure = adapter._analyze_document_structure(mock_content)
        print(f"[OK] Estructura detectada: {structure.get('heading_count', 0)} encabezados")
        
        # Probar estadísticas
        stats = adapter._calculate_content_stats(mock_content)
        print(f"[OK] Estadísticas: {stats.get('words', 0)} palabras, {stats.get('characters', 0)} caracteres")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error en metadatos: {e}")
        return False

def main():
    """Ejecuta todas las pruebas."""
    print("Iniciando pruebas de integración pymupdf4llm\n")
    
    tests = [
        ("Imports", test_imports),
        ("Adaptador básico", test_basic_adapter),
        ("Configuraciones del controlador", test_controller_configs),
        ("Estrategias de chunking", test_chunk_strategy),
        ("Generación de metadatos", test_metadata_generation),
        ("Procesamiento con PDF real", test_with_sample_pdf),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Prueba: {name}")
        print('='*60)
        
        try:
            if test_func():
                passed += 1
                print(f"[OK] {name}: PASÓ")
            else:
                print(f"❌ {name}: FALLÓ")
        except Exception as e:
            print(f"❌ {name}: ERROR - {e}")
            logger.exception(f"Error en prueba {name}")
    
    print(f"\n{'='*60}")
    print(f"📊 Resumen de pruebas: {passed}/{total} pasaron")
    print('='*60)
    
    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron! La integración pymupdf4llm está lista.")
        return 0
    else:
        print("⚠️ Algunas pruebas fallaron. Revisar la configuración.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
