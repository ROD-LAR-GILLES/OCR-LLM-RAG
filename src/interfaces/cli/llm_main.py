# interfaces/cli/llm_main.py
"""
Punto de entrada principal para la aplicación LLM-CLI con pymupdf4llm.

Este módulo actúa como bootstrap de la aplicación optimizada para
modelos de lenguaje, proporcionando un punto de entrada específico
para casos de uso de IA y procesamiento para LLMs.

Diferencias con main.py tradicional:
- Enfoque específico en pymupdf4llm
- Configuraciones predefinidas para casos de uso LLM
- Validaciones específicas para contenido dirigido a IA
- Mejor feedback para usuarios de sistemas RAG
"""
import sys
import os
import logging
from pathlib import Path

# Asegurarnos de que /app está en el PYTHONPATH
sys.path.append('/app')

# Configurar logging básico
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

try:
    from src.interfaces.cli.llm_menu import main_llm_menu
except ImportError as e:
    try:
        from interfaces.cli.llm_menu import main_llm_menu
    except ImportError:
        print(f"[ERROR] Error importando dependencias: {e}")
        print("[INFO] Asegúrate de que todas las dependencias estén instaladas:")
        print("   pip install pymupdf4llm questionary")
        sys.exit(1)


def check_dependencies() -> bool:
    """
    Verifica que todas las dependencias estén disponibles.
    
    Returns:
        bool: True si todas las dependencias están disponibles
    """
    missing_deps = []
    
    try:
        import pymupdf4llm
    except ImportError:
        missing_deps.append("pymupdf4llm")
    
    try:
        import questionary
    except ImportError:
        missing_deps.append("questionary")
    
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    if missing_deps:
        print("Dependencias faltantes:")
        for dep in missing_deps:
            print(f"   - {dep}")
        print("Instala las dependencias con:")
        print(f"   pip install {' '.join(missing_deps)}")
        return False
    
    return True


def check_directories() -> bool:
    """
    Verifica que los directorios necesarios existan.
    
    Returns:
        bool: True si los directorios están configurados correctamente
    """
    pdf_dir = Path("/app/pdfs")
    output_dir = Path("/app/resultado")
    
    issues = []
    
    if not pdf_dir.exists():
        issues.append(f"Directorio de entrada no existe: {pdf_dir}")
        
    if not output_dir.exists():
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[OK] Directorio de salida creado: {output_dir}")
        except Exception as e:
            issues.append(f"No se puede crear directorio de salida: {e}")
    
    if issues:
        print("[WARNING] Problemas con directorios:")
        for issue in issues:
            print(f"   - {issue}")
        print("Si usas Docker, verifica que los volúmenes estén montados:")
        print("   docker-compose up --build")
        return False
    
    return True


def main():
    """
    Función principal que inicializa y ejecuta la aplicación LLM-CLI.
    
    Realiza verificaciones previas y lanza la interfaz principal
    optimizada para casos de uso de modelos de lenguaje.
    """
    print("Iniciando PDF-to-LLM CLI...")
    
    # Verificar dependencias
    if not check_dependencies():
        sys.exit(1)
    
    # Verificar directorios
    if not check_directories():
        sys.exit(1)
    
    try:
        # Lanzar interfaz principal
        main_llm_menu()
        
    except KeyboardInterrupt:
        print("\n\nAplicación terminada por el usuario")
    except Exception as e:
        print(f"\n[ERROR] Error crítico en la aplicación: {e}")
        logging.exception("Error crítico en main()")
        sys.exit(1)


if __name__ == "__main__":
    """
    Entry point cuando se ejecuta como script principal.
    
    Formas de ejecución:
    1. Directa: python interfaces/cli/llm_main.py
    2. Como módulo: python -m interfaces.cli.llm_main
    3. Docker: especificado en docker-compose.yml para LLM
    
    Este punto de entrada está optimizado específicamente para:
    - Procesamiento con pymupdf4llm
    - Casos de uso de modelos de lenguaje
    - Sistemas RAG y embeddings
    - Análisis automatizado por IA
    """
    main()
