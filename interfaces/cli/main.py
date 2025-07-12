# interfaces/cli/main.py
"""
Punto de entrada principal para la aplicación CLI de OCR-CLI.

Este módulo actúa como bootstrap de la aplicación, proporcionando
un punto de entrada limpio y separando las responsabilidades entre
la inicialización de la aplicación y la lógica de la interfaz.

Principios aplicados:
- Single Responsibility: Solo se encarga de inicializar la aplicación
- Separation of Concerns: Lógica de interfaz separada en menu.py
- Clean Entry Point: Punto de entrada claro y simple

Arquitectura de entrada:
main.py (bootstrap) -> menu.py (interfaz) -> use_cases.py (lógica) -> adaptadores (implementación)
"""
from interfaces.cli.menu import main

if __name__ == "__main__":
    """
    Entry point de la aplicación CLI cuando se ejecuta como script principal.
    
    Este patrón de entry point es una práctica estándar en Python que:
    - Permite importar el módulo sin ejecutar código automáticamente
    - Proporciona un punto de entrada claro cuando se ejecuta directamente
    - Facilita testing al permitir importar sin ejecutar main()
    
    Formas de ejecución:
    1. Directa: python interfaces/cli/main.py
    2. Como módulo: python -m interfaces.cli.main
    3. Docker: CMD especificado en Dockerfile
    
    La separación main.py -> menu.py permite:
    - Testing: Importar menu sin ejecutar la aplicación
    - Reutilización: Usar menu.main() desde otros contextos
    - Flexibilidad: Agregar lógica de inicialización sin afectar la interfaz
    
    Future enhancements:
    - Argument parsing con argparse o click
    - Configuración via variables de entorno
    - Logging setup y configuración
    - Profile selection (development, production, testing)
    """
    main()