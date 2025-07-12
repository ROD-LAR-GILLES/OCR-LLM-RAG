#!/usr/bin/env python3
"""
Script de entrada simple para iniciar la interfaz de LLM.
"""
import sys
import os

# Configurar el entorno
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)
sys.path.insert(0, current_dir)

# Importar los módulos necesarios de forma explícita
from src.interfaces.cli.llm_menu import main_llm_menu

# Iniciar la interfaz
if __name__ == "__main__":
    try:
        main_llm_menu()
    except KeyboardInterrupt:
        print("\nAplicación terminada por el usuario")
    except Exception as e:
        print(f"\n[ERROR] Error en la aplicación: {e}")
        sys.exit(1)
