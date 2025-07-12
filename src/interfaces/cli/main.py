import sys
import os
from pathlib import Path

sys.path.insert(0, '/app/src')
sys.path.insert(0, '/app')

import questionary
from src.adapters.llm_pymupdf4llm_adapter import PyMuPDF4LLMAdapter
from src.utils.file_utils import discover_pdf_files
from src.interfaces.cli.llm_menu import process_ocr_classic_document

PDF_DIR = Path("/app/pdfs")
OUT_DIR = Path("/app/resultado")

def convert_pdf_to_markdown() -> None:
    pdf_files = [PDF_DIR / f for f in discover_pdf_files(PDF_DIR)]
    if not pdf_files:
        print("[ERROR] No se encontraron archivos PDF en el directorio.")
        print(f"   Coloca tus archivos PDF en: {PDF_DIR}")
        return
    choices = [questionary.Choice(f"{i+1}. {p.name}", p) for i, p in enumerate(pdf_files)]
    choices.append(questionary.Choice(f"{len(pdf_files)+1}. Salir", None))
    selected = questionary.select(
        "Selecciona el archivo PDF a convertir a Markdown:",
        choices=choices,
        instruction="Usa ↑↓ para navegar, Enter para seleccionar"
    ).ask()
    if selected is None:
        return
    adapter = PyMuPDF4LLMAdapter()
    try:
        result = adapter.extract_markdown(str(selected))
        output_file = OUT_DIR / (selected.stem + ".md")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"[OK] Conversión completada. Markdown guardado en: {output_file}")
    except Exception as e:
        print(f"[ERROR] Error al convertir el PDF: {e}")

def main_menu() -> None:
    print("\n" + "="*60)
    print("OCR-LLM-RAG - Menú Principal")
    print("="*60)
    while True:
        try:
            choice = questionary.select(
                "¿Qué deseas hacer?",
                choices=[
                    questionary.Choice("Convertir PDF a Markdown (PyMuPDF4LLM)", "markdown"),
                    questionary.Choice("Procesar PDF con OCR clásico (Tesseract/OpenCV)", "ocr_classic"),
                    # Puedes agregar aquí más opciones de configuración avanzada si aplica
                    questionary.Choice("Salir", "exit")
                ],
                instruction="Usa ↑↓ para navegar, Enter para seleccionar"
            ).ask()
            if choice == "markdown":
                convert_pdf_to_markdown()
            elif choice == "ocr_classic":
                process_ocr_classic_document()
            elif choice == "exit":
                print("\nHasta luego!")
                break
            else:
                print("\n[WARNING] Opción no válida")
        except KeyboardInterrupt:
            print("\n\nHasta luego!")
            break
        except Exception as e:
            print(f"\n[ERROR] Error en el menú: {str(e)}")

if __name__ == "__main__":
    main_menu()