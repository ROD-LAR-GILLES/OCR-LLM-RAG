"""
Interfaz de línea de comandos para la extracción de documentos PDF.
"""
import argparse
import os
import sys
from loguru import logger

from src.adapters.llm_pymupdf4llm_adapter import PyMuPDF4LLMAdapter
from src.application.use_cases import ExtractDocumentUseCase  # Importación corregida


def parse_args():
    parser = argparse.ArgumentParser(description="Extrae contenido de PDFs usando pymupdf4llm")
    
    parser.add_argument(
        "pdf_path",
        help="Ruta al archivo PDF a procesar"
    )
    
    parser.add_argument(
        "--format", "-f",
        choices=["markdown", "llama_documents"],
        default="markdown",
        help="Formato de salida (predeterminado: markdown)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Ruta al archivo de salida (opcional, por defecto muestra en consola)"
    )
    
    parser.add_argument(
        "--pages", "-p",
        help="Páginas a extraer, separadas por comas (ej: 0,1,3)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Verifica que el archivo exista
    if not os.path.exists(args.pdf_path):
        logger.error(f"El archivo no existe: {args.pdf_path}")
        sys.exit(1)
    
    # Crea instancias de las dependencias
    pymupdf_adapter = PyMuPDF4LLMAdapter()
    use_case = ExtractDocumentUseCase(pymupdf_adapter)
    
    # Prepara los parámetros
    pages = None
    if args.pages:
        try:
            pages = [int(p.strip()) for p in args.pages.split(",")]
        except ValueError:
            logger.error("Formato de páginas inválido. Use números separados por comas.")
            sys.exit(1)
    
    # Ejecuta el caso de uso
    try:
        result = use_case.execute(
            pdf_path=args.pdf_path,
            output_format=args.format,
            pages=pages
        )
        
        # Procesa el resultado
        if args.format == "markdown":
            output_content = result["content"]
        else:  # llama_documents
            # Simplificación - en un caso real habría que serializar mejor
            import json
            output_content = json.dumps(result["documents"], indent=2)
        
        # Guarda o muestra el resultado
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output_content)
            logger.success(f"Resultado guardado en: {args.output}")
        else:
            print(output_content)
        
    except Exception as e:
        logger.error(f"Error al procesar el documento: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
