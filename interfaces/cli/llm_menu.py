# interfaces/cli/llm_menu.py
"""
Interfaz CLI específica para procesamiento LLM con pymupdf4llm.

Este módulo proporciona una interfaz de línea de comandos simplificada
y optimizada específicamente para usar pymupdf4llm en aplicaciones de
modelos de lenguaje, sistemas RAG y análisis por IA.

Diferencias con menu.py tradicional:
- Enfoque específico en casos de uso LLM
- Configuraciones predefinidas para RAG, embeddings y análisis
- Feedback detallado sobre chunks y optimización
- Recomendaciones específicas para uso con LLMs
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

try:
    import questionary
except ImportError:
    raise ImportError("questionary es requerido. Instálalo con: pip install questionary")

from utils.file_utils import discover_pdf_files, validate_pdf_exists
from application.llm_controllers import LLMDocumentController, LLMProcessingConfig


# Configuración de directorios
PDF_DIR = Path("/app/pdfs")        # Directorio de entrada
OUT_DIR = Path("/app/resultado")   # Directorio de salida


def display_llm_welcome() -> None:
    """Muestra mensaje de bienvenida específico para LLM."""
    print("\n" + "="*60)
    print("PDF-to-LLM - Procesador optimizado para modelos de lenguaje")
    print("Convierte PDFs a formato markdown estructurado usando pymupdf4llm")
    print("="*60)
    print("\nCasos de uso soportados:")
    print("   • RAG Systems (Retrieval-Augmented Generation)")
    print("   • Vector Databases y Embeddings")
    print("   • Análisis automatizado por LLMs")
    print("   • Búsqueda semántica e indexación")
    print("\nTecnología: pymupdf4llm + arquitectura hexagonal")
    print("-"*60)


def select_use_case() -> str:
    """
    Permite al usuario seleccionar el caso de uso objetivo.
    
    Returns:
        str: Caso de uso seleccionado ('rag', 'embeddings', 'analysis')
    """
    print("\n🎯 Selecciona el caso de uso objetivo:")
    
    use_case = questionary.select(
        "¿Para qué vas a usar el contenido procesado?",
        choices=[
            questionary.Choice(
                title="Sistema RAG (Retrieval-Augmented Generation)",
                value="rag"
            ),
            questionary.Choice(
                title="🧮 Vector Database / Embeddings",
                value="embeddings"
            ),
            questionary.Choice(
                title="🔬 Análisis por LLMs",
                value="analysis"
            ),
            questionary.Choice(
                title="⚙️ Configuración personalizada",
                value="custom"
            ),
            questionary.Choice(
                title="🔙 Volver al menú principal",
                value="exit"
            )
        ],
        instruction="Usa ↑↓ para navegar, Enter para seleccionar"
    ).ask()
    
    return use_case or "exit"


def configure_custom_settings() -> LLMProcessingConfig:
    """
    Permite configuración personalizada de parámetros.
    
    Returns:
        LLMProcessingConfig: Configuración personalizada
    """
    print("\n⚙️ Configuración personalizada:")
    
    # Tamaño de chunk
    chunk_size = questionary.text(
        "Tamaño máximo de chunk (caracteres):",
        default="1000",
        validate=lambda x: x.isdigit() and 100 <= int(x) <= 10000
    ).ask()
    
    # Estrategia de chunking
    semantic_chunking = questionary.confirm(
        "¿Usar chunking semántico (recomendado)?",
        default=True
    ).ask()
    
    # Preservar estructura
    preserve_structure = questionary.confirm(
        "¿Preservar estructura de encabezados en chunks?",
        default=True
    ).ask()
    
    # Overlap
    overlap_percent = questionary.select(
        "Solapamiento entre chunks:",
        choices=[
            questionary.Choice("10% (mínimo)", "0.10"),
            questionary.Choice("20% (recomendado)", "0.20"),
            questionary.Choice("30% (máximo)", "0.30")
        ]
    ).ask()
    
    # Configuraciones adicionales
    include_images = questionary.confirm(
        "¿Incluir información sobre imágenes?",
        default=False
    ).ask()
    
    save_chunks = questionary.confirm(
        "¿Guardar chunks en archivos separados?",
        default=True
    ).ask()
    
    export_json = questionary.confirm(
        "¿Exportar datos completos en JSON?",
        default=True
    ).ask()
    
    return LLMProcessingConfig(
        chunk_size=int(chunk_size),
        chunk_overlap=int(int(chunk_size) * float(overlap_percent)),
        semantic_chunking=semantic_chunking,
        preserve_structure=preserve_structure,
        extract_images=include_images,
        save_chunks_separately=save_chunks,
        export_json=export_json,
        optimization_target="custom"
    )


def select_pdf_file() -> Optional[Path]:
    """
    Permite seleccionar un archivo PDF del directorio de entrada.
    
    Returns:
        Optional[Path]: Ruta al PDF seleccionado o None si se cancela
    """
    print(f"\nBuscando archivos PDF en: {PDF_DIR}")
    
    try:
        pdf_files = discover_pdf_files(PDF_DIR)
        
        if not pdf_files:
            print("[ERROR] No se encontraron archivos PDF en el directorio.")
            print(f"   Coloca tus archivos PDF en: {PDF_DIR}")
            return None
            
        print(f"[OK] Encontrados {len(pdf_files)} archivos PDF")
        
        # Crear opciones para el menú
        choices = []
        for pdf_file in pdf_files:
            file_size = pdf_file.stat().st_size / (1024 * 1024)  # MB
            title = f"📄 {pdf_file.name} ({file_size:.1f} MB)"
            choices.append(questionary.Choice(title, pdf_file))
        
        choices.append(questionary.Choice("🔙 Volver al menú principal", None))
        
        selected = questionary.select(
            "Selecciona el archivo PDF a procesar:",
            choices=choices,
            instruction="Usa ↑↓ para navegar, Enter para seleccionar"
        ).ask()
        
        return selected
        
    except Exception as e:
        print(f"[ERROR] Error buscando archivos PDF: {str(e)}")
        return None


def display_processing_progress(pdf_path: Path, config: LLMProcessingConfig) -> None:
    """
    Muestra información sobre el procesamiento que se va a realizar.
    
    Args:
        pdf_path (Path): Archivo que se va a procesar
        config (LLMProcessingConfig): Configuración que se va a usar
    """
    print(f"\n🔄 Procesando: {pdf_path.name}")
    print(f"📊 Configuración:")
    print(f"   • Caso de uso: {config.optimization_target}")
    print(f"   • Tamaño de chunk: {config.chunk_size} caracteres")
    print(f"   • Solapamiento: {config.chunk_overlap} caracteres")
    print(f"   • Chunking semántico: {'[OK]' if config.semantic_chunking else '[NO]'}")
    print(f"   • Preservar estructura: {'[OK]' if config.preserve_structure else '[NO]'}")
    print(f"   • Chunks separados: {'[OK]' if config.save_chunks_separately else '[NO]'}")
    print("\n⏳ Procesando con pymupdf4llm...")


def display_results(result: Dict[str, Any]) -> None:
    """
    Muestra los resultados del procesamiento de forma detallada.
    
    Args:
        result (Dict[str, Any]): Resultado del procesamiento
    """
    if not result.get("success", False):
        print(f"\n[ERROR] Error en el procesamiento:")
        print(f"   {result.get('error', 'Error desconocido')}")
        
        recommendations = result.get("recommendations", [])
        if recommendations:
            print(f"\n💡 Recomendaciones:")
            for rec in recommendations:
                print(f"   • {rec}")
        return
    
    print(f"\n[OK] Procesamiento completado exitosamente!")
    
    # Información básica
    chunk_info = result.get("chunk_info", {})
    processing_stats = result.get("processing_stats", {})
    
    print(f"\n📊 Estadísticas del documento:")
    print(f"   • Chunks generados: {chunk_info.get('total_chunks', 0)}")
    print(f"   • Tamaño promedio de chunk: {processing_stats.get('avg_chunk_size', 0):.0f} caracteres")
    print(f"   • Tokens estimados totales: {processing_stats.get('total_tokens_estimated', 0):.0f}")
    print(f"   • Coherencia semántica: {processing_stats.get('semantic_coherence', 0):.2f}")
    
    # Distribución por secciones
    sections_dist = chunk_info.get("sections_distribution", {})
    if sections_dist:
        print(f"\n📑 Distribución por secciones:")
        for section, count in list(sections_dist.items())[:5]:  # Mostrar top 5
            section_name = section if len(section) <= 40 else section[:37] + "..."
            print(f"   • {section_name}: {count} chunks")
    
    # Archivos generados
    files_generated = result.get("files_generated", [])
    print(f"\n📁 Archivos generados ({len(files_generated)}):")
    for file_path in files_generated[:10]:  # Mostrar primeros 10
        file_name = Path(file_path).name
        print(f"   • {file_name}")
    
    if len(files_generated) > 10:
        print(f"   ... y {len(files_generated) - 10} archivos más")
    
    # Recomendaciones específicas para LLMs
    recommendations = result.get("recommendations", [])
    if recommendations:
        print(f"\n💡 Recomendaciones para uso con LLMs:")
        for rec in recommendations:
            print(f"   • {rec}")
    
    # Información de salida
    output_info = result.get("controller_info", {})
    output_dir = output_info.get("output_directory", "desconocido")
    print(f"\nArchivos guardados en: {output_dir}")


def display_batch_results(batch_result: Dict[str, Any]) -> None:
    """
    Muestra resultados de procesamiento en lote.
    
    Args:
        batch_result (Dict[str, Any]): Resultado del procesamiento en lote
    """
    stats = batch_result.get("aggregate_stats", {})
    
    print(f"\n📊 Resumen del procesamiento en lote:")
    print(f"   • Documentos totales: {batch_result.get('total_documents', 0)}")
    print(f"   • Exitosos: {batch_result.get('successful', 0)}")
    print(f"   • Fallidos: {batch_result.get('failed', 0)}")
    print(f"   • Tasa de éxito: {stats.get('success_rate', 0):.1%}")
    print(f"   • Chunks totales: {stats.get('total_chunks_generated', 0)}")
    print(f"   • Archivos generados: {stats.get('total_files_generated', 0)}")


def process_single_document() -> None:
    """Flujo para procesar un solo documento."""
    # Seleccionar caso de uso
    use_case = select_use_case()
    if use_case == "exit":
        return
    
    # Seleccionar archivo
    pdf_path = select_pdf_file()
    if pdf_path is None:
        return
    
    # Configurar procesamiento
    controller = LLMDocumentController(OUT_DIR)
    
    if use_case == "custom":
        config = configure_custom_settings()
    else:
        config = controller.get_recommended_config(use_case)
    
    # Mostrar configuración
    display_processing_progress(pdf_path, config)
    
    # Procesar documento
    try:
        result = controller.process_document(pdf_path, config)
        display_results(result)
        
    except KeyboardInterrupt:
        print("\n⚠️ Procesamiento cancelado por el usuario")
    except Exception as e:
        print(f"\n[ERROR] Error inesperado: {str(e)}")


def process_batch_documents() -> None:
    """Flujo para procesar múltiples documentos en lote."""
    print(f"\n📁 Buscando archivos PDF en: {PDF_DIR}")
    
    try:
        pdf_files = discover_pdf_files(PDF_DIR)
        
        if not pdf_files:
            print("[ERROR] No se encontraron archivos PDF en el directorio.")
            return
            
        print(f"[OK] Encontrados {len(pdf_files)} archivos PDF")
        
        # Confirmar procesamiento en lote
        confirm = questionary.confirm(
            f"¿Procesar todos los {len(pdf_files)} archivos PDF?",
            default=False
        ).ask()
        
        if not confirm:
            return
        
        # Seleccionar configuración
        use_case = select_use_case()
        if use_case == "exit":
            return
        
        controller = LLMDocumentController(OUT_DIR)
        
        if use_case == "custom":
            config = configure_custom_settings()
        else:
            config = controller.get_recommended_config(use_case)
        
        print(f"\n🔄 Procesando {len(pdf_files)} documentos...")
        print("⏳ Esto puede tomar varios minutos...")
        
        # Procesar en lote
        batch_result = controller.process_multiple_documents(pdf_files, config)
        display_batch_results(batch_result)
        
    except KeyboardInterrupt:
        print("\n⚠️ Procesamiento en lote cancelado por el usuario")
    except Exception as e:
        print(f"\n[ERROR] Error en procesamiento en lote: {str(e)}")


def show_statistics() -> None:
    """Muestra estadísticas del directorio de salida."""
    controller = LLMDocumentController(OUT_DIR)
    stats = controller.get_processing_statistics()
    
    if "error" in stats:
        print(f"\n[ERROR] Error obteniendo estadísticas: {stats['error']}")
        return
    
    print(f"\n📊 Estadísticas del directorio de salida:")
    print(f"   • Archivos totales: {stats.get('total_files', 0)}")
    print(f"   • Tamaño total: {stats.get('total_size_mb', 0)} MB")
    print(f"   • Directorio: {stats.get('output_directory', 'desconocido')}")
    
    recent_files = stats.get("recent_files", [])
    if recent_files:
        print(f"\n📝 Archivos recientes:")
        for file_name in recent_files:
            print(f"   • {file_name}")


def main_llm_menu() -> None:
    """Menú principal para procesamiento LLM."""
    display_llm_welcome()
    
    while True:
        try:
            choice = questionary.select(
                "\n¿Qué deseas hacer?",
                choices=[
                    questionary.Choice(
                        "📄 Procesar un documento PDF",
                        "single"
                    ),
                    questionary.Choice(
                        "Procesar múltiples documentos (lote)",
                        "batch"
                    ),
                    questionary.Choice(
                        "📊 Ver estadísticas de procesamiento",
                        "stats"
                    ),
                    questionary.Choice(
                        "ℹ️ Información sobre casos de uso",
                        "info"
                    ),
                    questionary.Choice(
                        "🚪 Salir",
                        "exit"
                    )
                ],
                instruction="Usa ↑↓ para navegar, Enter para seleccionar"
            ).ask()
            
            if choice == "single":
                process_single_document()
            elif choice == "batch":
                process_batch_documents()
            elif choice == "stats":
                show_statistics()
            elif choice == "info":
                show_use_case_info()
            elif choice == "exit":
                print("\n👋 ¡Hasta luego!")
                break
            else:
                print("\n⚠️ Opción no válida")
                
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n[ERROR] Error en el menú: {str(e)}")


def show_use_case_info() -> None:
    """Muestra información detallada sobre los casos de uso."""
    print("\n📚 Información sobre casos de uso:")
    print("-" * 50)
    
    print("\n🔍 Sistema RAG (Retrieval-Augmented Generation):")
    print("   • Chunks de ~1000 caracteres con 20% de solapamiento")
    print("   • Preserva estructura de encabezados para contexto")
    print("   • Metadatos enriquecidos para mejor retrieval")
    print("   • Ideal para: ChatGPT, Claude, sistemas de Q&A")
    
    print("\n🧮 Vector Database / Embeddings:")
    print("   • Chunks más pequeños (~512 caracteres)")
    print("   • Menor solapamiento para evitar redundancia")
    print("   • Optimizado para modelos de embeddings")
    print("   • Ideal para: búsqueda semántica, clustering")
    
    print("\n🔬 Análisis por LLMs:")
    print("   • Chunks más grandes (~2000 caracteres)")
    print("   • Incluye imágenes y elementos visuales")
    print("   • Genera resúmenes automáticos")
    print("   • Ideal para: análisis de contenido, extracción de insights")
    
    print("\n💡 Tecnología utilizada:")
    print("   • pymupdf4llm: Extracción optimizada para LLMs")
    print("   • Markdown estructurado con preservación de jerarquía")
    print("   • Chunking semánticamente coherente")
    print("   • Metadatos enriquecidos para contexto")


if __name__ == "__main__":
    main_llm_menu()
