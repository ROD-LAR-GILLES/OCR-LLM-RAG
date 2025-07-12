# interfaces/cli/menu.py
"""
Interfaz de línea de comandos interactiva para OCR-CLI.

Este módulo implementa ÚNICAMENTE la capa de presentación e interacción con el usuario.
La lógica de negocio está separada en controladores y utilidades para permitir:
- Testing automatizado sin simulación de I/O
- Reutilización de lógica en otras interfaces (GUI, API)
- Mantenibilidad y extensibilidad mejoradas

Responsabilidades de este módulo (SOLO I/O):
- Captura de entrada del usuario (input())
- Presentación de información (print())
- Formateo y visualización de resultados
- Manejo de errores de interfaz

Lógica de negocio delegada a:
- utils.file_utils: Operaciones de archivos
- utils.menu_logic: Lógica de menús y validaciones
- application.controllers: Coordinación de casos de uso
"""

from pathlib import Path
from typing import List, Optional

from utils.file_utils import discover_pdf_files, validate_pdf_exists
from utils.menu_logic import (
    create_pdf_menu_options, 
    get_selected_pdf, 
    is_exit_selection,
    create_ocr_config_from_user_choices,
    validate_ocr_engine_choice,
    OCRConfig
)
from application.controllers import DocumentController

# Configuración de directorios Docker
# Estos paths son montados como volúmenes en docker-compose.yml
PDF_DIR = Path("/app/pdfs")        # Directorio de entrada (host: ./pdfs)
OUT_DIR = Path("/app/resultado")   # Directorio de salida (host: ./resultado)


def display_welcome_message() -> None:
    """
    Muestra mensaje de bienvenida y título de la aplicación.
    
    Función pura de presentación sin lógica de negocio.
    """
    print("\n" + "="*50)
    print("OCR-CLI - Procesador de documentos PDF")
    print("="*50)


def display_pdf_menu(pdf_files: List[str]) -> None:
    """
    Muestra el menú de selección de archivos PDF.
    
    Args:
        pdf_files (List[str]): Lista de archivos PDF disponibles
        
    Note:
        Función pura de presentación. La lógica de creación de opciones
        está en utils.menu_logic.create_pdf_menu_options()
    """
    print("Selecciona un PDF para procesar:")
    
    menu_options = create_pdf_menu_options(pdf_files)
    for option in menu_options:
        print(option.text)
    
    print("-" * 50)


def get_user_pdf_selection(total_options: int) -> int:
    """
    Captura y valida selección de PDF del usuario.
    
    Args:
        total_options (int): Total de opciones disponibles en el menú
        
    Returns:
        int: Opción seleccionada válida
        
    Note:
        Maneja la validación de entrada pero usa utils.menu_logic
        para la lógica de validación.
    """
    from utils.menu_logic import validate_menu_selection
    
    while True:
        try:
            choice = int(input(f"Ingresa tu opción (1-{total_options}): "))
            if validate_menu_selection(choice, total_options):
                return choice
            else:
                print(f"Opción inválida. Ingresa un número entre 1 y {total_options}.")
        except ValueError:
            print("Por favor ingresa un número válido.")
        except KeyboardInterrupt:
            print("\nSaliendo de la aplicación.")
            raise


def display_ocr_engine_menu() -> None:
    """
    Muestra el menú de selección de motor OCR.
    
    Función pura de presentación.
    """
    print("\nSelecciona el motor de OCR:")
    print("1. Tesseract básico (rápido)")
    print("2. Tesseract + OpenCV (alta calidad)")
    print("3. Volver al menú principal")


def get_user_ocr_selection() -> int:
    """
    Captura y valida selección de motor OCR del usuario.
    
    Returns:
        int: Opción de motor OCR seleccionada (1, 2, o 3)
    """
    while True:
        try:
            choice = int(input("Ingresa tu opción (1-3): "))
            if validate_ocr_engine_choice(choice):
                return choice
            else:
                print("Opción inválida. Ingresa 1, 2 o 3.")
        except ValueError:
            print("Por favor ingresa un número válido.")


def get_advanced_preprocessing_config() -> tuple[bool, bool, bool]:
    """
    Captura configuración avanzada de preprocesamiento del usuario.
    
    Returns:
        tuple: (enable_deskewing, enable_denoising, enable_contrast)
    """
    print("\nConfigurando preprocesamiento OpenCV.")
    
    enable_deskewing = input(
        "¿Corregir inclinación del documento? (recomendado para escaneos) (s/n): "
    ).lower().startswith('s')
    
    enable_denoising = input(
        "¿Aplicar eliminación de ruido? (recomendado para imágenes de baja calidad) (s/n): "
    ).lower().startswith('s')
    
    enable_contrast = input(
        "¿Mejorar contraste automáticamente? (recomendado para documentos con poca iluminación) (s/n): "
    ).lower().startswith('s')
    
    return enable_deskewing, enable_denoising, enable_contrast


def ask_for_advanced_config() -> bool:
    """
    Pregunta al usuario si desea configuración avanzada.
    
    Returns:
        bool: True si quiere configuración avanzada, False para valores por defecto
    """
    response = input("¿Configurar opciones avanzadas de preprocesamiento? (s/n): ")
    return response.lower().startswith('s')


def display_ocr_config_info(config: OCRConfig) -> None:
    """
    Muestra información sobre la configuración OCR seleccionada.
    
    Args:
        config (OCRConfig): Configuración del motor OCR
    """
    if config.engine_type == "basic":
        print("Usando Tesseract básico.")
    else:
        print("Usando Tesseract + OpenCV con preprocesamiento avanzado.")
        print(f"   - Corrección de inclinación: {'SI' if config.enable_deskewing else 'NO'}")
        print(f"   - Eliminación de ruido: {'SI' if config.enable_denoising else 'NO'}")
        print(f"   - Mejora de contraste: {'SI' if config.enable_contrast_enhancement else 'NO'}")


def display_processing_start(filename: str) -> None:
    """
    Muestra mensaje de inicio de procesamiento.
    
    Args:
        filename (str): Nombre del archivo a procesar
    """
    print(f"\nIniciando procesamiento de {filename}.")


def display_processing_success(processing_info: dict) -> None:
    """
    Muestra información de procesamiento exitoso.
    
    Args:
        processing_info (dict): Información del procesamiento exitoso
    """
    print(f"\n{processing_info['filename']} procesado exitosamente!")
    print(f"Tiempo de procesamiento: {processing_info['processing_time']:.2f} segundos")
    print(f"Archivos generados: {processing_info['files_count']}")
    print(f"   - Texto principal: {processing_info['main_text_file']}")
    print(f"   - Todos los archivos: {processing_info['generated_files']}")
    
    # Mostrar información adicional si se usó OpenCV
    if processing_info['ocr_config'].engine_type == "opencv":
        print("Preprocesamiento OpenCV aplicado con éxito")


def display_processing_error(error_info: dict) -> None:
    """
    Muestra información de error en el procesamiento.
    
    Args:
        error_info (dict): Información del error ocurrido
    """
    print(f"\nError procesando {error_info['filename']}:")
    print(f"   Error: {error_info['error']}")
    print(f"   Tiempo hasta error: {error_info['processing_time']:.2f} segundos")
    print("   Sugerencia: Prueba con el motor básico si el documento es de alta calidad")


def display_no_pdfs_message() -> None:
    """
    Muestra mensaje cuando no hay archivos PDF disponibles.
    """
    print("No hay PDFs en /pdfs. Añade archivos y reconstruye la imagen.")


def display_exit_message() -> None:
    """
    Muestra mensaje de salida de la aplicación.
    """
    print("Saliendo de la aplicación.")


def process_document_workflow(filename: str) -> None:
    """
    Ejecuta el flujo completo de procesamiento de un documento.
    
    Esta función coordina la interfaz de usuario para seleccionar opciones
    de OCR y delega el procesamiento real al controlador.
    
    Args:
        filename (str): Nombre del archivo PDF a procesar
        
    Note:
        Separación de responsabilidades:
        - Esta función: I/O y presentación
        - DocumentController: Lógica de procesamiento
        - Utilidades: Validación y configuración
    """
    # SELECCIÓN DEL MOTOR OCR
    display_ocr_engine_menu()
    
    ocr_choice = get_user_ocr_selection()
    
    if ocr_choice == 3:  # Volver al menú principal
        return
    
    # CONFIGURACIÓN DEL MOTOR OCR
    if ocr_choice == 1:
        # Configuración básica
        ocr_config = create_ocr_config_from_user_choices(1)
        
    elif ocr_choice == 2:
        # Configuración OpenCV
        if ask_for_advanced_config():
            # Configuración personalizada
            deskewing, denoising, contrast = get_advanced_preprocessing_config()
            ocr_config = create_ocr_config_from_user_choices(
                2, deskewing, denoising, contrast
            )
        else:
            # Configuración por defecto
            ocr_config = create_ocr_config_from_user_choices(2)
    
    # Mostrar configuración seleccionada
    display_ocr_config_info(ocr_config)
    
    # PROCESAMIENTO DEL DOCUMENTO
    display_processing_start(filename)
    
    # Crear controlador y procesar
    controller = DocumentController(PDF_DIR, OUT_DIR)
    success, processing_info = controller.process_document(filename, ocr_config)
    
    # MOSTRAR RESULTADOS
    if success:
        display_processing_success(processing_info)
    else:
        display_processing_error(processing_info)
    
    print()  # Línea en blanco para separación visual


def main() -> None:
    """
    Función principal que ejecuta el bucle interactivo de la aplicación.
    
    Implementa ÚNICAMENTE la lógica de presentación e interacción:
    - Muestra menús
    - Captura entrada del usuario  
    - Delega lógica de negocio a utilidades y controladores
    
    Separación de responsabilidades:
    - Esta función: I/O puro
    - utils.file_utils: Descubrimiento de archivos
    - utils.menu_logic: Validación de selecciones
    - DocumentController: Procesamiento de documentos
    """
    while True:
        try:
            # DESCUBRIMIENTO DE ARCHIVOS (delegado a utilidad)
            pdf_files = discover_pdf_files(PDF_DIR)
            
            # VALIDACIÓN DE DISPONIBILIDAD
            if not pdf_files:
                display_no_pdfs_message()
                break
            
            # PRESENTACIÓN DEL MENÚ
            display_welcome_message()
            display_pdf_menu(pdf_files)
            
            # CAPTURA DE SELECCIÓN
            total_options = len(pdf_files) + 1  # +1 para opción "Salir"
            selection = get_user_pdf_selection(total_options)
            
            # PROCESAMIENTO DE SELECCIÓN (usando lógica separada)
            if is_exit_selection(selection, len(pdf_files)):
                display_exit_message()
                return
            else:
                # Obtener archivo seleccionado (lógica delegada)
                selected_file = get_selected_pdf(pdf_files, selection)
                process_document_workflow(selected_file)
                
        except KeyboardInterrupt:
            print("\nSaliendo de la aplicación.")
            return
        except FileNotFoundError:
            print("Error: El directorio de PDFs no está disponible.")
            print("Verifica que el contenedor esté configurado correctamente.")
            break
        except Exception as e:
            print(f"Error inesperado: {e}")
            print("Contacta al administrador del sistema.")
            break


if __name__ == "__main__":
    """
    Punto de entrada cuando el módulo se ejecuta directamente.
    
    Permite ejecutar la aplicación con:
    python interfaces/cli/menu.py
    
    En el contexto Docker, este es el comando por defecto definido
    en el Dockerfile, ejecutándose automáticamente al iniciar el contenedor.
    """
    main()