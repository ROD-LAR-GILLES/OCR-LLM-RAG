"""
Lógica de menús y selección de opciones.

Este módulo contiene funciones puras para manejar la lógica de menús
sin depender de entrada/salida específica.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class MenuOption:
    """Representa una opción de menú con su identificador y descripción."""
    id: int
    text: str
    value: str = ""


@dataclass
class OCRConfig:
    """Configuración para el motor de OCR seleccionado."""
    engine_type: str  # "basic" o "opencv"
    enable_deskewing: bool = True
    enable_denoising: bool = True
    enable_contrast_enhancement: bool = True


def create_pdf_menu_options(pdf_files: List[str]) -> List[MenuOption]:
    """
    Crea opciones de menú a partir de una lista de archivos PDF.
    
    Args:
        pdf_files (List[str]): Lista de nombres de archivos PDF
        
    Returns:
        List[MenuOption]: Lista de opciones de menú con opción de salida
        
    Example:
        >>> files = ["doc1.pdf", "doc2.pdf"]
        >>> options = create_pdf_menu_options(files)
        >>> print(options[0].text)
        "1. doc1.pdf"
    """
    options = []
    for i, filename in enumerate(pdf_files, 1):
        options.append(MenuOption(id=i, text=f"{i}. {filename}", value=filename))
    
    # Agregar opción de salida
    exit_id = len(pdf_files) + 1
    options.append(MenuOption(id=exit_id, text=f"{exit_id}. Salir", value="exit"))
    
    return options


def validate_menu_selection(selection: int, total_options: int) -> bool:
    """
    Valida que una selección de menú esté dentro del rango válido.
    
    Args:
        selection (int): Número seleccionado por el usuario
        total_options (int): Total de opciones disponibles
        
    Returns:
        bool: True si la selección es válida, False en caso contrario
        
    Example:
        >>> validate_menu_selection(2, 5)
        True
        >>> validate_menu_selection(6, 5)
        False
    """
    return 1 <= selection <= total_options


def get_selected_pdf(pdf_files: List[str], selection: int) -> str:
    """
    Obtiene el archivo PDF seleccionado basado en la opción elegida.
    
    Args:
        pdf_files (List[str]): Lista de archivos PDF disponibles
        selection (int): Número de opción seleccionada (1-indexed)
        
    Returns:
        str: Nombre del archivo PDF seleccionado
        
    Raises:
        ValueError: Si la selección está fuera de rango
        IndexError: Si hay problemas con los índices
        
    Example:
        >>> files = ["doc1.pdf", "doc2.pdf"]
        >>> get_selected_pdf(files, 2)
        "doc2.pdf"
    """
    if not pdf_files:
        raise ValueError("No hay archivos PDF disponibles")
    
    if not validate_menu_selection(selection, len(pdf_files)):
        raise ValueError(f"Selección {selection} fuera de rango. Opciones válidas: 1-{len(pdf_files)}")
    
    try:
        return pdf_files[selection - 1]  # Convertir a 0-indexed
    except IndexError as e:
        raise IndexError(f"Error de índice al seleccionar archivo: {e}")


def is_exit_selection(selection: int, total_pdf_files: int) -> bool:
    """
    Determina si la selección corresponde a la opción de salida.
    
    Args:
        selection (int): Número seleccionado
        total_pdf_files (int): Total de archivos PDF disponibles
        
    Returns:
        bool: True si es selección de salida, False en caso contrario
        
    Example:
        >>> is_exit_selection(3, 2)  # 3 es "Salir" cuando hay 2 PDFs
        True
    """
    return selection == total_pdf_files + 1


def create_ocr_config_from_user_choices(
    engine_choice: int,
    enable_deskewing: bool = True,
    enable_denoising: bool = True,
    enable_contrast: bool = True
) -> OCRConfig:
    """
    Crea configuración de OCR basada en las elecciones del usuario.
    
    Args:
        engine_choice (int): 1 para básico, 2 para OpenCV
        enable_deskewing (bool): Habilitar corrección de inclinación
        enable_denoising (bool): Habilitar eliminación de ruido
        enable_contrast (bool): Habilitar mejora de contraste
        
    Returns:
        OCRConfig: Configuración del motor OCR
        
    Raises:
        ValueError: Si engine_choice no es válido
        
    Example:
        >>> config = create_ocr_config_from_user_choices(2, True, False, True)
        >>> print(config.engine_type)
        "opencv"
    """
    if engine_choice == 1:
        return OCRConfig(
            engine_type="basic",
            enable_deskewing=False,
            enable_denoising=False,
            enable_contrast_enhancement=False
        )
    elif engine_choice == 2:
        return OCRConfig(
            engine_type="opencv",
            enable_deskewing=enable_deskewing,
            enable_denoising=enable_denoising,
            enable_contrast_enhancement=enable_contrast
        )
    else:
        raise ValueError(f"Opción de motor inválida: {engine_choice}. Use 1 (básico) o 2 (OpenCV)")


def validate_ocr_engine_choice(choice: int) -> bool:
    """
    Valida que la elección del motor OCR sea válida.
    
    Args:
        choice (int): Opción elegida por el usuario
        
    Returns:
        bool: True si es válida (1, 2, o 3), False en caso contrario
    """
    return choice in [1, 2, 3]
