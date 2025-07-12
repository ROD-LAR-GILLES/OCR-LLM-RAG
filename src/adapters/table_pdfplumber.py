# adapters/table_pdfplumber.py
"""
Adaptador de extracción de tablas basado en pdfplumber.

Este módulo implementa el puerto TableExtractorPort utilizando pdfplumber,
una librería especializada en análisis estructural de documentos PDF que
puede detectar y extraer tablas manteniendo su formato original.
"""
from pathlib import Path
from typing import List

import pdfplumber
import pandas as pd

from application.ports import TableExtractorPort


class PdfPlumberAdapter(TableExtractorPort):
    """
    Adaptador para extracción de tablas que utiliza pdfplumber.
    
    pdfplumber es una librería que analiza la estructura interna de PDFs
    para detectar elementos como tablas, texto y metadatos sin necesidad
    de OCR, trabajando directamente con el contenido vectorial del PDF.
    
    Ventajas de pdfplumber:
    - Extracción precisa de tablas con bordes definidos
    - Mantiene la estructura original de celdas
    - Rápido (no requiere conversión a imagen ni OCR)
    - Detecta automáticamente límites de tabla
    - Soporta tablas complejas con celdas combinadas
    
    Limitaciones:
    - Solo funciona con PDFs nativos (no escaneados)
    - Requiere que las tablas tengan estructura clara
    - No funciona con tablas en imágenes incrustadas
    - Problemas con tablas sin bordes o con formato irregular
    
    Casos de uso ideales:
    - Reportes financieros generados digitalmente
    - Documentos empresariales con tablas estructuradas
    - PDFs creados desde Excel, Word, o herramientas similares
    """

    def extract_tables(self, pdf_path: Path) -> List[pd.DataFrame]:
        """
        Extrae todas las tablas detectadas en un documento PDF.
        
        Proceso de extracción:
        1. Abre el PDF y analiza su estructura interna
        2. Recorre cada página buscando elementos tabulares
        3. Para cada tabla detectada, extrae contenido celda por celda
        4. Convierte cada tabla a pandas DataFrame para facilitar manipulación
        
        Args:
            pdf_path (Path): Ruta al archivo PDF a procesar
            
        Returns:
            List[pd.DataFrame]: Lista de DataFrames, uno por cada tabla detectada.
                               Lista vacía si no se encuentran tablas.
                               
        Raises:
            FileNotFoundError: Si el archivo PDF no existe
            pdfplumber.pdf.PdfReadError: Si el PDF está corrupto o protegido
            pandas.errors.EmptyDataError: Si una tabla detectada está vacía
            
        Note:
            - pdfplumber.extract_tables() devuelve listas de listas (filas y columnas)
            - pandas.DataFrame convierte estas listas en estructuras de datos manipulables
            - El orden de las tablas en la lista corresponde al orden de aparición en el PDF
            - Las celdas vacías se representan como None en el DataFrame resultante
        """
        dfs: List[pd.DataFrame] = []
        
        # Abre el PDF usando pdfplumber, que analiza la estructura vectorial
        # del documento sin convertirlo a imagen
        with pdfplumber.open(pdf_path) as pdf:
            # Itera sobre cada página del documento
            for page in pdf.pages:
                # page.extract_tables() detecta automáticamente tablas en la página
                # Utiliza algoritmos de análisis de espaciado y líneas para identificar
                # estructuras tabulares basándose en:
                # - Líneas horizontales y verticales
                # - Espaciado consistente entre elementos
                # - Alineación de texto en columnas
                for table in page.extract_tables():
                    # Convierte cada tabla (lista de listas) a pandas DataFrame
                    df = pd.DataFrame(table)
                    
                    # Filtrar tablas con criterios de calidad
                    if self._is_valid_table(df):
                        dfs.append(df)
                    
        return dfs
    
    def _is_valid_table(self, df: pd.DataFrame) -> bool:
        """
        Valida si una tabla extraída es realmente significativa.
        
        Filtra tablas que probablemente sean falsos positivos:
        - Tablas con muy pocas filas o columnas
        - Tablas con demasiadas celdas vacías
        - Tablas que parecen ser listas simples o índices
        - Tablas con contenido muy repetitivo
        
        Args:
            df: DataFrame de la tabla extraída
            
        Returns:
            bool: True si la tabla es válida, False si debe descartarse
        """
        # Filtro 1: Tamaño mínimo - necesitamos al menos 2x2 para una tabla real
        if df.shape[0] < 2 or df.shape[1] < 2:
            return False
        
        # Filtro 2: Demasiadas celdas vacías
        total_cells = df.shape[0] * df.shape[1]
        empty_cells = df.isnull().sum().sum() + (df == '').sum().sum()
        if empty_cells / total_cells > 0.6:  # Más del 60% vacío
            return False
            
        # Filtro 3: Tabla demasiado pequeña (menos de 8 celdas total)
        if total_cells < 8:
            return False
            
        # Filtro 4: Detectar tablas que son solo listas numeradas (como la problemática)
        if df.shape[1] == 2:  # Exactamente dos columnas
            try:
                first_col = df.iloc[:, 0].astype(str).str.strip()
                second_col = df.iloc[:, 1].astype(str).str.strip()
                
                # Verificar si la primera columna son números consecutivos empezando en 0
                if len(first_col) >= 3:  # Al menos 3 filas
                    try:
                        nums = [int(x) for x in first_col if x.isdigit()]
                        if len(nums) == len(first_col) and nums == list(range(len(nums))):
                            # Es una secuencia 0,1,2,3... - probablemente un índice, no una tabla real
                            return False
                    except (ValueError, TypeError):
                        pass
                
                # Verificar si contiene palabras clave que indican que es un índice de contenido
                content_keywords = ['cabecera', 'cuerpo', 'tabla', 'trailer', 'índice', 'contenido']
                second_col_text = ' '.join(second_col).lower()
                keyword_matches = sum(1 for keyword in content_keywords if keyword in second_col_text)
                if keyword_matches >= 2:  # Si aparecen 2 o más palabras clave, es probablemente un índice
                    return False
                    
            except Exception:
                pass
        
        # Filtro 5: Contenido muy repetitivo o vacío
        unique_values = set()
        for col in df.columns:
            col_values = df[col].dropna().astype(str).str.strip()
            col_values = col_values[col_values != '']  # Remover strings vacíos
            unique_values.update(col_values.unique())
        
        # Si hay muy pocos valores únicos no vacíos
        if len(unique_values) < 3:
            return False
            
        # Filtro 6: Verificar que haya contenido sustantivo
        # Al menos el 50% de las celdas deben tener contenido no trivial
        non_trivial_content = 0
        for col in df.columns:
            for val in df[col]:
                if val is not None and str(val).strip() and len(str(val).strip()) > 1:
                    non_trivial_content += 1
        
        if non_trivial_content / total_cells < 0.5:
            return False
            
        return True