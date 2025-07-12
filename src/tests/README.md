# Tests para OCR-CLI

Este directorio contiene tests automatizados que demuestran los beneficios de la separación de I/O y lógica de negocio.

## Estructura de Tests

```
tests/
├── test_file_utils.py       # Tests para utilidades de archivos
├── test_menu_logic.py       # Tests para lógica de menús
├── test_controllers.py      # Tests para controladores
└── README.md               # Este archivo
```

## Beneficios de la Separación I/O

### Antes (Código Mezclado)
```python
def main():
    archivos = listar_pdfs()        # Lógica de negocio
    print("Selecciona PDF:")        # I/O
    choice = int(input("Opción: ")) # I/O
    procesar_archivo(archivos[choice - 1])  # Lógica mezclada
```

**Problemas:**
-  No se puede testear sin simular `input()` y `print()`
-  Difícil de reutilizar en otras interfaces
-  Tests complejos y frágiles

###  Después (Código Separado)
```python
# Lógica pura (testeable)
def get_selected_pdf(archivos: list, seleccion: int) -> str:
    if 1 <= seleccion <= len(archivos):
        return archivos[seleccion - 1]
    raise ValueError("Selección inválida")

# I/O separado
def main():
    archivos = discover_pdf_files(PDF_DIR)  # Función pura
    print("Selecciona PDF:")
    choice = int(input("Opción: "))
    archivo = get_selected_pdf(archivos, choice)  # Función pura
```

**Beneficios:**
-  Lógica testeable sin simulación de I/O
-  Reutilizable en GUI, API, tests
-  Tests simples y confiables

## Ejecutar Tests

### Usando pytest (recomendado)
```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar todos los tests
pytest tests/

# Ejecutar tests específicos
pytest tests/test_menu_logic.py

# Ejecutar con verbose
pytest -v tests/

# Ejecutar con coverage
pytest --cov=utils --cov=application tests/
```

### Usando unittest
```bash
# Ejecutar test específico
python -m unittest tests.test_controllers

# Ejecutar todos los tests
python -m unittest discover tests/
```

## Ejemplos de Tests

### 1. Test de Lógica de Menú (Sin I/O)
```python
def test_get_selected_pdf():
    archivos = ["doc1.pdf", "doc2.pdf"]
    
    # Test caso válido
    assert get_selected_pdf(archivos, 2) == "doc2.pdf"
    
    # Test caso inválido
    with pytest.raises(ValueError):
        get_selected_pdf(archivos, 5)
```

### 2. Test de Controlador (Con Mocks)
```python
@patch('application.controllers.ProcessDocument')
def test_process_document_success(mock_process):
    controller = DocumentController(pdf_dir, output_dir)
    config = OCRConfig("basic")
    
    success, info = controller.process_document("test.pdf", config)
    
    assert success is True
    assert info["filename"] == "test.pdf"
```

### 3. Test de Utilidades de Archivo
```python
def test_discover_pdf_files():
    with TemporaryDirectory() as tmpdir:
        directory = Path(tmpdir)
        (directory / "doc1.pdf").touch()
        (directory / "doc2.pdf").touch()
        
        result = discover_pdf_files(directory)
        
        assert len(result) == 2
        assert "doc1.pdf" in result
```

## Cobertura de Tests

Los tests cubren:

-  **Lógica de menús**: Validación de selecciones, creación de opciones
-  **Utilidades de archivos**: Descubrimiento, validación, información
-  **Controladores**: Procesamiento de documentos, manejo de errores
-  **Configuración OCR**: Creación y validación de configuraciones

## Tests de Integración

Para tests de integración completos:

```bash
# Test de flujo completo (requiere archivos PDF reales)
python tests/integration_test.py

# Test con Docker
docker-compose run app pytest tests/
```

## Debugging Tests

```bash
# Ejecutar test específico con debugging
pytest -v -s tests/test_menu_logic.py::TestGetSelectedPdf::test_get_selected_pdf_valid_selection

# Ver output de print statements
pytest -s tests/

# Parar en primer fallo
pytest -x tests/
```

## Ventajas Demostradas

1. **Testing Rápido**: Tests ejecutan en milisegundos sin I/O real
2. **Testing Confiable**: No dependen de archivos externos o entrada de usuario
3. **Testing Completo**: Cada función lógica tiene tests específicos
4. **Refactoring Seguro**: Tests detectan cambios que rompen funcionalidad
5. **Documentación Viva**: Tests sirven como ejemplos de uso

La separación de I/O convierte código difícil de testear en código que se puede testear automáticamente con confidence.
