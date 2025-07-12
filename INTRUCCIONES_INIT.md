# Introducción a PyMuPDF4LLM

PyMuPDF4LLM es un paquete diseñado para facilitar la extracción de contenido de PDFs en un formato óptimo para entornos de LLM/RAG (Large Language Models con Retrieval-Augmented Generation). Esta librería actúa como un wrapper de alto nivel sobre PyMuPDF (la biblioteca base de manipulación PDF) para convertir cada página de un PDF en texto Markdown estructurado, ideal para alimentar modelos de lenguaje. Su objetivo es conservar la mayor fidelidad posible al contenido y formato del documento original, detectando elementos como texto en múltiples columnas, tablas, imágenes y gráficos vectoriales, y exportándolos de forma coherente en Markdown. De esta manera, PyMuPDF4LLM unifica diferentes tipos de extracción (texto, tablas, imágenes) en una sola representación Markdown unificada de cada página.

Al usar Markdown como formato de salida, se aprovechan sus ventajas para los LLMs, por ejemplo: se preservan encabezados, negritas, cursivas, listas y bloques de código con sintaxis explícita, lo que aporta contexto semántico al modelo. Las líneas de título se identifican automáticamente por tamaño de fuente y se les antepone el número apropiado de almohadillas `#` en Markdown (por ejemplo, textos grandes se marcan como `# Título`). Igualmente, el texto en negrita, cursiva, monoespaciado o en formato de código se detecta y se formatea con los símbolos Markdown correspondientes. Listas ordenadas y viñetas también son reconocidas y transformadas en listados Markdown equivalentes.

En resumen, PyMuPDF4LLM permite extraer de forma rápida y fiable el contenido de documentos PDF complejos (múltiples columnas, distintos estilos de texto, tablas, imágenes) manteniendo su estructura lógica en Markdown. Esto resulta extremadamente útil para posteriormente fragmentar ese contenido y proporcionárselo a un LLM junto con sus estilos y contexto original, mejorando la relevancia de las respuestas generadas.

## Instalación y Requisitos Previos

Para incorporar PyMuPDF4LLM en tu proyecto, primero debes instalarlo mediante pip:

```bash
pip install pymupdf4llm
```

Este paquete instalará automáticamente PyMuPDF como dependencia si aún no lo tienes. Dado que PyMuPDF4LLM trabaja principalmente a nivel de software (no requiere complementos nativos adicionales aparte de PyMuPDF), la instalación es sencilla. Sin embargo, si planeas manejar PDFs escaneados (imágenes), necesitarás tener instalado Tesseract OCR en el sistema, ya que PyMuPDF puede aprovechar Tesseract para reconocer texto en imágenes. Asegúrate de que Tesseract esté disponible en tu PATH del sistema operativo y cuenta con los datos de idioma apropiados (por ejemplo, el paquete de idioma español `spa` si procesarás texto en español).

Nota: PyMuPDF4LLM también puede extenderse para procesar otros formatos (Word, Excel, PowerPoint, etc.) si dispones de PyMuPDF Pro. En ese caso, tras desbloquear PyMuPDF Pro con tu licencia, el mismo método `to_markdown` funcionará con archivos Office (DOCX, XLSX, etc.) de forma transparente. Esta característica es opcional y solo es relevante si tu proyecto requiere manejar esos formatos.

## Uso Básico: Extracción de PDF Digital a Markdown

Una vez instalado, el uso básico de PyMuPDF4LLM para PDFs digitales (aquellos que ya contienen texto, no puras imágenes) es muy directo. Solo se requieren un par de líneas de Python:

```python
import pymupdf4llm

md_text = pymupdf4llm.to_markdown("input.pdf")
```

Como resultado obtendrás en la variable `md_text` una cadena de texto en formato Markdown que representa todo el contenido del PDF de entrada. Por defecto, se procesan todas las páginas del documento y se concatenan en un solo string Markdown. Puedes, si lo deseas, limitar la extracción a ciertas páginas especificando el parámetro `pages` con una lista de números de página (basados en índice 0). Por ejemplo:

```python
pymupdf4llm.to_markdown("input.pdf", pages=[0, 1])
```

Después de obtener el Markdown, puedes guardarlo en un archivo para inspección o uso posterior:

```python
# Guardar la salida Markdown en un archivo UTF-8
import pathlib
pathlib.Path("output.md").write_bytes(md_text.encode("utf-8"))
print("Markdown guardado en output.md")
```

Este fragmento crea un archivo `output.md` con el contenido extraído. Ten en cuenta que la salida preserva la mayor parte de la estructura de formato original del PDF:

- Los encabezados del documento se verán reflejados como encabezados Markdown (`#`, `##`, etc.) según su nivel.
- El texto con énfasis (negritas/cursivas) aparecerá rodeado de `**` o `*` en la sintaxis Markdown.
- Los fragmentos de código o texto monoespaciado aparecerán formateados con backticks `` ` `` o bloques de código con sangría/apertura de triple backtick, según corresponda.
- Las listas y viñetas se convertirán en listas Markdown usando `-` o `1.` apropiadamente.
- Las tablas presentes en el PDF serán detectadas y volcadas como tablas en Markdown, manteniendo sus filas y columnas.

## Detección de Tablas y Formato Markdown

PyMuPDF4LLM detecta automáticamente las tablas en el PDF y las incorpora a la salida Markdown usando el formato de tablas de GitHub-Flavored Markdown (GFM). Por ejemplo, una tabla sencilla se representaría así en el Markdown de salida:

```markdown
| Columna1    | Columna2    |
|-------------|-------------|
| Celda (0,0) | Celda (0,1) |
| Celda (1,0) | Celda (1,1) |
| Celda (2,0) | Celda (2,1) |
```

Cada columna se separa con `|` y se incluye una división por guiones `---` tras la primera fila para indicar el encabezado de la tabla. Si la tabla original no tenía explícitamente una fila de encabezado, el extractor asumirá que la primera fila es el encabezado, ya que el formato Markdown requiere obligatoriamente un header para la tabla. Este comportamiento asegura que todas las tablas extraídas sean válidas en sintaxis Markdown. Además, el formato generado es lo más compacto posible (sin texto redundante), pensando en optimizar la entrada al LLM (menos tokens).

Internamente, PyMuPDF4LLM analiza la página para distinguir entre bloques de texto corrido y tablas. Realiza una “clasificación de áreas” en cada página, identificando regiones que son tablas (por la alineación en filas y columnas, líneas dibujadas, etc.) y extrayendo el contenido de las celdas en el orden correcto. De este modo, aunque en el PDF las tablas pudieran estar divididas en varios cuadros de texto o ser difíciles de leer linealmente, la salida Markdown presentará la tabla reconstruida de forma legible y consistente.

## Manejo de Imágenes y Gráficos

Otra capacidad importante es la extracción de imágenes y gráficos vectoriales. Por defecto, PyMuPDF4LLM no extrae las imágenes como archivos separados a menos que se le indique, pero sí puede incluir referencias a ellas. Hay dos maneras de manejar las imágenes en la salida:

- **Referenciando imágenes externamente**: Si llamamos a `to_markdown` con el parámetro `write_images=True`, el proceso exportará cada imagen o gráfico vectorial encontrado en las páginas a archivos de imagen en disco, e insertará en el Markdown resultante referencias del tipo `![](ruta/archivo.png)` apuntando a esos archivos. Por defecto, `write_images` es `False` (no exporta las imágenes). Al activarlo, podemos especificar opciones como:
  - `image_path`: la carpeta donde se guardarán las imágenes (por defecto, el directorio actual).
  - `image_format`: formato/extensión de imagen, por ejemplo "png" (por defecto) o "jpg", entre los soportados por PyMuPDF.
  - `dpi`: la resolución en DPI para renderizar las imágenes (150 por defecto; si necesitas más detalle pon, por ejemplo, 300).
  - `image_size_limit`: un filtro de tamaño relativo; por defecto 0.05, lo que significa que se ignorarán (no se exportarán) imágenes cuyo ancho y alto sean menores al 5% del tamaño de la página, asumiendo que son elementos insignificantes. Puedes ajustar este umbral si requieres extraer incluso imágenes muy pequeñas.

Los archivos de imagen generados seguirán un esquema de nombre: `NOMBREPDF-p{numPágina}-{index}.{ext}`. Por ejemplo, si el PDF se llama `informe.pdf` y en la página 0 hay dos imágenes, se podrían guardar como `informe.pdf-p0-0.png` e `informe.pdf-p0-1.png`. Cada imagen exportada mantiene las dimensiones originales que tenía en la página PDF (ancho y alto en píxeles equivalentes), salvo que modifiquemos el DPI para escalarla.

- **Incrustando imágenes en el Markdown**: Alternativamente, PyMuPDF4LLM permite `embed_images=True`, lo cual inserta directamente las imágenes codificadas en base64 dentro del Markdown (usando la sintaxis de image con un data URI). Sin embargo, no es recomendable incrustar imágenes grandes en base64 ya que el Markdown resultante crece mucho en tamaño. Normalmente es más manejable guardar las imágenes como archivos separados (opción anterior). Ten en cuenta que si `embed_images=True`, no necesitas especificar `image_path` ni formato, y `write_images` será ignorado.

Nota: De forma predeterminada, PyMuPDF4LLM incluye el texto subyacente de un área aunque también la exporte como imagen, para asegurar que no se pierda información. Pero esto puede resultar en texto duplicado (visible en la imagen y también como texto). Si deseas evitar duplicar texto contenido dentro de imágenes, puedes establecer `force_text=False`. Con `force_text=False` el texto que esté cubierto por una imagen no se incluirá por separado en la salida Markdown, asumiendo que quedará legible como parte de la imagen. Esto es útil, por ejemplo, si tienes páginas enteras escaneadas como imagen: probablemente prefieras manejar eso vía OCR en vez de incluir la imagen y repetir el texto. De lo contrario, con el valor por defecto (`force_text=True`), PyMuPDF4LLM intentará extraer texto sobrepuesto a imágenes (cuando existan ambos elementos) y lo añadirá después de la imagen en el Markdown, lo que en algunos casos podría verse redundante.

En caso de que no quieras ninguna imagen en la salida (por ejemplo, solo te interesa el texto, y quieres agilizar el procesamiento), puedes usar `ignore_images=True` para que el extractor simplemente omita las imágenes por completo. Similarmente `ignore_graphics=True` hará que se ignoren formas o gráficos vectoriales (líneas, diagramas). Estas opciones pueden ser útiles para documentos donde las imágenes no aportan valor a las consultas (p. ej., fondos decorativos, logos) y entorpecen la correcta extracción de texto.

## Salida Estructurada por Página y Metadatos

En su modo estándar, `pymupdf4llm.to_markdown()` devuelve un único string concatenando todas las páginas en secuencia. Sin embargo, la biblioteca ofrece un modo muy útil para ciertos casos: obtener la salida página por página junto con metadatos asociados. Activando el parámetro `page_chunks=True`, la función devolverá una lista de diccionarios, donde cada diccionario corresponde a una página del documento.

Por ejemplo:

```python
data_pages = pymupdf4llm.to_markdown("input.pdf", page_chunks=True)
print(len(data_pages))  # debería ser igual al número de páginas del PDF
primera_pagina = data_pages[0]
print(primera_pagina.keys())
```

Las claves presentes en cada uno de estos diccionarios por página incluyen:

- `metadata`: Un sub-diccionario con los metadatos del documento (autor, título, etc., obtenibles vía `Document.metadata` de PyMuPDF) enriquecido con información adicional de página, como el `file_path` (ruta/nombre del archivo PDF procesado), `page_count` (total de páginas del documento) y `page_number` (número de página actual, 1-indexado). Estos metadatos son útiles para conservar contexto del origen del fragmento, o para mostrar referencias al usuario.
- `toc_items`: Lista de elementos de la tabla de contenidos (Table of Contents) del PDF que apuntan a esta página. Si el PDF original tenía un índice o bookmarks jerarquizados, aquí se listarán aquellos items cuyo destino es la página en cuestión. Cada entrada es una tupla o lista del estilo `[nivel, título, num_página]` (por ejemplo `[2, "2.1 Metodología", 5]` indicaría un ítem de TOC de nivel 2 titulado "2.1 Metodología" que apunta a la página 5).
- `tables`: Una lista con información de cada tabla detectada en la página. Por cada tabla hay un diccionario que incluye al menos:
  - `bbox`: el bounding box (cuadrilátero delimitador) de la tabla en la página, dado como tupla `(x0, y0, x1, y1)` en coordenadas de PDF.
  - `row_count`: número de filas de la tabla.
  - `col_count`: número de columnas de la tabla.
- `images`: Una lista con los datos de cada imagen encontrada en la página. Cada elemento de la lista equivale a lo que retornaría PyMuPDF con `page.get_image_info()`: información como posición, tamaño, formato, máscara, etc. de la imagen. Esto permite saber, por ejemplo, cuántas imágenes tenía la página y sus características, independientemente de si se exportaron o no con `write_images`.
- `graphics`: Lista de los cuadros delimitadores (bounding boxes) de grupos de gráficos vectoriales en la página. PyMuPDF4LLM analiza los dibujos vectoriales (líneas, formas) y los agrupa; aquí proporciona sus ubicaciones (sirve para diagnosticar páginas con muchos elementos gráficos o posibles pseudo-textos dibujados). Por defecto, estos gráficos también se consideran para la detección de tablas (ya que las líneas dibujadas suelen delimitar celdas).
- `text`: Contiene el texto Markdown correspondiente solo a esa página. Es decir, es equivalente a lo que obtendríamos si procesáramos individualmente esa página a Markdown. Aquí es donde reside el contenido utilizable directamente para un LLM por página.
- `words`: Esta clave aparece solo si se invocó `to_markdown` con el parámetro `extract_words=True`. En tal caso, por cada página obtendremos también una lista de tuplas representando cada palabra individual junto con sus coordenadas en la página y otros índices. Cada tupla tiene el formato `(x0, y0, x1, y1, "palabra", bno, lno, wno)`, que corresponden a la caja delimitadora de la palabra y su texto, y referencias al bloque, línea y número de palabra. Importante: el orden de esta lista de palabras está sincronizado con el orden del texto en el Markdown (respetando flujo en columnas y filas de tablas). Esta opción es útil si se necesita un análisis más fino de la posición de las palabras (por ejemplo, para resaltar respuestas en el documento original). Ten en cuenta que al activar `extract_words=True`, automáticamente `page_chunks` se fuerza a `True` (porque necesita entregar estructuras por página) y además PyMuPDF4LLM desactiva el formato de código (`ignore_code=True`) para no distorsionar la correspondencia palabra por palabra.

En la práctica, el modo `page_chunks=True` con PyMuPDF4LLM resulta valioso si deseas implementar funcionalidades tipo highlight o referencias de página en las respuestas: puedes almacenar estas estructuras y saber exactamente de qué página proviene cada fragmento de texto que un LLM podría citar.

## Integración de PyMuPDF4LLM con OCR para PDFs Escaneados

Hasta ahora hemos descrito la extracción en PDFs que contienen texto embebido (los generados digitalmente). Sin embargo, en PDFs escaneados (documentos que son esencialmente imágenes escaneadas de páginas sin capa de texto), la extracción directa con PyMuPDF4LLM no obtendrá texto útil, ya que no hay texto que extraer (los métodos normales devolverían cadenas vacías). De hecho, la propia documentación indica que en PDFs completamente escaneados, la conversión a Markdown “no funcionará eficazmente directamente, ya que no hay capa de texto extraíble”. En estos casos es necesario integrar un paso de OCR (Reconocimiento Óptico de Caracteres) antes o durante la extracción.

Una ventaja es que PyMuPDF ofrece soporte OCR integrado utilizando Tesseract. Podemos combinar esta capacidad con PyMuPDF4LLM siguiendo un flujo de trabajo recomendado:

1.  Detectar páginas escaneadas: Abrimos el documento con PyMuPDF (fitz). Recorremos sus páginas y determinamos cuáles no tienen texto. Por ejemplo, usando `page.get_text("text")` – si devuelve una cadena vacía o muy poca información, es señal de que esa página es escaneada. Otra heurística es usar `page.get_text("words")` y ver si la lista viene vacía.
2.  Rasterizar la página a imagen: Para cada página sin texto, utilizamos `page.get_pixmap(dpi=300)` para renderizarla a una imagen (Pixmap) con resolución suficiente (300 DPI suele funcionar bien para OCR; se puede ajustar según la calidad del original).
3.  Aplicar OCR con Tesseract: Tomamos ese Pixmap e invocamos el método integrado `pix.pdfocr_tobytes(language="spa")` (por ejemplo, para español) el cual genera un PDF de una sola página en memoria, que contiene la imagen original pero ahora con una capa de texto oculto reconocida por Tesseract. En esencia, convierte la imagen en un PDF searchable (PDF/A) con el texto OCR.
4.  Reemplazar/Combinar la página OCR en el documento: Abrimos ese PDF de una página resultante (por ejemplo con `ocr_page_doc = fitz.open("pdf", ocr_bytes)` donde `ocr_bytes` es lo devuelto por `pdfocr_tobytes`) y la insertamos en un nuevo documento de salida. Podemos crear `out_doc = fitz.open()` vacío al inicio, e ir añadiéndole páginas. Si la página original tenía texto (no escaneada), podemos copiarla tal cual; si era escaneada, añadimos en su lugar la página OCR procesada. PyMuPDF provee `Document.insert_pdf`: por ejemplo `out_doc.insert_pdf(ocr_page_doc)` para anexar la página con texto reconocido. De esta manera, reconstruimos un documento donde todas las páginas tienen capa de texto (las originalmente digitales mantienen su texto, las escaneadas ahora tienen texto OCR oculto bajo la imagen).
5.  Extraer Markdown del documento combinado: Finalmente, aplicamos `pymupdf4llm.to_markdown()` sobre `out_doc` en vez del documento original. Como ahora cada página tiene texto accesible, PyMuPDF4LLM podrá extraerlo y formatearlo correctamente en Markdown.

Este flujo asegura que incluso los PDFs originalmente escaneados se conviertan en Markdown con texto real. Un punto crucial es que la imagen de la página se mantiene (como fondo) y la capa OCR es “oculta”, por lo que PyMuPDF leerá ese texto como si hubiera estado allí desde el principio. En la práctica, después de hacer OCR, puedes utilizar exactamente la misma lógica de PyMuPDF4LLM que con un PDF digital.

A continuación se muestra un ejemplo de implementación en código, integrado con una arquitectura de puertos y adaptadores:

```python
import fitz  # PyMuPDF
import pymupdf4llm
from pathlib import Path

class PyMuPDFAdapter(OCRPort):
    def extract_text(self, pdf_path: Path) -> str:
        # Abrir el documento PDF
        doc = fitz.open(pdf_path)
        # Verificar si alguna página requiere OCR
        needs_ocr = any(page.get_text().strip() == "" for page in doc)
        if needs_ocr:
            # Crear documento de salida combinando páginas OCR
            out_doc = fitz.open()
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                if page.get_text().strip() == "":
                    # Página escaneada: aplicar OCR
                    pix = page.get_pixmap(dpi=300)
                    # Idioma español (usar "eng", "eng+spa", etc. según necesites)
                    ocr_pdf_bytes = pix.pdfocr_tobytes(language="spa")
                    ocr_doc = fitz.open("pdf", ocr_pdf_bytes)
                    out_doc.insert_pdf(ocr_doc)  # agregar la página OCR al documento
                    ocr_doc.close()
                else:
                    # Página con texto: copiarla tal cual al documento de salida
                    out_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)
            # Extraer Markdown del documento combinado
            md_text = pymupdf4llm.to_markdown(out_doc)
            out_doc.close()
        else:
            # Si no necesita OCR, extraer directamente
            md_text = pymupdf4llm.to_markdown(doc)
        doc.close()
        return md_text
```

En este ejemplo, `PyMuPDFAdapter` implementa la interfaz `OCRPort` de tu arquitectura Clean/Hexagonal (puerto para extracción de texto OCR). Su método `extract_text` acepta la ruta de un PDF y devuelve un string con el contenido Markdown. La implementación verifica página por página; si alguna está escaneada, construye un `out_doc` aplicando OCR en las necesarias. Nótese que se usa `language="spa"` en `pdfocr_tobytes` – es importante establecer el idioma adecuado para que Tesseract tenga mejor precisión (por ejemplo "eng" para inglés, o múltiples idiomas "eng+spa" si el documento es bilingüe). Una vez listo el documento (original o combinado), se pasa directamente a `pymupdf4llm.to_markdown` para obtener el Markdown final.

Esta solución aprovecha la capacidad integrada de PyMuPDF para OCR, evitando tener que guardar imágenes temporales manualmente o usar herramientas externas. También mantiene el formato: la capa de texto OCR añadida está posicionada exactamente donde corresponde cada palabra en la página, así PyMuPDF4LLM puede reconstruir incluso columnas de texto OCR respetando la estructura original.

Por último, integrar este adaptador en la aplicación es sencillo. Por ejemplo, podrías inyectarlo en tu caso de uso de procesamiento LLM:

```python
# Ejemplo de uso del adaptador en el caso de uso RAG/LLM
process_llm = ProcessDocumentForLLM(
    ocr=PyMuPDFAdapter(),
    storage=FileStorage(output_dir),
    # otros adaptadores como table_extractor podrían no ser necesarios, PyMuPDF ya maneja tablas
)
process_llm.execute("ruta/al/documento.pdf")
```

De esta forma, reemplazamos en el contexto LLM el uso previo de OCR + PdfPlumber por esta solución unificada. PyMuPDF4LLM entregará un documento.md con texto y tablas, preservando estilos, listo para la fase de Retrieval-Augmented Generation.

## Próximos Pasos: Integración con RAG

Con el contenido de los PDFs convertido a Markdown estructurado, el siguiente paso típico es indexar y usar ese contenido para consultas inteligentes (RAG). Gracias a que el texto está organizado con contexto (por ejemplo, Markdown conserva la jerarquía de secciones y el formato de tablas/código), se puede segmentar en fragmentos semánticamente coherentes. Por ejemplo, usando LangChain podrías emplear un divisor de texto especializado para Markdown, como `MarkdownTextSplitter`, para fragmentar el contenido por secciones o por tamaño de token. Cada fragmento (chunk) se transforma en un embedding vectorial (usando, digamos, OpenAI o SentenceTransformers), y se almacena en un índice vectorial (como FAISS). Luego, las preguntas del usuario se atienden buscando los chunks más relevantes y alimentándolos al LLM para generar respuestas fundamentadas en el texto real del PDF.

Vale la pena mencionar que PyMuPDF4LLM incluso ofrece una integración directa con LlamaIndex a través de `LlamaMarkdownReader`, que te permite obtener objetos Document propios de LlamaIndex directamente desde un PDF. No obstante, implementar tu propio flujo RAG con el Markdown obtenido te da más control.

En conclusión, PyMuPDF4LLM simplifica enormemente la fase de extracción de conocimiento de PDFs. Hemos cubierto cómo usarlo para extraer texto, tablas, imágenes y metadatos, incluso manejando documentos escaneados. Con esta base, estás listo para construir la capa de RAG, indexando el Markdown y permitiendo consultas en lenguaje natural sobre tus documentos. ¡El diseño modular de tu proyecto (puertos y adaptadores) se beneficia de esta implementación limpia y extensible, y tu LLM podrá aprovechar información rica y estructurada proveniente de los PDFs!

---

Referencias: Las referencias citadas (p. ej. 【1】, 【2】, etc.) corresponden a fuentes conectadas que respaldan cada afirmación. Puedes hacer clic en cada referencia para ver el fragmento original del cual se obtuvo la información. ¡Espero que esta investigación te sea de ayuda para implementar PyMuPDF4LLM en tu proyecto!