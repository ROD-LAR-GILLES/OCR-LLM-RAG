Guía Técnica: Integración de PyMuPDF4LLM y RAG en Proyecto OCR-CLI

Introducción

En este informe se detalla cómo extender un proyecto OCR-CLI existente para incorporar un sistema de Retrieval-Augmented Generation (RAG), aprovechando la librería PyMuPDF4LLM (módulo pymupdf4llm). Se cubrirá el flujo completo: desde la extracción de texto/Markdown de PDFs (digitales y escaneados) hasta la indexación semántica con embeddings y la respuesta a consultas usando modelos LLM. Todo se integrará respetando los principios de arquitectura hexagonal (Clean Architecture), lo que permitirá evolucionar la aplicación de una CLI a una API web de forma mantenible.

¿Qué es PyMuPDF4LLM? Es una librería diseñada para facilitar la extracción de contenido de PDFs en formato Markdown, optimizado para entornos con LLM y RAG ￼. PyMuPDF4LLM se basa en PyMuPDF (también conocido como fitz) y agrega capacidades de conversión a Markdown enriquecido, con soporte para páginas multi-columna, tablas, imágenes y gráficos vectoriales incrustados en el texto Markdown ￼. Usar Markdown como formato base aporta beneficios en este contexto: preserva la estructura (encabezados, listas, tablas) y el estilo básico del documento, facilitando la comprensión para el modelo y permitiendo chunking más coherente por secciones ￼ ￼. En otras palabras, PyMuPDF4LLM nos permite obtener de un PDF un documento Markdown estructurado listo para alimentar a un pipeline de RAG ￼ ￼.

¿Qué implementaremos?
	1.	Extracción OCR + Markdown: Utilizaremos PyMuPDF4LLM para convertir PDFs a texto Markdown. En PDFs digitales, extraeremos texto, tablas e imágenes directamente. En PDFs escaneados (imágenes), aplicaremos OCR (Tesseract) para obtener el texto.
	2.	Indexación Semántica (RAG): Dividiremos el contenido Markdown en chunks manejables, generaremos embeddings vectoriales de cada fragmento y los almacenaremos en un índice vectorial para búsqueda semántica. Esto permite hacer consultas en lenguaje natural y recuperar los fragmentos relevantes.
	3.	LLM QA: Usaremos un modelo de lenguaje (inicialmente el API de OpenAI) para responder preguntas del usuario basadas en los documentos. El modelo recibirá como contexto los fragmentos recuperados (RAG) para generar respuestas con sustento factual.
	4.	Arquitectura Hexagonal: Diseñaremos puertos (interfaces) y adaptadores concretos para cada componente (OCR, extractor Markdown, embedder, LLM, etc.), de modo que podamos intercambiar fácilmente implementaciones – por ejemplo, cambiar de OpenAI a un LLM local con mínimas modificaciones.
	5.	CLI a API web: Mostraremos cómo la aplicación puede pasar de un modo CLI interactivo a exponer funcionalidades vía una API REST (usando FastAPI), simplemente añadiendo nuevos adaptadores (controladores web) que llamen a los casos de uso existentes.

Con este enfoque, obtendremos una solución modular y escalable: los componentes de OCR, procesamiento PDF, vector store y modelo de lenguaje estarán desacoplados, permitiendo evolucionar o reemplazar cada pieza (p. ej. usar SentenceTransformers en vez de OpenAI, o incorporar un modelo local) sin romper la arquitectura.

Instalación y Configuración en Docker

Para asegurar un entorno consistente, utilizaremos Docker como entorno de ejecución. A continuación, se detallan los requisitos y configuraciones necesarios en la imagen Docker:
	•	Base de imagen: Usar una imagen oficial de Python (ej. python:3.10-slim) para minimizar peso.
	•	Dependencias del sistema (APT):
	•	Tesseract-OCR: Imprescindible para OCR de imágenes escaneadas. Instalar paquetes tesseract-ocr y los idiomas necesarios (por ejemplo tesseract-ocr-spa para español, tesseract-ocr-eng para inglés). PyMuPDF puede integrarse con Tesseract si este está instalado ￼.
	•	Poppler o equivalente: Si se utilizan herramientas como pdf2image o OCRmyPDF, instalar poppler-utils (provee pdftoppm y pdftocairo para convertir PDF a imágenes).
	•	Libraries para PyMuPDF/fitz: No requieren instalacíon extra más allá de las proporcionadas por pip, pero es recomendable instalar libglib2.0-0 y otros que PyMuPDF pueda usar.
	•	OpenCV (opcional): Si se usa preprocesamiento de imágenes (como en la ampliación con OpenCV), instalar las librerías de sistema necesarias (e.g. libopencv-dev y dependencias listadas en la documentación) ￼.
	•	Dependencias Python (pip):
	•	PyMuPDF4LLM: Se instala vía pip e incluirá PyMuPDF como dependencia si no está presente ￼ ￼.
	•	pytesseract: Wrapper de Tesseract para Python (si integra OCR manualmente).
	•	pdfplumber/pdf2image: Si aún usaremos pdfplumber para tablas o pdf2image para imágenes, incluirlos en requirements.txt. Note que PyMuPDF4LLM ya extrae tablas en Markdown, así que pdfplumber podría no ser necesario en adelante.
	•	OpenAI SDK: openai para hacer llamadas al API de OpenAI (embeddings y chat completions).
	•	SentenceTransformers: Para posibles embeddings locales.
	•	Faiss u otro vector store: faiss-cpu para realizar búsqueda vectorial local en memoria. Alternativamente, Chroma (chromadb) o Milvus (requiere contenedor aparte) podrían usarse. En este proyecto usaremos Faiss para mantener todo en Python local.
	•	FastAPI (y uvicorn): Si se planea exponer una API REST.

Ejemplo de Dockerfile:

FROM python:3.10-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng \ 
    poppler-utils \
    libgtk-3-0 libgl1-mesa-glx libglib2.0-0 \ 
    && rm -rf /var/lib/apt/lists/*

# Copiar requerimientos e instalarlos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . /app
WORKDIR /app

# Comando por defecto (modo CLI)
CMD ["python", "main.py"]

En el requirements.txt incluiríamos por ejemplo:

pymupdf4llm==1.0.0    # versión hipotética
openai==0.27.0
faiss-cpu==1.7.4
fastapi==0.95.0
uvicorn==0.22.0
pytesseract==0.3.10
sentence-transformers==2.2.2

(Las versiones son de ejemplo; se recomienda usar versiones actuales en 2025).

Este Dockerfile instala Tesseract y poppler para OCR, luego las librerías Python necesarias. Se asume que main.py lanza la interfaz CLI; más adelante mostraremos cómo modificar para ejecutar un servidor API.

Configuraciones adicionales:
	•	Variables de entorno: Para OpenAI, exportar OPENAI_API_KEY con la clave de API en tiempo de ejecución (por seguridad, manejarlo fuera del Dockerfile, por ejemplo pasando -e OPENAI_API_KEY=... al ejecutar el contenedor).
	•	Montaje de volúmenes: En la CLI actual, se mapean probablemente volúmenes (p. ej. la carpeta local de PDFs montada en /pdfs en el contenedor) ￼. Esto debe mantenerse en la versión Dockerizada.
	•	Idioma OCR: Asegurarse de que la configuración de idioma que se pase a Tesseract (por ejemplo lang="spa+eng") corresponda a los paquetes instalados en el sistema.
	•	OpenCV (opcional): Si se utilizan funciones de OpenCV para preprocesamiento de imágenes antes del OCR, recordar incluir las dependencias de sistema necesarias ￼. En el ejemplo anterior se incluyeron libgtk-3-0 y similares para evitar errores al usar OpenCV.

Uso de PyMuPDF4LLM desde Cero

Con el entorno listo, veamos cómo emplear PyMuPDF4LLM en nuestro proyecto:

Instalación: Ya cubierto vía pip (pip install pymupdf4llm). Esto instalará PyMuPDF4LLM y PyMuPDF (llamado fitz). Verificamos la instalación importando el módulo:

import pymupdf4llm

Extracción de PDF a Markdown: La librería ofrece una interfaz muy sencilla. Para convertir un PDF en Markdown:

md_text = pymupdf4llm.to_markdown("archivo.pdf")

En una sola línea obtenemos todo el contenido del PDF en la cadena md_text, con formato Markdown ￼. Este proceso:
	•	Detecta texto estándar y tablas, manteniendo el flujo de lectura correcto y convirtiéndolos a Markdown (por ejemplo, tablas a formato pipe tables de GitHub) ￼ ￼.
	•	Identifica títulos a partir del tamaño de fuente, prefixándolos con # (Markdown headers) según su nivel ￼.
	•	Formatea texto en negrita, itálica, código monoespaciado, listas ordenadas y viñetas si corresponde ￼.
	•	Incluye referencias a imágenes y gráficos: si hay imágenes escaneadas o gráficos vectoriales, en lugar de texto (que no puede leer si son imágenes puras), PyMuPDF4LLM las inserta como referencias en el Markdown (por ejemplo ![Imagen1](image1.png)), facilitando el contexto para un LLM ￼.

Selección de páginas (opcional): Podemos extraer solo un subconjunto de páginas pasando una lista de índices de página (base 0) como segundo parámetro a to_markdown. Ejemplo:

md_text = pymupdf4llm.to_markdown("archivo.pdf", pages=[0,1,2])

Esto extraería solo las páginas 1, 2 y 3 del PDF. Si no se especifica, procesará todas ￼.

Guardar resultado: El texto Markdown puede guardarse en un archivo si se desea:

from pathlib import Path
Path("salida.md").write_text(md_text, encoding="utf-8")

(En la documentación sugieren usar write_bytes con encode(), pero write_text simplifica esta operación.)

Ejemplo rápido:
Supongamos un PDF report.pdf con dos columnas de texto, títulos y tablas. Con PyMuPDF4LLM:

import pymupdf4llm
md = pymupdf4llm.to_markdown("report.pdf")
print(md[:500])

La salida (truncada) podría verse así:

# Informe de Ventas 2024 Q1

El presente informe resume los resultados del primer trimestre. La **compañía ABC** muestra un crecimiento...

## Ventas por Región

| Región      | Ventas Q1    | Crecimiento |
| ----------- | ------------ | ----------- |
| Norteamérica| $1,200,000   | 5%          |
| Europa      | $950,000     | 7%          |
| Asia        | $1,500,000   | 10%         |

*Figura 1: Tendencia de ventas mensuales.* ![Grafico1](image0.png)

...

Como se observa, PyMuPDF4LLM detectó los encabezados # y ## según tamaños de fuente, convirtió la tabla en formato Markdown y dejó una referencia a una imagen (Figura 1) ￼ ￼. Esto facilita que un LLM entienda la estructura (sabe qué es un título, qué es una tabla, etc.) y acota el tamaño del prompt (el Markdown es conciso; por ejemplo, las tablas se representan en pocas líneas ￼).

Integración con LlamaIndex/LangChain: Cabe destacar que PyMuPDF4LLM ofrece también utilidades para integrarse con frameworks de RAG: por ejemplo, un loader para LlamaIndex llamado LlamaMarkdownReader que devuelve objetos Document listos para indexar ￼. También existe un loader comunitario para LangChain (PyMuPDFLoader) que internamente usa PyMuPDF para extraer datos ￼. En este proyecto mostraremos la integración manualmente (para conservar la arquitectura hexagonal propia), pero es bueno saber que estas opciones existen.

Procesamiento OCR: PDFs Escaneados vs. Digitales

No todos los PDFs son iguales. PDFs digitales (generados por exportación de Word, LaTeX, etc.) contienen texto seleccionable; PyMuPDF4LLM extraerá directamente ese contenido. PDFs escaneados son esencialmente imágenes incrustadas – no contienen texto, por lo que necesitamos aplicar OCR para obtenerlo. Nuestro flujo debe manejar ambas situaciones:

1. Detección del tipo de PDF:
Podemos implementar una función que analice una página del PDF para inferir si contiene texto: por ejemplo, usando PyMuPDF:

import fitz  # PyMuPDF
doc = fitz.open("file.pdf")
page = doc[0]
text = page.get_text().strip()
if text == "": 
    print("PDF escaneado (sin texto extraíble)")
else:
    print("PDF digital (texto presente)")

PyMuPDF devuelve una cadena vacía si no encuentra texto en la página. Otra heurística: comprobar si la proporción de objetos imagen en la página es alta y de texto es cero. Una tercera forma es usar la herramienta pdfinfo (poppler) para ver metadatos, pero la extracción directa es suficiente.

2. Caso PDF digital: Camino directo. Se utiliza PyMuPDF4LLM como se describió: md_text = pymupdf4llm.to_markdown(pdf_path). Esto nos dará todo el texto en Markdown, incluyendo tablas e imágenes referenciadas. No se requiere OCR pues el texto ya está embebido.

3. Caso PDF escaneado: Camino con OCR. PyMuPDF4LLM por sí solo no realiza OCR automático sobre imágenes (requiere que PyMuPDF tenga un texto que extraer). Hay varias opciones para incorporar OCR:
	•	Integración PyMuPDF + Tesseract: PyMuPDF (desde la v1.21) soporta OCR integrado con Tesseract si este está instalado ￼. Se puede usar page.get_textpage_ocr() para obtener un objeto de texto OCR de la página y luego extraer texto con page.get_text("text", textpage=my_ocr_textpage). Una estrategia es recorrer páginas, decidir si necesitan OCR (ej. no tienen texto) ￼, aplicar get_textpage_ocr() y luego usar ese resultado en to_markdown o en la extracción. Sin embargo, PyMuPDF4LLM actualmente no expone directamente un parámetro para OCR; tendríamos que hacer un proceso híbrido manual: generar un PDF con capa OCR o interceptar la extracción por página.
	•	OCRmyPDF (opción externa): Una solución práctica es usar la herramienta CLI OCRmyPDF para agregar una capa de texto oculto al PDF antes de procesarlo. OCRmyPDF usa Tesseract internamente para reconocer texto en cada página y crea un PDF searchable (el texto se agrega pero las imágenes originales permanecen). PyMuPDF4LLM podría entonces tratarlo como PDF digital y extraer Markdown. La ventaja es que OCRmyPDF mantiene el layout original – por ejemplo, si hay columnas, tras OCR PyMuPDF4LLM podría detectarlas correctamente. El costo es tener que llamar a un comando externo (se puede hacer con subprocess dentro de Docker).
	•	Pipeline personalizado (pdf2image + pytesseract): Dado que el proyecto ya implementaba OCR así, podemos continuar con esa vía dentro de un adaptador OCR. Es decir: convertir cada página en imagen (usando pdf2image o PyMuPDF pixmaps), aplicar pytesseract.image_to_string, y recolectar el texto. Este texto se podría luego formatear rudimentariamente en Markdown (por ejemplo, mantener saltos de línea, quizá identificar números de página). Sin embargo, formatear tablas o listas manualmente a Markdown desde OCR es complejo – se puede optar por extraer solo texto plano para PDFs escaneados.

Implementación recomendada: Para este proyecto, podemos seguir dos caminos según la complejidad:
	•	Camino simple: Utilizar OCRmyPDF para obtener un PDF con texto, luego pasar ese PDF a PyMuPDF4LLM. Esto nos da Markdown con estructura básica. La desventaja es que Tesseract no preserva estilos (todo será texto plano en el Markdown, sin bold/italic porque el OCR no los detecta). Aun así, conservaría la segmentación por párrafos.
	•	Camino controlado: Integrar la lógica de OCR en nuestro adaptador hexagonal. Ejemplo: crear un OcrAdapter que detecta si el PDF es escaneado. Si sí, realiza OCR (páginas a imagen, etc.) y devuelve un string con texto; si no, puede delegar a PyMuPDF4LLM directamente. Luego, tomar ese string y – opcionalmente – pasarlo por un formateador a Markdown. Aquí podríamos usar PyMuPDF4LLM parcialmente: por ejemplo, obtener las imágenes referenciadas y tablas detectadas si hubiera (difícil si es escaneado). Probablemente para escaneados nos contentemos con texto continuo.

En aras de aprovechar PyMuPDF4LLM, podemos hacer: si PDF es digital -> to_markdown completo; si es escaneado -> to_markdown igualmente, ya que PyMuPDF4LLM incluirá las referencias de imagen pero sin texto. Luego, por cada imagen referenciada en el Markdown, ejecutar OCR para ese área. Este enfoque es avanzado (requiere mapear coordenadas de imagen a texto), por lo que recomendamos el método OCR previo.

Conservando estructura básica: Una idea útil: si se conoce el documento (por ejemplo, que tiene columnas o secciones claras), se podría dividir la imagen en partes antes de OCR para imitar columnas y luego unir el texto. Sin embargo, esto puede ser overkill. Para MVP, extraeremos texto linear.

Resumen:
	•	Digital PDF: md_text = pymupdf4llm.to_markdown(pdf) directo.
	•	Scanned PDF:
	1.	Aplicar OCR a cada página (vía OCRmyPDF o pytesseract manualmente) y crear un PDF o texto resultante.
	2.	Si se obtuvo un PDF con capa OCR, usar PyMuPDF4LLM para Markdown. Si solo se obtuvo texto, podríamos insertarlo en un template Markdown básico (por ejemplo, cada página separada por un encabezado “Página X”).

En la arquitectura hexagonal, esto encaja así: podemos tener un Puerto OCR con un método extract_text(pdf_path) -> str que nuestras implementaciones TesseractOCRAdapter y NoOcrNeededAdapter implementen. En el caso digital, NoOcrNeededAdapter podría simplemente leer el PDF con fitz y devolver texto (o Markdown). Pero dado que PyMuPDF4LLM ya da Markdown completo, quizás reorganizamos: un Puerto ExtractorPDF que devuelva Markdown, con implementaciones PyMuPDFExtractor (digital) y OcrPyMuPDFExtractor (que combina OCR + PyMuPDF). Para simplificar: mantendremos un adaptador único que internamente haga la bifurcación.

Ejemplo de uso adaptador OCR (pseudo-código dentro de caso de uso):

pdf = "documento.pdf"
if is_scanned(pdf):
    ocr_text = tesseract_ocr_adapter.extract_text(pdf)
    md_text = plain_text_to_markdown(ocr_text)  # quizás formatear mínimamente
else:
    md_text = pymupdf4llm.to_markdown(pdf)

Luego md_text es el output unificado para ambos casos. La función plain_text_to_markdown podría simplemente envolver cada párrafo en una línea en blanco (para que Markdown los separe) y conservar listas si OCR detectó “- “ u “1.” al inicio de línea (Tesseract a veces mantiene viñetas como “*” o “-”). Esto no reconstruirá tablas ni formatos, pero es aceptable para tener contenido consultable.

Nota: En la práctica, los embeddings capturarán el significado del texto OCR aunque esté sin formato. El formato Markdown es más útil en documentos digitales donde aporta estructura; en documentos escaneados, el valor clave es obtener el texto bruto correctamente. Siempre podemos actualizar más adelante para detectar tablas en imágenes usando técnicas avanzadas (ej. visión computacional) y formatearlas, pero quedaría fuera del alcance inicial.

Indexación Semántica con Embeddings (RAG)

Con los documentos convertidos a texto/Markdown, el siguiente paso es habilitar la búsqueda semántica para RAG. Esto implica generar representaciones vectoriales (embeddings) de los textos y almacenarlas para su posterior recuperación mediante similitud de coseno. Detallaremos cómo indexar los documentos y luego cómo usarlos en consultas.

1. Segmentación (Chunking) del contenido:
Los LLM tienen un límite de contexto, por lo que no podemos darles todo el documento completo si es muy grande. Necesitamos dividir el texto en fragmentos relevantes. Idealmente, cada chunk corresponde a una unidad de significado (un párrafo, sección o subsección) para no romper contexto a la mitad.
	•	Si usamos el contenido Markdown, una estrategia es dividir por secciones o encabezados: por ejemplo, se puede separar por nivel de encabezado (H2, H3, etc.) y limitar tamaño. Otra estrategia es un splitter por longitud de texto (p. ej. 2000 caracteres o 300 tokens) procurando no cortar oraciones.
	•	Podemos aprovechar herramientas existentes: LangChain ofrece MarkdownTextSplitter que entiende bien el formato Markdown para cortes naturales ￼. Por ejemplo:

from langchain.text_splitter import MarkdownTextSplitter
splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(md_text)

Esto crearía fragmentos de ~500 tokens con un solapamiento de 50 para contexto ￼. El solapamiento evita que información en el borde de un fragmento se pierda en la búsqueda.

	•	También es posible implementar un splitter propio: recorrer el Markdown, dividir en secciones por ’\n## ’ etc., luego subdividir si exceden cierto tamaño.

Supongamos que tenemos md_text de un documento. Obtenemos una lista de fragmentos: chunks = [chunk1_text, chunk2_text, ...]. Cada fragmento lleva consigo quizá meta-información (por ejemplo, de qué documento y página proviene, útil para referencias). Podríamos encapsularlo en un objeto DocumentoFragmento con campos: id_doc, texto, num_pagina_inicio, ....

2. Generación de embeddings:
Cada fragmento de texto se convierte en un vector numérico de alta dimensión. En este proyecto consideramos dos opciones: usar el API de OpenAI (modelo text-embedding-ada-002) o usar un modelo local (por ejemplo, SentenceTransformers). Veamos primero con OpenAI:
	•	OpenAI Embeddings: El modelo text-embedding-ada-002 produce vectores de 1536 dimensiones por cada entrada de texto ￼ ￼. Es actualmente económico y de alta calidad semántica. Para usarlo:

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
response = openai.Embedding.create(input=chunks, model="text-embedding-ada-002")
embeddings = [r["embedding"] for r in response["data"]]

Aquí enviamos una lista de textos (chunks) y obtenemos una lista de vectores (embeddings). Cada vector es una lista de 1536 floats ￼ representando el texto en el espacio semántico de Ada. OpenAI permite hasta 8191 tokens de entrada por fragmento ￼, por lo que fragmentos de unas ~4-5 páginas A4 suelen caber (pero preferimos fragmentos menores para precisión).

	•	SentenceTransformers (modelo local): Para no depender de la nube, podemos usar modelos preentrenados como all-MiniLM-L6-v2 (384 dimensiones) o modelos más grandes como multi-qa-mpnet-base (768 dims) o incluso InstructorXL. Por ejemplo con sentence-transformers:

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(chunks)  # devuelve una lista de vectores numpy

Esto genera embeddings localmente. La calidad puede ser un poco menor que OpenAI Ada, pero mejora la privacidad y evita costos. Existen modelos multilingües si se requiere soporte en español específico.

En nuestra arquitectura hexagonal, definiremos un Puerto de Embedder (por ejemplo EmbedderPort con método embed(texts: List[str]) -> List[List[float]]) ￼. Tendremos dos adaptadores: OpenAIEmbedder y LocalEmbedder. El OpenAIEmbedder usará la API como se mostró (quizá loteando si hay muchos textos para respetar límites de tasa), y LocalEmbedder usará un modelo de SentenceTransformers cargado en inicialización. Ambas implementaciones devuelven listas de vectores flotantes de longitud fija (1536 en OpenAI ￼, dependiendo del modelo en local). Gracias a depender de la abstracción, cambiar de uno a otro es sencillo (Inversión de Dependencias) ￼.

Nota: El ejemplo en la documentación del proyecto muestra un OpenAIEmbedder ya planificado, configurando text-embedding-ada-002 con batch_size y timeout ￼, siguiendo exactamente este enfoque.

3. Almacenamiento de embeddings (Vector Store):
Para encontrar luego los documentos relevantes a una pregunta, usaremos un índice vectorial que permita búsqueda por similitud de coseno o distancia euclídea. Opciones:
	•	Faiss (Facebook AI Similarity Search): Es una librería muy eficiente en C++ con bindings Python, ideal para cientos de miles de vectores en RAM. Para pocos documentos, incluso una simple búsqueda lineal podría bastar, pero Faiss facilita escalabilidad.
Ejemplo:

import faiss
import numpy as np
dim = len(embeddings[0])  # 1536 por ejemplo
index = faiss.IndexFlatIP(dim)  # índice de producto interno (cosine similarity si vectores normalizados)
# Normalizar vectores para usar dot = cosine
xb = np.vstack([np.array(v)/np.linalg.norm(v) for v in embeddings]).astype('float32')
index.add(xb)

Aquí IndexFlatIP es un índice brute-force de similitud de coseno (inner product) de dimensión dada. Insertamos todos los embeddings. Podemos guardar también un mapeo de índice -> referencia del fragmento. Por ejemplo, mantener una lista paralela fragments = [...] donde fragments[i] corresponde al vector insertado xb[i]. Para persistencia sencilla, Faiss permite guardar a disco el índice, o podríamos recalcular embeddings en cada arranque (posible si no son muchos docs).

	•	Otras opciones: ChromaDB es una base de datos vectorial ligera en Python que soporta almacenamiento persistente (podría ser útil si queremos guardar metadata y embeddings sin mucho esfuerzo). Milvus es otra opción robusta (necesitaría su propio servicio). Dado el tamaño acotado de nuestro proyecto, Faiss es suficiente.

4. Proceso de consulta (RAG): Con el índice construido, cuando el usuario haga una pregunta, haremos:
a. Embeddear la pregunta: convertir la query del usuario en un vector usando el mismo método de embedding (OpenAI o local) ￼.
b. Buscar los k fragmentos más similares: usar el índice para obtener, por ejemplo, los top-5 fragmentos cuyo embedding tiene mayor similitud con la pregunta. Ejemplo con Faiss:

q = "¿Cuál fue el crecimiento de ventas en Europa?"
q_emb = embedder.embed([q])[0]  # obtener vector 1536-dim de la pregunta
q_vec = np.array(q_emb, dtype='float32')
faiss.normalize_L2(q_vec.reshape(1, -1))  # normalizar si usamos IP para coseno
D, I = index.search(q_vec.reshape(1, -1), k=5)
top_indices = I[0]  # indices de los 5 mejores matches
top_chunks = [fragments[i] for i in top_indices]

Aquí top_chunks sería la lista de textos más relevantes. Idealmente, concatenamos o elegimos los más pertinentes (podríamos filtrar por score D también).

c. Construir el prompt para el LLM: Hay varias estrategias. Una común es crear un mensaje del sistema que incluya los textos recuperados como contexto, luego la pregunta del usuario. Por ejemplo:

system_msg = ("Eres un asistente que responde preguntas basándose en los documentos proporcionados. "
              "Utiliza únicamente la información de los documentos para responder. "
              "Documentos:\n" + "\n---\n".join(top_chunks))
user_msg = pregunta_usuario

Y enviar esos mensajes al modelo. Alternativamente, formar una sola pregunta grande tipo: “Contexto: {doc1} {doc2} … Pregunta: {query}”.

Con OpenAI ChatCompletion:

messages = [
    {"role": "system", "content": system_msg},
    {"role": "user", "content": user_msg}
]
completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
answer = completion["choices"][0]["message"]["content"]

Con un modelo local, se usaría su propia interfaz (ver siguiente sección).

d. Post-proceso de la respuesta: Podemos incluir en la respuesta final referencias a las fuentes. Por ejemplo, podríamos enumerar los documentos o fragmentos usados (si llevamos identificadores de origen, como nombre de archivo o página, podemos adjuntarlos). Esto ayuda al usuario a saber de dónde salió la información (¡importante para confianza!). Este paso se puede hacer formateando la respuesta con citas estilo Markdown al final de la frase correspondiente, o listando las fuentes al final de la respuesta. Implementar esto manualmente requiere mapear fragmento -> documento original (lo cual podemos trackear). LangChain/LlamaIndex lo hacen automáticamente; en nuestra integración manual debemos hacerlo deliberadamente.

Ejemplo de flujo RAG completo:
	1.	Indexación (una vez): Supongamos tenemos 10 PDFs procesados a Markdown. Obtenemos en total 100 fragmentos tras chunking. Generamos embeddings (OpenAI) para los 100 fragmentos y los guardamos en Faiss.
	2.	Consulta del usuario: “Resumen de las ventas en Asia y Europa en Q1”.
	3.	Embedder convierte pregunta a vector. Faiss busca top-5 y retorna fragmentos de los documentos “Informe Q1” y “Reporte Ventas Globales”.
	4.	El prompt al LLM incluye esos fragmentos (que mencionan cifras de Asia y Europa).
	5.	El LLM (ChatGPT) responde con un párrafo comparando ventas en Asia vs Europa, citando los documentos: e.g. “Las ventas en Europa fueron de $950k con crecimiento 7%, mientras en Asia alcanzaron $1.5M con 10% de crecimiento ￼.” (Aquí la cita es ilustrativa de una fuente).

De esta forma, el usuario obtiene una respuesta precisa y con soporte documental, y el LLM se limita a no alucinar porque se circunscribe al contexto proporcionado ￼.

Arquitectura Hexagonal: Puertos y Adaptadores para RAG

Para integrar lo anterior en nuestro proyecto manteniendo Clean Architecture, identificaremos los nuevos puertos (interfaces) necesarios y sus adaptadores (implementaciones). La idea es seguir el mismo patrón que ya existe para OCR, extracción de tablas, almacenamiento, etc., de modo que nuestros casos de uso orquesten componentes abstractos sin conocer detalles de API externas.

Nuevos componentes lógicos que introduciremos:
	•	ExtractorPort: interfaz para extraer contenido de un PDF. Ya tenemos algo similar (OCRPort + TableExtractorPort combinados anteriormente). Sin embargo, con PyMuPDF4LLM podemos consolidar en un solo paso la extracción de texto y tablas a Markdown, así que podríamos definir un puerto DocumentExtractorPort con, por ejemplo, extract_markdown(pdf: Path) -> str. Implementaciones:
	•	PyMuPDFExtractorAdapter – usa pymupdf4llm.to_markdown directamente (asume PDF digital).
	•	OCRPyMuPDFExtractorAdapter – combina OCR si es necesario, como discutido. Podría heredar de la anterior o envolverla.
	•	Alternativamente, seguir con puertos separados (OCRPort, TablePort) y que el caso de uso los llame secuencialmente, luego unifique resultados. Pero aprovecharemos la potencia de PyMuPDF4LLM que ya extrae tablas e imágenes junto al texto ￼.
	•	EmbedderPort: interfaz embed(texts: List[str]) -> List[List[float]] ￼. Implementaciones:
	•	OpenAIEmbedder – llama a OpenAI API ￼.
	•	LocalEmbedder – usa SentenceTransformers localmente.
	•	(Podríamos have embed_one(text: str) -> List[float] también para conveniencia).
	•	VectorIndexPort: interfaz para la base de conocimiento vectorial, con métodos como index_documents(docs: List[DocumentFragment]) y query(vector: List[float], k: int) -> List[DocumentFragment]. Implementaciones:
	•	FaissIndex – internamente manejará el índice Faiss y los mappings.
	•	InMemoryIndex – incluso se podría hacer una implementación toy que calcula distancias en Python puro (para testing).
	•	En un inicio, podríamos simplificar y manejar el índice en el propio caso de uso, pero separarlo en un puerto facilita cambiar a otro backend (ej. llamar a Pinecone u otro servicio remoto en otra implementación).
	•	LLMPort: interfaz para el modelo generador que formará la respuesta final. Método ask(question: str, context: str) -> str o más genérico generate(prompt: str) -> str. Aquí la implementación puede ser:
	•	OpenAIChatCompletionAdapter – que formatea los messages y llama a openai.ChatCompletion.create como vimos.
	•	LocalLLMAdapter – que use por ejemplo HuggingFace Transformers o un wrapper de un modelo local (podría ser through transformers pipeline, o an API local like Ollama as in some stacks ￼, or llama.cpp Python binding).
	•	También se podría subdividir: un puerto para formateo de prompt (pero eso es más parte del caso de uso lógico), y otro para la llamada al modelo. En principio, LLMPort puede recibir ya el contexto listo y pregunta.

Además, podemos reutilizar/expandir puertos existentes:
	•	StoragePort: para guardar resultados (si queremos persistir el Markdown extraído o el índice).
	•	Logger/Monitor (opcional): para registrar operaciones, tiempo, costos, etc.

Diseño de casos de uso:
Podemos añadir un caso de uso principal nuevo: por ejemplo AnswerQueryUseCase o ChatUseCase, encargado de orquestar:
	1.	Dado un query del usuario, usar EmbedderPort para vectorizar la pregunta.
	2.	Llamar a VectorIndexPort.query con ese vector para obtener documentos relevantes.
	3.	Compilar el contexto a partir de esos docs (por ejemplo concatenar fragmentos, con quizás recorte si demasiado extenso).
	4.	Invocar LLMPort.ask con la pregunta y el contexto.
	5.	Devolver la respuesta (posiblemente enriquecida con referencias; la lógica de insertar referencias podría estar aquí usando metadatos de los fragmentos recuperados).

Este caso de uso no necesita saber cómo se hacen estos pasos, solo conoce las interfaces. Por ejemplo:

class AnswerQueryUseCase:
    def __init__(self, embedder: EmbedderPort, vector_index: VectorIndexPort, llm: LLMPort):
        self.embedder = embedder
        self.vector_index = vector_index
        self.llm = llm

    def __call__(self, query: str) -> str:
        q_vec = self.embedder.embed([query])[0]
        fragments = self.vector_index.query(q_vec, k=5)
        context = "\n".join([f.content for f in fragments])
        prompt = f"{context}\nPregunta: {query}"
        answer = self.llm.ask(prompt)
        # (Opcional: anexar referencias de fragments a 'answer')
        return answer

(Aquí asumimos que DocumentFragment.content tiene el texto, y probablemente haya fragment.source_info para citas.)

Otro caso de uso existente, ProcessDocumentUseCase (que probablemente ya exista para el OCR-CLI), se modificaría para usar DocumentExtractorPort en lugar de llamar separadamente a OCRPort y TablePort, ya que PyMuPDF4LLM combina eso. Por ejemplo:

class ProcessDocumentUseCase:
    def __init__(self, extractor: DocumentExtractorPort, storage: StoragePort):
        ...
    def __call__(self, pdf_path: Path) -> None:
        md = self.extractor.extract_markdown(pdf_path)
        self.storage.save_markdown(pdf_path.stem + ".md", md)
        # También podríamos indexarlo aquí directamente:
        fragments = splitter.split(md)
        vecs = embedder.embed(fragments)
        vector_index.add(vecs, fragments)

Aunque es debatible: podríamos separar la indexación a otro paso (quizá un caso de uso IndexDocumentsUseCase que se llame después de procesar todos los PDFs, o integrado en ProcessDocumentUseCase para indexar on-the-fly). Depende si queremos procesar e indexar en el mismo paso interactivo. Si la CLI actual procesaba PDF a texto/Markdown y terminaba, quizás ahora convenga después de procesar preguntar “¿Indexar este documento para preguntas?”. De cualquier modo, la arquitectura nos permite eso.

Inversión de dependencias: En la capa de infraestructura definiremos las clases concretas: PyMuPDFExtractorAdapter, OpenAIEmbedderAdapter, FaissIndexAdapter, OpenAIChatLLMAdapter, etc. Todas implementan sus respectivos puertos. La capa de aplicación (casos de uso) depende de puertos, no de implementaciones. Las implementaciones se inyectan probablemente desde la capa de interfaz (por ejemplo, en main.py o menu.py, se crean los objetos concretos pasando configuraciones). Esto ya se hace con OCRPort y TablePort en el proyecto actual ￼, y seguiremos el mismo patrón. Por ejemplo:

# Configuración de adaptadores
extractor = OCRPyMuPDFExtractorAdapter(ocr_lang="spa+eng")
embedder = OpenAIEmbedderAdapter(api_key=os.getenv("OPENAI_API_KEY"))
vector_index = FaissIndexAdapter(dim=1536)
llm = OpenAIChatAdapter(model="gpt-3.5-turbo")

# Inyección en caso de uso
ask_use_case = AnswerQueryUseCase(embedder, vector_index, llm)

Si mañana queremos usar un modelo local:

embedder = LocalEmbedderAdapter(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = LocalLLMAdapter(model="TheBloke/Llama-2-7B-GGML")  # ejemplo de modelo local

y nada más cambia en los casos de uso. Esto ejemplifica el principio DIP (Dep. Inversion): la lógica central no cambia al cambiar implementaciones, solo se sustituye la inyección ￼ ￼. Asimismo, el OCP (Open/Closed): estamos añadiendo adaptadores nuevos sin modificar el código existente de OCR, etc., que ya funciona ￼.

Organización por capas/carpetas: Manteniendo consistencia con el proyecto, podríamos estructurar así:

project/
├── domain/
│   └── ports.py         # Añadir DocumentExtractorPort, EmbedderPort, VectorIndexPort, LLMPort
├── application/
│   ├── use_cases.py     # Añadir AnswerQueryUseCase, o dividir en multiple files (process_document.py, answer_query.py)
│   └── models.py        # Podría definir DocumentFragment (con campos content, source_doc, page, etc.)
├── infrastructure/
│   ├── adapters/
│   │   ├── ocr_tesseract.py        # existente
│   │   ├── storage_filesystem.py   # existente
│   │   ├── extractor_pymupdf.py    # nuevo adapter usando PyMuPDF4LLM
│   │   ├── embed_openai.py         # adapter OpenAIEmbedder
│   │   ├── embed_local.py          # adapter SentenceTransformer
│   │   ├── vector_faiss.py         # adapter FaissIndex
│   │   ├── llm_openai.py           # adapter OpenAI Chat
│   │   └── llm_local.py            # adapter local LLM (si se implementa)
│   └── ... (otros adaptadores)
├── interface/
│   ├── cli.py           # CLI interface (menu.py en original)
│   └── api.py           # API interface (FastAPI endpoints)
└── main.py              # Orquestación: detecta modo CLI o API, crea adaptadores y lanza la interfaz correspondiente

Esta es solo una sugerencia. Lo importante es separar claramente las capas. Por ejemplo, vector_faiss.py en infraestructura manejará la dependencia a la librería Faiss. Si quisiéramos cambiar a Pinecone (un servicio cloud), haríamos vector_pinecone.py sin tocar la lógica de aplicación.

De CLI a API Web: Evolución con Clean Architecture

La transición de una aplicación de consola a una API web puede lograrse sin refactorizaciones masivas si seguimos Clean Architecture. En esencia, la capa de casos de uso sigue igual; solo añadimos una nueva forma de interactuar con ella: antes era un menú CLI, ahora será vía peticiones HTTP.

Framework sugerido: FastAPI por su facilidad con Python y pydantic, y porque permite exponer rápidamente endpoints async. En la documentación del proyecto ya se consideraba usar FastAPI ￼, lo que encaja perfectamente.

Organización: Podemos crear un módulo api.py o un paquete api/ dentro de interface. Allí definiremos la aplicación FastAPI. Podría estructurarse con routers por funcionalidad (ej. router_docs para endpoints de procesar documentos, router_query para endpoints de preguntas).

Inicialización de la app: En main.py, podríamos detectar vía variable de entorno o argumento si lanzar CLI o API. Por ejemplo, si MODE=api, entonces:

import api
api.run()  # esto iniciaría uvicorn programáticamente o similar

o simplemente tener un uvicorn configurado para apuntar a api:app.

Inyección de dependencias en API: En vez de variables globales, podemos usar Dependencias de FastAPI para proveer instancias de casos de uso a los endpoints. Sin embargo, dado que nuestros adaptadores no son request-scoped (son singleton essentially), podemos inicializarlos una vez. Por simplicidad:

# api.py
from fastapi import FastAPI, UploadFile, File, Depends, BackgroundTasks
app = FastAPI()

# Crear adaptadores y casos de uso globales (similar a main CLI)
extractor = OCRPyMuPDFExtractorAdapter(...)
storage = FileStorageAdapter(output_dir="outputs")
process_use_case = ProcessDocumentUseCase(extractor, storage)
ask_use_case = AnswerQueryUseCase(embedder, vector_index, llm)

@app.post("/process", summary="Procesar PDF a Markdown")
async def process_pdf(file: UploadFile = File(...)):
    # Guardar archivo subido temporalmente
    pdf_path = Path("/tmp") / file.filename
    with pdf_path.open("wb") as f:
        f.write(await file.read())
    # Procesar documento (OCR + markdown)
    process_use_case(pdf_path)
    return {"detail": "Documento procesado", "output": file.filename + ".md"}

@app.get("/ask")
def ask_question(q: str):
    answer = ask_use_case(q)
    return {"question": q, "answer": answer}

Esto es un esquema básico. Posiblemente, para manejar documentos grandes, conviene procesar en background (por eso incluimos BackgroundTasks en la firma en la documentación ￼). FastAPI permite delegar una tarea larga a un background task para no bloquear la respuesta. En el ejemplo de la documentación, suben el PDF y retornan inmediatamente un task_id ￼. Podríamos implementar eso: al subir PDF, lanzar BackgroundTasks que ejecuten process_use_case, y devolver un identificador para luego recoger el resultado. Dado el alcance de esta guía, dejamos esa posibilidad mencionada, pero se puede implementar fácilmente usando un simple dict global de estado o una librería de tasks.

CORS, Auth, etc.: Como es un proyecto interno, no entraremos en detalles, pero si es necesario exponer la API públicamente, configurar CORS (FastAPI middleware) y algún mecanismo de autenticación (tokens, etc.).

Reuso de lógica: Observemos que no duplicamos la lógica de procesamiento ni de respuesta en la API. Simplemente llamamos a los mismos casos de uso ya existentes. Esto confirma el beneficio de Clean Architecture: agregar una interfaz (puerto de entrada) nueva no afecta la lógica central; solo adaptamos la forma de invocar y de entregar resultados. Lo mismo ocurre con CLI: podríamos mantener ambas interfaces simultáneamente (por ejemplo, si se lanza la app con --api inicia FastAPI, sin argumentos lanza CLI).

Ejemplo de petición: Después de ejecutar la aplicación en modo API (por ejemplo uvicorn api:app), un usuario podría realizar:

POST /process  (con un PDF en el body form-data)

y luego:

GET /ask?q="¿Cuál es el resumen del documento X?"

y obtendría un JSON con la respuesta.

LLM en la Nube vs LLM Local: Consideraciones

Una decisión importante es si usar el backend LLM en la nube (OpenAI API) o un LLM local instalado en el servidor. Cada enfoque tiene ventajas y trade-offs que resumimos a continuación:

Criterio	OpenAI API (GPT-3.5/4)	LLM Local (ej: Llama 2 7B/13B, etc.)
Calidad del modelo	Muy alta (entrenado en masivo corpus, GPT-4 es SOTA) ￼.	Variable: modelos open-source han mejorado (Llama 2, Mistral, etc.), pero usualmente menor que GPT-4. Algunos afinados en español pueden rendir bien.
Costo	Pago por uso (e.g. ~$0.002/1K tokens en GPT-3.5). Puede sumar en grandes volúmenes ￼.	Una vez desplegado, inferencia local es “gratuita” (solo costo de hardware/energía). Conviene para uso intensivo continuo.
Latencia	Depende de Internet y carga de OpenAI. GPT-3.5 suele responder en 1-2 seg para consultas cortas. GPT-4 más lento.	Depende del hardware local. Con GPU adecuada, un modelo 7B puede responder en ~1-3 seg. En CPU puede ser mucho más lento. No hay overhead de red.
Escalabilidad	Altamente escalable (infra OpenAI se encarga). Rate limits aplican, pero se pueden aumentar con $$ o cuentas de empresa.	Limitada por tu hardware. Escalar = añadir más GPUs/servicios. Requiere arquitectura distribuida si muchas peticiones concurrentes.
Privacidad	Los datos de las preguntas y contexto van a servidores de OpenAI (aunque se comprometen a no entrenar con ellos por defecto). Puede ser inapropiado para info sensible.	100% privado/local: los datos nunca salen de tu entorno ￼. Ideal para documentos confidenciales.
Mantenimiento	Cero mantenimiento de modelo: OpenAI mejora modelos tras bambalinas.	Debes encargarte de descargar y actualizar modelos manualmente. Además, gestionar dependencias (CUDA, drivers, etc.).
Fine-tuning	OpenAI ofrece fine-tuning limitado (solo ciertos modelos como Ada, y con costo significativo; GPT-3.5 fine-tuning fue anunciado en 2023). Difícil ajustar a tu dominio.	Posible entrenar o afinar localmente si tienes datos (incluso técnicas como LoRA para Llama 2). Requiere know-how y recursos de cómputo.
Compatibilidad	Integración sencilla vía llamadas HTTP.	Requiere librerías (ej. Transformers, GGML libs) y posiblemente mucho VRAM/RAM. Puede necesitar int8/int4 quantization para caber en hardware más modesto.
Actualización	Nuevos modelos OpenAI se pueden usar cambiando un parámetro de modelo, sin infraestructura propia.	Cambiar de modelo = descargar nuevo y ajustarlo. Pero hay libertad de elegir cualquier modelo (incluso especializados por industria).

En muchos casos, una estrategia híbrida es recomendable: comenzar con OpenAI para tener algo funcionando rápido y con alta calidad, y en paralelo experimentar con un LLM local. Una vez que la funcionalidad esté madura o si se necesita offline, se puede cambiar. Gracias a la arquitectura de puertos y adaptadores, esto sería transparente para el sistema: bastará con inicializar el adaptador LLMPort distinto.

Por ejemplo, un buen candidato local actualmente es Mistral 7B o Llama-2 13B si se dispone de GPU con >16GB VRAM. Para usarlo podríamos emplear HuggingFace Transformers:

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
model = AutoModelForCausalLM.from_pretrained("nous-research/Llama-2-13b-hf", device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("nous-research/Llama-2-13b-hf")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=500)
response = pipe(prompt)  

Esta configuración usaría int8 para reducir memoria. La inferencia podría tardar unos segundos por respuesta. Otra opción es usar soluciones optimizadas como llama.cpp (modelos cuantizados ejecutándose en CPU, accesibles vía binding Python) o servidores como Ollama ￼ que exponen un endpoint local para un modelo.

Embeddings locales vs OpenAI: Similar análisis: OpenAI embeddings (Ada) son muy buenos pero implican envío de datos y costo marginal ￼. Los SentenceTransformers locales (por ej. multi-qa-MiniLM, all-MiniLM-L6-v2) pueden usarse offline aunque con vectores de menor dimensión. En pruebas, para preguntas directas sobre documentos, estos embeddings suelen ser suficientes para recuperar el texto correcto. Incluso existen modelos de embedding multilingües optimizados para búsqueda de información (ej: distiluse-base-multilingual-cased-v2). La arquitectura permite cambiar el adaptador de embedder igualmente.

En resumen, se recomienda:
	•	Desarrollo inicial: OpenAI para LLM y embeddings (rapidez de implementación y calidad).
	•	Luego, evaluar: Si la aplicación va a producción con datos sensibles o sin conexión, incorporar un modo local. Podría incluirse un flag en config para elegir el backend (e.g. USE_LOCAL_LLM=True).
	•	Pruebas A/B: Probar la calidad del LLM local en respuestas comparado con GPT. Ajustar el prompt o incluso considerar fine-tune instruct en el modelo local si las respuestas no son satisfactorias.
	•	Infraestructura: Asegurarse de que el contenedor Docker pueda soportar el modelo local (posiblemente usar imágenes base con CUDA y montar GPU al container, o usar CPU con quantization). Esto puede complicarse, por lo que reiteramos: inicialmente es más simple con OpenAI y luego migrar.

Ejemplos de Estructura de Proyecto, Flujo de Datos y Código

Para aterrizar todo lo anterior, presentaremos un ejemplo de cómo quedaría el flujo completo y snippets ilustrativos de código integrados:

Estructura de carpetas (simplificada):

ocr_cli_project/
├── domain/
│   └── ports.py               # Interfaces: DocumentExtractorPort, EmbedderPort, VectorIndexPort, LLMPort, etc.
├── application/
│   ├── use_cases.py           # ProcessDocumentUseCase, AnswerQueryUseCase
│   └── models.py              # DocumentFragment (con campos content, source, etc.)
├── infrastructure/
│   ├── extractor_pymupdf.py   # class PyMuPDFExtractorAdapter
│   ├── extractor_ocrmix.py    # class OcrPyMuPDFExtractorAdapter (combina Tesseract + PyMuPDF)
│   ├── embed_openai.py        # class OpenAIEmbedderAdapter
│   ├── embed_local.py         # class LocalEmbedderAdapter
│   ├── vector_faiss.py        # class FaissIndexAdapter
│   ├── llm_openai.py          # class OpenAIChatLLMAdapter
│   ├── llm_local.py           # class LocalLLMAdapter (ej. usando HF pipeline)
│   └── storage_filesystem.py  # ya existente, para guardar archivos
├── interface/
│   ├── cli_menu.py            # Interfaz de línea de comandos (usando questionary, etc.)
│   └── api.py                 # Servidor FastAPI con endpoints
└── main.py                    # Punto de entrada; decide CLI vs API y crea instancia de adaptadores

Flujo de datos completo:
	1.	Usuario ingresa un PDF a procesar (vía CLI o API).
	2.	Caso de uso ProcessDocument:
a. Llama a DocumentExtractorPort.extract_markdown(pdf) → (Adapter) PyMuPDF4LLM lee el PDF y devuelve Markdown ￼ (aplica OCR interno si adaptador lo contempla).
b. Llama a StoragePort.save_markdown(file.md, md_text) → guarda en sistema de ficheros o DB.
c. (Opcional) Llama a VectorIndexPort.index(document_id, md_text) → internamente fragmenta el md_text, embeddea cada chunk con EmbedderPort y los almacena en el índice vectorial. Esto puede hacerse aquí para indexar on-the-fly cada documento nuevo. Alternativamente, se podría tener un comando separado de indexación de todos los docs procesados.
	3.	Usuario realiza una pregunta (CLI interactivo o llamada GET/POST en API).
	4.	Caso de uso AnswerQuery:
a. Recibe la pregunta y utiliza EmbedderPort.embed([query]) → (Adapter) OpenAI o local produce vector de la pregunta ￼.
b. Usa VectorIndexPort.query(q_vector, k) → (Adapter) realiza búsqueda en Faiss y retorna p. ej. 3 fragmentos más similares.
c. Prepara el prompt con esos fragmentos de contexto (por ejemplo, los concatena con separadores y añade la pregunta).
d. Llama a LLMPort.ask(prompt) → (Adapter) OpenAI Chat o local genera la respuesta en lenguaje natural.
e. Retorna la respuesta (posiblemente enriquecida con referencias; p.ej., podemos post-procesar la respuesta para añadir 【fuente】 donde corresponda, o incluir al final algo como “Fuente: Documentox.pdf”).
	5.	La respuesta se muestra al usuario (en CLI se imprime texto, en API se envía en JSON).

Este flujo garantiza que el LLM no ve más contexto que los fragmentos relevantes, reduciendo coste y riesgo de dispersión, a la vez que le permite dar respuestas concretas basadas en los documentos ￼.

Snippet de código ilustrativo: a continuación, combinamos varios pasos en una función hipotética para demostrar cómo las piezas encajan (usando adaptadores OpenAI por brevedad):

def answer_question(query: str) -> str:
    # 1. Embed la pregunta
    q_vector = embedder_adapter.embed([query])[0]  # type: ignore
    
    # 2. Buscar fragmentos relevantes
    fragments = vector_index_adapter.query(q_vector, k=3)
    contexto = ""
    for frag in fragments:
        contexto += f"{frag.content}\n---\n"  # separador de fragmentos
    
    # 3. Formar prompt para LLM (con instrucciones simples)
    system_msg = ("Eres un asistente de IA que responde en español usando solo la información proporcionada.\n"
                  "Si la pregunta no puede responderse con ese contexto, indica que no tienes datos.\n"
                  f"Contexto:\n{contexto}")
    user_msg = f"Pregunta: {query}"
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    
    # 4. Invocar modelo de lenguaje (OpenAI ChatCompletion)
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    answer = response["choices"][0]["message"]["content"]
    return answer

En este ejemplo, embedder_adapter y vector_index_adapter son instancias de nuestros adaptadores de infraestructura. Obsérvese que se concatenan los fragmentos con un separador claro (---) y se instruye al LLM a usar solo ese contexto. Esta técnica, junto con el pipeline RAG completo, ayuda a obtener respuestas acertadas y con menor riesgo de alucinación, mejorando la calidad final ￼ ￼.

Conclusión: Con la arquitectura modular propuesta, hemos integrado satisfactoriamente OCR, extracción de PDFs a Markdown con PyMuPDF4LLM, y un sistema de RAG sobre dichos documentos. La aplicación puede iniciar como CLI y transicionar a API sin reescribir la lógica central, simplemente añadiendo adaptadores. Además, está preparada para cambiar de proveedores de AI (OpenAI a local) con mínimo esfuerzo gracias a la inversión de dependencias. Este documento sirve como un README técnico para desarrolladores Python, mostrando no solo el qué sino el cómo implementarlo con buenas prácticas, facilitando la mantenibilidad y extensión futura del proyecto.

Referencias Utilizadas: PyMuPDF4LLM Documentation ￼ ￼, PyMuPDF & RAG Guide ￼ ￼, Artifex Blog ￼ ￼ ￼, Documentación del proyecto (Roadmap) ￼ ￼ ￼, y experiencias recientes en construcción de chatbots con RAG ￼ ￼. Cada componente y decisión está sustentado en estas fuentes para asegurar una implementación alineada con el estado del arte al 2025. ¡Manos a la obra con la integración!