import os
import warnings
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_community.embeddings import HuggingFaceEmbeddings

# Ocultar warnings de deprecación
warnings.filterwarnings("ignore", category=DeprecationWarning)

# PALABRAS CLAVE MÉDICAS Y CLÍNICAS
KEYWORDS_MAP = {
    "criterios": [
        "criterios de inclusión", "criterios de exclusión", "consentimiento informado",
        "participante elegible", "diagnóstico", "recaídas", "EDSS", "selección", "aleatorización",
        "población de estudio", "población por protocolo", "enmienda de protocolo", "FCI", "firma de consentimiento"
    ],
    "farmacocinetica": [
        "PK", "farmacocinética", "farmacodinamia", "Cmax", "Tmax", "AUC",
        "vida media", "clearance", "concentración plasmática", "dosis", "curva PK", 
        "muestras de sangre", "absorción", "distribución", "metabolismo", "eliminación"
    ],
    "biomarcadores": [
        "biomarcador", "biomarcadores", "NfL", "cadena ligera de neurofilamentos",
        "expresión génica", "genómica", "proteómica", "biología molecular", "respuestas inmunes", 
        "marcadores inflamatorios", "panel de biomarcadores", "análisis de biomarcadores"
    ],
    "seguridad": [
        "eventos adversos", "EA", "eventos serios", "efectos secundarios", "farmacovigilancia",
        "seguimiento de seguridad", "desenlace", "complicaciones", "reacciones adversas",
        "mortalidad", "interrupción del tratamiento", "notificación de eventos"
    ],
    "procedimientos": [
        "visitas de seguimiento", "procedimiento", "examen físico", "ECG", "laboratorio clínico",
        "pruebas de embarazo", "evaluaciones", "subestudio", "toma de muestras", "monitoreo"
    ],
    "estadistica": [
        "análisis estadístico", "modelo NB", "regresión", "intervalo de confianza",
        "cociente de tasas", "mediana", "análisis de sensibilidad", "endpoint",
        "criterio de valoración", "p-valor", "power", "desviación estándar"
    ],
    "estructuras": [
        "tabla", "figura", "cuadro", "anexo", "apéndice", "tabla PK", "tabla de biomarcadores",
        "tabla de eventos adversos", "tabla de criterios", "tabla de análisis"
    ],
    "regulaciones": [
        "IRB", "IEC", "GCP", "ICH", "regulación europea", "consentimiento ético",
        "comité ético", "autorización", "aprobación regulatoria"
    ]
}

# PALABRAS CLAVE MÉDICAS GLOBALES (aplicables a cualquier estudio clínico)
KEYWORDS_MEDICAS = [
    "pk", "farmacocinética", "cmax", "tmax", "auc", "t1/2", "clearance",
    "exposición", "concentración plasmática", "niveles plasmáticos", "parámetros pk",
    "biomarcador", "biomarcadores", "nfl", "expresión génica", "proteínas",
    "niveles séricos", "respuesta biológica", "marcadores de laboratorio",
    "evaluaciones", "visitas", "cronograma", "tabla de evaluaciones",
    "parámetros medidos", "puntos temporales", "semana", "día", "muestreo",
    "procedimientos", "asignaciones", "recolecion de muestras",
    "eventos adversos", "seguridad", "efectos adversos", "acontecimientos adversos",
    "serious adverse event", "seguimiento", "monitorización",
    "eficacia", "objetivos secundarios", "objetivos primarios",
    "resultados clínicos", "endpoints", "análisis estadístico", "población de análisis"
]

PROMPT_TEMPLATE = """
Eres un especialista en análisis de documentos clínicos. Tu tarea es EXTRAER INFORMACIÓN PRECISA basándote ÚNICAMENTE en el contexto proporcionado.

CONTEXTO:
{context}

PREGUNTA: {question}

INSTRUCCIONES CRÍTICAS - DEBES SEGUIR ESTAS REGLAS:

**EXTRACCIÓN OBLIGATORIA:**
1. SI encuentras la información solicitada: EXTRAE y LISTA todos los elementos específicos
2. SI NO encuentras la información: Di claramente "No encontré [información solicitada] en el documento"
3. NO inventes, NO supongas, NO uses conocimiento externo

**PARA CRITERIOS/LISTAS/TABLAS:**
4. Si es una lista: Extrae TODOS los elementos, no solo algunos
5. Si es una tabla: Revisa TODAS las filas y columnas relevantes
6. Incluye números de página cuando estén disponibles

**BÚSQUEDA EXHAUSTIVA:**
7. Revisa TODO el contexto proporcionado antes de responder
8. Si hay información en múltiples secciones: CONSOLIDA y muestra TODOS los datos
9. Si hay discrepancias: Menciona todas las versiones encontradas

**PROHIBIDO:**
- "No hay información disponible en esta sección" (es redundante)
- "Revisé pero no encontré" (di directamente si encontraste o no)
- Dar ejemplos hipotéticos
- Usar lenguaje vago como "algunos", "varios", "entre otros"

**FORMATO DE RESPUESTA OBLIGATORIO:**
- Si ENCONTRASTE información: 
  "ENCONTRÉ [número] elementos:"
  [Lista numerada completa]
  "Fuente: [páginas/tablas donde se encontró]"

- Si NO ENCONTRASTE información:
  "No encontré [información específica] en el documento revisado."

Respuesta en español:
"""

# Configuración de Qdrant LOCAL
QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = ""
COLLECTION_NAME = "rag_documents"

def get_unified_llm():
    return Ollama(
        model="llama3.2:3b",
        temperature=0.1,
        top_p=0.7,
        num_predict=1200,
        timeout=600,
        top_k=40,
        repeat_penalty=1.1
    )

def get_embedding_function():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            'device': 'cpu',
            'trust_remote_code': True
        },
        encode_kwargs={
            'normalize_embeddings': True,
            'batch_size': 32
        }
    )

def query_rag(query_text: str):
    try:
        # Conectar a Qdrant
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        
        # Verificar que la colección existe
        collections = client.get_collections()
        collection_exists = any(col.name == COLLECTION_NAME for col in collections.collections)
        
        if not collection_exists:
            print("Error: La colección no existe. Ejecuta primero: python populate_database.py")
            return "Error: Base de datos no encontrada"
        
        # Obtener información de la colección
        collection_info = client.get_collection(COLLECTION_NAME)
        document_count = collection_info.points_count
        print(f"Documentos en la base de datos: {document_count}")
        
        if document_count == 0:
            print("La base de datos está vacía. Ejecuta: python populate_database.py")
            return "Error: Base de datos vacía"

        # Crear vector store
        embedding_function = get_embedding_function()
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embedding_function,
        )

        # BÚSQUEDA INTELIGENTE: Más resultados con umbral bajo
        results = vector_store.similarity_search_with_relevance_scores(query_text, k=80)
        
        if not results:
            print("No se encontraron resultados relevantes")
            return "No se encontraron documentos relevantes"
        
        # DETECCIÓN DE PALABRAS CLAVE: Identificar contexto médico
        user_query_lower = query_text.lower()
        matched_keywords = []
        detected_categories = []
        
        for category, keywords in KEYWORDS_MAP.items():
            for keyword in keywords:
                if keyword in user_query_lower:
                    matched_keywords.append(keyword)
                    if category not in detected_categories:
                        detected_categories.append(category)
        
        if matched_keywords:
            print(f"Palabras clave detectadas ({len(set(matched_keywords))}): {', '.join(set(matched_keywords)[:5])}...")
        
        # FILTRADO INTELIGENTE CON BOOST: Mantener resultados relevantes
        filtered_results = []
        for doc, score in results:
            boost = 0
            if matched_keywords and any(word in doc.page_content.lower() for word in matched_keywords):
                boost = 0.15
            
            final_score = score + boost
            
            if final_score > 0.15:
                filtered_results.append(doc)
        
        if not filtered_results:
            filtered_results = [doc for doc, _ in results[:50]]
        
        print(f"✅ Encontrados {len(filtered_results)} documentos relevantes")
        context_text = "\n\n---\n\n".join([f"[Documento {i+1}]:\n{doc.page_content}" for i, doc in enumerate(filtered_results)])

        # Construir el prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Generar respuesta con Ollama
        model = get_unified_llm()
        response = model.invoke(prompt)

        print(f"\n📄 Respuesta: {response.strip()}\n")
        return response
        
    except Exception as e:
        print(f"❌ Error conectando a Qdrant: {e}")
        print("💡 Verifica tu QDRANT_URL y QDRANT_API_KEY")
        return "Error de conexión con la base de datos"