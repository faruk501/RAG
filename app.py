import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain.schema import Document
import tempfile
import time
import os
import re

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = ""

@st.cache_resource
def get_embeddings():
    from get_embedding_function import get_embedding_function
    return get_embedding_function()

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=QDRANT_URL, 
        api_key=QDRANT_API_KEY,
        timeout=60.0,
        prefer_grpc=True
    )

# Cargar recursos cacheados
embeddings = get_embeddings()
client = get_qdrant_client()

llm = Ollama(
    model="llama3.2:3b",
    temperature=0.0,
    top_p=0.6,
    num_predict=1000,
    timeout=300,
)


def create_smart_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=300,
        length_function=len,
        separators=[
            "\n\n\n",
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )

def detect_content_type(text):
    """Detecta si el contenido es tabla, cronograma o texto"""
    text_lower = text.lower()
    
    has_pipes = text.count("|") > 5
    has_dashes = "---" in text or "___" in text
    has_columns = bool(re.search(r"\s{3,}", text))
    
    if has_pipes or (has_dashes and has_columns):
        return "table"
    
    if re.search(r"(w\d+|week\s*\d+|semana\s*\d+|visita\s*\d+)", text_lower):
        return "schedule"
    
    return "text"

def enrich_metadata(chunks):
    """Enriquece los fragmentos con metadatos de tipo de contenido"""
    for i, chunk in enumerate(chunks):
        content_type = detect_content_type(chunk.page_content)
        chunk.metadata["content_type"] = content_type
        chunk.metadata["chunk_index"] = i
        
        if re.search(r"(w\d+|week\s*\d+|semana\s*\d+)", chunk.page_content.lower()):
            chunk.metadata["has_timepoints"] = True
    
    return chunks

def smart_search(query_text, vector_store):
    """Búsqueda adaptativa que ajusta parámetros según el tipo de pregunta"""
    query_lower = query_text.lower()
    
    is_conceptual = any(word in query_lower for word in 
                       ["objetivo", "propósito", "finalidad", "qué es", "definición",
                        "describe", "explica", "resumen", "introducción", "antecedente",
                        "background", "rationale", "justificación", "motivo"])
    
    is_temporal = any(word in query_lower for word in 
                     ["semana", "week", "cuándo", "when", "qué día", "visita", "cronograma"])
    
    if is_conceptual:
        k = 20
        threshold = 0.3
    elif is_temporal:
        k = 50
        threshold = 0.1
    else:
        k = 30
        threshold = 0.2
    
    results = vector_store.similarity_search_with_relevance_scores(
        query_text,
        k=k,
        score_threshold=threshold
    )
    
    if not results:
        return []
    
    boosted = []
    for doc, score in results:
        content_type = doc.metadata.get("content_type", "")
        final_score = score
        
        if is_conceptual:
            if content_type == "table":
                final_score = score - 0.2
            else:
                final_score = score + 0.1
        
        elif is_temporal:
            final_score = score + (0.3 if content_type == "table" else 0)
            if doc.metadata.get("has_timepoints"):
                final_score += 0.2
        
        else:
            final_score = score + (0.15 if content_type == "table" else 0)
        
        boosted.append((doc, final_score))
    
    # Ordenar y retornar top resultados
    boosted.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in boosted[:25]]

def create_prompt(context, question):
    return f"""Eres un extractor de información médica. Tu trabajo es buscar y extraer EXACTAMENTE lo que está escrito.

CONTEXTO:
{context}

PREGUNTA: {question}

REGLAS:
1. Si ves una TABLA con marcas "X" o valores, extrae TODAS las filas relevantes
2. Si no encuentras información, di "No encontré información sobre [tema]"
3. NO inventes nada
4. Menciona la página si está disponible

FORMATO:
✅ Si encontraste:
• Elemento 1
• Elemento 2
...
📄 Fuente: Página X

❌ Si no encontraste:
No encontré información sobre [tema].

Respuesta:"""

def query_rag(query_text):
    if "documents_processed" not in st.session_state or not st.session_state.documents_processed:
        return "⚠️ Primero sube y procesa documentos PDF."
    
    try:
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=st.session_state.collection_name,
            embedding=embeddings,
        )
        
        docs = smart_search(query_text, vector_store)
        
        if not docs:
            return "❌ No encontré información relevante."
        
        query_lower = query_text.lower()
        
        is_conceptual = any(word in query_lower for word in 
                           ["objetivo", "propósito", "finalidad", "qué es", "definición",
                            "describe", "explica", "resumen", "introducción", "antecedente",
                            "background", "rationale", "justificación", "motivo"])
        is_temporal = any(word in query_lower for word in ["semana", "week", "visita", "cuándo"])
        
        if is_conceptual:
            docs = sorted(docs, 
                         key=lambda x: x.metadata.get("content_type") != "table", 
                         reverse=True)
        elif is_temporal:
            docs = sorted(docs, 
                         key=lambda x: x.metadata.get("content_type") == "table", 
                         reverse=True)
        
        context_parts = []
        pages = set()
        total_chars = 0
        MAX = 15000
        
        for doc in docs:
            content = doc.page_content
            if total_chars + len(content) > MAX:
                break
            
            page = doc.metadata.get('page', '?')
            content_type = doc.metadata.get('content_type', 'text')
            
            if content_type == "table":
                context_parts.append(f"📈 TABLA (Página {page}):\n{content}")
            else:
                context_parts.append(f"📄 Texto (Página {page}):\n{content}")
            
            total_chars += len(content)
            if page != '?':
                pages.add(str(page))
        
        context = "\n\n---\n\n".join(context_parts)
        
        if pages:
            context += f"\n\n📚 Páginas: {', '.join(sorted(pages, key=lambda x: int(x) if x.isdigit() else 999))}"
        
        # Generar respuesta
        prompt = create_prompt(context, query_text)
        response = llm.invoke(prompt)
        
        return response if response and response.strip() else "⚠️ No se generó respuesta."
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

def process_pdfs(uploaded_files, collection_name):
    try:
        all_chunks = []
        progress_bar = st.progress(0)
        status = st.empty()
        
        for idx, file in enumerate(uploaded_files):
            status.text(f"📄 {file.name}...")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            
            loader = PyMuPDFLoader(tmp_path)
            docs = loader.load()
            
            splitter = create_smart_splitter()
            chunks = splitter.split_documents(docs)
            chunks = enrich_metadata(chunks)
            
            for chunk in chunks:
                chunk.metadata["source"] = file.name
            
            all_chunks.extend(chunks)
            progress_bar.progress((idx + 1) / len(uploaded_files))
            os.unlink(tmp_path)
        
        status.text("Creando base de datos...")
        
        try:
            client.delete_collection(collection_name=collection_name)
        except:
            pass
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )
        
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        
        status.text("Subiendo documentos...")
        vector_store.add_documents(all_chunks)
        
        progress_bar.progress(1.0)
        status.empty()
        
        return len(uploaded_files), len(all_chunks)
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return 0, 0

st.set_page_config(page_title="Sistema RAG Médico", page_icon="🎋", layout="centered")
st.title("Sistema RAG Médico")
st.markdown("**Versión optimizada para tablas complejas**")

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "collection_name" not in st.session_state:
    st.session_state.collection_name = ""
if "history" not in st.session_state:
    st.session_state.history = []

uploaded_files = st.file_uploader(
    "📁 Subir documentos PDF", 
    type="pdf", 
    accept_multiple_files=True
)

if uploaded_files and not st.session_state.documents_processed:
    if st.button("🚀 Procesar", type="primary"):
        with st.spinner("Procesando..."):
            collection = f"docs_{int(time.time())}"
            num_docs, num_chunks = process_pdfs(uploaded_files, collection)
            
            if num_chunks > 0:
                st.session_state.collection_name = collection
                st.session_state.documents_processed = True
                st.success(f"✅ {num_docs} documentos ({num_chunks} fragmentos)")
                st.rerun()

if st.session_state.documents_processed:
    st.markdown("---")
    
    if st.button("🔄 Nuevo documento"):
        st.session_state.documents_processed = False
        st.session_state.history = []
        st.rerun()
    
    question = st.text_input(
        "🗣 Pregunta:",
        placeholder="Ej: ¿En qué semanas se toman muestras para PK?"
    )
    
    if st.button("🔎 Buscar", type="primary") and question.strip():
        with st.spinner("Buscando..."):
            answer = query_rag(question)
        st.session_state.history.append({"q": question, "a": answer})
        st.markdown("### 📈 Respuesta:")
        st.write(answer)
    
    if len(st.session_state.history) > 0:
        st.markdown("---")
        with st.expander("📚 Historial"):
            for i, item in enumerate(reversed(st.session_state.history)):
                st.markdown(f"**{i+1}. {item['q']}**")
                st.write(item['a'])
                st.markdown("---")