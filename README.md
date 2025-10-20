# Sistema RAG Avanzado para Análisis de Documentos Médicos

Un sistema de Retrieval-Augmented Generation (RAG) especializado en el análisis de documentos médicos, estudios clínicos y protocolos de investigación. Extrae información precisa de PDFs complejos con tablas extensas y datos estructurados.

## 🚀 Características Principales

### 🔍 Extracción Avanzada de Tablas
- **Extracción completa de tablas** con preservación de estructura
- **Detección automática** de encabezados y filas
- **Metadata enriquecida** para mejor recuperación
- **Agrupación inteligente** de fragmentos de tablas

### 🤖 Búsqueda Inteligente
- **Priorización automática** de contenido tabular vs textual
- **Recuperación de contexto extenso** (hasta 100 documentos)
- **Filtrado por relevancia** con umbrales adaptativos
- **Detección de preguntas tabulares** para respuestas completas

### 📊 Procesamiento Optimizado
- **Chunking inteligente** que respeta límites de tablas
- **Metadata especializada** para diferentes tipos de contenido
- **Procesamiento por lotes** con progreso en tiempo real
- **Manejo de documentos largos** (hasta 265+ páginas)

## 🛠️ Instalación

### Prerrequisitos
- Python 3.8+
- Ollama con modelo `llama3.1` instalado
- Cuenta en Qdrant Cloud

### 1. Clonar e instalar dependencias
```bash
git clone <repository-url>
cd rag-tutorial-v2
pip install -r requirements.txt
