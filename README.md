# Sistema RAG Avanzado para Análisis de Documentos Médicos

Un sistema de Retrieval-Augmented Generation (RAG) especializado en el análisis de documentos médicos, estudios clínicos y protocolos de investigación. Extrae información precisa de PDFs complejos con tablas extensas y datos estructurados.

⚡ Punto Crítico a Mejorar: Reducción de Latencia
🚨 PROBLEMA IDENTIFICADO
Larga latencia en respuestas - Actualmente el sistema puede tomar entre 1 a 2 minutos por consulta, lo que afecta la experiencia de usuario en entornos productivos.

🎯 RECOMENDACIÓN PRIORITARIA
🔧 IMPLEMENTAR PINECONE - Para Escala Profesional

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

### corre con esto ,  streamlit run app.py

