from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import warnings

warnings.filterwarnings('ignore')

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

QDRANT_URL = "http://localhost:6333"
QDRANT_API_KEY = ""
COLLECTION_NAME = "rag_documents"

def get_embedding_function():
    try:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                'device': 'cpu',
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'batch_size': 16
            }
        )
    except Exception as e:
        print(f"❌ Error cargando embeddings: {e}")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )