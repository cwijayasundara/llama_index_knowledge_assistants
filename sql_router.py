import warnings
import os
import chromadb
from dotenv import load_dotenv
import nest_asyncio
import llama_index.core
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI

nest_asyncio.apply()

warnings.filterwarnings('ignore')
_ = load_dotenv()

# setup Arize Phoenix for logging/observability
PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"

llama_index.core.set_global_handler(
    "arize_phoenix", endpoint="https://llamatrace.com/v1/traces"
)

# Vector Index from the auto-rag example
embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

llm = OpenAI(model="gpt-4o-mini")

# load from disk
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("research_papers")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)