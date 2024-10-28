import warnings
import chromadb
from dotenv import load_dotenv
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding

warnings.filterwarnings('ignore')
_ = load_dotenv()

embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

parser = LlamaParse(
    result_type="markdown"  # "markdown" and "text" are available
)

file_extractor = {".pdf": parser}
documents = SimpleDirectoryReader(input_files=['data/metra.pdf', 'data/selfrag.pdf', 'data/swebench.pdf'],
                                  file_extractor=file_extractor).load_data()

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("research_papers")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

# Test : Query Data from the persisted index
# query_engine = index.as_query_engine()
# response = query_engine.query("Explain how self-rag works?")
# print(response)