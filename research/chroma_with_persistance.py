import warnings
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb

warnings.filterwarnings('ignore')
_ = load_dotenv()

embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")

documents = SimpleDirectoryReader("./data/").load_data()

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

# load from disk
db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index_1 = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
)

# Query Data from the persisted index
query_engine = index_1.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)

# delete & update
doc_to_update = chroma_collection.get(limit=1)

doc_to_update["metadatas"][0] = {
    **doc_to_update["metadatas"][0],
    **{"author": "Paul Graham"},
}

chroma_collection.update(
    ids=[doc_to_update["ids"][0]], metadatas=[doc_to_update["metadatas"][0]]
)

updated_doc = chroma_collection.get(limit=1)
print(updated_doc["metadatas"][0])

# delete the last document
print("count before", chroma_collection.count())
chroma_collection.delete(ids=[doc_to_update["ids"][0]])
print("count after", chroma_collection.count())
