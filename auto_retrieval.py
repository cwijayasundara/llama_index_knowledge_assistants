import warnings
import chromadb
import json
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.vector_stores.types import VectorStoreInfo, VectorStoreQuerySpec, MetadataInfo, MetadataFilters
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Response
from functools import partial

warnings.filterwarnings('ignore')
_ = load_dotenv()

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

# Define LlamaCloud File/Chunk Retriever over Documents
doc_retriever = index.as_retriever(
    retrieval_mode="files_via_content",
    # retrieval_mode="files_via_metadata",
    files_top_k=1
)

chunk_retriever = index.as_retriever(
    retrieval_mode="chunks",
    rerank_top_n=5
)

# Setup Auto-Retrieval

SYS_PROMPT = """\
Your goal is to structure the user's query to match the request schema provided below.
You MUST call the tool in order to generate the query spec.

<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the \
following schema:

{schema_str}

The query string should contain only text that is expected to match the contents of \
documents. Any conditions in the filter should not be mentioned in the query as well.

Make sure that filters only refer to attributes that exist in the data source.
Make sure that filters take into account the descriptions of attributes.
Make sure that filters are only used as needed. If there are no filters that should be \
applied return [] for the filter value.\

If the user's query explicitly mentions number of documents to retrieve, set top_k to \
that number, otherwise do not set top_k.

The schema of the metadata filters in the vector db table is listed below, along with some example metadata dictionaries
 from relevant rows The user will send the input query string.

Data Source:
```json
{info_str}
```

Example metadata from relevant chunks:
{example_rows}

"""

example_rows_retriever = index.as_retriever(
    retrieval_mode="chunks",
    rerank_top_n=4
)

def get_example_rows_fn(**kwargs):
    """Retrieve relevant few-shot examples."""
    query_str = kwargs["query_str"]
    nodes = example_rows_retriever.retrieve(query_str)
    # get the metadata, join them
    metadata_list = [n.metadata for n in nodes]

    return "\n".join([json.dumps(m) for m in metadata_list])


# TODO: define function mapping for `example_rows`.
chat_prompt_tmpl = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        ("user", "{query_str}"),
    ],
    function_mappings={
        "example_rows": get_example_rows_fn
    }
)

## NOTE: this is a dataclass that contains information about the metadata
vector_store_info = VectorStoreInfo(
    content_info="contains content from various research papers",
    metadata_info=[
        MetadataInfo(
            name="file_name",
            type="str",
            description="Name of the source paper",
        ),
    ],
)

def auto_retriever_rag(query: str, retriever: BaseRetriever) -> Response:
    """Synthesizes an answer to your question by feeding in an entire relevant document as context."""
    print(f"> User query string: {query}")
    # Use structured predict to infer the metadata filters and query string.
    query_spec = llm.structured_predict(
        VectorStoreQuerySpec,
        chat_prompt_tmpl,
        info_str=vector_store_info.json(),
        schema_str=VectorStoreQuerySpec.schema_json(),
        query_str=query
    )
    # build retriever and query engine
    filters = MetadataFilters(filters=query_spec.filters) if len(query_spec.filters) > 0 else None
    print(f"> Inferred query string: {query_spec.query}")
    if filters:
        print(f"> Inferred filters: {filters.json()}")
    query_engine = RetrieverQueryEngine.from_args(
        retriever,
        llm=llm,
        response_mode="tree_summarize"
    )
    # run query
    return query_engine.query(query_spec.query)

# Try out Auto-Retrieval

auto_doc_rag = partial(auto_retriever_rag, retriever=doc_retriever)

auto_chunk_rag = partial(auto_retriever_rag, retriever=chunk_retriever)

response_1 = auto_chunk_rag("ELI5 the objective function in Metra")
print(str(response_1))

response_2 = auto_chunk_rag("How was SWE-Bench constructed? Tell me all the stages that went into it.")
print(str(response_2))

response_3 = auto_doc_rag("Give me a summary of the SWE-bench paper")
print(str(response_3))

response_4 = auto_doc_rag("Give me a summary of the Self-RAG paper")
print(str(response_4))
