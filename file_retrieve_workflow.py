import warnings
import os
import chromadb
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
import llama_index.core
from router_output_parser import RouterOutputParser, Answer, Answers, ROUTER_PROMPT
from typing import List, Optional, Any

from llama_index.core.query_engine import (
    BaseQueryEngine,
    RetrieverQueryEngine,
)
from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import LLM
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.workflow import (
    Workflow,
    Event,
    StartEvent,
    StopEvent,
    step,
)
import nest_asyncio
from typing import Dict, List

from llama_index.core.tools import BaseTool
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import ToolSelection
from llama_index.core.workflow import Context, Workflow
from llama_index.core.base.response.schema import Response
from llama_index.core.tools import FunctionTool

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

# Router Query Workflow
class ChooseQueryEngineEvent(Event):
    """Query engine event."""

    answers: Answers
    query_str: str

class SynthesizeAnswersEvent(Event):
    """Synthesize answers event."""

    responses: List[Any]
    query_str: str


class RouterQueryWorkflow(Workflow):
    """Router query workflow."""

    def __init__(
            self,
            query_engines: List[BaseQueryEngine],
            choice_descriptions: List[str],
            router_prompt: PromptTemplate,
            timeout: Optional[float] = 10.0,
            disable_validation: bool = False,
            verbose: bool = False,
            llm: Optional[LLM] = None,
            summarizer: Optional[TreeSummarize] = None,
    ):
        """Constructor"""

        super().__init__(timeout=timeout, disable_validation=disable_validation, verbose=verbose)

        self.query_engines: List[BaseQueryEngine] = query_engines
        self.choice_descriptions: List[str] = choice_descriptions
        self.router_prompt: PromptTemplate = router_prompt
        self.llm: LLM = llm or OpenAI(temperature=0, model="gpt-4o")
        self.summarizer: TreeSummarize = summarizer or TreeSummarize()

    def _get_choice_str(self, choices):
        """String of choices to feed into LLM."""

        choices_str = "\n\n".join([f"{idx + 1}. {c}" for idx, c in enumerate(choices)])
        return choices_str

    async def _query(self, query_str: str, choice_idx: int):
        """Query using query engine"""

        query_engine = self.query_engines[choice_idx]
        return await query_engine.aquery(query_str)

    @step()
    async def choose_query_engine(self, ev: StartEvent) -> ChooseQueryEngineEvent:
        """Choose query engine."""

        # get query str
        query_str = ev.get("query_str")
        if query_str is None:
            raise ValueError("'query_str' is required.")

        # partially format prompt with number of choices and max outputs
        router_prompt1 = self.router_prompt.partial_format(
            num_choices=len(self.choice_descriptions),
            max_outputs=len(self.choice_descriptions),
        )

        # get choices selected by LLM
        choices_str = self._get_choice_str(self.choice_descriptions)
        output = llm.structured_predict(
            Answers,
            router_prompt1,
            context_list=choices_str,
            query_str=query_str
        )

        if self._verbose:
            print(f"Selected choice(s):")
            for answer in output.answers:
                print(f"Choice: {answer.choice}, Reason: {answer.reason}")

        return ChooseQueryEngineEvent(answers=output, query_str=query_str)

    @step()
    async def query_each_engine(self, ev: ChooseQueryEngineEvent) -> SynthesizeAnswersEvent:
        """Query each engine."""

        query_str = ev.query_str
        answers = ev.answers

        # query using corresponding query engine given in Answers list
        responses = []

        for answer in answers.answers:
            choice_idx = answer.choice - 1
            response = await self._query(query_str, choice_idx)
            responses.append(response)

        return SynthesizeAnswersEvent(responses=responses, query_str=query_str)

    @step()
    async def synthesize_response(self, ev: SynthesizeAnswersEvent) -> StopEvent:
        """Synthesizes response."""

        responses = ev.responses
        query_str = ev.query_str

        # get result of responses
        if len(responses) == 1:
            return StopEvent(result=responses[0])
        else:
            response_strs = [str(r) for r in responses]
            result_response = self.summarizer.get_response(query_str, response_strs)
            return StopEvent(result=result_response)

# Define LlamaCloud Retriever over documents
doc_retriever = index.as_retriever(retrieval_mode="files_via_content", files_top_k=1)

query_engine_doc = RetrieverQueryEngine.from_args(
    doc_retriever, llm=llm, response_mode="tree_summarize"
)

chunk_retriever = index.as_retriever(retrieval_mode="chunks", rerank_top_n=10)

query_engine_chunk = RetrieverQueryEngine.from_args(
    chunk_retriever, llm=llm, response_mode="tree_summarize"
)

DOC_METADATA_EXTRA_STR = """\
Each document represents a complete 10K report for a given year (e.g. Apple in 2019).
Here's an example of relevant documents:
1. apple_2019.pdf
2. tesla_2020.pdf
"""

TOOL_DOC_DESC = f"""\
Synthesizes an answer to your question by feeding in an entire relevant document as context. Best used for higher-level summarization options.
Do NOT use if answer can be found in a specific chunk of a given document. Use the chunk_query_engine instead for that purpose.

Below we give details on the format of each document:
{DOC_METADATA_EXTRA_STR}
"""

TOOL_CHUNK_DESC = f"""\
Synthesizes an answer to your question by feeding in a relevant chunk as context. Best used for questions that are more pointed in nature.
Do NOT use if the question asks seems to require a general summary of any given document. Use the doc_query_engine instead for that purpose.

Below we give details on the format of each document:
{DOC_METADATA_EXTRA_STR}
"""

router_query_workflow = RouterQueryWorkflow(
    query_engines=[query_engine_doc, query_engine_chunk],
    choice_descriptions=[TOOL_DOC_DESC, TOOL_CHUNK_DESC],
    verbose=True,
    llm=llm,
    router_prompt=ROUTER_PROMPT,
    timeout=60
)

# Creating an Agent Around the Query Engine
class InputEvent(Event):
    """Input event."""

class GatherToolsEvent(Event):
    """Gather Tools Event"""

    tool_calls: Any

class ToolCallEvent(Event):
    """Tool Call event"""

    tool_call: ToolSelection

class ToolCallEventResult(Event):
    """Tool call event result."""

    msg: ChatMessage

class RouterOutputAgentWorkflow(Workflow):
    """Custom router output agent workflow."""

    def __init__(self,
                 rag_workflow: Workflow,
                 timeout: Optional[float] = 10.0,
                 disable_validation: bool = False,
                 verbose: bool = False,
                 llm: Optional[LLM] = None,
                 chat_history: Optional[List[ChatMessage]] = None,
                 ):
        """Constructor."""

        super().__init__(timeout=timeout, disable_validation=disable_validation, verbose=verbose)

        self.rag_workflow = rag_workflow

        def query_workflow(query_str: str) -> Response:
            """Queries 10k reports for a given year."""
            return self.rag_workflow.run(query_str=query_str)

        self.rag_workflow_tool = FunctionTool.from_defaults(query_workflow)

        self.llm: LLM = llm or OpenAI(temperature=0, model="gpt-4o")
        self.chat_history: List[ChatMessage] = chat_history or []

    def reset(self) -> None:
        """Resets Chat History"""

        self.chat_history = []

    @step()
    async def prepare_chat(self, ev: StartEvent) -> InputEvent:
        message = ev.get("message")
        if message is None:
            raise ValueError("'message' field is required.")

        # add msg to chat history
        chat_history = self.chat_history
        chat_history.append(ChatMessage(role="user", content=message))
        return InputEvent()

    @step()
    async def chat(self, ev: InputEvent) -> GatherToolsEvent | StopEvent:
        """Appends msg to chat history, then gets tool calls."""

        # Put msg into LLM with tools included
        chat_res = await self.llm.achat_with_tools(
            [self.rag_workflow_tool],
            chat_history=self.chat_history,
            verbose=self._verbose,
            allow_parallel_tool_calls=True
        )
        tool_calls = self.llm.get_tool_calls_from_response(chat_res, error_on_no_tool_call=False)

        ai_message = chat_res.message
        self.chat_history.append(ai_message)
        if self._verbose:
            print(f"Chat message: {ai_message.content}")

        # no tool calls, return chat message.
        if not tool_calls:
            return StopEvent(result=ai_message.content)

        return GatherToolsEvent(tool_calls=tool_calls)

    @step(pass_context=True)
    async def dispatch_calls(self, ctx: Context, ev: GatherToolsEvent) -> ToolCallEvent:
        """Dispatches calls."""

        tool_calls = ev.tool_calls
        await ctx.set("num_tool_calls", len(tool_calls))

        # trigger tool call events
        for tool_call in tool_calls:
            ctx.send_event(ToolCallEvent(tool_call=tool_call))

        return None

    @step()
    async def call_tool(self, ev: ToolCallEvent) -> ToolCallEventResult:
        """Calls tool."""

        tool_call = ev.tool_call

        # get tool ID and function call
        id_ = tool_call.tool_id

        if self._verbose:
            print(f"Calling function {tool_call.tool_name} with msg {tool_call.tool_kwargs}")

        # directly run workflow, don't call tools
        output = await self.rag_workflow.run(**tool_call.tool_kwargs)
        msg = ChatMessage(
            name=tool_call.tool_name,
            content=str(output),
            role="tool",
            additional_kwargs={
                "tool_call_id": id_,
                "name": tool_call.tool_name
            }
        )

        return ToolCallEventResult(msg=msg)

    @step(pass_context=True)
    async def gather(self, ctx: Context, ev: ToolCallEventResult) -> StopEvent | None:
        """Gathers tool calls."""
        # wait for all tool call events to finish.
        tool_events = ctx.collect_events(ev, [ToolCallEventResult] * await ctx.get("num_tool_calls"))
        if not tool_events:
            return None

        for tool_event in tool_events:
            # append tool call chat messages to history
            self.chat_history.append(tool_event.msg)

        # # after all tool calls finish, pass input event back, restart agent loop
        return InputEvent()

agent = RouterOutputAgentWorkflow(router_query_workflow, verbose=True, timeout=60)

from llama_index.utils.workflow import draw_all_possible_flows

draw_all_possible_flows(RouterOutputAgentWorkflow)

async def main():
    response = await agent.run(message="Tell me how self rag and corrective rag workflows differ.")
    return response

# To run the async function
import asyncio
result = asyncio.run(main())

