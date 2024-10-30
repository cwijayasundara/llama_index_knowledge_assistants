import json
from llama_index.core.bridge.pydantic import BaseModel
from typing import List
from llama_index.core.types import BaseOutputParser
from llama_index.core import PromptTemplate

# tells LLM to select choices given a list
ROUTER_PROMPT = PromptTemplate(
    "Some choices are given below. It is provided in a numbered list (1 to"
    " {num_choices}), where each item in the list corresponds to a"
    " summary.\n---------------------\n{context_list}\n---------------------\nUsing"
    " only the choices above and not prior knowledge, return the top choices"
    " (no more than {max_outputs}, but only select what is needed) that are"
    " most relevant to the question: '{query_str}'\n"
)

# tells LLM to format list of choices in a certain way
FORMAT_STR = """The output should be formatted as a JSON instance that conforms to 
the JSON schema below. 

Here is the output schema:
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "choice": {
        "type": "integer"
      },
      "reason": {
        "type": "string"
      }
    },
    "required": [
      "choice",
      "reason"
    ],
    "additionalProperties": false
  }
}
"""

class Answer(BaseModel):
    """Answer model."""
    choice: int
    reason: str

class Answers(BaseModel):
    """List of answers model."""
    answers: List[Answer]

class RouterOutputParser(BaseOutputParser):
    """Custom output parser."""

    def _escape_curly_braces(self, input_string: str):
        """Escape the brackets in the format string so contents are not treated as variables."""
        return input_string.replace("{", "{{").replace("}", "}}")

    def _marshal_output_to_json(self, output: str):
        """Find JSON string within response."""

        output = output.strip()
        left = output.find("[")
        right = output.find("]")
        output = output[left: right + 1]
        return output

    def parse(self, output: str) -> Answers:
        """Parse string"""

        json_output = self._marshal_output_to_json(output)
        json_dicts = json.loads(json_output)
        answers = [Answer.parse_obj(json_dict) for json_dict in json_dicts]
        return Answers(answers=answers)

    def format(self, query: str) -> str:
        return query + "\n\n" + self._escape_curly_braces(FORMAT_STR)