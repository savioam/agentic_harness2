from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class HarnessState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    target_file: str
    proposed_code: Optional[str]
    test_output: Optional[str]
    tests_passed: bool
    iteration_count: int
    max_iterations: int
    session_id: Optional[str]
    generator_error: Optional[str]
