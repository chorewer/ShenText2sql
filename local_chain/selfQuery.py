from typing import Any, Dict, List, Optional, TypedDict, Union

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda

from langchain.chains.sql_database.prompt import PROMPT, SQL_PROMPTS
from database.db import chroma_vs
def _strip(text: str) -> str:
    return text.strip()

class SQLInput(TypedDict):
    """Input for a SQL Chain."""

    question: str

def _output(text: str) -> str:
    print(text)
    return text

def _output_any(anything: Any) -> Any:
    print(anything)
    return anything
class SQLInputWithTables(TypedDict):
    """Input for a SQL Chain."""

    question: str
    table_names_to_use: List[str]

def create_sql_query_chain(
    llm: BaseLanguageModel,
    db: SQLDatabase,
    prompt: Optional[BasePromptTemplate] = None,
    k: int = 5,
) -> Runnable[Union[SQLInput, SQLInputWithTables, Dict[str, Any]], str]:
    if prompt is not None:
        prompt_to_use = prompt
    elif db.dialect in SQL_PROMPTS:
        prompt_to_use = SQL_PROMPTS[db.dialect]
    else:
        prompt_to_use = PROMPT
    if {"input", "top_k", "table_info", "relevant_msg"}.difference(prompt_to_use.input_variables):
        raise ValueError(
            f"Prompt must have input variables: 'input', 'top_k', "
            f"'table_info'. Received prompt with input variables: "
            f"{prompt_to_use.input_variables}. Full prompt:\n\n{prompt_to_use}"
        )
    if "dialect" in prompt_to_use.input_variables:
        prompt_to_use = prompt_to_use.partial(dialect=db.dialect)

    inputs = {
        "input": lambda x: x["question"] + "\nSQLQuery: ",
        "table_info": lambda x: db.get_table_info(
            table_names=x.get("table_names_to_use")
        ),
        "relevant_msg": lambda x: "".join(chroma_vs.get_related_documentation(question=x["question"])),
    }
    return (
        RunnablePassthrough.assign(**inputs)  # type: ignore
        |
        (
            lambda x: {
                k: v
                for k, v in x.items()
                if k not in ("question", "table_names_to_use")
            }
        )
        | prompt_to_use.partial(top_k=str(k))
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
        | _strip
    )
