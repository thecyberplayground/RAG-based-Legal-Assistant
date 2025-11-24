## import dependencies
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Literal

## prompt template
with open("./prompts/decide_query_complexity.md", "r", encoding="utf-8") as file:
    complexity_prompt = file.read()

class QueryComplexitySchema(BaseModel):
    complexity: Literal["simple_conversation", "complex", "multi_hop"] = Field(
        default="simple_conversation",
        description="""
        Indicates how the system should reason about and retrieve context for the user's query.

        `simple_conversation`:
        Purely conversational inputs—greetings, acknowledgments, chit-chat, or general dialogue.
        No retrieval or document processing is required.

        `complex`:
        Any user query that must be answered factually using the provided context or knowledge base.
        These queries trigger a MultiQuery retrieval workflow, where multiple reformulated queries
        are generated, documents are retrieved for each, and Reciprocal Rank Fusion (RRF) is applied
        to re-rank and optimize the final document set before answering.

        `multi_hop`:
        Queries requiring layered or chained reasoning across multiple distinct facts or documents.
        Retrieval occurs iteratively—each hop uses previously retrieved information to guide deeper,
        context-dependent discovery and inference.
        """
    )


class QueryComplexity:
    def __init__(
            self,
            user_query: str,
            memory,
            model,
            template: str = complexity_prompt,
            pydantic_schema=QueryComplexitySchema
    ):
        self.user_query = user_query
        self.memory = memory
        self.template = template
        self.model = model
        self.query_complexity_schema = pydantic_schema

    @staticmethod
    def __initiate_chain(template, model, pydantic_schema):
        parser = JsonOutputParser(pydantic_object=pydantic_schema)
        prompt = PromptTemplate(
            template=template,
            input_variables=["user_query", "memory"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        chain = prompt | model | parser
        return chain

    @property
    def decide_complexity(self):
        chain = self.__initiate_chain(
            template=self.template,
            model=self.model,
            pydantic_schema=self.query_complexity_schema
        )
        resp = chain.invoke(
            {
                "user_query": self.user_query,
                "memory": self.memory,
            }
        )
        return resp["complexity"]


## --- TEST EXECUTION ---
if __name__ == "__main__":
    pass