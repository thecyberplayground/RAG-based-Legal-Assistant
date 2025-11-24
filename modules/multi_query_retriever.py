## import dependencies
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from modules.conversation_history import ConversationSummaryMemory
from typing import List
from langchain.schema import Document
from modules.rrf_score import RRF

## multi query prompt
with open("./prompts/multiQuery-prompt.md", "r") as f:
    multiquery_template = f.read()

## defined schema for the multiquery queries
class MultiQuerySchema(BaseModel):
        generatedQueries: List[str] = Field(
            default=[], 
            description="List of semantically equivalent query variations for improved retrieval coverage. Empty list if documentRetrievalRequired is False. Contains only the original query if it's straightforward, or 5 alternative phrasings if the query is complex/ambiguous."
        )

## defined class
class MultiQueryRetriever:
    def __init__(
            self,
            model,
            memory: ConversationSummaryMemory,
            bm25_retriever,
            semantic_retriever,
            multiquery_prompt_template: str = multiquery_template,
            top_k=3,
            output_schema = MultiQuerySchema
    ) -> None:
        self.model = model
        self.memory = memory
        # self.user_query = user_query
        self.bm25_retriever = bm25_retriever
        self.semantic_retriever = semantic_retriever
        self.multiquery_prompt_template = multiquery_prompt_template
        self.output_schema = output_schema
        self.top_k = top_k
        self.document_retrieval_required = False

        self.sub_queries: List[str] = []
        self.all_documents: List[List[Document]] = []
    
    def __create_multiquery_chain(self):
        parser = JsonOutputParser(pydantic_object=self.output_schema)
        prompt = PromptTemplate(
            template=self.multiquery_prompt_template ,
            input_variables=["chat_history", "user_query"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        chain = prompt | self.model | parser
        return chain
    
    def __translate_queries(self, user_query) -> None:
        chain = self.__create_multiquery_chain()
        resp = chain.invoke({"chat_history": self.memory, "user_query": user_query})
        self.sub_queries.extend(resp.get("generatedQueries", [""]) if isinstance(resp, dict) else [""])

    def __retrieve(self) -> None:
        for query in self.sub_queries:
            # print(f"> Subquery: {query}")
            relevant_docs = self.bm25_retriever.invoke(query) + self.semantic_retriever.invoke(query) ## List[Document]
            self.all_documents.append(relevant_docs)

    def invoke(self, user_query: str) -> List[Document]:
        self.__translate_queries(user_query)
        self.__retrieve()
        rrf = RRF(self.all_documents)
        best_docs = rrf.rearrange(top_k=self.top_k)
        return best_docs

##--------------------- test the module -------------------------------
# if __name__ == "__main__":
#     from langchain_core.messages import HumanMessage, AIMessage
#     from langchain_ollama import ChatOllama
#     from modules.preprocess_documents import load_chunk_store
#     from modules.bm25_retriever import instantiate_bm25retriever
#     from modules.semantic_retriever import SemanticRetriever
#     from langchain_huggingface.embeddings import HuggingFaceEmbeddings
#
#     ## setup PATH
#     INPUT_DATA_PATH = "./data"
#     OUTPUT_DATA_PATH = "./data-ingestion-local"
#
#     ## set up environment
#     import os
#     from dotenv import load_dotenv
#     load_dotenv()
#     cohere_key = os.getenv("COHERE_API_KEY", None)
#     assert cohere_key, "[ATTENTION] Cohere API key is required."
#     os.environ['COHERE_API_KEY'] = cohere_key
#
#     dummy_chat_history = [
#         HumanMessage(content="I'm studying the Special Marriage Act for a legal automation project."),
#         AIMessage(
#             content="Nice! The Act has many procedural layers — notice, objections, inquiry, solemnization, registration, and even penalties. What aspects are you focusing on?"),
#         HumanMessage(
#             content="Mainly the notice and objection procedures, especially how Section 5 notice publication works and what Marriage Officers must do if an objection is raised."),
#         AIMessage(
#             content="Good area. Section 6 requires the Marriage Officer to keep the notice open for public inspection, and Section 7 allows any person to object on grounds that the marriage violates Section 4 conditions."),
#         HumanMessage(
#             content="Yes, but I’m also trying to understand how the timelines interact — especially the 30-day waiting period and how the inquiry under Section 8 should be conducted."),
#         AIMessage(
#             content="Right. Section 8 gives the Marriage Officer civil-court powers for inquiry, and Section 9 bars solemnization until the objection is resolved. The 30-day window under Section 7 acts as the minimum period before solemnization."),
#         HumanMessage(
#             content="Great. I also want to test how my system handles multi-step reasoning across Sections 4, 5, 6, 7, and 8."),
#         AIMessage(content="Sure. Tell me the exact kind of query you want me to generate.")
#     ]
#
#     user_query = (
#         "If someone objects to a Section 5 marriage notice claiming the groom already has a "
#         "living spouse, what sections must the Marriage Officer consult to decide the objection, "
#         "and can the marriage be solemnized before the inquiry is finished?"
#     )
#     llm = ChatOllama(model="qwen3:latest")
#
#     ## create chat history
#     memory = ConversationSummaryMemory(model=llm, k=2)
#     for item in dummy_chat_history:
#         memory.append(item)
#
#     ## retrievers
#     preprocessed_docs = load_chunk_store(data_path=INPUT_DATA_PATH)
#     bm25_retriever = instantiate_bm25retriever(documents=preprocessed_docs)
#
#     embedF = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     semantic_retriever = SemanticRetriever(
#         embedding_function=embedF,
#         prepped_docs=preprocessed_docs,
#         vectordb_output_path=OUTPUT_DATA_PATH
#     ).retriever
#
#     ## multiquery
#     mq = MultiQueryRetriever(model=llm, memory=memory, top_k=5, bm25_retriever=bm25_retriever, semantic_retriever=semantic_retriever)
#     print(f"> Best Docs:\n\n{mq.invoke(user_query=user_query)}")