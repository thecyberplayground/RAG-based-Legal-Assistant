## import dependencies
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from modules.rrf_score import RRF

with open("./prompts/multihop-prompt.md", "r", encoding="utf-8") as f:
    multihop_template = f.read()

class SubQuery(BaseModel):
    end_of_generation: bool = Field(
        default=False,
        description="""
        Indicates whether further query decomposition is required in a multi-hop
        retrieval process. Set this to True when the system has generated enough
        subqueries and gathered sufficient supporting documents, signaling that
        no additional subquery generation is needed.
        """
    )
    subquery: str = Field(
        default="",
        description="""
        The individual subquery derived from the user's original query. Each
        subquery should represent a logically independent step that contributes
        to answering the overall multi-hop question.
        """
    )

## define multi-hop retriever class
class MultiHopRetriever:
    def __init__(
            self,
            memory,
            model,
            bm25_retriever,
            semantic_retriever,
            multi_hop_prompt_template: str = multihop_template,
            pydantic_schema = SubQuery
    ):
        self.memory = memory
        self.multi_hop_prompt_template = multi_hop_prompt_template
        self.model = model
        self.bm25_retriever = bm25_retriever
        self.semantic_retriever = semantic_retriever
        self.pydantic_schema = pydantic_schema
        self.end_of_generation = False

        ## queries and documents
        self.subqueries = []
        self.retrieved_respective_documents = []

    def generate_subquery_content(self):
        starting_substring = "### Subqueries + Retrieved Documents\n"
        content = ""
        for index, (subquery, list_of_documents) in enumerate({subquery: list_of_docs for subquery, list_of_docs in zip(self.subqueries, self.retrieved_respective_documents)}.items(), start=1):
            content += f"{index}. Subquery: {subquery}\n-> Relevant Documents:\n{'\n'.join(doc.page_content for doc in list_of_documents)}"
            content +="\n"
        return starting_substring+content

    @staticmethod
    def __initiate_chain(template, model, pydantic_schema):
        parser = JsonOutputParser(pydantic_object=pydantic_schema)
        prompt = PromptTemplate(
            template=template,
            input_variables=["max_iteration_allowed", "user_query", "memory", "subqueries_and_relevant_documents"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        chain = prompt | model | parser
        return chain

    def invoke(self, user_query, max_iteration_allowed=5):
        for _ in range(max_iteration_allowed):
            # print(f"> Try: {i}") ## for debugging
            chain = self.__initiate_chain(
                template=self.multi_hop_prompt_template,
                model=self.model,
                pydantic_schema=self.pydantic_schema
            )
            resp = chain.invoke(
                {
                    "memory": self.memory,
                    "user_query": user_query,
                    "max_iteration_allowed": max_iteration_allowed,
                    "subqueries_and_relevant_documents": self.generate_subquery_content() if len(self.subqueries) > 0 else ""
                }
            )

            ## for debugging
            # print(f"> EoG Status: {resp['end_of_generation']}")
            # print(f"> Subquery: {resp['subquery']}")

            if not resp['end_of_generation']:
                """
                If the LLM has to continue generating more
                subqueries:
                -> append every subquery
                -> retrieve relevant documents using both retriever
                -> pack it in a List[List[Document]]
                -> fetch best documents via RRF
                -> append it to `retrieved_respective_documents`
                """
                self.subqueries.append(resp['subquery'])
                rrf = RRF(
                    [
                        self.bm25_retriever.invoke(resp['subquery']),  ## output -> List[Document]
                        self.semantic_retriever.invoke(resp['subquery']),  ## output -> List[Document]
                    ]  ## becomes: List[List[Document]]
                )
                best_relevant_docs = rrf.rearrange(top_k=3)
                self.retrieved_respective_documents.append(best_relevant_docs)
            else:
                break
        ## fetch subquery -> fetch relevant docs -> generate content -> return content
        return self.generate_subquery_content()

##---- TESTING EXECUTION ----
# if __name__ == "__main__":
#     from langchain_cohere import ChatCohere
#     # from langchain_ollama import ChatOllama
#     from modules.bm25_retriever import instantiate_bm25retriever
#     from modules.semantic_retriever import SemanticRetriever
#     from modules.preprocess_documents import load_chunk_store
#     from langchain_huggingface.embeddings import HuggingFaceEmbeddings
#
#     INPUT_DATA_PATH = "./data"
#     OUTPUT_DATA_PATH = "./data-ingestion-local"
#
#     import os
#     from dotenv import load_dotenv
#     load_dotenv()
#     cohere_key = os.getenv("COHERE_API_KEY", None)
#     assert cohere_key, "[ATTENTION] COHERE API KEY IS MISSING!"
#     os.environ["COHERE_API_KEY"] = cohere_key
#     #
#     # ## ---- TEST FOR THE CONTENT GENERATION PART ----
#     # # original_user_query = "Hey man how are you doing?"
#     # # subqueries = [
#     # #     "hello joesph",
#     # #     "how are you",
#     # #     "im fine"
#     # # ]
#     # # relevant_docs = [
#     # #     [Document(page_content="doc1 content"), Document(page_content="doc2 content")],
#     # #     [Document("doc3 content"), Document("doc4 content")],
#     # #     [Document("doc5 content"), Document("doc6 content")]
#     # # ]
#     # # multihop_obj = MultiHopRetriever(
#     # #     user_query=original_user_query,
#     # #     memory="",
#     # #     multi_hop_prompt_template="",
#     # #     model=ChatCohere(),
#     # #     bm25_retriever= None,
#     # #     semantic_retriever= None
#     # # )
#     # #
#     # # multihop_obj.subqueries = subqueries
#     # # multihop_obj.retrieved_respective_documents = relevant_docs
#     # #
#     # # print(multihop_obj.generate_subquery_content())
#     #
#     ## ---- TEST FOR THE ENTIRE MULTI-HOP RETRIEVAL SYSTEM ----
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
#     # with open("./prompts/multihop-prompt.md", "r", encoding="utf-8") as f:
#     #     multihop_template = f.read()
#
#     original_user_query = """
#     Which section sets the age requirement, what are the exact ages required for male and female, and does C.D.’s age (17) permit any discretion or exception under the Act (including references to Schedules, provisos, or the Act’s explanations)?
#     """
#     multihop_retriever = MultiHopRetriever(
#         model=ChatCohere(),
#         memory="",
#         bm25_retriever=bm25_retriever,
#         semantic_retriever=semantic_retriever
#     )
#
#     resp = multihop_retriever.run(user_query=original_user_query, max_iteration_allowed=7)
#     print(resp)