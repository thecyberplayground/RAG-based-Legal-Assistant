## import dependencies
import time
# from langchain_cohere import ChatCohere
# from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from modules.multi_query_retriever import MultiQueryRetriever
from modules.preprocess_documents import load_chunk_store
from modules.bm25_retriever import instantiate_bm25retriever
from modules.semantic_retriever import SemanticRetriever
from modules.multi_hop_retriever import MultiHopRetriever
from modules.decide_query_complexity import QueryComplexity
import pandas as pd
from typing import List

## supress langchain warning
import warnings
from langchain_core._api import LangChainBetaWarning
warnings.filterwarnings("ignore", category=LangChainBetaWarning)

## environment setup
import os
from dotenv import load_dotenv
load_dotenv()
google_key = os.getenv("GOOGLE_API_KEY", None)
assert google_key, "[ATTENTION] GOOGLE API KEY IS MISSING!"
os.environ["GOOGLE_API_KEY"] = google_key

## setup PATH
INPUT_DATA_PATH = "./data"
OUTPUT_DATA_PATH = "./data-ingestion-local"

## load the model
# llm = ChatCohere()
# llm = ChatOllama(model="qwen3:latest")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite"
)
# chat_history = ConversationSummaryMemory(model=llm, k=3)

## setup prompt
with open("./prompts/mainRAG-prompt-eval.md") as f:
    eval_rag_prompt = f.read()

## chatbot response
class ChatbotResponse:
    def __init__(self, model, user_query, rag_prompt_template, subquery_docs = "", documents = None) -> None:
        self.model = model
        self.user_query = user_query
        self.rag_prompt_template = rag_prompt_template
        self.subquery_docs = subquery_docs
        self.documents = documents

        self.context = ""
    def generate_response(self) -> str:
        if self.documents:
            self.context += "\n"
            for rank, doc in enumerate(self.documents, start=1):
                self.context += f"Relevant Document: {rank}" + doc.page_content + "\n"
        prompt = ChatPromptTemplate.from_template(self.rag_prompt_template)
        chain = prompt | self.model | StrOutputParser()
        resp = chain.invoke({
            "user_query": self.user_query,
            "context": self.context,
            "multi_hop_context": self.subquery_docs
        })
        return resp

## embedding model
embedF = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## process documents
preprocessed_docs = load_chunk_store(data_path=INPUT_DATA_PATH)

## set up retrievers - bm25, semantic and multi-hop(hybrid)
bm25_retriever = instantiate_bm25retriever(documents=preprocessed_docs)

semantic_retriever = SemanticRetriever(
    embedding_function=embedF,
    prepped_docs=preprocessed_docs,
    vectordb_output_path=OUTPUT_DATA_PATH
).retriever

multiquery_retriever = MultiQueryRetriever(
    model=llm,
    memory="",
    bm25_retriever=bm25_retriever,
    semantic_retriever=semantic_retriever
)

multihop_retriever = MultiHopRetriever(
    model=llm,
    memory="",
    bm25_retriever=bm25_retriever,
    semantic_retriever=semantic_retriever
)

### ------------- EVALUATION LOOP ----------------
eval_dataset = pd.read_csv("./RAGAS-dataset/eval-dataset.csv")
answers: List[str] = [] ## AI responses for the questions asked
master_retrieved_docs_container: List[List[str]] = [] ## an array containing list of documents for each query
try:
    for index, row in eval_dataset.iterrows():
        print(f"> Fetching answer for Q: {index+1}")
        retrieved_docs = []
        subquery_plus_docs = ""

        user_query = row["question"]
        print(f"> User: {user_query}")

        ## fetch complexity level
        complexity_tier_obj = QueryComplexity(user_query=user_query, memory="", model=llm)
        complexity_tier = complexity_tier_obj.decide_complexity

        ## fetch document based on the retriever type decided from `complexity_tier`
        if complexity_tier == "complex":
            retrieved_docs.extend(multiquery_retriever.invoke(user_query))
        elif complexity_tier == "super_complex":
            subquery_plus_docs += multihop_retriever.invoke(user_query)

        ## response generation
        resp = ChatbotResponse(
            model=llm,
            user_query=user_query,
            rag_prompt_template=eval_rag_prompt,
            subquery_docs=subquery_plus_docs,
            documents=None if len(retrieved_docs) == 0 else retrieved_docs,
        )
        ai_resp = resp.generate_response()
        print(f"> AI: (complexity_tier) {ai_resp}")
        print(f"> Context Injected: {resp.context}")
        answers.append(ai_resp) ## store the answers

        ## append documents according to the retriever type -> append item should be List[str]
        if complexity_tier == "complex":
            master_retrieved_docs_container.append([doc.page_content for doc in retrieved_docs])
        elif complexity_tier == "super_complex":
            master_retrieved_docs_container.append([doc.page_content for doc in multihop_retriever.retrieved_respective_documents])

        time.sleep(60)
except Exception as e:
    print(e)
finally:
    if len(master_retrieved_docs_container) == len(eval_dataset):
        eval_dataset['answer'] = answers
        eval_dataset['retrieved_contexts'] = master_retrieved_docs_container
        print(eval_dataset)
        eval_dataset.to_csv("./RAGAS-dataset/eval-dataset-final-2.csv", index=False)
    else:
        print(f"[ATTENTION] Incomplete Generation.\n> Documents Retriever: {len(master_retrieved_docs_container)}\n> Answers Generated: {len(answers)}")