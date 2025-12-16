## import dependencies
import time
# from langchain_cohere import ChatCohere
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from typing import List
from langchain.schema import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from modules.conversation_history import ConversationSummaryMemory
from modules.multi_query_retriever import MultiQueryRetriever
from modules.chatbot_response import ChatbotResponse
from modules.preprocess_documents import load_chunk_store
from modules.bm25_retriever import instantiate_bm25retriever
from modules.semantic_retriever import SemanticRetriever
from modules.multi_hop_retriever import MultiHopRetriever
from modules.decide_query_complexity import QueryComplexity

## supress langchain warning
# import warnings
# from langchain_core._api import LangChainBetaWarning
# warnings.filterwarnings("ignore", category=LangChainBetaWarning)

## READ THE EVAL DOC
import pandas as pd
eval_df = pd.read_csv("../RAGAS-dataset/eval-dataset.csv")


## environment setup
import os
from dotenv import load_dotenv
load_dotenv()
cohere_key = os.getenv("COHERE_API_KEY", None)
assert cohere_key, "[ATTENTION] COHERE API KEY IS MISSING!"
os.environ["COHERE_API_KEY"] = cohere_key

## setup PATH
INPUT_DATA_PATH = "../data"
OUTPUT_DATA_PATH = "../data-ingestion-local"

## load the model
llm = ChatOllama(model="qwen3:latest")

## conversation memory
chat_history = ConversationSummaryMemory(model=llm, k=3)

## main RAG prompt template
with open("../prompts/mainRAG-prompt.md") as f:
    rag_prompt_template = f.read()

## embedding model
embedF = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")

## process documents
preprocessed_docs = load_chunk_store(data_path=INPUT_DATA_PATH)

## set up retrievers - bm25, semantic and multi-hop(hybrid)
bm25_retriever = instantiate_bm25retriever(documents=preprocessed_docs)

semantic_retriever = SemanticRetriever(
    embedding_function=embedF,
    prepped_docs=preprocessed_docs,
    vectordb_output_path=OUTPUT_DATA_PATH
).retriever

## ---- START ----
welcome_message = "Welcome to the Legal Assistant Bot. How can I help you today? Write `exit` to quit."
chat_history.append(AIMessage(content=welcome_message))

## ---- SAVE ANSWER AND DOCS ----
answers: List[str] = []
master_retrieved_docs_container: List[List[Document]] = []

## ---- ANSWER GENERATION PART ----
for index, (_, row) in enumerate(eval_df.iterrows(), start=1):
    ## containers
    retrieved_docs = []
    subquery_plus_docs = ""

    print(f"> User Query[{index}]: {row['question']}")
    user_query = row['question']
    chat_history.append(HumanMessage(content=user_query))

    ## query complexity
    complexity_tier_obj = QueryComplexity(user_query=user_query, memory=chat_history, model=llm)
    complexity_tier = complexity_tier_obj.decide_complexity

    ## retrievers
    multiquery_retriever = MultiQueryRetriever(
        model=llm,
        memory=chat_history,
        bm25_retriever=bm25_retriever,
        semantic_retriever=semantic_retriever
    )

    multihop_retriever = MultiHopRetriever(
        model=llm,
        memory=chat_history,
        bm25_retriever=bm25_retriever,
        semantic_retriever=semantic_retriever
    )

    ## fetch document based on the retriever type decided from `complexity_tier`
    if complexity_tier == "complex":
        retrieved_docs.extend(multiquery_retriever.invoke(user_query))
        master_retrieved_docs_container.append([doc.page_content for doc in retrieved_docs])
    elif complexity_tier == "multi-hop":
        subquery_plus_docs += multihop_retriever.invoke(user_query)
        master_retrieved_docs_container.append(multihop_retriever.retrieved_respective_documents)

    ## response generation
    resp = ChatbotResponse(
        model=llm,
        memory=chat_history,
        rag_prompt_template=rag_prompt_template,
        subquery_docs=subquery_plus_docs,
        documents=None if len(retrieved_docs) == 0 else retrieved_docs,
    )
    ai_resp = resp.generate_response()

    print(f"> AI: ({complexity_tier}) {ai_resp}")
    print(f"> Context: {resp.context}")
    break