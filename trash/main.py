## import dependencies
import time
# from langchain_cohere import ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
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
INPUT_DATA_PATH = "../data"
OUTPUT_DATA_PATH = "../data-ingestion-local"

## load the model
# llm = ChatCohere()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite"
)
# llm = ChatOllama(model="qwen3:latest")

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

## print welcome message
print("\033[1m> AI:\033[0m ", end="")
for letter in welcome_message:
    print(letter, flush=True, end="")
    time.sleep(0.03)

### ------------- CONVERSATION LOOP ----------------
while True:
    try:
        retrieved_docs = []
        subquery_plus_docs = ""

        ## user input
        print("\n\033[1m> Human:\033[0m ", end="")
        user_query = input().strip()

        ## check if user wants to exit
        if user_query.lower().strip() == "exit":
            bye_message = "Chat ended! See ya."
            print("\033[1m> AI:\033[0m ", end="")
            for letter in bye_message:
                print(letter, flush=True, end="")
                time.sleep(0.03)
            break

        ## fetch complexity level
        complexity_tier_obj = QueryComplexity(user_query=user_query, memory=chat_history, model=llm)
        complexity_tier = complexity_tier_obj.decide_complexity

        ## instantiate retrievers for every turn
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
        elif complexity_tier == "multi-hop":
            subquery_plus_docs += multihop_retriever.invoke(user_query)

        ## STORE IN CHAT MEMORY
        chat_history.append(HumanMessage(content=user_query))  ## store it in the history

        # print(f">>> {complexity_tier}") ## for debugging
        ## response generation
        resp = ChatbotResponse(
            model=llm,
            memory=chat_history,
            rag_prompt_template=rag_prompt_template,
            subquery_docs=subquery_plus_docs,
            documents=None if len(retrieved_docs) == 0 else retrieved_docs,
        )
        ai_resp = resp.generate_response()

        ## print AI response
        print(f"\033[1m> AI:\033[0m ({complexity_tier}) ", end="")
        for letter in ai_resp:
            print(letter, flush=True, end="")
            time.sleep(0.03)

        ## append AI response to chat history
        chat_history.append(AIMessage(content=ai_resp))
    except KeyboardInterrupt:
        print("\n\n[ALERT] ^C: Keyboard Interruption detected. Session Terminated!!")
        break