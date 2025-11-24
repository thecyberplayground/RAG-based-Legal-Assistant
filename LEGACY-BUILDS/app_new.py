## set up environment
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['COHERE_API_KEY'] = os.getenv("COHERE_API_KEY")

## import dependencies
from langchain_cohere import ChatCohere
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from modules.conversation_history import ConversationSummaryMemory
from modules.multi_query_retriever import MultiQuery
from modules.rrf_score import RRF
from modules.chatbot_response import ChatbotResponse

## other dependencies
import time

## supress langchain warning
import warnings
from langchain_core._api import LangChainBetaWarning

warnings.filterwarnings("ignore", category=LangChainBetaWarning)

## setting up file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "data")
persistent_directory = os.path.join(current_dir, "data-ingestion-local")

## instantiate embedding model
embedF = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

## load local vector DB
# vectorDB = FAISS.load_local(persistent_directory, embeddings=embedF, allow_dangerous_deserialization=True)
vectorDB = Chroma(embedding_function=embedF, persist_directory=persistent_directory)

## set up retriever
kb_retriever = vectorDB.as_retriever(search_type="similarity",search_kwargs={"k": 5})

## main RAG prompt template
with open("../prompts/mainRAG-prompt.md") as f:
    rag_prompt_template = f.read()
with open("../prompts/multiQuery-prompt.md") as f:
    multi_query_prompt_template = f.read()

if __name__=="__main__":
    llm = ChatCohere(temperature=0.15)

    ## initial chat state
    chat_history = ConversationSummaryMemory(model=llm, k=2)
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

            chat_history.append(HumanMessage(content=user_query)) ## store it in the history
            
            ## multiquery
            mq = MultiQuery(model=llm, memory=chat_history, user_query=user_query, multiquery_template= multi_query_prompt_template)
            multiqueries = mq.multiqueries()

            if mq.document_retrieval_required:
                ## retrieve documents for each query
                for query in multiqueries:
                    retrieved_docs.append(kb_retriever.invoke(query))
                
                ## RRF score and rearrange documents in decending order
                rrf = RRF(documents=retrieved_docs)
                ranked_documents = rrf.rearrange(top_k=5)

                ## response generation
                resp = ChatbotResponse(
                    model=llm,
                    memory=chat_history,
                    rag_prompt_template=rag_prompt_template,
                    documents=ranked_documents
                )
                ai_resp = resp.generate_response()
            else:
                ## response generation
                resp = ChatbotResponse(
                    model=llm,
                    memory=chat_history,
                    rag_prompt_template=rag_prompt_template
                )
                ai_resp = resp.generate_response()
            
            ## print AI response
            print("\033[1m> AI:\033[0m ", end="")
            for letter in ai_resp:
                print(letter, flush=True, end="")
                time.sleep(0.03)
            
            ## append AI response to chat history
            chat_history.append(AIMessage(content=ai_resp))
        except KeyboardInterrupt:
            print("\n\n[ALERT] ^C: Keyboard Interruption detected. Session Terminated!!")
            break