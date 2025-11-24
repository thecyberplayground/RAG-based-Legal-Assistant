## import langchain dependencies
from langchain_community.retrievers import BM25Retriever
from nltk.tokenize import word_tokenize
from modules.preprocess_documents import load_chunk_store

## BM25 retriever
def instantiate_bm25retriever(documents, tokenizer = None):
    if not tokenizer:
        return BM25Retriever.from_documents(documents=documents, k=3)
    else:
        return BM25Retriever.from_documents(documents=documents, preprocess_func=tokenizer, k=3)

# ---------------------------------------------------------------------------------------------------------
if __name__=="__main__":
    ## set up directories
    # current_dir_path = os.path.dirname(os.path.abspath(__file__)) ## extract the directory name from the absolute path of this file
    # data_path = os.path.join(current_dir_path, "data") ## create path for the `data` folder
    
    ## load and chunk
    preprocessed_docs = load_chunk_store(data_path="./data")

    ## BM25 retriever
    bm25_retriever = instantiate_bm25retriever(documents=preprocessed_docs, tokenizer=word_tokenize)

    ## user query
    user_query = input("> Human: ")

    ## result
    retrieved_docs = bm25_retriever.invoke(user_query)

    ## print result
    # print(f"\n\n> Retriever Docs: ")
    # for index, docs in enumerate(retrieved_docs, start=1):
    #     print(f"{index}:\n\n\n{docs.page_content}")
    print(f"Output:{type(retrieved_docs)}")
    print(f"Output:{type(retrieved_docs[0])}")