## import dependencies
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

def load_chunk_store(data_path: str):
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"[ALERT] {data_path} doesn't exist. ⚠️⚠️"
        )

    ## list of all the PDFs
    pdfs = [pdf for pdf in os.listdir(data_path) if pdf.endswith(".pdf")] ## list of all file names as str that ends with `.pdf`

    doc_container = [] ## list of chunked documents

    ## take each item from `pdfs` and load it using PyPDFLoader
    for pdf in pdfs:
        loader = PyPDFLoader(file_path=os.path.join("data", pdf),
                                extract_images=False)
        docs_raw = loader.load() ## list of `Document` objects. Each such object has - 1. Page Content // 2. Metadata
        for doc in docs_raw:
            doc_container.append(doc) ## append each `Document` object to the previously declared container

    ## split the documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    prepped_docs = splitter.split_documents(documents=doc_container)
    return prepped_docs