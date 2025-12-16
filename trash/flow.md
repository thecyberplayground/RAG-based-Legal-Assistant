## Modules
### modules.preprocess_documents
- Take `document path` as input
- Check if the path actually exists:
  - if yes
    - list all the PDFs
    - load PDFs using PyPDFLoader
    - use RecursiveCharacterTextSplitter to chunk one by one
    - output -> List[Document]
  - if no
    - raise FileNotFound error

## modules.semantic_retriever
- there is no problem to use the same retriever instance multiple times to fetch documents for multiple queries in multiple turns

## modules.bm25_retriever
-  