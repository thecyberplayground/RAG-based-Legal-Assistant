## settings up the env
import os
from dotenv import load_dotenv
load_dotenv()

## langchain dependencies
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List
from langchain.schema import Document


class SemanticRetriever:
    """
    Semantic Retriever for FAISS:
    -> create an instance with embedding mode, prepped documents and output path to save the vector store locally
    -> access the semantic retriever by .retriever
    -> invoke by .retriever.invoke(user_query)
    """
    def __init__(
            self,
            embedding_function: HuggingFaceEmbeddings,
            prepped_docs: List[Document],
            vectordb_output_path: str
    ) -> None:
        self.embedding_function = embedding_function
        self.prepped_docs = prepped_docs
        self.vectordb_output_path = os.path.abspath(vectordb_output_path)

    def __build_vectordb(self) -> None:
        vector_db = FAISS.from_documents(documents=self.prepped_docs, embedding=self.embedding_function)
        vector_db.save_local(self.vectordb_output_path)

    @property
    def retriever(self):
        if not os.path.exists(self.vectordb_output_path):
            self.__build_vectordb()
        ## if already exists, skip building
        vector_db = FAISS.load_local(self.vectordb_output_path, embeddings=self.embedding_function, allow_dangerous_deserialization=True)
        semantic_retriever = vector_db.as_retriever(search_type="similarity",search_kwargs={"k": 5})
        return semantic_retriever