from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ChatbotResponse:
    def __init__(self, model, memory, rag_prompt_template, subquery_docs = "", documents = None) -> None:
        self.memory = memory
        self.model = model
        self.documents = documents
        self.subquery_docs = subquery_docs
        self.rag_prompt_template = rag_prompt_template
        # self.context = ""

    def generate_response(self) -> str:
        if not self.documents:
            context = ""
        else:
            context = "\n"
            for rank, doc in enumerate(self.documents, start=1):
                context += f"Relevant Document: {rank}" + doc.page_content + "\n"
        prompt = ChatPromptTemplate.from_template(self.rag_prompt_template)
        chain = prompt | self.model | StrOutputParser()
        resp = chain.invoke({
            "chat_history": self.memory,
            "context": context,
            "multi_hop_context": self.subquery_docs
        })
        return resp

    # def generate_response(self) -> str:
    #     if self.documents:
    #         self.context += "\n"
    #         for rank, doc in enumerate(self.documents, start=1):
    #             self.context += f"Relevant Document: {rank}" + doc.page_content + "\n"
    #     prompt = ChatPromptTemplate.from_template(self.rag_prompt_template)
    #     chain = prompt | self.model | StrOutputParser()
    #     resp = chain.invoke({
    #         "chat_history": self.memory,
    #         "context": self.context,
    #         "multi_hop_context": self.subquery_docs
    #     })
    #     return resp