from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Union, Any

## class for conversation memory
class ConversationSummaryMemory:
    def __init__(self, model: Any, k: int) -> None:
        self.conversations = []
        self.model = model
        self.window = k*2
        self._summary_cache = ""
    
    ## windowed conversation list
    @property
    def windowed_conversation(self):
        if len(self.conversations) > self.window:
            return self.conversations[-self.window:]
        return self.conversations

    ## method to pretty print
    @staticmethod
    def __pretty_print(conversations) -> str:
        entire_conversation = []
        for turn in conversations:
            entire_conversation.append(f"> {turn.type}: {turn.content}")
        return "\n".join(entire_conversation)

    ## private method for initiating a langchain chain
    def __initiate_chain(self):
        template = """
        You are an expert in summarizing conversations between a human and an AI chatbot. Your task is to write a short, clear summary that keeps the main facts, tone, and context of the conversation intact while removing unnecessary or repetitive parts. The summary should be concise but informative enough for another system to understand what the conversation was about and its overall direction.

        Input Conversation:
        {conversation}

        Output:
        A brief, well-written paragraph summarizing the conversation.
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["conversation"]
        )
        chain = prompt | self.model | StrOutputParser()
        return chain
    
    ## method to add conversation one-by-one
    def append(self, conversation: Union[AIMessage, HumanMessage]) -> None:
        self.conversations.append(conversation)
        if len(self.conversations) > self.window:
            prev = self._summary_cache
            total_conversation_context = (
                "Previous Conversation Summary: " + prev
                + "\nLatest Turns:\n" + self.__pretty_print(self.windowed_conversation)
            )
            chain = self.__initiate_chain()
            self._summary_cache = chain.invoke({"conversation": total_conversation_context})
        # else:
        #     total_conversation_context = self.__pretty_print(self.conversations)

    ## generate summary
    @property
    def summary(self) -> str:
        return self._summary_cache
    
    ## object print
    def __str__(self) -> str:
      if len(self.conversations) > self.window:
        return f"Last {self.window} turns of the Conversation:\n\n{self.__pretty_print(self.windowed_conversation)}\n\nPrevious Conversation Summary: {self.summary}"
      return f"Latest Conversation:\n\n{self.__pretty_print(self.windowed_conversation)}"