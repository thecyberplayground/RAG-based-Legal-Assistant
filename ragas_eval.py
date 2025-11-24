from datasets import Dataset
import os
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness
from langchain_ollama import ChatOllama
from ast import literal_eval

## environment setup
# import os
# from dotenv import load_dotenv
# load_dotenv()
# cohere_key = os.getenv("COHERE_API_KEY", None)
# assert cohere_key, "[ATTENTION] COHERE API KEY IS MISSING!"
# os.environ["COHERE_API_KEY"] = cohere_key

## evaluation
dataset = Dataset.from_csv("./RAGAS-dataset/eval-dataset-final.csv")
dataset['retrieved_contexts'] = dataset['retrieved_contexts'].apply(literal_eval)
llm = ChatOllama(model="qwen3:latest")
score = evaluate(dataset, llm=llm, metrics=[faithfulness, answer_correctness])
df = score.to_pandas()
df.to_csv('./RAGAS-dataset/eval_score.csv', index=False)