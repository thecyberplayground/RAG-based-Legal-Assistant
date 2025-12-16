## import dependencies
from datasets import Dataset
from ragas import evaluate
from modules.language_model import ragas_eval_llm, ragas_eval_embedF
from ragas.metrics import faithfulness, answer_correctness
from ragas.run_config import RunConfig
from ast import literal_eval

# Load as Dataset, then convert to pandas
dataset = Dataset.from_csv("RAGAS-dataset/eval-dataset-final.csv")
df = dataset.to_pandas()

# Now you can use apply
df['retrieved_contexts'] = df['retrieved_contexts'].apply(literal_eval)

# Convert back to Dataset if needed
dataset = Dataset.from_pandas(df)

## evaluation
score = evaluate(
    dataset=dataset,
    llm=ragas_eval_llm,
    embeddings=ragas_eval_embedF,
    metrics=[faithfulness, answer_correctness],
    run_config=RunConfig(timeout=1200)
)
df = score.to_pandas()
df.to_csv('./RAGAS-dataset/eval-score.csv', index=False)