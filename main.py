from src.vragas.metrics import F1Score

data = {
    "id": 1,
    "user_input": "What is the capital of France?",
    "response": "Paris",
    "references": ["Paris"],
}

f1_score = F1Score(reduction="mean")
f1_score.update(data)
result = f1_score.compute()
print(result)
# Output: {"f1": 1.0}