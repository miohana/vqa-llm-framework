# VQA Evaluation Beyond Accuracy: Leveraging Language Models for Semantic Assessment
We propose the creation of a framework that utilizes advanced language models to act as judges in the automatic evaluation of these tasks, offering a richer and more flexible perspective for this assessment. The framework aims to handle subtle variations in meaning that may be overlooked in traditional evaluations.

<img width="1200" alt="High Level Diagram" src="https://github.com/user-attachments/assets/66f99bf4-2e12-44b7-a1ab-e0e14ebbc4a9">


# Package

Sample for using `vragas` package:
```python
from vragas.metrics import F1Score

data = {
    "id": 1,
    "user_input": "What is the capital of France?",
    "response": "Paris",
    "references": ["Paris"],
}

f1_score = F1Score(reduction="mean")
f1_score.update(data)
f1_score.compute()
# Output: {"f1": 1.0}
```


# Developing

Requirements:
 - [PDM](https://pdm-project.org/en/latest/)
 - Python 3.12 or higher
 - In case of using Hugging Face models (e.g. PaliGemma), remember to set the HF_TOKEN (currently, we are expecting you to setup in our code).

## Setup
Run the command to install production and development dependencies:
```bash
pdm sync
```
