import numpy as np
from typing import Dict, List
from sentence_transformers import SentenceTransformer

from .base import Id, Metric, EvalInput, MetricValue


class EmbeddingSimilarity(Metric):
    """
        Calculate cosine similarity between two sentences
    """

    name = "embedding_similarity"

    def __init__(
        self,
        model: str,
        zero_min: bool = True,
        **stransformers_kwargs
    ) -> None:
        super().__init__()
        self.model = SentenceTransformer(
            model,
            **stransformers_kwargs,
            similarity_fn_name="cosine"
        )
        self.zero_min = zero_min

    def score(self, text: str, references: List[str]) -> float:
        embeddings = self.model.encode([text, *references])
        text_embedding, embeddings = embeddings[0], embeddings[1:]
        text_embedding = np.tile(text_embedding, (len(embeddings), 1))
        similarity = self.model.similarity_pairwise(text_embedding, embeddings)
        if self.zero_min:
            similarity = similarity.relu()

        return similarity.mean().item()

    def update(self, input: EvalInput) -> None:
        similarity = self.score(input["response"], input["references"])
        self.state[input["id"]] = similarity

    def report(self) -> Dict[Id, float]:
        return self.state.copy()

    def compute(self) -> MetricValue:
        values = list(self.state.values())
        return {
            self.name: float(np.mean(values))
        }
