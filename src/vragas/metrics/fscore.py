import string
from collections import Counter
from statistics import mean
from typing import Callable, List

from .base import EvalInput, MetricWithReduction, MetricValue


def tokenize(s: str) -> List[str]:
    # Normalize
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return s.lower().split()


class F1Score(MetricWithReduction):
    """
        F1 bag-of-words metric.
    """
    name = "f1"
    tokenizer: Callable[[str], List[str]] = tokenize

    def _f1(self, pred: str, ref: str) -> float:
        prediction_tokens = self.tokenizer(pred)
        reference_tokens = self.tokenizer(ref)
        common = Counter(prediction_tokens) & Counter(reference_tokens)
        intersection = sum(common.values())
        if intersection == 0:
            return 0

        precision = intersection / len(prediction_tokens)
        recall = intersection / len(reference_tokens)
        return (2 * precision * recall) / (precision + recall)

    def update(self, input: EvalInput) -> None:
        response = input["response"]
        references = input["references"]
        results = [
            self._f1(response, ref) for ref in references
        ]
        self.state[input["id"]] = results

    def _reduce_best(self):
        return sum(map(max, self.state.values()))

    def _reduce_mean(self):
        return sum(map(mean, self.state.values()))

    def compute(self) -> MetricValue:
        N = len(self.state)
        if self.reduction == "mean":
            num = self._reduce_mean()
        else:
            num = self._reduce_best()

        return {
            self.name: num/N
        }
