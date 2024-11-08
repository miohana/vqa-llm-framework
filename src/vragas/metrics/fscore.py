import string
from collections import Counter
from statistics import mean
from typing import Dict, Callable, List

from .base import EvalInput, MetricWithReduction, Id, MetricValue


def tokenize(s: str) -> List[str]:
    # Normalize
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return s.lower().split()


class F1Score(MetricWithReduction):
    """
        F1 bag-of-words metric.
    """
    name = "f1"
    tokenizer: Callable[[str], List[str]] = staticmethod(tokenize)

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

    def _report_best(self) -> Dict[Id, float]:
        return {
            id: max(results)
            for id, results in self.state.items()
        }

    def _report_mean(self) -> Dict[Id, float]:
        return {
            id: mean(results)
            for id, results in self.state.items()
        }

    def report(self) -> Dict[Id, float]:
        if self.reduction == "mean":
            return self._report_mean()
        elif self.reduction == "best":
            return self._report_best()
        else:
            raise ValueError(f"Unknown reduction {self.reduction}")

    def compute(self) -> MetricValue:
        return {
            self.name: mean(self.report().values())
        }
