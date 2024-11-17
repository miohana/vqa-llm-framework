from statistics import mean
from typing import Dict

from .base import EvalInput, MetricWithReduction, Id, MetricValue


def preprocess(s: str) -> str:
    return s.strip().lower()


class ExactMatch(MetricWithReduction):
    """
        Accuracy metric. Calculate whether a response string is exactly
        equals to any annotated reference.
    """
    name = "em"

    def update(self, input: EvalInput) -> None:
        response = preprocess(input["response"])
        references = list(
            map(preprocess, input["references"])
        )
        results = [
            response == ref for ref in references
        ]
        self.state[input["id"]] = results

    def _report_best(self) -> Dict[Id, float]:
        return {
            id: max(map(int, results))
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
