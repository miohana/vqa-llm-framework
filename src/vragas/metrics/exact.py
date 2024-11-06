from statistics import mean

from .base import EvalInput, MetricWithReduction, MetricValue


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
