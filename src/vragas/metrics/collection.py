from typing import Dict
from collections import defaultdict

from .base import EvalInput, Metric, Id, MetricValue


class MetricsCollection(Metric):
    def __init__(self, *metrics: Metric):
        self.metrics = metrics

    def update(self, input: EvalInput) -> None:
        for metric in self.metrics:
            metric.update(input)

    def report(self) -> Dict[Id, MetricValue]:
        result = defaultdict(dict)
        for metric in self.metrics:
            for key, value in metric.report().items():
                result[key][metric.name] = value
        return result

    def compute(self) -> MetricValue:
        result = dict()
        for metric in self.metrics:
            result.update(metric.compute())
        return result
