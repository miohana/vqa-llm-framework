from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Union, TypedDict


MetricValue = Dict[str, float]

Id = Union[int, str]

Reduction = Literal["mean", "best"]


class EvalInput(TypedDict):
    id: Id
    user_input: str
    image_input: str
    response: str
    references: List[str]


class Metric(ABC):
    name: str

    def __init__(self) -> None:
        self.state: Dict[Id, Any] = dict()

    @abstractmethod
    def update(self, input: EvalInput) -> None:
        """Override this method to update the state of your metric class."""
        return NotImplemented

    @abstractmethod
    def report(self) -> Dict[Id, Union[float, MetricValue]]:
        """Override this method to report the state of your metric class."""
        return NotImplemented

    @abstractmethod
    def compute(self) -> MetricValue:
        """Override this method to compute the final metric value."""
        return NotImplemented


class MetricWithReduction(Metric):
    def __init__(self, reduction: Reduction) -> None:
        super().__init__()
        self.reduction = reduction
