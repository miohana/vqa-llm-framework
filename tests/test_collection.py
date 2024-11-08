from src.vragas.metrics import F1Score, ExactMatch
from src.vragas.metrics.collection import MetricsCollection


def test_metrics_collection():
    data = [
        {
            "id": 1,
            "user_input": "What is the capital of France?",
            "response": "The capital of France is Paris.",
            "references": ["Paris"],
        },
        {
            "id": 2,
            "user_input": "Who wrote the play 'Romeo and Juliet'?",
            "response": "Shakespeare",
            "references": ["Shakespeare"],
        }
    ]

    metric = MetricsCollection(
        ExactMatch(reduction="best"),
        F1Score(reduction="best"),
    )

    for datum in data:
        metric.update(datum)

    result = metric.compute()
    assert result["em"] == 0.5
    assert result["f1"] > 0.5


def test_metrics_collection_report():
    data = [
        {
            "id": 1,
            "user_input": "What is the capital of France?",
            "response": "The capital of France is Paris.",
            "references": ["Paris"],
        },
        {
            "id": 2,
            "user_input": "Who wrote the play 'Romeo and Juliet'?",
            "response": "Shakespeare",
            "references": ["Shakespeare"],
        }
    ]

    metric = MetricsCollection(
        ExactMatch(reduction="best"),
        F1Score(reduction="best"),
    )

    for datum in data:
        metric.update(datum)

    result = metric.report()

    assert len(result) == 2
    assert "em" in result[1]
    assert "f1" in result[1]
    assert result[1]["em"] == 0
    assert result[1]["f1"] > 1/6
