from unittest import mock
from langchain_core.messages import AIMessage
from vragas.metrics import AnswerRelevancyScore, RelevancyMetric

def test_answer_relevancy():
    """Test the existing AnswerRelevancyScore with mock responses."""
    data = [
        {
            "id": 1,
            "user_input": "What is the capital of France?",
            "response": "Paris",
            "references": ["Paris"],
        },
        {
            "id": 2,
            "user_input": "Who wrote the play 'Romeo and Juliet'?",
            "response": "Shakespeare",
            "references": ["Shakespeare"],
        }
    ]

    mocked_language_model = mock.MagicMock()
    mocked_language_model.invoke.side_effect = [
        AIMessage(content="What is the capital of France?\nWhere is the Eiffel Tower?"),
        AIMessage(content="Who wrote 'Romeo and Juliet'?\nWho wrote 'Hamlet'?"),
    ]

    answer_relevancy = AnswerRelevancyScore(language_model=mocked_language_model)

    for datum in data:
        answer_relevancy.update(datum)

    result = answer_relevancy.report()
    assert len(result) == 2
    assert result[1] >= 0
    assert result[2] >= 0

    metric_value = answer_relevancy.compute()
    assert "answer_relevancy_score" in metric_value
    assert metric_value["answer_relevancy_score"] >= 0
    print("Existing AnswerRelevancyScore test passed with score:", metric_value)


def test_new_relevancy_metric():
    """Test the new RelevancyMetric using mock model responses."""
    data = [
        {
            "id": 1,
            "user_input": "What is the capital of France?",
            "response": "Paris",
            "references": ["Paris"],
        },
        {
            "id": 2,
            "user_input": "Who wrote the play 'Romeo and Juliet'?",
            "response": "Shakespeare",
            "references": ["Shakespeare"],
        }
    ]

    mocked_language_model = mock.MagicMock()
    mocked_language_model.invoke.side_effect = [
        AIMessage(content="What is the capital of France?\nWhere is the Eiffel Tower?"),
        AIMessage(content="Who wrote 'Romeo and Juliet'?\nWho wrote 'Hamlet'?"),
    ]

    relevancy_metric = RelevancyMetric(model_type="gpt-4o", reduction="mean")
    relevancy_metric.model = mocked_language_model

    for entry in data:
        relevancy_metric.update({
            "id": entry["id"],
            "question": entry["user_input"],
            "response": entry["response"],
            "references": entry["references"]
        })

    report = relevancy_metric.report()
    assert len(report) == 2
    assert report[1] in [0, 1], "Score should be binary (0 or 1)"
    assert report[2] in [0, 1], "Score should be binary (0 or 1)"

    metric_value = relevancy_metric.compute()
    assert "relevancymetric" in metric_value, "Metric value should include 'relevancymetric'"
    assert metric_value["relevancymetric"] >= 0.0, "Relevancy score should be non-negative"
    print("New RelevancyMetric test passed with score:", metric_value)