from unittest import mock
from langchain_core.messages import AIMessage

from vragas.metrics.relevancy_score import AnswerRelevancyScore


def test_answer_relevancy():
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
        AIMessage(
            content="What is the capital of France?\nWhere is the Eiffel Tower?"
        ),
        AIMessage(
            content="Who wrote 'Romeo and Juliet'?\nWho wrote 'Hamlet'?"
        ),
    ]
    answer_relevancy = AnswerRelevancyScore(
        language_model=mocked_language_model
    )

    for datum in data:
        answer_relevancy.update(datum)

    result = answer_relevancy.report()
    assert len(result) == 2
    assert result[1] >= 0
    assert result[2] >= 0

    metric_value = answer_relevancy.compute()
    assert "answer_relevancy_score" in metric_value
    assert metric_value["answer_relevancy_score"] >= 0
