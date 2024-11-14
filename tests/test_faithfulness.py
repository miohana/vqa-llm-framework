from langchain_core.language_models import FakeListLLM, FakeListChatModel
from src.vragas.metrics import FaithfulnessScore, FaithfulnessMetric


def test_faithfulness_score_with_llm():
    """Test the existing FaithfulnessScore using FakeListLLM."""
    data = [
        {
            "id": 1,
            "image_input": "http://images.cocodataset.org/train2017/000000517382.jpg",
            "user_input": "What activity do the man and the woman appear to be doing?",
            "response": "They are playing videogame.",
        },
        {
            "id": 2,
            "image_input": "http://images.cocodataset.org/train2017/000000370278.jpg",
            "user_input": "What type of meal is laid out on the table?",
            "response": "A hotel breakfast",
        }
    ]

    faithfulness = FaithfulnessScore(
        language_model=FakeListLLM(
            responses=[
                "A man and a woman are playing videogame", 
                "A hotel breakfast is laid out on the table"
            ]
        ),
        dtype="bfloat16"
    )

    faithfulness.update(data[0])
    faithfulness.update(data[1])

    report = faithfulness.report()
    assert all(v >= 0 for v in report.values())
    score = faithfulness.compute()
    assert "faithfulness_score" in score
    assert score["faithfulness_score"] >= 0.0


def test_faithfulness_score_with_chat_model():
    """Test the existing FaithfulnessScore using FakeListChatModel."""
    data = [
        {
            "id": 1,
            "image_input": "http://images.cocodataset.org/train2017/000000517382.jpg",
            "user_input": "What activity do the man and the woman appear to be doing?",
            "response": "They are playing videogame.",
        },
        {
            "id": 2,
            "image_input": "http://images.cocodataset.org/train2017/000000370278.jpg",
            "user_input": "What type of meal is laid out on the table?",
            "response": "A hotel breakfast",
        }
    ]

    faithfulness = FaithfulnessScore(
        language_model=FakeListChatModel(
            responses=[
                "A man and a woman are playing videogame", 
                "A hotel breakfast is laid out on the table"
            ]
        ),
        dtype="bfloat16"
    )

    faithfulness.update(data[0])
    faithfulness.update(data[1])

    report = faithfulness.report()
    assert all(v >= 0 for v in report.values())
    score = faithfulness.compute()
    assert "faithfulness_score" in score
    assert score["faithfulness_score"] >= 0.0


def test_new_faithfulness_metric_with_llm():
    """Test the new FaithfulnessMetric using FakeListLLM."""
    data = [
        {
            "id": 1,
            "image_input": "http://images.cocodataset.org/train2017/000000517382.jpg",
            "user_input": "What activity do the man and the woman appear to be doing?",
            "response": "They are playing videogame.",
        },
        {
            "id": 2,
            "image_input": "http://images.cocodataset.org/train2017/000000370278.jpg",
            "user_input": "What type of meal is laid out on the table?",
            "response": "A hotel breakfast",
        }
    ]

    faithfulness_metric = FaithfulnessMetric(model_type="gpt-4o", reduction="mean")
    
    faithfulness_metric.model = FakeListLLM(
        responses=[
            "A man and a woman are playing videogame",
            "A hotel breakfast is laid out on the table"
        ]
    )

    for entry in data:
        faithfulness_metric.update({
            "id": entry["id"],
            "question": entry["user_input"],
            "url": entry["image_input"],
            "response": entry["response"]
        })

    report = faithfulness_metric.report()
    assert all(v in [0, 1] for v in report.values())
    score = faithfulness_metric.compute()
    assert "faithfulnessmetric" in score
    assert score["faithfulnessmetric"] >= 0.0
    print("New FaithfulnessMetric with LLM:", score)


def test_new_faithfulness_metric_with_chat_model():
    """Test the new FaithfulnessMetric using FakeListChatModel."""
    data = [
        {
            "id": 1,
            "image_input": "http://images.cocodataset.org/train2017/000000517382.jpg",
            "user_input": "What activity do the man and the woman appear to be doing?",
            "response": "They are playing videogame.",
        },
        {
            "id": 2,
            "image_input": "http://images.cocodataset.org/train2017/000000370278.jpg",
            "user_input": "What type of meal is laid out on the table?",
            "response": "A hotel breakfast",
        }
    ]

    faithfulness_metric = FaithfulnessMetric(model_type="gpt-4o", reduction="mean")
    
    faithfulness_metric.model = FakeListChatModel(
        responses=[
            "A man and a woman are playing videogame",
            "A hotel breakfast is laid out on the table"
        ]
    )

    for entry in data:
        faithfulness_metric.update({
            "id": entry["id"],
            "question": entry["user_input"],
            "url": entry["image_input"],
            "response": entry["response"]
        })

    report = faithfulness_metric.report()
    assert all(v in [0, 1] for v in report.values())
    score = faithfulness_metric.compute()
    assert "faithfulnessmetric" in score
    assert score["faithfulnessmetric"] >= 0.0
    print("New FaithfulnessMetric with Chat Model:", score)