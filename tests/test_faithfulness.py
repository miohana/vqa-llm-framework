from langchain_core.language_models import FakeListLLM, FakeListChatModel

from src.vragas.metrics import FaithfulnessScore


def test_faithfulness_score_with_llm():
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
                "A man and a women are playing videogame", 
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
                "A man and a women are playing videogame", 
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