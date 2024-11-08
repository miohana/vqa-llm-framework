from src.vragas.metrics import F1Score, ExactMatch, SoftMatch


def test_f1_score():
    data = {
        "id": 1,
        "user_input": "What is the capital of France?",
        "response": "Paris",
        "references": ["Paris"],
    }

    f1_score = F1Score(reduction="mean")
    f1_score.update(data)
    result = f1_score.compute()
    assert result['f1'] == 1.0


def test_exact_match():
    data = {
        "id": 1,
        "user_input": "What is the capital of France?",
        "response": "Ennepe",
        "references": ["Paris"],
    }

    exact_match = ExactMatch(reduction="mean")
    exact_match.update(data)
    result = exact_match.compute()
    assert result['em'] == 0.0


def test_soft_match():
    data = {
        "id": 1,
        "user_input": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "references": ["Paris"],
    }

    soft_match = SoftMatch(reduction="mean")
    soft_match.update(data)
    result = soft_match.compute()
    assert result['sm'] == 1.0
