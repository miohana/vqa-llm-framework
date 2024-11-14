from binary_metric_with_model import BinaryMetricWithModel

class FaithfulnessMetric(BinaryMetricWithModel):
    """Metric to evaluate faithfulness."""

    def evaluate(self, question: str, image_url: str, model_answer: str) -> int:
        response = self.generate_response(question, model_answer, image_url, "faithfulness")
        return 1 if response == "yes" else 0

    def get_prompt(self, question: str, model_answer: str, evaluation_type: str) -> str:
        return (
                f"Evaluate the {evaluation_type} of the given **Answer** based on the **Image** and **Question**.\n\n"
                "Respond with either 'yes' or 'no' only, without any additional text.\n"
                "Examples:\n"
                "Question: Is the cat sitting on a chair? Answer: Yes, the cat is on the chair."
                "Reasoning: Since the cat is sitting in the char in the image, it is faithful.\nResponse: yes\n"
                "Question: Are there any cars in the picture? Answer: No, there are no cars."
                "Reasoning: I can see cars in the picture provided, which I may assume it is not a faithful answer provided.\nResponse: no\n\n"
                f"**Question**: {question}\n**Answer**: {model_answer}\nResponse: "
        )
