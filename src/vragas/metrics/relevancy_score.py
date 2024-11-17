from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from .embeddings import EmbeddingSimilarity, EvalInput
from ..util.image import load_image_as_base64


def format_image_content(image_url: str) -> dict:
    return {
        "image_url": {
            "url": load_image_as_base64(image_url)
        },
        "type": "image_url"
    }


def format_text_content(text: str) -> dict:
    return {
        "text": text,
        "type": "text"
    }


class AnswerRelevancyScore(EmbeddingSimilarity):
    """
      Calculates whether the answer addresses the question specifically.
    """
    name = "answer_relevancy_score"

    def __init__(
        self,
        language_model: BaseChatModel,
        n_questions: int = 3,
        use_image_input: bool = False,
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        **stransformers_kwargs
    ) -> None:
        super().__init__(embedding_model, zero_min=True, **stransformers_kwargs)
        self.language_model = language_model
        self.n_questions = n_questions
        self.use_image_input = use_image_input
        self.generated_questions = dict()

    def _get_system_prompt(self):
        if self.n_questions > 1:
            prompt = """
            Generate {} possible questions, one per line, for the given answer.
            """.strip().format(self.n_questions)
        else:
            prompt = "Generate a possible question for the given answer."

        if self.use_image_input:
            prompt += " Use the given image as support."

        return prompt

    def update(self, input: EvalInput) -> None:
        text_content = self._get_system_prompt()
        text_content += "\nAnswer: " + input["response"]
        prompt = HumanMessage(
            content=[format_text_content(text_content)]
        )
        if self.use_image_input:
            prompt.content.insert(
                0, format_image_content(input["image_input"])
            )
        ai_response = self.language_model.invoke([prompt])
        questions = ai_response.content.split("\n")
        self.generated_questions[input["id"]] = questions

        rewritten_input = {
            "id": input["id"],
            "response": input["user_input"],
            "references": questions
        }
        return super().update(rewritten_input)
