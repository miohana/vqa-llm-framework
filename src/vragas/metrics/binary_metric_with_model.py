import os
import requests
from PIL import Image
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import Dict, List
from base import MetricWithReduction, EvalInput, Id, MetricValue
from statistics import mean


class BinaryMetricWithModel(MetricWithReduction):
    """Base class for binary metrics (faithfulness and relevancy) that use models for evaluation."""

    def __init__(self, model_type="gpt-4o", reduction="mean"):
        super().__init__(reduction)
        self.model_type = model_type
        self._initialize_model()

    def _initialize_model(self):
        self.token = os.environ.get("OPENAI_API_KEY") if self.model_type == "gpt-4o" else os.environ.get("HF_TOKEN")
        if not self.token:
            raise ValueError(f"{self.model_type} token not set in environment variables.")

        if self.model_type == "paligemma":
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                "google/paligemma-3b-mix-224", use_auth_token=self.token).eval()
            self.processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224", use_auth_token=self.token)
        elif self.model_type == "gpt-4o":
            self.model = ChatOpenAI(model="gpt-4o", temperature=0, api_key=self.token)
        else:
            raise ValueError("Invalid model type.")

    def fetch_image(self, image_url: str) -> Image.Image:
        """Fetch image from the provided URL."""
        response = requests.get(image_url, stream=True)
        response.raise_for_status()
        return Image.open(response.raw)

    def generate_evaluation(self, prompt: str, image_url: str) -> str:
        """Generate a binary evaluation response using the model."""
        if self.model_type == "gpt-4o":
            message = HumanMessage(content=[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}])
            return self.model.invoke([message]).content.strip().lower()
        else:
            image = self.fetch_image(image_url)
            model_inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            input_len = model_inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(**model_inputs, max_new_tokens=5)
                response = self.processor.decode(generation[0][input_len:], skip_special_tokens=True).strip().lower()
            return response

    def _evaluate(self, question: str, image_url: str, model_answer: str, evaluation_type: str) -> int:
        """Evaluate the response using the specific evaluation type."""
        prompt = (
            f"Evaluate the {evaluation_type} of the given answer based on the question and image content.\n"
            f"Question: {question}\nAnswer: {model_answer}\nResponse: "
        )
        response = self.generate_evaluation(prompt, image_url)
        return 1 if response == "yes" else 0

    def update(self, input: EvalInput) -> None:
        response = input["response"]
        question = input["question"]
        image_url = input["url"]
        evaluation_type = input["evaluation_type"]

        result = self._evaluate(question, image_url, response, evaluation_type)
        self.state[input["id"]].append(result)

    def _report_best(self) -> Dict[Id, float]:
        return {id: max(results) for id, results in self.state.items()}

    def _report_mean(self) -> Dict[Id, float]:
        return {id: mean(results) for id, results in self.state.items()}

    def report(self) -> Dict[Id, float]:
        if self.reduction == "mean":
            return self._report_mean()
        elif self.reduction == "best":
            return self._report_best()
        else:
            raise ValueError("Invalid reduction method.")

    def compute(self) -> MetricValue:
        return {self.name: mean(self.report().values())}
