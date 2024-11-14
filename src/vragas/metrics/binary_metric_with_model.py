import os
import requests
from PIL import Image
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import Dict, Any
from base import MetricWithReduction


class BinaryMetricWithModel(MetricWithReduction):
    """Base class for binary metrics that use models for evaluation."""

    def __init__(self, model_type="gpt-4o", reduction="mean"):
        super().__init__(reduction)
        self.token = os.environ.get("OPENAI_API_KEY") if model_type == "gpt-4o" else os.environ.get("HF_TOKEN")
        if not self.token:
            raise ValueError(f"{model_type} token not set in environment variables.")
        self.model_type = model_type
        self._initialize_model()

    def _initialize_model(self):
        if self.model_type == "paligemma":
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                "google/paligemma-3b-mix-224", use_auth_token=self.token
            ).eval()
            self.processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224", use_auth_token=self.token)
        elif self.model_type == "gpt-4o":
            self.model = ChatOpenAI(model="gpt-4o", temperature=0, api_key=self.token)
        else:
            raise ValueError("Invalid model type.")

    def fetch_image(self, image_url: str) -> Image.Image:
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
            return Image.open(response.raw)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error fetching image: {e}")

    def generate_response(self, question: str, model_answer: str, image_url: str, evaluation_type: str) -> str:
        prompt = self.get_prompt(question, model_answer, evaluation_type)
        if self.model_type == "gpt-4o":
            message = HumanMessage(content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ])
            return self.model.invoke([message]).content.strip().lower()

        elif self.model_type == "paligemma":
            image = self.fetch_image(image_url)
            model_inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            input_len = model_inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                generation = self.model.generate(**model_inputs, max_new_tokens=5, do_sample=False)
                return self.processor.decode(generation[0][input_len:], skip_special_tokens=True).strip().lower()
        return ""

    def get_prompt(self, question: str, model_answer: str, evaluation_type: str) -> str:
        raise NotImplementedError("Subclasses must implement the get_prompt method.")
