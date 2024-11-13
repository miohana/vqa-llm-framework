import torch
import requests
from PIL import Image
from statistics import mean
from transformers import CLIPModel, CLIPProcessor
from typing import get_args, Dict, Union, Literal

from .base import Id, EvalInput, Metric, MetricValue


DType = Literal["float16", "bfloat16", "float32"]
Device = Literal["cpu", "cuda"]


class CLIPScore(Metric):

    name = "clip_score"

    def __init__(
        self,
        model: str,
        device: Union[Device, torch.device] = None,
        dtype: Union[DType, torch.dtype] = torch.float32,
    ):
        super().__init__()
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif isinstance(device, str) and device not in get_args(Device):
            raise ValueError(f"Invalid device: {device}")
        if isinstance(dtype, str) and dtype in get_args(DType):
            dtype = getattr(torch, dtype)
        elif isinstance(dtype, str):
            raise ValueError(f"Unknown dtype: {dtype}")

        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(model).to(device, dtype=dtype)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model)

    def score(self, text: str, image: Image.Image) -> float:
        inputs = self.processor(
            text=text, images=image,
            return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            ouputs = self.model(**inputs)
            text_embeds, image_embeds = ouputs.text_embeds, ouputs.image_embeds

        similarity: torch.Tensor = (text_embeds * image_embeds).sum(dim=-1)
        return similarity.relu().item()

    def update(self, input: EvalInput) -> None:
        """
            Update metric state with input response
        """
        image = Image.open(
            requests.get(input["image_input"], stream=True).raw
        )
        score = self.score(input["response"], image)
        self.state[input["id"]] = score

    def report(self) -> Dict[Id, float]:
        return self.state.copy()

    def compute(self) -> MetricValue:
        return {
            self.name: mean(self.state.values())
        }
