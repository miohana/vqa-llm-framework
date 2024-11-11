from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import requests
import torch
import os

class ResponseGenerator:
    def __init__(self, model_id="google/paligemma-3b-mix-224"):
        self.token = os.environ.get("HF_TOKEN")
        if not self.token:
            raise ValueError("Hugging Face token (HF_TOKEN) not set in environment variables.")
        
        try:
            print("Loading model and processor...")
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval()
            self.processor = AutoProcessor.from_pretrained(model_id)
            print("Model and processor loaded successfully.")
        except Exception as e:
            raise ValueError(f"Error loading model or processor: {e}")

    def fetch_image(self, image_url: str) -> Image.Image:
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
            image = Image.open(response.raw)
            return image
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error fetching image: {e}")

    def generate_response(self, question: str, image_url: str) -> str:
        try:
            image = self.fetch_image(image_url)
            
            model_inputs = self.processor(text=question, images=image, return_tensors="pt")
            input_len = model_inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
                generation = generation[0][input_len:]
                response = self.processor.decode(generation, skip_special_tokens=True)
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return "An error occurred while generating the response."

    def generate_batch_responses(self, questions: list, image_urls: list) -> list:
        responses = []
        for question, image_url in zip(questions, image_urls):
            response = self.generate_response(question, image_url)
            responses.append(response)
        return responses
    
if __name__ == "__main__":
    generator = ResponseGenerator()

    question = "What is the person doing?"
    image_url = "http://images.cocodataset.org/val2014/COCO_val2014_000000293832.jpg"
    response = generator.generate_response(question, image_url)
    print(f"Response: {response}")

    questions = ["What is in the image?", "How many people are there?"]
    image_urls = [
        "http://images.cocodataset.org/val2014/COCO_val2014_000000129592.jpg",
        "http://images.cocodataset.org/val2014/COCO_val2014_000000293832.jpg"
    ]
    batch_responses = generator.generate_batch_responses(questions, image_urls)
    print(f"Batch Responses: {batch_responses}")
