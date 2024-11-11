import json
import os
import requests
import argparse
from PIL import Image
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

class ResponseGenerator:
    def __init__(self, model_id="google/paligemma-3b-mix-224"):
        self.token = os.environ.get("HF_TOKEN")
        if not self.token:
            raise ValueError("Hugging Face token (HF_TOKEN) not set in environment variables.")
        
        try:
            print("Loading model and processor...")
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, use_auth_token=self.token).eval()
            self.processor = AutoProcessor.from_pretrained(model_id, use_auth_token=self.token)
            print("Model and processor loaded successfully.")
        except Exception as e:
            raise ValueError(f"Error loading model or processor: {e}")

    def fetch_image(self, image_url: str) -> Image.Image:
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
            return Image.open(response.raw)
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

    def generate_batch_responses(self, data: list) -> list:
        responses = []
        for entry in data:
            question = entry["question"]
            image_url = entry["url"]
            response = self.generate_response(question, image_url)
            responses.append({
                "id": entry["id"],
                "question": question,
                "url": image_url,
                "model_answer": response,
                "model_name": "paligemma"
            })
        return responses


def main(dataset_type: str):
    if dataset_type == "vqa":
        input_path = "data/1_question_datasets/vqa_questions.json"
        output_path = "data/2_answered_question_datasets/vqa_answered.json"
    elif dataset_type == "coco":
        input_path = "data/1_question_datasets/coco_questions.json"
        output_path = "data/2_answered_question_datasets/coco_answered.json"
    else:
        raise ValueError("Invalid dataset type. Use 'vqa' or 'coco'.")

    with open(input_path, 'r') as file:
        input_data = json.load(file)

    generator = ResponseGenerator()

    responses = generator.generate_batch_responses(input_data)

    with open(output_path, 'w') as outfile:
        json.dump(responses, outfile, indent=4)
    print(f"Responses saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using PaliGemma model.")
    parser.add_argument('--dataset_type', type=str, required=True, help='Type of dataset (vqa or coco)')

    args = parser.parse_args()

    main(args.dataset_type)
