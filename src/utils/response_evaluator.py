import json
import os
import requests
from PIL import Image
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import argparse


class ResponseEvaluator:
    def __init__(self, model_id="google/paligemma-3b-mix-224"):
        self.token = os.environ.get("HF_TOKEN")
        if not self.token:
            raise ValueError("Hugging Face token (HF_TOKEN) not set in environment variables.")
        
        print("Loading PaliGemma model and processor...")
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, use_auth_token=self.token
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_id, use_auth_token=self.token
        )
        print("Model loaded successfully.")

    def fetch_image(self, image_url: str) -> Image.Image:
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
            print(f"Image fetched successfully from {image_url}")
            return Image.open(response.raw)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching image from {image_url}: {e}")
            return None

    def generate_evaluation(self, question: str, image_url: str, model_answer: str, evaluation_type: str) -> int:
        try:
            image = self.fetch_image(image_url)
            if image is None:
                return 0

            prompt = (
                f"Evaluate the {evaluation_type} of the given answer based on the image and question. "
                f"Question: {question}. Model Answer: {model_answer}. "
                "Respond with 1 if the answer is supported, otherwise respond with 0."
            )
            
            model_inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            input_len = model_inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(**model_inputs, max_new_tokens=10, do_sample=False)
                generation = generation[0][input_len:]
                response = self.processor.decode(generation, skip_special_tokens=True).strip()

            print(f"Generated response for {evaluation_type}: {response}")
            return int(response) if response in ["0", "1"] else 0
        except Exception as e:
            print(f"Error during evaluation for {evaluation_type}: {e}")
            return 0

    def assess_batch(self, data: list) -> list:
        results = []
        for entry in data:
            question = entry.get("question", "")
            image_url = entry.get("url", "")
            model_answer = entry.get("model_answer", "")

            print(f"Evaluating question ID {entry['id']}...")

            faithfulness = self.generate_evaluation(question, image_url, model_answer, "faithfulness")
            relevancy = self.generate_evaluation(question, image_url, model_answer, "relevancy")
            
            results.append({
                "id": entry["id"],
                "model_name": entry.get("model_name", "unknown"),
                "question": question,
                "url": image_url,
                "model_answer": model_answer,
                "faithfulness": faithfulness,
                "relevancy": relevancy
            })

        return results


def main(dataset_type: str):
    if dataset_type == "vqa":
        input_path = "data/2_answered_question_datasets/vqa_answered.json"
        output_path = "data/3_assessment_results_datasets/vqa_assessment_results.json"
    elif dataset_type == "coco":
        input_path = "data/2_answered_question_datasets/coco_answered.json"
        output_path = "data/3_assessment_results_datasets/coco_assessment_results.json"
    else:
        print("Invalid dataset type. Please use 'vqa' or 'coco'.")
        return

    print(f"Loading data from {input_path}...")
    with open(input_path, 'r') as file:
        input_data = json.load(file)
    print(f"Data loaded successfully with {len(input_data)} entries.")

    evaluator = ResponseEvaluator()
    assessment_results = evaluator.assess_batch(input_data)

    print(f"Saving assessment results to {output_path}...")
    with open(output_path, 'w') as outfile:
        json.dump(assessment_results, outfile, indent=4)
    print(f"Assessment results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate responses using PaliGemma model.")
    parser.add_argument('--dataset_type', type=str, required=True, help='Type of dataset (vqa or coco)')

    args = parser.parse_args()

    main(args.dataset_type)