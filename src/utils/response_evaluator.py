import json
import os
import requests
from PIL import Image
import torch
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

class ResponseEvaluator:
    def __init__(self, model_type="gpt-4o"):
        self.token = os.environ.get("OPENAI_API_KEY") if model_type=="gpt-4o" else os.environ.get("HF_TOKEN")
        if not self.token:
            raise ValueError(f"{model_type} token not set in environment variables.")
        
        if model_type == "paligemma":
            print("Loading PaliGemma model and processor...")
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                "google/paligemma-3b-mix-224", use_auth_token=self.token
            ).eval()
            self.processor = AutoProcessor.from_pretrained(
                "google/paligemma-3b-mix-224", use_auth_token=self.token
            )
            print("PaliGemma model loaded successfully.")
        elif model_type == "gpt-4o":
            print("Loading GPT-4o model...")
            self.model = ChatOpenAI(model="gpt-4o", temperature=0, api_key=self.token)
            print("GPT-4o model loaded successfully.")
        else:
            raise ValueError("Invalid model type. Use 'paligemma' or 'gpt-4o'.")
        
        self.model_type = model_type

    def fetch_image(self, image_url: str) -> Image.Image:
        """Fetch image from the provided URL."""
        try:
            response = requests.get(image_url, stream=True, timeout=10)
            response.raise_for_status()
            return Image.open(response.raw)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error fetching image: {e}")

    def generate_evaluation(self, question: str, image_url: str, model_answer: str, evaluation_type: str) -> int:
        """
        Generate an evaluation score (1 for "yes" or 0 for "no") for faithfulness or relevancy.
        """
        try:
            image = self.fetch_image(image_url)
            if image is None:
                return 0
            print(self.model_type)
            prompt = (
                f"Evaluate the {evaluation_type} of the given **Answer** based on the **Image** and **Question**.\n\n"
                "Respond with either 'yes' or 'no' only, without any additional text.\n"
                "Examples:\n"
                "Question: Is the cat sitting on a chair? Answer: Yes, the cat is on the chair."
                "Reasoning: Since the cat is sitting in the char in the image, it is faithful.\nResponse: yes\n"
                "Question: Are there any cars in the picture? Answer: No, there are no cars."
                "Reasoning: I can see cars in the picture provided, which I may assume it is not a faithful answer provided.\nResponse: no\n\n"
                f"**Question**: {question}\n**Answer**: {model_answer}\nResponse: "
            )

            if self.model_type == "gpt-4o":
                message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                )
                response = self.model.invoke([message]).content.strip().lower()
            elif self.model_type == "paligemma":
                model_inputs = self.processor(text=prompt, images=image, return_tensors="pt")
                input_len = model_inputs["input_ids"].shape[-1]

                with torch.inference_mode():
                    generation = self.model.generate(
                        **model_inputs,
                        max_new_tokens=5,
                        do_sample=False,
                        num_return_sequences=1
                    )
                    generation = generation[0][input_len:]
                    response = self.processor.decode(generation, skip_special_tokens=True).strip().lower()

            print(f"Generated response for {evaluation_type}: '{response}'")

            if response == "yes":
                return 1
            elif response == "no":
                return 0
            else:
                print(f"Unexpected response: '{response}'. Defaulting to 0.")
                return 0

        except Exception as e:
            print(f"Error during evaluation for {evaluation_type}: {e}")
            return 0

    def assess_batch(self, data: list) -> list:
        results = []
        for entry in data:
            question = entry["question"]
            image_url = entry["url"]
            model_answer = entry["model_answer"]

            faithfulness = self.generate_evaluation(
                question, image_url, model_answer, "faithfulness"
            )

            relevancy = self.generate_evaluation(
                question, image_url, model_answer, "relevancy"
            )

            results.append({
                "id": entry["id"],
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
        raise ValueError("Invalid dataset type. Use 'vqa' or 'coco'.")

    # Load input data
    with open(input_path, 'r') as file:
        input_data = json.load(file)

    evaluator = ResponseEvaluator()
    assessment_results = evaluator.assess_batch(input_data)

    with open(output_path, 'w') as outfile:
        json.dump(assessment_results, outfile, indent=4)
    print(f"Assessment results saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate responses using PaliGemma or GPT-4o models.")
    parser.add_argument('--dataset_type', type=str, required=True, help='Type of dataset (vqa or coco)')

    args = parser.parse_args()
    main(args.dataset_type)