import json
import pandas as pd
from typing import List, Dict

class VQADatasetTransformer:

    def __init__(self, input_path: str, output_path: str) -> None:
        self.input_path = input_path
        self.output_path = output_path

    def load_data(self) -> List[Dict]:
        with open(self.input_path, 'r') as file:
            data = json.load(file)
        return data

    def transform_data(self, data: List[Dict]) -> pd.DataFrame:
        transformed_data = [
            {
                "id": entry["id"],
                "question": entry["question"],
                "url": entry["url"],
                "ground_truth_answer": entry["multiple_choice_answer"]
            }
            for entry in data
        ]
        return pd.DataFrame(transformed_data)

    def save_data(self, df: pd.DataFrame) -> None:
        data = df.to_dict(orient='records')
        with open(self.output_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)
        print(f"Dataset transformed and saved to {self.output_path}")

    def run(self) -> None:
        data = self.load_data()
        transformed_df = self.transform_data(data)
        self.save_data(transformed_df)


if __name__ == "__main__":
    input_path = "data/0_source_datasets/vqa_eval.json"
    output_path = "data/1_question_datasets/vqa_questions.json"

    transformer = VQADatasetTransformer(input_path, output_path)
    transformer.run()