import os
import json
import argparse
import pathlib
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from langchain_openai import ChatOpenAI
from typing import Literal, Iterator, Optional


from vragas.metrics import *
from vragas.util.monitor import LLMUsageMonitor


def get_chat_model(callbacks=None) -> ChatOpenAI:
    """
        We will use Llama 3.2 as the chat model
    """
    from google import auth
    from google.auth.transport.requests import Request
    credentials, _ = auth.default()
    auth_request = Request()
    credentials.refresh(auth_request)
    return ChatOpenAI(
        base_url=os.environ["VERTEX_LLAMA_ENDPOINT"],
        api_key=credentials.token,
        model="meta/llama-3.2-90b-vision-instruct-maas",
        max_tokens=4096,
        temperature=0.0,
        callbacks=callbacks
    )


def get_metric(
    language_model: ChatOpenAI,
    mode: Literal["vqa", "coco"]
) -> Metric:
    if mode.lower() == "vqa":
        metrics = [
            ExactMatch("best"),
            SoftMatch("best"),
            F1Score("best")
        ]
    else:
        metrics = []

    metrics.extend([
        FaithfulnessScore(
            language_model,
            clip_model="openai/clip-vit-base-patch16"
        ),
        AnswerRelevancyScore(
            language_model,
            n_questions=1,
            use_image_input=True
        )
    ])

    return MetricsCollection(*metrics)


def get_generations(collection: MetricsCollection) -> pd.DataFrame:
    faithfulness = next(
        m for m in collection.metrics if isinstance(m, FaithfulnessScore)
    )
    relevancy = next(
        m for m in collection.metrics if isinstance(m, AnswerRelevancyScore)
    )
    generations = [
        {
            "id": id,
            "generated_sentences": response,
            "generated_question": relevancy.generated_questions.get(id, ""),

        }
        for id, response in faithfulness.responses.items()
    ]

    return pd.DataFrame(generations)


class Loader:
    def __init__(
        self,
        answer_file: pathlib.Path,
        source_file: Optional[pathlib.Path] = None,
        mode: Literal["vqa", "coco"] = "vqa"
    ) -> None:
        if not answer_file.is_file():
            raise ValueError("Invalid answer file")
        self.answers = pd.read_csv(answer_file)
        self.references = defaultdict(list)
        if mode == "vqa":
            if source_file is None or not source_file.is_file():
                raise ValueError("Source file must be provided in 'vqa' mode")
            source = pd.read_json(source_file)
            # Find references answers
            for _, row in source.iterrows():
                self.references[row["id"]].extend(row["answers"])

    def __len__(self) -> int:
        return self.answers.shape[0]

    def __iter__(self) -> Iterator[dict]:
        for _, row in self.answers.iterrows():
            data = {
                "id": row["id"],
                "image_input": row["url"],
                "user_input": row["question"],
                "response": row["model_answer"],
                "references": self.references[row["id"]]
            }
            yield data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Image Question Answering Responses"
    )
    parser.add_argument(
        "--answer",
        type=pathlib.Path, required=True, help="Model answers file"
    )
    parser.add_argument(
        "--source",
        type=pathlib.Path, default=None, help="Source image file"
    )
    parser.add_argument(
        "--mode",
        choices=["vqa", "coco"], default="vqa", help="Evaluation mode"
    )
    args = parser.parse_args()

    data_loader = Loader(args.answer, args.source, args.mode)
    monitor = LLMUsageMonitor()
    llm = get_chat_model([monitor])
    metrics = get_metric(llm, args.mode)
    for data in tqdm(data_loader):
        metrics.update(data)

    print("LLM Usage", monitor)
    results = pd.DataFrame([
        {
            "id": id,
            **values
        }
        for id, values in metrics.report().items()
    ])
    summary = metrics.compute()

    results_file = pathlib.Path(
        "data", "results", f"{args.answer.stem}-results.csv"
    )
    results.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    summary_file = pathlib.Path(
        "data", "results", f"{args.answer.stem}-summary.json"
    )
    with open(summary_file, "w+") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"Summary saved to {summary_file}")

    generations = get_generations(metrics)
    generations_file = pathlib.Path(
        "data", "results", f"{args.answer.stem}-assessment-generations.csv"
    )
    generations.to_csv(generations_file, index=False)
    print(f"Assessments saved to {generations_file}")
