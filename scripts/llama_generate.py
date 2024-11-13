import os
import json
import time
import base64
import requests
import pandas as pd
from tqdm import tqdm
from google import auth
from google.auth.transport.requests import Request
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from vragas.util.monitor import LLMUsageMonitor


def get_image_data(image_url: str):
    image_data = base64.b64encode(
        requests.get(image_url).content
    ).decode("utf-8")
    return f"data:image/jpeg;base64,{image_data}"


def load_chat_llm(callbacks=None):
    credentials, _ = auth.default()
    auth_request = Request()
    credentials.refresh(auth_request)
    return ChatOpenAI(
        base_url=os.environ["VERTEX_LLAMA_ENDPOINT"],
        api_key=credentials.token,
        model="meta/llama-3.2-90b-vision-instruct-maas",
        max_tokens=4096,
        temperature=0.5,
        callbacks=callbacks
    )


def get_human_message(question: str, image_url: str):
    input = [
        {"text": f"{question} Answer in one sentence.", "type": "text"}
    ]
    if image_url:
        image_message = {
            "image_url": {
                "url": get_image_data(image_url)
            },
            "type": "image_url",
        }
        input.insert(0, image_message)

    return HumanMessage(content=input)


if __name__ == "__main__":
    with open("data/coco-eval-questions.json") as fp:
        data = json.load(fp)

    monitor = LLMUsageMonitor()
    llm = load_chat_llm([monitor])
    results = []
    for sample in tqdm(data):
        message = get_human_message(sample["question"], sample["url"])
        response = llm.invoke([message])
        results.append({
            "id": sample["id"],
            "url": sample["url"],
            "question": sample["question"],
            "model_answer": response.content,
            "model_name": "llama-3.2-90b"
        })
        time.sleep(0.6)

    print(monitor)
    frame = pd.DataFrame(results)
    frame.to_csv("llama-coco.csv", index=False)
