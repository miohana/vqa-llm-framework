from typing import Union
from langchain.llms.base import BaseLLM
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import AIMessage

from .clip import CLIPScore, EvalInput


INSTRUCTION = """
Given a question and answer, create a statement that summarizes
the question and answer main proposition. Follow the examples structure:

Examples:

question: What kind of drink is inside the glass?
answer: It seem to be white wine.
statement: The glass is filled with white wine.
---
question: What activity do the man and the woman appear to be doing?
answer: They seem to be playing a videogame.
statement: A man and a woman are playing videogame.
---
question: What appears to be the purpose of the microwave being placed on the asphalt?
answer: The microwave appears to have been thrown away.
statement: A microwave was thrown away and placed on the asphalt.
---
question: Is the train a passenger or freight train?
answer: It's a freight train.
statement: The train is a freight train.
---
""".strip()


TEMPLATE = """
question: {question}
answer: {answer}
statement:
""".strip()


def parse_output_statements(output: Union[str, AIMessage]) -> str:
    if isinstance(output, AIMessage):
        output = output.content
    return output


def get_chain_for_model(model):
    if isinstance(model, BaseLLM):
        prompt = PromptTemplate.from_template(
            INSTRUCTION + "\n" + TEMPLATE
        )
    elif isinstance(model, BaseChatModel):
        prompt = ChatPromptTemplate.from_messages([
            ("system", INSTRUCTION),
            ("human", TEMPLATE)
        ])
    else:
        raise ValueError(f"invalid model class {type(model)}")
    
    return prompt | model | parse_output_statements


class FaithfulnessScore(CLIPScore):
    name = "faithfulness_score"

    def __init__(
        self,
        language_model: Union[BaseLLM, BaseChatModel],
        clip_model: str = "openai/clip-vit-base-patch32",
        **clip_kwargs
    ):
        super().__init__(clip_model, **clip_kwargs)
        self.responses = dict()
        self.chain = get_chain_for_model(language_model)
    
    def update(self, input: EvalInput) -> None:
        sentence = self.chain.invoke({
            "question": input["user_input"],
            "answer": input["response"]
        })
        self.responses[input["id"]] = sentence
        rewritten_input = {
            "id": input["id"],
            "image_input": input["image_input"],
            "response": sentence
        }
        return super().update(rewritten_input)
