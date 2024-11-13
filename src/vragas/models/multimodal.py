import torch
from typing import List, Union
from transformers import Pipeline
from transformers.image_utils import ImageInput, load_image


TextInput = Union[str, List[str]]


class ImageTextToTextPipeline(Pipeline):
    """

    A pipeline that generates text from an image and input prompt.

    Code sample:
    ```python
    model_id = "google/paligemma-3b-mix-224"
    processor = AutoProcessor.from_pretrained(model_id)
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)

    generator = ImageTextToTextPipeline(
        model=model,
        processor=processor,
        task="image-text-to-text",
        device="cuda",
        torch_dtype="bfloat16"
    )


    print(
        generator(
            "What are they doing? Answer in a single sentence",
            images="http://images.cocodataset.org/train2017/000000517382.jpg",
            max_new_tokens=100,
            do_sample=False
        )
    )
    ```
    """
    _load_processor = True
    _load_image_processor = False
    _load_feature_extractor = False
    _load_tokenizer = False

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "images" in kwargs:
            preprocess_kwargs["images"] = kwargs.pop("images")

        forward_kwargs = kwargs.copy()
        return preprocess_kwargs, forward_kwargs, {}

    def __call__(self, inputs: TextInput, images: ImageInput, **kwargs):
        """
        Generate texts from image and input prompt

        Args:
            inputs (`str`, `List[str]`):
                The input promp sequence that will be used as input for the generation.

            images (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a HTTP(s) link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images.

            kwargs (`Any`):
                Arguments passed into `generate`
        Return:
            A list or a list of list of `dict`: Each result comes as a dictionary with the following key:

            - **generated_text** (`str`) -- The generated text.
        """
        kwargs["images"] = images
        return super().__call__(inputs, **kwargs)

    def preprocess(self, inputs: TextInput, images: ImageInput):
        # Fetch images
        if isinstance(images, (list, tuple)):
            loaded_images = [load_image(img) for img in images]
        else:
            loaded_images = load_image(images)

        # Preprocess
        model_inputs = self.processor(
            text=inputs,
            images=loaded_images,
            return_tensors=self.framework
        )
        if hasattr(self, "device") and isinstance(self.device, torch.device):
            model_inputs = model_inputs.to(self.device)

        return model_inputs

    def _forward(self, model_inputs, **generate_kwargs):
        if "generation_config" not in generate_kwargs:
            generate_kwargs["generation_config"] = self.generation_config

        model_outputs = self.model.generate(
            **model_inputs, **generate_kwargs
        )
        return model_outputs

    def postprocess(self, model_outputs):
        if hasattr(self.processor, "decode"):
            decoder = self.processor.decode
        else:
            decoder = self.processor.tokenizer.decode

        return [
            {
                "generated_text": decoder(
                    output,
                    skip_special_tokens=True
                )
            }
            for output in model_outputs
        ]
