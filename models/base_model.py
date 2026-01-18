from typing import List
from abc import ABC, abstractmethod
from mydatasets.BaseDataset import PageContent, Figure


class BaseModel:
    model_name: str

    def __init__(self, config):
        """
        Base model constructor to initialize common attributes.
        :param config: A dictionary containing model configuration parameters.
        """
        self.create_ask_message = lambda question: {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ans},
            ],
        }
        self.config = config

    def predict(
        self,
        question: str,
        texts: List | None = None,
        images: List | None = None,
        history=None,
    ):
        pass

    def create_text_message(self, texts: List[str], question):
        content = []
        for text in texts:
            content.append({"type": "text", "text": text})
        content.append({"type": "text", "text": question})
        message = {"role": "user", "content": content}
        return message

    def create_image_message(self, images: List[str], question):
        content = []
        for image_path in images:
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": question})
        message = {"role": "user", "content": content}
        return message

    def process_message(self, question, texts, images, history):
        if history is not None:
            assert self.is_valid_history(history)
            messages = history
        else:
            messages = []

        if texts is not None:
            messages.append(self.create_text_message(texts, question))
        if images is not None:
            messages.append(self.create_image_message(images, question))

        if (texts is None or len(texts) == 0) and (images is None or len(images) == 0):
            messages.append(self.create_ask_message(question))

        return messages

    def is_valid_history(self, history):
        return True

    @abstractmethod
    def query(
        self,
        messages: list,
    ) -> tuple[str, list]:
        pass

    @abstractmethod
    def content2message(
        self,
        prompt: str,
        question: str,
        page_contents: list[PageContent],
        figure_contents: list[Figure],
    ) -> list:
        pass


class BaseLLM:
    model_name: str

    @abstractmethod
    def query(
        self,
        messages: list,
    ) -> tuple[str, list]:
        pass

    @abstractmethod
    def content2message(
        self,
        prompt: str,
        question: str,
        page_contents: list[PageContent],
        figure_contents: list[Figure],
    ) -> list:
        pass

    def create_text_message(self, texts: List[str], question):
        content = []
        for text in texts:
            content.append({"type": "text", "text": text})
        content.append({"type": "text", "text": question})
        message = {"role": "user", "content": content}
        return message

    def create_image_message(self, images: List[str], question):
        content = []
        for image_path in images:
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": question})
        message = {"role": "user", "content": content}
        return message

    def is_valid_history(self, history):
        return True

    def process_message(self, question, texts=None, images=None, history=None):
        if history is not None:
            assert self.is_valid_history(history)
            messages = history
        else:
            messages = []
        return messages

    def predict(self, question, history):
        pass
