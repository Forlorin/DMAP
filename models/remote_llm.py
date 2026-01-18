from models.base_model import BaseModel, BaseLLM
from openai import OpenAI, APIConnectionError, RateLimitError
from typing import Any
import base64
import os
from pathlib import Path
import time
from tqdm import tqdm

from mydatasets.BaseDataset import Figure, PageContent


class Qwen3VL(BaseModel):
    def __init__(
        self,
        config=None,
        api_key: str | None = os.getenv("DASHSCOPE_API_KEY"),
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name: str = "qwen-vl-plus",
    ):
        super().__init__(config)
        self.model = OpenAI(api_key=api_key, base_url=base_url)
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
        self.model_name = model_name

    def create_text_message(self, texts, question):
        content = []
        for text in texts:
            content.append({"type": "text", "text": text})
        content.append({"type": "text", "text": question})
        message = {"role": "user", "content": content}
        return message

    def process_message(
        self, question, texts, images, history, figure_message_contents=[]
    ):
        if history is not None:
            assert self.is_valid_history(history)
            messages = history
        else:
            messages = []
        if texts is not None:
            messages.append(self.create_text_message(texts, question))
        if images is not None:
            messages.append(
                self.create_image_message(
                    images, question, figure_message_contents=figure_message_contents
                )
            )

        if (texts is None or len(texts) == 0) and (images is None or len(images) == 0):
            messages.append(self.create_ask_message(question))

        return messages

    def create_image_message(self, images, question, figure_message_contents=[]):
        if figure_message_contents:
            content = figure_message_contents
            content.append(
                {
                    "type": "text",
                    "text": "The next are some document pages that may contain the required information.\n",
                }
            )
        else:
            content = []
        for image_path in images:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )
        content.append({"type": "text", "text": question})
        message = {"role": "user", "content": content}
        return message

    def content2message(
        self,
        prompt: str,
        question: str,
        page_contents: list[PageContent],
        figure_contents: list[Figure],
    ) -> list:
        message_contents: list[dict[str, Any]] = [
            {"type": "text", "text": f"{prompt}{question}"}
        ]
        if figure_contents:
            message_contents.append(
                {
                    "type": "text",
                    "text": "The next are some figures ,tables or charts that may contain the required information.\n",
                }
            )
            for figure in figure_contents:
                with open(figure.image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                message_contents.extend(
                    [
                        {"type": "text", "text": f"{figure.text}:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ]
                )
        if page_contents:
            message_contents.append(
                {
                    "type": "text",
                    "text": "The next are some document pages that may contain the required information.I will give you page images and texts\n",
                }
            )
            for page in page_contents:
                with open(page.image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                message_contents.extend(
                    [
                        {"type": "text", "text": f"Page {page.page}:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ]
                )
        messages = [{"role": "user", "content": message_contents}]
        return messages

    def query(self, messages: list) -> tuple[str, list]:
        output = self.model.chat.completions.create(
            model=self.model_name, messages=messages
        )
        output_text = output.choices[0].message.content
        if output_text is None:
            output_text = ""
        return output_text, messages

    def create_figure_message(
        self, figure_contents: list[dict[str, str]], figure_prompt: str
    ):
        contents = []
        # content 内容
        # {
        #     "img_path": img_path,
        #     "caption": item.get("caption", ""),
        # }
        for content in figure_contents:
            try:
                with open(content["img_path"], "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                c = [
                    {"type": "text", "text": content["caption"]},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ]
                contents.extend(c)
            except:
                continue
        if contents:
            contents.insert(0, {"type": "text", "text": figure_prompt})
        return contents

    def predict(self, question, texts=None, images=None, history=None) -> tuple[str, list]:  # type: ignore
        messages = self.process_message(
            question,
            texts,
            images,
            history,
        )
        output = self.model.chat.completions.create(
            model=self.model_name, messages=messages
        )
        output_text = output.choices[0].message.content
        if output_text == None:
            output_text = ""
        return output_text, messages

    def is_valid_history(self, history):  # type: ignore
        if not isinstance(history, list):
            return False
        for item in history:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
            if not isinstance(item["role"], str) or not isinstance(
                item["content"], list
            ):
                return False
            for content in item["content"]:
                if not isinstance(content, dict):
                    return False
                if "type" not in content:
                    return False
                if content["type"] not in content:
                    return False
        return True


class Qwen3(BaseModel):
    def __init__(
        self,
        config=None,
        api_key: str | None = os.getenv("DASHSCOPE_API_KEY"),
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name: str = "qwen-plus",
    ):
        super().__init__(config)
        self.model = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def create_text_message(self, texts, question):
        prompt = "\n".join(texts)
        message = {
            "role": "user",
            "content": f"{prompt}\n{question}",
        }
        return message

    def predict(self, question, texts=None, images=None, history=None) -> tuple[str, list]:  # type: ignore
        messages = self.process_message(question, texts, None, None)
        output_text = (
            self.model.chat.completions.create(
                model=self.model_name,
                messages=messages,
                extra_body={"enable_thinking": False},  # 在这里配置 enable_thinking
            )
            .choices[0]
            .message.content
        )
        if output_text is None:
            output_text = ""
        return output_text, messages

    def is_valid_history(self, history):  # type: ignore
        if not isinstance(history, list):
            return False
        for item in history:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
            if not isinstance(item["role"], str) or not isinstance(
                item["content"], str
            ):
                return False
        return True

    def content2message(
        self,
        prompt: str,
        question: str,
        page_contents: list[PageContent],
        figure_contents: list[Figure] = [],
    ) -> list:
        """
        如果不传入page content，则将prompt和question结合进行回答，否则生成page内容
        """
        text = ""
        if page_contents:
            for page in page_contents:
                text += f"{page.text.replace('\n',' ')} \n"
            text = f"The following is the reference content you can use. Answer the question mentioned above based on the content below.{text}\nQuestion:{question}"
        messages = [{"role": "user", "content": f"{prompt}{question}\n{text}"}]
        return messages

    def query(self, messages: list) -> tuple[str, list]:
        output = self.model.chat.completions.create(
            model=self.model_name, messages=messages
        )
        output_text = output.choices[0].message.content
        if output_text is None:
            output_text = ""
        return output_text, messages


class QwenLong:
    def __init__(
        self,
        api_key: str | None = os.getenv("DASHSCOPE_API_KEY"),
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name: str = "qwen-long",
    ):
        self.model = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        file_stk = self.model.files.list().model_dump()
        while file_stk["has_more"] == True:
            new_file_stk = self.model.files.list(
                after=file_stk["data"][-1]["id"]
            ).model_dump()
            file_stk["data"].append(new_file_stk["data"])
            file_stk["has_more"] = new_file_stk["has_more"]
        self.file_objects: list[dict[str, Any]] = file_stk["data"]

    def get_remote_file_objects(self):
        file_stk = self.model.files.list().model_dump()
        while file_stk["has_more"] == True:
            new_file_stk = self.model.files.list(
                after=file_stk["data"][-1]["id"]
            ).model_dump()
            file_stk["data"].append(new_file_stk["data"])
            file_stk["has_more"] = new_file_stk["has_more"]
        self.file_objects: list[dict[str, Any]] = file_stk["data"]

    def get_file_id(self, file_path: str) -> str:
        file_name = Path(file_path).name
        file_object = None
        for fo in self.file_objects:
            if fo.get("filename") == file_name:
                file_object = fo
                print("find file in remote")
                break
        if file_object is None:
            file_object = self.model.files.create(
                file=Path(file_path), purpose="file-extract"
            ).model_dump()
            self.file_objects.append(file_object)
            print(f"upload file {file_path}")
        return file_object["id"]

    def doc_query(
        self,
        question: str,
        file_path: str,
        system_prompt: str = "You are a helpful assistant.",
    ):
        file_id = self.get_file_id(file_path=file_path)
        completion = self.model.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"fileid://{file_id}"},
                {"role": "user", "content": question},
            ],
        )
        return completion.choices[0].message.content

    def delete_all_docs(self):
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        all_file_objects = client.files.list().model_dump()["data"]
        assert isinstance(all_file_objects, list)
        for fo in all_file_objects:
            client.files.delete(fo["id"])
        file_stk = client.files.list()
        print(f"now files:{file_stk.model_dump_json()}")


class ImageLLM(BaseLLM):
    def __init__(
        self,
        api_key: str | None = os.getenv("OPENAI_API_KEY"),
        model_name: str = "gpt-4o-mini",
    ):
        self.model = OpenAI(api_key=api_key)
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
        self.model_name = model_name

    # def __init__(
    #     self,
    #     api_key: str | None = os.getenv("DASHSCOPE_API_KEY"),
    #     model_name: str = "qwen-vl-plus",
    # ):
    #     self.base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    #     self.model = OpenAI(api_key=api_key, base_url=self.base_url)
    #     self.create_ask_message = lambda question: {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": question},
    #         ],
    #     }
    #     self.create_ans_message = lambda ans: {
    #         "role": "assistant",
    #         "content": [
    #             {"type": "text", "text": ans},
    #         ],
    #     }
    #     self.model_name = model_name

    def create_text_message(self, texts, question):
        content = []
        for text in texts:
            content.append({"type": "text", "text": text})
        content.append({"type": "text", "text": question})
        message = {"role": "user", "content": content}
        return message

    def process_message(  # type: ignore
        self, question, texts, images, history, figure_message_contents=[]
    ):
        if history is not None:
            assert self.is_valid_history(history)
            messages = history
        else:
            messages = []
        if texts is not None:
            messages.append(self.create_text_message(texts, question))
        if images is not None:
            messages.append(
                self.create_image_message(
                    images, question, figure_message_contents=figure_message_contents
                )
            )

        if (texts is None or len(texts) == 0) and (images is None or len(images) == 0):
            messages.append(self.create_ask_message(question))

        return messages

    def create_image_message(self, images, question, figure_message_contents=[]):
        if figure_message_contents:
            content = figure_message_contents
            content.append(
                {
                    "type": "text",
                    "text": "The next are some document pages that may contain the required information.\n",
                }
            )
        else:
            content = []
        for image_path in images:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )
        content.append({"type": "text", "text": question})
        message = {"role": "user", "content": content}
        return message

    def content2message(
        self,
        prompt: str,
        question: str,
        page_contents: list[PageContent],
        figure_contents: list[Figure],
    ) -> list:
        message_contents: list[dict[str, Any]] = [
            {"type": "text", "text": f"{prompt}{question}"}
        ]
        if figure_contents:
            message_contents.append(
                {
                    "type": "text",
                    "text": "The next are some figures ,tables or charts that may contain the required information.\n",
                }
            )
            for figure in figure_contents:
                with open(figure.image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                message_contents.extend(
                    [
                        {"type": "text", "text": f"{figure.text}:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ]
                )
        if page_contents:
            message_contents.append(
                {
                    "type": "text",
                    "text": "The next are some document pages that may contain the required information.I will give you page images and texts\n",
                }
            )
            for page in page_contents:
                with open(page.image_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                message_contents.extend(
                    [
                        {"type": "text", "text": f"Page {page.page}:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ]
                )
        messages = [{"role": "user", "content": message_contents}]
        return messages

    def query(self, messages: list) -> tuple[str, list]:
        max_retries = 10
        retry_count = 0

        while retry_count < max_retries:
            try:
                output = self.model.chat.completions.create(
                    model=self.model_name, messages=messages
                )
                output_text = output.choices[0].message.content
                if output_text is None:
                    output_text = ""
                return output_text, messages
            except (APIConnectionError, RateLimitError) as e:
                print(
                    f"Connection error occurred: {e}. Retrying ({retry_count + 1}/{max_retries})..."
                )
                retry_count += 1
                time.sleep(1)
        raise RuntimeError

    def create_figure_message(
        self, figure_contents: list[dict[str, str]], figure_prompt: str
    ):
        contents = []
        # content 内容
        # {
        #     "img_path": img_path,
        #     "caption": item.get("caption", ""),
        # }
        for content in figure_contents:
            try:
                with open(content["img_path"], "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                c = [
                    {"type": "text", "text": content["caption"]},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ]
                contents.extend(c)
            except:
                continue
        if contents:
            contents.insert(0, {"type": "text", "text": figure_prompt})
        return contents

    def predict(self, question, texts=None, images=None, history=None) -> tuple[str, list]:  # type: ignore
        messages = self.process_message(
            question,
            texts,
            images,
            history,
        )
        output = self.model.chat.completions.create(
            model=self.model_name, messages=messages
        )
        output_text = output.choices[0].message.content
        if output_text == None:
            output_text = ""
        return output_text, messages

    def is_valid_history(self, history):  # type: ignore
        if not isinstance(history, list):
            return False
        for item in history:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
            if not isinstance(item["role"], str) or not isinstance(
                item["content"], list
            ):
                return False
            for content in item["content"]:
                if not isinstance(content, dict):
                    return False
                if "type" not in content:
                    return False
                if content["type"] not in content:
                    return False
        return True


class TextLLM(BaseLLM):
    def __init__(
        self,
        api_key: str | None = os.getenv("OPENAI_API_KEY"),
        model_name: str = "gpt-4o-mini",
    ):
        self.model = OpenAI(api_key=api_key)
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
        self.model_name = model_name

    # def __init__(
    #     self,
    #     api_key: str | None = os.getenv("DASHSCOPE_API_KEY"),
    #     model_name: str = "qwen-vl-plus",
    # ):
    #     self.base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    #     self.model = OpenAI(api_key=api_key, base_url=self.base_url)
    #     self.create_ask_message = lambda question: {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": question},
    #         ],
    #     }
    #     self.create_ans_message = lambda ans: {
    #         "role": "assistant",
    #         "content": [
    #             {"type": "text", "text": ans},
    #         ],
    #     }
    #     self.model_name = model_name

    def create_text_message(self, texts, question):
        prompt = "\n".join(texts)
        message = {
            "role": "user",
            "content": f"{prompt}\n{question}",
        }
        return message

    def is_valid_history(self, history):  # type: ignore
        if not isinstance(history, list):
            return False
        for item in history:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
            if not isinstance(item["role"], str) or not isinstance(
                item["content"], str
            ):
                return False
        return True

    def content2message(
        self,
        prompt: str,
        question: str,
        page_contents: list[PageContent],
        figure_contents: list[Figure] = [],
    ) -> list:
        """
        如果不传入page content，则将prompt和question结合进行回答，否则生成page内容
        """
        text = ""
        if page_contents:
            for page in page_contents:
                text += f"{page.text.replace('\n',' ')} \n"
            text = f"The following is the reference content you can use. Answer the question mentioned above based on the content below.{text}\nQuestion:{question}"
        messages = [{"role": "user", "content": f"{prompt}{question}\n{text}"}]
        return messages

    def predict(self, question, history=None) -> tuple[str, list]:  # type: ignore
        messages = self.process_message(
            question=question,
            history=history,
        )
        output = self.model.chat.completions.create(
            model=self.model_name, messages=messages
        )
        output_text = output.choices[0].message.content
        if output_text == None:
            output_text = ""
        return output_text, messages

    def query(self, messages: list) -> tuple[str, list]:
        max_retries = 10
        retry_count = 0

        while retry_count < max_retries:
            try:
                output = self.model.chat.completions.create(
                    model=self.model_name, messages=messages
                )
                output_text = output.choices[0].message.content
                if output_text is None:
                    output_text = ""
                return output_text, messages
            except (APIConnectionError, RateLimitError) as e:
                print(
                    f"Connection error occurred: {e}. Retrying ({retry_count + 1}/{max_retries})..."
                )
                retry_count += 1
                time.sleep(1)
        raise RuntimeError


class DocLLM:
    def __init__(
        self,
        api_key: str | None = os.getenv("OPENAI_API_KEY"),
        model_name: str = "gpt-4o",
    ):
        self.model = OpenAI(api_key=api_key)
        self.model_name = model_name

    def query(self, doc_id: str, question: str, prompt: str):
        max_retries = 10
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = self.model.responses.create(
                    model="gpt-4o",
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_file",
                                    "file_id": doc_id,
                                },
                                {
                                    "type": "input_text",
                                    "text": f"{prompt}{question}",
                                },
                            ],
                        }
                    ],
                )
                return response.output_text
            except (APIConnectionError, RateLimitError) as e:
                print(
                    f"Connection error occurred: {e}. Retrying ({retry_count + 1}/{max_retries})..."
                )
                retry_count += 1
                time.sleep(1)
        return None
