import os
from base64 import b64encode
from typing import Any
from models.remote_llm import ImageLLM, DocLLM
from mydatasets.BaseDataset import BaseDataset, PageContent
from concurrent.futures import ThreadPoolExecutor, as_completed
from mydatasets.FileUploader import FileUploader
import json


class DocumentSummarizer:
    def __init__(self, dataset: BaseDataset, prompt: str):
        self.dataset = dataset
        self.dataset_name = dataset.dataset_name
        self.doc_name = dataset.document_path
        self.prompt = prompt

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return b64encode(image_file.read()).decode("utf-8")

    def build_message(
        self, summary: str, Pages: list[PageContent] = []
    ) -> list[dict[str, Any]]:
        messages: list = [{"role": "system", "content": self.prompt}]
        message_contents: list = []
        if Pages:
            for page in Pages:
                if not os.path.exists(page.image_path):
                    continue
                base64_image = self.encode_image(page.image_path)
                message_contents.extend(
                    [
                        {"type": "text", "text": f"{page.text},Page {page.page}:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ]
                )
        message_contents.append(
            {
                "type": "text",
                "text": f"The current summary:\n{summary}\nNow, give me the new summary:",
            }
        )
        messages.append({"role": "user", "content": message_contents})
        return messages

    def summarize_page(
        self, pages: list[PageContent], summary: str, model: ImageLLM
    ) -> str:
        response, _ = model.query(self.build_message(summary=summary, Pages=pages))
        return response

    def summarize_document(self, document_path: str, output: bool = False) -> str:
        os.makedirs(self.dataset.summary_path, exist_ok=True)
        document_name = os.path.splitext(os.path.basename(document_path))[0]
        result_path = os.path.join(
            self.dataset.summary_path, f"{document_name}_summary.md"
        )
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                print(f"find {result_path}")
                return f.read()
        model = ImageLLM()
        summary = ""
        page_contents = self.dataset.load_processed_content({"doc_id": document_path})
        if not page_contents:
            raise ValueError(
                f"No processed content found for document: {document_path}"
            )
        previous_page = None
        for page in page_contents:
            pages: list[PageContent] = []
            page.text = "The new page"
            if previous_page:
                previous_page.text = "The previous page "
                pages.append(previous_page)
            pages.append(page)
            previous_page = page
            new_summary = self.summarize_page(pages, summary, model)
            new_summary = "\n".join(
                line
                for line in new_summary.splitlines()
                if ("no figure" not in line and "no table" not in line)
            )
            summary = summary + f"\n{new_summary}"
        if output:
            print(f"Final summary for {document_path}:\n{summary}")
        with open(result_path, "w") as f:
            f.write(summary)
        print(f"Summary saved to {result_path}")
        return summary

    def index_document(
        self, uploader: FileUploader, docLLM: DocLLM, document_path: str, prompt: str
    ) -> str:
        os.makedirs(self.dataset.index_path, exist_ok=True)
        document_name = os.path.splitext(os.path.basename(document_path))[0]
        result_path = os.path.join(self.dataset.index_path, f"{document_name}_index.md")
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                content = f.read()
                if content[0].isdigit():
                    print(f"find {result_path}")
                    return content
                os.remove(result_path)
        result = ""
        file_id = uploader.get_fileID_by_path(filepath=document_path)
        if file_id:
            try:
                result = docLLM.query(file_id, "", prompt)
            except:
                result = ""
        else:
            print("No file id")
        result = result if result else ""
        with open(result_path, "w") as f:
            f.write(result)
        return result

    def page_index(self, document_name: str, prompt: str):
        result_path = os.path.join(
            self.dataset.index_path, f"{document_name}_index.txt"
        )
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                print(f"find {result_path}")
                return f.read()
        index_path = os.path.join(self.dataset.index_path, f"{document_name}_index.md")
        indexes = {}
        with open(index_path, "r") as f:
            for line in f:
                line = line.strip()
                parts = line.split(":", maxsplit=1)
                if len(parts) == 2 and parts[0][0].isdigit():
                    indexes[parts[0]] = f"{parts[1]} <|> "
        pages = self.dataset.load_processed_content({"doc_id": document_name})
        if not pages:
            raise ValueError(
                f"No processed content found for document: {document_name}"
            )
        previous_page: PageContent = None  # type: ignore
        model = ImageLLM()
        for page in pages:
            page2: list[PageContent] = []
            page.text = f"The new page."
            page2.append(page)
            if previous_page:
                previous_page.text = f"The previous page."
                page2.append(previous_page)
            previous_page = page
            messages: list = [{"role": "system", "content": prompt}]
            message_contents = []
            for p in page2:
                if not os.path.exists(p.image_path):
                    continue
                base64_image = self.encode_image(p.image_path)
                message_contents.extend(
                    [
                        {"type": "text", "text": f"{p.text}. Page {p.page}:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                    ]
                )
            now_content = "\n".join(f"{key}: {value}" for key, value in indexes.items())
            message_contents.append(
                {
                    "type": "text",
                    "text": f"The current outline:\n{now_content}\nNow, give me the section numbers the page belongs to in the required format.",
                }
            )
            print(now_content)
            messages.append({"role": "user", "content": message_contents})
            try:
                output, _ = model.query(messages)
            except:
                continue
            print(output)
            output = output.replace("```json", "").replace("```", "")
            try:
                section_numbers = json.loads(output)["section_numbers"]
                for number in section_numbers:
                    if number in indexes:
                        print(indexes[number] + f"{page.page}, ")
                        indexes[number] += f"{page.page}, "
            except:
                continue

        with open(result_path, "w") as f:
            f.write("\n".join(f"{key}: {value}" for key, value in indexes.items()))

    def index_dataset(self, prompt: str):
        directory_path = self.dataset.document_path
        if not os.path.isdir(directory_path):
            raise ValueError(f"The provided path {directory_path} is not a directory.")
        file_paths = [
            os.path.join(directory_path, fname)
            for fname in os.listdir(directory_path)
            if fname.endswith(".pdf")
        ]
        uploader = FileUploader(self.dataset)
        docLLM = DocLLM()
        for path in file_paths:
            self.index_document(
                uploader=uploader, docLLM=docLLM, document_path=path, prompt=prompt
            )

    def summarize_dataset(self, max_workers: int = 4):
        directory_path = self.dataset.document_path
        if not os.path.isdir(directory_path):
            raise ValueError(f"The provided path {directory_path} is not a directory.")
        file_paths = [
            os.path.join(directory_path, fname)
            for fname in os.listdir(directory_path)
            if fname.endswith(".pdf")
        ]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(self.summarize_document, path): path
                for path in file_paths
            }

            for future in as_completed(future_to_path):
                pass
